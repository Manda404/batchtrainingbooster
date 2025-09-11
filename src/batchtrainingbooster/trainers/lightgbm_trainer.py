from copy import deepcopy
from lightgbm import LGBMClassifier
from typing import Any, Optional, cast, List, Dict
from pandas import DataFrame as PandasDataFrame
from pyspark.sql import DataFrame as SparkDataFrame
from batchtrainingbooster.core.base_trainer import BatchTrainer
from batchtrainingbooster.core.weights import OptimizedWeightCalculator


class LGBMTrainer(BatchTrainer):  # lightgbm.LGBMClassifier
    def __init__(self):
        super().__init__()
        self.global_train_loss: List[List[float]] = []  # keep track of training loss
        self.global_valid_loss: List[List[float]] = []  # keep track of validation loss
        self.global_iterations: List[int] = []  # keep track of iterations
        self.model: Optional[LGBMClassifier] = None  # Initialize model attribute
        self.lr_schedulers: List[float] = []
        self.categorical_features: Optional[List[str]] = None  #
        self.weight_calculator = OptimizedWeightCalculator()

    def fit(
        self,
        train_dataframe: Optional[SparkDataFrame],
        valid_dataframe: Optional[SparkDataFrame],
        target_column: str,
        **kwargs,
    ) -> None:
        """
        Train a LightGBM classifier in mini-batches from Spark DataFrames, with optional
        validation (early stopping), per-batch sample weights, and a learning-rate scheduler.

        Args:
            train_dataframe: Spark DataFrame used for training.
            valid_dataframe: Spark DataFrame for validation (enables metric tracking & early stopping) or None.
            target_column: Name of the target column.
            **kwargs:
                - config_model (dict): LightGBM hyperparams; `learning_rate` is updated per batch via the scheduler.
                - config_training (dict): training options (`num_batches`, `eval_metric`, `use_sample_weight`).
                - config_lr_scheduler (dict | callable | None): learning-rate scheduler config.

        Returns:
            None. Sets `self.model` to the best trained model.
        """
        # Configuration par défaut avec validation
        config_model: dict[str, Any] = cast(
            dict[str, Any], kwargs.get("config_model", {})
        )
        config_training: dict[str, Any] = cast(
            dict[str, Any], kwargs.get("config_training", {})
        )
        lr_scheduler_config: Optional[dict[str, Any]] = cast(
            Optional[dict[str, Any]], kwargs.get("config_lr_scheduler")
        )
        num_batches = config_training.get("num_batches", 10)

        # Validation des paramètres d'entrée
        self._validate_input_parameters(
            train_dataframe, valid_dataframe, target_column, num_batches
        )

        # Préparation des données d'entraînement
        dataframe_generator = self._apply_pandas_processing_to_generator(
            train_dataframe, target_column, num_batches
        )
        # Préparation des données de validation
        valid_data = self._prepare_validation_data(valid_dataframe, target_column)

        self.logger.info(f"🚀 Starting XGBoost training with {num_batches} batches")
        training_state = {
            "previous_model": None,
            "best_model": None,
            "patience_counter": 0,
            "best_valid_loss": float("inf"),
            "should_stop": False,
            "eval_metric": config_training.get("eval_metric", "binary_logloss"),
            "use_sample_weight": config_training.get("use_sample_weight", False),
        }
        self.logger.info(f"🚀 Starting LGBM training with {num_batches} batches")
        # Boucle d'entraînement principal
        for batch_id, processed_batch in enumerate(dataframe_generator):
            self.logger.info(
                f"\n--- 📦 Processing batch {batch_id + 1}/{num_batches} ---"
            )

            # Traitement du batch courant
            current_batch_data = self._prepare_batch_data(
                processed_batch, target_column, batch_id + 1
            )
            # Retrieve the current learning rate scheduler
            current_lr = self._get_current_learning_rate(
                lr_scheduler_config, batch_id + 1
            )
            self.logger.info("Mise à jour du modèle LGBM.")
            params_batch = {**config_model, "learning_rate": current_lr}
            self.logger.info(
                "LGBMClassifier configuration",
                extra={
                    "batch": batch_id,
                    "learning_rate": current_lr,
                    "lgbm_config": params_batch,
                },
            )
            self.logger.info(f"🏋️ Training LGBM on batch {batch_id + 1}")

            # Instancie un nouveau classifieur LightGBM pour ce batch (hyperparamètres dans params_batch).
            model = LGBMClassifier(**params_batch)

            # Entraînement avec evaluation sets
            self._train_batch(
                model=model,
                target_column=target_column,
                batch_data=current_batch_data,
                valid_data=valid_data,
                training_state=training_state,
            )

            # Evaluation des performances du model
            self._evaluate_trained_model(
                model=model,
                training_state=training_state,
                config_training=config_training,
                batch_id=batch_id,
            )

            # Check the early-stopping condition (based on validation metrics)
            if training_state["should_stop"]:
                self.logger.info("Early stopping triggered - Training completed")
                break

        # Persist the best-performing model found during training
        self.model = training_state["best_model"]
        self.logger.info(f"clean the cache of {type(self.weight_calculator).__name__}")
        self.weight_calculator.clear_cache()

        # Visualize training diagnostics (learning curves, validation metrics, LR schedule)
        self._visualize(
            training_state=training_state,
            config_training=config_training,
            lr_scheduler_config=lr_scheduler_config,
        )

    def _train_batch(
        self,
        model: LGBMClassifier,
        target_column: str,
        batch_data: dict,
        valid_data: PandasDataFrame,
        training_state: dict,
    ) -> None:
        """Train model using batch training"""
        # Entraînement avec evaluation sets
        model.fit(
            batch_data["features"],
            batch_data["target"],
            eval_set=[
                (batch_data["features"], batch_data["target"]),
                (
                    valid_data.drop(columns=[target_column]),
                    valid_data[target_column],
                ),
            ],
            eval_names=["train", "valid"],
            eval_metric=training_state.get("eval_metric"),
            categorical_feature=self.categorical_features
            if self.categorical_features is not None
            else "auto",
            sample_weight=batch_data["sample_weight"]
            if training_state.get("use_sample_weight")
            else None,
            init_model=training_state["previous_model"]
            if training_state["previous_model"] is not None
            else None,
        )
        # Mise à jour du modèle précédent
        training_state["previous_model"] = deepcopy(model)

    def _evaluate_trained_model(
        self,
        model: LGBMClassifier,
        training_state: dict[str, Any],
        config_training: dict[str, Any],
        batch_id: int,
    ):
        # Retrieve per-iteration evaluation history (requires .fit() with eval_set/eval_metric)
        evals: Dict[str, Dict[str, list[Any]]] = model.evals_result_
        eval_metric: str = cast(
            str, training_state.get("eval_metric", "binary_logloss")
        )
        # keys cohérentes avec eval_names=["train","valid"]
        train_scores = evals["train"][eval_metric]
        valid_scores = evals["valid"][eval_metric]
        self.global_train_loss.append(train_scores)
        self.global_valid_loss.append(valid_scores)
        self.global_iterations.append(batch_id + 1)

        # Scores du dernier epoch
        current_train_loss = train_scores[-1]
        current_valid_loss = valid_scores[-1]
        self.logger.info(
            "Batch %d - Train (last): %.5f | Valid (last): %.5f",
            batch_id + 1,
            current_train_loss,
            current_valid_loss,
        )
        # Logique d'early stopping
        if current_valid_loss < training_state["best_valid_loss"]:
            improvement = training_state["best_valid_loss"] - current_valid_loss
            self.logger.info(
                f"New best validation loss: {current_valid_loss:.5f} "
                f"(improvement: {improvement:.5f})"
            )
            training_state["best_valid_loss"] = current_valid_loss
            training_state["patience_counter"] = 0
            training_state["best_model"] = deepcopy(model)
            training_state["should_stop"] = False
        else:
            training_state["patience_counter"] += 1
            self.logger.info(
                f"⏳ No improvement - Patience: {training_state['patience_counter']}/{config_training.get('max_patience', 5)}"
            )
            training_state["should_stop"] = training_state.get(
                "patience_counter"
            ) >= config_training.get("max_patience", 5)

    def _visualize(
        self,
        training_state: dict[str, Any],
        config_training: dict[str, Any],
        lr_scheduler_config: Optional[dict[str, Any]] = None,  # <- accepter None
    ) -> None:
        if config_training.get("show_learning_curve"):
            # Génération des visualisations
            if lr_scheduler_config is not None and self.lr_schedulers:
                self._plot_lr_schedule("Exponential Decay", self.lr_schedulers)

            # Training and validation curve
            self._plot_learning_curve(
                self.global_train_loss,
                self.global_valid_loss,
                self.global_iterations,
                "LGBM",
                training_state["eval_metric"],
            )

    def _validate_input_parameters(
        self,
        train_dataframe: Optional[SparkDataFrame],
        valid_dataframe: Optional[SparkDataFrame],
        target_column: str,
        num_batches: int,
    ) -> None:
        """Validation centralisée des paramètres d'entrée."""
        self.logger.info("Validating input parameters...")
        if train_dataframe is None or valid_dataframe is None:
            raise ValueError("train dataframe and validation dataframe cannot be None")

        if num_batches <= 0:
            raise ValueError("the number of batches must be >= 1")

        if target_column not in train_dataframe.columns:
            raise ValueError(f"train dataframe must contain '{target_column}' column")

        if target_column not in valid_dataframe.columns:
            raise ValueError(f"valid dataframe must contain '{target_column}' column")
        self.logger.info("Input parameters validation passed.")

    def _prepare_validation_data(
        self,
        valid_dataframe: Optional[SparkDataFrame],
        target_column: str,
    ) -> PandasDataFrame:
        """Prépare les données de validation et détecte les variables catégorielles."""
        self.logger.info("Preparing validation data...")
        # Spark -> pandas (+ prétraitements éventuels)
        valid_dataframe_processed: PandasDataFrame = self._apply_pandas_processing(
            valid_dataframe
        )

        # Convertit les colonnes object en category si présent
        valid_dataframe_processed, cat_is_present = (
            self._convert_object_to_category_dtype(
                valid_dataframe_processed, target_column
            )
        )

        # Détecte les variables catégorielles sur les données de validation
        if cat_is_present:
            self.logger.info("Categorical features detected in validation set")
            self.categorical_features = (
                valid_dataframe_processed.drop(columns=[target_column])
                .select_dtypes(include=["category"])
                .columns.tolist()
            )
            self.logger.info(
                f"Number of categorical features identified: {len(self.categorical_features)}"
            )

        return valid_dataframe_processed

    def _get_current_learning_rate(
        self, lr_scheduler_config: Optional[dict[str, Any]], batch_id: int
    ) -> float:
        """Calcul du taux d'apprentissage courant avec scheduler."""
        if lr_scheduler_config is None:
            return 0.1

        current_lr = self._exponential_lr_schedule(
            initial_lr=lr_scheduler_config.get("initial_lr", 0.1),
            decay_rate=lr_scheduler_config.get("decay_rate", 0.95),
            batch_id=batch_id + 1,
        )

        self.logger.info(f"Learning rate for batch {batch_id + 1}: {current_lr:.6f}")
        self.lr_schedulers.append(current_lr)

        return current_lr

    def _prepare_batch_data(
        self, processed_batch: PandasDataFrame, target_column: str, batch_num: int
    ) -> dict[str, Any]:
        """Préparation des données du batch courant."""
        self.logger.info(f"Calculating sample weights for batch {batch_num}")
        sample_weight = self.weight_calculator.calculate_sample_weights(
            processed_batch[target_column]
        )
        processed_batch, _ = self._convert_object_to_category_dtype(
            processed_batch, target_column
        )

        return {
            "features": processed_batch.drop(columns=[target_column]),
            "target": processed_batch[target_column],
            "sample_weight": sample_weight,
        }

    def get_trained_model(self) -> Any:
        """
        Retourne l'instance du modèle entraîné ou initialisé.

        Returns
        -------
        Any
            L'objet du modèle (par exemple un `CatBoostClassifier`, `XGBClassifier`, etc.),
            selon l'implémentation spécifique de la classe.
        """
        return self.model
