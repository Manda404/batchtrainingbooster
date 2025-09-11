from copy import deepcopy
from catboost import CatBoostClassifier
from pandas import DataFrame as PandasDataFrame
from typing import Optional, List, Dict, Any
from pyspark.sql import DataFrame as SparkDataFrame
from batchtrainingbooster.core.base_trainer import BatchTrainer


class CatBoostTrainer(BatchTrainer):
    def __init__(self):
        super().__init__()
        self.global_train_loss: List[List[float]] = []  # keep track of training loss
        self.global_valid_loss: List[List[float]] = []  # keep track of validation loss
        self.global_iterations: List[int] = []  # keep track of iterations
        self.model: Optional[CatBoostClassifier] = None  # type: ignore
        self.categorical_features: Optional[List[str]] = None

    def fit(
        self,
        train_dataframe: Optional[SparkDataFrame],
        valid_dataframe: Optional[SparkDataFrame],
        target_column: str,
        **kwargs,
    ) -> None:
        """
        Entraînement par batch du modèle CatBoost avec early stopping global.
        """
        # Configuration par défaut avec validation
        config_model: Dict[str, Any] = deepcopy(kwargs.get("config_model", {}))
        config_training: Dict[str, Any] = deepcopy(kwargs.get("config_training", {}))
        num_batches: int = config_training.get("num_batches", 10)

        # Validation des paramètres d'entrée
        self.logger.info("Validating input parameters")
        self._validate_input_parameters(
            train_dataframe, valid_dataframe, target_column, num_batches
        )

        # Préparation des données d'entraînement
        self.logger.info("Preparing training data")
        dataframe_generator = self._apply_pandas_processing_to_generator(
            train_dataframe, target_column, num_batches
        )

        # Préparation des données de validation
        self.logger.info("Preparing validation data")
        valid_df_processed = self._prepare_validation_data(
            valid_dataframe, target_column
        )
        # Initialisation de l'état d'entraînement
        self.logger.info("Initializing training state for CatBoost")
        training_state: Dict[str, Any] = self._initialize_training_state(
            config_training=config_training, config_model=config_model
        )
        training_state["config_model"] = config_model
        self.logger.info(f"🚀 Starting CatBoost training with {num_batches} batches")
        try:
            # Boucle d'entraînement principal
            for batch_id, processed_batch in enumerate(dataframe_generator):
                self.logger.info(f"📦 Processing batch {batch_id + 1}/{num_batches}")

                # Entraînement du batch courant
                current_model = self._train_batch(
                    processed_batch,
                    valid_df_processed,
                    target_column,
                    batch_id + 1,
                    training_state,
                )

                # Évaluation et mise à jour du meilleur modèle
                should_stop = self._evaluate_and_update_best_model(
                    current_model,
                    training_state,
                    batch_id,
                )

                if should_stop:
                    self.logger.info("Early stopping triggered - Training completed")
                    break

        except Exception as e:
            self.logger.error(f"Training failed at batch {batch_id + 1}: {str(e)}")
            raise

        # Finalisation de l'entraînement
        self._finalize_training(training_state)

    def _train_batch(
        self,
        processed_batch: PandasDataFrame,
        valid_dataframe: PandasDataFrame,
        target_column: str,
        batch_num: int,
        training_state: Dict[str, Any],
    ) -> CatBoostClassifier:
        """Entraînement sur un batch avec warm restart."""
        self.logger.info(
            f"🏋️ Training CatBoost model on batch {batch_num}/{training_state['num_batches']}"
        )

        # Préparation des features et target
        X_train = processed_batch.drop(columns=[target_column])
        y_train = processed_batch[target_column]
        X_valid = valid_dataframe.drop(columns=[target_column])
        y_valid = valid_dataframe[target_column]

        # Initialisation du modèle
        model = CatBoostClassifier(**training_state.get("config_model", {}))

        # Configuration d'entraînement
        fit_params = {
            "X": X_train,
            "y": y_train,
            "init_model": training_state.get("previous_model"),
            "eval_set": [(X_train, y_train), (X_valid, y_valid)],
            "verbose": True,  # Évite le spam des logs CatBoost
            "use_best_model": False,  # Géré manuellement avec early stopping global
        }

        # Ajouter cat_features seulement si des features catégorielles existent
        if self.categorical_features:
            fit_params["cat_features"] = self.categorical_features
            self.logger.debug(
                f"🔧 Using {len(self.categorical_features)} categorical features"
            )
        else:
            self.logger.debug(
                "🔧 No categorical features - using all numerical features"
            )

        # Entraînement
        model.fit(**fit_params)

        self.logger.info(
            f"✅ Model trained on batch {batch_num}/{training_state['num_batches']}"
        )
        return model

    def _initialize_training_state(
        self, config_model: Dict[str, Any], config_training: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Initialisation de l'état d'entraînement."""
        return {
            "best_model": None,
            "previous_model": None,
            "best_valid_loss": float("inf"),
            "patience_counter": 0,
            "max_patience": config_training.get("max_patience", 5),
            "eval_metric": config_model.get("eval_metric", "logloss"),
            "num_batches": config_training.get("num_batches", 10),
            "show_learning_curve": config_training.get("show_learning_curve", True),
        }

    def _validate_input_parameters(
        self,
        train_dataframe: Optional[SparkDataFrame],
        valid_dataframe: Optional[SparkDataFrame],
        target_column: str,
        num_batches: int,
    ) -> None:
        """Validation centralisée des paramètres d'entrée."""
        if train_dataframe is None or valid_dataframe is None:
            raise ValueError("train_dataframe and valid_dataframe cannot be None")

        if num_batches <= 0:
            raise ValueError("num_batches must be >= 1")

        if target_column not in train_dataframe.columns:
            raise ValueError(f"train_dataframe must contain '{target_column}' column")

        if target_column not in valid_dataframe.columns:
            raise ValueError(f"valid_dataframe must contain '{target_column}' column")

    def _prepare_validation_data(
        self,
        valid_dataframe: Optional[SparkDataFrame],
        target_column: str,
    ) -> PandasDataFrame:
        """Prépare les données de validation et détecte les variables catégorielles."""
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

    def _evaluate_and_update_best_model(
        self,
        current_model: CatBoostClassifier,
        training_state: Dict[str, Any],
        batch_id: int,
    ) -> bool:
        """Évaluation du modèle et mise à jour du meilleur modèle avec early stopping."""

        # Mise à jour du modèle précédent
        training_state["previous_model"] = deepcopy(current_model)

        # Extraction des courbes d'apprentissage
        evals_result = current_model.get_evals_result()
        eval_metric = training_state["eval_metric"]

        train_curve = evals_result["validation_0"][eval_metric]
        valid_curve = evals_result["validation_1"][eval_metric]

        # Scores finaux
        train_loss = train_curve[-1]
        valid_loss = valid_curve[-1]

        # Sauvegarde des courbes d'apprentissage globales
        self.global_train_loss.append(train_curve)
        self.global_valid_loss.append(valid_curve)
        self.global_iterations.append(batch_id)

        self.logger.info(
            f"Batch {batch_id + 1} - Train: {train_loss:.5f} | Valid: {valid_loss:.5f}"
        )

        # Logique d'early stopping
        if valid_loss < training_state["best_valid_loss"]:
            improvement = training_state["best_valid_loss"] - valid_loss
            training_state["best_valid_loss"] = valid_loss
            training_state["patience_counter"] = 0
            training_state["best_model"] = deepcopy(current_model)

            self.logger.info(
                f"🎉 New best model found with Valid Loss: {valid_loss:.5f} "
                f"(improvement: {improvement:.5f})"
            )
            return False
        else:
            training_state["patience_counter"] += 1
            self.logger.info(
                f"⏳ No improvement - Patience: {training_state['patience_counter']}/{training_state['max_patience']}"
            )

            if training_state["patience_counter"] >= training_state["max_patience"]:
                self.logger.warning("Early stopping triggered")
                return True

        return False

    def _finalize_training(
        self,
        training_state: Dict[str, Any],
    ) -> None:
        """Finalisation de l'entraînement avec visualisations et sélection du modèle final."""

        # Sélection du modèle final
        final_model = (
            training_state["best_model"]
            if training_state["best_model"] is not None
            else training_state["previous_model"]
        )

        if final_model is None:
            raise RuntimeError("No model was successfully trained")

        self.model = final_model

        # Génération des courbes d'apprentissage
        if training_state["show_learning_curve"]:
            self._plot_learning_curve(
                self.global_train_loss,
                self.global_valid_loss,
                self.global_iterations,
                "CatBoost",
                training_state["eval_metric"],
            )

        # Logs de fin
        total_batches = len(self.global_iterations)
        final_valid_loss = training_state["best_valid_loss"]
        model_type = "best" if training_state["best_model"] is not None else "last"

        self.logger.info("CatBoost training completed successfully!")
        self.logger.info(f"Total batches processed: {total_batches}")
        self.logger.info(f"Best validation loss: {final_valid_loss:.5f}")
        self.logger.info(f"Using {model_type} model")

        if self.categorical_features:
            self.logger.info(
                f"Categorical features used: {len(self.categorical_features)}"
            )

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
