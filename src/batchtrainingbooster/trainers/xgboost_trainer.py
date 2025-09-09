from copy import deepcopy
from typing import Optional, Union
from numpy import ndarray, asarray, unique, vectorize
from pandas import Series
from sklearn.utils.class_weight import compute_class_weight
from xgboost import XGBClassifier
from pandas import DataFrame as PandasDataFrame
from pyspark.sql import DataFrame as SparkDataFrame
from batchtrainingbooster.core.base_trainer import BatchTrainer
from matplotlib.pyplot import (
    figure,
    plot,
    title,
    xlabel,
    ylabel,
    legend,
    grid,
    show,
)


class XGBoostTrainer(BatchTrainer):
    def __init__(self):
        super().__init__()
        self.global_train_loss: list[list[float]] = []  # keep track of training loss
        self.global_valid_loss: list[list[float]] = []  # keep track of validation loss
        self.global_iterations: list[int] = []  # keep track of iterations
        self.model: Optional[XGBClassifier] = None  # Initialize model attribute
        self.lr_schedulers: list[float] = []

    def fit(
        self,
        train_dataframe: Optional[SparkDataFrame],
        valid_dataframe: Optional[SparkDataFrame],
        target_column: str,
        **kwargs,
    ) -> None:
        # Configuration par défaut avec validation
        config_model = kwargs.get("config_model", {})
        config_training = kwargs.get("config_training", {})
        num_batches = config_training.get("num_batches", 10)

        # Validation des paramètres d'entrée
        self._validate_input_parameters(
            train_dataframe, valid_dataframe, target_column, num_batches
        )

        # Après validation, on sait que les dataframes ne sont plus None
        assert train_dataframe is not None and valid_dataframe is not None

        # Préparation des données
        dataframe_generator = self._apply_pandas_processing_to_generator(
            train_dataframe, target_column, num_batches
        )

        valid_dataframe_processed = self._prepare_validation_data(
            valid_dataframe, target_column, config_model
        )

        # Initialisation des variables de suivi avec types explicites
        training_state = self._initialize_training_state(config_training, config_model)
        lr_scheduler_config = kwargs.get("config_lr_scheduler", None)

        self.logger.info(f"🚀 Starting XGBoost training with {num_batches} batches")
        try:
            # Boucle d'entraînement principal
            for batch_id, processed_batch in enumerate(dataframe_generator):
                self.logger.info(f"Processing batch {batch_id + 1}/{num_batches}")

                # Traitement du batch courant
                current_batch_data = self._prepare_batch_data(
                    processed_batch, target_column, batch_id + 1
                )
                current_lr = self._get_current_learning_rate(
                    lr_scheduler_config, batch_id
                )

                # Mise à jour ou initialisation du modèle
                training_state["booster"] = self._update_or_initialize_model(
                    training_state["booster"], config_model, current_lr, batch_id
                )

                # Entraînement du batch
                self._train_batch(
                    training_state["booster"],
                    current_batch_data,
                    valid_dataframe_processed,
                    target_column,
                    training_state["best_model"],
                    batch_id,
                )

                # Évaluation et mise à jour du meilleur modèle
                should_stop = self._evaluate_and_update_best_model(
                    training_state, batch_id, training_state["eval_metric"]
                )

                if should_stop:
                    self.logger.info("Early stopping triggered - Training completed")
                    break

        except Exception as e:
            self.logger.error(f"Training failed at batch {batch_id + 1}: {str(e)}")
            raise

        # Post-traitement et visualisations
        self._finalize_training(training_state, lr_scheduler_config, kwargs)

    def _calculate_sample_weights(
        self, y_train_batch: Union[ndarray, Series]
    ) -> ndarray:
        """
        Calcule des poids d'échantillons équilibrés pour un batch d'entraînement,
        quel que soit le nombre/les valeurs des classes.

        Paramètres
        ----------
        y_train_batch : array-like (n_samples,)
            Cibles du batch (numpy array, liste ou pandas Series).

        Retour
        ------
        np.ndarray shape (n_samples,)
            Poids d'échantillons alignés sur y_train_batch.
        """
        # Convertir en array 1D
        y = asarray(y_train_batch).ravel()

        # Classes présentes dans le batch
        classes = unique(y)

        # Poids "balanced" pour chaque classe
        class_weights = compute_class_weight(
            class_weight="balanced", classes=classes, y=y
        )

        # Mapping classe -> poids
        class_to_weight = dict(zip(classes, class_weights))

        # Assigner le poids correspondant à chaque échantillon
        sample_weight = vectorize(class_to_weight.get)(y)

        return sample_weight

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
        self, valid_dataframe: SparkDataFrame, target_column: str, config_model: dict
    ) -> PandasDataFrame:
        """Préparation des données de validation avec gestion des catégories."""
        valid_dataframe_processed: PandasDataFrame = self._apply_pandas_processing(
            valid_dataframe
        )
        valid_dataframe_processed, cat_is_present = (
            self._convert_object_to_category_dtype(
                valid_dataframe_processed, target_column
            )
        )

        if cat_is_present:
            self.logger.info(
                "Categorical features detected - Enabling categorical support"
            )
            config_model["enable_categorical"] = True

        return valid_dataframe_processed

    def _initialize_training_state(
        self, config_training: dict, config_model: dict
    ) -> dict:
        """Initialisation de l'état d'entraînement."""
        return {
            "best_model": None,
            "booster": None,
            "best_valid_loss": float("inf"),
            "patience_counter": 0,
            "max_patience": config_training.get("max_patience", 5),
            "eval_metric": config_model.get(
                "eval_metric", "logloss"
            ),  # Standardisé en minuscule
            "previous_model": None,
            # "show_learning_curve": config_training.get("show_learning_curve", True),
        }

    def _prepare_batch_data(
        self, processed_batch: PandasDataFrame, target_column: str, batch_num: int
    ) -> dict:
        """Préparation des données du batch courant."""
        self.logger.info(f"Calculating sample weights for batch {batch_num}")
        sample_weight = self._calculate_sample_weights(processed_batch[target_column])

        processed_batch, _ = self._convert_object_to_category_dtype(
            processed_batch, target_column
        )

        return {
            "features": processed_batch.drop(columns=[target_column]),
            "target": processed_batch[target_column],
            "sample_weight": sample_weight,
        }

    def _exponential_lr_schedule(
        self, initial_lr: float, decay_rate: float, batch_id: int
    ) -> float:
        """
        Calcule le learning rate à un batch donné selon une décroissance exponentielle.

        Paramètres
        ----------
        initial_lr : float, default=0.1
            Learning rate initial
        decay_rate : float, default=0.95
            Facteur de décroissance (0 < decay_rate < 1)
        batch_id : int, default=0
            Identifiant du batch courant

        Retour
        ------
        float
            Nouveau learning rate
        """
        return initial_lr * (decay_rate**batch_id)

    def _get_current_learning_rate(
        self, lr_scheduler_config: Optional[dict], batch_id: int
    ) -> float:
        """Calcul du taux d'apprentissage courant avec scheduler."""
        if lr_scheduler_config is None:
            return 0.1

        current_lr = self._exponential_lr_schedule(
            initial_lr=lr_scheduler_config.get("initial_lr", 0.1),
            decay_rate=lr_scheduler_config.get("decay_rate", 0.95),
            batch_id=batch_id,
        )

        self.logger.info(f"Learning rate for batch {batch_id + 1}: {current_lr:.6f}")
        self.lr_schedulers.append(current_lr)

        return current_lr

    def _update_or_initialize_model(
        self,
        current_booster: Optional[XGBClassifier],
        config_model: dict,
        current_lr: float,
        batch_id: int,
    ) -> XGBClassifier:
        """Mise à jour ou initialisation du modèle XGBoost."""
        if current_booster is None:
            # Première initialisation
            config_model["learning_rate"] = current_lr
            booster = XGBClassifier(**config_model)
            self.logger.info("XGBoost model initialized with parameters:")
            for param, value in booster.get_params().items():
                if param in ["learning_rate", "n_estimators", "max_depth", "subsample"]:
                    self.logger.info(f"  {param}: {value}")
            return booster
        else:
            # Mise à jour des paramètres existants
            current_booster.set_params(
                learning_rate=current_lr,
                n_estimators=config_model.get(
                    "n_estimators", current_booster.n_estimators
                ),
            )
            return current_booster

    def _train_batch(
        self,
        booster: XGBClassifier,
        batch_data: dict,
        valid_dataframe_processed: PandasDataFrame,
        target_column: str,
        best_model: Optional[XGBClassifier],
        batch_id: int,
    ) -> None:
        """Entraînement sur le batch courant avec warm restart."""
        self.logger.info(f"🏋️ Training XGBoost on batch {batch_id + 1}")

        # Détermination du modèle de warm restart
        xgb_model_param = None
        if batch_id > 0 and best_model is not None:
            xgb_model_param = best_model.get_booster()
            self.logger.debug("Using warm restart from best model")

        # Entraînement avec evaluation sets
        booster.fit(
            batch_data["features"],
            batch_data["target"],
            eval_set=[
                (batch_data["features"], batch_data["target"]),
                (
                    valid_dataframe_processed.drop(columns=[target_column]),
                    valid_dataframe_processed[target_column],
                ),
            ],
            xgb_model=xgb_model_param,
            sample_weight=batch_data["sample_weight"],
            # verbose=True,  # Évite le spam de logs XGBoost
        )

    def _evaluate_and_update_best_model(
        self, training_state: dict, batch_id: int, eval_metric: str
    ) -> bool:
        """Évaluation du modèle et mise à jour du meilleur modèle avec early stopping."""
        booster = training_state["booster"]

        # Extraction des métriques d'évaluation
        evals_result = booster.evals_result()
        train_scores = evals_result["validation_0"][eval_metric]
        valid_scores = evals_result["validation_1"][eval_metric]

        # Scores du dernier epoch
        current_train_loss = train_scores[-1]
        current_valid_loss = valid_scores[-1]

        # Sauvegarde des courbes d'apprentissage
        self.global_train_loss.append(train_scores)
        self.global_valid_loss.append(valid_scores)
        self.global_iterations.append(batch_id)

        self.logger.info(
            f"Batch {batch_id + 1} - Train: {current_train_loss:.5f} | Valid: {current_valid_loss:.5f}"
        )

        # Mise à jour du modèle précédent
        training_state["previous_model"] = deepcopy(booster)

        # Logique d'early stopping
        if current_valid_loss < training_state["best_valid_loss"]:
            improvement = training_state["best_valid_loss"] - current_valid_loss
            self.logger.info(
                f"New best validation loss: {current_valid_loss:.5f} "
                f"(improvement: {improvement:.5f})"
            )
            training_state["best_valid_loss"] = current_valid_loss
            training_state["patience_counter"] = 0
            training_state["best_model"] = deepcopy(booster)
            return False
        else:
            training_state["patience_counter"] += 1
            self.logger.info(
                f"⏳ No improvement - Patience: {training_state['patience_counter']}/{training_state['max_patience']}"
            )

            return training_state["patience_counter"] >= training_state["max_patience"]

    def _finalize_training(
        self, training_state: dict, lr_scheduler_config: Optional[dict], kwargs: dict
    ) -> None:
        """Finalisation de l'entraînement avec visualisations."""
        # Sélection du modèle final
        final_model = (
            training_state["best_model"]
            if training_state["best_model"] is not None
            else training_state["previous_model"]
        )

        if final_model is None:
            raise RuntimeError("No model was successfully trained")

        self.model = final_model

        if kwargs.get("config_training", {}).get("show_learning_curve"):
            # Génération des visualisations
            if lr_scheduler_config is not None and self.lr_schedulers:
                self._plot_lr_schedule("Exponential Decay", self.lr_schedulers)

            # Training and validation curve
            self._plot_learning_curve(
                self.global_train_loss,
                self.global_valid_loss,
                self.global_iterations,
                "XGBoost",
                training_state["eval_metric"],
            )

        # Logs de fin
        total_batches = len(self.global_iterations)
        final_valid_loss = training_state["best_valid_loss"]

        self.logger.info("Training completed successfully!")
        self.logger.info(f"Total batches processed: {total_batches}")
        self.logger.info(f"Best validation loss: {final_valid_loss:.5f}")
        self.logger.info(
            f"Using {'best' if training_state['best_model'] is not None else 'last'} model"
        )

    def _plot_lr_schedule(
        self, name: str, lrs: list, titlename: str = "Learning Rate Scheduler"
    ):
        """
        Trace la courbe d'un seul Learning Rate Scheduler.

        Paramètres
        ----------
        name : str
            Nom du scheduler (ExponentialLR, CyclicalLR, etc.)
        lrs : list
            Liste des learning rates déjà calculés
        title : str
            Titre du graphique
        """
        batch_ids = list(range(len(lrs)))

        figure(figsize=(16, 4))
        plot(batch_ids, lrs, marker="o", linestyle="-", label=name, color="orange")
        title(f"{titlename} - {name}")
        xlabel("Batch ID")
        ylabel("Learning Rate")
        legend()
        grid(True)
        show()

    def predict(
        self,
        dataframe: SparkDataFrame,
        target_column: str,
    ):
        pass
