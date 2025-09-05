from pandas import Series
from copy import deepcopy
from typing import Tuple, Union
from xgboost import XGBClassifier
from pyspark.sql import DataFrame as SparkDataFrame
from pandas import DataFrame as PandasDataFrame
from batchtrainingbooster.core import BatchTrainer
from numpy import ndarray, asarray, unique, vectorize
from sklearn.utils.class_weight import compute_class_weight
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
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.global_train_loss: list[list[float]] = []  # keep track of training loss
        self.global_valid_loss: list[list[float]] = []  # keep track of validation loss
        self.global_iterations: list[int] = []  # keep track of iterations
        self.model = None
        self.lr_schedulers: list[float] = []

    def calculate_sample_weights(
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

    def exponential_lr_schedule(
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

    def plot_lr_schedule(
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

    def convert_object_to_category_dtype(
        self, data: PandasDataFrame, target_column: str = ""
    ) -> Tuple[PandasDataFrame, bool]:
        """
        Convert object and category columns in the provided PandasDataFrame to the 'category' dtype.

        This function identifies columns with object or category types in the PandasDataFrame
        and converts them to the 'category' dtype for improved memory usage and performance.

        Parameters
        ----------
        data : PandasDataFrame
            The dataset containing features.

        target_column : str
            The name of the target column that should not be converted.

        Returns
        -------
        PandasDataFrame
            The modified DataFrame with categorical columns converted.
        """
        # Identify categorical features
        cat_features = [
            col
            for col in data.select_dtypes(include=["object"]).columns.tolist()
            if col != target_column
        ]

        # Check if there are categorical features that are not already of type 'category'
        if cat_features:
            # Convert identified features to 'category' dtype
            data[cat_features] = data[cat_features].astype("category")
            is_present = True

        return data, is_present

    def fit(
        self,
        train_dataframe: SparkDataFrame,
        valid_dataframe: SparkDataFrame,
        target_column: str,
        **kwargs,
    ):
        if train_dataframe is None and valid_dataframe is None:
            raise ValueError(
                "Both training and validation dataframes are None. Cannot proceed with training."
            )

        if target_column is None:
            raise ValueError("Target column is None. Cannot proceed with training.")

        # add number of batches then apply batch split
        num_batches = kwargs.get("num_batches", 10)

        # Create a generator for the training dataframe
        dataframe_generator = self._apply_pandas_processing_to_generator(
            train_dataframe, target_column, num_batches
        )

        # process the evaluation dataframe
        valid_dataframe = self._apply_pandas_processing_to_validation_set(
            valid_dataframe,
        )
        valid_dataframe, cat_is_present = self.convert_object_to_category_dtype(
            valid_dataframe, target_column
        )
        # initialize best and previous model, tvvo search optimal performances
        best_model, booster = None, None
        max_patience = kwargs.get("config_training", {}).get("max_patience", 5)
        best_valid_loss, patience_counter = float("inf"), 0
        config_model = kwargs.get("config_model", {})
        eval_metric = kwargs.get("config_model", {}).get("eval_metric", "Logloss")
        lr_scheduler = kwargs.get("config_lr_scheduler", None)
        if cat_is_present:
            self.logger.info(
                "Categorical features are present in the train and validation set."
            )
            config_model["enable_categorical"] = True

        for batch_id, processed_batch in enumerate(dataframe_generator):
            self.logger.info("Processing batch %d / %d", batch_id + 1, num_batches)

            # Calculate sample weights based on the target variable for the current batch
            self.logger.info("Calculating sample weights for batch %d", batch_id + 1)
            sample_weight_batch = self.calculate_sample_weights(
                processed_batch[target_column]
            )

            processed_batch, cat_is_present = self.convert_object_to_category_dtype(
                processed_batch, target_column
            )

            # print(sample_weight_batch)

            # self.logger.info("Aspect implementation here !!!")

            # Set current learning rate
            current_lr = (
                self.exponential_lr_schedule(
                    initial_lr=lr_scheduler.get("initial_lr", 0.1),
                    decay_rate=lr_scheduler.get("decay_rate", 0.95),
                    batch_id=batch_id,
                )
                if lr_scheduler is not None
                else 0.1
            )
            print(
                f"Learning rate for batch {batch_id + 1}/{num_batches} : {current_lr}"
            )
            # Append current learning rate to the list
            self.lr_schedulers.append(current_lr)

            # If first batch → initialize XGBClassifier
            if booster is None:
                config_model["learning_rate"] = current_lr
                booster = XGBClassifier(**config_model)
                params = booster.get_params()
                self.logger.info(
                    "XGBoostClassifier parameters before initialization: %s",
                    ", ".join([f"{param}: {value}" for param, value in params.items()]),
                )
            else:
                # update learning rate + n_estimators
                booster.set_params(
                    learning_rate=current_lr, n_estimators=config_model["n_estimators"]
                )

            # Fit with warm start (xgb_model)
            self.logger.info(
                f"Fitting the model XGBoost in batch {batch_id + 1}/{num_batches} for Batch ID {batch_id}."
            )
            booster.fit(
                processed_batch.drop(columns=[target_column]),
                processed_batch[target_column],
                eval_set=[
                    (
                        processed_batch.drop(columns=[target_column]),
                        processed_batch[target_column],
                    ),
                    (
                        valid_dataframe.drop(columns=[target_column]),
                        valid_dataframe[target_column],
                    ),
                ],
                # verbose=verbose,
                xgb_model=best_model.get_booster()
                if batch_id + 1 > 1
                else None,  # J'hésite encore entre utiliser 'booster' ou 'best_model'.
                sample_weight=sample_weight_batch,
            )
            self.logger.info(f"Completed fitting model for Batch {batch_id}.")

            # Updating previous model
            previous_model = deepcopy(booster)

            # Extract training and validation loss
            list_train_loss = booster.evals_result()["validation_0"][eval_metric]
            list_valid_curve = booster.evals_result()["validation_1"][eval_metric]
            train_loss, valid_loss = list_train_loss[-1], list_valid_curve[-1]

            self.logger.info(
                f"Train Loss: {train_loss:.5f} | Valid Loss: {valid_loss:.5f}"
            )
            self.global_train_loss.append(list_train_loss)
            self.global_valid_loss.append(list_valid_curve)
            self.global_iterations.append(batch_id)
            # Check for improvement and save best model if needed
            # Early stopping global
            if valid_loss < best_valid_loss:
                self.logger.info(
                    f"👍 - New best validation loss achieved: {valid_loss:.4f} (previous best: {best_valid_loss:.4f})"
                )
                best_valid_loss = valid_loss
                patience_counter = 0
                best_model = deepcopy(booster)
            else:
                self.logger.info(
                    f"👎 - No improvement in the validation log loss. Patience counter: {patience_counter}/{max_patience}"
                )
                patience_counter += 1
                if patience_counter >= max_patience:
                    self.logger.info(
                        "⏳ - Global early stopping triggered after reaching maximum patience."
                    )
                    break

        if lr_scheduler is not None:
            self.plot_lr_schedule("Exponential Decay", self.lr_schedulers)
        if kwargs.get("show_learning_curve", True):
            self._plot_learning_curve(
                self.global_train_loss,
                self.global_valid_loss,
                self.global_iterations,
                "XGBoost",
                eval_metric,
            )
            self.logger.info(f"Completed fitting model for Batch {batch_id + 1}.")
        self.model = best_model if best_model else previous_model

    def predict(
        self,
        dataframe: SparkDataFrame,
        target_column: str,
    ):
        pass
