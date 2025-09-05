from copy import deepcopy
from typing import Union

from catboost import CatBoostClassifier
from pyspark.sql import DataFrame as SparkDataFrame
from sklearn.pipeline import Pipeline

from incrementaltraining.core import BatchTrainer


class CatBoostTrainer(BatchTrainer):
    def __init__(self):
        super().__init__()
        self.global_train_loss = []  # keep track of training loss
        self.global_valid_loss = []  # keep track of validation loss
        self.global_iterations = []  # keep track of iterations

    def fit(
        self,
        train_dataframe: SparkDataFrame,
        valid_dataframe: SparkDataFrame,
        target_column: str,
        pipeline: Union[Pipeline, None] = None,
        **kwargs,
    ):
        # add number of batches then apply batch split
        num_batches = kwargs.get("num_batches", 10)
        dataframe_generator = self._apply_pandas_processing_to_generator(
            train_dataframe, target_column, pipeline, num_batches
        )
        # process the evalutation dateframe
        if valid_dataframe is not None:
            valid_dataframe = self._apply_pandas_processing_to_validation_set(
                valid_dataframe, pipeline
            )
        # initialize best and previous model, to search optimal performances
        best_model, previous_model = None, None
        eval_metric = kwargs.get("config_model", {}).get("eval_metric", "Logloss")
        max_patience = kwargs.get("config_training", {}).get("max_patience", 5)
        best_valid_loss, patience_counter = float("inf"), 0
        config_model = kwargs.get("config_model", {})

        for batch_id, processed_batch in enumerate(dataframe_generator):
            self.logger.info("Processing batch %d / %d", batch_id + 1, num_batches)

            if previous_model is None:
                categorical_features = (
                    processed_batch.drop(columns=[target_column])
                    .select_dtypes(include=["category", "object"])
                    .columns.tolist()
                )

                self.logger.info(
                    "Identified categorical features: %s", categorical_features
                )

            self.logger.info(
                "Training model on batch %d / %d", batch_id + 1, num_batches
            )
            model = CatBoostClassifier(**config_model)
            model.fit(
                processed_batch.drop(columns=[target_column]),
                processed_batch[target_column],
                cat_features=categorical_features,
                init_model=previous_model,
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
            )
            self.logger.info(
                "Model trained on batch %d / %d", batch_id + 1, num_batches
            )

            # Updating previous model
            previous_model = deepcopy(model)

            # Extract training and validation loss
            train_curve = model.get_evals_result()["validation_0"][eval_metric]
            valid_curve = model.get_evals_result()["validation_1"][eval_metric]
            train_loss, valid_loss = train_curve[-1], valid_curve[-1]

            self.logger.info(
                f"Train Loss: {train_loss:.5f} | Valid Loss: {valid_loss:.5f}"
            )
            self.global_train_loss.append(train_curve)
            self.global_valid_loss.append(valid_curve)
            self.global_iterations.append(batch_id)

            # Check for improvement and save best model if needed
            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                patience_counter = 0
                best_model = deepcopy(model)
                self.logger.info(
                    f"New best model found with Valid Loss: {best_valid_loss:.5f}"
                )
            else:
                patience_counter += 1
                self.logger.info(
                    f"No improvement - patience {patience_counter}/{max_patience}"
                )
                if patience_counter >= max_patience:
                    self.logger.warning("Early stopping triggered")
                    break

        if kwargs.get("show_learning_curve", True):
            self._plot_learning_curve(
                self.global_train_loss,
                self.global_valid_loss,
                self.global_iterations,
                "CatBoost",
                eval_metric,
            )

        self.logger.info("Training completed")
        return best_model if best_model else previous_model

    def predict(
        self,
        dataframe: SparkDataFrame,
        target_column: str,
        pipeline: Union[Pipeline, None] = None,
    ):
        pass
