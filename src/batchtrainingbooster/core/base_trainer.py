from typing import List
from numpy import cumsum
from pyspark.sql import Window
from abc import ABC, abstractmethod
from logger.logger import setup_logger
from pandas import DataFrame as PandasDataFrame
from pyspark.sql import DataFrame as SparkDataFrame
from pyspark.sql.functions import col, ntile, rand
from matplotlib.pyplot import (
    figure,
    plot,
    gca,
    title,
    xlabel,
    ylabel,
    legend,
    grid,
    show,
)


class BatchTrainer(ABC):
    def __init__(self):
        self.logger = setup_logger(__name__)

    @abstractmethod
    def fit(
        self,
        dataframe: SparkDataFrame,
        target_column: str,
        **kwargs,
    ):
        pass

    @abstractmethod
    def predict(
        self,
        dataframe: SparkDataFrame,
        target_column: str,
        **kwargs,
    ):
        pass

    def _create_and_apply_batches(self, dataframe, target_column: str, **kwargs):
        """
        Create balanced batches per target class using ntile over a random order.

        Args:
            dataframe (SparkDataFrame): The input DataFrame.
            target_columns (str): The target columns for batching.
            **kwargs: Additional keyword arguments.
        """
        num_batches = int(kwargs.get("num_batches", 10))
        if num_batches <= 0:
            raise ValueError("num_batches must be >= 1")

        if target_column not in dataframe.columns:
            raise ValueError(f"DataFrame must contain '{target_column}' column")

        self.logger.info("Creating batches...")
        # Stratify by target; randomize order inside each class (seed for reproducibility)
        w = Window.partitionBy(target_column).orderBy(rand(seed=42))

        # ntile K → buckets 1..K. Convert to 0..K-1 if tu préfères.
        df = dataframe.withColumn("batch_id", ntile(num_batches).over(w) - 1)
        self.logger.info(
            "Created %d batches stratified by %s", num_batches, target_column
        )
        self.logger.info("Batches created successfully.")

        return df

    def _apply_pandas_processing_to_generator(
        self,
        dataframe: SparkDataFrame,
        batch_column: str,
        num_batches: int,
    ) -> PandasDataFrame:
        """
        Apply pandas processing to a Spark DataFrame in a streaming fashion.

        Args:
            dataframe (SparkDataFrame): The input Spark DataFrame.
            batch_column (str): The column used for batching.
            pipeline (Union[Pipeline, None]): The sklearn pipeline to apply.
            num_batches (int): The number of batches to create.

        Yields:
            DataFrame: The processed DataFrame for each batch.
        """
        self.logger.info("Starting to process DataFrame in batches...")
        # Create and apply batches
        dataframe = self._create_and_apply_batches(
            dataframe, batch_column, num_batches=num_batches
        )
        self.logger.info("Batches created and applied to DataFrame.")

        # Process each batch
        for batch_id in range(num_batches):
            self.logger.info(f"Filtering and processing batch {batch_id}")
            batch_dataframe = dataframe.filter(col("batch_id") == batch_id)

            # Convert Spark DataFrame to pandas DataFrame
            self.logger.info(
                f"Converting Spark DataFrame to pandas DataFrame for batch {batch_id}"
            )
            pandas_df = batch_dataframe.toPandas().drop(columns=["batch_id"])

            yield pandas_df

    def _apply_pandas_processing_to_validation_set(
        self,
        dataframe: SparkDataFrame,
    ) -> PandasDataFrame:
        """
        Convert a Spark DataFrame to a pandas DataFrame and optionally apply a
        pandas-compatible pipeline (`.transform(pd.DataFrame) -> pd.DataFrame`).
        """
        # Spark -> pandas conversion
        self.logger.info(
            "Converting Spark DataFrame to pandas DataFrame for validation."
        )
        pandas_df: PandasDataFrame = dataframe.toPandas()

        self.logger.debug(
            "Pandas DataFrame created with %d rows and %d columns",
            pandas_df.shape[0],
            pandas_df.shape[1],
        )
        return pandas_df

    def _plot_learning_curve(
        self,
        global_train_loss: List[List[float]],
        global_valid_loss: List[List[float]],
        global_iterations: List[int],
        model_name: str = "CatBoost",
        eval_metric: str = "Logloss",
    ) -> None:
        """
        Plot the global flattened learning curve over all batches.

        Args:
            global_train_loss (List[float]): Flattened list of training loss values across all batches.
            global_valid_loss (List[float]): Flattened list of validation loss values across all batches.
            global_iterations (List[int]): Global iteration indices corresponding to the losses.
            model_name (str): Model name for the plot title.

        Returns:
            None: Displays the plot.
        """
        self.logger.info(f"Plotting learning curve of {model_name} model...")
        flattened_train_loss = [
            loss for batch_curve in global_train_loss for loss in batch_curve
        ]
        flattened_val_loss = [
            loss for batch_curve in global_valid_loss for loss in batch_curve
        ]
        batch_start_indices = cumsum(
            [0] + [len(curve) for curve in global_train_loss[:-1]]
        )

        figure(figsize=(20, 6))
        plot(flattened_train_loss, label="Train Logloss (flattened)")
        plot(flattened_val_loss, label="Validation Logloss (flattened)")

        ax1 = gca()
        epoch = 0
        for idx in range(len(batch_start_indices)):
            batch_num = global_iterations[idx]
            ax1.axvline(
                x=batch_start_indices[idx], color="red", linestyle="--", linewidth=0.8
            )
            ax1.text(
                batch_start_indices[idx],
                ax1.get_ylim()[1] * 0.95,
                f"Epoch {epoch} - Batch {batch_num}",
                ha="center",
                fontsize=10,
                color="gray",
            )

        title(f"Global Flattened Learning Curve Over All Batches ({model_name})")
        xlabel("Global Iteration (All Batches Concatenated)")
        ylabel(f"{eval_metric}")
        legend()
        grid()
        show()
        self.logger.info("Learning curve plotted successfully.")