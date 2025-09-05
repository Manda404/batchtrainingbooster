
from copy import deepcopy
from typing import Optional
from catboost import CatBoostClassifier  # type: ignore
from pyspark.sql import DataFrame as SparkDataFrame
from batchtrainingbooster.core.base_trainer import BatchTrainer  # Import corrigé


class CatBoostTrainer(BatchTrainer):
    def __init__(self):
        super().__init__()
        self.global_train_loss: list[list[float]] = []  # keep track of training loss
        self.global_valid_loss: list[list[float]] = []  # keep track of validation loss
        self.global_iterations: list[int] = []  # keep track of iterations
        self.model: Optional[CatBoostClassifier] = None  # Initialize model attribute
        self.lr_schedulers: list[float] = []

    def fit(
        self,
        train_dataframe: Optional[SparkDataFrame],
        valid_dataframe: Optional[SparkDataFrame],
        target_column: str,
        **kwargs,
    ) -> Optional[CatBoostClassifier]: 
        
        # add number of batches then apply batch split
        num_batches = kwargs.get("num_batches", 10)

        if train_dataframe is None or valid_dataframe is None:
            raise ValueError("train_dataframe cannot be None or valid_dataframe cannot be None")
        
        if num_batches <= 0:
            raise ValueError("num_batches must be >= 1")
        
        if target_column not in train_dataframe.columns:
            raise ValueError(f"train_dataframe must contain '{target_column}' column")
        
        if target_column not in valid_dataframe.columns:
            raise ValueError(f"valid_dataframe must contain '{target_column}' column")
        
        
        
        
        return self.model

