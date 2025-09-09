from typing import Optional, Union
from numpy import ndarray, asarray, unique, vectorize
from sklearn.utils.class_weight import compute_class_weight
from lightgbm import LGBMClassifier
from pandas import DataFrame as PandasDataFrame, Series
from pyspark.sql import DataFrame as SparkDataFrame
from batchtrainingbooster.core.base_trainer import BatchTrainer

class LGBMTrainer(BatchTrainer):      #lightgbm.LGBMClassifier
    def __init__(self):
        super().__init__()
        self.global_train_loss: list[list[float]] = []  # keep track of training loss
        self.global_valid_loss: list[list[float]] = []  # keep track of validation loss
        self.global_iterations: list[int] = []  # keep track of iterations
        self.model: Optional[LGBMClassifier] = None  # Initialize model attribute
        self.lr_schedulers: list[float] = []

    def fit(
        self,
        train_dataframe: Optional[SparkDataFrame],
        valid_dataframe: Optional[SparkDataFrame],
        target_column: str,
        **kwargs,
    ) -> None:
        
        pass


    def predict(
        self,
        dataframe: SparkDataFrame,
        target_column: str,
    ):
        pass
