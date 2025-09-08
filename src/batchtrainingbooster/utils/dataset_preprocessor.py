from pandas import DataFrame as PandasDataFrame, concat
from typing import Tuple, Optional
from sklearn.model_selection import train_test_split
from pyspark.sql import SparkSession, DataFrame as SparkDataFrame


# 1) Train/Validation Split with stratification
def make_train_valid_split(
    df: PandasDataFrame,
    target: str,
    train_size: float = 0.8,
    val_size: float = 0.2,
    **kwargs,
) -> Tuple[PandasDataFrame, PandasDataFrame]:
    """
    Split a pandas DataFrame into train/validation sets with optional stratification.

    Parameters
    ----------
    df : PandasDataFrame
        Full dataset (must include the target column).
    target : str
        Target column name.
    train_size : float, default=0.8
        Proportion for the train set.
    val_size : float, default=0.2
        Proportion for the validation set.
    **kwargs :
        Extra keyword arguments passed to sklearn's `train_test_split`.

    Returns
    -------
    (train_df, valid_df) : Tuple[PandasDataFrame, PandasDataFrame]
        Two pandas DataFrames containing all original columns.
    """
    if target not in df.columns:
        raise KeyError(f"Target column '{target}' is missing in DataFrame.")

    total = train_size + val_size
    if not (0.999 <= total <= 1.001):
        raise ValueError(
            f"train_size ({train_size}) + val_size ({val_size}) must sum to 1.0."
        )

    X = df.drop(columns=[target])
    y = df[target]

    # Default stratify behavior unless overridden in kwargs
    stratify_arg = kwargs.get("stratify", y)
    kwargs["stratify"] = stratify_arg if stratify_arg is not None else None

    X_train, X_valid, y_train, y_valid = train_test_split(
        X,
        y,
        test_size=val_size,
        **kwargs,
    )

    train_df = concat([X_train, y_train], axis=1)[df.columns]
    valid_df = concat([X_valid, y_valid], axis=1)[df.columns]

    print(f"[SPLIT] Train shape: {train_df.shape} | Validation shape: {valid_df.shape}")
    return train_df, valid_df


# 2) Pandas → Spark conversion
def to_spark_dfs(
    train_df: PandasDataFrame,
    valid_df: PandasDataFrame,
    spark: Optional[SparkSession] = None,
    app_name: str = "PandasToSpark",
    **kwargs,
) -> Tuple[SparkDataFrame, SparkDataFrame]:
    """
    Convert two pandas DataFrames into Spark DataFrames.

    Parameters
    ----------
    train_df : PandasDataFrame
        Training set in pandas.
    valid_df : PandasDataFrame
        Validation set in pandas.
    spark : Optional[SparkSession], default=None
        Existing Spark session. If None, a new one will be created.
    app_name : str, default="PandasToSpark"
        Name of the Spark application if a session is created.
    **kwargs :
        Extra keyword arguments passed to `spark.createDataFrame`.

    Returns
    -------
    (spark_train_df, spark_valid_df) : Tuple[DataFrame, DataFrame]
        Two Spark DataFrames.
    """
    if spark is None:
        spark = SparkSession.builder.appName(app_name).getOrCreate()

    spark_train_df = spark.createDataFrame(train_df, **kwargs)
    spark_valid_df = spark.createDataFrame(valid_df, **kwargs)

    print("[SPARK] Conversion successful → Spark DataFrames created.")
    print(
        f"[SPARK] Train rows: {spark_train_df.count()} | Validation rows: {spark_valid_df.count()}"
    )

    return spark_train_df, spark_valid_df
