from typing import Tuple, Optional
from pandas import DataFrame as PandasDataFrame, concat
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


def stratified_split_dataset(
    df: PandasDataFrame,
    target_column: str,
    train_ratio: float = 0.7,
    valid_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_state: int = 42,
    **kwargs
) -> Tuple[PandasDataFrame, PandasDataFrame, PandasDataFrame]:
    """
    Split a dataset into train, validation, and test sets with stratification.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataset.
    target_column : str
        Target column name for stratification.
    train_ratio : float, default=0.7
        Proportion of data to use for training.
    valid_ratio : float, default=0.15
        Proportion of data to use for validation.
    test_ratio : float, default=0.15
        Proportion of data to use for testing.
    random_state : int, default=42
        Random seed for reproducibility.
    **kwargs :
        Extra keyword arguments passed to `train_test_split`.

    Returns
    -------
    (train_df, valid_df, test_df) : tuple of pd.DataFrame
        Stratified splits of the dataset.
    """
    # Vérifier que la somme des ratios = 1
    total = train_ratio + valid_ratio + test_ratio
    if not abs(total - 1.0) < 1e-6:
        raise ValueError("train_ratio + valid_ratio + test_ratio must equal 1.0")

    # Split train vs temp
    train_df, temp_df = train_test_split(
        df,
        stratify=df[target_column],
        test_size=(1 - train_ratio),
        random_state=random_state,
        **kwargs
    )

    # Proportion relative de valid/test
    valid_size = valid_ratio / (valid_ratio + test_ratio)

    # Split valid vs test
    valid_df, test_df = train_test_split(
        temp_df,
        stratify=temp_df[target_column],
        test_size=(1 - valid_size),
        random_state=random_state,
        **kwargs
    )

    # Print dimensions + nb classes
    print("Dataset split results:")
    print(f" - Train set: {train_df.shape} → {len(train_df)} samples, "
          f"{train_df[target_column].nunique()} distinct classes")
    print(f" - Valid set: {valid_df.shape} → {len(valid_df)} samples, "
          f"{valid_df[target_column].nunique()} distinct classes")
    print(f" - Test set : {test_df.shape} → {len(test_df)} samples, "
          f"{test_df[target_column].nunique()} distinct classes")

    return train_df, valid_df, test_df


def stop_spark_session(spark: Optional[SparkSession]) -> None:
    """
    Stoppe proprement une session Spark si elle existe.

    Parameters
    ----------
    spark : Optional[SparkSession]
        La session Spark à arrêter.
    """
    if spark is not None:
        print("[SPARK] Stopping Spark session...")
        spark.stop()
        print("[SPARK] Session stopped successfully.")
    else:
        print("[SPARK] No active Spark session to stop.")
