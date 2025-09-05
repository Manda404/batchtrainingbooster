from pyspark.sql import DataFrame, functions as F


def add_bmi(
    df: DataFrame, height_col: str = "Height", weight_col: str = "Weight"
) -> DataFrame:
    """Add BMI = Weight / Height^2 when both columns exist."""
    if height_col in df.columns and weight_col in df.columns:
        return df.withColumn("BMI", F.col(weight_col) / (F.col(height_col) ** 2))
    return df


