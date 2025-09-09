import random
from random import randint, choice, uniform
from typing import Tuple, List, Optional
from pyspark.sql import DataFrame as SparkDataFrame
from dataclasses import dataclass, asdict
from pyspark.sql import SparkSession, DataFrame as SparkDataFrame
from pyspark.sql import types as T
from pyspark.sql import functions as F
from pyspark.sql import Window as W

@dataclass
class ObesityRow:
    Age: float
    Gender: str
    Height: float
    Weight: float
    FAVC: str
    FCVC: float
    NCP: float
    SCC: str
    SMOKE: str
    CH2O: float
    family_history_with_overweight: str
    FAF: float
    TUE: float
    CAEC: str
    CALC: float
    MTRANS: str
    NObeyesdad: str


def generate_mock_row() -> ObesityRow:
    """Construit une ligne ObesityRow avec des valeurs aléatoires réalistes."""
    return ObesityRow(
        Age=float(randint(18, 60)),
        Gender=choice(["Male", "Female"]),
        Height=round(uniform(1.50, 1.90), 2),
        Weight=round(uniform(45, 140), 1),
        FAVC=choice(["yes", "no", "Frequently"]),
        FCVC=round(uniform(1, 3), 1),
        NCP=round(uniform(2, 5), 1),
        SCC=choice(["yes", "no"]),
        SMOKE=choice(["yes", "no"]),
        CH2O=round(uniform(1, 3), 1),
        family_history_with_overweight=choice(["yes", "no"]),
        FAF=round(uniform(0, 3), 1),
        TUE=round(uniform(0, 2), 1),
        CAEC=choice(["no", "Sometimes", "Frequently", "Always"]),
        CALC=round(uniform(0, 3), 1),
        MTRANS=choice(["Walking", "Bike", "Automobile", "Public_Transportation"]),
        NObeyesdad=choice([
            "Underweight",
            "Normal_Weight",
            "Overweight_Level_I",
            "Overweight_Level_II",
            "Obesity_Type_I",
            "Obesity_Type_II",
            "Obesity_Type_III",
        ]),
    )

def build_mock_dataset(n: int, seed: Optional[int] = None) -> List[ObesityRow]:
    """Construit une liste de n ObesityRow (optionnellement reproductible via seed)."""
    if seed is not None:
        random.seed(seed)
    return [generate_mock_row() for _ in range(n)]

# --- PySpark ---
SCHEMA = T.StructType([
    T.StructField("Age", T.DoubleType(), False),
    T.StructField("Gender", T.StringType(), False),
    T.StructField("Height", T.DoubleType(), False),
    T.StructField("Weight", T.DoubleType(), False),
    T.StructField("FAVC", T.StringType(), False),
    T.StructField("FCVC", T.DoubleType(), False),
    T.StructField("NCP", T.DoubleType(), False),
    T.StructField("SCC", T.StringType(), False),
    T.StructField("SMOKE", T.StringType(), False),
    T.StructField("CH2O", T.DoubleType(), False),
    T.StructField("family_history_with_overweight", T.StringType(), False),
    T.StructField("FAF", T.DoubleType(), False),
    T.StructField("TUE", T.DoubleType(), False),
    T.StructField("CAEC", T.StringType(), False),
    T.StructField("CALC", T.DoubleType(), False),
    T.StructField("MTRANS", T.StringType(), False),
    T.StructField("NObeyesdad", T.StringType(), False),
])

def build_mock_spark_df(spark: SparkSession, n: int, seed: Optional[int] = None) -> SparkDataFrame:
    """
    Génère n lignes mockées et retourne un DataFrame Spark avec un schéma explicite.
    """
    rows = build_mock_dataset(n, seed=seed)
    data = [asdict(r) for r in rows]
    return spark.createDataFrame(data, schema=SCHEMA)

def stratified_split_sparkdf(
    sparkdf: SparkDataFrame,
    target_col: str = "NObeyesdad",
    valid_size: float = 0.2,
    seed: int = 42
) -> Tuple[SparkDataFrame, SparkDataFrame]:
    """
    Split stratifié d'un DataFrame Spark en (train_df, valid_df) en préservant
    les proportions de la classe cible `target_col`.

    Args:
        sparkdf (SparkDataFrame): DataFrame d'entrée.
        target_col (str): Nom de la colonne cible (stratification). Défaut: 'NObeyesdad'.
        valid_size (float): Proportion du split validation (0 < valid_size < 1). Défaut: 0.2.
        seed (int): graine pour la reproductibilité du tirage aléatoire. Défaut: 42.

    Returns:
        Tuple[SparkDataFrame, SparkDataFrame]: (train_df, valid_df)
    """
    if not (0.0 < valid_size < 1.0):
        raise ValueError("valid_size doit être dans l'intervalle (0, 1).")

    # Numéroter aléatoirement (mais de façon reproductible) les lignes dans chaque classe
    w = W.partitionBy(target_col).orderBy(F.rand(seed))
    df_rn = sparkdf.withColumn("_rn", F.row_number().over(w))

    # Calculer le nombre de lignes par classe et le seuil de validation par classe
    class_counts = sparkdf.groupBy(target_col).agg(F.count(F.lit(1)).alias("_cnt"))
    thresholds = class_counts.withColumn("_valid_k", F.ceil(F.col("_cnt") * F.lit(valid_size)))

    # Joindre les informations de seuil par classe
    df_join = df_rn.join(thresholds, on=target_col, how="inner")

    # Split stratifié
    valid_df = df_join.where(F.col("_rn") <= F.col("_valid_k")).drop("_rn", "_cnt", "_valid_k")
    train_df = df_join.where(F.col("_rn") > F.col("_valid_k")).drop("_rn", "_cnt", "_valid_k")

    return train_df, valid_df
