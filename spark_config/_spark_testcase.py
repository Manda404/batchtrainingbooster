# config_spark/_spark_testcase.py
import gc
import unittest
import warnings

from pyspark.sql import SparkSession


class SparkTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Option : ignorer les ResourceWarning (sockets Py4J lors de l’extinction)
        warnings.filterwarnings("ignore", category=ResourceWarning)

        cls.spark = (
            SparkSession.builder.master("local[*]")
            .appName(f"unittest-{cls.__name__}")
            .config("spark.ui.enabled", "false")
            # 👉 Force l’IP locale pour éviter le warning "loopback"
            .config("spark.driver.bindAddress", "127.0.0.1")
            .config("spark.driver.host", "127.0.0.1")
            # 👉 Moins de partitions = plus rapide pour petits DF
            .config("spark.sql.shuffle.partitions", "1")
            .getOrCreate()
        )
        cls.spark.sparkContext.setLogLevel("WARN")

    @classmethod
    def tearDownClass(cls):
        # Stoppe Spark proprement
        cls.spark.stop()
        # Aide le GC à fermer les sockets Py4J rapidement
        gc.collect()