from data._data_mocks import build_mock_spark_df, stratified_split_sparkdf
from spark_config._spark_testcase import SparkTestCase
from batchtrainingbooster.trainers.catboost_trainer import CatBoostTrainer
from unittest import main


class TestFitCatBoostTrainer(SparkTestCase):
    """
    Tests pour CatBoostTrainer avec organisation structurée
    """

    # Configuration par défaut
    DEFAULT_CONFIG_MULTICLASS = {
        "data": {
            "n_samples": 500,
            "target_column": "NObeyesdad",
            "train_ratio": 0.8,
            "seed": 42,
        },
        "model": {
            "loss_function": "MultiClass",
            "eval_metric": "TotalF1",
            "iterations": 50,
            "learning_rate": 0.03,
            "depth": 6,
            "l2_leaf_reg": 3.0,
            "bootstrap_type": "Bernoulli",  # 👈 ajouté
            "subsample": 0.8,  # 👈 maintenant valide
            "random_seed": 42,
            "verbose": 100,
        },
        "training": {
            "num_batches": 2,
            "max_patience": 5,
            "show_learning_curve": False,
        },
    }

    def setUp(self):
        """Configuration initiale pour chaque test"""
        super().setUp()
        self.trainer = CatBoostTrainer()  # Initialisation du trainer CatBoost
        self.train_df = None
        self.valid_df = None

    def tearDown(self):
        """Nettoyage après chaque test"""
        try:
            if hasattr(self, "train_df") and self.train_df is not None:
                self.train_df.unpersist()
            if hasattr(self, "valid_df") and self.valid_df is not None:
                self.valid_df.unpersist()
        except Exception as e:
            print(f"Warning: Erreur lors du nettoyage: {e}")
        super().tearDown()

    def _create_mock_data(self, config=None):
        """
        Création des données mock avec split stratifié
        """
        if config is None:
            config = self.DEFAULT_CONFIG_MULTICLASS["data"]

        print(f"Création du DataFrame mock ({config['n_samples']} lignes)")
        df = build_mock_spark_df(self.spark, n=config["n_samples"], seed=config["seed"])

        # Split stratifié
        self.train_df, self.valid_df = stratified_split_sparkdf(
            df,
            target_col=config["target_column"],
            valid_size=1 - config["train_ratio"],
            seed=config["seed"],
        )

        # Cache pour performance
        self.train_df.cache()
        self.valid_df.cache()

        return self.train_df, self.valid_df

    def _display_data_info(self, target_column):
        """
        Affichage des informations sur les données
        """
        print(f"Train: {self.train_df.count()} | Valid: {self.valid_df.count()}")

        # Distribution des classes
        train_dist = self.train_df.groupBy(target_column).count().orderBy(target_column)
        valid_dist = self.valid_df.groupBy(target_column).count().orderBy(target_column)

        print("\n=== Distribution Train ===")
        train_dist.show(truncate=False)

        print("\n=== Distribution Valid ===")
        valid_dist.show(truncate=False)

    def test_fit_catboost(self):
        """
        Test de la méthode fit avec configuration par défaut
        """
        print("\n" + "=" * 50)
        print("TEST: fit avec configuration par défaut")
        print("=" * 50)

        # Étape 1: Création des données
        train_df, valid_df = self._create_mock_data()
        target_column = self.DEFAULT_CONFIG_MULTICLASS["data"]["target_column"]

        # Étape 2: Affichage des infos données
        self._display_data_info(target_column)

        # Étape 3: Entraînement
        print("\n=== Début de l'entraînement batch ===")
        self.trainer.fit(
            train_dataframe=train_df,
            valid_dataframe=valid_df,
            target_column=target_column,
            config_training=self.DEFAULT_CONFIG_MULTICLASS["training"],
            config_model=self.DEFAULT_CONFIG_MULTICLASS["model"],
        )
        print("Entraînement batch terminé avec succès")

        # Étape 4: Validation
        self.assertIsNotNone(self.trainer, "Trainer est None après fit")
        self.assertTrue(
            hasattr(self.trainer, "model") and self.trainer.model is not None,
            "Trainer instancié mais le modèle interne n’a pas été entraîné",
        )


if __name__ == "__main__":
    main(verbosity=2)
