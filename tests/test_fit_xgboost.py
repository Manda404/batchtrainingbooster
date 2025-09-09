from unittest import main
from pyspark.sql import functions as F
from spark_config._spark_testcase import SparkTestCase
from batchtrainingbooster.trainers.xgboost_trainer import XGBoostTrainer
from data._data_mocks import build_mock_spark_df, stratified_split_sparkdf


class TestFitXGBoostTrainer(SparkTestCase):
    """
    Tests pour CatBoostTrainer avec organisation structurée
    """
    
    # Configuration pour classification multiclasse
    DEFAULT_CONFIG_MULTICLASS = {
        'data': {
            'n_samples': 500,
            'target_column': 'NObeyesdad',  # Multiclasse: 7 classes
            'train_ratio': 0.8,
            'seed': 42
        },
        'model': {
            "objective": "multi:softprob",
            "eval_metric": "mlogloss",
            "n_estimators": 100,
            "learning_rate": 0.05,
            "max_depth": 6,
            "reg_lambda": 3.0,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "random_state": 42,
            "early_stopping_rounds": 10,
        },
        'lr_scheduler': {
            "initial_lr": 0.1,
            "decay_rate": 0.20,
        },
        'training': {
            "num_batches": 2,  # nombre de lots pour l'entraînement
            "max_patience": 5,  # patience pour early stopping global
            "show_learning_curve": False,  # afficher la courbe d'apprentissage
        }
    }

    CLASS_ORDER = [
        "Underweight",
        "Normal_Weight",
        "Overweight_Level_I",
        "Overweight_Level_II",
        "Obesity_Type_I",
        "Obesity_Type_II",
        "Obesity_Type_III",
    ]

    def setUp(self):
        """Configuration initiale pour chaque test"""
        super().setUp()
        self.trainer = XGBoostTrainer() # Initialisation du trainer CatBoost
        self.train_df = None
        self.valid_df = None

    def tearDown(self):
        """Nettoyage après chaque test"""
        try:
            if hasattr(self, 'train_df') and self.train_df is not None:
                self.train_df.unpersist()
            if hasattr(self, 'valid_df') and self.valid_df is not None:
                self.valid_df.unpersist()
        except Exception as e:
            print(f"Warning: Erreur lors du nettoyage: {e}")
        super().tearDown()

    def _create_mock_data(self, config=None):
        """
        Création des données mock avec split stratifié
        """
        if config is None:
            config = self.DEFAULT_CONFIG_MULTICLASS['data']
        
        print(f"Création du DataFrame mock ({config['n_samples']} lignes)")
        df = build_mock_spark_df(
            self.spark, 
            n=config['n_samples'], 
            seed=config['seed']
        )
        
        # Split stratifié
        self.train_df, self.valid_df = stratified_split_sparkdf(
            df, 
            target_col=config['target_column'], 
            valid_size=1 - config['train_ratio'], 
            seed=config['seed']
        )
        
        # Cache pour performance
        self.train_df.cache()
        self.valid_df.cache()
        
        return self.train_df, self.valid_df


    def _encode_target_inplace(self, df, target_col: str):
        """
        Encode la cible string -> int selon CLASS_ORDER (remplace la colonne).
        """
        mapping = {lbl: i for i, lbl in enumerate(self.CLASS_ORDER)}
        # create_map exige une liste plate [k1, v1, k2, v2, ...]
        kv = []
        for k, v in mapping.items():
            kv.extend([F.lit(k), F.lit(v)])
        map_expr = F.create_map(*kv)
        return df.withColumn(target_col, map_expr[F.col(target_col)].cast("int"))

    def _create_mock_data(self, config=None):
        if config is None:
            config = self.DEFAULT_CONFIG_MULTICLASS['data']
        print(f"Création du DataFrame mock ({config['n_samples']} lignes)")
        df = build_mock_spark_df(self.spark, n=config['n_samples'], seed=config['seed'])

        # Split stratifié (sur labels string)
        self.train_df, self.valid_df = stratified_split_sparkdf(
            df,
            target_col=config['target_column'],
            valid_size=1 - config['train_ratio'],
            seed=config['seed']
        )

        # Encodage de la cible (string -> int) pour éviter les erreurs “Expected: [0..6]”
        tgt = config['target_column']
        self.train_df = self._encode_target_inplace(self.train_df, tgt).cache()
        self.valid_df = self._encode_target_inplace(self.valid_df, tgt).cache()

        # Sanity: pas de valeurs manquantes après encodage
        nulls_train = self.train_df.filter(F.col(tgt).isNull()).count()
        nulls_valid = self.valid_df.filter(F.col(tgt).isNull()).count()
        assert nulls_train == 0 and nulls_valid == 0, "Encodage labels → int a produit des NULL"
        
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
        print("\n" + "="*50)
        print("TEST: fit avec configuration par défaut")
        print("="*50)
        
        # Étape 1: Création des données
        train_df, valid_df = self._create_mock_data()
        target_column = self.DEFAULT_CONFIG_MULTICLASS['data']['target_column']
        
        # Étape 2: Affichage des infos données
        self._display_data_info(target_column)
        
        # Étape 3: Entraînement
        print("\n=== Début de l'entraînement batch ===")
        self.trainer.fit(
            train_dataframe=train_df,
            valid_dataframe=valid_df,
            target_column=target_column,
            config_training=self.DEFAULT_CONFIG_MULTICLASS['training'],
            config_model=self.DEFAULT_CONFIG_MULTICLASS['model'],
            config_lr_scheduler=self.DEFAULT_CONFIG_MULTICLASS['lr_scheduler'],
        )
        print("Entraînement batch terminé avec succès")
        
        # Étape 4: Validation
        self.assertIsNotNone(
            self.trainer,
            "Trainer est None après fit"
        )
        self.assertTrue(
            hasattr(self.trainer, "model") and self.trainer.model is not None,
            "Trainer instancié mais le modèle interne n’a pas été entraîné"
        )

if __name__ == "__main__":
    main(verbosity=2)