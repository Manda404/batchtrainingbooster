from unittest import TestCase
from batchtrainingbooster.trainers.xgboost_trainer import XGBoostTrainer

class TestInitXGBTrainer(TestCase):
    def setUp(self):
        try:
            self.trainer = XGBoostTrainer()
        except Exception as e:
            self.fail(f"Instanciation XGBoostTrainer a échoué dans setUp: {e}")

    def test_instantiated(self):
        self.assertIsNotNone(self.trainer)

    def test_has_minimal_attrs(self):
        # ajuste selon ton implémentation
        self.assertTrue(hasattr(self.trainer, "fit"))
        self.assertTrue(callable(getattr(self.trainer, "fit")))

