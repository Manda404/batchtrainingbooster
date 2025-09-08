# tests/test_catboost_trainer_instantiation.py
from unittest import TestCase
from batchtrainingbooster.trainers.catboost_trainer import CatBoostTrainer


class TestInitCatBoostTrainer(TestCase):
    def setUp(self):
        try:
            self.trainer = CatBoostTrainer()
        except Exception as e:
            self.fail(f"Instanciation CatBoostTrainer a échoué dans setUp: {e}")

    def test_instantiated(self):
        """Vérifie que l’objet est bien instancié"""
        self.assertIsNotNone(self.trainer)

    def test_has_minimal_attrs(self):
        """Vérifie que l’objet possède au moins la méthode fit"""
        self.assertTrue(hasattr(self.trainer, "fit"))
        self.assertTrue(callable(getattr(self.trainer, "fit")))
