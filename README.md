# BatchTrainingBooster

**Framework d'entraînement incrémental par batch avec XGBoost/CatBoost et Apache Spark**

## 🎯 Objectif du Projet

BatchTrainingBooster est un framework conçu pour résoudre les défis de l'entraînement de modèles de machine learning sur de très grandes données distribuées. Il combine la puissance d'Apache Spark pour la gestion des données avec l'efficacité des algorithmes de gradient boosting (XGBoost, CatBoost) pour créer un système d'entraînement incrémental par batch.

### 🔍 Problématique Adressée

- **Données massives** : Impossibilité de charger l'ensemble des données en mémoire
- **Entraînement séquentiel** : Nécessité d'entraîner sur des sous-ensembles de données de manière cohérente
- **Optimisation continue** : Ajustement dynamique des hyperparamètres (learning rate, early stopping)
- **Équilibrage des classes** : Maintien de la distribution des classes dans chaque batch

## 🏗️ Architecture

Le framework s'articule autour de plusieurs composants clés :
```
batchtrainingbooster/
│
├── __init__.py
│
├── core/
│   ├── __init__.py               # expose OptimizedWeightCalculator
│   ├── base_trainer.py           # classe abstraite: fit(), get_trained_model(), save(), load(), etc.
│   └── weights.py                # OptimizedWeightCalculator (cache, smoothing, normalize, labels_all)
│
├── trainers/
│   ├── __init__.py
│   ├── xgboost_trainer.py        # utilise OptimizedWeightCalculator -> sample_weight par batch
│   └── catboost_trainer.py       # idem pour CatBoost (fit avec sample_weight)
│
└── logger/
    └── logger.py                 # logger centralisé (handlers, niveaux, format)
```

## ⚡ Fonctionnalités Implémentées

### 1. **Entraînement Incrémental par Batch**
- **Stratification automatique** : Chaque batch maintient la distribution des classes cibles
- **Ordonnancement aléatoire** : Randomisation avec seed pour la reproductibilité
- **Warm restart** : Continuité de l'entraînement entre les batches

### 2. **Optimisation Avancée**
- **Learning Rate Scheduling** : Décroissance exponentielle automatique
- **Early Stopping Global** : Arrêt basé sur la performance sur l'ensemble de validation
- **Équilibrage des classes** : Calcul automatique des poids d'échantillonnage

### 3. **Monitoring et Visualisation**
- **Courbes d'apprentissage** : Visualisation des métriques train/validation
- **Suivi des hyperparamètres** : Évolution du learning rate par batch
- **Logging détaillé** : Traçabilité complète du processus d'entraînement

### 4. **Support Multi-Modèles**

#### XGBoostTrainer
```python
trainer = XGBoostTrainer()
model = trainer.fit(
    train_dataframe=spark_train_df,
    valid_dataframe=spark_valid_df,
    target_column="NObeyesdad",
    config_model={
        "objective": "multi:softprob",
        "n_estimators": 100,
        "num_class": spark_valid_df.select("NObeyesdad").distinct().count(),
        "max_depth": 6,
        "eval_metric": "mlogloss"
    },
    config_training = {
        "num_batches": 2,  # nombre de lots pour l'entraînement
        "max_patience": 5,  # patience pour early stopping global
        "show_learning_curve": True,
    },
    config_lr_scheduler={
        "initial_lr": 0.1,
        "decay_rate": 0.95
    }
)