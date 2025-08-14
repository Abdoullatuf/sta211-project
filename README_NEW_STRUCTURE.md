# Projet STA211 - Nouvelle Structure Modulaire

## Vue d'ensemble

Ce projet a été restructuré pour améliorer la modularité, la réutilisabilité et la maintenabilité du code. La nouvelle architecture sépare clairement les différentes fonctionnalités en modules spécialisés.

## Structure du projet

```
sta211-project/
├── modules/                          # Modules Python du projet
│   ├── config.py                     # Configuration centralisée
│   ├── utils/                        # Utilitaires généraux
│   │   ├── __init__.py
│   │   └── storage.py               # Fonctions de sauvegarde/rechargement
│   ├── evaluation/                   # Évaluation et métriques
│   │   ├── __init__.py
│   │   └── metrics.py
│   ├── modeling/                     # Modélisation d'ensemble
│   │   ├── __init__.py
│   │   └── ensembles.py
│   ├── notebook1_preprocessing.py    # Fonctions pour EDA/Preprocessing
│   ├── notebook2_modeling.py        # Fonctions pour modélisation individuelle
│   └── [anciens modules conservés]   # Compatibilité
├── notebooks/                        # Notebooks Jupyter
│   ├── 00_Test_Migration.ipynb      # Test de la nouvelle structure
│   ├── 01_EDA_Preprocessing.ipynb
│   ├── 02_Modelisation_et_Optimisation.ipynb
│   └── 03_Stacking_et_predictions_finales.ipynb
├── data/                            # Données du projet
│   ├── raw/                         # Données brutes
│   └── processed/                   # Données traitées
├── artifacts/                       # Artefacts d'entraînement (nouveau)
│   ├── imputers/                    # Imputers sauvegardés
│   ├── transformers/                # Transformateurs
│   ├── selectors/                   # Sélecteurs de features
│   └── models/                      # Modèles entraînés
├── outputs/                         # Sorties du projet
│   ├── figures/                     # Graphiques
│   └── reports/                     # Rapports
├── MIGRATION_GUIDE.md              # Guide de migration
└── README_NEW_STRUCTURE.md         # Ce fichier
```

## Modules principaux

### 1. Configuration centralisée (`modules/config.py`)

Point central de configuration du projet :

```python
from modules.config import cfg

# Configuration du projet
print(cfg.project.project_name)    # "STA211 Ads"
print(cfg.project.random_state)    # 42
print(cfg.project.scoring)         # "f1"

# Chemins automatiquement configurés
print(cfg.paths.root)              # Racine du projet
print(cfg.paths.data)              # Répertoire des données
print(cfg.paths.models)            # Répertoire des modèles
```

**Avantages :**
- Configuration unifiée et cohérente
- Chemins automatiquement créés
- Logging centralisé
- Variables d'environnement gérées

### 2. Utilitaires de stockage (`modules/utils/storage.py`)

Fonctions standardisées pour la sauvegarde/rechargement :

```python
from modules.utils import save_artifact, load_artifact

# Sauvegarder un objet
path = save_artifact(model, "my_model.pkl", cfg.paths.models)

# Recharger un objet
model = load_artifact("my_model.pkl", cfg.paths.models)
```

**Avantages :**
- Interface unifiée pour tous les types d'objets
- Support automatique JSON/Pickle selon l'extension
- Gestion automatique des répertoires
- Logging intégré

### 3. Prétraitement (`modules/notebook1_preprocessing.py`)

Fonctions spécialisées pour l'EDA et le preprocessing :

```python
from modules.notebook1_preprocessing import (
    load_and_clean_data,
    perform_knn_imputation,
    apply_optimal_transformations,
    detect_and_cap_outliers,
    generate_polynomial_features,
    preprocess_complete_pipeline
)
```

**Fonctionnalités :**
- Chargement et nettoyage des données
- Analyse des patterns de valeurs manquantes
- Imputations KNN et MICE
- Transformations optimales (Yeo-Johnson, Box-Cox)
- Détection et traitement des outliers
- Génération de features polynomiales
- Pipeline complet intégré

### 4. Modélisation individuelle (`modules/notebook2_modeling.py`)

Fonctions pour l'entraînement et l'optimisation des modèles individuels :

```python
from modules.notebook2_modeling import (
    train_and_optimize_model,
    perform_feature_selection,
    optimize_classification_threshold,
    evaluate_model_performance,
    train_all_models
)
```

**Fonctionnalités :**
- Entraînement avec validation croisée
- Optimisation d'hyperparamètres
- Sélection de features (RFE, importance, permutation)
- Optimisation des seuils de classification
- Évaluation complète des modèles
- Pipeline d'entraînement automatisé

### 5. Modélisation d'ensemble (`modules/modeling/ensembles.py`)

Création et optimisation d'ensembles de modèles :

```python
from modules.modeling import (
    create_voting_ensemble,
    create_stacking_ensemble,
    train_all_ensembles,
    compare_models_vs_ensembles
)
```

**Fonctionnalités :**
- Ensembles Voting (hard/soft)
- Ensembles Bagging
- Ensembles Stacking avec méta-modèles
- Optimisation automatique
- Comparaison modèles individuels vs ensembles

### 6. Évaluation et métriques (`modules/evaluation/metrics.py`)

Outils complets d'évaluation :

```python
from modules.evaluation import (
    calculate_basic_metrics,
    optimize_threshold,
    plot_evaluation_dashboard,
    generate_evaluation_report
)
```

**Fonctionnalités :**
- Métriques complètes (F1, précision, rappel, AUC, etc.)
- Optimisation des seuils
- Analyse de sensibilité
- Tableaux de bord visuels
- Rapports automatisés
- Comparaisons de modèles

## Utilisation rapide

### Notebook 1 - Preprocessing

```python
from modules.config import cfg
from modules.notebook1_preprocessing import preprocess_complete_pipeline

# Pipeline complet de preprocessing
results = preprocess_complete_pipeline(
    df=raw_data,
    target_col='outcome',
    continuous_cols=['X1', 'X2', 'X3'],
    imputation_method='knn',
    transformation_mapping={'X1': 'yeo-johnson', 'X2': 'box-cox'},
    save_artifacts=True
)

df_final = results['final_dataframe']
```

### Notebook 2 - Modélisation

```python
from modules.config import cfg
from modules.notebook2_modeling import train_all_models

# Entraînement de tous les modèles
results = train_all_models(
    X_train, y_train, X_val, y_val,
    models_to_train=['randforest', 'xgboost', 'svm'],
    save_dir=cfg.paths.models
)

best_model = results['champion_model']['model']
```

### Notebook 3 - Ensembles

```python
from modules.modeling import train_all_ensembles
from modules.evaluation import compare_models_visualization

# Entraînement des ensembles
ensemble_results = train_all_ensembles(
    base_models, X_train, y_train, X_val, y_val
)

# Comparaison visuelle
compare_models_visualization(
    {'Individual': individual_results, 'Ensembles': ensemble_results}
)
```

## Migration depuis l'ancienne structure

Consultez `MIGRATION_GUIDE.md` pour :
- Correspondances détaillées des imports
- Exemples de migration par notebook
- Fonctions deprecated et leurs remplacements
- Configuration des nouveaux chemins

## Test de la nouvelle structure

Exécutez le notebook `00_Test_Migration.ipynb` pour :
- Valider tous les imports
- Tester les fonctionnalités principales
- Vérifier la compatibilité
- Diagnostiquer d'éventuels problèmes

## Avantages de la restructuration

1. **Modularité** : Code organisé en modules logiques et réutilisables
2. **Maintenabilité** : Séparation claire des responsabilités
3. **Réutilisabilité** : Fonctions facilement réutilisables entre projets
4. **Standardisation** : Approche unifiée pour la configuration et la sauvegarde
5. **Documentation** : Modules et fonctions bien documentés
6. **Testabilité** : Structure facilitant les tests unitaires
7. **Évolutivité** : Ajout facile de nouvelles fonctionnalités

## Compatibilité

- Les anciens modules sont conservés pour la compatibilité
- Migration progressive possible notebook par notebook
- Documentation complète des changements
- Tests automatisés pour valider la migration

## Support et développement

Pour toute question ou amélioration :
1. Consultez la documentation dans chaque module
2. Utilisez le notebook de test pour diagnostiquer
3. Référez-vous au guide de migration
4. Les logs détaillés facilitent le débogage

---

**Auteur :** Maoulida Abdoullatuf  
**Version :** 2.0 (Restructurée)  
**Dernière mise à jour :** 2025