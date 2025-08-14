# Guide de Migration - Nouvelle Structure Modulaire

Ce document détaille les changements d'imports suite à la refactorisation de la structure modulaire du projet STA211.

## Résumé des changements

### Nouvelle structure des modules

```
modules/
├── config.py                     # Configuration centralisée
├── utils/
│   ├── __init__.py
│   └── storage.py                # Fonctions save_artifact/load_artifact
├── evaluation/
│   ├── __init__.py
│   └── metrics.py                # Évaluation et métriques
├── modeling/
│   ├── __init__.py
│   └── ensembles.py             # Modèles d'ensemble
├── notebook1_preprocessing.py    # Fonctions pour Notebook 1
└── notebook2_modeling.py        # Fonctions pour Notebook 2
```

## Guide de migration des imports

### 1. Configuration

**Avant :**
```python
from modules.config import cfg
from modules.config.project_config import CLASS_LABELS as LABEL_MAP
```

**Après :**
```python
from modules.config import cfg
# CLASS_LABELS maintenant dans cfg.project si nécessaire
```

### 2. Fonctions de sauvegarde/rechargement

**Avant :**
```python
import joblib
joblib.dump(obj, path)
obj = joblib.load(path)
```

**Après :**
```python
from modules.utils import save_artifact, load_artifact
save_artifact(obj, filename, cfg.paths.models)
obj = load_artifact(filename, cfg.paths.models)
```

### 3. Prétraitement (Notebook 1)

**Avant :**
```python
from modules.preprocessing.transformation_optimale_mixte import appliquer_transformation_optimale
```

**Après :**
```python
from modules.notebook1_preprocessing import apply_optimal_transformations
```

**Fonctions disponibles dans `notebook1_preprocessing` :**
- `load_and_clean_data()`
- `analyze_missing_patterns()`
- `perform_knn_imputation()`
- `perform_mice_imputation()`
- `apply_optimal_transformations()`
- `detect_and_cap_outliers()`
- `generate_polynomial_features()`
- `preprocess_complete_pipeline()`

### 4. Modélisation individuelle (Notebook 2)

**Avant :**
```python
from modules.modeling import (
    train_and_optimize_model,
    optimize_all_thresholds,
    load_optimized_thresholds
)
```

**Après :**
```python
from modules.notebook2_modeling import (
    train_and_optimize_model,
    optimize_classification_threshold,
    evaluate_model_performance,
    train_all_models
)
```

**Fonctions disponibles dans `notebook2_modeling` :**
- `get_default_param_grids()`
- `create_model_estimators()`
- `train_and_optimize_model()`
- `perform_feature_selection()`
- `optimize_classification_threshold()`
- `evaluate_model_performance()`
- `train_all_models()`

### 5. Modélisation d'ensemble (Notebook 3)

**Avant :**
```python
from modules.modeling import (
    run_stacking_with_refit,
    run_stacking_no_refit
)
```

**Après :**
```python
from modules.modeling import (
    create_voting_ensemble,
    create_stacking_ensemble,
    train_all_ensembles,
    compare_models_vs_ensembles
)
```

**Fonctions disponibles dans `modeling.ensembles` :**
- `create_voting_ensemble()`
- `create_bagging_ensemble()`
- `create_stacking_ensemble()`
- `optimize_ensemble()`
- `train_all_ensembles()`
- `compare_models_vs_ensembles()`
- `load_ensemble()`
- `get_ensemble_feature_importance()`

### 6. Évaluation et métriques

**Nouveau module :**
```python
from modules.evaluation import (
    calculate_basic_metrics,
    optimize_threshold,
    plot_evaluation_dashboard,
    generate_evaluation_report
)
```

**Fonctions disponibles dans `evaluation.metrics` :**
- `calculate_basic_metrics()`
- `calculate_detailed_metrics()`
- `optimize_threshold()`
- `analyze_threshold_sensitivity()`
- `plot_evaluation_dashboard()`
- `compare_models_visualization()`
- `generate_evaluation_report()`
- `export_metrics_to_csv()`

## Exemple de migration complète

### Notebook 1 - EDA & Preprocessing

```python
# Configuration
from modules.config import cfg

# Prétraitement
from modules.notebook1_preprocessing import (
    load_and_clean_data,
    analyze_missing_patterns,
    perform_knn_imputation,
    apply_optimal_transformations,
    detect_and_cap_outliers,
    preprocess_complete_pipeline
)

# Utilitaires
from modules.utils import save_artifact, load_artifact
```

### Notebook 2 - Modélisation

```python
# Configuration
from modules.config import cfg

# Modélisation individuelle
from modules.notebook2_modeling import (
    train_and_optimize_model,
    perform_feature_selection,
    train_all_models
)

# Évaluation
from modules.evaluation import (
    optimize_threshold,
    plot_evaluation_dashboard,
    generate_evaluation_report
)

# Utilitaires
from modules.utils import save_artifact, load_artifact
```

### Notebook 3 - Ensembles

```python
# Configuration
from modules.config import cfg

# Ensembles
from modules.modeling import (
    create_voting_ensemble,
    create_stacking_ensemble,
    train_all_ensembles,
    compare_models_vs_ensembles
)

# Évaluation
from modules.evaluation import (
    calculate_basic_metrics,
    plot_evaluation_dashboard,
    compare_models_visualization
)

# Utilitaires
from modules.utils import save_artifact, load_artifact
```

## Avantages de la nouvelle structure

1. **Organisation claire** : Séparation logique des fonctionnalités
2. **Réutilisabilité** : Modules indépendants et réutilisables
3. **Maintenance** : Code plus facile à maintenir et à tester
4. **Standardisation** : Approche uniforme pour la sauvegarde/rechargement
5. **Configuration centralisée** : Un seul point de configuration

## Configuration des chemins

La nouvelle configuration gère automatiquement les chemins :

```python
from modules.config import cfg

# Chemins disponibles
cfg.paths.root        # Racine du projet
cfg.paths.data        # Données
cfg.paths.raw         # Données brutes
cfg.paths.processed   # Données traitées
cfg.paths.outputs     # Sorties
cfg.paths.figures     # Graphiques
cfg.paths.artifacts   # Artefacts (sous-répertoires ci-dessous)
cfg.paths.imputers    # Imputers
cfg.paths.transformers # Transformateurs
cfg.paths.selectors   # Sélecteurs de features
cfg.paths.models      # Modèles
```

## Notes importantes

1. Les anciens modules (`data_processing.py`, `modeling.py`, `utils.py`) sont conservés pour compatibilité
2. Les nouveaux modules utilisent la configuration centralisée (`cfg`)
3. La sauvegarde/rechargement est standardisée via `save_artifact`/`load_artifact`
4. Tous les répertoires sont créés automatiquement au chargement de la configuration