# 🔍 Analyse du Notebook 02 - Modélisation et Optimisation

## ❌ **PROBLÈMES IDENTIFIÉS**

### 1. **Imports obsolètes**
```python
# ❌ Imports qui ne fonctionneront plus
from modules.config.project_config import CLASS_LABELS as LABEL_MAP
from modules.modeling import (
    optimize_threshold,
    optimize_multiple, 
    save_optimized_thresholds,
    load_best_pipelines,
    optimize_all_thresholds,
    load_optimized_thresholds,
    evaluate_all_models_on_test,
    plot_test_performance,
    plot_best_roc_curves_comparison
)
```

### 2. **Chemins de sauvegarde mixtes**
```python
# ❌ Ancien système de chemins
MODELS_DIR = cfg.paths.MODELS_DIR  # pointe vers outputs/modeling
save_dir = MODELS_DIR / "notebook2" / name

# ❌ Usage direct de joblib partout
joblib.dump(cols, save_dir / f"columns_{name}.pkl")
joblib.dump(pipeline, pipeline_path)
```

### 3. **Structure de sauvegarde incohérente**
- Sauvegarde dans `outputs/modeling/notebook2/` (ancien système)
- Devrait être dans `artifacts/models/` ou `models/` (nouveau système)

## ✅ **CORRECTIONS NÉCESSAIRES**

### 1. **Corriger les imports**

**Cellule de configuration :**
```python
# ✅ Imports corrigés
from modules.config import cfg
from modules.notebook2_modeling import (
    train_and_optimize_model,
    perform_feature_selection, 
    optimize_classification_threshold,
    evaluate_model_performance,
    train_all_models
)
from modules.utils import save_artifact, load_artifact
```

**Pour les fonctions spécifiques manquantes, ajouter :**
```python
# Import depuis l'ancien module si nécessaire
from modules.modeling import (
    optimize_all_thresholds,  # Si cette fonction existe encore
    load_optimized_thresholds
)
```

### 2. **Remplacer joblib par save_artifact/load_artifact**

**Avant :**
```python
# ❌ Ancien code
joblib.dump(pipeline, pipeline_path)
pipeline = joblib.load(pipeline_path)
```

**Après :**
```python
# ✅ Nouveau code
from modules.utils import save_artifact, load_artifact
save_artifact(pipeline, "pipeline_name.pkl", cfg.paths.models)
pipeline = load_artifact("pipeline_name.pkl", cfg.paths.models)
```

### 3. **Corriger les chemins de sauvegarde**

**Avant :**
```python
# ❌ Ancien système
MODELS_DIR = cfg.paths.MODELS_DIR  # outputs/modeling
save_dir = MODELS_DIR / "notebook2"
```

**Après :**
```python
# ✅ Nouveau système 
models_dir = cfg.paths.models  # artifacts/models ou models/
save_dir = models_dir / "notebook2"
```

## 🔧 **SCRIPT DE CORRECTION AUTOMATIQUE**

Voici un script pour corriger automatiquement le notebook :

```python
import json
from pathlib import Path

def fix_notebook02():
    notebook_path = Path("notebooks/02_Modelisation_et_Optimisation.ipynb")
    
    with open(notebook_path, 'r', encoding='utf-8') as f:
        notebook = json.load(f)
    
    replacements = {
        # Imports
        "from modules.config.project_config import CLASS_LABELS as LABEL_MAP": 
        "# CLASS_LABELS supprimé - utiliser directement les labels",
        
        "from modules.modeling import": 
        "from modules.notebook2_modeling import",
        
        "optimize_all_thresholds": "# optimize_all_thresholds - fonction à migrer",
        "load_optimized_thresholds": "# load_optimized_thresholds - fonction à migrer",
        
        # Sauvegarde 
        "joblib.dump(": "save_artifact(",
        "joblib.load(": "load_artifact(",
        
        # Chemins
        "cfg.paths.MODELS_DIR": "cfg.paths.models",
        "MODELS_DIR": "cfg.paths.models"
    }
    
    # Appliquer les corrections...
    # (code de remplacement similaire au notebook 01)
```

## 📋 **FONCTIONS MANQUANTES À IMPLÉMENTER**

Ces fonctions du notebook 02 n'existent pas dans la nouvelle structure :

### Dans `modules/notebook2_modeling.py`, ajouter :

```python
def optimize_all_thresholds(pipelines_dict, splits_dict, output_dir):
    """Optimise les seuils pour tous les modèles et datasets."""
    # À implémenter

def load_optimized_thresholds(imputation, version, thresholds_dir):
    """Charge les seuils optimisés."""  
    # À implémenter

def evaluate_all_models_on_test(pipelines, test_data, thresholds):
    """Évalue tous les modèles sur le test set."""
    # À implémenter

def plot_best_roc_curves_comparison(results):
    """Trace la comparaison des courbes ROC."""
    # À implémenter
```

## 🚨 **PROBLÈMES CRITIQUES**

1. **Le notebook 02 ne fonctionnera PAS** en l'état actuel
2. **Beaucoup de fonctions manquantes** dans la nouvelle structure
3. **Chemins de sauvegarde incohérents** avec la nouvelle architecture

## 💡 **RECOMMANDATIONS**

### Option 1: Migration complète (recommandée)
1. Implémenter les fonctions manquantes dans `notebook2_modeling.py`
2. Corriger tous les imports
3. Adopter `save_artifact/load_artifact`
4. Utiliser les nouveaux chemins

### Option 2: Compatibilité temporaire  
1. Garder les imports de `modules.modeling` pour les fonctions existantes
2. Corriger progressivement cellule par cellule
3. Avertissements de dépréciation

### Option 3: Exécution avec l'ancien système
1. Temporairement, utiliser l'ancien `modules.modeling` 
2. Planifier la migration pour plus tard

## 🎯 **ACTIONS IMMÉDIATES**

1. **Vérifier que `modules/modeling.py` contient encore les fonctions utilisées**
2. **Tester l'import** : `from modules.modeling import optimize_all_thresholds`
3. **Si ça ne marche pas, implémenter les fonctions manquantes**
4. **Utiliser save_artifact/load_artifact progressivement**

Le notebook 02 nécessite plus de travail que le 01 pour être compatible avec la nouvelle structure ! 🔧