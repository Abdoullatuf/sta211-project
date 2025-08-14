# Fix pour le Notebook 01 - EDA & Preprocessing

## Problème identifié

Le notebook `01_EDA_Preprocessing.ipynb` utilise encore l'ancien import :

```python
from modules.preprocessing.transformation_optimale_mixte import appliquer_transformation_optimale
```

## Solutions proposées

### Solution 1 : Import corrigé (recommandé)

Remplacez l'import dans le notebook par :

```python
# Ancien import (à supprimer)
# from modules.preprocessing.transformation_optimale_mixte import appliquer_transformation_optimale

# Nouvel import
from modules.notebook1_preprocessing import apply_optimal_transformations
```

Et remplacez tous les appels de fonction :

```python
# Ancien appel
# df_transformed = appliquer_transformation_optimale(df)

# Nouveau appel
df_transformed, transformers = apply_optimal_transformations(
    df=df,
    continuous_cols=['X1', 'X2', 'X3'],  # spécifiez vos colonnes
    method_mapping={'X1': 'yeo-johnson', 'X2': 'box-cox', 'X3': 'yeo-johnson'}
)
```

### Solution 2 : Fonction de compatibilité (temporaire)

Si vous voulez garder temporairement l'ancien code, utilisez :

```python
from modules.notebook1_preprocessing import appliquer_transformation_optimale
```

⚠️ **Note** : Cette fonction affichera un avertissement de dépréciation et sera supprimée dans une version future.

## Instructions étape par étape

### Étape 1 : Ajouter la configuration des chemins

Au début du notebook, dans la cellule d'imports, ajoutez :

```python
import sys
from pathlib import Path

# Configuration des chemins pour les imports
project_root = Path.cwd().parent if Path.cwd().name == 'notebooks' else Path.cwd()
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Import de la configuration
from modules.config import cfg
```

### Étape 2 : Remplacer les anciens imports

Cherchez cette ligne :
```python
from modules.preprocessing.transformation_optimale_mixte import appliquer_transformation_optimale
```

Et remplacez-la par :
```python
from modules.notebook1_preprocessing import apply_optimal_transformations
```

### Étape 3 : Adapter les appels de fonction

L'ancienne fonction :
```python
df_transformed = appliquer_transformation_optimale(df)
```

Devient :
```python
# Définir les colonnes continues et les méthodes
continuous_cols = ['X1', 'X2', 'X3']  # adaptez selon vos données
method_mapping = {
    'X1': 'yeo-johnson',
    'X2': 'yeo-johnson', 
    'X3': 'yeo-johnson'
}

# Appliquer les transformations
df_transformed, transformers = apply_optimal_transformations(
    df=df,
    continuous_cols=continuous_cols,
    method_mapping=method_mapping,
    save_transformers=True
)
```

### Étape 4 : Utiliser les nouveaux chemins de sauvegarde

Si vous sauvegardez des objets, utilisez maintenant :

```python
from modules.utils import save_artifact, load_artifact

# Sauvegarder
save_artifact(transformers, "my_transformers.pkl", cfg.paths.transformers)

# Recharger
transformers = load_artifact("my_transformers.pkl", cfg.paths.transformers)
```

## Migration complète recommandée

Pour une migration complète vers la nouvelle structure, nous recommandons de :

1. **Utiliser la nouvelle fonction** `apply_optimal_transformations`
2. **Adopter la nouvelle approche de sauvegarde** avec `save_artifact/load_artifact`  
3. **Utiliser la configuration centralisée** via `cfg.paths`
4. **Suivre les nouvelles conventions** documentées dans `MIGRATION_GUIDE.md`

## Support

Si vous rencontrez des difficultés :

1. Consultez le `MIGRATION_GUIDE.md` pour tous les détails
2. Exécutez `notebooks/00_Test_Migration.ipynb` pour valider votre environnement
3. Utilisez le script `test_imports.py` pour diagnostiquer les problèmes d'import

## Exemple complet

Voici un exemple complet d'utilisation dans le notebook :

```python
# Configuration des imports
import sys
from pathlib import Path
project_root = Path.cwd().parent if Path.cwd().name == 'notebooks' else Path.cwd()
sys.path.insert(0, str(project_root))

# Imports de la nouvelle structure
from modules.config import cfg
from modules.notebook1_preprocessing import (
    load_and_clean_data,
    apply_optimal_transformations,
    detect_and_cap_outliers
)
from modules.utils import save_artifact, load_artifact

# Chargement des données
df, info = load_and_clean_data('path/to/your/data.csv')

# Transformation optimale
continuous_cols = ['X1', 'X2', 'X3']
method_mapping = {'X1': 'yeo-johnson', 'X2': 'box-cox', 'X3': 'yeo-johnson'}

df_transformed, transformers = apply_optimal_transformations(
    df=df,
    continuous_cols=continuous_cols,
    method_mapping=method_mapping
)

# Sauvegarde
save_artifact(transformers, "transformers.pkl", cfg.paths.transformers)
```

Cette approche garantit la compatibilité avec la nouvelle structure et vous bénéficiez de toutes les améliorations apportées.