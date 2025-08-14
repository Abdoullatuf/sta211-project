# Fix pour la cellule d'import du Notebook 01

## Problème identifié

La variable `base_path` pointe vers `c:\sta211-project\notebooks` au lieu de `c:\sta211-project`, ce qui fait que l'import `from modules.config import cfg` échoue.

## Solution

Remplacez le contenu de la cellule problématique par ce code corrigé :

```python
# ============================================================================
# 2.1 PARAMÈTRES ET IMPORTS CORRIGÉS
# ============================================================================

import logging
import sys
from pathlib import Path

# --- 1. CORRECTION ET VÉRIFICATION DU BASE_PATH ---

if 'base_path' not in locals():
    raise NameError("La variable 'base_path' n'a pas été définie. Exécutez la première cellule du notebook.")

# CORRECTION : Si base_path pointe vers notebooks/, remonter au niveau parent
if base_path.name == 'notebooks':
    base_path = base_path.parent
    print(f"📍 Base path corrigé vers la racine du projet: {base_path}")
else:
    print(f"📍 Base path détecté: {base_path}")

# Vérifier que le dossier modules existe
modules_path = base_path / "modules"
if not modules_path.exists():
    raise FileNotFoundError(f"❌ Dossier modules non trouvé: {modules_path}")

print(f"✅ Dossier modules trouvé: {modules_path}")

# --- 2. AJOUT AU PYTHON PATH ---

# Ajouter la racine du projet au sys.path (pas le dossier modules!)
if str(base_path) not in sys.path:
    sys.path.insert(0, str(base_path))
    print(f"✅ Ajouté au sys.path: {base_path}")

# --- 3. IMPORT DE LA CONFIGURATION ---

try:
    from modules.config import cfg
    print("✅ Configuration importée avec succès")
except ImportError as e:
    print(f"❌ Erreur d'import: {e}")
    print(f"Contenu du dossier modules: {list(modules_path.iterdir())}")
    raise

print("\n✅ Objet 'cfg' importé avec succès.")

# --- 4. UTILISATION DES CHEMINS DE LA CONFIGURATION ---

# Utiliser les chemins déjà configurés dans cfg
DATA_DIR = cfg.paths.raw
OUTPUTS_DIR = cfg.paths.outputs  
DATA_PROCESSED = cfg.paths.processed
MODELS_DIR = cfg.paths.models

print("\n📁 Chemins configurés automatiquement :")
print(f" • Racine projet : {cfg.paths.root}")
print(f" • Données brutes : {DATA_DIR}")
print(f" • Données traitées : {DATA_PROCESSED}")
print(f" • Sorties : {OUTPUTS_DIR}")
print(f" • Modèles : {MODELS_DIR}")

# Vérification de l'existence des dossiers principaux
print(f"\n🔍 Vérification des dossiers :")
for name, path in [("Données brutes", DATA_DIR), ("Sorties", OUTPUTS_DIR)]:
    exists = "✅" if path.exists() else "❌"
    print(f" {exists} {name}: {path}")

# --- 5. CONFIGURATION DU LOGGER ---
logger = cfg.get_logger(__name__)
logger.info("Configuration de la cellule 2.1 terminée avec succès.")
print("\n🎉 Configuration terminée avec succès !")
```

## Pourquoi cette correction fonctionne

1. **Détection automatique** : Le code détecte si `base_path` pointe vers le dossier `notebooks/` et le corrige automatiquement vers la racine du projet.

2. **Vérification du dossier modules** : On s'assure que le dossier `modules/` existe avant d'essayer d'importer.

3. **sys.path correct** : On ajoute la racine du projet au `sys.path`, pas le dossier `modules/`.

4. **Utilisation de la configuration centralisée** : On utilise `cfg.paths` au lieu de redéfinir les chemins manuellement.

5. **Diagnostics intégrés** : Le code affiche des informations utiles pour déboguer en cas de problème.

## Alternative simple

Si vous voulez juste une correction minimale, remplacez ces lignes :

```python
# Ancien code problématique
sys.path.insert(0, str(base_path/ "modules"))

# Nouveau code corrigé  
if base_path.name == 'notebooks':
    base_path = base_path.parent
sys.path.insert(0, str(base_path))
```

Cette correction devrait résoudre immédiatement votre problème d'import !