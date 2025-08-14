# Cellule améliorée pour Notebook 02

## Votre cellule actuelle est très bien ! Voici juste quelques améliorations mineures :

```python
import sys, os, logging
from pathlib import Path

# ── 0. Logger clair (avec Rich si dispo)
try:
    from rich.logging import RichHandler
    logging.basicConfig(level="INFO",
                        format="%(message)s",
                        handlers=[RichHandler(rich_tracebacks=True, markup=True)],
                        force=True)
except ModuleNotFoundError:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s - %(levelname)s - %(message)s",
                        stream=sys.stdout,
                        force=True)
logger = logging.getLogger(__name__)

# ── 1. Détection environnement Colab
def _in_colab() -> bool:
    try: import google.colab
    except ImportError: return False
    else: return True

# ── 2. Montage Drive manuel rapide
if _in_colab():
    from google.colab import drive
    if not Path("/content/drive/MyDrive/Colab Notebooks").exists():
        logger.info("🔗 Montage de Google Drive en cours…")
        drive.mount("/content/drive", force_remount=False)

# ── 3. Localisation racine projet STA211
def find_project_root() -> Path:
    env_path = os.getenv("STA211_PROJECT_PATH")
    if env_path and (Path(env_path) / "modules").exists():
        return Path(env_path).expanduser().resolve()

    # Chemin Colab correct
    default_colab = Path("/content/drive/MyDrive/Colab Notebooks/projet_sta211_2025")
    if _in_colab() and (default_colab / "modules").exists():
        return default_colab.resolve()

    # Local - chercher depuis le répertoire courant
    cwd = Path.cwd()
    
    # Si on est dans notebooks/, remonter au parent
    if cwd.name == "notebooks":
        parent = cwd.parent
        if (parent / "modules").exists():
            return parent.resolve()
    
    # Chercher dans la hiérarchie
    for p in [cwd, *cwd.parents]:
        if (p / "modules").exists():
            return p.resolve()

    raise FileNotFoundError("❌ Impossible de localiser un dossier contenant 'modules/'.")

# ── 4. Définition racine + PYTHONPATH
try:
    ROOT_DIR = find_project_root()
    os.environ["STA211_PROJECT_PATH"] = str(ROOT_DIR)
    if str(ROOT_DIR) not in sys.path:
        sys.path.insert(0, str(ROOT_DIR))
    logger.info(f"📂 Racine projet détectée : {ROOT_DIR}")
    logger.info(f"PYTHONPATH ← {ROOT_DIR}")
except FileNotFoundError as e:
    logger.error(f"❌ {e}")
    logger.info("💡 Assurez-vous que le dossier 'modules/' existe dans votre projet")
    raise

# ── 5. Initialisation de la configuration projet
try:
    from modules.config import cfg
    logger.info("✅ Configuration importée avec succès")
except ImportError as e:
    logger.error(f"❌ Erreur d'import de la configuration: {e}")
    logger.info(f"📁 Vérifiez que {ROOT_DIR / 'modules' / 'config.py'} existe")
    raise

# ✅ CORRECTION : Redéfinir les chemins avec la racine correcte
# NOTE: cfg utilise maintenant une structure différente (cfg.paths.root, cfg.paths.models, etc.)
# Mais on maintient la compatibilité avec l'ancienne structure

# Vérifier si cfg utilise la nouvelle structure
if hasattr(cfg.paths, 'root'):
    # Nouvelle structure
    logger.info("📋 Utilisation de la nouvelle structure cfg.paths")
    ROOT_DIR_CFG = ROOT_DIR
    RAW_DATA_DIR = cfg.paths.raw if hasattr(cfg.paths, 'raw') else ROOT_DIR / "data" / "raw"
    DATA_PROCESSED = cfg.paths.processed if hasattr(cfg.paths, 'processed') else ROOT_DIR / "data" / "processed"
    OUTPUTS_DIR = cfg.paths.outputs if hasattr(cfg.paths, 'outputs') else ROOT_DIR / "outputs"
    FIGURES_NB2_DIR = OUTPUTS_DIR / "figures" / "notebook2"
    THRESHOLDS_DIR = OUTPUTS_DIR / "modeling" / "thresholds"
    MODELS_DIR = cfg.paths.models if hasattr(cfg.paths, 'models') else ROOT_DIR / "outputs" / "modeling"
else:
    # Ancienne structure - redéfinir les chemins
    logger.info("📋 Configuration des chemins (structure de compatibilité)")
    cfg.paths.ROOT_DIR = ROOT_DIR
    cfg.paths.MODULE_DIR = ROOT_DIR / "modules"
    cfg.paths.RAW_DATA_DIR = ROOT_DIR / "data" / "raw"
    cfg.paths.DATA_PROCESSED = ROOT_DIR / "data" / "processed"
    cfg.paths.OUTPUTS_DIR = ROOT_DIR / "outputs"
    cfg.paths.FIGURES_NB2_DIR = ROOT_DIR / "outputs" / "figures" / "notebook2"
    cfg.paths.THRESHOLDS_DIR = ROOT_DIR / "outputs" / "modeling" / "thresholds"
    cfg.paths.MODELS_DIR = ROOT_DIR / "outputs" / "modeling"
    
    # Variables pour le notebook
    RAW_DATA_DIR = cfg.paths.RAW_DATA_DIR
    DATA_PROCESSED = cfg.paths.DATA_PROCESSED
    OUTPUTS_DIR = cfg.paths.OUTPUTS_DIR
    FIGURES_NB2_DIR = cfg.paths.FIGURES_NB2_DIR
    THRESHOLDS_DIR = cfg.paths.THRESHOLDS_DIR
    MODELS_DIR = cfg.paths.MODELS_DIR

# ── 6. Affichage des chemins utiles
def display_paths(style: bool = True):
    import pandas as pd
    paths_dict = {
        "ROOT_DIR": ROOT_DIR,
        "RAW_DATA_DIR": RAW_DATA_DIR,
        "DATA_PROCESSED": DATA_PROCESSED,
        "OUTPUTS_DIR": OUTPUTS_DIR,
        "FIGURES_NB2_DIR": FIGURES_NB2_DIR,
        "THRESHOLDS_DIR": THRESHOLDS_DIR,
        "MODELS_DIR": MODELS_DIR
    }
    rows = [{"Clé": k, "Chemin": str(v)} for k, v in paths_dict.items()]
    df = pd.DataFrame(rows).set_index("Clé")
    
    # Vérification de l'existence des dossiers
    df["Existe"] = [
        "✅" if Path(v).exists() else "❌"
        for v in paths_dict.values()
    ]
    
    from IPython.display import display
    display(df.style.set_table_styles([
        {"selector": "th", "props": [("text-align", "left")]},
        {"selector": "td", "props": [("text-align", "left")]},
    ]) if style else df)

# ── 7. Correction pour CLASS_LABELS (problème identifié)
# Définir manuellement car modules.config.project_config n'existe plus
CLASS_LABELS = {0: "Non-Ad", 1: "Ad"}
LABEL_MAP = CLASS_LABELS  # Alias pour compatibilité

# ── 8. Import des utilitaires de la nouvelle structure (si disponibles)
try:
    from modules.utils import save_artifact, load_artifact
    logger.info("✅ Utilitaires de stockage importés (nouvelle structure)")
    USING_NEW_STORAGE = True
except ImportError:
    logger.info("⚠️ Ancienne structure de stockage (joblib)")
    USING_NEW_STORAGE = False

display_paths()

logger.info("✅ Initialisation complète réussie - Notebook 02 prêt !")
logger.info(f"🔧 Structure de stockage: {'Nouvelle (save_artifact)' if USING_NEW_STORAGE else 'Ancienne (joblib)'}")
```

## 🔍 **Principales améliorations :**

### 1. **Gestion robuste des erreurs**
- Try/catch pour les imports
- Messages d'erreur plus clairs

### 2. **Détection automatique de la structure**
- Compatible avec ancienne ET nouvelle structure cfg
- Adaptation automatique des chemins

### 3. **Correction CLASS_LABELS**
- Définition manuelle pour éviter l'erreur d'import
- Alias LABEL_MAP pour compatibilité

### 4. **Vérification des dossiers**
- Colonne "Existe" dans l'affichage des chemins
- Diagnostic plus facile

### 5. **Import conditionnel des nouveaux utilitaires**
- Détection automatique save_artifact/load_artifact
- Fallback vers joblib si nécessaire

## 💡 **Recommandation :**

**Votre cellule actuelle devrait déjà fonctionner !** Ces améliorations sont optionnelles mais rendent le code plus robuste.

La correction la plus importante était déjà présente dans votre code : la redéfinition des chemins cfg.paths avec ROOT_DIR.