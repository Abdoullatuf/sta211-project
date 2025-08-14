# config.py
"""
Configuration centrale du projet STA211

Ce module unifie :
- Configuration de l'environnement (affichage, warnings, random seed)
- Gestion des chemins du projet
- Configuration du projet (modélisation, métadonnées)

Utilisation :
    from modules.config import cfg
"""

import os
import sys
import json
import time
import logging
import warnings
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Tuple, Union

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.exceptions import ConvergenceWarning, FitFailedWarning

# ============================================================================
# CONSTANTES GLOBALES
# ============================================================================

RANDOM_STATE: int = 42
PROJECT_NAME: str = "STA211 Ads"

# ============================================================================
# CONFIGURATION DE BASE
# ============================================================================

@dataclass
class ProjectConfig:
    """Configuration centrale du projet"""
    project_name: str = PROJECT_NAME
    author: str = "Maoulida Abdoullatuf"
    version: str = "2.0"
    random_state: int = RANDOM_STATE
    test_size: float = 0.2
    scoring: str = "f1"
    cv: int = 5

    def to_dict(self) -> dict:
        return asdict(self)

    def save_json(self, path: Union[str, Path]) -> None:
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=4)


@dataclass
class PathsConfig:
    """Configuration des chemins du projet"""
    
    def __init__(self, root: Path):
        self.root = root
        
        # Répertoires de données
        self.data = self.root / "data"
        self.raw = self.data / "raw"
        self.processed = self.data / "processed"
        
        # Répertoires de sortie
        self.outputs = self.root / "outputs"
        self.figures = self.outputs / "figures"
        
        # Répertoires de sauvegarde des artefacts
        self.artifacts = self.root / "artifacts"
        self.imputers = self.artifacts / "imputers"
        self.transformers = self.artifacts / "transformers"
        self.selectors = self.artifacts / "selectors"
        self.models = self.artifacts / "models"
        
        # Créer les répertoires nécessaires
        self._create_directories()
    
    def _create_directories(self):
        """Crée les répertoires nécessaires"""
        for path_attr in ['outputs', 'figures', 'artifacts', 'imputers', 
                         'transformers', 'selectors', 'models', 'processed']:
            path = getattr(self, path_attr)
            path.mkdir(parents=True, exist_ok=True)

    def to_dict(self) -> dict:
        return {k: str(v) for k, v in self.__dict__.items() if isinstance(v, Path)}


class Config:
    """Classe de configuration principale"""
    
    def __init__(self):
        self.project = ProjectConfig()
        
        # Détection de la racine du projet
        self.root = self._find_project_root()
        self.paths = PathsConfig(self.root)
        
        # Configuration du logging
        self._setup_logging()
        self._setup_environment()
    
    def _find_project_root(self, marker_file: str = '.git') -> Path:
        """Trouve la racine du projet en cherchant un marqueur."""
        current_dir = Path.cwd()
        for parent in [current_dir] + list(current_dir.parents):
            if (parent / marker_file).exists() or (parent / '.project_root').exists():
                return parent
        return current_dir
    
    def _setup_logging(self):
        """Configure le logging du projet"""
        # Configuration basique du logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
    
    def get_logger(self, name: str) -> logging.Logger:
        """Retourne un logger configuré pour un module"""
        return logging.getLogger(name)
    
    def _setup_environment(self):
        """Configure l'environnement (warnings, etc.)"""
        # Configuration des warnings
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        warnings.filterwarnings("ignore", category=FitFailedWarning)
        warnings.filterwarnings("ignore", category=UserWarning, module='sklearn')
        
        # Configuration de numpy et pandas
        np.random.seed(self.project.random_state)
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', None)
        pd.set_option('display.max_colwidth', 50)
        
        # Configuration de matplotlib
        plt.style.use('seaborn-v0_8')
        plt.rcParams['figure.figsize'] = (12, 8)
        plt.rcParams['font.size'] = 11
    
    def to_dict(self) -> dict:
        return {
            "project": self.project.to_dict(),
            "paths": self.paths.to_dict()
        }
    
    def save_config(self, filename: str = "project_config.json"):
        """Sauvegarde la configuration dans un fichier JSON"""
        config_path = self.paths.outputs / filename
        with open(config_path, 'w') as f:
            json.dump(self.to_dict(), f, indent=4)
        self.logger.info(f"Configuration sauvegardée : {config_path}")


# ============================================================================
# INSTANCIATION GLOBALE
# ============================================================================

# Instance globale de configuration
cfg = Config()

# Message de confirmation
cfg.logger.info("✅ Configuration chargée depuis config.py")
cfg.logger.info(f"📁 Racine du projet: {cfg.root}")