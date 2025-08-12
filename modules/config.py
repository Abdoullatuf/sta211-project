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
DEFAULT_PATHS = {
    "ROOT_DIR": None,
    "MODULE_DIR": "modules",
    "RAW_DATA_DIR": "data/raw",
    "DATA_PROCESSED": "data/processed",
    "OUTPUTS_DIR": "outputs",
    "FIGURES_DIR": "outputs/figures",
    "THRESHOLDS_DIR": "outputs/modeling/thresholds"
}

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
    model_dir: Optional[Path] = None # Accept model_dir here

    def to_dict(self) -> dict:
        return asdict(self)

    def save_json(self, path: Union[str, Path]) -> None:
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=4)

@dataclass
class PathsConfig:
    """Configuration des chemins du projet"""
    ROOT_DIR: Path
    MODULE_DIR: Path
    RAW_DATA_DIR: Path
    DATA_PROCESSED: Path
    OUTPUTS_DIR: Path
    FIGURES_DIR: Path
    THRESHOLDS_DIR: Path
    MODELS_DIR: Path # Add MODELS_DIR here

    def to_dict(self) -> dict:
        return {k: str(v) for k, v in asdict(self).items()}

    def save_json(self, path: Union[str, Path]) -> None:
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=4)

@dataclass
class Config:
    """Classe de configuration principale"""
    project: ProjectConfig
    paths: PathsConfig
    logger: logging.Logger

    def to_dict(self) -> dict:
        return {
            "project": self.project.to_dict(),
            "paths": self.paths.to_dict()
        }

# ============================================================================
# INITIALISATION DES CHEMINS
# ============================================================================

def _find_project_root(marker_file: str = '.git') -> Path:
    """Tente de trouver la racine du projet en cherchant un marqueur."""
    current_dir = Path.cwd()
    for parent in [current_dir] + list(current_dir.parents):
        if (parent / marker_file).exists() or (parent / '.project_root').exists():
            return parent
    # Fallback if no marker found
    return current_dir

# Get the project_path from the global scope if available (from ProjectMetadata)
PROJECT_ROOT = globals().get('project_path', _find_project_root())

# ============================================================================
# MISE EN PLACE DES CHEMINS
# ============================================================================

# Use the PROJECT_ROOT from the global scope or the fallback
ROOT_DIR = PROJECT_ROOT
MODULE_DIR = ROOT_DIR / DEFAULT_PATHS["MODULE_DIR"]
RAW_DATA_DIR = ROOT_DIR / DEFAULT_PATHS["RAW_DATA_DIR"]
DATA_PROCESSED = ROOT_DIR / DEFAULT_PATHS["DATA_PROCESSED"]
OUTPUTS_DIR = ROOT_DIR / DEFAULT_PATHS["OUTPUTS_DIR"]
FIGURES_DIR = ROOT_DIR / DEFAULT_PATHS["FIGURES_DIR"]
THRESHOLDS_DIR = ROOT_DIR / DEFAULT_PATHS["THRESHOLDS_DIR"]
MODELS_DIR = ROOT_DIR / "models" # Define MODELS_DIR here

# Ensure output directories exist
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
THRESHOLDS_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================================
# CONFIGURATION DU LOGGER
# ============================================================================

# Basic logger setup (can be expanded)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ============================================================================
# INSTANCIATION DE LA CONFIGURATION GLOBALE
# ============================================================================

# Instantiate the Config and assign it to cfg so it can be imported
cfg = Config(
    project=ProjectConfig(model_dir=MODELS_DIR), # Pass MODELS_DIR to ProjectConfig
    paths=PathsConfig(
        ROOT_DIR=ROOT_DIR,
        MODULE_DIR=MODULE_DIR,
        RAW_DATA_DIR=RAW_DATA_DIR,
        DATA_PROCESSED=DATA_PROCESSED,
        OUTPUTS_DIR=OUTPUTS_DIR,
        FIGURES_DIR=FIGURES_DIR,
        THRESHOLDS_DIR=THRESHOLDS_DIR,
        MODELS_DIR=MODELS_DIR # Pass MODELS_DIR to PathsConfig
    ),
    logger=logger
)

# Optional: Add a simple print to confirm config is loaded
print("✅ Configuration chargée depuis config.py")

# ============================================================================
# CONFIGURATION DES WARNINGS
# ============================================================================

warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=FitFailedWarning)
warnings.filterwarnings("ignore", category=UserWarning, module='sklearn') # Ignore general sklearn warnings