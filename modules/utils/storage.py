"""
Utilitaires centralisés pour la sauvegarde/rechargement des artefacts.

Permet de sauvegarder et recharger des objets (pickle, JSON) de manière
standardisée à travers tout le projet.
"""

from pathlib import Path
import joblib
import json
import logging
from modules.config import cfg

log = logging.getLogger(__name__)


def save_artifact(obj, filename: str, subdir: Path, verbose: bool = True) -> Path:
    """
    Sauvegarde un objet (pickle ou JSON) dans un sous-dossier du projet.
    
    Args:
        obj: Objet à sauvegarder
        filename: Nom du fichier avec extension
        subdir: Sous-répertoire (doit provenir de cfg.paths)
        verbose: Si True, affiche un message de confirmation
        
    Returns:
        Path: Chemin complet du fichier sauvegardé
    """
    path = subdir / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    
    # Choisir le format selon l'extension
    if path.suffix.lower() == ".json":
        with open(path, "w") as f:
            json.dump(obj, f, indent=2)
    else:
        joblib.dump(obj, path)
    
    if verbose:
        log.info(f"✅ Sauvegarde de {path}")
    return path


def load_artifact(filename: str, subdir: Path):
    """
    Recharge un objet (pickle ou JSON) depuis un sous-dossier du projet.
    
    Args:
        filename: Nom du fichier
        subdir: Sous-répertoire (doit provenir de cfg.paths)
        
    Returns:
        L'objet rechargé
        
    Raises:
        FileNotFoundError: Si le fichier n'existe pas
    """
    path = subdir / filename
    if not path.exists():
        raise FileNotFoundError(f"❌ Fichier introuvable : {path}")
    
    if path.suffix.lower() == ".json":
        with open(path, "r") as f:
            return json.load(f)
    
    return joblib.load(path)


def list_artifacts(subdir: Path, pattern: str = "*") -> list[Path]:
    """
    Liste les artefacts dans un répertoire.
    
    Args:
        subdir: Sous-répertoire à scanner
        pattern: Pattern de fichiers (ex: "*.pkl", "*.json")
        
    Returns:
        Liste des chemins trouvés
    """
    if not subdir.exists():
        return []
    
    return list(subdir.glob(pattern))


def artifact_exists(filename: str, subdir: Path) -> bool:
    """
    Vérifie si un artefact existe.
    
    Args:
        filename: Nom du fichier
        subdir: Sous-répertoire
        
    Returns:
        True si le fichier existe
    """
    return (subdir / filename).exists()


def get_artifact_info(filename: str, subdir: Path) -> dict:
    """
    Retourne des informations sur un artefact.
    
    Args:
        filename: Nom du fichier
        subdir: Sous-répertoire
        
    Returns:
        Dict avec les infos (taille, date de modification, etc.)
    """
    path = subdir / filename
    if not path.exists():
        return {}
    
    stat = path.stat()
    return {
        'path': str(path),
        'size_bytes': stat.st_size,
        'size_mb': stat.st_size / (1024 * 1024),
        'modified': stat.st_mtime,
        'exists': True
    }