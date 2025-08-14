#!/usr/bin/env python3
"""
Script de test simple pour vérifier les imports de la nouvelle structure modulaire.
Exécutez depuis la racine du projet : python test_imports.py
"""

import sys
import os
from pathlib import Path

def test_imports():
    """Teste tous les imports de la nouvelle structure."""
    
    print("🔍 TEST DES IMPORTS - STA211 PROJECT")
    print("=" * 50)
    
    # Vérifier qu'on est dans le bon répertoire
    current_dir = Path.cwd()
    print(f"📁 Répertoire actuel: {current_dir}")
    
    # Vérifier la présence du dossier modules
    modules_dir = current_dir / 'modules'
    if not modules_dir.exists():
        print(f"❌ Dossier 'modules' non trouvé dans {current_dir}")
        print("💡 Assurez-vous d'exécuter ce script depuis la racine du projet")
        return False
    
    print(f"✅ Dossier modules trouvé: {modules_dir}")
    
    # Test des imports principaux
    print(f"\n📦 Test des imports:")
    
    modules_to_test = [
        ('modules.config', 'Configuration centralisée'),
        ('modules.utils.storage', 'Utilitaires de stockage'),  
        ('modules.evaluation.metrics', 'Module d\'évaluation'),
        ('modules.modeling.ensembles', 'Module d\'ensembles'),
        ('modules.notebook1_preprocessing', 'Prétraitement (Notebook 1)'),
        ('modules.notebook2_modeling', 'Modélisation (Notebook 2)')
    ]
    
    success_count = 0
    
    for module_name, description in modules_to_test:
        try:
            __import__(module_name)
            print(f"  ✅ {module_name} - {description}")
            success_count += 1
        except ImportError as e:
            print(f"  ❌ {module_name} - {description}: {e}")
        except Exception as e:
            print(f"  ⚠️ {module_name} - {description}: {e}")
    
    # Test des imports spécifiques
    print(f"\n🧪 Test des fonctions spécifiques:")
    
    try:
        from modules.config import cfg
        print(f"  ✅ cfg importé - Racine: {cfg.root}")
    except Exception as e:
        print(f"  ❌ cfg: {e}")
        return False
    
    try:
        from modules.utils import save_artifact, load_artifact
        print(f"  ✅ Fonctions de stockage importées")
    except Exception as e:
        print(f"  ❌ Fonctions de stockage: {e}")
    
    try:
        from modules.evaluation import calculate_basic_metrics
        print(f"  ✅ Fonctions d'évaluation importées")
    except Exception as e:
        print(f"  ❌ Fonctions d'évaluation: {e}")
    
    try:
        from modules.modeling import create_voting_ensemble
        print(f"  ✅ Fonctions d'ensemble importées")
    except Exception as e:
        print(f"  ❌ Fonctions d'ensemble: {e}")
    
    # Test rapide avec des données
    print(f"\n🧮 Test fonctionnel:")
    
    try:
        import numpy as np
        import pandas as pd
        
        # Test de sauvegarde/rechargement
        test_data = {"test": "data", "value": 42}
        from modules.utils import save_artifact, load_artifact
        
        # Sauvegarder
        saved_path = save_artifact(test_data, "test.pkl", cfg.paths.models)
        print(f"  ✅ Sauvegarde: {saved_path}")
        
        # Recharger
        loaded_data = load_artifact("test.pkl", cfg.paths.models)
        if loaded_data == test_data:
            print(f"  ✅ Rechargement: données identiques")
        else:
            print(f"  ❌ Rechargement: données différentes")
            
    except Exception as e:
        print(f"  ⚠️ Test fonctionnel échoué: {e}")
    
    # Résumé
    print(f"\n📊 RÉSUMÉ:")
    print(f"  Modules testés: {len(modules_to_test)}")
    print(f"  Succès: {success_count}")
    print(f"  Échecs: {len(modules_to_test) - success_count}")
    
    if success_count == len(modules_to_test):
        print(f"\n🎉 TOUS LES TESTS RÉUSSIS !")
        print(f"🚀 La nouvelle structure modulaire fonctionne parfaitement.")
        return True
    else:
        print(f"\n⚠️ CERTAINS TESTS ONT ÉCHOUÉ")
        print_troubleshooting()
        return False

def print_troubleshooting():
    """Affiche les instructions de dépannage."""
    print(f"\n🔧 DÉPANNAGE:")
    print(f"1. Vérifiez que vous êtes dans le bon répertoire:")
    print(f"   pwd  # doit afficher le chemin vers sta211-project")
    print(f"")
    print(f"2. Installez les dépendances manquantes:")
    print(f"   pip install -r requirements.txt")
    print(f"")
    print(f"3. Si vous utilisez un environnement virtuel, activez-le:")
    print(f"   source venv/bin/activate  # Linux/Mac")
    print(f"   venv\\Scripts\\activate     # Windows")
    print(f"")
    print(f"4. Vérifiez la structure des dossiers:")
    print(f"   ls -la modules/  # doit montrer config.py, utils/, etc.")

def check_dependencies():
    """Vérifie les dépendances Python."""
    print(f"\n📦 VÉRIFICATION DES DÉPENDANCES:")
    
    required_packages = [
        'pandas', 'numpy', 'scikit-learn', 'matplotlib', 
        'seaborn', 'xgboost', 'joblib', 'scipy'
    ]
    
    optional_packages = ['chardet', 'missingno']
    
    missing_required = []
    missing_optional = []
    
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
            print(f"  ✅ {package}")
        except ImportError:
            print(f"  ❌ {package}")
            missing_required.append(package)
    
    for package in optional_packages:
        try:
            __import__(package.replace('-', '_'))
            print(f"  ✅ {package} (optionnel)")
        except ImportError:
            print(f"  ⚠️ {package} (optionnel)")
            missing_optional.append(package)
    
    if missing_required:
        print(f"\n⚠️ Packages requis manquants: {', '.join(missing_required)}")
        print(f"💡 Installez avec: pip install {' '.join(missing_required)}")
        return False
    
    if missing_optional:
        print(f"\n💡 Packages optionnels manquants: {', '.join(missing_optional)}")
    
    return True

if __name__ == "__main__":
    print(f"Python {sys.version}")
    print(f"OS: {os.name}")
    
    # Vérifier les dépendances d'abord
    deps_ok = check_dependencies()
    
    # Puis tester les imports
    if deps_ok:
        success = test_imports()
        sys.exit(0 if success else 1)
    else:
        print(f"\n❌ Installez d'abord les dépendances manquantes")
        sys.exit(1)