#!/usr/bin/env python3
"""
Test rapide de la correction des erreurs de stacking
"""

import sys
import os
sys.path.append('/home/aellatuf/sta211-project')

# Importer le module de configuration d'abord
from modules.config import cfg

# Tester l'importation et la fonction de prédiction
try:
    from modules.utils import generate_final_predictions
    print("✅ Import réussi de generate_final_predictions")
    
    # Tester la génération de prédictions avec stacking
    print("🔄 Test de génération de prédictions avec stacking...")
    stacking_submission = generate_final_predictions(
        use_stacking=True,
        auto_select=True,
        stacking_dir="artifacts/models/notebook3/stacking",
        base_dir=str(cfg.paths.root)
    )
    
    print("✅ Prédictions générées avec succès !")
    print(f"📊 Forme des prédictions : {stacking_submission.shape}")
    print(f"📈 Distribution des classes : \n{stacking_submission['outcome'].value_counts()}")
    
except Exception as e:
    print(f"❌ Erreur : {e}")
    import traceback
    traceback.print_exc()