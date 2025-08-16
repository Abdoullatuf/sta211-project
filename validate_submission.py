#!/usr/bin/env python3
"""
Script de validation des fichiers de soumission selon les critères spécifiés.
Reproduit la logique de la fonction R de validation.
"""

import csv
from pathlib import Path

def validate_submission(file_path):
    """
    Valide un fichier de soumission selon les critères :
    - Doit contenir exactement 820 lignes
    - Chaque ligne doit contenir uniquement "ad." ou "noad."
    - Pas d'en-tête
    """
    
    file_path = Path(file_path)
    print(f"🔍 Validation du fichier : {file_path.name}")
    
    if not file_path.exists():
        print(f"❌ Fichier introuvable : {file_path}")
        return False
    
    try:
        with open(file_path, 'r') as f:
            lines = f.readlines()
            
        # Nettoyer les lignes (enlever les \n)
        lines = [line.strip() for line in lines if line.strip()]
        num_lines = len(lines)
        
        print(f"📊 Nombre de lignes : {num_lines}")
        
        # Vérifier le nombre exact de lignes
        if num_lines != 820:
            print(f"❌ Nombre de lignes incorrect. Attendu: 820, Trouvé: {num_lines}")
            return False
        
        # Vérifier le contenu de chaque ligne
        valid_outcomes = {'ad.', 'noad.'}
        ad_count = 0
        noad_count = 0
        
        for i, line in enumerate(lines):
            # Vérifier l'outcome
            if line not in valid_outcomes:
                print(f"❌ Ligne {i+1}: contenu invalid. Attendu: 'ad.' ou 'noad.', Trouvé: '{line}'")
                return False
            
            # Compter les outcomes
            if line == 'ad.':
                ad_count += 1
            else:
                noad_count += 1
            
        # Statistiques finales
        ad_percent = 100 * ad_count / num_lines
        noad_percent = 100 * noad_count / num_lines
        
        print(f"✅ Validation réussie !")
        print(f"📈 Répartition : {ad_count} 'ad.' ({ad_percent:.1f}%), {noad_count} 'noad.' ({noad_percent:.1f}%)")
        
        return True
            
    except Exception as e:
        print(f"❌ Erreur lors de la lecture du fichier : {e}")
        return False

def validate_all_submissions():
    """Valide tous les fichiers de soumission générés."""
    
    project_root = Path(__file__).parent
    predictions_dir = project_root / "outputs" / "predictions"
    
    submission_files = [
        "submission_stacking_with_refit_mice_820.csv",
        "submission_stacking_with_refit_knn_820.csv"
    ]
    
    print("🚀 Validation des fichiers de soumission\n")
    
    all_valid = True
    
    for filename in submission_files:
        file_path = predictions_dir / filename
        print(f"{'='*50}")
        is_valid = validate_submission(file_path)
        all_valid = all_valid and is_valid
        print()
    
    print(f"{'='*50}")
    if all_valid:
        print("🎉 Tous les fichiers de soumission sont valides !")
    else:
        print("❌ Certains fichiers de soumission ont des erreurs.")
    
    return all_valid

if __name__ == "__main__":
    validate_all_submissions()