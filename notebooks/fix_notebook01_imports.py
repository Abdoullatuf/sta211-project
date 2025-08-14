#!/usr/bin/env python3
"""
Script pour corriger les imports dans le notebook 01_EDA_Preprocessing.ipynb
"""

import json
from pathlib import Path

def fix_notebook_imports():
    """Corrige les imports dans le notebook 01."""
    
    notebook_path = Path("01_EDA_Preprocessing.ipynb")
    
    if not notebook_path.exists():
        print(f"❌ Notebook non trouvé: {notebook_path}")
        return False
    
    print(f"🔧 Correction des imports dans {notebook_path}")
    
    # Lire le notebook
    with open(notebook_path, 'r', encoding='utf-8') as f:
        notebook = json.load(f)
    
    corrections_made = 0
    
    # Parcourir toutes les cellules
    for cell in notebook.get('cells', []):
        if cell.get('cell_type') == 'code':
            source = cell.get('source', [])
            
            # Convertir en string si nécessaire
            if isinstance(source, list):
                source_text = ''.join(source)
            else:
                source_text = source
            
            # Corrections à effectuer
            replacements = {
                'from modules.preprocessing.transformation_optimale_mixte import appliquer_transformation_optimale': 
                'from modules.notebook1_preprocessing import apply_optimal_transformations',
                
                'appliquer_transformation_optimale(': 
                'apply_optimal_transformations(',
                
                'appliquer_transformation_optimale ': 
                'apply_optimal_transformations '
            }
            
            # Appliquer les corrections
            new_source_text = source_text
            for old, new in replacements.items():
                if old in source_text:
                    new_source_text = new_source_text.replace(old, new)
                    corrections_made += 1
                    print(f"  ✅ Corrigé: {old[:50]}... -> {new[:50]}...")
            
            # Remettre dans la cellule si modifié
            if new_source_text != source_text:
                cell['source'] = new_source_text.split('\n')
                # Garder les retours à la ligne
                cell['source'] = [line + '\n' if i < len(cell['source'])-1 else line 
                                 for i, line in enumerate(cell['source'])]
    
    if corrections_made > 0:
        # Sauvegarder le notebook corrigé
        backup_path = notebook_path.with_suffix('.ipynb.backup')
        
        # Faire une sauvegarde
        with open(backup_path, 'w', encoding='utf-8') as f:
            json.dump(notebook, f, indent=1, ensure_ascii=False)
        
        # Sauvegarder le notebook corrigé
        with open(notebook_path, 'w', encoding='utf-8') as f:
            json.dump(notebook, f, indent=1, ensure_ascii=False)
        
        print(f"✅ {corrections_made} corrections appliquées")
        print(f"💾 Sauvegarde créée: {backup_path}")
        print(f"📝 Notebook corrigé: {notebook_path}")
        
        return True
    else:
        print("ℹ️ Aucune correction nécessaire")
        return True

if __name__ == "__main__":
    success = fix_notebook_imports()
    exit(0 if success else 1)