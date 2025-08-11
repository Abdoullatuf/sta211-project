# =============================================================================
# MODULE DE TRANSFORMATION OPTIMALE MIXTE (Version Corrigée)
# Yeo-Johnson pour X1, X2 - Box-Cox pour X3 (approche optimale)
# =============================================================================

import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.preprocessing import PowerTransformer
import joblib
import os
import matplotlib.pyplot as plt
from typing import List, Union
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')

# =============================================================================
# CLASSE PRINCIPALE DE TRANSFORMATION
# =============================================================================

class TransformationOptimaleMixte:
    """
    Classe pour appliquer la transformation optimale mixte et sauvegarder les modèles.
    """
    
    def __init__(self, models_dir: Union[str, Path] = 'models/notebook1/', verbose: bool = True):
        """
        Initialise la classe avec un chemin de sauvegarde pour les modèles.
        """
        self.transformer_yj = PowerTransformer(method='yeo-johnson', standardize=True)
        self.transformer_bc = PowerTransformer(method='box-cox', standardize=True)
        self.variables_yj = ['X1', 'X2']
        self.variables_bc = ['X3']
        self.is_fitted = False
        self.verbose = verbose
        self.models_dir = Path(models_dir)

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Applique la transformation optimale mixte complète."""
        if self.verbose:
            print("🔧 TRANSFORMATION OPTIMALE MIXTE")
            print("=" * 50)
        
        self._diagnostic_initial(df)
        df_result = df.copy()
        
        # Yeo-Johnson pour X1, X2
        df_result[self.variables_yj] = self.transformer_yj.fit_transform(df[self.variables_yj])
        
        # Box-Cox pour X3
        df_x3 = df[self.variables_bc].copy()
        if (df_x3['X3'] <= 0).any():
            offset = abs(df_x3['X3'].min()) + 1e-6
            df_x3['X3'] += offset
            if self.verbose:
                print(f"⚠️ X3 corrigé : un offset de {offset:.6f} a été ajouté.")
        
        df_result[self.variables_bc] = self.transformer_bc.fit_transform(df_x3)
        
        self.is_fitted = True
        self._rename_columns(df_result)
        self._rapport_transformation(df, df_result)
        self.save_transformers()
        
        return df_result

    def _rename_columns(self, df: pd.DataFrame):
        """Renomme les colonnes après transformation."""
        rename_map = {col: f"{col}_transformed" for col in self.variables_yj + self.variables_bc}
        df.rename(columns=rename_map, inplace=True)

    def save_transformers(self):
        """Sauvegarde les transformateurs dans le dossier spécifié."""
        os.makedirs(self.models_dir, exist_ok=True)
        joblib.dump(self.transformer_yj, self.models_dir / 'yeo_johnson_transformer.pkl')
        joblib.dump(self.transformer_bc, self.models_dir / 'box_cox_transformer.pkl')
        if self.verbose:
            print(f"\n💾 Transformateurs sauvegardés dans : {self.models_dir}")
            
    # --- Fonctions de rapport et diagnostic ---
    def _diagnostic_initial(self, df):
        if self.verbose:
            print("\n📊 Diagnostic initial (Asymétrie):")
            for var in self.variables_yj + self.variables_bc:
                print(f"  {var}: {df[var].skew():.3f}")

    def _rapport_transformation(self, df_original, df_transformed):
        if self.verbose:
            print("\n📊 Rapport de Transformation :")
            for var in self.variables_yj + self.variables_bc:
                original_skew = df_original[var].skew()
                transformed_skew = df_transformed[f'{var}_transformed'].skew()
                print(f"  {var}: Asymétrie avant={original_skew:+.3f} → après={transformed_skew:+.3f}")

# =============================================================================
# FONCTIONS UTILITAIRES (POUR NOTEBOOK)
# =============================================================================

def appliquer_transformation_optimale(
    df: pd.DataFrame, 
    models_dir: Union[str, Path], 
    verbose: bool = True
) -> pd.DataFrame:
    """Fonction principale pour appliquer la transformation et sauvegarder les modèles."""
    transformer = TransformationOptimaleMixte(models_dir=models_dir, verbose=verbose)
    df_transformed = transformer.fit_transform(df)
    if verbose:
        print("\n✅ TRANSFORMATION OPTIMALE TERMINÉE")
    return df_transformed


def generer_graphiques_comparaison(
    df_original: pd.DataFrame, 
    df_transformed: pd.DataFrame, 
    figures_dir: Union[str, Path]
):
    """Génère et sauvegarde des graphiques de comparaison avant/après transformation."""
    figures_dir = Path(figures_dir)
    os.makedirs(figures_dir, exist_ok=True)
    
    fig, axes = plt.subplots(3, 2, figsize=(12, 10))
    fig.suptitle('Comparaison Avant/Après Transformation', fontsize=16, fontweight='bold')
    
    variables = ['X1', 'X2', 'X3']
    for i, var in enumerate(variables):
        # Distribution originale
        sns.histplot(df_original[var], kde=True, ax=axes[i, 0], color='salmon')
        axes[i, 0].set_title(f'{var} - Original (Skew: {df_original[var].skew():.2f})')
        
        # Distribution transformée
        sns.histplot(df_transformed[f'{var}_transformed'], kde=True, ax=axes[i, 1], color='mediumseagreen')
        axes[i, 1].set_title(f'{var} - Transformé (Skew: {df_transformed[f"{var}_transformed"].skew():.2f})')
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    save_path = figures_dir / "transformation_comparison.png"
    plt.savefig(save_path, dpi=150)
    print(f"📊 Graphique de comparaison sauvegardé dans : {save_path}")
    plt.show()


def plot_outlier_comparison(
    df_before: pd.DataFrame, 
    df_after: pd.DataFrame, 
    cols: List[str], 
    figures_dir: Union[str, Path]
):
    """Affiche et sauvegarde les boxplots avant/après traitement des outliers."""
    figures_dir = Path(figures_dir)
    os.makedirs(figures_dir, exist_ok=True)

    for col in cols:
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        fig.suptitle(f'Traitement des Outliers pour {col}', fontsize=14)
        
        sns.boxplot(x=df_before[col], ax=axes[0], color='salmon')
        axes[0].set_title("Avant Traitement")

        sns.boxplot(x=df_after[col], ax=axes[1], color='mediumseagreen')
        axes[1].set_title("Après Traitement")
        
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        save_path = figures_dir / f"outlier_comparison_{col}.png"
        plt.savefig(save_path, dpi=150)
        plt.show()
        
    print(f"📊 Graphiques de comparaison des outliers sauvegardés dans : {figures_dir}")


if __name__ == "__main__":
    print("Ce module contient des outils de prétraitement et doit être importé.")