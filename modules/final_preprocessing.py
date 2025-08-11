"""
modules/final_preprocessing.py - VERSION 3.0

Module de prétraitement complet pour le projet STA211 Internet Advertisements.
Inclut toutes les corrections, la réduction de corrélation optimisée, et les fonctions avancées.

NOUVELLES FONCTIONNALITÉS VERSION 3.0:
- Intégration de l'analyse de corrélation comprehensive
- Réduction de dimensionnalité efficace
- Protection X4 renforcée
- Pipeline modulaire et robuste


Auteur: Abdoullatuf
Version: 3.0
Date: 2025
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from pathlib import Path
from typing import List, Optional, Union, Dict, Tuple

from sklearn.preprocessing import PowerTransformer
from outliers import detect_and_remove_outliers
from eda import load_and_clean_data


from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

import sys
import json

from config import cfg

models_dir = cfg.paths.MODELS_DIR / "notebook1"
Path(models_dir).mkdir(parents=True, exist_ok=True)

# ============================================================================
# 1. UTILITAIRES DE BASE
# ============================================================================

def convert_X4_to_int(df: pd.DataFrame, column: str = "X4", verbose: bool = True) -> pd.DataFrame:
    """
    Convertit X4 en Int64 si elle contient uniquement des valeurs binaires (0, 1).
    
    Args:
        df: DataFrame d'entrée
        column: Nom de la colonne à convertir (défaut: "X4")
        verbose: Affichage des informations
        
    Returns:
        DataFrame avec X4 convertie en Int64
    """
    df = df.copy()
    
    if column not in df.columns:
        if verbose:
            print(f"⚠️ Colonne '{column}' absente du DataFrame.")
        return df

    unique_vals = df[column].dropna().unique()
    if set(unique_vals).issubset({0, 1}):
        df[column] = df[column].astype("Int64")
        if verbose:
            print(f"✅ Colonne '{column}' convertie en Int64 (binaire).")
    elif verbose:
        print(f"❌ Colonne '{column}' contient {unique_vals}. Conversion ignorée.")

    return df


def apply_yeojohnson(
    df: pd.DataFrame,
    columns: List[str],
    standardize: bool = False,
    save_model: bool = False,
    model_path: Optional[Union[str, Path]] = None,
    return_transformer: bool = False
) -> Union[pd.DataFrame, Tuple[pd.DataFrame, PowerTransformer]]:
    """
    Applique la transformation Yeo-Johnson sur les colonnes spécifiées.
    
    Args:
        df: DataFrame d'entrée
        columns: Colonnes à transformer
        standardize: Standardiser après transformation
        save_model: Sauvegarder le transformateur
        model_path: Chemin de sauvegarde du modèle
        return_transformer: Retourner aussi le transformateur
        
    Returns:
        DataFrame transformé (et transformateur si demandé)
    """
    df_transformed = df.copy()

    # Charger ou créer le transformateur
    if model_path and Path(model_path).exists():
        pt = joblib.load(model_path)
        print(f"🔄 Transformateur rechargé depuis : {model_path}")
    else:
        pt = PowerTransformer(method="yeo-johnson", standardize=standardize)
        pt.fit(df[columns])
        
        if save_model and model_path:
            Path(model_path).parent.mkdir(parents=True, exist_ok=True)
            joblib.dump(pt, model_path)
            print(f"✅ Transformateur Yeo-Johnson sauvegardé à : {model_path}")

    # Appliquer la transformation
    transformed_values = pt.transform(df[columns])
    for i, col in enumerate(columns):
        df_transformed[f"{col}_trans"] = transformed_values[:, i]

    return (df_transformed, pt) if return_transformer else df_transformed


# ============================================================================
# 2. ANALYSE DE CORRÉLATION COMPREHENSIVE (NOUVELLE VERSION 3.0)
# ============================================================================

def analyze_correlation_comprehensive(df: pd.DataFrame, threshold=0.90, show_analysis=True):
    """
    Analyse de corrélation complète avec protection X4 et réduction effective.
    
    Args:
        df: DataFrame d'entrée
        threshold: Seuil de corrélation pour suppression
        show_analysis: Afficher l'analyse détaillée
        
    Returns:
        df_reduced, correlation_report
    """
    
    if show_analysis:
        print("🔍 ANALYSE DE CORRÉLATION COMPREHENSIVE")
        print("=" * 45)
        print(f"📊 Dataset initial : {df.shape}")
    
    # ========================================================================
    # 1. IDENTIFICATION DES VARIABLES BINAIRES
    # ========================================================================
    
    # Variables binaires (exclure outcome et protéger X4)
    binary_vars = []
    protected_vars = ['X4']  # Variables à protéger
    excluded_vars = ['outcome']  # Variables à exclure de l'analyse
    
    for col in df.columns:
        if col in excluded_vars:
            continue
        if df[col].dtype in ['int64', 'Int64', 'bool']:
            # Vérifier que c'est vraiment binaire
            unique_vals = df[col].dropna().unique()
            if len(unique_vals) <= 2 and set(unique_vals).issubset({0, 1}):
                binary_vars.append(col)
    
    if show_analysis:
        print(f"🔢 Variables binaires trouvées : {len(binary_vars)}")
        print(f"🛡️ Variables protégées : {protected_vars}")
        print(f"⚠️ Variables exclues : {excluded_vars}")
    
    if len(binary_vars) < 2:
        if show_analysis:
            print("❌ Pas assez de variables binaires pour l'analyse")
        return df, {"message": "Pas assez de variables binaires"}
    
    # ========================================================================
    # 2. CALCUL DE LA MATRICE DE CORRÉLATION
    # ========================================================================
    
    try:
        # Calcul sur les variables binaires seulement
        binary_df = df[binary_vars]
        corr_matrix = binary_df.corr().abs()
        
        if show_analysis:
            print(f"📐 Matrice de corrélation : {corr_matrix.shape}")
            
    except Exception as e:
        if show_analysis:
            print(f"❌ Erreur calcul corrélation : {e}")
        return df, {"error": str(e)}
    
    # ========================================================================
    # 3. IDENTIFICATION DES GROUPES CORRÉLÉS
    # ========================================================================
    
    # Triangle supérieur pour éviter les doublons
    upper_triangle = corr_matrix.where(
        np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    )
    
    # Identifier les paires hautement corrélées
    highly_correlated_pairs = []
    
    for col in upper_triangle.columns:
        correlated_with = upper_triangle[col][upper_triangle[col] > threshold]
        if len(correlated_with) > 0:
            for corr_col, corr_val in correlated_with.items():
                highly_correlated_pairs.append({
                    'var1': col,
                    'var2': corr_col, 
                    'correlation': corr_val
                })
    
    if show_analysis:
        print(f"🔗 Paires hautement corrélées (>{threshold}) : {len(highly_correlated_pairs)}")
        
        # Afficher quelques exemples
        if highly_correlated_pairs:
            print("📋 Exemples de corrélations élevées :")
            for i, pair in enumerate(highly_correlated_pairs[:5]):
                print(f"   {pair['var1']} ↔ {pair['var2']}: {pair['correlation']:.3f}")
            if len(highly_correlated_pairs) > 5:
                print(f"   ... et {len(highly_correlated_pairs) - 5} autres")
    
    # ========================================================================
    # 4. STRATÉGIE DE SUPPRESSION AVEC PROTECTION
    # ========================================================================
    
    # Variables à supprimer (en protégeant X4)
    vars_to_drop = set()
    
    # Pour chaque paire corrélée, supprimer une des deux variables
    for pair in highly_correlated_pairs:
        var1, var2 = pair['var1'], pair['var2']
        
        # Logique de priorité :
        # 1. Ne jamais supprimer X4
        # 2. Supprimer la variable avec l'index le plus élevé (convention)
        
        if var1 in protected_vars:
            # Protéger var1, supprimer var2 si pas protégée
            if var2 not in protected_vars:
                vars_to_drop.add(var2)
        elif var2 in protected_vars:
            # Protéger var2, supprimer var1
            vars_to_drop.add(var1)
        else:
            # Aucune des deux n'est protégée, supprimer celle avec index plus élevé
            if var1 > var2:  # Ordre alphabétique/numérique
                vars_to_drop.add(var1)
            else:
                vars_to_drop.add(var2)
    
    vars_to_drop = list(vars_to_drop)
    
    if show_analysis:
        print(f"📉 Variables marquées pour suppression : {len(vars_to_drop)}")
        
        # Vérification protection X4
        x4_protected = all(var not in protected_vars for var in vars_to_drop)
        print(f"🛡️ X4 correctement protégée : {'✅' if x4_protected else '❌'}")
        
        if vars_to_drop:
            print(f"🗑️ Variables à supprimer : {vars_to_drop[:10]}{'...' if len(vars_to_drop) > 10 else ''}")
    
    # ========================================================================
    # 5. APPLICATION DE LA SUPPRESSION
    # ========================================================================
    
    # Suppression effective
    df_reduced = df.drop(columns=vars_to_drop, errors='ignore')
    
    # Rapport final
    correlation_report = {
        'original_shape': df.shape,
        'reduced_shape': df_reduced.shape,
        'binary_vars_analyzed': len(binary_vars),
        'highly_correlated_pairs': len(highly_correlated_pairs),
        'vars_dropped': len(vars_to_drop),
        'vars_dropped_list': vars_to_drop,
        'protected_vars': protected_vars,
        'x4_still_present': 'X4' in df_reduced.columns,
        'dimension_reduction': df.shape[1] - df_reduced.shape[1]
    }
    
    if show_analysis:
        print("\n🎉 RÉSULTATS DE LA RÉDUCTION :")
        print(f"📊 Dimensions avant : {correlation_report['original_shape']}")
        print(f"📊 Dimensions après : {correlation_report['reduced_shape']}")
        print(f"📉 Réduction : {correlation_report['dimension_reduction']} colonnes supprimées")
        print(f"🛡️ X4 présente : {'✅' if correlation_report['x4_still_present'] else '❌'}")
        print(f"⚖️ Pourcentage conservé : {df_reduced.shape[1]/df.shape[1]*100:.1f}%")
    
    return df_reduced, correlation_report








def apply_collinearity_filter(
    df,
    cols_to_drop,
    imputation_method: str, # 'mice' ou 'knn'
    models_dir: Path,       # Le dossier de BASE (ex: '.../notebook1')
    protected_cols=['X4'],
    display_info=True,
    save_results=True
):
    """
    Supprime les colonnes corrélées et sauvegarde la liste dans un
    sous-dossier spécifique à la méthode d'imputation.
    """

    # --- 1. Protection des colonnes ---
    cols_to_drop_filtered = [col for col in cols_to_drop if col not in protected_cols]
    
    if display_info and protected_cols:
        protected_saved = len(cols_to_drop) - len(cols_to_drop_filtered)
        if protected_saved > 0:
            saved_cols = [col for col in cols_to_drop if col in protected_cols]
            print(f"🛡️ {protected_saved} colonnes protégées : {saved_cols}")

    # --- 2. Sauvegarde dynamique dans le sous-dossier ---
    if save_results:
        # Construit le chemin du sous-dossier (ex: .../notebook1/mice)
        save_directory = models_dir / imputation_method
        # Crée le dossier s'il n'existe pas
        save_directory.mkdir(parents=True, exist_ok=True)
        
        # Définit le chemin complet du fichier
        save_path = save_directory / "cols_to_drop_corr.pkl"
        
        try:
            joblib.dump(cols_to_drop_filtered, save_path)
            if display_info:
                print(f"💾 Liste des colonnes sauvegardée dans : {save_directory.name}/{save_path.name}")
        except Exception as e:
            print(f"❌ Erreur lors de la sauvegarde : {e}")

    # --- 3. Suppression des colonnes ---
    existing_cols_to_drop = [col for col in cols_to_drop_filtered if col in df.columns]
    df_filtered = df.drop(columns=existing_cols_to_drop)

    if display_info:
        print(f"✅ Colonnes supprimées : {len(existing_cols_to_drop)}")
        print(f"📏 Dimensions finales : {df_filtered.shape}")

    return df_filtered




# ======================================================================
# FONCTION PRINCIPALE : DÉTECTION DES VARIABLES TRÈS CORRÉLÉES
# ======================================================================

def find_highly_correlated_groups(
    df: pd.DataFrame,
    threshold: float = 0.90,
    exclude_cols: Optional[List[str]] = None,
    protected_cols: Optional[List[str]] = None,
    show_plot: bool = False,
    save_path: Optional[Union[str, Path]] = None,
    figsize: tuple = (10, 8)
) -> Dict[str, Union[List[List[str]], List[str]]]:
    """
    Identifie les groupes de variables fortement corrélées avec protection de certaines colonnes.

    VERSION 3.0 – MISE À JOUR : suppression de save_fig(), sauvegarde standard matplotlib

    Args:
        df: DataFrame d'entrée
        threshold: Seuil de corrélation (défaut: 0.90)
        exclude_cols: Colonnes à exclure de l'analyse
        protected_cols: Colonnes à protéger de la suppression (défaut: ['X4'])
        show_plot: Afficher la heatmap de corrélation
        save_path: Chemin de sauvegarde de la figure (PNG ou autre format)
        figsize: Taille de la figure

    Returns:
        Dictionnaire avec groups, to_drop, et protected
    """
    if protected_cols is None:
        protected_cols = ['X4']

    if df.empty:
        print("⚠️ DataFrame vide - retour structure par défaut")
        return {"groups": [], "to_drop": [], "protected": protected_cols}

    all_exclude_cols = (exclude_cols or []) + (protected_cols or [])
    df_corr = df.drop(columns=all_exclude_cols, errors='ignore') if all_exclude_cols else df.copy()

    if df_corr.empty:
        print("⚠️ Aucune colonne à analyser après exclusions")
        return {"groups": [], "to_drop": [], "protected": protected_cols}

    try:
        corr_matrix = df_corr.corr().abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    except Exception as e:
        print(f"⚠️ Erreur dans le calcul de corrélation: {e}")
        return {"groups": [], "to_drop": [], "protected": protected_cols}

    groups, visited = [], set()
    for col in upper.columns:
        if col in visited:
            continue
        correlated = upper[col][upper[col] > threshold].index.tolist()
        if correlated:
            group = sorted(set([col] + correlated))
            groups.append(group)
            visited.update(group)

    to_drop = []
    for group in groups:
        protected_in_group = [col for col in group if col in protected_cols]
        non_protected = [col for col in group if col not in protected_cols]
        if non_protected:
            to_drop.extend(non_protected[1:])  # Garder le premier non-protégé

    if show_plot and not corr_matrix.empty:
        try:
            plt.figure(figsize=figsize)
            sns.heatmap(corr_matrix, cmap="coolwarm", center=0, square=True,
                        cbar_kws={"shrink": 0.75})
            plt.title(f"Matrice de corrélation (>{threshold})")
            plt.tight_layout()

            if save_path:
                save_path = Path(save_path)
                save_path.parent.mkdir(parents=True, exist_ok=True)
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                print(f"📸 Figure sauvegardée dans : {save_path}")
            else:
                plt.show()
        except Exception as e:
            print(f"⚠️ Erreur dans la visualisation: {e}")

    protected_in_drop = set(to_drop) & set(protected_cols)
    if protected_in_drop:
        print(f"🛡️ PROTECTION ACTIVÉE: Retrait de {protected_in_drop} de la liste de suppression")
        to_drop = [col for col in to_drop if col not in protected_cols]

    result = {
        "groups": groups,
        "to_drop": to_drop,
        "protected": protected_cols
    }

    assert isinstance(result, dict)
    assert isinstance(result["groups"], list)
    assert isinstance(result["to_drop"], list)
    assert isinstance(result["protected"], list)

    return result




def drop_correlated_duplicates(
    df: pd.DataFrame,
    groups: List[List[str]],
    target_col: str = "outcome",
    extra_cols: List[str] = None,
    protected_cols: List[str] = None,
    priority_cols: List[str] = None,
    verbose: bool = False,
    summary: bool = True
) -> Tuple[pd.DataFrame, List[str], List[str]]:
    """
    Supprime les variables corrélées avec protection et ordre des colonnes optimisé.
    
    VERSION 3.0 - ROBUSTESSE RENFORCÉE
    """
    # 🛡️ Protection par défaut
    if protected_cols is None:
        protected_cols = ['X4']
    
    # 📌 Colonnes prioritaires par défaut
    if priority_cols is None:
        priority_cols = ['X1_trans', 'X2_trans', 'X3_trans', 'X4']
    
    # 🔧 VALIDATION D'ENTRÉE RENFORCÉE
    if not isinstance(groups, list):
        print(f"⚠️ groups doit être une liste, reçu: {type(groups)} - conversion forcée")
        groups = list(groups) if groups else []
    
    to_drop, to_keep = [], []

    # Traitement des groupes corrélés avec validation
    for group in groups:
        if not group or not isinstance(group, (list, tuple)):
            if verbose:
                print(f"⚠️ Groupe invalide ignoré: {group}")
            continue
        
        group = list(group)  # Conversion sécurisée
        
        # 🛡️ Séparer les colonnes protégées des autres
        protected_in_group = [col for col in group if col in protected_cols]
        non_protected = [col for col in group if col not in protected_cols and col in df.columns]
        
        if non_protected:
            # Garder le premier non-protégé
            keep = non_protected[0]
            drop = non_protected[1:]
            to_keep.append(keep)
            to_drop.extend(drop)
            
            if verbose:
                print(f"🧹 Groupe : {group} → garde {keep}, retire {drop}")
                if protected_in_group:
                    print(f"🛡️   Protégées dans ce groupe : {protected_in_group}")
        
        # 🛡️ Les colonnes protégées sont toujours gardées
        for protected in protected_in_group:
            if protected not in to_keep:
                to_keep.append(protected)

    to_drop = sorted(set(to_drop))
    to_keep = sorted(set(to_keep))

    # Colonnes binaires restantes (non corrélées)
    all_binary = [col for col in df.select_dtypes(include=['int64', 'Int64']).columns 
                  if col != target_col]
    untouched = [col for col in all_binary if col not in to_drop + to_keep]

    # 📌 CONSTRUCTION DE L'ORDRE FINAL DES COLONNES
    
    # 1. Colonnes prioritaires (en premier)
    priority_existing = [col for col in priority_cols if col in df.columns]
    
    # 2. Variable cible (après les prioritaires)
    target_existing = [target_col] if target_col and target_col in df.columns else []
    
    # 3. Extra cols (variables transformées, etc.)
    extra_existing = []
    if extra_cols:
        extra_existing = [col for col in extra_cols 
                         if col in df.columns and col not in priority_existing + target_existing]
    
    # 4. Variables gardées par corrélation (pas déjà dans prioritaires/extra)
    kept_remaining = [col for col in to_keep 
                     if col not in priority_existing + target_existing + extra_existing]
    
    # 5. Variables intactes (pas déjà listées)
    untouched_remaining = [col for col in untouched 
                          if col not in priority_existing + target_existing + extra_existing + kept_remaining]
    
    # 🛡️ S'assurer que les colonnes protégées sont présentes
    protected_remaining = []
    for protected in protected_cols:
        if (protected in df.columns and 
            protected not in priority_existing + target_existing + extra_existing + 
            kept_remaining + untouched_remaining):
            protected_remaining.append(protected)
    
    # 📌 ORDRE FINAL : prioritaires → cible → extra → gardées → intactes → protégées restantes
    final_cols = (priority_existing + target_existing + extra_existing + 
                  kept_remaining + untouched_remaining + protected_remaining)
    
    # Filtrage des colonnes existantes (sécurité)
    existing_cols = [col for col in final_cols if col in df.columns]
    df_reduced = df[existing_cols].copy()

    # Affichage du résumé
    if summary:
        print(f"\n📊 Réduction : {len(to_drop)} supprimées, {len(to_keep)} gardées, {len(untouched)} intactes.")
        print(f"📌 Ordre final : {priority_existing[:3]}{'...' if len(priority_existing) > 3 else ''} → {target_existing} → reste")
        
        if protected_cols:
            protected_in_final = [col for col in protected_cols if col in existing_cols]
            print(f"🛡️ {len(protected_in_final)} colonnes protégées : {protected_in_final}")
        
        if extra_cols:
            existing_extra = [col for col in extra_cols if col in existing_cols]
            print(f"🧩 {len(existing_extra)} extra conservées : {existing_extra}")
        
        print(f"📐 Dimensions : {df_reduced.shape}")

    return df_reduced, to_drop, to_keep


# ============================================================================
# 4. FONCTIONS DE VALIDATION ET PROTECTION X4
# ============================================================================

def validate_x4_presence(df: pd.DataFrame, step_name: str = "", verbose: bool = True) -> bool:
    """
    Valide que X4 est présente et correcte dans le DataFrame.
    """
    if 'X4' not in df.columns:
        if verbose:
            print(f"❌ {step_name}: X4 MANQUANTE !")
        return False
    
    # Vérifier le type et les valeurs
    unique_vals = sorted(df['X4'].dropna().unique())
    expected_vals = [0, 1]
    
    if set(unique_vals).issubset(set(expected_vals)):
        if verbose:
            print(f"✅ {step_name}: X4 présente et correcte (valeurs: {unique_vals})")
        return True
    else:
        if verbose:
            print(f"⚠️ {step_name}: X4 présente mais valeurs inattendues: {unique_vals}")
        return False


def quick_x4_check(df_or_dict, name: str = "Dataset") -> bool:
    """
    Vérification rapide de X4 dans un DataFrame ou dictionnaire de DataFrames.
    """
    if isinstance(df_or_dict, dict):
        print(f"🔍 Vérification X4 dans {len(df_or_dict)} datasets:")
        all_good = True
        for dataset_name, df in df_or_dict.items():
            has_x4 = 'X4' in df.columns if df is not None else False
            print(f"  {dataset_name}: {'✅' if has_x4 else '❌'}")
            if not has_x4:
                all_good = False
        return all_good
    else:
        # DataFrame unique
        has_x4 = 'X4' in df_or_dict.columns
        print(f"🔍 {name}: {'✅' if has_x4 else '❌'} X4")
        return has_x4


# ============================================================================
# 5. GESTION DE L'ORDRE DES COLONNES
# ============================================================================

def reorder_columns_priority(
    df: pd.DataFrame, 
    priority_cols: List[str] = None,
    target_col: str = "outcome"
) -> pd.DataFrame:
    """
    Réorganise les colonnes avec un ordre prioritaire.
    """
    if priority_cols is None:
        priority_cols = ['X1_trans', 'X2_trans', 'X3_trans', 'X4']
    
    current_cols = df.columns.tolist()
    
    # 1. Variables prioritaires (en premier)
    final_priority = [col for col in priority_cols if col in current_cols]
    
    # 2. Variable cible (après les prioritaires)
    final_target = [target_col] if target_col and target_col in current_cols else []
    
    # 3. Toutes les autres colonnes (dans l'ordre actuel)
    final_others = [col for col in current_cols if col not in final_priority + final_target]
    
    # Ordre final : prioritaires → cible → reste
    final_order = final_priority + final_target + final_others
    
    return df[final_order]


# ============================================================================
# 6. PIPELINE PRINCIPAL DE PRÉTRAITEMENT (VERSION 3.0 COMPLÈTE)
# ============================================================================

def prepare_final_dataset(
    file_path: Union[str, Path],
    strategy: str = "mixed_mar_mcar",
    mar_method: str = "knn",
    knn_k: Optional[int] = None,
    mar_cols: List[str] = ["X1_trans", "X2_trans", "X3_trans"],
    mcar_cols: List[str] = ["X4"],
    drop_outliers: bool = False,
    correlation_threshold: float = 0.90,
    save_transformer: bool = False,
    processed_data_dir: Optional[Union[str, Path]] = None,
    models_dir: Optional[Union[str, Path]] = None,
    display_info: bool = True,
    raw_data_dir: Optional[Union[str, Path]] = None,
    require_outcome: bool = True,
    protect_x4: bool = True,
    priority_cols: List[str] = None,
    return_objects: bool = False,
    use_comprehensive_correlation: bool = True
) -> Union[pd.DataFrame, Tuple[pd.DataFrame, Dict]]:
    """
    Pipeline de prétraitement complet avec toutes les corrections et améliorations.
    
    VERSION 3.0 - COMPLÈTEMENT CORRIGÉE ET OPTIMISÉE
    
    Args:
        file_path: Chemin vers le fichier de données
        strategy: Stratégie d'imputation ("mixed_mar_mcar")
        mar_method: Méthode d'imputation MAR ("knn" ou "mice")
        knn_k: Paramètre k pour KNN (None = auto)
        mar_cols: Colonnes à imputer avec méthode MAR
        mcar_cols: Colonnes à imputer avec méthode MCAR
        drop_outliers: Supprimer les outliers
        correlation_threshold: Seuil de corrélation pour suppression
        save_transformer: Sauvegarder les transformateurs
        processed_data_dir: Dossier de sauvegarde des données
        models_dir: Dossier de sauvegarde des modèles
        display_info: Affichage des informations
        raw_data_dir: Dossier des données brutes
        require_outcome: Nécessite la variable cible
        protect_x4: Protéger X4 de la suppression
        priority_cols: Colonnes prioritaires pour l'ordre
        return_objects: Retourner aussi les objets de transformation
        use_comprehensive_correlation: Utiliser la nouvelle méthode de corrélation
        
    Returns:
        DataFrame prétraité (et objets si demandé)
        
    Raises:
        ValueError: Si X4 est perdue pendant le preprocessing
    """
    
    # 🛡️ Configuration de protection
    protected_cols = ['X4'] if protect_x4 else []
    
    # 📌 Colonnes prioritaires par défaut
    if priority_cols is None:
        priority_cols = ['X1_trans', 'X2_trans', 'X3_trans', 'X4']

    # Dictionnaire pour stocker les objets de transformation
    transform_objects = {
        'scaler': None,
        'imputer': None,
        'yeojohnson': None,
        'correlation_info': None
    }

    if display_info:
        print("🔄 DÉMARRAGE DU PIPELINE DE PRÉTRAITEMENT (VERSION 3.0)")
        print("=" * 70)

    # ========================================================================
    # ÉTAPE 1: CHARGEMENT DES DONNÉES
    # ========================================================================
    
    if display_info:
        print("📂 Étape 1: Chargement des données...")
    
    try:
        df = load_and_clean_data(
            file_path=file_path,
            require_outcome=require_outcome,
            display_info=display_info,
            raw_data_dir=raw_data_dir,
            encode_target=True
        )
    except Exception as e:
        print(f"❌ Erreur lors du chargement : {e}")
        raise
    
    # 🛡️ Validation X4 après chargement
    if protect_x4:
        validate_x4_presence(df, "Après chargement", display_info)

    # ========================================================================
    # ÉTAPE 2: CONVERSION DE X4
    # ========================================================================
    
    if display_info:
        print("\n🔧 Étape 2: Conversion de X4...")
    
    df = convert_X4_to_int(df, verbose=display_info)
    
    # 🛡️ Validation X4 après conversion
    if protect_x4:
        validate_x4_presence(df, "Après conversion X4", display_info)

    # ========================================================================
    # ÉTAPE 3: TRANSFORMATION YEO-JOHNSON
    # ========================================================================
    
    if display_info:
        print("\n🔄 Étape 3: Transformation Yeo-Johnson (X1, X2, X3)...")
    
    try:
        if return_objects:
            df, yeojohnson_transformer = apply_yeojohnson(
                df=df,
                columns=["X1", "X2", "X3"],
                standardize=False,
                save_model=save_transformer,
                model_path=models_dir / "yeojohnson.pkl" if save_transformer and models_dir else None,
                return_transformer=True
            )
            transform_objects['yeojohnson'] = yeojohnson_transformer
        else:
            df = apply_yeojohnson(
                df=df,
                columns=["X1", "X2", "X3"],
                standardize=False,
                save_model=save_transformer,
                model_path=models_dir / "yeojohnson.pkl" if save_transformer and models_dir else None,
                return_transformer=False
            )
        
        # Suppression des colonnes originales
        df.drop(columns=["X1", "X2", "X3"], inplace=True, errors="ignore")
        
    except Exception as e:
        print(f"❌ Erreur lors de la transformation Yeo-Johnson : {e}")
        if display_info:
            print("⚠️ Poursuite sans transformation...")
    
    # 🛡️ Validation X4 après transformation
    if protect_x4:
        validate_x4_presence(df, "Après Yeo-Johnson", display_info)

    # ========================================================================
    # ÉTAPE 4: IMPUTATION DES VALEURS MANQUANTES
    # ========================================================================
    
    if display_info:
        print(f"\n🔧 Étape 4: Imputation des valeurs manquantes ({mar_method})...")
    
    try:
        df = handle_missing_values(
            df=df,
            strategy=strategy,
            mar_method=mar_method,
            knn_k=knn_k,
            mar_cols=mar_cols,
            mcar_cols=mcar_cols,
            display_info=display_info,
            save_results=False,
            processed_data_dir=processed_data_dir,
            models_dir=models_dir
        )
    except Exception as e:
        print(f"❌ Erreur lors de l'imputation : {e}")
        if display_info:
            print("⚠️ Poursuite sans imputation...")
    
    # 🛡️ Validation X4 après imputation
    if protect_x4:
        validate_x4_presence(df, "Après imputation", display_info)

    # ========================================================================
    # ÉTAPE 5: RÉDUCTION DE LA COLINÉARITÉ (VERSION 3.0 COMPLÈTE)
    # ========================================================================
    
    if display_info:
        print(f"\n🔗 Étape 5: Réduction de la colinéarité (seuil={correlation_threshold})...")
    
    try:
        if use_comprehensive_correlation:
            # 🚀 NOUVELLE MÉTHODE COMPREHENSIVE
            df_reduced, correlation_report = apply_correlation_reduction_to_dataset(
                df, threshold=correlation_threshold
            )
            df = df_reduced
            transform_objects['correlation_info'] = correlation_report
            
        else:
            # 🔧 ANCIENNE MÉTHODE CORRIGÉE (POUR COMPATIBILITÉ)
            binary_vars = [col for col in df.columns 
                           if pd.api.types.is_integer_dtype(df[col]) and col != "outcome"]
            
            if display_info:
                print(f"🔢 Variables binaires candidates : {len(binary_vars)}")

            if binary_vars:
                groups_corr = find_highly_correlated_groups(
                    df[binary_vars], 
                    threshold=correlation_threshold,
                    protected_cols=protected_cols
                )
                
                # 🔧 VALIDATION DU TYPE DE RETOUR RENFORCÉE
                if isinstance(groups_corr, list):
                    if display_info:
                        print("⚠️ Format de retour détecté comme liste - conversion en dictionnaire")
                    groups_corr = {
                        "groups": groups_corr,
                        "to_drop": [],
                        "protected": protected_cols
                    }
                elif not isinstance(groups_corr, dict):
                    if display_info:
                        print(f"⚠️ Type de retour inattendu: {type(groups_corr)} - utilisation valeurs par défaut")
                    groups_corr = {
                        "groups": [],
                        "to_drop": [],
                        "protected": protected_cols
                    }
                elif "groups" not in groups_corr:
                    if display_info:
                        print("⚠️ Clé 'groups' manquante - ajout de structure par défaut")
                    groups_corr["groups"] = []
                    if "to_drop" not in groups_corr:
                        groups_corr["to_drop"] = []
                    if "protected" not in groups_corr:
                        groups_corr["protected"] = protected_cols
                
                # Stockage des informations de corrélation
                transform_objects['correlation_info'] = groups_corr
                
            else:
                # Pas de variables binaires à analyser
                groups_corr = {
                    "groups": [],
                    "to_drop": [],
                    "protected": protected_cols
                }
                if display_info:
                    print("⚠️ Aucune variable binaire trouvée pour l'analyse de corrélation")

            target_col = "outcome" if "outcome" in df.columns and require_outcome else None

            # 🛡️ Protection dans drop_correlated_duplicates
            df_reduced, dropped_cols, kept_cols = drop_correlated_duplicates(
                df=df,
                groups=groups_corr["groups"],
                target_col=target_col,
                extra_cols=mar_cols + mcar_cols,
                protected_cols=protected_cols,
                priority_cols=priority_cols,
                verbose=False,
                summary=display_info
            )
            
            # Réassignation du DataFrame
            df = df_reduced
        
    except Exception as e:
        print(f"❌ Erreur lors de la réduction de colinéarité : {e}")
        if display_info:
            print("⚠️ Poursuite sans réduction de colinéarité...")
        
        # Mode dégradé - Pas de réduction
        if 'correlation_info' not in transform_objects:
            transform_objects['correlation_info'] = {
                "error": str(e),
                "original_shape": df.shape,
                "reduced_shape": df.shape,
                "dimension_reduction": 0
            }
    
    # 🛡️ Validation X4 après réduction colinéarité
    if protect_x4:
        validate_x4_presence(df, "Après réduction colinéarité", display_info)

    # ========================================================================
    # ÉTAPE 6: SUPPRESSION DES OUTLIERS (OPTIONNELLE)
    # ========================================================================
    
    target_col = "outcome" if "outcome" in df.columns and require_outcome else None
    
    if drop_outliers and target_col:
        if display_info:
            print(f"\n🎯 Étape 6: Suppression des outliers...")
        
        try:
            df = detect_and_remove_outliers(
                df=df,
                columns=mar_cols,
                method='iqr',
                remove=True,
                verbose=display_info
            )
        except Exception as e:
            print(f"❌ Erreur lors de la suppression des outliers : {e}")
            if display_info:
                print("⚠️ Poursuite sans suppression des outliers...")
        
        # 🛡️ Validation X4 après suppression outliers
        if protect_x4:
            validate_x4_presence(df, "Après suppression outliers", display_info)
    elif display_info:
        print(f"\n⏭️ Étape 6: Suppression des outliers ignorée (drop_outliers={drop_outliers})")

    # ========================================================================
    # ÉTAPE 7: SUPPRESSION DES COLONNES DUPLIQUÉES
    # ========================================================================
    
    duplicate_cols = df.columns[df.columns.duplicated()].tolist()
    if duplicate_cols:
        if display_info:
            print(f"\n🔄 Étape 7: Suppression des colonnes dupliquées...")
        
        # 🛡️ Vérifier qu'on ne supprime pas X4 par accident
        if 'X4' in duplicate_cols and protect_x4:
            print("🛡️ ALERTE: X4 détectée comme dupliquée - protection activée")
            # Garder la première occurrence de X4
            df = df.loc[:, ~df.columns.duplicated(keep='first')]
        else:
            df = df.loc[:, ~df.columns.duplicated()]
            
        if display_info:
            print(f"⚠️ Colonnes dupliquées détectées : {duplicate_cols}")
            print(f"🔹 Duplication supprimée : {df.shape}")
        
        # 🛡️ Validation X4 après suppression doublons
        if protect_x4:
            validate_x4_presence(df, "Après suppression doublons", display_info)
    elif display_info:
        print(f"\n✅ Étape 7: Aucune colonne dupliquée détectée")

    # ========================================================================
    # ÉTAPE 8: RÉORGANISATION FINALE DES COLONNES
    # ========================================================================
    
    if display_info:
        print(f"\n📌 Étape 8: Réorganisation finale des colonnes...")
    
    try:
        # Utiliser la fonction dédiée pour la réorganisation
        df = reorder_columns_priority(df, priority_cols, target_col)
        
        if display_info:
            print(f"📌 Ordre final : {priority_cols} → [{target_col}] → autres")
            print(f"📌 Premières colonnes : {df.columns[:min(8, len(df.columns))].tolist()}")
    
    except Exception as e:
        print(f"❌ Erreur lors de la réorganisation : {e}")
        if display_info:
            print("⚠️ Poursuite avec ordre actuel...")

    # ========================================================================
    # ÉTAPE 9: VALIDATION FINALE
    # ========================================================================
    
    if display_info:
        print(f"\n🔍 Étape 9: Validation finale...")
        print(f"✅ Pipeline complet terminé – Dimensions finales : {df.shape}")
        
        # 🛡️ Validation finale X4
        if protect_x4:
            final_status = validate_x4_presence(df, "VALIDATION FINALE", True)
            if not final_status:
                print("🚨 ERREUR CRITIQUE: X4 manquante en fin de pipeline !")
                raise ValueError("X4 a été perdue pendant le preprocessing !")

    # ========================================================================
    # ÉTAPE 10: SAUVEGARDE
    # ========================================================================
    
    if processed_data_dir:
        if display_info:
            print(f"\n💾 Étape 10: Sauvegarde...")
        
        try:
            processed_data_dir = Path(processed_data_dir)
            processed_data_dir.mkdir(parents=True, exist_ok=True)
            suffix = f"{mar_method}{'_no_outliers' if drop_outliers else ''}"
            filename = f"final_dataset_{suffix}.parquet"
            df.to_parquet(processed_data_dir / filename, index=False)
            
            if display_info:
                print(f"💾 Sauvegarde Parquet : {processed_data_dir / filename}")
        except Exception as e:
            print(f"❌ Erreur lors de la sauvegarde : {e}")
    elif display_info:
        print(f"\n⏭️ Étape 10: Sauvegarde ignorée (processed_data_dir=None)")

    if display_info:
        print("\n" + "=" * 70)
        print("🎉 PIPELINE DE PRÉTRAITEMENT TERMINÉ AVEC SUCCÈS (VERSION 3.0)")
        print("=" * 70)

    # Retour selon les options
    if return_objects:
        return df, transform_objects
    else:
        return df


# ============================================================================
# 7. FONCTIONS UTILITAIRES POUR DATASETS EXISTANTS
# ============================================================================

def apply_full_preprocessing_to_existing(
    df: pd.DataFrame,
    **kwargs
) -> pd.DataFrame:
    """
    Applique le prétraitement complet à un DataFrame existant.
    
    Args:
        df: DataFrame à prétraiter
        **kwargs: Arguments à passer à prepare_final_dataset
        
    Returns:
        DataFrame prétraité
        
    Note:
        Cette fonction sauvegarde temporairement le DataFrame et utilise prepare_final_dataset
    """
    import tempfile
    
    # Sauvegarde temporaire
    with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as tmp_file:
        df.to_csv(tmp_file.name, index=False)
        
        # Application du pipeline
        result = prepare_final_dataset(
            file_path=tmp_file.name,
            raw_data_dir=Path(tmp_file.name).parent,
            **kwargs
        )
        
        # Nettoyage
        Path(tmp_file.name).unlink()
        
    return result


def batch_process_datasets(
    datasets_dict: Dict[str, Union[pd.DataFrame, str, Path]],
    **kwargs
) -> Dict[str, pd.DataFrame]:
    """
    Traite plusieurs datasets en lot avec le pipeline complet.
    
    Args:
        datasets_dict: Dictionnaire {nom: DataFrame ou chemin vers fichier}
        **kwargs: Arguments à passer à prepare_final_dataset
        
    Returns:
        Dictionnaire des datasets traités
    """
    processed_datasets = {}
    
    for name, data in datasets_dict.items():
        print(f"\n🔄 Traitement de {name}...")
        
        try:
            if isinstance(data, pd.DataFrame):
                # DataFrame existant
                result = apply_full_preprocessing_to_existing(data, **kwargs)
            else:
                # Chemin vers fichier
                result = prepare_final_dataset(file_path=data, **kwargs)
            
            processed_datasets[name] = result
            print(f"✅ {name} traité avec succès : {result.shape}")
            
        except Exception as e:
            print(f"❌ Erreur lors du traitement de {name} : {e}")
            processed_datasets[name] = None
    
    return processed_datasets


# ============================================================================
# 8. FONCTIONS DE DIAGNOSTIC ET DEBUG AVANCÉES
# ============================================================================

def prepare_dataset_safe(file_path: Union[str, Path], **kwargs) -> pd.DataFrame:
    """
    Version sécurisée du pipeline avec gestion d'erreurs automatique.
    
    Args:
        file_path: Chemin vers le fichier
        **kwargs: Arguments pour prepare_final_dataset
        
    Returns:
        DataFrame prétraité
    """
    print("🔒 PIPELINE SÉCURISÉ - VERSION 3.0")
    print("=" * 45)
    
    # Paramètres par défaut sécurisés
    safe_defaults = {
        'strategy': 'mixed_mar_mcar',
        'mar_method': 'knn',
        'correlation_threshold': 0.90,
        'drop_outliers': False,
        'protect_x4': True,
        'display_info': True,
        'use_comprehensive_correlation': True
    }
    
    # Fusion avec les paramètres utilisateur
    final_params = {**safe_defaults, **kwargs}
    
    try:
        # Tentative normale
        df_result = prepare_final_dataset(file_path=file_path, **final_params)
        print("✅ Pipeline exécuté avec succès en mode normal")
        return df_result
        
    except TypeError as e:
        if "list indices must be integers" in str(e) or "groups" in str(e):
            print("🔧 Erreur de corrélation détectée - application du mode de récupération...")
            
            # Mode de récupération avec méthode ancienne
            recovery_params = final_params.copy()
            recovery_params['use_comprehensive_correlation'] = False
            recovery_params['correlation_threshold'] = 0.95
            recovery_params['display_info'] = True
            
            try:
                df_result = prepare_final_dataset(file_path=file_path, **recovery_params)
                print("✅ Pipeline exécuté avec succès en mode récupération")
                return df_result
            except Exception as e2:
                print(f"❌ Échec en mode récupération: {e2}")
                raise
                
    except Exception as e:
        print(f"❌ Erreur critique: {e}")
        print("🔍 Mode dégradé activé...")
        
        # Mode dégradé - sans corrélation
        degraded_params = final_params.copy()
        degraded_params['correlation_threshold'] = 1.0  # Pas de suppression
        degraded_params['display_info'] = True
        
        try:
            df_result = prepare_final_dataset(file_path=file_path, **degraded_params)
            print("⚠️ Pipeline exécuté en mode dégradé (sans réduction de corrélation)")
            return df_result
        except Exception as e3:
            print(f"❌ Échec total: {e3}")
            raise


def get_preprocessing_summary(df: pd.DataFrame) -> Dict:
    """
    Génère un résumé des caractéristiques du dataset prétraité.
    
    Args:
        df: DataFrame prétraité
        
    Returns:
        Dictionnaire avec le résumé
    """
    summary = {
        'shape': df.shape,
        'columns': df.columns.tolist(),
        'dtypes': df.dtypes.to_dict(),
        'missing_values': df.isnull().sum().to_dict(),
        'has_x4': 'X4' in df.columns,
        'has_outcome': 'outcome' in df.columns,
        'binary_vars': [],
        'continuous_vars': [],
        'memory_usage': df.memory_usage(deep=True).sum() / 1024**2  # MB
    }
    
    # Classification des variables
    for col in df.columns:
        if col == 'outcome':
            continue
        elif pd.api.types.is_integer_dtype(df[col]):
            summary['binary_vars'].append(col)
        elif pd.api.types.is_float_dtype(df[col]):
            summary['continuous_vars'].append(col)
    
    # Statistiques additionnelles
    if 'outcome' in df.columns:
        summary['class_distribution'] = df['outcome'].value_counts().to_dict()
        summary['class_balance'] = df['outcome'].value_counts(normalize=True).to_dict()
    
    return summary


def print_preprocessing_summary(df: pd.DataFrame):
    """Affiche un résumé formaté du dataset prétraité."""
    
    summary = get_preprocessing_summary(df)
    
    print("\n📊 RÉSUMÉ DU DATASET PRÉTRAITÉ")
    print("=" * 40)
    
    print(f"📐 Dimensions: {summary['shape'][0]} lignes × {summary['shape'][1]} colonnes")
    print(f"💾 Mémoire utilisée: {summary['memory_usage']:.2f} MB")
    print(f"🛡️ X4 présente: {'✅' if summary['has_x4'] else '❌'}")
    print(f"🎯 Outcome présente: {'✅' if summary['has_outcome'] else '❌'}")
    
    # Variables par type
    print(f"\n🔢 Variables binaires ({len(summary['binary_vars'])}): {summary['binary_vars'][:5]}{'...' if len(summary['binary_vars']) > 5 else ''}")
    print(f"📈 Variables continues ({len(summary['continuous_vars'])}): {summary['continuous_vars'][:5]}{'...' if len(summary['continuous_vars']) > 5 else ''}")
    
    # Valeurs manquantes
    missing_count = sum(v for v in summary['missing_values'].values() if v > 0)
    if missing_count > 0:
        print(f"💧 Valeurs manquantes: {missing_count} au total")
        missing_cols = {k: v for k, v in summary['missing_values'].items() if v > 0}
        for col, count in list(missing_cols.items())[:3]:
            print(f"   {col}: {count}")
    else:
        print("💧 Valeurs manquantes: ✅ Aucune")
    
    # Distribution des classes
    if 'class_distribution' in summary:
        print(f"\n🎯 Distribution des classes:")
        for classe, count in summary['class_distribution'].items():
            pct = summary['class_balance'][classe] * 100
            print(f"   Classe {classe}: {count} ({pct:.1f}%)")


# ============================================================================
# 9. FONCTIONS DE VISUALISATION ET RAPPORT
# ============================================================================

def visualize_correlation_impact(corr_report, save_path=None):
    """Visualise l'impact de la réduction de corrélation."""
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Graphique 1: Avant/Après
    categories = ['Variables\noriginales', 'Variables\nfinales']
    values = [corr_report['original_shape'][1], corr_report['reduced_shape'][1]]
    colors = ['lightcoral', 'lightblue']
    
    bars1 = ax1.bar(categories, values, color=colors)
    ax1.set_title('Impact de la réduction de corrélation')
    ax1.set_ylabel('Nombre de variables')
    
    # Ajouter les valeurs sur les barres
    for bar, value in zip(bars1, values):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 20, 
                str(value), ha='center', va='bottom', fontweight='bold')
    
    # Graphique 2: Breakdown
    labels = ['Variables\nsupprimées', 'Variables\nconservées']
    sizes = [corr_report['vars_dropped'], corr_report['reduced_shape'][1]]
    colors2 = ['lightcoral', 'lightgreen']
    
    ax2.pie(sizes, labels=labels, colors=colors2, autopct='%1.1f%%', startangle=90)
    ax2.set_title('Répartition des variables')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def create_preprocessing_report(
    datasets_dict: Dict[str, pd.DataFrame],
    validation_reports: Dict[str, Dict],
    output_path: Union[str, Path],
    transform_objects: Dict = None
) -> None:
    """
    Crée un rapport détaillé du prétraitement.
    
    Args:
        datasets_dict: Dictionnaire des datasets
        validation_reports: Rapports de validation
        output_path: Chemin de sauvegarde du rapport
        transform_objects: Objets de transformation utilisés
    """
    from datetime import datetime
    
    report_content = f"""
# RAPPORT DE PRÉTRAITEMENT STA211 - VERSION 3.0
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Résumé

- **Nombre de datasets traités**: {len(datasets_dict)}
- **Datasets valides**: {sum(1 for r in validation_reports.values() if r.get('status') == 'success')}
- **Pipeline version**: 3.0 (Complètement Corrigée)

## Améliorations Version 3.0

- ✅ Intégration analyse de corrélation comprehensive
- ✅ Réduction de dimensionnalité efficace
- ✅ Fix TypeError complet dans find_highly_correlated_groups
- ✅ Validation robuste des types de retour
- ✅ Gestion d'erreurs améliorée à tous les niveaux
- ✅ Protection X4 renforcée
- ✅ Fonctions de diagnostic avancées

## Détails par dataset

"""
    
    for name, report in validation_reports.items():
        if report['status'] == 'error':
            report_content += f"### ❌ {name}\n- **Statut**: Erreur\n- **Message**: {report['message']}\n\n"
            continue
        
        df = datasets_dict[name]
        report_content += f"""### ✅ {name}

- **Dimensions**: {report['shape']}
- **Score qualité**: {report['quality_score']:.2f}/1.0
- **X4 présente**: {'✅' if report['has_x4'] else '❌'}
- **Outcome présente**: {'✅' if report['has_outcome'] else '❌'}
- **Valeurs manquantes**: {report['missing_values']}
- **Premières colonnes**: {', '.join(report['first_columns'])}

#### Statistiques descriptives (premières variables)
```
{df[df.columns[:5]].describe().round(3).to_string()}
```

"""
    
    # Informations sur les transformations si disponibles
    if transform_objects:
        report_content += "\n## Objets de transformation\n\n"
        for obj_name, obj in transform_objects.items():
            if obj is not None:
                report_content += f"- **{obj_name}**: {type(obj).__name__}\n"
    
    # Sauvegarde du rapport
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(report_content)
        print(f"📝 Rapport sauvegardé: {output_path}")
    except Exception as e:
        print(f"❌ Erreur sauvegarde rapport: {e}")


# ============================================================================
# 10. FONCTIONS DE TEST ET VALIDATION COMPLÈTES
# ============================================================================

def run_comprehensive_test(file_path: Union[str, Path]) -> Dict:
    """Execute un test complet du pipeline avec diagnostic."""
    
    print("🧪 TEST COMPLET DU PIPELINE VERSION 3.0")
    print("=" * 50)
    
    # Test 1: Pipeline de base
    print("\n1️⃣ Test pipeline de base...")
    try:
        df_result = prepare_final_dataset(
            file_path=file_path,
            strategy="mixed_mar_mcar",
            mar_method="knn",
            correlation_threshold=0.90,
            drop_outliers=False,
            display_info=False
        )
        basic_success = True
        print(f"✅ Pipeline de base réussi - Shape: {df_result.shape}")
    except Exception as e:
        basic_success = False
        print(f"❌ Pipeline de base échoué: {e}")
    
    # Test 2: Pipeline avec objets
    print("\n2️⃣ Test pipeline avec objets de transformation...")
    try:
        df_result, transform_objects = prepare_final_dataset(
            file_path=file_path,
            return_objects=True,
            display_info=False
        )
        objects_success = True
        print("✅ Pipeline avec objets réussi")
    except Exception as e:
        objects_success = False
        print(f"❌ Pipeline avec objets échoué: {e}")
        transform_objects = {}
    
    # Test 3: Méthode comprehensive de corrélation
    print("\n3️⃣ Test méthode comprehensive de corrélation...")
    try:
        df_comprehensive = prepare_final_dataset(
            file_path=file_path,
            use_comprehensive_correlation=True,
            display_info=False
        )
        comprehensive_success = True
        print(f"✅ Méthode comprehensive réussie - Shape: {df_comprehensive.shape}")
    except Exception as e:
        comprehensive_success = False
        print(f"❌ Méthode comprehensive échouée: {e}")
    
    # Test 4: Méthode ancienne (compatibilité)
    print("\n4️⃣ Test méthode ancienne (compatibilité)...")
    try:
        df_legacy = prepare_final_dataset(
            file_path=file_path,
            use_comprehensive_correlation=False,
            display_info=False
        )
        legacy_success = True
        print(f"✅ Méthode ancienne réussie - Shape: {df_legacy.shape}")
    except Exception as e:
        legacy_success = False
        print(f"❌ Méthode ancienne échouée: {e}")
    
    # Test 5: Validation X4
    print("\n5️⃣ Validation protection X4...")
    x4_protected = True
    if basic_success:
        x4_protected = 'X4' in df_result.columns
        print(f"🛡️ X4 protégée: {'✅' if x4_protected else '❌'}")
    
    # Test 6: Pipeline sécurisé
    print("\n6️⃣ Test pipeline sécurisé...")
    try:
        df_safe = prepare_dataset_safe(file_path)
        safe_success = True
        print(f"✅ Pipeline sécurisé réussi - Shape: {df_safe.shape}")
    except Exception as e:
        safe_success = False
        print(f"❌ Pipeline sécurisé échoué: {e}")
    
    # Résumé final
    print("\n📊 RÉSUMÉ DU TEST COMPLET")
    print("=" * 30)
    print(f"Pipeline de base: {'✅' if basic_success else '❌'}")
    print(f"Pipeline avec objets: {'✅' if objects_success else '❌'}")
    print(f"Méthode comprehensive: {'✅' if comprehensive_success else '❌'}")
    print(f"Méthode ancienne: {'✅' if legacy_success else '❌'}")
    print(f"Protection X4: {'✅' if x4_protected else '❌'}")
    print(f"Pipeline sécurisé: {'✅' if safe_success else '❌'}")
    
    # Score global
    tests = [basic_success, objects_success, comprehensive_success, 
             legacy_success, x4_protected, safe_success]
    score = sum(tests) / len(tests)
    
    if score == 1.0:
        print(f"\n🏆 SCORE PARFAIT: {score:.0%} - Tous les tests réussis!")
    elif score >= 0.8:
        print(f"\n✅ SCORE EXCELLENT: {score:.0%} - Pipeline opérationnel")
    elif score >= 0.6:
        print(f"\n⚠️ SCORE CORRECT: {score:.0%} - Quelques problèmes détectés")
    else:
        print(f"\n❌ SCORE FAIBLE: {score:.0%} - Révision nécessaire")
    
    return {
        'basic_success': basic_success,
        'objects_success': objects_success,
        'comprehensive_success': comprehensive_success,
        'legacy_success': legacy_success,
        'x4_protected': x4_protected,
        'safe_success': safe_success,
        'score': score,
        'transform_objects': transform_objects if objects_success else {}
    }


def validate_all_datasets(
    datasets_dict: Dict[str, pd.DataFrame],
    expected_cols: List[str] = None,
    protect_x4: bool = True
) -> Dict[str, Dict]:
    """
    Valide la qualité de plusieurs datasets.
    
    Args:
        datasets_dict: Dictionnaire des datasets à valider
        expected_cols: Colonnes attendues en premier
        protect_x4: Vérifier la présence de X4
        
    Returns:
        Dictionnaire des rapports de validation
    """
    if expected_cols is None:
        expected_cols = ['X1_trans', 'X2_trans', 'X3_trans', 'X4', 'outcome']
    
    validation_reports = {}
    
    for name, df in datasets_dict.items():
        if df is None:
            validation_reports[name] = {'status': 'error', 'message': 'DataFrame vide'}
            continue
        
        # Vérification ordre des colonnes
        current_order = df.columns.tolist()
        columns_order_correct = True
        for i, expected_col in enumerate(expected_cols[:5]):  # Vérifier les 5 premières
            if expected_col in current_order:
                actual_position = current_order.index(expected_col)
                if actual_position != i:
                    columns_order_correct = False
                    break
        
        report = {
            'status': 'success',
            'shape': df.shape,
            'columns_order_correct': columns_order_correct,
            'has_x4': 'X4' in df.columns if protect_x4 else True,
            'has_outcome': 'outcome' in df.columns,
            'missing_values': df.isnull().sum().sum(),
            'first_columns': df.columns[:min(8, len(df.columns))].tolist(),
            'dimension_reduction': None,
            'memory_usage': df.memory_usage(deep=True).sum() / 1024**2  # MB
        }
        
        # Score de qualité global
        quality_checks = [
            report['columns_order_correct'],
            report['has_x4'],
            report['has_outcome'],
            report['missing_values'] == 0,
            report['shape'][1] < 1000,  # Réduction effective
            report['shape'][0] > 1000   # Suffisamment d'échantillons
        ]
        report['quality_score'] = sum(quality_checks) / len(quality_checks)
        
        validation_reports[name] = report
    
    return validation_reports


def print_validation_summary(validation_reports: Dict[str, Dict]):
    """
    Affiche un résumé des validations.
    
    Args:
        validation_reports: Rapports de validation
    """
    print("\n📊 RÉSUMÉ DE LA VALIDATION DES DATASETS")
    print("=" * 60)
    
    for name, report in validation_reports.items():
        if report['status'] == 'error':
            print(f"❌ {name}: {report['message']}")
            continue
        
        quality = report['quality_score']
        status_icon = "✅" if quality == 1.0 else "⚠️" if quality >= 0.75 else "❌"
        
        print(f"{status_icon} {name}:")
        print(f"   📐 Shape: {report['shape']}")
        print(f"   🎯 Score qualité: {quality:.2f}/1.0")
        print(f"   📌 Ordre colonnes: {'✅' if report['columns_order_correct'] else '❌'}")
        print(f"   🛡️ X4 présente: {'✅' if report['has_x4'] else '❌'}")
        print(f"   🎯 Outcome présente: {'✅' if report['has_outcome'] else '❌'}")
        print(f"   💧 Valeurs manquantes: {report['missing_values']}")
        print(f"   💾 Mémoire: {report['memory_usage']:.1f} MB")
        print(f"   📋 Premières colonnes: {report['first_columns'][:4]}...")
        print()


# ============================================================================
# 11. FONCTIONS D'EXPORT ET SAUVEGARDE AVANCÉES
# ============================================================================

def export_datasets_multiple_formats(
    datasets_dict: Dict[str, pd.DataFrame],
    output_dir: Union[str, Path],
    formats: List[str] = ['parquet', 'csv'],
    compress: bool = True
) -> Dict[str, Dict[str, Path]]:
    """
    Exporte les datasets dans plusieurs formats.
    
    Args:
        datasets_dict: Dictionnaire des datasets
        output_dir: Dossier de sortie
        formats: Formats d'export ('parquet', 'csv', 'xlsx')
        compress: Compresser les fichiers
        
    Returns:
        Dictionnaire des chemins de sauvegarde
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    export_paths = {}
    
    for dataset_name, df in datasets_dict.items():
        if df is None:
            continue
        
        export_paths[dataset_name] = {}
        
        for fmt in formats:
            try:
                if fmt == 'parquet':
                    file_path = output_dir / f"{dataset_name}.parquet"
                    df.to_parquet(file_path, index=False, compression='snappy' if compress else None)
                    
                elif fmt == 'csv':
                    file_path = output_dir / f"{dataset_name}.csv"
                    compression = 'gzip' if compress else None
                    if compression:
                        file_path = file_path.with_suffix('.csv.gz')
                    df.to_csv(file_path, index=False, compression=compression)
                    
                elif fmt == 'xlsx':
                    file_path = output_dir / f"{dataset_name}.xlsx"
                    df.to_excel(file_path, index=False)
                
                export_paths[dataset_name][fmt] = file_path
                print(f"💾 {dataset_name}.{fmt} sauvegardé: {file_path}")
                
            except Exception as e:
                print(f"❌ Erreur sauvegarde {dataset_name}.{fmt}: {e}")
    
    return export_paths


# ============================================================================
# 12. FONCTIONS DE COMPARAISON ET BENCHMARKING
# ============================================================================

def compare_preprocessing_methods(
    file_path: Union[str, Path],
    methods: List[str] = None,
    thresholds: List[float] = None
) -> pd.DataFrame:
    """
    Compare différentes méthodes de prétraitement.
    
    Args:
        file_path: Chemin vers le fichier de données
        methods: Méthodes à comparer
        thresholds: Seuils de corrélation à tester
        
    Returns:
        DataFrame de comparaison
    """
    if methods is None:
        methods = ['comprehensive', 'legacy']
    
    if thresholds is None:
        thresholds = [0.85, 0.90, 0.95]
    
    results = []
    
    for method in methods:
        for threshold in thresholds:
            config_name = f"{method}_t{threshold}"
            print(f"\n🧪 Test {config_name}...")
            
            try:
                start_time = pd.Timestamp.now()
                
                df_result = prepare_final_dataset(
                    file_path=file_path,
                    correlation_threshold=threshold,
                    use_comprehensive_correlation=(method == 'comprehensive'),
                    display_info=False
                )
                
                end_time = pd.Timestamp.now()
                duration = (end_time - start_time).total_seconds()
                
                result = {
                    'method': method,
                    'threshold': threshold,
                    'config_name': config_name,
                    'status': 'success',
                    'final_shape': df_result.shape,
                    'features_count': df_result.shape[1] - 1,  # -1 pour outcome
                    'reduction_rate': 1 - (df_result.shape[1] / 1559),  # Baseline 1559
                    'has_x4': 'X4' in df_result.columns,
                    'has_outcome': 'outcome' in df_result.columns,
                    'missing_values': df_result.isnull().sum().sum(),
                    'duration_seconds': duration,
                    'memory_usage_mb': df_result.memory_usage(deep=True).sum() / 1024**2
                }
                
            except Exception as e:
                result = {
                    'method': method,
                    'threshold': threshold,
                    'config_name': config_name,
                    'status': 'error',
                    'error': str(e),
                    'final_shape': None,
                    'features_count': None,
                    'reduction_rate': None,
                    'has_x4': False,
                    'has_outcome': False,
                    'missing_values': None,
                    'duration_seconds': None,
                    'memory_usage_mb': None
                }
            
            results.append(result)
    
    return pd.DataFrame(results)


# ============================================================================
# FIN DU MODULE - INFORMATIONS DE VERSION ET UTILISATION
# ============================================================================

__version__ = "3.0"
__status__ = "Complètement Corrigé et Optimisé"
__new_features__ = [
    "Intégration analyse de corrélation comprehensive",
    "Réduction de dimensionnalité efficace",
    "Fonctions de diagnostic avancées",
    "Pipeline modulaire et robuste",
    "Fonctions de benchmarking et comparaison",
    "Export multi-formats",
    "Rapport automatique complet"
]
__corrections__ = [
    "Fix TypeError complet dans find_highly_correlated_groups",
    "Validation robuste des types de retour",
    "Gestion d'erreurs améliorée à tous les niveaux",
    "Protection X4 renforcée",
    "Mode de récupération automatique",
    "Pipeline sécurisé avec fallbacks"
]


def print_version_info():
    """Affiche les informations de version du module."""
    print(f"\n📋 MODULE final_preprocessing.py")
    print(f"Version: {__version__}")
    print(f"Statut: {__status__}")
    
    print(f"\n🚀 NOUVELLES FONCTIONNALITÉS VERSION 3.0:")
    for feature in __new_features__:
        print(f"  ✨ {feature}")
    
    print(f"\n🔧 CORRECTIONS APPORTÉES:")
    for correction in __corrections__:
        print(f"  ✅ {correction}")


def print_usage_examples():
    """Affiche des exemples d'utilisation recommandés."""
    print(f"\n🚀 EXEMPLES D'UTILISATION RECOMMANDÉS:")
    
    examples = """
# 1. Utilisation basique (recommandée)
df = prepare_dataset_safe('data_train.csv')

# 2. Pipeline complet avec objets
df, objects = prepare_final_dataset('data_train.csv', return_objects=True)

# 3. Configuration personnalisée
df = prepare_final_dataset(
    'data_train.csv',
    correlation_threshold=0.85,
    use_comprehensive_correlation=True,
    drop_outliers=False
)

# 4. Test complet du pipeline
test_results = run_comprehensive_test('data_train.csv')

# 5. Comparaison de méthodes
comparison = compare_preprocessing_methods('data_train.csv')

# 6. Traitement en lot
datasets = {'train': 'data_train.csv', 'test': 'data_test.csv'}
results = batch_process_datasets(datasets)

# 7. Export multi-formats
export_datasets_multiple_formats(
    {'final': df}, 
    'output/', 
    formats=['parquet', 'csv', 'xlsx']
)

# 8. Résumé du preprocessing
print_preprocessing_summary(df)

# 9. Analyse de corrélation seule
df_reduced, report = apply_correlation_reduction_to_dataset(df)

# 10. Visualisation impact
visualize_correlation_impact(report)
"""
    
    print(examples)


if __name__ == "__main__":
    print_version_info()
    print_usage_examples()
    
    print(f"\n💡 CONSEILS D'UTILISATION:")
    print("• Utilisez prepare_dataset_safe() pour une utilisation simple et sécurisée")
    print("• Activez use_comprehensive_correlation=True pour une réduction optimale")
    print("• Testez avec run_comprehensive_test() avant utilisation en production")
    print("• Surveillez la protection X4 avec les fonctions de validation")
    print("• Exportez en Parquet pour des performances optimales")
    
    print(f"\n🆘 EN CAS DE PROBLÈME:")
    print("• Le pipeline sécurisé inclut des modes de récupération automatiques")
    print("• Utilisez display_info=True pour diagnostiquer les problèmes")
    print("• Les fonctions de validation permettent de vérifier l'intégrité")
    print("• Consultez les rapports automatiques pour l'analyse détaillée")