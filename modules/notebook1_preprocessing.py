"""
Module de prétraitement pour le Notebook 1 - EDA et préparation des données
Extrait et adapté de data_processing.py

Fonctionnalités:
- Chargement et nettoyage des données
- Analyse exploratoire (EDA) 
- Imputations KNN et MICE
- Transformations optimales (Yeo-Johnson, Box-Cox)
- Détection et traitement des outliers
- Génération des features polynomiales

Auteur: Maoulida Abdoullatuf  
Version: 1.0 (restructuré)
"""

import os
import re
import time
import chardet
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Union
from IPython.display import display, Markdown

# Imports du projet
from modules.config import cfg
from modules.utils.storage import save_artifact, load_artifact

# Scikit-learn imports
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer, KNNImputer
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.linear_model import BayesianRidge
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder, PowerTransformer
from scipy import stats

warnings.filterwarnings('ignore', category=pd.errors.DtypeWarning)

# Configuration du logging
log = cfg.get_logger(__name__)

# =============================================================================
# 1. CHARGEMENT ET NETTOYAGE DES DONNÉES
# =============================================================================

def load_and_clean_data(
    file_path: Union[str, Path], 
    target_col: str = "outcome",
    display_info: bool = True,
    encoding: Optional[str] = None,
    max_file_size_mb: float = 500
) -> Tuple[pd.DataFrame, Dict]:
    """
    Charge un fichier CSV proprement et vérifie la présence de la colonne cible.
    
    Args:
        file_path: Chemin vers le fichier CSV
        target_col: Nom de la colonne cible  
        display_info: Afficher les informations de chargement
        encoding: Encodage du fichier (auto-détecté si None)
        max_file_size_mb: Taille maximale autorisée en MB
        
    Returns:
        Tuple contenant le DataFrame et un dictionnaire d'informations
    """
    
    start_time = time.time()
    file_path = Path(file_path)
    
    # Vérifications préliminaires
    if not file_path.exists():
        raise FileNotFoundError(f"Fichier non trouvé : {file_path}")
        
    file_size_mb = file_path.stat().st_size / (1024 * 1024)
    if file_size_mb > max_file_size_mb:
        raise ValueError(f"Fichier trop volumineux : {file_size_mb:.1f}MB > {max_file_size_mb}MB")
    
    # Auto-détection de l'encodage si nécessaire
    if encoding is None:
        with open(file_path, 'rb') as f:
            raw_data = f.read(10000)  # Lire les premiers 10KB
            encoding_info = chardet.detect(raw_data)
            encoding = encoding_info['encoding']
            if display_info:
                print(f"🔍 Encodage détecté : {encoding} (confiance: {encoding_info['confidence']:.2%})")
    
    # Chargement des données
    try:
        df = pd.read_csv(file_path, encoding=encoding)
        load_time = time.time() - start_time
        
        if display_info:
            print(f"✅ Données chargées avec succès !")
            print(f"📊 Shape: {df.shape}")
            print(f"⏱️  Temps de chargement: {load_time:.2f}s")
            print(f"💾 Mémoire utilisée: {df.memory_usage(deep=True).sum() / 1024 / 1024:.1f} MB")
        
        # Vérification de la colonne cible
        target_present = target_col in df.columns
        if not target_present and display_info:
            print(f"⚠️  Colonne cible '{target_col}' non trouvée (mode test/prédiction)")
            
        # Informations de base
        info_dict = {
            'shape': df.shape,
            'columns': list(df.columns),
            'dtypes': df.dtypes.to_dict(),
            'target_present': target_present,
            'missing_values': df.isnull().sum().sum(),
            'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024 / 1024,
            'load_time_seconds': load_time,
            'file_size_mb': file_size_mb
        }
        
        return df, info_dict
        
    except Exception as e:
        raise Exception(f"Erreur lors du chargement : {str(e)}")

# =============================================================================
# 2. ANALYSE EXPLORATOIRE (EDA)
# =============================================================================

def analyze_missing_patterns(df: pd.DataFrame, target_col: Optional[str] = None, 
                           save_path: Optional[Path] = None, figsize: Tuple[int, int] = (15, 10)) -> Dict:
    """
    Analyse complète des patterns de valeurs manquantes.
    
    Args:
        df: DataFrame à analyser
        target_col: Colonne cible (optionnelle)
        save_path: Chemin pour sauvegarder les graphiques
        figsize: Taille des graphiques
        
    Returns:
        Dictionnaire avec les statistiques des valeurs manquantes
    """
    
    print("🔍 ANALYSE DES VALEURS MANQUANTES")
    print("=" * 50)
    
    # Statistiques générales
    missing_stats = {}
    total_cells = df.shape[0] * df.shape[1]
    total_missing = df.isnull().sum().sum()
    
    missing_stats['total_cells'] = total_cells
    missing_stats['total_missing'] = total_missing
    missing_stats['missing_percentage'] = (total_missing / total_cells) * 100
    
    print(f"📊 Cellules totales : {total_cells:,}")
    print(f"🕳️  Valeurs manquantes : {total_missing:,} ({missing_stats['missing_percentage']:.2f}%)")
    
    # Par colonne
    missing_by_col = df.isnull().sum()
    missing_by_col = missing_by_col[missing_by_col > 0].sort_values(ascending=False)
    
    if len(missing_by_col) > 0:
        print(f"\n📈 Colonnes avec valeurs manquantes ({len(missing_by_col)}/{len(df.columns)}):")
        
        missing_df = pd.DataFrame({
            'Missing_Count': missing_by_col,
            'Missing_Percentage': (missing_by_col / len(df)) * 100
        })
        
        # Top 10 des colonnes avec le plus de valeurs manquantes
        top_missing = missing_df.head(10)
        for col, row in top_missing.iterrows():
            print(f"  • {col}: {row['Missing_Count']:,} ({row['Missing_Percentage']:.1f}%)")
            
        missing_stats['by_column'] = missing_df.to_dict()
        
        # Visualisations
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        
        # 1. Heatmap des valeurs manquantes
        if len(df.columns) <= 50:  # Éviter les heatmaps trop denses
            sns.heatmap(df.isnull(), cbar=True, ax=axes[0,0], cmap='viridis')
            axes[0,0].set_title('Pattern des valeurs manquantes')
        else:
            # Prendre un échantillon de colonnes
            sample_cols = missing_by_col.head(20).index.tolist()
            sns.heatmap(df[sample_cols].isnull(), cbar=True, ax=axes[0,0], cmap='viridis')
            axes[0,0].set_title('Pattern des valeurs manquantes (top 20 colonnes)')
        
        # 2. Distribution du pourcentage de valeurs manquantes par colonne
        axes[0,1].hist(missing_df['Missing_Percentage'], bins=20, edgecolor='black', alpha=0.7)
        axes[0,1].set_xlabel('Pourcentage de valeurs manquantes')
        axes[0,1].set_ylabel('Nombre de colonnes')
        axes[0,1].set_title('Distribution des taux de valeurs manquantes')
        
        # 3. Top 15 colonnes avec valeurs manquantes
        top_15 = missing_df.head(15)
        axes[1,0].barh(range(len(top_15)), top_15['Missing_Percentage'])
        axes[1,0].set_yticks(range(len(top_15)))
        axes[1,0].set_yticklabels([col[:20] + '...' if len(col) > 20 else col for col in top_15.index])
        axes[1,0].set_xlabel('Pourcentage manquant')
        axes[1,0].set_title('Top 15 colonnes avec valeurs manquantes')
        
        # 4. Valeurs manquantes par ligne
        missing_by_row = df.isnull().sum(axis=1)
        axes[1,1].hist(missing_by_row, bins=30, edgecolor='black', alpha=0.7)
        axes[1,1].set_xlabel('Nombre de valeurs manquantes par ligne')
        axes[1,1].set_ylabel('Nombre de lignes')
        axes[1,1].set_title('Distribution des valeurs manquantes par ligne')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path / "missing_values_analysis.png", dpi=300, bbox_inches='tight')
            print(f"💾 Graphique sauvegardé : {save_path / 'missing_values_analysis.png'}")
        
        plt.show()
        
    else:
        print("✅ Aucune valeur manquante détectée !")
        missing_stats['by_column'] = {}
    
    return missing_stats


# =============================================================================
# 3. IMPUTATIONS
# =============================================================================

def perform_knn_imputation(df: pd.DataFrame, 
                          cols_to_impute: List[str],
                          n_neighbors: int = 7,
                          save_imputer: bool = True,
                          random_state: int = 42) -> Tuple[pd.DataFrame, object]:
    """
    Effectue une imputation KNN sur les colonnes spécifiées.
    
    Args:
        df: DataFrame avec valeurs manquantes
        cols_to_impute: Liste des colonnes à imputer  
        n_neighbors: Nombre de voisins pour KNN
        save_path: Chemin pour sauvegarder l'imputer
        random_state: Graine aléatoire
        
    Returns:
        Tuple (DataFrame imputé, imputer entraîné)
    """
    
    print(f"🔧 IMPUTATION KNN (k={n_neighbors})")
    print("=" * 40)
    
    df_imputed = df.copy()
    
    # Vérification des colonnes
    missing_cols = []
    for col in cols_to_impute:
        if col not in df.columns:
            print(f"⚠️  Colonne '{col}' non trouvée, ignorée")
            continue
        missing_count = df[col].isnull().sum()
        if missing_count > 0:
            missing_cols.append(col)
            print(f"  • {col}: {missing_count:,} valeurs manquantes ({missing_count/len(df)*100:.1f}%)")
    
    if not missing_cols:
        print("✅ Aucune valeur manquante dans les colonnes spécifiées")
        return df_imputed, None
    
    # Configuration de l'imputer KNN
    knn_imputer = KNNImputer(
        n_neighbors=n_neighbors,
        weights='uniform'  # Poids uniformes pour plus de stabilité
    )
    
    # Imputation
    start_time = time.time()
    imputed_values = knn_imputer.fit_transform(df[missing_cols])
    impute_time = time.time() - start_time
    
    # Mise à jour du DataFrame
    df_imputed[missing_cols] = imputed_values
    
    print(f"✅ Imputation terminée en {impute_time:.2f}s")
    print(f"🎯 Colonnes imputées: {len(missing_cols)}")
    
    # Vérification post-imputation
    remaining_missing = df_imputed[missing_cols].isnull().sum().sum()
    if remaining_missing > 0:
        print(f"⚠️  {remaining_missing} valeurs toujours manquantes après imputation")
    else:
        print("✅ Toutes les valeurs manquantes ont été imputées")
    
    # Sauvegarde de l'imputer
    if save_imputer and knn_imputer is not None:
        filename = f"imputer_knn_k{n_neighbors}.pkl"
        save_artifact(knn_imputer, filename, cfg.paths.imputers)
        log.info(f"💾 Imputer KNN sauvegardé : {filename}")
    
    return df_imputed, knn_imputer


def perform_mice_imputation(df: pd.DataFrame,
                           cols_to_impute: List[str],
                           max_iter: int = 10,
                           estimator=None,
                           save_imputer: bool = True,
                           random_state: int = 42) -> Tuple[pd.DataFrame, object]:
    """
    Effectue une imputation MICE (Multiple Imputation by Chained Equations).
    
    Args:
        df: DataFrame avec valeurs manquantes
        cols_to_impute: Liste des colonnes à imputer
        max_iter: Nombre maximum d'itérations
        estimator: Estimateur à utiliser (BayesianRidge par défaut)
        save_path: Chemin pour sauvegarder l'imputer
        random_state: Graine aléatoire
        
    Returns:
        Tuple (DataFrame imputé, imputer entraîné)
    """
    
    print(f"🔧 IMPUTATION MICE (max_iter={max_iter})")
    print("=" * 40)
    
    df_imputed = df.copy()
    
    # Estimateur par défaut
    if estimator is None:
        estimator = BayesianRidge()
        print("🧠 Estimateur: BayesianRidge (défaut)")
    else:
        print(f"🧠 Estimateur: {type(estimator).__name__}")
    
    # Vérification des colonnes
    missing_cols = []
    for col in cols_to_impute:
        if col not in df.columns:
            print(f"⚠️  Colonne '{col}' non trouvée, ignorée")
            continue
        missing_count = df[col].isnull().sum()
        if missing_count > 0:
            missing_cols.append(col)
            print(f"  • {col}: {missing_count:,} valeurs manquantes ({missing_count/len(df)*100:.1f}%)")
    
    if not missing_cols:
        print("✅ Aucune valeur manquante dans les colonnes spécifiées")
        return df_imputed, None
    
    # Configuration de l'imputer MICE
    mice_imputer = IterativeImputer(
        estimator=estimator,
        max_iter=max_iter,
        random_state=random_state,
        verbose=1  # Pour suivre la progression
    )
    
    # Imputation
    start_time = time.time()
    print("🔄 Démarrage de l'imputation MICE...")
    imputed_values = mice_imputer.fit_transform(df[missing_cols])
    impute_time = time.time() - start_time
    
    # Mise à jour du DataFrame
    df_imputed[missing_cols] = imputed_values
    
    print(f"✅ Imputation terminée en {impute_time:.2f}s")
    print(f"🎯 Colonnes imputées: {len(missing_cols)}")
    print(f"🔄 Itérations effectuées: {mice_imputer.n_iter_}")
    
    # Vérification post-imputation
    remaining_missing = df_imputed[missing_cols].isnull().sum().sum()
    if remaining_missing > 0:
        print(f"⚠️  {remaining_missing} valeurs toujours manquantes après imputation")
    else:
        print("✅ Toutes les valeurs manquantes ont été imputées")
    
    # Sauvegarde de l'imputer
    if save_imputer and mice_imputer is not None:
        filename = f"imputer_mice_maxiter{max_iter}.pkl"
        save_artifact(mice_imputer, filename, cfg.paths.imputers)
        log.info(f"💾 Imputer MICE sauvegardé : {filename}")
    
    return df_imputed, mice_imputer

# =============================================================================
# 4. TRANSFORMATIONS OPTIMALES
# =============================================================================

def apply_optimal_transformations(df: pd.DataFrame,
                                 continuous_cols: List[str],
                                 method_mapping: Dict[str, str],
                                 save_transformers: bool = True) -> Tuple[pd.DataFrame, Dict]:
    """
    Applique les transformations optimales (Yeo-Johnson, Box-Cox) aux colonnes.
    
    Args:
        df: DataFrame à transformer
        continuous_cols: Liste des colonnes continues
        method_mapping: Dict {colonne: méthode} où méthode in ['yeo-johnson', 'box-cox']
        save_path: Chemin pour sauvegarder les transformateurs
        
    Returns:
        Tuple (DataFrame transformé, dict des transformateurs)
    """
    
    print("🔄 APPLICATION DES TRANSFORMATIONS OPTIMALES")
    print("=" * 50)
    
    df_transformed = df.copy()
    transformers = {}
    
    for col in continuous_cols:
        if col not in df.columns:
            print(f"⚠️  Colonne '{col}' non trouvée, ignorée")
            continue
            
        method = method_mapping.get(col, 'yeo-johnson')  # défaut
        print(f"🔧 {col}: {method}")
        
        if method == 'yeo-johnson':
            transformer = PowerTransformer(method='yeo-johnson', standardize=False)
        elif method == 'box-cox':
            transformer = PowerTransformer(method='box-cox', standardize=False) 
            # S'assurer que les valeurs sont positives pour Box-Cox
            min_val = df[col].min()
            if min_val <= 0:
                shift = abs(min_val) + 1e-6
                df_transformed[col] = df_transformed[col] + shift
                print(f"  📈 Décalage appliqué: +{shift:.6f}")
        else:
            print(f"  ⚠️ Méthode '{method}' non supportée, Yeo-Johnson utilisée")
            transformer = PowerTransformer(method='yeo-johnson', standardize=False)
        
        # Application de la transformation
        try:
            df_transformed[[col]] = transformer.fit_transform(df_transformed[[col]])
            transformers[col] = transformer
            print(f"  ✅ Transformation appliquée")
            
            # Renommer la colonne pour indiquer la transformation
            new_col_name = f"{col}_transformed"
            df_transformed = df_transformed.rename(columns={col: new_col_name})
            transformers[new_col_name] = transformers.pop(col)  # Update key
            
        except Exception as e:
            print(f"  ❌ Erreur: {str(e)}")
            continue
    
    # Sauvegarde des transformateurs
    if save_transformers and transformers:
        # Sauvegarder par type de transformation  
        yj_transformers = {k: v for k, v in transformers.items() 
                          if hasattr(v, 'method') and v.method == 'yeo-johnson'}
        bc_transformers = {k: v for k, v in transformers.items() 
                          if hasattr(v, 'method') and v.method == 'box-cox'}
        
        if yj_transformers:
            save_artifact(yj_transformers, "yeo_johnson_transformers.pkl", cfg.paths.transformers)
            log.info("💾 Transformateurs Yeo-Johnson sauvegardés")
            
        if bc_transformers:
            save_artifact(bc_transformers, "box_cox_transformers.pkl", cfg.paths.transformers)
            log.info("💾 Transformateurs Box-Cox sauvegardés")
    
    print(f"✅ Transformations terminées: {len(transformers)} colonnes")
    return df_transformed, transformers

# =============================================================================
# 5. DÉTECTION ET TRAITEMENT DES OUTLIERS
# =============================================================================

def detect_and_cap_outliers(df: pd.DataFrame,
                           cols_to_process: List[str], 
                           method: str = 'iqr',
                           factor: float = 1.5,
                           save_params: bool = True) -> Tuple[pd.DataFrame, Dict]:
    """
    Détecte et traite les outliers par capping (écrêtage).
    
    Args:
        df: DataFrame à traiter
        cols_to_process: Colonnes à traiter
        method: 'iqr' ou 'zscore'
        factor: Facteur multiplicateur (1.5 pour IQR, 3 pour z-score)
        save_params_path: Chemin pour sauvegarder les paramètres de capping
        
    Returns:
        Tuple (DataFrame avec outliers traités, paramètres de capping)
    """
    
    print(f"🎯 DÉTECTION ET TRAITEMENT DES OUTLIERS ({method.upper()})")
    print("=" * 50)
    
    df_capped = df.copy()
    capping_params = {}
    
    for col in cols_to_process:
        if col not in df.columns:
            print(f"⚠️  Colonne '{col}' non trouvée, ignorée")
            continue
            
        print(f"🔍 Analyse de {col}:")
        
        if method == 'iqr':
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - factor * IQR
            upper_bound = Q3 + factor * IQR
            
        elif method == 'zscore':
            mean_val = df[col].mean()
            std_val = df[col].std()
            lower_bound = mean_val - factor * std_val
            upper_bound = mean_val + factor * std_val
            
        else:
            print(f"  ❌ Méthode '{method}' non supportée")
            continue
        
        # Compter les outliers avant traitement
        outliers_mask = (df[col] < lower_bound) | (df[col] > upper_bound)
        n_outliers = outliers_mask.sum()
        outlier_percentage = (n_outliers / len(df)) * 100
        
        print(f"  📊 Outliers détectés: {n_outliers} ({outlier_percentage:.1f}%)")
        print(f"  📏 Bornes: [{lower_bound:.3f}, {upper_bound:.3f}]")
        
        if n_outliers > 0:
            # Appliquer le capping
            df_capped[col] = df_capped[col].clip(lower=lower_bound, upper=upper_bound)
            
            # Sauvegarder les paramètres
            capping_params[col] = (lower_bound, upper_bound)
            
            print(f"  ✅ Capping appliqué")
        else:
            print(f"  ℹ️  Aucun outlier détecté")
    
    # Sauvegarde des paramètres
    if save_params and capping_params:
        filename = f"capping_params_{method}_factor{factor}.pkl"
        save_artifact(capping_params, filename, cfg.paths.transformers)
        log.info(f"💾 Paramètres de capping sauvegardés : {filename}")
    
    print(f"✅ Traitement terminé: {len(capping_params)} colonnes traitées")
    return df_capped, capping_params

# =============================================================================
# 6. GÉNÉRATION DE FEATURES POLYNOMIALES
# =============================================================================

def generate_polynomial_features(df: pd.DataFrame,
                                feature_cols: List[str],
                                degree: int = 2,
                                interaction_only: bool = False,
                                save_transformer: bool = True) -> Tuple[pd.DataFrame, object]:
    """
    Génère des features polynomiales à partir des colonnes spécifiées.
    
    Args:
        df: DataFrame source
        feature_cols: Colonnes pour générer les features
        degree: Degré polynomial
        interaction_only: Inclure seulement les interactions (pas les puissances)
        save_path: Chemin pour sauvegarder le transformateur
        
    Returns:
        Tuple (DataFrame avec nouvelles features, transformateur polynomial)
    """
    
    from sklearn.preprocessing import PolynomialFeatures
    
    print(f"🔢 GÉNÉRATION DE FEATURES POLYNOMIALES (degré={degree})")
    print("=" * 50)
    
    # Vérification des colonnes
    valid_cols = [col for col in feature_cols if col in df.columns]
    if not valid_cols:
        print("❌ Aucune colonne valide trouvée")
        return df, None
        
    missing_cols = [col for col in feature_cols if col not in df.columns]
    if missing_cols:
        print(f"⚠️  Colonnes manquantes ignorées: {missing_cols}")
    
    print(f"📊 Colonnes utilisées: {valid_cols}")
    
    # Configuration du générateur polynomial
    poly_transformer = PolynomialFeatures(
        degree=degree,
        interaction_only=interaction_only,
        include_bias=False  # Pas de terme constant
    )
    
    # Génération des features
    start_time = time.time()
    poly_features = poly_transformer.fit_transform(df[valid_cols])
    generation_time = time.time() - start_time
    
    # Noms des nouvelles features
    feature_names = poly_transformer.get_feature_names_out(valid_cols)
    
    # Création du nouveau DataFrame
    df_with_poly = df.copy()
    
    # Ajouter les nouvelles features (exclure les features originales)
    n_original = len(valid_cols)
    new_features = poly_features[:, n_original:]  # Exclure les colonnes originales
    new_feature_names = feature_names[n_original:]
    
    for i, name in enumerate(new_feature_names):
        df_with_poly[name] = new_features[:, i]
    
    print(f"✅ Génération terminée en {generation_time:.2f}s")
    print(f"🎯 Features créées: {len(new_feature_names)}")
    print(f"📈 Shape finale: {df_with_poly.shape}")
    
    # Quelques exemples de nouvelles features
    if new_feature_names:
        print(f"📝 Exemples de nouvelles features:")
        for name in new_feature_names[:5]:  # Top 5
            print(f"  • {name}")
        if len(new_feature_names) > 5:
            print(f"  ... et {len(new_feature_names) - 5} autres")
    
    # Sauvegarde du transformateur
    if save_transformer and poly_transformer is not None:
        filename = f"poly_transformer_degree{degree}_interact{interaction_only}.pkl"
        save_artifact(poly_transformer, filename, cfg.paths.transformers)
        log.info(f"💾 Transformateur polynomial sauvegardé : {filename}")
    
    return df_with_poly, poly_transformer

# =============================================================================
# 7. PIPELINE COMPLET DE PRÉTRAITEMENT
# =============================================================================

def preprocess_complete_pipeline(df: pd.DataFrame,
                                target_col: str,
                                continuous_cols: List[str],
                                imputation_method: str = 'knn',
                                transformation_mapping: Optional[Dict] = None,
                                save_artifacts: bool = True,
                                **kwargs) -> Dict:
    """
    Pipeline complet de prétraitement combinant toutes les étapes.
    
    Args:
        df: DataFrame source
        target_col: Colonne cible
        continuous_cols: Colonnes continues à traiter
        imputation_method: 'knn' ou 'mice'
        transformation_mapping: Mapping des transformations par colonne
        save_dir: Répertoire pour sauvegarder tous les artefacts
        **kwargs: Paramètres supplémentaires pour les étapes
        
    Returns:
        Dictionnaire contenant tous les résultats et artefacts
    """
    
    print("🚀 PIPELINE COMPLET DE PRÉTRAITEMENT")
    print("=" * 60)
    
    results = {
        'original_shape': df.shape,
        'steps_completed': [],
        'artifacts': {},
        'processing_time': {}
    }
    
    # Configuration du logging et sauvegarde
    log.info("🚀 Démarrage du pipeline de prétraitement")
    log.info(f"📊 Données d'entrée: {df.shape}")
    
    # 1. Imputation
    step_start = time.time()
    if imputation_method == 'knn':
        df_imputed, imputer = perform_knn_imputation(
            df, continuous_cols, 
            save_imputer=save_artifacts,
            **kwargs.get('knn_params', {})
        )
    else:
        df_imputed, imputer = perform_mice_imputation(
            df, continuous_cols,
            save_imputer=save_artifacts, 
            **kwargs.get('mice_params', {})
        )
    
    results['steps_completed'].append('imputation')
    results['artifacts']['imputer'] = imputer
    results['processing_time']['imputation'] = time.time() - step_start
    
    # 2. Transformations optimales
    if transformation_mapping:
        step_start = time.time()
        df_transformed, transformers = apply_optimal_transformations(
            df_imputed, continuous_cols, transformation_mapping,
            save_transformers=save_artifacts
        )
        results['steps_completed'].append('transformations')
        results['artifacts']['transformers'] = transformers
        results['processing_time']['transformations'] = time.time() - step_start
    else:
        df_transformed = df_imputed
    
    # 3. Traitement des outliers
    step_start = time.time()
    outlier_cols = [f"{col}_transformed" for col in continuous_cols if f"{col}_transformed" in df_transformed.columns]
    if not outlier_cols:
        outlier_cols = continuous_cols
        
    df_capped, capping_params = detect_and_cap_outliers(
        df_transformed, outlier_cols,
        save_params=save_artifacts,
        **kwargs.get('outlier_params', {})
    )
    results['steps_completed'].append('outlier_treatment')
    results['artifacts']['capping_params'] = capping_params  
    results['processing_time']['outlier_treatment'] = time.time() - step_start
    
    # 4. Features polynomiales
    step_start = time.time()
    poly_cols = [col for col in outlier_cols if col in df_capped.columns]
    df_final, poly_transformer = generate_polynomial_features(
        df_capped, poly_cols,
        save_transformer=save_artifacts,
        **kwargs.get('poly_params', {})
    )
    results['steps_completed'].append('polynomial_features')
    results['artifacts']['poly_transformer'] = poly_transformer
    results['processing_time']['polynomial_features'] = time.time() - step_start
    
    # Résultats finaux
    results['final_shape'] = df_final.shape
    results['final_dataframe'] = df_final
    results['total_time'] = sum(results['processing_time'].values())
    
    print(f"🎉 PIPELINE TERMINÉ !")
    print(f"⏱️  Temps total: {results['total_time']:.2f}s")
    print(f"📊 Shape: {results['original_shape']} → {results['final_shape']}")
    print(f"✅ Étapes: {' → '.join(results['steps_completed'])}")
    
    return results