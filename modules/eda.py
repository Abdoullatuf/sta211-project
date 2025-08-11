# -*- coding: utf-8 -*-
# utils.py

import os
import pandas as pd
from pathlib import Path
from IPython.display import display, Markdown

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder

from scipy import stats
from typing import List, Dict, Optional, Tuple, Union


# Scikit-learn
from sklearn.experimental import enable_iterative_imputer   # Doit rester au-dessus de IterativeImputer
from sklearn.impute import IterativeImputer, KNNImputer


from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error

# Machine learning models
from sklearn.linear_model import BayesianRidge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
import warnings

# Utilities
import joblib
import time

import re
import chardet
import warnings
warnings.filterwarnings('ignore', category=pd.errors.DtypeWarning)

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
        target_col: Nom de la colonne cible à rechercher (défaut: "outcome")
        display_info: Afficher les informations du dataset
        encoding: Encodage du fichier (auto-détection si None)
        max_file_size_mb: Taille maximale du fichier en MB
        
    Returns:
        Tuple[pd.DataFrame, Dict]: DataFrame chargé et informations sur le chargement
        
    Raises:
        FileNotFoundError: Si le fichier n'existe pas
        ValueError: Si les paramètres sont invalides
        MemoryError: Si le fichier est trop volumineux
    """
    
    # --- 1. VALIDATION ---
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"❌ Fichier introuvable : {file_path}")
    
    # Vérifier la taille du fichier
    file_size_mb = file_path.stat().st_size / (1024 * 1024)
    if file_size_mb > max_file_size_mb:
        raise MemoryError(f"❌ Fichier trop volumineux ({file_size_mb:.1f}MB > {max_file_size_mb}MB)")
    
    # --- 2. DÉTECTION DE L'ENCODAGE ---
    if encoding is None:
        encoding = _detect_encoding(file_path)
    
    # --- 3. CHARGEMENT AVEC DÉTECTION DU SÉPARATEUR ---
    df, separator_used = _load_with_separator_detection(file_path, encoding)
    
    # --- 4. NETTOYAGE MINIMAL (NOMS DE COLONNES UNIQUEMENT) ---
    df = _clean_column_names(df)
    
    # --- 5. GESTION DES COLONNES DUPLIQUÉES ---
    df = _handle_duplicate_columns(df)
    
    # --- 6. VÉRIFICATION DE LA COLONNE CIBLE ---
    has_target = target_col in df.columns
    target_info = _analyze_target_column(df, target_col) if has_target else None
    
    # --- 7. CRÉATION DU RAPPORT ---
    report = {
        'file_info': {
            'name': file_path.name,
            'size_mb': round(file_size_mb, 2),
            'encoding': encoding,
            'separator': separator_used
        },
        'data_info': {
            'shape': df.shape,
            'columns': list(df.columns),
            'dtypes': df.dtypes.to_dict(),
            'memory_usage_mb': round(df.memory_usage(deep=True).sum() / (1024 * 1024), 2)
        },
        'target_column': {
            'name': target_col,
            'present': has_target,
            'info': target_info
        }
    }
    
    # --- 8. AFFICHAGE DES INFORMATIONS ---
    if display_info:
        _display_loading_info(df, report)
    
    return df, report


def _detect_encoding(file_path: Path, sample_size: int = 10000) -> str:
    """Détecte l'encodage du fichier."""
    try:
        with open(file_path, 'rb') as f:
            raw_data = f.read(sample_size)
        result = chardet.detect(raw_data)
        detected_encoding = result['encoding'] or 'utf-8'
        
        # Fallback vers des encodages courants si la détection échoue
        if result['confidence'] < 0.7:
            for encoding_try in ['utf-8', 'latin-1', 'cp1252']:
                try:
                    with open(file_path, 'r', encoding=encoding_try) as f:
                        f.read(1000)  # Test de lecture
                    return encoding_try
                except:
                    continue
        
        return detected_encoding
    except Exception:
        return 'utf-8'


def _load_with_separator_detection(file_path: Path, encoding: str) -> Tuple[pd.DataFrame, str]:
    """Détecte le bon séparateur et charge le DataFrame."""
    
    # Valeurs à considérer comme NaN (minimal, sans imputation)
    na_values = ['', 'NULL', 'null', 'NA', 'N/A', 'NaN', 'nan', '#N/A']
    
    # Tester différents séparateurs par ordre de probabilité
    separators_to_test = [',', ';', '\t', '|']
    
    best_df = None
    best_separator = ','
    max_columns = 0
    
    # Tester d'abord avec un échantillon pour éviter de charger tout le fichier
    for sep in separators_to_test:
        try:
            sample_df = pd.read_csv(
                file_path, 
                sep=sep, 
                encoding=encoding,
                nrows=100,  # Échantillon pour test
                na_values=na_values,
                on_bad_lines='skip'
            )
            
            # Critères : nombre de colonnes et qualité des données
            if sample_df.shape[1] > max_columns and sample_df.shape[1] > 1:
                # Vérifier que ce n'est pas du texte mal parsé
                if sample_df.iloc[:, 0].astype(str).str.len().mean() < 100:  # Pas de texte très long
                    max_columns = sample_df.shape[1]
                    best_separator = sep
                    
        except Exception:
            continue
    
    # Charger le fichier complet avec le meilleur séparateur
    try:
        df = pd.read_csv(
            file_path,
            sep=best_separator,
            encoding=encoding,
            na_values=na_values,
            keep_default_na=True,
            on_bad_lines='skip',
            low_memory=False
        )
    except Exception as e:
        raise IOError(f"❌ Erreur lors du chargement du fichier : {e}")
    
    return df, best_separator


def _clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Nettoie uniquement les noms de colonnes sans toucher aux données."""
    # Sauvegarder les noms originaux pour debug
    original_columns = df.columns.tolist()
    
    # Nettoyage minimal : espaces et caractères problématiques
    cleaned_columns = []
    for col in df.columns:
        cleaned = str(col).strip()
        cleaned = re.sub(r'["\'\s]+', ' ', cleaned)  # Guillemets et espaces multiples
        cleaned = re.sub(r'^\s+|\s+$', '', cleaned)  # Espaces début/fin
        cleaned_columns.append(cleaned)
    
    df.columns = cleaned_columns
    
    return df


def _handle_duplicate_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Gère les colonnes dupliquées (pattern .1, .2, etc.) en gardant la première."""
    duplicate_pattern = re.compile(r'^(.+)\.(\d+)$')
    cols_to_drop = []
    
    for col in df.columns:
        match = duplicate_pattern.match(str(col))
        if match:
            base_name = match.group(1)
            if base_name in df.columns:
                # Toujours garder la colonne originale, supprimer les .1, .2, etc.
                cols_to_drop.append(col)
    
    if cols_to_drop:
        df = df.drop(columns=cols_to_drop)
    
    return df


def _analyze_target_column(df: pd.DataFrame, target_col: str) -> Dict:
    """Analyse la colonne cible sans faire d'imputation."""
    target_series = df[target_col]
    
    return {
        'dtype': str(target_series.dtype),
        'total_values': len(target_series),
        'non_null_values': target_series.notna().sum(),
        'null_values': target_series.isnull().sum(),
        'null_percentage': round((target_series.isnull().sum() / len(target_series)) * 100, 2),
        'unique_values': target_series.nunique(),
        'unique_values_list': target_series.dropna().unique().tolist()[:10]  # Max 10 pour éviter l'affichage massif
    }


def _display_loading_info(df: pd.DataFrame, report: Dict):
    """Affiche les informations de chargement."""
    try:
        from IPython.display import display, Markdown
        
        # Titre
        display(Markdown(f"# 📊 Chargement : `{report['file_info']['name']}`"))
        
        # Informations fichier
        file_info = report['file_info']
        print(f"📁 **Taille** : {file_info['size_mb']} MB")
        print(f"🔤 **Encodage** : {file_info['encoding']}")
        print(f"🔄 **Séparateur** : '{file_info['separator']}'")
        
        # Informations dataset
        data_info = report['data_info']
        print(f"📏 **Dimensions** : {data_info['shape'][0]:,} lignes × {data_info['shape'][1]} colonnes")
        print(f"💾 **Mémoire** : {data_info['memory_usage_mb']} MB")
        
        # Informations colonne cible
        target_info = report['target_column']
        print(f"\n🎯 **Colonne cible '{target_info['name']}'** : ", end="")
        if target_info['present']:
            info = target_info['info']
            print(f"✅ PRÉSENTE")
            print(f"   - Type : {info['dtype']}")
            print(f"   - Valeurs non-nulles : {info['non_null_values']:,}/{info['total_values']:,} ({100-info['null_percentage']:.1f}%)")
            print(f"   - Valeurs uniques : {info['unique_values']}")
            if info['unique_values'] <= 20:  # Afficher seulement si pas trop de valeurs
                print(f"   - Valeurs : {info['unique_values_list']}")
        else:
            print("❌ ABSENTE")
            print(f"   Colonnes disponibles : {', '.join(data_info['columns'][:5])}{'...' if len(data_info['columns']) > 5 else ''}")
        
        # Aperçu des données
        display(Markdown("## 👀 Aperçu"))
        display(df.head(3))
        
        # Informations détaillées
        display(Markdown("## 🔍 Informations Détaillées"))
        print("Types de données :")
        type_counts = df.dtypes.value_counts()
        for dtype, count in type_counts.items():
            print(f"  - {dtype}: {count} colonnes")
        
        # Valeurs manquantes globales
        total_missing = df.isnull().sum().sum()
        total_values = df.size
        missing_pct = (total_missing / total_values) * 100
        print(f"\nValeurs manquantes globales : {total_missing:,}/{total_values:,} ({missing_pct:.2f}%)")
        
        # Colonnes avec valeurs manquantes
        missing_by_col = df.isnull().sum()
        missing_cols = missing_by_col[missing_by_col > 0]
        if len(missing_cols) > 0:
            print("Colonnes avec valeurs manquantes :")
            for col, count in missing_cols.head(10).items():
                pct = (count / len(df)) * 100
                print(f"  - {col}: {count:,} ({pct:.1f}%)")
        
    except ImportError:
        # Fallback sans Jupyter
        print(f"\n✅ Fichier chargé : {report['file_info']['name']}")
        print(f"Dimensions : {df.shape}")
        print(f"Colonne cible '{target_info['name']}' : {'✅ PRÉSENTE' if target_info['present'] else '❌ ABSENTE'}")
        print("\nAperçu :")
        print(df.head())


# Fonction utilitaire pour vérifier rapidement un fichier
def check_file_structure(file_path: Union[str, Path], target_col: str = "outcome") -> Dict:
    """
    Vérifie rapidement la structure d'un fichier sans le charger complètement.
    
    Args:
        file_path: Chemin vers le fichier
        target_col: Nom de la colonne cible
        
    Returns:
        Dict: Informations sur la structure du fichier
    """
    file_path = Path(file_path)
    
    if not file_path.exists():
        return {"error": "Fichier introuvable"}
    
    try:
        # Charger seulement les premières lignes
        sample_df = pd.read_csv(file_path, nrows=5)
        
        return {
            "file_exists": True,
            "sample_shape": sample_df.shape,
            "columns": list(sample_df.columns),
            "has_target": target_col in sample_df.columns,
            "estimated_rows": "Inconnu (échantillon de 5 lignes)",
            "file_size_mb": round(file_path.stat().st_size / (1024 * 1024), 2)
        }
    except Exception as e:
        return {"error": f"Erreur de lecture : {e}"}







def visualize_distributions_and_boxplots(df, continuous_cols = ["X1", "X2", "X3"], output_dir=None):
    """
    Visualise les distributions et les boîtes à moustaches pour les variables spécifiées.
    Enregistre les figures si un dossier est spécifié.
    """
    print("\n--- Distribution des variables numériques ---")
    
    if output_dir:
        Path(output_dir).mkdir(parents=True, exist_ok=True)

    for var in continuous_cols:
        if pd.api.types.is_numeric_dtype(df[var]):
            fig, axes = plt.subplots(1, 2, figsize=(15, 4.5))

            sns.histplot(df[var], kde=True, ax=axes[0], color='blue')
            axes[0].set_title(f'Distribution de {var}', fontsize=14)

            sns.boxplot(x=df[var], ax=axes[1], color='skyblue')
            axes[1].set_title(f'Boîte à moustaches de {var}', fontsize=14)

            plt.tight_layout()

            # ✅ Enregistrement si output_dir est fourni
            if output_dir:
                fig_path = Path(output_dir) / f"{var}_distribution_boxplot.png"
                plt.savefig(fig_path, dpi=300, bbox_inches='tight')
                print(f"📁 Figure enregistrée : {fig_path}")

            plt.show()
        else:
            print(f"⚠️ '{var}' n'est pas une variable numérique.")



def plot_correlation_heatmap(df, target_variable, output_dir='outputs/figures/eda'):
    """
    Calcule et affiche la matrice de corrélation, en se concentrant sur la variable cible.
    Gère l'encodage de la variable cible si elle est catégorielle.
    """
    print(f"\n--- Analyse de corrélation avec la cible '{target_variable}' ---")

    df_corr = df.copy()

    # Correction : Encoder la variable cible si elle est de type object/catégorielle
    if df_corr[target_variable].dtype == 'object':
        print(f"Encodage de la variable cible '{target_variable}' pour le calcul de corrélation.")
        le = LabelEncoder()
        df_corr[target_variable] = le.fit_transform(df_corr[target_variable])
        # Affiche le mapping pour référence
        print("Mapping de l'encodage :", {cl: i for i, cl in enumerate(le.classes_)})

    # Calculer les corrélations
    correlations = df_corr.corr()[target_variable].sort_values(ascending=False)

    print("\nTop 10 des variables les plus corrélées (positivement) avec la cible :")
    print(correlations.head(11)) # Inclut la cible elle-même

    print("\nTop 10 des variables les moins corrélées (négativement) avec la cible :")
    print(correlations.tail(10))

    # Il est souvent peu pratique d'afficher une heatmap avec >1500 variables.
    # On se concentre sur les plus corrélées.
    top_corr_vars = correlations.abs().nlargest(30).index

    plt.figure(figsize=(12, 10))
    sns.heatmap(df_corr[top_corr_vars].corr(), annot=True, cmap='coolwarm', fmt=".2f")
    plt.title(f'Matrice de corrélation des 30 variables les plus corrélées avec {target_variable}')

    # S'assurer que le dossier de sortie existe
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, 'correlation_heatmap.png')
    plt.savefig(save_path)
    print(f"\nGraphique de corrélation sauvegardé dans : {save_path}")
    plt.show()







def analyze_continuous_variables(df: pd.DataFrame, continuous_cols: List[str], target_col: str = "y", save_figures_path: str = None) -> dict:
    """
    Analyse complète des variables continues :
    - Statistiques descriptives
    - Asymétrie, aplatissement, test de normalité (Shapiro)
    - Outliers (IQR)
    - Corrélation avec la cible
    - Matrice de corrélation + heatmap

    :param df: DataFrame d'entrée
    :param continuous_cols: Liste des colonnes continues
    :param target_col: Nom de la variable cible
    :param save_figures_path: Chemin vers le dossier de sauvegarde des figures
    :return: Dictionnaire des résultats
    """
    summary_stats = df[continuous_cols].describe()
    print("📊 Statistiques descriptives :")
    print(summary_stats)

    skew_kurtosis_results = {}
    outliers_summary = {}
    correlations = {}

    print("\n📊 Analyse de la distribution :")
    for col in continuous_cols:
        data = df[col].dropna()
        skew = stats.skew(data)
        kurt = stats.kurtosis(data)
        stat, p_value = stats.shapiro(data.sample(min(5000, len(data))))
        print(f"\n{col}:")
        print(f"  - Skewness (asymétrie) : {skew:.3f}")
        print(f"  - Kurtosis (aplatissement) : {kurt:.3f}")
        print(f"  - Test de Shapiro-Wilk : p-value = {p_value:.4f}")
        if p_value < 0.01:
            print("    → Distribution non normale (nécessite transformation)")
        else:
            print("    → Distribution approximativement normale")
        skew_kurtosis_results[col] = {"skewness": skew, "kurtosis": kurt, "shapiro_p": p_value}

    print("\n🔍 Détection des outliers (méthode IQR) :")
    for col in continuous_cols:
        data = df[col].dropna()
        Q1 = data.quantile(0.25)
        Q3 = data.quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        outliers = data[(data < lower) | (data > upper)]
        pct = len(outliers) / len(data) * 100
        print(f"\n{col}:")
        print(f"  - Limites : [{lower:.2f}, {upper:.2f}]")
        print(f"  - Outliers : {len(outliers)} ({pct:.2f}%)")
        outliers_summary[col] = {"count": len(outliers), "percentage": pct, "lower": lower, "upper": upper}

    print("\n🎯 Corrélation avec la variable cible (y) :")
    for col in continuous_cols:
        valid = df[col].notna() & df[target_col].notna()
        corr = df.loc[valid, col].corr(df.loc[valid, target_col])
        print(f"  - {col}: {corr:.4f}")
        correlations[col] = corr

    print("\n📊 Matrice de corrélation entre variables continues :")
    corr_matrix = df[continuous_cols].corr()
    print(corr_matrix)

    plt.figure(figsize=(6, 4))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0,
                square=True, linewidths=1, cbar_kws={"shrink": 0.8})
    plt.title('Matrice de corrélation des variables continues', fontsize=14)
    plt.tight_layout()
    if save_figures_path:
        plt.savefig(save_figures_path + "/continuous_correlation_matrix.png", dpi=300, bbox_inches='tight')
    plt.show()

    total_outliers = sum(info['count'] for info in outliers_summary.values())
    print("\n💡 Résumé et recommandations :")
    print("  - Les trois variables continues montrent des distributions fortement asymétriques")
    print("  - Transformation Yeo-Johnson recommandée pour normaliser les distributions")
    print(f"  - Outliers détectés : {total_outliers} au total")
    print("  - Corrélations faibles avec la cible, mais potentiellement utiles après transformation")

    return {
        "summary_stats": summary_stats,
        "skew_kurtosis": skew_kurtosis_results,
        "outliers": outliers_summary,
        "correlations": correlations,
        "corr_matrix": corr_matrix
    }




def bivariate_analysis(
    data: pd.DataFrame,
    use_transformed: bool = True,
    display_correlations: bool = True,
    top_n: int = 10,
    show_plot: bool = True
):
    """
    Analyse bivariée complète :
    - Corrélations continues et binaires avec la cible
    - Redondances internes (variables continues & binaires)

    Args:
        data: DataFrame contenant les données
        use_transformed: True = utilise X1_trans, X2_trans, X3_trans
        display_correlations: affiche les résultats dans la console
        top_n: nombre de variables à afficher dans le top
        show_plot: génère un barplot des meilleures variables

    Returns:
        corr_df: DataFrame trié des corrélations avec la cible
        high_corr_pairs: paires de variables continues fortement corrélées
        binary_corr_pairs: paires binaires très redondantes
    """
    print("\n=== Analyse Bivariée ===")

    # 1. Encodage cible binaire
    target_numeric = (data['outcome'] == 'ad.').astype(int)

    # 2. Sélection des variables
    continuous_vars = [col for col in ['X1_trans', 'X2_trans', 'X3_trans'] if col in data.columns] \
        if use_transformed else [col for col in ['X1', 'X2', 'X3'] if col in data.columns]

    binary_vars = [
        col for col in data.columns
        if data[col].dropna().nunique() == 2 and col != 'outcome'
    ]

    # 3. Corrélation avec la cible
    correlations = []
    for col in continuous_vars:
        corr = data[col].corr(target_numeric)
        correlations.append((col, corr))

    for col in binary_vars:
        try:
            corr, _ = pointbiserialr(data[col], target_numeric)
            correlations.append((col, corr))
        except Exception:
            continue

    corr_df = pd.DataFrame(correlations, columns=['feature', 'correlation']).dropna()
    corr_df = corr_df.sort_values('correlation', key=abs, ascending=False)

    if display_correlations:
        print(f"\n🔝 Top {top_n} variables les plus corrélées à la cible :")
        print(corr_df.head(top_n))

    if show_plot:
        plt.figure(figsize=(8, 5))
        sns.barplot(data=corr_df.head(top_n), y='feature', x='correlation', palette='viridis')
        plt.title(f"Top {top_n} variables corrélées à la cible")
        plt.tight_layout()
        plt.show()

    # 4. Corrélation interne (variables continues)
    high_corr_pairs = []
    if continuous_vars:
        corr_matrix = data[continuous_vars].corr()
        for i in range(len(corr_matrix.columns)):
            for j in range(i + 1, len(corr_matrix.columns)):
                val = corr_matrix.iloc[i, j]
                if abs(val) > 0.9:
                    high_corr_pairs.append((corr_matrix.columns[i], corr_matrix.columns[j], val))

    # 5. Corrélation interne (variables binaires)
    binary_corr_pairs = []
    if len(binary_vars) > 1:
        bin_corr = data[binary_vars].corr()
        for i in range(len(bin_corr.columns)):
            for j in range(i + 1, len(bin_corr.columns)):
                val = bin_corr.iloc[i, j]
                if abs(val) > 0.95:
                    binary_corr_pairs.append((bin_corr.columns[i], bin_corr.columns[j], val))

    return corr_df, high_corr_pairs, binary_corr_pairs




def handle_missing_values(
    df: pd.DataFrame,
    strategy: str = 'mixed_mar_mcar',
    mar_method: str = 'knn',
    knn_k: Optional[int] = None,
    mar_cols: Optional[List[str]] = None,
    mcar_cols: Optional[List[str]] = None,
    display_info: bool = True,
    save_results: bool = True,
    processed_data_dir: Optional[Union[str, Path]] = None,
    imputers_dir: Optional[Union[str, Path]] = None,
    custom_filename: Optional[str] = None,
    auto_optimize_k: bool = False,
    validate_imputation: bool = True,
    backup_method: str = 'median',
    mice_estimator: Optional[object] = None,
    # ✅ NOUVEAU : Paramètre pour contrôler le traitement des autres colonnes
    treat_other_cols: bool = False
) -> pd.DataFrame:

    def median_fill(df_local, cols_local):
        for col in cols_local:
            if col in df_local.columns and df_local[col].isnull().any():
                df_local[col] = df_local[col].fillna(df_local[col].median())

    if display_info:
        print("🔧 Début de l'imputation des valeurs manquantes")
        print("=" * 50)

    df_proc = df.copy()
    suffix = ''

    initial_missing = df_proc.isnull().sum().sum()
    if display_info:
        print(f" Valeurs manquantes initiales: {initial_missing}")

    if strategy == 'all_median':
        num_cols = df_proc.select_dtypes(include=[np.number]).columns
        median_fill(df_proc, num_cols)
        suffix = 'median_all'

    elif strategy == 'mixed_mar_mcar':
        if mar_cols is None:
            mar_cols = ['X1', 'X2', 'X3']
        if mcar_cols is None:
            mcar_cols = ['X4']

        available_mar_cols = [col for col in mar_cols if col in df_proc.columns]

        if available_mar_cols:
            if display_info:
                print(f" Variables MAR à imputer: {len(available_mar_cols)}")
                for col in available_mar_cols:
                    count = df_proc[col].isnull().sum()
                    print(f"   • {col}: {count} valeurs manquantes")

            try:
                if mar_method == 'knn':
                    if knn_k is None:
                        knn_k = 5
                        if display_info:
                            print("⚠️ k non spécifié, utilisation de k = 5 par défaut.")

                    imputer = KNNImputer(n_neighbors=knn_k)
                    df_proc[available_mar_cols] = imputer.fit_transform(df_proc[available_mar_cols])
                    suffix = f'knn_k{knn_k}'

                elif mar_method == 'mice':
                    if mice_estimator is None:
                        if display_info:
                            print("⚙️ Utilisation de MICE avec BayesianRidge par défaut")
                        mice_estimator = BayesianRidge()

                    complete_rows = df_proc[available_mar_cols].dropna().shape[0]
                    if complete_rows < 10:
                        raise ValueError("Pas assez de données complètes pour utiliser MICE efficacement.")

                    imputer = IterativeImputer(
                        estimator=mice_estimator,
                        max_iter=50,
                        random_state=42
                    )

                    df_proc[available_mar_cols] = imputer.fit_transform(df_proc[available_mar_cols])
                    suffix = 'mice_custom'

                else:
                    raise ValueError("❌ mar_method doit être 'knn' ou 'mice'.")

                if save_results and imputers_dir:
                    imputers_dir = Path(imputers_dir)
                    imputers_dir.mkdir(parents=True, exist_ok=True)
                    imp_path = imputers_dir / f"imputer_{suffix}.pkl"
                    joblib.dump(imputer, imp_path)
                    if display_info:
                        print(f"💾 Modèle d'imputation sauvegardé: {imp_path}")

            except Exception as e:
                print(f"❌ Erreur lors de l'imputation MAR: {e}")
                print(f"🔄 Utilisation de la méthode de secours: {backup_method}")
                median_fill(df_proc, available_mar_cols)
                suffix = f'{backup_method}_backup'

        # ✅ MODIFICATION : Traitement conditionnel des autres colonnes
        if treat_other_cols:
            available_mcar_cols = [col for col in mcar_cols if col in df_proc.columns]
            if available_mcar_cols:
                if display_info:
                    print(f"📊 Variables MCAR à imputer avec médiane: {len(available_mcar_cols)}")
                    for col in available_mcar_cols:
                        count = df_proc[col].isnull().sum()
                        print(f"   • {col}: {count} valeurs manquantes")
                median_fill(df_proc, available_mcar_cols)
        else:
            if display_info:
                print("ℹ️ Traitement des autres colonnes désactivé (treat_other_cols=False)")

    else:
        raise ValueError("❌ strategy doit être 'all_median' ou 'mixed_mar_mcar'.")

    final_missing = df_proc.isnull().sum().sum()
    if display_info:
        print("\n📊 Résumé de l'imputation:")
        print(f"   • Valeurs manquantes avant: {initial_missing}")
        print(f"   • Valeurs manquantes après: {final_missing}")
        
        # ✅ Vérification des colonnes traitées
        if final_missing > 0:
            remaining_cols = df_proc.columns[df_proc.isnull().any()].tolist()
            print(f"ℹ️ Colonnes avec valeurs manquantes restantes: {remaining_cols}")
            for col in remaining_cols:
                count = df_proc[col].isnull().sum()
                print(f"   • {col}: {count} valeurs manquantes")

    if save_results:
        if processed_data_dir is None:
            raise ValueError("processed_data_dir doit être fourni si save_results=True.")
        processed_data_dir = Path(processed_data_dir)

        if processed_data_dir.suffix:
            raise ValueError("processed_data_dir doit être un dossier, pas un fichier.")

        processed_data_dir.mkdir(parents=True, exist_ok=True)

        filename = custom_filename or f"df_imputed_{suffix}.csv"
        filepath = processed_data_dir / filename
        df_proc.to_csv(filepath, index=False)

        if display_info:
            print(f"💾 Données imputées sauvegardées: {filepath}")

    if display_info:
        print("=" * 50)
        print("✅ Imputation terminée avec succès")

    return df_proc







def find_optimal_k(
    df: pd.DataFrame,
    columns_to_impute: List[str],
    k_range: range = range(3, 21, 2),
    cv_folds: int = 5,
    sample_size: Optional[int] = 1000,
    test_size: float = 0.2,
    random_state: int = 42,
    metric: str = 'mse',
    plot_results: bool = True,
    verbose: bool = True,
    figsize: Tuple[int, int] = (10, 6)
) -> Dict:
    """
    Trouve la valeur optimale de K pour l'imputation KNN par validation croisée.
    Version corrigée pour être compatible avec les anciennes versions de scikit-learn.
    """

    # ... (Le début de la fonction reste identique) ...
    start_time = time.time()

    if verbose:
        print("🔍 Recherche de la valeur optimale K pour l'imputation KNN")
        print("=" * 60)
        print(f"📊 Colonnes à évaluer      : {columns_to_impute}")
        print(f"🎯 Plage K à tester        : {list(k_range)}")
        print(f"🔄 Validation croisée      : {cv_folds} folds")
        print(f"📏 Métrique d'évaluation  : {metric.upper()}")
        print("-" * 60)

    if not columns_to_impute:
        raise ValueError("❌ La liste 'columns_to_impute' ne peut pas être vide.")

    for col in columns_to_impute:
        if col not in df.columns:
            raise ValueError(f"❌ Colonne '{col}' introuvable dans le DataFrame.")

    df_work = df[columns_to_impute].copy()
    df_complete = df_work.dropna()

    if len(df_complete) < cv_folds:
        raise ValueError("Moins de lignes complètes que de folds pour la CV.")

    if sample_size is not None and len(df_complete) > sample_size:
        if verbose:
            print(f"Échantillonnage de {sample_size} lignes parmi {len(df_complete)} pour l'optimisation.")
        df_complete = df_complete.sample(n=sample_size, random_state=random_state)

    if verbose:
        print(f"\n🧮 Données utilisées pour le test : {len(df_complete):,} lignes.")

    results = {'k_values': list(k_range), 'scores_mean': [], 'scores_std': []}

    for k in k_range:
        if verbose:
            print(f"\n🔄 Test K={k:2d} ", end="")

        fold_scores = []
        kf = KFold(n_splits=cv_folds, shuffle=True, random_state=random_state)

        for fold_idx, (train_idx, test_idx) in enumerate(kf.split(df_complete)):
            df_train_fold, df_test_fold = df_complete.iloc[train_idx], df_complete.iloc[test_idx]

            df_test_masked = df_test_fold.copy()
            mask = np.random.RandomState(random_state + fold_idx).rand(*df_test_masked.shape) < test_size
            mask[df_test_masked.isnull()] = False

            true_values = df_test_fold.to_numpy()[mask]
            df_test_masked.iloc[mask] = np.nan

            if df_test_masked.isnull().values.sum() == 0:
                continue

            imputer = KNNImputer(n_neighbors=k)
            df_imputed = pd.DataFrame(imputer.fit(df_train_fold).transform(df_test_masked),
                                      columns=df_test_masked.columns, index=df_test_masked.index)

            imputed_values = df_imputed.to_numpy()[mask]

            if len(true_values) > 0 and len(imputed_values) > 0:
                # --- DÉBUT DE LA CORRECTION ---
                # 1. Calculez toujours le MSE
                mse = mean_squared_error(true_values, imputed_values)

                # 2. Adaptez le score en fonction de la métrique demandée
                if metric.lower() == 'rmse':
                    score = np.sqrt(mse)
                elif metric.lower() == 'mae':
                    # Le MAE est une métrique différente, il faut l'importer ou la calculer manuellement
                    from sklearn.metrics import mean_absolute_error
                    score = mean_absolute_error(true_values, imputed_values)
                else: # 'mse' est le cas par défaut
                    score = mse
                

                fold_scores.append(score)

        if fold_scores:
            mean_score, std_score = np.mean(fold_scores), np.std(fold_scores)
            results['scores_mean'].append(mean_score)
            results['scores_std'].append(std_score)
            if verbose:
                print(f"→ {metric.upper()}: {mean_score:.4f} (±{std_score:.4f})")
        else:
            results['scores_mean'].append(np.inf)
            results['scores_std'].append(0)
            if verbose:
                print("→ ❌ Échec (pas de scores valides)")

    valid_scores = [(i, score) for i, score in enumerate(results['scores_mean']) if np.isfinite(score)]
    if not valid_scores:
        raise RuntimeError("❌ Aucune valeur K n'a produit de résultats valides.")

    best_idx, best_score = min(valid_scores, key=lambda x: x[1])
    optimal_k = results['k_values'][best_idx]

    if plot_results and len(valid_scores) > 1:
        plt.figure(figsize=figsize)
        k_vals = [results['k_values'][i] for i, _ in valid_scores]
        scores = [score for _, score in valid_scores]
        stds = [results['scores_std'][i] for i, _ in valid_scores]
        plt.errorbar(k_vals, scores, yerr=stds, marker='o', capsize=5, label='Score moyen par K')
        plt.axvline(x=optimal_k, color='red', linestyle='--', label=f'K optimal = {optimal_k}')
        plt.scatter([optimal_k], [best_score], color='red', s=150, zorder=5, marker='*')
        plt.title(f'Optimisation K pour Imputation KNN ({metric.upper()})', fontsize=14)
        plt.xlabel('Nombre de voisins (K)')
        plt.ylabel(f'Score ({metric.upper()})')
        plt.legend()
        plt.grid(True, which='both', linestyle='--', linewidth=0.5)
        plt.show()

    computation_time = time.time() - start_time
    if verbose:
        print("\n" + "=" * 60)
        print("🎯 RÉSULTATS DE L'OPTIMISATION")
        print("=" * 60)
        print(f"🏆 K optimal              : {optimal_k}")
        print(f"📊 Meilleur score ({metric.upper()})  : {best_score:.4f}")
        print(f"⏱️  Temps de calcul        : {computation_time:.2f}s")

    return {
        'optimal_k': optimal_k,
        'best_score': best_score,
        'results_df': pd.DataFrame(results)
    }





import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Dict, Union, Tuple

def full_correlation_analysis(
    df_study: pd.DataFrame,
    continuous_cols: List[str],
    presence_rates: Dict[str, float],
    fig_dir: Union[str, Path],
    top_n_corr_features: int = 20,
    figsize_corr_matrix: Tuple[int, int] = (8, 6)
) -> None:
    """
    Analyse combinée des corrélations :
    - Variables continues + 40 variables binaires sélectionnées par taux de présence
    - Corrélation avec la cible y
    - Heatmap des corrélations
    - Détection des paires fortement corrélées
    - Résumé imprimé

    :param df_study: DataFrame complet
    :param continuous_cols: Liste des colonnes continues
    :param presence_rates: Dictionnaire des taux de présence des variables binaires
    :param fig_dir: Répertoire où sauvegarder la heatmap
    :param top_n_corr_features: Nombre de variables à afficher dans le résumé
    :param figsize_corr_matrix: Taille de la heatmap
    """

    print("🔗 Analyse combinée des corrélations (features ↔ cible, features ↔ features)")
    print("=" * 80)

    fig_dir = Path(fig_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)

    # 🎯 Sélection des variables binaires par quartile de taux de présence
    presence_series = pd.Series(presence_rates)
    quartiles = presence_series.quantile([0.25, 0.5, 0.75])

    vars_q1 = presence_series[presence_series <= quartiles[0.25]].sample(10, random_state=42).index.tolist()
    vars_q2 = presence_series[(presence_series > quartiles[0.25]) & (presence_series <= quartiles[0.5])].sample(10, random_state=42).index.tolist()
    vars_q3 = presence_series[(presence_series > quartiles[0.5]) & (presence_series <= quartiles[0.75])].sample(10, random_state=42).index.tolist()
    vars_q4 = presence_series[presence_series > quartiles[0.75]].sample(10, random_state=42).index.tolist()

    selected_vars = continuous_cols + vars_q1 + vars_q2 + vars_q3 + vars_q4
    print(f"📌 Variables sélectionnées : {len(selected_vars)} (3 continues + 40 binaires)")

    # 🧮 Matrice de corrélation
    corr_matrix = df_study[selected_vars + ['y']].corr()

    # 🎯 Corrélations avec la cible
    target_corr = corr_matrix['y'].drop('y').sort_values(ascending=False)
    print("\n🎯 Top 10 corrélations avec la cible (y) :")
    for var, corr in target_corr.head(10).items():
        print(f"  - {var:<20} : {corr:.4f}")

    print("\n🎯 Bottom 10 corrélations avec la cible (y) :")
    for var, corr in target_corr.tail(10).items():
        print(f"  - {var:<20} : {corr:.4f}")

    # 🔥 Heatmap
    plt.figure(figsize=figsize_corr_matrix)
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(
        corr_matrix,
        mask=mask,
        cmap='coolwarm',
        center=0,
        square=True,
        linewidths=0.5,
        cbar_kws={"shrink": 0.8},
        vmin=-0.5,
        vmax=0.5
    )
    plt.title('Matrice de corrélation (échantillon)', fontsize=14)
    plt.tight_layout()
    heatmap_path = fig_dir / "correlation_matrix_sample.png"
    plt.savefig(heatmap_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\n📈 Heatmap sauvegardée dans : {heatmap_path}")

    # 🔎 Corrélations entre features
    upper_triangle = corr_matrix.where(np.triu(np.ones_like(corr_matrix, dtype=bool), k=1))
    high_corr_pairs = [
        {'var1': c1, 'var2': c2, 'corr': upper_triangle.loc[c1, c2]}
        for c1 in upper_triangle.columns
        for c2 in upper_triangle.columns
        if c1 != c2 and not pd.isna(upper_triangle.loc[c1, c2]) and abs(upper_triangle.loc[c1, c2]) > 0.8
    ]

    print("\n🔍 Paires fortement corrélées entre variables (|r| > 0.8) :")
    if high_corr_pairs:
        print(f"⚠️ {len(high_corr_pairs)} paires trouvées.")
        for pair in high_corr_pairs[:5]:
            print(f"  - {pair['var1']} vs {pair['var2']} : r = {pair['corr']:.3f}")
    else:
        print("✅ Aucune paire avec |r| > 0.8")

    # 🔚 Résumé
    print("\n💡 Résumé final :")
    print(f"  - Corrélation max avec la cible y : {abs(target_corr).max():.3f}")
    print(f"  - Total de variables analysées : {len(selected_vars)}")
    print(f"  - Multicolinéarité modérée : {'Oui' if high_corr_pairs else 'Non'}")







def detect_outliers_iqr(series: pd.Series, multiplier: float = 1.5) -> pd.Series:
    """
    Détecte les outliers selon la règle de l'IQR.
    
    Parameters:
    -----------
    series : pd.Series
        Série à analyser
    multiplier : float
        Multiplicateur pour l'IQR (par défaut 1.5)
        
    Returns:
    --------
    pd.Series : Masque booléen (True = outlier)
    """
    Q1 = series.quantile(0.25)
    Q3 = series.quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - multiplier * IQR
    upper = Q3 + multiplier * IQR
    return (series < lower) | (series > upper)




def find_highly_correlated_groups(df: pd.DataFrame, threshold: float = 0.90):
    """
    Identifie les groupes de variables fortement corrélées (|corr| > threshold).
    
    Args:
        df (pd.DataFrame): données binaires (0/1)
        threshold (float): seuil de corrélation absolue

    Returns:
        List[List[str]]: groupes de variables corrélées
    """
    corr_matrix = df.corr().abs()
    upper_triangle = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

    groups = []
    visited = set()

    for col in upper_triangle.columns:
        if col in visited:
            continue
        correlated = upper_triangle[col][upper_triangle[col] > threshold].index.tolist()
        if correlated:
            group = sorted(set([col] + correlated))
            groups.append(group)
            visited.update(group)

    return groups





def drop_correlated_duplicates(
    df: pd.DataFrame,
    groups: List[List[str]],
    target_col: str = "outcome",
    extra_cols: List[str] = None,
    verbose: bool = False,
    summary: bool = True
) -> Tuple[pd.DataFrame, List[str], List[str]]:
    """
    Supprime toutes les variables d'un groupe corrélé sauf la première.

    Args:
        df (pd.DataFrame): DataFrame d'origine
        groups (List[List[str]]): groupes de variables fortement corrélées
        target_col (str): nom de la variable cible (à exclure des calculs binaires)
        extra_cols (List[str]): variables à réintégrer à la fin (X1-X4 etc.)
        verbose (bool): affiche les groupes traités
        summary (bool): affiche le résumé final

    Returns:
        - df_reduced: DataFrame nettoyé
        - to_drop: colonnes supprimées
        - to_keep: colonnes conservées
    """
    to_drop = []
    to_keep = []

    for group in groups:
        if not group:
            continue
        keep = group[0]
        drop = [col for col in group[1:] if col in df.columns]
        to_keep.append(keep)
        to_drop.extend(drop)
        if verbose:
            print(f"🧹 Groupe : {group} → garde {keep}, retire {drop}")

    # Dédupliquer
    to_drop = sorted(set(to_drop))
    to_keep = sorted(set(to_keep))

    # Colonnes binaires restantes (non corrélées)
    all_binary = [col for col in df.select_dtypes(include='int64').columns if col != target_col]
    untouched = [col for col in all_binary if col not in to_drop and col not in to_keep]

    # Colonnes finales conservées
    final_cols = to_keep + untouched
    if extra_cols:
        final_cols += [col for col in extra_cols if col in df.columns]

    df_reduced = df[final_cols + [target_col]].copy()

    # Résumé
    if summary:
        print(f"\n📊 Résumé de la réduction :")
        print(f"🔻 {len(to_drop)} colonnes binaires supprimées (corrélées)")
        print(f"✅ {len(to_keep)} colonnes binaires conservées (1 par groupe)")
        print(f"➕ {len(untouched)} colonnes binaires non corrélées conservées")
        if extra_cols:
            print(f"🧩 {len(extra_cols)} variables continues / contextuelles ajoutées : {extra_cols}")
        print(f"📐 DataFrame final : {df_reduced.shape[1]} colonnes, {df_reduced.shape[0]} lignes")

    return df_reduced, to_drop, to_keep



def summary_statistics(data, pca, target_col='outcome'):
    """Affiche un résumé des statistiques et résultats."""
    print("\n=== Résumé des Résultats ===")
    print(f"Nombre total d'observations : {len(data)}")
    print(f"Nombre de variables : {data.shape[1]}")
    
    if target_col and target_col in data.columns:
        print(f"Distribution des classes :\n{data[target_col].value_counts(normalize=True)}")
    else:
        print(f"Attention : La colonne cible '{target_col}' n'a pas été trouvée dans le DataFrame.")
    
    numeric_cols = data.select_dtypes(include=['float64', 'int64']).columns
    print(f"Nombre de variables numériques : {len(numeric_cols)}")
    print(f"Pourcentage de valeurs manquantes : {data[numeric_cols].isnull().mean().mean()*100:.2f}%")
    
    if pca is not None:
        print("\n=== Analyse des Composantes Principales ===")
        print(f"Variance expliquée par les deux premières composantes : {pca.explained_variance_ratio_.sum()*100:.2f}%")
        print(f"Variance expliquée par la première composante : {pca.explained_variance_ratio_[0]*100:.2f}%")
        print(f"Variance expliquée par la deuxième composante : {pca.explained_variance_ratio_[1]*100:.2f}%")




################ VISUALISATION #############################


def save_fig(
    fname: str,
    directory: Union[str, Path],
    dpi: int = 150,
    figsize: Optional[Tuple[float, float]] = None,
    format: str = "png",
    close: bool = False,
    show: bool = True,
    **kwargs
) -> Path:
    """
    Sauvegarde la figure matplotlib courante.
    """
    if plt.get_fignums() == 0:
        raise ValueError("Aucune figure Matplotlib active à sauvegarder.")

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / fname
    if not path.suffix:
        path = path.with_suffix(f".{format}")

    if figsize:
        plt.gcf().set_size_inches(figsize)

    plt.savefig(path, dpi=dpi, bbox_inches="tight", format=format, **kwargs)
    if show:
        plt.show()
    if close:
        plt.close(plt.gcf())
    print(f"✅ Figure sauvegardée : {path}")
    return path

def visualize_distributions_and_boxplots(df: pd.DataFrame, continuous_cols: List[str], output_dir: Path) -> None:
    num_cols = len(continuous_cols)
    fig, axes = plt.subplots(2, num_cols, figsize=(6 * num_cols, 10))
    axes = axes.flatten()

    for i, col in enumerate(continuous_cols):
        data = df[col].dropna()
        # Histogramme + KDE
        ax1 = axes[i]
        data.hist(bins=50, ax=ax1, alpha=0.7, color='skyblue', edgecolor='black')
        ax2 = ax1.twinx()
        data.plot(kind='kde', ax=ax2, color='red', linewidth=2)
        ax1.set_title(f'Distribution de {col}')
        ax1.set_xlabel(col)
        ax1.set_ylabel('Fréquence')
        ax2.set_ylabel('Densité')
        # Boxplot
        ax3 = axes[i + num_cols]
        sns.boxplot(y=df[col], ax=ax3, color='lightgrey')
        ax3.set_title(f'Box plot de {col}')
        ax3.set_ylabel('Valeur')

    plt.tight_layout()
    save_fig(
        fname="continuous_distributions_boxplots.png",
        directory=output_dir,
        dpi=150,
        format="png",
        show=True,
        close=True
    )

def plot_continuous_by_class(df, continuous_cols, output_dir, figsize=(15, 5)):
    fig, axes = plt.subplots(1, len(continuous_cols), figsize=figsize)
    for i, col in enumerate(continuous_cols):
        df_clean = df[[col, 'y']].dropna()
        df_clean['y_label'] = df_clean['y'].map({0: 'Non-pub', 1: 'Pub'})
        sns.violinplot(data=df_clean, x='y_label', y=col, ax=axes[i])
        axes[i].set_title(f'{col} par classe')
    plt.tight_layout()
    save_fig('continuous_by_class.png', directory=output_dir, figsize=figsize)

def plot_continuous_target_corr(df, continuous_cols, output_dir):
    fig, ax = plt.subplots(figsize=(8, 4))
    corr_data = df[continuous_cols + ['y']].corr()['y'][:-1].to_frame()
    sns.heatmap(corr_data, annot=True, cmap='coolwarm', center=0,
                cbar_kws={'label': 'Corrélation'}, fmt='.3f', ax=ax)
    ax.set_title('Corrélations avec la cible')
    plt.tight_layout()
    save_fig('continuous_target_correlation.png', directory=output_dir, figsize=(8, 4))

def plot_binary_sparsity(df, binary_cols, output_dir, sample_size=100):
    sample_vars = np.random.choice(binary_cols, size=min(sample_size, len(binary_cols)), replace=False)
    sample_data = df[sample_vars].head(sample_size)
    plt.figure(figsize=(8, 4))
    plt.imshow(sample_data.values, cmap='binary', aspect='auto')
    plt.colorbar(label='Valeur (0 ou 1)')
    plt.title('Sparsité (100 obs × 100 var binaires)')
    plt.xlabel('Variables')
    plt.ylabel('Observations')
    plt.tight_layout()
    save_fig('sparsity_visualization.png', directory=output_dir, figsize=(8, 4))



def plot_eda_summary(df, continuous_cols, binary_cols, target_corr, sparsity, imbalance_ratio, output_dir, presence_series):
    fig = plt.figure(figsize=(14, 8))

    # Distribution de la cible
    ax1 = plt.subplot(2, 3, 1)
    df['y'].map({0: 'noad.', 1: 'ad.'}).value_counts().plot.pie(
        ax=ax1, autopct='%1.1f%%', colors=['#3498db', '#e74c3c'])
    ax1.set_title('Cible')
    ax1.set_ylabel('')

    # Valeurs manquantes (hardcodé)
    ax2 = plt.subplot(2, 3, 2)
    pd.Series({'X1': 27.41, 'X2': 27.37, 'X3': 27.61}).plot.bar(
        ax=ax2, color='coral')
    ax2.set_title('Valeurs manquantes')
    ax2.set_ylabel('%')

    # Sparsité
    ax3 = plt.subplot(2, 3, 3)
    pd.Series({'Zéros': sparsity, 'Uns': 100-sparsity}).plot.pie(
        ax=ax3, autopct='%1.1f%%', colors=['lightgray', 'darkgray'])
    ax3.set_title('Sparsité')

    # Corrélations ↔ cible
    ax4 = plt.subplot(2, 3, 4)
    target_corr.abs().nlargest(10).plot.barh(ax=ax4, color='skyblue')
    ax4.set_title('Top 10 corrélations')

    # Taux de présence
    ax5 = plt.subplot(2, 3, 5)
    presence_series.hist(ax=ax5, bins=30, color='lightgreen', edgecolor='black')
    ax5.axvline(presence_series.mean(), color='red', linestyle='--')
    ax5.set_title('Taux de présence')
    ax5.legend([f'Moy: {presence_series.mean():.1f}%'])

    # Statistiques clés
    ax6 = plt.subplot(2, 3, 6)
    ax6.axis('off')
    summary = f"""
    Observations: {len(df):,}
    Variables: {df.shape[1]:,}
    Déséquilibre: {imbalance_ratio:.1f}:1
    Sparsité: {sparsity:.1f}%
    Corr. max (y): {abs(target_corr).max():.3f}
    Binaires: {len(binary_cols)} | Continues: {len(continuous_cols)}
    """
    ax6.text(0, 0.5, summary, fontsize=12)

    plt.suptitle("Résumé visuel de l'EDA")
    plt.tight_layout()
    save_fig('eda_summary.png', directory=output_dir, figsize=(14, 8))




def plot_outlier_comparison(
    df_before: pd.DataFrame,
    df_after: pd.DataFrame,
    cols: List[str],
    output_dir: Union[str, Path],
    show: bool = True,
    save: bool = True,
    dpi: int = 100
):
    """
    Affiche les boxplots avant/après traitement des outliers pour chaque variable.

    Args:
        df_before (pd.DataFrame): Données avant traitement.
        df_after (pd.DataFrame): Données après traitement.
        cols (list): Colonnes à analyser.
        output_dir (str or Path): Dossier de sauvegarde des figures.
        show (bool): Affiche les figures si True.
        save (bool): Sauvegarde les figures si True.
        dpi (int): Résolution de l’image sauvegardée.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for col in cols:
        fig, axes = plt.subplots(1, 2, figsize=(8, 3))

        sns.boxplot(x=df_before[col], ax=axes[0], color='salmon')
        axes[0].set_title(f"{col} - Avant traitement")

        sns.boxplot(x=df_after[col], ax=axes[1], color='mediumseagreen')
        axes[1].set_title(f"{col} - Après traitement")

        plt.tight_layout()

        if save:
            save_fig(
                fname=f"{col}_outliers_comparison.png",
                directory=output_dir,
                figsize=(10, 3.5),
                dpi=dpi,
                show=show,
                close=not show
            )
        elif show:
            plt.show()
        else:
            plt.close()
