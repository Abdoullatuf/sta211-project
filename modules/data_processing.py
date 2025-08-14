"""
Module consolidé de traitement des données - STA211 Project
Consolidation de: eda.py, outliers.py, transformation_optimale_mixte.py, final_preprocessing.py

Auteur: Maoulida Abdoullatuf  
Version: 4.0 (consolidée)
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
import joblib
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Union
from IPython.display import display, Markdown

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

# =============================================================================
# 1. CHARGEMENT ET NETTOYAGE DES DONNÉES (ex-eda.py)
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
    """
    
    # Validation
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"❌ Fichier introuvable : {file_path}")
    
    # Vérifier la taille du fichier
    file_size_mb = file_path.stat().st_size / (1024 * 1024)
    if file_size_mb > max_file_size_mb:
        raise MemoryError(f"❌ Fichier trop volumineux ({file_size_mb:.1f}MB > {max_file_size_mb}MB)")
    
    # Détection de l'encodage
    if encoding is None:
        encoding = _detect_encoding(file_path)
    
    # Chargement avec détection du séparateur
    df, separator_used = _load_with_separator_detection(file_path, encoding)
    
    # Nettoyage minimal (noms de colonnes uniquement)
    df = _clean_column_names(df)
    
    # Gestion des colonnes dupliquées
    df = _handle_duplicate_columns(df)
    
    # Vérification de la colonne cible
    has_target = target_col in df.columns
    target_info = _analyze_target_column(df, target_col) if has_target else None
    
    # Création du rapport
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
    
    # Affichage des informations
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
        display(Markdown(f" Chargement : `{report['file_info']['name']}`"))
        
        # Informations fichier
        file_info = report['file_info']
        print(f" **Taille** : {file_info['size_mb']} MB")
        print(f" **Encodage** : {file_info['encoding']}")
        print(f" **Séparateur** : '{file_info['separator']}'")
        
        # Informations dataset
        data_info = report['data_info']
        print(f"📏 **Dimensions** : {data_info['shape'][0]:,} lignes × {data_info['shape'][1]} colonnes")
        print(f"💾 **Mémoire** : {data_info['memory_usage_mb']} MB")
        
        # Informations colonne cible
        target_info = report['target_column']
        print(f"\n **Colonne cible '{target_info['name']}'** : ", end="")
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
        display(Markdown("Aperçu"))
        display(df.head(3))
        
        # Informations détaillées
        display(Markdown(" Informations Détaillées"))
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


def check_file_structure(file_path: Union[str, Path], target_col: str = "outcome") -> Dict:
    """
    Vérifie rapidement la structure d'un fichier sans le charger complètement.
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

# =============================================================================
# 2. DÉTECTION ET SUPPRESSION DES OUTLIERS (ex-outliers.py)
# =============================================================================

def detect_and_remove_outliers(
    df: pd.DataFrame,
    columns: List[str],
    method: str = 'iqr',
    iqr_multiplier: float = 1.5,
    remove: bool = True,
    verbose: bool = True,
    save_path: Optional[Union[str, Path]] = None
) -> pd.DataFrame:
    """
    Détecte (et optionnellement supprime) les outliers selon la méthode IQR.
    """
    if method != 'iqr':
        raise NotImplementedError("Seule la méthode 'iqr' est implémentée.")

    df = df.copy()
    initial_shape = df.shape
    mask = pd.Series(True, index=df.index)

    for col in columns:
        if col not in df.columns:
            continue

        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - iqr_multiplier * iqr
        upper_bound = q3 + iqr_multiplier * iqr
        col_mask = df[col].between(lower_bound, upper_bound)

        if verbose:
            n_outliers = (~col_mask).sum()
            print(f"📉 {col} : {n_outliers} outliers détectés")

        mask &= col_mask

    df_result = df[mask] if remove else df

    if verbose and remove:
        print(f"\n✅ Total supprimé : {initial_shape[0] - df_result.shape[0]} lignes")
        print(f" Dimensions finales : {df_result.shape}")

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        if save_path.suffix == ".csv":
            df_result.to_csv(save_path, index=False)
        elif save_path.suffix in [".parquet", ".pq"]:
            df_result.to_parquet(save_path, index=False)
        else:
            raise ValueError("❌ Format non supporté : utilisez .csv ou .parquet")

        if verbose:
            print(f"💾 Données sauvegardées : {save_path}")

    return df_result

# =============================================================================
# 3. TRANSFORMATIONS OPTIMALES (ex-transformation_optimale_mixte.py)
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
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
        if verbose:
            print("TransformationOptimaleMixte initialisée")
            print(f"📁 Répertoire des modèles: {self.models_dir}")
    
    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Applique la transformation optimale mixte complète."""
        if self.verbose:
            print(" TRANSFORMATION OPTIMALE MIXTE")
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
                print(f" X3 corrigé : un offset de {offset:.6f} a été ajouté.")
        
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
        joblib.dump(self.transformer_yj, self.models_dir / 'yeo_johnson_transformer.pkl')
        joblib.dump(self.transformer_bc, self.models_dir / 'box_cox_transformer.pkl')
        if self.verbose:
            print(f"\n Transformateurs sauvegardés dans : {self.models_dir}")
            
    def _diagnostic_initial(self, df):
        """Diagnostic initial des asymétries."""
        if self.verbose:
            print("\n Diagnostic initial (Asymétrie):")
            for var in self.variables_yj + self.variables_bc:
                print(f"  {var}: {df[var].skew():.3f}")

    def _rapport_transformation(self, df_original, df_transformed):
        """Rapport de transformation avec comparaison des asymétries."""
        if self.verbose:
            print("\n Rapport de Transformation :")
            for var in self.variables_yj + self.variables_bc:
                original_skew = df_original[var].skew()
                transformed_skew = df_transformed[f'{var}_transformed'].skew()
                print(f"  {var}: Asymétrie avant={original_skew:+.3f} → après={transformed_skew:+.3f}")


# =============================================================================
# 4. IMPUTATION DES VALEURS MANQUANTES
# =============================================================================

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
    treat_other_cols: bool = False
) -> pd.DataFrame:
    """
    Gère l'imputation des valeurs manquantes avec différentes stratégies.
    """

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
                            print(" k non spécifié, utilisation de k = 5 par défaut.")

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
                        print(f" Modèle d'imputation sauvegardé: {imp_path}")

            except Exception as e:
                print(f"❌ Erreur lors de l'imputation MAR: {e}")
                print(f" Utilisation de la méthode de secours: {backup_method}")
                median_fill(df_proc, available_mar_cols)
                suffix = f'{backup_method}_backup'

        # Traitement conditionnel des autres colonnes
        if treat_other_cols:
            available_mcar_cols = [col for col in mcar_cols if col in df_proc.columns]
            if available_mcar_cols:
                if display_info:
                    print(f" Variables MCAR à imputer avec médiane: {len(available_mcar_cols)}")
                    for col in available_mcar_cols:
                        count = df_proc[col].isnull().sum()
                        print(f"   • {col}: {count} valeurs manquantes")
                median_fill(df_proc, available_mcar_cols)
        else:
            if display_info:
                print(" Traitement des autres colonnes désactivé (treat_other_cols=False)")

    else:
        raise ValueError("❌ strategy doit être 'all_median' ou 'mixed_mar_mcar'.")

    final_missing = df_proc.isnull().sum().sum()
    if display_info:
        print("\n Résumé de l'imputation:")
        print(f"   • Valeurs manquantes avant: {initial_missing}")
        print(f"   • Valeurs manquantes après: {final_missing}")
        
        # Vérification des colonnes traitées
        if final_missing > 0:
            remaining_cols = df_proc.columns[df_proc.isnull().any()].tolist()
            print(f" Colonnes avec valeurs manquantes restantes: {remaining_cols}")
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

    start_time = time.time()

    if verbose:
        print("🔍 Recherche de la valeur optimale K pour l'imputation KNN")
        print("=" * 60)
        print(f" Colonnes à évaluer      : {columns_to_impute}")
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
                # Calculer le score selon la métrique demandée
                mse = mean_squared_error(true_values, imputed_values)

                if metric.lower() == 'rmse':
                    score = np.sqrt(mse)
                elif metric.lower() == 'mae':
                    score = mean_absolute_error(true_values, imputed_values)
                else:  # 'mse' est le cas par défaut
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
        print(f" Meilleur score ({metric.upper()})  : {best_score:.4f}")
        print(f"⏱️  Temps de calcul        : {computation_time:.2f}s")

    return {
        'optimal_k': optimal_k,
        'best_score': best_score,
        'results_df': pd.DataFrame(results)
    }

# =============================================================================
# 5. FONCTIONS D'ANALYSE ET DE VISUALISATION
# =============================================================================

def analyze_continuous_variables(df: pd.DataFrame, continuous_cols: List[str], target_col: str = "y", save_figures_path: str = None) -> dict:
    """
    Analyse complète des variables continues :
    - Statistiques descriptives
    - Asymétrie, aplatissement, test de normalité (Shapiro)
    - Outliers (IQR)
    - Corrélation avec la cible
    - Matrice de corrélation + heatmap
    """
    summary_stats = df[continuous_cols].describe()
    print(" Statistiques descriptives :")
    print(summary_stats)

    skew_kurtosis_results = {}
    outliers_summary = {}
    correlations = {}

    print("\n Analyse de la distribution :")
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

    print("\n Matrice de corrélation entre variables continues :")
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

            # Enregistrement si output_dir est fourni
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
    print(correlations.head(11))  # Inclut la cible elle-même

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

# =============================================================================
# 6. FONCTIONS UTILITAIRES DE RÉDUCTION DE CORRÉLATION
# =============================================================================

def find_highly_correlated_groups(
    df: pd.DataFrame, 
    threshold: float = 0.90, 
    exclude_cols: List[str] = None,
    show_plot: bool = False,
    save_path: Union[str, Path] = None
):
    """
    Identifie les groupes de variables fortement corrélées (|corr| > threshold).
    
    Args:
        df: DataFrame à analyser
        threshold: Seuil de corrélation
        exclude_cols: Colonnes à exclure de l'analyse
        show_plot: Afficher la heatmap
        save_path: Chemin de sauvegarde de la heatmap
    
    Returns:
        dict: Résultats avec groupes et métadonnées
    """
    if exclude_cols is None:
        exclude_cols = []
    
    # Filtrer les colonnes à analyser
    cols_to_analyze = [col for col in df.columns if col not in exclude_cols]
    df_filtered = df[cols_to_analyze]
    
    corr_matrix = df_filtered.corr().abs()
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

    # Optionnel: générer la heatmap
    if show_plot or save_path:
        plt.figure(figsize=(10, 8))
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        sns.heatmap(corr_matrix, mask=mask, annot=False, cmap='coolwarm', 
                   center=0, square=True, linewidths=0.5)
        plt.title(f'Matrice de corrélation (seuil: {threshold})')
        plt.tight_layout()
        
        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✅ Heatmap sauvegardée : {save_path}")
        
        if show_plot:
            plt.show()
        else:
            plt.close()

    # Calculer les variables à supprimer (toutes sauf la première de chaque groupe)
    to_drop = []
    for group in groups:
        if len(group) > 1:
            to_drop.extend(group[1:])  # Garder la première, supprimer les autres

    return {
        'groups': groups,
        'to_drop': to_drop,
        'threshold': threshold,
        'total_groups': len(groups),
        'excluded_columns': exclude_cols,
        'analyzed_columns': cols_to_analyze
    }


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
        print(f"\n Résumé de la réduction :")
        print(f"🔻 {len(to_drop)} colonnes binaires supprimées (corrélées)")
        print(f"✅ {len(to_keep)} colonnes binaires conservées (1 par groupe)")
        print(f"➕ {len(untouched)} colonnes binaires non corrélées conservées")
        if extra_cols:
            print(f"🧩 {len(extra_cols)} variables continues / contextuelles ajoutées : {extra_cols}")
        print(f"📐 DataFrame final : {df_reduced.shape[1]} colonnes, {df_reduced.shape[0]} lignes")

    return df_reduced, to_drop, to_keep


def detect_outliers_iqr(series: pd.Series, multiplier: float = 1.5) -> pd.Series:
    """
    Détecte les outliers selon la règle de l'IQR.
    """
    Q1 = series.quantile(0.25)
    Q3 = series.quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - multiplier * IQR
    upper = Q3 + multiplier * IQR
    return (series < lower) | (series > upper)

# =============================================================================
# 7. FONCTIONS DE PREPROCESSING FINAL
# =============================================================================

def prepare_final_dataset_with_correlation_reduction(
    data_path: Union[str, Path],
    target_col: str = 'outcome',
    continuous_cols: List[str] = None,
    correlation_threshold: float = 0.95,
    save_dir: Optional[Union[str, Path]] = None,
    verbose: bool = True
) -> Tuple[pd.DataFrame, Dict]:
    """
    Pipeline complet de préparation des données avec réduction de corrélation.
    
    Returns:
        Tuple[pd.DataFrame, Dict]: (données_finales, métadonnées)
    """
    if continuous_cols is None:
        continuous_cols = ['X1', 'X2', 'X3', 'X4']
    
    if verbose:
        print("🚀 Début du preprocessing complet avec réduction de corrélation")
        print("=" * 70)
    
    # 1. Chargement des données
    df, load_report = load_and_clean_data(data_path, target_col, display_info=verbose)
    
    # 2. Identification des groupes corrélés (seulement variables binaires)
    binary_cols = [col for col in df.select_dtypes(include='int64').columns if col != target_col]
    
    if len(binary_cols) > 1:
        if verbose:
            print(f"\n🔍 Recherche de corrélations élevées parmi {len(binary_cols)} variables binaires...")
        
        correlated_groups = find_highly_correlated_groups(
            df[binary_cols], 
            threshold=correlation_threshold
        )
        
        if verbose:
            print(f" {len(correlated_groups)} groupes de variables corrélées identifiés")
        
        # 3. Réduction de dimensionnalité
        df_reduced, dropped_cols, kept_cols = drop_correlated_duplicates(
            df, 
            correlated_groups,
            target_col=target_col,
            extra_cols=continuous_cols,
            verbose=verbose,
            summary=verbose
        )
    else:
        df_reduced = df.copy()
        dropped_cols = []
        kept_cols = []
        if verbose:
            print(" Pas assez de variables binaires pour l'analyse de corrélation")
    
    # 4. Métadonnées finales
    metadata = {
        'original_shape': df.shape,
        'final_shape': df_reduced.shape,
        'correlation_threshold': correlation_threshold,
        'correlated_groups_found': len(correlated_groups) if 'correlated_groups' in locals() else 0,
        'dropped_columns': dropped_cols,
        'kept_columns': kept_cols,
        'continuous_columns': continuous_cols,
        'reduction_ratio': (len(dropped_cols) / df.shape[1]) * 100 if df.shape[1] > 0 else 0
    }
    
    # 5. Sauvegarde optionnelle
    if save_dir:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Sauvegarder les données réduites
        data_path = save_dir / "df_correlation_reduced.csv"
        df_reduced.to_csv(data_path, index=False)
        
        # Sauvegarder les métadonnées
        metadata_path = save_dir / "correlation_reduction_metadata.json"
        import json
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        
        if verbose:
            print(f"\n💾 Données sauvegardées : {data_path}")
            print(f"💾 Métadonnées sauvegardées : {metadata_path}")
    
    if verbose:
        print("\n" + "=" * 70)
        print("✅ Preprocessing complet terminé avec succès")
        print(f" Réduction : {df.shape[1]} → {df_reduced.shape[1]} colonnes ({metadata['reduction_ratio']:.1f}% réduit)")
    
    return df_reduced, metadata

def apply_collinearity_filter(
    df: pd.DataFrame,
    cols_to_drop: List[str],
    imputation_method: str = 'knn',
    models_dir: Union[str, Path] = None,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Applique un filtre de collinéarité en supprimant les colonnes spécifiées.
    
    Args:
        df: DataFrame à filtrer
        cols_to_drop: Liste des colonnes à supprimer
        imputation_method: Méthode d'imputation utilisée (pour info)
        models_dir: Répertoire des modèles (pour cohérence avec l'interface)
        verbose: Affichage des informations
    
    Returns:
        DataFrame filtré
    """
    if verbose:
        print(f" DataFrame initial: {df.shape}")
        print(f" Colonnes à supprimer: {len(cols_to_drop)}")
        if cols_to_drop:
            print(f"   Variables: {cols_to_drop[:10]}{'...' if len(cols_to_drop) > 10 else ''}")
    
    # Supprimer les colonnes qui existent dans le DataFrame
    existing_cols_to_drop = [col for col in cols_to_drop if col in df.columns]
    df_filtered = df.drop(columns=existing_cols_to_drop)
    
    if verbose:
        print(f" DataFrame filtré: {df_filtered.shape}")
        print(f"✅ {len(existing_cols_to_drop)} colonnes supprimées")
        if len(existing_cols_to_drop) != len(cols_to_drop):
            missing_cols = set(cols_to_drop) - set(existing_cols_to_drop)
            print(f" {len(missing_cols)} colonnes introuvables: {list(missing_cols)[:5]}{'...' if len(missing_cols) > 5 else ''}")
    
    return df_filtered

# =============================================================================
# 5. FONCTIONS DE VISUALISATION MANQUANTES
# =============================================================================

def plot_continuous_by_class(df, continuous_cols, output_dir, figsize=(15, 5)):
    """Distribution des variables continues par classe."""
    fig, axes = plt.subplots(1, len(continuous_cols), figsize=figsize)
    for i, col in enumerate(continuous_cols):
        df_clean = df[[col, 'y']].dropna()
        df_clean['y_label'] = df_clean['y'].map({0: 'Non-pub', 1: 'Pub'})
        sns.violinplot(data=df_clean, x='y_label', y=col, ax=axes[i])
        axes[i].set_title(f'{col} par classe')
    plt.tight_layout()
    
    output_path = Path(output_dir) / 'continuous_by_class.png'
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✅ Figure sauvegardée : {output_path}")
    plt.show()

def plot_continuous_target_corr(df, continuous_cols, output_dir):
    """Corrélations des variables continues avec la cible."""
    fig, ax = plt.subplots(figsize=(8, 4))
    corr_data = df[continuous_cols + ['y']].corr()['y'][:-1].to_frame()
    sns.heatmap(corr_data, annot=True, cmap='coolwarm', center=0,
                cbar_kws={'label': 'Corrélation'}, fmt='.3f', ax=ax)
    ax.set_title('Corrélations avec la cible')
    plt.tight_layout()
    
    output_path = Path(output_dir) / 'continuous_target_correlation.png'
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✅ Figure sauvegardée : {output_path}")
    plt.show()

def plot_binary_sparsity(df, binary_cols, output_dir, sample_size=100):
    """Visualisation de la sparsité des variables binaires."""
    sample_vars = np.random.choice(binary_cols, size=min(sample_size, len(binary_cols)), replace=False)
    sample_data = df[sample_vars].head(sample_size)
    plt.figure(figsize=(8, 4))
    plt.imshow(sample_data.values, cmap='binary', aspect='auto')
    plt.colorbar(label='Valeur (0 ou 1)')
    plt.title('Sparsité (100 obs × 100 var binaires)')
    plt.xlabel('Variables')
    plt.ylabel('Observations')
    plt.tight_layout()
    
    output_path = Path(output_dir) / 'sparsity_visualization.png'
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✅ Figure sauvegardée : {output_path}")
    plt.show()

def plot_eda_summary(df, continuous_cols, binary_cols, target_corr, sparsity, imbalance_ratio, output_dir, presence_series):
    """Résumé visuel de l'EDA."""
    fig = plt.figure(figsize=(14, 8))

    # Distribution de la cible
    ax1 = plt.subplot(2, 3, 1)
    df['y'].map({0: 'noad.', 1: 'ad.'}).value_counts().plot.pie(
        ax=ax1, autopct='%1.1f%%', colors=['#3498db', '#e74c3c'])
    ax1.set_title('Cible')
    ax1.set_ylabel('')

    # Valeurs manquantes
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

    # Corrélations avec cible
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
    
    output_path = Path(output_dir) / 'eda_summary.png'
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✅ Figure sauvegardée : {output_path}")
    plt.show()

def plot_outlier_comparison(
    df_before: pd.DataFrame,
    df_after: pd.DataFrame,
    cols: List[str],
    output_dir: Union[str, Path],
    show: bool = True,
    save: bool = True,
    dpi: int = 100
):
    """Affiche les boxplots avant/après traitement des outliers."""
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
            output_path = output_dir / f"{col}_outliers_comparison.png"
            plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
            print(f"✅ Figure sauvegardée : {output_path}")
        
        if show:
            plt.show()
        
        plt.close()

def generer_graphiques_comparaison(df_avant, df_apres, figures_dir):
    """Génère des graphiques de comparaison avant/après transformation."""
    figures_dir = Path(figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)
    
    vars_to_compare = ['X1', 'X2', 'X3']
    transformed_vars = ['X1_transformed', 'X2_transformed', 'X3_transformed']
    
    for original, transformed in zip(vars_to_compare, transformed_vars):
        if original in df_avant.columns and transformed in df_apres.columns:
            fig, axes = plt.subplots(1, 2, figsize=(12, 4))
            
            # Avant transformation
            df_avant[original].dropna().hist(bins=50, ax=axes[0], alpha=0.7, color='lightcoral')
            axes[0].set_title(f'{original} - Avant transformation')
            axes[0].set_xlabel(original)
            axes[0].set_ylabel('Fréquence')
            
            # Après transformation
            df_apres[transformed].dropna().hist(bins=50, ax=axes[1], alpha=0.7, color='lightgreen')
            axes[1].set_title(f'{transformed} - Après transformation')
            axes[1].set_xlabel(transformed)
            axes[1].set_ylabel('Fréquence')
            
            plt.tight_layout()
            
            output_path = figures_dir / f'transformation_comparison_{original}.png'
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            print(f"✅ Figure sauvegardée : {output_path}")
            plt.show()
            plt.close()

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

    # Sélection des variables binaires par quartile de taux de présence
    presence_series = pd.Series(presence_rates)
    quartiles = presence_series.quantile([0.25, 0.5, 0.75])

    vars_q1 = presence_series[presence_series <= quartiles[0.25]].sample(10, random_state=42).index.tolist()
    vars_q2 = presence_series[(presence_series > quartiles[0.25]) & (presence_series <= quartiles[0.5])].sample(10, random_state=42).index.tolist()
    vars_q3 = presence_series[(presence_series > quartiles[0.5]) & (presence_series <= quartiles[0.75])].sample(10, random_state=42).index.tolist()
    vars_q4 = presence_series[presence_series > quartiles[0.75]].sample(10, random_state=42).index.tolist()

    selected_vars = continuous_cols + vars_q1 + vars_q2 + vars_q3 + vars_q4
    print(f"Variables sélectionnées : {len(selected_vars)} (3 continues + 40 binaires)")

    # Matrice de corrélation
    corr_matrix = df_study[selected_vars + ['y']].corr()

    # Corrélations avec la cible
    target_corr = corr_matrix['y'].drop('y').sort_values(ascending=False)
    print("\nTop 10 corrélations avec la cible (y) :")
    for var, corr in target_corr.head(10).items():
        print(f"  - {var:<20} : {corr:.4f}")

    print("\nBottom 10 corrélations avec la cible (y) :")
    for var, corr in target_corr.tail(10).items():
        print(f"  - {var:<20} : {corr:.4f}")

    # Heatmap
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
    print(f"\nHeatmap sauvegardée dans : {heatmap_path}")

    # Corrélations entre features
    upper_triangle = corr_matrix.where(np.triu(np.ones_like(corr_matrix, dtype=bool), k=1))
    high_corr_pairs = [
        {'var1': c1, 'var2': c2, 'corr': upper_triangle.loc[c1, c2]}
        for c1 in upper_triangle.columns
        for c2 in upper_triangle.columns
        if c1 != c2 and not pd.isna(upper_triangle.loc[c1, c2]) and abs(upper_triangle.loc[c1, c2]) > 0.8
    ]

    print("\nPaires fortement corrélées entre variables (|r| > 0.8) :")
    if high_corr_pairs:
        print(f"⚠️ {len(high_corr_pairs)} paires trouvées.")
        for pair in high_corr_pairs[:5]:
            print(f"  - {pair['var1']} vs {pair['var2']} : r = {pair['corr']:.3f}")
    else:
        print("✅ Aucune paire avec |r| > 0.8")

    # Résumé
    print("\nRésumé final :")
    print(f"  - Corrélation max avec la cible y : {abs(target_corr).max():.3f}")
    print(f"  - Total de variables analysées : {len(selected_vars)}")
    print(f"  - Multicolinéarité modérée : {'Oui' if high_corr_pairs else 'Non'}")

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

def plot_transformed_distributions(df, transformed_vars, output_dir, figsize=(6, 2)):
    """Génère des histogrammes et boxplots pour les variables transformées."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for col in transformed_vars:
        if col not in df.columns:
            print(f"⚠️ Colonne '{col}' introuvable, ignorée.")
            continue
            
        fig, ax = plt.subplots(1, 2, figsize=figsize)

        # Histogramme + KDE
        sns.histplot(x=df[col], bins=30, kde=True, ax=ax[0], color="mediumseagreen")
        ax[0].set_title(f"{col} - Histogramme")
        ax[0].set_xlabel(col)

        # Boxplot
        sns.boxplot(x=df[col], ax=ax[1], color="salmon")
        ax[1].set_title(f"{col} - Boxplot")

        plt.tight_layout()

        # Sauvegarde
        fig_path = output_dir / f"{col}_distribution_boxplot.png"
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        plt.show()

        print(f"✅ Figure sauvegardée : {fig_path}")
        plt.close()