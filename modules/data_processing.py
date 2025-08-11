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
        print(f"🔢 Dimensions finales : {df_result.shape}")

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
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.verbose = verbose
        self.transformers = {}
        
        if verbose:
            print(f"🏭 TransformationOptimaleMixte initialisée")
            print(f"📁 Répertoire des modèles: {self.models_dir}")
    
    def fit_transform(
        self, 
        data: pd.DataFrame, 
        method_mapping: Dict[str, str] = None,
        save_transformers: bool = True,
        prefix: str = ""
    ) -> pd.DataFrame:
        """
        Applique les transformations optimales selon le mapping spécifié.
        
        Args:
            data: DataFrame contenant les variables à transformer
            method_mapping: dict {colonne: methode} où méthode = 'yeo-johnson' ou 'box-cox'
            save_transformers: Sauvegarder les transformateurs
            prefix: Préfixe pour les fichiers de sauvegarde
            
        Returns:
            DataFrame transformé
        """
        if method_mapping is None:
            method_mapping = {
                'X1': 'yeo-johnson',
                'X2': 'yeo-johnson', 
                'X3': 'box-cox'
            }
        
        data_transformed = data.copy()
        
        for col, method in method_mapping.items():
            if col not in data.columns:
                if self.verbose:
                    print(f"⚠️ Colonne '{col}' introuvable, ignorée.")
                continue
            
            if self.verbose:
                print(f"🔄 Transformation {method} pour {col}...")
            
            # Créer et ajuster le transformateur
            if method == 'yeo-johnson':
                transformer = PowerTransformer(method='yeo-johnson', standardize=False)
            elif method == 'box-cox':
                transformer = PowerTransformer(method='box-cox', standardize=False)
            else:
                raise ValueError(f"Méthode '{method}' non supportée")
            
            # S'assurer que les valeurs sont positives pour Box-Cox
            if method == 'box-cox':
                min_val = data[col].min()
                if min_val <= 0:
                    data_transformed[col] = data[col] - min_val + 1
            
            # Ajuster et transformer
            data_transformed[col] = transformer.fit_transform(
                data_transformed[[col]]
            ).flatten()
            
            # Sauvegarder le transformateur
            self.transformers[col] = transformer
            
            if save_transformers:
                transformer_filename = f"{prefix}{method.replace('-', '_')}_transformer.pkl"
                transformer_path = self.models_dir / transformer_filename
                joblib.dump(transformer, transformer_path)
                
                if self.verbose:
                    print(f"💾 Transformateur sauvegardé: {transformer_path}")
        
        return data_transformed
    
    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Applique les transformations déjà ajustées.
        """
        if not self.transformers:
            raise ValueError("Aucun transformateur ajusté. Utilisez fit_transform() d'abord.")
        
        data_transformed = data.copy()
        
        for col, transformer in self.transformers.items():
            if col in data.columns:
                data_transformed[col] = transformer.transform(data_transformed[[col]]).flatten()
        
        return data_transformed
    
    def load_transformers(self, prefix: str = ""):
        """
        Charge les transformateurs depuis les fichiers sauvegardés.
        """
        transformer_files = {
            'X1': f"{prefix}yeo_johnson_transformer.pkl",
            'X2': f"{prefix}yeo_johnson_transformer.pkl", 
            'X3': f"{prefix}box_cox_transformer.pkl"
        }
        
        for col, filename in transformer_files.items():
            transformer_path = self.models_dir / filename
            if transformer_path.exists():
                self.transformers[col] = joblib.load(transformer_path)
                if self.verbose:
                    print(f"✅ Transformateur chargé pour {col}: {transformer_path}")
            else:
                if self.verbose:
                    print(f"⚠️ Transformateur non trouvé pour {col}: {transformer_path}")


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

        # Traitement conditionnel des autres colonnes
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
        
        # Vérification des colonnes traitées
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
        print(f"📊 Meilleur score ({metric.upper()})  : {best_score:.4f}")
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

def find_highly_correlated_groups(df: pd.DataFrame, threshold: float = 0.90):
    """
    Identifie les groupes de variables fortement corrélées (|corr| > threshold).
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
            print(f"📊 {len(correlated_groups)} groupes de variables corrélées identifiés")
        
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
            print("ℹ️ Pas assez de variables binaires pour l'analyse de corrélation")
    
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
        print(f"🔢 Réduction : {df.shape[1]} → {df_reduced.shape[1]} colonnes ({metadata['reduction_ratio']:.1f}% réduit)")
    
    return df_reduced, metadata