"""
Module consolidé d'utilitaires - STA211 Project
Consolidation de: build_comparison_table.py, visualization.py, prediction.py

Auteur: Maoulida Abdoullatuf
Version: 4.0 (consolidée)
"""

import pandas as pd
import numpy as np
import json
import joblib
import logging
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from sklearn.feature_selection import RFECV
from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Configuration du logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# =============================================================================
# 1. CONSTRUCTION DE TABLES DE COMPARAISON (ex-build_comparison_table.py)
# =============================================================================

def build_comparison_table(json_results_paths: List[Union[str, Path]], model_details: Dict) -> pd.DataFrame:
    """
    Construit un DataFrame comparatif à partir de fichiers JSON de résultats.

    Args:
        json_results_paths (list): Liste de chemins vers les fichiers JSON de résultats.
        model_details (dict): Dictionnaire {nom_fichier_json: {details_du_modèle}}.

    Returns:
        pd.DataFrame: DataFrame des comparaisons trié par F1-score test.
    """
    results = []
    for json_path in json_results_paths:
        json_path = Path(json_path)
        filename = json_path.name
        try:
            with open(json_path, 'r') as f:
                data = json.load(f)

            base_info = model_details.get(filename, {})
            nom_affiche = base_info.get("Nom Affiché", filename.replace(".json", ""))
            type_model = base_info.get("Type", "Inconnu")
            imputation = base_info.get("Imputation", "Inconnue")

            perf = data.get("performance", {})
            # Priorité au F1-score sur test, sinon sur val
            f1_test = perf.get("f1_score_test")
            f1_val = perf.get("f1_score_val")
            f1_to_use = f1_test if f1_test is not None else f1_val

            precision_test = perf.get("precision_test")
            recall_test = perf.get("recall_test")
            
            # Si F1-test n'est pas dispo, on peut utiliser F1-val ou le définir à None
            precision_to_use = precision_test if f1_test is not None else perf.get("precision_val")
            recall_to_use = recall_test if f1_test is not None else perf.get("recall_val")

            threshold = data.get("threshold")

            if f1_to_use is not None:
                results.append({
                    'Modèle': nom_affiche,
                    'Type': type_model,
                    'Imputation': imputation,
                    'F1-score (test)': f1_to_use,
                    'Précision (test)': precision_to_use,
                    'Rappel (test)': recall_to_use,
                    'Seuil utilisé': threshold
                })
            else:
                print(f"⚠️ Score F1 manquant dans {json_path}")

        except FileNotFoundError:
            print(f"❌ Fichier non trouvé : {json_path}")
        except json.JSONDecodeError:
            print(f"❌ Erreur de décodage JSON dans {json_path}")
        except Exception as e:
             print(f"❌ Erreur lors du traitement de {json_path} : {e}")

    if not results:
        print("Aucun résultat valide à afficher.")
        return pd.DataFrame()

    df_results = pd.DataFrame(results)
    df_results = df_results.sort_values(by='F1-score (test)', ascending=False).reset_index(drop=True)
    return df_results

# =============================================================================
# 2. FONCTIONS DE VISUALISATION (ex-visualization.py) 
# =============================================================================

def plot_feature_selection_performance(cv_scores: np.ndarray, n_features_list: List[int], 
                                      title: str = "Performance par nombre de features",
                                      figsize: Tuple[int, int] = (10, 6)) -> None:
    """
    Visualise les performances de sélection de features.
    """
    plt.figure(figsize=figsize)
    
    # Calculer les statistiques
    mean_scores = np.mean(cv_scores, axis=1)
    std_scores = np.std(cv_scores, axis=1)
    
    # Plot avec barres d'erreur
    plt.plot(n_features_list, mean_scores, 'b-', marker='o', markersize=5)
    plt.fill_between(n_features_list, 
                     mean_scores - std_scores, 
                     mean_scores + std_scores, 
                     alpha=0.2, color='blue')
    
    # Trouver le meilleur score
    best_idx = np.argmax(mean_scores)
    best_n_features = n_features_list[best_idx]
    best_score = mean_scores[best_idx]
    
    plt.axvline(x=best_n_features, color='red', linestyle='--', alpha=0.7,
                label=f'Optimal: {best_n_features} features (score={best_score:.4f})')
    
    plt.xlabel('Nombre de features')
    plt.ylabel('Score de validation croisée')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def plot_feature_importance_comparison(importance_data: Dict[str, Dict], 
                                     top_n: int = 15,
                                     figsize: Tuple[int, int] = (12, 8)) -> None:
    """
    Compare l'importance des features entre différentes méthodes.
    
    Args:
        importance_data: Dict avec {methode: {feature: importance}}
        top_n: Nombre de top features à afficher
        figsize: Taille de la figure
    """
    if not importance_data:
        print("Aucune donnée d'importance fournie")
        return
    
    # Créer un DataFrame pour faciliter la visualisation
    all_features = set()
    for method_data in importance_data.values():
        all_features.update(method_data.keys())
    
    df_importance = pd.DataFrame(index=sorted(all_features))
    
    for method, features_imp in importance_data.items():
        df_importance[method] = pd.Series(features_imp)
    
    # Remplir les NaN avec 0
    df_importance = df_importance.fillna(0)
    
    # Calculer l'importance moyenne et prendre le top_n
    df_importance['mean_importance'] = df_importance.mean(axis=1)
    top_features = df_importance.nlargest(top_n, 'mean_importance')
    
    # Supprimer la colonne moyenne pour le plot
    top_features = top_features.drop('mean_importance', axis=1)
    
    # Créer le graphique
    fig, ax = plt.subplots(figsize=figsize)
    top_features.plot(kind='bar', ax=ax, width=0.8)
    
    plt.title(f'Comparaison de l\'importance des features (Top {top_n})')
    plt.xlabel('Features')
    plt.ylabel('Importance')
    plt.xticks(rotation=45, ha='right')
    plt.legend(title='Méthode', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def plot_model_performance_comparison(results_df: pd.DataFrame, 
                                    metric_col: str = 'F1-score (test)',
                                    figsize: Tuple[int, int] = (12, 8)) -> None:
    """
    Visualise la comparaison des performances entre modèles.
    """
    if results_df.empty:
        print("DataFrame vide - aucune donnée à visualiser")
        return
    
    plt.figure(figsize=figsize)
    
    # Créer le graphique en barres
    bars = plt.bar(range(len(results_df)), results_df[metric_col], 
                   color=['green' if i == 0 else 'lightblue' for i in range(len(results_df))])
    
    # Ajouter les valeurs sur les barres
    for i, (bar, value) in enumerate(zip(bars, results_df[metric_col])):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                f'{value:.4f}', ha='center', va='bottom', fontweight='bold' if i == 0 else 'normal')
    
    plt.xlabel('Modèles')
    plt.ylabel(metric_col)
    plt.title(f'Comparaison des modèles - {metric_col}')
    plt.xticks(range(len(results_df)), results_df['Modèle'], rotation=45, ha='right')
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.show()

# =============================================================================
# 3. PIPELINE DE PRÉDICTIONS (ex-prediction.py)
# =============================================================================

class PredictionPipeline:
    """Pipeline complet pour générer les prédictions finales avec les vrais modèles"""
    
    def __init__(self, base_dir: Union[str, Path] = "."):
        """
        Initialise le pipeline avec les chemins du projet.
        """
        self.base_dir = Path(base_dir)
        self.models_dir = self.base_dir / "models"
        self.outputs_dir = self.base_dir / "outputs"
        self.data_dir = self.base_dir / "data"
        
        logger.info(f"Pipeline initialisé avec base_dir: {self.base_dir}")
    
    def load_and_preprocess_test_data(self) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Charge et prétraite les données de test exactement comme dans les notebooks.
        """
        logger.info("Chargement et prétraitement des données de test...")
        
        # Charger le fichier CSV
        test_path = self.data_dir / "raw" / "data_test.csv"
        logger.info(f"Chargement des données depuis : {test_path}")
        
        # Lire d'abord pour voir la structure
        df_test = pd.read_csv(test_path)
        logger.info(f"Données chargées : {df_test.shape[0]} lignes, {df_test.shape[1]} colonnes")
        
        # Convertir toutes les colonnes (sauf la première qui pourrait être l'ID) en numérique
        for col in df_test.columns:
            if col not in ['id']:  # Garder les IDs comme strings
                df_test[col] = pd.to_numeric(df_test[col], errors='coerce')
        
        # Vérifier les conversions
        numeric_cols = df_test.select_dtypes(include=[np.number]).columns
        logger.info(f"Colonnes numériques converties : {len(numeric_cols)}")
        
        # Vérifier s'il y a des NaN après conversion (sauf pour les valeurs manquantes légitimes)
        problematic_cols = []
        for col in df_test.columns:
            if col not in ['id'] and df_test[col].isna().any():
                na_count = df_test[col].isna().sum()
                if na_count < len(df_test) * 0.9:  # Si moins de 90% de NaN, c'est un problème de conversion
                    problematic_cols.append(f"{col}: {na_count} NaN")
        
        if problematic_cols:
            logger.warning(f"Colonnes avec NaN après conversion : {problematic_cols[:5]}")  # Limite à 5 pour le log
        
        # Séparer features et IDs
        if 'id' in df_test.columns:
            ids = df_test['id']
            features = df_test.drop(columns=['id'])
        else:
            ids = pd.Series(range(len(df_test)))
            features = df_test
        
        return features, ids
    
    def apply_notebook1_preprocessing(self, features: pd.DataFrame, imputation_method: str = "knn") -> pd.DataFrame:
        """
        Applique le prétraitement exact du notebook 1.
        """
        logger.info(f"Application du prétraitement notebook 1 avec {imputation_method.upper()}...")
        
        df = features.copy()
        
        # Définir les chemins selon la méthode d'imputation
        if imputation_method == "knn":
            base_path = self.outputs_dir / "modeling" / "notebook1" / "knn"
        else:
            base_path = self.outputs_dir / "modeling" / "notebook1" / "mice"
        
        # 1. Imputation X4 avec médiane
        median_path = base_path / "median_imputer_X4.pkl"
        if median_path.exists():
            median_value = joblib.load(median_path)
            df['X4'] = df['X4'].fillna(median_value)
            logger.info(f"X4 imputé avec médiane : {median_value}")
        
        # 2. Imputation des variables continues X1, X2, X3
        continuous_cols = ['X1', 'X2', 'X3']
        if imputation_method == "knn":
            imputer_path = base_path / "imputer_knn_k7.pkl"
        else:
            imputer_path = base_path / "imputer_mice_custom.pkl"
        
        if imputer_path.exists():
            imputer = joblib.load(imputer_path)
            df[continuous_cols] = imputer.transform(df[continuous_cols])
            logger.info(f"Variables continues imputées avec {imputation_method.upper()}")
        
        # 3. Transformations Yeo-Johnson et Box-Cox
        transformers_dir = base_path / f"{imputation_method}_transformers"
        
        # Yeo-Johnson pour X1 et X2
        yj_path = transformers_dir / "yeo_johnson_transformer.pkl"
        if yj_path.exists():
            yj_transformer = joblib.load(yj_path)
            df[['X1', 'X2']] = yj_transformer.transform(df[['X1', 'X2']])
            df.rename(columns={'X1': 'X1_transformed', 'X2': 'X2_transformed'}, inplace=True)
        
        # Box-Cox pour X3
        bc_path = transformers_dir / "box_cox_transformer.pkl"
        if bc_path.exists():
            bc_transformer = joblib.load(bc_path)
            # Assurer que X3 > 0 pour Box-Cox
            df['X3'] = df['X3'].clip(lower=1e-6)
            df[['X3']] = bc_transformer.transform(df[['X3']])
            df.rename(columns={'X3': 'X3_transformed'}, inplace=True)
        
        # 4. Capping des outliers
        capping_path = base_path / f"capping_params_{imputation_method}.pkl"
        if capping_path.exists():
            capping_params = joblib.load(capping_path)
            for col in ['X1_transformed', 'X2_transformed', 'X3_transformed']:
                if col in capping_params and col in df.columns:
                    lower, upper = capping_params[col]
                    df[col] = df[col].clip(lower=lower, upper=upper)
            logger.info("Capping des outliers appliqué")
        
        # 5. Features polynomiales
        poly_path = base_path / f"poly_transformer_{imputation_method}_no_outliers.pkl"
        if poly_path.exists():
            poly_transformer = joblib.load(poly_path)
            transformed_cols = ['X1_transformed', 'X2_transformed', 'X3_transformed']
            if all(col in df.columns for col in transformed_cols):
                poly_features = poly_transformer.transform(df[transformed_cols])
                poly_names = poly_transformer.get_feature_names_out(transformed_cols)
                
                for i, name in enumerate(poly_names):
                    df[name] = poly_features[:, i]
                
                logger.info(f"Features polynomiales ajoutées : {len(poly_names)} colonnes")
        
        # 6. Supprimer les colonnes corrélées
        cols_to_drop_path = base_path / "cols_to_drop_corr.pkl"
        if cols_to_drop_path.exists():
            cols_to_drop = joblib.load(cols_to_drop_path)
            cols_to_drop = [col for col in cols_to_drop if col in df.columns]
            if cols_to_drop:
                df = df.drop(columns=cols_to_drop)
                logger.info(f"Colonnes corrélées supprimées : {len(cols_to_drop)}")
        
        return df
    
    def generate_predictions_with_best_model(self) -> pd.DataFrame:
        """
        Génère les prédictions avec le meilleur modèle (GradBoost KNN Reduced).
        """
        logger.info("="*80)
        logger.info("GÉNÉRATION DES PRÉDICTIONS AVEC LE MODÈLE CHAMPION")
        logger.info("="*80)
        
        # 1. Charger et prétraiter les données
        features, ids = self.load_and_preprocess_test_data()
        X_processed = self.apply_notebook1_preprocessing(features, imputation_method="knn")
        
        # 2. Charger le modèle champion et ses paramètres
        logger.info("Chargement du modèle champion : GradBoost KNN Reduced")
        
        # Pipeline du modèle
        model_path = self.models_dir / "best_gradboost_knn_reduced.joblib"
        if not model_path.exists():
            # Essayer un autre chemin
            model_path = self.models_dir / "pipeline_gradboost_knn_reduced.joblib"
        
        if not model_path.exists():
            raise FileNotFoundError(f"Modèle non trouvé : {model_path}")
        
        model = joblib.load(model_path)
        logger.info(f"Modèle chargé depuis : {model_path}")
        
        # Seuil optimal
        threshold_path = self.outputs_dir / "modeling" / "thresholds" / "optimized_thresholds_knn_reduced.json"
        if threshold_path.exists():
            with open(threshold_path, 'r') as f:
                thresholds = json.load(f)
                optimal_threshold = thresholds.get('GradBoost', {}).get('threshold', 0.285)
        else:
            optimal_threshold = 0.285  # Valeur du champion selon vos résultats
        
        logger.info(f"Seuil optimal : {optimal_threshold}")
        
        # 3. Sélection des features si modèle reduced
        selected_features_path = self.models_dir / "knn" / "reduced" / "selected_columns_knn.pkl"
        if selected_features_path.exists():
            selected_columns = joblib.load(selected_features_path)
            # Garder seulement les colonnes qui existent
            selected_columns = [col for col in selected_columns if col in X_processed.columns]
            X_final = X_processed[selected_columns]
            logger.info(f"Features sélectionnées : {len(selected_columns)} colonnes")
        else:
            X_final = X_processed
        
        # 4. Générer les prédictions
        logger.info("Génération des prédictions...")
        y_proba = model.predict_proba(X_final)[:, 1]
        predictions = (y_proba >= optimal_threshold).astype(int)
        
        # 5. Créer le DataFrame de soumission
        submission = pd.DataFrame({
            'id': ids,
            'outcome': predictions
        })
        
        # Convertir en format attendu
        submission['outcome'] = submission['outcome'].map({0: 'noad.', 1: 'ad.'})
        
        # 6. Sauvegarder
        output_path = self.outputs_dir / "predictions" / "submission_champion.csv"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        submission.to_csv(output_path, index=False)
        
        # 7. Statistiques
        logger.info("\nRÉSULTATS FINAUX")
        logger.info("-"*40)
        logger.info(f"Modèle : GradBoost KNN Reduced")
        logger.info(f"Seuil : {optimal_threshold:.4f}")
        logger.info(f"Total prédictions : {len(submission)}")
        logger.info(f"Publicités (ad.) : {np.sum(submission['outcome'] == 'ad.')} ({np.mean(submission['outcome'] == 'ad.')*100:.1f}%)")
        logger.info(f"Non-publicités (noad.) : {np.sum(submission['outcome'] == 'noad.')} ({np.mean(submission['outcome'] == 'noad.')*100:.1f}%)")
        logger.info(f"\nFichier sauvegardé : {output_path}")
        
        return submission
    
    def generate_predictions_with_stacking(self) -> pd.DataFrame:
        """
        Génère les prédictions avec le modèle de stacking.
        """
        logger.info("="*80)
        logger.info("GÉNÉRATION DES PRÉDICTIONS AVEC STACKING")
        logger.info("="*80)
        
        # 1. Charger et prétraiter les données pour KNN et MICE
        features, ids = self.load_and_preprocess_test_data()
        X_knn = self.apply_notebook1_preprocessing(features, imputation_method="knn")
        X_mice = self.apply_notebook1_preprocessing(features, imputation_method="mice")
        
        # 2. Charger les modèles de stacking
        stacking_knn_path = self.outputs_dir / "modeling" / "notebook3" / "stacking" / "stacking_knn_with_refit.joblib"
        stacking_mice_path = self.outputs_dir / "modeling" / "notebook3" / "stacking" / "stacking_mice_with_refit.joblib"
        
        if not stacking_knn_path.exists() or not stacking_mice_path.exists():
            logger.warning("Modèles de stacking non trouvés, utilisation du modèle champion")
            return self.generate_predictions_with_best_model()
        
        stacking_knn = joblib.load(stacking_knn_path)
        stacking_mice = joblib.load(stacking_mice_path)
        
        # 3. Générer les prédictions
        logger.info("Génération des prédictions avec stacking...")
        proba_knn = stacking_knn.predict_proba(X_knn)[:, 1]
        proba_mice = stacking_mice.predict_proba(X_mice)[:, 1]
        
        # Moyenne pondérée
        proba_final = 0.5 * proba_knn + 0.5 * proba_mice
        
        # Seuil optimal pour le stacking (basé sur vos résultats)
        optimal_threshold = 0.42
        predictions = (proba_final >= optimal_threshold).astype(int)
        
        # 4. Créer et sauvegarder la soumission
        submission = pd.DataFrame({
            'id': ids,
            'outcome': predictions
        })
        submission['outcome'] = submission['outcome'].map({0: 'noad.', 1: 'ad.'})
        
        output_path = self.outputs_dir / "predictions" / "submission_stacking.csv"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        submission.to_csv(output_path, index=False)
        
        logger.info(f"\nFichier sauvegardé : {output_path}")
        return submission

# =============================================================================
# 4. FONCTIONS UTILITAIRES GÉNÉRALES
# =============================================================================

def generate_final_predictions(use_stacking: bool = False, base_dir: Union[str, Path] = ".") -> pd.DataFrame:
    """
    Génère les prédictions finales avec les vrais modèles.
    
    Args:
        use_stacking: Si True, utilise le stacking. Sinon, utilise le modèle champion.
        base_dir: Répertoire de base du projet
    """
    pipeline = PredictionPipeline(base_dir)
    
    if use_stacking:
        return pipeline.generate_predictions_with_stacking()
    else:
        return pipeline.generate_predictions_with_best_model()

def calculate_model_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_proba: Optional[np.ndarray] = None) -> Dict[str, float]:
    """
    Calcule les métriques de performance d'un modèle.
    
    Args:
        y_true: Vraies étiquettes
        y_pred: Prédictions binaires
        y_proba: Probabilités prédites (optionnel)
    
    Returns:
        Dictionnaire des métriques
    """
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1_score': f1_score(y_true, y_pred, zero_division=0)
    }
    
    if y_proba is not None:
        from sklearn.metrics import roc_auc_score
        try:
            metrics['roc_auc'] = roc_auc_score(y_true, y_proba)
        except ValueError:
            # En cas d'une seule classe dans y_true
            metrics['roc_auc'] = np.nan
    
    return metrics

def save_results_to_json(results: Dict, output_path: Union[str, Path]) -> None:
    """
    Sauvegarde les résultats dans un fichier JSON.
    
    Args:
        results: Dictionnaire des résultats
        output_path: Chemin de sauvegarde
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Convertir les types numpy en types Python pour la sérialisation JSON
    def convert_numpy_types(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: convert_numpy_types(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy_types(item) for item in obj]
        return obj
    
    results_serializable = convert_numpy_types(results)
    
    with open(output_path, 'w') as f:
        json.dump(results_serializable, f, indent=2)
    
    logger.info(f"Résultats sauvegardés dans : {output_path}")
