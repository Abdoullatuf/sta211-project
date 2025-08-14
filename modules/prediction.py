# modules/prediction.py
"""
Module unifié pour la génération des prédictions finales
Projet STA211 - Classification de publicités
Version avec vrais modèles uniquement
"""

import pandas as pd
import numpy as np
import joblib
import json
import logging
from pathlib import Path
from typing import Tuple, Optional, Union

# Configuration du logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

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
    
    def load_and_preprocess_test_data(self):
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
    
    def apply_notebook1_preprocessing(self, features: pd.DataFrame, imputation_method: str = "knn"):
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
    
    def generate_predictions_with_best_model(self):
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
    
    def generate_predictions_with_stacking(self):
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

# Fonction principale
def generate_final_predictions(use_stacking: bool = False):
    """
    Génère les prédictions finales avec les vrais modèles.
    
    Args:
        use_stacking: Si True, utilise le stacking. Sinon, utilise le modèle champion.
    """
    pipeline = PredictionPipeline()
    
    if use_stacking:
        return pipeline.generate_predictions_with_stacking()
    else:
        return pipeline.generate_predictions_with_best_model()

if __name__ == "__main__":
    # Générer avec le modèle champion
    submission = generate_final_predictions(use_stacking=False)
    print("\n✅ Prédictions générées avec le modèle champion !")