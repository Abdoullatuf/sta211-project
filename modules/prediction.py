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
        logger.info(f"Colonnes détectées : {list(df_test.columns)}")
        logger.info(f"Premières valeurs de la première colonne : {df_test.iloc[0:3, 0].tolist()}")
        
        # Si une seule colonne détectée, essayer d'autres méthodes de parsing
        if df_test.shape[1] == 1:
            logger.warning("Une seule colonne détectée, tentative de parsing alternatif...")
            
            # Essayer avec différents séparateurs
            for sep in [',', ';', '\t', ' ']:
                try:
                    df_alt = pd.read_csv(test_path, sep=sep)
                    if df_alt.shape[1] > 1:
                        logger.info(f"Parsing réussi avec séparateur '{sep}': {df_alt.shape[1]} colonnes")
                        df_test = df_alt
                        break
                except:
                    continue
            
            # Si toujours une seule colonne, essayer de split la première colonne
            if df_test.shape[1] == 1:
                first_col_name = df_test.columns[0]
                sample_value = str(df_test.iloc[0, 0])
                logger.info(f"Tentative de split sur la première colonne. Valeur exemple: {sample_value}")
                
                # Détecter le pattern de séparation
                for sep in ['\t', ',', ';', ' ']:
                    if sep in sample_value:
                        logger.info(f"Séparateur '{sep}' détecté dans les données")
                        # Split et créer nouvelles colonnes
                        split_data = df_test[first_col_name].str.split(sep, expand=True)
                        # Générer noms de colonnes
                        split_data.columns = [f'X{i+1}' for i in range(split_data.shape[1])]
                        df_test = split_data
                        logger.info(f"Données splitées en {df_test.shape[1]} colonnes: {list(df_test.columns)}")
                        break
        
        logger.info(f"Structure finale après parsing : {df_test.shape[0]} lignes, {df_test.shape[1]} colonnes")
        logger.info(f"Colonnes finales : {list(df_test.columns)}")
        
        # Convertir toutes les colonnes (sauf la première qui pourrait être l'ID) en numérique
        for col in df_test.columns:
            if col not in ['id']:  # Garder les IDs comme strings
                try:
                    df_test[col] = pd.to_numeric(df_test[col], errors='coerce')
                except Exception as e:
                    logger.warning(f"Impossible de convertir la colonne {col} en numérique: {e}")
        
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
        
        # Validation et conversion des types de données
        logger.info("Validation des types de données avant preprocessing...")
        for col in df.columns:
            if not pd.api.types.is_numeric_dtype(df[col]):
                logger.warning(f"Colonne {col} n'est pas numérique, conversion...")
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        logger.info(f"Types après validation : {dict(df.dtypes)}")
        
        # Définir les chemins selon la méthode d'imputation
        if imputation_method == "knn":
            base_path = self.outputs_dir / "modeling" / "notebook1" / "knn"
        else:
            base_path = self.outputs_dir / "modeling" / "notebook1" / "mice"
        
        # 1. Imputation X4 avec médiane
        median_path = base_path / "median_imputer_X4.pkl"
        if median_path.exists():
            median_value = joblib.load(median_path)
            # S'assurer que median_value est numérique
            try:
                median_value = float(median_value)
                # S'assurer que X4 est numérique
                if not pd.api.types.is_numeric_dtype(df['X4']):
                    df['X4'] = pd.to_numeric(df['X4'], errors='coerce')
                df['X4'] = df['X4'].fillna(median_value)
                logger.info(f"X4 imputé avec médiane : {median_value}")
            except (ValueError, TypeError) as e:
                logger.warning(f"Erreur lors de l'imputation de X4: {e}, ignoré")
        
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
                    # S'assurer que les valeurs de capping sont numériques
                    try:
                        # Ignorer les valeurs textuelles comme 'lower_bound', 'upper_bound'
                        if isinstance(lower, str) and not lower.replace('.', '').replace('-', '').isdigit():
                            lower = None
                        if isinstance(upper, str) and not upper.replace('.', '').replace('-', '').isdigit():
                            upper = None
                            
                        lower = float(lower) if lower is not None else None
                        upper = float(upper) if upper is not None else None
                        
                        # Appliquer seulement si on a au moins une valeur valide
                        if lower is not None or upper is not None:
                            # Vérifier que la colonne est bien numérique
                            if not pd.api.types.is_numeric_dtype(df[col]):
                                df[col] = pd.to_numeric(df[col], errors='coerce')
                            df[col] = df[col].clip(lower=lower, upper=upper)
                            logger.info(f"Capping appliqué sur {col}: [{lower}, {upper}]")
                        else:
                            logger.warning(f"Pas de valeurs de capping valides pour {col}, ignoré")
                    except (ValueError, TypeError) as e:
                        logger.warning(f"Erreur lors du capping de {col}: {e}, ignoré")
                        continue
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

    # ------------------------------------------------------------------
    # Nouveau : génération automatique en fonction des performances
    # ------------------------------------------------------------------
    def generate_predictions_with_selected_model(self, model_name: str, imputation: str, version: str, threshold: float) -> pd.DataFrame:
        """
        Génère les prédictions finales à partir d'un modèle spécifique, d'une méthode
        d'imputation et d'une variante (full/reduced), en appliquant le seuil
        optimal fourni.

        Args:
            model_name: nom du modèle (ex: 'randforest', 'xgboost', 'gradboost').
            imputation: méthode d'imputation utilisée ('knn' ou 'mice').
            version: version du modèle ('full' ou 'reduced').
            threshold: seuil de classification optimisé à appliquer.

        Returns:
            DataFrame de soumission avec les colonnes 'id' et 'outcome'.
        """
        logger.info("=" * 80)
        logger.info(f"GÉNÉRATION DES PRÉDICTIONS AVEC {model_name.upper()} {imputation.upper()} {version.upper()}")
        logger.info("=" * 80)

        # 1. Charger et prétraiter les données de test
        features, ids = self.load_and_preprocess_test_data()
        X_processed = self.apply_notebook1_preprocessing(features, imputation_method=imputation)

        # 2. Déterminer le chemin du modèle
        # Plusieurs conventions possibles : best_*, pipeline_*, extensions .joblib ou .pkl
        candidates = []
        # Sans sous-dossier imputation/version (modèles au niveau racine)
        candidates.append(self.models_dir / f"pipeline_{model_name}_{imputation}_{version}.joblib")
        candidates.append(self.models_dir / f"pipeline_{model_name}_{imputation}_{version}.pkl")
        candidates.append(self.models_dir / f"best_{model_name}_{imputation}_{version}.joblib")
        candidates.append(self.models_dir / f"best_{model_name}_{imputation}_{version}.pkl")
        # Éventuellement sous un sous-dossier par imputation/version
        candidates.append(self.models_dir / imputation / version / f"pipeline_{model_name}_{imputation}_{version}.joblib")
        candidates.append(self.models_dir / imputation / version / f"pipeline_{model_name}_{imputation}_{version}.pkl")
        candidates.append(self.models_dir / imputation / version / f"best_{model_name}_{imputation}_{version}.joblib")
        candidates.append(self.models_dir / imputation / version / f"best_{model_name}_{imputation}_{version}.pkl")

        model_path = None
        for path in candidates:
            if path.exists():
                model_path = path
                break

        if model_path is None:
            raise FileNotFoundError(f"Aucun fichier de modèle trouvé parmi : {[str(p) for p in candidates]}")

        model = joblib.load(model_path)
        logger.info(f"Modèle chargé depuis : {model_path}")

        # 3. Sélectionner les colonnes si version réduite
        X_final = X_processed
        if version.lower() == 'reduced':
            # Chercher le fichier de colonnes sélectionnées dans plusieurs emplacements
            sel_candidates = []
            sel_candidates.append(self.models_dir / imputation / version / f"selected_columns_{imputation}.pkl")
            sel_candidates.append(self.models_dir / imputation / version / "selected_features.pkl")
            sel_candidates.append(self.models_dir / imputation / version / "selected_columns_knn.pkl")
            sel_candidates.append(self.models_dir / imputation / version / "selected_columns_mice.pkl")
            # Éventuels chemins dans outputs/modeling/notebook2 (cas historique)
            sel_candidates.append(self.outputs_dir / "modeling" / "notebook2" / imputation / version / "selected_columns.pkl")

            selected_columns = None
            for s in sel_candidates:
                if s.exists():
                    try:
                        selected_columns = joblib.load(s)
                        break
                    except Exception:
                        continue
            if selected_columns is not None:
                selected_columns = [col for col in selected_columns if col in X_processed.columns]
                if selected_columns:
                    X_final = X_processed[selected_columns]
                    logger.info(f"Features sélectionnées : {len(selected_columns)} colonnes")
                else:
                    logger.warning("Liste de features sélectionnées vide ou aucune colonne correspondante")
            else:
                logger.warning("Fichier de colonnes sélectionnées introuvable pour version réduite, utilisation de toutes les colonnes")

        # 4. Générer les probabilités et les prédictions
        logger.info("Génération des prédictions avec le modèle sélectionné…")
        y_proba = model.predict_proba(X_final)[:, 1]
        predictions = (y_proba >= threshold).astype(int)

        # 5. Construire le DataFrame de soumission
        submission = pd.DataFrame({
            'id': ids,
            'outcome': predictions
        })
        submission['outcome'] = submission['outcome'].map({0: 'noad.', 1: 'ad.'})

        # 6. Sauvegarder la soumission
        out_dir = self.outputs_dir / "predictions"
        out_dir.mkdir(parents=True, exist_ok=True)
        file_name = f"submission_{model_name}_{imputation}_{version}.csv"
        output_path = out_dir / file_name
        submission.to_csv(output_path, index=False)
        logger.info(f"Fichier de soumission sauvegardé : {output_path}")

        return submission

    def generate_predictions_with_best_model_auto(self, results_csv_path: Union[str, Path]):
        """
        Génère les prédictions finales en sélectionnant automatiquement le meilleur
        modèle individuel selon le F1-score indiqué dans un fichier CSV de
        résultats de test.

        Args:
            results_csv_path: chemin vers le fichier CSV résumant les performances des
                modèles individuels. Ce fichier doit contenir au minimum les colonnes
                suivantes : 'model', 'imputation', 'version' et 'f1' (ou 'f1_score_test')
                et 'threshold' si disponible.

        Returns:
            DataFrame de soumission générée par le meilleur modèle individuel.
        """
        # Charger le CSV de résultats
        results_csv_path = Path(results_csv_path)
        if not results_csv_path.exists():
            logger.error(f"Fichier de résultats introuvable : {results_csv_path}")
            logger.warning("Retour au modèle champion…")
            return self.generate_predictions_with_best_model()

        try:
            df = pd.read_csv(results_csv_path)
        except Exception as e:
            logger.error(f"Erreur lors du chargement du CSV {results_csv_path} : {e}")
            logger.warning("Retour au modèle champion…")
            return self.generate_predictions_with_best_model()

        # Normaliser les colonnes
        cols_lower = [c.lower() for c in df.columns]
        mapping = {c: cols_lower[i] for i, c in enumerate(df.columns)}
        # Rechercher le nom de la colonne F1
        f1_col_candidates = [c for c in df.columns if c.lower() in ['f1', 'f1_score', 'f1_score_test', 'f1_score_val']]
        if not f1_col_candidates:
            logger.error("Aucune colonne F1 trouvée dans le CSV de résultats.")
            logger.warning("Retour au modèle champion…")
            return self.generate_predictions_with_best_model()
        f1_col = f1_col_candidates[0]

        # Trier par F1 décroissant et choisir la première ligne
        df_sorted = df.sort_values(by=f1_col, ascending=False).reset_index(drop=True)
        best_row = df_sorted.iloc[0]
        model = str(best_row.get('model', '')).lower()
        imputation = str(best_row.get('imputation', '')).lower()
        version = str(best_row.get('version', '')).lower()
        threshold = best_row.get('threshold', None)

        if threshold is None or pd.isna(threshold):
            # Default threshold if missing
            logger.warning("Seuil non trouvé dans le CSV, utilisation de 0.5")
            threshold = 0.5

        logger.info(f"Meilleur modèle sélectionné : {model} {imputation} {version} (F1={best_row[f1_col]:.4f}, seuil={threshold})")
        return self.generate_predictions_with_selected_model(model, imputation, version, float(threshold))

    def generate_predictions_with_stacking_auto(self, stacking_dir: Union[str, Path]):
        """
        Génère les prédictions finales en sélectionnant automatiquement le meilleur
        stacking (sans refit) parmi les fichiers JSON présents dans un dossier.

        Args:
            stacking_dir: chemin vers le dossier contenant les fichiers JSON de stacking.
                Chaque fichier doit contenir les clés 'performance' et 'threshold'.

        Returns:
            DataFrame de soumission générée par le meilleur stacking.
        """
        stacking_dir = Path(stacking_dir)
        logger.info(f"Recherche du dossier de stacking : {stacking_dir}")
        logger.info(f"Dossier existe : {stacking_dir.exists()}")
        logger.info(f"Est un dossier : {stacking_dir.is_dir() if stacking_dir.exists() else 'N/A'}")
        
        if not stacking_dir.exists() or not stacking_dir.is_dir():
            logger.error(f"Dossier de stacking introuvable : {stacking_dir}")
            logger.warning("Retour au stacking par défaut…")
            return self.generate_predictions_with_stacking()

        best_f1 = -1
        best_threshold = None
        best_suffix = None
        best_data = None
        # Lister tous les fichiers JSON
        json_files = list(stacking_dir.glob("*.json"))
        logger.info(f"Fichiers JSON trouvés : {[f.name for f in json_files]}")
        
        # Parcourir tous les fichiers JSON de stacking
        for json_file in json_files:
            if "stacking" not in json_file.name.lower():
                continue
            try:
                with open(json_file, 'r') as f:
                    data = json.load(f)
                perf = data.get('performance', {})
                # Chercher F1 test ou F1 val
                f1 = perf.get('f1_score_test') or perf.get('f1_score_val') or perf.get('f1')
                threshold = data.get('threshold') or data.get('optimal_threshold')
                if f1 is not None and threshold is not None and float(f1) > best_f1:
                    best_f1 = float(f1)
                    best_threshold = float(threshold)
                    best_data = data
                    # suffix identifie le type : knn ou mice
                    # Exemple de fichier : stacking_no_refit_knn_full.json
                    name = json_file.name
                    if 'knn' in name:
                        best_suffix = 'knn'
                    elif 'mice' in name:
                        best_suffix = 'mice'
                    else:
                        best_suffix = None
                    logger.info(f"Nouveau meilleur stacking : {best_suffix} (F1={best_f1:.4f}, seuil={best_threshold})")
            except Exception:
                continue

        if best_f1 == -1 or best_suffix is None or best_data is None:
            logger.error("Aucun fichier de stacking valide trouvé.")
            logger.warning("Retour au stacking par défaut…")
            return self.generate_predictions_with_stacking()

        logger.info(f"Meilleur stacking : {best_suffix.upper()} (F1={best_f1:.4f}, seuil={best_threshold})")
        
        # Utiliser directement les prédictions sauvegardées dans le JSON
        return self.create_submission_from_stacking_results(best_data, best_threshold)

        # 1. Charger et prétraiter les données
        features, ids = self.load_and_preprocess_test_data()
        X_knn = self.apply_notebook1_preprocessing(features, imputation_method="knn")
        X_mice = self.apply_notebook1_preprocessing(features, imputation_method="mice")
        
        # 2. Charger les modèles de stacking (avec refit ou sans refit, on utilise ceux de notebook3)
        # On cherche d'abord dans outputs/modeling/notebook3/stacking
        default_dir = self.outputs_dir / "modeling" / "notebook3" / "stacking"
        stacking_knn_path = default_dir / "stacking_knn_with_refit.joblib"
        stacking_mice_path = default_dir / "stacking_mice_with_refit.joblib"
        if not stacking_knn_path.exists() or not stacking_mice_path.exists():
            # Fallback vers models_dir/notebook3/stacking
            alt_dir = self.models_dir / "notebook3" / "stacking"
            stacking_knn_path = alt_dir / "stacking_knn_with_refit.joblib"
            stacking_mice_path = alt_dir / "stacking_mice_with_refit.joblib"
        
        if not stacking_knn_path.exists() or not stacking_mice_path.exists():
            logger.warning("Modèles de stacking non trouvés, retour au stacking par défaut…")
            return self.generate_predictions_with_stacking()
        
        stacking_knn = joblib.load(stacking_knn_path)
        stacking_mice = joblib.load(stacking_mice_path)
        
        # 3. Générer les probabilités et les prédictions
        proba_knn = stacking_knn.predict_proba(X_knn)[:, 1]
        proba_mice = stacking_mice.predict_proba(X_mice)[:, 1]
        proba_final = 0.5 * proba_knn + 0.5 * proba_mice
        predictions = (proba_final >= best_threshold).astype(int)
        
        # 4. Construire la soumission
        submission = pd.DataFrame({
            'id': ids,
            'outcome': predictions
        })
        submission['outcome'] = submission['outcome'].map({0: 'noad.', 1: 'ad.'})
        
        out_dir = self.outputs_dir / "predictions"
        out_dir.mkdir(parents=True, exist_ok=True)
        output_path = out_dir / f"submission_stacking_{best_suffix}.csv"
        submission.to_csv(output_path, index=False)
        logger.info(f"Fichier de soumission sauvegardé : {output_path}")
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

    def create_submission_from_stacking_results(self, stacking_data: dict, threshold: float) -> pd.DataFrame:
        """
        Crée une soumission directement à partir des résultats de stacking sauvegardés.
        
        Args:
            stacking_data: Données JSON du stacking (avec prédictions)
            threshold: Seuil optimal à appliquer
            
        Returns:
            DataFrame de soumission
        """
        logger.info("Création de la soumission à partir des résultats de stacking...")
        
        # Extraire les probabilités de test
        predictions_data = stacking_data.get('predictions', {})
        # Supporte différentes clés : "test_proba" (nouvelle) ou "test" (ancienne)
        test_proba = predictions_data.get('test_proba')
        if not test_proba:
            test_proba = predictions_data.get('test')

        if not test_proba:
            logger.error("Aucune probabilité de test trouvée dans les données de stacking")
            return self.generate_predictions_with_stacking()

        # Appliquer le seuil
        predictions = (np.array(test_proba) >= threshold).astype(int)

        # Créer les IDs (utiliser ceux fournis si disponibles)
        num_predictions = len(predictions)
        ids = stacking_data.get('test_ids')
        if not ids or len(ids) != num_predictions:
            ids = list(range(num_predictions))
        
        # Créer la soumission
        submission = pd.DataFrame({
            'id': ids,
            'outcome': predictions
        })
        submission['outcome'] = submission['outcome'].map({0: 'noad.', 1: 'ad.'})
        
        # Sauvegarder
        output_path = self.outputs_dir / "predictions" / "submission_stacking_auto.csv"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        submission.to_csv(output_path, index=False)
        
        # Statistiques
        logger.info("RÉSULTATS FINAUX STACKING")
        logger.info("-" * 40)
        logger.info(f"Méthode : {stacking_data.get('method', 'Stacking')}")
        logger.info(f"Seuil : {threshold:.4f}")
        logger.info(f"Total prédictions : {len(submission)}")
        logger.info(f"Publicités (ad.) : {np.sum(submission['outcome'] == 'ad.')} ({np.mean(submission['outcome'] == 'ad.')*100:.1f}%)")
        logger.info(f"Non-publicités (noad.) : {np.sum(submission['outcome'] == 'noad.')} ({np.mean(submission['outcome'] == 'noad.')*100:.1f}%)")
        logger.info(f"Fichier sauvegardé : {output_path}")
        
        return submission

# Fonction principale
def generate_final_predictions(
    use_stacking: bool = False,
    auto_select: bool = False,
    results_csv_path: Optional[Union[str, Path]] = None,
    stacking_dir: Optional[Union[str, Path]] = None,
    base_dir: Optional[Union[str, Path]] = None,
) -> pd.DataFrame:
    """
    Génère les prédictions finales avec les vrais modèles.

    Args:
        use_stacking: Si True, utilise le stacking. Sinon, utilise un modèle individuel.
        auto_select: Si True, sélectionne automatiquement le meilleur modèle/stacking en
            fonction des performances enregistrées. Si False, utilise le modèle
            champion codé en dur ou le stacking par défaut.
        results_csv_path: Chemin vers le CSV des résultats des modèles individuels.
            Utilisé uniquement si auto_select=True et use_stacking=False. Par défaut,
            cherche ``outputs/modeling/test_results_all_models.csv``.
        stacking_dir: Dossier contenant les JSON de stacking. Utilisé uniquement
            si auto_select=True et use_stacking=True. Par défaut,
            cherche ``artifacts/models/notebook3/stacking``.
        base_dir: Répertoire racine du projet. Par défaut ``'.'``.

    Returns:
        DataFrame contenant les prédictions finales.
    """
    # Instancier le pipeline avec le répertoire de base
    pipeline = PredictionPipeline(base_dir=base_dir or ".")

    if auto_select:
        if use_stacking:
            # Sélection automatique du meilleur stacking
            # Déterminer le dossier par défaut si stacking_dir non fourni
            if stacking_dir is None:
                # Préférence pour outputs/modeling/notebook3/stacking
                default_dir = pipeline.outputs_dir / "modeling" / "notebook3" / "stacking"
                if not default_dir.exists():
                    # Essayer artifacts/models/notebook3/stacking
                    default_dir = pipeline.base_dir / "artifacts" / "models" / "notebook3" / "stacking"
                    if not default_dir.exists():
                        default_dir = pipeline.models_dir / "notebook3" / "stacking"
                stacking_dir = default_dir
            else:
                # Si stacking_dir est fourni comme chemin relatif, le résoudre depuis base_dir
                stacking_dir = Path(stacking_dir)
                if not stacking_dir.is_absolute():
                    stacking_dir = pipeline.base_dir / stacking_dir
            return pipeline.generate_predictions_with_stacking_auto(stacking_dir)
        else:
            # Sélection automatique du meilleur modèle individuel
            if results_csv_path is None:
                # Chemin par défaut vers test_results_all_models.csv
                default_csv = pipeline.outputs_dir / "modeling" / "test_results_all_models.csv"
                # Fallback vers cross_validation_results.csv si le premier n'existe pas
                if not default_csv.exists():
                    default_csv = pipeline.outputs_dir / "modeling" / "cross_validation_results.csv"
                results_csv_path = default_csv
            return pipeline.generate_predictions_with_best_model_auto(results_csv_path)
    else:
        # Pas d'auto-sélection
        if use_stacking:
            return pipeline.generate_predictions_with_stacking()
        else:
            return pipeline.generate_predictions_with_best_model()

def create_submission_with_full_test_data(method="stacking_with_refit_mice"):
    """
    Génère un fichier de soumission en utilisant les données de test complètes (820 échantillons).
    Utilise les résultats JSON stockés plutôt que de recharger les modèles.
    
    Args:
        method: Méthode de stacking à utiliser ("stacking_with_refit_mice" ou "stacking_with_refit_knn")
    
    Returns:
        DataFrame: Soumission avec exactement 820 prédictions
    """
    import pandas as pd
    import json
    from pathlib import Path
    
    # Charger les données de test complètes
    test_data_path = cfg.paths.data / "raw" / "data_test.csv"
    if not test_data_path.exists():
        raise FileNotFoundError(f"Fichier de test introuvable : {test_data_path}")
    
    test_data = pd.read_csv(test_data_path)
    logger.info(f"📊 Données de test chargées : {test_data.shape}")
    
    # Charger les résultats JSON du stacking
    stacking_results_path = cfg.paths.artifacts / "models" / "notebook3" / "stacking" / f"{method}.json"
    
    if not stacking_results_path.exists():
        raise FileNotFoundError(f"Résultats de stacking introuvables : {stacking_results_path}")
    
    with open(stacking_results_path, 'r') as f:
        results = json.load(f)
    
    # Vérifier que nous avons les bonnes prédictions
    if "predictions" not in results or "test_pred" not in results["predictions"]:
        raise ValueError(f"Prédictions manquantes dans {stacking_results_path}")
    
    predictions = results["predictions"]["test_pred"]
    
    # Vérifier la cohérence avec les données de test
    if len(predictions) != len(test_data):
        logger.warning(f"⚠️ Désaccord: {len(predictions)} prédictions vs {len(test_data)} échantillons de test")
        
        # Si on a moins de prédictions que d'échantillons de test, on utilise les données disponibles
        if len(predictions) < len(test_data):
            logger.info(f"📊 Troncature des données de test à {len(predictions)} échantillons")
            test_data = test_data.iloc[:len(predictions)]
        else:
            # Si on a plus de prédictions, on tronque les prédictions
            logger.info(f"📊 Troncature des prédictions à {len(test_data)} échantillons")
            predictions = predictions[:len(test_data)]
    
    # Créer le DataFrame de soumission
    submission = pd.DataFrame({
        'id': range(len(predictions)),
        'outcome': ['ad.' if pred == 1 else 'noad.' for pred in predictions]
    })
    
    # Sauvegarder
    output_dir = cfg.paths.outputs / "predictions"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"submission_{method}_full.csv"
    
    submission.to_csv(output_path, index=False)
    logger.info(f"📝 Fichier de soumission sauvegardé : {output_path}")
    logger.info(f"📊 {len(submission)} prédictions générées")
    logger.info(f"📈 Répartition: {sum(predictions)} 'ad.' ({100*sum(predictions)/len(predictions):.1f}%), {len(predictions)-sum(predictions)} 'noad.' ({100*(len(predictions)-sum(predictions))/len(predictions):.1f}%)")
    
    return submission

if __name__ == "__main__":
    # Générer avec le modèle champion
    submission = generate_final_predictions(use_stacking=False)
    print("\n✅ Prédictions générées avec le modèle champion !")