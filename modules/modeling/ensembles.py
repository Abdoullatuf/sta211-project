"""
Module de modélisation d'ensemble pour le Notebook 3.

Fonctionnalités:
- Entraînement d'ensembles (Voting, Bagging, Stacking)
- Optimisation des méta-modèles
- Évaluation comparée des ensembles vs modèles individuels
- Sauvegarde et rechargement des pipelines d'ensemble

Auteur: Maoulida Abdoullatuf
Version: 1.0
"""

import time
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any
import logging

# Scikit-learn imports
from sklearn.ensemble import (
    VotingClassifier, BaggingClassifier, RandomForestClassifier,
    ExtraTreesClassifier, AdaBoostClassifier, GradientBoostingClassifier
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    f1_score, precision_score, recall_score, roc_auc_score,
    classification_report, confusion_matrix
)

# Imports du projet
from modules.config import cfg
from modules.utils.storage import save_artifact, load_artifact

# Configuration du logging
log = cfg.get_logger(__name__)


# =============================================================================
# 1. CRÉATION D'ENSEMBLES
# =============================================================================

def create_voting_ensemble(base_models: Dict[str, Any],
                          voting: str = 'soft',
                          weights: Optional[List[float]] = None) -> VotingClassifier:
    """
    Crée un ensemble de type Voting.
    
    Args:
        base_models: Dict {nom: modèle} des modèles de base
        voting: Type de vote ('hard' ou 'soft')
        weights: Poids optionnels pour chaque modèle
        
    Returns:
        VotingClassifier configuré
    """
    
    log.info(f"🗳️ Création d'un ensemble Voting ({voting})")
    log.info(f"📊 Modèles de base: {list(base_models.keys())}")
    
    # Conversion en liste de tuples pour VotingClassifier
    estimators = [(name, model) for name, model in base_models.items()]
    
    voting_ensemble = VotingClassifier(
        estimators=estimators,
        voting=voting,
        weights=weights
    )
    
    return voting_ensemble


def create_bagging_ensemble(base_estimator,
                           n_estimators: int = 10,
                           max_samples: float = 1.0,
                           max_features: float = 1.0,
                           random_state: int = None) -> BaggingClassifier:
    """
    Crée un ensemble de type Bagging.
    
    Args:
        base_estimator: Estimateur de base
        n_estimators: Nombre d'estimateurs dans l'ensemble
        max_samples: Fraction d'échantillons pour chaque estimateur
        max_features: Fraction de features pour chaque estimateur
        random_state: Graine aléatoire
        
    Returns:
        BaggingClassifier configuré
    """
    
    if random_state is None:
        random_state = cfg.project.random_state
    
    log.info(f"🎒 Création d'un ensemble Bagging")
    log.info(f"📊 Estimateur de base: {type(base_estimator).__name__}")
    log.info(f"🔢 Nombre d'estimateurs: {n_estimators}")
    
    bagging_ensemble = BaggingClassifier(
        base_estimator=base_estimator,
        n_estimators=n_estimators,
        max_samples=max_samples,
        max_features=max_features,
        random_state=random_state,
        n_jobs=-1
    )
    
    return bagging_ensemble


def create_stacking_ensemble(base_models: Dict[str, Any],
                            meta_classifier=None,
                            cv: int = 5,
                            use_probas: bool = True,
                            use_features_in_secondary: bool = False) -> Any:
    """
    Crée un ensemble de type Stacking.
    
    Args:
        base_models: Dict {nom: modèle} des modèles de base
        meta_classifier: Méta-classificateur (LogisticRegression par défaut)
        cv: Nombre de folds pour la validation croisée
        use_probas: Utiliser les probabilités comme features
        use_features_in_secondary: Inclure les features originales
        
    Returns:
        StackingClassifier configuré
    """
    
    from sklearn.ensemble import StackingClassifier
    
    if meta_classifier is None:
        meta_classifier = LogisticRegression(
            random_state=cfg.project.random_state,
            max_iter=1000
        )
    
    log.info(f"🥞 Création d'un ensemble Stacking")
    log.info(f"📊 Modèles de base: {list(base_models.keys())}")
    log.info(f"🧠 Méta-classificateur: {type(meta_classifier).__name__}")
    
    # Conversion en liste de tuples
    estimators = [(name, model) for name, model in base_models.items()]
    
    # Configuration de la CV
    cv_strategy = StratifiedKFold(
        n_splits=cv, 
        shuffle=True, 
        random_state=cfg.project.random_state
    )
    
    stacking_ensemble = StackingClassifier(
        estimators=estimators,
        final_estimator=meta_classifier,
        cv=cv_strategy,
        stack_method='predict_proba' if use_probas else 'predict',
        passthrough=use_features_in_secondary,
        n_jobs=-1
    )
    
    return stacking_ensemble


# =============================================================================
# 2. OPTIMISATION DES ENSEMBLES
# =============================================================================

def optimize_ensemble(ensemble,
                     X_train: pd.DataFrame,
                     y_train: pd.Series,
                     X_val: pd.DataFrame,
                     y_val: pd.Series,
                     param_grid: Optional[Dict] = None,
                     cv: int = 5,
                     scoring: str = 'f1',
                     verbose: bool = True) -> Dict:
    """
    Optimise les hyperparamètres d'un ensemble.
    
    Args:
        ensemble: Ensemble à optimiser
        X_train, y_train: Données d'entraînement
        X_val, y_val: Données de validation
        param_grid: Grille de paramètres à tester
        cv: Nombre de folds pour la validation croisée
        scoring: Métrique d'optimisation
        verbose: Affichage détaillé
        
    Returns:
        Dictionnaire avec les résultats d'optimisation
    """
    
    ensemble_name = type(ensemble).__name__
    
    if verbose:
        log.info(f"🔧 OPTIMISATION ENSEMBLE: {ensemble_name}")
        log.info(f"📊 Données: {X_train.shape[0]} train, {X_val.shape[0]} validation")
    
    # Grille par défaut si non fournie
    if param_grid is None:
        param_grid = _get_default_ensemble_params(ensemble)
    
    if verbose and param_grid:
        log.info(f"🎯 Paramètres à optimiser: {list(param_grid.keys())}")
    
    # Configuration de la validation croisée
    cv_strategy = StratifiedKFold(
        n_splits=cv,
        shuffle=True, 
        random_state=cfg.project.random_state
    )
    
    # Grid Search
    start_time = time.time()
    if param_grid:
        grid_search = GridSearchCV(
            estimator=ensemble,
            param_grid=param_grid,
            cv=cv_strategy,
            scoring=scoring,
            n_jobs=-1,
            verbose=1 if verbose else 0
        )
        grid_search.fit(X_train, y_train)
        best_ensemble = grid_search.best_estimator_
        best_params = grid_search.best_params_
        best_cv_score = grid_search.best_score_
    else:
        # Pas d'optimisation, juste entraînement
        ensemble.fit(X_train, y_train)
        best_ensemble = ensemble
        best_params = {}
        best_cv_score = None
    
    training_time = time.time() - start_time
    
    # Évaluation sur validation
    y_val_pred = best_ensemble.predict(X_val)
    y_val_pred_proba = best_ensemble.predict_proba(X_val)[:, 1]
    
    # Métriques
    val_metrics = {
        'f1': f1_score(y_val, y_val_pred),
        'precision': precision_score(y_val, y_val_pred),
        'recall': recall_score(y_val, y_val_pred),
        'auc_roc': roc_auc_score(y_val, y_val_pred_proba)
    }
    
    if verbose:
        log.info(f"✅ Optimisation terminée en {training_time:.2f}s")
        if best_cv_score:
            log.info(f"🏆 Meilleur score CV: {best_cv_score:.4f}")
        log.info("📈 Métriques de validation:")
        for metric, value in val_metrics.items():
            log.info(f"   • {metric}: {value:.4f}")
    
    results = {
        'ensemble_name': ensemble_name,
        'best_ensemble': best_ensemble,
        'best_params': best_params,
        'best_cv_score': best_cv_score,
        'training_time': training_time,
        'validation_metrics': val_metrics,
        'predictions': {
            'y_val_pred': y_val_pred,
            'y_val_pred_proba': y_val_pred_proba
        }
    }
    
    return results


def _get_default_ensemble_params(ensemble) -> Dict:
    """Retourne les paramètres par défaut selon le type d'ensemble."""
    
    ensemble_name = type(ensemble).__name__
    
    if ensemble_name == 'VotingClassifier':
        return {
            'weights': [None, [1, 1, 1], [2, 1, 1], [1, 2, 1], [1, 1, 2]]
        }
    
    elif ensemble_name == 'BaggingClassifier':
        return {
            'n_estimators': [10, 20, 50],
            'max_samples': [0.8, 1.0],
            'max_features': [0.8, 1.0]
        }
    
    elif ensemble_name == 'StackingClassifier':
        return {
            'final_estimator__C': [0.1, 1, 10],
            'final_estimator__max_iter': [1000]
        }
    
    return {}


# =============================================================================
# 3. ENTRAÎNEMENT COMPLET D'ENSEMBLES
# =============================================================================

def train_all_ensembles(base_models: Dict[str, Any],
                       X_train: pd.DataFrame,
                       y_train: pd.Series,
                       X_val: pd.DataFrame,
                       y_val: pd.Series,
                       ensemble_types: Optional[List[str]] = None,
                       save_ensembles: bool = True,
                       verbose: bool = True) -> Dict:
    """
    Entraîne tous les types d'ensembles spécifiés.
    
    Args:
        base_models: Dict des modèles de base à utiliser
        X_train, y_train: Données d'entraînement
        X_val, y_val: Données de validation
        ensemble_types: Types d'ensembles à créer
        save_ensembles: Sauvegarder les ensembles
        verbose: Affichage détaillé
        
    Returns:
        Dictionnaire avec tous les résultats d'ensembles
    """
    
    if ensemble_types is None:
        ensemble_types = ['voting_soft', 'voting_hard', 'bagging', 'stacking']
    
    if verbose:
        log.info("🚀 ENTRAÎNEMENT DE TOUS LES ENSEMBLES")
        log.info("=" * 50)
        log.info(f"📊 Types d'ensembles: {ensemble_types}")
        log.info(f"🧠 Modèles de base: {list(base_models.keys())}")
    
    results = {
        'ensemble_results': {},
        'comparison_data': [],
        'best_ensembles': {}
    }
    
    for ensemble_type in ensemble_types:
        if verbose:
            log.info(f"\n{'='*20} {ensemble_type.upper()} {'='*20}")
        
        try:
            # Création de l'ensemble
            if ensemble_type == 'voting_soft':
                ensemble = create_voting_ensemble(base_models, voting='soft')
            elif ensemble_type == 'voting_hard':
                ensemble = create_voting_ensemble(base_models, voting='hard')
            elif ensemble_type == 'bagging':
                # Utilise le premier modèle comme base pour le bagging
                base_model = list(base_models.values())[0]
                ensemble = create_bagging_ensemble(base_model)
            elif ensemble_type == 'stacking':
                ensemble = create_stacking_ensemble(base_models)
            else:
                log.warning(f"Type d'ensemble '{ensemble_type}' non supporté")
                continue
            
            # Optimisation
            ensemble_results = optimize_ensemble(
                ensemble, X_train, y_train, X_val, y_val,
                verbose=verbose
            )
            
            # Stockage des résultats
            results['ensemble_results'][ensemble_type] = ensemble_results
            results['best_ensembles'][ensemble_type] = ensemble_results['best_ensemble']
            
            # Données pour comparaison
            results['comparison_data'].append({
                'Ensemble': ensemble_type,
                'F1_Score': ensemble_results['validation_metrics']['f1'],
                'Precision': ensemble_results['validation_metrics']['precision'],
                'Recall': ensemble_results['validation_metrics']['recall'],
                'AUC_ROC': ensemble_results['validation_metrics']['auc_roc'],
                'CV_Score': ensemble_results.get('best_cv_score', 'N/A'),
                'Training_Time': ensemble_results['training_time']
            })
            
            # Sauvegarde
            if save_ensembles:
                filename = f"ensemble_{ensemble_type}.pkl"
                save_artifact(
                    ensemble_results['best_ensemble'], 
                    filename, 
                    cfg.paths.models
                )
                log.info(f"💾 Ensemble sauvegardé: {filename}")
            
        except Exception as e:
            log.error(f"❌ Erreur avec {ensemble_type}: {str(e)}")
            continue
    
    # Tableau de comparaison
    if results['comparison_data']:
        comparison_df = pd.DataFrame(results['comparison_data'])
        comparison_df = comparison_df.sort_values('F1_Score', ascending=False)
        results['comparison_dataframe'] = comparison_df
        
        if verbose:
            log.info("\n🏆 TABLEAU DE COMPARAISON DES ENSEMBLES")
            log.info("=" * 50)
            print(comparison_df.round(4).to_string(index=False))
        
        # Meilleur ensemble
        best_ensemble_type = comparison_df.iloc[0]['Ensemble']
        results['champion_ensemble'] = {
            'type': best_ensemble_type,
            'model': results['best_ensembles'][best_ensemble_type],
            'metrics': results['ensemble_results'][best_ensemble_type]['validation_metrics']
        }
        
        if verbose:
            log.info(f"\n🥇 MEILLEUR ENSEMBLE: {best_ensemble_type.upper()}")
            log.info(f"   F1-Score: {comparison_df.iloc[0]['F1_Score']:.4f}")
    
    if verbose:
        log.info(f"\n🎉 ENTRAÎNEMENT TERMINÉ - {len(results['best_ensembles'])} ensembles")
    
    return results


# =============================================================================
# 4. ÉVALUATION ET COMPARAISON
# =============================================================================

def compare_models_vs_ensembles(individual_results: Dict,
                               ensemble_results: Dict,
                               save_comparison: bool = True) -> pd.DataFrame:
    """
    Compare les performances des modèles individuels vs ensembles.
    
    Args:
        individual_results: Résultats des modèles individuels
        ensemble_results: Résultats des ensembles
        save_comparison: Sauvegarder le tableau de comparaison
        
    Returns:
        DataFrame de comparaison
    """
    
    log.info("📊 COMPARAISON MODÈLES INDIVIDUELS vs ENSEMBLES")
    log.info("=" * 60)
    
    comparison_data = []
    
    # Modèles individuels
    if 'comparison_data' in individual_results:
        for row in individual_results['comparison_data']:
            row_copy = row.copy()
            row_copy['Type'] = 'Individual'
            comparison_data.append(row_copy)
    
    # Ensembles
    if 'comparison_data' in ensemble_results:
        for row in ensemble_results['comparison_data']:
            row_copy = row.copy()
            row_copy['Type'] = 'Ensemble'
            # Renommer la colonne si nécessaire
            if 'Ensemble' in row_copy:
                row_copy['Model'] = row_copy['Ensemble']
                del row_copy['Ensemble']
            comparison_data.append(row_copy)
    
    # Création du DataFrame
    comparison_df = pd.DataFrame(comparison_data)
    
    if not comparison_df.empty:
        # Tri par F1-Score
        comparison_df = comparison_df.sort_values('F1_Score', ascending=False)
        comparison_df = comparison_df.reset_index(drop=True)
        
        # Affichage
        log.info("🏆 CLASSEMENT GLOBAL:")
        print(comparison_df[['Type', 'Model', 'F1_Score', 'Precision', 'Recall', 'AUC_ROC']].round(4).to_string(index=False))
        
        # Sauvegarde
        if save_comparison:
            comparison_path = cfg.paths.outputs / "models_vs_ensembles_comparison.csv"
            comparison_df.to_csv(comparison_path, index=False)
            log.info(f"💾 Comparaison sauvegardée: {comparison_path}")
    
    return comparison_df


# =============================================================================
# 5. UTILITAIRES
# =============================================================================

def load_ensemble(ensemble_type: str) -> Any:
    """
    Charge un ensemble sauvegardé.
    
    Args:
        ensemble_type: Type d'ensemble à charger
        
    Returns:
        L'ensemble chargé
    """
    
    filename = f"ensemble_{ensemble_type}.pkl"
    try:
        ensemble = load_artifact(filename, cfg.paths.models)
        log.info(f"✅ Ensemble chargé: {filename}")
        return ensemble
    except FileNotFoundError:
        log.error(f"❌ Ensemble non trouvé: {filename}")
        raise


def get_ensemble_feature_importance(ensemble, 
                                  feature_names: List[str],
                                  top_n: int = 20) -> pd.DataFrame:
    """
    Extrait l'importance des features d'un ensemble (si disponible).
    
    Args:
        ensemble: Modèle d'ensemble
        feature_names: Noms des features
        top_n: Nombre de top features à retourner
        
    Returns:
        DataFrame avec l'importance des features
    """
    
    ensemble_name = type(ensemble).__name__
    
    if hasattr(ensemble, 'feature_importances_'):
        # Ensembles basés sur des arbres
        importances = ensemble.feature_importances_
    elif ensemble_name == 'VotingClassifier':
        # Moyenne des importances des estimateurs (si disponible)
        importances = []
        for name, estimator in ensemble.named_estimators_.items():
            if hasattr(estimator, 'feature_importances_'):
                importances.append(estimator.feature_importances_)
        
        if importances:
            importances = np.mean(importances, axis=0)
        else:
            log.warning("Aucune importance de features disponible")
            return pd.DataFrame()
    else:
        log.warning(f"Importance des features non disponible pour {ensemble_name}")
        return pd.DataFrame()
    
    # Création du DataFrame
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    }).sort_values('Importance', ascending=False)
    
    return importance_df.head(top_n)