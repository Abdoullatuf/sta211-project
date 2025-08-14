"""
Module de modélisation pour le Notebook 2 - Entraînement des modèles individuels
Extrait et adapté de modeling.py

Fonctionnalités:
- Entraînement des modèles individuels (RF, XGB, SVM, MLP, GradBoost)
- Optimisation des hyperparamètres avec validation croisée
- Sélection de features avec RFE et importance
- Évaluation et comparaison des modèles
- Optimisation des seuils de classification
- Sauvegarde des pipelines optimisés

Auteur: Maoulida Abdoullatuf
Version: 1.0 (restructuré)
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

# Imports du projet
from modules.config import cfg
from modules.utils.storage import save_artifact, load_artifact

# Scikit-learn imports
from sklearn.metrics import (
    precision_recall_curve, f1_score, precision_score, recall_score, 
    roc_auc_score, confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc,
    classification_report
)
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.feature_selection import RFECV
from sklearn.inspection import permutation_importance
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier

# Configuration du logging
log = cfg.get_logger(__name__)

# =============================================================================
# 1. CONFIGURATION DES MODÈLES ET HYPERPARAMÈTRES
# =============================================================================

def get_default_param_grids() -> Dict[str, Dict]:
    """
    Retourne les grilles d'hyperparamètres par défaut pour chaque modèle.
    
    Returns:
        Dictionnaire avec les paramètres par modèle
    """
    
    param_grids = {
        'randforest': {
            'clf__n_estimators': [100, 200],
            'clf__max_depth': [10, 20, None],
            'clf__min_samples_split': [2, 5],
            'clf__min_samples_leaf': [1, 2],
            'clf__max_features': ['sqrt', 'log2']
        },
        
        'xgboost': {
            'clf__n_estimators': [100, 200],
            'clf__max_depth': [6, 10],
            'clf__learning_rate': [0.01, 0.1],
            'clf__subsample': [0.8, 1.0],
            'clf__colsample_bytree': [0.8, 1.0]
        },
        
        'gradboost': {
            'clf__n_estimators': [100, 200],
            'clf__max_depth': [6, 10], 
            'clf__learning_rate': [0.01, 0.1],
            'clf__subsample': [0.8, 1.0],
            'clf__colsample_bytree': [0.8, 1.0]
        },
        
        'svm': {
            'clf__C': [0.1, 1, 10],
            'clf__kernel': ['rbf', 'linear'],
            'clf__gamma': ['scale', 'auto']
        },
        
        'mlp': {
            'clf__hidden_layer_sizes': [(100,), (100, 50), (200, 100)],
            'clf__alpha': [0.0001, 0.001],
            'clf__learning_rate': ['constant', 'adaptive'],
            'clf__max_iter': [500]
        }
    }
    
    return param_grids


def create_model_estimators() -> Dict[str, object]:
    """
    Crée les instances des estimateurs par défaut.
    
    Returns:
        Dictionnaire contenant les estimateurs
    """
    
    estimators = {
        'randforest': RandomForestClassifier(
            random_state=42,
            n_jobs=-1
        ),
        
        'xgboost': XGBClassifier(
            random_state=42,
            eval_metric='logloss',
            verbosity=0
        ),
        
        'gradboost': XGBClassifier(
            random_state=42,
            eval_metric='logloss', 
            verbosity=0
        ),
        
        'svm': SVC(
            random_state=42,
            probability=True  # Nécessaire pour predict_proba
        ),
        
        'mlp': MLPClassifier(
            random_state=42,
            early_stopping=True,
            validation_fraction=0.1
        )
    }
    
    return estimators


# =============================================================================
# 2. ENTRAÎNEMENT ET OPTIMISATION DES MODÈLES
# =============================================================================

def train_and_optimize_model(X_train: pd.DataFrame,
                            y_train: pd.Series,
                            X_val: pd.DataFrame, 
                            y_val: pd.Series,
                            model_name: str,
                            param_grid: Optional[Dict] = None,
                            cv_folds: int = 5,
                            scoring: str = 'f1',
                            n_jobs: int = -1,
                            verbose: bool = True) -> Dict:
    """
    Entraîne et optimise un modèle avec validation croisée.
    
    Args:
        X_train, y_train: Données d'entraînement
        X_val, y_val: Données de validation
        model_name: Nom du modèle
        param_grid: Grille d'hyperparamètres
        cv_folds: Nombre de folds pour la validation croisée
        scoring: Métrique d'optimisation
        n_jobs: Nombre de processus parallèles
        verbose: Affichage détaillé
        
    Returns:
        Dictionnaire contenant les résultats d'optimisation
    """
    
    if verbose:
        print(f"🚀 OPTIMISATION DU MODÈLE: {model_name.upper()}")
        print("=" * 50)
    
    # Récupérer l'estimateur et les paramètres
    estimators = create_model_estimators()
    default_grids = get_default_param_grids()
    
    if model_name not in estimators:
        raise ValueError(f"Modèle '{model_name}' non supporté")
    
    estimator = estimators[model_name]
    if param_grid is None:
        param_grid = default_grids.get(model_name, {})
    
    if verbose:
        print(f"📊 Données: {X_train.shape[0]} train, {X_val.shape[0]} validation")
        print(f"🔧 Hyperparamètres à tester: {len(param_grid)} groupes")
        print(f"🎯 Métrique d'optimisation: {scoring}")
    
    # Configuration de la validation croisée
    cv_strategy = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    
    # Création du pipeline (si nécessaire)
    from sklearn.pipeline import Pipeline
    pipeline = Pipeline([('clf', estimator)])
    
    # Grid Search
    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        cv=cv_strategy,
        scoring=scoring,
        n_jobs=n_jobs,
        verbose=1 if verbose else 0,
        return_train_score=True
    )
    
    # Entraînement
    start_time = time.time()
    grid_search.fit(X_train, y_train)
    training_time = time.time() - start_time
    
    # Meilleur modèle
    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_
    best_cv_score = grid_search.best_score_
    
    # Prédictions sur validation
    y_val_pred = best_model.predict(X_val)
    y_val_pred_proba = best_model.predict_proba(X_val)[:, 1]
    
    # Métriques de validation
    val_f1 = f1_score(y_val, y_val_pred)
    val_precision = precision_score(y_val, y_val_pred)
    val_recall = recall_score(y_val, y_val_pred)
    val_auc = roc_auc_score(y_val, y_val_pred_proba)
    
    if verbose:
        print(f"✅ Optimisation terminée en {training_time:.2f}s")
        print(f"🏆 Meilleur score CV ({scoring}): {best_cv_score:.4f}")
        print(f"📈 Métriques de validation:")
        print(f"   • F1-score: {val_f1:.4f}")
        print(f"   • Précision: {val_precision:.4f}")  
        print(f"   • Rappel: {val_recall:.4f}")
        print(f"   • AUC-ROC: {val_auc:.4f}")
    
    # Résultats complets
    results = {
        'model_name': model_name,
        'best_model': best_model,
        'best_params': best_params,
        'best_cv_score': best_cv_score,
        'cv_results': grid_search.cv_results_,
        'training_time': training_time,
        'validation_metrics': {
            'f1': val_f1,
            'precision': val_precision,
            'recall': val_recall,
            'auc_roc': val_auc
        },
        'predictions': {
            'y_val_pred': y_val_pred,
            'y_val_pred_proba': y_val_pred_proba
        }
    }
    
    return results


# =============================================================================
# 3. SÉLECTION DE FEATURES
# =============================================================================

def perform_feature_selection(X_train: pd.DataFrame,
                             y_train: pd.Series,
                             X_val: pd.DataFrame,
                             y_val: pd.Series, 
                             estimator,
                             method: str = 'rfecv',
                             cv_folds: int = 5,
                             scoring: str = 'f1',
                             save_selector: bool = True,
                             verbose: bool = True) -> Dict:
    """
    Effectue la sélection de features avec différentes méthodes.
    
    Args:
        X_train, y_train: Données d'entraînement
        X_val, y_val: Données de validation
        estimator: Estimateur pour la sélection
        method: 'rfecv', 'importance', ou 'permutation'
        cv_folds: Nombre de folds
        scoring: Métrique pour l'évaluation
        save_path: Chemin pour sauvegarder le sélecteur
        verbose: Affichage détaillé
        
    Returns:
        Dictionnaire avec les résultats de sélection
    """
    
    if verbose:
        print(f"🔍 SÉLECTION DE FEATURES - {method.upper()}")
        print("=" * 40)
    
    results = {
        'method': method,
        'original_features': X_train.shape[1],
        'selected_features': None,
        'feature_names': None,
        'scores': None,
        'selector': None
    }
    
    if method == 'rfecv':
        # Recursive Feature Elimination with CV
        cv_strategy = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
        
        selector = RFECV(
            estimator=estimator,
            cv=cv_strategy,
            scoring=scoring,
            n_jobs=-1,
            verbose=1 if verbose else 0
        )
        
        # Fit du sélecteur
        start_time = time.time()
        selector.fit(X_train, y_train)
        selection_time = time.time() - start_time
        
        # Features sélectionnées
        selected_mask = selector.support_
        selected_features = X_train.columns[selected_mask].tolist()
        n_selected = selector.n_features_
        
        results.update({
            'selected_features': n_selected,
            'feature_names': selected_features,
            'scores': selector.cv_results_['mean_test_score'],
            'selector': selector,
            'selection_time': selection_time
        })
        
        if verbose:
            print(f"✅ RFECV terminé en {selection_time:.2f}s")
            print(f"🎯 Features sélectionnées: {n_selected}/{X_train.shape[1]}")
            print(f"🏆 Meilleur score: {selector.cv_results_['mean_test_score'][n_selected-1]:.4f}")
    
    elif method == 'importance':
        # Feature importance native du modèle
        start_time = time.time()
        estimator.fit(X_train, y_train)
        selection_time = time.time() - start_time
        
        if hasattr(estimator, 'feature_importances_'):
            importances = estimator.feature_importances_
            
            # Trier par importance
            indices = np.argsort(importances)[::-1]
            feature_names = X_train.columns[indices].tolist()
            
            results.update({
                'scores': importances,
                'feature_names': feature_names,
                'selection_time': selection_time,
                'estimator': estimator
            })
            
            if verbose:
                print(f"✅ Importance calculée en {selection_time:.2f}s")
                print("🏆 Top 10 features les plus importantes:")
                for i in range(min(10, len(feature_names))):
                    idx = indices[i]
                    print(f"   {i+1}. {X_train.columns[idx]}: {importances[idx]:.4f}")
        else:
            print("❌ L'estimateur ne supporte pas feature_importances_")
            
    elif method == 'permutation':
        # Permutation importance
        start_time = time.time()
        estimator.fit(X_train, y_train)
        
        perm_importance = permutation_importance(
            estimator, X_val, y_val,
            scoring=scoring,
            n_repeats=5,
            random_state=42,
            n_jobs=-1
        )
        selection_time = time.time() - start_time
        
        # Trier par importance
        indices = np.argsort(perm_importance.importances_mean)[::-1]
        feature_names = X_train.columns[indices].tolist()
        
        results.update({
            'scores': perm_importance.importances_mean,
            'scores_std': perm_importance.importances_std,
            'feature_names': feature_names,
            'selection_time': selection_time,
            'estimator': estimator
        })
        
        if verbose:
            print(f"✅ Permutation importance calculée en {selection_time:.2f}s")
            print("🏆 Top 10 features les plus importantes:")
            for i in range(min(10, len(feature_names))):
                idx = indices[i]
                mean_imp = perm_importance.importances_mean[idx]
                std_imp = perm_importance.importances_std[idx]
                print(f"   {i+1}. {X_train.columns[idx]}: {mean_imp:.4f} ± {std_imp:.4f}")
    
    else:
        raise ValueError(f"Méthode '{method}' non supportée")
    
    # Sauvegarde du sélecteur
    if save_selector and results['selector'] is not None:
        filename = f"selector_{method}_{scoring}.pkl"
        save_artifact(results['selector'], filename, cfg.paths.selectors)
        if verbose:
            log.info(f"💾 Sélecteur sauvegardé : {filename}")
    
    return results


# =============================================================================
# 4. OPTIMISATION DES SEUILS DE CLASSIFICATION
# =============================================================================

def optimize_classification_threshold(y_true: np.ndarray,
                                    y_proba: np.ndarray,
                                    metric: str = 'f1',
                                    verbose: bool = True) -> Dict:
    """
    Optimise le seuil de classification pour maximiser une métrique donnée.
    
    Args:
        y_true: Labels vrais
        y_proba: Probabilités prédites
        metric: Métrique à optimiser ('f1', 'precision', 'recall')
        verbose: Affichage détaillé
        
    Returns:
        Dictionnaire avec le seuil optimal et les métriques
    """
    
    if verbose:
        print(f"🎯 OPTIMISATION DU SEUIL - {metric.upper()}")
        print("=" * 40)
    
    # Générer une gamme de seuils
    thresholds = np.arange(0.1, 1.0, 0.01)
    
    scores = []


# =============================================================================
# 5. SÉLECTION DE VARIABLES AVEC RFECV
# =============================================================================

def perform_rfecv_selection(data_splits: Dict, 
                          imputation_method: str, 
                          save_dir_base: Path) -> Dict:
    """
    Effectue la sélection RFECV pour un type d'imputation donné.
    
    Args:
        data_splits: Dict contenant X_train, y_train
        imputation_method: str ("knn" ou "mice") 
        save_dir_base: Path - répertoire de base pour sauvegarde
    
    Returns:
        dict: {"selector": selector_fitted, "selected_features": list, "scores": array}
    """
    # Données
    X = data_splits["X_train"]
    y = data_splits["y_train"]
    
    print(f"🔍 RFECV pour {imputation_method.upper()}...")
    
    # Estimateur RFECV
    estimator = GradientBoostingClassifier(
        n_estimators=100,
        learning_rate=0.05,
        max_depth=4,
        min_samples_split=10,
        min_samples_leaf=5,
        subsample=0.8,
        random_state=42
    )
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    selector = RFECV(estimator=estimator, step=1, cv=cv, scoring='f1', n_jobs=-1, verbose=0)
    selector.fit(X, y)
    
    # Résultats
    mask = selector.support_
    selected_features = X.columns[mask].tolist()
    
    # Sauvegarde
    rfecv_dir = save_dir_base / "notebook2" / imputation_method
    rfecv_dir.mkdir(parents=True, exist_ok=True)
    
    save_artifact(selector, f"rfecv_selector_{imputation_method}.pkl", rfecv_dir)
    save_artifact(selected_features, f"selected_features_{imputation_method}.pkl", rfecv_dir)
    
    print(f"✅ {imputation_method.upper()}: {len(selected_features)} variables sélectionnées")
    print("🔝 Top 10 :", selected_features[:10])
    
    return {
        "selector": selector,
        "selected_features": selected_features,
        "scores": selector.cv_results_["mean_test_score"]
    }


def plot_rfecv_results(results_dict: Dict, imputation_method: str) -> None:
    """
    Trace les résultats RFECV.
    
    Args:
        results_dict: Résultats de perform_rfecv_selection
        imputation_method: Nom de la méthode d'imputation
    """
    selector = results_dict["selector"]
    scores = results_dict["scores"]
    
    optimal_num_features = selector.n_features_
    optimal_score = max(scores)
    
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(scores) + 1), scores, marker='o', label='Score F1')
    plt.axvline(optimal_num_features, color='red', linestyle='--', 
                label=f'Optimal: {optimal_num_features} variables')
    plt.scatter(optimal_num_features, optimal_score, color='red')
    plt.text(optimal_num_features + 0.5, optimal_score, f'{optimal_score:.3f}', color='blue')
    plt.xlabel("Nombre de variables sélectionnées")
    plt.ylabel("Score F1 moyen (validation croisée)")
    plt.title(f"Évolution du score F1 - {imputation_method.upper()}")
    plt.legend()
    plt.grid(True, linestyle="--", linewidth=0.5)
    plt.tight_layout()
    plt.show()


def save_reduced_datasets(splits_dict: Dict, 
                         rfecv_results_dict: Dict, 
                         save_dir_base: Path,
                         imputation_methods: List[str] = ["knn", "mice"]) -> None:
    """
    Sauvegarde les jeux de données réduits avec les variables sélectionnées par RFECV.
    
    Args:
        splits_dict: Dict contenant les splits pour chaque méthode
        rfecv_results_dict: Dict contenant les résultats RFECV 
        save_dir_base: Répertoire de base pour la sauvegarde
        imputation_methods: Liste des méthodes d'imputation à traiter
    """
    for method in imputation_methods:
        print(f"💾 Sauvegarde des datasets réduits - {method.upper()}")
        
        # Récupérer les features sélectionnées
        selected_features = rfecv_results_dict[method]["selected_features"]
        
        # Dossier de sauvegarde
        reduced_dir = save_dir_base / "notebook2" / method / "reduced"
        reduced_dir.mkdir(parents=True, exist_ok=True)
        
        # Splits pour cette méthode
        splits_method = splits_dict[method]
        
        # Sauvegarde des jeux de données réduits
        for subset_name in ["train", "val", "test"]:
            X_subset = splits_method[f"X_{subset_name}"][selected_features]
            y_subset = splits_method[f"y_{subset_name}"]
            
            save_artifact({"X": X_subset, "y": y_subset}, 
                         f"{method}_{subset_name}_reduced.pkl", 
                         reduced_dir)
            print(f"  ✅ {subset_name.capitalize()} réduit sauvegardé")
        
        # Sauvegarde des noms de variables sélectionnées
        save_artifact(selected_features, f"selected_columns_{method}.pkl", reduced_dir)
        print(f"  ✅ Variables sélectionnées ({len(selected_features)}) sauvegardées")
        print("-" * 50)


# =============================================================================
# 5. ÉVALUATION ET COMPARAISON DES MODÈLES
# =============================================================================

def evaluate_model_performance(y_true: np.ndarray,
                              y_pred: np.ndarray,
                              y_proba: np.ndarray,
                              model_name: str,
                              save_path: Optional[Path] = None,
                              verbose: bool = True) -> Dict:
    """
    Évalue les performances d'un modèle avec métriques complètes.
    
    Args:
        y_true: Labels vrais
        y_pred: Prédictions
        y_proba: Probabilités prédites
        model_name: Nom du modèle
        save_path: Chemin pour sauvegarder les graphiques
        verbose: Affichage détaillé
        
    Returns:
        Dictionnaire avec toutes les métriques
    """
    
    if verbose:
        print(f"📊 ÉVALUATION: {model_name.upper()}")
        print("=" * 40)
    
    # Métriques de base
    metrics = {
        'model_name': model_name,
        'f1': f1_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred),
        'recall': recall_score(y_true, y_pred),
        'auc_roc': roc_auc_score(y_true, y_proba)
    }
    
    # Matrice de confusion
    cm = confusion_matrix(y_true, y_pred)
    metrics['confusion_matrix'] = cm
    
    # Rapport de classification
    class_report = classification_report(y_true, y_pred, output_dict=True)
    metrics['classification_report'] = class_report
    
    if verbose:
        print("📈 Métriques principales:")
        print(f"   • F1-score: {metrics['f1']:.4f}")
        print(f"   • Précision: {metrics['precision']:.4f}")
        print(f"   • Rappel: {metrics['recall']:.4f}")
        print(f"   • AUC-ROC: {metrics['auc_roc']:.4f}")
        
        print("\n🎯 Matrice de confusion:")
        print(cm)
    
    # Visualisations
    if save_path:
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        # 1. Matrice de confusion
        ConfusionMatrixDisplay(cm, display_labels=['Non-Ad', 'Ad']).plot(ax=axes[0])
        axes[0].set_title(f'Matrice de Confusion - {model_name}')
        
        # 2. Courbe ROC
        fpr, tpr, _ = roc_curve(y_true, y_proba)
        auc_score = auc(fpr, tpr)
        axes[1].plot(fpr, tpr, label=f'ROC (AUC = {auc_score:.3f})')
        axes[1].plot([0, 1], [0, 1], 'k--', label='Aléatoire')
        axes[1].set_xlabel('Taux de Faux Positifs')
        axes[1].set_ylabel('Taux de Vrais Positifs')
        axes[1].set_title(f'Courbe ROC - {model_name}')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        # 3. Courbe Précision-Rappel
        precision_vals, recall_vals, _ = precision_recall_curve(y_true, y_proba)
        axes[2].plot(recall_vals, precision_vals, label=f'PR (F1 = {metrics["f1"]:.3f})')
        axes[2].set_xlabel('Rappel')
        axes[2].set_ylabel('Précision')
        axes[2].set_title(f'Courbe Précision-Rappel - {model_name}')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Sauvegarder
        plot_path = save_path / f'{model_name}_evaluation.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        if verbose:
            print(f"💾 Graphiques sauvegardés : {plot_path}")
    
    return metrics


# =============================================================================
# 6. PIPELINE COMPLET D'ENTRAÎNEMENT
# =============================================================================

def train_all_models(X_train: pd.DataFrame,
                     y_train: pd.Series,
                     X_val: pd.DataFrame,
                     y_val: pd.Series,
                     models_to_train: Optional[List[str]] = None,
                     param_grids: Optional[Dict] = None,
                     save_dir: Optional[Path] = None,
                     cv_folds: int = 5,
                     scoring: str = 'f1',
                     n_jobs: int = -1,
                     verbose: bool = True) -> Dict:
    """
    Entraîne tous les modèles spécifiés avec optimisation complète.
    
    Args:
        X_train, y_train: Données d'entraînement
        X_val, y_val: Données de validation  
        models_to_train: Liste des modèles à entraîner
        param_grids: Grilles d'hyperparamètres personnalisées
        save_dir: Répertoire pour sauvegarder tous les artefacts
        cv_folds: Nombre de folds pour la CV
        scoring: Métrique d'optimisation
        n_jobs: Processus parallèles
        verbose: Affichage détaillé
        
    Returns:
        Dictionnaire avec tous les résultats d'entraînement
    """
    
    if models_to_train is None:
        models_to_train = ['randforest', 'xgboost', 'gradboost', 'svm', 'mlp']
    
    if param_grids is None:
        param_grids = get_default_param_grids()
    
    if verbose:
        print("🚀 ENTRAÎNEMENT DE TOUS LES MODÈLES")
        print("=" * 60)
        print(f"📊 Modèles à entraîner: {', '.join(models_to_train)}")
        print(f"💾 Sauvegarde: {'Oui' if save_dir else 'Non'}")
    
    results = {
        'training_summary': {},
        'best_models': {},
        'all_metrics': {},
        'comparison_data': []
    }
    
    if save_dir:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
    
    for model_name in models_to_train:
        if verbose:
            print(f"\n{'='*20} {model_name.upper()} {'='*20}")
        
        try:
            # 1. Entraînement et optimisation
            training_results = train_and_optimize_model(
                X_train, y_train, X_val, y_val,
                model_name=model_name,
                param_grid=param_grids.get(model_name),
                cv_folds=cv_folds,
                scoring=scoring,
                n_jobs=n_jobs,
                verbose=verbose
            )
            
            # 2. Optimisation du seuil
            threshold_results = optimize_classification_threshold(
                y_val, training_results['predictions']['y_val_pred_proba'],
                metric='f1', verbose=verbose
            )
            
            # 3. Évaluation complète
            evaluation_results = evaluate_model_performance(
                y_val, 
                threshold_results['y_pred_optimal'],
                training_results['predictions']['y_val_pred_proba'],
                model_name=model_name,
                save_path=save_dir / 'evaluation_plots' if save_dir else None,
                verbose=verbose
            )
            
            # Stocker les résultats
            results['training_summary'][model_name] = training_results
            results['best_models'][model_name] = training_results['best_model']
            results['all_metrics'][model_name] = {
                **training_results['validation_metrics'],
                **threshold_results['optimal_metrics'],
                'best_threshold': threshold_results['best_threshold']
            }
            
            # Données pour comparaison
            results['comparison_data'].append({
                'Model': model_name,
                'F1_Score': evaluation_results['f1'],
                'Precision': evaluation_results['precision'],
                'Recall': evaluation_results['recall'],
                'AUC_ROC': evaluation_results['auc_roc'],
                'Best_Threshold': threshold_results['best_threshold'],
                'CV_Score': training_results['best_cv_score'],
                'Training_Time': training_results['training_time']
            })
            
            # Sauvegarde des artefacts
            if save_dir:
                # Pipeline complet
                pipeline_filename = f'pipeline_{model_name}.pkl'
                save_artifact(training_results['best_model'], pipeline_filename, save_dir)
                
                # Paramètres
                params_filename = f'best_params_{model_name}.json'
                save_artifact(training_results['best_params'], params_filename, save_dir)
                
                # Seuil optimal
                threshold_data = {
                    'best_threshold': threshold_results['best_threshold'],
                    'optimized_metric': 'f1',
                    'optimal_metrics': threshold_results['optimal_metrics']
                }
                threshold_filename = f'threshold_{model_name}.json'
                save_artifact(threshold_data, threshold_filename, save_dir)
                
                if verbose:
                    print(f"💾 Artefacts sauvegardés pour {model_name}")
            
        except Exception as e:
            print(f"❌ Erreur lors de l'entraînement de {model_name}: {str(e)}")
            continue
    
    # Tableau de comparaison
    if results['comparison_data']:
        comparison_df = pd.DataFrame(results['comparison_data'])
        comparison_df = comparison_df.sort_values('F1_Score', ascending=False)
        results['comparison_dataframe'] = comparison_df
        
        if verbose:
            print("\n🏆 TABLEAU DE COMPARAISON FINAL")
            print("=" * 60)
            print(comparison_df.round(4).to_string(index=False))
        
        # Meilleur modèle
        best_model_name = comparison_df.iloc[0]['Model']
        results['champion_model'] = {
            'name': best_model_name,
            'model': results['best_models'][best_model_name],
            'metrics': results['all_metrics'][best_model_name]
        }
        
        if verbose:
            print(f"\n🥇 MODÈLE CHAMPION: {best_model_name.upper()}")
            print(f"   F1-Score: {comparison_df.iloc[0]['F1_Score']:.4f}")
        
        # Sauvegarder le tableau de comparaison
        if save_dir:
            comparison_path = save_dir / 'models_comparison.csv'
            comparison_df.to_csv(comparison_path, index=False)
            
            # Sauvegarder le modèle champion séparément
            champion_filename = f'champion_{best_model_name}.pkl'
            save_artifact(results['champion_model']['model'], champion_filename, save_dir)
            
            if verbose:
                print(f"💾 Comparaison sauvegardée : {comparison_path}")
                print(f"💾 Modèle champion sauvegardé : {champion_filename}")
    
    if verbose:
        print(f"\n🎉 ENTRAÎNEMENT TERMINÉ - {len(results['best_models'])} modèles entraînés")
    
    return results


# Utilitaires supplémentaires
import time

def clean_xgb_params(params: Dict) -> Dict:
    """Nettoie les paramètres XGBoost des clés incompatibles."""
    params_clean = params.copy()
    params_to_remove = ['use_label_encoder', 'eval_metric', 'feature_weights']
    
    for param in params_to_remove:
        if param in params_clean:
            del params_clean[param]
    
    return params_clean


def clean_svm_params(params: Dict) -> Dict:
    """Nettoie les paramètres SVM pour s'assurer de la compatibilité."""
    params_clean = params.copy()
    
    # S'assurer que probability=True pour predict_proba
    params_clean['probability'] = True
    return params_clean