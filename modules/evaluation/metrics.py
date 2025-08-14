"""
Module d'évaluation et de métriques pour le projet STA211.

Fonctionnalités:
- Évaluation complète des modèles avec métriques détaillées
- Optimisation des seuils de classification
- Visualisations (courbes ROC, Precision-Recall, matrice de confusion)
- Comparaison de modèles
- Export de rapports d'évaluation

Auteur: Maoulida Abdoullatuf
Version: 1.0
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any
import logging

# Scikit-learn imports
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score,
    confusion_matrix, classification_report,
    roc_curve, precision_recall_curve, auc,
    ConfusionMatrixDisplay
)

# Imports du projet
from modules.config import cfg
from modules.utils.storage import save_artifact

# Configuration du logging
log = cfg.get_logger(__name__)


# =============================================================================
# 1. MÉTRIQUES DE BASE
# =============================================================================

def calculate_basic_metrics(y_true: np.ndarray, 
                           y_pred: np.ndarray,
                           y_proba: np.ndarray = None) -> Dict[str, float]:
    """
    Calcule les métriques de base pour la classification.
    
    Args:
        y_true: Labels vrais
        y_pred: Prédictions
        y_proba: Probabilités prédites (optionnel)
        
    Returns:
        Dictionnaire des métriques
    """
    
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1': f1_score(y_true, y_pred, zero_division=0),
        'specificity': None,  # Calculé ci-dessous
        'npv': None  # Negative Predictive Value
    }
    
    # Matrice de confusion pour spécificité et NPV
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    metrics['npv'] = tn / (tn + fn) if (tn + fn) > 0 else 0.0
    
    # Métriques basées sur les probabilités
    if y_proba is not None:
        metrics['auc_roc'] = roc_auc_score(y_true, y_proba)
        metrics['auc_pr'] = average_precision_score(y_true, y_proba)
    
    return metrics


def calculate_detailed_metrics(y_true: np.ndarray, 
                              y_pred: np.ndarray,
                              y_proba: np.ndarray = None,
                              target_names: List[str] = None) -> Dict:
    """
    Calcule des métriques détaillées incluant le rapport de classification.
    
    Args:
        y_true: Labels vrais
        y_pred: Prédictions
        y_proba: Probabilités prédites (optionnel)
        target_names: Noms des classes
        
    Returns:
        Dictionnaire avec métriques détaillées
    """
    
    if target_names is None:
        target_names = ['Non-Ad', 'Ad']
    
    # Métriques de base
    basic_metrics = calculate_basic_metrics(y_true, y_pred, y_proba)
    
    # Rapport de classification
    class_report = classification_report(
        y_true, y_pred, 
        target_names=target_names, 
        output_dict=True,
        zero_division=0
    )
    
    # Matrice de confusion
    cm = confusion_matrix(y_true, y_pred)
    
    detailed_metrics = {
        'basic_metrics': basic_metrics,
        'classification_report': class_report,
        'confusion_matrix': cm,
        'confusion_matrix_normalized': cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    }
    
    return detailed_metrics


# =============================================================================
# 2. OPTIMISATION DES SEUILS
# =============================================================================

def optimize_threshold(y_true: np.ndarray,
                      y_proba: np.ndarray,
                      metric: str = 'f1',
                      thresholds: np.ndarray = None) -> Dict:
    """
    Optimise le seuil de classification pour maximiser une métrique.
    
    Args:
        y_true: Labels vrais
        y_proba: Probabilités prédites
        metric: Métrique à optimiser ('f1', 'precision', 'recall', 'youden')
        thresholds: Seuils à tester (auto-générés si None)
        
    Returns:
        Dictionnaire avec le seuil optimal et les métriques
    """
    
    if thresholds is None:
        thresholds = np.arange(0.1, 1.0, 0.01)
    
    log.info(f"🎯 Optimisation du seuil pour {metric}")
    
    scores = []
    best_threshold = 0.5
    best_score = 0.0
    
    for threshold in thresholds:
        y_pred = (y_proba >= threshold).astype(int)
        
        if metric == 'f1':
            score = f1_score(y_true, y_pred, zero_division=0)
        elif metric == 'precision':
            score = precision_score(y_true, y_pred, zero_division=0)
        elif metric == 'recall':
            score = recall_score(y_true, y_pred, zero_division=0)
        elif metric == 'youden':
            # Youden's J statistic = Sensitivity + Specificity - 1
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
            sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            score = sensitivity + specificity - 1
        else:
            raise ValueError(f"Métrique '{metric}' non supportée")
        
        scores.append(score)
        
        if score > best_score:
            best_score = score
            best_threshold = threshold
    
    # Métriques avec le seuil optimal
    y_pred_optimal = (y_proba >= best_threshold).astype(int)
    optimal_metrics = calculate_basic_metrics(y_true, y_pred_optimal, y_proba)
    
    results = {
        'best_threshold': best_threshold,
        'best_score': best_score,
        'optimized_metric': metric,
        'threshold_scores': list(zip(thresholds, scores)),
        'optimal_metrics': optimal_metrics,
        'y_pred_optimal': y_pred_optimal
    }
    
    log.info(f"🏆 Seuil optimal: {best_threshold:.3f} (score: {best_score:.4f})")
    
    return results


def analyze_threshold_sensitivity(y_true: np.ndarray,
                                 y_proba: np.ndarray,
                                 thresholds: np.ndarray = None,
                                 plot: bool = True,
                                 save_path: Path = None) -> pd.DataFrame:
    """
    Analyse la sensibilité des métriques aux différents seuils.
    
    Args:
        y_true: Labels vrais
        y_proba: Probabilités prédites
        thresholds: Seuils à analyser
        plot: Créer des graphiques
        save_path: Chemin pour sauvegarder les graphiques
        
    Returns:
        DataFrame avec les métriques pour chaque seuil
    """
    
    if thresholds is None:
        thresholds = np.arange(0.1, 0.95, 0.05)
    
    log.info(f"📊 Analyse de sensibilité pour {len(thresholds)} seuils")
    
    sensitivity_data = []
    
    for threshold in thresholds:
        y_pred = (y_proba >= threshold).astype(int)
        metrics = calculate_basic_metrics(y_true, y_pred, y_proba)
        
        row = {'threshold': threshold}
        row.update(metrics)
        sensitivity_data.append(row)
    
    sensitivity_df = pd.DataFrame(sensitivity_data)
    
    if plot:
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Graphique 1: F1, Precision, Recall
        axes[0, 0].plot(sensitivity_df['threshold'], sensitivity_df['f1'], 'o-', label='F1-Score')
        axes[0, 0].plot(sensitivity_df['threshold'], sensitivity_df['precision'], 's-', label='Precision')
        axes[0, 0].plot(sensitivity_df['threshold'], sensitivity_df['recall'], '^-', label='Recall')
        axes[0, 0].set_xlabel('Seuil')
        axes[0, 0].set_ylabel('Score')
        axes[0, 0].set_title('Évolution des métriques principales')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Graphique 2: Accuracy et Specificity
        axes[0, 1].plot(sensitivity_df['threshold'], sensitivity_df['accuracy'], 'o-', label='Accuracy')
        axes[0, 1].plot(sensitivity_df['threshold'], sensitivity_df['specificity'], 's-', label='Specificity')
        axes[0, 1].set_xlabel('Seuil')
        axes[0, 1].set_ylabel('Score')
        axes[0, 1].set_title('Accuracy et Spécificité')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Graphique 3: Trade-off Precision-Recall
        axes[1, 0].plot(sensitivity_df['recall'], sensitivity_df['precision'], 'o-')
        axes[1, 0].set_xlabel('Recall')
        axes[1, 0].set_ylabel('Precision')
        axes[1, 0].set_title('Trade-off Precision-Recall')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Graphique 4: Seuils optimaux par métrique
        optimal_f1_idx = sensitivity_df['f1'].idxmax()
        optimal_prec_idx = sensitivity_df['precision'].idxmax()
        optimal_rec_idx = sensitivity_df['recall'].idxmax()
        
        axes[1, 1].axvline(sensitivity_df.loc[optimal_f1_idx, 'threshold'], 
                          color='blue', linestyle='--', label='Optimal F1')
        axes[1, 1].axvline(sensitivity_df.loc[optimal_prec_idx, 'threshold'], 
                          color='red', linestyle='--', label='Optimal Precision')
        axes[1, 1].axvline(sensitivity_df.loc[optimal_rec_idx, 'threshold'], 
                          color='green', linestyle='--', label='Optimal Recall')
        
        axes[1, 1].plot(sensitivity_df['threshold'], sensitivity_df['f1'], 'o-', alpha=0.7)
        axes[1, 1].set_xlabel('Seuil')
        axes[1, 1].set_ylabel('F1-Score')
        axes[1, 1].set_title('Seuils optimaux')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            log.info(f"💾 Graphique sauvegardé: {save_path}")
        
        plt.show()
    
    return sensitivity_df


# =============================================================================
# 3. VISUALISATIONS
# =============================================================================

def plot_evaluation_dashboard(y_true: np.ndarray,
                            y_pred: np.ndarray,
                            y_proba: np.ndarray,
                            model_name: str = "Model",
                            save_path: Path = None,
                            figsize: Tuple[int, int] = (20, 12)) -> None:
    """
    Crée un tableau de bord complet d'évaluation.
    
    Args:
        y_true: Labels vrais
        y_pred: Prédictions
        y_proba: Probabilités
        model_name: Nom du modèle
        save_path: Chemin de sauvegarde
        figsize: Taille de la figure
    """
    
    fig, axes = plt.subplots(2, 4, figsize=figsize)
    fig.suptitle(f'Tableau de Bord d\'Évaluation - {model_name}', fontsize=16, fontweight='bold')
    
    # 1. Matrice de confusion
    cm = confusion_matrix(y_true, y_pred)
    ConfusionMatrixDisplay(cm, display_labels=['Non-Ad', 'Ad']).plot(ax=axes[0, 0])
    axes[0, 0].set_title('Matrice de Confusion')
    
    # 2. Matrice de confusion normalisée
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    sns.heatmap(cm_norm, annot=True, fmt='.2%', cmap='Blues', ax=axes[0, 1])
    axes[0, 1].set_title('Matrice Normalisée')
    axes[0, 1].set_xlabel('Prédictions')
    axes[0, 1].set_ylabel('Vraies valeurs')
    
    # 3. Courbe ROC
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    roc_auc = auc(fpr, tpr)
    axes[0, 2].plot(fpr, tpr, label=f'ROC (AUC = {roc_auc:.3f})', linewidth=2)
    axes[0, 2].plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Aléatoire')
    axes[0, 2].set_xlabel('Taux de Faux Positifs')
    axes[0, 2].set_ylabel('Taux de Vrais Positifs')
    axes[0, 2].set_title('Courbe ROC')
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)
    
    # 4. Courbe Precision-Recall
    precision_vals, recall_vals, _ = precision_recall_curve(y_true, y_proba)
    pr_auc = auc(recall_vals, precision_vals)
    axes[0, 3].plot(recall_vals, precision_vals, label=f'PR (AUC = {pr_auc:.3f})', linewidth=2)
    axes[0, 3].set_xlabel('Recall')
    axes[0, 3].set_ylabel('Precision')
    axes[0, 3].set_title('Courbe Precision-Recall')
    axes[0, 3].legend()
    axes[0, 3].grid(True, alpha=0.3)
    
    # 5. Distribution des scores de probabilité
    axes[1, 0].hist(y_proba[y_true == 0], bins=30, alpha=0.7, label='Non-Ad', density=True)
    axes[1, 0].hist(y_proba[y_true == 1], bins=30, alpha=0.7, label='Ad', density=True)
    axes[1, 0].set_xlabel('Probabilité prédite')
    axes[1, 0].set_ylabel('Densité')
    axes[1, 0].set_title('Distribution des Probabilités')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # 6. Métriques par seuil
    thresholds = np.arange(0.1, 0.95, 0.05)
    f1_scores = []
    precision_scores = []
    recall_scores = []
    
    for thresh in thresholds:
        pred_thresh = (y_proba >= thresh).astype(int)
        f1_scores.append(f1_score(y_true, pred_thresh, zero_division=0))
        precision_scores.append(precision_score(y_true, pred_thresh, zero_division=0))
        recall_scores.append(recall_score(y_true, pred_thresh, zero_division=0))
    
    axes[1, 1].plot(thresholds, f1_scores, 'o-', label='F1')
    axes[1, 1].plot(thresholds, precision_scores, 's-', label='Precision')
    axes[1, 1].plot(thresholds, recall_scores, '^-', label='Recall')
    axes[1, 1].set_xlabel('Seuil')
    axes[1, 1].set_ylabel('Score')
    axes[1, 1].set_title('Métriques par Seuil')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    # 7. Calibration des probabilités (diagramme de fiabilité)
    from sklearn.calibration import calibration_curve
    try:
        fraction_pos, mean_pred = calibration_curve(y_true, y_proba, n_bins=10)
        axes[1, 2].plot(mean_pred, fraction_pos, "s-", label="Modèle", linewidth=2)
        axes[1, 2].plot([0, 1], [0, 1], "k:", label="Parfaitement calibré")
        axes[1, 2].set_xlabel('Probabilité moyenne prédite')
        axes[1, 2].set_ylabel('Fraction de positifs')
        axes[1, 2].set_title('Calibration des Probabilités')
        axes[1, 2].legend()
        axes[1, 2].grid(True, alpha=0.3)
    except Exception as e:
        axes[1, 2].text(0.5, 0.5, f"Erreur de calibration:\n{str(e)}", 
                       ha='center', va='center', transform=axes[1, 2].transAxes)
        axes[1, 2].set_title('Calibration (Erreur)')
    
    # 8. Résumé des métriques
    metrics = calculate_basic_metrics(y_true, y_pred, y_proba)
    metrics_text = "\n".join([f"{k.upper()}: {v:.3f}" for k, v in metrics.items() if v is not None])
    axes[1, 3].text(0.1, 0.9, metrics_text, transform=axes[1, 3].transAxes, 
                   fontsize=12, verticalalignment='top', fontfamily='monospace',
                   bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    axes[1, 3].set_title('Résumé des Métriques')
    axes[1, 3].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        log.info(f"💾 Dashboard sauvegardé: {save_path}")
    
    plt.show()


def compare_models_visualization(results_dict: Dict[str, Dict],
                               metric: str = 'f1',
                               save_path: Path = None,
                               figsize: Tuple[int, int] = (12, 8)) -> None:
    """
    Visualise la comparaison entre plusieurs modèles.
    
    Args:
        results_dict: Dict {nom_modèle: résultats_évaluation}
        metric: Métrique principale pour la comparaison
        save_path: Chemin de sauvegarde
        figsize: Taille de la figure
    """
    
    if not results_dict:
        log.warning("Aucun résultat fourni pour la comparaison")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    fig.suptitle('Comparaison des Modèles', fontsize=16, fontweight='bold')
    
    models = list(results_dict.keys())
    
    # Extraire les métriques
    metrics_data = {
        'f1': [],
        'precision': [],
        'recall': [],
        'auc_roc': []
    }
    
    for model_name in models:
        result = results_dict[model_name]
        if 'basic_metrics' in result:
            for metric_name in metrics_data.keys():
                value = result['basic_metrics'].get(metric_name, 0)
                metrics_data[metric_name].append(value if value is not None else 0)
        else:
            # Fallback si structure différente
            for metric_name in metrics_data.keys():
                metrics_data[metric_name].append(0)
    
    # 1. Graphique en barres des métriques principales
    x = np.arange(len(models))
    width = 0.2
    
    for i, (metric_name, values) in enumerate(metrics_data.items()):
        axes[0, 0].bar(x + i*width, values, width, label=metric_name.upper())
    
    axes[0, 0].set_xlabel('Modèles')
    axes[0, 0].set_ylabel('Score')
    axes[0, 0].set_title('Métriques Principales')
    axes[0, 0].set_xticks(x + width * 1.5)
    axes[0, 0].set_xticklabels(models, rotation=45, ha='right')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Graphique radar (si plus de 3 modèles)
    if len(models) <= 5:
        from math import pi
        
        categories = list(metrics_data.keys())
        N = len(categories)
        
        angles = [n / float(N) * 2 * pi for n in range(N)]
        angles += angles[:1]  # Fermer le cercle
        
        axes[0, 1].set_theta_offset(pi / 2)
        axes[0, 1].set_theta_direction(-1)
        axes[0, 1].set_thetagrids(np.degrees(angles[:-1]), categories)
        
        for i, model_name in enumerate(models):
            values = [metrics_data[cat][i] for cat in categories]
            values += values[:1]  # Fermer le cercle
            
            axes[0, 1].plot(angles, values, 'o-', linewidth=2, label=model_name)
            axes[0, 1].fill(angles, values, alpha=0.25)
        
        axes[0, 1].set_ylim(0, 1)
        axes[0, 1].set_title('Vue Radar')
        axes[0, 1].legend(loc='upper right', bbox_to_anchor=(1.3, 1))
    else:
        axes[0, 1].text(0.5, 0.5, 'Trop de modèles\npour vue radar', 
                       ha='center', va='center', transform=axes[0, 1].transAxes)
        axes[0, 1].set_title('Vue Radar (Non disponible)')
    
    # 3. Classement par métrique sélectionnée
    if metric in metrics_data:
        sorted_indices = np.argsort(metrics_data[metric])[::-1]
        sorted_models = [models[i] for i in sorted_indices]
        sorted_values = [metrics_data[metric][i] for i in sorted_indices]
        
        bars = axes[1, 0].barh(range(len(sorted_models)), sorted_values)
        axes[1, 0].set_yticks(range(len(sorted_models)))
        axes[1, 0].set_yticklabels(sorted_models)
        axes[1, 0].set_xlabel(f'Score {metric.upper()}')
        axes[1, 0].set_title(f'Classement par {metric.upper()}')
        axes[1, 0].grid(True, alpha=0.3, axis='x')
        
        # Colorer le meilleur modèle
        if bars:
            bars[0].set_color('gold')
    
    # 4. Tableau récapitulatif
    table_data = []
    for model in models:
        row = [model] + [f"{metrics_data[m][models.index(model)]:.3f}" 
                        for m in ['f1', 'precision', 'recall', 'auc_roc']]
        table_data.append(row)
    
    table = axes[1, 1].table(
        cellText=table_data,
        colLabels=['Modèle', 'F1', 'Precision', 'Recall', 'AUC-ROC'],
        cellLoc='center',
        loc='center'
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.5)
    axes[1, 1].axis('off')
    axes[1, 1].set_title('Tableau Récapitulatif')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        log.info(f"💾 Comparaison sauvegardée: {save_path}")
    
    plt.show()


# =============================================================================
# 4. RAPPORTS D'ÉVALUATION
# =============================================================================

def generate_evaluation_report(results_dict: Dict[str, Any],
                             model_name: str,
                             save_report: bool = True) -> str:
    """
    Génère un rapport d'évaluation détaillé.
    
    Args:
        results_dict: Résultats d'évaluation
        model_name: Nom du modèle
        save_report: Sauvegarder le rapport
        
    Returns:
        Rapport au format texte
    """
    
    log.info(f"📝 Génération du rapport pour {model_name}")
    
    report_lines = []
    report_lines.append("=" * 80)
    report_lines.append(f"RAPPORT D'ÉVALUATION - {model_name.upper()}")
    report_lines.append("=" * 80)
    report_lines.append("")
    
    # Métriques de base
    if 'basic_metrics' in results_dict:
        report_lines.append("MÉTRIQUES PRINCIPALES:")
        report_lines.append("-" * 30)
        for metric, value in results_dict['basic_metrics'].items():
            if value is not None:
                report_lines.append(f"{metric.upper():<15}: {value:.4f}")
        report_lines.append("")
    
    # Rapport de classification détaillé
    if 'classification_report' in results_dict:
        report_lines.append("RAPPORT DE CLASSIFICATION:")
        report_lines.append("-" * 30)
        
        class_report = results_dict['classification_report']
        for class_name, metrics in class_report.items():
            if isinstance(metrics, dict) and class_name not in ['accuracy', 'macro avg', 'weighted avg']:
                report_lines.append(f"\nClasse: {class_name}")
                for metric, value in metrics.items():
                    if metric != 'support':
                        report_lines.append(f"  {metric:<12}: {value:.4f}")
                    else:
                        report_lines.append(f"  {metric:<12}: {value}")
        
        # Moyennes
        if 'macro avg' in class_report:
            report_lines.append(f"\nMoyenne macro:")
            for metric, value in class_report['macro avg'].items():
                if metric != 'support':
                    report_lines.append(f"  {metric:<12}: {value:.4f}")
        
        if 'weighted avg' in class_report:
            report_lines.append(f"\nMoyenne pondérée:")
            for metric, value in class_report['weighted avg'].items():
                if metric != 'support':
                    report_lines.append(f"  {metric:<12}: {value:.4f}")
        
        report_lines.append("")
    
    # Matrice de confusion
    if 'confusion_matrix' in results_dict:
        report_lines.append("MATRICE DE CONFUSION:")
        report_lines.append("-" * 30)
        cm = results_dict['confusion_matrix']
        report_lines.append("                Prédictions")
        report_lines.append("                Non-Ad    Ad")
        report_lines.append(f"Vraies  Non-Ad  {cm[0,0]:6d} {cm[0,1]:6d}")
        report_lines.append(f"valeurs Ad      {cm[1,0]:6d} {cm[1,1]:6d}")
        report_lines.append("")
        
        # Calculs dérivés
        tn, fp, fn, tp = cm.ravel()
        report_lines.append("CALCULS DÉRIVÉS:")
        report_lines.append("-" * 20)
        report_lines.append(f"Vrais Positifs (TP):   {tp}")
        report_lines.append(f"Vrais Négatifs (TN):   {tn}")
        report_lines.append(f"Faux Positifs (FP):    {fp}")
        report_lines.append(f"Faux Négatifs (FN):    {fn}")
        report_lines.append(f"Total échantillons:    {tp + tn + fp + fn}")
        report_lines.append("")
    
    # Seuil optimal si disponible
    if 'optimal_threshold' in results_dict:
        threshold_info = results_dict['optimal_threshold']
        report_lines.append("SEUIL OPTIMAL:")
        report_lines.append("-" * 20)
        report_lines.append(f"Seuil:                 {threshold_info['best_threshold']:.3f}")
        report_lines.append(f"Métrique optimisée:    {threshold_info['optimized_metric']}")
        report_lines.append(f"Score optimal:         {threshold_info['best_score']:.4f}")
        report_lines.append("")
    
    # Informations sur l'entraînement
    if 'training_time' in results_dict:
        report_lines.append("INFORMATIONS ENTRAÎNEMENT:")
        report_lines.append("-" * 30)
        report_lines.append(f"Temps d'entraînement:  {results_dict['training_time']:.2f}s")
        if 'best_cv_score' in results_dict and results_dict['best_cv_score']:
            report_lines.append(f"Meilleur score CV:     {results_dict['best_cv_score']:.4f}")
        report_lines.append("")
    
    report_lines.append("=" * 80)
    report_text = "\n".join(report_lines)
    
    if save_report:
        report_path = cfg.paths.outputs / f"evaluation_report_{model_name.lower()}.txt"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report_text)
        log.info(f"💾 Rapport sauvegardé: {report_path}")
    
    return report_text


# =============================================================================
# 5. UTILITAIRES
# =============================================================================

def export_metrics_to_csv(results_dict: Dict[str, Dict],
                         filename: str = "model_metrics.csv") -> Path:
    """
    Exporte les métriques de plusieurs modèles vers un CSV.
    
    Args:
        results_dict: Dict {nom_modèle: résultats}
        filename: Nom du fichier CSV
        
    Returns:
        Chemin du fichier sauvegardé
    """
    
    export_data = []
    
    for model_name, results in results_dict.items():
        row = {'model': model_name}
        
        if 'basic_metrics' in results:
            row.update(results['basic_metrics'])
        
        if 'training_time' in results:
            row['training_time'] = results['training_time']
        
        if 'best_cv_score' in results:
            row['cv_score'] = results['best_cv_score']
        
        export_data.append(row)
    
    export_df = pd.DataFrame(export_data)
    export_path = cfg.paths.outputs / filename
    export_df.to_csv(export_path, index=False)
    
    log.info(f"💾 Métriques exportées: {export_path}")
    return export_path