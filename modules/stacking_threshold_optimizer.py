# modules/stacking_threshold_optimizer.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import joblib
import json
from sklearn.metrics import f1_score, precision_score, recall_score, roc_curve, auc
from sklearn.model_selection import GridSearchCV
import logging

log = logging.getLogger(__name__)

def optimize_stacking_thresholds(X_train_knn, X_val_knn, y_train_knn, y_val_knn,
                                X_train_mice, X_val_mice, y_train_mice, y_val_mice,
                                optimization_method="f1", verbose=True):
    """
    Optimise les seuils pour les modèles de stacking KNN et MICE.
    
    Parameters:
    -----------
    X_train_knn, X_val_knn, y_train_knn, y_val_knn : données KNN
    X_train_mice, X_val_mice, y_train_mice, y_val_mice : données MICE
    optimization_method : str, méthode d'optimisation ("f1", "precision", "recall")
    verbose : bool, affichage détaillé
    
    Returns:
    --------
    dict : résultats d'optimisation
    """
    if verbose:
        print("Optimisation des seuils pour les modèles de stacking...")
    
    results = {}
    
    # Cette fonction nécessite vos modèles de stacking
    # Elle sera utilisée si vous voulez entraîner et optimiser en même temps
    if verbose:
        print("Note: Cette fonction nécessite des modèles de stacking pré-entraînés")
    
    return results

def optimize_stacking_thresholds_with_trained_models(stacking_knn, stacking_mice,
                                                    X_val_knn, y_val_knn,
                                                    X_val_mice, y_val_mice,
                                                    verbose=True):
    """
    Optimise les seuils en utilisant des modèles de stacking déjà entraînés.
    
    Parameters:
    -----------
    stacking_knn : modèle de stacking KNN entraîné
    stacking_mice : modèle de stacking MICE entraîné
    X_val_knn, y_val_knn : données de validation KNN
    X_val_mice, y_val_mice : données de validation MICE
    verbose : bool, affichage détaillé
    
    Returns:
    --------
    dict : résultats d'optimisation
    """
    if verbose:
        print("Optimisation des seuils avec modèles entraînés...")
    
    results = {}
    
    # Optimisation pour KNN
    if stacking_knn is not None:
        if verbose:
            print("Optimisation seuils KNN...")
        
        try:
            # Prédictions sur validation
            y_proba_knn = stacking_knn.predict_proba(X_val_knn)[:, 1]
            
            # Optimisation du seuil
            thresholds = np.arange(0.1, 0.9, 0.01)
            best_f1 = 0
            best_threshold = 0.5
            best_precision = 0
            best_recall = 0
            
            for threshold in thresholds:
                y_pred = (y_proba_knn >= threshold).astype(int)
                f1 = f1_score(y_val_knn, y_pred)
                precision = precision_score(y_val_knn, y_pred)
                recall = recall_score(y_val_knn, y_pred)
                
                if f1 > best_f1:
                    best_f1 = f1
                    best_threshold = threshold
                    best_precision = precision
                    best_recall = recall
            
            results['knn'] = {
                'best_threshold': best_threshold,
                'best_f1': best_f1,
                'best_precision': best_precision,
                'best_recall': best_recall,
                'y_proba': y_proba_knn
            }
            
            if verbose:
                print(f"KNN - Meilleur seuil: {best_threshold:.3f}, F1: {best_f1:.3f}")
                print(f"     Precision: {best_precision:.3f}, Recall: {best_recall:.3f}")
                
        except Exception as e:
            if verbose:
                print(f"Erreur lors de l'optimisation KNN: {e}")
            results['knn'] = None
    
    # Optimisation pour MICE
    if stacking_mice is not None:
        if verbose:
            print("Optimisation seuils MICE...")
        
        try:
            # Prédictions sur validation
            y_proba_mice = stacking_mice.predict_proba(X_val_mice)[:, 1]
            
            # Optimisation du seuil
            thresholds = np.arange(0.1, 0.9, 0.01)
            best_f1 = 0
            best_threshold = 0.5
            best_precision = 0
            best_recall = 0
            
            for threshold in thresholds:
                y_pred = (y_proba_mice >= threshold).astype(int)
                f1 = f1_score(y_val_mice, y_pred)
                precision = precision_score(y_val_mice, y_pred)
                recall = recall_score(y_val_mice, y_pred)
                
                if f1 > best_f1:
                    best_f1 = f1
                    best_threshold = threshold
                    best_precision = precision
                    best_recall = recall
            
            results['mice'] = {
                'best_threshold': best_threshold,
                'best_f1': best_f1,
                'best_precision': best_precision,
                'best_recall': best_recall,
                'y_proba': y_proba_mice
            }
            
            if verbose:
                print(f"MICE - Meilleur seuil: {best_threshold:.3f}, F1: {best_f1:.3f}")
                print(f"      Precision: {best_precision:.3f}, Recall: {best_recall:.3f}")
                
        except Exception as e:
            if verbose:
                print(f"Erreur lors de l'optimisation MICE: {e}")
            results['mice'] = None
    
    return results

def plot_optimization_results(results, y_val_knn=None, y_val_mice=None, save_path=None):
    """
    Génère les graphiques des résultats d'optimisation.
    
    Parameters:
    -----------
    results : dict, résultats d'optimisation
    y_val_knn, y_val_mice : vraies valeurs pour calculer les courbes ROC
    save_path : str, chemin de sauvegarde (optionnel)
    """
    if not results:
        print("Aucun résultat à afficher")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Courbes ROC
    if 'knn' in results and results['knn'] is not None and 'y_proba' in results['knn']:
        if y_val_knn is not None:
            fpr_knn, tpr_knn, _ = roc_curve(y_val_knn, results['knn']['y_proba'])
            auc_knn = auc(fpr_knn, tpr_knn)
            axes[0, 0].plot(fpr_knn, tpr_knn, label=f'KNN (AUC={auc_knn:.3f})')
    
    if 'mice' in results and results['mice'] is not None and 'y_proba' in results['mice']:
        if y_val_mice is not None:
            fpr_mice, tpr_mice, _ = roc_curve(y_val_mice, results['mice']['y_proba'])
            auc_mice = auc(fpr_mice, tpr_mice)
            axes[0, 0].plot(fpr_mice, tpr_mice, label=f'MICE (AUC={auc_mice:.3f})')
    
    axes[0, 0].plot([0, 1], [0, 1], 'k--', alpha=0.5)
    axes[0, 0].set_xlabel('Taux de faux positifs')
    axes[0, 0].set_ylabel('Taux de vrais positifs')
    axes[0, 0].set_title('Courbes ROC')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Comparaison des F1 scores
    methods = []
    f1_scores = []
    
    if 'knn' in results and results['knn'] is not None:
        methods.append('KNN')
        f1_scores.append(results['knn']['best_f1'])
    
    if 'mice' in results and results['mice'] is not None:
        methods.append('MICE')
        f1_scores.append(results['mice']['best_f1'])
    
    if methods:
        colors = ['skyblue', 'lightcoral'][:len(methods)]
        axes[0, 1].bar(methods, f1_scores, color=colors)
        axes[0, 1].set_ylabel('F1 Score')
        axes[0, 1].set_title('Comparaison F1 Scores')
        axes[0, 1].set_ylim(0, 1)
        
        # Ajouter les valeurs sur les barres
        for i, v in enumerate(f1_scores):
            axes[0, 1].text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom')
    
    # Seuils optimaux
    thresholds = []
    if 'knn' in results and results['knn'] is not None:
        thresholds.append(results['knn']['best_threshold'])
    if 'mice' in results and results['mice'] is not None:
        thresholds.append(results['mice']['best_threshold'])
    
    if thresholds:
        colors = ['skyblue', 'lightcoral'][:len(thresholds)]
        axes[1, 0].bar(methods, thresholds, color=colors)
        axes[1, 0].set_ylabel('Seuil optimal')
        axes[1, 0].set_title('Seuils optimaux')
        axes[1, 0].set_ylim(0, 1)
        
        for i, v in enumerate(thresholds):
            axes[1, 0].text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom')
    
    # Résumé textuel
    axes[1, 1].axis('off')
    summary_text = "Résumé de l'optimisation:\n\n"
    
    if 'knn' in results and results['knn'] is not None:
        summary_text += f"KNN:\n"
        summary_text += f"  - Seuil optimal: {results['knn']['best_threshold']:.3f}\n"
        summary_text += f"  - F1 Score: {results['knn']['best_f1']:.3f}\n"
        summary_text += f"  - Precision: {results['knn']['best_precision']:.3f}\n"
        summary_text += f"  - Recall: {results['knn']['best_recall']:.3f}\n\n"
    
    if 'mice' in results and results['mice'] is not None:
        summary_text += f"MICE:\n"
        summary_text += f"  - Seuil optimal: {results['mice']['best_threshold']:.3f}\n"
        summary_text += f"  - F1 Score: {results['mice']['best_f1']:.3f}\n"
        summary_text += f"  - Precision: {results['mice']['best_precision']:.3f}\n"
        summary_text += f"  - Recall: {results['mice']['best_recall']:.3f}\n"
    
    axes[1, 1].text(0.1, 0.9, summary_text, transform=axes[1, 1].transAxes, 
                   fontsize=10, verticalalignment='top')
    
    plt.tight_layout()
    
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Graphique sauvegardé: {save_path}")
    
    plt.show()

def save_optimization_results(results, save_dir=None):
    """
    Sauvegarde les résultats d'optimisation.
    
    Parameters:
    -----------
    results : dict, résultats d'optimisation
    save_dir : str, répertoire de sauvegarde (optionnel)
    """
    if save_dir is None:
        save_dir = Path("outputs/modeling/thresholds")
    else:
        save_dir = Path(save_dir)
    
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Sauvegarde JSON
    json_path = save_dir / "optimization_results.json"
    
    # Conversion pour JSON serializable
    json_results = {}
    for key, value in results.items():
        if value is not None and isinstance(value, dict):
            json_results[key] = {k: float(v) if isinstance(v, (np.float32, np.float64)) else v 
                               for k, v in value.items() if k != 'y_proba'}
        else:
            json_results[key] = value
    
    with open(json_path, 'w') as f:
        json.dump(json_results, f, indent=2, default=str)
    
    # Sauvegarde CSV
    if json_results:
        df_results = []
        for method, data in json_results.items():
            if isinstance(data, dict):
                row = {'method': method}
                row.update(data)
                df_results.append(row)
        
        if df_results:
            df = pd.DataFrame(df_results)
            csv_path = save_dir / "optimization_results.csv"
            df.to_csv(csv_path, index=False)
            print(f"Résultats sauvegardés: {csv_path}")
    
    print(f"Résultats sauvegardés: {json_path}")

def generate_mean_proba(y_proba_knn, y_proba_mice, weights=None):
    """
    Génère les probabilités moyennes pondérées.
    
    Parameters:
    -----------
    y_proba_knn : array, probabilités KNN
    y_proba_mice : array, probabilités MICE
    weights : list, poids pour chaque méthode [knn_weight, mice_weight]
    
    Returns:
    --------
    array : probabilités moyennes
    """
    if weights is None:
        weights = [0.5, 0.5]  # Poids égaux par défaut
    
    # Normalisation des poids
    weights = np.array(weights) / np.sum(weights)
    
    # Calcul de la moyenne pondérée
    mean_proba = weights[0] * y_proba_knn + weights[1] * y_proba_mice
    
    return mean_proba