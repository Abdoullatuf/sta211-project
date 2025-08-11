"""
Module pour les graphiques de sélection de features
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
import seaborn as sns

def plot_rfecv_evolution(results_dict, model_name="Model", save_path=None, figsize=(12, 8)):
    """
    Crée un graphique montrant l'évolution du score en fonction du nombre de variables.
    
    Args:
        results_dict (dict): Dictionnaire contenant les résultats de l'analyse RFECV
        model_name (str): Nom du modèle pour le titre
        save_path (str/Path): Chemin pour sauvegarder le graphique
        figsize (tuple): Taille de la figure
    """
    
    if 'rfecv' not in results_dict or results_dict['rfecv'] is None:
        print("❌ Aucune donnée RFECV disponible dans les résultats")
        return
    
    rfecv_data = results_dict['rfecv']
    
    # Extraire les données
    n_features_range = range(1, len(rfecv_data['grid_scores_mean']) + 1)
    scores_mean = rfecv_data['grid_scores_mean']
    scores_std = rfecv_data['grid_scores_std']
    optimal_n_features = rfecv_data['n_features_optimal']
    best_score = rfecv_data['best_score']
    
    # Créer la figure
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, height_ratios=[3, 1])
    
    # Graphique principal : Score vs Nombre de features
    ax1.plot(n_features_range, scores_mean, 'b-', linewidth=2, label='Score moyen')
    ax1.fill_between(n_features_range, 
                     np.array(scores_mean) - np.array(scores_std),
                     np.array(scores_mean) + np.array(scores_std),
                     alpha=0.3, color='blue', label='± 1 écart-type')
    
    # Marquer le point optimal
    ax1.axvline(x=optimal_n_features, color='red', linestyle='--', 
                linewidth=2, label=f'Optimal: {optimal_n_features} features')
    ax1.axhline(y=best_score, color='green', linestyle=':', 
                linewidth=2, label=f'Meilleur score: {best_score:.4f}')
    
    # Personnalisation du graphique principal
    ax1.set_xlabel('Nombre de features', fontsize=12)
    ax1.set_ylabel('Score F1', fontsize=12)
    ax1.set_title(f'Évolution du Score RFECV - {model_name}', fontsize=14, fontweight='bold')
    ax1.legend(loc='best')
    ax1.grid(True, alpha=0.3)
    
    # Graphique secondaire : Écart-type
    ax2.plot(n_features_range, scores_std, 'r-', linewidth=2, alpha=0.7)
    ax2.axvline(x=optimal_n_features, color='red', linestyle='--', linewidth=2)
    ax2.set_xlabel('Nombre de features', fontsize=12)
    ax2.set_ylabel('Écart-type', fontsize=12)
    ax2.set_title('Stabilité du Score (Écart-type)', fontsize=12)
    ax2.grid(True, alpha=0.3)
    
    # Ajuster l'espacement
    plt.tight_layout()
    
    # Sauvegarder si demandé
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"📊 Graphique sauvegardé : {save_path}")
    
    plt.show()
    
    # Afficher les statistiques
    print(f"\n📈 Statistiques RFECV - {model_name}:")
    print(f"   • Nombre optimal de features : {optimal_n_features}")
    print(f"   • Meilleur score F1 : {best_score:.4f}")
    print(f"   • Score avec toutes les features : {scores_mean[-1]:.4f}")
    print(f"   • Amélioration : {best_score - scores_mean[-1]:.4f}")
    
    return fig

def plot_simple_rfecv_evolution(results_dict, model_name="Model", save_path=None, figsize=(12, 8)):
    """
    Crée un graphique simple montrant uniquement l'évolution du score RFECV.
    
    Args:
        results_dict (dict): Dictionnaire contenant les résultats de l'analyse RFECV
        model_name (str): Nom du modèle pour le titre
        save_path (str/Path): Chemin pour sauvegarder le graphique
        figsize (tuple): Taille de la figure
    """
    
    if 'rfecv' not in results_dict or results_dict['rfecv'] is None:
        print("❌ Aucune donnée RFECV disponible dans les résultats")
        return
    
    rfecv_data = results_dict['rfecv']
    
    # Extraire les données
    n_features_range = range(1, len(rfecv_data['grid_scores_mean']) + 1)
    scores_mean = rfecv_data['grid_scores_mean']
    scores_std = rfecv_data['grid_scores_std']
    optimal_n_features = rfecv_data['n_features_optimal']
    best_score = rfecv_data['best_score']
    
    # Créer la figure
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    
    # Graphique principal : Score vs Nombre de features
    ax.plot(n_features_range, scores_mean, 'b-', linewidth=3, label='Score F1 moyen', color='#2E86AB')
    ax.fill_between(n_features_range, 
                     np.array(scores_mean) - np.array(scores_std),
                     np.array(scores_mean) + np.array(scores_std),
                     alpha=0.3, color='#2E86AB', label='± 1 écart-type')
    
    # Marquer le point optimal
    ax.axvline(x=optimal_n_features, color='red', linestyle='--', 
                linewidth=3, label=f'Optimal: {optimal_n_features} features')
    ax.axhline(y=best_score, color='green', linestyle=':', 
                linewidth=2, label=f'Meilleur score: {best_score:.4f}')
    
    # Personnalisation du graphique
    ax.set_xlabel('Nombre de features', fontsize=14, fontweight='bold')
    ax.set_ylabel('Score F1', fontsize=14, fontweight='bold')
    ax.set_title(f'Évolution du Score en Fonction du Nombre de Variables\n{model_name}', 
                 fontsize=16, fontweight='bold', pad=20)
    ax.legend(loc='best', fontsize=12)
    ax.grid(True, alpha=0.3)
    
    # Améliorer l'apparence
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Ajuster l'espacement
    plt.tight_layout()
    
    # Sauvegarder si demandé
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"📊 Graphique simple sauvegardé : {save_path}")
    
    plt.show()
    
    # Afficher les statistiques
    print(f"\n📈 Statistiques RFECV - {model_name}:")
    print(f"   • Nombre optimal de features : {optimal_n_features}")
    print(f"   • Meilleur score F1 : {best_score:.4f}")
    print(f"   • Score avec toutes les features : {scores_mean[-1]:.4f}")
    print(f"   • Amélioration : {best_score - scores_mean[-1]:.4f}")
    
    return fig

def plot_feature_importance_comparison(results_dict, model_name="Model", save_path=None, figsize=(15, 10)):
    """
    Crée un graphique comparatif des différentes méthodes d'importance des features.
    
    Args:
        results_dict (dict): Dictionnaire contenant les résultats de l'analyse
        model_name (str): Nom du modèle pour le titre
        save_path (str/Path): Chemin pour sauvegarder le graphique
        figsize (tuple): Taille de la figure
    """
    
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    fig.suptitle(f'Analyse d\'Importance des Features - {model_name}', fontsize=16, fontweight='bold')
    
    # 1. Top features par permutation importance
    if ('permutation' in results_dict and results_dict['permutation'] is not None and 
        'dataframe' in results_dict['permutation'] and results_dict['permutation']['dataframe']):
        
        try:
            perm_data = pd.DataFrame(results_dict['permutation']['dataframe'])
            if not perm_data.empty and 'importance_mean' in perm_data.columns:
                top_10_perm = perm_data.head(10)
                
                # Créer le graphique en barres horizontales
                y_pos = range(len(top_10_perm))
                axes[0, 0].barh(y_pos, top_10_perm['importance_mean'], color='skyblue', alpha=0.7)
                axes[0, 0].set_yticks(y_pos)
                axes[0, 0].set_yticklabels(top_10_perm['feature'])
                axes[0, 0].set_xlabel('Importance moyenne')
                axes[0, 0].set_title('Top 10 - Permutation Importance')
                axes[0, 0].grid(True, alpha=0.3)
                
                # Ajouter les valeurs sur les barres
                for i, v in enumerate(top_10_perm['importance_mean']):
                    axes[0, 0].text(v + 0.001, i, f'{v:.3f}', va='center', fontsize=9)
            else:
                axes[0, 0].text(0.5, 0.5, 'Données permutation\nnon disponibles', 
                               ha='center', va='center', transform=axes[0, 0].transAxes,
                               bbox=dict(boxstyle='round', facecolor='lightgray'))
                axes[0, 0].set_title('Top 10 - Permutation Importance')
        except Exception as e:
            axes[0, 0].text(0.5, 0.5, f'Erreur permutation:\n{str(e)[:50]}', 
                           ha='center', va='center', transform=axes[0, 0].transAxes,
                           bbox=dict(boxstyle='round', facecolor='lightcoral'))
            axes[0, 0].set_title('Top 10 - Permutation Importance')
    else:
        axes[0, 0].text(0.5, 0.5, 'Permutation Importance\nnon disponible', 
                       ha='center', va='center', transform=axes[0, 0].transAxes,
                       bbox=dict(boxstyle='round', facecolor='lightgray'))
        axes[0, 0].set_title('Top 10 - Permutation Importance')
    
    # 2. Évolution RFECV (version simplifiée)
    if 'rfecv' in results_dict and results_dict['rfecv'] is not None:
        rfecv_data = results_dict['rfecv']
        n_features_range = range(1, len(rfecv_data['grid_scores_mean']) + 1)
        scores_mean = rfecv_data['grid_scores_mean']
        
        axes[0, 1].plot(n_features_range, scores_mean, 'b-', linewidth=2, label='Score F1')
        axes[0, 1].axvline(x=rfecv_data['n_features_optimal'], color='red', linestyle='--', 
                           label=f'Optimal: {rfecv_data["n_features_optimal"]}')
        axes[0, 1].set_xlabel('Nombre de features')
        axes[0, 1].set_ylabel('Score F1')
        axes[0, 1].set_title('Évolution RFECV')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
    else:
        axes[0, 1].text(0.5, 0.5, 'Données RFECV\nnon disponibles', 
                       ha='center', va='center', transform=axes[0, 1].transAxes,
                       bbox=dict(boxstyle='round', facecolor='lightgray'))
        axes[0, 1].set_title('Évolution RFECV')
    
    # 3. Distribution des scores RFECV
    if 'rfecv' in results_dict and results_dict['rfecv'] is not None:
        rfecv_data = results_dict['rfecv']
        scores_mean = rfecv_data['grid_scores_mean']
        
        axes[1, 0].hist(scores_mean, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        axes[1, 0].axvline(x=rfecv_data['best_score'], color='red', linestyle='--', 
                           label=f'Meilleur: {rfecv_data["best_score"]:.4f}')
        axes[1, 0].set_xlabel('Score F1')
        axes[1, 0].set_ylabel('Fréquence')
        axes[1, 0].set_title('Distribution des Scores RFECV')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
    else:
        axes[1, 0].text(0.5, 0.5, 'Données RFECV\nnon disponibles', 
                       ha='center', va='center', transform=axes[1, 0].transAxes,
                       bbox=dict(boxstyle='round', facecolor='lightgray'))
        axes[1, 0].set_title('Distribution des Scores RFECV')
    
    # 4. Résumé des méthodes appliquées et statistiques
    methods_applied = results_dict.get('methods_applied', [])
    
    # Créer un résumé plus détaillé
    summary_text = f"Méthodes appliquées:\n" + "\n".join(methods_applied)
    
    # Ajouter des statistiques si disponibles
    if 'rfecv' in results_dict and results_dict['rfecv'] is not None:
        rfecv_data = results_dict['rfecv']
        summary_text += f"\n\nRFECV - Optimal: {rfecv_data['n_features_optimal']} features"
        summary_text += f"\nMeilleur score: {rfecv_data['best_score']:.4f}"
    
    if 'permutation' in results_dict and results_dict['permutation'] is not None:
        summary_text += f"\n\nPermutation - Top features analysées"
    
    if 'shap' in results_dict and results_dict['shap'] is not None:
        summary_text += f"\n\nSHAP - {results_dict['shap'].get('explainer_type', 'N/A')}"
    
    axes[1, 1].text(0.05, 0.95, summary_text, 
                     transform=axes[1, 1].transAxes, fontsize=10, 
                     verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightgray'))
    axes[1, 1].set_title('Résumé de l\'Analyse')
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    
    # Sauvegarder si demandé
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"📊 Graphique comparatif sauvegardé : {save_path}")
    
    plt.show()
    
    return fig

def create_feature_selection_summary(results_dict, model_name="Model"):
    """
    Crée un résumé textuel de l'analyse de sélection de features.
    
    Args:
        results_dict (dict): Dictionnaire contenant les résultats de l'analyse
        model_name (str): Nom du modèle
    """
    
    print(f"\n{'='*60}")
    print(f"📊 RÉSUMÉ DE L'ANALYSE - {model_name}")
    print(f"{'='*60}")
    
    # RFECV
    if 'rfecv' in results_dict and results_dict['rfecv'] is not None:
        rfecv_data = results_dict['rfecv']
        print(f"\n🎯 RFECV (Recursive Feature Elimination with Cross-Validation):")
        print(f"   • Nombre optimal de features : {rfecv_data['n_features_optimal']}")
        print(f"   • Meilleur score F1 : {rfecv_data['best_score']:.4f}")
        print(f"   • Features sélectionnées : {len(rfecv_data['selected_features'])}")
        
        if len(rfecv_data['selected_features']) <= 10:
            print(f"   • Top features : {', '.join(rfecv_data['selected_features'][:5])}")
    
    # Permutation Importance
    if 'permutation' in results_dict and results_dict['permutation'] is not None:
        perm_data = pd.DataFrame(results_dict['permutation']['dataframe'])
        print(f"\n🔄 Permutation Importance:")
        print(f"   • Top 5 features : {', '.join(perm_data.head(5)['feature'].tolist())}")
        print(f"   • Importance max : {perm_data['importance_mean'].max():.4f}")
        print(f"   • Importance min : {perm_data['importance_mean'].min():.4f}")
    
    # SHAP
    if 'shap' in results_dict and results_dict['shap'] is not None:
        print(f"\n📊 SHAP Analysis:")
        print(f"   • Type d'explainer : {results_dict['shap']['explainer_type']}")
        print(f"   • Échantillon utilisé : {results_dict['shap']['n_samples_used']} samples")
    
    print(f"\n{'='*60}") 





import joblib, json
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import roc_curve, auc
import pandas as pd
import logging

log = logging.getLogger(__name__)

def plot_best_roc_curves_comparison(models_dir: Path, figures_dir: Path, splits: dict):
    """
    Trace les courbes ROC comparées des meilleurs modèles KNN et MICE sur le jeu de validation.

    Paramètres :
    ------------
    models_dir : Path
        Dossier contenant les fichiers pipeline + seuils + df_all_thresholds.csv
    figures_dir : Path
        Dossier de sortie pour enregistrer la figure
    splits : dict
        Dictionnaire contenant les données de validation (X_val, y_val)
        pour les combinaisons : knn_full, knn_reduced, mice_full, mice_reduced
    """
    df_path = models_dir / "df_all_thresholds.csv"
    if not df_path.exists():
        raise FileNotFoundError(f"❌ Fichier df_all_thresholds.csv introuvable : {df_path}")

    df_all = pd.read_csv(df_path)
    df_all.columns = df_all.columns.str.lower()

    if "imputation" not in df_all.columns or "version" not in df_all.columns:
        raise KeyError("❌ Les colonnes 'imputation' et 'version' sont requises dans df_all_thresholds.csv")

    best_combinations = (
        df_all.sort_values("f1", ascending=False)
              .groupby(["imputation", "version"], as_index=False)
              .first()
    )

    color_map = {
        ("knn", "full"): "#1f77b4",
        ("knn", "reduced"): "#2ca02c",
        ("mice", "full"): "#ff7f0e",
        ("mice", "reduced"): "#d62728",
    }

    plt.figure(figsize=(8, 6))

    for _, row in best_combinations.iterrows():
        model = row["model"]
        imp = row["imputation"]
        version = row["version"]

        key = f"{imp}_{version}".lower() # Convert key to lowercase

        # --- Debugging Print Statements ---
        log.info(f"Attempting to get split for key: {key}")
        split = splits.get(key)
        if split is None:
            log.warning(f"⚠️ Data missing for key: {key}. Split dictionary is None.")
            continue
        log.info(f"Split dictionary for key {key}: {split.keys()}")
        if "val" not in split:
             log.warning(f"⚠️ Data missing for key: {key}. 'val' not in split dictionary.")
             continue
        log.info(f"Accessing 'val' key. Contains: {split['val'].keys()}")
        # --- End Debugging Print Statements ---


        # Corrected pipeline path construction
        model_specific_dir = models_dir / model.lower() / version.lower()
        pipe_path = model_specific_dir / f"pipeline_{model.lower()}_{imp.lower()}_{version.lower()}.joblib"


        if not pipe_path.exists():
            print(f"❌ Pipeline manquant : {pipe_path.name}")
            continue

        pipe = joblib.load(pipe_path)


        # Access validation data using the consistent structure
        X_val = split["val"].get("X")
        y_val = split["val"].get("y")


        if X_val is None or y_val is None:
            print(f"❌ Données de validation incomplètes pour {key}")
            continue

        y_scores = pipe.predict_proba(X_val)[:, 1] if hasattr(pipe, "predict_proba") else pipe.decision_function(X_val)
        fpr, tpr, _ = roc_curve(y_val, y_scores)
        roc_auc = auc(fpr, tpr)

        label = f"{model} ({imp.upper()}-{version.upper()}) – AUC={roc_auc:.3f}"
        color = color_map.get((imp, version), None)
        plt.plot(fpr, tpr, lw=2, label=label, color=color)

    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", alpha=0.5)
    plt.xlabel("Taux de faux positifs")
    plt.ylabel("Taux de vrais positifs")
    plt.title("Comparaison des meilleures courbes ROC (Validation)", fontsize=13)
    plt.legend(loc="lower right", fontsize=9)
    plt.grid(alpha=0.3)
    plt.tight_layout()

    figures_dir.mkdir(parents=True, exist_ok=True)
    save_path = figures_dir / "roc_comparison_best_models.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"✅ Courbe ROC sauvegardée → {save_path}")
    plt.show()