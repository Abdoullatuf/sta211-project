"""
Module consolidé de modélisation et analyse - STA211 Project
Consolidation de: modeling.py, stacking.py, stacking_threshold_optimizer.py

"""

import joblib
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

# Scikit-learn imports
from sklearn.metrics import (
    precision_recall_curve, f1_score, precision_score, recall_score, 
    roc_auc_score, confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc,
    classification_report, mean_squared_error, mean_absolute_error
)
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.ensemble import StackingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.feature_selection import RFECV
from sklearn.inspection import permutation_importance

# Configuration du logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
log = logging.getLogger(__name__)

# =============================================================================
# 1. OPTIMISATION DES SEUILS (ex-modeling.py)
# =============================================================================

def optimize_threshold(pipeline, X_val, y_val, *, plot=True, label="model") -> dict:
    """Optimise le seuil de classification pour maximiser le F1-score."""
    if hasattr(pipeline, "predict_proba"):
        y_scores = pipeline.predict_proba(X_val)[:, 1]
    elif hasattr(pipeline, "decision_function"):
        y_scores = pipeline.decision_function(X_val)
    else:
        raise AttributeError(f"Pipeline {label} n'a ni predict_proba ni decision_function")

    prec, rec, thr = precision_recall_curve(y_val, y_scores)
    f1 = 2 * prec * rec / (prec + rec + 1e-9)
    best_idx = np.argmax(f1)
    best_thr = float(thr[best_idx])
    y_pred_opt = (y_scores >= best_thr).astype(int)

    metrics = {
        "threshold": round(best_thr, 4),
        "f1": round(f1_score(y_val, y_pred_opt), 4),
        "precision": round(precision_score(y_val, y_pred_opt, zero_division=0), 4),
        "recall": round(recall_score(y_val, y_pred_opt, zero_division=0), 4),
    }

    if plot:
        plt.figure(figsize=(4, 3))
        plt.plot(rec, prec, lw=1.2)
        plt.scatter(rec[best_idx], prec[best_idx], c="red", s=50,
                    label=f"F1={metrics['f1']:.3f}")
        plt.xlabel("Recall"); plt.ylabel("Precision")
        plt.title(f"PR Curve – {label}\nThreshold = {metrics['threshold']}")
        plt.legend(); plt.grid(True, alpha=0.3); plt.tight_layout(); plt.show()

    return metrics


def optimize_multiple(pipes: dict, X_val, y_val, *, plot_each=False, show_df=True) -> pd.DataFrame:
    """Optimise les seuils pour plusieurs modèles."""
    rows = []
    print("Optimisation des seuils sur jeu de VALIDATION")
    print(f"📊 {len(pipes)} modèles à optimiser...\n" + "-" * 50)

    for name, obj in pipes.items():
        try:
            pipe = joblib.load(obj) if isinstance(obj, (str, Path)) else obj
            metr = optimize_threshold(pipe, X_val, y_val, plot=plot_each, label=name)
            metr["model"] = name
            rows.append(metr)
            print(f"✅ {name:<12}: F1_VAL={metr['f1']:.4f}, seuil={metr['threshold']:.3f}")
        except Exception as e:
            print(f"❌ {name:<12}: Erreur → {e}")

    df = pd.DataFrame(rows).sort_values("f1", ascending=False).reset_index(drop=True)

    if show_df and not df.empty:
        print("\n📋 Résultats triés par F1-score (validation)")
        try:
            display(df.style.format({
                "f1": "{:.4f}", "precision": "{:.4f}",
                "recall": "{:.4f}", "threshold": "{:.3f}"
            }).background_gradient(subset=["f1"], cmap="Greens"))
        except NameError:
            print(df)

        print(f"\n🏆 Meilleur seuil (VALIDATION) : {df.iloc[0]['model']} (F1={df.iloc[0]['f1']:.4f})")

    return df


def save_optimized_thresholds(df: pd.DataFrame, imputation: str, version: str, output_dir: Path):
    """Sauvegarde les seuils optimisés dans un fichier JSON."""
    assert imputation in ["knn", "mice"]
    assert version in ["full", "reduced"]

    path = output_dir / f"optimized_thresholds_{imputation}_{version}.json"
    thresholds_dict = {
        row["model"]: {
            "threshold": row["threshold"],
            "f1": row["f1"],
            "precision": row["precision"],
            "recall": row["recall"]
        }
        for _, row in df.iterrows()
    }

    with open(path, "w") as f:
        json.dump(thresholds_dict, f, indent=2)

    print(f"💾 Seuils optimisés sauvegardés → {path.name}")


def load_optimized_thresholds(imputation: str, version: str, input_dir: Path) -> pd.DataFrame:
    """Charge les seuils optimisés depuis un fichier JSON."""
    assert imputation in ["knn", "mice"]
    assert version in ["full", "reduced"]

    path = input_dir / f"optimized_thresholds_{imputation}_{version}.json"
    if not path.exists():
        raise FileNotFoundError(f"❌ Fichier seuils introuvable : {path}")

    with open(path, "r") as f:
        data = json.load(f)

    df = pd.DataFrame(data).T
    df["model"] = df.index
    df["Imputation"] = imputation.upper()
    df["Version"] = version.upper()
    return df.reset_index(drop=True)


def optimize_all_thresholds(pipelines_dict, splits_dict, output_dir):
    """
    Optimise et sauvegarde les seuils pour toutes les combinaisons KNN/MICE × Full/Reduced.
    
    Args:
        pipelines_dict: Dict des pipelines {key: {model_name: pipeline}}
        splits_dict: Dict des splits avec structure uniforme {key: {"X_val": df, "y_val": series}}
        output_dir: Répertoire de sauvegarde
    
    Note: 
        Supporte maintenant la structure uniforme 6-splits pour tous les types de données.
        Plus besoin de distinction entre structure "full" et "reduced".
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    combinations = [
        ("knn", "full"),
        ("knn", "reduced"),
        ("mice", "full"),
        ("mice", "reduced"),
    ]

    for imput, version in combinations:
        key = f"{imput}_{version}"
        print(f"--- {imput.upper()} {version.upper()} ---")

        pipes = pipelines_dict.get(key)
        if pipes is None:
            print(f"⚠️ Aucun pipeline trouvé pour {key}")
            continue
            
        # ✅ STRUCTURE UNIFORME: Tous les splits utilisent maintenant X_val/y_val
        splits = splits_dict.get(key)
        if splits is None:
            print(f"⚠️ Aucun split trouvé pour {key}")
            continue
            
        val_X = splits["X_val"]
        val_y = splits["y_val"]

        thresholds = optimize_multiple(
            pipes, val_X, val_y, plot_each=False
        )
        save_optimized_thresholds(thresholds, imput, version, output_dir)

# =============================================================================
# 2. ÉVALUATION SUR DONNÉES DE TEST
# =============================================================================

def evaluate_all_models_on_test(pipelines_dict, thresholds_dict, splits_dict, output_dir: Path):
    """Évalue tous les modèles avec leurs seuils sur les jeux de test."""
    output_dir.mkdir(parents=True, exist_ok=True)
    all_results = []

    for key in pipelines_dict:
        print(f"\n🔍 Évaluation : {key.upper()}")
        pipes = pipelines_dict[key]
        thresholds = thresholds_dict[key]

        # Support des splits réduits avec structure différente
        split = splits_dict[key]
        # Logique corrigée pour accéder aux données de test
        if "test" in split and isinstance(split["test"], dict):
            test_X = split["test"]["X"]
            test_y = split["test"]["y"]
        elif "X_test" in split and "y_test" in split:
            test_X = split["X_test"]
            test_y = split["y_test"]
        else:
            raise ValueError(f"Could not find test data in split dictionary for key: {key}")

        for model_name, pipe in pipes.items():
            if model_name not in thresholds:
                print(f"⚠️ Seuil manquant pour {model_name} ({key}) – ignoré.")
                continue

            # Check if the model has predict_proba or decision_function
            if hasattr(pipe, "predict_proba"):
                y_scores = pipe.predict_proba(test_X)[:, 1]
            elif hasattr(pipe, "decision_function"):
                y_scores = pipe.decision_function(test_X)
            else:
                print(f"⚠️ Modèle {model_name} ({key}) ne supporte ni predict_proba ni decision_function – ignoré.")
                continue

            thr = thresholds[model_name]["threshold"]
            y_pred = (y_scores >= thr).astype(int)

            f1 = f1_score(test_y, y_pred)
            precision = precision_score(test_y, y_pred)
            recall = recall_score(test_y, y_pred)
            auc_score = roc_auc_score(test_y, y_scores)

            result = {
                "model": model_name,
                "imputation": key.split("_")[0].upper(),
                "version": key.split("_")[1].upper(),
                "threshold": thr,
                "f1": f1,
                "precision": precision,
                "recall": recall,
                "auc": auc_score
            }
            all_results.append(result)

            # Affichage de la matrice de confusion
            print(f"\n📌 {model_name} ({key}) - F1={f1:.4f}, Précision={precision:.4f}, Rappel={recall:.4f}, AUC={auc_score:.4f}")
            cm = confusion_matrix(test_y, y_pred)
            disp = ConfusionMatrixDisplay(cm)
            disp.plot(cmap="Blues")
            plt.title(f"{model_name} – {key.upper()} – TEST")
            plt.tight_layout()
            plt.gcf().set_size_inches(4, 4)
            plt.show()

    df_results = pd.DataFrame(all_results)
    path = output_dir / "test_results_all_models.csv"
    df_results.to_csv(path, index=False)
    print(f"\n💾 Résultats sauvegardés → {path.name}")
    return df_results


def plot_test_performance(df_results):
    """Affiche un barplot des performances F1 sur TEST."""
    if df_results.empty:
        print("❌ Données vides. Rien à afficher.")
        return

    df_sorted = df_results.sort_values("f1", ascending=True)
    plt.figure(figsize=(8, 4))
    ax = sns.barplot(data=df_sorted, x="f1", y="model", hue="imputation", dodge=True)
    plt.title("F1-score sur TEST (toutes configurations)", fontsize=14)
    plt.xlabel("F1-score"); plt.ylabel("Modèle")
    plt.grid(axis="x", alpha=0.3, linestyle="--")
    plt.legend(title="Imputation")
    plt.tight_layout()
    plt.show()

# =============================================================================
# 3. COURBES ROC DE COMPARAISON
# =============================================================================

def plot_best_roc_curves_comparison(models_dir: Path, figures_dir: Path, splits: dict):
    """Trace les courbes ROC comparées des meilleurs modèles KNN et MICE sur le jeu de validation."""
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

        key = f"{imp}_{version}".lower()

        # Debugging
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

        # Pipeline path construction
        model_specific_dir = models_dir / model.lower() / version.lower()
        pipe_path = model_specific_dir / f"pipeline_{model.lower()}_{imp.lower()}_{version.lower()}.joblib"

        if not pipe_path.exists():
            print(f"❌ Pipeline manquant : {pipe_path.name}")
            continue

        pipe = joblib.load(pipe_path)

        # Access validation data
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

# =============================================================================
# 4. STACKING AVEC ET SANS REFIT (ex-stacking.py)
# =============================================================================

def load_stacking_artefacts(paths):
    """Recharge tous les artefacts nécessaires pour le début du Notebook 03 (Stacking)."""
    print("Rechargement de tous les artefacts pour le Notebook 03 (Stacking)...")
    print("=" * 60)

    # Initialize variables to None
    splits = None
    all_optimized_pipelines = {}
    all_thresholds = {}
    df_all_thr = None
    feature_cols = None

    # Utiliser le chemin fourni
    MODELS_DIR = paths.get("MODELS_DIR")
    if MODELS_DIR is None:
        raise ValueError("Le chemin MODELS_DIR doit être fourni dans le dictionnaire paths")

    # 1. Charger les données splitées
    print("Chargement des données splitées (train, val, test)...")
    try:
        knn_split_dir = MODELS_DIR / "knn"
        mice_split_dir = MODELS_DIR / "mice"

        if not knn_split_dir.exists() or not mice_split_dir.exists():
            raise FileNotFoundError("Les dossiers de split KNN ou MICE n'existent pas.")

        splits = {
            "knn": {
                "X_train": joblib.load(knn_split_dir / "knn_train.pkl")["X"],
                "y_train": joblib.load(knn_split_dir / "knn_train.pkl")["y"],
                "X_val": joblib.load(knn_split_dir / "knn_val.pkl")["X"],
                "y_val": joblib.load(knn_split_dir / "knn_val.pkl")["y"],
                "X_test": joblib.load(knn_split_dir / "knn_test.pkl")["X"],
                "y_test": joblib.load(knn_split_dir / "knn_test.pkl")["y"],
            },
            "mice": {
                "X_train": joblib.load(mice_split_dir / "mice_train.pkl")["X"],
                "y_train": joblib.load(mice_split_dir / "mice_train.pkl")["y"],
                "X_val": joblib.load(mice_split_dir / "mice_val.pkl")["X"],
                "y_val": joblib.load(mice_split_dir / "mice_val.pkl")["y"],
                "X_test": joblib.load(mice_split_dir / "mice_test.pkl")["X"],
                "y_test": joblib.load(mice_split_dir / "mice_test.pkl")["y"],
            }
        }
        print("Données splitées chargées.")

    except Exception as e:
        print(f"Erreur lors du chargement des splits : {e}")

    # 2. Charger tous les pipelines optimisés
    print("Chargement de tous les pipelines optimisés...")
    models_notebook2_dir = MODELS_DIR

    # Rechercher les fichiers de pipeline et threshold
    pipeline_files = list(models_notebook2_dir.glob("pipeline_*.joblib"))
    threshold_files = list(models_notebook2_dir.glob("best_params_*.json"))

    for pipeline_file in pipeline_files:
        # Extraire le nom du modèle et l'imputation du nom de fichier
        filename = pipeline_file.stem
        parts = filename.split("_")
        
        if len(parts) >= 4:
            model_name = parts[1]
            imputation = parts[2] 
            version = parts[3]
            
            dict_key = f"{model_name}_{imputation}_{version}"
            
            try:
                all_optimized_pipelines[dict_key] = joblib.load(pipeline_file)
                print(f"Pipeline chargé pour {dict_key}")
            except Exception as e:
                print(f"Erreur lors du chargement de {pipeline_file}: {e}")

    # 3. Charger les seuils
    for threshold_file in threshold_files:
        filename = threshold_file.stem
        parts = filename.split("_")
        
        if len(parts) >= 5:
            model_name = parts[2]
            imputation = parts[3]
            version = parts[4]
            
            dict_key = f"{model_name}_{imputation}_{version}"
            
            try:
                with open(threshold_file, "r") as f:
                    all_thresholds[dict_key] = json.load(f)
                print(f"Seuils chargés pour {dict_key}")
            except Exception as e:
                print(f"Erreur lors du chargement de {threshold_file}: {e}")

    # 4. Charger le tableau récapitulatif des seuils optimaux
    print("Chargement du tableau des seuils optimaux...")
    thresholds_csv_path = MODELS_DIR / "df_all_thresholds.csv"
    try:
        df_all_thr = pd.read_csv(thresholds_csv_path)
        print("Tableau des seuils optimaux chargé.")
    except Exception as e:
        print(f"Erreur lors du chargement de {thresholds_csv_path} : {e}")

    # 5. Charger les feature columns
    print("Chargement des feature columns...")
    knn_cols_path = MODELS_DIR / "knn" / "columns_knn.pkl"
    mice_cols_path = MODELS_DIR / "mice" / "columns_mice.pkl"

    try:
        feature_cols_knn = joblib.load(knn_cols_path)
        feature_cols_mice = joblib.load(mice_cols_path)
        
        feature_cols = {
            "knn": feature_cols_knn,
            "mice": feature_cols_mice
        }
        print(f"Feature columns chargées : KNN ({len(feature_cols_knn)} cols), MICE ({len(feature_cols_mice)} cols).")

    except Exception as e:
        print(f"Erreur lors du chargement des feature columns : {e}")

    print("Artefacts chargés pour Notebook 03.")
    return splits, all_optimized_pipelines, all_thresholds, df_all_thr, feature_cols


def strip_clf_prefix(params):
    """Supprime le préfixe 'clf__' des paramètres."""
    return {k.replace("clf__", ""): v for k, v in params.items()}


def clean_xgb_params(params):
    """Nettoie les paramètres XGBoost."""
    params_clean = params.copy()
    params_to_remove = ['use_label_encoder', 'eval_metric', 'feature_weights']
    
    for param in params_to_remove:
        if param in params_clean:
            params_clean.pop(param)
    
    return params_clean


def clean_svm_params(params):
    """Nettoie les paramètres SVM et s'assure que probability=True."""
    params_clean = params.copy()
    params_clean['probability'] = True
    return params_clean


def load_best_params_from_saved_files(model_name, imputation="knn", models_dir=None):
    """Charge les meilleurs paramètres depuis les fichiers JSON sauvegardés."""
    try:
        if models_dir is None:
            # Par défaut, chercher dans outputs/modeling/notebook2
            models_dir = Path("outputs/modeling/notebook2")
        else:
            models_dir = Path(models_dir)
        
        # Construire les noms de fichiers possibles
        model_mapping = {
            'gradboost': 'gradboost',
            'gradientboosting': 'gradboost', 
            'xgboost': 'xgboost',
            'randforest': 'randforest',
            'randomforest': 'randforest',
            'svm': 'svm',
            'mlp': 'mlp'
        }
        
        base_name = model_mapping.get(model_name.lower())
        if base_name is None:
            raise ValueError(f"Modèle {model_name} non reconnu")
        
        # Essayer différentes variantes (priorité aux "reduced" car meilleurs résultats)
        possible_files = [
            f"best_params_{base_name}_{imputation}_reduced.json",
            f"best_params_{base_name}_{imputation}_full.json",
            f"best_params_{base_name}_{imputation}.json"
        ]
        
        # Essayer dans plusieurs répertoires
        search_dirs = [
            models_dir,
            models_dir / "notebook2",
            Path("outputs/modeling/notebook2"),
            Path("outputs/modeling")
        ]
        
        file_path = None
        for search_dir in search_dirs:
            if not search_dir.exists():
                continue
            for file_name in possible_files:
                test_path = search_dir / file_name
                if test_path.exists():
                    file_path = test_path
                    break
            if file_path is not None:
                break
        
        if file_path is None:
            print(f"Aucun fichier trouvé pour {model_name}_{imputation}")
            return None
            
        # Charger les paramètres
        with open(file_path, 'r') as f:
            params = json.load(f)
        
        # Nettoyer les paramètres
        params = strip_clf_prefix(params)
        
        if model_name.lower() in ['gradboost', 'xgboost']:
            params = clean_xgb_params(params)
        elif model_name.lower() in ['svm']:
            params = clean_svm_params(params)
        
        print(f"{model_name}_{imputation}: {len(params)} paramètres chargés")
        return params
        
    except Exception as e:
        print(f"Erreur lors du chargement des paramètres pour {model_name}_{imputation}: {e}")
        return None


def create_stacking_models(imputation_method="both", models_dir=None, verbose=True):
    """Crée les modèles de stacking avec les paramètres optimisés sauvegardés."""
    
    if verbose:
        print("=" * 80)
        print(f"CRÉATION DES MODÈLES DE STACKING - {imputation_method.upper()}")
        print("=" * 80)
    
    result = {}
    
    # Définir les méthodes d'imputation à traiter
    if imputation_method.lower() == "knn":
        imputation_methods = ["knn"]
    elif imputation_method.lower() == "mice":
        imputation_methods = ["mice"]
    else:  # "both"
        imputation_methods = ["knn", "mice"]
    
    for imputation in imputation_methods:
        if verbose:
            print(f"\nTraitement {imputation.upper()}...")
            print("-" * 50)
        
        # Chargement des paramètres optimisés
        best_gradboost_params = load_best_params_from_saved_files("gradboost", imputation, models_dir)
        best_mlp_params = load_best_params_from_saved_files("mlp", imputation, models_dir)
        best_randforest_params = load_best_params_from_saved_files("randforest", imputation, models_dir)
        best_svm_params = load_best_params_from_saved_files("svm", imputation, models_dir)
        best_xgboost_params = load_best_params_from_saved_files("xgboost", imputation, models_dir)
        
        # Vérification que tous les paramètres ont été chargés
        loaded_params = {
            f"gradboost_{imputation}": best_gradboost_params,
            f"mlp_{imputation}": best_mlp_params,
            f"randforest_{imputation}": best_randforest_params,
            f"svm_{imputation}": best_svm_params,
            f"xgboost_{imputation}": best_xgboost_params
        }
        
        missing_params = [name for name, params in loaded_params.items() if params is None]
        
        if missing_params:
            print(f"Paramètres manquants pour {imputation}: {missing_params}")
            continue
        
        if verbose:
            print(f"Tous les paramètres {imputation} chargés !")
        
        # Création des modèles avec les meilleurs paramètres
        try:
            gradboost_model = XGBClassifier(**best_gradboost_params)
            mlp_model = MLPClassifier(**best_mlp_params)
            randforest_model = RandomForestClassifier(**best_randforest_params)
            svm_model = SVC(**best_svm_params)
            xgboost_model = XGBClassifier(**best_xgboost_params)
            
            if verbose:
                print(f"Tous les modèles {imputation} créés avec succès !")
            
        except Exception as e:
            print(f"Erreur lors de la création des modèles {imputation} : {e}")
            continue
        
        # Création du stacking classifier
        try:
            estimators = [
                ('gradboost', gradboost_model),
                ('mlp', mlp_model),
                ('randforest', randforest_model),
                ('svm', svm_model),
                ('xgboost', xgboost_model)
            ]
            
            stacking_classifier = StackingClassifier(
                estimators=estimators,
                final_estimator=LogisticRegression(random_state=42),
                cv=5,
                stack_method='predict_proba',
                n_jobs=-1
            )
            
            if verbose:
                print(f"Stacking classifier {imputation} créé avec succès !")
            
        except Exception as e:
            print(f"Erreur lors de la création du stacking {imputation} : {e}")
            continue
        
        # Stockage des résultats
        result[f"stacking_classifier_{imputation}"] = stacking_classifier
        result[f"gradboost_{imputation}"] = gradboost_model
        result[f"mlp_{imputation}"] = mlp_model
        result[f"randforest_{imputation}"] = randforest_model
        result[f"svm_{imputation}"] = svm_model
        result[f"xgboost_{imputation}"] = xgboost_model
        
        if verbose:
            print(f"STACKING {imputation.upper()} PRÊT !")
    
    return result


def generate_mean_proba(pipelines, X):
    """Génère les probabilités moyennes de plusieurs pipelines."""
    probas = []
    for pipeline in pipelines:
        if pipeline is not None:
            if hasattr(pipeline, 'predict_proba'):
                proba = pipeline.predict_proba(X)[:, 1]
            else:
                proba = pipeline.decision_function(X)
            probas.append(proba)
    
    if not probas:
        raise ValueError("Aucun pipeline valide fourni")
    
    return np.mean(probas, axis=0)


def load_pipeline(model_name, imputation_method, version, models_dir=None):
    """Charge un pipeline depuis un fichier."""
    if models_dir is None:
        raise ValueError("models_dir doit être fourni")
    
    filename = f"pipeline_{model_name}_{imputation_method}_{version}.joblib"
    # Fix: Force le chemin passé en paramètre
    filepath = Path(str(models_dir)) / filename
    
    print(f"Debug: Tentative de chargement {filepath}")  # Debug temporaire
    
    if filepath.exists():
        return joblib.load(filepath)
    else:
        print(f"Fichier non trouvé : {filepath}")
        return None


def load_best_pipelines(imputation_method, version="", models_dir=None, model_dir=None):
    """Charge tous les meilleurs pipelines pour une méthode d'imputation donnée."""
    # Compatibilité avec les deux noms de paramètres
    if model_dir is not None:
        models_dir = model_dir
    
    if models_dir is None:
        models_dir = Path("models")
    else:
        models_dir = Path(models_dir)
    
    version_suffix = f"_{version}" if version else ""
    
    models_to_load = ["gradboost", "mlp", "randforest", "svm", "xgboost"]
    pipelines = {}
    
    for model_name in models_to_load:
        # Essayer différents patterns de noms de fichiers
        patterns = [
            f"pipeline_{model_name}_{imputation_method}{version_suffix}.joblib",
            f"best_{model_name}_{imputation_method}{version_suffix}.joblib",
            f"{model_name}_{imputation_method}{version_suffix}.joblib"
        ]
        
        loaded = False
        for pattern in patterns:
            filepath = models_dir / pattern
            if filepath.exists():
                try:
                    pipelines[model_name] = joblib.load(filepath)
                    print(f"✅ {model_name}_{imputation_method} chargé depuis {filepath}")
                    loaded = True
                    break
                except Exception as e:
                    print(f"❌ Erreur lors du chargement de {filepath}: {e}")
        
        if not loaded:
            print(f"⚠️ Pipeline {model_name}_{imputation_method} non trouvé")
    
    return pipelines


def run_stacking_with_refit(X_train, y_train, X_val, y_val, X_test, y_test,
                            imputation_method, models_dir, output_dir=None,
                            model_name_suffix="", create_stacking_func=None,
                            threshold_optimization_func=None,
                            stacking_model_key=None, context_name="notebook3"):
    """Exécute le processus complet de stacking avec refit."""
    log.info(f"Démarrage du Stacking avec Refit - {imputation_method.upper()}")

    if output_dir is None:
        stacking_dir = Path(models_dir) / context_name / "stacking"
    else:
        stacking_dir = Path(output_dir)
    stacking_dir.mkdir(parents=True, exist_ok=True)

    results = {
        'model': None,
        'metrics': {},
        'threshold': None,
        'paths': {}
    }

    try:
        # Création du modèle
        if create_stacking_func is None:
            models_output = create_stacking_models(imputation_method=imputation_method, 
                                                 models_dir=models_dir, verbose=True)
        else:
            models_output = create_stacking_func(imputation_method=imputation_method, verbose=True)

        # Extraction du modèle
        if isinstance(models_output, dict):
            if stacking_model_key is None:
                stacking_model_key = f"stacking_classifier_{imputation_method}"
            
            if stacking_model_key not in models_output:
                raise KeyError(f"Clé '{stacking_model_key}' non trouvée")
            
            stacking_model = models_output[stacking_model_key]
        else:
            stacking_model = models_output

        if stacking_model is None:
            raise ValueError(f"Échec de la création du modèle Stacking pour {imputation_method}")

        log.info(f"Modèle Stacking {imputation_method.upper()} créé.")

        # Entraînement
        stacking_model.fit(X_train, y_train)
        log.info(f"Modèle Stacking {imputation_method.upper()} entraîné.")

        # Prédictions sur validation
        y_proba_val = stacking_model.predict_proba(X_val)[:, 1]

        # Optimisation du seuil
        if threshold_optimization_func is None:
            best_threshold, best_f1_val = _optimize_threshold_internal(y_val, y_proba_val)
        else:
            best_threshold, best_f1_val = threshold_optimization_func(y_val, y_proba_val)

        log.info(f"Seuil optimal {imputation_method.upper()}: {best_threshold:.3f}")
        results['threshold'] = best_threshold

        # Évaluation sur test
        y_proba_test = stacking_model.predict_proba(X_test)[:, 1]
        y_pred_test = (y_proba_test >= best_threshold).astype(int)

        f1_test = f1_score(y_test, y_pred_test)
        precision_test = precision_score(y_test, y_pred_test)
        recall_test = recall_score(y_test, y_pred_test)
        cm_test = confusion_matrix(y_test, y_pred_test)

        results['model'] = stacking_model
        y_pred_val_opt = (y_proba_val >= best_threshold).astype(int)
        results['metrics'] = {
            'f1_score_val': float(best_f1_val),
            'precision_val': float(precision_score(y_val, y_pred_val_opt)),
            'recall_val': float(recall_score(y_val, y_pred_val_opt)),
            'f1_score_test': float(f1_test),
            'precision_test': float(precision_test),
            'recall_test': float(recall_test),
            'confusion_matrix_test': cm_test.tolist()
        }

        # Sauvegarde
        model_filename = f"stacking_{imputation_method}_with_refit{model_name_suffix}.joblib"
        model_path = stacking_dir / model_filename
        joblib.dump(stacking_model, model_path)
        results['paths']['model'] = model_path

        results_filename = f"stacking_with_refit_{imputation_method}{model_name_suffix}.json"
        results_path = stacking_dir / results_filename
        results_to_save = {
            "method": f"stacking_with_refit_{imputation_method}{model_name_suffix}",
            "threshold": float(best_threshold),
            "performance": results['metrics'],
            "predictions": {
                "test_proba": y_proba_test.tolist(),
                "test_pred": y_pred_test.tolist()
            }
        }
        with open(results_path, 'w') as f:
            json.dump(results_to_save, f, indent=4)
        results['paths']['results'] = results_path

    except Exception as e:
        log.error(f"Erreur lors de l'exécution du Stacking: {e}")
        return None

    return results


def _optimize_threshold_internal(y_true, y_proba, metric='f1', thresholds=None):
    """Optimise un seuil de classification pour une métrique donnée."""
    if thresholds is None:
        thresholds = np.linspace(0.2, 0.8, 61)

    best_score = -np.inf
    best_threshold = 0.5

    metric_funcs = {
        'f1': f1_score,
        'precision': precision_score,
        'recall': recall_score
    }

    if metric not in metric_funcs:
        raise ValueError(f"Métrique '{metric}' non supportée.")

    metric_func = metric_funcs[metric]

    for thr in thresholds:
        y_pred = (y_proba >= thr).astype(int)
        try:
            score = metric_func(y_true, y_pred)
            if np.isfinite(score) and score > best_score:
                best_score = score
                best_threshold = thr
        except Exception:
            pass

    return best_threshold, best_score


def run_stacking_no_refit(X_val, y_val, X_test, y_test, imputation_method, models_dir, output_dir=None):
    """Exécute le processus complet de stacking sans refit."""
    print(f"STACKING SANS REFIT - {imputation_method.upper()}")
    print("=" * 80)

    if output_dir is None:
        stacking_dir = Path(models_dir) / "notebook3" / "stacking"
    else:
        stacking_dir = Path(output_dir)
    stacking_dir.mkdir(parents=True, exist_ok=True)

    results_summary = {
        'metrics': {},
        'threshold': None,
        'paths': {}
    }

    try:
        # Chargement des pipelines
        print(f"Chargement des pipelines pour {imputation_method.upper()}...")
        pipelines = [
            load_pipeline('randforest', imputation_method, 'full', models_dir),
            load_pipeline('xgboost', imputation_method, 'full', models_dir),
            load_pipeline('svm', imputation_method, 'full', models_dir),
            load_pipeline('mlp', imputation_method, 'full', models_dir),
            load_pipeline('gradboost', imputation_method, 'full', models_dir)
        ]
        
        valid_pipelines = [p for p in pipelines if p is not None]
        if not valid_pipelines:
            raise RuntimeError("Échec du chargement des pipelines.")
        
        print(f"{len(valid_pipelines)} pipelines chargés avec succès.")

        # Calcul des probabilités moyennes
        print("Calcul des probabilités moyennes...")
        proba_mean_val = generate_mean_proba(valid_pipelines, X_val)
        proba_mean_test = generate_mean_proba(valid_pipelines, X_test)

        # Optimisation du seuil
        print("Optimisation du seuil...")
        thresholds = np.linspace(0.2, 0.8, 61)
        best_f1 = -1
        best_threshold = 0.5

        for thr in thresholds:
            y_pred_val = (proba_mean_val >= thr).astype(int)
            f1 = f1_score(y_val, y_pred_val)
            if np.isfinite(f1) and f1 > best_f1:
                best_f1 = f1
                best_threshold = thr

        print(f"Seuil optimal: {best_threshold:.3f} (F1-val: {best_f1:.4f})")
        results_summary['threshold'] = best_threshold

        # Évaluation sur test
        y_pred_test = (proba_mean_test >= best_threshold).astype(int)
        f1_test = f1_score(y_test, y_pred_test)
        precision_test = precision_score(y_test, y_pred_test)
        recall_test = recall_score(y_test, y_pred_test)

        print(f"\nRÉSULTATS STACKING SANS REFIT {imputation_method.upper()}:")
        print(f"   F1-score (test) : {f1_test:.4f}")
        print(f"   Précision (test): {precision_test:.4f}")
        print(f"   Rappel (test)   : {recall_test:.4f}")

        results_summary['metrics'] = {
            'f1_score_val': float(best_f1),
            'f1_score_test': float(f1_test),
            'precision_test': float(precision_test),
            'recall_test': float(recall_test)
        }

        # Sauvegarde
        stacking_no_refit_results = {
            "method": f"stacking_no_refit_{imputation_method}_full",
            "threshold": float(best_threshold),
            "performance": results_summary['metrics'],
            "predictions": {
                "validation": proba_mean_val.tolist(),
                "test": proba_mean_test.tolist()
            }
        }

        results_filename = f"stacking_no_refit_{imputation_method}_full.json"
        results_path = stacking_dir / results_filename

        with open(results_path, 'w') as f:
            json.dump(stacking_no_refit_results, f, indent=2)

        print(f"Résultats sauvegardés dans: {results_path}")
        results_summary['paths']['results'] = results_path

    except Exception as e:
        print(f"Erreur lors du stacking sans refit - {imputation_method.upper()}: {e}")
        return None

    print(f"\nSTACKING SANS REFIT {imputation_method.upper()} TERMINÉ !")
    return results_summary

# =============================================================================
# 5. ÉVALUATION DES PRÉDICTIONS
# =============================================================================

def evaluate_predictions(y_true, y_pred, label=""):
    """Affiche les métriques principales, le rapport de classification et une jolie matrice de confusion."""
    # Calcul des métriques
    f1 = f1_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)

    print(f"\n=== ÉVALUATION : {label.upper()} ===")
    print(f"F1-score   : {f1:.4f}")
    print(f"Précision  : {precision:.4f}")
    print(f"Rappel     : {recall:.4f}")
    print("\n--- Rapport détaillé ---")
    print(classification_report(y_true, y_pred, digits=4))

    # Matrice de confusion
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(3.5, 3))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=["Prédit: No-AD", "Prédit: AD"],
                yticklabels=["Réel: No-AD", "Réel: AD"])
    plt.title(f"Matrice de Confusion – {label.upper()}", fontsize=11)
    plt.xlabel("Valeurs Prédites")
    plt.ylabel("Valeurs Réelles")
    plt.tight_layout()
    plt.grid(False)
    plt.show()


def evaluate_from_probabilities(y_true, y_proba, threshold=0.5, label=""):
    """Évalue les performances à partir des probabilités prédites avec un seuil donné."""
    y_pred = (np.array(y_proba) >= threshold).astype(int)

    # Métriques
    f1 = f1_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)

    print(f"\n=== ÉVALUATION : {label.upper()} ===")
    print(f"Seuil appliqué : {threshold:.3f}")
    print(f"F1-score       : {f1:.4f}")
    print(f"Précision      : {precision:.4f}")
    print(f"Rappel         : {recall:.4f}")
    print("\n--- Rapport détaillé ---")
    print(classification_report(y_true, y_pred, digits=4))

    # Matrice de confusion
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(3.5, 3))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=["Prédit: No-AD", "Prédit: AD"],
                yticklabels=["Réel: No-AD", "Réel: AD"])
    plt.title(f"Matrice de Confusion – {label.upper()}", fontsize=11)
    plt.xlabel("Valeurs Prédites")
    plt.ylabel("Valeurs Réelles")
    plt.tight_layout()
    plt.grid(False)
    plt.show()

    return {
        "threshold": threshold,
        "f1": f1,
        "precision": precision,
        "recall": recall
    }


def analyze_model_performance(result_dict, X_test, y_test, method_name):
    """Analyse les performances d'un modèle à partir du dictionnaire de résultats."""
    if result_dict is None:
        print(f"❌ Aucun résultat disponible pour {method_name}.")
        return

    model = result_dict['model']
    threshold = result_dict['threshold']
    
    if model is None:
        print(f"❌ Aucun modèle disponible dans les résultats pour {method_name}.")
        return

    # Prédictions
    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= threshold).astype(int)

    # Matrice de confusion
    cm = confusion_matrix(y_test, y_pred)
    print(f"\n--- Matrice de Confusion - {method_name} ---")
    print(cm)

    # Classification Report
    print(f"\n--- Rapport de Classification - {method_name} ---")
    print(classification_report(y_test, y_pred))

    # Visualisation de la matrice de confusion
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", 
                xticklabels=["Non-admis", "Admis"], 
                yticklabels=["Non-admis (Réel)", "Admis (Réel)"])
    plt.title(f'Matrice de Confusion - {method_name}\n(Seuil: {threshold:.3f}, F1-test: {result_dict["metrics"]["f1_score_test"]:.4f})')
    plt.xlabel('Prédit')
    plt.ylabel('Réel')
    plt.tight_layout()
    plt.show()

    # Afficher les métriques déjà calculées
    print(f"\n--- Métriques Test (Seuil Optimal {threshold:.3f}) - {method_name} ---")
    metrics = result_dict['metrics']
    print(f"F1-score (test): {metrics['f1_score_test']:.4f}")
    print(f"Précision (test): {metrics['precision_test']:.4f}")
    print(f"Rappel (test): {metrics['recall_test']:.4f}")

# =============================================================================
# 6. TABLEAUX DE COMPARAISON
# =============================================================================

def build_comparison_table(json_results_paths, model_details):
    """Construit un DataFrame comparatif à partir de fichiers JSON de résultats."""
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
# 7. ANALYSE D'IMPORTANCE DES VARIABLES
# =============================================================================

def get_correct_feature_names(feature_cols_dict, imputation_method, X_data):
    """
    Récupère les noms de features corrects pour l'analyse d'importance.
    
    Args:
        feature_cols_dict: Dictionnaire avec clés 'knn' et 'mice'
        imputation_method: 'knn' ou 'mice'
        X_data: Données features pour vérifier la compatibilité
    
    Returns:
        List: Noms de features appropriés
    """
    if isinstance(feature_cols_dict, dict):
        if imputation_method in feature_cols_dict:
            feature_names = feature_cols_dict[imputation_method]
        else:
            log.warning(f"Méthode '{imputation_method}' non trouvée dans feature_cols_dict")
            feature_names = None
    else:
        feature_names = feature_cols_dict
    
    # Vérifier la compatibilité
    if feature_names is not None and len(feature_names) != X_data.shape[1]:
        log.warning(f"Incompatibilité: {len(feature_names)} noms vs {X_data.shape[1]} features")
        log.warning("Génération de noms génériques...")
        feature_names = [f"feature_{i:03d}" for i in range(X_data.shape[1])]
    elif feature_names is None:
        feature_names = [f"feature_{i:03d}" for i in range(X_data.shape[1])]
    
    return feature_names

def analyze_feature_importance(model, X_train, y_train, X_eval, y_eval, feature_names,
                               method='all', cv_folds=5, n_repeats_perm=20,
                               output_dir=None, model_name="model", save_results=True,
                               scoring='f1'):
    """
    Analyse l'importance des variables pour un modèle entraîné en utilisant RFECV,
    Permutation Importance et SHAP (si disponible) avec gestion d'erreurs robuste.
    """
    # Gestion robuste des feature_names
    if feature_names is None:
        feature_names = [f"feature_{i}" for i in range(X_train.shape[1])]
    elif isinstance(feature_names, dict):
        # Si un dictionnaire est passé, montrer les clés disponibles et lever une erreur claire
        available_keys = list(feature_names.keys())
        raise ValueError(f"feature_names doit être une liste, pas un dictionnaire. "
                         f"Clés disponibles: {available_keys}. "
                         f"Utilisez feature_names['{available_keys[0]}'] pour sélectionner la bonne méthode.")
    else:
        feature_names = list(feature_names)
    
    # Vérification finale de compatibilité
    if len(feature_names) != X_train.shape[1]:
        log.warning(f"Incompatibilité détectée: {len(feature_names)} noms fournis pour {X_train.shape[1]} features")
        log.warning("Génération automatique de noms de features...")
        feature_names = [f"feature_{i:03d}" for i in range(X_train.shape[1])]

    results = {
        'model_name': model_name,
        'methods_applied': [],
        'rfecv': None,
        'permutation': None,
        'shap': None
    }

    # Déterminer les méthodes à exécuter
    methods_to_run = method
    if method == 'all':
        methods_to_run = ['rfecv', 'permutation', 'shap']
    if isinstance(methods_to_run, str):
        methods_to_run = [methods_to_run]

    # === RFECV ===
    if 'rfecv' in methods_to_run:
        try:
            cv_splitter = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)

            # Définir un getter d'importance compatible avec les versions récentes de scikit‑learn
            def custom_importance_getter(estimator):
                if hasattr(estimator, 'feature_importances_'):
                    return estimator.feature_importances_
                elif hasattr(estimator, 'coef_'):
                    # Pour les modèles multi‑classes, on moyenne les coefficients absolus
                    coefs = np.array(estimator.coef_)
                    return np.mean(np.abs(coefs), axis=0)
                else:
                    return np.ones(estimator.n_features_in_)

            # Choisir si l'on doit passer un importance_getter explicite
            if hasattr(model, 'feature_importances_') or hasattr(model, 'coef_'):
                rfecv_selector = RFECV(estimator=model, step=1, cv=cv_splitter,
                                       scoring=scoring, n_jobs=-1)
            else:
                rfecv_selector = RFECV(estimator=model, step=1, cv=cv_splitter,
                                       scoring=scoring, n_jobs=-1,
                                       importance_getter=custom_importance_getter)

            rfecv_selector.fit(X_train, y_train)

            # Calcul du meilleur score en fonction des attributs disponibles
            best_score = None
            if hasattr(rfecv_selector, 'best_score_'):
                best_score = float(rfecv_selector.best_score_)
            elif hasattr(rfecv_selector, 'cv_results_'):
                idx = np.argmax(rfecv_selector.cv_results_['mean_test_score'])
                best_score = float(rfecv_selector.cv_results_['mean_test_score'][idx])

            # Vérification de la compatibilité des dimensions pour RFECV
            if len(feature_names) != rfecv_selector.support_.shape[0]:
                log.warning(f"Incompatibilité dimensions RFECV: {len(feature_names)} noms vs {rfecv_selector.support_.shape[0]} features")
                # Ajuster la liste des noms de features
                if len(feature_names) > rfecv_selector.support_.shape[0]:
                    adjusted_names = feature_names[:rfecv_selector.support_.shape[0]]
                else:
                    adjusted_names = feature_names + [f"feature_{i}" for i in range(len(feature_names), rfecv_selector.support_.shape[0])]
            else:
                adjusted_names = feature_names

            results['rfecv'] = {
                'n_features_optimal': int(rfecv_selector.n_features_),
                'best_score': best_score,
                'support': rfecv_selector.support_.tolist(),
                'ranking': rfecv_selector.ranking_.tolist(),
                'grid_scores_mean': rfecv_selector.cv_results_['mean_test_score'].tolist(),
                'grid_scores_std': rfecv_selector.cv_results_['std_test_score'].tolist(),
                'selected_features': [adjusted_names[i] for i, s in enumerate(rfecv_selector.support_) if s]
            }
            results['methods_applied'].append('rfecv')
            log.info(f"RFECV réussi pour {model_name}")
        except Exception as e:
            log.error(f"Erreur lors de RFECV pour {model_name}: {e}")

    # === Importance par permutation ===
    if 'permutation' in methods_to_run:
        try:
            # Vérifier la compatibilité des dimensions
            model_features_expected = getattr(model, 'n_features_in_', X_eval.shape[1])
            
            if X_eval.shape[1] != model_features_expected:
                log.warning(f"Dimensions incompatibles: X_eval a {X_eval.shape[1]} features, "
                             f"mais le modèle attend {model_features_expected} features")
                
                # Essayer de redimensionner X_eval si possible
                if X_eval.shape[1] > model_features_expected:
                    # Prendre les premières features
                    X_eval_adjusted = X_eval[:, :model_features_expected]
                    tmp_names = feature_names[:model_features_expected] if len(feature_names) >= model_features_expected else feature_names
                    log.info(f"X_eval redimensionné de {X_eval.shape[1]} à {X_eval_adjusted.shape[1]} features")
                else:
                    # Ajouter des features factices si nécessaire
                    padding = np.zeros((X_eval.shape[0], model_features_expected - X_eval.shape[1]))
                    X_eval_adjusted = np.hstack([X_eval, padding])
                    tmp_names = feature_names + [f"feature_{i}" for i in range(len(feature_names), model_features_expected)]
                    log.info(f"X_eval complété de {X_eval.shape[1]} à {X_eval_adjusted.shape[1]} features")
            else:
                X_eval_adjusted = X_eval
                tmp_names = feature_names

            # S'assurer que tmp_names a la bonne longueur
            if len(tmp_names) != X_eval_adjusted.shape[1]:
                tmp_names = tmp_names[:X_eval_adjusted.shape[1]] if len(tmp_names) > X_eval_adjusted.shape[1] else tmp_names + [f"feature_{i}" for i in range(len(tmp_names), X_eval_adjusted.shape[1])]

            perm_result = permutation_importance(model, X_eval_adjusted, y_eval,
                                                 n_repeats=n_repeats_perm,
                                                 random_state=42, n_jobs=-1,
                                                 scoring=scoring)

            df_perm_importance = pd.DataFrame({
                'feature': tmp_names,
                'importance_mean': perm_result.importances_mean,
                'importance_std': perm_result.importances_std
            }).sort_values(by='importance_mean', ascending=False)

            results['permutation'] = {
                'dataframe': df_perm_importance.to_dict(orient='list'),
                'top_20_features': df_perm_importance.head(20)['feature'].tolist()
            }
            results['methods_applied'].append('permutation')
            log.info(f"Permutation importance calculée avec succès pour {model_name}")
        except Exception as e:
            log.error(f"Erreur lors de la permutation pour {model_name}: {e}")
            # Ajouter un résultat vide pour éviter les erreurs dans les graphiques
            results['permutation'] = {
                'dataframe': {
                    'feature': feature_names[:10] if len(feature_names) >= 10 else feature_names,
                    'importance_mean': [0.0] * min(10, len(feature_names)),
                    'importance_std': [0.0] * min(10, len(feature_names))
                },
                'top_20_features': feature_names[:10] if len(feature_names) >= 10 else feature_names
            }

    # === SHAP ===
    if 'shap' in methods_to_run:
        try:
            import shap

            # Vérifier la compatibilité des dimensions pour SHAP aussi
            if X_eval.shape[1] != model.n_features_in_:
                if X_eval.shape[1] > model.n_features_in_:
                    X_sample = X_eval[:min(100, X_eval.shape[0]), :model.n_features_in_]
                else:
                    padding = np.zeros((min(100, X_eval.shape[0]), model.n_features_in_ - X_eval.shape[1]))
                    X_sample = np.hstack([X_eval[:min(100, X_eval.shape[0])], padding])
            else:
                X_sample = X_eval[:min(100, X_eval.shape[0])]

            shap_values_for_plot = None
            # Si modèle d'arbres avec predict_proba
            if hasattr(model, 'predict_proba') and hasattr(model, 'estimators_'):
                try:
                    # Essayer d'abord avec predict_proba
                    explainer = shap.TreeExplainer(model, model_output="predict_proba")
                    shap_values = explainer.shap_values(X_sample)
                    # shap_values est une liste (une par classe) : on prend la classe positive s'il y en a plusieurs
                    if isinstance(shap_values, list):
                        shap_values_for_plot = shap_values[1] if len(shap_values) > 1 else shap_values[0]
                    else:
                        shap_values_for_plot = shap_values
                except Exception as e:
                    log.warning(f"Erreur avec predict_proba, essai avec raw: {e}")
                    # Fallback vers raw
                    explainer = shap.TreeExplainer(model, model_output="raw")
                    shap_values = explainer.shap_values(X_sample)
                    if isinstance(shap_values, list):
                        shap_values_for_plot = shap_values[1] if len(shap_values) > 1 else shap_values[0]
                    else:
                        shap_values_for_plot = shap_values
            else:
                # Explainer générique pour les autres modèles
                explainer = shap.Explainer(model, X_sample)
                shap_values = explainer(X_sample)
                # shap_values.values : (n_samples, n_features, n_outputs) éventuel
                if hasattr(shap_values, 'values'):
                    sv = shap_values.values
                    if sv.ndim == 3:
                        shap_values_for_plot = sv[:, :, 1] if sv.shape[2] > 1 else sv[:, :, 0]
                    else:
                        shap_values_for_plot = sv
                else:
                    shap_values_for_plot = shap_values

            results['shap'] = {
                'explainer_type': type(explainer).__name__,
                'n_samples_used': X_sample.shape[0],
            }
            results['methods_applied'].append('shap')
            log.info(f"Analyse SHAP réussie pour {model_name}")

            # Ici, on pourrait générer et sauvegarder des graphiques si output_dir est fourni
        except ImportError:
            log.warning("Le module SHAP n'est pas installé ; l'analyse SHAP est ignorée.")
        except Exception as e:
            log.error(f"Erreur lors de l'analyse SHAP pour {model_name}: {e}")
            # Ajouter un résultat vide pour éviter les erreurs dans les graphiques
            results['shap'] = {
                'explainer_type': 'Error',
                'n_samples_used': 0,
            }

    return results

# =============================================================================
# 8. OPTIMISATION DE SEUILS POUR STACKING (ex-stacking_threshold_optimizer.py)
# =============================================================================

def optimize_stacking_thresholds_with_trained_models(stacking_knn, stacking_mice, 
                                                   X_val_knn, y_val_knn, X_val_mice, y_val_mice,
                                                   optimization_method="f1", verbose=True):
    """
    Optimise les seuils pour des modèles de stacking déjà entraînés.
    
    Parameters:
    -----------
    stacking_knn : modèle de stacking KNN déjà entraîné
    stacking_mice : modèle de stacking MICE déjà entraîné  
    X_val_knn, y_val_knn : données de validation KNN
    X_val_mice, y_val_mice : données de validation MICE
    optimization_method : str, méthode d'optimisation ("f1", "precision", "recall")
    verbose : bool, affichage détaillé
    
    Returns:
    --------
    dict: Dictionnaire contenant les résultats d'optimisation pour KNN et MICE
    """
    
    if verbose:
        print("🎯 Optimisation des seuils avec modèles pré-entraînés")
        print("=" * 60)
    
    results = {}
    
    # Données pour chaque modèle
    models_data = {
        'knn': (stacking_knn, X_val_knn, y_val_knn),
        'mice': (stacking_mice, X_val_mice, y_val_mice)
    }
    
    for imputation_name, (model, X_val, y_val) in models_data.items():
        if verbose:
            print(f"\n--- Optimisation pour {imputation_name.upper()} ---")
        
        try:
            # Prédictions de probabilité
            y_proba = model.predict_proba(X_val)[:, 1]
            
            # Optimisation du seuil
            best_threshold, best_score = _optimize_threshold_internal(
                y_val, y_proba, metric=optimization_method
            )
            
            # Prédictions finales avec seuil optimal
            y_pred_opt = (y_proba >= best_threshold).astype(int)
            
            # Calcul des métriques
            f1_opt = f1_score(y_val, y_pred_opt)
            precision_opt = precision_score(y_val, y_pred_opt)
            recall_opt = recall_score(y_val, y_pred_opt)
            
            results[imputation_name] = {
                'model': model,
                'threshold': float(best_threshold),
                f'{optimization_method}_score': float(best_score),
                'f1_score': float(f1_opt),
                'precision': float(precision_opt),
                'recall': float(recall_opt),
                'y_proba': y_proba,
                'y_pred_opt': y_pred_opt
            }
            
            if verbose:
                print(f"✅ Seuil optimal : {best_threshold:.3f}")
                print(f"   F1-score : {f1_opt:.4f}")
                print(f"   Précision : {precision_opt:.4f}")
                print(f"   Rappel : {recall_opt:.4f}")
                
        except Exception as e:
            if verbose:
                print(f"❌ Erreur pour {imputation_name}: {e}")
            results[imputation_name] = None
    
    return results


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
    dict: Dictionnaire contenant les résultats d'optimisation pour KNN et MICE
    """
    
    if verbose:
        print("🎯 Optimisation des seuils pour Stacking")
        print("=" * 50)
    
    results = {}
    
    # Données pour chaque imputation
    datasets = {
        'knn': (X_train_knn, X_val_knn, y_train_knn, y_val_knn),
        'mice': (X_train_mice, X_val_mice, y_train_mice, y_val_mice)
    }
    
    for imputation_name, (X_train, X_val, y_train, y_val) in datasets.items():
        if verbose:
            print(f"\n--- Optimisation pour {imputation_name.upper()} ---")
        
        try:
            # Créer le modèle de stacking pour cette imputation
            stacking_models = create_stacking_models(
                imputation_method=imputation_name, 
                models_dir=None,  # Sera géré dans la fonction
                verbose=False
            )
            
            stacking_classifier = stacking_models.get(f"stacking_classifier_{imputation_name}")
            
            if stacking_classifier is None:
                if verbose:
                    print(f"❌ Impossible de créer le modèle de stacking pour {imputation_name}")
                continue
            
            # Entraîner le modèle
            stacking_classifier.fit(X_train, y_train)
            
            # Prédire les probabilités sur validation
            y_proba_val = stacking_classifier.predict_proba(X_val)[:, 1]
            
            # Optimiser le seuil
            best_threshold, best_score = _optimize_threshold_internal(
                y_val, y_proba_val, metric=optimization_method
            )
            
            # Calculer toutes les métriques au seuil optimal
            y_pred_val_optimal = (y_proba_val >= best_threshold).astype(int)
            
            metrics = {
                'threshold': float(best_threshold),
                'f1': float(f1_score(y_val, y_pred_val_optimal)),
                'precision': float(precision_score(y_val, y_pred_val_optimal)),
                'recall': float(recall_score(y_val, y_pred_val_optimal)),
                'best_score': float(best_score)
            }
            
            results[imputation_name] = {
                'model': stacking_classifier,
                'metrics': metrics,
                'probabilities': y_proba_val.tolist()
            }
            
            if verbose:
                print(f"✅ Seuil optimal: {best_threshold:.3f}")
                print(f"   F1-score: {metrics['f1']:.4f}")
                print(f"   Précision: {metrics['precision']:.4f}")
                print(f"   Rappel: {metrics['recall']:.4f}")
        
        except Exception as e:
            if verbose:
                print(f"❌ Erreur lors de l'optimisation pour {imputation_name}: {e}")
            continue
    
    if verbose:
        print(f"\n Optimisation terminée pour {len(results)} configurations")
    
    return results