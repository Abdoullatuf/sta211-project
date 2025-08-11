import pandas as pd
import json
from pathlib import Path

def build_comparison_table(json_results_paths, model_details):
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

# --- Exemple d'utilisation avec tous les modèles ---
# json_paths = [
#     MODELS_DIR / "notebook3" / "stacking" / "stacking_no_refit_knn_full.json",
#     MODELS_DIR / "notebook3" / "stacking" / "stacking_no_refit_mice_full.json",
#     MODELS_DIR / "notebook3" / "stacking" / "stacking_with_refit_knn.json", # Généré par run_stacking_with_refit
#     MODELS_DIR / "notebook3" / "stacking" / "stacking_with_refit_mice.json", # Généré par run_stacking_with_refit
#     # Ajouter d'autres chemins si nécessaire (Random Forest, XGBoost, réduits, etc.)
#     # Ex: MODELS_DIR / "notebook2" / "randforest_knn_full.json", # Si ces fichiers existent et ont le bon format
# ]

# details = {
#     "stacking_no_refit_knn_full.json": {"Nom Affiché": "Stacking sans refit KNN", "Type": "Complet", "Imputation": "KNN"},
#     "stacking_no_refit_mice_full.json": {"Nom Affiché": "Stacking sans refit MICE", "Type": "Complet", "Imputation": "MICE"},
#     "stacking_with_refit_knn.json": {"Nom Affiché": "Stacking avec refit KNN", "Type": "Complet", "Imputation": "KNN"},
#     "stacking_with_refit_mice.json": {"Nom Affiché": "Stacking avec refit MICE", "Type": "Complet", "Imputation": "MICE"},
#     # Ajouter les détails pour les autres modèles
#     # "randforest_knn_full.json": {"Nom Affiché": "Random Forest KNN", "Type": "Complet", "Imputation": "KNN"},
# }

# df_comparison = build_comparison_table(json_paths, details)
# print(df_comparison.to_string(index=False))
