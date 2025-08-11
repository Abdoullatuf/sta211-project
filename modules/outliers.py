# modules/preprocessing/outliers.py


import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Optional, Union

def detect_and_remove_outliers(
    df: pd.DataFrame,
    columns: List[str],
    method: str = 'iqr',
    iqr_multiplier: float = 1.5,
    remove: bool = True,
    verbose: bool = True,
    save_path: Optional[Union[str, Path]] = None
) -> pd.DataFrame:
    """
    Détecte (et optionnellement supprime) les outliers selon la méthode IQR.

    Args:
        df (pd.DataFrame): Données d’entrée.
        columns (List[str]): Colonnes à analyser.
        method (str): Méthode à utiliser ('iqr' uniquement pour l’instant).
        iqr_multiplier (float): Coefficient de l’IQR.
        remove (bool): Si True, supprime les lignes outliers. Sinon, retourne le DataFrame d’origine.
        verbose (bool): Affiche les statistiques si True.
        save_path (str or Path): Fichier de sortie facultatif (.csv ou .parquet).

    Returns:
        pd.DataFrame: DataFrame filtré ou original selon `remove`.
    """
    if method != 'iqr':
        raise NotImplementedError("Seule la méthode 'iqr' est implémentée.")

    df = df.copy()
    initial_shape = df.shape
    mask = pd.Series(True, index=df.index)

    for col in columns:
        if col not in df.columns:
            continue

        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - iqr_multiplier * iqr
        upper_bound = q3 + iqr_multiplier * iqr
        col_mask = df[col].between(lower_bound, upper_bound)

        if verbose:
            n_outliers = (~col_mask).sum()
            print(f"📉 {col} : {n_outliers} outliers détectés")

        mask &= col_mask

    df_result = df[mask] if remove else df

    if verbose and remove:
        print(f"\n✅ Total supprimé : {initial_shape[0] - df_result.shape[0]} lignes")
        print(f"🔢 Dimensions finales : {df_result.shape}")

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        if save_path.suffix == ".csv":
            df_result.to_csv(save_path, index=False)
        elif save_path.suffix in [".parquet", ".pq"]:
            df_result.to_parquet(save_path, index=False)
        else:
            raise ValueError("❌ Format non supporté : utilisez .csv ou .parquet")

        if verbose:
            print(f"💾 Données sauvegardées : {save_path}")

    return df_result
