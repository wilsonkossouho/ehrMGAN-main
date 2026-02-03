"""
utils/utils.py
Fonctions utilitaires de base pour le preprocessing eICU
"""

import pandas as pd
import numpy as np
from pathlib import Path


def dataframe_from_csv(filepath, index_col=None, **kwargs):
    """
    Charger un CSV/CSV.GZ en DataFrame avec gestion de compression automatique

    Args:
        filepath: Chemin vers le fichier CSV
        index_col: Colonne à utiliser comme index
        **kwargs: Arguments supplémentaires pour pd.read_csv

    Returns:
        pd.DataFrame
    """
    filepath = Path(filepath)

    # Détection automatique de la compression
    if filepath.suffix == '.gz':
        compression = 'gzip'
    else:
        compression = None

    print(f"  📂 Chargement de {filepath.name}...")

    df = pd.read_csv(
        filepath,
        compression=compression,
        index_col=index_col,
        **kwargs
    )

    print(f"     ✅ {len(df):,} lignes chargées")

    return df


def save_pickle(obj, filepath):
    """
    Sauvegarder un objet en pickle

    Args:
        obj: Objet à sauvegarder
        filepath: Chemin de destination
    """
    import pickle

    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    with open(filepath, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

    print(f"  💾 Sauvegardé : {filepath}")


def load_pickle(filepath):
    """
    Charger un objet depuis pickle

    Args:
        filepath: Chemin du fichier

    Returns:
        Objet chargé
    """
    import pickle

    with open(filepath, 'rb') as f:
        obj = pickle.load(f)

    print(f"  📂 Chargé : {filepath}")
    return obj


def print_stats(df, name="DataFrame"):
    """
    Afficher des statistiques de base sur un DataFrame

    Args:
        df: DataFrame à analyser
        name: Nom du DataFrame pour l'affichage
    """
    print(f"\n📊 Stats pour {name}:")
    print(f"  • Lignes : {len(df):,}")
    print(f"  • Colonnes : {len(df.columns)}")
    print(f"  • Valeurs manquantes : {df.isnull().sum().sum():,}")
    print(f"  • Mémoire : {df.memory_usage(deep=True).sum() / 1024 ** 2:.2f} MB")