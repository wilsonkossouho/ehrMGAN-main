"""
utils/pat_utils.py
Fonctions de filtrage et transformation pour les données patients eICU
"""

import pandas as pd
import numpy as np


def filter_patients_on_age(pats, min_age=15, max_age=89):
    """
    Filtrer les patients selon l'âge

    Args:
        pats: DataFrame des patients
        min_age: Âge minimum
        max_age: Âge maximum

    Returns:
        DataFrame filtré
    """
    print(f"  🔍 Filtrage par âge ({min_age}-{max_age} ans)...")
    initial_count = len(pats)

    # Gérer les valeurs spéciales dans eICU
    # Age peut être: nombre, '', '> 89'
    pats = pats[pats['age'] != ''].copy()

    # Remplacer '> 89' par 90 pour le filtrage
    pats['age_numeric'] = pats['age'].replace('> 89', '90').astype(float)

    # Filtrer
    pats = pats[
        (pats['age_numeric'] >= min_age) &
        (pats['age_numeric'] <= max_age)
        ]

    print(f"     ✅ {initial_count:,} → {len(pats):,} patients")

    return pats


def filter_one_unit_stay(pats):
    """
    Garder seulement les patients avec un seul séjour ICU
    (première admission seulement)

    Args:
        pats: DataFrame des patients

    Returns:
        DataFrame filtré
    """
    print(f"  🔍 Filtrage : un séjour par patient...")
    initial_count = len(pats)

    # Grouper par patienthealthsystemstayid et garder le premier séjour ICU
    # (le plus petit unitvisitnumber)
    pats = pats.sort_values(['patienthealthsystemstayid', 'unitvisitnumber'])
    pats = pats.groupby('patienthealthsystemstayid').first().reset_index()

    print(f"     ✅ {initial_count:,} → {len(pats):,} patients")

    return pats


def filter_max_hours(pats, max_hours=24, thres=240):
    """
    Filtrer selon la durée du séjour ICU

    Args:
        pats: DataFrame des patients
        max_hours: Nombre d'heures minimum (en heures)
        thres: Seuil en minutes (240 min = 4h minimum pour avoir données)

    Returns:
        DataFrame filtré
    """
    print(f"  🔍 Filtrage : séjour >= {max_hours}h...")
    initial_count = len(pats)

    # unitdischargeoffset est en minutes
    min_offset_minutes = max_hours * 60

    pats = pats[
        (pats['unitdischargeoffset'] >= min_offset_minutes) &
        (pats['unitdischargeoffset'] >= thres)
        ]

    print(f"     ✅ {initial_count:,} → {len(pats):,} patients")

    return pats


def filter_patients_on_columns(pats):
    """
    Filtrer les patients avec des valeurs manquantes critiques

    Args:
        pats: DataFrame des patients

    Returns:
        DataFrame filtré
    """
    print(f"  🔍 Filtrage : colonnes critiques non-nulles...")
    initial_count = len(pats)

    # Colonnes qui ne doivent pas être nulles
    critical_columns = [
        'patientunitstayid',
        'age',
        'gender',
        'ethnicity',
        'unitdischargeoffset',
        'unitdischargestatus',
        'hospitaldischargestatus'
    ]

    # Filtrer
    for col in critical_columns:
        if col in pats.columns:
            pats = pats[pats[col].notna()]
            pats = pats[pats[col] != '']

    print(f"     ✅ {initial_count:,} → {len(pats):,} patients")

    return pats


def transform_gender(gender_series):
    """
    Transformer la colonne gender en valeurs numériques

    Args:
        gender_series: pd.Series contenant 'Male', 'Female', etc.

    Returns:
        pd.Series avec valeurs numériques
    """
    print(f"  🔄 Transformation : gender...")

    gender_map = {
        'Male': 1,
        'Female': 0,
        'Unknown': -1,
        '': -1
    }

    return gender_series.map(gender_map).fillna(-1).astype(int)


def transform_ethnicity(ethnicity_series):
    """
    Transformer la colonne ethnicity en valeurs numériques

    Args:
        ethnicity_series: pd.Series

    Returns:
        pd.Series avec valeurs numériques
    """
    print(f"  🔄 Transformation : ethnicity...")

    ethnicity_map = {
        'Caucasian': 0,
        'African American': 1,
        'Hispanic': 2,
        'Asian': 3,
        'Native American': 4,
        'Other/Unknown': 5,
        '': 5
    }

    return ethnicity_series.map(ethnicity_map).fillna(5).astype(int)


def transform_hospital_discharge_status(status_series):
    """
    Transformer hospitaldischargestatus en valeurs numériques

    Args:
        status_series: pd.Series

    Returns:
        pd.Series avec valeurs numériques
    """
    print(f"  🔄 Transformation : hospital discharge status...")

    status_map = {
        'Alive': 0,
        'Expired': 1,
        '': -1
    }

    return status_series.map(status_map).fillna(-1).astype(int)


def transform_unit_discharge_status(status_series):
    """
    Transformer unitdischargestatus en valeurs numériques

    Args:
        status_series: pd.Series

    Returns:
        pd.Series avec valeurs numériques
    """
    print(f"  🔄 Transformation : unit discharge status...")

    status_map = {
        'Alive': 0,
        'Expired': 1,
        'Stepdown': 2,
        '': -1
    }

    return status_series.map(status_map).fillna(-1).astype(int)


def transform_dx_into_id(pats):
    """
    Transformer les diagnostics en IDs numériques

    Args:
        pats: DataFrame des patients

    Returns:
        DataFrame modifié
    """
    print(f"  🔄 Transformation : diagnostics...")

    # Si la colonne apacheadmissiondx existe, la garder
    if 'apacheadmissiondx' in pats.columns:
        # Créer un mapping des diagnostics vers des IDs
        unique_dx = pats['apacheadmissiondx'].unique()
        dx_map = {dx: idx for idx, dx in enumerate(unique_dx)}
        pats['dx_id'] = pats['apacheadmissiondx'].map(dx_map)
    else:
        # Si pas de diagnostic, mettre -1
        pats['dx_id'] = -1

    return pats


def filter_patients_on_columns_model(pats):
    """
    Sélectionner seulement les colonnes nécessaires pour le modèle

    Args:
        pats: DataFrame des patients

    Returns:
        DataFrame avec colonnes sélectionnées
    """
    print(f"  🔍 Sélection des colonnes pour le modèle...")

    # Colonnes à garder
    keep_columns = [
        'patientunitstayid',
        'patienthealthsystemstayid',
        'age',
        'age_numeric',
        'gender',
        'ethnicity',
        'unitdischargeoffset',
        'unitdischargestatus',
        'hospitaldischargestatus',
        'unittype',
        'unitadmittime24',
        'unitdischargetime24'
    ]

    # Ajouter dx_id si elle existe
    if 'dx_id' in pats.columns:
        keep_columns.append('dx_id')

    # Garder seulement les colonnes qui existent
    available_columns = [col for col in keep_columns if col in pats.columns]

    pats = pats[available_columns].copy()

    print(f"     ✅ {len(available_columns)} colonnes conservées")

    return pats


def create_labels(pats):
    """
    Créer les labels pour l'entraînement (mortalité ICU)

    Args:
        pats: DataFrame des patients

    Returns:
        DataFrame avec colonne 'label' ajoutée
    """
    print(f"  🏷️  Création des labels (mortalité ICU)...")

    # Label = 1 si décédé à l'ICU, 0 sinon
    if 'unitdischargestatus' in pats.columns:
        pats['label'] = (pats['unitdischargestatus'] == 1).astype(int)
    else:
        # Fallback sur hospital discharge status
        pats['label'] = (pats['hospitaldischargestatus'] == 1).astype(int)

    n_positive = pats['label'].sum()
    n_total = len(pats)

    print(f"     ✅ Labels créés : {n_positive}/{n_total} décès ({n_positive / n_total * 100:.1f}%)")

    return pats


def get_patient_icustay_hours(pats):
    """
    Calculer la durée du séjour ICU en heures

    Args:
        pats: DataFrame des patients

    Returns:
        pd.Series avec durée en heures
    """
    # unitdischargeoffset est en minutes
    return pats['unitdischargeoffset'] / 60


def filter_short_stays(pats, min_hours=24):
    """
    Filtrer les séjours trop courts

    Args:
        pats: DataFrame des patients
        min_hours: Durée minimum en heures

    Returns:
        DataFrame filtré
    """
    print(f"  🔍 Filtrage : séjours >= {min_hours}h...")
    initial_count = len(pats)

    icustay_hours = get_patient_icustay_hours(pats)
    pats = pats[icustay_hours >= min_hours]

    print(f"     ✅ {initial_count:,} → {len(pats):,} patients")

    return pats