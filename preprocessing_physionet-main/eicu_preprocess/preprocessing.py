"""
preprocessing_eicu_complete.py

Script complet de preprocessing : eICU-CRD Demo → Format EHR-M-GAN
Génère les 4 fichiers requis en une seule exécution :
  1. vital_sign_24hrs.pkl
  2. med_interv_24hrs.pkl
  3. statics.pkl
  4. norm_stats.npz

Usage:
    python preprocessing_eicu_complete.py
"""

import pandas as pd
import numpy as np
import pickle
import os
from pathlib import Path
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================

# Chemins
EICU_PATH = Path('data/real/eicu/raw/eicu-collaborative-research-database-demo-2.0.1')
OUTPUT_PATH = Path('data/real/eicu')
OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

# Paramètres
WINDOW_HOURS = 24  # Fenêtre de 24 heures
MIN_ICU_STAY_HOURS = 24  # Séjour minimum 24h
MIN_AGE = 15
MAX_AGE = 89
MIN_MEASUREMENTS = 12  # Au moins 12 mesures par patient

print("=" * 80)
print(" " * 20 + "🔄 PREPROCESSING eICU → EHR-M-GAN")
print("=" * 80)
print(f"\n⚙️  Configuration:")
print(f"  • Chemin eICU : {EICU_PATH}")
print(f"  • Fenêtre temporelle : {WINDOW_HOURS} heures")
print(f"  • Séjour minimum : {MIN_ICU_STAY_HOURS}h")
print(f"  • Âge : {MIN_AGE}-{MAX_AGE} ans")
print("=" * 80)

# ============================================================================
# ÉTAPE 1 : CHARGER ET FILTRER LES PATIENTS
# ============================================================================

print("\n" + "=" * 80)
print("📋 ÉTAPE 1/6 : Chargement et filtrage des patients")
print("=" * 80)

# Charger la table patients
print("\n  📂 Chargement de patient.csv.gz...")
patients = pd.read_csv(EICU_PATH / 'patient.csv.gz', compression='gzip')
print(f"     ✅ {len(patients):,} admissions chargées")

initial_count = len(patients)

# Filtrer par âge
print("\n  🔍 Filtrage par âge...")
patients = patients[patients['age'] != ''].copy()
patients['age_numeric'] = patients['age'].replace('> 89', '90').astype(float)
patients = patients[(patients['age_numeric'] >= MIN_AGE) & (patients['age_numeric'] <= MAX_AGE)]
print(f"     ✅ {initial_count:,} → {len(patients):,} patients")

# Un séjour par patient (premier séjour seulement)
print("\n  🔍 Filtrage : premier séjour ICU uniquement...")
initial_count = len(patients)
patients = patients.sort_values(['patienthealthsystemstayid', 'unitvisitnumber'])
patients = patients.groupby('patienthealthsystemstayid').first().reset_index()
print(f"     ✅ {initial_count:,} → {len(patients):,} patients")

# Filtrer par durée de séjour
print(f"\n  🔍 Filtrage : séjour >= {MIN_ICU_STAY_HOURS}h...")
initial_count = len(patients)
min_offset = MIN_ICU_STAY_HOURS * 60  # Convertir en minutes
patients = patients[patients['unitdischargeoffset'] >= min_offset]
print(f"     ✅ {initial_count:,} → {len(patients):,} patients")

# Créer le label de mortalité
patients['label'] = (patients['unitdischargestatus'] == 'Expired').astype(int)

print(f"\n  ✅ Cohorte finale : {len(patients):,} patients")
print(f"     • Taux de mortalité ICU : {patients['label'].mean() * 100:.1f}%")

# ============================================================================
# ÉTAPE 2 : EXTRAIRE LES SIGNAUX VITAUX (DONNÉES CONTINUES)
# ============================================================================

print("\n" + "=" * 80)
print("📊 ÉTAPE 2/6 : Extraction des signaux vitaux (continues)")
print("=" * 80)

# Features vitales à extraire
VITAL_FEATURES = [
    'heartrate',  # Fréquence cardiaque
    'respiration',  # Fréquence respiratoire
    'spo2',  # Saturation en oxygène
    'temperature',  # Température
    'systemicsystolic',  # Pression systolique
    'systemicdiastolic',  # Pression diastolique
    'systemicmean'  # Pression moyenne
]

print(f"\n  📂 Chargement de vitalPeriodic.csv.gz...")
vitals = pd.read_csv(EICU_PATH / 'vitalPeriodic.csv.gz', compression='gzip')
print(f"     ✅ {len(vitals):,} mesures chargées")

# Filtrer sur la cohorte
valid_ids = set(patients['patientunitstayid'])
vitals = vitals[vitals['patientunitstayid'].isin(valid_ids)]

print(f"\n  🔄 Extraction des fenêtres 24h pour {len(valid_ids):,} patients...")

continuous_data_list = []
valid_patient_ids = []
failed_patients = 0

for _, patient in tqdm(patients.iterrows(), total=len(patients), desc="  Traitement"):
    patient_id = patient['patientunitstayid']
    discharge_offset = patient['unitdischargeoffset']

    # Fenêtre : [discharge - 24h, discharge]
    start_offset = discharge_offset - (WINDOW_HOURS * 60)
    end_offset = discharge_offset

    # Extraire les vitaux dans cette fenêtre
    patient_vitals = vitals[
        (vitals['patientunitstayid'] == patient_id) &
        (vitals['observationoffset'] >= start_offset) &
        (vitals['observationoffset'] <= end_offset)
        ].copy()

    if len(patient_vitals) < MIN_MEASUREMENTS:
        failed_patients += 1
        continue

    # Convertir offset en heures relatives (0-24)
    patient_vitals['hour'] = (patient_vitals['observationoffset'] - start_offset) / 60
    patient_vitals['hour'] = patient_vitals['hour'].clip(0, WINDOW_HOURS)
    patient_vitals['hour_bin'] = patient_vitals['hour'].astype(int).clip(0, WINDOW_HOURS - 1)

    # Sélectionner les features
    vital_cols = [col for col in VITAL_FEATURES if col in patient_vitals.columns]

    # Agrégation horaire (moyenne)
    vitals_hourly = patient_vitals.groupby('hour_bin')[vital_cols].mean()

    # Reindex pour avoir exactement 24 heures
    vitals_hourly = vitals_hourly.reindex(range(WINDOW_HOURS))

    # Imputation : forward fill → backward fill → mean
    vitals_hourly = vitals_hourly.fillna(method='ffill').fillna(method='bfill')
    vitals_hourly = vitals_hourly.fillna(vitals_hourly.mean())

    # Si toujours des NaN, skip
    if vitals_hourly.isnull().any().any():
        failed_patients += 1
        continue

    continuous_data_list.append(vitals_hourly.values)
    valid_patient_ids.append(patient_id)

continuous_data = np.array(continuous_data_list)

print(f"\n  ✅ Extraction terminée :")
print(f"     • Patients traités avec succès : {len(valid_patient_ids):,}")
print(f"     • Patients échoués (données insuffisantes) : {failed_patients}")
print(f"     • Shape finale : {continuous_data.shape}")
print(f"     • Features : {len(VITAL_FEATURES)} ({', '.join(VITAL_FEATURES[:3])}...)")

# ============================================================================
# ÉTAPE 3 : EXTRAIRE LES INTERVENTIONS MÉDICALES (DONNÉES DISCRÈTES)
# ============================================================================

print("\n" + "=" * 80)
print("💊 ÉTAPE 3/6 : Extraction des interventions médicales (discrètes)")
print("=" * 80)

# Features d'interventions
DISCRETE_FEATURES = [
    'mechanical_ventilation',
    'vasopressor',
    'dialysis'
]

print(f"\n  📂 Chargement de treatment.csv.gz et infusiondrug.csv.gz...")
treatment = pd.read_csv(EICU_PATH / 'treatment.csv.gz', compression='gzip')
infusion = pd.read_csv(EICU_PATH / 'infusiondrug.csv.gz', compression='gzip')
print(f"     ✅ {len(treatment):,} traitements + {len(infusion):,} infusions")

# Filtrer sur la cohorte valide
treatment = treatment[treatment['patientunitstayid'].isin(valid_patient_ids)]
infusion = infusion[infusion['patientunitstayid'].isin(valid_patient_ids)]

print(f"\n  🔄 Extraction des interventions 24h pour {len(valid_patient_ids):,} patients...")

discrete_data_list = []
vasopressor_drugs = ['norepinephrine', 'epinephrine', 'dopamine',
                     'vasopressin', 'phenylephrine', 'dobutamine', 'milrinone']

for patient_id in tqdm(valid_patient_ids, desc="  Traitement"):

    patient_info = patients[patients['patientunitstayid'] == patient_id].iloc[0]
    discharge_offset = patient_info['unitdischargeoffset']
    start_offset = discharge_offset - (WINDOW_HOURS * 60)
    end_offset = discharge_offset

    # Initialiser matrice [24, 3]
    interventions = np.zeros((WINDOW_HOURS, len(DISCRETE_FEATURES)))

    # 1. VENTILATION MÉCANIQUE
    vent_keywords = ['ventilation', 'intubation', 'mechanical vent', 'intubated']
    vent = treatment[
        (treatment['patientunitstayid'] == patient_id) &
        (treatment['treatmentstring'].str.contains('|'.join(vent_keywords),
                                                   case=False, na=False)) &
        (treatment['treatmentoffset'] >= start_offset) &
        (treatment['treatmentoffset'] <= end_offset)
        ]

    for _, row in vent.iterrows():
        hour_bin = int((row['treatmentoffset'] - start_offset) / 60)
        hour_bin = max(0, min(hour_bin, WINDOW_HOURS - 1))
        interventions[hour_bin, 0] = 1

    # Forward fill : si ventilé à t, reste ventilé jusqu'à changement
    for i in range(1, WINDOW_HOURS):
        if interventions[i, 0] == 0 and interventions[i - 1, 0] == 1:
            interventions[i, 0] = 1

    # 2. VASOPRESSEURS
    vaso = infusion[
        (infusion['patientunitstayid'] == patient_id) &
        (infusion['drugname'].str.lower().isin(vasopressor_drugs)) &
        (infusion['infusionoffset'] >= start_offset) &
        (infusion['infusionoffset'] <= end_offset)
        ]

    for _, row in vaso.iterrows():
        hour_bin = int((row['infusionoffset'] - start_offset) / 60)
        hour_bin = max(0, min(hour_bin, WINDOW_HOURS - 1))
        interventions[hour_bin, 1] = 1

    # 3. DIALYSE
    dialysis_keywords = ['dialysis', 'CRRT', 'hemodialysis', 'hemofiltration']
    dial = treatment[
        (treatment['patientunitstayid'] == patient_id) &
        (treatment['treatmentstring'].str.contains('|'.join(dialysis_keywords),
                                                   case=False, na=False)) &
        (treatment['treatmentoffset'] >= start_offset) &
        (treatment['treatmentoffset'] <= end_offset)
        ]

    for _, row in dial.iterrows():
        hour_bin = int((row['treatmentoffset'] - start_offset) / 60)
        hour_bin = max(0, min(hour_bin, WINDOW_HOURS - 1))
        interventions[hour_bin, 2] = 1

    discrete_data_list.append(interventions)

discrete_data = np.array(discrete_data_list)

print(f"\n  ✅ Extraction terminée :")
print(f"     • Shape finale : {discrete_data.shape}")
print(f"     • Features : {DISCRETE_FEATURES}")
print(f"     • Taux d'activation :")
for i, feature in enumerate(DISCRETE_FEATURES):
    rate = (discrete_data[:, :, i].sum() / discrete_data[:, :, i].size) * 100
    print(f"       - {feature}: {rate:.1f}%")

# ============================================================================
# ÉTAPE 4 : CRÉER LES LABELS STATIQUES
# ============================================================================

print("\n" + "=" * 80)
print("🏷️  ÉTAPE 4/6 : Création des labels statiques")
print("=" * 80)

# Filtrer patients sur la cohorte valide
patients_final = patients[patients['patientunitstayid'].isin(valid_patient_ids)]
patients_final = patients_final.set_index('patientunitstayid').loc[valid_patient_ids].reset_index()

# Extraire les labels
statics_label = patients_final[['label']].values

print(f"\n  ✅ Labels créés :")
print(f"     • Shape : {statics_label.shape}")
print(f"     • Mortalité ICU : {statics_label.sum()}/{len(statics_label)} ({statics_label.mean() * 100:.1f}%)")

# ============================================================================
# ÉTAPE 5 : NORMALISATION MIN-MAX
# ============================================================================

print("\n" + "=" * 80)
print("📏 ÉTAPE 5/6 : Normalisation des données continues")
print("=" * 80)

print("\n  🔄 Application de la normalisation min-max...")

# Calculer min/max sur toutes les dimensions (patients + temps)
min_val = continuous_data.min(axis=(0, 1), keepdims=True)
max_val = continuous_data.max(axis=(0, 1), keepdims=True)

# Éviter division par zéro
range_val = max_val - min_val
range_val[range_val == 0] = 1.0

# Normaliser
continuous_data_normalized = (continuous_data - min_val) / range_val

print(f"\n  ✅ Normalisation terminée :")
print(f"     • Range : [0, 1]")
print(f"     • Min par feature : {min_val.squeeze()}")
print(f"     • Max par feature : {max_val.squeeze()}")

# ============================================================================
# ÉTAPE 6 : SAUVEGARDER LES FICHIERS
# ============================================================================

print("\n" + "=" * 80)
print("💾 ÉTAPE 6/6 : Sauvegarde des fichiers finaux")
print("=" * 80)

print(f"\n  📁 Répertoire de sortie : {OUTPUT_PATH.absolute()}\n")

# 1. vital_sign_24hrs.pkl
filepath = OUTPUT_PATH / 'vital_sign_24hrs.pkl'
with open(filepath, 'wb') as f:
    # pickle.dump(continuous_data_normalized, f, pickle.HIGHEST_PROTOCOL)
    pickle.dump(continuous_data_normalized, f, 4)
print(f"  ✅ {filepath.name}")
print(f"     Shape: {continuous_data_normalized.shape}, dtype: {continuous_data_normalized.dtype}")

# 2. med_interv_24hrs.pkl
filepath = OUTPUT_PATH / 'med_interv_24hrs.pkl'
with open(filepath, 'wb') as f:
    # pickle.dump(discrete_data, f, pickle.HIGHEST_PROTOCOL)
    pickle.dump(discrete_data, f, 4)
print(f"  ✅ {filepath.name}")
print(f"     Shape: {discrete_data.shape}, dtype: {discrete_data.dtype}")

# 3. statics.pkl
filepath = OUTPUT_PATH / 'statics.pkl'
with open(filepath, 'wb') as f:
    # pickle.dump(statics_label, f, pickle.HIGHEST_PROTOCOL)
    pickle.dump(statics_label, f, 4)
print(f"  ✅ {filepath.name}")
print(f"     Shape: {statics_label.shape}, dtype: {statics_label.dtype}")

# 4. norm_stats.npz
filepath = OUTPUT_PATH / 'norm_stats.npz'
np.savez(filepath,
         min_val=min_val.squeeze(),
         max_val=max_val.squeeze())
print(f"  ✅ {filepath.name}")
print(f"     Min/Max stats pour dénormalisation")

# ============================================================================
# RÉSUMÉ FINAL
# ============================================================================

print("\n" + "=" * 80)
print(" " * 25 + "✅ PREPROCESSING TERMINÉ !")
print("=" * 80)

print(f"\n📊 RÉSUMÉ :")
print(f"  • Patients traités : {len(valid_patient_ids):,}")
print(f"  • Données continues : {continuous_data_normalized.shape}")
print(f"    → Features : {VITAL_FEATURES}")
print(f"  • Données discrètes : {discrete_data.shape}")
print(f"    → Features : {DISCRETE_FEATURES}")
print(f"  • Labels : {statics_label.shape}")
print(f"  • Taux de mortalité : {statics_label.mean() * 100:.1f}%")

print(f"\n📁 FICHIERS CRÉÉS :")
print(f"  1. vital_sign_24hrs.pkl")
print(f"  2. med_interv_24hrs.pkl")
print(f"  3. statics.pkl")
print(f"  4. norm_stats.npz")

print(f"\n🚀 PROCHAINE ÉTAPE :")
print(f"  Lancez l'entraînement EHR-M-GAN avec :")
print(f"  cd ../../ehrMGAN-main")
print(f"  python main_train.py --dataset eicu")

print("\n" + "=" * 80)