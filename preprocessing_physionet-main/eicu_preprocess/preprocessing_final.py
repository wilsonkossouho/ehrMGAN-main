#!/usr/bin/env python3
"""
Preprocessing eICU-CRD Demo pour EHR-M-GAN
- 12 features continues  (7 réelles + 5 dummy) → compatible visualise_gan (num_dim=12)
- 12 features discrètes  (3 réelles + 9 dummy) → compatible visualise_gan (num_dim=12)
- visualise_vae attend num_dim=8 → OK car 12 >= 8

Fix principal : interpolation + seuil bas pour maximiser le nb de patients
"""

import pandas as pd
import numpy as np
import pickle
import os
from pathlib import Path

# ============================================================================
# CONFIGURATION - MODIFIER CES CHEMINS SI NÉCESSAIRE
# ============================================================================
EICU_RAW_PATH = Path("../../data/real/eicu/raw/eicu-collaborative-research-database-demo-2.0.1")
OUTPUT_DIR    = Path("../../data/real/eicu")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

WINDOW_HOURS   = 24
NUM_TIMESTEPS  = 24          # 1 bin par heure
MIN_STAY_HOURS = 24
MIN_AGE        = 15
MAX_AGE        = 89
MIN_MEASURES   = 3           # ← très bas : on accepte les patients avec >= 3 mesures vitales
N_CONTINUOUS   = 12          # features continues totales attendues par le modèle
N_DISCRETE     = 12          # features discrètes totales attendues par le modèle
N_REAL_CONT    = 7           # vraies features continues
N_REAL_DISC    = 3           # vraies features discrètes

print("=" * 80)
print("      PREPROCESSING eICU → EHR-M-GAN  (12c + 12d features)")
print("=" * 80)
print(f"  Source  : {EICU_RAW_PATH}")
print(f"  Sortie  : {OUTPUT_DIR}")
print(f"  Features continues : {N_REAL_CONT} réelles + {N_CONTINUOUS - N_REAL_CONT} dummy = {N_CONTINUOUS}")
print(f"  Features discrètes : {N_REAL_DISC} réelles + {N_DISCRETE - N_REAL_DISC} dummy = {N_DISCRETE}")
print("=" * 80)

# ============================================================================
# ÉTAPE 1 : PATIENTS
# ============================================================================
print("\n[1/6] Chargement et filtrage des patients...")

patients = pd.read_csv(EICU_RAW_PATH / 'patient.csv.gz', compression='gzip')
print(f"  Total admissions : {len(patients)}")

# Filtrage âge
patients = patients[patients['age'].notna() & (patients['age'] != '> 89') & (patients['age'] != '')]
patients['age'] = pd.to_numeric(patients['age'], errors='coerce')
patients = patients[(patients['age'] >= MIN_AGE) & (patients['age'] <= MAX_AGE)]
print(f"  Après filtre âge : {len(patients)}")

# Filtrage durée séjour (unitdischargeoffset en minutes)
patients = patients[patients['unitdischargeoffset'] >= MIN_STAY_HOURS * 60]
print(f"  Après filtre durée >= 24h : {len(patients)}")

pid_list = patients['patientunitstayid'].tolist()
print(f"  ✅ Cohorte : {len(pid_list)} patients")

# ============================================================================
# ÉTAPE 2 : SIGNAUX VITAUX (7 features continues réelles)
# ============================================================================
print("\n[2/6] Chargement des signaux vitaux...")

vitals_raw = pd.read_csv(EICU_RAW_PATH / 'vitalPeriodic.csv.gz', compression='gzip')
vitals_raw = vitals_raw[vitals_raw['patientunitstayid'].isin(pid_list)]
print(f"  Mesures vitales (dataset filtré) : {len(vitals_raw)}")

VITAL_COLS = ['heartrate', 'respiration', 'sao2', 'temperature',
              'systemicsystolic', 'systemicdiastolic', 'systemicmean']

# ============================================================================
# ÉTAPE 3 : INTERVENTIONS (3 features discrètes réelles)
# ============================================================================
print("\n[3/6] Chargement des interventions...")

resp_care  = pd.read_csv(EICU_RAW_PATH / 'respiratoryCare.csv.gz', compression='gzip')
infusions  = pd.read_csv(EICU_RAW_PATH / 'infusiondrug.csv.gz',    compression='gzip')
treatments = pd.read_csv(EICU_RAW_PATH / 'treatment.csv.gz',       compression='gzip')

resp_care  = resp_care [resp_care ['patientunitstayid'].isin(pid_list)]
infusions  = infusions [infusions ['patientunitstayid'].isin(pid_list)]
treatments = treatments[treatments['patientunitstayid'].isin(pid_list)]

vasopressor_keywords = ['norepinephrine','epinephrine','dopamine','vasopressin','phenylephrine','neosynephrine']
vaso = infusions[infusions['drugname'].str.lower().str.contains('|'.join(vasopressor_keywords), na=False)]

dialysis_keywords = ['dialysis','crrt','cvvh','cvvhd','cvvhdf']
dial = treatments[treatments['treatmentstring'].str.lower().str.contains('|'.join(dialysis_keywords), na=False)]

print(f"  Ventilation : {resp_care['patientunitstayid'].nunique()} patients")
print(f"  Vasopresseurs : {vaso['patientunitstayid'].nunique()} patients")
print(f"  Dialyse : {dial['patientunitstayid'].nunique()} patients")

# ============================================================================
# ÉTAPE 4 : CONSTRUCTION DES MATRICES PATIENT PAR PATIENT
# ============================================================================
print("\n[4/6] Construction des matrices temporelles...")

# Lookup rapide par patient
vitals_by_pid    = {pid: grp for pid, grp in vitals_raw.groupby('patientunitstayid')}
resp_by_pid      = {pid: grp for pid, grp in resp_care.groupby('patientunitstayid')}
vaso_by_pid      = {pid: grp for pid, grp in vaso.groupby('patientunitstayid')}
dial_by_pid      = {pid: grp for pid, grp in dial.groupby('patientunitstayid')}
discharge_lookup = dict(zip(patients['patientunitstayid'], patients['unitdischargeoffset']))

good_pids   = []
vital_mats  = []
interv_mats = []

skipped_no_vitals   = 0
skipped_few_measures = 0

for pid in pid_list:

    if pid not in vitals_by_pid:
        skipped_no_vitals += 1
        continue

    pv = vitals_by_pid[pid]
    discharge = discharge_lookup[pid]

    # Fenêtre 24h avant sortie
    t_start = discharge - WINDOW_HOURS * 60
    pv_win  = pv[(pv['observationoffset'] >= t_start) & (pv['observationoffset'] <= discharge)]

    # Filtre : minimum MIN_MEASURES lignes avec au moins une valeur vitale non-nulle
    valid_rows = pv_win[VITAL_COLS].notna().any(axis=1).sum()
    if valid_rows < MIN_MEASURES:
        skipped_few_measures += 1
        continue

    # ---- Matrice continue 24×7 ----
    vital_mat = np.zeros((NUM_TIMESTEPS, N_REAL_CONT))

    for hour in range(NUM_TIMESTEPS):
        # heure 0 = heure la plus récente (juste avant sortie)
        t_end_h   = discharge - hour * 60
        t_start_h = t_end_h - 60
        mask = (pv_win['observationoffset'] >= t_start_h) & (pv_win['observationoffset'] < t_end_h)
        hour_rows = pv_win.loc[mask, VITAL_COLS]
        if len(hour_rows) > 0:
            vital_mat[NUM_TIMESTEPS - 1 - hour, :] = hour_rows.mean(axis=0).fillna(0).values

    # Interpolation linéaire colonne par colonne pour combler les zéros
    for f in range(N_REAL_CONT):
        series = vital_mat[:, f].copy()
        nonzero_idx = np.where(series != 0)[0]
        if len(nonzero_idx) == 0:
            continue
        # forward-fill
        last = series[nonzero_idx[0]]
        for t in range(NUM_TIMESTEPS):
            if series[t] != 0:
                last = series[t]
            else:
                series[t] = last
        # backward-fill (pour les premiers points)
        first = series[nonzero_idx[-1]]
        for t in range(NUM_TIMESTEPS - 1, -1, -1):
            if series[t] != 0:
                first = series[t]
            else:
                series[t] = first
        vital_mat[:, f] = series

    # ---- Matrice discrète 24×3 ----
    interv_mat = np.zeros((NUM_TIMESTEPS, N_REAL_DISC))

    for hour in range(NUM_TIMESTEPS):
        t_end_h   = discharge - hour * 60
        t_start_h = t_end_h - 60
        row_idx   = NUM_TIMESTEPS - 1 - hour

        # Feature 0 : ventilation mécanique
        if pid in resp_by_pid:
            pr = resp_by_pid[pid]
            if 'respCareStatusOffset' in pr.columns:
                m = (pr['respCareStatusOffset'] >= t_start_h) & (pr['respCareStatusOffset'] < t_end_h)
                if m.any():
                    interv_mat[row_idx, 0] = 1

        # Feature 1 : vasopresseurs
        if pid in vaso_by_pid:
            pva = vaso_by_pid[pid]
            if 'infusionoffset' in pva.columns:
                m = (pva['infusionoffset'] >= t_start_h) & (pva['infusionoffset'] < t_end_h)
                if m.any():
                    interv_mat[row_idx, 1] = 1

        # Feature 2 : dialyse
        if pid in dial_by_pid:
            pd_ = dial_by_pid[pid]
            if 'treatmentoffset' in pd_.columns:
                m = (pd_['treatmentoffset'] >= t_start_h) & (pd_['treatmentoffset'] < t_end_h)
                if m.any():
                    interv_mat[row_idx, 2] = 1

    # ---- Dummy features ----
    n_dummy_cont = N_CONTINUOUS - N_REAL_CONT   # 5
    n_dummy_disc = N_DISCRETE  - N_REAL_DISC    # 9

    dummy_cont = np.clip(np.random.normal(0.5, 0.1, (NUM_TIMESTEPS, n_dummy_cont)), 0, 1)
    dummy_disc = np.random.binomial(1, 0.15, (NUM_TIMESTEPS, n_dummy_disc)).astype(float)

    vital_mat_full  = np.hstack([vital_mat,  dummy_cont])   # (24, 12)
    interv_mat_full = np.hstack([interv_mat, dummy_disc])   # (24, 12)

    good_pids.append(pid)
    vital_mats.append(vital_mat_full)
    interv_mats.append(interv_mat_full)

print(f"  ✅ Patients retenus        : {len(good_pids)}")
print(f"  ❌ Sans données vitales    : {skipped_no_vitals}")
print(f"  ❌ Trop peu de mesures (<{MIN_MEASURES}) : {skipped_few_measures}")

if len(good_pids) == 0:
    raise RuntimeError("Aucun patient retenu ! Vérifiez les chemins et le dataset.")

vital_mats  = np.array(vital_mats,  dtype=np.float64)   # (N, 24, 12)
interv_mats = np.array(interv_mats, dtype=np.float64)   # (N, 24, 12)
print(f"  Shape continues  : {vital_mats.shape}")
print(f"  Shape discrètes  : {interv_mats.shape}")

# ============================================================================
# ÉTAPE 5 : NORMALISATION MIN-MAX [0, 1]
# ============================================================================
print("\n[5/6] Normalisation min-max...")

min_vals = np.zeros(N_CONTINUOUS)
max_vals = np.ones(N_CONTINUOUS)

for f in range(N_REAL_CONT):
    vals = vital_mats[:, :, f].flatten()
    vals = vals[vals > 0]
    if len(vals) > 0:
        lo = np.percentile(vals, 1)
        hi = np.percentile(vals, 99)
        min_vals[f] = lo
        max_vals[f] = hi if hi > lo else lo + 1e-6
        vital_mats[:, :, f] = np.clip(
            (vital_mats[:, :, f] - lo) / (max_vals[f] - lo), 0, 1)

# dummy déjà dans [0,1]
for f in range(N_REAL_CONT, N_CONTINUOUS):
    min_vals[f] = 0.0
    max_vals[f] = 1.0

print(f"  ✅ Normalisation terminée")

# ============================================================================
# ÉTAPE 6 : LABELS STATIQUES (mortalité ICU)
# ============================================================================
print("\n[6/6] Sauvegarde...")

patient_subset = patients[patients['patientunitstayid'].isin(good_pids)].set_index('patientunitstayid')
mortality = [int(patient_subset.loc[pid, 'unitdischargestatus'] == 'Expired') for pid in good_pids]
statics   = [[m] for m in mortality]

# ============================================================================
# SAUVEGARDE
# ============================================================================
with open(OUTPUT_DIR / 'vital_sign_24hrs.pkl', 'wb') as f:
    pickle.dump(vital_mats, f, protocol=4)

with open(OUTPUT_DIR / 'med_interv_24hrs.pkl', 'wb') as f:
    pickle.dump(interv_mats, f, protocol=4)

with open(OUTPUT_DIR / 'statics.pkl', 'wb') as f:
    pickle.dump(statics, f, protocol=4)

np.savez(OUTPUT_DIR / 'norm_stats.npz', min_val=min_vals, max_val=max_vals)

# ============================================================================
# RÉSUMÉ
# ============================================================================
print("\n" + "=" * 80)
print("  ✅ PREPROCESSING TERMINÉ")
print("=" * 80)
print(f"  Patients traités           : {len(good_pids)}")
print(f"  Features continues totales : {N_CONTINUOUS}  (dont {N_REAL_CONT} réelles)")
print(f"  Features discrètes totales : {N_DISCRETE}  (dont {N_REAL_DISC} réelles)")
print(f"  Taux mortalité ICU         : {np.mean(mortality)*100:.1f}%")
print(f"  Fichiers générés dans      : {OUTPUT_DIR}")
print("=" * 80)
print("\n  Lancez maintenant :")
print("  python main_train.py --dataset eicu\n")