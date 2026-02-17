"""
inspect_results.py
Inspecter les données synthétiques générées par EHR-M-GAN
"""

import numpy as np
from pathlib import Path

print("="*70)
print("🔍 Inspection des données synthétiques EHR-M-GAN")
print("="*70)

# ============================================================================
# 1. CHARGER LES DONNÉES
# ============================================================================

# Dernières données générées (epoch 699)
data_path = Path('data/fake/epoch699/gen_data.npz')

if not data_path.exists():
    print(f"❌ Fichier introuvable : {data_path}")
    print("   Vérifiez que l'entraînement s'est terminé")
    exit(1)

print(f"\n📂 Chargement de {data_path}...")
data = np.load(data_path)

# Extraire les arrays
c_gen_data = data['c_gen_data']  # Continues (signaux vitaux)
d_gen_data = data['d_gen_data']  # Discrètes (interventions)

print("✅ Chargement terminé\n")

# ============================================================================
# 2. INFORMATIONS DE BASE
# ============================================================================

print("="*70)
print("📊 INFORMATIONS DE BASE")
print("="*70)

print(f"\n🔵 Données continues (signaux vitaux) :")
print(f"  • Shape : {c_gen_data.shape}")
print(f"  • Interprétation : ({c_gen_data.shape[0]} patients, "
      f"{c_gen_data.shape[1]} heures, {c_gen_data.shape[2]} features)")
print(f"  • Type : {c_gen_data.dtype}")
print(f"  • Range : [{c_gen_data.min():.3f}, {c_gen_data.max():.3f}]")

print(f"\n🟣 Données discrètes (interventions médicales) :")
print(f"  • Shape : {d_gen_data.shape}")
print(f"  • Interprétation : ({d_gen_data.shape[0]} patients, "
      f"{d_gen_data.shape[1]} heures, {d_gen_data.shape[2]} interventions)")
print(f"  • Type : {d_gen_data.dtype}")
print(f"  • Valeurs uniques : {np.unique(d_gen_data)}")

# ============================================================================
# 3. EXEMPLE DE PATIENT SYNTHÉTIQUE
# ============================================================================

print("\n" + "="*70)
print("👤 EXEMPLE : PATIENT SYNTHÉTIQUE #0")
print("="*70)

patient_id = 0
patient_vitals = c_gen_data[patient_id]  # [24, 7]
patient_interventions = d_gen_data[patient_id]  # [24, 3]

print(f"\n📊 Signaux vitaux sur 24h :")
print(f"  Shape : {patient_vitals.shape}")
print(f"\n  Première heure (h=0) :")
print(f"    {patient_vitals[0]}")
print(f"\n  Dernière heure (h=23) :")
print(f"    {patient_vitals[23]}")

print(f"\n💊 Interventions médicales sur 24h :")
print(f"  Shape : {patient_interventions.shape}")
print(f"\n  Actives à chaque heure :")
for hour in range(24):
    active = patient_interventions[hour].astype(int)
    print(f"    h={hour:02d}: {active}  ", end="")
    if (hour + 1) % 6 == 0:
        print()  # Nouvelle ligne tous les 6

# ============================================================================
# 4. STATISTIQUES GLOBALES
# ============================================================================

print("\n" + "="*70)
print("📈 STATISTIQUES GLOBALES")
print("="*70)

print(f"\n🔵 Signaux vitaux :")
for feat_idx in range(c_gen_data.shape[2]):
    feat_data = c_gen_data[:, :, feat_idx]
    print(f"  Feature {feat_idx+1} : "
          f"mean={feat_data.mean():.3f}, "
          f"std={feat_data.std():.3f}, "
          f"min={feat_data.min():.3f}, "
          f"max={feat_data.max():.3f}")

print(f"\n🟣 Interventions médicales (taux d'activation) :")
intervention_names = ['Ventilation mécanique', 'Vasopresseurs', 'Dialyse']
for feat_idx in range(d_gen_data.shape[2]):
    feat_data = d_gen_data[:, :, feat_idx]
    activation_rate = (feat_data > 0.5).mean() * 100
    print(f"  {intervention_names[feat_idx]} : {activation_rate:.1f}%")

# ============================================================================
# 5. QUALITÉ DES DONNÉES
# ============================================================================

print("\n" + "="*70)
print("✅ VÉRIFICATION QUALITÉ")
print("="*70)

# NaN ou Inf
has_nan = np.isnan(c_gen_data).any() or np.isnan(d_gen_data).any()
has_inf = np.isinf(c_gen_data).any() or np.isinf(d_gen_data).any()

if has_nan:
    print("  ⚠️  Contient des NaN")
else:
    print("  ✅ Pas de NaN")

if has_inf:
    print("  ⚠️  Contient des Inf")
else:
    print("  ✅ Pas de Inf")

# Range correct
if c_gen_data.min() >= 0 and c_gen_data.max() <= 1:
    print("  ✅ Signaux vitaux normalisés [0,1]")
else:
    print(f"  ⚠️  Signaux vitaux hors range : [{c_gen_data.min()}, {c_gen_data.max()}]")

if set(np.unique(d_gen_data)).issubset({0, 1}):
    print("  ✅ Interventions binaires {0,1}")
else:
    print(f"  ⚠️  Interventions non-binaires : {np.unique(d_gen_data)}")

print("\n" + "="*70)
print("🎉 Inspection terminée !")
print("="*70)