"""
visualize_synthetic_patient.py
Visualiser un patient synthétique complet
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Charger données
data = np.load('data/fake/epoch699/gen_data.npz')
c_gen = data['c_gen_data']
d_gen = data['d_gen_data']

# Prendre le patient 0
patient_vitals = c_gen[0]  # [24, 7]
patient_interventions = d_gen[0]  # [24, 3]

# Noms des features
vital_names = ['Heart Rate', 'Respiration', 'SpO2', 'Temperature',
               'Systolic BP', 'Diastolic BP', 'Mean BP']
interv_names = ['Mechanical Ventilation', 'Vasopressor', 'Dialysis']

# Créer figure
fig, axes = plt.subplots(2, 1, figsize=(14, 8))
fig.suptitle('Patient Synthétique #0 - 24 heures avant sortie ICU',
             fontsize=14, fontweight='bold')

time_axis = np.arange(24)

# PLOT 1 : Signaux vitaux
ax = axes[0]
for feat_idx in range(min(5, patient_vitals.shape[1])):
    ax.plot(time_axis, patient_vitals[:, feat_idx],
           label=vital_names[feat_idx], marker='o', alpha=0.7)

ax.set_title('Signaux Vitaux (Normalisés [0,1])', fontsize=12)
ax.set_xlabel('Temps (heures avant sortie)')
ax.set_ylabel('Valeur normalisée')
ax.legend(loc='best', fontsize=9)
ax.grid(True, alpha=0.3)
ax.set_xlim(-1, 24)

# PLOT 2 : Interventions
ax = axes[1]
for feat_idx in range(patient_interventions.shape[1]):
    offset = feat_idx * 1.5
    ax.step(time_axis, patient_interventions[:, feat_idx] + offset,
           where='post', label=interv_names[feat_idx],
           linewidth=2, alpha=0.8)

ax.set_title('Interventions Médicales', fontsize=12)
ax.set_xlabel('Temps (heures avant sortie)')
ax.set_ylabel('Intervention Active (avec offset)')
ax.legend(loc='best', fontsize=9)
ax.grid(True, alpha=0.3)
ax.set_xlim(-1, 24)
ax.set_ylim(-0.5, 5)

plt.tight_layout()
plt.savefig('synthetic_patient_example.png', dpi=150, bbox_inches='tight')
print("✅ Visualisation sauvegardée : synthetic_patient_example.png")
plt.show()