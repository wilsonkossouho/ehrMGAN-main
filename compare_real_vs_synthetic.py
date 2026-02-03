"""
compare_real_vs_synthetic.py
Comparer distributions réelles vs synthétiques
"""

import numpy as np
import matplotlib.pyplot as plt
import pickle

# Charger données RÉELLES
with open('data/real/eicu/vital_sign_24hrs.pkl', 'rb') as f:
    c_real = pickle.load(f)

with open('data/real/eicu/med_interv_24hrs.pkl', 'rb') as f:
    d_real = pickle.load(f)

# Charger données SYNTHÉTIQUES
data_syn = np.load('data/fake/epoch699/gen_data.npz')
c_syn = data_syn['c_gen_data']
d_syn = data_syn['d_gen_data']

print(f"📊 Données réelles : {c_real.shape[0]} patients")
print(f"📊 Données synthétiques : {c_syn.shape[0]} patients")

# COMPARAISON : Distributions globales
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Comparaison Données Réelles vs Synthétiques',
             fontsize=14, fontweight='bold')

# PLOT 1 : Histogramme signaux vitaux
ax = axes[0, 0]
ax.hist(c_real.flatten(), bins=50, alpha=0.6, label='Réelles', density=True)
ax.hist(c_syn.flatten(), bins=50, alpha=0.6, label='Synthétiques', density=True)
ax.set_title('Distribution Signaux Vitaux')
ax.set_xlabel('Valeur')
ax.set_ylabel('Densité')
ax.legend()
ax.grid(True, alpha=0.3)

# PLOT 2 : Box plot signaux vitaux
ax = axes[0, 1]
ax.boxplot([c_real.flatten(), c_syn.flatten()],
           labels=['Réelles', 'Synthétiques'])
ax.set_title('Statistiques Signaux Vitaux')
ax.set_ylabel('Valeur')
ax.grid(True, alpha=0.3)

# PLOT 3 : Taux d'activation interventions
ax = axes[1, 0]
interv_names = ['Ventilation', 'Vasopresseur', 'Dialyse']
real_rates = [(d_real[:,:,i] > 0.5).mean()*100 for i in range(3)]
syn_rates = [(d_syn[:,:,i] > 0.5).mean()*100 for i in range(3)]

x = np.arange(len(interv_names))
width = 0.35
ax.bar(x - width/2, real_rates, width, label='Réelles', alpha=0.8)
ax.bar(x + width/2, syn_rates, width, label='Synthétiques', alpha=0.8)
ax.set_title('Taux d\'Activation Interventions')
ax.set_ylabel('% Activation')
ax.set_xticks(x)
ax.set_xticklabels(interv_names)
ax.legend()
ax.grid(True, alpha=0.3, axis='y')

# PLOT 4 : Moyennes par feature
ax = axes[1, 1]
real_means = [c_real[:,:,i].mean() for i in range(c_real.shape[2])]
syn_means = [c_syn[:,:,i].mean() for i in range(c_syn.shape[2])]

x = np.arange(len(real_means))
width = 0.35
ax.bar(x - width/2, real_means, width, label='Réelles', alpha=0.8)
ax.bar(x + width/2, syn_means, width, label='Synthétiques', alpha=0.8)
ax.set_title('Moyennes par Feature Vitale')
ax.set_ylabel('Valeur moyenne')
ax.set_xlabel('Feature ID')
ax.legend()
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('comparison_real_vs_synthetic.png', dpi=150, bbox_inches='tight')
print("✅ Comparaison sauvegardée : comparison_real_vs_synthetic.png")
plt.show()