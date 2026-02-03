"""
visualise.py
Fonctions de visualisation pour EHR-M-GAN
Compatible Python 3.7 + TensorFlow 1.15

Génère des plots pour :
- Reconstructions VAE (continues et discrètes)
- Données générées par GAN
"""

import numpy as np
import matplotlib

matplotlib.use('Agg')  # Mode non-interactif (pas besoin d'affichage)
import matplotlib.pyplot as plt
import os
from pathlib import Path

# Configuration
VIZ_OUTPUT_DIR = Path('logs/visualizations')
VIZ_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def visualise_vae(continuous_real, continuous_recon, discrete_real, discrete_recon, inx=0):
    """
    Visualiser les reconstructions du VAE

    Args:
        continuous_real: Données continues réelles [batch, timesteps, features]
        continuous_recon: Reconstructions continues [batch, timesteps, features]
        discrete_real: Données discrètes réelles [batch, timesteps, features]
        discrete_recon: Reconstructions discrètes [batch, timesteps, features]
        inx: Index/epoch pour nommer le fichier
    """
    try:
        print(f"  📊 Génération visualisation VAE epoch {inx}...")

        # Vérifier que les données ne sont pas vides
        if continuous_real.size == 0 or continuous_recon.size == 0:
            print(f"     ⚠️ Données continues vides, skip visualisation")
            return

        if discrete_real.size == 0 or discrete_recon.size == 0:
            print(f"     ⚠️ Données discrètes vides, skip visualisation")
            return

        # Prendre le premier patient comme exemple
        cont_real_sample = continuous_real[0]  # [timesteps, features]
        cont_recon_sample = continuous_recon[0]
        disc_real_sample = discrete_real[0]
        disc_recon_sample = discrete_recon[0]

        timesteps = cont_real_sample.shape[0]

        # Créer la figure avec 2 lignes (continues + discrètes)
        fig, axes = plt.subplots(2, 1, figsize=(12, 8))
        fig.suptitle(f'VAE Reconstruction - Epoch {inx}', fontsize=14, fontweight='bold')

        # ===== PLOT 1 : Données continues =====
        ax1 = axes[0]

        # Plot toutes les features continues (max 5 pour lisibilité)
        num_features = min(cont_real_sample.shape[1], 5)
        time_axis = np.arange(timesteps)

        for feat_idx in range(num_features):
            # Réel en ligne continue
            ax1.plot(time_axis, cont_real_sample[:, feat_idx],
                     label=f'Real F{feat_idx + 1}', linestyle='-', alpha=0.7)
            # Reconstruction en pointillés
            ax1.plot(time_axis, cont_recon_sample[:, feat_idx],
                     label=f'Recon F{feat_idx + 1}', linestyle='--', alpha=0.7)

        ax1.set_title('Continuous Features (Vital Signs)', fontsize=12)
        ax1.set_xlabel('Time (hours)')
        ax1.set_ylabel('Normalized Value')
        ax1.legend(loc='upper right', fontsize=8, ncol=2)
        ax1.grid(True, alpha=0.3)

        # ===== PLOT 2 : Données discrètes =====
        ax2 = axes[1]

        num_discrete = disc_real_sample.shape[1]

        for feat_idx in range(num_discrete):
            # Décalage vertical pour séparer les features
            offset = feat_idx * 1.5

            # Réel
            ax2.step(time_axis, disc_real_sample[:, feat_idx] + offset,
                     where='post', label=f'Real Intervention {feat_idx + 1}',
                     linewidth=2, alpha=0.8)
            # Reconstruction
            ax2.step(time_axis, disc_recon_sample[:, feat_idx] + offset,
                     where='post', label=f'Recon Intervention {feat_idx + 1}',
                     linewidth=2, linestyle='--', alpha=0.8)

        ax2.set_title('Discrete Features (Medical Interventions)', fontsize=12)
        ax2.set_xlabel('Time (hours)')
        ax2.set_ylabel('Intervention Active (with offset)')
        ax2.legend(loc='upper right', fontsize=8)
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(-0.5, num_discrete * 1.5 + 0.5)

        # Sauvegarder
        plt.tight_layout()
        output_path = VIZ_OUTPUT_DIR / f'vae_reconstruction_epoch_{inx:04d}.png'
        plt.savefig(output_path, dpi=100, bbox_inches='tight')
        plt.close()

        print(f"     ✅ Sauvegardé : {output_path.name}")

    except Exception as e:
        print(f"     ❌ Erreur visualisation VAE : {e}")
        # Ne pas crasher l'entraînement à cause d'un plot raté
        pass


def visualise_gan(continuous_real, continuous_fake, discrete_real, discrete_fake, inx=0):
    """
    Visualiser les données générées par le GAN

    Args:
        continuous_real: Données continues réelles [batch, timesteps, features]
        continuous_fake: Données continues générées [batch, timesteps, features]
        discrete_real: Données discrètes réelles [batch, timesteps, features]
        discrete_fake: Données discrètes générées [batch, timesteps, features]
        inx: Index/epoch pour nommer le fichier
    """
    try:
        print(f"  📊 Génération visualisation GAN epoch {inx}...")

        # Vérifier que les données ne sont pas vides
        if continuous_real.size == 0 or continuous_fake.size == 0:
            print(f"     ⚠️ Données continues vides, skip visualisation")
            return

        if discrete_real.size == 0 or discrete_fake.size == 0:
            print(f"     ⚠️ Données discrètes vides, skip visualisation")
            return

        # Prendre 2 exemples : 1 réel + 1 généré
        cont_real_sample = continuous_real[0]
        cont_fake_sample = continuous_fake[0]
        disc_real_sample = discrete_real[0]
        disc_fake_sample = discrete_fake[0]

        timesteps = cont_real_sample.shape[0]

        # Créer figure 2x2
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(f'GAN Generation - Epoch {inx}', fontsize=14, fontweight='bold')

        time_axis = np.arange(timesteps)

        # ===== HAUT GAUCHE : Continues réelles =====
        ax = axes[0, 0]
        num_features = min(cont_real_sample.shape[1], 5)

        for feat_idx in range(num_features):
            ax.plot(time_axis, cont_real_sample[:, feat_idx],
                    label=f'Feature {feat_idx + 1}', alpha=0.8)

        ax.set_title('Real Continuous Data', fontsize=11)
        ax.set_xlabel('Time (hours)')
        ax.set_ylabel('Normalized Value')
        ax.legend(loc='best', fontsize=8)
        ax.grid(True, alpha=0.3)

        # ===== HAUT DROITE : Continues générées =====
        ax = axes[0, 1]

        for feat_idx in range(num_features):
            ax.plot(time_axis, cont_fake_sample[:, feat_idx],
                    label=f'Feature {feat_idx + 1}', alpha=0.8, linestyle='--')

        ax.set_title('Generated Continuous Data', fontsize=11)
        ax.set_xlabel('Time (hours)')
        ax.set_ylabel('Normalized Value')
        ax.legend(loc='best', fontsize=8)
        ax.grid(True, alpha=0.3)

        # ===== BAS GAUCHE : Discrètes réelles =====
        ax = axes[1, 0]
        num_discrete = disc_real_sample.shape[1]

        for feat_idx in range(num_discrete):
            offset = feat_idx * 1.5
            ax.step(time_axis, disc_real_sample[:, feat_idx] + offset,
                    where='post', label=f'Intervention {feat_idx + 1}',
                    linewidth=2, alpha=0.8)

        ax.set_title('Real Discrete Interventions', fontsize=11)
        ax.set_xlabel('Time (hours)')
        ax.set_ylabel('Active (with offset)')
        ax.legend(loc='best', fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-0.5, num_discrete * 1.5 + 0.5)

        # ===== BAS DROITE : Discrètes générées =====
        ax = axes[1, 1]

        for feat_idx in range(num_discrete):
            offset = feat_idx * 1.5
            ax.step(time_axis, disc_fake_sample[:, feat_idx] + offset,
                    where='post', label=f'Intervention {feat_idx + 1}',
                    linewidth=2, alpha=0.8, linestyle='--')

        ax.set_title('Generated Discrete Interventions', fontsize=11)
        ax.set_xlabel('Time (hours)')
        ax.set_ylabel('Active (with offset)')
        ax.legend(loc='best', fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-0.5, num_discrete * 1.5 + 0.5)

        # Sauvegarder
        plt.tight_layout()
        output_path = VIZ_OUTPUT_DIR / f'gan_generation_epoch_{inx:04d}.png'
        plt.savefig(output_path, dpi=100, bbox_inches='tight')
        plt.close()

        print(f"     ✅ Sauvegardé : {output_path.name}")

    except Exception as e:
        print(f"     ❌ Erreur visualisation GAN : {e}")
        pass


def plot_training_losses(losses_dict, output_name='training_losses.png'):
    """
    Plot l'évolution des losses pendant l'entraînement

    Args:
        losses_dict: Dict avec clés comme 'vae_loss', 'gan_loss_g', etc.
        output_name: Nom du fichier de sortie
    """
    try:
        fig, ax = plt.subplots(figsize=(10, 6))

        for loss_name, loss_values in losses_dict.items():
            if len(loss_values) > 0:
                ax.plot(loss_values, label=loss_name, alpha=0.8)

        ax.set_title('Training Losses', fontsize=14, fontweight='bold')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)

        output_path = VIZ_OUTPUT_DIR / output_name
        plt.savefig(output_path, dpi=100, bbox_inches='tight')
        plt.close()

        print(f"  ✅ Loss plot sauvegardé : {output_path.name}")

    except Exception as e:
        print(f"  ❌ Erreur plot losses : {e}")


def visualise_distributions(real_data, fake_data, title='Distribution Comparison',
                            output_name='distribution_comparison.png'):
    """
    Comparer les distributions réelles vs générées

    Args:
        real_data: Données réelles [samples, timesteps, features]
        fake_data: Données générées [samples, timesteps, features]
        title: Titre du plot
        output_name: Nom du fichier
    """
    try:
        # Flatten pour comparer les distributions globales
        real_flat = real_data.flatten()
        fake_flat = fake_data.flatten()

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle(title, fontsize=14, fontweight='bold')

        # Histogramme
        ax = axes[0]
        ax.hist(real_flat, bins=50, alpha=0.6, label='Real', density=True)
        ax.hist(fake_flat, bins=50, alpha=0.6, label='Generated', density=True)
        ax.set_xlabel('Value')
        ax.set_ylabel('Density')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_title('Value Distribution')

        # Box plot
        ax = axes[1]
        ax.boxplot([real_flat, fake_flat], labels=['Real', 'Generated'])
        ax.set_ylabel('Value')
        ax.grid(True, alpha=0.3)
        ax.set_title('Distribution Statistics')

        output_path = VIZ_OUTPUT_DIR / output_name
        plt.savefig(output_path, dpi=100, bbox_inches='tight')
        plt.close()

        print(f"  ✅ Distribution plot sauvegardé : {output_path.name}")

    except Exception as e:
        print(f"  ❌ Erreur visualisation distributions : {e}")


# Message au démarrage
print(f"📊 Module visualise.py chargé")
print(f"   Visualisations sauvegardées dans : {VIZ_OUTPUT_DIR.absolute()}")