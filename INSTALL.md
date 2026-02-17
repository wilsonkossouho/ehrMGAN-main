# 📦 Guide d'Installation — EHR-M-GAN

## Prérequis

- Ubuntu 20.04 / 22.04 / 24.04 (ou WSL2)
- Python 3.7 installé
- Compte PhysioNet (gratuit) → https://physionet.org/register/
- Git installé

---

## 🚀 Installation automatique (recommandé)

```bash
bash setup.sh
```

Le script fait tout automatiquement (étapes 1 à 7 ci-dessous).

---

## 🔧 Installation manuelle (étape par étape)

### Étape 1 — Cloner le dépôt

```bash
git clone https://github.com/wilsonkossouho/ehrMGAN-main.git
cd ehrMGAN-main
```

---

### Étape 2 — Créer l'environnement Python 3.7

```bash
python3.7 -m venv venv37
source venv37/bin/activate
```

Vérification :
```bash
python --version
# Python 3.7.x
```

---

### Étape 3 — Installer les dépendances

```bash
pip install --upgrade pip
pip install tensorflow==1.15
pip install torch==1.12.1
pip install numpy==1.18.5
pip install pandas==0.25.3
pip install matplotlib==3.1.3
pip install seaborn==0.12.2
pip install scikit-learn scipy tqdm
```

> ⚠️ Ne pas upgrader numpy au-delà de 1.18.5 — incompatible avec TensorFlow 1.15

---

### Étape 4 — Télécharger le dataset eICU-CRD Demo

Le dataset est disponible gratuitement sur PhysioNet en **accès libre**, aucun compte requis.

```bash
# Créer le dossier de destination
mkdir -p data/real/eicu/raw/eicu-collaborative-research-database-demo-2.0.1

# Télécharger les fichiers nécessaires
cd data/real/eicu/raw/eicu-collaborative-research-database-demo-2.0.1

wget https://physionet.org/files/eicu-crd-demo/2.0.1/patient.csv.gz \
     https://physionet.org/files/eicu-crd-demo/2.0.1/vitalPeriodic.csv.gz \
     https://physionet.org/files/eicu-crd-demo/2.0.1/respiratoryCare.csv.gz \
     https://physionet.org/files/eicu-crd-demo/2.0.1/infusiondrug.csv.gz \
     https://physionet.org/files/eicu-crd-demo/2.0.1/treatment.csv.gz

# Revenir à la racine
cd ../../../../..
```

Structure attendue :
```
ehrMGAN-main/
└── data/
    └── real/
        └── eicu/
            └── raw/
                └── eicu-collaborative-research-database-demo-2.0.1/
                    ├── patient.csv.gz
                    ├── vitalPeriodic.csv.gz
                    ├── respiratoryCare.csv.gz
                    ├── infusiondrug.csv.gz
                    └── treatment.csv.gz
```

---

### Étape 5 — Lancer le preprocessing

```bash
# Aller dans le dossier preprocessing
cd preprocessing_physionet-main/eicu_preprocess

# Lancer le script
python preprocessing_final.py

# Revenir à la racine
cd ../..
```

Résultat attendu :
```
✅ PREPROCESSING TERMINÉ
  Patients traités           : ~1496
  Features continues totales : 12
  Features discrètes totales : 12
```

Fichiers générés dans `data/real/eicu/` :
- `vital_sign_24hrs.pkl`
- `med_interv_24hrs.pkl`
- `statics.pkl`
- `norm_stats.npz`

---

### Étape 6 — Lancer l'entraînement

```bash
# Depuis la racine du projet
python main_train.py --dataset eicu
```

Vous devriez voir :
```
start pretraining
pretraining epoch 0
pretraining epoch 1
...
```

> ⏳ Durée estimée : 6-8 heures sur CPU

---

## 📁 Structure du projet

```
ehrMGAN-main/
├── main_train.py                  ← Point d'entrée entraînement
├── m3gan.py                       ← Architecture principale
├── networks.py                    ← Réseaux VAE + GAN
├── setup.sh                       ← Script installation automatique
├── venv37/                        ← Environnement virtuel (non versionné)
├── data/
│   ├── real/eicu/                 ← Données prétraitées (.pkl)
│   │   └── raw/                   ← Données brutes eICU (non versionnées)
│   └── fake/                      ← Données synthétiques générées
├── preprocessing_physionet-main/
│   └── eicu_preprocess/
│       └── preprocessing_final.py ← Script de preprocessing
├── evaluation_metrics/
│   └── visualise.py               ← Visualisations
└── logs/                          ← TensorBoard + visualisations
```

---

## ⚠️ Erreurs courantes

| Erreur | Solution |
|--------|----------|
| `No module named seaborn` | `pip install seaborn` |
| `numpy incompatible` | `pip install numpy==1.18.5 --force-reinstall` |
| `ValueError: Sample larger than population` | Relancer le preprocessing (12 features requises) |
| `17 patients seulement` | Utiliser `preprocessing_final.py` (pas les anciennes versions) |
| `HTTP 408 timeout` | `git config --global http.postBuffer 524288000` |

---

## 📊 Résultats attendus

Après entraînement complet (800 epochs) :
- Données synthétiques dans `data/fake/`
- Visualisations dans `logs/visualizations/`
- Modèle sauvegardé dans `data/checkpoint/`