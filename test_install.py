"""
test_install.py
Script de vérification d'installation pour EHR-M-GAN + Preprocessing

Vérifie que toutes les dépendances sont correctement installées
"""

import sys
from pathlib import Path

print("=" * 80)
print(" " * 25 + "🧪 TEST D'INSTALLATION")
print("=" * 80)

# ============================================================================
# 1. PYTHON VERSION
# ============================================================================

print("\n📋 ÉTAPE 1/7 : Vérification de Python")
print("-" * 80)

python_version = sys.version_info
print(f"  🐍 Version Python : {python_version.major}.{python_version.minor}.{python_version.micro}")

if python_version.major == 3 and 8 <= python_version.minor <= 10:
    print("  ✅ Version Python compatible")
elif python_version.major == 3 and python_version.minor == 11:
    print("  ⚠️  Python 3.11 peut avoir des problèmes avec TensorFlow 2.10")
    print("      Recommandation : utiliser Python 3.8, 3.9, ou 3.10")
else:
    print("  ❌ Version Python non compatible")
    print("      Requis : Python 3.8, 3.9, ou 3.10")

# ============================================================================
# 2. PACKAGES CORE
# ============================================================================

print("\n📦 ÉTAPE 2/7 : Vérification des packages core")
print("-" * 80)

packages_status = []

# NumPy
try:
    import numpy as np

    print(f"  ✅ NumPy {np.__version__}")
    packages_status.append(("NumPy", True, np.__version__))
except ImportError as e:
    print(f"  ❌ NumPy non installé : {e}")
    packages_status.append(("NumPy", False, None))

# Pandas
try:
    import pandas as pd

    print(f"  ✅ Pandas {pd.__version__}")
    packages_status.append(("Pandas", True, pd.__version__))
except ImportError as e:
    print(f"  ❌ Pandas non installé : {e}")
    packages_status.append(("Pandas", False, None))

# tqdm
try:
    from tqdm import tqdm
    import tqdm as tqdm_module

    print(f"  ✅ tqdm {tqdm_module.__version__}")
    packages_status.append(("tqdm", True, tqdm_module.__version__))
except ImportError as e:
    print(f"  ❌ tqdm non installé : {e}")
    packages_status.append(("tqdm", False, None))

# ============================================================================
# 3. TENSORFLOW
# ============================================================================

print("\n🤖 ÉTAPE 3/7 : Vérification de TensorFlow")
print("-" * 80)

try:
    import tensorflow as tf

    print(f"  ✅ TensorFlow {tf.__version__}")
    packages_status.append(("TensorFlow", True, tf.__version__))

    # Vérifier GPU
    print("\n  🎮 Vérification GPU...")
    gpus = tf.config.list_physical_devices('GPU')

    if gpus:
        print(f"     ✅ {len(gpus)} GPU(s) disponible(s) :")
        for i, gpu in enumerate(gpus):
            print(f"        [{i}] {gpu.name}")
            try:
                # Vérifier mémoire GPU
                gpu_details = tf.config.experimental.get_memory_info(gpu.name)
                print(f"            Mémoire : {gpu_details.get('current', 0) / 1024 ** 3:.2f} GB")
            except:
                pass
    else:
        print("     💻 Mode CPU uniquement (pas de GPU détecté)")
        print("        → L'entraînement sera plus lent mais fonctionnel")

    # Test simple TensorFlow
    print("\n  🧪 Test TensorFlow rapide...")
    try:
        x = tf.constant([[1.0, 2.0], [3.0, 4.0]])
        y = tf.constant([[1.0], [1.0]])
        result = tf.matmul(x, y)
        print("     ✅ Opérations TensorFlow fonctionnelles")
    except Exception as e:
        print(f"     ❌ Erreur lors du test TensorFlow : {e}")

except ImportError as e:
    print(f"  ❌ TensorFlow non installé : {e}")
    print("     → Installer avec : pip install tensorflow==2.10.1")
    packages_status.append(("TensorFlow", False, None))

# ============================================================================
# 4. MACHINE LEARNING
# ============================================================================

print("\n🔬 ÉTAPE 4/7 : Vérification des packages ML")
print("-" * 80)

# Scikit-learn
try:
    import sklearn

    print(f"  ✅ Scikit-learn {sklearn.__version__}")
    packages_status.append(("Scikit-learn", True, sklearn.__version__))
except ImportError as e:
    print(f"  ❌ Scikit-learn non installé : {e}")
    packages_status.append(("Scikit-learn", False, None))

# Scipy
try:
    import scipy

    print(f"  ✅ SciPy {scipy.__version__}")
    packages_status.append(("SciPy", True, scipy.__version__))
except ImportError as e:
    print(f"  ❌ SciPy non installé : {e}")
    packages_status.append(("SciPy", False, None))

# ============================================================================
# 5. VISUALISATION
# ============================================================================

print("\n📊 ÉTAPE 5/7 : Vérification des packages de visualisation")
print("-" * 80)

# Matplotlib
try:
    import matplotlib

    print(f"  ✅ Matplotlib {matplotlib.__version__}")
    packages_status.append(("Matplotlib", True, matplotlib.__version__))
except ImportError as e:
    print(f"  ⚠️  Matplotlib non installé (optionnel) : {e}")
    packages_status.append(("Matplotlib", False, None))

# Seaborn
try:
    import seaborn

    print(f"  ✅ Seaborn {seaborn.__version__}")
    packages_status.append(("Seaborn", True, seaborn.__version__))
except ImportError as e:
    print(f"  ⚠️  Seaborn non installé (optionnel) : {e}")
    packages_status.append(("Seaborn", False, None))

# ============================================================================
# 6. STRUCTURE DES DOSSIERS
# ============================================================================

print("\n📁 ÉTAPE 6/7 : Vérification de la structure des dossiers")
print("-" * 80)

required_dirs = [
    "data/real/eicu",
    "data/checkpoint",
    "data/fake",
    "evaluation_metrics"
]

for dir_path in required_dirs:
    path = Path(dir_path)
    if path.exists():
        print(f"  ✅ {dir_path}/")
    else:
        print(f"  ⚠️  {dir_path}/ manquant (sera créé automatiquement)")

# Vérifier données eICU
eicu_data_path = Path("data/real/eicu/raw/eicu-collaborative-research-database-demo-2.0.1")
if eicu_data_path.exists():
    csv_files = list(eicu_data_path.glob("*.csv.gz"))
    print(f"\n  ✅ Données eICU trouvées : {len(csv_files)} fichiers CSV.GZ")

    # Vérifier fichiers clés
    key_files = ["patient.csv.gz", "vitalPeriodic.csv.gz", "treatment.csv.gz", "infusiondrug.csv.gz"]
    missing_files = []
    for file in key_files:
        if not (eicu_data_path / file).exists():
            missing_files.append(file)

    if missing_files:
        print(f"  ⚠️  Fichiers manquants : {', '.join(missing_files)}")
    else:
        print(f"  ✅ Tous les fichiers clés présents")
else:
    print(f"  ❌ Données eICU non trouvées dans {eicu_data_path}")
    print("     → Télécharger depuis : https://physionet.org/content/eicu-crd-demo/2.0.1/")

# ============================================================================
# 7. FICHIERS PREPROCESSING
# ============================================================================

print("\n📄 ÉTAPE 7/7 : Vérification des fichiers de preprocessing")
print("-" * 80)

preprocessing_files = [
    "preprocessing_eicu_complete.py",
    "main_train.py",
    "train_config.py",
    "networks.py",
    "m3gan.py"
]

for file in preprocessing_files:
    path = Path(file)
    if path.exists():
        print(f"  ✅ {file}")
    else:
        print(f"  ⚠️  {file} manquant")

# Vérifier fichiers preprocessés (s'ils existent)
processed_files = [
    "data/real/eicu/vital_sign_24hrs.pkl",
    "data/real/eicu/med_interv_24hrs.pkl",
    "data/real/eicu/statics.pkl",
    "data/real/eicu/norm_stats.npz"
]

processed_exist = all(Path(f).exists() for f in processed_files)
if processed_exist:
    print(f"\n  ✅ Données preprocessées trouvées (prêt pour l'entraînement !)")
else:
    print(f"\n  ℹ️  Données pas encore preprocessées")
    print("     → Lancer : python preprocessing_eicu_complete.py")

# ============================================================================
# RÉSUMÉ
# ============================================================================

print("\n" + "=" * 80)
print(" " * 30 + "📊 RÉSUMÉ")
print("=" * 80)

# Compter succès/échecs
core_packages = ["NumPy", "Pandas", "tqdm", "TensorFlow"]
core_status = [status for name, status, _ in packages_status if name in core_packages]

if all(core_status):
    print("\n✅ TOUS LES PACKAGES CORE SONT INSTALLÉS")
    print("\n🎉 Installation réussie ! Vous pouvez commencer le preprocessing.")
    print("\nProchaines étapes :")
    print("  1. Si pas encore fait : python preprocessing_eicu_complete.py")
    print("  2. Lancer l'entraînement : python main_train.py --dataset eicu")
else:
    print("\n⚠️  PACKAGES MANQUANTS DÉTECTÉS")
    print("\nPackages à installer :")
    for name, status, version in packages_status:
        if name in core_packages and not status:
            print(f"  ❌ {name}")

    print("\nCommande d'installation :")
    print("  pip install -r requirements.txt")

# Tableau récapitulatif
print("\n" + "-" * 80)
print(f"{'Package':<20} {'Status':<10} {'Version':<15}")
print("-" * 80)
for name, status, version in packages_status:
    status_icon = "✅" if status else "❌"
    version_str = version if version else "Non installé"
    print(f"{name:<20} {status_icon:<10} {version_str:<15}")
print("-" * 80)

print("\n" + "=" * 80)
print("🔧 Pour plus d'aide, consultez le guide d'installation complet")
print("=" * 80)