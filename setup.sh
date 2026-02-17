#!/bin/bash
# =============================================================================
#  SETUP AUTOMATIQUE - EHR-M-GAN
#  Usage : bash setup.sh
# =============================================================================

set -e  # Arrêt immédiat si une commande échoue

REPO_URL="https://github.com/wilsonkossouho/ehrMGAN-main.git"
PROJECT_DIR="ehrMGAN-main"
EICU_URL="https://physionet.org/files/eicu-crd-demo/2.0.1/"
EICU_DIR="data/real/eicu/raw/eicu-collaborative-research-database-demo-2.0.1"
PREPROCESS_DIR="preprocessing_physionet-main/eicu_preprocess"
PREPROCESS_SCRIPT="preprocessing_final.py"

echo "========================================================================"
echo "   🚀 SETUP EHR-M-GAN - DÉMARRAGE"
echo "========================================================================"

# ─────────────────────────────────────────────────────────────────────────────
# ÉTAPE 1 : Vérifier Python 3.7
# ─────────────────────────────────────────────────────────────────────────────
echo ""
echo "[1/7] Vérification de Python 3.7..."

if ! command -v python3.7 &> /dev/null; then
    echo "  ❌ Python 3.7 non trouvé. Installation..."
    sudo apt update
    sudo apt install -y python3.7 python3.7-venv python3.7-dev
else
    echo "  ✅ Python 3.7 trouvé : $(python3.7 --version)"
fi

# ─────────────────────────────────────────────────────────────────────────────
# ÉTAPE 2 : Git clone
# ─────────────────────────────────────────────────────────────────────────────
echo ""
echo "[2/7] Clonage du dépôt..."

if [ -d "$PROJECT_DIR" ]; then
    echo "  ⚠️  Dossier $PROJECT_DIR déjà existant, mise à jour..."
    cd "$PROJECT_DIR"
    git pull
else
    git clone "$REPO_URL"
    cd "$PROJECT_DIR"
    echo "  ✅ Dépôt cloné dans $(pwd)"
fi

# ─────────────────────────────────────────────────────────────────────────────
# ÉTAPE 3 : Environnement virtuel Python 3.7
# ─────────────────────────────────────────────────────────────────────────────
echo ""
echo "[3/7] Création de l'environnement virtuel Python 3.7..."

if [ ! -d "venv37" ]; then
    python3.7 -m venv venv37
    echo "  ✅ Environnement venv37 créé"
else
    echo "  ⚠️  venv37 déjà existant, on continue..."
fi

# Activer l'environnement
source venv37/bin/activate
echo "  ✅ Environnement activé : $(python --version)"

# ─────────────────────────────────────────────────────────────────────────────
# ÉTAPE 4 : Installation des dépendances
# ─────────────────────────────────────────────────────────────────────────────
echo ""
echo "[4/7] Installation des dépendances..."

pip install --upgrade pip --quiet
pip install tensorflow==1.15 --quiet
pip install torch==1.12.1 --quiet
pip install numpy==1.18.5 --quiet
pip install pandas==0.25.3 --quiet
pip install matplotlib==3.1.3 --quiet
pip install seaborn==0.12.2 --quiet
pip install scikit-learn scipy tqdm --quiet

echo "  ✅ Dépendances installées"

# ─────────────────────────────────────────────────────────────────────────────
# ÉTAPE 5 : Téléchargement du dataset eICU-CRD Demo
# ─────────────────────────────────────────────────────────────────────────────
echo ""
echo "[5/7] Téléchargement du dataset eICU-CRD Demo..."

mkdir -p "$EICU_DIR"

# Liste des fichiers nécessaires
FILES=(
    "patient.csv.gz"
    "vitalPeriodic.csv.gz"
    "respiratoryCare.csv.gz"
    "infusiondrug.csv.gz"
    "treatment.csv.gz"
)

echo "  ℹ️  Dataset open access (aucun identifiant requis)"

for FILE in "${FILES[@]}"; do
    if [ -f "$EICU_DIR/$FILE" ]; then
        echo "  ✅ $FILE déjà présent, skip"
    else
        echo "  📥 Téléchargement de $FILE..."
        wget -q -O "$EICU_DIR/$FILE" "${EICU_URL}${FILE}" && \
            echo "  ✅ $FILE téléchargé" || \
            echo "  ❌ Erreur téléchargement $FILE"
    fi
done

# ─────────────────────────────────────────────────────────────────────────────
# ÉTAPE 6 : Preprocessing
# ─────────────────────────────────────────────────────────────────────────────
echo ""
echo "[6/7] Lancement du preprocessing..."

# Aller dans le dossier preprocessing
cd "$PREPROCESS_DIR"
echo "  📂 Dossier actuel : $(pwd)"

python "$PREPROCESS_SCRIPT"

# Revenir à la racine du projet
cd ../..
echo "  ✅ Preprocessing terminé"
echo "  📂 Retour à la racine : $(pwd)"

# ─────────────────────────────────────────────────────────────────────────────
# ÉTAPE 7 : Lancement de l'entraînement
# ─────────────────────────────────────────────────────────────────────────────
echo ""
echo "[7/7] Lancement de l'entraînement EHR-M-GAN..."
echo "  ⏳ Durée estimée : 6-8 heures sur CPU"
echo ""

python main_train.py --dataset eicu

echo ""
echo "========================================================================"
echo "   ✅ SETUP ET ENTRAÎNEMENT TERMINÉS !"
echo "========================================================================"