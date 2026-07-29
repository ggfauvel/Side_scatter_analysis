#!/bin/bash
# ============================================================
#  Spectro Multi-Fibres - lanceur macOS (double-cliquez-moi)
#  Si macOS bloque l'ouverture : clic droit > Ouvrir.
# ============================================================
cd "$(dirname "$0")"
if ! command -v python3 &> /dev/null; then
    echo "Python 3 n'est pas installe. Installez-le depuis https://www.python.org/downloads/"
    read -p "Appuyez sur Entree pour fermer."
    exit 1
fi
if [ ! -d ".venv" ]; then
    echo "Premiere installation : creation de l'environnement..."
    python3 -m venv .venv
    source .venv/bin/activate
    python -m pip install --upgrade pip
    pip install -r requirements.txt
else
    source .venv/bin/activate
fi
python run.py
