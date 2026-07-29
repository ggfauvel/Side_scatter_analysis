#!/bin/bash
# ============================================================
#  Spectro Multi-Fibres - lanceur Linux
#  Rendez-le executable si besoin : chmod +x Lancer_Linux.sh
# ============================================================
cd "$(dirname "$0")"
if ! command -v python3 &> /dev/null; then
    echo "Python 3 n'est pas installe (ex : sudo apt install python3 python3-venv)."
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
