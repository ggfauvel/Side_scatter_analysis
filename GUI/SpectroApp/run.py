"""
run.py — Demarre le serveur local et ouvre le navigateur.
Usage : python run.py   (ou double-clic sur le lanceur de votre systeme)
"""
import os
import sys
import threading
import webbrowser

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

HOST = "127.0.0.1"
PORT = 8050


def open_browser():
    webbrowser.open(f"http://{HOST}:{PORT}")


if __name__ == "__main__":
    from app.main import app
    print("=" * 60)
    print("  Spectro Multi-Fibres — serveur local")
    print(f"  Ouvrez http://{HOST}:{PORT} si le navigateur ne s'ouvre pas.")
    print("  Fermez cette fenetre pour arreter l'application.")
    print("=" * 60)
    threading.Timer(1.2, open_browser).start()
    app.run(host=HOST, port=PORT, debug=False)
