"""
core/session.py — État global de l'application (mono-utilisateur, local).

Choix assumé : l'application tourne en local pour UN utilisateur ; l'état
(calibration, tables, chemins) vit dans un singleton serveur plutôt que dans
des stores navigateur. C'est plus simple, plus rapide (la calibration est un
gros objet numpy non sérialisable proprement en JSON) et honnête vis-à-vis
de l'usage réel. Contrepartie documentée : ne PAS exposer ce serveur sur un
réseau multi-utilisateurs.
"""
from __future__ import annotations

import hashlib
import json
import re
import threading
from pathlib import Path

import numpy as np

from core import spectro_functions as sf

# ── Paramètres par défaut = cellule 2/4 du notebook, à l'identique ──────────
DEFAULT_PARAMS = {
    "N_FIBERS": 80,
    "HALF_WIDTH": 6,
    "BG_FILTER_SIZE": 55,
    "PEAK_MIN_PROM": 50,
    "WL_SHIFT_NM": 0.0,
    "WL_METHOD": "auto",          # "auto" (comb-matching) | "manual"
    "BG_COLUMN_GAP": 4,
    "SUBTRACT_BG": True,
    "USE_INT_CALIBRATION": False,
    "USE_ND_CORRECTION": False,   # divide spectra by ND filter transmission
    "USE_WL_AXIS": True,
    # Pre-rotation of images (degrees clockwise: 0/90/180/270), applied on
    # load. Different campaigns may export shots and calibration in different
    # orientations, so the two are independent.
    "SHOT_ROTATION": 0,
    "CALIB_ROTATION": 0,
    # Positions des fibres : "manual" = tableau du pipeline (campagne
    # actuelle), "auto" = détection automatique multi-couches (généralisable)
    "FIBER_MODE": "manual",
    "FIBER_AUTO_N_SCIENCE": 5,
    # Préférences d'affichage (appliquées à tous les graphiques interactifs)
    "PLOT_FONT_SIZE": 13,
    "PLOT_TEMPLATE": "plotly_white",
}

# Paires manuelles de secours — valeurs du notebook (cellule 2)
DEFAULT_WL_CALIB_PAIRS = [
    (733, 730.04), (1359, 809.32), (1409, 815.56), (1871, 871.66),
    (274, 667.73), (481, 696.54), (557, 706.72), (617, 714.7),
    (797, 738.4), (899, 750.925), (993, 763.51), (1063, 772.38),
    (1242, 794.82), (1289, 800.62), (1498, 826.45), (1622, 841.64),
    (1709, 852.14), (2212, 912.3), (2298, 922.45),
]

APP_DIR = Path(__file__).resolve().parent.parent
CONFIG_PATH = APP_DIR / "config.json"

_IMG_RE = re.compile(r"shot\s*_?\s*0*(\d+)\.tiff?$", re.IGNORECASE)


class Session:
    def __init__(self):
        self.lock = threading.Lock()
        # Chemins
        self.images_dir: str = ""
        self.hgar_path: str = ""
        self.excel_path: str = ""
        self.pulses_dir: str = ""
        self.workspace: str = ""
        # Paramètres du pipeline
        self.params = dict(DEFAULT_PARAMS)
        self.wl_calib_pairs = list(DEFAULT_WL_CALIB_PAIRS)
        # État calculé
        self.calib = None                    # dict retourné par sf.build_calibration
        self.int_calib_factors = None        # (80, n_px)
        self.calib_log: str = ""
        self.image_dict: dict[str, Path] = {}    # "shot431" -> Path
        self.energy_table: dict[int, float] = {}
        self.last_error: str = ""
        # Cache de la sonde de format des images (voir unusable_images)
        self._bad_img_key = None
        self._bad_img: dict = {}
        # Métadonnées de campagne (Final.xlsx) et configurations angulaires
        self.metadata: dict[int, dict] = {}
        # Fichiers de transmission ND : {"1": chemin, "2": chemin, ...}
        self.nd_files: dict[str, str] = {}
        # En-tete de la colonne ND a lire dans le shotbook (None = auto).
        # Les shotbooks de campagnes differentes ont des colonnes ND
        # differentes (SRS ND, side-SRS ND, Resolved-SSRS ND, ...).
        self.nd_column: str | None = None
        # Calibration absolue (ADU -> J) : facteurs par fibre + parametres.
        # Les gros tableaux par-lambda (A, B, tau) vivent dans abs_cal_arrays
        # (memoire seule, non persistes) et sont recalcules au besoin.
        self.abs_cal: dict | None = None
        # Parametres complets de la calibration absolue (geometrie du
        # wattmetre, ND, filtre passe-haut, lissage). Persistes pour que la
        # calibration soit reconstruite a l'identique au redemarrage.
        self.abs_cal_params: dict | None = None
        self.abs_cal_arrays: dict | None = None
        # Unites d'affichage des spectres : "adu" ou "uJ" (energie/nm).
        # "uJ" n'est possible qu'apres une calibration absolue.
        self.display_units: str = "adu"
        self.structure_path: str = ""
        # Fichier unique de configuration des fibres (format "table plate" :
        # une ligne par fibre avec ses angles et sa position)
        self.flat_angle_path: str = ""
        # Une campagne peut avoir PLUSIEURS plans de fibres (un par
        # configuration) : novembre en a trois. On garde donc une liste.
        self.flat_angle_paths: list[str] = []
        # Cache choisi explicitement par l'utilisateur (voir cache_dir).
        self.cache_override: str = ""
        # Identite de la campagne validee la derniere fois, pour detecter un
        # changement de campagne et purger ce qui ne lui appartient plus.
        self.campaign_id: str = ""
        self.angle_registry: dict = {}
        self.angle_report: list = []
        # Détection automatique des fibres (une fois par campagne)
        self.fiber_auto: dict | None = None      # {positions, conf, overrides,
        #                                           science_used, hgar_key}
        self.fiber_diag: dict | None = None      # diagnostics (RAM seulement)

    # ── Application des paramètres au module sf (comme la cellule 2) ────────
    def apply_params_to_sf(self):
        p = self.params
        sf.N_FIBERS = int(p["N_FIBERS"])
        sf.HALF_WIDTH = int(p["HALF_WIDTH"])
        sf.BG_FILTER_SIZE = int(p["BG_FILTER_SIZE"])
        sf.PEAK_MIN_PROM = float(p["PEAK_MIN_PROM"])
        sf.WL_SHIFT_NM = float(p["WL_SHIFT_NM"])
        sf.WL_METHOD = str(p["WL_METHOD"])
        sf.BG_COLUMN_GAP = int(p["BG_COLUMN_GAP"])
        sf.WL_CALIB_PAIRS = [tuple(x) for x in self.wl_calib_pairs]

    # ── Scan des images science ──────────────────────────────────────────────
    def scan_images(self) -> dict:
        """Scanne images_dir pour tous les TIFF 'shot<N>' (préfixes tolérés).
        Retourne un résumé {n, first, last, duplicates}."""
        self.image_dict = {}
        dups = []
        d = Path(self.images_dir) if self.images_dir else None
        if not d or not d.is_dir():
            return {"n": 0, "first": None, "last": None, "duplicates": [],
                    "error": "Images folder not found."}
        for f in sorted(d.iterdir()):
            if not f.is_file():
                continue
            m = _IMG_RE.search(f.name)
            if not m:
                continue
            key = f"shot{int(m.group(1)):03d}"
            if key in self.image_dict:
                dups.append(f.name)
                continue
            self.image_dict[key] = f
        keys = sorted(self.image_dict)
        return {"n": len(keys), "first": keys[0] if keys else None,
                "last": keys[-1] if keys else None, "duplicates": dups}

    # ── Sonde des dimensions (lecture des en-têtes TIFF, rapide) ────────────
    def probe_image_sizes(self, max_files=None) -> dict:
        """Retourne {(largeur, hauteur): [shot_keys...]} pour les images
        science, via la lecture des seuls en-têtes TIFF (pas les pixels)."""
        from PIL import Image
        sizes: dict[tuple, list] = {}
        keys = sorted(self.image_dict)
        if max_files:
            keys = keys[:max_files]
        for k in keys:
            try:
                with Image.open(self.image_dict[k]) as im:
                    sizes.setdefault(im.size, []).append(k)
            except Exception:
                sizes.setdefault(("illisible", "?"), []).append(k)
        return sizes

    def probe_image_formats(self, max_files=None) -> dict:
        """{mode PIL: [shot_keys...]} — lecture des seuls en-tetes.

        Complete probe_image_sizes : deux images peuvent avoir exactement les
        memes dimensions et n'etre pas comparables pour autant (16 bits
        monochrome contre RGBA 8 bits re-exporte).
        """
        from PIL import Image
        modes: dict[str, list] = {}
        keys = sorted(self.image_dict)
        if max_files:
            keys = keys[:max_files]
        for k in keys:
            try:
                with Image.open(self.image_dict[k]) as im:
                    modes.setdefault(im.mode, []).append(k)
            except Exception:
                modes.setdefault("unreadable", []).append(k)
        return modes

    def unusable_images(self) -> dict:
        """{shot_key: raison} pour les images qui ne peuvent pas etre
        extraites (format non conforme). Le resultat est mis en cache et
        invalide des que la liste des images change : la sonde ouvre les
        en-tetes de tous les TIFF du dossier, il ne faut pas la relancer a
        chaque construction de page.
        """
        from core import analysis
        key = tuple(sorted(self.image_dict))
        if getattr(self, "_bad_img_key", None) == key:
            return self._bad_img
        bad = {}
        try:
            for mode, keys in self.probe_image_formats().items():
                why = ("unreadable file" if mode == "unreadable"
                       else analysis.image_format_problem(mode))
                if why:
                    for k in keys:
                        bad[k] = why
        except Exception:
            # Un incident de sondage (droits, disque reseau, PIL) ne doit pas
            # empecher d'ouvrir la page : on retombe sur « rien de signale »
            # plutot que de tout bloquer.
            bad = {}
        self._bad_img_key = key
        self._bad_img = bad
        return bad

    def hgar_size(self):
        from PIL import Image
        if not self.hgar_path or not Path(self.hgar_path).exists():
            return None
        try:
            with Image.open(self.hgar_path) as im:
                return im.size  # (largeur, hauteur)
        except Exception:
            return None

    # ── Résolution flexible des CSV de pulse ────────────────────────────────
    def resolve_pulse_csv(self, shot_num: int) -> Path | None:
        """Essaie les conventions de nommage rencontrées :
        'shot 431.csv', 'shot_431.csv', 'shot431.csv', variantes zéro-paddées."""
        if not self.pulses_dir:
            return None
        d = Path(self.pulses_dir)
        if not d.is_dir():
            return None
        candidates = [
            f"shot {shot_num}.csv", f"shot_{shot_num}.csv", f"shot{shot_num}.csv",
            f"shot {shot_num:03d}.csv", f"shot_{shot_num:03d}.csv",
            f"shot{shot_num:03d}.csv",
        ]
        for c in candidates:
            p = d / c
            if p.exists():
                return p
        # dernier recours : regex sur le dossier
        rx = re.compile(rf"shot\s*_?\s*0*{shot_num}\.csv$", re.IGNORECASE)
        for f in d.iterdir():
            if rx.search(f.name):
                return f
        return None

    def list_pulse_shots(self) -> list[int]:
        """Numéros de shots pour lesquels un CSV de pulse existe."""
        out = set()
        if not self.pulses_dir:
            return []
        d = Path(self.pulses_dir)
        if not d.is_dir():
            return []
        rx = re.compile(r"shot\s*_?\s*0*(\d+)\.csv$", re.IGNORECASE)
        for f in d.iterdir():
            m = rx.search(f.name)
            if m:
                out.add(int(m.group(1)))
        return sorted(out)

    # ── Identite de campagne et remise a zero ───────────────────────────────
    def campaign_key(self) -> str:
        """Empreinte de la campagne : dossier d'images + HgAr + shotbook.

        Deux campagnes differentes ne peuvent pas partager une calibration,
        une table d'energies ni une calibration absolue. Cette cle sert a
        detecter le changement et a purger ce qui ne suit pas.
        """
        h = hashlib.md5()
        for v in (self.images_dir, self.hgar_path, self.excel_path):
            h.update(str(v or "").encode())
        return h.hexdigest()[:12]

    def reset_computed_state(self) -> list[str]:
        """Efface tout ce qui a ete CALCULE pour une campagne, en gardant les
        chemins et les preferences.

        Sans cela, changer de dossier d'images laissait en place la
        calibration, la table d'energies, les metadonnees et la calibration
        absolue de la campagne precedente : les resultats etaient un melange
        des deux et il fallait redemarrer l'application pour s'en sortir.
        """
        cleared = []
        if self.calib is not None:
            cleared.append("wavelength calibration")
        if self.abs_cal is not None:
            cleared.append("absolute calibration")
        if self.energy_table:
            cleared.append("energy table")
        if self.metadata:
            cleared.append("campaign metadata")
        if self.angle_registry:
            cleared.append("angular configurations")
        if self.fiber_auto:
            cleared.append("automatic fiber positions")
        self.calib = None
        self.int_calib_factors = None
        self.calib_log = ""
        self.energy_table = {}
        self.metadata = {}
        self.abs_cal = None
        self.abs_cal_params = None
        self.abs_cal_arrays = None
        self.display_units = "adu"
        self.angle_registry = {}
        self.angle_report = []
        self.fiber_auto = None
        self.fiber_diag = None
        self.cache_override = ""
        self._bad_img_key = None
        self._bad_img = {}
        self.last_error = ""
        return cleared

    # ── Hash de calibration → clé de cache ──────────────────────────────────
    def calib_hash(self) -> str:
        """Identifie de manière unique (calibration + mode d'extraction).
        Amélioration par rapport au notebook : le cache .npy y était réutilisé
        même si la calibration changeait. Ici, tout changement de paramètre ou
        d'image HgAr invalide le cache automatiquement."""
        h = hashlib.md5()
        hg = Path(self.hgar_path) if self.hgar_path else None
        if hg and hg.exists():
            h.update(f"{hg.name}|{self._file_digest(hg)}".encode())
        payload = {k: self.params[k] for k in sorted(self.params)
                   if k not in ("USE_INT_CALIBRATION", "USE_WL_AXIS",
                                "USE_ND_CORRECTION",
                                "PLOT_FONT_SIZE", "PLOT_TEMPLATE",
                                "FIBER_AUTO_N_SCIENCE")}
        payload["pairs"] = self.wl_calib_pairs
        # Les positions effectives des fibres font partie de l'identite du
        # cache : mode auto + corrections manuelles incluses.
        if self.params.get("FIBER_MODE") == "auto" and self.fiber_auto:
            pos = list(self.fiber_auto.get("positions", []))
            for k, v in (self.fiber_auto.get("overrides") or {}).items():
                pos[int(k)] = float(v)
            payload["fiber_positions"] = [round(float(p), 3) for p in pos]
        h.update(json.dumps(payload, sort_keys=True).encode())
        return h.hexdigest()[:10]

    _digest_cache: dict = {}

    @classmethod
    def _file_digest(cls, path: Path) -> str:
        """Empreinte du CONTENU d'un fichier.

        La cle de cache utilisait la date de modification. Copier le fichier
        HgAr, restaurer une sauvegarde ou deplacer le dossier de campagne
        changeait donc la cle, et un cache parfaitement valide devenait
        inutilisable sans que rien ne l'explique. Le contenu, lui, ne bouge
        pas. Le resultat est memorise pour ne lire le fichier qu'une fois.
        """
        st = path.stat()
        key = (str(path), st.st_size, int(st.st_mtime))
        hit = cls._digest_cache.get(key)
        if hit:
            return hit
        h = hashlib.md5()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        out = h.hexdigest()[:16]
        cls._digest_cache[key] = out
        return out

    # ── Dossiers de travail ──────────────────────────────────────────────────
    def ensure_workspace(self) -> Path:
        ws = Path(self.workspace) if self.workspace else (APP_DIR / "workspace")
        ws.mkdir(parents=True, exist_ok=True)
        self.workspace = str(ws)
        return ws

    def cache_dir(self) -> Path:
        """Dossier de cache courant.

        Si l'utilisateur a explicitement adopte un ancien cache, c'est celui-la
        qui sert : un cache deja rempli doit pouvoir etre reutilise, meme
        quand la cle de calibration a change pour une raison exterieure
        (fichier recopie, dossier deplace).
        """
        if self.cache_override:
            d = Path(self.cache_override)
            if d.is_dir():
                return d
            self.cache_override = ""
        d = self.ensure_workspace() / f"cache_{self.calib_hash()}"
        d.mkdir(parents=True, exist_ok=True)
        self.write_cache_meta(d)
        return d

    def cache_meta(self, d: Path) -> dict:
        """Provenance d'un dossier de cache (fiche meta.json)."""
        try:
            return json.loads((Path(d) / "meta.json").read_text("utf-8"))
        except Exception:
            return {}

    def write_cache_meta(self, d: Path):
        """Note a cote des .npy ce qui les a produits.

        Un cache anonyme ne peut etre ni reconnu ni reutilise en confiance.
        Avec cette fiche, l'application peut dire a quelle campagne et a
        quelle calibration un dossier appartient, et prevenir avant de le
        reutiliser a tort.
        """
        f = Path(d) / "meta.json"
        meta = {
            "calib_hash": self.calib_hash(),
            "campaign_id": self.campaign_key(),
            "images_dir": self.images_dir,
            "hgar": Path(self.hgar_path).name if self.hgar_path else "",
            "n_fibers": int(self.params.get("N_FIBERS", 0)),
            "fiber_mode": self.params.get("FIBER_MODE", ""),
        }
        if f.exists():
            old = self.cache_meta(d)
            meta.setdefault("created", old.get("created"))
            if old == {**old, **meta}:
                return
        import datetime
        meta.setdefault("created",
                        datetime.datetime.now().isoformat(timespec="seconds"))
        try:
            f.write_text(json.dumps(meta, indent=2), encoding="utf-8")
        except Exception:
            pass

    def outputs_dir(self) -> Path:
        d = self.ensure_workspace() / "outputs"
        d.mkdir(parents=True, exist_ok=True)
        return d

    # ── Persistance de la configuration ─────────────────────────────────────
    def save_config(self):
        cfg = {
            "images_dir": self.images_dir, "hgar_path": self.hgar_path,
            "excel_path": self.excel_path, "pulses_dir": self.pulses_dir,
            "workspace": self.workspace,
            "structure_path": self.structure_path,
            "flat_angle_path": self.flat_angle_path,
            "flat_angle_paths": self.flat_angle_paths,
            "campaign_id": self.campaign_id,
            "cache_override": self.cache_override,
            "params": self.params, "wl_calib_pairs": self.wl_calib_pairs,
            "nd_files": self.nd_files, "nd_column": self.nd_column,
            "abs_cal": self.abs_cal, "display_units": self.display_units,
            "abs_cal_params": self.abs_cal_params,
        }
        CONFIG_PATH.write_text(json.dumps(cfg, indent=2), encoding="utf-8")

    def load_config(self):
        if not CONFIG_PATH.exists():
            return
        try:
            cfg = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
        except Exception:
            return
        self.images_dir = cfg.get("images_dir", "")
        self.hgar_path = cfg.get("hgar_path", "")
        self.excel_path = cfg.get("excel_path", "")
        self.pulses_dir = cfg.get("pulses_dir", "")
        self.workspace = cfg.get("workspace", "")
        self.structure_path = cfg.get("structure_path", "")
        self.flat_angle_path = cfg.get("flat_angle_path", "")
        self.flat_angle_paths = list(cfg.get("flat_angle_paths") or
                                     ([self.flat_angle_path]
                                      if self.flat_angle_path else []))
        self.campaign_id = cfg.get("campaign_id", "")
        self.cache_override = cfg.get("cache_override", "")
        self.nd_files = dict(cfg.get("nd_files", {}) or {})
        self.nd_column = cfg.get("nd_column")
        self.abs_cal = cfg.get("abs_cal")
        self.abs_cal_params = cfg.get("abs_cal_params")
        self.display_units = cfg.get("display_units", "adu")
        self.params.update(cfg.get("params", {}))
        pairs = cfg.get("wl_calib_pairs")
        if pairs:
            self.wl_calib_pairs = [tuple(p) for p in pairs]

    # ── Journal d'analyses (historique) ──────────────────────────────────────
    def log_history(self, kind: str, payload: dict):
        """Journal JSONL : une ligne par action d'analyse (traçabilité)."""
        import datetime
        rec = {"ts": datetime.datetime.now().isoformat(timespec="seconds"),
               "kind": kind, **payload}
        p = self.ensure_workspace() / "history.jsonl"
        with open(p, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")

    def read_history(self, n=200) -> list[dict]:
        p = self.ensure_workspace() / "history.jsonl"
        if not p.exists():
            return []
        lines = p.read_text(encoding="utf-8").strip().splitlines()[-n:]
        out = []
        for ln in reversed(lines):
            try:
                out.append(json.loads(ln))
            except Exception:
                pass
        return out


    def hgar_key(self) -> str:
        hg = Path(self.hgar_path) if self.hgar_path else None
        if not hg or not hg.exists():
            return ""
        st = hg.stat()
        return f"{hg.name}|{st.st_size}|{int(st.st_mtime)}"

    def fiber_auto_path(self) -> Path:
        return self.ensure_workspace() / "fiber_auto.json"

    def load_fiber_auto(self):
        p = self.fiber_auto_path()
        if not p.exists():
            self.fiber_auto = None
            return
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            self.fiber_auto = None
            return
        # invalide si l'image HgAr a change
        if data.get("hgar_key") != self.hgar_key():
            self.fiber_auto = None
            return
        self.fiber_auto = data

    def save_fiber_auto(self):
        if self.fiber_auto is None:
            return
        self.fiber_auto_path().write_text(
            json.dumps(self.fiber_auto, indent=1), encoding="utf-8")

    def load_angle_registry(self):
        """Charge le registre de configurations angulaires du workspace."""
        from core import angles as _angles
        try:
            self.angle_registry = _angles.load_registry(self.ensure_workspace())
        except Exception:
            self.angle_registry = {}


SESSION = Session()
SESSION.load_config()
if SESSION.workspace:
    SESSION.load_angle_registry()
