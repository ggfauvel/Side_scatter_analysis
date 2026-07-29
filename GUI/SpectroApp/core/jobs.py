"""
core/jobs.py — Batch d'extraction en arriere-plan.

Un seul job a la fois (suffisant pour un outil local). Le thread ecrit sa
progression dans PROGRESS ; l'interface la lit par polling (dcc.Interval).
Reprend la logique de la cellule 10 : skip si deja en cache, sauvegarde .npy
brute + CSV calibre en intensite via sf.save_spectra.
"""
from __future__ import annotations

import threading
import traceback
from pathlib import Path

import numpy as np

from core import spectro_functions as sf
from core.session import SESSION

PROGRESS = {
    "running": False, "cancel": False, "total": 0, "done": 0, "skipped": 0,
    "current": "", "errors": [], "finished_msg": "",
}
_lock = threading.Lock()


def _reset(total):
    with _lock:
        PROGRESS.update(running=True, cancel=False, total=total, done=0,
                        skipped=0, current="", errors=[], finished_msg="")


def _worker(shots: list[str], export_csv: bool, ignore_cache: bool = False):
    s = SESSION
    cache = s.cache_dir()
    out = s.outputs_dir() / "spectra_csv"
    if export_csv:
        out.mkdir(parents=True, exist_ok=True)
    n_done = n_skip = 0
    try:
        s.apply_params_to_sf()
        for k, key in enumerate(shots):
            if PROGRESS["cancel"]:
                break
            with _lock:
                PROGRESS["current"] = key
            cache_f = cache / f"{key}.npy"
            try:
                if cache_f.exists() and not ignore_cache:
                    n_skip += 1
                    spectra_k = None
                else:
                    from core import analysis as _an
                    err = _an.check_image_compat(key)
                    if err:
                        raise ValueError(err)
                    spectra_k, _, _ = sf.extract_all_spectra(
                        s.image_dict[key], s.calib,
                        subtract_bg=bool(s.params["SUBTRACT_BG"]))
                    np.save(cache_f, spectra_k)
                    n_done += 1
                if export_csv:
                    if spectra_k is None:
                        spectra_k = np.load(cache_f)
                    sf.save_spectra(
                        sf.apply_intensity_calibration(
                            spectra_k, s.int_calib_factors),
                        str(out / key), wl_axis=s.calib["wl_axis"])
            except Exception as e:
                with _lock:
                    PROGRESS["errors"].append(f"{key} : {e}")
            with _lock:
                PROGRESS["done"] = k + 1
                PROGRESS["skipped"] = n_skip
    except Exception:
        with _lock:
            PROGRESS["errors"].append(traceback.format_exc(limit=2))
    finally:
        msg = (f"Done: {n_done} extracted, {n_skip} already cached"
               + (f", {len(PROGRESS['errors'])} errors" if PROGRESS["errors"] else "")
               + ("." if not PROGRESS["cancel"] else " (interrompu)."))
        with _lock:
            PROGRESS["running"] = False
            PROGRESS["finished_msg"] = msg
        SESSION.log_history("batch", {"n_asked": len(shots), "n_done": n_done,
                                      "n_skip": n_skip,
                                      "n_err": len(PROGRESS["errors"])})


def start_batch(shots: list[str], export_csv: bool = True,
                ignore_cache: bool = False) -> tuple[bool, str]:
    if PROGRESS["running"] or SNR_PROGRESS["running"]:
        return False, "A run is already in progress."
    if SESSION.calib is None:
        return False, "Effectuez d'abord la calibration."
    if not shots:
        return False, "No image to process."
    _reset(len(shots))
    threading.Thread(target=_worker, args=(shots, export_csv, ignore_cache),
                     daemon=True).start()
    return True, f"Processing of {len(shots)} images started."


# ── Gestion du cache ─────────────────────────────────────────────────────────
def cache_inventory() -> list[dict]:
    """Liste tous les dossiers de cache du workspace (celui de la calibration
    courante et les anciens), avec taille et nombre de fichiers."""
    import shutil  # noqa: F401  (reste dispo pour clear)
    ws = SESSION.ensure_workspace()
    current = f"cache_{SESSION.calib_hash()}"
    out = []
    for d in sorted(ws.glob("cache_*")):
        if not d.is_dir():
            continue
        files = list(d.glob("*.npy"))
        size = sum(f.stat().st_size for f in files)
        meta = SESSION.cache_meta(d)
        out.append({"name": d.name, "path": d, "n": len(files),
                    "size_mb": size / 1e6,
                    "current": d.name == current,
                    "in_use": bool(SESSION.cache_override)
                    and Path(SESSION.cache_override) == d,
                    "meta": meta,
                    # Un cache est reutilisable en confiance s'il a ete
                    # produit pour la MEME campagne. Sinon il reste
                    # adoptable, mais sous avertissement.
                    "same_campaign": bool(meta.get("campaign_id"))
                    and meta.get("campaign_id") == SESSION.campaign_key()})
    return out


def adopt_cache(path) -> str:
    """Utiliser explicitement un dossier de cache deja rempli.

    La cle de calibration peut changer pour des raisons exterieures a la
    physique (fichier HgAr recopie, campagne deplacee). Un cache valide
    devenait alors inutilisable et il fallait tout re-extraire. On laisse
    donc l'utilisateur le designer, apres l'avoir informe de sa provenance.
    """
    d = Path(path)
    if not d.is_dir():
        return "Folder not found."
    if PROGRESS["running"]:
        return "Not possible: a run is in progress."
    SESSION.cache_override = "" if d.name == f"cache_{SESSION.calib_hash()}" \
        else str(d)
    SESSION.save_config()
    n = len(list(d.glob("*.npy")))
    return (f"Cache '{d.name}' now in use ({n} spectra). "
            f"Extraction will skip these shots.")


def clear_cache(which: str = "current") -> str:
    """which : 'current' (cache de la calibration en cours), 'others'
    (anciens caches), 'all'. Retourne un message de bilan."""
    import shutil
    if PROGRESS["running"] or SNR_PROGRESS["running"]:
        return "Not possible: a run is in progress."
    inv = cache_inventory()
    n_dirs = n_files = 0
    for item in inv:
        hit = (which == "all"
               or (which == "current" and item["current"])
               or (which == "others" and not item["current"]))
        if hit:
            shutil.rmtree(item["path"], ignore_errors=True)
            n_dirs += 1
            n_files += item["n"]
    return f"{n_files} spectres supprimes ({n_dirs} dossier(s) de cache)."


# ── Tache SNR (carte SNR sur toutes les images extraites) ────────────────────
SNR_PROGRESS = {"running": False, "cancel": False, "total": 0, "done": 0,
                "errors": [], "finished_msg": "", "result": None}


def _snr_worker(shots: list[str]):
    import numpy as _np
    from core import analysis as _an
    rows, names = [], []
    try:
        for k, key in enumerate(shots):
            if SNR_PROGRESS["cancel"]:
                break
            try:
                sp = _an.get_spectra(key)
                rows.append([sf.compute_snr(sp[j]) for j in range(sp.shape[0])])
                names.append(key)
            except Exception as e:
                SNR_PROGRESS["errors"].append(f"{key} : {e}")
            SNR_PROGRESS["done"] = k + 1
    finally:
        SNR_PROGRESS["running"] = False
        if names:
            SNR_PROGRESS["result"] = (_np.array(rows, dtype=float), names)
            SNR_PROGRESS["finished_msg"] = (
                f"SNR map computed on {len(names)} images"
                + (f" ({len(SNR_PROGRESS['errors'])} errors)"
                   if SNR_PROGRESS["errors"] else "") + ".")
        else:
            SNR_PROGRESS["finished_msg"] = "No usable image."


def start_snr(shots: list[str]) -> tuple[bool, str]:
    if PROGRESS["running"] or SNR_PROGRESS["running"]:
        return False, "A run is already in progress."
    if not shots:
        return False, "Aucune image extraite : lancez d'abord l'extraction."
    SNR_PROGRESS.update(running=True, cancel=False, total=len(shots), done=0,
                        errors=[], finished_msg="", result=None)
    threading.Thread(target=_snr_worker, args=(shots,), daemon=True).start()
    return True, f"Calcul du SNR sur {len(shots)} images lance."


def cancel():
    with _lock:
        PROGRESS["cancel"] = True
