"""
core/analysis.py — Orchestration des calculs.

REGLE ABSOLUE de ce module : aucun calcul scientifique n'est re-implemente.
Tout passe par core.spectro_functions (sf), avec les MEMES parametres et les
MEMES enchainements que le notebook. Ce module ne fait que :
  - materialiser l'etat inter-cellules du notebook (session),
  - gerer le cache .npy (identique au notebook, mais invalide par hash),
  - repliquer a l'identique deux helpers definis DANS le notebook et non dans
    sf : le fit polynomial + IC (cellule "_poly_fit_and_plot") et le fit
    exponentiel + IC (cellule 66), ainsi que la constante BEAM_AREA_CM2.
"""
from __future__ import annotations

import io
import re
import warnings
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np
from scipy.optimize import curve_fit
from scipy.special import gamma as _gamma
from scipy.stats import t as student_t

from core import spectro_functions as sf
from core.session import SESSION

# ── Image normalisation hook (grayscale + user rotation) ─────────────────────
# The scientific pipeline loads every image through sf.load_image(path). We
# wrap that single function (WITHOUT touching spectro_functions.py) so that,
# on ANY campaign:
#   • multi-channel images (RGB/RGBA exported by some acquisition tools) are
#     collapsed to grayscale — the pipeline expects a 2-D detector array;
#   • images are pre-rotated by a user-chosen multiple of 90° so a campaign
#     whose shots (or calibration) were exported sideways can still be
#     analysed. Calibration and science shots have independent rotations
#     because they are sometimes exported in different orientations.
# The wrapper is a strict no-op when images are already 2-D and rotations are
# 0 (the original campaign), so it cannot change any validated result.
import os as _os

_ORIG_LOAD_IMAGE = sf.load_image


def _to_grayscale(arr: np.ndarray) -> np.ndarray:
    if arr.ndim == 3:
        # average the colour channels, dropping a 4th (alpha) channel if any
        arr = arr[..., :3].astype(np.float64).mean(axis=-1)
    return arr.astype(np.float64)


def _rotation_k_for(path) -> int:
    """np.rot90 k (number of 90° CCW turns) for a given image path, from the
    user setting. Calibration image uses CALIB_ROTATION, everything else
    SHOT_ROTATION. Stored settings are in degrees clockwise."""
    s = SESSION
    deg = s.params.get("SHOT_ROTATION", 0) or 0
    try:
        hg = s.hgar_path
        if hg and _os.path.normcase(_os.path.abspath(str(path))) == \
                _os.path.normcase(_os.path.abspath(str(hg))):
            deg = s.params.get("CALIB_ROTATION", 0) or 0
    except Exception:
        pass
    try:
        deg = int(deg) % 360
    except (TypeError, ValueError):
        deg = 0
    return (-(deg // 90)) % 4   # clockwise degrees -> CCW turns for np.rot90


def _load_image_normalized(path):
    arr = _to_grayscale(_ORIG_LOAD_IMAGE(path))
    k = _rotation_k_for(path)
    if k:
        arr = np.ascontiguousarray(np.rot90(arr, k))
    return arr


# Install the hook once. Internal sf.* functions call load_image by the module
# global name, so rebinding the attribute covers every code path (calibration,
# extraction, previews, fiber detection).
sf.load_image = _load_image_normalized

# ── Constante faisceau (cellule 65, a l'identique) ───────────────────────────
# I(r) = I0 * exp(-(r/w)^(2n)) ; A_eff = (pi/n) * w^2 * Gamma(1/n)
SG_N = 8
SG_W_CM = 111e-4  # 111 um -> cm
BEAM_AREA_CM2 = (np.pi / SG_N) * SG_W_CM**2 * _gamma(1.0 / SG_N)


def beam_area_cm2(sg_n: float = SG_N, sg_w_um: float = 111.0) -> float:
    """EXACT effective area of a super-Gaussian focal spot.

    Beam profile convention (identical to the pipeline, cell 65):
        I(r) = I0 * exp(-(r/w)^(2n))
    The effective area is defined so that  E = I_peak * A_eff * dt, i.e.
        A_eff = integral(I dA) / I0 = 2*pi * int_0^inf exp(-(r/w)^(2n)) r dr
              = (pi/n) * w^2 * Gamma(1/n)          [closed form, exact]
    With the defaults n=8, w=111 um this reproduces BEAM_AREA_CM2.
    """
    n = float(sg_n)
    w_cm = float(sg_w_um) * 1e-4
    if n <= 0 or w_cm <= 0:
        raise ValueError("super-Gaussian order and radius must be > 0")
    return (np.pi / n) * w_cm**2 * _gamma(1.0 / n)


# ── Positions des fibres ─────────────────────────────────────────────────────
# Copie immaculee du tableau manuel du pipeline (pour pouvoir en sortir)
_FIBER_Y_PIPELINE = sf.FIBER_Y_MANUAL.copy()


def effective_fiber_positions() -> np.ndarray | None:
    """Positions a utiliser selon le mode. None = tableau du pipeline."""
    s = SESSION
    if s.params.get("FIBER_MODE") != "auto":
        return None
    if s.fiber_auto is None:
        s.load_fiber_auto()
    if s.fiber_auto is None:
        return None
    pos = np.array(s.fiber_auto["positions"], float)
    for k, v in (s.fiber_auto.get("overrides") or {}).items():
        pos[int(k)] = float(v)
    return np.sort(pos)


def run_fiber_detection(n_science: int | None = None) -> tuple[bool, str]:
    """Detection automatique des fibres sur l'image HgAr redressee,
    renforcee par n_science images de tir reparties sur la campagne.
    Calcul unique par campagne : resultat persiste dans le workspace."""
    from core import fiberdetect as fd
    s = SESSION
    if not s.hgar_path or not Path(s.hgar_path).exists():
        return False, "HgAr image not found."
    if n_science is None:
        n_science = int(s.params.get("FIBER_AUTO_N_SCIENCE", 5))
    s.apply_params_to_sf()
    try:
        arr_raw = sf.load_image(s.hgar_path).astype(float)
        arr = sf._rotate_image(arr_raw, sf._measure_image_rotation(arr_raw))
        sci, sci_names = [], []
        keys = sorted(s.image_dict)
        if n_science > 0 and keys:
            step = max(1, len(keys) // n_science)
            for k in keys[::step][:n_science]:
                try:
                    a = sf.load_image(s.image_dict[k]).astype(float)
                    if a.shape != arr.shape:
                        continue
                    sci.append(sf._rotate_image(
                        a, sf._measure_image_rotation(a)))
                    sci_names.append(k)
                except Exception:
                    continue
        pos, conf, diag = fd.detect_fibers_auto(
            arr, n_fibers=int(s.params["N_FIBERS"]), science_arrays=sci)
    except Exception as e:
        return False, f"Detection failed: {e}"
    s.fiber_auto = {
        "hgar_key": s.hgar_key(),
        "positions": [round(float(p), 3) for p in pos],
        "conf": [str(c) for c in conf],
        "overrides": {},
        "science_used": sci_names,
        # diagnostics compacts persistes pour que les plots de verification
        # survivent au redemarrage de l'application
        "diag": {
            "profile": [round(float(v), 1) for v in diag["profile"]],
            "band": [int(diag["band"][0]), int(diag["band"][1])],
            "period": float(diag["period"]),
            "candidates": [round(float(c), 1) for c in diag["candidates"]],
            "n_science": int(diag.get("n_science", 0)),
        },
    }
    s.save_fiber_auto()
    s.fiber_diag = diag
    n_dir = sum(1 for c in conf if str(c).startswith("directe"))
    s.log_history("detection_fibres", {
        "n_fibres": len(pos), "directes": n_dir,
        "science": sci_names, "periode_px": round(diag["period"], 2)})
    return True, (f"{len(pos)} fibers detected ({n_dir} direct, "
                  f"{len(pos) - n_dir} enfouies/interpolees a verifier).")


def get_fiber_diag() -> dict | None:
    """Diagnostics de detection : version RAM si disponible (detection faite
    dans cette session), sinon reconstruction depuis fiber_auto.json (survit
    au redemarrage)."""
    s = SESSION
    if s.fiber_diag is not None:
        return s.fiber_diag
    if s.fiber_auto is None:
        s.load_fiber_auto()
    if s.fiber_auto is None:
        return None
    d = s.fiber_auto.get("diag")
    if not d:
        return None
    return {"profile": np.array(d["profile"], float),
            "band": tuple(d["band"]), "period": float(d["period"]),
            "candidates": np.array(d["candidates"], float),
            "n_science": int(d.get("n_science", 0))}


def load_shot_derotated(shot_key: str) -> np.ndarray:
    """Image de tir redressee par SON propre angle (meme convention que
    l'extraction) : les positions master s'y superposent directement."""
    s = SESSION
    arr = sf.load_image(s.image_dict[shot_key]).astype(float)
    return sf._rotate_image(arr, sf._measure_image_rotation(arr))


def set_fiber_overrides(overrides: dict) -> None:
    s = SESSION
    if s.fiber_auto is None:
        return
    s.fiber_auto["overrides"] = {str(k): float(v)
                                 for k, v in overrides.items()}
    s.save_fiber_auto()


# ── Calibration ───────────────────────────────────────────────────────────────
def run_calibration() -> tuple[bool, str]:
    """Cellule 3 du notebook : build_calibration + calibration d'intensite.
    Capture la sortie texte pour l'afficher dans l'interface."""
    # Les graphiques deja affiches deviennent faux si la calibration
    # change : on vide les resultats memorises, pas les reglages.
    try:
        from core import uistate
        uistate.clear_outputs()
    except Exception:
        pass

    s = SESSION
    if not s.hgar_path or not Path(s.hgar_path).exists():
        return False, "HgAr image not found: set it on the Data page."
    s.apply_params_to_sf()
    # Positions des fibres selon le mode (le pipeline lit FIBER_Y_MANUAL)
    if s.params.get("FIBER_MODE") == "auto":
        pos = effective_fiber_positions()
        if pos is None:
            ok, msg = run_fiber_detection()
            if not ok:
                return False, msg
            pos = effective_fiber_positions()
        if len(pos) != int(s.params["N_FIBERS"]):
            return False, (f"Detection: {len(pos)} fibers instead of "
                           f"{s.params['N_FIBERS']}.")
        sf.FIBER_Y_MANUAL = np.asarray(pos, float)
    else:
        sf.FIBER_Y_MANUAL = _FIBER_Y_PIPELINE.copy()
    buf = io.StringIO()
    try:
        with redirect_stdout(buf):
            calib = sf.build_calibration(
                s.hgar_path,
                n_fibers=int(s.params["N_FIBERS"]),
                half_width=int(s.params["HALF_WIDTH"]),
                wl_method=str(s.params["WL_METHOD"]),
            )
            if s.params["USE_INT_CALIBRATION"]:
                icf = sf.build_relative_intensity_calibration(
                    calib, ref_fiber=0, verbose=True)
            else:
                icf = np.ones((calib["n_fibers"], len(calib["wl_axis"])),
                              dtype=float)
    except Exception as e:  # message utilisateur explicite, log complet
        s.calib_log = buf.getvalue()
        s.last_error = f"{type(e).__name__}: {e}"
        return False, f"Calibration failed: {e}"
    with s.lock:
        s.calib = calib
        s.int_calib_factors = icf
        s.calib_log = buf.getvalue()
    s.log_history("calibration", {
        "hgar": Path(s.hgar_path).name, "hash": s.calib_hash(),
        "rms_nm": float(np.sqrt(np.mean(calib["wl_residuals"] ** 2))),
        "n_pairs": len(calib["wl_pairs"]),
        "wl_range": [float(calib["wl_axis"][0]), float(calib["wl_axis"][-1])],
    })
    return True, "Calibration complete."


def calib_summary() -> dict | None:
    s = SESSION
    if s.calib is None:
        return None
    c = s.calib
    wl = c["wl_axis"]
    return {
        "n_fibers": c["n_fibers"],
        "fiber_period": float(c.get("fiber_period", 0) or 0),
        "wl_min": float(wl[0]), "wl_max": float(wl[-1]),
        "dispersion": float((wl[-1] - wl[0]) / len(wl)),
        "rms": float(np.sqrt(np.mean(c["wl_residuals"] ** 2))),
        "n_pairs": len(c["wl_pairs"]),
        "wl_shift": float(SESSION.params["WL_SHIFT_NM"]),
        "hash": s.calib_hash(),
    }


def check_image_compat(shot_key: str) -> str | None:
    """Verifie que l'image science a la meme geometrie que la calibration.
    Retourne None si tout va bien, sinon un message d'erreur explicite.
    Rationale : la calibration pixel->nm et les positions de fibres sont
    definies sur la grille de l'image HgAr ; une image science acquise avec
    un binning different (ex : 1280 px de large contre 2560) ne peut pas etre
    analysee avec cette calibration — ni ici, ni dans le notebook d'origine."""
    from PIL import Image
    s = SESSION
    if s.calib is None:
        return "Calibration not performed."
    try:
        with Image.open(s.image_dict[shot_key]) as im:
            w, h = im.size
            mode = im.mode
    except Exception as e:
        return f"Unreadable image: {e}"
    n_wl = len(s.calib["wl_axis"])
    if w != n_wl:
        return (f"width {w} px incompatible with the calibration "
                f"({n_wl} px, HgAr image). Different binning or sensor — use "
                f"an HgAr image acquired in the same conditions as this "
                f"shot.")
    err = image_format_problem(mode)
    if err:
        return err
    return None


# Modes PIL d'une VRAIE image de detecteur : un seul canal, 16 ou 32 bits.
GRAY_MODES = {"I;16", "I;16B", "I;16L", "I;16N", "I", "F", "L"}


def image_format_problem(mode: str) -> str | None:
    """Message d'erreur si le TIFF n'est pas une image de detecteur brute.

    Motif (cas reel, shot135 d'une campagne) : un shot avait ete re-exporte
    en RGBA 8 bits au lieu du 16 bits monochrome d'origine. Les dimensions
    etaient identiques, donc le controle de largeur le laissait passer ; mais
    le tableau charge devient (H, W, 4) au lieu de (H, W) et les valeurs
    plafonnent a 255 au lieu de 65535. Selon le chemin, cela plantait
    l'extraction ou produisait un spectre ~257 fois trop faible, presente
    comme un resultat physique. On refuse donc explicitement, en nommant la
    cause, plutot que de laisser passer un nombre faux.
    """
    m = str(mode)
    if m in GRAY_MODES:
        if m == "L":
            return ("8-bit greyscale image (PIL mode 'L') where detector "
                    "images are 16-bit: levels are capped at 255 instead of "
                    "65535, so its intensity is not comparable with the "
                    "other shots. Re-export the original 16-bit TIFF.")
        return None
    n_ch = {"RGB": 3, "RGBA": 4, "LA": 2, "CMYK": 4, "P": 1, "1": 1}.get(m)
    return (f"image in '{m}' mode"
            + (f" ({n_ch} channels)" if n_ch and n_ch > 1 else "")
            + ": this is not a raw detector image but a re-export (colour "
              "and/or 8-bit). Values are capped at 255 instead of 65535 and "
              "the array has the wrong shape. Use the original 16-bit "
              "greyscale TIFF.")


# ── Extraction avec cache (logique cellule 10, invalidation par hash) ────────
def extract_cached(shot_key: str, use_cache: bool = True) -> np.ndarray:
    """Spectres BRUTS (avant calibration intensite), comme le cache du
    notebook. La calibration d'intensite est appliquee a la volee ensuite."""
    s = SESSION
    if s.calib is None:
        raise RuntimeError("Calibration not performed.")
    if shot_key not in s.image_dict:
        raise KeyError(f"Image '{shot_key}' not found in the folder.")
    cache_f = s.cache_dir() / f"{shot_key}.npy"
    if use_cache and cache_f.exists():
        return np.load(cache_f)
    err = check_image_compat(shot_key)
    if err:
        raise ValueError(err)
    s.apply_params_to_sf()
    spectra, _, _ = sf.extract_all_spectra(
        s.image_dict[shot_key], s.calib,
        subtract_bg=bool(s.params["SUBTRACT_BG"]))
    np.save(cache_f, spectra)
    return spectra


def get_spectra(shot_key: str, use_cache: bool = True) -> np.ndarray:
    """Intensity-calibrated spectra = what every analysis cell of the
    notebook manipulates. Optionally ND-corrected (applied on the fly, after
    the cache, exactly like the relative intensity calibration)."""
    raw = extract_cached(shot_key, use_cache=use_cache)
    sp = sf.apply_intensity_calibration(raw, SESSION.int_calib_factors)
    return apply_nd_correction(sp, shot_key)


def shot_nd_value(shot_key: str):
    """OD value of the ND filter for a shot (column 'side-SRS ND' of
    Final.xlsx), or None if unknown. Loads the metadata lazily."""
    s = SESSION
    sn = shot_num(shot_key)
    if sn is None:
        return None
    if not s.metadata:
        load_metadata()
    m = s.metadata.get(sn)
    return m.get("nd") if m else None


def apply_nd_correction(spectra: np.ndarray, shot_key: str) -> np.ndarray:
    """Divide the spectra by the ND filter transmission T(lambda) of the
    shot, when USE_ND_CORRECTION is on. Measured curve (interpolated from
    the user-supplied Excel file) when available, flat 10^-OD otherwise.
    No-op if the option is off, the OD is 0, or the shot is unknown."""
    s = SESSION
    if not s.params.get("USE_ND_CORRECTION"):
        return spectra
    if s.calib is None or s.calib.get("wl_axis") is None:
        return spectra
    nd = shot_nd_value(shot_key)
    if nd in (None, 0, 0.0):
        return spectra
    from core import ndfilters as ndf
    factor, _src = ndf.correction_factor(nd, s.calib["wl_axis"], s.nd_files)
    return spectra * factor[np.newaxis, :]


def nd_status_table() -> list[dict]:
    """One row per distinct OD value in the shotbook: number of shots,
    registered file, correction source actually used. For the Calibration
    page."""
    from core import ndfilters as ndf
    s = SESSION
    if not s.metadata:
        load_metadata()
    counts: dict[str, int] = {}
    for m in s.metadata.values():
        try:
            od = float(m.get("nd"))
        except (TypeError, ValueError):
            continue
        if od == 0.0:
            continue
        counts[f"{od:g}"] = counts.get(f"{od:g}", 0) + 1
    rows = []
    wl_axis = (s.calib or {}).get("wl_axis")
    for key in sorted(counts, key=float):
        path = s.nd_files.get(key, "")
        source, detail = "theory:10^-OD", ""
        if path:
            try:
                wl, tr = ndf.get_curve(path)
                source = "file"
                detail = (f"{wl.min():.0f}-{wl.max():.0f} nm, "
                          f"{len(wl)} points")
                if wl_axis is not None and (
                        wl_axis.min() < wl.min() - 1e-9
                        or wl_axis.max() > wl.max() + 1e-9):
                    source = "file+clamp"
            except ValueError as e:
                source = "theory:10^-OD (file unreadable)"
                detail = str(e)
        rows.append({"od": key, "n_shots": counts[key], "path": path,
                     "source": source, "detail": detail})
    return rows


def extract_full(shot_key: str):
    """Extraction complete (spectres calibres, positions, angle) pour
    l'inspection detaillee — equivalent cellule 7."""
    s = SESSION
    s.apply_params_to_sf()
    spectra_raw, fiber_y_rot, angle_deg = sf.extract_all_spectra(
        s.image_dict[shot_key], s.calib,
        subtract_bg=bool(s.params["SUBTRACT_BG"]))
    cache_f = s.cache_dir() / f"{shot_key}.npy"
    if not cache_f.exists():
        np.save(cache_f, spectra_raw)
    spectra = sf.apply_intensity_calibration(spectra_raw, s.int_calib_factors)
    return spectra, fiber_y_rot, angle_deg


def cached_shots() -> list[str]:
    d = SESSION.cache_dir()
    return sorted(p.stem for p in d.glob("shot*.npy"))


# ── Table d'energie ───────────────────────────────────────────────────────────
def load_energy() -> tuple[bool, str]:
    s = SESSION
    if not s.excel_path or not Path(s.excel_path).exists():
        return False, "Excel file not found."
    try:
        s.energy_table = sf.load_energy_table(s.excel_path)
    except Exception as e:
        return False, f"Could not read Excel: {e}"
    if not s.energy_table:
        return False, ("No energy read. Check that column C contains "
                       "les numeros de shot et la colonne F l'energie 2w (J).")
    return True, f"{len(s.energy_table)} energies loaded."


def shot_num(shot_key: str) -> int | None:
    m = re.search(r"(\d+)", shot_key)
    return int(m.group(1)) if m else None


def load_metadata() -> tuple[bool, str]:
    """Load the full shotbook table (profile, target, E2w, per-shot fiber
    config, ND value) for the automatic groups, the angular-configuration
    resolution and the ND correction."""
    from core import metadata as md
    s = SESSION
    if not s.excel_path or not Path(s.excel_path).exists():
        return False, "Excel file not found."
    try:
        s.metadata = md.load_final_table(s.excel_path,
                                         nd_column=s.nd_column)
    except Exception as e:
        return False, f"Could not read the metadata: {e}"
    n_cfg = sum(1 for m in s.metadata.values() if m.get("fiberpos"))
    return True, (f"{len(s.metadata)} shots in the shotbook "
                  f"({n_cfg} with fiber configuration).")


def resolve_config(shot_key: str) -> tuple[str | None, str]:
    """Configuration angulaire d'un shot. Priorite :
    1. colonne 'side-SRS fibrePos' de Final.xlsx (generalisable),
    2. plages de shots codees dans le pipeline (campagne actuelle).
    Verifie en outre que les deux sources sont d'accord quand elles existent
    toutes les deux (0 desaccord constate sur cette campagne).
    Retourne (nom_config | None, source)."""
    from core import angles as ang
    sn = shot_num(shot_key)
    if sn is None:
        return None, "unreadable number"
    known = {c.lower() for c in ang.known_configs(SESSION.angle_registry)}
    meta_cfg = None
    m = SESSION.metadata.get(sn)
    if m and m.get("fiberpos"):
        cand = m["fiberpos"].strip()
        if cand.lower() in known:
            meta_cfg = cand.lower() if cand.lower() in sf.FIBER_CONFIGS                 else cand
    range_cfg = sf.get_config_for_shot(sn)
    if meta_cfg and range_cfg and meta_cfg.lower() != range_cfg.lower():
        return meta_cfg, (f"Final.xlsx ({meta_cfg}) — ATTENTION : desaccord "
                          f"avec les plages du pipeline ({range_cfg})")
    if meta_cfg:
        return meta_cfg, "Final.xlsx"
    if range_cfg:
        return range_cfg, "plages du pipeline"
    return None, "no source"


def fiber_angles(config_name: str, sign_rules=None):
    """Angles (phis, thetas) indexes par indice d'extraction — pipeline exact
    pour les configs codees en dur, meme inversion physique pour les configs
    importees des Excel.

    `sign_rules` = {'theta': [quadrants], 'phi': [quadrants]} : convention de
    signe choisie par l'utilisateur, appliquee aux seules configurations
    importees (les configs du pipeline gardent leur chemin exact)."""
    from core import angles as ang
    return ang.get_fiber_angles_any(config_name, SESSION.angle_registry,
                                    sign_rules=sign_rules)


# ── Statistiques de groupe (cellules 11bis/12, a l'identique) ────────────────
def group_fiber_spectra(shots: list[str], fiber_idx: int):
    """Liste des spectres (1D) d'une fibre pour chaque shot disponible.

    Retourne aussi la liste des noms de shots REELLEMENT presents, dans le
    meme ordre que les spectres, pour que l'appelant puisse nommer un spectre
    aberrant sans se decaler quand un shot manque.
    """
    out, missing, present = [], [], []
    unreadable = []
    for name in shots:
        if name not in SESSION.image_dict:
            missing.append(name)
            continue
        try:
            sp = np.asarray(get_spectra(name)[fiber_idx], dtype=float)
        except Exception as e:
            # Un seul shot illisible (format, dimensions, fichier tronque) ne
            # doit pas emporter toute la comparaison : on le met de cote et
            # l'appelant le nomme. Avant, l'exception remontait et la page
            # entiere restait vide, ce qui se lisait comme « la detection ne
            # marche pas ».
            unreadable.append((name, str(e)))
            continue
        if sp.ndim != 1:
            unreadable.append((name, f"spectrum of shape {sp.shape} instead "
                                     f"of a 1D curve — non-conforming image"))
            continue
        out.append(sp)
        present.append(name)
    group_fiber_spectra.last_unreadable = unreadable
    return out, missing, present


group_fiber_spectra.last_unreadable = []


def mean_std(spectra_group: list[np.ndarray], kind: str = "std"):
    """Moyenne et deviation exactement comme plot_mean_and_deviation()."""
    if not spectra_group:
        return None, None
    data = np.vstack(spectra_group)
    mean_sp = np.nanmean(data, axis=0)
    if kind == "sem":
        n_v = np.sum(~np.isnan(data), axis=0)
        dev = np.nanstd(data, axis=0) / np.sqrt(np.maximum(n_v, 1))
    else:
        dev = np.nanstd(data, axis=0)
    return mean_sp, dev


def group_stack_all_fibers(shots: list[str], wl_mask: np.ndarray):
    """(n_images, n_fibers, n_wl) comme load_group_all_fibers (cellule 11bis-3D)."""
    stack = []
    for name in shots:
        if name not in SESSION.image_dict:
            continue
        try:
            sp = get_spectra(name)
        except Exception:
            continue                 # meme exclusion que group_fiber_spectra
        if sp.ndim != 2:
            continue
        stack.append(sp[:, wl_mask])
    if not stack:
        return None
    return np.array(stack)


# ── Fits (répliques exactes des helpers du notebook) ─────────────────────────
def poly_fit_ci(x, y, deg, conf_level=0.95):
    """Replique _poly_fit_and_plot : np.polyfit(cov=True), bande de confiance
    Var[y_hat] = V C V^T avec quantile de Student. Retourne un dict."""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    n = x.size
    if n < deg + 1:
        return {"ok": False, "msg": f"Pas assez de points ({n}) pour un degre {deg}."}
    try:
        RankWarning = np.RankWarning
    except AttributeError:
        try:
            from numpy.polynomial.polyutils import RankWarning
        except ImportError:
            class RankWarning(Warning):
                pass
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RankWarning)
        p, cov = np.polyfit(x, y, deg, cov=True)
    x_fit = np.linspace(x.min(), x.max(), 800)
    y_fit = np.polyval(p, x_fit)
    df = max(1, n - (deg + 1))
    tval = student_t.ppf(1.0 - (1 - conf_level) / 2.0, df)
    if cov is not None:
        V = np.vander(x_fit, N=deg + 1)
        var_pred = np.einsum("ij,jk,ik->i", V, cov, V)
        var_pred = np.where(var_pred < 0, 0.0, var_pred)
        std_pred = np.sqrt(var_pred)
    else:
        resid = y - np.polyval(p, x)
        std_pred = np.full_like(x_fit, np.sqrt(np.nansum(resid**2) / df))
    lower = y_fit - tval * std_pred
    upper = y_fit + tval * std_pred
    y_model = np.polyval(p, x)
    ss_res = np.nansum((y - y_model) ** 2)
    ss_tot = np.nansum((y - np.nanmean(y)) ** 2)
    R2 = 1.0 - ss_res / ss_tot if ss_tot != 0 else np.nan
    rmse = np.sqrt(ss_res / df)
    stderr = (np.sqrt(np.abs(np.diag(cov))) if cov is not None
              else np.full(p.shape, np.nan))
    # Equation lisible (texte simple, pas LaTeX : rendu Plotly)
    terms = []
    for i, coef in enumerate(p):
        pw = deg - i
        if abs(coef) < 1e-30:
            continue
        sgn = "+" if coef >= 0 else "-"
        ca = f"{abs(coef):.3g}"
        tc = ca if pw == 0 else (f"{ca}*x" if pw == 1 else f"{ca}*x^{pw}")
        terms.append((sgn, tc))
    if terms:
        eq = ("-" if terms[0][0] == "-" else "") + terms[0][1]
        for sgn, tc in terms[1:]:
            eq += f" {sgn} {tc}"
    else:
        eq = "0"
    return {"ok": True, "p": p, "cov": cov, "x_fit": x_fit, "y_fit": y_fit,
            "lower": lower, "upper": upper, "R2": float(R2),
            "rmse": float(rmse), "stderr": stderr, "tval": float(tval),
            "df": int(df), "eq": eq}


def _exp_model(x, a, b):
    return a * np.exp(b * x)


def exp_fit_ci(x, y, conf_level=0.95):
    """Replique _exp_fit_and_plot (cellule 66) : init par regression sur
    log(y), curve_fit, bande par propagation du jacobien + Student."""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    mask = np.isfinite(x) & np.isfinite(y) & (y > 0)
    x, y = x[mask], y[mask]
    if len(x) < 3:
        return {"ok": False, "msg": "Pas assez de points valides (>0) pour un fit exponentiel."}
    p_lin = np.polyfit(x, np.log(y), 1)
    a0, b0 = np.exp(p_lin[1]), p_lin[0]
    try:
        popt, pcov = curve_fit(_exp_model, x, y, p0=(a0, b0), maxfev=10000)
    except Exception as e:
        return {"ok": False, "msg": f"Fit exponentiel impossible : {e}"}
    a, b = popt
    n = len(x)
    dof = max(0, n - 2)
    xfit = np.linspace(np.min(x), np.max(x), 300)
    yfit = _exp_model(xfit, a, b)
    lower = upper = None
    if dof > 0 and np.all(np.isfinite(pcov)):
        tval = student_t.ppf(1.0 - (1 - conf_level) / 2.0, dof)
        expbx = np.exp(b * xfit)
        J = np.vstack([expbx, a * xfit * expbx]).T
        var_pred = np.einsum("ij,jk,ik->i", J, pcov, J)
        var_pred = np.maximum(var_pred, 0.0)
        delta = tval * np.sqrt(var_pred)
        lower, upper = yfit - delta, yfit + delta
    yhat = _exp_model(x, a, b)
    ss_res = np.sum((y - yhat) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
    eq = f"y = {a:.3e} * exp({b:.3e} * x)"
    return {"ok": True, "a": float(a), "b": float(b), "x_fit": xfit,
            "y_fit": yfit, "lower": lower, "upper": upper,
            "R2": float(r2), "eq": eq}


# ── Chaine energie / puissance / intensite crete ─────────────────────────────
def shot_power(shot_key: str, header_lines=25, frac_rise=0.03, frac_fall=0.05):
    """P_crete d'un shot : CSV pulse + E_2w (Final.xlsx), via les fonctions sf.
    Retourne (P_watts, message_erreur|None)."""
    s = SESSION
    sn = shot_num(shot_key)
    if sn is None:
        return np.nan, "unreadable shot number"
    csv = s.resolve_pulse_csv(sn)
    if csv is None:
        return np.nan, "pulse CSV not found"
    if sn not in s.energy_table:
        return np.nan, "E_2w missing from the Excel table"
    try:
        pdata = sf.load_pulse_profile(str(csv), header_lines=header_lines,
                                      frac_rise=frac_rise, frac_fall=frac_fall)
    except Exception as e:
        return np.nan, f"pulse read: {e}"
    P = sf.compute_pulse_power(pdata, s.energy_table[sn])
    if np.isnan(P):
        return np.nan, "power = NaN"
    return float(P), None


def correlation_dataset(shots: list[str], x_kind: str, fiber_idx,
                        wl_range=None, header_lines=25,
                        frac_rise=0.03, frac_fall=0.05,
                        beam_area=BEAM_AREA_CM2, units="adu"):
    """Construit (x, y, labels, skipped) pour les cellules 13-19/20-24.
    x_kind : 'energy' | 'power' | 'intensity'.
    y = aire spectrale de la fibre (ou moyenne des 80 si fiber_idx None)."""
    s = SESSION
    xs, ys, labels, skipped = [], [], [], []
    for name in shots:
        if name not in s.image_dict:
            skipped.append((name, "image missing from the folder"))
            continue
        sn = shot_num(name)
        if x_kind == "energy":
            if sn not in s.energy_table:
                skipped.append((name, "E_2w missing from the table"))
                continue
            xval = s.energy_table[sn]
        else:
            P, err = shot_power(name, header_lines, frac_rise, frac_fall)
            if err:
                skipped.append((name, err))
                continue
            xval = P if x_kind == "power" else P / beam_area
        sp = get_spectra(name)
        if units == "uJ":
            sp = to_absolute_energy(sp)
        areas = sf.compute_spectral_area(sp, s.calib["wl_axis"], wl_range=wl_range)
        yval = areas[fiber_idx] if fiber_idx is not None else float(areas.mean())
        xs.append(float(xval))
        ys.append(float(yval))
        labels.append(name)
    if xs:
        x = np.array(xs)
        y = np.array(ys)
        idx = np.argsort(x)
        x, y = x[idx], y[idx]
        labels = [labels[i] for i in idx]
    else:
        x = np.array([])
        y = np.array([])
    return x, y, labels, skipped


# ── Barycentre vs theta a phi fixe + exports (cellules 29/31, a l'identique) ─
def centroid_theta_profile(shot_key: str, phi_target: float, phi_tol: float,
                           wl_range=(600, 1000), bg_percentile=10,
                           sg_window_nm=5.0, config_name=None):
    s = SESSION
    from core import angles as ang
    sp = get_spectra(shot_key)
    if config_name is None:
        config_name, _src = resolve_config(shot_key)
    if config_name is None:
        raise ValueError(
            f"Undetermined angular configuration for {shot_key}. "
            f"Pick it manually among "
            f"{ang.known_configs(SESSION.angle_registry)}.")
    phis, thetas = fiber_angles(config_name)
    centroids = sf.compute_spectral_centroid(
        sp, s.calib["wl_axis"], wl_range=wl_range,
        bg_percentile=bg_percentile, savgol_window_nm=sg_window_nm)
    mask_phi = np.abs(phis - phi_target) <= phi_tol
    mask_valid = mask_phi & np.isfinite(thetas) & np.isfinite(centroids)
    if mask_valid.sum() == 0:
        phi_avail = np.unique(phis[np.isfinite(phis)])
        raise ValueError(
            f"No fiber for phi = {phi_target} +/- {phi_tol} deg. "
            f"Available values: {phi_avail}")
    theta_sel = thetas[mask_valid]
    centroid_sel = centroids[mask_valid]
    fiber_sel = np.where(mask_valid)[0]
    order = np.argsort(theta_sel)
    theta_sel, centroid_sel, fiber_sel = (theta_sel[order], centroid_sel[order],
                                          fiber_sel[order])
    phys_fiber_sel = sf.physical_fiber_index(fiber_sel)
    wl_axis = s.calib["wl_axis"]
    wl_mask = (wl_axis >= wl_range[0]) & (wl_axis <= wl_range[1])
    wl_export = wl_axis[wl_mask]
    spectra_export = sp[np.ix_(fiber_sel, wl_mask)]
    lambda_c_min = float(np.nanmin(centroid_sel))
    lambda_c_norm = centroid_sel - lambda_c_min
    return {
        "config": config_name, "theta": theta_sel, "centroid": centroid_sel,
        "fiber_idx": fiber_sel, "phys_fiber": phys_fiber_sel,
        "lambda_c_min": lambda_c_min, "lambda_c_norm": lambda_c_norm,
        "wl_export": wl_export, "spectra_export": spectra_export,
        "phi_target": float(phi_target), "phi_tol": float(phi_tol),
        "wl_range": tuple(wl_range), "bg_percentile": bg_percentile,
        "sg_window_nm": sg_window_nm, "image": shot_key,
    }


def export_centroid_npz_txt(res: dict, out_dir: Path) -> tuple[Path, Path]:
    """Exports .npz et .txt STRICTEMENT identiques a la cellule 31."""
    s = SESSION
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    npz_name = (f"{res['image']}_{res['config']}"
                f"_phi{int(res['phi_target']):+d}deg"
                f"_wl{res['wl_range'][0]}-{res['wl_range'][1]}nm.npz")
    npz_path = out_dir / npz_name
    meta = dict(
        image=res["image"], config=res["config"],
        phi_deg=float(res["phi_target"]), phi_tol_deg=float(res["phi_tol"]),
        wl_range_nm=list(res["wl_range"]), bg_percentile=res["bg_percentile"],
        sg_window_nm=res["sg_window_nm"],
        subtract_bg=bool(s.params["SUBTRACT_BG"]),
        int_calib=bool(s.params["USE_INT_CALIBRATION"]),
        description=(
            "spectra[i] = intensite calibree (ADU) de la fibre, a theta[i] degres, "
            "phi fixe. fiber_idx = indice d'extraction (ligne de la matrice spectra) ; "
            "phys_fiber = numero de fibre PHYSIQUE (le faisceau est monte inverse, "
            "phys = 79 - fiber_idx). lambda_c_norm = lambda_c - min(lambda_c)."),
    )
    np.savez(
        str(npz_path),
        wl_axis=res["wl_export"], spectra=res["spectra_export"],
        theta=res["theta"], phi=np.array(float(res["phi_target"])),
        fiber_idx=res["fiber_idx"], phys_fiber=res["phys_fiber"],
        lambda_c=res["centroid"], lambda_c_norm=res["lambda_c_norm"],
        meta=np.array(meta, dtype=object),
    )
    txt_path = npz_path.with_suffix(".txt")
    with open(txt_path, "w", encoding="utf-8") as fout:
        fout.write("=" * 70 + "\n")
        fout.write("EXPORT SPECTRES SRS\n")
        fout.write("=" * 70 + "\n\n")
        fout.write("[METADATA]\n")
        for k, v in meta.items():
            fout.write(f"  {k:<20} = {v}\n")
        fout.write("\n[CENTROID TABLE]\n")
        fout.write(f"  {'extr_idx':>9}  {'phys_fiber':>10}  {'theta (deg)':>12}  ")
        fout.write(f"{'lambda_c (nm)':>15}  {'lambda_c_norm (nm)':>20}\n")
        fout.write("  " + "-" * 74 + "\n")
        for fi, pf, th, lc, lcn in zip(res["fiber_idx"], res["phys_fiber"],
                                       res["theta"], res["centroid"],
                                       res["lambda_c_norm"]):
            fout.write(f"  {fi:>9}  {pf:>10}  {th:>12.4f}  {lc:>15.6f}  {lcn:>20.6f}\n")
        fout.write("\n[SPECTRA]\n")
        fout.write("  Une section par fibre, triees par theta croissant.\n")
        fout.write("  Colonnes : wl_nm <TAB> intensity_ADU\n\n")
        for fi, pf, th, sp_row in zip(res["fiber_idx"], res["phys_fiber"],
                                      res["theta"], res["spectra_export"]):
            fout.write(f"  --- extr_idx={fi}  phys_fiber={pf}  "
                       f"theta={th:.4f}deg  phi={res['phi_target']}deg ---\n")
            fout.write(f"  {'wl_nm':>12}\t{'intensity_ADU':>16}\n")
            for wl_val, i_val in zip(res["wl_export"], sp_row):
                fout.write(f"  {wl_val:>12.6f}\t{i_val:>16.6f}\n")
            fout.write("\n")
    return npz_path, txt_path


# ── Divers ────────────────────────────────────────────────────────────────────
def load_image_preview(path, max_dim=640):
    """Image 2D sous-echantillonnee POUR AFFICHAGE UNIQUEMENT (jamais pour un
    calcul). Retourne (array_reduit, facteur)."""
    arr = sf.load_image(path).astype(float)
    f = max(1, int(np.ceil(max(arr.shape) / max_dim)))
    return arr[::f, ::f], f


# ── Group pulse profiles (all pulses of a group + mean ± std) ────────────────
def group_pulse_profiles(shots: list[str], header_lines=25,
                         frac_rise=0.03, frac_fall=0.05, n_grid=600):
    """Load the windowed pulse profile of every shot in a group, align them
    on the rising edge (t = 0 at the frac_rise threshold crossing — the same
    windowing already used for the power computation), normalise each pulse
    to its own peak, and resample onto a common time grid to build the
    mean +/- standard-deviation envelope.

    Returns
    -------
    dict with:
        'pulses'  : list of (shot_key, t_ns, I_norm) — full resolution
        'grid_ns' : (n_grid,) common time axis (ns, 0 = rising edge)
        'matrix'  : (n_pulses, n_grid) resampled normalised pulses
                    (NaN outside each pulse's own window)
        'mean', 'std' : (n_grid,) computed over the valid pulses per bin
        'n_valid' : (n_grid,) number of pulses contributing per bin
        'skipped' : list of (shot_key, reason)
    or None if no pulse could be loaded.
    """
    s = SESSION
    pulses, skipped = [], []
    for name in shots:
        sn = shot_num(name)
        if sn is None:
            skipped.append((name, "unreadable shot number"))
            continue
        csv = s.resolve_pulse_csv(sn)
        if csv is None:
            skipped.append((name, "pulse CSV not found"))
            continue
        try:
            pdata = sf.load_pulse_profile(
                str(csv), header_lines=header_lines,
                frac_rise=frac_rise, frac_fall=frac_fall)
        except Exception as e:
            skipped.append((name, f"pulse read error: {e}"))
            continue
        t = np.asarray(pdata["time_pulse"], float)
        I = np.asarray(pdata["intensity_pulse"], float)
        if len(t) < 3 or np.max(I) <= 0:
            skipped.append((name, "empty or flat pulse window"))
            continue
        t_ns = (t - t[0]) * 1e9           # rising-edge alignment
        pulses.append((name, t_ns, I / np.max(I)))
    if not pulses:
        return None

    t_max = max(t[-1] for _, t, _ in pulses)
    grid = np.linspace(0.0, t_max, int(n_grid))
    mat = np.full((len(pulses), len(grid)), np.nan)
    for i, (_, t, I) in enumerate(pulses):
        inside = grid <= t[-1]
        mat[i, inside] = np.interp(grid[inside], t, I)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        mean = np.nanmean(mat, axis=0)
        std = np.nanstd(mat, axis=0)
    n_valid = np.sum(np.isfinite(mat), axis=0)
    return {"pulses": pulses, "grid_ns": grid, "matrix": mat,
            "mean": mean, "std": std, "n_valid": n_valid,
            "skipped": skipped}


# ── Absolute (physical) calibration ──────────────────────────────────────────
# Turns detector ADU into physical spectral energy. The calibration is a CURVE
# per fiber, C_i(lambda) in J/(nm.count), so that for any science image
#
#       E_i(lambda) [J/nm] = spectra_i(lambda) [ADU] * C_i(lambda)
#
# and no exposure time of the science image is involved: C already contains the
# division by the exposure time of the lamp image.
#
# Chain, per fiber i (see core/abscal.py for each step):
#   1. A_i(lambda)        after-fiber lamp image, background-subtracted [ADU]
#      /= T_ND(lambda)    ND filter in front of the spectrometer, if any
#      /= t_after         -> counts per second
#   2. d(lambda)          lamp spectrum measured BEFORE the fibers, normalised
#                         to unit area [1/nm]
#   3. c_pm               power-meter spectral correction R(set)/<R>_spectrum
#   4. P_i                power-meter reading -> power entering the fiber [W],
#                         including the sphere geometry (distance, aperture
#                         area, fiber core area)
#   5. f_lp               fraction of d(lambda) transmitted by the long-pass
#      d_lp = d.T_lp/f_lp normalised in-band density [1/nm]
#   6. C_i^raw(lambda)    = P_i.f_lp.d_lp(lambda) / (A_i(lambda)/t_after)
#   7. smoothing in log space, optional replacement of bad fibers by the mean
#      curve, hard zero below the physical cut-off.
#
# The historical single-scalar factor g_i is still exposed (median of C_i over
# the band, converted to uJ) so older code paths keep working, but every plot
# now uses the full matrix.
def _extract_after_fiber(image_path: str,
                         intensity_calib: bool = True) -> np.ndarray:
    """Spectres (ADU) de l'image lampe apres fibres, extraits EXACTEMENT
    comme les images science.

    Point critique. `get_spectra` applique aux tirs la calibration relative
    d'intensite (`int_calib_factors`). Si on ne l'applique pas ici, la
    calibration absolue est etablie sur une grandeur differente de celle a
    laquelle on l'applique ensuite, et le rapport ne se compense plus : tout
    facteur porte par ces coefficients se retrouve directement sur l'energie
    finale. Les deux chemins doivent donc subir le meme traitement.
    """
    s = SESSION
    if s.calib is None:
        raise RuntimeError("Run the wavelength/fiber calibration first.")
    s.apply_params_to_sf()
    spectra, _, _ = sf.extract_all_spectra(
        image_path, s.calib, subtract_bg=bool(s.params["SUBTRACT_BG"]))
    if intensity_calib and s.int_calib_factors is not None:
        spectra = sf.apply_intensity_calibration(spectra, s.int_calib_factors)
    return spectra


ABS_CAL_DEFAULTS = {
    # files
    "after_image": "", "spf2": "", "power_file": "",
    "background_image": "", "responsivity_file": "", "longpass_curve": "",
    # exposures
    "t_after_ms": 1000.0, "t_before_ms": 100.0,
    # power meter
    "power_unit": "uW", "power_set_nm": 750.0,
    "power_geometry": "sphere",
    # sphere geometry
    "sphere_radius_cm": 44.0, "sensor_closer_cm": 0.0,
    # Mode alternatif : on saisit les deux distances lampe->plan, plutot
    # qu'un rayon et un decalage. Beaucoup moins source d'erreur de signe.
    "dist_mode": "offset", "dist_unit": "cm",
    "dist_lamp_fiber": 44.0, "dist_lamp_meter": 44.0,
    "aperture_mm": 9.5, "core_um": 100.0,
    "fiber_na": 0.22, "apply_na": False,
    # ND filter on the lamp image
    "nd_od": 0.0,
    # long-pass filter
    "cut_nm": 650.0, "zero_below_nm": 660.0, "force_zero": True,
    # smoothing
    "smooth_method": "moving_average", "smooth_log": True,
    "smooth_window_nm": 10.0,
    # bad fibers
    "bad_fibers": [], "replace_bad": True,
}


def abs_cal_params() -> dict:
    """Current absolute-calibration parameters, defaults filled in."""
    p = dict(ABS_CAL_DEFAULTS)
    p.update(getattr(SESSION, "abs_cal_params", None) or {})
    return p


def build_absolute_calibration(**kw) -> tuple[bool, str]:
    """Build the per-fiber, per-wavelength ADU -> energy calibration.

    Every parameter is a keyword taken from ABS_CAL_DEFAULTS, so the caller
    (the page) can pass only what the user changed.
    """
    from core import abscal, ndfilters
    s = SESSION
    if s.calib is None:
        return False, "Run the wavelength/fiber calibration first."

    P = abs_cal_params()
    P.update({k: v for k, v in kw.items() if v is not None})

    # Tous les graphiques deja affiches en uJ/nm deviennent faux des que la
    # calibration absolue change. Sans cet appel, l'interface reaffichait la
    # version memorisee (voir core/uistate) et l'utilisateur croyait que le
    # nouveau calcul n'avait rien change.
    try:
        from core import uistate
        uistate.clear_outputs()
    except Exception:
        pass

    for label, key in [("after-fiber lamp image", "after_image"),
                       ("before-fiber .spf2 spectrum", "spf2"),
                       ("per-fiber power file", "power_file")]:
        f = str(P.get(key) or "").strip()
        if not f or not Path(f).exists():
            return False, f"Missing or unreadable {label}."

    wl = np.asarray(s.calib["wl_axis"], float)
    notes: list[str] = []

    # ── 1. lamp image -> counts per second ────────────────────────────────
    try:
        A = _extract_after_fiber(P["after_image"])
    except Exception as e:
        return False, f"Could not extract the after-fiber image: {e}"

    if P.get("background_image"):
        bgp = str(P["background_image"]).strip()
        if Path(bgp).exists():
            try:
                Abg = _extract_after_fiber(bgp)
                if Abg.shape == A.shape:
                    A = A - Abg
                    notes.append("Dark image subtracted fiber by fiber.")
                else:
                    notes.append("Dark image ignored: different shape.")
            except Exception as e:
                notes.append(f"Dark image ignored ({e}).")
        else:
            notes.append("Dark image ignored: file not found.")
    A = np.clip(np.asarray(A, float), 0.0, None)

    od = float(P.get("nd_od") or 0.0)
    nd_T = np.ones_like(wl)
    if od > 0:
        nd_T, src = ndfilters.transmission_on_axis(od, wl, s.nd_files or {})
        nd_T = np.clip(np.asarray(nd_T, float), 1e-12, None)
        A = A / nd_T[None, :]
        notes.append(f"ND OD {od:g} removed from the lamp image "
                     f"({src}); mean transmission "
                     f"{float(np.nanmean(nd_T)):.3e}.")

    t_after_s = float(P["t_after_ms"]) / 1000.0
    if t_after_s <= 0:
        return False, "The lamp-image exposure time must be > 0."
    # Controle croise : la pose est souvent lisible dans le TIFF ou dans le
    # nom du fichier. Un facteur 10 ici se propage tel quel sur l'energie.
    t_guess = abscal.tiff_exposure_ms(P["after_image"]) or \
        abscal.exposure_from_name(P["after_image"])
    if t_guess and abs(t_guess - float(P["t_after_ms"])) / t_guess > 0.05:
        notes.append(
            f"⚠ Exposure entered {float(P['t_after_ms']):.0f} ms, but the "
            f"file itself indicates {t_guess:.0f} ms — the calibration scales "
            f"directly with this number. Check which one is right.")
    counts_per_s = A / t_after_s

    # ── 2. lamp spectrum before the fibers ────────────────────────────────
    try:
        wl_b, B_raw = abscal.read_spf2(P["spf2"]) if \
            str(P["spf2"]).lower().endswith(".spf2") else \
            abscal.read_two_columns(P["spf2"])
        wl_b, dens = abscal.normalized_density(wl_b, B_raw)
    except Exception as e:
        return False, f"Could not read the before-fiber spectrum: {e}"

    # ── 3. power-meter spectral correction ────────────────────────────────
    c_pm, pm_info = 1.0, None
    if P.get("responsivity_file") and Path(str(P["responsivity_file"])).exists():
        try:
            pm_info = abscal.powermeter_spectral_correction(
                wl_b, dens, P["responsivity_file"],
                float(P["power_set_nm"]))
            c_pm = float(pm_info["correction"])
            notes.append(f"Power-meter spectral correction "
                         f"x{c_pm:.4f} (set at {P['power_set_nm']:.0f} nm).")
        except Exception as e:
            notes.append(f"Power-meter responsivity ignored ({e}); "
                         f"correction forced to 1.")
    else:
        notes.append("No responsivity curve supplied: the power-meter "
                     "spectral correction is 1 (the reading is assumed "
                     "already correct for this lamp).")

    # ── 4. power entering each fiber ──────────────────────────────────────
    try:
        readings = abscal.read_power_txt(P["power_file"])
    except Exception as e:
        return False, f"Could not read the power file: {e}"
    if not readings:
        return False, "No 'fiber / power' line found in the power file."

    nfib = A.shape[0]
    P_read = np.full(nfib, np.nan)
    for k, v in readings.items():
        i = int(k) - 1
        if 0 <= i < nfib:
            P_read[i] = float(v)
    r_fiber_cm = float(P["sphere_radius_cm"])
    closer_cm = float(P["sensor_closer_cm"])
    if str(P.get("dist_mode")) == "explicit":
        k = 100.0 if str(P.get("dist_unit")) == "m" else 1.0
        r_fiber_cm = float(P["dist_lamp_fiber"]) * k
        r_meter_cm = float(P["dist_lamp_meter"]) * k
        closer_cm = r_fiber_cm - r_meter_cm
        notes.append(
            f"Distances entered directly: lamp→fibers {r_fiber_cm:.1f} cm, "
            f"lamp→meter {r_meter_cm:.1f} cm (meter "
            f"{'closer to' if closer_cm > 0 else 'further from'} the lamp by "
            f"{abs(closer_cm):.1f} cm).")
    try:
        geo = abscal.power_entering_fiber(
            P_read, unit=str(P["power_unit"]), spectral_correction=c_pm,
            geometry=str(P["power_geometry"]),
            sphere_radius_cm=r_fiber_cm,
            sensor_closer_by_cm=closer_cm,
            aperture_diameter_mm=float(P["aperture_mm"]),
            core_diameter_um=float(P["core_um"]),
            na=float(P["fiber_na"]), apply_na=bool(P["apply_na"]))
    except Exception as e:
        return False, f"Invalid power-meter geometry: {e}"
    P_fiber_W = np.asarray(geo["power_W"], float)

    # ── 5. long-pass filter and in-band density ───────────────────────────
    dlam = abscal.bin_widths(wl)
    d_axis = abscal.density_on_axis(wl_b, dens, wl)
    T_lp = abscal.longpass_transmission(wl, float(P["cut_nm"]),
                                        P.get("longpass_curve") or None)
    if bool(P.get("force_zero", True)):
        T_lp = np.where(wl >= float(P["zero_below_nm"]), T_lp, 0.0)
    f_lp = float(np.sum(d_axis * T_lp * dlam))
    if f_lp <= 0:
        return False, ("The long-pass filter removes the whole lamp spectrum "
                       "on the science wavelength axis — check the cut and "
                       "the wavelength calibration.")
    d_lp = d_axis * T_lp / f_lp                      # normalised, 1/nm

    # ── 6. raw calibration curves ─────────────────────────────────────────
    P_band_W = P_fiber_W * f_lp
    power_spectral = P_band_W[:, None] * d_lp[None, :]        # W/nm
    with np.errstate(divide="ignore", invalid="ignore"):
        C_raw = np.where(counts_per_s > 0, power_spectral / counts_per_s,
                         np.nan)                              # J/(nm.count)

    # ── 7. smoothing, bad fibers, hard cut-off ────────────────────────────
    C = np.full_like(C_raw, np.nan)
    for i in range(nfib):
        if not np.isfinite(P_band_W[i]):
            continue
        try:
            C[i] = abscal.smooth_curve(
                wl, C_raw[i], method=str(P["smooth_method"]),
                log_space=bool(P["smooth_log"]),
                window_nm=float(P["smooth_window_nm"]),
                sigma_nm=float(P["smooth_window_nm"]) / 2.5,
                savgol_window_nm=float(P["smooth_window_nm"]) * 1.5,
                median_window_nm=float(P["smooth_window_nm"]) * 0.8)
        except Exception:
            C[i] = C_raw[i]

    bad = sorted({int(b) - 1 for b in (P.get("bad_fibers") or [])
                  if 1 <= int(b) <= nfib})
    if bad and bool(P.get("replace_bad", True)):
        good = [i for i in range(nfib) if i not in bad]
        if good:
            mean_curve = np.nanmean(C[good], axis=0)
            for i in bad:
                C[i] = mean_curve
            notes.append(f"{len(bad)} fibers replaced by the mean curve of "
                         f"the {len(good)} others: "
                         + ", ".join(str(b + 1) for b in bad[:12])
                         + ("…" if len(bad) > 12 else "") + ".")

    blocked = wl < float(P["zero_below_nm"]) if bool(P.get("force_zero", True)) \
        else np.zeros_like(wl, bool)
    C[:, blocked] = 0.0
    C_raw[:, blocked] = 0.0

    # ── scalar summary factor, for the older code paths ───────────────────
    band = ~blocked & np.isfinite(C).any(axis=0)
    with np.errstate(invalid="ignore"):
        g_uJ = np.nanmedian(np.where(band[None, :], C, np.nan), axis=1) * 1e6
    outliers = _abs_cal_outliers(g_uJ)

    s.abs_cal_params = {k: P[k] for k in ABS_CAL_DEFAULTS}
    s.abs_cal = {
        # Empreinte de la campagne : une calibration absolue ne vaut que pour
        # les tirs qui l'ont produite. Sans ce marqueur, celle d'une campagne
        # continuait a offrir l'affichage en J/nm sur la suivante.
        "campaign_key": s.campaign_key(),
        "g_uJ": g_uJ.tolist(), "power_uW": (P_read).tolist(),
        "power_fiber_W": P_fiber_W.tolist(),
        "I_after": np.nansum(A, axis=1).tolist(),
        "fraction_above": f_lp, "cut_nm": float(P["cut_nm"]),
        "zero_below_nm": float(P["zero_below_nm"]),
        "filter_on": "after", "t_after_ms": float(P["t_after_ms"]),
        "t_before_ms": float(P["t_before_ms"]), "n_fibers": int(nfib),
        "after_image": P["after_image"], "spf2": P["spf2"],
        "power_file": P["power_file"], "outliers": outliers,
        "pm_correction": c_pm, "geometry": geo["distance_factor"],
        "area_ratio": geo["area_ratio"], "core_um": float(P["core_um"]),
        "sphere_radius_cm": r_fiber_cm,
        "dist_lamp_meter_cm": r_fiber_cm - closer_cm,
        "notes": notes,
    }
    s.abs_cal_arrays = {
        "wl": wl, "A": A, "wl_b": wl_b, "B": B_raw,
        "B_on_axis": d_axis, "d_lp": d_lp, "T_lp": T_lp, "nd_T": nd_T,
        "counts_per_s": counts_per_s, "power_spectral": power_spectral,
        "C": C, "C_raw": C_raw, "dlam": dlam,
        "tau": np.where(counts_per_s > 0, counts_per_s, np.nan)
        / np.where(d_axis > 0, d_axis, np.nan)[None, :],
    }
    s.save_config()

    n_ok = int(np.isfinite(C).any(axis=1).sum())
    med = float(np.nanmedian(g_uJ)) if np.isfinite(g_uJ).any() else np.nan
    warn = ""
    if outliers:
        warn = (f" ⚠ {len(outliers)} outlier fiber(s): "
                + ", ".join(str(o) for o in outliers[:12])
                + ("…" if len(outliers) > 12 else "") + ".")
    return True, (
        f"Absolute calibration built for {n_ok}/{nfib} fibers. "
        f"In-band fraction of the lamp = {f_lp:.4f}. "
        f"Median calibration = {med * 1e-6:.3e} J/(nm·count).{warn}")


def abs_cal_matrix() -> np.ndarray | None:
    """Per-fiber, per-wavelength factor in µJ/(ADU·nm), or None.

    Rebuilt on demand after a restart. Older sessions that only stored the
    scalar factor still work: the scalar is broadcast over lambda.
    """
    s = SESSION
    if s.abs_cal is None:
        return None
    arr = getattr(s, "abs_cal_arrays", None)
    if arr and arr.get("C") is not None:
        return np.asarray(arr["C"], float) * 1e6
    if abs_cal_rebuild_arrays():
        arr = s.abs_cal_arrays
        if arr and arr.get("C") is not None:
            return np.asarray(arr["C"], float) * 1e6
    g = np.asarray(s.abs_cal["g_uJ"], float)
    n = len(np.asarray(s.calib["wl_axis"], float))
    return np.repeat(g[:, None], n, axis=1)


def _abs_cal_outliers(g: np.ndarray) -> list[int]:
    """1-based fiber numbers whose factor is a robust outlier (MAD-based)."""
    g = np.asarray(g, float)
    finite = g[np.isfinite(g)]
    if finite.size < 8:
        return []
    med = np.median(finite)
    mad = np.median(np.abs(finite - med))
    if mad <= 0:
        return []
    z = 0.6745 * (g - med) / mad
    return [int(i) + 1 for i in np.where(np.abs(z) > 5.0)[0]]


def abs_cal_ready() -> bool:
    """Vraie seulement si une calibration absolue existe POUR CETTE campagne.

    Le simple fait qu'un objet traine en session ne suffit pas : il doit
    porter l'empreinte de la campagne courante. C'est ce qui empeche de
    proposer des J/nm sur une campagne ou la calibration absolue a ete
    sautee, faute de mesure au wattmetre.
    """
    ac = getattr(SESSION, "abs_cal", None)
    if not ac:
        return False
    key = ac.get("campaign_key") if isinstance(ac, dict) else None
    if key and key != SESSION.campaign_key():
        return False
    return True


# Unite physique affichee partout dans l'application. On travaille et on
# affiche desormais en J/nm, en notation scientifique, comme la reference.
ENERGY_UNIT = "J/nm"
ENERGY_LABEL = "Spectral energy (J/nm)"


def to_absolute_energy(spectra: np.ndarray) -> np.ndarray:
    """Convert an (nfib, N) ADU array to spectral energy density [J/nm].

    Uses the per-wavelength calibration curve of each fiber (C is stored in
    J/(nm·count)). Fibers without a curve become NaN rather than zero, so they
    are visibly missing instead of silently contributing nothing.
    """
    C = abs_cal_matrix()
    if C is None:
        raise RuntimeError("Build the absolute calibration first.")
    # abs_cal_matrix renvoie des µJ/(ADU·nm) ; on repasse en J/(ADU·nm).
    C = C * 1e-6
    out = np.asarray(spectra, float).copy()
    n = min(C.shape[0], out.shape[0])
    m = min(C.shape[1], out.shape[1])
    res = np.full(out.shape, np.nan)
    res[:n, :m] = out[:n, :m] * C[:n, :m]
    return res


def abs_cal_rebuild_arrays() -> bool:
    """Recompute the heavy per-λ arrays (after a restart) from the stored
    paths and parameters, so the diagnostics and the µJ/nm conversion work
    without the user rebuilding anything by hand."""
    s = SESSION
    if getattr(s, "abs_cal", None) is None:
        return False
    if getattr(s, "abs_cal_arrays", None) is not None:
        return True
    params = getattr(s, "abs_cal_params", None)
    if not params:
        return False
    try:
        ok, _ = build_absolute_calibration(**params)
        return bool(ok)
    except Exception:
        return False


# ── Display units (ADU vs physical energy) ───────────────────────────────────
def display_units_available() -> bool:
    """True when spectra can be shown in physical energy (calibration done)."""
    return abs_cal_ready()


def current_units() -> str:
    """'uJ' only if selected AND the absolute calibration exists; else 'adu'."""
    if getattr(SESSION, "display_units", "adu") == "uJ" and abs_cal_ready():
        return "uJ"
    return "adu"


def units_ylabel(base: str = "Intensity") -> str:
    return (ENERGY_LABEL if current_units() == "uJ"
            else f"{base} (ADU)")


def to_display_units(spectra: np.ndarray) -> tuple[np.ndarray, str]:
    """Convert a (nfib, N) ADU array to the currently selected display units.
    Returns (array, y-axis label). No-op (returns ADU) unless energy units are
    selected and available."""
    if current_units() == "uJ":
        return to_absolute_energy(spectra), ENERGY_LABEL
    return spectra, "Intensity (ADU)"


# ── Extrapolation de l'energie collectee sur la sphere ───────────────────────
# Une fibre de coeur a_c placee a la distance R du point source intercepte
# l'angle solide
#       Omega = a_c / R^2          [sr]
# L'energie qu'elle mesure est donc un echantillon de l'intensite radiante
#       J(theta, phi) = E_fibre / Omega     [J/sr]
# Extrapoler revient a integrer J sur un domaine plus large que celui couvert
# par les fibres :
#       E_domaine = int J(theta, phi) sin(theta) dtheta dphi
# Les deux directions sont independantes :
#   * horizontale (phi)  : la couronne azimutale mesuree est etendue a 360 deg
#   * verticale  (theta) : la bande polaire mesuree est etendue a [0, 180] deg
#   * les deux ensemble  : integration sur 4*pi sr
# Hors du domaine mesure, J est prolonge par la valeur du bord le plus proche
# (aucune structure n'est inventee). C'est une HYPOTHESE, affichee comme telle
# dans l'interface : elle n'est exacte que si l'emission est reellement plate
# dans la direction extrapolee.
def fiber_solid_angle_sr() -> float | None:
    """Angle solide vu par une fibre, d'apres la geometrie de la page 3."""
    ac = getattr(SESSION, "abs_cal", None) or {}
    core_um = ac.get("core_um")
    radius_cm = ac.get("sphere_radius_cm")
    if not core_um or not radius_cm:
        p = abs_cal_params()
        core_um = core_um or p.get("core_um")
        radius_cm = radius_cm or p.get("sphere_radius_cm")
    try:
        a_core_cm2 = np.pi * (float(core_um) * 1e-4 / 2.0) ** 2
        r_cm = float(radius_cm)
    except (TypeError, ValueError):
        return None
    if not (r_cm > 0 and a_core_cm2 > 0):
        return None
    return float(a_core_cm2 / r_cm ** 2)


def extrapolate_sphere_energy(per_fiber_energy, phis_deg, thetas_deg,
                              extend_phi=False, extend_theta=False,
                              n_grid=181) -> dict:
    """Integre l'energie collectee sur la sphere.

    per_fiber_energy : energie par fibre, meme unite en entree et en sortie
                       (uJ si les spectres sont en uJ/nm).
    Retourne un dictionnaire avec l'energie sur le domaine mesure et sur le
    domaine etendu, plus les bornes utilisees, pour que l'utilisateur voie
    exactement ce qui a ete suppose.
    """
    E = np.asarray(per_fiber_energy, float)
    ph = np.asarray(phis_deg, float)
    th = np.asarray(thetas_deg, float)
    ok = np.isfinite(E) & np.isfinite(ph) & np.isfinite(th)
    out = {"n_fibers": int(ok.sum()), "collected": float(np.nansum(E[ok])),
           "omega_sr": fiber_solid_angle_sr()}
    if out["omega_sr"] is None or ok.sum() < 3:
        out["error"] = ("Not enough fibers with angles, or the fiber core / "
                        "sphere radius is unknown (set them on page 3).")
        return out

    E, ph, th = E[ok], ph[ok], th[ok]
    # Une fibre ne collecte pas d'energie negative : les valeurs < 0 viennent
    # d'une sur-soustraction de fond. On les ramene a 0 avant d'extrapoler,
    # sinon l'integrale peut devenir negative, ce qui n'a pas de sens physique.
    n_neg = int((E < 0).sum())
    if n_neg:
        out["n_negative_fibers"] = n_neg
        E = np.clip(E, 0.0, None)
    omega = out["omega_sr"]
    J = E / omega                                   # intensite radiante, J/sr

    ph0, ph1 = float(ph.min()), float(ph.max())
    th0, th1 = float(th.min()), float(th.max())
    out["phi_range"] = (ph0, ph1)
    out["theta_range"] = (th0, th1)

    # grille reguliere sur le domaine demande
    pg0, pg1 = (0.0, 360.0) if extend_phi else (ph0, ph1)
    tg0, tg1 = (0.0, 180.0) if extend_theta else (th0, th1)
    pgrid = np.linspace(pg0, pg1, n_grid)
    tgrid = np.linspace(tg0, tg1, n_grid)
    PG, TG = np.meshgrid(pgrid, tgrid, indexing="ij")

    # prolongement par le bord : on interroge l'interpolateur sur la position
    # mesuree la plus proche, jamais au-dela.
    PQ = np.clip(PG, ph0, ph1)
    TQ = np.clip(TG, th0, th1)

    from scipy.interpolate import griddata
    pts = np.column_stack([ph, th])
    q = np.column_stack([PQ.ravel(), TQ.ravel()])
    Jg = griddata(pts, J, q, method="linear")
    miss = ~np.isfinite(Jg)
    if miss.any():
        Jg[miss] = griddata(pts, J, q[miss], method="nearest")
    Jg = Jg.reshape(PG.shape)

    def integrate(Jgrid, pg, tg):
        # |dΩ| = |sin θ| dθ dφ, et on integre toujours sur des axes croissants :
        # une plage saisie a l'envers ne doit pas produire d'angle solide ou
        # d'energie negatifs (un angle solide est positif par definition).
        tr = np.radians(np.sort(tg))
        pr = np.radians(np.sort(pg))
        w = np.abs(np.sin(tr))
        trapz = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
        inner = trapz(Jgrid * w[None, :], tr, axis=1)     # sur theta
        return abs(float(trapz(inner, pr)))               # sur phi

    out["energy_extrapolated"] = integrate(Jg, pgrid, tgrid)
    out["solid_angle_sr"] = integrate(np.ones_like(Jg), pgrid, tgrid)
    out["extend_phi"] = bool(extend_phi)
    out["extend_theta"] = bool(extend_theta)
    out["is_4pi"] = bool(extend_phi and extend_theta)

    # meme integrale restreinte au domaine reellement mesure, pour comparer
    pm = np.linspace(ph0, ph1, n_grid)
    tm = np.linspace(th0, th1, n_grid)
    PM, TM = np.meshgrid(pm, tm, indexing="ij")
    qm = np.column_stack([PM.ravel(), TM.ravel()])
    Jm = griddata(pts, J, qm, method="linear")
    miss = ~np.isfinite(Jm)
    if miss.any():
        Jm[miss] = griddata(pts, J, qm[miss], method="nearest")
    Jm = Jm.reshape(PM.shape)
    out["energy_measured_cone"] = integrate(Jm, pm, tm)
    out["solid_angle_measured_sr"] = integrate(np.ones_like(Jm), pm, tm)
    return out


def abs_cal_audit(shot_key: str | None = None) -> list[dict]:
    """Deroule toute la chaine de calibration en nombres, une ligne par etape.

    Sert a comparer terme a terme avec une autre implementation : quand deux
    codes ne donnent pas la meme energie, l'ecart se voit immediatement sur la
    ligne fautive plutot que sur le resultat final.
    """
    s = SESSION
    ac = getattr(s, "abs_cal", None)
    arr = getattr(s, "abs_cal_arrays", None)
    if ac is None or not arr:
        return []
    P = abs_cal_params()
    wl = arr["wl"]
    A, cps, C = arr["A"], arr["counts_per_s"], arr["C"]
    Pr = np.asarray(ac["power_uW"], float)
    Pf = np.asarray(ac["power_fiber_W"], float)
    ref = int(np.nanargmax(np.nansum(np.nan_to_num(A), axis=1)))

    def row(step, value, unit, comment):
        return {"step": step, "value": value, "unit": unit,
                "comment": comment}

    r_fib = float(ac.get("sphere_radius_cm", np.nan))
    r_pm = float(ac.get("dist_lamp_meter_cm", np.nan))
    rows = [
        row("Reference fiber used below", f"{ref + 1}", "",
            "brightest fiber of the lamp image"),
        row("Power-meter reading", f"{Pr[ref]:.6g}", P["power_unit"],
            "as written in the power file"),
        row("Spectral correction", f"{ac.get('pm_correction', 1.0):.6g}", "×",
            "R(set)/⟨R⟩; 1 if no responsivity curve was given"),
        row("Lamp → fibers", f"{r_fib:.4g}", "cm", "geometry"),
        row("Lamp → meter", f"{r_pm:.4g}", "cm",
            "meter " + ("closer to" if r_pm < r_fib else "further from")
            + " the lamp"),
        row("Distance factor", f"{ac.get('geometry', 1.0):.6g}", "×",
            "(R_meter/R_fiber)²"),
        row("Meter aperture area", f"{np.pi * (0.1 * float(P['aperture_mm']) / 2) ** 2:.6g}",
            "cm²", f"Ø {P['aperture_mm']} mm"),
        row("Fiber core area", f"{np.pi * (float(P['core_um']) * 1e-4 / 2) ** 2:.6g}",
            "cm²", f"Ø {P['core_um']} µm"),
        row("Core/aperture ratio", f"{ac.get('area_ratio', 1.0):.6g}", "×",
            "fraction of the beam intercepted by the core"),
        row("Power entering the fiber", f"{Pf[ref]:.6g}", "W",
            "after every geometric factor"),
        row("In-band lamp fraction", f"{ac['fraction_above']:.6g}", "×",
            f"above {ac.get('zero_below_nm', ac['cut_nm']):.0f} nm, on the "
            f"science axis"),
        row("Power in band", f"{Pf[ref] * ac['fraction_above']:.6g}", "W",
            "what the calibrated band actually carries"),
        row("Lamp-image exposure", f"{P['t_after_ms'] / 1000.0:.6g}", "s",
            "divides the counts"),
        row("ND on the lamp image", f"{P['nd_od']:.6g}", "OD",
            f"counts multiplied by {10 ** float(P['nd_od']):.6g}"),
        row("Peak counts/s of that fiber",
            f"{np.nanmax(cps[ref]):.6g}", "count/s", "after ND and exposure"),
        row("Calibration at that peak",
            f"{np.nanmax(np.where(np.isfinite(C[ref]), C[ref], np.nan)):.6g}",
            "J/(nm·count)", "smoothed and saved"),
        row("Median calibration",
            f"{np.nanmedian(np.where(C > 0, C, np.nan)):.6g}",
            "J/(nm·count)", "over every fiber and wavelength"),
    ]
    if shot_key:
        try:
            raw = sf.apply_intensity_calibration(
                extract_cached(shot_key, use_cache=True),
                SESSION.int_calib_factors)          # AVANT correction ND
            S = get_spectra(shot_key, use_cache=True)   # APRES correction ND
            E = to_absolute_energy(S)
            od_shot = shot_nd_value(shot_key)
            from core import ndfilters as _ndf
            _fac, _src = _ndf.correction_factor(
                od_shot or 0, SESSION.calib["wl_axis"], SESSION.nd_files)
            icf = SESSION.int_calib_factors
            with np.errstate(invalid="ignore", divide="ignore"):
                nd_gain = float(np.nanmax(S) / np.nanmax(raw)) \
                    if np.nanmax(raw) else float("nan")
            C_med = np.nanmedian(np.where(C > 0, C, np.nan))
            rows += [
                row("── science shot ──", shot_key, "", ""),
                row("ND of the shot itself",
                    "none" if od_shot in (None, 0) else f"{od_shot:g}", "OD",
                    "column 'side-SRS ND' of the shotbook"),
                row("ND factor actually applied",
                    f"{np.nanmin(_fac):.4g} … {np.nanmax(_fac):.4g}", "×",
                    f"source '{_src}' — should be 10^OD = "
                    f"{10 ** float(od_shot or 0):.4g}. A measured curve read "
                    f"outside its own range ('file+clamp') can be far off."),
                row("Intensity-calibration factors",
                    "none" if icf is None else
                    f"{np.nanmin(icf):.4g} … {np.nanmax(icf):.4g}", "×",
                    "applied to BOTH the lamp image and the shots since "
                    "v15; if this range is far from 1 it used to land "
                    "entirely on the energy"),
                row("Peak ADU before ND correction",
                    f"{np.nanmax(raw):.6g}", "ADU",
                    "what the detector actually recorded"),
                row("Peak ADU after ND correction",
                    f"{np.nanmax(S):.6g}", "ADU",
                    f"×{nd_gain:.6g} — the pipeline removes the shot's own ND"),
                row("Peak energy (this application)",
                    f"{np.nanmax(E):.4e}", "J/nm", "ND-corrected shot"),
                row("Peak energy WITHOUT the shot's ND",
                    f"{np.nanmax(raw) * C_med * 1e-6:.4e}", "J/nm",
                    "the quantity a script that ignores the shot's ND plots"),
            ]
        except Exception as e:
            rows.append(row("Science shot", "—", "", f"not available ({e})"))
    return rows
