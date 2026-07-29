"""
core/abscal.py — Absolute (physical) calibration of the fibers.

Turns the detector ADU spectra into physical units (W/nm, then J/nm) using
three measurements, all mandatory:

  1. after-fiber lamp image  (.tiff, science detector) — the lamp seen
     THROUGH the fibers; gives, per fiber, the ADU spectral shape on the
     science wavelength axis. Usually carries a long-pass filter (e.g. 650 nm
     high-pass), so the fibers are only characterised above the cut.
  2. before-fiber lamp spectrum (.spf2, fiber spectrometer) — the SAME lamp
     measured BEFORE the fibers, no filter; a single reference spectrum
     spanning the full range, used to estimate the fraction of light below
     the filter cut (which the after-fiber image cannot see).
  3. per-fiber power (.txt) — the absolute optical power (µW) measured through
     each fiber with a power meter; the absolute anchor of the calibration.

The two images have DIFFERENT integration times, entered by the user; the
filter cut wavelength (if any) is entered by the user too.

This module only READS and prepares the inputs. The physics is assembled in
core/analysis.py (build_absolute_calibration), so the assumptions stay in one
documented place.
"""
from __future__ import annotations

import re
import struct
from pathlib import Path

import numpy as np

try:
    _trapz = np.trapezoid   # numpy >= 2.0
except AttributeError:      # numpy < 2.0
    _trapz = np.trapz


# ── Per-fiber power meter file ───────────────────────────────────────────────
def read_power_txt(path) -> dict[int, float]:
    """Parse the power-meter file: lines like 'fiber / power' -> 'N / value'.

    Returns {fiber_number_1based: power_uW}. Tolerant of a free-text header,
    blank lines, trailing spaces and both '/' and ':' separators."""
    out: dict[int, float] = {}
    text = Path(path).read_text(errors="ignore")
    for line in text.splitlines():
        s = line.strip()
        if not s:
            continue
        m = re.match(r"^(\d+)\s*[/:]\s*([-+]?\d+(?:\.\d+)?)\s*$", s)
        if m:
            out[int(m.group(1))] = float(m.group(2))
    return out


# ── Before-fiber spectrometer file (.spf2) ───────────────────────────────────
def read_spf2(path) -> tuple[np.ndarray, np.ndarray]:
    """Read a .spf2 fiber-spectrometer file (Ocean-Optics-style binary).

    Layout (reverse-engineered): a header, then a float32 little-endian
    wavelength array (monotonically increasing, nm), immediately followed by
    a float32 little-endian intensity array of the same length. Returns
    (wavelength_nm, intensity). Raises ValueError if no plausible wavelength
    array is found.
    """
    data = Path(path).read_bytes()
    # locate the wavelength array: first offset giving a long, strictly
    # increasing float32 run within a plausible optical range.
    best = None
    for off in range(0, min(len(data) - 4 * 64, 20000)):
        try:
            probe = np.frombuffer(data, dtype="<f4", count=64, offset=off)
        except ValueError:
            continue
        if not np.all(np.isfinite(probe)):
            continue
        d = np.diff(probe)
        if (np.all(d > 0) and probe[0] > 150 and probe[-1] < 1200
                and 0.02 < np.median(d) < 5):
            best = off
            break
    if best is None:
        raise ValueError("no wavelength array found in this .spf2 file")

    full = np.frombuffer(data, dtype="<f4", offset=best)
    n = 1
    while n < len(full) and full[n] > full[n - 1] and full[n] < 1200:
        n += 1
    wl = np.ascontiguousarray(full[:n].astype(np.float64))
    inten_off = best + n * 4
    inten = np.frombuffer(data, dtype="<f4", offset=inten_off,
                          count=n).astype(np.float64)
    return wl, np.ascontiguousarray(inten)


# ── Integration time from a PVCAM TIFF description ───────────────────────────
def tiff_exposure_ms(path) -> float | None:
    """Try to read the exposure time (ms) from a PVCAM TIFF ImageDescription
    (contains a line 'expTime=1000ms'). Returns None if unavailable."""
    try:
        import tifffile
        with tifffile.TiffFile(path) as t:
            desc = t.pages[0].tags.get("ImageDescription")
            desc = desc.value if desc is not None else ""
    except Exception:
        return None
    m = re.search(r"expTime\s*=\s*([-+]?\d+(?:\.\d+)?)\s*(ms|us|s)\b", desc)
    if not m:
        return None
    val, unit = float(m.group(1)), m.group(2)
    return val * {"ms": 1.0, "us": 1e-3, "s": 1000.0}[unit]


# ── Background-subtracted integral of a spectrum ─────────────────────────────
def integrate_without_background(wl: np.ndarray, spectrum: np.ndarray,
                                 lo: float | None = None,
                                 hi: float | None = None,
                                 bg_percentile: float = 15.0) -> float:
    """∫ spectrum dλ over [lo, hi] after removing a flat background, so the
    integral counts signal only (not the pedestal) — mirroring the noise
    handling used when plotting spectra.

    The background is the ``bg_percentile`` percentile of the spectrum inside
    the window (robust to the emission peaks); it is subtracted and negative
    residuals are clipped to zero before integrating with the trapezoidal
    rule. ``lo``/``hi`` default to the full range.
    """
    wl = np.asarray(wl, float)
    y = np.asarray(spectrum, float)
    m = np.isfinite(wl) & np.isfinite(y)
    wl, y = wl[m], y[m]
    order = np.argsort(wl)
    wl, y = wl[order], y[order]
    if lo is None:
        lo = wl.min()
    if hi is None:
        hi = wl.max()
    sel = (wl >= lo) & (wl <= hi)
    if sel.sum() < 2:
        return 0.0
    wls, ys = wl[sel], y[sel]
    bg = np.percentile(ys, bg_percentile)
    ys = np.clip(ys - bg, 0.0, None)
    return float(_trapz(ys, wls))


def fraction_above_cut(wl: np.ndarray, spectrum: np.ndarray,
                       cut_nm: float, bg_percentile: float = 15.0) -> float:
    """Fraction of the (background-subtracted) spectral content that lies
    above ``cut_nm``: ∫_cut^∞ / ∫_-∞^∞ . Used to rescale a full-range power
    measurement to the band the filtered after-fiber image can actually see.
    Returns 1.0 if there is effectively no content below the cut.
    """
    total = integrate_without_background(wl, spectrum,
                                         bg_percentile=bg_percentile)
    if total <= 0:
        return 1.0
    above = integrate_without_background(wl, spectrum, lo=cut_nm,
                                         bg_percentile=bg_percentile)
    return float(np.clip(above / total, 0.0, 1.0))


# =============================================================================
# CHAINE PHOTOMETRIQUE COMPLETE (methode de reference)
# =============================================================================
# La version historique de ce module ramenait la calibration a UN scalaire par
# fibre : kappa = P_mesure / integrale_ADU. Cela suppose implicitement que le
# wattmetre lit exactement la puissance qui entre dans la fibre, et ignore la
# dependance en lambda de la reponse fibre + spectrometre + camera.
#
# La chaine ci-dessous reprend la methode de reference et corrige les deux
# points :
#
#   1. GEOMETRIE. Le wattmetre n'est pas la fibre. Sur la sphere, il mesure
#      l'eclairement de la lampe sur SA propre ouverture, a une distance qui
#      n'est pas forcement celle du plan des fibres. La puissance qui entre
#      reellement dans une fibre est donc
#          P_fibre = P_vraie / A_wattmetre * (R_wm/R_fibre)^2 * A_coeur
#      Le rapport A_coeur/A_wattmetre vaut ~1e-4 pour un coeur de 100 um face
#      a une ouverture de 9.5 mm : c'est la source principale des ordres de
#      grandeur d'ecart.
#
#   2. SPECTRE. La reponse n'est pas plate. La calibration devient une COURBE
#      par fibre, C_i(lambda) en J/(nm*count) :
#          C_i(lambda) = P_bande,i * d_bande(lambda) / (counts_i(lambda)/t)
#      ou d_bande est la densite spectrale normalisee de la lampe apres le
#      filtre passe-haut. Une energie spectrale s'obtient alors directement
#      par  E(lambda) = counts * C_i(lambda), independamment du temps de pose
#      de l'image science.
#
# Chaque etape est isolee dans une fonction testable et tous les parametres
# physiques sont des arguments : rien n'est code en dur pour une campagne.

# --- Lecture d'un fichier a deux colonnes (reponse du wattmetre, filtre) -----
def read_two_columns(path) -> tuple[np.ndarray, np.ndarray]:
    """Lit un fichier CSV/TXT a deux colonnes numeriques (x, y).

    Tolerant : entete libre, separateurs varies, lignes de commentaire. Les
    deux premieres colonnes contenant au moins 5 valeurs numeriques sont
    retenues.
    """
    p = Path(path)
    text = p.read_text(errors="ignore")
    rows = []
    for line in text.splitlines():
        nums = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", line)
        if len(nums) >= 2:
            try:
                rows.append((float(nums[0]), float(nums[1])))
            except ValueError:
                continue
    if len(rows) < 2:
        raise ValueError(f"no two readable numeric columns in {p.name}")
    arr = np.asarray(rows, float)
    x, y = arr[:, 0], arr[:, 1]
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    o = np.argsort(x)
    return x[o], y[o]


# --- Densite spectrale normalisee ------------------------------------------
def normalized_density(wl: np.ndarray, signal: np.ndarray,
                       floor_percentile: float = 1.0
                       ) -> tuple[np.ndarray, np.ndarray]:
    """(wl, d) avec d >= 0 et integrale(d dlambda) = 1.

    Le plancher de bruit du spectrometre est retire par un percentile bas
    avant normalisation, sinon un offset constant sur 400 nm de plage domine
    l'integrale et fausse toutes les fractions de bande.
    """
    wl = np.asarray(wl, float)
    y = np.asarray(signal, float)
    m = np.isfinite(wl) & np.isfinite(y)
    wl, y = wl[m], y[m]
    o = np.argsort(wl)
    wl, y = wl[o], y[o]
    if wl.size < 2:
        raise ValueError("spectrum too short to be normalised")
    if floor_percentile is not None:
        y = y - np.nanpercentile(y, float(floor_percentile))
    y = np.clip(y, 0.0, None)
    area = float(_trapz(y, wl))
    if not np.isfinite(area) or area <= 0:
        raise ValueError("spectrum integral is zero or negative")
    return wl, y / area


def density_on_axis(src_wl, src_density, target_wl) -> np.ndarray:
    """Interpole une densite spectrale sur un autre axe, 0 hors du domaine."""
    d = np.interp(np.asarray(target_wl, float), np.asarray(src_wl, float),
                  np.asarray(src_density, float), left=0.0, right=0.0)
    d[~np.isfinite(d)] = 0.0
    return np.clip(d, 0.0, None)


def bin_widths(wl: np.ndarray) -> np.ndarray:
    """Largeur de chaque bin spectral (bords a mi-chemin des centres)."""
    wl = np.asarray(wl, float)
    edges = np.empty(wl.size + 1, float)
    edges[1:-1] = 0.5 * (wl[:-1] + wl[1:])
    edges[0] = wl[0] - 0.5 * (wl[1] - wl[0])
    edges[-1] = wl[-1] + 0.5 * (wl[-1] - wl[-2])
    return np.abs(np.diff(edges))


# --- Filtre passe-haut ------------------------------------------------------
def longpass_transmission(target_wl, cut_nm: float,
                          curve_path=None) -> np.ndarray:
    """T(lambda) du passe-haut : marche ideale, ou courbe mesuree si fournie.

    Une courbe en pourcents (max > 1.5) est ramenee en fraction.
    """
    target_wl = np.asarray(target_wl, float)
    if not curve_path:
        return (target_wl >= float(cut_nm)).astype(float)
    wl, tr = read_two_columns(curve_path)
    tr = np.asarray(tr, float)
    if np.nanmax(tr) > 1.5:
        tr = tr / 100.0
    tr = np.clip(tr, 0.0, 1.0)
    return np.interp(target_wl, wl, tr, left=0.0, right=float(tr[-1]))


# --- Correction spectrale du wattmetre --------------------------------------
def powermeter_spectral_correction(led_wl, led_density, responsivity_path,
                                   set_wavelength_nm: float) -> dict:
    """Facteur de correction du wattmetre regle sur UNE longueur d'onde.

    Un wattmetre affiche P_lu = I / R(lambda_reglee). Face a une source large
    bande, le photocourant vaut en realite I = P_vraie * <R> ou <R> est la
    responsivite moyennee par le spectre de la source. D'ou

        P_vraie = P_lu * R(lambda_reglee) / <R>

    Le facteur peut atteindre plusieurs dizaines de pourcents quand la source
    est large et la responsivite pentue.
    """
    rwl, resp = read_two_columns(responsivity_path)
    lo = max(float(np.nanmin(led_wl)), float(np.nanmin(rwl)))
    hi = min(float(np.nanmax(led_wl)), float(np.nanmax(rwl)))
    if not (hi > lo):
        raise ValueError("the lamp spectrum and the responsivity curve do not "
                         "overlap")
    grid = np.linspace(lo, hi, 5000)
    d = density_on_axis(led_wl, led_density, grid)
    area = float(_trapz(d, grid))
    if area <= 0:
        raise ValueError("lamp/responsivity overlap has a zero integral")
    d = d / area
    r = np.interp(grid, rwl, resp)
    r_eff = float(_trapz(d * r, grid))
    r_set = float(np.interp(float(set_wavelength_nm), rwl, resp))
    if r_eff <= 0:
        raise ValueError("spectrum-weighted responsivity is zero")
    return {"r_set": r_set, "r_eff": r_eff, "correction": r_set / r_eff,
            "overlap_nm": (lo, hi),
            "resp_range_nm": (float(rwl[0]), float(rwl[-1]))}


# --- Geometrie wattmetre -> fibre -------------------------------------------
POWER_UNIT_SCALE = {"uW": 1e-6, "µW": 1e-6, "mW": 1e-3, "W": 1.0, "nW": 1e-9}


def power_entering_fiber(power_reading, unit: str = "uW",
                         spectral_correction: float = 1.0,
                         geometry: str = "sphere",
                         sphere_radius_cm: float = 44.0,
                         sensor_closer_by_cm: float = 0.0,
                         aperture_diameter_mm: float = 9.5,
                         core_diameter_um: float = 100.0,
                         na: float = 0.22,
                         apply_na: bool = False) -> dict:
    """Puissance reellement injectee dans une fibre, en W.

    geometry = 'sphere' : le wattmetre mesure l'eclairement de la lampe sur la
        sphere, a la place d'une fibre. On corrige de la distance puis on ne
        garde que la fraction interceptee par le coeur :
            P_fibre = P_vraie / A_ouverture * (R_wm/R_fibre)^2 * A_coeur
    geometry = 'through_fiber' : le wattmetre est place en sortie de fibre et
        lit deja la puissance transmise. Aucune correction geometrique.

    sensor_closer_by_cm > 0 signifie que le detecteur etait PLUS PRES de la
    lampe que le plan des fibres ; l'eclairement au plan des fibres est alors
    plus faible d'un facteur ((R - offset)/R)^2.
    """
    scale = POWER_UNIT_SCALE.get(str(unit), None)
    if scale is None:
        raise ValueError(f"unsupported power unit: {unit!r}")
    p = np.asarray(power_reading, float) * scale
    p_true = p * float(spectral_correction)

    if str(geometry) == "through_fiber":
        return {"power_W": p_true, "distance_factor": 1.0,
                "area_ratio": 1.0, "na_factor": 1.0,
                "aperture_area_cm2": np.nan, "core_area_cm2": np.nan}

    r_fib = float(sphere_radius_cm)
    r_pm = r_fib - float(sensor_closer_by_cm)
    if r_fib <= 0 or r_pm <= 0:
        raise ValueError("invalid sphere geometry: the sphere radius and the "
                         "fiber-plane distance must both be > 0 (check the "
                         "radius and the sensor offset)")
    dist = (r_pm / r_fib) ** 2
    a_pm = np.pi * (0.1 * float(aperture_diameter_mm) / 2.0) ** 2      # cm^2
    a_core = np.pi * (float(core_diameter_um) * 1e-4 / 2.0) ** 2       # cm^2
    na_factor = float(na) ** 2 if apply_na else 1.0
    irradiance = p_true / a_pm * dist                                  # W/cm^2
    return {"power_W": irradiance * a_core * na_factor,
            "distance_factor": dist, "area_ratio": a_core / a_pm,
            "na_factor": na_factor, "aperture_area_cm2": a_pm,
            "core_area_cm2": a_core, "irradiance_W_cm2": irradiance}


# --- Lissage des courbes de calibration -------------------------------------
def _odd_window(wl, window_nm, minimum=3) -> int:
    wl = np.asarray(wl, float)
    f = np.isfinite(wl)
    dw = np.nanmedian(np.abs(np.diff(wl[f]))) if f.sum() >= 2 else 1.0
    if not np.isfinite(dw) or dw <= 0:
        dw = 1.0
    n = max(int(minimum), int(np.ceil(float(window_nm) / dw)))
    return n if n % 2 else n + 1


def _fill_nonfinite(y):
    y = np.asarray(y, float).copy()
    idx = np.arange(y.size, dtype=float)
    f = np.isfinite(y)
    if f.sum() == 0:
        return np.zeros_like(y)
    if f.sum() == 1:
        y[~f] = y[f][0]
        return y
    y[~f] = np.interp(idx[~f], idx[f], y[f])
    return y


def smooth_curve(wl, y, method="moving_average", log_space=True,
                 window_nm=10.0, sigma_nm=4.0, savgol_window_nm=15.0,
                 savgol_poly=3, median_window_nm=8.0,
                 spline_factor=0.05, spline_order=3,
                 clip_sigma=6.0, min_points=30) -> np.ndarray:
    """Lisse une courbe de calibration le long de lambda.

    Le lissage se fait par defaut sur log10(C) : la calibration est un rapport
    (puissance connue)/(counts mesures) qui devient tres bruite la ou la lampe
    est faible ; le bruit y est multiplicatif, donc additif en log, et le
    resultat reste strictement positif.

    Retourne un tableau de meme taille que wl, avec NaN la ou aucune valeur
    exploitable n'existe.
    """
    wl = np.asarray(wl, float)
    y = np.asarray(y, float)
    out = np.full(wl.size, np.nan)
    good = np.isfinite(y) & (y > 0) if log_space else np.isfinite(y)
    if good.sum() < max(5, int(min_points)) or str(method) == "none":
        out[good] = y[good]
        return out

    x = wl[good]
    w = np.log10(y[good]) if log_space else y[good].copy()

    # rejet des points aberrants isoles avant lissage
    if clip_sigma:
        from scipy.ndimage import median_filter
        win = min(_odd_window(x, median_window_nm, 7),
                  x.size if x.size % 2 else x.size - 1)
        if win >= 3:
            trend = median_filter(_fill_nonfinite(w), size=win, mode="nearest")
            res = w - trend
            mad = np.median(np.abs(res - np.median(res)))
            sig = 1.4826 * mad
            if np.isfinite(sig) and sig > 0:
                keep = np.abs(res) <= float(clip_sigma) * sig
                if keep.sum() >= max(5, int(min_points)):
                    x, w = x[keep], w[keep]

    m = str(method)
    if m in ("moving_average", "boxcar"):
        win = _odd_window(x, window_nm, 3)
        k = np.ones(win)
        num = np.convolve(_fill_nonfinite(w), k, mode="same")
        den = np.convolve(np.ones_like(w), k, mode="same")
        sm = num / np.where(den > 0, den, np.nan)
    elif m == "gaussian":
        from scipy.ndimage import gaussian_filter1d
        dx = np.nanmedian(np.abs(np.diff(x))) or 1.0
        sm = gaussian_filter1d(_fill_nonfinite(w),
                               sigma=max(float(sigma_nm) / dx, 0.01),
                               mode="nearest")
    elif m in ("savgol", "savitzky_golay"):
        from scipy.signal import savgol_filter
        win = min(_odd_window(x, savgol_window_nm, int(savgol_poly) + 3),
                  x.size if x.size % 2 else x.size - 1)
        poly = max(0, min(int(savgol_poly), win - 2))
        sm = (savgol_filter(_fill_nonfinite(w), win, poly, mode="interp")
              if win >= poly + 2 and win >= 3 else _fill_nonfinite(w))
    elif m in ("median_filter", "median"):
        from scipy.ndimage import median_filter
        win = _odd_window(x, median_window_nm, 3)
        sm = median_filter(_fill_nonfinite(w), size=win, mode="nearest")
    elif m == "spline":
        from scipy.interpolate import UnivariateSpline
        ww = _fill_nonfinite(w)
        s = float(spline_factor) * x.size * max(np.var(ww), 1e-12)
        k = int(min(max(spline_order, 1), 5))
        if x.size <= k:
            sm = ww
        else:
            sm = UnivariateSpline(x, ww, k=k, s=s)(x)
    else:
        raise ValueError(f"unknown smoothing method: {method!r}")

    vals = 10.0 ** sm if log_space else sm
    # ré-interpolation sur l'axe complet (des points ont pu etre rejetes)
    out = np.interp(wl, x, vals, left=np.nan, right=np.nan)
    return out


def exposure_from_name(path) -> float | None:
    """Temps de pose (ms) devine depuis le nom de fichier.

    Les images de calibration portent souvent la pose dans leur nom
    ("..._10sacq_...", "..._500ms_..."). Le script de reference se sert
    uniquement de cela ; on l'utilise ici comme controle croise du champ
    saisi, jamais comme valeur imposee.
    """
    name = Path(path).name
    for pat, mult in ((r"(\d+(?:\.\d+)?)\s*s\s*acq", 1000.0),
                      (r"(\d+(?:\.\d+)?)\s*ms", 1.0),
                      (r"(\d+(?:\.\d+)?)\s*us", 1e-3),
                      (r"(\d+(?:\.\d+)?)\s*s(?![a-rt-z])", 1000.0)):
        m = re.search(pat, name, flags=re.IGNORECASE)
        if m:
            return float(m.group(1)) * mult
    return None
