"""

spectro_functions.py

====================

Toutes les fonctions du pipeline d'extraction de spectres multi-fibres.

Appelé par le notebook Jupyter analyse.ipynb.

 

Pipeline :

  1. Calibration spatiale  : positions + tilt des 80 fibres (image HgAr)

  2. Calibration spectrale : pixels -> longueurs d'onde (raies HgAr connues)

  3. Extraction            : 80 spectres par image (extraction optimale pondérée)

  4. Analyse               : détection fibres vides, SNR, comparaisons

 

Auteur  : généré automatiquement

Dépend. : numpy scipy matplotlib Pillow

"""

 

import numpy as np

import warnings

from pathlib import Path

from PIL import Image

from scipy.signal import find_peaks, savgol_filter

from scipy.ndimage import median_filter, gaussian_filter1d, uniform_filter1d, rotate as ndimage_rotate

from scipy.interpolate import interp1d

from scipy.optimize import curve_fit

 

 

# =============================================================================

# CATALOGUE DES RAIES HgAr (nm) — sources NIST

# =============================================================================

 

HGAR_LINES = {

    # Mercure (Hg)

     871.66: 'Hg', 730.04: 'Hg', 809.32: 'Hg', 815.56: 'Hg', 871.68: 'Hg',

    # Argon (Ar)

    667.73: 'Ar', 696.54: 'Ar', 706.72: 'Ar', 714.70: 'Ar',

    727.29: 'Ar', 738.40: 'Ar', 750.92: 'Ar', 763.51: 'Ar',

    772.38: 'Ar', 794.82: 'Ar', 800.62: 'Ar',

    811.53: 'Ar', 826.45: 'Ar', 841.64: 'Ar',

    852.14: 'Ar', 866.79: 'Ar',  912.30: 'Ar', 922.45: 'Ar',


}

 

# Paires pixel<->longueur d'onde identifiées manuellement sur l'image de calib.

# Format : (pixel_centre, longueur_onde_nm)

# Ces valeurs ont été vérifiées avec le setup 150 gr/mm centré à 800 nm.

# Ajustez-les si votre setup diffère (utilisez plot_wl_calibration pour vérifier).

WL_CALIB_PAIRS = [

    ( 457,  576.96),   # Hg

    (1242,  706.72),   # Ar

    (1291,  714.70),   # Ar

    (1363,  727.29),   # Ar

    (1495,  750.39),   # Ar

    (1618,  772.38),   # Ar

    (1862,  811.53),   # Ar (raie de référence principale)

]

 

# Degré du polynôme de dispersion (3 recommandé)

WL_POLY_DEG = 3

# ── Calibration spectrale AUTOMATIQUE (no-guess, universelle) ────────────────
# Paramètres du comb-matching pixel→nm sans paires codées en dur.
# La technique : détecter les pics de raies dans le spectre HgAr moyen, puis
# trouver la dispersion qui apparie le PLUS de pics au catalogue HGAR_LINES,
# via une recherche à deux ancres (aucune longueur d'onde supposée a priori).
WL_AUTO_SMOOTH_PX      = 5      # lissage (moving average) avant find_peaks
WL_AUTO_PROM_FRAC      = 0.010  # prominence mini = frac × dynamique du spectre
WL_AUTO_MIN_DIST_PX    = 6      # distance mini entre pics (px)
WL_AUTO_MAX_PEAKS      = 60     # nb max de pics détectés gardés
WL_AUTO_N_BRIGHT       = 16     # nb de pics les plus intenses testés comme ancres
WL_AUTO_MATCH_TOL_NM   = 2.0    # tolérance d'appariement pic↔catalogue (nm)
WL_AUTO_MIN_MATCHED    = 6      # nb mini de raies appariées pour accepter
WL_AUTO_RMS_MAX_NM     = 0.2    # RMS max accepté (au-delà = solution aliasée)
WL_AUTO_MIN_PIX_SEP    = 200    # écart pixel mini entre les 2 ancres (stabilité)
# Fourchette de dispersion physiquement plausible |nm/px|. Large par défaut
# (universel) mais borné pour éviter les solutions aliasées absurdes.
WL_AUTO_DISP_RANGE     = (0.02, 1.0)
# "auto" = comb-matching (défaut, universel) ; "manual" = anciennes paires.
WL_METHOD              = "auto"

 

 

# =============================================================================

# PARAMÈTRES DU PIPELINE (modifiables)

# =============================================================================

 

N_FIBERS       = 80

HALF_WIDTH     = 6

BG_FILTER_SIZE = 55

PEAK_MIN_DIST  = 10

PEAK_MIN_PROM  = 50

GAP_RATIO      = 1.7
EDGE_GAP_RATIO = 3.5   # For _trim_edge_peaks: max gap (in periods) before a
                        # peak is considered isolated noise.  Must be > 2 so
                        # that 1–2 missed fibers near the edge don't cause
                        # trimming of legitimate peaks beyond the gap.
                        # Noise spikes (Y<250 or Y>1750) have gaps of 6–13×
                        # period, so 3.5 is a safe threshold.
MAX_EXTEND     = 3      # Max extra grid slots beyond the last detected peak

N_TILT_COLS    = 15

CALIB_FRAC     = (0.2, 0.85)

 

# ── Fond colonne par colonne (modifiable cellule 2) ─────────────────────────

BG_N_ROWS = 30   # Nombre de lignes en haut de l'image (avant la 1re fibre)

                  # utilisées pour estimer le fond de chaque colonne.

                  # Vérifier avec plot_spatial_calibration que la 1re fibre

                  # se trouve bien au-delà de cette valeur en Y.

TILT_ASSOC_TOL = 25

# Systematic wavelength shift applied after pixel→nm polynomial
# (positive = shift towards longer wavelengths)
# Example: set to -15.0 if peaks appear 15 nm too high
WL_SHIFT_NM = 0.0

# ── Positions Y manuelles des 80 fibres (image HgAr redressée) ────────────
# Identifiées visuellement sur l'image de calibration HgAr après rotation.
# Ce sont les positions MASTER utilisées pour toutes les extractions.
# Format : FIBER_Y_MANUAL[i] = position Y en pixels de la fibre i (0-based).
FIBER_Y_MANUAL = np.array([
    328, 348, 366, 383, 403, 422, 441, 460, 478, 495,   # fibres  0– 9
    515, 534, 550, 570, 588, 609, 626, 644, 663, 682,   # fibres 10–19
    700, 719, 738, 757, 777, 794, 812, 830, 850, 868,   # fibres 20–29
    887, 908, 925, 943, 962, 980, 999, 1017, 1035, 1055, # fibres 30–39
    1073, 1091, 1110, 1128, 1148, 1166, 1184, 1205, 1222, 1240, # fibres 40–49
    1259, 1277, 1296, 1315, 1334, 1354, 1371, 1389, 1407, 1427, # fibres 50–59
    1445, 1465, 1482, 1500, 1518, 1537, 1557, 1575, 1594, 1611, # fibres 60–69
    1630, 1647, 1668, 1687, 1705, 1723, 1741, 1761, 1780, 1799, # fibres 70–79
], dtype=float)


# ── Inversion physique du faisceau de fibres ─────────────────────────────────
# Constat expérimental : le faisceau de fibres est monté à l'envers. La trace
# située en HAUT du détecteur (donc l'indice d'extraction 0, c'est-à-dire
# spectra[0]) correspond en réalité à la fibre PHYSIQUE 79 ; l'indice
# d'extraction 1 ↔ fibre physique 78 ; … ; indice 79 ↔ fibre physique 0.
#
# Conséquence : seule l'ASSOCIATION indice→angle (via FIBER_CONFIGS, qui est
# indexé par numéro de fibre PHYSIQUE) était fausse. Tout le reste du pipeline
# (détection, extraction, calibration intensité, SNR, aires…) est interne et
# cohérent en « ordre détecteur » : il n'est PAS affecté par l'inversion et ne
# doit donc PAS être modifié. On corrige l'inversion en UN SEUL endroit logique,
# au moment où un indice de fibre acquiert un sens physique (angle θ, φ).
#
# Mettre False pour désactiver (p.ex. si le faisceau est un jour remonté à
# l'endroit) — c'est l'unique commutateur à toucher.
FIBER_INDEX_REVERSED = True


def physical_fiber_index(det_index, n_fibers=None):
    """
    Convertit un indice d'EXTRACTION/DÉTECTEUR (0 = trace du haut = spectra[0])
    en numéro de fibre PHYSIQUE (celui utilisé comme clé dans FIBER_CONFIGS).

    Si FIBER_INDEX_REVERSED est True : physique = (N-1) - det_index.
    Sinon : identité (physique = det_index).

    Accepte un scalaire ou un array numpy.
    """
    if n_fibers is None:
        n_fibers = N_FIBERS
    det_index = np.asarray(det_index)
    if FIBER_INDEX_REVERSED:
        out = (n_fibers - 1) - det_index
    else:
        out = det_index.copy()
    # Renvoyer un int pur si l'entrée était scalaire (ergonomie)
    return int(out) if out.ndim == 0 else out


def get_fiber_angles(config_name, n_fibers=None):
    """
    Renvoie (phis, thetas), deux arrays de longueur N_FIBERS, INDEXÉS PAR
    L'INDICE D'EXTRACTION/DÉTECTEUR (le même que spectra[i]), avec l'inversion
    physique du faisceau déjà appliquée.

    FIBER_CONFIGS est indexé par numéro de fibre PHYSIQUE. Cette fonction est
    le SEUL endroit qui fait le pont indice-détecteur → fibre-physique → angle :
        phis[i], thetas[i] = angles de la fibre dont le spectre est spectra[i].

    Les fibres absentes de la config (ou hors plage) reçoivent NaN, exactement
    comme l'ancien code `cfg[i] if i in cfg else nan`.
    """
    if config_name not in FIBER_CONFIGS:
        raise ValueError(f"Configuration '{config_name}' inconnue. "
                         f"Disponibles : {list(FIBER_CONFIGS.keys())}")
    if n_fibers is None:
        n_fibers = N_FIBERS
    cfg    = FIBER_CONFIGS[config_name]['fibres']
    phis   = np.full(n_fibers, np.nan)
    thetas = np.full(n_fibers, np.nan)
    for det_i in range(n_fibers):
        phys_i = physical_fiber_index(det_i, n_fibers)
        if phys_i in cfg:
            phis[det_i]   = cfg[phys_i][0]
            thetas[det_i] = cfg[phys_i][1]
    return phis, thetas


# =============================================================================

# I/O

# =============================================================================

 

def load_image(path):

    """Charge une image TIFF (8 ou 16 bit) en array float64."""

    return np.array(Image.open(path)).astype(np.float64)

 

 

def list_science_images(folder, pattern="shot{:03d}.tif", start=0, end=546):

    """

    Retourne la liste des images science existantes dans `folder`.

    Seules les images réellement présentes sur le disque sont incluses.

    """

    folder = Path(folder)

    images = []

    for i in range(start, end + 1):

        p = folder / pattern.format(i)

        if p.exists():

            images.append(p)

    return images

 

 

def save_spectra(spectra, path, wl_axis=None):

    """

    Sauvegarde les spectres extraits.

      - .npy  : array numpy brut

      - .csv  : texte (première colonne = longueur d'onde si wl_axis fourni)

    """

    p = Path(path)

    np.save(str(p.with_suffix('.npy')), spectra)

    if wl_axis is not None:

        data = np.column_stack([wl_axis, spectra.T])   # (n_px, 1+n_fibres)

        header = "wl_nm," + ",".join(f"fibre{i}" for i in range(spectra.shape[0]))

    else:

        data   = spectra

        header = f"{spectra.shape[0]} fibres x {spectra.shape[1]} px"

    np.savetxt(str(p.with_suffix('.csv')), data, delimiter=',', header=header)

 

 

# =============================================================================

# CALIBRATION SPATIALE : positions des fibres + tilt

# =============================================================================

 

def find_calib_columns(arr, n=N_TILT_COLS):

    """Colonnes X correspondant aux raies d'émission les plus fortes."""

    x0     = int(arr.shape[1] * CALIB_FRAC[0])

    x1     = int(arr.shape[1] * CALIB_FRAC[1])

    xproj  = arr.mean(axis=0)[x0:x1]

    signal = np.clip(xproj - np.percentile(xproj, 20), 0, None)

    peaks, props = find_peaks(signal,

                              height=signal.max() * 0.05,

                              distance=15,

                              prominence=signal.max() * 0.03)

    if len(peaks) == 0:

        return np.argsort(xproj)[::-1][:n] + x0

    order = np.argsort(props['peak_heights'])[::-1]

    return np.sort(peaks[order[:min(n, len(order))]] + x0)

 

 

def column_profile(arr, col, half=8):

    """Profil Y moyen sur une bande de largeur 2*half+1 centrée en `col`."""

    c0 = max(0, col - half)

    c1 = min(arr.shape[1], col + half + 1)

    return arr[:, c0:c1].mean(axis=1)

 

 

def detect_fibers_in_profile(profile):
    """
    Detect fiber peaks in a Y column profile.
    Returns only the RAW detected peaks (no gap-filling here).

    Returns
    -------
    positions : float array  — sub-pixel Y positions of detected peaks
    is_real   : bool array   — all True (every returned position is real)
    """
    from scipy.ndimage import percentile_filter

    prof = gaussian_filter1d(median_filter(profile.astype(float), 3), sigma=1.0)

    bg  = percentile_filter(prof, percentile=10, size=BG_FILTER_SIZE)
    sig = np.clip(prof - bg, 0, None)

    peaks, _ = find_peaks(sig, distance=PEAK_MIN_DIST, prominence=PEAK_MIN_PROM)

    if len(peaks) < max(3, N_FIBERS // 4):
        soft_prom = max(PEAK_MIN_PROM * 0.30,
                        np.percentile(sig[sig > 0], 20) if sig.max() > 0 else 1.0)
        peaks, _ = find_peaks(sig, distance=PEAK_MIN_DIST, prominence=soft_prom)

    if len(peaks) < 3:
        return np.array([]), np.array([], dtype=bool)

    refined = _refine_peaks_subpixel(peaks, sig)
    refined = _trim_edge_peaks(refined)

    if len(refined) < 3:
        return np.array([]), np.array([], dtype=bool)

    is_real = np.ones(len(refined), dtype=bool)
    return refined, is_real


def _refine_peaks_subpixel(peaks, sig):
    """Sub-pixel refinement by intensity-weighted centroid around each peak."""
    refined = []
    for p in peaks:
        y0, y1 = max(0, p - 4), min(len(sig), p + 5)
        seg = np.clip(sig[y0:y1], 0, None)
        ys  = np.arange(y0, y1)
        refined.append(float(np.sum(ys * seg) / seg.sum()) if seg.sum() > 0 else float(p))
    return np.array(refined)


def _trim_edge_peaks(positions):
    """
    Remove isolated peaks at the edges of the detector that sit beyond
    a large gap (> EDGE_GAP_RATIO × estimated period) from the main cluster.
    """
    if len(positions) < 4:
        return positions

    spacings = np.diff(positions)
    med_sp   = np.median(spacings)
    single   = spacings[spacings < 1.5 * med_sp]
    period   = float(np.median(single)) if len(single) >= 2 else float(med_sp)
    threshold = EDGE_GAP_RATIO * period

    left = 0
    for i in range(len(spacings)):
        if spacings[i] <= threshold:
            left = i
            break
    else:
        return positions

    right = len(spacings)
    for i in range(len(spacings) - 1, -1, -1):
        if spacings[i] <= threshold:
            right = i + 1
            break

    return positions[left:right + 1]


# =============================================================================
# DÉTECTION ROBUSTE PER-IMAGE — deux passes
# =============================================================================

def _estimate_period(positions):
    """Robust inter-fiber period from detected peaks (median of single-gap spacings)."""
    if len(positions) < 2:
        return 18.0   # fallback raisonnable pour 80 fibres sur ~1500 px
    spacings = np.diff(positions)
    med_sp   = np.median(spacings)
    single   = spacings[spacings < 1.5 * med_sp]
    return float(np.median(single)) if len(single) >= 2 else float(med_sp)


def detect_fibers_robust(profile, n_fibers=None):
    """
    Détection robuste des fibres dans un profil Y — stratégie en 2 passes.

    Passe 1 : détection standard (find_peaks avec prominence forte).
              On en tire la période inter-fibre et le domaine Y actif.

    Passe 2 : on construit un grid régulier ancré sur le premier pic réel.
              À chaque position du grid sans pic assigné, on fait une
              recherche CIBLÉE de maximum local dans le profil brut
              (fenêtre ±0.4 × period) avec un seuil très bas.
              Cela récupère les fibres faibles que la passe 1 a ratées.

    Différence clé avec l'ancien code :
    - L'ancien code INTERPOLAIT mathématiquement les trous (positions fictives).
    - Ce code CHERCHE le signal réel à la position attendue.
    - Si aucun signal n'est trouvé, la position est quand même interpolée
      linéairement, mais ces cas sont flaggés is_real=False.

    Returns
    -------
    positions : float array (n_found,) — Y positions (sub-pixel)
    is_real   : bool array  (n_found,) — True si détecté, False si interpolé
    period    : float — estimated inter-fiber spacing in pixels
    """
    from scipy.ndimage import percentile_filter
    if n_fibers is None:
        n_fibers = N_FIBERS

    prof = gaussian_filter1d(median_filter(profile.astype(float), 3), sigma=1.0)
    bg   = percentile_filter(prof, percentile=10, size=BG_FILTER_SIZE)
    sig  = np.clip(prof - bg, 0, None)

    # ── Passe 1 : pics forts ────────────────────────────────────────────────
    peaks1, props1 = find_peaks(sig, distance=PEAK_MIN_DIST, prominence=PEAK_MIN_PROM)

    if len(peaks1) < 5:
        # Seuil adaptatif si presque rien n'est trouvé
        soft_prom = max(PEAK_MIN_PROM * 0.25,
                        np.percentile(sig[sig > 0], 15) if sig.max() > 0 else 1.0)
        peaks1, props1 = find_peaks(sig, distance=PEAK_MIN_DIST, prominence=soft_prom)

    if len(peaks1) < 3:
        return np.array([]), np.array([], dtype=bool), 18.0

    pass1 = _refine_peaks_subpixel(peaks1, sig)
    pass1 = _trim_edge_peaks(pass1)

    if len(pass1) < 3:
        return np.array([]), np.array([], dtype=bool), 18.0

    period = _estimate_period(pass1)

    # ── Construction du grid attendu ────────────────────────────────────────
    y_first = float(pass1[0])
    y_last  = float(pass1[-1])
    n_slots = int(round((y_last - y_first) / period)) + 1
    # Ne pas dépasser n_fibers
    n_slots = min(n_slots, n_fibers)
    grid    = y_first + np.arange(n_slots) * period

    # ── Assigner les pics de passe 1 au grid ────────────────────────────────
    assigned_y    = np.full(n_slots, np.nan)
    assigned_real = np.zeros(n_slots, dtype=bool)
    used          = np.zeros(len(pass1), dtype=bool)
    tol           = 0.5 * period

    for gi in range(n_slots):
        dists = np.abs(pass1 - grid[gi])
        best  = int(np.argmin(dists))
        if dists[best] < tol and not used[best]:
            assigned_y[gi]    = pass1[best]
            assigned_real[gi] = True
            used[best]        = True

    # ── Passe 2 : recherche guidée aux positions manquantes ─────────────────
    # Pour chaque slot vide, chercher un maximum local dans le signal brut
    # dans une fenêtre centrée sur la position attendue du grid.
    search_hw = int(0.4 * period)
    # Seuil bas pour la passe 2 : on accepte tout ce qui dépasse le bruit
    noise_floor = np.percentile(sig[sig > 0], 10) if (sig > 0).any() else 0.0
    pass2_min_height = max(noise_floor * 2.0, PEAK_MIN_PROM * 0.15)

    for gi in range(n_slots):
        if assigned_real[gi]:
            continue

        expected_y = grid[gi]
        y0 = max(0, int(expected_y) - search_hw)
        y1 = min(len(sig), int(expected_y) + search_hw + 1)

        if y1 <= y0:
            continue

        window = sig[y0:y1]
        local_peaks, local_props = find_peaks(window, distance=3, prominence=pass2_min_height * 0.5)

        if len(local_peaks) == 0:
            # Fallback : prendre le maximum brut de la fenêtre si significatif
            if window.max() > pass2_min_height:
                best_local = int(np.argmax(window))
                local_peaks = np.array([best_local])
            else:
                continue

        # Parmi les pics locaux, prendre le plus proche de expected_y
        local_abs = local_peaks + y0
        dists     = np.abs(local_abs.astype(float) - expected_y)
        best_idx  = int(np.argmin(dists))
        best_pos  = local_abs[best_idx]

        # Raffinement sub-pixel par centroid
        cy0 = max(0, best_pos - 4)
        cy1 = min(len(sig), best_pos + 5)
        seg = np.clip(sig[cy0:cy1], 0, None)
        ys  = np.arange(cy0, cy1, dtype=float)
        if seg.sum() > 0:
            refined_pos = float(np.sum(ys * seg) / seg.sum())
        else:
            refined_pos = float(best_pos)

        # Vérifier que cette position ne chevauche pas un voisin déjà assigné
        conflict = False
        for gj in range(n_slots):
            if gj == gi or np.isnan(assigned_y[gj]):
                continue
            if abs(refined_pos - assigned_y[gj]) < 0.5 * period:
                conflict = True
                break

        if not conflict:
            assigned_y[gi]    = refined_pos
            assigned_real[gi] = True   # trouvé dans le signal, c'est un vrai pic

    # ── Passe 3 : interpolation linéaire pour les rares slots encore vides ──
    missing = np.isnan(assigned_y)
    n_missing = int(missing.sum())
    if n_missing > 0 and n_missing < n_slots:
        valid_idx = np.where(~missing)[0]
        valid_y   = assigned_y[~missing]
        fy = interp1d(valid_idx, valid_y, kind='linear',
                       bounds_error=False, fill_value='extrapolate')
        for gi in range(n_slots):
            if np.isnan(assigned_y[gi]):
                assigned_y[gi]    = float(fy(gi))
                assigned_real[gi] = False  # interpolé, pas détecté

    return assigned_y, assigned_real, period


def detect_fibers_for_image(arr_rot, n_fibers=None, col_ref=None, verbose=False):
    """
    Détection autonome des fibres pour UNE image (déjà rotée).

    Stratégie :
    1. Moyenne le profil Y sur plusieurs colonnes (plus robuste qu'une seule)
    2. Appelle detect_fibers_robust() pour la détection en 2 passes
    3. Vérifie la cohérence (espacements, nombre de fibres)

    Parameters
    ----------
    arr_rot   : ndarray (H, W) — image rotée (fibres horizontales)
    n_fibers  : int — nombre attendu de fibres (défaut: N_FIBERS)
    col_ref   : int or None — colonne de référence (None = auto-détection)
    verbose   : bool

    Returns
    -------
    fiber_y   : float array (n_found,) — Y position of each fiber
    is_real   : bool array  (n_found,) — True if detected, False if interpolated
    period    : float — inter-fiber spacing in pixels
    col_used  : int — colonne de référence utilisée
    """
    if n_fibers is None:
        n_fibers = N_FIBERS

    H, W = arr_rot.shape

    # ── Choix de la colonne de référence ────────────────────────────────────
    if col_ref is None:
        # Tester plusieurs colonnes, garder celle avec le plus de détections
        test_cols = np.linspace(W * 0.2, W * 0.8, 7, dtype=int)
        best_col, best_n = int(test_cols[0]), 0
        for c in test_cols:
            prof = column_profile(arr_rot, int(c), half=12)
            p, _, _ = detect_fibers_robust(prof, n_fibers)
            if len(p) > best_n:
                best_n, best_col = len(p), int(c)
        col_ref = best_col

    # ── Profil multi-colonnes (moyenne sur ±3 colonnes autour de col_ref) ──
    n_avg_cols = 5
    cols_to_avg = np.clip(
        np.arange(col_ref - n_avg_cols * 20, col_ref + n_avg_cols * 20 + 1, 20),
        0, W - 1
    ).astype(int)
    # Garder uniquement les colonnes suffisamment espacées
    cols_to_avg = np.unique(cols_to_avg)

    profiles = []
    for c in cols_to_avg:
        profiles.append(column_profile(arr_rot, int(c), half=12))
    avg_profile = np.mean(profiles, axis=0)

    # ── Détection robuste ──────────────────────────────────────────────────
    fiber_y, is_real, period = detect_fibers_robust(avg_profile, n_fibers)

    if verbose:
        n_det = int(is_real.sum())
        n_int = len(fiber_y) - n_det
        print(f"  Per-image detection: col_ref={col_ref}  "
              f"|  {n_det} detected + {n_int} interpolated = {len(fiber_y)} fibers  "
              f"|  period={period:.2f} px")

    return fiber_y, is_real, period, col_ref


def plot_fiber_detection_diagnostic(arr_rot, fiber_y, is_real, period,
                                     col_ref, image_name=""):
    """
    Diagnostic plots pour la détection per-image.

    4 panneaux :
    1. Image rotée avec traces des fibres (cyan=détecté, orange=interpolé)
    2. Profil Y à col_ref avec pics marqués
    3. Inter-fiber spacing (doit être ~constant autour de period)
    4. Résidu position vs grid régulier (doit être petit)

    Usage dans le notebook :
        fiber_y, is_real, period, col = sf.detect_fibers_for_image(arr_rot)
        sf.plot_fiber_detection_diagnostic(arr_rot, fiber_y, is_real, period, col, "shot300")
    """
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec

    n = len(fiber_y)
    valid = ~np.isnan(fiber_y)
    n_det = int(is_real.sum()) if len(is_real) > 0 else 0

    fig = plt.figure(figsize=(20, 12))
    gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.30)

    # ── Panel 1 : Image avec traces ──────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    vmax = np.percentile(arr_rot, 99.5)
    ax1.imshow(arr_rot, cmap='hot', aspect='auto', vmin=0, vmax=vmax)
    for i in range(n):
        if np.isnan(fiber_y[i]):
            continue
        col = 'cyan' if is_real[i] else 'orange'
        ax1.axhline(fiber_y[i], color=col, lw=0.5, alpha=0.7)
    ax1.set_title(f"{image_name} — {n} fibers\n"
                  f"cyan={n_det} detected  |  orange={n - n_det} interpolated")
    ax1.set_xlabel("X (px)"); ax1.set_ylabel("Y (px)")

    # ── Panel 2 : Profil Y avec détections ───────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    prof = column_profile(arr_rot, col_ref, half=12)
    yy   = np.arange(len(prof))
    ax2.plot(prof, yy, 'steelblue', lw=0.7)
    for i in range(n):
        if np.isnan(fiber_y[i]):
            continue
        col = 'cyan' if is_real[i] else 'orange'
        ms  = 6 if is_real[i] else 4
        mk  = 'o' if is_real[i] else 's'
        y_pos = fiber_y[i]
        # Intensité approximative à cette position
        y_int = int(round(y_pos))
        if 0 <= y_int < len(prof):
            ax2.plot(prof[y_int], y_pos, mk, color=col, ms=ms, zorder=5)
    ax2.invert_yaxis()
    ax2.set_xlabel("Intensity (ADU)"); ax2.set_ylabel("Y (px)")
    ax2.set_title(f"Y profile — col {col_ref}")
    ax2.grid(True, alpha=0.2)

    # ── Panel 3 : Inter-fiber spacing ────────────────────────────────────────
    ax3 = fig.add_subplot(gs[1, 0])
    valid_y = fiber_y[valid]
    if len(valid_y) > 1:
        spacings = np.diff(valid_y)
        fib_idx  = np.arange(len(spacings))
        colors_s = ['tomato' if abs(s - period) > 0.3 * period else 'steelblue'
                     for s in spacings]
        ax3.bar(fib_idx, spacings, color=colors_s, width=0.8, edgecolor='none')
        ax3.axhline(period, ls='--', color='k', lw=1.5,
                     label=f"Period = {period:.2f} px")
        ax3.axhspan(period * 0.85, period * 1.15, alpha=0.1, color='green',
                     label="±15% tolerance")
        ax3.set_xlabel("Fiber pair index")
        ax3.set_ylabel("Spacing (px)")
        ax3.set_title("Inter-fiber spacing (red = deviant)")
        ax3.legend(fontsize=8)
        ax3.grid(True, alpha=0.2)
    else:
        ax3.text(0.5, 0.5, "Not enough fibers", transform=ax3.transAxes,
                 ha='center', fontsize=14)

    # ── Panel 4 : Résidu vs grid régulier ────────────────────────────────────
    ax4 = fig.add_subplot(gs[1, 1])
    if len(valid_y) > 1:
        # Grid parfait depuis la première fibre
        ideal_grid = valid_y[0] + np.arange(len(valid_y)) * period
        residuals  = valid_y - ideal_grid
        colors_r   = ['cyan' if is_real[valid][i] else 'orange'
                       for i in range(len(valid_y))]
        ax4.bar(np.arange(len(residuals)), residuals, color=colors_r,
                 width=0.8, edgecolor='none')
        ax4.axhline(0, ls='-', color='k', lw=0.8)
        ax4.axhspan(-period * 0.1, period * 0.1, alpha=0.1, color='green')
        ax4.set_xlabel("Fiber index")
        ax4.set_ylabel("Residual (px)")
        ax4.set_title(f"Position residual vs regular grid\n"
                       f"RMS = {np.sqrt(np.mean(residuals**2)):.2f} px")
        ax4.grid(True, alpha=0.2)
    else:
        ax4.text(0.5, 0.5, "Not enough fibers", transform=ax4.transAxes,
                 ha='center', fontsize=14)

    plt.suptitle(f"FIBER DETECTION DIAGNOSTIC — {image_name}\n"
                 f"{n_det} detected + {n - n_det} interpolated = {n} fibers  "
                 f"|  period = {period:.2f} px",
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.show()


def _best_ref_column(arr, calib_cols):

    """Colonne parmi calib_cols qui donne le plus de fibres détectables."""

    best_col, best_n = int(calib_cols[0]), 0

    for col in calib_cols:

        p, _ = detect_fibers_in_profile(column_profile(arr, int(col)))

        if len(p) > best_n:

            best_n, best_col = len(p), int(col)

    return best_col

 

 

def measure_tilt(arr, calib_cols, verbose=True):

    """

    Mesure la pente dY/dX (tilt) de chaque fibre sur plusieurs colonnes.

 

    Returns dict avec col_ref, fiber_y_ref, tilt_slopes, fiber_is_real.

    """

    col_ref = _best_ref_column(arr, calib_cols)

    p_ref, r_ref = detect_fibers_in_profile(column_profile(arr, col_ref))

    n = len(p_ref)

    if verbose:

        print(f"  Reference col.: {col_ref}  |  "

              f"{int(r_ref.sum())} detected + {int((~r_ref).sum())} interpolated = {n}")

 

    col_data = {}

    for col in calib_cols:

        col = int(col)

        peaks, _ = detect_fibers_in_profile(column_profile(arr, col))

        if len(peaks) < 3:

            continue

        matched, used = np.full(n, np.nan), np.zeros(len(peaks), dtype=bool)

        for i, ry in enumerate(p_ref):

            d = np.abs(peaks - ry)

            b = np.argmin(d)

            if d[b] < TILT_ASSOC_TOL and not used[b]:

                matched[i], used[b] = peaks[b], True

        col_data[col] = matched

 

    cols_arr = np.array(sorted(col_data.keys()), dtype=float)

    slopes   = np.zeros(n)

    for i in range(n):

        ys = np.array([col_data[c][i] for c in cols_arr.astype(int)])

        v  = ~np.isnan(ys)

        if v.sum() >= 2:

            slopes[i], _ = np.polyfit(cols_arr[v] - col_ref, ys[v], 1)

 

    if verbose:

        print(f"  Tilt: mean={slopes.mean():.4f}  "

              f"min={slopes.min():.4f}  max={slopes.max():.4f} px/px")

    return {'col_ref': col_ref, 'fiber_y_ref': p_ref,

            'tilt_slopes': slopes, 'fiber_is_real': r_ref}

 

 

def assign_to_grid(info, n_fibers=N_FIBERS, max_extend=MAX_EXTEND,
                   verbose=True):
    """
    Assign detected fiber peaks to a regular grid of n_fibers slots.

    Key rules
    ---------
    • The grid is anchored to the FIRST detected peak.
    • It covers at least up to the LAST detected peak.
    • If n_use < n_fibers, the grid may be EXTENDED by up to `max_extend`
      extra slots beyond the last detected peak.  These extrapolated slots
      are marked is_real=False and their positions/slopes are linearly
      extrapolated from the matched peaks.
    • Slots beyond this extended range remain NaN (no signal).

    Algorithm
    ---------
    1. Estimate the inter-fiber period from "single-gap" spacings.
    2. Build the grid starting at y_first, stepping by period, up to
       n_slots = round((y_last - y_first) / period) + 1 + max_extend
       (clamped to n_fibers).
    3. Match each detected peak to its nearest grid slot.
    4. Interpolate/extrapolate within the extended range using the
       matched peaks as anchors.
    5. Slots beyond the extended range remain NaN.
    """
    y, sl, real = info['fiber_y_ref'], info['tilt_slopes'], info['fiber_is_real']

    if len(y) == 0:
        ay  = np.full(n_fibers, np.nan)
        asl = np.zeros(n_fibers)
        ar  = np.zeros(n_fibers, dtype=bool)
        return {'col_ref': info['col_ref'], 'fiber_y_ref': ay,
                'tilt_slopes': asl, 'fiber_is_real': ar}

    if len(y) == n_fibers:
        return info

    # ── Step 1: robust period estimate ───────────────────────────────────────
    spacings = np.diff(y)
    med_sp   = np.median(spacings)
    # Only "single-gap" spacings (< 1.5 × median) contribute to the period
    single   = spacings[spacings < 1.5 * med_sp]
    period   = float(np.median(single)) if len(single) >= 2 else float(med_sp)

    if verbose:
        print(f"  Period estimate : {period:.2f} px/fiber  "
              f"(from {len(single)}/{len(spacings)} single-gap spacings)")

    # ── Step 2: build grid anchored to first detected peak ───────────────────
    y_first = float(y[0])   # first real peak  — grid starts HERE
    y_last  = float(y[-1])  # last real peak

    # Slots from first to last detected peak
    n_core  = int(round((y_last - y_first) / period)) + 1
    # Allow extending by up to max_extend extra slots beyond last peak
    n_use   = min(n_core + max_extend, n_fibers)

    grid = y_first + np.arange(n_use) * period

    if verbose:
        print(f"  Signal zone     : Y={y_first:.1f} – {y_last:.1f} px  "
              f"→ {n_core} core + {n_use - n_core} extended = {n_use} grid slots "
              f"(of {n_fibers} total)")

    # ── Step 3: match detected peaks to grid slots ────────────────────────────
    ay   = np.full(n_fibers, np.nan, dtype=float)
    asl  = np.zeros(n_fibers, dtype=float)
    ar   = np.zeros(n_fibers, dtype=bool)
    used = np.zeros(len(y), dtype=bool)

    tol = 0.55 * period
    for gi in range(n_use):
        d = np.abs(y - grid[gi])
        b = int(np.argmin(d))
        if d[b] < tol and not used[b]:
            ay[gi], asl[gi], ar[gi] = y[b], sl[b], real[b]
            used[b] = True

    # ── Step 4: fill gaps by interpolation + limited extrapolation ────────────
    # Internal gaps (between matched peaks) → interpolate
    # Extended slots (beyond last matched peak) → extrapolate linearly
    valid = ~np.isnan(ay[:n_use])
    n_matched = int(valid.sum())
    if n_matched >= 2:
        idx_v = np.where(valid)[0]
        # Use fill_value='extrapolate' for the extended region
        fy  = interp1d(idx_v, ay[:n_use][valid],  kind='linear',
                       bounds_error=False, fill_value='extrapolate')
        fsl = interp1d(idx_v, asl[:n_use][valid], kind='linear',
                       bounds_error=False, fill_value='extrapolate')
        for gi in range(n_use):
            if np.isnan(ay[gi]):
                ay[gi]  = float(fy(gi))
                asl[gi] = float(fsl(gi))
                # ar[gi] stays False — these are interpolated/extrapolated

    valid2      = ~np.isnan(ay)
    n_filled    = int(valid2.sum()) - n_matched
    n_extended  = max(0, n_use - n_core)
    n_empty     = n_fibers - int(valid2.sum())
    if verbose:
        print(f"  Grid: {n_matched}/{n_fibers} matched  "
              f"|  {n_filled} gap-filled (incl. {n_extended} extended)  "
              f"|  {n_empty} empty/NaN (outside signal zone)")

    return {'col_ref': info['col_ref'], 'fiber_y_ref': ay,
            'tilt_slopes': asl, 'fiber_is_real': ar}



# =============================================================================

# CALIBRATION SPECTRALE : pixels -> longueurs d'onde

# =============================================================================

 

# =============================================================================
# CALIBRATION SPECTRALE AUTOMATIQUE — comb-matching "no-guess" (universel)
# =============================================================================
# Idée importée de HG-FOAM mais rendue plus universelle : AUCUNE longueur d'onde
# n'est supposée a priori (leur code fixe "pic le plus grand = 871.667 nm", ce
# qui est propre à leur montage). Ici on cherche la dispersion pixel→nm qui
# apparie le PLUS de pics détectés au catalogue HGAR_LINES, via une recherche à
# DEUX ancres (deux pics ↔ deux raies définissent une droite). On garde ensuite
# la meilleure solution, on la raffine en polynôme, et on refuse si trop peu de
# raies matchent (repli sur les paires manuelles).

def _spectral_peaks_subpixel(spectrum, smooth_px=WL_AUTO_SMOOTH_PX,
                             prom_frac=WL_AUTO_PROM_FRAC,
                             min_dist=WL_AUTO_MIN_DIST_PX,
                             max_peaks=WL_AUTO_MAX_PEAKS):
    """
    Détecte les pics de raies dans un spectre 1D et raffine leur position en
    sous-pixel par interpolation parabolique (3 points), moins biaisée qu'un
    centroïde en présence de fond/asymétrie.

    Returns
    -------
    peaks_px : float array — positions X (sous-pixel), triées croissant
    heights  : float array — intensité (spectre lissé) au pic
    """
    sp = np.asarray(spectrum, dtype=float)
    # Lissage par moyenne glissante (fenêtre impaire)
    w = int(smooth_px)
    if w > 1:
        if w % 2 == 0:
            w += 1
        kernel = np.ones(w) / w
        sm = np.convolve(np.nan_to_num(sp, nan=np.nanmedian(sp)), kernel, mode='same')
    else:
        sm = sp.copy()

    dyn = np.nanpercentile(sm, 99) - np.nanpercentile(sm, 1)
    if not np.isfinite(dyn) or dyn <= 0:
        return np.array([]), np.array([])

    peaks, props = find_peaks(sm, prominence=prom_frac * dyn,
                              distance=max(1, int(min_dist)))
    if len(peaks) == 0:
        return np.array([]), np.array([])

    # Raffinement parabolique sous-pixel
    refined = []
    for k in peaks:
        if 0 < k < len(sm) - 1:
            y0, y1, y2 = sm[k - 1], sm[k], sm[k + 1]
            denom = (y0 - 2 * y1 + y2)
            delta = 0.5 * (y0 - y2) / denom if denom != 0 else 0.0
            delta = float(np.clip(delta, -1.0, 1.0))
            refined.append(k + delta)
        else:
            refined.append(float(k))
    refined = np.array(refined)
    heights = sm[peaks]

    # Garder les plus intenses si trop nombreux
    if max_peaks is not None and len(refined) > int(max_peaks):
        keep = np.argsort(heights)[::-1][:int(max_peaks)]
        refined, heights = refined[keep], heights[keep]

    order = np.argsort(refined)
    return refined[order], heights[order]


def _greedy_unique_match(pred_wl, cat_wl, tol_nm):
    """
    Appariement 1-à-1 (chaque pic ↔ au plus une raie, et inversement), en
    commençant par les couples les plus proches. Renvoie une liste de triplets
    (index_pic, index_raie, erreur_nm).
    """
    cand = []
    for i, pw in enumerate(pred_wl):
        d = np.abs(cat_wl - pw)
        j = np.where(d <= tol_nm)[0]
        for jj in j:
            cand.append((i, int(jj), float(d[jj])))
    cand.sort(key=lambda t: t[2])
    used_p, used_l, out = set(), set(), []
    for i, j, e in cand:
        if i in used_p or j in used_l:
            continue
        used_p.add(i); used_l.add(j); out.append((i, j, e))
    return out


def catalog_wavelengths(catalog=None, dedupe_tol=0.05):
    """Longueurs d'onde uniques (triées) du catalogue HGAR_LINES."""
    if catalog is None:
        catalog = HGAR_LINES
    wl = np.array(sorted(catalog.keys()), dtype=float)
    if len(wl) == 0:
        return wl
    keep = [wl[0]]
    for v in wl[1:]:
        if v - keep[-1] > dedupe_tol:
            keep.append(v)
    return np.array(keep)


def auto_wl_calibration(mean_spectrum, n_cols=None, catalog=None,
                        poly_deg=WL_POLY_DEG,
                        match_tol_nm=WL_AUTO_MATCH_TOL_NM,
                        min_matched=WL_AUTO_MIN_MATCHED,
                        rms_max_nm=WL_AUTO_RMS_MAX_NM,
                        n_bright=WL_AUTO_N_BRIGHT,
                        min_pix_sep=WL_AUTO_MIN_PIX_SEP,
                        disp_range=WL_AUTO_DISP_RANGE,
                        verbose=True):
    """
    Calibration spectrale AUTOMATIQUE sans paires codées en dur (universelle).

    Algorithme
    ----------
    1. Détecte les pics de raies dans `mean_spectrum` (sous-pixel).
    2. Recherche à DEUX ANCRES : pour chaque couple (pic_a, pic_b) parmi les
       N pics les plus intenses et chaque couple compatible (raie_i, raie_j) du
       catalogue, deux points → une droite pixel→nm. On prédit les longueurs
       d'onde de tous les pics et on les apparie au catalogue (tolérance).
       On garde la droite qui apparie LE PLUS de pics (puis plus petit RMS,
       puis plus grand recouvrement). Aucune longueur d'onde n'est supposée.
    3. Raffinement polynomial itératif sur les paires appariées.
    4. Garde-fou : refuse si < `min_matched` raies appariées.

    Returns
    -------
    dict avec :
        'ok'         : bool — True si la solution passe les garde-fous
        'wl_axis'    : (n_cols,) axe pixel→nm (si ok)
        'coeffs'     : coefficients polynomiaux (np.polyfit, degré effectif)
        'pairs'      : liste de (pixel, wl_nm) appariées (= WL_CALIB_PAIRS auto)
        'residuals'  : résidus (nm) sur les paires appariées
        'rms_nm'     : RMS des résidus
        'n_matched'  : nb de raies appariées
        'peaks_px'   : positions des pics détectés
        'reason'     : message si échec
    """
    cat = catalog_wavelengths(catalog)
    peaks_px, heights = _spectral_peaks_subpixel(mean_spectrum)

    fail = {'ok': False, 'wl_axis': None, 'coeffs': None, 'pairs': [],
            'residuals': np.array([]), 'rms_nm': np.nan, 'n_matched': 0,
            'peaks_px': peaks_px}

    if len(peaks_px) < min_matched or len(cat) < 2:
        fail['reason'] = (f"Trop peu de pics ({len(peaks_px)}) ou de raies "
                          f"catalogue ({len(cat)}).")
        return fail

    # Pics-ancres candidats = les plus intenses
    bright = np.argsort(heights)[::-1][:int(n_bright)]
    dmin, dmax = disp_range

    # Scoring VECTORISÉ : pour un mapping candidat, on compte les pics dont la
    # longueur d'onde prédite tombe à < tol d'une raie du catalogue (via
    # searchsorted, O(P log L), sans boucle Python). C'est un proxy fidèle du
    # nombre d'appariements, et ~1000× plus rapide que le matching exact appelé
    # dans la boucle chaude. Le matching 1-à-1 exact n'est fait qu'une fois, sur
    # le meilleur (slope, intercept) retenu.
    # Score pondéré par l'INTENSITÉ : les vraies raies sont brillantes, les pics
    # de bruit faibles. Sommer la hauteur des pics appariés fait ressortir la
    # vraie dispersion (qui explique les pics intenses) au-dessus des alias (qui
    # n'accrochent que des pics faibles) — cf. diagnostic couverture 15/15 vs 2/15.
    heights_norm = heights / (np.max(heights) + 1e-12)

    def _fast_score(pred):
        idx = np.clip(np.searchsorted(cat, pred), 1, len(cat) - 1)
        d = np.minimum(np.abs(pred - cat[idx - 1]), np.abs(pred - cat[idx]))
        within = d < match_tol_nm
        weighted = float(np.sum(heights_norm[within]))
        return weighted, -float(d[within].sum())

    candidates = []   # (fast_key, slope, intercept)
    for ia in range(len(bright)):
        for ib in range(ia + 1, len(bright)):
            xa = peaks_px[bright[ia]]
            xb = peaks_px[bright[ib]]
            dx = xb - xa
            if abs(dx) < min_pix_sep:
                continue
            lo, hi = dmin * abs(dx), dmax * abs(dx)
            for i in range(len(cat)):
                wa = cat[i]
                sep = np.abs(cat - wa)
                for j in np.where((sep >= lo) & (sep <= hi))[0]:
                    slope = (cat[j] - wa) / dx
                    intercept = wa - slope * xa
                    wscore, negd = _fast_score(slope * peaks_px + intercept)
                    if wscore > 0:
                        candidates.append(((wscore, negd), slope, intercept))

    if not candidates:
        fail['reason'] = "Aucune correspondance comb trouvée."
        return fail

    if n_cols is None:
        n_cols = int(np.ceil(peaks_px.max())) + 1
    x_full = np.arange(n_cols)

    # Ajustement ROBUSTE (sigma-clipping) : quelques appariements parasites
    # (pics de bruit tombant par hasard près d'une raie) ne doivent pas gonfler
    # le RMS d'une bonne solution ni la faire rejeter. On ajuste, on écarte les
    # résidus aberrants (> clip_nm), on ré-ajuste, en gardant le degré le plus
    # élevé (≤ poly_deg) qui donne un axe strictement monotone.
    clip_nm = max(0.4, 2.0 * rms_max_nm)

    def _robust_fit(x_m, y_m):
        mask = np.ones(len(x_m), dtype=bool)
        coeffs = None
        for _ in range(4):
            n_in = int(mask.sum())
            if n_in < 2:
                break
            deg_ok = None
            for deg in range(max(1, min(poly_deg, n_in - 1)), 0, -1):
                c = np.polyfit(x_m[mask], y_m[mask], deg)
                ax = np.polyval(c, x_full)
                dd = np.diff(ax)
                if np.all(dd > 0) or np.all(dd < 0):
                    deg_ok, coeffs = deg, c
                    break
            if coeffs is None:
                coeffs = np.polyfit(x_m[mask], y_m[mask], 1)
            resid_all = y_m - np.polyval(coeffs, x_m)
            new_mask = np.abs(resid_all) < clip_nm
            if new_mask.sum() < 2:
                break
            if np.array_equal(new_mask, mask):
                break
            mask = new_mask
        if coeffs is None:
            return None
        rms = float(np.sqrt(np.mean((y_m[mask] - np.polyval(coeffs, x_m[mask])) ** 2)))
        return coeffs, mask, rms

    def _evaluate(slope, intercept):
        pred = slope * peaks_px + intercept
        matches = _greedy_unique_match(pred, cat, match_tol_nm)
        if len(matches) < 2:
            return None
        pk = np.array([m[0] for m in matches]); ln = np.array([m[1] for m in matches])
        fit = _robust_fit(peaks_px[pk], cat[ln])
        if fit is None:
            return None
        coeffs, inlier, rms = fit
        inlier_matches = [matches[k] for k in range(len(matches)) if inlier[k]]
        return inlier_matches, rms, coeffs

    # Garder les K meilleurs candidats (fast-score), les évaluer robustement,
    # ne retenir que ceux au RMS acceptable, puis choisir (n_inliers, RMS mini).
    import heapq
    topk = heapq.nlargest(40, candidates, key=lambda c: c[0])

    best = None   # ((n_inliers, -rms), matches, coeffs)
    for _, slope, intercept in topk:
        ev = _evaluate(slope, intercept)
        if ev is None:
            continue
        inlier_matches, rms, coeffs = ev
        if rms > rms_max_nm or len(inlier_matches) < 2:
            continue
        key = (len(inlier_matches), -rms)
        if best is None or key > best[0]:
            best = (key, inlier_matches, coeffs)

    if best is None:
        fail['reason'] = ("Aucun candidat au RMS acceptable "
                          f"(< {rms_max_nm} nm) après évaluation robuste.")
        return fail

    matches = best[1]

    # ── Raffinement : re-matcher avec le fit courant, ré-ajuster robustement ──
    for _ in range(6):
        pk_i = np.array([m[0] for m in matches]); ln_i = np.array([m[1] for m in matches])
        fit = _robust_fit(peaks_px[pk_i], cat[ln_i])
        if fit is None:
            break
        coeffs = fit[0]
        pred = np.polyval(coeffs, peaks_px)
        new_matches = _greedy_unique_match(pred, cat, match_tol_nm)
        # garder seulement les inliers du nouveau matching
        if len(new_matches) >= 2:
            pk2 = np.array([m[0] for m in new_matches]); ln2 = np.array([m[1] for m in new_matches])
            f2 = _robust_fit(peaks_px[pk2], cat[ln2])
            if f2 is not None:
                new_matches = [new_matches[k] for k in range(len(new_matches)) if f2[1][k]]
        if {(m[0], m[1]) for m in new_matches} == {(m[0], m[1]) for m in matches}:
            matches = new_matches
            break
        if len(new_matches) >= 2:
            matches = new_matches

    pk_i = np.array([m[0] for m in matches]); ln_i = np.array([m[1] for m in matches])
    x_m = peaks_px[pk_i]; y_m = cat[ln_i]
    fit = _robust_fit(x_m, y_m)
    coeffs, inlier, rms = fit
    x_m, y_m = x_m[inlier], y_m[inlier]
    resid = y_m - np.polyval(coeffs, x_m)

    n_matched = len(x_m)
    # ── Garde-fous anti-alias / mauvaise solution ──────────────────────────
    # 1) nombre mini de raies appariées
    if n_matched < min_matched:
        fail['reason'] = (f"Seulement {n_matched} raies appariées "
                          f"(min {min_matched}).")
        fail['n_matched'] = n_matched
        return fail
    # 2) RMS : une vraie calibration HgAr a un RMS sub-nm ; un alias-miroir a un
    #    RMS élevé. C'est le test qui distingue le mieux une solution correcte.
    if rms > rms_max_nm:
        fail['reason'] = (f"RMS trop élevé ({rms:.3f} > {rms_max_nm} nm) : "
                          f"solution probablement aliasée.")
        fail['n_matched'] = n_matched
        return fail
    # 3) couverture pixel : les raies appariées doivent couvrir une fraction
    #    notable du détecteur (un alias apparie souvent un sous-ensemble regroupé).
    if n_cols is not None and np.ptp(x_m) < 0.35 * n_cols:
        fail['reason'] = (f"Couverture pixel insuffisante "
                          f"({np.ptp(x_m):.0f} < {0.35*n_cols:.0f} px).")
        fail['n_matched'] = n_matched
        return fail
    # 4) couverture des pics BRILLANTS : une vraie solution explique la plupart
    #    des pics les plus intenses (les parasites de bruit sont faibles). Un
    #    alias les laisse non appariés. Test très discriminant (15/15 vs 2/15).
    n_bright_chk = min(12, len(peaks_px))
    bright_px = peaks_px[np.argsort(heights)[::-1][:n_bright_chk]]
    matched_px = x_m
    covered = sum(1 for bp in bright_px
                  if np.min(np.abs(matched_px - bp)) < 2.0)
    if covered < 0.6 * n_bright_chk:
        fail['reason'] = (f"Trop de pics brillants non expliqués "
                          f"({covered}/{n_bright_chk} couverts) : solution suspecte.")
        fail['n_matched'] = n_matched
        return fail

    order = np.argsort(x_m)
    pairs = [(float(x_m[k]), float(y_m[k])) for k in order]

    wl_axis = np.polyval(coeffs, np.arange(n_cols)) + WL_SHIFT_NM

    if verbose:
        print(f"  [WL auto] {n_matched} raies appariées  "
              f"|  RMS={rms:.3f} nm  |  dispersion≈"
              f"{(wl_axis[-1]-wl_axis[0])/n_cols:.4f} nm/px  |  degré {len(coeffs)-1}")
        for (px, wl), r in zip(pairs, resid[order]):
            sym = HGAR_LINES.get(wl, HGAR_LINES.get(round(wl, 2), '?'))
            print(f"    px={px:7.1f} -> {wl:.2f} nm ({sym})  résidu={r:+.3f} nm")

    return {'ok': True, 'wl_axis': wl_axis, 'coeffs': coeffs, 'pairs': pairs,
            'residuals': resid[order], 'rms_nm': rms, 'n_matched': n_matched,
            'peaks_px': peaks_px, 'reason': ''}


def plot_wl_calibration_diagnostic(mean_spectrum, auto_result=None,
                                   wl_axis=None, figsize=(14, 7)):
    """
    Diagnostic visuel de la calibration spectrale automatique.

    Trois panneaux :
      (haut)   spectre HgAr moyen en fonction de la longueur d'onde calibrée,
               avec les pics détectés (gris) et les raies appariées au catalogue
               (vert, annotées) → on VOIT si les raies tombent au bon endroit.
      (bas-g)  résidus d'appariement (nm) vs pixel → doit être plat, sub-0.1 nm.
      (bas-d)  courbe de dispersion pixel→nm.

    Paramètres
    ----------
    mean_spectrum : spectre HgAr moyen (1D)
    auto_result   : dict renvoyé par auto_wl_calibration (recalculé si None)
    wl_axis       : axe calibré (pris dans auto_result si None)
    """
    import matplotlib.pyplot as plt
    if auto_result is None:
        auto_result = auto_wl_calibration(mean_spectrum,
                                          n_cols=len(mean_spectrum),
                                          verbose=False)
    if not auto_result.get('ok'):
        print(f"[diagnostic] calibration auto non valide : "
              f"{auto_result.get('reason','?')}")
    if wl_axis is None:
        wl_axis = auto_result.get('wl_axis')
        if wl_axis is None:
            wl_axis = np.arange(len(mean_spectrum))

    peaks_px = auto_result.get('peaks_px', np.array([]))
    pairs    = auto_result.get('pairs', [])
    resid    = auto_result.get('residuals', np.array([]))
    rms      = auto_result.get('rms_nm', np.nan)

    px_axis = np.arange(len(mean_spectrum))

    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(2, 2, height_ratios=[2, 1])

    # Panneau haut : spectre calibré + pics + raies appariées
    ax0 = fig.add_subplot(gs[0, :])
    ax0.plot(wl_axis, mean_spectrum, lw=0.8, color='0.3', label='Spectre HgAr moyen')
    # pics détectés (position en nm via l'axe)
    if len(peaks_px):
        pk_wl = np.interp(peaks_px, px_axis, wl_axis)
        ax0.plot(pk_wl, np.interp(peaks_px, px_axis, mean_spectrum), 'v',
                 color='0.6', ms=5, label=f'{len(peaks_px)} pics détectés')
    # raies appariées
    for (px, wl) in pairs:
        wl_pos = np.interp(px, px_axis, wl_axis)
        ax0.axvline(wl_pos, color='tab:green', alpha=0.5, lw=0.8)
        ax0.text(wl_pos, ax0.get_ylim()[1]*0.92, f'{wl:.1f}',
                 rotation=90, va='top', ha='right', fontsize=7, color='tab:green')
    ax0.set_xlabel('Longueur d\'onde calibrée (nm)')
    ax0.set_ylabel('Intensité')
    ax0.set_title(f"Calibration WL auto — {len(pairs)} raies appariées, "
                  f"RMS = {rms:.3f} nm")
    ax0.legend(loc='upper right', fontsize=8)

    # Panneau bas-gauche : résidus
    ax1 = fig.add_subplot(gs[1, 0])
    if len(pairs) and len(resid):
        px_m = np.array([p[0] for p in pairs])
        ax1.axhline(0, color='0.7', lw=0.8)
        ax1.plot(px_m, resid, 'o', color='tab:red', ms=4)
        ax1.axhspan(-rms, rms, color='tab:red', alpha=0.12)
    ax1.set_xlabel('Pixel')
    ax1.set_ylabel('Résidu (nm)')
    ax1.set_title('Résidus d\'appariement')

    # Panneau bas-droite : dispersion
    ax2 = fig.add_subplot(gs[1, 1])
    ax2.plot(px_axis, wl_axis, color='tab:blue', lw=1.2)
    if len(pairs):
        ax2.plot([p[0] for p in pairs], [p[1] for p in pairs],
                 'o', color='tab:green', ms=4)
    ax2.set_xlabel('Pixel')
    ax2.set_ylabel('λ (nm)')
    ax2.set_title('Courbe de dispersion')

    fig.tight_layout()
    return fig


def build_wl_calibration(arr_calib, calib_pairs=None, poly_deg=WL_POLY_DEG,
                          mean_spectrum=None, method=None, verbose=True):
    """
    Construit la solution de dispersion pixels -> nm à partir des raies HgAr.

    method :
      • "auto"   -> comb-matching universel (auto_wl_calibration), aucune paire
                   codee en dur. Repli automatique sur les paires manuelles si
                   la detection auto echoue (garde-fous non satisfaits).
      • "manual" -> ancien comportement : polynome sur calib_pairs (ou
                   WL_CALIB_PAIRS par defaut).
      • None     -> utilise le defaut du module WL_METHOD.

    Renvoie (wl_axis, coeffs, residuals, used_pairs) ou used_pairs est la liste
    des paires (pixel, nm) REELLEMENT utilisees (auto ou manuelle) : ainsi
    residuals et used_pairs ont toujours la meme longueur.
    """
    if method is None:
        method = WL_METHOD

    n_cols = arr_calib.shape[1]

    # ── Voie AUTOMATIQUE (universelle) ──────────────────────────────────────
    if str(method).lower() == "auto":
        if mean_spectrum is None:
            mean_spectrum = arr_calib.mean(axis=0)
        if verbose:
            print("  Spectral calibration — AUTO comb-matching (no-guess)")
        auto = auto_wl_calibration(mean_spectrum, n_cols=n_cols,
                                   poly_deg=poly_deg, verbose=verbose)
        if auto['ok']:
            return (auto['wl_axis'], auto['coeffs'],
                    np.asarray(auto['residuals']), list(auto['pairs']))
        warnings.warn(
            f"Auto WL calibration a echoue ({auto.get('reason','?')}). "
            f"Repli sur les paires manuelles WL_CALIB_PAIRS.")
        if verbose:
            print(f"  -> Repli manuel (raison: {auto.get('reason','?')})")

    # ── Voie MANUELLE (paires codees) ───────────────────────────────────────
    if calib_pairs is None:
        calib_pairs = WL_CALIB_PAIRS

    pixels_k = np.array([p[0] for p in calib_pairs], dtype=float)
    wl_k     = np.array([p[1] for p in calib_pairs], dtype=float)

    coeffs   = np.polyfit(pixels_k, wl_k, poly_deg)
    wl_fit   = np.polyval(coeffs, pixels_k)
    residuals = wl_k - wl_fit
    rms       = np.sqrt(np.mean(residuals ** 2))

    wl_axis  = np.polyval(coeffs, np.arange(n_cols)) + WL_SHIFT_NM

    if verbose:
        print(f"  Spectral calibration — polynomial deg {poly_deg}")
        print(f"  Range: {wl_axis[0]:.1f} – {wl_axis[-1]:.1f} nm")
        print(f"  Dispersion: {(wl_axis[-1]-wl_axis[0])/n_cols:.4f} nm/px")
        print(f"  RMS residuals: {rms:.3f} nm")
        for (px, wl), res in zip(calib_pairs, residuals):
            sym = HGAR_LINES.get(wl, '?')
            print(f"    px={px:5d} -> {wl:.2f} nm ({sym})  résidu={res:+.3f} nm")

    return wl_axis, coeffs, residuals, list(calib_pairs)


def auto_detect_wl_lines(arr_calib, wl_axis, tol_nm=3.0, mean_spectrum=None):

    """

    Détection automatique des raies dans le spectre de calibration

    et association aux raies HgAr du catalogue.

    """

    if mean_spectrum is not None:

        xproj = mean_spectrum.copy()

    else:

        xproj = arr_calib.mean(axis=0)

 

    xproj = gaussian_filter1d(median_filter(xproj.astype(float), 3), 1.5)

    bg    = np.percentile(xproj, 10)

    sig   = np.clip(xproj - bg, 0, None)

 

    peaks, props = find_peaks(sig,

                              height=sig.max() * 0.03,

                              distance=15,

                              prominence=sig.max() * 0.02)

 

    catalog_wls = np.array(sorted(HGAR_LINES.keys()))

    results = []

    for pk in peaks:

        wl_meas = wl_axis[pk]

        diffs   = np.abs(catalog_wls - wl_meas)

        best    = np.argmin(diffs)

        if diffs[best] < tol_nm:

            results.append((pk, catalog_wls[best], wl_meas, float(diffs[best])))

    return results

 

 

# =============================================================================

# EXTRACTION DES SPECTRES

# =============================================================================

 

def build_calibration(calib_path, n_fibers=N_FIBERS, half_width=HALF_WIDTH,

                      calib_pairs=None, wl_method=None, verbose=True):

    """

    Calibration complète (spatiale + spectrale) à partir de l'image HgAr.

    Positions spatiales : FIBER_Y_MANUAL (identifiées manuellement).
    Tilt : mesuré automatiquement sur les raies du HgAr.
    Spectral : polynomial pixel → nm depuis les paires de calibration.
    """

    if verbose:

        print(f"\n{'='*60}")

        print(f"  CALIBRATION : {Path(calib_path).name}")

        print(f"{'='*60}")

 

    arr_raw = load_image(calib_path)
    if verbose:
        print(f"  Image: {arr_raw.shape[1]}×{arr_raw.shape[0]} px  "
              f"[{arr_raw.min():.0f} – {arr_raw.max():.0f}]")

    # Straighten the calibration image before fiber detection.
    if verbose: print("\n[0] Straightening calibration image")
    calib_angle_deg = _measure_image_rotation(arr_raw.astype(float))
    arr             = _rotate_image(arr_raw.astype(float), calib_angle_deg)
    if verbose: print(f"  Rotation angle: {calib_angle_deg:.4f} deg")

    # ── Positions spatiales : manuelles ─────────────────────────────────────
    if verbose: print("\n[1] Spatial calibration (manual fiber positions)")
    fiber_y = FIBER_Y_MANUAL.copy()
    assert len(fiber_y) == n_fibers, (
        f"FIBER_Y_MANUAL has {len(fiber_y)} entries, expected {n_fibers}")
    is_real = np.ones(n_fibers, dtype=bool)   # toutes manuelles = toutes réelles

    spacings = np.diff(fiber_y)
    period   = float(np.median(spacings))
    if verbose:
        print(f"  {n_fibers} fiber positions loaded (manual)")
        print(f"  Y range: {fiber_y[0]:.0f} – {fiber_y[-1]:.0f} px")
        print(f"  Period: median={period:.1f}  min={spacings.min():.0f}  "
              f"max={spacings.max():.0f} px")

    # ── Auto-select reference column ────────────────────────────────────────
    calib_cols = find_calib_columns(arr)
    col_ref    = _best_ref_column(arr, calib_cols)
    if verbose:
        print(f"  Reference column: {col_ref}")
        print(f"  Emission lines detected (X): {calib_cols}")

    # ── Tilt measurement ────────────────────────────────────────────────────
    if verbose: print("\n[1b] Tilt measurement")
    tilt_slopes = np.zeros(n_fibers)
    col_data = {}
    for col in calib_cols:
        col = int(col)
        peaks, _ = detect_fibers_in_profile(column_profile(arr, col))
        if len(peaks) < 3:
            continue
        matched = np.full(n_fibers, np.nan)
        used    = np.zeros(len(peaks), dtype=bool)
        for i in range(n_fibers):
            d = np.abs(peaks - fiber_y[i])
            b = np.argmin(d)
            if d[b] < TILT_ASSOC_TOL and not used[b]:
                matched[i], used[b] = peaks[b], True
        col_data[col] = matched

    if col_data:
        cols_arr = np.array(sorted(col_data.keys()), dtype=float)
        for i in range(n_fibers):
            ys = np.array([col_data[c][i] for c in cols_arr.astype(int)])
            v  = ~np.isnan(ys)
            if v.sum() >= 2:
                tilt_slopes[i], _ = np.polyfit(cols_arr[v] - col_ref, ys[v], 1)

    if verbose:
        print(f"  Tilt: mean={tilt_slopes.mean():.4f}  "
              f"min={tilt_slopes.min():.4f}  max={tilt_slopes.max():.4f} px/px")

    # ── Spectral calibration ───────────────────────────────────────────────
    if verbose: print("\n[2] Spectral calibration (wavelengths)")

    mean_sp = _mean_spectrum_along_fibers(
        arr, fiber_y, tilt_slopes, col_ref, half_width)

    wl_axis, wl_coeffs, wl_residuals, wl_used_pairs = build_wl_calibration(
        arr, calib_pairs=calib_pairs, mean_spectrum=mean_sp,
        method=wl_method, verbose=verbose)

    calib = {
        'n_fibers'        : n_fibers,
        'half_width'      : half_width,
        'col_ref'         : col_ref,
        'fiber_y_ref'     : fiber_y,
        'tilt_slopes'     : tilt_slopes,
        'fiber_is_real'   : is_real,
        'calib_cols'      : calib_cols,
        'image_shape'     : arr.shape,
        'calib_path'      : str(calib_path),
        'calib_angle_deg' : calib_angle_deg,
        'wl_axis'         : wl_axis,
        'wl_coeffs'       : wl_coeffs,
        'wl_residuals'    : wl_residuals,
        'wl_pairs'        : wl_used_pairs,
        'mean_calib_spectrum': mean_sp,
        'fiber_period'    : period,
    }

    if verbose:
        print(f"\n{'='*60}")
        print("  Calibration complete ✓")
        print(f"{'='*60}\n")

    return calib


def build_calibration_multi(calib_paths, n_fibers=N_FIBERS,
                            half_width=HALF_WIDTH, calib_pairs=None,
                            wl_method=None,
                            outlier_tol_px=2.0, hgar_path=None,
                            verbose=True):
    """
    Multi-image spatial calibration by consensus.

    Fiber positions are determined by running detection on every image in
    `calib_paths` and taking the median across images (consensus).

    Spectral calibration and intensity calibration reference are taken from
    `hgar_path` (HgAr lamp image).  If `hgar_path` is None, the first image
    in `calib_paths` is used for both (legacy behaviour).

    Parameters
    ----------
    calib_paths    : list of str/Path — science images for fiber detection
    n_fibers       : int   (default 80)
    half_width     : int   (default 6)
    calib_pairs    : list of (pixel, wl_nm) or None
    outlier_tol_px : float — fiber flagged as outlier if position std > this (px)
    hgar_path      : str/Path or None — HgAr lamp image used for:
                       • spectral calibration (pixel → nm)
                       • intensity calibration reference (fiber 0 spectrum)
                     If None, calib_paths[0] is used (legacy).
    verbose        : bool

    Returns
    -------
    calib : dict — same keys as build_calibration(), plus:
        'multi_image_data'   : list of per-image detection dicts
        'fiber_n_detections' : int array (n_fibers,)
        'fiber_y_std'        : float array (n_fibers,) — std of Y across images
        'fiber_outlier'      : bool array (n_fibers,)
    """
    calib_paths = [Path(p) for p in calib_paths]
    n_images    = len(calib_paths)

    if n_images == 0:
        raise ValueError("calib_paths is empty")
    if n_images == 1:
        if verbose:
            print("  [multi] Only 1 image provided — falling back to "
                  "single-image calibration.")
        calib = build_calibration(calib_paths[0], n_fibers=n_fibers,
                                 half_width=half_width, calib_pairs=calib_pairs,
                                 verbose=verbose)
        calib['multi_image_data']    = None
        calib['fiber_n_detections']  = calib['fiber_is_real'].astype(int)
        calib['fiber_y_std']         = np.full(n_fibers, np.nan)
        calib['fiber_outlier']       = np.zeros(n_fibers, dtype=bool)
        return calib

    if verbose:
        print(f"\n{'='*60}")
        print(f"  MULTI-IMAGE CALIBRATION — {n_images} images")
        print(f"{'='*60}")
        for k, p in enumerate(calib_paths):
            print(f"    [{k}] {p.name}")

    # ── Per-image detection ──────────────────────────────────────────────────
    per_image = []

    for k, cp in enumerate(calib_paths):
        if verbose:
            print(f"\n── Image {k}: {cp.name} ──")

        arr_raw   = load_image(cp)
        angle_deg = _measure_image_rotation(arr_raw.astype(float))
        arr       = _rotate_image(arr_raw.astype(float), angle_deg)

        if verbose:
            print(f"  Shape: {arr.shape[1]}x{arr.shape[0]} px  "
                  f"rotation: {angle_deg:.4f} deg")

        calib_cols = find_calib_columns(arr)
        info       = measure_tilt(arr, calib_cols, verbose=verbose)
        info       = assign_to_grid(info, n_fibers=n_fibers, verbose=verbose)

        per_image.append({
            'path'          : str(cp),
            'angle_deg'     : angle_deg,
            'arr_shape'     : arr.shape,
            'calib_cols'    : calib_cols,
            'col_ref'       : info['col_ref'],
            'fiber_y_ref'   : info['fiber_y_ref'].copy(),
            'tilt_slopes'   : info['tilt_slopes'].copy(),
            'fiber_is_real' : info['fiber_is_real'].copy(),
        })

    # ── Merge by consensus ───────────────────────────────────────────────────
    if verbose:
        print(f"\n{'='*60}")
        print(f"  MERGING {n_images} images by consensus")
        print(f"{'='*60}")

    all_y    = np.full((n_images, n_fibers), np.nan)
    all_sl   = np.full((n_images, n_fibers), np.nan)
    all_real = np.zeros((n_images, n_fibers), dtype=bool)

    for k, img in enumerate(per_image):
        for i in range(n_fibers):
            if not np.isnan(img['fiber_y_ref'][i]):
                all_y[k, i]  = img['fiber_y_ref'][i]
                all_sl[k, i] = img['tilt_slopes'][i]
            all_real[k, i] = img['fiber_is_real'][i]

    consensus_y    = np.full(n_fibers, np.nan)
    consensus_sl   = np.zeros(n_fibers)
    consensus_real = np.zeros(n_fibers, dtype=bool)
    n_detections   = np.zeros(n_fibers, dtype=int)
    fiber_y_std    = np.full(n_fibers, np.nan)

    for i in range(n_fibers):
        real_mask = all_real[:, i]
        n_det     = int(real_mask.sum())
        n_detections[i] = n_det

        valid_y  = all_y[:, i][~np.isnan(all_y[:, i])]
        valid_sl = all_sl[:, i][~np.isnan(all_sl[:, i])]

        if len(valid_y) > 0:
            consensus_y[i]  = float(np.median(valid_y))
            consensus_sl[i] = float(np.median(valid_sl))
        if len(valid_y) >= 2:
            fiber_y_std[i] = float(np.std(valid_y))

        consensus_real[i] = n_det > 0

    fiber_outlier = np.zeros(n_fibers, dtype=bool)
    for i in range(n_fibers):
        if np.isfinite(fiber_y_std[i]) and fiber_y_std[i] > outlier_tol_px:
            fiber_outlier[i] = True

    # Use col_ref from image with most detections
    n_real_per_img = [int(img['fiber_is_real'].sum()) for img in per_image]
    best_img_idx   = int(np.argmax(n_real_per_img))
    col_ref        = per_image[best_img_idx]['col_ref']

    if verbose:
        n_any = int((n_detections > 0).sum())
        n_all = int((n_detections == n_images).sum())
        n_out = int(fiber_outlier.sum())
        n_tot = int(np.isfinite(consensus_y).sum())
        print(f"\n  Consensus results:")
        print(f"    Fibers detected on >= 1 image   : {n_any}/{n_fibers}")
        print(f"    Fibers detected on ALL images    : {n_all}/{n_fibers}")
        print(f"    Total valid fibers (incl. interp.) : {n_tot}/{n_fibers}")
        print(f"    Outliers (std > {outlier_tol_px} px) : {n_out}")
        if n_out > 0:
            out_idx = np.where(fiber_outlier)[0]
            print(f"    Outlier fibers: {list(out_idx)}")
            for idx in out_idx:
                vals = all_y[:, idx][~np.isnan(all_y[:, idx])]
                print(f"      Fiber {idx}: positions = "
                      f"{', '.join(f'{v:.1f}' for v in vals)}  "
                      f"std = {fiber_y_std[idx]:.2f} px")

        print(f"\n  Per-image detection:")
        for k, img in enumerate(per_image):
            n_r = int(img['fiber_is_real'].sum())
            n_v = int(np.isfinite(img['fiber_y_ref']).sum())
            print(f"    [{k}] {Path(img['path']).name}: "
                  f"{n_r} detected, {n_v - n_r} interpolated, "
                  f"{n_fibers - n_v} empty")

    # ── Spectral calibration + intensity reference ───────────────────────────
    # Use HgAr image if provided, otherwise fall back to first science image.
    if hgar_path is not None:
        hgar_path_obj   = Path(hgar_path)
        arr_raw_wl      = load_image(hgar_path_obj)
        # Straighten using the rotation angle from the best science image
        angle_wl        = per_image[best_img_idx]['angle_deg']
        arr_wl          = _rotate_image(arr_raw_wl.astype(float), angle_wl)
        wl_source_name  = hgar_path_obj.name
        calib_path_out  = str(hgar_path_obj)
        calib_angle_out = angle_wl
        if verbose:
            print(f"\n[2] Spectral calibration (from HgAr: {wl_source_name})")
    else:
        arr_wl          = _rotate_image(load_image(calib_paths[0]).astype(float),
                                        per_image[0]['angle_deg'])
        wl_source_name  = calib_paths[0].name
        calib_path_out  = str(calib_paths[0])
        calib_angle_out = per_image[0]['angle_deg']
        if verbose:
            print(f"\n[2] Spectral calibration (from {wl_source_name})")

    mean_sp = _mean_spectrum_along_fibers(
        arr_wl, consensus_y, consensus_sl, col_ref, half_width)

    wl_axis, wl_coeffs, wl_residuals, wl_used_pairs = build_wl_calibration(
        arr_wl, calib_pairs=calib_pairs, mean_spectrum=mean_sp,
        method=wl_method, verbose=verbose)

    # ── Build output dict ────────────────────────────────────────────────────
    calib = {
        'n_fibers'          : n_fibers,
        'half_width'        : half_width,
        'col_ref'           : col_ref,
        'fiber_y_ref'       : consensus_y,
        'tilt_slopes'       : consensus_sl,
        'fiber_is_real'     : consensus_real,
        'calib_cols'        : per_image[best_img_idx]['calib_cols'],
        'image_shape'       : per_image[0]['arr_shape'],
        # calib_path points to HgAr if provided: used by _extract_calib_spectra
        # for intensity calibration (fiber 0 of HgAr as reference)
        'calib_path'        : calib_path_out,
        'calib_angle_deg'   : calib_angle_out,
        'wl_axis'           : wl_axis,
        'wl_coeffs'         : wl_coeffs,
        'wl_residuals'      : wl_residuals,
        'wl_pairs'          : wl_used_pairs,
        'mean_calib_spectrum': mean_sp,
        # Multi-image specific
        'multi_image_data'   : per_image,
        'fiber_n_detections' : n_detections,
        'fiber_y_std'        : fiber_y_std,
        'fiber_outlier'      : fiber_outlier,
    }

    if verbose:
        print(f"\n{'='*60}")
        print(f"  Multi-image calibration complete  "
              f"({n_images} images, {int(np.isfinite(consensus_y).sum())} fibers)")
        print(f"{'='*60}\n")

    return calib


def plot_multi_calib_diagnostic(calib):
    """
    Diagnostic plots for multi-image calibration.

    Requires a calib dict produced by build_calibration_multi().

    Panel 1 : Detection matrix (fiber x image)
              green = detected | yellow = interpolated | dark = missing
              red X = outlier fiber
    Panel 2 : Y position per fiber (one curve per image + consensus)
    Panel 3 : Inter-image scatter (std) per fiber with outlier threshold
    Panel 4 : Consensus inter-fiber spacing
    Panel 5 : Tilt slopes per image + consensus
    Panel 6 : Detection count histogram

    Usage
    -----
    >>> calib = sf.build_calibration_multi(CALIB_PATHS)
    >>> sf.plot_multi_calib_diagnostic(calib)
    """
    mid = calib.get('multi_image_data')
    if mid is None:
        print("  No multi-image data available (single-image calibration).")
        print("  Use build_calibration_multi() with >= 2 images.")
        return

    n_fibers  = calib['n_fibers']
    n_images  = len(mid)
    fib_idx   = np.arange(n_fibers)
    n_det     = calib['fiber_n_detections']
    y_std     = calib['fiber_y_std']
    outlier   = calib['fiber_outlier']
    consensus = calib['fiber_y_ref']
    outlier_tol = 2.0  # default, matches build_calibration_multi

    fig = plt.figure(figsize=(22, 14))
    gs_ = gridspec.GridSpec(3, 2, figure=fig, hspace=0.45, wspace=0.30)

    # ── Panel 1: Detection matrix ────────────────────────────────────────────
    ax1 = fig.add_subplot(gs_[0, 0])
    det_matrix = np.zeros((n_images, n_fibers))
    for k, img in enumerate(mid):
        for i in range(n_fibers):
            if img['fiber_is_real'][i]:
                det_matrix[k, i] = 2
            elif np.isfinite(img['fiber_y_ref'][i]):
                det_matrix[k, i] = 1
            else:
                det_matrix[k, i] = 0

    from matplotlib.colors import ListedColormap, BoundaryNorm
    cmap_det = ListedColormap(['#2d2d2d', '#ffc107', '#4caf50'])
    norm_det = BoundaryNorm([-0.5, 0.5, 1.5, 2.5], cmap_det.N)
    ax1.imshow(det_matrix, aspect='auto', cmap=cmap_det, norm=norm_det,
               extent=[-0.5, n_fibers - 0.5, n_images - 0.5, -0.5])
    ax1.set_xlabel("Fiber index")
    ax1.set_ylabel("Image")
    ax1.set_yticks(range(n_images))
    ax1.set_yticklabels([Path(img['path']).name for img in mid], fontsize=7)
    ax1.set_title("Detection matrix\n"
                  "green = detected | yellow = interpolated | dark = missing")
    for i in np.where(outlier)[0]:
        for k in range(n_images):
            ax1.plot(i, k, 'rx', ms=8, mew=2)

    # ── Panel 2: Y position per fiber ────────────────────────────────────────
    ax2 = fig.add_subplot(gs_[0, 1])
    colors_img = plt.cm.tab10(np.linspace(0, 1, max(n_images, 1)))
    for k, img in enumerate(mid):
        y_k = img['fiber_y_ref']
        mask = np.isfinite(y_k)
        ax2.plot(fib_idx[mask], y_k[mask], 'o-', ms=2, lw=0.6,
                 color=colors_img[k], alpha=0.6,
                 label=Path(img['path']).name)
    mask_c = np.isfinite(consensus)
    ax2.plot(fib_idx[mask_c], consensus[mask_c], 'k-', lw=2.5,
             alpha=0.8, label='Consensus (median)', zorder=10)
    for i in np.where(outlier)[0]:
        if np.isfinite(consensus[i]):
            ax2.plot(i, consensus[i], 'r*', ms=14, zorder=11)
    ax2.set_xlabel("Fiber index")
    ax2.set_ylabel("Y position (px)")
    ax2.set_title("Fiber Y positions — per image + consensus")
    ax2.legend(fontsize=6, ncol=2, loc='upper left')
    ax2.grid(True, alpha=0.25)

    # ── Panel 3: Inter-image scatter ─────────────────────────────────────────
    ax3 = fig.add_subplot(gs_[1, 0])
    colors_bar = np.where(outlier, 'tomato', 'steelblue')
    valid_std = np.where(np.isfinite(y_std), y_std, 0)
    ax3.bar(fib_idx, valid_std, color=colors_bar, width=1.0, edgecolor='none')
    ax3.axhline(outlier_tol, color='red', ls='--', lw=1.2,
                label=f"Outlier threshold ({outlier_tol} px)")
    ax3.set_xlabel("Fiber index")
    ax3.set_ylabel("Std of Y position across images (px)")
    ax3.set_title(f"Inter-image position scatter — "
                  f"{int(outlier.sum())} outliers (red)")
    ax3.legend(fontsize=8)
    ax3.grid(True, alpha=0.25)

    # ── Panel 4: Consensus inter-fiber spacing ───────────────────────────────
    ax4 = fig.add_subplot(gs_[1, 1])
    sp = np.diff(consensus)
    valid_sp = np.isfinite(sp)
    ax4.plot(fib_idx[:-1][valid_sp], sp[valid_sp], 'o-', ms=3, color='tomato')
    if valid_sp.any():
        med_sp = float(np.nanmedian(sp))
        ax4.axhline(med_sp, ls='--', color='k',
                    label=f"Median = {med_sp:.2f} px")
    ax4.set_xlabel("Fiber index")
    ax4.set_ylabel("Spacing (px)")
    ax4.set_title("Consensus inter-fiber spacing")
    ax4.legend(fontsize=9)
    ax4.grid(True, alpha=0.3)

    # ── Panel 5: Tilt slopes per image ───────────────────────────────────────
    ax5 = fig.add_subplot(gs_[2, 0])
    for k, img in enumerate(mid):
        sl_k = img['tilt_slopes']
        mask = img['fiber_is_real']
        ax5.plot(fib_idx[mask], sl_k[mask], 'o-', ms=2, lw=0.6,
                 color=colors_img[k], alpha=0.6,
                 label=Path(img['path']).name)
    ax5.plot(fib_idx, calib['tilt_slopes'], 'k-', lw=2, alpha=0.8,
             label='Consensus', zorder=10)
    ax5.axhline(0, ls='--', color='gray', lw=0.8)
    ax5.set_xlabel("Fiber index")
    ax5.set_ylabel("Slope dY/dX (px/px)")
    ax5.set_title("Tilt slopes — per image + consensus")
    ax5.legend(fontsize=6, ncol=2)
    ax5.grid(True, alpha=0.25)

    # ── Panel 6: Detection count histogram ───────────────────────────────────
    ax6 = fig.add_subplot(gs_[2, 1])
    counts = np.bincount(n_det, minlength=n_images + 1)
    bars   = np.arange(len(counts))
    bar_colors = ['#2d2d2d' if b == 0 else '#ffc107' if b < n_images
                  else '#4caf50' for b in bars]
    ax6.bar(bars, counts, color=bar_colors, edgecolor='white', linewidth=0.5)
    ax6.set_xlabel("Number of images where fiber was detected")
    ax6.set_ylabel("Number of fibers")
    ax6.set_title(f"Detection count distribution — {n_images} images")
    ax6.set_xticks(bars)
    for b, c in zip(bars, counts):
        if c > 0:
            ax6.text(b, c + 0.3, str(c), ha='center', fontsize=10,
                     fontweight='bold')
    ax6.grid(True, alpha=0.25, axis='y')

    plt.suptitle(f"MULTI-IMAGE CALIBRATION DIAGNOSTIC — {n_images} images",
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()


def extract_spectrum_optimal(arr, y_ref, slope, col_ref, half_width=HALF_WIDTH):

    """

    Extraction optimale (pondération gaussienne, Horne 1986 simplifié)

    d'un spectre le long de sa trace inclinée.

    """

    n_rows, n_cols = arr.shape

    spectrum, sigma = np.zeros(n_cols), half_width / 2.5

    for x in range(n_cols):

        yc = y_ref + slope * (x - col_ref)

        y0 = max(0, int(yc) - half_width)

        y1 = min(n_rows, int(yc) + half_width + 1)

        if y1 <= y0:

            continue

        ys = np.arange(y0, y1, dtype=float)

        w  = np.exp(-0.5 * ((ys - yc) / sigma) ** 2)

        ws = w.sum()

        spectrum[x] = (arr[y0:y1, x] * w).sum() / ws if ws > 0 else 0.

    return spectrum

 

 

# =============================================================================

# SPECTRE MOYEN DE CALIBRATION (sans biais de tilt)

# =============================================================================

 

def _mean_spectrum_along_fibers(arr, fiber_y_ref, tilt_slopes, col_ref,

                                 half_width=HALF_WIDTH):

    """

    Calcule le spectre moyen de l'image de calibration en extrayant chaque

    fibre le long de sa TRACE RÉELLE (avec son tilt propre), puis en

    moyennant tous les spectres.

    """

    n_fibers = len(fiber_y_ref)

    n_cols   = arr.shape[1]

    spectra  = np.zeros((n_fibers, n_cols))

 

    for i in range(n_fibers):

        if np.isnan(fiber_y_ref[i]):

            continue

        spectra[i] = extract_spectrum_optimal(

            arr, fiber_y_ref[i], tilt_slopes[i], col_ref, half_width)

 

    normed = np.zeros_like(spectra)

    for i in range(n_fibers):

        m = spectra[i].max()

        if m > 0:

            normed[i] = spectra[i] / m

 

    mean_sp = normed.mean(axis=0)

    if mean_sp.max() > 0:

        mean_sp = mean_sp / mean_sp.max()

    return mean_sp

 

 

# =============================================================================

# CORRECTION DE TILT PAR IMAGE

# =============================================================================

 

def _rotate_image(arr, angle_deg):

    """Applique une rotation à l'image (anti-horaire si positif)."""

    if abs(angle_deg) < 1e-4:

        return arr.copy()

    return ndimage_rotate(arr, angle_deg, reshape=False, order=3, cval=0.0)

 

 

def _measure_image_rotation(arr):

    """

    Mesure l'angle de rotation des fibres dans une image par

    cross-corrélation de profils Y entre colonnes distantes.

    """

    H, W = arr.shape

    y0_active = int(H * 0.12)

    y1_active = int(H * 0.88)

    x0 = int(W * 0.15)

    x1 = int(W * 0.85)

    all_cols = np.linspace(x0, x1, 16, dtype=int)

 

    def clean_profile(col):

        p  = arr[y0_active:y1_active,

                  max(0, col-10):min(W, col+11)].mean(axis=1)

        p  = gaussian_filter1d(median_filter(p.astype(float), 3), 1.5)

        bg = uniform_filter1d(p, size=60)

        return p - bg

 

    separation = 500

    slopes = []

 

    for c1 in all_cols:

        c2 = int(c1 + separation)

        if c2 >= W:

            continue

        p1 = clean_profile(int(c1))

        p2 = clean_profile(c2)

        corr = np.correlate(p2, p1, mode='full')

        lags = np.arange(-(len(p1) - 1), len(p1))

        pk   = int(np.argmax(corr))

        if 1 <= pk <= len(corr) - 2:

            a, b, c = corr[pk-1], corr[pk], corr[pk+1]

            denom = a - 2*b + c

            sub   = 0.5*(a - c) / denom if denom != 0 else 0.0

            shift = lags[pk] + sub

        else:

            shift = float(lags[pk])

        slopes.append(shift / (c2 - c1))

 

    if not slopes:

        return 0.0

 

    median_slope = float(np.median(slopes))

    angle_deg    = +np.degrees(np.arctan(median_slope))

    return angle_deg

 

 

def _refine_fibers_local(arr_rot, master_y, col_ref, search_hw=3):
    """
    Micro-refinement des positions master par centroid local.

    Pour chaque fibre, regarde le signal dans une fenêtre de ±search_hw pixels
    autour de la position master et recalcule le centroid pondéré.

    Propriétés clés :
    - Ne peut JAMAIS rater une fibre (pas de re-détection)
    - Ne peut JAMAIS décaler l'indexation
    - Si pas de signal local, garde la position master inchangée
    - Correction typique : 0–2 px (sub-pixel, lié à la rotation)

    Parameters
    ----------
    arr_rot    : ndarray (H, W) — image rotée
    master_y   : ndarray (n_fibers,) — positions Y master (de la calibration HgAr)
    col_ref    : int — colonne de référence
    search_hw  : int — demi-largeur de recherche (px)

    Returns
    -------
    refined_y : ndarray (n_fibers,) — positions Y raffinées
    """
    # Profil moyen sur plusieurs colonnes pour plus de robustesse
    H, W = arr_rot.shape
    n_avg = 5
    cols = np.clip(
        np.arange(col_ref - n_avg * 15, col_ref + n_avg * 15 + 1, 15),
        0, W - 1
    ).astype(int)
    cols = np.unique(cols)

    profiles = [column_profile(arr_rot, int(c), half=10) for c in cols]
    prof = np.mean(profiles, axis=0)

    # Seuil : une fibre a du signal si le profil local dépasse significativement
    # le fond. On prend le 20e percentile du profil comme estimation du fond.
    bg_level = np.percentile(prof, 20)

    refined = master_y.copy()

    for i in range(len(master_y)):
        if np.isnan(master_y[i]):
            continue

        y_expect = master_y[i]
        y0 = max(0, int(round(y_expect)) - search_hw)
        y1 = min(len(prof), int(round(y_expect)) + search_hw + 1)

        if y1 <= y0:
            continue

        segment = prof[y0:y1].astype(float)

        # Signal significatif = max local > fond + 50% de la dynamique locale
        if segment.max() > bg_level * 1.3:
            ys = np.arange(y0, y1, dtype=float)
            w  = np.clip(segment - np.min(segment), 0, None)
            if w.sum() > 0:
                new_y = float(np.sum(ys * w) / w.sum())
                # Sanity check : ne pas bouger de plus de search_hw
                if abs(new_y - y_expect) <= search_hw:
                    refined[i] = new_y

    return refined


def _detect_fibers_after_rotation(arr_rot, calib):

    """

    Raffine les positions master des fibres sur une image science rotée.

    Utilise le micro-refinement local (centroid ±3 px) autour des
    positions de la calibration HgAr. Ne re-détecte PAS les fibres.

    Returns
    -------
    fiber_y : ndarray (n_fibers,) — positions Y raffinées
    is_real : ndarray (n_fibers,) bool — copie de calib['fiber_is_real']
    """

    master_y = calib['fiber_y_ref'].copy()
    col_ref  = calib['col_ref']

    refined_y = _refine_fibers_local(arr_rot, master_y, col_ref, search_hw=3)

    return refined_y, calib['fiber_is_real'].copy()

 

 

def _assign_to_n(positions, is_real, n_fibers, max_extend=MAX_EXTEND):
    """
    Assign detected peaks to a grid of n_fibers slots.
    Grid anchored to first detected peak; extends up to max_extend slots
    beyond the last detected peak to catch undetected edge fibers.
    """
    if len(positions) == 0:
        return np.full(n_fibers, np.nan), np.zeros(n_fibers, dtype=bool)

    spacings = np.diff(positions)
    med_sp   = np.median(spacings) if len(spacings) > 0 else 18.0
    single   = spacings[spacings < 1.5 * med_sp]
    period   = float(np.median(single)) if len(single) >= 1 else float(med_sp)

    y_first = float(positions[0])
    y_last  = float(positions[-1])
    n_core  = int(round((y_last - y_first) / period)) + 1
    n_use   = min(n_core + max_extend, n_fibers)
    grid    = y_first + np.arange(n_use) * period

    ay   = np.full(n_fibers, np.nan)
    ar   = np.zeros(n_fibers, dtype=bool)
    used = np.zeros(len(positions), dtype=bool)

    tol = 0.55 * period
    for gi in range(n_use):
        d = np.abs(positions - grid[gi])
        b = np.argmin(d)
        if d[b] < tol and not used[b]:
            ay[gi], ar[gi], used[b] = positions[b], is_real[b], True

    # Interpolate internal gaps + extrapolate extended slots
    valid = ~np.isnan(ay[:n_use])
    if valid.sum() >= 2:
        idx_v = np.where(valid)[0]
        fy    = interp1d(idx_v, ay[:n_use][valid], kind='linear',
                         bounds_error=False, fill_value='extrapolate')
        for gi in range(n_use):
            if np.isnan(ay[gi]):
                ay[gi] = float(fy(gi))

    return ay, ar



# =============================================================================
# TRAITEMENT DU BRUIT — pipeline en 3 niveaux
# =============================================================================
#
# Niveau 1 : subtract_background_interfiber()  [image-level, avant extraction]
#   Estime le fond à partir des GAPS entre les fibres pour chaque colonne.
#   Beaucoup plus fidèle que les top-N rows car il capture la lumière
#   diffusée (scattered light) qui varie en Y et en λ.
#
# Niveau 2 : remove_spectral_baseline()  [spectrum-level, après extraction]
#   Soustrait le piédestal résiduel par morphological opening (rolling
#   minimum + lissage). Standard en spectroscopie (SNIP-like).
#
# Niveau 3 : apply_edge_taper()  [spectrum-level, après extraction]
#   Détecte les bords du signal utile et applique une transition
#   sigmoïde vers zéro (pas de mur vertical).
#
# Le tout est orchestré par clean_spectra() qui enchaîne les 3 niveaux.
# extract_all_spectra() appelle ce pipeline automatiquement.


def subtract_background_columns(arr, calib=None, hw=None):
    """
    Legacy wrapper — redirige vers subtract_background_interfiber si les
    positions de fibres sont disponibles dans calib, sinon fallback sur
    l'ancienne méthode (top BG_N_ROWS rows).
    """
    if calib is not None and 'fiber_y_ref' in calib:
        return subtract_background_interfiber(arr, calib)
    # Fallback legacy
    H, W = arr.shape
    n    = min(BG_N_ROWS, H // 10)
    bg   = np.median(arr[:n, :], axis=0)
    return np.clip(arr - bg[np.newaxis, :], 0.0, None)


def subtract_background_interfiber(arr, calib, bg_smooth_y=15, bg_smooth_x=51):
    """
    Soustraction de fond basée sur les GAPS INTER-FIBRES.

    Principe
    --------
    Pour chaque colonne x (= chaque longueur d'onde), on échantillonne
    l'image aux positions Y situées À MI-CHEMIN entre deux fibres
    adjacentes. Ces points sont hors de l'ouverture d'extraction et
    mesurent donc uniquement le fond (scattered light + bias + dark).

    On interpole ensuite un fond lisse 2D (Y, X) par interpolation
    linéaire en Y + lissage Savitzky-Golay en X.

    Avantage par rapport à l'ancienne méthode
    ------------------------------------------
    • Capture la variation spatiale du fond DANS le champ des fibres
      (la méthode top-N-rows ne voyait que le fond au-dessus du champ)
    • Élimine le piédestal résiduel dû à la lumière diffusée qui
      augmente entre les fibres brillantes

    Paramètres
    ----------
    arr         : ndarray (H, W)  — image rotée (fibres horizontales)
    calib       : dict — doit contenir 'fiber_y_ref'
    bg_smooth_y : int — lissage vertical du fond interpolé (pixels)
    bg_smooth_x : int — lissage spectral du fond (pixels, doit être impair)

    Retourne
    --------
    arr_sub : ndarray (H, W) — image avec fond soustrait, clippée à 0
    """
    H, W = arr.shape
    fiber_y = calib['fiber_y_ref']
    valid_y = fiber_y[np.isfinite(fiber_y)]

    if len(valid_y) < 2:
        # Pas assez de fibres → fallback
        n = min(BG_N_ROWS, H // 10)
        bg = np.median(arr[:n, :], axis=0)
        return np.clip(arr - bg[np.newaxis, :], 0.0, None)

    # ── Positions Y inter-fibres ────────────────────────────────────────────
    # Points de mesure du fond : milieu de chaque gap entre fibres adjacentes
    mid_y = 0.5 * (valid_y[:-1] + valid_y[1:])

    # Ajouter des points de mesure AVANT la première fibre et APRÈS la dernière
    spacing = float(np.median(np.diff(valid_y)))
    extra_before = [valid_y[0] - spacing * k for k in range(1, 3)
                    if valid_y[0] - spacing * k > 2]
    extra_after  = [valid_y[-1] + spacing * k for k in range(1, 3)
                    if valid_y[-1] + spacing * k < H - 2]

    sample_y = np.sort(np.concatenate([extra_before, mid_y, extra_after]))
    sample_y = np.clip(sample_y, 1, H - 2).astype(int)

    # ── Échantillonnage du fond (médiane sur ±2 pixels en Y à chaque point) ─
    n_samples = len(sample_y)
    bg_samples = np.zeros((n_samples, W))

    for k, y_pos in enumerate(sample_y):
        y0 = max(0, y_pos - 2)
        y1 = min(H, y_pos + 3)
        bg_samples[k] = np.median(arr[y0:y1, :], axis=0)

    # ── Lissage spectral des échantillons (élimine le bruit pixel-à-pixel) ──
    win = bg_smooth_x if bg_smooth_x % 2 == 1 else bg_smooth_x + 1
    win = min(win, W - 2 if W > 3 else 1)
    if win >= 5:
        for k in range(n_samples):
            try:
                bg_samples[k] = savgol_filter(bg_samples[k], win, 3)
            except ValueError:
                bg_samples[k] = gaussian_filter1d(bg_samples[k], win / 4)

    # ── Interpolation 2D du fond (linéaire en Y) ────────────────────────────
    # Pour chaque colonne x, interpoler les n_samples points en Y pour
    # obtenir le fond à chaque ligne de l'image
    from scipy.interpolate import interp1d as _interp1d

    bg_2d = np.zeros_like(arr)
    all_rows = np.arange(H, dtype=float)

    for x in range(W):
        f = _interp1d(sample_y.astype(float), bg_samples[:, x],
                      kind='linear', bounds_error=False,
                      fill_value=(bg_samples[0, x], bg_samples[-1, x]))
        bg_2d[:, x] = f(all_rows)

    # ── Lissage vertical léger du fond 2D ───────────────────────────────────
    if bg_smooth_y > 1:
        bg_2d = gaussian_filter1d(bg_2d, sigma=bg_smooth_y / 3, axis=0)

    return np.clip(arr - bg_2d, 0.0, None)


# =============================================================================
# FOND "AVANT/APRÈS" — soustraction douce basée UNIQUEMENT sur les zones
# sans fibre (au-dessus de la 1re fibre et en-dessous de la dernière).
# =============================================================================
#
# Motivation
# ----------
# subtract_background_interfiber() échantillonne le fond ENTRE les fibres.
# Aux longueurs d'onde des raies, toutes les fibres sont brillantes et
# débordent (aile de PSF) dans les gaps → le fond y est surestimé, on
# sur-soustrait. Ici on ne mesure le fond QUE là où aucune fibre ne peut
# déborder : la marge du haut et celle du bas du détecteur.
#
# Compromis assumé (mesuré sur les données)
# -----------------------------------------
# Le fond diffus est LÉGÈREMENT plus élevé dans la zone dense des fibres
# qu'en marge (≈ quelques dizaines d'ADU sur un fond de ~180). Cette méthode
# sous-soustrait donc d'autant. C'est le prix à payer pour éliminer TOUT
# biais de débordement. Pour pouvoir vérifier ce résidu, la soustraction
# ne clippe PAS à zéro par défaut (clip=False) : le fond corrigé doit
# osciller autour de 0 hors des raies ; s'il se pose à +30, tu vois la
# sous-soustraction et tu peux décider de la corriger.


def estimate_background_edges(arr, calib=None, fiber_y=None,
                              sensor_margin=50, top_margin=20,
                              bottom_margin=15, min_band_px=40,
                              min_valid_frac=0.5, smooth_x=51,
                              return_components=False):
    """
    Estime un fond 2D (H, W) à partir des SEULES zones sans fibre.

    Principe
    --------
    1. Détermine l'étendue verticale des fibres [y_first, y_last]
       (depuis calib['fiber_y_ref'], sinon `fiber_y`, sinon FIBER_Y_MANUAL).
    2. Définit deux bandes de fond PROPRES :
         • bande HAUT : Y ∈ [sensor_margin, y_first - top_margin]
         • bande BAS  : Y ∈ [y_last + bottom_margin, H - sensor_margin]
    3. Pour chaque colonne x, fond de bande = MÉDIANE des pixels valides
       de la bande (les pixels à ~0 issus des coins de rotation sont
       ignorés). La médiane est non biaisée sur du fond pur et robuste
       aux pixels chauds / rayons cosmiques.
    4. Colonnes sans assez de pixels valides (coins de rotation aux
       bords X) → remplies par la colonne valide la plus proche.
    5. Lissage spectral (Savitzky-Golay, fenêtre smooth_x) : le fond est
       large bande, le lisser évite d'INJECTER son bruit pixel-à-pixel
       dans chaque spectre lors de la soustraction.
    6. Fond 2D = interpolation LINÉAIRE en Y entre la bande haut et la
       bande bas (extrapolation plate au-delà des centres de bande).

    Pourquoi linéaire en Y et pas constant
    ---------------------------------------
    Sur ces données le gradient vertical du fond est faible (~-4 ADU
    haut→bas), mais l'interpolation linéaire le capture gratuitement et
    reste correcte si un gradient plus marqué apparaît sur d'autres tirs.

    Paramètres
    ----------
    arr             : ndarray (H, W) — image rotée (fibres horizontales)
    calib           : dict ou None — si présent, utilise calib['fiber_y_ref']
    fiber_y         : ndarray ou None — positions Y des fibres (override)
    sensor_margin   : int — pixels ignorés en tout haut/bas du capteur
                      (bords + coins de rotation)
    top_margin      : int — garde entre la bande haut et la 1re fibre
    bottom_margin   : int — garde entre la dernière fibre et la bande bas
    min_band_px     : int — hauteur minimale exigée pour une bande ;
                      si non atteinte → fallback sur l'autre bande seule
    min_valid_frac  : float — fraction de pixels non nuls exigée dans une
                      colonne de bande pour la juger valide
    smooth_x        : int — fenêtre de lissage spectral du fond (impair)
    return_components : bool — si True, renvoie aussi les diagnostics

    Retourne
    --------
    bg_2d : ndarray (H, W) — fond estimé (non soustrait)
    (si return_components) dict avec bg_top, bg_bot, bandes, centres Y
    """
    H, W = arr.shape

    # ── 1. Étendue verticale des fibres ─────────────────────────────────────
    if fiber_y is None:
        if calib is not None and 'fiber_y_ref' in calib:
            fiber_y = np.asarray(calib['fiber_y_ref'], dtype=float)
        else:
            fiber_y = FIBER_Y_MANUAL
    fy = fiber_y[np.isfinite(fiber_y)]
    if len(fy) < 2:
        # Aucune info de fibre → fond global par colonne (médiane bandes bord)
        y_first, y_last = int(0.15 * H), int(0.85 * H)
    else:
        y_first, y_last = int(np.floor(fy.min())), int(np.ceil(fy.max()))

    # ── 2. Bandes de fond ───────────────────────────────────────────────────
    top_y0 = max(0, sensor_margin)
    top_y1 = max(top_y0 + 1, y_first - top_margin)
    bot_y0 = min(H - 1, y_last + bottom_margin)
    bot_y1 = min(H, H - sensor_margin)

    have_top = (top_y1 - top_y0) >= min_band_px
    have_bot = (bot_y1 - bot_y0) >= min_band_px

    # ── 3. Médiane par colonne (ignorant les zéros de rotation) ─────────────
    def _band_median(y0, y1):
        band = arr[y0:y1, :].astype(float).copy()
        band[band < 1.0] = np.nan          # masque les coins zéro-remplis
        valid = np.mean(np.isfinite(band), axis=0)
        bg = np.full(band.shape[1], np.nan)
        ok = valid >= min_valid_frac       # colonnes exploitables uniquement
        if ok.any():
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', category=RuntimeWarning)
                bg[ok] = np.nanmedian(band[:, ok], axis=0)
        return bg

    bg_top = _band_median(top_y0, top_y1) if have_top else np.full(W, np.nan)
    bg_bot = _band_median(bot_y0, bot_y1) if have_bot else np.full(W, np.nan)

    # ── 4. Remplissage des colonnes invalides (bords X) par plus proche ─────
    def _fill_nearest(v):
        v = v.copy()
        good = np.isfinite(v)
        if not good.any():
            return np.zeros_like(v)
        idx = np.arange(len(v))
        v[~good] = np.interp(idx[~good], idx[good], v[good])
        return v

    # Si une seule bande est exploitable, on l'utilise pour les deux niveaux
    if have_top and not have_bot:
        bg_top = _fill_nearest(bg_top); bg_bot = bg_top.copy()
    elif have_bot and not have_top:
        bg_bot = _fill_nearest(bg_bot); bg_top = bg_bot.copy()
    else:
        bg_top = _fill_nearest(bg_top)
        bg_bot = _fill_nearest(bg_bot)

    # ── 5. Lissage spectral (le fond est large bande) ───────────────────────
    def _smooth(v):
        win = int(smooth_x)
        win = win if win % 2 == 1 else win + 1
        win = min(win, (W - 1) if (W - 1) % 2 == 1 else (W - 2))
        if win >= 5:
            try:
                return savgol_filter(v, win, 3)
            except ValueError:
                return gaussian_filter1d(v, win / 4)
        return v

    bg_top_s = _smooth(bg_top)
    bg_bot_s = _smooth(bg_bot)

    # ── 6. Interpolation linéaire en Y (extrapolation plate hors bandes) ────
    yc_top = 0.5 * (top_y0 + top_y1)
    yc_bot = 0.5 * (bot_y0 + bot_y1)
    yy = np.arange(H, dtype=float)[:, None]
    denom = (yc_bot - yc_top) if (yc_bot - yc_top) != 0 else 1.0
    f = np.clip((yy - yc_top) / denom, 0.0, 1.0)      # clamp = extrapolation plate
    bg_2d = (1.0 - f) * bg_top_s[None, :] + f * bg_bot_s[None, :]

    if return_components:
        return bg_2d, {
            'bg_top': bg_top, 'bg_bot': bg_bot,
            'bg_top_smooth': bg_top_s, 'bg_bot_smooth': bg_bot_s,
            'top_band': (top_y0, top_y1), 'bot_band': (bot_y0, bot_y1),
            'yc_top': yc_top, 'yc_bot': yc_bot,
            'y_first': y_first, 'y_last': y_last,
            'have_top': have_top, 'have_bot': have_bot,
        }
    return bg_2d


def subtract_background_edges(arr, calib=None, clip=False, **kwargs):
    """
    Soustrait le fond avant/après estimé par estimate_background_edges().

    Paramètres
    ----------
    arr   : ndarray (H, W) — image rotée
    calib : dict ou None
    clip  : bool — si True, force les valeurs négatives à 0.
            DÉFAUT False (recommandé) : garder le résidu signé permet de
            VÉRIFIER que la soustraction est non biaisée. Ne mettre True
            que pour l'affichage final, jamais avant un calcul d'aire.
    **kwargs : transmis à estimate_background_edges (top_margin, smooth_x, …)

    Retourne
    --------
    arr_sub : ndarray (H, W)
    """
    bg_2d = estimate_background_edges(arr, calib=calib, **kwargs)
    arr_sub = arr - bg_2d
    if clip:
        arr_sub = np.clip(arr_sub, 0.0, None)
    return arr_sub


def plot_background_edges_diagnostic(arr, calib=None, fiber_y=None,
                                     col_examples=None, **kwargs):
    """
    Diagnostic visuel de la soustraction de fond avant/après.

    4 panneaux :
      (a) image rotée avec les bandes de fond (haut/bas) et l'étendue des
          fibres surlignées → vérifier qu'aucune fibre n'entre dans les bandes
      (b) profil vertical médian (image) avec les bandes marquées
      (c) fond par colonne : bande haut vs bas, brut vs lissé
          → vérifier que le fond varie bien en X et que le lissage est doux
      (d) quelques coupes verticales (colonnes) : image vs fond interpolé
          → vérifier que le fond passe SOUS le signal, pas dedans

    Renvoie la figure (utile pour export).
    """
    import matplotlib.pyplot as plt

    bg_2d, comp = estimate_background_edges(
        arr, calib=calib, fiber_y=fiber_y, return_components=True, **kwargs)
    H, W = arr.shape
    t0, t1 = comp['top_band']; b0, b1 = comp['bot_band']

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # (a) image + bandes
    ax = axes[0, 0]
    vmax = np.percentile(arr[arr > 1], 99)
    ax.imshow(arr, aspect='auto', cmap='gray', vmin=0, vmax=vmax)
    ax.axhspan(t0, t1, color='deepskyblue', alpha=0.25, label='bande HAUT')
    ax.axhspan(b0, b1, color='orange', alpha=0.25, label='bande BAS')
    ax.axhline(comp['y_first'], color='lime', lw=0.8, ls='--')
    ax.axhline(comp['y_last'], color='lime', lw=0.8, ls='--')
    ax.set_title('(a) Image rotée + bandes de fond', fontsize=10)
    ax.set_xlabel('X (px)'); ax.set_ylabel('Y (px)')
    ax.legend(fontsize=7, loc='upper right')

    # (b) profil vertical médian
    ax = axes[0, 1]
    prof = np.median(arr, axis=1)
    ax.plot(prof, np.arange(H), color='k', lw=0.7)
    ax.axhspan(t0, t1, color='deepskyblue', alpha=0.25)
    ax.axhspan(b0, b1, color='orange', alpha=0.25)
    ax.invert_yaxis()
    ax.set_title('(b) Profil vertical médian', fontsize=10)
    ax.set_xlabel('ADU'); ax.set_ylabel('Y (px)')
    ax.set_xlim(0, np.percentile(prof, 98) * 1.5)

    # (c) fond par colonne
    ax = axes[1, 0]
    x = np.arange(W)
    ax.plot(x, comp['bg_top'], color='deepskyblue', lw=0.4, alpha=0.4,
            label='haut (brut)')
    ax.plot(x, comp['bg_bot'], color='orange', lw=0.4, alpha=0.4,
            label='bas (brut)')
    ax.plot(x, comp['bg_top_smooth'], color='navy', lw=1.2, label='haut (lissé)')
    ax.plot(x, comp['bg_bot_smooth'], color='darkred', lw=1.2, label='bas (lissé)')
    ax.set_title('(c) Fond par colonne (axe spectral)', fontsize=10)
    ax.set_xlabel('X (px)'); ax.set_ylabel('ADU')
    ax.legend(fontsize=7); ax.grid(True, alpha=0.2)

    # (d) coupes verticales : image vs fond
    ax = axes[1, 1]
    if col_examples is None:
        col_examples = [int(W * f) for f in (0.2, 0.5, 0.8)]
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(col_examples)))
    for c, xc in zip(colors, col_examples):
        ax.plot(np.arange(H), arr[:, xc], color=c, lw=0.5, alpha=0.7,
                label=f'image X={xc}')
        ax.plot(np.arange(H), bg_2d[:, xc], color=c, lw=1.4, ls='--')
    ax.set_title('(d) Coupes Y : image (—) vs fond interpolé (- -)',
                 fontsize=10)
    ax.set_xlabel('Y (px)'); ax.set_ylabel('ADU')
    ax.set_ylim(0, np.percentile(arr[arr > 1], 97))
    ax.legend(fontsize=7); ax.grid(True, alpha=0.2)

    plt.suptitle('DIAGNOSTIC FOND AVANT/APRÈS', fontsize=13, fontweight='bold')
    plt.tight_layout()
    return fig


def remove_spectral_baseline(spectrum, wl_axis=None, window_nm=40.0,
                              window_px=None, n_iter=3):
    """
    Supprime le piédestal/baseline résiduel d'un spectre extrait.

    Algorithme : morphological opening itéré (SNIP simplifié).
    1. Rolling minimum sur une fenêtre large → enveloppe inférieure brute
    2. Lissage de cette enveloppe (gaussian_filter1d)
    3. Soustraction
    4. Itérer (chaque passe capture un peu mieux la baseline
       car les pics ont été réduits)

    Pourquoi c'est mieux que rien
    -----------------------------
    Le fond inter-fibre soustrait au niveau image capture le scattered light
    SPATIAL. Mais il reste un piédestal SPECTRAL : le spectre de la lumière
    diffusée n'est pas plat, il a une forme qui suit grossièrement
    l'enveloppe spectrale. Ce piédestal crée les "marches" que tu vois.

    Limites
    -------
    • Si la fenêtre est trop petite, elle mange les ailes des raies larges
    • Si trop grande, elle ne capture pas les variations rapides du fond
    • 40 nm est un bon compromis pour un réseau 150 gr/mm centré à 800 nm
      (raies d'émission larges de 10–20 nm au max)

    Paramètres
    ----------
    spectrum   : ndarray (n_cols,)  — spectre 1D
    wl_axis    : ndarray (n_cols,) ou None — pour convertir window_nm en pixels
    window_nm  : float — fenêtre en nm (utilisé si wl_axis est fourni)
    window_px  : int ou None — fenêtre en pixels (priorité sur window_nm)
    n_iter     : int — nombre d'itérations (2–4 typique)

    Retourne
    --------
    spectrum_clean : ndarray (n_cols,) — spectre avec baseline soustraite
    baseline       : ndarray (n_cols,) — baseline estimée (pour diagnostic)
    """
    spec = spectrum.astype(float).copy()
    n = len(spec)

    # Déterminer la fenêtre en pixels
    if window_px is not None:
        hw = window_px // 2
    elif wl_axis is not None and len(wl_axis) > 1:
        disp = abs(float(wl_axis[-1] - wl_axis[0])) / (len(wl_axis) - 1)
        hw = max(5, int(window_nm / disp / 2))
    else:
        hw = max(5, n // 50)

    # Morphological opening itéré
    baseline = spec.copy()
    for _ in range(n_iter):
        # Rolling minimum (= érosion)
        eroded = np.array([baseline[max(0, i - hw):min(n, i + hw + 1)].min()
                           for i in range(n)])
        # Lissage (= dilatation douce) — σ = hw/2 pour lisser sans remonter
        baseline = gaussian_filter1d(eroded, sigma=hw / 2)
        # La baseline ne peut pas dépasser le spectre original
        baseline = np.minimum(baseline, spec)

    spectrum_clean = np.clip(spec - baseline, 0.0, None)
    return spectrum_clean, baseline


def apply_edge_taper(spectrum, wl_axis=None, noise_percentile=15,
                     taper_width_nm=10.0, taper_width_px=None,
                     min_signal_ratio=2.0):
    """
    Applique une transition sigmoïde douce aux bords du spectre.

    Problème résolu
    ---------------
    Aux extrémités de la plage spectrale, le rapport S/N tombe en dessous
    de 1 et le spectre montre du bruit pur ou un résidu de fond qui crée
    des "marches" abruptes. Plutôt que de couper net (clip binaire),
    on applique une transition continue vers zéro.

    Algorithme
    ----------
    1. Estimer le niveau de bruit (percentile bas du spectre)
    2. Pour chaque bord, trouver le dernier pixel où le signal dépasse
       min_signal_ratio × bruit
    3. Appliquer une demi-sigmoïde de largeur taper_width entre ce pixel
       et le bord

    Paramètres
    ----------
    spectrum         : ndarray (n_cols,)
    wl_axis          : ndarray (n_cols,) ou None
    noise_percentile : float — percentile pour estimer le bruit (10–20)
    taper_width_nm   : float — largeur de la zone de transition en nm
    taper_width_px   : int ou None — largeur en pixels (priorité)
    min_signal_ratio : float — S/bruit minimum pour considérer qu'il y a
                       du signal (2.0 = signal > 2× le bruit)

    Retourne
    --------
    spectrum_tapered : ndarray (n_cols,)
    taper_mask       : ndarray (n_cols,) — le masque sigmoïde [0, 1]
    """
    spec = spectrum.astype(float).copy()
    n = len(spec)

    # Largeur de transition en pixels
    if taper_width_px is not None:
        tw = taper_width_px
    elif wl_axis is not None and len(wl_axis) > 1:
        disp = abs(float(wl_axis[-1] - wl_axis[0])) / (len(wl_axis) - 1)
        tw = max(5, int(taper_width_nm / disp))
    else:
        tw = max(5, n // 100)

    # Estimation du bruit
    positive = spec[spec > 0]
    if len(positive) < 10:
        return spec, np.ones(n)
    noise_level = np.percentile(positive, noise_percentile)
    threshold = noise_level * min_signal_ratio

    # Lisser le spectre pour détecter les bords de manière robuste
    spec_smooth = gaussian_filter1d(spec, sigma=tw / 3)

    # Trouver le bord gauche : premier pixel soutenu au-dessus du seuil
    left_edge = 0
    above = spec_smooth > threshold
    # Chercher une zone contiguë d'au moins tw/2 pixels au-dessus du seuil
    run_length = 0
    for i in range(n):
        if above[i]:
            run_length += 1
            if run_length >= max(3, tw // 3):
                left_edge = i - run_length + 1
                break
        else:
            run_length = 0

    # Trouver le bord droit
    right_edge = n - 1
    run_length = 0
    for i in range(n - 1, -1, -1):
        if above[i]:
            run_length += 1
            if run_length >= max(3, tw // 3):
                right_edge = i + run_length - 1
                break
        else:
            run_length = 0

    # Construire le masque sigmoïde
    taper_mask = np.ones(n)
    x = np.arange(n, dtype=float)

    # Taper gauche : sigmoïde centrée sur left_edge, montant de 0 à 1
    if left_edge > 0:
        # sigmoid(t) = 1 / (1 + exp(-k*t)) avec t = (x - center) / scale
        center_l = float(left_edge)
        scale_l = max(1.0, float(tw) / 6.0)  # 6σ ≈ transition complète
        sigmoid_l = 1.0 / (1.0 + np.exp(-(x - center_l) / scale_l))
        taper_mask = np.minimum(taper_mask, sigmoid_l)

    # Taper droit : sigmoïde descendante
    if right_edge < n - 1:
        center_r = float(right_edge)
        scale_r = max(1.0, float(tw) / 6.0)
        sigmoid_r = 1.0 / (1.0 + np.exp((x - center_r) / scale_r))
        taper_mask = np.minimum(taper_mask, sigmoid_r)

    spectrum_tapered = spec * taper_mask
    return spectrum_tapered, taper_mask


def clean_spectra(spectra, wl_axis=None, calib=None,
                  do_baseline=True, do_taper=True,
                  baseline_window_nm=40.0, baseline_n_iter=3,
                  taper_width_nm=10.0, noise_percentile=15,
                  min_signal_ratio=2.0,
                  verbose=False):
    """
    Pipeline complet de nettoyage post-extraction.

    Applique séquentiellement :
    1. remove_spectral_baseline  (si do_baseline=True)
    2. apply_edge_taper          (si do_taper=True)

    Pensé pour être appelé après extract_spectrum_optimal(), soit
    automatiquement par extract_all_spectra(), soit manuellement.

    Paramètres
    ----------
    spectra            : ndarray (n_fibers, n_cols)
    wl_axis            : ndarray (n_cols,) — axe en nm (recommandé)
    calib              : dict ou None — si fourni, wl_axis est extrait
    do_baseline        : bool — appliquer la suppression de baseline
    do_taper           : bool — appliquer le taper aux bords
    baseline_window_nm : float — fenêtre du morphological opening (nm)
    baseline_n_iter    : int — itérations du baseline removal
    taper_width_nm     : float — largeur de transition aux bords (nm)
    noise_percentile   : float — percentile pour estimer le bruit
    min_signal_ratio   : float — rapport S/N minimum aux bords
    verbose            : bool

    Retourne
    --------
    spectra_clean : ndarray (n_fibers, n_cols)
    """
    spectra = np.asarray(spectra, dtype=float)

    if wl_axis is None and calib is not None:
        wl_axis = calib.get('wl_axis', None)

    n_fibers, n_cols = spectra.shape
    spectra_clean = spectra.copy()

    # Seuil pour ignorer les fibres vides
    global_max = spectra.max()
    empty_thresh = 0.01 * global_max if global_max > 0 else 0

    for i in range(n_fibers):
        sp = spectra_clean[i]

        # Ignorer les fibres vides
        if sp.max() < empty_thresh:
            continue

        # Étape 1 : suppression baseline (piédestal)
        if do_baseline:
            sp, baseline = remove_spectral_baseline(
                sp, wl_axis=wl_axis,
                window_nm=baseline_window_nm,
                n_iter=baseline_n_iter)

            if verbose and i % 10 == 0:
                print(f"  Fiber {i:2d}: baseline max={baseline.max():.1f} ADU")

        # Étape 2 : taper aux bords
        if do_taper:
            sp, _ = apply_edge_taper(
                sp, wl_axis=wl_axis,
                noise_percentile=noise_percentile,
                taper_width_nm=taper_width_nm,
                min_signal_ratio=min_signal_ratio)

        spectra_clean[i] = sp

    return spectra_clean


def plot_noise_diagnostic(spectra_raw, spectra_clean, wl_axis,
                          fiber_indices=None, n_cols_plot=4):
    """
    Diagnostic visuel du pipeline de nettoyage.

    Pour chaque fibre, affiche sur le même graphe :
      • Spectre brut (gris)        — avant nettoyage
      • Baseline estimée (orange)  — ce qui est soustrait
      • Spectre nettoyé (bleu)     — résultat final
      • Masque de taper (vert)     — zone de transition aux bords

    Paramètres
    ----------
    spectra_raw   : ndarray (n_fibers, n_cols) — avant clean_spectra()
    spectra_clean : ndarray (n_fibers, n_cols) — après clean_spectra()
    wl_axis       : ndarray (n_cols,)
    fiber_indices : list ou None → auto (16 premières actives)
    n_cols_plot   : int
    """
    import matplotlib.pyplot as plt

    if fiber_indices is None:
        global_max = spectra_raw.max()
        active = [i for i in range(spectra_raw.shape[0])
                  if spectra_raw[i].max() >= 0.01 * global_max][:16]
        fiber_indices = active

    n_plot = len(fiber_indices)
    n_rows = int(np.ceil(n_plot / n_cols_plot))
    fig, axes = plt.subplots(n_rows, n_cols_plot,
                             figsize=(5 * n_cols_plot, 3.5 * n_rows),
                             squeeze=False)

    for k, fidx in enumerate(fiber_indices):
        row, col = divmod(k, n_cols_plot)
        ax = axes[row][col]

        raw   = spectra_raw[fidx]
        clean = spectra_clean[fidx]

        # Recalculer la baseline pour l'affichage
        _, baseline = remove_spectral_baseline(raw, wl_axis=wl_axis)
        _, taper_mask = apply_edge_taper(
            np.clip(raw - baseline, 0, None), wl_axis=wl_axis)

        ax.plot(wl_axis, raw,   color='lightgray', lw=0.7, label='Brut')
        ax.plot(wl_axis, baseline, color='darkorange', lw=1.2,
                ls='--', label='Baseline')
        ax.plot(wl_axis, clean, color='steelblue', lw=0.9, label='Nettoyé')

        # Afficher le taper en vert sur l'axe secondaire
        ax2 = ax.twinx()
        ax2.plot(wl_axis, taper_mask, color='green', lw=0.8, alpha=0.5)
        ax2.set_ylim(-0.05, 1.15)
        ax2.set_ylabel('Taper', fontsize=7, color='green')
        ax2.tick_params(axis='y', labelsize=6, colors='green')

        reduction = 0
        if raw.max() > 0:
            reduction = (1 - clean.max() / raw.max()) * 100

        ax.set_title(f'Fibre {fidx} | -{reduction:.0f}% peak', fontsize=9)
        ax.set_xlabel('λ (nm)', fontsize=8)
        ax.set_ylabel('ADU', fontsize=8)
        ax.tick_params(labelsize=7)
        ax.legend(fontsize=6, loc='upper left')
        ax.grid(True, alpha=0.2)

    for k in range(n_plot, n_rows * n_cols_plot):
        row, col = divmod(k, n_cols_plot)
        axes[row][col].set_visible(False)

    plt.suptitle('DIAGNOSTIC NETTOYAGE — baseline + edge taper',
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.show()

 

 

# =============================================================================
# INTENSITY CALIBRATION — relative, from HgAr calibration image
# =============================================================================
#
# Method: fiber 0 is taken as the intensity reference.
# For each wavelength λ:
#   factor_i(λ) = spectrum_fiber0(λ) / spectrum_fiber_i(λ)
#   corrected_i(λ) = raw_i(λ) × factor_i(λ)
#
# This normalises all fibres relative to fibre 0 and corrects for
# fibre-to-fibre throughput differences across the spectral range.
#
# Usage:
#   calib = sf.build_calibration(CALIB_PATH, ...)  # must be run first
#   int_calib_factors = sf.build_relative_intensity_calibration(calib)
#   spectra_cal = sf.apply_intensity_calibration(spectra, int_calib_factors)



def build_relative_intensity_calibration(calib, ref_fiber=0,
                                          min_signal_frac=0.01,
                                          verbose=True):
    """
    Build relative intensity calibration factors from the HgAr calibration image.

    Fiber `ref_fiber` (0-based, default 0) is taken as the intensity reference.
    For each wavelength pixel k:
        factor_i(k) = spectrum_ref(k) / spectrum_i(k)

    Applying factors:
        corrected_i = raw_i * factor_i
    normalises all fibers to the throughput of fiber `ref_fiber`.

    Pixels where spectrum_i < min_signal_frac * max(spectrum_i) yield NaN factors
    to avoid dividing by noise.

    Parameters
    ----------
    calib           : dict  — output of build_calibration()
    ref_fiber       : int   — 0-based index of the reference fiber (default 0)
    min_signal_frac : float — minimum relative signal threshold (default 0.01)
    verbose         : bool

    Returns
    -------
    calib_factors : ndarray (n_fibers, n_cols)
    """
    n_fibers = calib['n_fibers']
    n_cols   = calib['image_shape'][1]

    calib_spectra = _extract_calib_spectra(calib)
    ref_sp        = calib_spectra[ref_fiber].copy()

    if verbose:
        print(f"\n── Relative intensity calibration (reference: fiber {ref_fiber}) ──")
        print(f"  Reference max: {ref_sp.max():.1f} ADU")

    calib_factors = np.ones((n_fibers, n_cols), dtype=float)

    for i in range(n_fibers):
        sp_i       = calib_spectra[i]
        threshold  = min_signal_frac * sp_i.max() if sp_i.max() > 0 else 1e-9
        valid      = (sp_i > threshold) & (ref_sp > threshold)

        factor = np.full(n_cols, np.nan, dtype=float)
        factor[valid] = ref_sp[valid] / sp_i[valid]
        calib_factors[i] = factor

        if verbose:
            n_v = int(valid.sum())
            with np.errstate(invalid='ignore'):
                fmin = np.nanmin(factor)
                fmax = np.nanmax(factor)
            print(f"  Fiber {i:2d}: {n_v}/{n_cols} valid pixels  "
                  f"factor [{fmin:.3f} – {fmax:.3f}]")

    if verbose:
        print(f"  Done. Shape: {calib_factors.shape}")

    return calib_factors


def _extract_calib_spectra(calib):
    """
    Re-extract individual fiber spectra from the (already straightened)
    calibration image stored in calib['calib_path'].
    """
    arr_raw   = load_image(calib['calib_path'])
    arr       = _rotate_image(arr_raw.astype(float), calib['calib_angle_deg'])
    n_fibers  = calib['n_fibers']
    n_cols    = calib['image_shape'][1]
    hw        = calib['half_width']
    col_ref   = calib['col_ref']

    spectra = np.zeros((n_fibers, n_cols))
    for i in range(n_fibers):
        if np.isnan(calib['fiber_y_ref'][i]):
            continue
        spectra[i] = extract_spectrum_optimal(
            arr, calib['fiber_y_ref'][i], calib['tilt_slopes'][i], col_ref, hw)
    return spectra


def apply_intensity_calibration(spectra, calib_factors, floor=0):
    """
    Apply relative intensity calibration factors to an array of spectra.

        corrected[i, k] = raw[i, k] * factor[i, k]

    Pixels where factor is NaN or <= floor are set to NaN in the output.

    Parameters
    ----------
    spectra       : ndarray (n_fibers, n_cols)
    calib_factors : ndarray (n_fibers, n_cols) — from build_relative_intensity_calibration
    floor         : float — minimum valid factor (default 0)

    Returns
    -------
    spectra_cal : ndarray (n_fibers, n_cols)
    """
    spectra       = np.asarray(spectra,       dtype=float)
    calib_factors = np.asarray(calib_factors, dtype=float)

    if spectra.shape != calib_factors.shape:
        raise ValueError(
            f"Shape mismatch: spectra {spectra.shape} vs factors {calib_factors.shape}"
        )

    spectra_cal = np.full_like(spectra, np.nan, dtype=float)
    valid = np.isfinite(calib_factors) & (calib_factors > floor)
    spectra_cal[valid] = spectra[valid] * calib_factors[valid]
    return spectra_cal


def plot_intensity_calibration(calib_factors, wl_axis, fiber_indices=None):
    """
    Visualise relative intensity calibration factors.

    Panel 1: 2D heatmap — factor value as a function of fiber index and wavelength.
    Panel 2: individual factor curves for a subset of fibers.

    Parameters
    ----------
    calib_factors : ndarray (n_fibers, n_cols)
    wl_axis       : ndarray (n_cols,) in nm
    fiber_indices : list of 0-based indices, or None (auto: first 16 non-trivial)
    """
    import matplotlib.pyplot as plt

    calib_factors = np.asarray(calib_factors, dtype=float)
    wl_axis       = np.asarray(wl_axis,       dtype=float)
    n_fibers      = calib_factors.shape[0]

    non_trivial = [i for i in range(n_fibers)
                   if not np.all(np.isnan(calib_factors[i]))]
    if fiber_indices is None:
        fiber_indices = non_trivial[:16]

    fig, axes = plt.subplots(1, 2, figsize=(18, 5))

    # Panel 1: heatmap
    with np.errstate(invalid='ignore'):
        vmax = np.nanpercentile(calib_factors, 95)
        vmin = np.nanpercentile(calib_factors,  5)
    im = axes[0].imshow(
        calib_factors, aspect='auto', cmap='RdBu_r',
        vmin=vmin, vmax=vmax,
        extent=[wl_axis[0], wl_axis[-1], n_fibers - 0.5, -0.5]
    )
    plt.colorbar(im, ax=axes[0], label='Calibration factor')
    axes[0].set_xlabel('Wavelength (nm)')
    axes[0].set_ylabel('Fiber index (0-based)')
    axes[0].set_title('Relative intensity calibration factors\n'
                      '(fiber 0 = reference, factor=1)')

    # Panel 2: individual curves
    cols = plt.cm.tab20(np.linspace(0, 1, max(len(fiber_indices), 1)))
    for j, fidx in enumerate(fiber_indices):
        axes[1].plot(wl_axis, calib_factors[fidx], lw=0.9,
                     color=cols[j % len(cols)], label=f'F{fidx}')
    axes[1].axhline(1.0, color='k', lw=0.8, ls='--', alpha=0.5, label='factor=1')
    axes[1].set_xlabel('Wavelength (nm)')
    axes[1].set_ylabel('Calibration factor')
    axes[1].set_title(f'Factor curves — {len(fiber_indices)} example fibers')
    axes[1].legend(fontsize=6, ncol=4)
    axes[1].grid(True, alpha=0.25)

    plt.suptitle('RELATIVE INTENSITY CALIBRATION — fiber 0 as reference',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.show()


# =============================================================================

# MÉTRIQUES SPECTRALES

# =============================================================================

 

def compute_spectral_area(spectra, wl_axis, wl_range=None):

    """

    Calcule la somme du signal spectral pour chaque fibre.

    """

    if wl_range is not None:

        wl_min, wl_max = wl_range

        mask = (wl_axis >= wl_min) & (wl_axis <= wl_max)

        y = spectra[:, mask]

    else:

        y = spectra

    return y.sum(axis=1)


# =============================================================================

# DÉTECTION DE λ_MAX — version améliorée

# =============================================================================
#
# Stratégie : Savitzky-Golay modéré + fit gaussien local
#
# Problèmes de l'ancienne approche :
#   1. Lissage gaussien fort → déplace le pic (biais de convolution si spectre
#      asymétrique)
#   2. Seuil plateau 99.5 % → très sensible au bruit résiduel
#   3. Pas de validation → un pic spurieux n'est jamais rejeté
#
# Nouvelle stratégie (méthode 'gaussian_fit', défaut) :
#   1. Savitzky-Golay modéré : élimine le bruit sans déplacer les pics
#      (fit polynomial local ≠ convolution gaussienne)
#   2. find_peaks sur le SG → pic le plus proéminent
#   3. Fit gaussien local ±fit_window_nm → λ_max sub-pixel physiquement motivé
#   4. Repli : barycentre FWHM (> 50 % du max) si le fit gaussien diverge
#
# Méthodes disponibles :
#   'gaussian_fit'  (défaut) — le plus précis pour pics d'émission
#   'half_max'               — barycentre FWHM, robuste, sans fit
#   'multiscale'             — valide le pic à 3 niveaux de lissage ;
#                              retourne NaN si le pic est instable


def _gaussian_bg(x, A, mu, sigma, bg):
    """Gaussienne + fond plat."""
    return A * np.exp(-0.5 * ((x - mu) / sigma) ** 2) + bg


def _fit_gaussian_peak(wl, spec, peak_idx, window_nm=20.0):
    """
    Fit une gaussienne autour de `peak_idx` dans une fenêtre ±window_nm.

    Returns
    -------
    mu      : float — position du pic en nm (NaN si le fit échoue)
    sigma   : float — largeur gaussienne en nm
    success : bool
    """
    disp   = abs(wl[1] - wl[0])
    hw_px  = max(5, int(window_nm / disp / 2))

    i0 = max(0, peak_idx - hw_px)
    i1 = min(len(wl), peak_idx + hw_px + 1)
    if (i1 - i0) < 5:
        return np.nan, np.nan, False

    x = wl[i0:i1]
    y = spec[i0:i1]
    if y.max() <= 0:
        return np.nan, np.nan, False

    A0    = float(y.max() - np.percentile(y, 10))
    mu0   = float(x[np.argmax(y)])
    sig0  = window_nm / 6.0
    bg0   = float(np.percentile(y, 10))

    try:
        popt, _ = curve_fit(
            _gaussian_bg, x, y,
            p0=[A0, mu0, sig0, bg0],
            bounds=([0,        x[0],       disp,        -np.inf],
                    [np.inf,   x[-1],  window_nm * 2,    np.inf]),
            maxfev=2000
        )
        A_fit, mu_fit, sig_fit, _ = popt

        if A_fit < 0.05 * A0 or not (x[0] < mu_fit < x[-1]):
            return np.nan, np.nan, False

        return float(mu_fit), float(abs(sig_fit)), True

    except (RuntimeError, ValueError):
        return np.nan, np.nan, False


def _find_main_peak(wl, spec, savgol_window_nm=10.0, savgol_poly=3):
    """
    Localise l'indice du pic principal après Savitzky-Golay.
    Retourne l'indice (int) du pic le plus proéminent, ou None.
    """
    disp    = abs(wl[1] - wl[0])
    win_px  = int(savgol_window_nm / disp)
    win_px  = win_px if win_px % 2 == 1 else win_px + 1
    win_px  = max(savgol_poly + 2 if (savgol_poly + 2) % 2 == 1
                  else savgol_poly + 3, win_px)

    try:
        smooth = savgol_filter(spec.astype(float), win_px, savgol_poly)
    except ValueError:
        smooth = gaussian_filter1d(spec.astype(float), sigma=win_px / 4)

    smooth = np.clip(smooth, 0, None)
    if smooth.max() == 0:
        return None

    peaks, props = find_peaks(
        smooth,
        height=smooth.max() * 0.05,
        prominence=smooth.max() * 0.03,
        distance=max(3, int(5.0 / disp))
    )

    if len(peaks) == 0:
        return int(np.argmax(smooth))

    return int(peaks[np.argmax(props['prominences'])])


def _half_max_centroid(wl, spec_smooth):
    """
    Barycentre sur la région à > 50 % du max (zone FWHM).
    Plus robuste que le plateau 99.5 % de l'ancienne version.
    """
    vmax = spec_smooth.max()
    if vmax <= 0:
        return np.nan
    mask = spec_smooth >= 0.5 * vmax
    if not mask.any():
        return float(wl[np.argmax(spec_smooth)])
    return float(np.average(wl[mask], weights=spec_smooth[mask]))


def _lambda_max_gaussian_fit(wl, spec, savgol_window_nm=10.0,
                              savgol_poly=3, fit_window_nm=20.0):
    """Savitzky-Golay + fit gaussien local. Repli : barycentre FWHM."""
    disp   = abs(wl[1] - wl[0])
    win_px = int(savgol_window_nm / disp)
    win_px = win_px if win_px % 2 == 1 else win_px + 1
    win_px = max(savgol_poly + 2 if (savgol_poly + 2) % 2 == 1
                 else savgol_poly + 3, win_px)

    try:
        spec_sg = savgol_filter(spec.astype(float), win_px, savgol_poly)
    except ValueError:
        spec_sg = gaussian_filter1d(spec.astype(float), sigma=win_px / 4)

    spec_sg = np.clip(spec_sg, 0, None)

    peak_idx = _find_main_peak(wl, spec, savgol_window_nm, savgol_poly)
    if peak_idx is None:
        return np.nan

    mu, _, ok = _fit_gaussian_peak(wl, spec_sg, peak_idx, fit_window_nm)
    if ok:
        return mu

    # Repli : barycentre FWHM
    return _half_max_centroid(wl, spec_sg)


def _lambda_max_multiscale(wl, spec, savgol_window_nm=10.0,
                            savgol_poly=3, fit_window_nm=20.0,
                            stability_tol_nm=2.0):
    """
    Fit gaussien validé à 3 niveaux de lissage.
    Retourne NaN si le pic bouge de plus de stability_tol_nm entre les échelles.
    """
    scales_nm = [savgol_window_nm * 0.5,
                 savgol_window_nm,
                 savgol_window_nm * 2.5]
    mus = []

    for win_nm in scales_nm:
        disp   = abs(wl[1] - wl[0])
        win_px = int(win_nm / disp)
        win_px = win_px if win_px % 2 == 1 else win_px + 1
        win_px = max(savgol_poly + 2 if (savgol_poly + 2) % 2 == 1
                     else savgol_poly + 3, win_px)
        try:
            sg = savgol_filter(spec.astype(float), win_px, savgol_poly)
        except ValueError:
            sg = gaussian_filter1d(spec.astype(float), sigma=win_px / 4)
        sg = np.clip(sg, 0, None)

        pk = _find_main_peak(wl, spec, win_nm, savgol_poly)
        if pk is None:
            mus.append(np.nan)
            continue

        mu, _, ok = _fit_gaussian_peak(wl, sg, pk, fit_window_nm)
        mus.append(mu if ok else float(wl[pk]))

    mus   = np.array(mus)
    valid = mus[np.isfinite(mus)]

    if len(valid) < 2:
        return np.nan
    if np.ptp(valid) > stability_tol_nm:
        return np.nan   # pic instable → NaN explicite

    return float(np.median(valid))


def compute_lambda_max(
    spectra,
    wl_axis,
    wl_range=None,
    method='gaussian_fit',
    savgol_window_nm=10.0,
    savgol_poly=3,
    fit_window_nm=20.0,
    stability_tol_nm=2.0,
    # --- paramètre legacy conservé pour compatibilité ascendante ---
    kernel_sigma=None,
    debug=False
):
    """
    Détermine la longueur d'onde du maximum d'émission pour chaque fibre.

    Paramètres
    ----------
    spectra          : ndarray (n_fibers, n_cols) — spectres bruts
    wl_axis          : ndarray (n_cols,)          — axe en nm
    wl_range         : (wl_min, wl_max) ou None   — plage d'analyse
    method           : str
        'gaussian_fit'  Savitzky-Golay + fit gaussien local  [RECOMMANDÉ]
        'half_max'      Barycentre sur la région FWHM (> 50 % du max)
        'multiscale'    Fit gaussien validé à 3 échelles ; NaN si instable
    savgol_window_nm : float — fenêtre Savitzky-Golay en nm  (défaut 10 nm)
    savgol_poly      : int   — degré polynomial SG            (défaut 3)
    fit_window_nm    : float — demi-fenêtre fit gaussien en nm (défaut 20 nm)
    stability_tol_nm : float — tolérance multi-échelle en nm  (défaut 2 nm)
    kernel_sigma     : ignoré (conservé pour compatibilité avec l'ancienne API)
    debug            : bool  — affiche les résultats fibre par fibre

    Retourne
    --------
    lambda_max : ndarray (n_fibers,) — λ_max en nm
                 NaN pour les fibres vides ou dont le pic est instable
                 (méthode 'multiscale' uniquement)

    Différences clés avec l'ancienne version
    -----------------------------------------
    • Savitzky-Golay au lieu de gaussien pur → pas de biais de convolution
    • Fit gaussien local → résultat sub-pixel physiquement motivé
    • Barycentre à 50 % du max (FWHM) au lieu de 99.5 % → bien plus robuste
    • Méthode 'multiscale' pour rejeter les pics spurieux explicitement
    """
    spectra = np.asarray(spectra, dtype=float)
    wl_axis = np.asarray(wl_axis, dtype=float)

    if wl_range is not None:
        mask    = (wl_axis >= wl_range[0]) & (wl_axis <= wl_range[1])
        wl      = wl_axis[mask]
        sp_all  = spectra[:, mask]
    else:
        wl     = wl_axis
        sp_all = spectra

    if len(wl) < 10:
        raise ValueError("wl_range trop étroite : moins de 10 pixels disponibles.")

    n_fibers   = sp_all.shape[0]
    lambda_max = np.full(n_fibers, np.nan)
    global_max = sp_all.max()

    for i in range(n_fibers):
        spec = sp_all[i, :]

        # Fibre vide : max < 1 % du max global
        if global_max > 0 and spec.max() < 0.01 * global_max:
            if debug:
                print(f"Fibre {i+1:2d} : VIDE (max={spec.max():.1f})")
            continue

        if not np.any(np.isfinite(spec)) or spec.max() <= 0:
            continue

        if method == 'gaussian_fit':
            lm = _lambda_max_gaussian_fit(
                wl, spec,
                savgol_window_nm=savgol_window_nm,
                savgol_poly=savgol_poly,
                fit_window_nm=fit_window_nm
            )
        elif method == 'half_max':
            disp   = abs(wl[1] - wl[0])
            win_px = int(savgol_window_nm / disp)
            win_px = win_px if win_px % 2 == 1 else win_px + 1
            win_px = max(savgol_poly + 2 if (savgol_poly + 2) % 2 == 1
                         else savgol_poly + 3, win_px)
            try:
                sg = np.clip(savgol_filter(spec, win_px, savgol_poly), 0, None)
            except ValueError:
                sg = np.clip(gaussian_filter1d(spec, sigma=win_px / 4), 0, None)
            lm = _half_max_centroid(wl, sg)
        elif method == 'multiscale':
            lm = _lambda_max_multiscale(
                wl, spec,
                savgol_window_nm=savgol_window_nm,
                savgol_poly=savgol_poly,
                fit_window_nm=fit_window_nm,
                stability_tol_nm=stability_tol_nm
            )
        else:
            raise ValueError(f"Méthode inconnue : '{method}'. "
                             f"Choisir parmi 'gaussian_fit', 'half_max', 'multiscale'.")

        lambda_max[i] = lm

        if debug:
            status = f"λ_max = {lm:.2f} nm" if np.isfinite(lm) else "NaN (pic instable)"
            print(f"Fibre {i+1:2d} : {status}  (max brut = {spec.max():.1f})")

    return lambda_max


def plot_lambda_max_diagnostic(spectra, wl_axis, fiber_indices=None,
                                wl_range=None,
                                method='gaussian_fit',
                                savgol_window_nm=10.0,
                                savgol_poly=3,
                                fit_window_nm=20.0,
                                n_cols_plot=4):
    """
    Visualisation du fit λ_max pour un sous-ensemble de fibres.

    Pour chaque fibre, affiche :
      • Spectre brut (bleu clair)
      • Spectre lissé Savitzky-Golay (bleu foncé)
      • Fit gaussien local (orange tirets)
      • λ_max estimé (ligne rouge verticale)

    Paramètres
    ----------
    fiber_indices : liste d'indices 0-based ou None → 16 premières fibres actives
    n_cols_plot   : colonnes dans la figure (défaut 4)

    Exemple d'utilisation
    ---------------------
    sf.plot_lambda_max_diagnostic(spectra, calib['wl_axis'],
                                   wl_range=(700, 850),
                                   method='gaussian_fit')
    """
    import matplotlib.pyplot as plt

    spectra = np.asarray(spectra, dtype=float)
    wl_axis = np.asarray(wl_axis, dtype=float)

    if wl_range is not None:
        mask   = (wl_axis >= wl_range[0]) & (wl_axis <= wl_range[1])
        wl     = wl_axis[mask]
        sp_all = spectra[:, mask]
    else:
        wl     = wl_axis
        sp_all = spectra

    if fiber_indices is None:
        global_max = sp_all.max()
        active = [i for i in range(sp_all.shape[0])
                  if sp_all[i].max() >= 0.01 * global_max][:16]
        fiber_indices = active

    n_plot = len(fiber_indices)
    n_rows = int(np.ceil(n_plot / n_cols_plot))
    fig, axes = plt.subplots(n_rows, n_cols_plot,
                             figsize=(5 * n_cols_plot, 3.5 * n_rows),
                             squeeze=False)

    disp   = abs(wl[1] - wl[0])
    win_px = int(savgol_window_nm / disp)
    win_px = win_px if win_px % 2 == 1 else win_px + 1
    win_px = max(savgol_poly + 2 if (savgol_poly + 2) % 2 == 1
                 else savgol_poly + 3, win_px)

    for k, fidx in enumerate(fiber_indices):
        row, col = divmod(k, n_cols_plot)
        ax  = axes[row][col]
        sp  = sp_all[fidx]

        try:
            sg = np.clip(savgol_filter(sp, win_px, savgol_poly), 0, None)
        except ValueError:
            sg = np.clip(gaussian_filter1d(sp, sigma=win_px / 4), 0, None)

        lm = compute_lambda_max(
            sp[np.newaxis, :], wl,
            method=method,
            savgol_window_nm=savgol_window_nm,
            savgol_poly=savgol_poly,
            fit_window_nm=fit_window_nm
        )[0]

        ax.plot(wl, sp, color='lightsteelblue', lw=0.7, alpha=0.8, label='Brut')
        ax.plot(wl, sg, color='steelblue',      lw=1.2,
                label=f'SG {savgol_window_nm:.0f} nm')

        pk = _find_main_peak(wl, sp, savgol_window_nm, savgol_poly)
        if pk is not None:
            mu, sigma_fit, ok = _fit_gaussian_peak(wl, sg, pk, fit_window_nm)
            if ok:
                x_fit = np.linspace(wl[0], wl[-1], 800)
                hw_px = int(fit_window_nm / disp / 2)
                i0 = max(0, pk - hw_px);  i1 = min(len(wl), pk + hw_px + 1)
                bg0 = float(np.percentile(sg[i0:i1], 10))
                A0  = float(sg[pk] - bg0)
                y_fit = _gaussian_bg(x_fit, A0, mu, sigma_fit, bg0)
                ax.plot(x_fit, y_fit, color='darkorange', lw=1.8,
                        ls='--', label=f'Fit G σ={sigma_fit:.1f} nm')

        if np.isfinite(lm):
            ax.axvline(lm, color='red', lw=1.5,
                       label=f'λ_max = {lm:.1f} nm')
        else:
            ax.set_facecolor('#fff0f0')

        ax.set_title(f'Fibre {fidx + 1}', fontsize=9)
        ax.set_xlabel('λ (nm)', fontsize=8)
        ax.set_ylabel('ADU',    fontsize=8)
        ax.tick_params(labelsize=7)
        ax.legend(fontsize=6.5, loc='upper left')
        ax.grid(True, alpha=0.25)

    for k in range(n_plot, n_rows * n_cols_plot):
        row, col = divmod(k, n_cols_plot)
        axes[row][col].set_visible(False)

    plt.suptitle(
        f"Diagnostic λ_max — méthode : '{method}'  |  "
        f"SG {savgol_window_nm:.0f} nm  |  fenêtre fit {fit_window_nm:.0f} nm",
        fontsize=12, fontweight='bold'
    )
    plt.tight_layout()
    plt.show()


# =============================================================================
# BARYCENTRE SPECTRAL
# =============================================================================
#
# Motivation : λ_max fluctue au gré du bruit sur le pic.
# Le barycentre λ_c = Σ(λ·I) / Σ(I) utilise TOUT le spectre → insensible
# à un pixel bruité isolé. Il mesure le "centre de gravité" spectral, donc
# exactement le déplacement global en λ avec l'angle — ce qui nous intéresse.
#
# Pipeline de calcul :
#   1. Restriction à wl_range (optionnel)
#   2. Soustraction du fond : percentile bas du spectre (fond diffus + biais CCD)
#   3. Clip à 0 (pas de poids négatifs)
#   4. Lissage SG léger (réduit l'influence des pics de bruit isolés sans déplacer
#      le centre de masse — le lissage n'affecte PAS la position du barycentre
#      si le spectre est symétrique, et l'affecte très peu sinon)
#   5. Barycentre pondéré
#
# Variante robuste disponible : 'trimmed' — exclut les x% extrêmes en λ
# (utile si le bord de la plage spectrale est bruité).


def compute_spectral_centroid(
    spectra,
    wl_axis,
    wl_range=None,
    bg_percentile=10,
    savgol_window_nm=5.0,
    savgol_poly=3,
    trim_frac=0.0,
    debug=False,
):
    """
    Calcule le barycentre spectral (λ moyen pondéré par l'intensité) pour chaque fibre.

        λ_c = Σ(λ_i · max(I_i - bg, 0)) / Σ(max(I_i - bg, 0))

    Plus robuste que λ_max : utilise tout le spectre → un pic de bruit isolé
    ne peut déplacer λ_c que de quelques centièmes de nm.

    Paramètres
    ----------
    spectra          : ndarray (n_fibers, n_cols)
    wl_axis          : ndarray (n_cols,) en nm
    wl_range         : (wl_min, wl_max) ou None — plage d'intégration
    bg_percentile    : float (0–30) — percentile utilisé pour estimer le fond.
                       10 % est conservateur ; augmenter à 20–25 si la ligne
                       de base est haute et variable.
    savgol_window_nm : float — lissage SG avant calcul (réduit les pics isolés
                       sans déplacer le barycentre). Mettre 0 pour désactiver.
    savgol_poly      : int — degré du filtre SG
    trim_frac        : float (0.0–0.3) — fraction des bords spectraux à exclure
                       du calcul (utile si les bords de la plage sont bruités).
                       Ex : 0.05 ignore les 5 % de pixels aux deux extrémités.
    debug            : bool — affiche les résultats fibre par fibre

    Retourne
    --------
    centroid : ndarray (n_fibers,) — λ_c en nm (NaN pour fibres vides)
    """
    spectra = np.asarray(spectra, dtype=float)
    wl_axis = np.asarray(wl_axis, dtype=float)

    if wl_range is not None:
        mask    = (wl_axis >= wl_range[0]) & (wl_axis <= wl_range[1])
        wl      = wl_axis[mask]
        sp_all  = spectra[:, mask]
    else:
        wl     = wl_axis
        sp_all = spectra

    if len(wl) < 5:
        raise ValueError("wl_range trop étroite : moins de 5 pixels disponibles.")

    # Masque de trim sur les bords
    n_trim  = max(0, int(len(wl) * trim_frac))
    use_mask = np.ones(len(wl), dtype=bool)
    if n_trim > 0:
        use_mask[:n_trim]  = False
        use_mask[-n_trim:] = False

    # Paramètres SG
    disp    = abs(wl[1] - wl[0])
    do_sg   = savgol_window_nm > 0
    if do_sg:
        win_px = int(savgol_window_nm / disp)
        win_px = win_px if win_px % 2 == 1 else win_px + 1
        win_px = max(savgol_poly + 2 if (savgol_poly + 2) % 2 == 1
                     else savgol_poly + 3, win_px)

    n_fibers  = sp_all.shape[0]
    centroid  = np.full(n_fibers, np.nan)
    global_max = sp_all.max()

    for i in range(n_fibers):
        spec = sp_all[i, :]

        # Fibre vide
        if global_max > 0 and spec.max() < 0.01 * global_max:
            if debug:
                print(f"Fibre {i+1:2d} : VIDE")
            continue
        if not np.any(np.isfinite(spec)) or spec.max() <= 0:
            continue

        # Lissage SG léger
        if do_sg:
            try:
                spec_w = savgol_filter(spec.astype(float), win_px, savgol_poly)
            except ValueError:
                spec_w = gaussian_filter1d(spec.astype(float), sigma=win_px / 4)
        else:
            spec_w = spec.astype(float)

        # Soustraction fond
        bg       = np.percentile(spec_w, bg_percentile)
        weights  = np.clip(spec_w - bg, 0.0, None)
        weights  = weights * use_mask   # applique trim

        total = weights.sum()
        if total <= 0:
            if debug:
                print(f"Fibre {i+1:2d} : poids nuls après soustraction fond")
            continue

        lc = float(np.dot(wl, weights) / total)
        centroid[i] = lc

        if debug:
            print(f"Fibre {i+1:2d} : λ_c = {lc:.2f} nm  "
                  f"(max brut = {spec.max():.1f},  fond = {bg:.1f})")

    return centroid


def plot_spectral_centroid_diagnostic(
    spectra,
    wl_axis,
    fiber_indices=None,
    wl_range=None,
    bg_percentile=10,
    savgol_window_nm=5.0,
    savgol_poly=3,
    trim_frac=0.0,
    n_cols_plot=4,
):
    """
    Visualisation du barycentre spectral pour un sous-ensemble de fibres.

    Pour chaque fibre :
      • Spectre brut (bleu clair)
      • Spectre lissé SG + fond soustrait (bleu foncé, poids utilisés)
      • Zone grisée = contribution au barycentre (I - fond, clippée)
      • Ligne rouge = λ_c (barycentre)
      • Ligne orange pointillée = fond estimé (bg_percentile)

    Paramètres
    ----------
    fiber_indices : liste d'indices 0-based ou None → 16 premières fibres actives
    n_cols_plot   : colonnes dans la figure (défaut 4)

    Exemple
    -------
    sf.plot_spectral_centroid_diagnostic(spectra, calib['wl_axis'],
                                          wl_range=(700, 850))
    """
    import matplotlib.pyplot as plt

    spectra = np.asarray(spectra, dtype=float)
    wl_axis = np.asarray(wl_axis, dtype=float)

    if wl_range is not None:
        mask   = (wl_axis >= wl_range[0]) & (wl_axis <= wl_range[1])
        wl     = wl_axis[mask]
        sp_all = spectra[:, mask]
    else:
        wl     = wl_axis
        sp_all = spectra

    # Sélection des fibres
    if fiber_indices is None:
        global_max = sp_all.max()
        active = [i for i in range(sp_all.shape[0])
                  if sp_all[i].max() >= 0.01 * global_max][:16]
        fiber_indices = active

    # Tous les barycentres (pour afficher les statistiques)
    all_centroids = compute_spectral_centroid(
        sp_all, wl,
        bg_percentile=bg_percentile,
        savgol_window_nm=savgol_window_nm,
        savgol_poly=savgol_poly,
        trim_frac=trim_frac,
    )

    # Paramètres SG
    disp   = abs(wl[1] - wl[0])
    do_sg  = savgol_window_nm > 0
    if do_sg:
        win_px = int(savgol_window_nm / disp)
        win_px = win_px if win_px % 2 == 1 else win_px + 1
        win_px = max(savgol_poly + 2 if (savgol_poly + 2) % 2 == 1
                     else savgol_poly + 3, win_px)

    # Masque trim
    n_trim   = max(0, int(len(wl) * trim_frac))
    use_mask = np.ones(len(wl), dtype=bool)
    if n_trim > 0:
        use_mask[:n_trim] = False; use_mask[-n_trim:] = False

    n_plot = len(fiber_indices)
    n_rows = int(np.ceil(n_plot / n_cols_plot))
    fig, axes = plt.subplots(n_rows, n_cols_plot,
                             figsize=(5 * n_cols_plot, 3.5 * n_rows),
                             squeeze=False)

    for k, fidx in enumerate(fiber_indices):
        row, col = divmod(k, n_cols_plot)
        ax  = axes[row][col]
        sp  = sp_all[fidx]

        # Lissage SG
        if do_sg:
            try:
                sp_sg = savgol_filter(sp.astype(float), win_px, savgol_poly)
            except ValueError:
                sp_sg = gaussian_filter1d(sp.astype(float), sigma=win_px / 4)
        else:
            sp_sg = sp.astype(float)

        bg      = np.percentile(sp_sg, bg_percentile)
        weights = np.clip(sp_sg - bg, 0.0, None) * use_mask
        lc      = all_centroids[fidx]

        # Tracé
        ax.plot(wl, sp,   color='lightsteelblue', lw=0.7, alpha=0.8, label='Brut')
        ax.plot(wl, sp_sg, color='steelblue',      lw=1.1,
                label=f'SG {savgol_window_nm:.0f} nm')
        ax.fill_between(wl, bg, sp_sg, where=(weights > 0),
                        alpha=0.25, color='steelblue',
                        label='Poids barycentre')
        ax.axhline(bg, color='darkorange', lw=1.0, ls='--',
                   alpha=0.8, label=f'Fond (p{bg_percentile:.0f})')

        if np.isfinite(lc):
            ax.axvline(lc, color='red', lw=1.8,
                       label=f'λ_c = {lc:.1f} nm')
        else:
            ax.set_facecolor('#fff0f0')

        ax.set_title(f'Fibre {fidx + 1}', fontsize=9)
        ax.set_xlabel('λ (nm)', fontsize=8)
        ax.set_ylabel('ADU',    fontsize=8)
        ax.tick_params(labelsize=7)
        ax.legend(fontsize=6.0, loc='upper left')
        ax.grid(True, alpha=0.25)

    # Axes vides
    for k in range(n_plot, n_rows * n_cols_plot):
        row, col = divmod(k, n_cols_plot)
        axes[row][col].set_visible(False)

    # Statistiques globales dans le titre
    valid_c = all_centroids[np.isfinite(all_centroids)]
    stats   = (f"  |  λ_c : [{valid_c.min():.1f} – {valid_c.max():.1f}] nm"
               f"  moy={valid_c.mean():.1f}  σ={valid_c.std():.2f} nm"
               if len(valid_c) > 0 else "")

    plt.suptitle(
        f"Diagnostic barycentre spectral  |  "
        f"SG {savgol_window_nm:.0f} nm  |  fond p{bg_percentile:.0f}"
        f"{stats}",
        fontsize=11, fontweight='bold'
    )
    plt.tight_layout()
    plt.show()


def plot_centroid_angular_map(
    spectra,
    config_name,
    wl_axis,
    wl_range=None,
    bg_percentile=10,
    savgol_window_nm=5.0,
    savgol_poly=3,
    trim_frac=0.0,
    cmap='plasma',
    n_grid=200,
    smoothing=2.0,
    title="",
    theta_shift=0.0,
    coverage_deg=None,
):
    """
    Carte angulaire 2D du barycentre spectral (θ vs φ).

    Affiche deux panneaux côte à côte :
      1. Scatter brut : 80 points colorés par λ_c
      2. Interpolation RBF continue (thin-plate spline)

    Paramètres
    ----------
    spectra       : ndarray (n_fibers, n_cols)
    config_name   : str — clé dans FIBER_CONFIGS
    wl_axis       : ndarray (n_cols,) en nm
    wl_range      : (wl_min, wl_max) ou None
    bg_percentile : fond soustrait avant calcul du barycentre
    savgol_window_nm : lissage SG (nm) avant calcul
    cmap          : colormap (défaut 'plasma')
    n_grid        : résolution de la grille RBF
    smoothing     : lissage de l'interpolation RBF (0 = exact)
    title         : titre de la figure

    Exemple
    -------
    sf.plot_centroid_angular_map(spectra, 'config3_e', calib['wl_axis'],
                                  wl_range=(700, 850))
    """
    # Angles indexés par indice d'extraction (inversion physique appliquée),
    # pour que phis[i]/thetas[i] correspondent bien à spectra[i] = centroids[i].
    phis, thetas = get_fiber_angles(config_name)

    centroids = compute_spectral_centroid(
        spectra, wl_axis,
        wl_range=wl_range,
        bg_percentile=bg_percentile,
        savgol_window_nm=savgol_window_nm,
        savgol_poly=savgol_poly,
        trim_frac=trim_frac,
    )

    label = "Spectral centroid λ_c (nm)"
    full_title = title or f"Spectral centroid — {config_name}"
    if wl_range is not None:
        full_title += f"  [{wl_range[0]:.0f}–{wl_range[1]:.0f} nm]"

    plot_angular_metric(phis, thetas, centroids,
                        label=label, title=full_title,
                        cmap=cmap, n_grid=n_grid, smoothing=smoothing,
                        theta_shift=theta_shift, coverage_deg=coverage_deg)


# =============================================================================

# EXTRACTION

# =============================================================================

 

def extract_all_spectra(image_path, calib, subtract_bg=True,
                        bg_method='edges', bg_clip=False,
                        clean_noise=False, show_detection=False,
                        baseline_window_nm=40.0, baseline_n_iter=3,
                        taper_width_nm=10.0, noise_percentile=15,
                        min_signal_ratio=2.0):
    """
    Extrait les N spectres d'une image science.

    Pipeline (mode par défaut = SOBRE : fond seul, sans baseline ni taper)
    ---------------------------------------------------------------------
    1. Rotation de l'image (fibres horizontales)
    2. Soustraction de fond (si subtract_bg=True) — méthode bg_method
    3. Micro-refinement des positions de fibre
    4. Extraction pondérée gaussienne (Horne 1986 simplifié)
    5. (optionnel) Nettoyage post-extraction baseline+taper si clean_noise=True

    Changement de défaut (important)
    --------------------------------
    • bg_method='edges'  : fond mesuré UNIQUEMENT au-dessus de la 1re fibre
      et en-dessous de la dernière (zones sans débordement possible).
      Alternative 'interfiber' = ancienne méthode (fond entre fibres).
    • bg_clip=False      : la soustraction ne force PAS les négatifs à 0,
      pour permettre de vérifier qu'elle est non biaisée.
    • clean_noise=False  : baseline SNIP + edge taper DÉSACTIVÉS par défaut
      (jugés trop agressifs). Les repasser à True pour retrouver l'ancien
      comportement. NB : si tu utilises un cache .npy, VIDE-LE après ce
      changement, sinon load_or_extract relira les anciens spectres.

    Parameters
    ----------
    image_path         : str/Path
    calib              : dict — calibration (spatiale HgAr + spectrale)
    subtract_bg        : bool — soustraire le fond
    bg_method          : 'edges' | 'interfiber' — méthode de fond
    bg_clip            : bool — clipper le fond soustrait à 0 (déconseillé)
    clean_noise        : bool — appliquer baseline removal + edge taper
    show_detection     : bool — afficher le diagnostic de positionnement
    baseline_window_nm : float — fenêtre du morphological opening (nm)
    baseline_n_iter    : int — itérations du baseline removal
    taper_width_nm     : float — largeur de transition aux bords (nm)
    noise_percentile   : float — percentile pour estimer le bruit aux bords
    min_signal_ratio   : float — rapport S/N minimum pour le taper

    Returns
    -------
    spectra     : ndarray (n_fibers, n_cols)
    fiber_y_rot : ndarray (n_fibers,) — Y positions in rotated image
    angle_deg   : float — rotation angle applied
    """
    arr_raw = load_image(image_path)
    angle_deg = _measure_image_rotation(arr_raw.astype(float))
    arr_rot = _rotate_image(arr_raw.astype(float), angle_deg)

    if subtract_bg:
        if bg_method == 'edges':
            arr_rot = subtract_background_edges(arr_rot, calib, clip=bg_clip)
        elif bg_method == 'interfiber':
            arr_rot = subtract_background_columns(arr_rot, calib)
        else:
            raise ValueError(f"bg_method inconnu: {bg_method!r} "
                             "(attendu 'edges' ou 'interfiber')")

    fiber_y_rot, is_real = _detect_fibers_after_rotation(arr_rot, calib)

    if show_detection:
        period = calib.get('fiber_period', _estimate_period(
            fiber_y_rot[~np.isnan(fiber_y_rot)]))
        plot_fiber_detection_diagnostic(
            arr_rot, fiber_y_rot, is_real, period, calib['col_ref'],
            image_name=Path(image_path).name)

    n       = calib['n_fibers']
    hw      = calib['half_width']
    spectra = np.zeros((n, arr_rot.shape[1]))

    for i in range(n):
        if np.isnan(fiber_y_rot[i]):
            continue
        spectra[i] = extract_spectrum_optimal(
            arr_rot,
            y_ref      = fiber_y_rot[i],
            slope      = 0.0,
            col_ref    = calib['col_ref'],
            half_width = hw)

    # ── Nettoyage post-extraction (niveaux 2 + 3) ────────────────────────
    if clean_noise:
        wl_axis = calib.get('wl_axis', None)
        spectra = clean_spectra(
            spectra, wl_axis=wl_axis,
            do_baseline=True,
            do_taper=True,
            baseline_window_nm=baseline_window_nm,
            baseline_n_iter=baseline_n_iter,
            taper_width_nm=taper_width_nm,
            noise_percentile=noise_percentile,
            min_signal_ratio=min_signal_ratio)

    return spectra, fiber_y_rot, angle_deg

 

 

def batch_extract(image_list, calib, output_dir=None, subtract_bg=True,
                  bg_method='edges', bg_clip=False, clean_noise=False,
                  verbose=True):
    """Extrait les spectres de toutes les images de `image_list`."""
    all_spectra = {}
    n = len(image_list)
    for k, img_path in enumerate(image_list):
        img_path = Path(img_path)
        if verbose and (k % 50 == 0 or k == n - 1):
            print(f"  [{k+1}/{n}] {img_path.name}")
        spectra, _, angle = extract_all_spectra(img_path, calib,
                                             subtract_bg=subtract_bg,
                                             bg_method=bg_method,
                                             bg_clip=bg_clip,
                                             clean_noise=clean_noise)
        all_spectra[img_path.stem] = spectra
        if output_dir is not None:
            p = Path(output_dir) / img_path.stem
            np.save(str(p.with_suffix('.npy')), spectra)
    if verbose:
        print(f"  Extraction terminée — {n} images traitées.")
    return all_spectra


def load_or_extract(image_path, calib, cache_dir=None, subtract_bg=True,
                    bg_method='edges', bg_clip=False, clean_noise=False):
    """Charge depuis le cache (.npy) si disponible, sinon extrait et met en cache.

    ATTENTION : le cache est indexé par NOM d'image seulement, pas par les
    options de traitement. Si tu changes bg_method / clean_noise / bg_clip,
    vide cache_dir (ou change de dossier) sinon tu reliras les anciens .npy.
    """
    image_path = Path(image_path)
    if cache_dir is not None:
        cache_file = Path(cache_dir) / (image_path.stem + '.npy')
        if cache_file.exists():
            return np.load(str(cache_file))

    spectra, _, _ = extract_all_spectra(image_path, calib,
                                         subtract_bg=subtract_bg,
                                         bg_method=bg_method,
                                         bg_clip=bg_clip,
                                         clean_noise=clean_noise)

    if cache_dir is not None:
        Path(cache_dir).mkdir(exist_ok=True)
        np.save(str(cache_file), spectra)

    return spectra

 

 

# =============================================================================

# ANALYSE

# =============================================================================

 

def detect_empty_fibers(spectra, sigma_thresh=3.0):

    """Identifie les fibres vides/faibles. Returns bool array (n_fibers,), True = vide."""

    signals = np.array([s.max() - np.percentile(s, 20) for s in spectra])

    med     = np.median(signals)

    mad     = np.median(np.abs(signals - med))

    thresh  = max(med - sigma_thresh * 1.4826 * mad, med * 0.1)

    return signals < thresh

 

 

def compute_snr(spectrum, signal_pct=(70, 99), noise_pct=(5, 25)):

    """SNR d'un spectre : max_signal / std_bruit."""

    sig_region   = np.percentile(spectrum, signal_pct)

    noise_region = np.percentile(spectrum, noise_pct)

    signal_val = sig_region[1] - sig_region[0]

    noise_std  = np.std(spectrum[(spectrum >= noise_region[0]) &

                                  (spectrum <= noise_region[1])])

    return signal_val / noise_std if noise_std > 0 else 0.

 

 

def snr_map(all_spectra):

    """Calcule la carte SNR pour toutes les images et fibres."""

    names = sorted(all_spectra.keys())

    n_img = len(names)

    n_fib = next(iter(all_spectra.values())).shape[0]

    snr   = np.zeros((n_img, n_fib))

    for i, name in enumerate(names):

        for j in range(n_fib):

            snr[i, j] = compute_snr(all_spectra[name][j])

    return snr, names

 

 

# =============================================================================

# CONFIGURATIONS ANGULAIRES (phi, theta) → (x, y, z)

# =============================================================================

 

FIBER_CONFIGS = {

 

    'config3_e': {

        'images': (280, 365),

        'fibres': {

             0: ( 80, -78.52),   1: ( 80, -72.78),   2: ( 80, -67.04),

             3: ( 80, -61.30),   4: ( 80, -55.56),   5: ( 80, -49.82),

             6: ( 50,  -9.64),   7: ( 80,  72.78),   8: ( 90, -67.04),

            9: ( 90, -61.30),  10: ( 90, -55.56),  11: ( 90, -49.82),

            12: ( 80,  84.26),  13: ( 80,  78.52),  14: ( 80,  67.04),

            15: ( 80,  61.30),  16: ( 80,  55.56),  17: ( 80,  49.82),

            18: ( 70,  84.26),  19: ( 70,  78.52),  20: ( 70,  72.78),

            21: ( 70,  67.04),  22: ( 70,  61.30),  23: ( 70,  55.56),

            24: ( 60,  78.52),  25: ( 60,  72.78),  26: ( 60,  67.04),

            27: ( 60,  61.30),  28: ( 60,  55.56),  29: ( 60,  49.82),

            30: ( 60,  44.08),  31: ( 60,  38.34),  32: ( 50,  72.78),

            33: ( 50,  67.04),  34: ( 50,  61.30),  35: ( 50,  55.56),

            36: ( 50,  49.82),  37: ( 50,  44.08),  38: ( 50,  38.34),

            39: ( 50,  32.60),  40: ( 40,  72.78),  41: ( 40,  67.04),

            42: ( 40,  61.30),  43: ( 40,  55.56),  44: ( 40,  49.82),

            45: ( 40,  44.08),  46: ( 40,  38.34),  47: ( 40,  32.60),

            48: ( 30,  67.04),  49: ( 30,  61.30),  50: ( 30,  55.56),

            51: ( 30,  49.82),  52: ( 30,  44.08),  53: ( 30,  38.34),

            54: ( 30,  32.60),  55: ( 30,  26.86),  56: ( 20,  67.04),

            57: ( 20,  61.30),  58: ( 20,  55.56),  59: ( 20,  49.82),

            60: ( 20,  44.08),  61: ( 20,  38.34),  62: ( 20,  32.60),

            63: ( 20,  26.86),  64: ( 10,  61.30),  65: ( 10,  55.56),

            66: ( 10,  49.82),  67: ( 10,  44.08),  68: ( 10,  38.34),

            69: ( 10,  32.60),  70: ( 10,  26.86),  71: ( 10,  21.12),

            72: ( 10,  67.04),  73: (-10,  55.56),  74: (-10,  49.82),

            75: (-10,  44.08),  76: (-10,  38.34),  77: (-10,  32.60),

            78: (-10,  26.86),  79: (-10,  21.12),

        },

    },

 

    'config3_d': {

        'images': (265, 279),

        'fibres': {

             0: ( 80, -78.52),   1: ( 80, -72.78),   2: ( 80, -67.04),

             3: ( 80, -61.30),   4: ( 80, -55.56),   5: ( 80, -49.82),

             6: ( 90,  -78.52),   7: ( 90,  -72.78),   8: ( 90, -67.04),

            9: ( 90, -61.30),  10: ( 90, -55.56),  11: ( 90, -49.82),

            12: ( 80,  84.26),  13: ( 80,  78.52),  14: ( 80,  67.04),

            15: ( 80,  61.30),  16: ( 80,  55.56),  17: ( 80,  49.82),

            18: ( 70,  84.26),  19: ( 70,  78.52),  20: ( 70,  72.78),

            21: ( 70,  67.04),  22: ( 70,  61.30),  23: ( 70,  55.56),

            24: ( 60,  78.52),  25: ( 60,  72.78),  26: ( 60,  67.04),

            27: ( 60,  61.30),  28: ( 60,  55.56),  29: ( 60,  49.82),

            30: ( 60,  44.08),  31: ( 60,  38.34),  32: ( 50,  72.78),

            33: ( 50,  67.04),  34: ( 50,  61.30),  35: ( 50,  55.56),

            36: ( 50,  49.82),  37: ( 50,  44.08),  38: ( 50,  38.34),

            39: ( 50,  32.60),  40: ( 40,  72.78),  41: ( 40,  67.04),

            42: ( 40,  61.30),  43: ( 40,  55.56),  44: ( 40,  49.82),

            45: ( 40,  44.08),  46: ( 40,  38.34),  47: ( 40,  32.60),

            48: ( 30,  67.04),  49: ( 30,  61.30),  50: ( 30,  55.56),

            51: ( 30,  49.82),  52: ( 30,  44.08),  53: ( 30,  38.34),

            54: ( 30,  32.60),  55: ( 30,  26.86),  56: ( 20,  67.04),

            57: ( 20,  61.30),  58: ( 20,  55.56),  59: ( 20,  49.82),

            60: ( 20,  44.08),  61: ( 20,  38.34),  62: ( 20,  32.60),

            63: ( 20,  26.86),  64: ( 10,  61.30),  65: ( 10,  55.56),

            66: ( 10,  49.82),  67: ( 10,  44.08),  68: ( 10,  38.34),

            69: ( 10,  32.60),  70: ( 10,  26.86),  71: ( 10,  21.12),

            72: ( 10,  67.04),  73: (-10,  55.56),  74: (-10,  49.82),

            75: (-10,  44.08),  76: (-10,  38.34),  77: (-10,  32.60),

            78: (-10,  26.86),  79: (-10,  21.12),

        },

    },

 

    'config3_f': {

        'images': (366, 546),

        'fibres': {

             0: ( 80, -78.52),   1: ( 80, -72.78),   2: ( 80, -67.04),

             3: ( 80, -61.30),   4: ( 80, -55.56),   5: ( 80, -49.82),

             6: ( 90,  -78.52),   7: ( 80,  72.78),   8: ( 90, -67.04),

            9: ( 90, -61.30),  10: ( 90, -55.56),  11: ( 90, -49.82),

            12: ( 80,  84.26),  13: ( 80,  78.52),  14: ( 80,  67.04),

            15: ( 80,  61.30),  16: ( 80,  55.56),  17: ( 80,  49.82),

            18: ( 70,  84.26),  19: ( 70,  78.52),  20: ( 70,  72.78),

            21: ( 70,  67.04),  22: ( 70,  61.30),  23: ( 70,  55.56),

            24: ( 60,  78.52),  25: ( 60,  72.78),  26: ( 60,  67.04),

            27: ( 60,  61.30),  28: ( 60,  55.56),  29: ( 60,  49.82),

            30: ( 60,  44.08),  31: ( 60,  38.34),  32: ( 50,  72.78),

            33: ( 50,  67.04),  34: ( 50,  61.30),  35: ( 50,  55.56),

            36: ( 50,  49.82),  37: ( 50,  44.08),  38: ( 50,  38.34),

            39: ( 50,  32.60),  40: ( 40,  72.78),  41: ( 40,  67.04),

            42: ( 40,  61.30),  43: ( 40,  55.56),  44: ( 40,  49.82),

            45: ( 40,  44.08),  46: ( 40,  38.34),  47: ( 40,  32.60),

            48: ( 30,  67.04),  49: ( 30,  61.30),  50: ( 30,  55.56),

            51: ( 30,  49.82),  52: ( 30,  44.08),  53: ( 30,  38.34),

            54: ( 30,  32.60),  55: ( 30,  26.86),  56: ( 20,  67.04),

            57: ( 20,  61.30),  58: ( 20,  55.56),  59: ( 20,  49.82),

            60: ( 20,  44.08),  61: ( 20,  38.34),  62: ( 20,  32.60),

            63: ( 20,  26.86),  64: ( 10,  61.30),  65: ( 10,  55.56),

            66: ( 10,  49.82),  67: ( 10,  44.08),  68: ( 10,  38.34),

            69: ( 10,  32.60),  70: ( 10,  26.86),  71: ( 10,  21.12),

            72: ( 10,  67.04),  73: (-10,  55.56),  74: (-10,  49.82),

            75: (-10,  44.08),  76: (-10,  38.34),  77: (-10,  32.60),

            78: (-10,  26.86),  79: (-10,  21.12),

        },

    },

}

 

 

def get_config_for_shot(shot_number):

    """Retourne le nom de la configuration correspondant à un numéro de shot."""

    for name, cfg in FIBER_CONFIGS.items():

        s0, s1 = cfg['images']

        if s0 <= shot_number <= s1:

            return name

    return None

 

 

def _fiber_angles_to_xyz(phi_stored_deg, theta_stored_deg, roll=0.0):

    """

    Convertit les angles stockés dans FIBER_CONFIGS en coordonnées (x, y, z).

    """

    theta_vals = np.asarray(phi_stored_deg,   dtype=float).copy()

    phi_vals   = np.asarray(theta_stored_deg, dtype=float).copy()

 

    args_min = phi_vals < 0

    phi_vals[args_min]   *= -1

    theta_vals[args_min]  = 90.0 + (90.0 - theta_vals[args_min])

    theta_vals           += -90.0

 

    theta_r = np.radians(theta_vals)

    phi_r   = np.radians(phi_vals)

 

    sin_phi = np.sin(phi_r)

    x =  np.sin(theta_r) * sin_phi

    y =  np.cos(phi_r)

    z =  np.cos(theta_r) * sin_phi

 

    if roll != 0.0:

        cos_r = np.cos(roll);  sin_r = np.sin(roll)

        x, y = cos_r * x - sin_r * y, sin_r * x + cos_r * y

 

    return x, y, z

 

 

def get_fiber_xyz(config_name, roll=0.0):

    """Retourne un array (n_fibers, 3) des positions (x, y, z) sur la sphère unité."""

    if config_name not in FIBER_CONFIGS:

        raise ValueError(f"Configuration '{config_name}' inconnue. "

                         f"Disponibles : {list(FIBER_CONFIGS.keys())}")

    # Angles indexés par indice d'extraction (inversion physique appliquée).
    phis, thetas = get_fiber_angles(config_name)

    xyz = np.full((80, 3), np.nan)

    valid = ~np.isnan(phis)

    xs, ys, zs = _fiber_angles_to_xyz(phis[valid], thetas[valid], roll=roll)

    xyz[valid] = np.column_stack([xs, ys, zs])

    return xyz

 

 

def spectra_to_angular_cube(spectra, config_name, wl_axis, wl_range=None):

    """Assemble les spectres en une valeur par direction (x, y, z)."""

    xyz = get_fiber_xyz(config_name)

 

    if wl_range is not None:

        wl_min, wl_max = wl_range

        mask = (wl_axis >= wl_min) & (wl_axis <= wl_max)

        if mask.sum() == 0:

            raise ValueError(f"Aucun pixel dans la plage {wl_min}–{wl_max} nm")

        values = spectra[:, mask].mean(axis=1)

    else:

        values = spectra.copy()

 

    return xyz, values

 

 

def plot_spectra_3d(
        spectra, config_name, wl_axis,
        wl_range=None, title="",
        elev=25, azim=45, cmap='plasma',
        roll=0.0, n_grid=300, smoothing=1.0,
        coverage_deg=12.0,
        log_scale=False,
        exclude_fibers=None,
        AXLABEL_FONTSIZE=20,
        COLORBAR_LABELSIZE=20,
        COLORBAR_TICKSIZE=13
    ):
    """
    Deux figures angulaires sur la sphère unité : scatter + surface RBF continue,
    avec valeurs numériques supprimées des axes (affichage du label seul).
    Tailles des polices des axes/colorbar personnalisables.

    log_scale : bool (defaut False)
        False -> echelle de couleur lineaire (Normalize) : comportement inchange.
        True  -> echelle de couleur logarithmique (LogNorm). Les valeurs restent
                 en ADU ; seule la correspondance valeur->couleur devient log,
                 et la colorbar affiche des graduations log. Utile quand la
                 dynamique entre fibres est forte (le lineaire ecrase alors les
                 faibles signaux). log(<=0) n'existant pas : les valeurs <= 0
                 sont exclues du calcul des bornes, le plancher de la surface RBF
                 est fixe a vmin (>0) et les points <= 0 sont masques. Si aucune
                 valeur n'est positive, on retombe automatiquement en lineaire.

    exclude_fibers : liste/array d'int ou None (defaut None)
        Indices de fibres a RETIRER completement du plot (scatter ET surface
        RBF). Convention = indice d'EXTRACTION (= l'etiquette affichee a cote de
        chaque point, = la ligne de spectra). Les fibres retirees ne sont ni
        tracees ni utilisees pour l'interpolation : la surface se recalcule sur
        les fibres restantes, et le masque coverage_deg cache les zones devenues
        sans donnees. Typiquement, retirer les fibres qui captent le back-SRS
        pour mieux voir le side-SRS.
        NB : si vous disposez d'une liste de fibres PHYSIQUES, convertissez-les
        en indices d'extraction avec sf.physical_fiber_index(liste) (la fonction
        est sa propre reciproque : phys<->extraction).
    """

    from mpl_toolkits.mplot3d import Axes3D   # noqa
    from scipy.interpolate import RBFInterpolator
    from matplotlib.colors import LogNorm

    # Angles indexés par indice d'extraction (inversion physique appliquée).
    phis, thetas = get_fiber_angles(config_name)

    if wl_range is not None:
        wl_min, wl_max = wl_range
        mask      = (wl_axis >= wl_min) & (wl_axis <= wl_max)
        vals      = spectra[:, mask].mean(axis=1).astype(float)
        val_label = f"Mean intensity {wl_range[0]:.0f}–{wl_range[1]:.0f} nm (ADU)"
    else:
        vals      = spectra.max(axis=1).astype(float)
        val_label = "Max intensity (ADU)"

    valid  = ~np.isnan(phis) & ~np.isnan(vals)

    # Exclusion manuelle de fibres (indices d'extraction).
    if exclude_fibers is not None:
        excl = np.atleast_1d(np.asarray(exclude_fibers, dtype=int)).ravel()
        excl = excl[(excl >= 0) & (excl < len(valid))]
        valid[excl] = False
        n_excl = int(np.isin(np.where(~np.isnan(phis) & ~np.isnan(vals))[0],
                             excl).sum())
        if n_excl:
            print(f"  [plot_spectra_3d] {n_excl} fibre(s) exclue(s) du plot : "
                  f"{sorted(set(excl.tolist()))}")

    xs, ys, zs = _fiber_angles_to_xyz(phis[valid], thetas[valid], roll=roll)
    v    = vals[valid]
    fids = np.where(valid)[0]

    cm   = plt.get_cmap(cmap)
    rbf_floor = 0.0          # plancher pour la surface RBF (0 en lineaire)
    if log_scale:
        v_pos = v[np.isfinite(v) & (v > 0)]
        if v_pos.size >= 2:
            vmin = float(np.nanpercentile(v_pos, 2))
            vmax = float(np.nanpercentile(v_pos, 98))
            if not (vmin > 0):
                vmin = float(v_pos.min())
            if vmax <= vmin:
                vmax = float(v_pos.max()) if v_pos.max() > vmin else vmin * 10.0
            norm = LogNorm(vmin=vmin, vmax=vmax)
            rbf_floor = vmin               # surface clampee a vmin > 0
            val_label = val_label + " (log)"
        else:
            print("  [plot_spectra_3d] log_scale demande mais aucune valeur > 0 "
                  "exploitable -> retour en echelle lineaire.")
            log_scale = False
    if not log_scale:
        vmin = np.nanpercentile(v, 2)
        vmax = np.nanpercentile(v, 98)
        norm = plt.Normalize(vmin=vmin, vmax=vmax)

    # --- Première figure : SCATTER ---
    fig1 = plt.figure(figsize=(10, 8))
    ax1  = fig1.add_subplot(111, projection='3d')
    sc = ax1.scatter(xs, ys, zs, c=v, cmap=cmap, norm=norm,
                     s=80, depthshade=True, edgecolors='w', linewidths=0.4)
    for x, y, z, fid in zip(xs, ys, zs, fids):
        ax1.text(x*1.05, y*1.05, z*1.05, str(fid), fontsize=6, alpha=0.75, ha='center')
    cbar = plt.colorbar(sc, ax=ax1, shrink=0.50, pad=0.08)
    cbar.set_label(val_label, fontsize=COLORBAR_LABELSIZE)
    cbar.ax.tick_params(labelsize=COLORBAR_TICKSIZE)  # taille valeurs colorbar

    ax1.set_xlabel('x', fontsize=AXLABEL_FONTSIZE)
    ax1.set_ylabel('y', fontsize=AXLABEL_FONTSIZE)
    ax1.set_zlabel('z', fontsize=AXLABEL_FONTSIZE)

    # Cacher toutes les valeurs (ticks) des axes x, y, z mais montrer le label :
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax1.set_zticks([])

    ax1.set_box_aspect([1,1,1])
    ax1.view_init(elev=elev, azim=azim)
    ax1.set_title(f"80-fiber scatter — {title or config_name}", fontsize=AXLABEL_FONTSIZE)
    plt.tight_layout()
    plt.show()

    # --- Deuxième figure : RBF SURFACE ---
    pts_all = np.column_stack([xs, ys, zs])
    pts_u, inv = np.unique(pts_all.round(5), axis=0, return_inverse=True)
    v_u = np.array([v[inv == k].mean() for k in range(len(pts_u))])
    rbf = RBFInterpolator(pts_u, v_u, kernel='thin_plate_spline', smoothing=smoothing)

    theta_g = np.linspace(0, np.pi, n_grid)
    phi_g   = np.linspace(0, 2*np.pi, n_grid)
    Tg, Pg  = np.meshgrid(theta_g, phi_g, indexing='ij')
    Xg = np.sin(Tg)*np.cos(Pg); Yg = np.sin(Tg)*np.sin(Pg); Zg = np.cos(Tg)
    grid_pts = np.column_stack([Xg.ravel(), Yg.ravel(), Zg.ravel()])
    Ig = np.clip(rbf(grid_pts).reshape(n_grid, n_grid), rbf_floor, None)

    cos_thresh = np.cos(np.radians(coverage_deg))
    dot_max    = (grid_pts @ pts_u.T).max(axis=1)
    mask_cov   = (dot_max >= cos_thresh).reshape(n_grid, n_grid)
    facecolors = cm(norm(Ig))
    facecolors[~mask_cov, 3] = 0.0

    fig2 = plt.figure(figsize=(11, 9))
    ax2  = fig2.add_subplot(111, projection='3d')
    ax2.plot_surface(Xg, Yg, Zg, facecolors=facecolors,
                     rstride=1, cstride=1, antialiased=False, shade=False)
    ax2.plot_surface(Xg, Yg, Zg, color='lightgrey', alpha=0.06,
                     rstride=4, cstride=4, linewidth=0, antialiased=False, shade=False)
    ax2.scatter(xs, ys, zs, c='k', s=18, zorder=6,
                depthshade=False, edgecolors='w', linewidths=0.5)
    for x, y, z, fid in zip(xs, ys, zs, fids):
        ax2.text(x*1.04, y*1.04, z*1.04, str(fid), fontsize=5.5, alpha=0.7, ha='center')
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])

    cbar2 = plt.colorbar(sm, ax=ax2, shrink=0.50, pad=0.08)
    cbar2.set_label(val_label, fontsize=COLORBAR_LABELSIZE)
    cbar2.ax.tick_params(labelsize=COLORBAR_TICKSIZE)

    ax2.set_xlabel('x', fontsize=AXLABEL_FONTSIZE)
    ax2.set_ylabel('y', fontsize=AXLABEL_FONTSIZE)
    ax2.set_zlabel('z', fontsize=AXLABEL_FONTSIZE)

    ax2.set_xticks([])
    ax2.set_yticks([])
    ax2.set_zticks([])

    ax2.set_box_aspect([1,1,1])
    ax2.view_init(elev=elev, azim=azim)
    ax2.set_title("Maximum intensity as a function of X, Y and Z", fontsize=AXLABEL_FONTSIZE)
    plt.tight_layout()
    plt.show()


def make_gif_3d_rotation(spectra, config_name, wl_axis,
                          gif_path="rbf_rotation.gif",
                          wl_range=None,
                          elev=25,
                          n_frames=72,
                          fps=20,
                          dpi=90,
                          cmap='plasma',
                          roll=0.0,
                          n_grid=150,
                          smoothing=1.0,
                          coverage_deg=12.0,
                          title="",
                          verbose=True):
    """
    Generate a rotating GIF of the RBF surface (interpolated 3D angular map).

    The azimuth angle sweeps 360° over `n_frames` frames, keeping `elev` fixed.
    Only the RBF surface plot is animated (not the scatter plot).

    Parameters
    ----------
    spectra      : ndarray (n_fibers, n_cols)
    config_name  : str — key in FIBER_CONFIGS
    wl_axis      : ndarray (n_cols,) in nm
    gif_path     : str — output file path (e.g. "rotation.gif")
    wl_range     : (wl_min, wl_max) or None
    elev         : float — elevation angle kept fixed (deg)
    n_frames     : int   — number of frames for a full 360° rotation
    fps          : int   — frames per second in the output GIF
    dpi          : int   — resolution per frame (lower = faster/smaller file)
    n_grid       : int   — RBF sphere grid resolution (lower = faster)
    smoothing    : float — RBF smoothing
    coverage_deg : float — mask radius around data points (deg)
    title        : str   — figure title
    verbose      : bool  — print progress

    Returns
    -------
    gif_path : str — path to the saved GIF
    """
    from scipy.interpolate import RBFInterpolator
    from PIL import Image as _PILImage
    import io, tempfile

    # Angles indexés par indice d'extraction (inversion physique appliquée).
    phis, thetas = get_fiber_angles(config_name)

    if wl_range is not None:
        mask  = (wl_axis >= wl_range[0]) & (wl_axis <= wl_range[1])
        vals  = spectra[:, mask].mean(axis=1).astype(float)
        vlbl  = f"Mean intensity {wl_range[0]:.0f}–{wl_range[1]:.0f} nm (ADU)"
    else:
        vals  = spectra.max(axis=1).astype(float)
        vlbl  = "Max intensity (ADU)"

    valid = ~np.isnan(phis) & ~np.isnan(vals)
    xs, ys, zs = _fiber_angles_to_xyz(phis[valid], thetas[valid], roll=roll)
    v = vals[valid]

    vmin = np.nanpercentile(v, 2)
    vmax = np.nanpercentile(v, 98)
    norm = plt.Normalize(vmin=vmin, vmax=vmax)
    cm   = plt.get_cmap(cmap)

    # ── Build RBF surface once (expensive) ───────────────────────────────────
    if verbose:
        print(f"  Building RBF surface ({n_grid}×{n_grid} grid)…")

    pts_all = np.column_stack([xs, ys, zs])
    pts_u, inv = np.unique(pts_all.round(5), axis=0, return_inverse=True)
    v_u = np.array([v[inv == k].mean() for k in range(len(pts_u))])
    rbf = RBFInterpolator(pts_u, v_u, kernel='thin_plate_spline', smoothing=smoothing)

    theta_g = np.linspace(0, np.pi,    n_grid)
    phi_g   = np.linspace(0, 2*np.pi,  n_grid)
    Tg, Pg  = np.meshgrid(theta_g, phi_g, indexing='ij')
    Xg = np.sin(Tg)*np.cos(Pg)
    Yg = np.sin(Tg)*np.sin(Pg)
    Zg = np.cos(Tg)

    grid_pts  = np.column_stack([Xg.ravel(), Yg.ravel(), Zg.ravel()])
    Ig        = np.clip(rbf(grid_pts).reshape(n_grid, n_grid), 0, None)

    cos_thresh = np.cos(np.radians(coverage_deg))
    dot_max    = (grid_pts @ pts_u.T).max(axis=1)
    mask_cov   = (dot_max >= cos_thresh).reshape(n_grid, n_grid)

    facecolors = cm(norm(Ig))
    facecolors[~mask_cov, 3] = 0.0

    # ── Render frames ─────────────────────────────────────────────────────────
    azimuths = np.linspace(0, 360, n_frames, endpoint=False)
    frames   = []

    if verbose:
        print(f"  Rendering {n_frames} frames…")

    for k, az in enumerate(azimuths):
        if verbose and (k % max(1, n_frames // 10) == 0):
            print(f"    Frame {k+1}/{n_frames}  azim={az:.1f}°")

        fig = plt.figure(figsize=(8, 7))
        ax  = fig.add_subplot(111, projection='3d')

        ax.plot_surface(Xg, Yg, Zg, facecolors=facecolors,
                        rstride=1, cstride=1, antialiased=False, shade=False)
        ax.plot_surface(Xg, Yg, Zg, color='lightgrey', alpha=0.06,
                        rstride=4, cstride=4, linewidth=0,
                        antialiased=False, shade=False)
        ax.scatter(xs, ys, zs, c='k', s=14, zorder=6,
                   depthshade=False, edgecolors='w', linewidths=0.4)

        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        plt.colorbar(sm, ax=ax, shrink=0.45, pad=0.08, label=vlbl)

        ax.set_xlabel('x', fontsize=8)
        ax.set_ylabel('y', fontsize=8)
        ax.set_zlabel('z', fontsize=8)
        ax.set_box_aspect([1, 1, 1])
        ax.view_init(elev=elev, azim=az)
        ax.set_title(f"RBF surface — {title or config_name}  (azim={az:.0f}°)",
                     fontsize=10)
        plt.tight_layout()

        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=dpi, bbox_inches='tight')
        plt.close(fig)
        buf.seek(0)
        frames.append(_PILImage.open(buf).copy())

    # ── Save GIF ──────────────────────────────────────────────────────────────
    duration_ms = int(1000 / fps)
    frames[0].save(
        gif_path,
        save_all=True,
        append_images=frames[1:],
        duration=duration_ms,
        loop=0,
    )
    if verbose:
        import os
        size_kb = os.path.getsize(gif_path) / 1024
        print(f"  GIF saved → {gif_path}  ({size_kb:.0f} KB, "
              f"{n_frames} frames @ {fps} fps)")
    return gif_path


# =============================================================================

# CARTES ANGULAIRES 2D (θ vs φ)

# =============================================================================

 

def _interp_angular_2d(phis, thetas, values, n_grid=200, smoothing=2.0):

    """Interpolation RBF 2D dans l'espace (theta, phi)."""

    from scipy.interpolate import RBFInterpolator

 

    valid = ~np.isnan(phis) & ~np.isnan(values)

    pts = np.column_stack([thetas[valid], phis[valid]])

    v   = values[valid]

    pts_u, inv = np.unique(pts.round(4), axis=0, return_inverse=True)

    v_u = np.array([v[inv == k].mean() for k in range(len(pts_u))])

    rbf = RBFInterpolator(pts_u, v_u, kernel='thin_plate_spline', smoothing=smoothing)

    t_lin = np.linspace(thetas[valid].min() - 2, thetas[valid].max() + 2, n_grid)

    p_lin = np.linspace(phis[valid].min()   - 2, phis[valid].max()   + 2, n_grid)

    TT, PP = np.meshgrid(t_lin, p_lin)

    ZZ = rbf(np.column_stack([TT.ravel(), PP.ravel()])).reshape(n_grid, n_grid)

    return TT, PP, ZZ

 

 

def plot_angular_metric(phis, thetas, values, label="Value",
                        title="", cmap='plasma',
                        n_grid=200, smoothing=2.0,
                        theta_shift=0.0,
                        coverage_deg=None):
    """
    Two angular maps (θ on x, φ on y): scatter + RBF interpolation.

    Parameters
    ----------
    theta_shift  : float — offset added to all θ values before plotting (degrees)
    coverage_deg : float or None — if given, the RBF map is masked at grid
                   points farther than coverage_deg from any data point
                   (prevents unphysical interpolation in large empty regions)
    """
    valid = ~np.isnan(phis) & ~np.isnan(values)

    # Apply theta shift
    thetas_shifted = np.asarray(thetas, dtype=float) + theta_shift

    th_v  = thetas_shifted[valid];  ph_v = phis[valid];  val_v = values[valid]

    fids  = np.where(valid)[0]

    vmin = np.nanpercentile(val_v, 2)

    vmax = np.nanpercentile(val_v, 98)

 

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

 

    sc1 = ax1.scatter(th_v, ph_v, c=val_v, cmap=cmap,

                      vmin=vmin, vmax=vmax,

                      s=90, edgecolors='k', linewidths=0.3)

    plt.colorbar(sc1, ax=ax1, label=label)

    for j in range(len(fids)):

        ax1.annotate(str(fids[j]), (th_v[j], ph_v[j]),

                     fontsize=5, ha='center', va='center',

                     color='white', fontweight='bold')

    ax1.set_xlabel('θ (°)', fontsize=11); ax1.set_ylabel('φ (°)', fontsize=11)

    ax1.set_title(f"Scatter — {title}", fontsize=11)

    ax1.grid(True, alpha=0.25)

 

    TT, PP, ZZ = _interp_angular_2d(phis, thetas_shifted, values,
                                     n_grid=n_grid, smoothing=smoothing)

    ZZ = np.clip(ZZ, vmin, vmax)

    # Coverage mask: hide interpolated points far from any data point
    if coverage_deg is not None:
        pts_data = np.column_stack([th_v, ph_v])
        grid_pts = np.column_stack([TT.ravel(), PP.ravel()])
        from scipy.spatial import cKDTree
        tree = cKDTree(pts_data)
        dist, _ = tree.query(grid_pts, k=1)
        mask_cov = dist.reshape(TT.shape) > coverage_deg
        ZZ_plot = np.ma.masked_where(mask_cov, ZZ)
    else:
        ZZ_plot = ZZ

    im = ax2.pcolormesh(TT, PP, ZZ_plot, cmap=cmap, vmin=vmin, vmax=vmax, shading='auto')

    plt.colorbar(im, ax=ax2, label=label)

    ax2.scatter(th_v, ph_v, c='k', s=25, edgecolors='w', linewidths=0.4, zorder=5)

    ax2.set_xlabel('θ (°)', fontsize=11); ax2.set_ylabel('φ (°)', fontsize=11)

    ax2.set_title(f"RBF interpolation — {title}", fontsize=11)

    ax2.grid(True, alpha=0.20)

 

    plt.suptitle(title, fontsize=13, fontweight='bold')

    plt.tight_layout()

    plt.show()

 

 

# =============================================================================

# FONCTIONS DE VISUALISATION

# =============================================================================

 

import matplotlib.pyplot as plt

import matplotlib.gridspec as gridspec

 

CMAP_IMAGE   = 'hot'

CMAP_SPECTRA = 'plasma'

FIG_DPI      = 110

plt.rcParams['figure.dpi'] = FIG_DPI

 

 

def _wl_or_px(calib, use_wl=True):

    """Retourne l'axe X (en nm ou en pixels) et le label associé."""

    if use_wl and calib is not None:

        return calib['wl_axis'], "Wavelength (nm)"

    return np.arange(calib['image_shape'][1] if calib else 2560), "Spectral pixel"

 

 

def plot_wl_calibration(calib, arr_calib=None):
    """Spectral calibration visualisation: pixel -> wavelength mapping."""
    wl_axis   = calib['wl_axis']   # physical wavelength axis
    coeffs    = calib['wl_coeffs']
    pairs     = calib['wl_pairs']
    residuals = calib['wl_residuals']
    n_cols    = len(wl_axis)
    pixels_k  = np.array([p[0] for p in pairs])
    wl_k      = np.array([p[1] for p in pairs])
    px_arr    = np.arange(n_cols, dtype=float)

    def _wl_to_px(w):  return np.interp(w, wl_axis, px_arr)
    def _px_to_wl(px): return np.interp(px, px_arr, wl_axis)

    if 'mean_calib_spectrum' in calib:
        xproj = calib['mean_calib_spectrum'].copy()
    elif arr_calib is not None:
        xproj = gaussian_filter1d(arr_calib.mean(axis=0), 1.5)
    else:
        xproj = None
    if xproj is not None and xproj.max() > 0:
        xproj = xproj / xproj.max()

    fig = plt.figure(figsize=(18, 10))
    gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.55, wspace=0.35)

    # Panel 1: HgAr spectrum + catalogue lines
    ax_sp = fig.add_subplot(gs[0, :2])
    if xproj is not None:
        ax_sp.plot(wl_axis, xproj, color='steelblue', lw=0.9,
                   label='Mean spectrum (along traces)')
        ax_sp.fill_between(wl_axis, xproj, alpha=0.15, color='steelblue')
    for px, wl in pairs:
        sym = HGAR_LINES.get(wl, '?')
        ax_sp.axvline(wl, color='red', lw=1.2, alpha=0.9)
        ax_sp.text(wl, 1.03, f"{wl:.1f} nm\n(px {px})\n({sym})",
                   ha='center', fontsize=6.5, color='red',
                   rotation=90, va='bottom')
    for wl_cat in HGAR_LINES:
        if wl_axis[0] < wl_cat < wl_axis[-1]:
            ax_sp.axvline(wl_cat, color='gray', lw=0.4, alpha=0.35, ls='--')
    ax_sp.set_xlim(wl_axis[0], wl_axis[-1])
    ax_sp.set_ylim(-0.05, 1.45)
    ax_sp.set_xlabel("Wavelength (nm)", fontsize=10)
    ax_sp.set_ylabel("Normalised intensity")
    ax_sp.set_title("HgAr spectrum — calibration lines (red) | full catalogue (grey)")
    ax_sp.legend(fontsize=8)
    ax_px_top = ax_sp.secondary_xaxis('top', functions=(_wl_to_px, _px_to_wl))
    ax_px_top.set_xlabel("Pixel", fontsize=9, labelpad=6)
    ax_px_top.tick_params(axis='x', labelsize=8)

    # Panel 2: dispersion curve pixel -> nm
    ax_disp = fig.add_subplot(gs[0, 2])
    ax_disp.plot(px_arr, wl_axis, 'steelblue', lw=1.5,
                 label=f'Polynomial deg {WL_POLY_DEG}')
    ax_disp.scatter(pixels_k, wl_k, color='red', zorder=5, s=50,
                    label='Calibration pairs')
    for px, wl in pairs:
        ax_disp.annotate(f"{wl:.1f}", (px, wl),
                         textcoords="offset points", xytext=(4, 2),
                         fontsize=6, color='red')
    ax_disp.set_xlabel("Pixel")
    ax_disp.set_ylabel("Wavelength (nm)")
    ax_disp.set_title("Dispersion curve (pixel -> nm)")
    ax_disp.legend(fontsize=8)
    ax_disp.grid(True, alpha=0.3)

    # Panel 3: polynomial fit residuals
    ax_res = fig.add_subplot(gs[1, :2])
    cols_r = ['tomato' if r > 0 else 'steelblue' for r in residuals]
    ax_res.bar(wl_k, residuals, width=2.0, color=cols_r, alpha=0.8)
    ax_res.axhline(0, color='k', lw=0.8)
    rms = np.sqrt(np.mean(residuals ** 2))
    ax_res.axhline( rms, color='orange', ls='--', lw=1.2, label=f'+-RMS = {rms:.3f} nm')
    ax_res.axhline(-rms, color='orange', ls='--', lw=1.2)
    for (px, wl), r in zip(pairs, residuals):
        sym = HGAR_LINES.get(wl, '?')
        ax_res.text(wl, r + 0.03 * np.sign(r) if r != 0 else 0.03,
                    f"{wl:.1f} nm\npx {px}\n({sym})",
                    ha='center', fontsize=6.5,
                    va='bottom' if r >= 0 else 'top')
    ax_res.set_xlabel("Wavelength (nm)")
    ax_res.set_ylabel("Residual (nm)")
    ax_res.set_title(f"Polynomial fit residuals -- RMS = {rms:.3f} nm")
    ax_res.legend(fontsize=9)
    ax_res.grid(True, alpha=0.3)
    ax_res_px = ax_res.secondary_xaxis('top', functions=(_wl_to_px, _px_to_wl))
    ax_res_px.set_xlabel("Pixel", fontsize=8, labelpad=4)
    ax_res_px.tick_params(axis='x', labelsize=7)

    # Panel 4: summary table
    ax_tab = fig.add_subplot(gs[1, 2])
    ax_tab.axis('off')
    rows = [[f"{p[0]}", f"{p[1]:.2f}", HGAR_LINES.get(p[1], '?'),
             f"{r:+.3f}", "OK" if abs(r) < 0.5 else "~" if abs(r) < 1.0 else "X"]
            for p, r in zip(pairs, residuals)]
    table = ax_tab.table(
        cellText=rows,
        colLabels=["Pixel", "lambda cat (nm)", "Elem.", "Residual (nm)", "Quality"],
        loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.15, 1.5)
    for i, (_, r) in enumerate(zip(pairs, residuals)):
        color = '#d4edda' if abs(r) < 0.5 else '#fff3cd' if abs(r) < 1.0 else '#f8d7da'
        for j in range(5):
            table[(i + 1, j)].set_facecolor(color)
    ax_tab.set_title("Calibration pairs  (green<0.5nm | orange<1nm | red>=1nm)",
                     pad=12, fontsize=8)

    plt.suptitle("SPECTRAL CALIBRATION -- HgAr", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()


def plot_spatial_calibration(calib, calib_path):
    """
    Spatial calibration visualisation.

    Displays the STRAIGHTENED calibration image (rotated by calib_angle_deg)
    so that the detected fiber traces are perfectly consistent with what is
    shown — no spurious crossings between fibres.
    """
    angle_deg = calib.get('calib_angle_deg', 0.0)
    arr = _rotate_image(load_image(calib_path).astype(float), angle_deg)

    xs     = np.arange(arr.shape[1])
    vmax_c = np.percentile(arr, 99.5)
    cr     = calib['col_ref']
    n_real = int(calib['fiber_is_real'].sum())

    fig = plt.figure(figsize=(20, 13))
    gs  = gridspec.GridSpec(3, 3, figure=fig, hspace=0.4, wspace=0.35)

    ax_img = fig.add_subplot(gs[0:2, 0:2])
    ax_img.imshow(arr, cmap=CMAP_IMAGE, aspect='auto', vmin=0, vmax=vmax_c)
    for y0, sl, real in zip(calib['fiber_y_ref'], calib['tilt_slopes'],
                             calib['fiber_is_real']):
        ys = y0 + sl * (xs - cr)
        ax_img.plot(xs, ys, '-', color='cyan' if real else 'lime', lw=0.5, alpha=0.8)
    ax_img.set_title(
        f"Calibration image (straightened {angle_deg:+.4f}°) — {calib['n_fibers']} traces\n"
        f"cyan = {n_real} detected  |  lime = {calib['n_fibers'] - n_real} interpolated"
    )
    ax_img.set_xlabel("X (px)"); ax_img.set_ylabel("Y (px)")

    ax_prof = fig.add_subplot(gs[0:2, 2])
    prof = column_profile(arr, cr)

    bg   = uniform_filter1d(gaussian_filter1d(median_filter(prof, 3), 1.), BG_FILTER_SIZE)

    yy   = np.arange(len(prof))

    ax_prof.plot(prof, yy, 'steelblue', lw=0.8, label='profile')

    ax_prof.plot(bg,   yy, 'orange', lw=1.2, ls='--', label='local background')

    for y0, real in zip(calib['fiber_y_ref'], calib['fiber_is_real']):

        ax_prof.axhline(y0, color='cyan' if real else 'lime', lw=0.6, alpha=0.8)

    ax_prof.invert_yaxis()

    ax_prof.set_xlabel("Intensity (ADU)"); ax_prof.set_ylabel("Y (px)")

    ax_prof.set_title(f"Y profile — col. {cr}"); ax_prof.legend(fontsize=8)

 

    ax_pos = fig.add_subplot(gs[2, 0])

    # Fiber Y positions at x=0 (ref col) AND x=x_max
    # If both curves overlap → fibers are perfectly horizontal
    x_max   = arr.shape[1] - 1
    y_at_x0 = [calib['fiber_y_ref'][i] + calib['tilt_slopes'][i] * (0 - cr)
               for i in range(calib['n_fibers'])]
    y_at_xm = [calib['fiber_y_ref'][i] + calib['tilt_slopes'][i] * (x_max - cr)
               for i in range(calib['n_fibers'])]
    ax_pos.plot(y_at_x0, np.arange(calib['n_fibers']),
               'o-', ms=3, color='steelblue', label='x=0')
    ax_pos.plot(y_at_xm, np.arange(calib['n_fibers']),
               's--', ms=3, color='tomato', alpha=0.8, label=f'x={x_max}')
    ax_pos.set_xlabel("Y position (px)"); ax_pos.set_ylabel("Fiber index")
    ax_pos.set_title("Fiber positions — x=0 vs x=x_max\n(overlap = horizontal fibers)")
    ax_pos.legend(fontsize=8); ax_pos.grid(True, alpha=0.3)

 

    ax_sp = fig.add_subplot(gs[2, 1])

    sp = np.diff(calib['fiber_y_ref'])

    ax_sp.plot(sp, 'o-', ms=3, color='tomato')

    ax_sp.axhline(np.median(sp), ls='--', color='k', label=f"médiane = {np.median(sp):.1f} px")

    ax_sp.set_xlabel("Index"); ax_sp.set_ylabel("Spacing (px)")

    ax_sp.set_title("Inter-fiber spacing"); ax_sp.legend(fontsize=9); ax_sp.grid(True, alpha=0.3)

 

    ax_tilt = fig.add_subplot(gs[2, 2])

    ax_tilt.plot(calib['tilt_slopes'], np.arange(calib['n_fibers']), 'o-', ms=3, color='mediumpurple')

    ax_tilt.axvline(0, ls='--', color='k', lw=0.8)

    ax_tilt.set_xlabel("Slope dY/dX (px/px)"); ax_tilt.set_ylabel("Fiber index")

    ax_tilt.set_title("Tilt per fiber"); ax_tilt.grid(True, alpha=0.3)

 

    plt.suptitle("SPATIAL CALIBRATION", fontsize=14, fontweight='bold')

    plt.show()

 

 

def plot_image_overview(image_path, calib=None):

    """Vue d'ensemble d'une image science brute."""

    arr  = load_image(image_path)

    name = Path(image_path).name

 

    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    axes[0].imshow(arr, cmap=CMAP_IMAGE, aspect='auto',

                   vmin=0, vmax=np.percentile(arr, 99.5))

    if calib is not None:

        xs = np.arange(arr.shape[1]); cr = calib['col_ref']

        for y0, sl in zip(calib['fiber_y_ref'], calib['tilt_slopes']):

            ys = y0 + sl * (xs - cr)

            axes[0].plot(xs, ys, 'cyan', lw=0.3, alpha=0.5)

    axes[0].set_title(f"Raw image — {name}")

    axes[0].set_xlabel("X (px)"); axes[0].set_ylabel("Y (px)")

 

    vals = arr.flatten()

    axes[1].hist(vals[vals > np.percentile(vals, 1)], bins=300, color='steelblue', log=True)

    axes[1].axvline(np.percentile(arr, 5),  color='orange', ls='--', label='fond (p5)')

    axes[1].axvline(np.percentile(arr, 99), color='red',    ls='--', label='signal (p99)')

    axes[1].set_xlabel("Intensity (ADU)"); axes[1].set_ylabel("Pixel count")

    axes[1].set_title("Intensity distribution"); axes[1].legend(fontsize=9)

 

    x_ax = calib['wl_axis'] if calib is not None else np.arange(arr.shape[1])

    xlabel = "Wavelength (nm)" if calib is not None else "Spectral pixel"

    axes[2].plot(x_ax, arr.mean(axis=0), lw=0.8, color='steelblue', label="X projection (spectral)")

    ax2b = axes[2].twinx()

    ax2b.plot(arr.mean(axis=1), color='tomato', lw=0.8, label="Y projection (fibers)")

    axes[2].set_xlabel(xlabel); axes[2].set_ylabel("Mean intensity", color='steelblue')

    ax2b.set_ylabel("Mean intensity", color='tomato')

    axes[2].set_title("X and Y projections")

    axes[2].legend(loc='upper left', fontsize=8); ax2b.legend(loc='upper right', fontsize=8)

 

    plt.suptitle(f"IMAGE — {name}", fontsize=13, fontweight='bold')

    plt.show()

 

 

def plot_all_spectra(spectra, calib, image_name="", use_wl=True):

    """Vue d'ensemble de tous les spectres d'une image."""

    x_ax, xlabel = _wl_or_px(calib, use_wl)

    empty = detect_empty_fibers(spectra)

    vpos  = spectra[spectra > 0]

    vmax  = np.percentile(vpos, 99.5) if len(vpos) > 0 else 1.

 

    fig, axes = plt.subplots(1, 2, figsize=(20, 7))

    im = axes[0].imshow(spectra, aspect='auto', cmap=CMAP_IMAGE, vmin=0, vmax=vmax,

                        extent=[x_ax[0], x_ax[-1], spectra.shape[0], 0])

    for idx in np.where(empty)[0]:

        axes[0].axhline(idx, color='cyan', lw=0.5, alpha=0.7)

    plt.colorbar(im, ax=axes[0], label='Intensity (ADU)')

    if use_wl:

        for wl_cat in HGAR_LINES:

            if x_ax[0] < wl_cat < x_ax[-1]:

                axes[0].axvline(wl_cat, color='lime', lw=0.4, alpha=0.3)

    axes[0].set_title(f"{image_name} — tous les spectres\n"

                      f"(cyan = fibres vides : {list(np.where(empty)[0])})")

    axes[0].set_xlabel(xlabel); axes[0].set_ylabel("Fiber index")

 

    n     = spectra.shape[0]

    step  = max(1, n // 10)

    cols  = plt.cm.plasma(np.linspace(0, 0.9, 10))

    scale = spectra.max() if spectra.max() > 0 else 1.

    for j, idx in enumerate(range(0, n, step)):

        if j >= 10: break

        sp  = spectra[idx] / scale

        col = 'lightgray' if empty[idx] else cols[j]

        axes[1].plot(x_ax, sp + j * 0.09, lw=0.8, color=col, label=f"Fiber {idx}")

    axes[1].set_xlabel(xlabel); axes[1].set_ylabel("Normalised intensity + offset")

    axes[1].set_title("Spectrum cascade"); axes[1].legend(fontsize=8, ncol=2)

 

    plt.suptitle(f"OVERVIEW — {image_name}", fontsize=13, fontweight='bold')

    plt.show()

 

 

def plot_fiber_inspection(spectra, calib, fiber_idx, image_path,

                          fiber_y_rot=None, angle_deg=None, use_wl=True):

    """Inspection détaillée d'une fibre sur une image science."""

    arr_raw = load_image(image_path)

    name    = Path(image_path).name

 

    if fiber_y_rot is None or angle_deg is None:

        _, fiber_y_rot, angle_deg = extract_all_spectra(image_path, calib, subtract_bg=True)

 

    arr_rot = _rotate_image(arr_raw, angle_deg)

    x_ax, xlabel = _wl_or_px(calib, use_wl)

    y_ref  = fiber_y_rot[fiber_idx]

    cr     = calib['col_ref']

    sp     = spectra[fiber_idx]

    empty  = detect_empty_fibers(spectra)[fiber_idx]

    xs     = np.arange(arr_rot.shape[1])

    vmax_s = np.percentile(arr_rot, 99.5)

    hw     = calib['half_width']

 

    fig = plt.figure(figsize=(20, 12))

    gs  = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.35)

 

    ax_sci = fig.add_subplot(gs[0:2, 0:2])

    ax_sci.imshow(arr_rot, cmap=CMAP_IMAGE, aspect='auto', vmin=0, vmax=vmax_s)

    for y0_ in fiber_y_rot:

        if not np.isnan(y0_):

            ax_sci.axhline(y0_, color='gray', lw=0.3, alpha=0.3)

    if not np.isnan(y_ref):

        ax_sci.axhline(y_ref, color='cyan', lw=2, label=f'Fibre {fiber_idx}  (y={y_ref:.1f} px)')

        ax_sci.axhspan(y_ref - hw, y_ref + hw, alpha=0.25, color='cyan', label=f'±{hw} px')

    ax_sci.set_title(f"{name} — rotated image ({angle_deg:.4f}°)\nFiber {fiber_idx}"

                     + (" [EMPTY]" if empty else ""))

    ax_sci.set_xlabel("X (px)"); ax_sci.set_ylabel("Y (px)"); ax_sci.legend(fontsize=9)

 

    ax_zoom = fig.add_subplot(gs[0:2, 2])

    yc_int  = int(round(y_ref)) if not np.isnan(y_ref) else arr_rot.shape[0]//2

    y0z     = max(0, yc_int - 40); y1z = min(arr_rot.shape[0], yc_int + 40)

    ax_zoom.imshow(arr_rot[y0z:y1z, :], cmap=CMAP_IMAGE, aspect='auto',

                   vmin=0, vmax=vmax_s, extent=[0, arr_rot.shape[1], y1z, y0z])

    if not np.isnan(y_ref):

        ax_zoom.axhline(y_ref, color='cyan', lw=2)

        ax_zoom.axhspan(y_ref - hw, y_ref + hw, alpha=0.25, color='cyan')

    ax_zoom.set_title(f"Zoom — fiber {fiber_idx} (±40 px)")

    ax_zoom.set_xlabel("X (px)"); ax_zoom.set_ylabel("Y (px)")

 

    ax_sp = fig.add_subplot(gs[2, 0:2])

    ax_sp.plot(x_ax, sp, color='steelblue', lw=0.9)

    ax_sp.fill_between(x_ax, sp, alpha=0.2, color='steelblue')

    if use_wl:

        for wl_cat, sym in HGAR_LINES.items():

            if x_ax[0] < wl_cat < x_ax[-1]:

                ax_sp.axvline(wl_cat, color='red', lw=0.6, alpha=0.4)

                ax_sp.text(wl_cat, sp.max() * 1.01, f"{wl_cat:.0f}", fontsize=6,

                           rotation=90, ha='center', color='red', va='bottom')

    ax_sp.set_xlabel(xlabel); ax_sp.set_ylabel("Intensity (ADU)")

    ax_sp.set_title(f"Spectrum — fiber {fiber_idx}"

                    + (" [EMPTY]" if empty else f"  |  max={sp.max():.0f} ADU"))

    ax_sp.grid(True, alpha=0.3)

 

    ax_cut = fig.add_subplot(gs[2, 2])

    yy     = np.arange(max(0, yc_int-30), min(arr_rot.shape[0], yc_int+30))

    cut    = arr_rot[yy, cr]

    ax_cut.plot(cut, yy, color='tomato', lw=1.)

    if not np.isnan(y_ref):

        ax_cut.axhline(y_ref, color='cyan', ls='--', lw=1.5, label=f"y={y_ref:.1f}")

        ax_cut.axhspan(y_ref - hw, y_ref + hw, alpha=0.2, color='cyan')

    ax_cut.invert_yaxis()

    ax_cut.set_xlabel("Intensity (ADU)"); ax_cut.set_ylabel("Y (px)")

    ax_cut.set_title(f"Cross-section\nx = {cr} (ref. col.) — rotated image")

    ax_cut.legend(fontsize=8); ax_cut.grid(True, alpha=0.3)

 

    flag = "⚠ EMPTY FIBER" if empty else "✓ Active fiber"

    plt.suptitle(f"INSPECTION — Fiber {fiber_idx}  |  {flag}  |  {name}",

                 fontsize=13, fontweight='bold')

    plt.show()

 

    snr = compute_snr(sp)

    print(f"\n── Résumé fibre {fiber_idx} ──────────────────────────────")

    print(f"  Image            : {name}")

    print(f"  Rotation angle   : {angle_deg:.4f} deg")

    print(f"  Y position (rot) : {y_ref:.2f} px  (horizontal)")

    print(f"  Max intensity    : {sp.max():.1f} ADU")

    print(f"  Median intensity : {np.median(sp):.1f} ADU")

    print(f"  Estimated SNR    : {snr:.1f}")

    print(f"  Status           : {'EMPTY' if empty else 'OK'}")

    print(f"  Spectral range   : {x_ax[0]:.1f} – {x_ax[-1]:.1f} nm")

 

 

def plot_fiber_comparison(spectra, calib, fiber_indices, image_name="",

                          use_wl=True, normalize=False):

    """Compare plusieurs fibres sur une même image (cascade verticale)."""

    x_ax, xlabel = _wl_or_px(calib, use_wl)

    empty  = detect_empty_fibers(spectra)

    n_comp = len(fiber_indices)

    colors = plt.cm.plasma(np.linspace(0, 0.9, n_comp))

 

    fig, axes = plt.subplots(n_comp, 1, figsize=(18, 2.3 * n_comp), sharex=True)

    if n_comp == 1:

        axes = [axes]

 

    for ax, idx, col in zip(axes, fiber_indices, colors):

        sp  = spectra[idx]

        if normalize and sp.max() > 0:

            sp = sp / sp.max()

        c   = 'lightgray' if empty[idx] else col

        ax.plot(x_ax, sp, color=c, lw=0.9)

        ax.fill_between(x_ax, sp, alpha=0.15, color=c)

        snr = compute_snr(spectra[idx])

        lab = f"Fiber {idx}  |  max={spectra[idx].max():.0f} ADU  |  SNR≈{snr:.0f}"

        if empty[idx]: lab += "  [VIDE]"

        ax.set_ylabel(f"F{idx}", fontsize=9); ax.set_title(lab, fontsize=9, pad=2)

        ax.grid(True, alpha=0.2)

        if use_wl:

            for wl_cat in HGAR_LINES:

                if x_ax[0] < wl_cat < x_ax[-1]:

                    ax.axvline(wl_cat, color='red', lw=0.4, alpha=0.25)

 

    axes[-1].set_xlabel(xlabel)

    plt.suptitle(f"SPECTRUM COMPARISON — {image_name}", fontsize=13, fontweight='bold')

    plt.show()

 

 

def plot_fiber_across_images(all_spectra, calib, fiber_idx,

                             image_names=None, use_wl=True, n_show=12):

    """Évolution d'une même fibre sur plusieurs images (waterfall temporel)."""

    x_ax, xlabel = _wl_or_px(calib, use_wl)

    names = image_names if image_names else sorted(all_spectra.keys())

    names = names[:n_show]; n = len(names)

 

    fig, axes = plt.subplots(1, 2, figsize=(20, max(6, n * 0.5 + 2)))

    mat = np.array([all_spectra[nm][fiber_idx] for nm in names])

    vpos = mat[mat > 0]; vmax = np.percentile(vpos, 99.5) if len(vpos) > 0 else 1.

    axes[0].imshow(mat, aspect='auto', cmap=CMAP_IMAGE, vmin=0, vmax=vmax,

                   extent=[x_ax[0], x_ax[-1], len(names), 0])

    axes[0].set_yticks(np.arange(n) + 0.5); axes[0].set_yticklabels(names, fontsize=7)

    axes[0].set_xlabel(xlabel); axes[0].set_title(f"Fiber {fiber_idx} — {n} images (heatmap)")

 

    scale  = mat.max() if mat.max() > 0 else 1.

    colors = plt.cm.viridis(np.linspace(0, 0.9, n))

    for j, (nm, col) in enumerate(zip(names, colors)):

        sp = all_spectra[nm][fiber_idx] / scale

        axes[1].plot(x_ax, sp + j * 0.06, lw=0.7, color=col, label=nm)

    axes[1].set_xlabel(xlabel); axes[1].set_ylabel("Intensité norm. + offset")

    axes[1].set_title(f"Fiber {fiber_idx} — temporal cascade")

    if n <= 20: axes[1].legend(fontsize=6, ncol=2)

 

    plt.suptitle(f"TEMPORAL EVOLUTION — Fiber {fiber_idx}", fontsize=13, fontweight='bold')

    plt.show()

 

 

def plot_snr_map(all_spectra, title="Carte SNR"):

    """Affiche la carte SNR (images × fibres)."""

    snr, names = snr_map(all_spectra)

    fig, axes  = plt.subplots(1, 2, figsize=(18, 6))

    im = axes[0].imshow(snr, aspect='auto', cmap='viridis')

    plt.colorbar(im, ax=axes[0], label='SNR')

    axes[0].set_xlabel("Fiber index"); axes[0].set_ylabel("Image")

    if len(names) <= 30:

        axes[0].set_yticks(np.arange(len(names))); axes[0].set_yticklabels(names, fontsize=7)

    axes[0].set_title("SNR map (image × fiber)")

 

    snr_mean = snr.mean(axis=0); snr_std = snr.std(axis=0)

    axes[1].plot(snr_mean, label='SNR moyen')

    axes[1].fill_between(np.arange(len(snr_mean)),

                         snr_mean - snr_std, snr_mean + snr_std,

                         alpha=0.25, label='±1σ')

    axes[1].set_xlabel("Fiber index"); axes[1].set_ylabel("SNR")

    axes[1].set_title("Mean SNR per fiber (all images)")

    axes[1].legend(); axes[1].grid(True, alpha=0.3)

 

    plt.suptitle(title, fontsize=13, fontweight='bold')

    plt.show()


# =============================================================================
# PROFILS TEMPORELS DES PULSES & CALCUL DE PUISSANCE
# =============================================================================

def load_pulse_profile(csv_path, header_lines=25, frac_rise=0.1, frac_fall=0.15):
    """
    Charge le profil temporel d'un pulse depuis un fichier CSV oscilloscope.

    Format attendu :
      - Lignes 1 à `header_lines` : en-tête (métadonnées, ignorées)
      - Lignes suivantes : valeurs séparées par des virgules
        col 0 = temps (s), col 1 = intensité (V ou u.a.)

    Fenêtrage asymétrique du pulse :
      - Front montant : premier passage au-dessus de frac_rise × Imax
      - Front descendant : dernier passage au-dessus de frac_fall × Imax

    Returns
    -------
    dict avec clés :
        'time_full', 'intensity_full'   — données complètes
        'time_pulse', 'intensity_pulse' — données fenêtrées
        'idx_start', 'idx_end'          — indices de la fenêtre
        'dt'                            — pas de temps moyen (s)
    """
    csv_path = Path(csv_path)
    times, intensities = [], []

    with open(csv_path, 'r', errors='replace') as f:
        for i, line in enumerate(f):
            if i < header_lines:
                continue
            line = line.strip()
            if not line:
                continue
            parts = line.split(',')
            if len(parts) < 2:
                continue
            try:
                t = float(parts[0].strip())
                v = float(parts[2].strip())
                times.append(t)
                intensities.append(v)
            except ValueError:
                continue

    if len(times) == 0:
        raise ValueError(f"Aucune donnée numérique trouvée dans {csv_path}")

    time_full = np.array(times)
    intensity_full = np.array(intensities)

    # ── Fenêtrage asymétrique du pulse ──────────────────────────────────
    i_max = np.max(intensity_full)
    i_peak_idx = np.argmax(intensity_full)

    # Front montant : premier index où I >= frac_rise × Imax
    thresh_rise = frac_rise * i_max
    rising_mask = intensity_full[:i_peak_idx + 1] >= thresh_rise
    idx_start = int(np.argmax(rising_mask)) if np.any(rising_mask) else 0

    # Front descendant : dernier index où I >= frac_fall × Imax
    thresh_fall = frac_fall * i_max
    falling_mask = intensity_full[i_peak_idx:] >= thresh_fall
    idx_end = (i_peak_idx + int(np.max(np.where(falling_mask)))
               if np.any(falling_mask) else len(intensity_full) - 1)

    idx_start = max(0, idx_start)
    idx_end = min(len(intensity_full) - 1, idx_end)

    time_pulse = time_full[idx_start:idx_end + 1]
    intensity_pulse = intensity_full[idx_start:idx_end + 1]
    dt = float(np.mean(np.diff(time_full))) if len(time_full) > 1 else 1.0

    return {
        'time_full': time_full,
        'intensity_full': intensity_full,
        'time_pulse': time_pulse,
        'intensity_pulse': intensity_pulse,
        'idx_start': idx_start,
        'idx_end': idx_end,
        'dt': dt,
    }


def load_energy_table(xlsx_path):
    """
    Charge la table d'énergie 2ω depuis Final.xlsx.

    Colonnes attendues (Excel 1-indexed) :
      C (index 2) : numéro de shot   (à partir de la ligne 2)
      F (index 5) : énergie 2ω en J  (à partir de la ligne 2)

    Returns
    -------
    dict {shot_number (int): energy_2omega_J (float)}
    """
    import openpyxl
    xlsx_path = Path(xlsx_path)
    wb = openpyxl.load_workbook(xlsx_path, data_only=True, read_only=True)
    ws = wb.active

    energy_table = {}
    for row in ws.iter_rows(min_row=2, values_only=True):
        if len(row) < 6:
            continue
        shot_val, e2w_val = row[2], row[5]
        if shot_val is None or e2w_val is None:
            continue
        try:
            energy_table[int(shot_val)] = float(e2w_val)
        except (ValueError, TypeError):
            continue

    wb.close()
    return energy_table


def compute_pulse_power(pulse_data, energy_2omega_J):
    """
    Puissance crête du pulse.

    Formule :  P = (Imax / (Σ I_pulse × dt)) × E_2ω × 0.92

    Le terme Imax/(Σ×dt) est un facteur de forme (crête/moyenne temporelle).
    Multiplié par l'énergie et le facteur 0.92, cela donne une puissance (W).
    """
    I_pulse = pulse_data['intensity_pulse']
    dt      = pulse_data['dt']
    I_max   = np.max(I_pulse)
    area    = np.sum(I_pulse) * dt

    if area == 0 or np.isnan(area):
        return np.nan

    return (I_max / area) * energy_2omega_J * 0.92