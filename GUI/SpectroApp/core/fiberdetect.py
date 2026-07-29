"""Detection automatique des fibres — algorithme final multi-couches.

Couches :
  1. Bande eclairee + periode (mode des ecarts entre pics forts).
  2. Candidats sur-complets : maxima d'intensite (2 lissages) U maxima de
     courbure (epaulements) U maxima du profil de votes.
  3. Votes = persistance multi-canaux : pics detectes independamment sur
     chaque colonne de raie HgAr (sous-pixel) ET sur chaque image science
     redressee fournie (les fibres eteintes sur la lampe sont souvent
     brillantes en tir).
  4. Selection exacte de n_fibers candidats par programmation dynamique
     (espacement contraint a [0.70, 1.30] x periode, somme des poids max).
  5. Precision : barycentre guide par canal (fenetre +-hw, base lineaire par
     les bords, porte SNR), mediane robuste sur les canaux ; pour les fibres
     sans canal valide (enfouies), apex de courbure dans la fenetre bornee
     par les mi-distances aux voisines ; sinon position de grille.
Sortie : positions + confiance par fibre + diagnostics pour les plots.
"""
import numpy as np
from scipy.ndimage import gaussian_filter1d, median_filter, percentile_filter
from scipy.signal import argrelextrema, find_peaks


def _parab(prof, i):
    y0, y1, y2 = prof[i - 1], prof[i], prof[i + 1]
    d = y0 - 2 * y1 + y2
    return i + (float(np.clip(0.5 * (y0 - y2) / d, -1, 1))
                if abs(d) > 1e-12 else 0.0)


def band_envelope(profile, period_guess=19.0):
    env = percentile_filter(profile, 95, size=int(3 * period_guess) | 1)
    dark = np.percentile(env, 8)
    margin = env[env <= np.percentile(env, 15)]
    noise = 1.4826 * np.median(np.abs(margin - np.median(margin))) + 1e-9
    lit = np.where(env > dark + 8 * noise)[0]
    return (int(lit[0]), int(lit[-1])) if len(lit) else (0, len(profile) - 1)


def estimate_period(profile, band):
    sm = gaussian_filter1d(profile, 2.0)
    pk, props = find_peaks(sm[band[0]:band[1]], prominence=0)
    if len(pk) < 10:
        return 19.0
    order = np.argsort(props["prominences"])[::-1][:max(20, len(pk) // 3)]
    gaps = np.diff(np.sort(pk[order]))
    gaps = gaps[(gaps >= 10) & (gaps <= 35)]
    return float(np.median(gaps)) if len(gaps) else 19.0


def _votes_from_profile(p, period, weight, votes, subpix=True):
    yy = np.arange(len(votes))
    sm = gaussian_filter1d(p, 1.5)
    base = median_filter(sm, int(2.6 * period) | 1)
    r = sm - base
    mad = 1.4826 * np.median(np.abs(r - np.median(r))) + 1e-9
    pk, _ = find_peaks(r, height=2 * mad, distance=int(period * 0.45))
    for y in pk:
        y2 = _parab(sm, int(y)) if subpix else float(y)
        votes += weight * np.exp(-0.5 * ((yy - y2) / 1.6) ** 2)
    curv = -np.gradient(np.gradient(gaussian_filter1d(p, 2.2)))
    cmad = 1.4826 * np.median(np.abs(curv)) + 1e-9
    pk2, _ = find_peaks(curv, height=2 * cmad, distance=int(period * 0.45))
    for y in pk2:
        votes += 0.6 * weight * np.exp(-0.5 * ((yy - float(y)) / 1.6) ** 2)


def build_votes(arr_hgar, line_cols, science_profiles, period):
    votes = np.zeros(arr_hgar.shape[0])
    for c in line_cols:
        p = arr_hgar[:, max(0, c - 3):c + 4].sum(axis=1)
        _votes_from_profile(p, period, 1.0, votes)
    for p in science_profiles:
        _votes_from_profile(p, period, 0.8, votes)
    return votes


def candidates(profile, votes, band, sigmas=(1.5, 2.5)):
    lo, hi = band
    pos = []
    for s in sigmas:
        sm = gaussian_filter1d(profile, s)
        pos += list(argrelextrema(sm, np.greater_equal, order=3)[0])
        curv = -np.gradient(np.gradient(gaussian_filter1d(profile, s * 1.3)))
        pos += list(argrelextrema(curv, np.greater_equal, order=3)[0])
    vm = argrelextrema(votes, np.greater_equal, order=2)[0]
    pos += list(vm[votes[vm] > 0.4])
    pos = np.unique(np.asarray(pos, float))
    return pos[(pos >= lo - 3) & (pos <= hi + 3)]


def dp_select(pos, weights, period, n_fibers, gmin_f=0.70, gmax_f=1.30):
    gmin, gmax = gmin_f * period, gmax_f * period
    n = len(pos)
    NEG = -1e18
    dp = np.full((n, n_fibers), NEG)
    parent = np.full((n, n_fibers), -1, dtype=int)
    dp[:, 0] = weights
    for c in range(1, n_fibers):
        prev = dp[:, c - 1]
        for i in range(n):
            js = np.where((pos >= pos[i] - gmax) & (pos <= pos[i] - gmin))[0]
            if len(js) == 0:
                continue
            j = js[np.argmax(prev[js])]
            if prev[j] <= NEG / 2:
                continue
            dp[i, c] = prev[j] + weights[i]
            parent[i, c] = j
    end = int(np.argmax(dp[:, n_fibers - 1]))
    if dp[end, n_fibers - 1] <= NEG / 2:
        raise RuntimeError(
            "Impossible d'assembler une chaine de fibres compatible : "
            "verifiez l'image de calibration (bande, nombre de fibres).")
    path = [end]
    for c in range(n_fibers - 1, 0, -1):
        path.append(parent[path[-1], c])
    return pos[np.array(path[::-1])]


def guided_centroid(channels, pos, hw=7, snr_gate=3.0, iters=2):
    """Barycentre guide multi-canaux (version validee RMS 0.94)."""
    pos = np.asarray(pos, float).copy()
    n_used = np.zeros(len(pos), int)
    disp = np.zeros(len(pos))
    noise = [np.std(np.diff(p)) / np.sqrt(2) for p in channels]
    for _ in range(iters):
        new = pos.copy()
        for i, y in enumerate(pos):
            ests = []
            for p, nz in zip(channels, noise):
                i0, i1 = int(round(y)) - hw, int(round(y)) + hw + 1
                if i0 < 1 or i1 > len(p) - 1:
                    continue
                xw = np.arange(i0, i1, dtype=float)
                w = p[i0:i1]
                xa, ya = xw[[0, 1, -2, -1]], w[[0, 1, -2, -1]]
                coef = np.linalg.lstsq(
                    np.vstack([xa, np.ones_like(xa)]).T, ya, rcond=None)[0]
                r = np.clip(w - (coef[0] * xw + coef[1]), 0, None)
                if r.sum() <= 0 or r.max() < snr_gate * nz:
                    continue
                est = (xw * r).sum() / r.sum()
                if abs(est - y) <= 3.5:
                    ests.append(est)
            if len(ests) >= 3:
                new[i] = np.median(ests)
                n_used[i] = len(ests)
                disp[i] = 1.4826 * np.median(np.abs(np.array(ests) - new[i]))
            else:
                n_used[i] = len(ests)
        pos = new
    return pos, n_used, disp

def flank_gauss_refine(channels, pos, idx, period, sig=2.0):
    """Fibres enfouies : modele 2 flancs exponentiels + gaussienne, mediane
    multi-canaux. Les flancs log-lineaires des voisines brillantes sont
    captures explicitement au lieu de biaiser une ligne de base."""
    from scipy.optimize import curve_fit
    out = pos.copy()
    used = {}
    for i in idx:
        muL = pos[i - 1] if i > 0 else pos[i] - period
        muR = pos[i + 1] if i < len(pos) - 1 else pos[i] + period
        y0 = pos[i]
        ests = []
        for p in channels:
            a, b = int(muL) + 1, int(muR)
            if a < 0 or b > len(p) or b - a < 11:
                continue
            xs = np.arange(a, b, dtype=float)
            w = p[a:b]
            x0 = xs.mean()

            def model(x, aL, kL, aR, kR, A, mu, c):
                return (aL * np.exp(-kL * (x - x0))
                        + aR * np.exp(kR * (x - x0))
                        + A * np.exp(-0.5 * ((x - mu) / sig) ** 2) + c)
            wmin = w.min()
            noise = np.std(np.diff(p)) / np.sqrt(2) + 1e-9
            try:
                popt, _ = curve_fit(
                    model, xs, w,
                    p0=[max(w[0] - wmin, 1), 0.15, max(w[-1] - wmin, 1), 0.15,
                        max((w[len(w) // 2] - wmin) * 0.5, 1), y0, wmin],
                    bounds=([0, 0.01, 0, 0.01, 0, y0 - 5, -np.inf],
                            [np.inf, 1.5, np.inf, 1.5, np.inf, y0 + 5, np.inf]),
                    maxfev=20000)
            except Exception:
                continue
            if popt[4] < 2 * noise:
                continue
            ests.append(popt[5])
        if len(ests) >= 3:
            out[i] = float(np.median(ests))
            used[i] = len(ests)
    return out, used


def shoulder_refine(profile, pos, idx, period):
    """Fibres enfouies : apex de courbure dans la fenetre bornee voisines."""
    out = pos.copy()
    used = []
    curv = -np.gradient(np.gradient(gaussian_filter1d(profile, 2.0)))
    for i in idx:
        y = pos[i]
        lo = (pos[i - 1] + y) / 2 + 1.5 if i > 0 else y - period / 2
        hi = (pos[i + 1] + y) / 2 - 1.5 if i < len(pos) - 1 else y + period / 2
        i0, i1 = int(np.ceil(lo)), int(np.floor(hi)) + 1
        if i1 - i0 < 5 or i0 < 1 or i1 > len(curv) - 1:
            continue
        j = i0 + int(np.argmax(curv[i0:i1]))
        if j <= i0 or j >= i1 - 1:
            continue
        est = _parab(curv, j)
        if abs(est - y) <= 0.45 * period:
            out[i] = est
            used.append(i)
    return out, used


def detect_fibers_auto(arr_hgar, n_fibers=80, science_arrays=None,
                       find_calib_columns=None):
    """arr_hgar et science_arrays : images DEJA redressees.
    Retourne (positions, confiances, diagnostics)."""
    if find_calib_columns is None:
        from core import spectro_functions as _sf
        find_calib_columns = _sf.find_calib_columns
    line_cols = find_calib_columns(arr_hgar)
    colmask = np.zeros(arr_hgar.shape[1], bool)
    for c in line_cols:
        colmask[max(0, c - 3):c + 4] = True
    profile = arr_hgar[:, colmask].sum(axis=1)
    band = band_envelope(profile)
    period = estimate_period(profile, band)
    science_profiles = [a.sum(axis=1) for a in (science_arrays or [])]
    votes = build_votes(arr_hgar, line_cols, science_profiles, period)
    cand = candidates(profile, votes, band)
    sm = gaussian_filter1d(profile, 1.5)
    imax = argrelextrema(sm, np.greater_equal, order=3)[0]
    base = median_filter(sm, int(2.6 * period) | 1)
    contrast = np.clip((sm - base) / (np.abs(base) + 1e-9), 0, 1)
    w = (3.0 * np.array([votes[max(0, int(c) - 1):int(c) + 2].max()
                         for c in cand])
         + 1.0 * np.isin(cand.astype(int), imax)
         + 0.5 * contrast[cand.astype(int)])
    sel = dp_select(cand, w, period, n_fibers)
    # canaux de precision : colonnes HgAr + profils science
    channels = [gaussian_filter1d(
        arr_hgar[:, max(0, c - 3):c + 4].sum(axis=1), 1.0)
        for c in line_cols]
    channels += [gaussian_filter1d(p, 1.0) for p in science_profiles]
    pos, n_used, disp = guided_centroid(channels, sel)
    weak = [i for i in range(n_fibers) if n_used[i] < 3]
    pos, fg_used = flank_gauss_refine(channels, pos, weak, period)
    still_weak = [i for i in weak if i not in fg_used]
    pos, shoulder_ok = shoulder_refine(profile, pos, still_weak, period)
    order = np.argsort(pos)
    pos = pos[order]
    conf = np.empty(n_fibers, dtype=object)
    n_used, disp = n_used[order], disp[order]
    inv = {int(o): k for k, o in enumerate(order)}
    for i in range(n_fibers):
        if n_used[i] >= 3:
            conf[i] = f"directe ({n_used[i]} canaux, ±{disp[i]:.2f}px)"
        elif any(inv[w] == i for w in fg_used):
            w0 = [w for w in fg_used if inv[w] == i][0]
            conf[i] = f"enfouie (fit flancs, {fg_used[w0]} canaux)"
        elif any(inv[s] == i for s in shoulder_ok):
            conf[i] = "epaulement (courbure)"
        else:
            conf[i] = "grille (interpolee)"
    diag = dict(profile=profile, votes=votes, band=band, period=period,
                candidates=cand, selection=sel, n_used=n_used, disp=disp,
                line_cols=list(map(int, line_cols)),
                n_science=len(science_profiles))
    return pos, conf, diag


