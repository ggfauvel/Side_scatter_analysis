"""
core/plotting.py — Figures Plotly.

Principe : chaque figure est construite a partir des MEMES tableaux numpy que
la figure matplotlib correspondante du notebook (memes fonctions sf, memes
percentiles 2-98 pour les echelles de couleur, meme interpolation RBF, meme
masque de couverture). Seul le moteur de rendu change (interactif au lieu de
statique). Les apparences (polices, palettes par defaut) different donc de
matplotlib ; les DONNEES tracees sont identiques.
"""
from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from core import spectro_functions as sf

TEMPLATE = "plotly_white"
FIBER_DISPLAY_OFFSET = 1   # les fibres sont AFFICHÉES 1–80 (internes : 0–79)


def apply_prefs(fig):
    """Applies the global display preferences (font size, theme) set by
    the user. Called by components.graph()."""
    try:
        from core.session import SESSION
        fig.update_layout(
            template=SESSION.params.get("PLOT_TEMPLATE", TEMPLATE),
            font=dict(size=int(SESSION.params.get("PLOT_FONT_SIZE", 13))))
    except Exception:
        pass
    return fig
CMAP_IMAGE = "Hot"       # equivalent Plotly du cmap 'hot' du notebook
CMAP_METRIC = "Plasma"   # equivalent du cmap 'plasma'


def _fix_si_prefix(fig, ylab):
    """Empeche la double prefixation SI sur les axes d'energie.

    Plotly ecrit spontanement "600µ" pour 6e-4. Sur un axe DEJA intitule
    « µJ/nm », cela se lit « 600 µJ/nm » alors que la valeur vaut 0,0006 :
    six ordres de grandeur d'erreur de lecture, et rien dans le graphique ne
    le signale. On force donc la notation scientifique sur ces axes.
    """
    t = str(ylab)
    if "µJ" in t or "J/nm" in t or "J·nm" in t:
        fig.update_yaxes(exponentformat="e", showexponent="all",
                         tickformat=".3g")
    return fig


def _base_layout(fig, title="", xlab="", ylab="", height=450):
    fig.update_layout(template=TEMPLATE, title=title, height=height,
                      margin=dict(l=60, r=30, t=60, b=50),
                      xaxis_title=xlab, yaxis_title=ylab)
    return _fix_si_prefix(fig, ylab)


def empty_fig(msg="No data"):
    fig = go.Figure()
    fig.add_annotation(text=msg, showarrow=False, font=dict(size=14, color="grey"))
    fig.update_layout(template=TEMPLATE, height=300,
                      xaxis_visible=False, yaxis_visible=False)
    return fig


# ── Images detecteur ─────────────────────────────────────────────────────────
def fig_image(arr_small, factor=1, fiber_y=None, title="", log=True, height=520):
    """Vue detecteur (sous-echantillonnee pour l'affichage uniquement)."""
    z = np.log10(np.clip(arr_small, 1, None)) if log else arr_small
    fig = go.Figure(go.Heatmap(
        z=z, colorscale=CMAP_IMAGE,
        colorbar=dict(title="log10(ADU)" if log else "ADU"),
        y=np.arange(arr_small.shape[0]) * factor,
        x=np.arange(arr_small.shape[1]) * factor,
        hovertemplate="x=%{x}px  y=%{y}px  val=%{z:.2f}<extra></extra>"))
    if fiber_y is not None:
        ys = np.asarray(fiber_y, float)
        ys = ys[np.isfinite(ys)]
        fig.add_trace(go.Scatter(
            x=np.full_like(ys, 8.0), y=ys, mode="markers",
            marker=dict(symbol="triangle-right", size=7, color="cyan"),
            name="fibers", hovertemplate="fiber y=%{y:.1f}px<extra></extra>"))
    fig.update_yaxes(autorange="reversed", title="Y (px)")
    fig.update_xaxes(title="X (px)")
    fig.update_layout(template=TEMPLATE, title=title, height=height,
                      margin=dict(l=60, r=30, t=50, b=50))
    return fig


# ── Calibration spectrale ────────────────────────────────────────────────────
def fig_calib_spectrum(calib, height=430):
    wl = calib["wl_axis"]
    spec = calib.get("mean_calib_spectrum")
    fig = go.Figure()
    if spec is not None and len(spec) == len(wl):
        fig.add_trace(go.Scatter(x=wl, y=spec, mode="lines",
                                 line=dict(width=1, color="#333"),
                                 name="spectre HgAr moyen"))
        ymax = float(np.nanmax(spec))
    else:
        ymax = 1.0
    for px, wl_cat in calib.get("wl_pairs", []):
        elem = sf.HGAR_LINES.get(wl_cat, "?")
        fig.add_vline(x=wl_cat, line=dict(color="seagreen", width=1, dash="dot"))
        fig.add_annotation(x=wl_cat, y=ymax, text=f"{wl_cat:.1f} {elem}",
                           showarrow=False, textangle=-90, yshift=10,
                           font=dict(size=9, color="seagreen"))
    return _base_layout(fig, "Spectre de calibration HgAr et raies appariees",
                        "Wavelength (nm)", "Intensity (ADU)", height)


def fig_calib_residuals(calib, height=350):
    pairs = calib.get("wl_pairs", [])
    res = calib.get("wl_residuals", np.array([]))
    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=("Residus d'appariement",
                                        "Courbe de dispersion pixel -> nm"))
    if len(pairs) and len(res) == len(pairs):
        wls = [p[1] for p in pairs]
        fig.add_trace(go.Scatter(x=wls, y=res, mode="markers",
                                 marker=dict(size=8, color="steelblue"),
                                 hovertemplate="%{x:.2f} nm : %{y:+.3f} nm<extra></extra>",
                                 showlegend=False), row=1, col=1)
        fig.add_hline(y=0, line=dict(color="grey", width=1), row=1, col=1)
        rms = float(np.sqrt(np.mean(np.asarray(res) ** 2)))
        fig.add_annotation(text=f"RMS = {rms:.3f} nm", xref="x domain",
                           yref="y domain", x=0.02, y=0.95, showarrow=False,
                           row=1, col=1)
    wl = calib["wl_axis"]
    fig.add_trace(go.Scatter(x=np.arange(len(wl)), y=wl, mode="lines",
                             line=dict(color="#333", width=1.2),
                             showlegend=False), row=1, col=2)
    if len(pairs):
        fig.add_trace(go.Scatter(x=[p[0] for p in pairs], y=[p[1] for p in pairs],
                                 mode="markers",
                                 marker=dict(size=7, color="seagreen"),
                                 showlegend=False), row=1, col=2)
    fig.update_xaxes(title_text="lambda (nm)", row=1, col=1)
    fig.update_yaxes(title_text="residu (nm)", row=1, col=1)
    fig.update_xaxes(title_text="pixel", row=1, col=2)
    fig.update_yaxes(title_text="lambda (nm)", row=1, col=2)
    fig.update_layout(template=TEMPLATE, height=height,
                      margin=dict(l=60, r=30, t=50, b=50))
    return fig


def fig_intensity_calib(factors, wl_axis, fibers=None, height=400):
    fig = go.Figure()
    if fibers is None:
        fibers = list(range(0, factors.shape[0], 10))
    for i in fibers:
        fig.add_trace(go.Scatter(x=wl_axis, y=factors[i], mode="lines",
                                 line=dict(width=1), name=f"fiber {i}"))
    return _base_layout(fig, "Facteurs de calibration d'intensite relative",
                        "Wavelength (nm)", "Factor", height)


# ── Spectres ─────────────────────────────────────────────────────────────────
def fig_all_spectra_heatmap(spectra, x_axis, xlab, title="", height=520):
    v2, v98 = np.nanpercentile(spectra, [2, 98])
    fig = go.Figure(go.Heatmap(
        z=spectra, x=x_axis,
        y=np.arange(spectra.shape[0]) + FIBER_DISPLAY_OFFSET,
        colorscale=CMAP_METRIC, zmin=v2, zmax=v98,
        colorbar=dict(title="ADU"),
        hovertemplate=xlab + "=%{x:.1f}  fiber=%{y}  I=%{z:.0f}<extra></extra>"))
    fig.update_yaxes(title="Fiber (1–80, detector order)")
    fig.update_xaxes(title=xlab)
    fig.update_layout(template=TEMPLATE, title=title, height=height,
                      margin=dict(l=60, r=30, t=50, b=50))
    return fig


def fig_fiber_lines(spectra, fibers, x_axis, xlab, normalize=False,
                    title="", colors=None, height=450):
    fig = go.Figure()
    for k, i in enumerate(fibers):
        y = np.asarray(spectra[int(i)], float)
        if normalize:
            m = np.nanmax(np.abs(y))
            y = y / m if m else y
        kw = {}
        if colors:
            kw["line"] = dict(width=1.1, color=colors[k % len(colors)])
        else:
            kw["line"] = dict(width=1.1)
        fig.add_trace(go.Scatter(x=x_axis, y=y, mode="lines",
                                 name=f"Fiber {int(i) + FIBER_DISPLAY_OFFSET}",
                                 **kw))
    ylab = "Normalised intensity" if normalize else "Intensity (ADU)"
    return _base_layout(fig, title, xlab, ylab, height)


def fig_fiber_across_images(spectra_by_image: dict, fiber_idx, x_axis, xlab,
                            height=450):
    fig = go.Figure()
    names = list(spectra_by_image)
    for k, name in enumerate(names):
        c = 0.05 + 0.9 * k / max(1, len(names) - 1)
        fig.add_trace(go.Scatter(
            x=x_axis, y=spectra_by_image[name][int(fiber_idx)], mode="lines",
            line=dict(width=1.0), name=name,
            marker_color=None, opacity=0.9,
            legendgroup=name))
        fig.data[-1].line.color = f"rgba({int(30+200*c)},{int(60*(1-c)+30)},{int(180*(1-c)+40)},0.9)"
    return _base_layout(fig,
                        f"Fiber {int(fiber_idx) + FIBER_DISPLAY_OFFSET} across images",
                        xlab, "Intensity (ADU)", height)


def fig_snr(snr_arr, names, height=520):
    fig = go.Figure(go.Heatmap(
        z=snr_arr, x=np.arange(snr_arr.shape[1]) + FIBER_DISPLAY_OFFSET,
        y=names,
        colorscale="Viridis", colorbar=dict(title="SNR"),
        hovertemplate="fiber=%{x}  %{y}  SNR=%{z:.1f}<extra></extra>"))
    fig.update_xaxes(title="Fiber (1–80)")
    fig.update_yaxes(title="Image", autorange="reversed",
                     tickmode="auto", nticks=20)
    fig.update_layout(template=TEMPLATE, title="SNR map", height=height,
                      margin=dict(l=90, r=30, t=50, b=50))
    return fig


# ── Groupes ──────────────────────────────────────────────────────────────────
def fig_groups_individual(x_axis, xlab, groups_data, title="", alpha=0.35,
                          height=470):
    """groups_data : liste de dicts {label, color, spectra: [1D...]}"""
    fig = go.Figure()
    for g in groups_data:
        for i, spec in enumerate(g["spectra"]):
            fig.add_trace(go.Scatter(
                x=x_axis, y=spec, mode="lines",
                line=dict(width=1.0, color=g["color"]),
                opacity=alpha, legendgroup=g["label"],
                name=g["label"], showlegend=(i == 0)))
    return _base_layout(fig, title, xlab, "Intensity (ADU)", height)


def _hex_to_rgba(hex_color, alpha):
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


def fig_groups_mean_std(x_axis, xlab, groups_stats, title="",
                        std_alpha=0.25, height=470):
    """groups_stats : liste de dicts {label, color, mean, dev}"""
    fig = go.Figure()
    for g in groups_stats:
        if g["mean"] is None:
            continue
        m, d = g["mean"], g["dev"]
        fig.add_trace(go.Scatter(x=x_axis, y=m + d, mode="lines",
                                 line=dict(width=0), showlegend=False,
                                 legendgroup=g["label"], hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=x_axis, y=m - d, mode="lines",
                                 line=dict(width=0), fill="tonexty",
                                 fillcolor=_hex_to_rgba(g["color"], std_alpha),
                                 showlegend=False, legendgroup=g["label"],
                                 hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=x_axis, y=m, mode="lines",
                                 line=dict(width=2.4, color=g["color"]),
                                 name=g["label"], legendgroup=g["label"]))
    return _base_layout(fig, title, xlab, "Intensity (ADU)", height)


def fig_groups_3d(x_plot, groups_stats, std_alpha=0.18, overlay_title="",
                  height=650, z_title="Intensity (ADU)"):
    """Surfaces 3D moyenne (+/- std en enveloppes) par groupe — cellule 11bis-3D.
    Avantage Plotly : rotation en direct, plus besoin d'elev/azim fixes."""
    fig = go.Figure()
    for g in groups_stats:
        if g["mean"] is None:
            continue
        mean, std, color, label = g["mean"], g["dev"], g["color"], g["label"]
        n_fib = mean.shape[0]
        fibs = np.arange(n_fib) + FIBER_DISPLAY_OFFSET
        scale = [[0, color], [1, color]]
        fig.add_trace(go.Surface(
            x=x_plot, y=fibs, z=mean, colorscale=scale, showscale=False,
            opacity=0.75, name=label, showlegend=True,
            hovertemplate=("lambda=%{x:.1f}  fiber=%{y}  "
                           "z=%{z:.4g}"
                           f"<br>{label}<extra></extra>")))
        for zz in (mean + std, mean - std):
            fig.add_trace(go.Surface(
                x=x_plot, y=fibs, z=zz, colorscale=scale, showscale=False,
                opacity=std_alpha, showlegend=False, hoverinfo="skip"))
    fig.update_layout(
        template=TEMPLATE, height=height, title=overlay_title,
        scene=dict(xaxis_title="Wavelength (nm)",
                   yaxis_title="Fiber (1–80)",
                   zaxis_title=z_title),
        margin=dict(l=10, r=10, t=50, b=10))
    return fig


def fig_area_map_si_profile(results: dict, profile_order, missing=None,
                            smoothing=0.0, height=470):
    """Cellule 12-MAP : scatter annote + carte RBF (coordonnees normalisees
    /15 et /2, thin_plate_spline, smoothing=0), memes calculs."""
    from scipy.interpolate import RBFInterpolator
    prof_to_idx = {p: i for i, p in enumerate(profile_order)}
    si_vals = np.array([k[0] for k in results], float)
    prof_idx = np.array([prof_to_idx[k[1]] for k in results], float)
    areas = np.array(list(results.values()), float)

    fig = make_subplots(rows=1, cols=2, horizontal_spacing=0.14,
                        subplot_titles=("Measured points", "RBF interpolation"))
    fig.add_trace(go.Scatter(
        x=si_vals, y=prof_idx, mode="markers+text",
        text=[f"{a:.2e}" for a in areas], textposition="top center",
        textfont=dict(size=9),
        marker=dict(size=26, color=areas, colorscale=CMAP_METRIC,
                    line=dict(color="black", width=1.2),
                    colorbar=dict(title="Area (ADU)", x=0.42)),
        showlegend=False,
        hovertemplate="Si=%{x}%  profile=%{y}  area=%{marker.color:.3e}<extra></extra>"),
        row=1, col=1)
    if missing:
        for si, prof in missing:
            fig.add_trace(go.Scatter(
                x=[si], y=[prof_to_idx[prof]], mode="markers+text",
                text=["missing"], textposition="top center",
                textfont=dict(size=9, color="red"),
                marker=dict(symbol="x", size=14, color="red",
                            line=dict(width=2)),
                showlegend=False, hoverinfo="skip"), row=1, col=1)

    if len(areas) >= 3:
        si_n = si_vals / 15.0
        pr_n = prof_idx / 2.0
        rbf = RBFInterpolator(np.column_stack([si_n, pr_n]), areas,
                              kernel="thin_plate_spline", smoothing=smoothing)
        gx = np.linspace(0, 15, 200)
        gy = np.linspace(0, len(profile_order) - 1, 200)
        GX, GY = np.meshgrid(gx, gy)
        Z = rbf(np.column_stack([GX.ravel() / 15.0,
                                 GY.ravel() / 2.0])).reshape(GX.shape)
        fig.add_trace(go.Heatmap(
            x=gx, y=gy, z=Z, colorscale=CMAP_METRIC,
            colorbar=dict(title="Area (ADU)"),
            hovertemplate="Si=%{x:.1f}%  y=%{y:.2f}  aire=%{z:.3e}<extra></extra>"),
            row=1, col=2)
        fig.add_trace(go.Scatter(
            x=si_vals, y=prof_idx, mode="markers",
            marker=dict(size=9, color="black",
                        line=dict(color="white", width=1)),
            showlegend=False, hoverinfo="skip"), row=1, col=2)
    for c in (1, 2):
        fig.update_xaxes(title_text="Si (%)", tickvals=[0, 5, 15], row=1, col=c)
        fig.update_yaxes(title_text="Pulse profile",
                         tickvals=list(range(len(profile_order))),
                         ticktext=profile_order, row=1, col=c)
    fig.update_layout(template=TEMPLATE, height=height,
                      margin=dict(l=60, r=30, t=60, b=50))
    return fig


# ── Cartes angulaires 2D (memes calculs que plot_angular_metric) ─────────────
def fig_angular_map(phis, thetas, values, label="Value", title="",
                    n_grid=200, smoothing=2.0, theta_shift=0.0,
                    coverage_deg=None, height=480):
    valid = ~np.isnan(phis) & ~np.isnan(values)
    thetas_shifted = np.asarray(thetas, float) + theta_shift
    th_v, ph_v, val_v = (thetas_shifted[valid], phis[valid], values[valid])
    fids = np.where(valid)[0] + FIBER_DISPLAY_OFFSET
    vmin = float(np.nanpercentile(val_v, 2))
    vmax = float(np.nanpercentile(val_v, 98))

    fig = make_subplots(rows=1, cols=2, horizontal_spacing=0.14,
                        subplot_titles=("Points (fibers)", "RBF interpolation"))
    fig.add_trace(go.Scatter(
        x=th_v, y=ph_v, mode="markers+text",
        text=[str(f) for f in fids], textposition="middle center",
        textfont=dict(size=7, color="white"),
        marker=dict(size=16, color=val_v, cmin=vmin, cmax=vmax,
                    colorscale=CMAP_METRIC, line=dict(color="black", width=0.4),
                    colorbar=dict(title=label, x=0.42)),
        showlegend=False,
        hovertemplate="fiber %{text}: theta=%{x:.2f}  phi=%{y:.1f}  val=%{marker.color:.4g}<extra></extra>"),
        row=1, col=1)

    TT, PP, ZZ = sf._interp_angular_2d(phis, thetas_shifted, values,
                                       n_grid=n_grid, smoothing=smoothing)
    ZZ = np.clip(ZZ, vmin, vmax)
    if coverage_deg is not None:
        from scipy.spatial import cKDTree
        tree = cKDTree(np.column_stack([th_v, ph_v]))
        dist, _ = tree.query(np.column_stack([TT.ravel(), PP.ravel()]), k=1)
        ZZ = np.where(dist.reshape(TT.shape) > coverage_deg, np.nan, ZZ)
    fig.add_trace(go.Heatmap(
        x=TT[0], y=PP[:, 0], z=ZZ, colorscale=CMAP_METRIC,
        zmin=vmin, zmax=vmax, colorbar=dict(title=label),
        hovertemplate="theta=%{x:.2f}  phi=%{y:.1f}  val=%{z:.4g}<extra></extra>"),
        row=1, col=2)
    fig.add_trace(go.Scatter(
        x=th_v, y=ph_v, mode="markers",
        marker=dict(size=6, color="black", line=dict(color="white", width=0.5)),
        showlegend=False, hoverinfo="skip"), row=1, col=2)
    for c in (1, 2):
        fig.update_xaxes(title_text="theta (deg)", row=1, col=c)
        fig.update_yaxes(title_text="phi (deg)", row=1, col=c)
    fig.update_layout(template=TEMPLATE, title=title, height=height,
                      margin=dict(l=60, r=30, t=70, b=50))
    return fig


# ── Vue 3D sphere (memes calculs que plot_spectra_3d) ────────────────────────
def fig_sphere(spectra, phis, thetas, wl_axis, wl_range=None, smoothing=1.0,
               coverage_deg=12.0, n_grid=160, log_scale=False,
               exclude_fibers=None, show_laser=True,
               laser_origin=(0, 1, 0), laser_dir=(0, -1, 0), laser_len=1.5,
               title="", height=680, unit_label="ADU", xyz=None):
    """Sphere de collection.

    unit_label : unite affichee sur la barre de couleur ("ADU" ou "µJ/nm").
                 Suit le selecteur d'unites de la page, comme les axes.
    xyz        : (3, n_fibres) optionnel — coordonnees cartesiennes fournies
                 directement (colonnes x, y, z du fichier de configuration).
                 Si None, elles sont recalculees depuis (phi, theta) avec la
                 convention historique du pipeline.
    """
    from scipy.interpolate import RBFInterpolator
    if wl_range is not None:
        wl_min, wl_max = wl_range
        mask = (wl_axis >= wl_min) & (wl_axis <= wl_max)
        vals = spectra[:, mask].mean(axis=1).astype(float)
        val_label = f"Mean I {wl_min:.0f}-{wl_max:.0f} nm ({unit_label})"
    else:
        vals = spectra.max(axis=1).astype(float)
        val_label = f"I max ({unit_label})"
    valid = ~np.isnan(phis) & ~np.isnan(vals)
    if exclude_fibers:
        excl = np.atleast_1d(np.asarray(exclude_fibers, int)).ravel()
        excl = excl[(excl >= 0) & (excl < len(valid))]
        valid[excl] = False
    if xyz is not None:
        XYZ = np.asarray(xyz, float)
        valid &= np.all(np.isfinite(XYZ), axis=0)
        xs, ys, zs = XYZ[0][valid], XYZ[1][valid], XYZ[2][valid]
    else:
        xs, ys, zs = sf._fiber_angles_to_xyz(phis[valid], thetas[valid])
    v = vals[valid]
    fids = np.where(valid)[0] + FIBER_DISPLAY_OFFSET

    if log_scale:
        v_pos = v[np.isfinite(v) & (v > 0)]
        if v_pos.size >= 2:
            vmin = float(np.nanpercentile(v_pos, 2))
            vmax = float(np.nanpercentile(v_pos, 98))
            if not vmin > 0:
                vmin = float(v_pos.min())
            if vmax <= vmin:
                vmax = float(v_pos.max()) if v_pos.max() > vmin else vmin * 10
        else:
            log_scale = False
    if not log_scale:
        vmin = float(np.nanpercentile(v, 2))
        vmax = float(np.nanpercentile(v, 98))

    # RBF sur points uniques (arrondi 1e-5), thin plate, comme le notebook
    pts_all = np.column_stack([xs, ys, zs])
    pts_u, inv = np.unique(pts_all.round(5), axis=0, return_inverse=True)
    v_u = np.array([v[inv == k].mean() for k in range(len(pts_u))])
    rbf = RBFInterpolator(pts_u, v_u, kernel="thin_plate_spline",
                          smoothing=smoothing)
    theta_g = np.linspace(0, np.pi, n_grid)
    phi_g = np.linspace(0, 2 * np.pi, n_grid)
    Tg, Pg = np.meshgrid(theta_g, phi_g, indexing="ij")
    Xg = np.sin(Tg) * np.cos(Pg)
    Yg = np.sin(Tg) * np.sin(Pg)
    Zg = np.cos(Tg)
    grid_pts = np.column_stack([Xg.ravel(), Yg.ravel(), Zg.ravel()])
    floor = vmin if log_scale else 0.0
    Ig = np.clip(rbf(grid_pts).reshape(n_grid, n_grid), floor, None)
    cos_thresh = np.cos(np.radians(coverage_deg))
    dot_max = (grid_pts @ pts_u.T).max(axis=1)
    mask_cov = (dot_max >= cos_thresh).reshape(n_grid, n_grid)

    surfcolor = np.log10(Ig) if log_scale else Ig
    cmin = np.log10(vmin) if log_scale else vmin
    cmax = np.log10(vmax) if log_scale else vmax
    Xm, Ym, Zm = Xg.copy(), Yg.copy(), Zg.copy()
    Xm[~mask_cov] = np.nan
    Ym[~mask_cov] = np.nan
    Zm[~mask_cov] = np.nan

    fig = go.Figure()
    # Sphere grise de repere
    fig.add_trace(go.Surface(x=Xg, y=Yg, z=Zg,
                             surfacecolor=np.zeros_like(Ig),
                             colorscale=[[0, "lightgrey"], [1, "lightgrey"]],
                             opacity=0.10, showscale=False, hoverinfo="skip"))
    fig.add_trace(go.Surface(
        x=Xm, y=Ym, z=Zm, surfacecolor=surfcolor, cmin=cmin, cmax=cmax,
        colorscale=CMAP_METRIC,
        colorbar=dict(title=val_label + (" [log10]" if log_scale else "")),
        opacity=1.0, name="RBF",
        hovertemplate="val=%{surfacecolor:.4g}<extra></extra>"))
    color_pts = np.clip(v, vmin, vmax)
    fig.add_trace(go.Scatter3d(
        x=xs * 1.02, y=ys * 1.02, z=zs * 1.02, mode="markers+text",
        text=[str(f) for f in fids], textfont=dict(size=8),
        textposition="top center",
        marker=dict(size=4.5, color=color_pts, cmin=vmin, cmax=vmax,
                    colorscale=CMAP_METRIC,
                    line=dict(color="white", width=1)),
        name="fibers",
        hovertemplate="fiber %{text}: val=%{marker.color:.4g}<extra></extra>"))
    if show_laser:
        d = np.asarray(laser_dir, float)
        d = d / np.linalg.norm(d) * laser_len
        ox, oy, oz = laser_origin
        fig.add_trace(go.Scatter3d(
            x=[ox, ox + d[0]], y=[oy, oy + d[1]], z=[oz, oz + d[2]],
            mode="lines+text", text=["laser", ""],
            textfont=dict(color="limegreen", size=14),
            line=dict(color="limegreen", width=10), name="laser",
            hoverinfo="skip"))
        fig.add_trace(go.Cone(
            x=[ox + d[0]], y=[oy + d[1]], z=[oz + d[2]],
            u=[d[0] * 0.25], v=[d[1] * 0.25], w=[d[2] * 0.25],
            colorscale=[[0, "limegreen"], [1, "limegreen"]],
            showscale=False, sizemode="absolute", sizeref=0.12,
            hoverinfo="skip"))
    fig.update_layout(
        template=TEMPLATE, height=height, title=title,
        scene=dict(xaxis=dict(title="x", showticklabels=False),
                   yaxis=dict(title="y", showticklabels=False),
                   zaxis=dict(title="z", showticklabels=False),
                   aspectmode="cube"),
        margin=dict(l=10, r=10, t=50, b=10))
    return fig


# ── Barycentre vs theta ──────────────────────────────────────────────────────
def fig_centroid_theta(res, height=440):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=res["theta"], y=res["lambda_c_norm"], mode="lines+markers+text",
        text=[str(int(p) + FIBER_DISPLAY_OFFSET) for p in res["phys_fiber"]],
        textposition="top center", textfont=dict(size=8, color="dimgray"),
        line=dict(color="steelblue", width=1.6),
        marker=dict(size=8, color="white",
                    line=dict(color="steelblue", width=1.5)),
        hovertemplate=("theta=%{x:.2f} deg  lambda_c-min=%{y:.4f} nm"
                       "<br>phys. fiber %{text}<extra></extra>"),
        showlegend=False))
    fig.add_hline(y=0, line=dict(color="grey", dash="dash", width=1))
    t = (f"Normalised spectral centroid vs theta — {res['image']} "
         f"({res['config']}) — phi = {res['phi_target']} deg "
         f"[ref. min = {res['lambda_c_min']:.2f} nm]")
    return _base_layout(fig, t, "theta (deg)",
                        "lambda_c - lambda_c,min (nm)", height)


def fig_selected_spectra(res, height=440):
    fig = go.Figure()
    n = len(res["theta"])
    for k, (th, sp_row, fi) in enumerate(zip(res["theta"],
                                             res["spectra_export"],
                                             res["fiber_idx"])):
        c = k / max(1, n - 1)
        color = f"rgb({int(60+195*c)},{int(90*(1-c)+40)},{int(200*(1-c)+55)})"
        fig.add_trace(go.Scatter(
            x=res["wl_export"], y=sp_row, mode="lines",
            line=dict(width=1.0, color=color),
            name=f"theta={th:.1f} (f{fi + FIBER_DISPLAY_OFFSET})"))
    t = (f"Selected spectra — {res['image']} ({res['config']}) — "
         f"phi = {res['phi_target']} deg")
    return _base_layout(fig, t, "Wavelength (nm)", "Intensity (ADU)",
                        height)


# ── Correlations (scatter + fit + bande de confiance) ────────────────────────
def fig_scatter_fit(x, y, labels, fit=None, group_ids=None, group_names=None,
                    group_colors=None, xlab="", ylab="", title="",
                    height=470):
    fig = go.Figure()
    if group_ids is None:
        fig.add_trace(go.Scatter(
            x=x, y=y, mode="markers+text", text=labels,
            textposition="top right", textfont=dict(size=8, color="grey"),
            marker=dict(size=10, color="steelblue",
                        line=dict(color="black", width=0.6)),
            name="measurements",
            hovertemplate="%{text}<br>x=%{x:.4g}  y=%{y:.4g}<extra></extra>"))
    else:
        group_ids = np.asarray(group_ids)
        for gid in np.unique(group_ids):
            m = group_ids == gid
            fig.add_trace(go.Scatter(
                x=np.asarray(x)[m], y=np.asarray(y)[m], mode="markers+text",
                text=[labels[i] for i in np.where(m)[0]],
                textposition="top right", textfont=dict(size=8, color="grey"),
                marker=dict(size=10, color=group_colors[int(gid)],
                            line=dict(color="black", width=0.6)),
                name=group_names[int(gid)],
                hovertemplate="%{text}<br>x=%{x:.4g}  y=%{y:.4g}<extra></extra>"))
    if fit and fit.get("ok"):
        if fit.get("lower") is not None:
            fig.add_trace(go.Scatter(x=fit["x_fit"], y=fit["upper"],
                                     mode="lines", line=dict(width=0),
                                     showlegend=False, hoverinfo="skip"))
            fig.add_trace(go.Scatter(
                x=fit["x_fit"], y=fit["lower"], mode="lines",
                line=dict(width=0), fill="tonexty",
                fillcolor="rgba(255,140,0,0.25)", name="IC",
                hoverinfo="skip"))
        fig.add_trace(go.Scatter(
            x=fit["x_fit"], y=fit["y_fit"], mode="lines",
            line=dict(color="darkorange", width=2.8),
            name=f"fit : {fit['eq']}  (R2={fit['R2']:.4f})"))
    fig.update_layout(legend=dict(orientation="h", yanchor="bottom", y=1.02))
    return _base_layout(fig, title, xlab, ylab, height)


def fig_group_pulses(gp, group_name="", height=430):
    """All pulses of a group overlaid (rising-edge aligned, peak-normalised)."""
    fig = go.Figure()
    n = len(gp["pulses"])
    for k, (name, t, I) in enumerate(gp["pulses"]):
        c = k / max(1, n - 1)
        color = f"rgb({int(60+195*c)},{int(90*(1-c)+40)},{int(200*(1-c)+55)})"
        fig.add_trace(go.Scatter(
            x=t, y=I, mode="lines", line=dict(width=1.1, color=color),
            name=name,
            hovertemplate=name + "<br>t=%{x:.2f} ns  I=%{y:.3f}"
                          "<extra></extra>"))
    if n > 25:
        fig.update_layout(showlegend=False)
    return _base_layout(
        fig, f"{group_name} — {n} pulses (rising-edge aligned)",
        "Time from rising edge (ns)", "Intensity (normalised to peak)",
        height)


def fig_group_pulse_meanstd(gp, group_name="", height=430):
    """Mean +/- standard deviation of the group's pulses on the common grid."""
    t = gp["grid_ns"]
    mean, std = gp["mean"], gp["std"]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=t, y=mean + std, mode="lines",
                             line=dict(width=0), showlegend=False,
                             hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=t, y=np.clip(mean - std, 0, None),
                             mode="lines", line=dict(width=0),
                             fill="tonexty",
                             fillcolor="rgba(70,130,180,0.25)",
                             name="±1σ", hoverinfo="skip"))
    fig.add_trace(go.Scatter(
        x=t, y=mean, mode="lines", line=dict(color="#1d3557", width=2.4),
        name="mean",
        hovertemplate="t=%{x:.2f} ns  mean=%{y:.3f}<extra></extra>"))
    n = len(gp["pulses"])
    return _base_layout(
        fig, f"{group_name} — mean ± σ ({n} pulses)",
        "Time from rising edge (ns)", "Intensity (normalised to peak)",
        height)


def fig_pulse(pdata, title="", height=380):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=pdata["time_full"] * 1e9,
                             y=pdata["intensity_full"], mode="lines",
                             line=dict(color="#999", width=1),
                             name="full signal"))
    fig.add_trace(go.Scatter(x=pdata["time_pulse"] * 1e9,
                             y=pdata["intensity_pulse"], mode="lines",
                             line=dict(color="crimson", width=1.6),
                             name="pulse window"))
    return _base_layout(fig, title, "Time (ns)", "Signal (V)", height)


# ── Automatic fiber detection ─────────────────────────────────────────
def _conf_color(c):
    c = str(c)
    if c.startswith("directe"):
        return "#2e8b57"
    if c.startswith("enfouie"):
        return "#e6a817"
    if c.startswith("epaulement"):
        return "#e07020"
    return "#d62728"


def fig_fiber_detection(diag, positions, conf, manual=None, height=520):
    """Detection profile: band, candidates, final positions coloured by
    confidence, and the pipeline manual positions when comparable."""
    prof = np.asarray(diag["profile"], float)
    y = np.arange(len(prof))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=y, y=prof, mode="lines",
                             line=dict(color="#333", width=1),
                             name="profile (sum of lines)"))
    lo, hi = diag["band"]
    fig.add_vrect(x0=lo, x1=hi, fillcolor="rgba(70,130,180,0.07)",
                  line_width=0)
    fig.add_trace(go.Scatter(
        x=diag["candidates"], y=np.interp(diag["candidates"], y, prof),
        mode="markers", marker=dict(symbol="line-ns-open", size=7,
                                    color="#999"),
        name=f"candidates ({len(diag['candidates'])})"))
    seen = set()
    for p, c in zip(positions, conf):
        col = _conf_color(c)
        label = str(c).split(" ")[0]
        fig.add_trace(go.Scatter(
            x=[p], y=[float(np.interp(p, y, prof))], mode="markers",
            marker=dict(size=9, color=col, symbol="triangle-down",
                        line=dict(color="black", width=0.5)),
            name=label if label not in seen else None,
            showlegend=label not in seen,
            hovertemplate=f"y={p:.2f}px<br>{c}<extra></extra>"))
        seen.add(label)
    if manual is not None and len(manual) == len(positions):
        fig.add_trace(go.Scatter(
            x=manual, y=np.interp(manual, y, prof), mode="markers",
            marker=dict(size=6, color="rgba(30,30,200,0.55)", symbol="x"),
            name="manual positions (pipeline)"))
    fig.update_yaxes(type="log", title="Intensity (ADU, log)")
    fig.update_xaxes(title="Detector Y (px)")
    fig.update_layout(template=TEMPLATE, height=height,
                      title=f"Fiber detection — period "
                            f"{diag['period']:.1f} px, "
                            f"{diag.get('n_science', 0)} images science en renfort",
                      legend=dict(orientation="h", yanchor="bottom", y=1.02),
                      margin=dict(l=60, r=30, t=80, b=50))
    return fig


def fig_fiber_compare(manual, auto, conf, height=380):
    """Detection − manual gap, fiber by fiber (1–80 display)."""
    d = np.asarray(auto, float) - np.asarray(manual, float)
    xs = np.arange(len(d)) + FIBER_DISPLAY_OFFSET
    fig = go.Figure(go.Bar(
        x=xs, y=d, marker_color=[_conf_color(c) for c in conf],
        hovertemplate="fiber %{x}: Δ=%{y:.2f}px<extra></extra>"))
    fig.add_hline(y=0, line=dict(color="grey", width=1))
    for lim in (2, -2):
        fig.add_hline(y=lim, line=dict(color="orange", dash="dot", width=1))
    rms = float(np.sqrt(np.mean(d ** 2)))
    return _base_layout(
        fig, f"Detection − manual positions: RMS = {rms:.2f} px "
             f"(manual pointing: ±0.6 px intrinsic noise)",
        "Fiber (1–80)", "Gap (px)", height)


def fig_fiber_zooms(profile, positions, conf, flagged, hw=15, height=None):
    """Small zoomed multiples on the fibers to check/correct."""
    n = len(flagged)
    if n == 0:
        return empty_fig("No fiber to check: all direct.")
    ncols = min(3, n)
    nrows = int(np.ceil(n / ncols))
    fig = make_subplots(rows=nrows, cols=ncols, subplot_titles=[
        f"fiber {i + FIBER_DISPLAY_OFFSET} — {str(conf[i]).split(' ')[0]}"
        for i in flagged])
    prof = np.asarray(profile, float)
    for k, i in enumerate(flagged):
        r, c = k // ncols + 1, k % ncols + 1
        p0 = positions[i]
        lo, hi = int(p0) - hw, int(p0) + hw + 1
        xs = np.arange(lo, hi)
        fig.add_trace(go.Scatter(x=xs, y=prof[lo:hi], mode="lines",
                                 line=dict(color="#333", width=1),
                                 showlegend=False), row=r, col=c)
        fig.add_vline(x=p0, line=dict(color=_conf_color(conf[i]), width=2),
                      row=r, col=c)
        for j in (i - 1, i + 1):
            if 0 <= j < len(positions):
                fig.add_vline(x=positions[j],
                              line=dict(color="grey", dash="dot", width=1),
                              row=r, col=c)
    fig.update_layout(template=TEMPLATE,
                      height=height or (230 * nrows + 60),
                      title="Fibers to check (coloured line = detected "
                            "position; dashed = neighbours)",
                      margin=dict(l=50, r=20, t=70, b=40))
    return fig


def fig_shot_fibers_overlay(arr, positions, conf, title="", x_downsample=3,
                            height=760):
    """Real (derotated) image with the 80 fibers overlaid.
    Green = direct measurement; orange/red = fibers to check.
    Raster rendering (go.Image): full resolution in Y (the one that matters
    to check positions), smooth zoom even on 2160 lines."""
    import matplotlib
    try:
        cmap = matplotlib.colormaps["hot"]
    except AttributeError:            # matplotlib < 3.6
        from matplotlib import cm
        cmap = cm.get_cmap("hot")

    a = np.log10(np.clip(arr[:, ::x_downsample], 1, None))
    v2, v98 = np.nanpercentile(a, [2, 99.5])
    norm = np.clip((a - v2) / max(v98 - v2, 1e-9), 0, 1)
    rgb = (cmap(norm)[:, :, :3] * 255).astype(np.uint8)
    fig = go.Figure(go.Image(z=rgb, dx=x_downsample, x0=0))

    xmax = arr.shape[1]
    direct_y, flagged, flagged_cols = [], [], []
    for i, (p, c) in enumerate(zip(positions, conf)):
        if str(c).startswith("directe"):
            direct_y.append(p)
        else:
            flagged.append((i, p))
            flagged_cols.append(_conf_color(c))
    # direct fibers: thin green segments (one trace, separated by None)
    xs, ys = [], []
    for p in direct_y:
        xs += [0, xmax, None]
        ys += [p, p, None]
    fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines",
                             line=dict(color="rgba(0,255,140,0.55)", width=1),
                             name="directes", hoverinfo="skip"))
    # numbered markers on the left edge (hover = # + position)
    fig.add_trace(go.Scatter(
        x=[8] * len(positions), y=list(positions), mode="markers",
        marker=dict(size=4, color="rgba(0,255,140,0.8)"),
        text=[f"fiber {i + FIBER_DISPLAY_OFFSET} — {c}"
              for i, c in enumerate(conf)],
        hovertemplate="%{text}<br>y = %{y:.2f} px<extra></extra>",
        name="fibers (hover)", showlegend=False))
    # flagged fibers: thick coloured lines + label
    for (i, p), col in zip(flagged, flagged_cols):
        fig.add_trace(go.Scatter(
            x=[0, xmax], y=[p, p], mode="lines",
            line=dict(color=col, width=2.4, dash="dash"),
            name=f"fiber {i + FIBER_DISPLAY_OFFSET} (to check)",
            hovertemplate=(f"fiber {i + FIBER_DISPLAY_OFFSET}: "
                           f"y = {p:.2f} px<extra></extra>")))
        fig.add_annotation(x=xmax * 0.985, y=p,
                           text=f"f{i + FIBER_DISPLAY_OFFSET}",
                           showarrow=False, font=dict(color=col, size=11),
                           bgcolor="rgba(0,0,0,0.5)")
    fig.update_yaxes(autorange="reversed", title="Y (px)")
    fig.update_xaxes(title="X (px)")
    fig.update_layout(template=TEMPLATE, height=height, title=title,
                      legend=dict(orientation="h", yanchor="bottom", y=1.02),
                      margin=dict(l=60, r=30, t=80, b=50),
                      dragmode="zoom")
    return fig


# ── Absolute calibration diagnostics ─────────────────────────────────────────
def fig_abs_before(wl_b, B, cut_nm, frac, height=380):
    """Before-fiber lamp spectrum with the filter cut and the excluded band
    shaded, annotating the fraction kept above the cut."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=wl_b, y=B, mode="lines", name="before-fiber lamp",
                             line=dict(color="#274b73")))
    ymax = float(np.nanmax(B)) if np.isfinite(np.nanmax(B)) else 1.0
    fig.add_vrect(x0=float(np.nanmin(wl_b)), x1=cut_nm, fillcolor="#ff6b6b",
                  opacity=0.12, line_width=0,
                  annotation_text=f"excluded (<{cut_nm:.0f} nm)",
                  annotation_position="top left")
    fig.add_vline(x=cut_nm, line=dict(color="#ff6b6b", dash="dash"))
    fig.add_annotation(x=cut_nm, y=ymax, yshift=6,
                       text=f"kept above cut: {frac*100:.1f}%",
                       showarrow=False, font=dict(color="#c0392b"))
    return _base_layout(fig, "Before-fiber spectrum (band-fraction correction)",
                        "Wavelength (nm)", "Intensity (a.u.)", height)


def fig_abs_after(wl, A, cut_nm, height=400):
    """After-fiber per-fiber spectra (ADU): all fibers faint + the mean."""
    fig = go.Figure()
    n = A.shape[0]
    for i in range(n):
        fig.add_trace(go.Scatter(x=wl, y=A[i], mode="lines",
                                 line=dict(width=0.6, color="rgba(39,75,115,0.25)"),
                                 hoverinfo="skip", showlegend=False))
    fig.add_trace(go.Scatter(x=wl, y=np.nanmean(A, axis=0), mode="lines",
                             name="mean", line=dict(color="#e6194B", width=2)))
    fig.add_vline(x=cut_nm, line=dict(color="#ff6b6b", dash="dash"))
    return _base_layout(fig, "After-fiber spectra (80 fibers, ADU)",
                        "Wavelength (nm)", "Intensity (ADU)", height)


def fig_abs_transfer(wl, tau, cut_nm, height=400):
    """Per-fiber transfer function τ = (A/t_after)/(B/t_before), above the cut."""
    fig = go.Figure()
    n = tau.shape[0]
    for i in range(n):
        fig.add_trace(go.Scatter(x=wl, y=tau[i], mode="lines",
                                 line=dict(width=0.6, color="rgba(60,140,90,0.30)"),
                                 hoverinfo="skip", showlegend=False))
    fig.add_trace(go.Scatter(x=wl, y=np.nanmean(tau, axis=0), mode="lines",
                             name="mean transfer", line=dict(color="#118a3d", width=2)))
    fig.add_vline(x=cut_nm, line=dict(color="#ff6b6b", dash="dash"))
    return _base_layout(fig, "Fiber transfer function (after/before, time-normalised)",
                        "Wavelength (nm)", "Transfer (a.u.)", height)


def fig_abs_factor(g, outliers=None, height=340):
    """Per-fiber ADU→energy factor g_i (µJ per ADU·nm), bars 1–80. Outlier
    fibers (unreliable factor) are drawn in red."""
    g = np.asarray(g, float)
    x = np.arange(len(g)) + FIBER_DISPLAY_OFFSET
    outset = set(outliers or [])
    colors = ["#e6194B" if int(v) in outset else "#4363d8" for v in x]
    fig = go.Figure(go.Bar(x=x, y=g, marker_color=colors,
                           hovertemplate="fiber %{x}: %{y:.3e}<extra></extra>"))
    if outset:
        fig.add_annotation(x=0, y=1, xref="paper", yref="paper",
                           text="red = outlier (unreliable)", showarrow=False,
                           font=dict(color="#e6194B", size=11),
                           xanchor="left", yanchor="bottom")
    return _base_layout(fig, "Absolute factor per fiber",
                        "Fiber (1–80)", "g (µJ / ADU·nm)", height)


def fig_abs_power_vs_adu(P, I_after, height=380):
    """Power meter vs after-fiber ADU integral, per fiber — outliers here flag
    a bad fiber alignment or a saturated/dark fiber."""
    P = np.asarray(P, float)
    I = np.asarray(I_after, float)
    x = np.arange(len(P)) + FIBER_DISPLAY_OFFSET
    fig = go.Figure(go.Scatter(
        x=I, y=P, mode="markers+text", text=[str(int(v)) for v in x],
        textposition="top center", textfont=dict(size=8),
        marker=dict(size=7, color=x, colorscale="Viridis",
                    colorbar=dict(title="fiber")),
        hovertemplate="fiber %{text}: ∫ADU=%{x:.2e}, P=%{y:.2f} µW<extra></extra>"))
    return _base_layout(fig, "Power meter vs after-fiber ADU integral",
                        "∫ after-fiber (ADU·nm, >cut)", "Power (µW)", height)


def fig_abs_energy(wl, E, cut_nm, fibers=None, height=400):
    """A calibrated spectrum in energy density (µJ/nm) for selected fibers."""
    fig = go.Figure()
    n = E.shape[0]
    idx = fibers if fibers is not None else range(n)
    for i in idx:
        fig.add_trace(go.Scatter(x=wl, y=E[i], mode="lines",
                                 name=f"fiber {i + FIBER_DISPLAY_OFFSET}",
                                 line=dict(width=1)))
    fig.add_vline(x=cut_nm, line=dict(color="#ff6b6b", dash="dash"))
    return _base_layout(fig, "Calibrated spectrum (energy density)",
                        "Wavelength (nm)", "Spectral energy (J/nm)", height)


# ── Calibration absolue : courbes J/(nm·count) vs lambda ─────────────────────
def fig_abs_calibration_curves(wl, C, C_raw=None, fibers=None, cut_nm=None,
                               log_y=True, replaced=None, height=520):
    """Courbe de calibration de chaque fibre, en J/(nm·count).

    C, C_raw : (nfib, nwl) en J/(nm·count). Les courbes brutes sont tracees en
    pointille fin derriere les courbes lissees, comme dans le script de
    reference : c'est le seul moyen de voir si le lissage suit la tendance ou
    s'il invente une forme.
    """
    C = np.asarray(C, float)
    nfib = C.shape[0]
    if fibers is None:
        step = max(1, nfib // 9)
        fibers = list(range(0, nfib, step))[:9]
    fig = go.Figure()
    for i in fibers:
        if not (0 <= i < nfib):
            continue
        col = _fiber_color(i, nfib)
        if C_raw is not None:
            y = np.asarray(C_raw, float)[i]
            fig.add_trace(go.Scatter(
                x=wl, y=np.where(y > 0, y, np.nan), mode="lines",
                line=dict(color=col, width=0.7, dash="dot"),
                opacity=0.40, showlegend=False, hoverinfo="skip"))
        y = C[i]
        lab = f"fiber {i + FIBER_DISPLAY_OFFSET}"
        if replaced is not None and i in set(replaced):
            lab += " (replaced)"
        fig.add_trace(go.Scatter(
            x=wl, y=np.where(y > 0, y, np.nan), mode="lines", name=lab,
            line=dict(color=col, width=1.6),
            hovertemplate="λ=%{x:.1f} nm<br>C=%{y:.3e} J/(nm·count)"
                          f"<br>{lab}<extra></extra>"))
    if cut_nm:
        fig.add_vline(x=float(cut_nm), line=dict(color="crimson", width=1.2,
                                                 dash="dash"),
                      annotation_text=f"cut {cut_nm:.0f} nm")
    _base_layout(fig, title="", xlab="Wavelength (nm)",
                 ylab="Calibration (J·nm⁻¹·count⁻¹)", height=height)
    if log_y:
        fig.update_yaxes(type="log")
    return fig


def fig_abs_calibration_map(wl, C, cut_nm=None, height=480):
    """Carte fibre x lambda de la calibration (log10). Une bande horizontale
    anormale designe immediatement une fibre a probleme."""
    C = np.asarray(C, float)
    with np.errstate(divide="ignore", invalid="ignore"):
        Z = np.log10(np.where(C > 0, C, np.nan))
    fig = go.Figure(go.Heatmap(
        x=wl, y=np.arange(C.shape[0]) + FIBER_DISPLAY_OFFSET, z=Z,
        colorscale=CMAP_METRIC,
        colorbar=dict(title="log₁₀ C<br>[J/(nm·count)]"),
        hovertemplate="λ=%{x:.1f}  fiber=%{y}<br>"
                      "log₁₀C=%{z:.2f}<extra></extra>"))
    if cut_nm:
        fig.add_vline(x=float(cut_nm),
                      line=dict(color="white", width=1.2, dash="dash"))
    _base_layout(fig, title="", xlab="Wavelength (nm)", ylab="Fiber (1–80)",
                 height=height)
    return fig


def fig_abs_power_chain(labels, values, height=380):
    """Cascade des facteurs multiplicatifs du wattmetre vers la fibre.

    Rend visible d'un coup d'oeil quel terme domine les ordres de grandeur —
    en pratique le rapport des aires coeur/ouverture.
    """
    fig = go.Figure(go.Bar(
        x=labels, y=values, marker_color="#3d6ea8",
        text=[f"{v:.3e}" for v in values], textposition="outside",
        hovertemplate="%{x}: %{y:.4e}<extra></extra>"))
    _base_layout(fig, title="", xlab="", ylab="Multiplicative factor",
                 height=height)
    fig.update_yaxes(type="log")
    return fig


def _fiber_color(i, n):
    import plotly.colors as pc
    t = 0.0 if n <= 1 else i / (n - 1)
    return pc.sample_colorscale("Turbo", [t])[0]
