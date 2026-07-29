"""Page 4 — Exploration: inspect an image, a fiber, the SNR."""
from __future__ import annotations

from collections import OrderedDict

import numpy as np
import dash_bootstrap_components as dbc
from dash import Input, Output, State, dcc, html
from core.uistate import callback

from app.components import (card, guard_alert, labeled, page_header, graph,
                            units_radio)
from core import analysis, plotting
from core import spectro_functions as sf
from core.session import SESSION

# Small in-memory cache (the on-disk .npy remain the source of truth)
_MEM: OrderedDict[str, np.ndarray] = OrderedDict()
_MEM_MAX = 30


def _spectra(shot_key):
    if shot_key in _MEM:
        _MEM.move_to_end(shot_key)
        return _MEM[shot_key]
    sp = analysis.get_spectra(shot_key)
    _MEM[shot_key] = sp
    while len(_MEM) > _MEM_MAX:
        _MEM.popitem(last=False)
    return sp


def _apply_units(sp, units):
    """Set the global unit choice and return (converted spectra, y-label)."""
    SESSION.display_units = units or "adu"
    return analysis.to_display_units(sp)


def _x_axis():
    s = SESSION
    if s.params["USE_WL_AXIS"]:
        return s.calib["wl_axis"], "Wavelength (nm)"
    return np.arange(len(s.calib["wl_axis"])), "Pixel"


def _img_options():
    """Options du menu des images science.

    Une image dont le format interdit l'extraction reste VISIBLE mais est
    desactivee et etiquetee : la faire disparaitre laisserait croire qu'elle
    n'a pas ete fournie.

    Volontairement local plutot que partage : une page ne doit pas pouvoir
    devenir inaccessible parce qu'un fichier commun n'a pas ete mis a jour
    lors de l'installation.
    """
    bad = SESSION.unusable_images()
    return [{"label": (f"\u26a0 {k} \u2014 unusable file" if k in bad else k),
             "value": k, "disabled": k in bad}
            for k in sorted(SESSION.image_dict)]


def _first_usable():
    """Premiere image selectionnable, pour la valeur par defaut."""
    bad = SESSION.unusable_images()
    return next((k for k in sorted(SESSION.image_dict) if k not in bad), None)


def _spectra_guard(img):
    """(spectres, None) ou (None, alerte). Sans ce garde-fou, une image au
    mauvais format faisait lever le callback : Dash gardait alors la sortie
    precedente et l'utilisateur voyait le spectre du shot d'AVANT, en croyant
    regarder celui qu'il venait de choisir."""
    bad = SESSION.unusable_images().get(img)
    if bad:
        return None, guard_alert(f"{img}: {bad}", "danger")
    try:
        return _spectra(img), None
    except Exception as e:
        return None, guard_alert(f"{img} cannot be extracted: {e}", "danger")


def layout():
    ready = SESSION.calib is not None
    return html.Div([
        page_header(
            "Exploration",
            "Freely look at an image, a spectrum, a fiber.",
            "Step 5. Pick an image and open the tabs. If the image has "
            "not been extracted yet, it is extracted on the fly (2–3 s) "
            "and cached."),
        (guard_alert("Run the calibration first (step 2).")
         if not ready else html.Div()),
        card("Selection", dbc.Row([
            labeled("Image", dcc.Dropdown(id="x-image",
                                          options=_img_options(),
                                          value=_first_usable()),
                    width=4),
            labeled("Fiber (1–80, detector order)",
                    dbc.Input(id="x-fiber", type="number", value=63, min=1,
                              max=80, step=1), width=2),
            labeled("Fibers to compare (e.g. 38,39,40,41)",
                    dbc.Input(id="x-fibers-multi", value="38,39,40,41,42,43"),
                    width=3),
            dbc.Col(dbc.Checklist(id="x-normalize", switch=True,
                                  options=[{"label": " Normalise",
                                            "value": "n"}], value=[],
                                  className="mt-4"), width=2),
        ]), icon="bi-crosshair"),
        units_radio("x-units"),
        dbc.Tabs([
            dbc.Tab(dcc.Loading(html.Div(id="x-tab-image")),
                    label="Detector view", tab_id="t-img"),
            dbc.Tab(dcc.Loading(html.Div(id="x-tab-all")),
                    label="All 80 spectra", tab_id="t-all"),
            dbc.Tab(dcc.Loading(html.Div(id="x-tab-fiber")),
                    label="Fiber inspection", tab_id="t-fib"),
            dbc.Tab(dcc.Loading(html.Div(id="x-tab-multi")),
                    label="Fiber comparison", tab_id="t-multi"),
            dbc.Tab(dcc.Loading(html.Div(id="x-tab-evol")),
                    label="Temporal evolution", tab_id="t-evol"),
            dbc.Tab(dcc.Loading(html.Div(id="x-tab-snr")),
                    label="SNR map", tab_id="t-snr"),
        ], id="x-tabs", active_tab="t-img", className="mt-2"),
    ])


@callback(Output("x-tab-image", "children"),
          Input("x-tabs", "active_tab"), Input("x-image", "value"))
def tab_image(tab, img):
    if tab != "t-img" or not img:
        return html.Div()
    if SESSION.calib is None:
        return guard_alert("Calibration required.")
    arr_small, factor = analysis.load_image_preview(SESSION.image_dict[img])
    fy = SESSION.calib.get("fiber_y_ref",
                           getattr(sf, "FIBER_Y_MANUAL", None))
    return html.Div([
        graph(plotting.fig_image(
            arr_small, factor, fiber_y=fy,
            title=f"{img} — detector view (log scale, master fiber positions)")),
        html.Small("Display is downsampled for smoothness; every "
                   "computation uses the full-resolution image.",
                   className="text-muted"),
    ])


@callback(Output("x-tab-all", "children"),
          Input("x-tabs", "active_tab"), Input("x-image", "value"),
          Input("x-units", "value"))
def tab_all(tab, img, units):
    if tab != "t-all" or not img:
        return html.Div()
    if SESSION.calib is None:
        return guard_alert("Calibration required.")
    raw, err = _spectra_guard(img)
    if err:
        return err
    sp, ylab = _apply_units(raw, units)
    x, xlab = _x_axis()
    fig = plotting.fig_all_spectra_heatmap(
        sp, x, xlab, title=f"{img} — 80 spectra (2–98 percentile scale)")
    fig.update_traces(colorbar_title_text=("J/nm" if units == "uJ"
                                           else "ADU"))
    return graph(fig)


@callback(Output("x-tab-fiber", "children"),
          Input("x-tabs", "active_tab"), Input("x-image", "value"),
          Input("x-fiber", "value"), Input("x-units", "value"))
def tab_fiber(tab, img, fiber, units):
    if tab != "t-fib" or not img or fiber is None:
        return html.Div()
    if SESSION.calib is None:
        return guard_alert("Calibration required.")
    fiber = int(fiber) - 1   # display 1-80 -> internal 0-79
    raw, err = _spectra_guard(img)
    if err:
        return err
    sp, ylab = _apply_units(raw, units)
    x, xlab = _x_axis()
    areas = sf.compute_spectral_area(sp, SESSION.calib["wl_axis"])
    area_unit = "J" if units == "uJ" else "ADU"
    cent = sf.compute_spectral_centroid(
        sp[fiber:fiber + 1], SESSION.calib["wl_axis"])[0]
    snr = sf.compute_snr(sp[fiber])
    cfg, cfg_src = analysis.resolve_config(img)
    phys = sf.physical_fiber_index(fiber)
    info = dbc.Row([
        dbc.Col(dbc.Card(dbc.CardBody([
            html.Div(f"{v}", className="fw-bold"),
            html.Small(k, className="text-muted")]),
            className="text-center shadow-sm"), width="auto")
        for k, v in [
            ("Physical fiber", phys + 1),
            ("Angular config",
             f"{cfg} ({cfg_src})" if cfg else "undetermined"),
            (f"Spectral area ({area_unit})", f"{areas[fiber]:.3e}"),
            ("Centroid (nm)", f"{cent:.2f}" if np.isfinite(cent) else "N/A"),
            ("SNR", f"{snr:.1f}"),
        ]], className="g-2 mb-2")
    fig = plotting.fig_fiber_lines(
        sp, [fiber], x, xlab, title=f"{img} — fiber {fiber + 1}")
    fig.update_yaxes(title=ylab)
    return html.Div([info, graph(fig)])


@callback(Output("x-tab-multi", "children"),
          Input("x-tabs", "active_tab"), Input("x-image", "value"),
          Input("x-fibers-multi", "value"), Input("x-normalize", "value"),
          Input("x-units", "value"))
def tab_multi(tab, img, fibers_txt, norm, units):
    if tab != "t-multi" or not img:
        return html.Div()
    if SESSION.calib is None:
        return guard_alert("Calibration required.")
    try:
        fibers = [int(t) - 1 for t in str(fibers_txt or "").replace(";", ",")
                  .split(",") if t.strip() != ""]   # 1–80 -> 0–79
        fibers = [f for f in fibers if 0 <= f < SESSION.calib["n_fibers"]]
    except ValueError:
        return guard_alert("Unreadable fiber list — use comma-separated "
                           "numbers.")
    if not fibers:
        return guard_alert("No valid fiber in the list.")
    raw, err = _spectra_guard(img)
    if err:
        return err
    sp, ylab = _apply_units(raw, units)
    x, xlab = _x_axis()
    fig = plotting.fig_fiber_lines(
        sp, fibers, x, xlab, normalize=bool(norm),
        title=f"{img} — comparison of {len(fibers)} fibers")
    if not norm:
        fig.update_yaxes(title=ylab)
    return graph(fig)


@callback(Output("x-tab-evol", "children"),
          Input("x-tabs", "active_tab"), Input("x-fiber", "value"))
def tab_evol(tab, fiber):
    if tab != "t-evol":
        return html.Div()
    if SESSION.calib is None:
        return guard_alert("Calibration required.")
    cached = analysis.cached_shots()
    if len(cached) < 2:
        return guard_alert("Temporal evolution uses already-extracted "
                           "images: run the extraction first (step 3).")
    step = max(1, len(cached) // 12)
    subset = cached[::step][:12]
    return html.Div([
        labeled("Displayed images (already cached)",
                dcc.Dropdown(id="x-evol-imgs", multi=True,
                             options=[{"label": k, "value": k} for k in cached],
                             value=subset)),
        dcc.Loading(html.Div(id="x-evol-fig")),
    ])


@callback(Output("x-evol-fig", "children"),
          Input("x-evol-imgs", "value"), State("x-fiber", "value"),
          State("x-units", "value"))
def evol_fig(imgs, fiber, units):
    if not imgs:
        return html.Div()
    fiber = int(fiber or 1) - 1
    x, xlab = _x_axis()
    data = {k: _apply_units(_spectra(k), units)[0] for k in imgs}
    fig = plotting.fig_fiber_across_images(data, fiber, x, xlab)
    fig.update_yaxes(title=("Spectral energy (J/nm)" if units == "uJ"
                            else "Intensity (ADU)"))
    return graph(fig)


@callback(Output("x-tab-snr", "children"),
          Input("x-tabs", "active_tab"))
def tab_snr(tab):
    if tab != "t-snr":
        return html.Div()
    if SESSION.calib is None:
        return guard_alert("Calibration required.")
    n_cached = len(analysis.cached_shots())
    if not n_cached:
        return guard_alert("The SNR map uses all the extracted images: "
                           "run the extraction first (step 3).")
    return html.Div([
        html.P(html.Small(
            f"{n_cached} extracted images available. The computation runs "
            f"in the background with a progress bar (≈ 1 min for 500 "
            f"images); any failing image is listed, never silently "
            f"skipped.", className="text-muted")),
        dbc.Button([html.I(className="bi bi-play-fill me-1"),
                    "Compute the SNR map"], id="x-snr-run", color="primary"),
        dcc.Interval(id="x-snr-interval", interval=900, disabled=False),
        html.Div(id="x-snr-progress", className="mt-2"),
        html.Div(id="x-snr-fig"),
    ])


@callback(Output("x-snr-progress", "children"),
          Output("x-snr-interval", "disabled"),
          Input("x-snr-run", "n_clicks"), prevent_initial_call=True)
def snr_start(_):
    from core import jobs
    ok, msg = jobs.start_snr(analysis.cached_shots())
    # Re-enable the interval to follow this new computation
    return dbc.Alert(msg, color="success" if ok else "warning",
                     className="py-2"), False


@callback(Output("x-snr-fig", "children"),
          Output("x-snr-interval", "disabled", allow_duplicate=True),
          Input("x-snr-interval", "n_intervals"),
          prevent_initial_call=True)
def snr_poll(_):
    """Follows the progress then renders the map ONCE: as soon as the
    figure is rendered, the interval is disabled (otherwise the page
    re-rendered every 0.9 s and became unusable)."""
    from core import jobs
    from dash import no_update
    p = jobs.SNR_PROGRESS
    if p["running"]:
        pct = int(100 * p["done"] / max(1, p["total"]))
        return dbc.Progress(value=pct, label=f"{p['done']}/{p['total']}",
                            striped=True, animated=True,
                            style={"height": "20px"}), False
    if p["result"] is None:
        if p["finished_msg"]:
            return guard_alert(p["finished_msg"]), True
        return no_update, no_update    # nothing running: leave the DOM alone
    snr_arr, names = p["result"]
    col_mean = snr_arr.mean(axis=0)
    best = int(np.nanargmax(col_mean))
    worst = int(np.nanargmin(col_mean))
    out = [html.P([
        dbc.Badge(f"Global mean SNR: {np.nanmean(snr_arr):.1f}",
                  color="primary", className="me-2"),
        dbc.Badge(f"Best fiber: {best + 1} ({col_mean[best]:.1f})",
                  color="success", className="me-2"),
        dbc.Badge(f"Weakest: {worst + 1} ({col_mean[worst]:.1f})",
                  color="secondary")]),
        graph(plotting.fig_snr(snr_arr, names))]
    if p["errors"]:
        out.append(dbc.Accordion([dbc.AccordionItem(
            html.Ul([html.Li(e) for e in p["errors"][:20]]),
            title=f"{len(p['errors'])} failing images")],
            start_collapsed=True))
    return html.Div(out), True   # figure rendered -> interval disabled
