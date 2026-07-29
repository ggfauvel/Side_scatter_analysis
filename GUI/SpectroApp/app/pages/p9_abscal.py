"""Page 9 — Absolute calibration: turn ADU spectra into physical energy.

The calibration is a CURVE per fiber, C_i(λ) in J/(nm·count), so that any
science spectrum converts as  E_i(λ) = counts_i(λ) · C_i(λ)  [J/nm], with no
dependence on the exposure time of the science image.

Everything campaign-specific is a user input: the power-meter geometry, the ND
filter of the lamp image, the long-pass cut, the smoothing. Nothing is
hard-coded, so a different bench only means different numbers here.
"""
from __future__ import annotations

from pathlib import Path

import dash_bootstrap_components as dbc
import numpy as np
from dash import Input, Output, State, dcc, html
from core.uistate import callback

from app.components import card, graph, guard_alert, labeled, page_header
from core import analysis, plotting
from core.session import SESSION


def _d():
    return analysis.abs_cal_params()


def _num(id_, label, value, help_txt=None, width=3, min_=None):
    kw = {"min": min_} if min_ is not None else {}
    return labeled(label,
                   dbc.Input(id=id_, type="number", value=value, step="any",
                             **kw),
                   help_txt=help_txt, width=width)


def _files_card(d):
    return card("1 · Measurement files", [
        html.Small("The first three are mandatory. The last three refine the "
                   "result and can be left empty — the page then reports which "
                   "correction was skipped rather than applying a silent "
                   "default.", className="text-muted d-block mb-2"),
        labeled("After-fiber lamp image (.tiff) — the lamp seen through the "
                "fibers, on the science detector *",
                dbc.Input(id="ab-after", value=d["after_image"],
                          placeholder="full path")),
        labeled("Before-fiber lamp spectrum (.spf2 or 2-column .csv) — same "
                "lamp measured before the fibers, unfiltered *",
                dbc.Input(id="ab-spf2", value=d["spf2"],
                          placeholder="full path")),
        labeled("Per-fiber power-meter readings (.txt, lines 'fiber / value') *",
                dbc.Input(id="ab-power", value=d["power_file"],
                          placeholder="full path")),
        labeled("Dark / background image (.tiff, same exposure, lamp off)",
                dbc.Input(id="ab-bg", value=d["background_image"],
                          placeholder="optional — subtracted fiber by fiber")),
        labeled("Power-meter responsivity curve (2 columns: λ nm, A/W)",
                dbc.Input(id="ab-resp", value=d["responsivity_file"],
                          placeholder="optional — corrects a meter set on a "
                                      "single wavelength")),
        labeled("Measured long-pass transmission curve (2 columns: λ nm, T)",
                dbc.Input(id="ab-lpcurve", value=d["longpass_curve"],
                          placeholder="optional — otherwise an ideal step at "
                                      "the cut")),
    ], icon="bi-paperclip")


def _powermeter_card(d):
    return card("2 · Power meter and geometry", [
        html.Small("This block sets the absolute scale, and it is where orders "
                   "of magnitude are won or lost. On the sphere the meter "
                   "measures the lamp irradiance over its own aperture, so "
                   "only the fraction intercepted by the fiber core actually "
                   "enters the fiber: the core/aperture area ratio alone is of "
                   "order 1e-4.", className="text-muted d-block mb-2"),
        dbc.Row([
            labeled("What the meter measured", dcc.Dropdown(
                id="ab-geom", clearable=False, value=d["power_geometry"],
                options=[
                    {"label": "Lamp irradiance at the fiber port on the sphere "
                              "(full geometry below)", "value": "sphere"},
                    {"label": "Power at the fiber output (reading used as is)",
                     "value": "through_fiber"}]), width=6),
            labeled("Unit of the file", dcc.Dropdown(
                id="ab-punit", clearable=False, value=d["power_unit"],
                options=[{"label": u, "value": u}
                         for u in ("nW", "uW", "mW", "W")]), width=2),
            _num("ab-pset", "Meter wavelength setting (nm)", d["power_set_nm"],
                 "Wavelength the meter was set to during the readings.",
                 width=4),
        ]),
        dbc.Row([
            labeled("How to enter the geometry", dcc.Dropdown(
                id="ab-distmode", clearable=False, value=d["dist_mode"],
                options=[
                    {"label": "Two distances lamp → plane (recommended)",
                     "value": "explicit"},
                    {"label": "Sphere radius + offset", "value": "offset"}]),
                    help_txt="The two-distance form removes any sign "
                             "ambiguity: just measure lamp→fibers and "
                             "lamp→meter.", width=4),
            labeled("Distance unit", dcc.Dropdown(
                id="ab-distunit", clearable=False, value=d["dist_unit"],
                options=[{"label": "cm", "value": "cm"},
                         {"label": "m", "value": "m"}]), width=2),
            _num("ab-dfiber", "Lamp → fibers", d["dist_lamp_fiber"],
                 "Distance from the lamp to the fiber entrance plane.",
                 width=3),
            _num("ab-dmeter", "Lamp → power meter", d["dist_lamp_meter"],
                 "Distance from the lamp to the meter sensor plane during "
                 "the readings.", width=3),
        ]),
        dbc.Row([
            _num("ab-radius", "Sphere radius (cm)", d["sphere_radius_cm"],
                 "Used only in 'radius + offset' mode.", width=3),
            _num("ab-closer", "Sensor closer to the lamp by (cm)",
                 d["sensor_closer_cm"],
                 "Positive = meter nearer the lamp than the fibers; negative "
                 "= further. Used only in 'radius + offset' mode.", width=3),
            _num("ab-aperture", "Meter input aperture Ø (mm)", d["aperture_mm"],
                 "From the sensor datasheet.", width=3),
            _num("ab-core", "Fiber core Ø (µm)", d["core_um"],
                 "Sets both the collected fraction and the solid angle used "
                 "on page 7.", width=3),
        ]),
        dbc.Row([
            _num("ab-na", "Fiber NA", d["fiber_na"], width=3),
            dbc.Col(dbc.Checklist(
                id="ab-naopt", switch=True,
                options=[{"label": " Apply the NA² solid-angle factor",
                          "value": "na"}],
                value=["na"] if d["apply_na"] else [], className="mt-4"),
                width=6),
        ]),
        html.Div(id="ab-geom-preview", className="mt-1"),
    ], icon="bi-bullseye")


def _optics_card(d):
    return card("3 · Optics on the lamp image", [
        dbc.Row([
            _num("ab-tafter", "Lamp-image exposure (ms)", d["t_after_ms"],
                 "Filled automatically from the TIFF; the energy scales "
                 "directly with it.", width=3, min_=0),
            _num("ab-nd", "ND optical density on the lamp image", d["nd_od"],
                 "0 = no ND. Uses the measured ND curves registered on page 2 "
                 "when available, otherwise 10^(−OD).", width=3, min_=0),
            _num("ab-cut", "Long-pass cut (nm)", d["cut_nm"],
                 "Where the filter starts transmitting.", width=3, min_=0),
            _num("ab-zero", "Force zero energy below (nm)", d["zero_below_nm"],
                 "Counts below this cannot come through the filter, so their "
                 "calibration is set to exactly zero.", width=3, min_=0),
        ]),
        html.Div(id="ab-expo-note"),
        html.Div(id="ab-nd-note", className="mt-1"),
        html.Div(id="ab-sat-note", className="mt-1"),
        dbc.Checklist(id="ab-forcezero", switch=True,
                      options=[{"label": " Apply the hard zero cut-off",
                                "value": "z"}],
                      value=["z"] if d["force_zero"] else []),
    ], icon="bi-funnel")


def _smoothing_card(d):
    return card("4 · Smoothing and bad fibers", [
        html.Small("The raw calibration is a ratio (known power)/(measured "
                   "counts). Where the lamp is weak the denominator is almost "
                   "pure noise, so the raw curve is unusable there. Smoothing "
                   "is done on log₁₀(C): the noise is multiplicative, and the "
                   "result stays positive. The raw curves remain visible under "
                   "the smoothed ones so the fit can be checked.",
                   className="text-muted d-block mb-2"),
        dbc.Row([
            labeled("Method", dcc.Dropdown(
                id="ab-smooth", clearable=False, value=d["smooth_method"],
                options=[{"label": l, "value": v} for l, v in [
                    ("Moving average", "moving_average"),
                    ("Gaussian", "gaussian"),
                    ("Savitzky-Golay", "savgol"),
                    ("Median filter", "median_filter"),
                    ("Smoothing spline", "spline"),
                    ("None (raw curves)", "none")]]), width=3),
            _num("ab-window", "Window (nm)", d["smooth_window_nm"],
                 "Width in wavelength; the other methods scale from it.",
                 width=2),
            dbc.Col(dbc.Checklist(
                id="ab-smoothlog", switch=True,
                options=[{"label": " Smooth in log space", "value": "log"}],
                value=["log"] if d["smooth_log"] else [], className="mt-4"),
                width=3),
            labeled("Fibers to replace by the mean curve (1–80)",
                    dbc.Input(id="ab-bad",
                              value=", ".join(str(b) for b in
                                              (d["bad_fibers"] or [])),
                              placeholder="e.g. 1, 2, 39, 78"), width=4),
        ]),
    ], icon="bi-sliders")


def layout():
    s = SESSION
    d = _d()
    ready = s.calib is not None
    return html.Div([
        page_header(
            "Absolute calibration",
            "Convert the ADU spectra into physical spectral energy (J/nm), "
            "per fiber and per wavelength.",
            "Step 3 (optional, but do it before the analysis if you want "
            "physical units). It reuses the wavelength axis and fiber "
            "positions of step 2, so run the calibration first. The result is "
            "a curve C(λ) per fiber, not a single number, so the spectral "
            "response of each fiber is preserved."),
        (guard_alert("Run the wavelength/fiber calibration (step 2) first.",
                     "info") if not ready else html.Div()),
        _files_card(d), _powermeter_card(d), _optics_card(d),
        _smoothing_card(d),
        dbc.Button([html.I(className="bi bi-magic me-1"),
                    "Build the absolute calibration"],
                   id="ab-build", color="primary", size="lg",
                   disabled=not ready),
        dcc.Loading(html.Div(id="ab-msg", className="mt-2")),
        dcc.Loading(html.Div(id="ab-diagnostics")),
        html.Div(dbc.Button("Next step: Extraction →", href="/extraction",
                            color="secondary", outline=True,
                            className="mt-3"), className="text-end"),
    ])


@callback(Output("ab-geom-preview", "children"),
          Input("ab-geom", "value"), Input("ab-radius", "value"),
          Input("ab-closer", "value"), Input("ab-aperture", "value"),
          Input("ab-core", "value"), Input("ab-na", "value"),
          Input("ab-naopt", "value"), Input("ab-distmode", "value"),
          Input("ab-distunit", "value"), Input("ab-dfiber", "value"),
          Input("ab-dmeter", "value"))
def geom_preview(geom, radius, closer, aperture, core, na, naopt,
                 distmode, distunit, dfiber, dmeter):
    from core import abscal
    if geom == "through_fiber":
        return html.Small("Reading used as is — no geometric factor applied.",
                          className="text-muted")
    if distmode == "explicit":
        k = 100.0 if distunit == "m" else 1.0
        r_fib = float(dfiber or 44) * k
        r_pm = float(dmeter or 44) * k
        radius, closer = r_fib, r_fib - r_pm
    try:
        r = abscal.power_entering_fiber(
            1.0, "W", 1.0, "sphere", float(radius or 44), float(closer or 0),
            float(aperture or 9.5), float(core or 100), float(na or 0.22),
            "na" in (naopt or []))
    except Exception as e:
        return guard_alert(str(e))
    tot = float(r["power_W"])
    return html.P([
        dbc.Badge(f"distance ×{r['distance_factor']:.4f}", color="secondary",
                  className="me-1"),
        dbc.Badge(f"core/aperture area ×{r['area_ratio']:.4e}",
                  color="secondary", className="me-1"),
        dbc.Badge(f"NA ×{r['na_factor']:.4g}", color="secondary",
                  className="me-1"),
        dbc.Badge(f"total ×{tot:.4e}", color="primary"),
        html.Br(),
        html.Small(f"1 W read on the meter → {tot:.4e} W entering a fiber. "
                   f"Core area {r['core_area_cm2']:.3e} cm², aperture area "
                   f"{r['aperture_area_cm2']:.3e} cm².",
                   className="text-muted"),
    ], className="mb-0")


@callback(Output("ab-tafter", "value"), Output("ab-expo-note", "children"),
          Input("ab-after", "value"), prevent_initial_call=True)
def autofill_exposure(path):
    """Lit la pose directement dans le TIFF. C'est ecrit dans le fichier
    (tag ImageDescription, 'expTime=10000ms') : aucune raison de laisser
    l'utilisateur la retaper, un facteur 10 ici passe tel quel sur l'energie."""
    from core import abscal
    from dash import no_update
    p = (path or "").strip().strip('"')
    if not p or not Path(p).exists():
        return no_update, ""
    t = abscal.tiff_exposure_ms(p) or abscal.exposure_from_name(p)
    if not t:
        return no_update, html.Small(
            "Exposure not readable from this file — enter it by hand.",
            className="text-warning")
    return t, html.Small(f"Exposure read from the file: {t:.0f} ms.",
                         className="text-success")


@callback(Output("ab-nd-note", "children"),
          Input("ab-after", "value"), Input("ab-nd", "value"))
def nd_consistency(path, od):
    """Le nom du fichier porte presque toujours le ND ('..._ND2A_...').
    Une incoherence entre ce nom et la valeur saisie vaut un facteur 10^OD
    sur toutes les energies : elle merite d'etre signalee tout de suite."""
    import re as _re
    name = Path((path or "").strip().strip('"')).name
    if not name:
        return ""
    m = _re.search(r"ND\s*([0-9]+(?:\.[0-9]+)?)", name, _re.IGNORECASE)
    in_name = float(m.group(1)) if m else 0.0
    od = float(od or 0)
    if abs(in_name - od) < 1e-9:
        return html.Small(
            f"File name and ND field agree (OD {od:g})."
            if od else "No ND in the file name, and OD 0 entered — consistent.",
            className="text-success")
    return guard_alert(
        f"The file name suggests OD {in_name:g} but OD {od:g} is entered. "
        f"That is a factor {10 ** abs(in_name - od):.0f} on every energy. "
        f"Set it to what was physically in the beam when this image was taken.",
        "warning")


@callback(Output("ab-sat-note", "children"), Input("ab-after", "value"),
          prevent_initial_call=True)
def saturation_note(path):
    """Une image saturee tronque le pic : l'energie absolue est alors
    sous-estimee, et aucune calibration ne peut le rattraper."""
    p = (path or "").strip().strip('"')
    if not p or not Path(p).exists():
        return ""
    try:
        from PIL import Image
        Image.MAX_IMAGE_PIXELS = None
        im = Image.open(p)
        a = np.array(im)
        full = 65535 if a.dtype == np.uint16 else (255 if a.dtype == np.uint8
                                                   else float(a.max()))
        frac = float((a >= full).mean())
    except Exception:
        return ""
    if frac <= 0:
        return html.Small("No saturated pixel.", className="text-success")
    return guard_alert(
        f"{frac * 100:.3f} % of the pixels are at full scale ({full:g}). "
        f"Saturated pixels clip the peak, so the absolute energy is "
        f"under-estimated there and no calibration can recover it.",
        "warning")


def _parse_ints(txt):
    out = []
    for t in str(txt or "").replace(";", ",").split(","):
        t = t.strip()
        if t:
            try:
                out.append(int(float(t)))
            except ValueError:
                pass
    return out


@callback(Output("ab-msg", "children"), Output("ab-diagnostics", "children"),
          Input("ab-build", "n_clicks"),
          State("ab-after", "value"), State("ab-spf2", "value"),
          State("ab-power", "value"), State("ab-bg", "value"),
          State("ab-resp", "value"), State("ab-lpcurve", "value"),
          State("ab-geom", "value"), State("ab-punit", "value"),
          State("ab-pset", "value"), State("ab-radius", "value"),
          State("ab-closer", "value"), State("ab-distmode", "value"),
          State("ab-distunit", "value"), State("ab-dfiber", "value"),
          State("ab-dmeter", "value"), State("ab-aperture", "value"),
          State("ab-core", "value"), State("ab-na", "value"),
          State("ab-naopt", "value"), State("ab-tafter", "value"),
          State("ab-nd", "value"), State("ab-cut", "value"),
          State("ab-zero", "value"), State("ab-forcezero", "value"),
          State("ab-smooth", "value"), State("ab-window", "value"),
          State("ab-smoothlog", "value"), State("ab-bad", "value"),
          prevent_initial_call=True)
def build(_, after, spf2, power, bg, resp, lpcurve, geom, punit, pset, radius,
          closer, distmode, distunit, dfiber, dmeter, aperture, core, na,
          naopt, tafter, nd, cut, zero, forcezero, smooth, window, smoothlog,
          bad):
    def clean(x):
        return (x or "").strip().strip('"')
    ok, msg = analysis.build_absolute_calibration(
        after_image=clean(after), spf2=clean(spf2), power_file=clean(power),
        background_image=clean(bg), responsivity_file=clean(resp),
        longpass_curve=clean(lpcurve),
        power_geometry=geom or "sphere", power_unit=punit or "uW",
        power_set_nm=float(pset or 750), sphere_radius_cm=float(radius or 44),
        sensor_closer_cm=float(closer or 0),
        dist_mode=distmode or "offset", dist_unit=distunit or "cm",
        dist_lamp_fiber=float(dfiber or 44),
        dist_lamp_meter=float(dmeter or 44),
        aperture_mm=float(aperture or 9.5), core_um=float(core or 100),
        fiber_na=float(na or 0.22), apply_na="na" in (naopt or []),
        t_after_ms=float(tafter or 1000), nd_od=float(nd or 0),
        cut_nm=float(cut or 0), zero_below_nm=float(zero or 0),
        force_zero="z" in (forcezero or []),
        smooth_method=smooth or "moving_average",
        smooth_window_nm=float(window or 10),
        smooth_log="log" in (smoothlog or []), bad_fibers=_parse_ints(bad))
    if not ok:
        return guard_alert(msg, "danger"), None
    return dbc.Alert(msg, color="success", className="py-2"), _diagnostics()


def _diagnostics():
    s = SESSION
    ac = s.abs_cal
    arr = getattr(s, "abs_cal_arrays", None)
    if ac is None or not arr:
        return None
    wl, C, C_raw = arr["wl"], arr["C"], arr["C_raw"]
    g = np.asarray(ac["g_uJ"], float)
    cut = ac["cut_nm"]
    bad = [int(b) - 1 for b in
           (analysis.abs_cal_params().get("bad_fibers") or [])]

    n_ok = int(np.isfinite(C).any(axis=1).sum())
    med = float(np.nanmedian(g))
    badges = dbc.Row([
        dbc.Col(dbc.Card(dbc.CardBody([html.Div(v, className="fs-6 fw-bold"),
                                       html.Small(k, className="text-muted")]),
                         className="text-center shadow-sm"), width="auto")
        for k, v in [
            ("Fibers calibrated", f"{n_ok}/{ac['n_fibers']}"),
            ("In-band lamp fraction", f"{ac['fraction_above']:.4f}"),
            ("Meter spectral corr.", f"×{ac.get('pm_correction', 1):.4f}"),
            ("Core/aperture area", f"×{ac.get('area_ratio', 1):.3e}"),
            ("Distance factor", f"×{ac.get('geometry', 1):.4f}"),
            ("Median C", f"{med * 1e-6:.3e} J/(nm·count)"),
        ]], className="g-2 mb-3")

    notes = ac.get("notes") or []
    notes_block = dbc.Alert(
        [html.B("What was applied"), html.Br(),
         html.Ul([html.Li(n) for n in notes], className="mb-0",
                 style={"fontSize": "0.85rem"})],
        color="light", className="border py-2") if notes else html.Div()

    chain_labels = ["meter reading (1 W)", "spectral corr.", "distance",
                    "core/aperture", "in-band fraction"]
    chain_vals = [1.0, float(ac.get("pm_correction", 1.0)),
                  float(ac.get("geometry", 1.0)),
                  float(ac.get("area_ratio", 1.0)),
                  float(ac["fraction_above"])]

    audit = analysis.abs_cal_audit(None)      # calibration seule
    audit_tbl = dbc.Table(
        [html.Thead(html.Tr([html.Th("Step"), html.Th("Value"),
                             html.Th("Unit"), html.Th("Meaning")]))] +
        [html.Tbody([html.Tr([html.Td(r["step"]),
                              html.Td(r["value"],
                                      style={"fontFamily": "monospace"}),
                              html.Td(r["unit"]),
                              html.Td(html.Small(r["comment"],
                                                 className="text-muted"))])
                     for r in audit])],
        size="sm", hover=True, striped=True, className="mb-0") if audit \
        else html.Div()

    keys = sorted(SESSION.image_dict)
    demo = html.Div()
    if keys:
        demo = card("Apply to a science shot (check)", [
            dbc.Row([
                dbc.Col(dcc.Dropdown(id="ab-demo-shot",
                                     options=[{"label": k, "value": k}
                                              for k in keys],
                                     value=keys[len(keys) // 2],
                                     clearable=False), width=5),
                dbc.Col(dbc.Button("Convert to J/nm", id="ab-demo-btn",
                                   color="primary", size="sm",
                                   className="mt-1"), width=3),
            ]),
            dcc.Loading(html.Div(id="ab-demo-fig")),
        ], icon="bi-lightning-charge")

    # Diagnostics essentiels, affiches par defaut.
    essential = html.Div([
        badges,
        card("Calibration curves — J·nm⁻¹·count⁻¹ vs λ",
             [html.Small("Solid = smoothed and saved; faint dotted = raw. A "
                         "fiber whose smoothed curve leaves its own raw cloud "
                         "belongs in the replacement list above.",
                         className="text-muted d-block mb-2"),
              graph(plotting.fig_abs_calibration_curves(
                  wl, C, C_raw=C_raw, cut_nm=cut, replaced=bad))],
             icon="bi-graph-down"),
        card("Calibration map (all fibers)",
             graph(plotting.fig_abs_calibration_map(wl, C, cut_nm=cut)),
             icon="bi-grid-3x3"),
        demo,
    ])

    # Details ranges dans un depliant : utiles pour un diagnostic pousse, mais
    # plus affiches en permanence une fois la calibration validee.
    details = dbc.Accordion(dbc.AccordionItem([
        notes_block,
        card("Where the orders of magnitude come from",
             graph(plotting.fig_abs_power_chain(chain_labels, chain_vals)),
             icon="bi-diagram-3"),
        card("Numeric audit of the chain",
             [html.Small("Every step with its number, for the brightest fiber "
                         "of the lamp image (calibration only).",
                         className="text-muted d-block mb-2"), audit_tbl],
             icon="bi-list-columns"),
        card("Lamp spectrum before the fibers & in-band fraction",
             graph(plotting.fig_abs_before(arr["wl_b"], arr["B"], cut,
                                           ac["fraction_above"])),
             icon="bi-align-start"),
        card("Lamp seen through the fibers (shape per fiber)",
             graph(plotting.fig_abs_after(wl, arr["A"], cut)),
             icon="bi-bar-chart"),
    ], title="Detailed diagnostics (open only if a value looks off)"),
        start_collapsed=True, className="mt-2")

    return html.Div([essential, details])


@callback(Output("ab-demo-fig", "children"),
          Input("ab-demo-btn", "n_clicks"), State("ab-demo-shot", "value"),
          prevent_initial_call=True)
def demo_shot(_, shot):
    s = SESSION
    if s.abs_cal is None or not shot:
        return guard_alert("Build the calibration first.")
    try:
        S = analysis.get_spectra(shot, use_cache=True)
        E = analysis.to_absolute_energy(S)
    except Exception as e:
        return guard_alert(f"Could not convert: {e}", "danger")
    wl = np.asarray(s.calib["wl_axis"], float)
    cut = s.abs_cal["cut_nm"]
    trapz = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
    m = wl >= cut
    per_fiber = trapz(np.nan_to_num(E[:, m]), wl[m], axis=1)
    ok = np.isfinite(per_fiber) & (per_fiber != 0)
    show = [int(i) for i in range(0, E.shape[0], max(1, E.shape[0] // 8))]
    rows = analysis.abs_cal_audit(shot)
    rows = rows[next((i for i, r in enumerate(rows)
                      if r["step"].startswith("── science")), len(rows)):]
    per_shot = dbc.Table(
        [html.Thead(html.Tr([html.Th("Step"), html.Th("Value"),
                             html.Th("Unit"), html.Th("Meaning")]))] +
        [html.Tbody([html.Tr([html.Td(r["step"]),
                              html.Td(r["value"],
                                      style={"fontFamily": "monospace"}),
                              html.Td(r["unit"]),
                              html.Td(html.Small(r["comment"],
                                                 className="text-muted"))])
                     for r in rows])],
        size="sm", hover=True, striped=True) if rows else html.Div()
    return html.Div([
        graph(plotting.fig_abs_energy(wl, E, cut, fibers=show)),
        dbc.Alert([
            dbc.Badge(f"total over {int(ok.sum())} fibers: "
                      f"{float(per_fiber[ok].sum()):.4e} J", color="primary",
                      className="me-2"),
            dbc.Badge(f"median per fiber: "
                      f"{float(np.nanmedian(per_fiber[ok])):.4e} J",
                      color="secondary"),
            html.Br(),
            html.Small("Energy actually collected by the fiber cores. Page 7 "
                       "extrapolates it over the sphere.",
                       className="text-muted"),
        ], color="light", className="border py-2"),
        html.Hr(),
        html.Small([html.B(f"Numbers for {shot}"), " — these describe exactly "
                    "the curve above. The application removes this shot's own "
                    "ND filter before converting; a script that does not must "
                    "be compared with the last line, not the one above it."],
                   className="text-muted d-block mb-2"),
        per_shot,
    ])
