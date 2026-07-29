"""Page 7 — Laser correlations: spectral area vs E_2w / peak power /
peak intensity, plus pulse-profile inspection (single pulse or whole group)."""
from __future__ import annotations

import numpy as np
import dash_bootstrap_components as dbc
from dash import Input, Output, State, dcc, html
from core.uistate import callback

from app.components import (card, guard_alert, labeled, page_header,
                            graph, units_radio)
from core import analysis, plotting
from core import spectro_functions as sf
from core.campaign import load_groups
from core.session import SESSION


def _group_options():
    return [{"label": g["name"], "value": g["name"]}
            for g in load_groups(SESSION.ensure_workspace())]


def _beam_area(sg_n, sg_w):
    """Effective area (cm^2) from the UI inputs, with safe defaults."""
    try:
        return analysis.beam_area_cm2(float(sg_n or 8), float(sg_w or 111.0))
    except (ValueError, TypeError):
        return analysis.BEAM_AREA_CM2


def _shot_sets_from_source(source, categories, group_names, shots_txt):
    """Resolve the selected shot source into [(name, color, [shot keys])].
    Returns (shot_sets, error_component|None)."""
    s = SESSION
    if source == "manual":
        manual = []
        for tok in str(shots_txt or "").replace(";", ",").split(","):
            tok = tok.strip()
            if tok:
                n = analysis.shot_num(tok)
                if n is not None:
                    manual.append(f"shot{n:03d}")
        if not manual:
            return None, guard_alert("Enter at least one shot number.")
        return [("selection", "#4682b4", manual)], None
    if source == "categories":
        from core.metadata import PALETTE
        if not categories:
            return None, guard_alert("Select at least one category "
                                     "(target × profile).")
        if not s.metadata:
            ok, msg = analysis.load_metadata()
            if not ok:
                return None, guard_alert(msg)
        shot_sets = []
        for i, cat in enumerate(categories):
            target, profile = cat.split("|||")
            shots = [f"shot{shot:03d}"
                     for shot, m in sorted(s.metadata.items())
                     if m["target"] == target and m["profile"] == profile]
            shot_sets.append((f"{target} — {profile}",
                              PALETTE[i % len(PALETTE)], shots))
        return shot_sets, None
    # saved groups
    groups = load_groups(s.ensure_workspace())
    if not group_names:
        return None, guard_alert("Select at least one saved group.")
    return [(g["name"], g["color"], g["shots"])
            for g in groups if g["name"] in group_names], None


def layout():
    return html.Div([
        page_header(
            "Laser correlations",
            "Spectral area versus 2ω energy, peak power or peak intensity.",
            "Energy comes from the Excel shotbook (column C = shot, "
            "F = E_2ω). Peak power additionally needs the shot's "
            "oscilloscope CSV: P = (Imax/∫I·dt)·E·0.92. Peak intensity "
            "divides P by the effective area of the super-Gaussian focal "
            "spot, A_eff = (π/n)·w²·Γ(1/n) for I(r) = I₀·exp(−(r/w)^(2n)) — "
            "the order n and radius w are adjustable below because the "
            "focal spot may change between campaigns. Shots with missing "
            "data are listed, never invented."),
        card("Parameters", [
            dbc.Row([
                labeled("X axis", dcc.Dropdown(
                    id="r-xkind", clearable=False, value="energy",
                    options=[{"label": "2ω energy (J)", "value": "energy"},
                             {"label": "Peak power (W)", "value": "power"},
                             {"label": "Peak intensity (W/cm²)",
                              "value": "intensity"}]), width=3),
                labeled("Shot source", dcc.Dropdown(
                    id="r-source", clearable=False, value="categories",
                    options=[
                        {"label": "Campaign categories — all energies "
                                  "(recommended for an E/P/I scan)",
                         "value": "categories"},
                        {"label": "Saved groups (as defined)",
                         "value": "groups"},
                        {"label": "Manual shot list",
                         "value": "manual"},
                    ]), width=9),
            ]),
            dbc.Row([
                dbc.Col(labeled(
                    "Categories (target × profile, from Final.xlsx) — every "
                    "shot of the category, no energy filter",
                    dcc.Dropdown(id="r-categories", multi=True,
                                 options=[], placeholder="click to load "
                                 "the categories…")),
                    id="r-col-categories", width=6),
                dbc.Col(labeled("Groups (one colour per group)",
                                dcc.Dropdown(id="r-groups", multi=True,
                                             options=_group_options())),
                        id="r-col-groups", width=6,
                        style={"display": "none"}),
                dbc.Col(labeled("Shots (e.g. 319,320,321)",
                                dbc.Input(id="r-shots")),
                        id="r-col-manual", width=6,
                        style={"display": "none"}),
            ]),
            dbc.Row([
                labeled("Fiber 1–80 (empty = mean of all 80)",
                        dbc.Input(id="r-fiber", type="number", value=64,
                                  min=1, max=80),
                        width=2),
                labeled("λ range (nm, empty = all)",
                        dbc.Input(id="r-wl", placeholder="e.g. 700,900"),
                        width=2),
                labeled("Fit type", dcc.Dropdown(
                    id="r-fitkind", clearable=False, value="poly",
                    options=[{"label": "Polynomial", "value": "poly"},
                             {"label": "Exponential a·exp(b·x)",
                              "value": "exp"},
                             {"label": "None", "value": "none"}]), width=2),
                labeled("Degree (poly)", dbc.Input(id="r-deg", type="number",
                                                   value=2, min=1, max=6),
                        width=1),
                labeled("Confidence", dbc.Input(id="r-conf", type="number",
                                                value=0.95, min=0.5,
                                                max=0.999,
                                                step=0.01), width=2),
                dbc.Col(dbc.Button([html.I(className="bi bi-graph-up me-1"),
                                    "Compute"], id="r-run", color="primary",
                                   className="mt-4"), width=2),
            ]),
            card("Focal spot (super-Gaussian) — used for peak intensity", [
                dbc.Row([
                    labeled("Order n",
                            dbc.Input(id="r-sgn", type="number", value=8,
                                      min=0.1, step="any"),
                            help_txt="I(r) = I₀·exp(−(r/w)^(2n)); n = 1 is "
                                     "a Gaussian, larger n a flatter top. "
                                     "Any real value is accepted (2.1, 3.75…).",
                            width=3),
                    labeled("Radius w (µm)",
                            dbc.Input(id="r-sgw", type="number", value=111.0,
                                      min=0.1, step="any"),
                            help_txt="1/e radius of the intensity profile.",
                            width=3),
                    dbc.Col(html.Div(id="r-sgarea", className="mt-4 fw-bold"),
                            width=6),
                ]),
            ], icon="bi-bullseye"),
            dbc.Accordion([dbc.AccordionItem(dbc.Row([
                labeled("CSV header lines", dbc.Input(
                    id="r-header", type="number", value=25), width=3),
                labeled("Rising-edge threshold (× Imax)", dbc.Input(
                    id="r-rise", type="number", value=0.03, step=0.01),
                    width=3),
                labeled("Falling-edge threshold (× Imax)", dbc.Input(
                    id="r-fall", type="number", value=0.05, step=0.01),
                    width=3),
            ]), title="Pulse reading parameters (advanced)")],
                start_collapsed=True, className="mt-2"),
        ], icon="bi-sliders"),
        units_radio("r-units"),
        dcc.Loading(html.Div(id="r-out")),
        card("Pulse profiles", [
            html.Small(
                "Inspect a single pulse, or every pulse of a group side by "
                "side with the mean ± standard deviation to check the "
                "consistency of the pulses within the group. Pulses are "
                "aligned on their rising edge (the same threshold as the "
                "power computation) and normalised to their own peak.",
                className="text-muted d-block mb-2"),
            dbc.Row([
                labeled("Mode", dcc.Dropdown(
                    id="r-pulse-mode", clearable=False, value="single",
                    options=[{"label": "Single pulse", "value": "single"},
                             {"label": "All pulses of a category",
                              "value": "category"},
                             {"label": "All pulses of a saved group",
                              "value": "group"}]), width=3),
                dbc.Col(labeled("Shot number",
                                dbc.Input(id="r-pulse-shot", type="number",
                                          placeholder="e.g. 498")),
                        id="r-pulse-col-single", width=3),
                dbc.Col(labeled("Category",
                                dcc.Dropdown(id="r-pulse-category",
                                             options=[],
                                             placeholder="click to load…")),
                        id="r-pulse-col-category", width=5,
                        style={"display": "none"}),
                dbc.Col(labeled("Group",
                                dcc.Dropdown(id="r-pulse-group",
                                             options=_group_options())),
                        id="r-pulse-col-group", width=5,
                        style={"display": "none"}),
                dbc.Col(dbc.Button("Show", id="r-pulse-run",
                                   color="secondary", className="mt-4"),
                        width=2),
            ]),
            dcc.Loading(html.Div(id="r-pulse-out")),
        ], icon="bi-activity"),
    ])


@callback(Output("r-col-categories", "style"),
          Output("r-col-groups", "style"),
          Output("r-col-manual", "style"),
          Input("r-source", "value"))
def toggle_source(source):
    show, hide = {}, {"display": "none"}
    return (show if source == "categories" else hide,
            show if source == "groups" else hide,
            show if source == "manual" else hide)


@callback(Output("r-pulse-col-single", "style"),
          Output("r-pulse-col-category", "style"),
          Output("r-pulse-col-group", "style"),
          Input("r-pulse-mode", "value"))
def toggle_pulse_mode(mode):
    show, hide = {}, {"display": "none"}
    return (show if mode == "single" else hide,
            show if mode == "category" else hide,
            show if mode == "group" else hide)


@callback(Output("r-sgarea", "children"),
          Input("r-sgn", "value"), Input("r-sgw", "value"))
def show_beam_area(sg_n, sg_w):
    try:
        a = analysis.beam_area_cm2(float(sg_n), float(sg_w))
    except (ValueError, TypeError):
        return dbc.Badge("Invalid n or w — the pipeline default "
                         f"({analysis.BEAM_AREA_CM2:.4e} cm²) will be used.",
                         color="warning")
    return [dbc.Badge(f"A_eff = {a:.4e} cm²", color="primary",
                      className="me-2"),
            html.Small("exact: A_eff = (π/n)·w²·Γ(1/n) — n may be non-integer",
                       className="text-muted")]


def _category_options():
    """Categories (target × profile) from Final.xlsx, with counts.
    Counts separately the shots whose image is present in the folder."""
    s = SESSION
    if not s.metadata:
        ok, _ = analysis.load_metadata()
        if not ok:
            return []
    if not s.image_dict:
        s.scan_images()
    counts = {}
    for shot, m in s.metadata.items():
        if m["target"] in ("?", "") or m["profile"] in ("?", ""):
            continue
        key = (m["target"], m["profile"])
        c = counts.setdefault(key, [0, 0])
        c[0] += 1
        if f"shot{shot:03d}" in s.image_dict:
            c[1] += 1
    opts = []
    for (target, profile), (n_tot, n_avail) in sorted(counts.items()):
        opts.append({
            "label": f"{target} — {profile}  ({n_avail} images / "
                     f"{n_tot} shots)",
            "value": f"{target}|||{profile}"})
    return opts


@callback(Output("r-categories", "options"),
          Input("r-source", "value"))
def load_categories(source):
    if source != "categories":
        from dash import no_update
        return no_update
    return _category_options()


@callback(Output("r-pulse-category", "options"),
          Input("r-pulse-mode", "value"))
def load_pulse_categories(mode):
    if mode != "category":
        from dash import no_update
        return no_update
    return _category_options()


@callback(Output("r-out", "children"),
          Input("r-run", "n_clicks"),
          State("r-xkind", "value"), State("r-source", "value"),
          State("r-categories", "value"), State("r-groups", "value"),
          State("r-shots", "value"), State("r-fiber", "value"),
          State("r-wl", "value"), State("r-fitkind", "value"),
          State("r-deg", "value"), State("r-conf", "value"),
          State("r-header", "value"), State("r-rise", "value"),
          State("r-fall", "value"),
          State("r-sgn", "value"), State("r-sgw", "value"),
          State("r-units", "value"),
          prevent_initial_call=True)
def run(_, xkind, source, categories, group_names, shots_txt, fiber, wl_txt,
        fitkind, deg, conf, header, rise, fall, sg_n, sg_w, units):
    s = SESSION
    if s.calib is None:
        return guard_alert("Run the calibration first (step 2).")
    if xkind == "energy" and not s.energy_table:
        return guard_alert("Energy table missing: set Final.xlsx on the "
                           "Data page.")
    if xkind in ("power", "intensity") and not s.list_pulse_shots():
        return guard_alert("No pulse CSV detected: set the pulses folder "
                           "on the Data page.")

    wl_range = None
    toks = [t for t in str(wl_txt or "").split(",") if t.strip()]
    if len(toks) == 2:
        try:
            wl_range = (float(toks[0]), float(toks[1]))
        except ValueError:
            wl_range = None
    fiber_idx = int(fiber) - 1 if fiber not in (None, "") else None
    conf = float(conf or 0.95)
    beam_area = _beam_area(sg_n, sg_w)
    units = units if (units == "uJ" and analysis.abs_cal_ready()) else "adu"

    shot_sets, err = _shot_sets_from_source(source, categories, group_names,
                                            shots_txt)
    if err is not None:
        return err

    xs, ys, labels, gids, skipped = [], [], [], [], []
    gnames, gcolors = [], []
    for gid, (gname, gcolor, shots) in enumerate(shot_sets):
        gnames.append(gname)
        gcolors.append(gcolor)
        x, y, lab, skip = analysis.correlation_dataset(
            shots, xkind, fiber_idx, wl_range=wl_range,
            header_lines=int(header or 25), frac_rise=float(rise or 0.03),
            frac_fall=float(fall or 0.05), beam_area=beam_area,
            units=units)
        xs += list(x)
        ys += list(y)
        labels += lab
        gids += [gid] * len(x)
        skipped += [(gname, n, r) for n, r in skip]
    if not xs:
        details = html.Ul([html.Li(f"{g} / {n}: {r}")
                           for g, n, r in skipped[:15]])
        return guard_alert(["No usable data point. Reasons: ", details],
                           "danger")

    x = np.array(xs)
    y = np.array(ys)
    order = np.argsort(x)
    x, y = x[order], y[order]
    labels = [labels[i] for i in order]
    gids = np.array(gids)[order]

    fit = None
    if fitkind == "poly":
        fit = analysis.poly_fit_ci(x, y, int(deg or 2), conf)
    elif fitkind == "exp":
        fit = analysis.exp_fit_ci(x, y, conf)
    fit_alert = None
    if fit is not None and not fit.get("ok"):
        fit_alert = guard_alert(fit.get("msg", "Fit failed."))
        fit = None

    xlabels = {"energy": "2ω energy (J)", "power": "Peak power (W)",
               "intensity": "Peak intensity (W/cm²)"}
    ylab = ("Spectral integral (%s)" % ("J" if units == "uJ" else "ADU")
            + (f" [{wl_range[0]:.0f}–{wl_range[1]:.0f} nm]" if wl_range
               else ""))
    title = (f"Area vs {xlabels[xkind]} — fiber "
             f"{fiber_idx + 1 if fiber_idx is not None else 'mean'}")
    fig = plotting.fig_scatter_fit(
        x, y, labels, fit=fit,
        group_ids=gids if len(shot_sets) > 1 else None,
        group_names=gnames, group_colors=gcolors,
        xlab=xlabels[xkind], ylab=ylab, title=title)

    out = [graph(fig)]
    if xkind == "intensity":
        out.append(dbc.Alert(
            f"Peak intensity computed with A_eff = {beam_area:.4e} cm² "
            f"(super-Gaussian n = {float(sg_n or 8):.4g}, "
            f"w = {float(sg_w or 111):g} µm).",
            color="light", className="border py-2"))
    if fit_alert is not None:
        out.append(fit_alert)
    if fit is not None and fit.get("ok") and fitkind == "poly":
        rows = []
        degn = int(deg or 2)
        for i, c in enumerate(fit["p"]):
            d = degn - i
            se = fit["stderr"][i]
            ci = fit["tval"] * se if np.isfinite(se) else np.nan
            rows.append(html.Tr([
                html.Td(f"a{d}"), html.Td(f"{c:.6g}"),
                html.Td(f"± {ci:.3g}" if np.isfinite(ci) else "N/A")]))
        out.append(card("Fit details", [
            html.P([dbc.Badge(f"R² = {fit['R2']:.4f}", color="primary",
                              className="me-2"),
                    dbc.Badge(f"RMSE = {fit['rmse']:.4g}", color="secondary",
                              className="me-2"),
                    dbc.Badge(f"dof = {fit['df']}", color="secondary")]),
            dbc.Table([html.Thead(html.Tr([html.Th("Coefficient"),
                                           html.Th("Value"),
                                           html.Th(f"CI {int(conf*100)}%")])),
                       html.Tbody(rows)], size="sm"),
        ], icon="bi-calculator"))
    if fit is not None and fit.get("ok") and fitkind == "exp":
        out.append(dbc.Alert(
            f"Exponential fit: {fit['eq']}   (R² = {fit['R2']:.4f})",
            color="light", className="border"))
    if len(shot_sets) > 1 and fit is not None:
        out.append(dbc.Alert(
            "The fit is global over all groups: if the groups occupy "
            "distinct abscissa ranges, it mixes the effect of the condition "
            "with the effect of the intensity. Interpret with care, or fit "
            "one group at a time.", color="info", className="py-2"))
    if skipped:
        out.append(dbc.Accordion([dbc.AccordionItem(
            html.Ul([html.Li(f"{g} / {n}: {r}") for g, n, r in skipped]),
            title=f"{len(skipped)} shots skipped (missing data)")],
            start_collapsed=True))
    SESSION.log_history("correlation", {
        "x": xkind, "source": source,
        "selection": categories or group_names or "manual",
        "n_points": len(x),
        "fiber": fiber_idx, "fit": fitkind,
        "beam_area_cm2": beam_area if xkind == "intensity" else None,
        "R2": (fit or {}).get("R2") if fit else None})
    return html.Div(out)


@callback(Output("r-pulse-out", "children"),
          Input("r-pulse-run", "n_clicks"),
          State("r-pulse-mode", "value"), State("r-pulse-shot", "value"),
          State("r-pulse-category", "value"),
          State("r-pulse-group", "value"),
          State("r-header", "value"), State("r-rise", "value"),
          State("r-fall", "value"),
          State("r-sgn", "value"), State("r-sgw", "value"),
          prevent_initial_call=True)
def show_pulse(_, mode, shot, category, group_name, header, rise, fall,
               sg_n, sg_w):
    header = int(header or 25)
    rise = float(rise or 0.03)
    fall = float(fall or 0.05)

    if mode == "single":
        if shot in (None, ""):
            return guard_alert("Enter a shot number.")
        csv = SESSION.resolve_pulse_csv(int(shot))
        if csv is None:
            return guard_alert(f"CSV for shot {int(shot)} not found in the "
                               f"pulses folder.")
        try:
            pdata = sf.load_pulse_profile(str(csv), header_lines=header,
                                          frac_rise=rise, frac_fall=fall)
        except Exception as e:
            return guard_alert(f"Could not read the pulse: {e}", "danger")
        E = SESSION.energy_table.get(int(shot))
        extra = ""
        if E is not None:
            P = sf.compute_pulse_power(pdata, E)
            beam_area = _beam_area(sg_n, sg_w)
            extra = (f" — E_2ω = {E:.2f} J → P_peak = {P:.3e} W, "
                     f"I_peak = {P / beam_area:.3e} W/cm²")
        dur = (pdata["time_pulse"][-1] - pdata["time_pulse"][0]) * 1e9
        return graph(plotting.fig_pulse(
            pdata,
            title=f"Pulse shot {int(shot)} — window {dur:.2f} ns{extra}"))

    # group / category mode: all pulses + mean ± std
    if mode == "category":
        if not category:
            return guard_alert("Select a category.")
        s = SESSION
        if not s.metadata:
            ok, msg = analysis.load_metadata()
            if not ok:
                return guard_alert(msg)
        target, profile = category.split("|||")
        shots = [f"shot{n:03d}" for n, m in sorted(s.metadata.items())
                 if m["target"] == target and m["profile"] == profile]
        name = f"{target} — {profile}"
    else:
        if not group_name:
            return guard_alert("Select a saved group.")
        groups = load_groups(SESSION.ensure_workspace())
        match = [g for g in groups if g["name"] == group_name]
        if not match:
            return guard_alert("Group not found.")
        shots = match[0]["shots"]
        name = group_name

    gp = analysis.group_pulse_profiles(shots, header_lines=header,
                                       frac_rise=rise, frac_fall=fall)
    if gp is None:
        return guard_alert(f"No pulse could be loaded for '{name}' "
                           f"({len(shots)} shots — check the pulses "
                           f"folder).", "danger")
    out = [dbc.Row([
        dbc.Col(graph(plotting.fig_group_pulses(gp, group_name=name)), md=6),
        dbc.Col(graph(plotting.fig_group_pulse_meanstd(gp, group_name=name)),
                md=6),
    ])]
    # quick consistency figure: mean relative std inside the pulse
    core = gp["n_valid"] >= max(2, int(0.8 * len(gp["pulses"])))
    if np.any(core):
        rel = float(np.nanmean(gp["std"][core]))
        out.insert(0, html.P([
            dbc.Badge(f"{len(gp['pulses'])} pulses loaded", color="primary",
                      className="me-2"),
            dbc.Badge(f"mean σ over the common window = {rel:.3f} "
                      f"(normalised units)", color="secondary"),
        ]))
    if gp["skipped"]:
        out.append(dbc.Accordion([dbc.AccordionItem(
            html.Ul([html.Li(f"{n}: {r}") for n, r in gp["skipped"]]),
            title=f"{len(gp['skipped'])} shots skipped (no usable pulse)")],
            start_collapsed=True))
    SESSION.log_history("group_pulses", {
        "selection": name, "n_pulses": len(gp["pulses"]),
        "n_skipped": len(gp["skipped"])})
    return html.Div(out)
