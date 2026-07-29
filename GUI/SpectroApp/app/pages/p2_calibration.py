"""Page 2 — Calibration: parameters, run, diagnostics, ND correction."""
from __future__ import annotations

import numpy as np
import dash
import dash_bootstrap_components as dbc
from dash import Input, Output, State, dcc, html
from core.uistate import callback

from app.components import card, guard_alert, labeled, page_header, graph
from core import analysis, plotting
from core import spectro_functions as sf
from core.session import SESSION


def _params_form():
    p = SESSION.params
    return dbc.Row([
        labeled("Number of fibers",
                dbc.Input(id="c-nfibers", type="number", value=p["N_FIBERS"],
                          min=1, step=1), width=3),
        labeled("Extraction half-width (px)",
                dbc.Input(id="c-halfwidth", type="number",
                          value=p["HALF_WIDTH"], min=1, step=1), width=3),
        labeled("Fiber positions",
                dcc.Dropdown(id="c-fibermode",
                             options=[{"label": "Manual (pipeline — current campaign)",
                                       "value": "manual"},
                                      {"label": "Automatic detection (new campaign)",
                                       "value": "auto"}],
                             value=SESSION.params.get("FIBER_MODE", "manual"),
                             clearable=False), width=3),
        labeled("Science images as reinforcement (auto mode)",
                dbc.Input(id="c-nscience", type="number",
                          value=SESSION.params.get("FIBER_AUTO_N_SCIENCE", 5),
                          min=0, max=20), width=3),
        labeled("λ calibration method",
                dcc.Dropdown(id="c-wlmethod",
                             options=[{"label": "Automatic (comb-matching, recommended)",
                                       "value": "auto"},
                                      {"label": "Manual (pixel↔nm pairs)",
                                       "value": "manual"}],
                             value=p["WL_METHOD"], clearable=False), width=3),
        labeled("Systematic λ shift (nm)",
                dbc.Input(id="c-wlshift", type="number", value=p["WL_SHIFT_NM"],
                          step=0.1), width=3),
        labeled("Background filter (px)",
                dbc.Input(id="c-bgfilter", type="number",
                          value=p["BG_FILTER_SIZE"], min=1, step=2), width=3),
        labeled("Min. peak prominence",
                dbc.Input(id="c-peakprom", type="number",
                          value=p["PEAK_MIN_PROM"], min=0), width=3),
        labeled("Inter-fiber background margin (px)",
                dbc.Input(id="c-bggap", type="number", value=p["BG_COLUMN_GAP"],
                          min=0, step=1), width=3),
        labeled("Rotate science images",
                dcc.Dropdown(
                    id="c-shotrot", clearable=False,
                    value=int(p.get("SHOT_ROTATION", 0) or 0),
                    options=[{"label": "None", "value": 0},
                             {"label": "90° clockwise", "value": 90},
                             {"label": "180°", "value": 180},
                             {"label": "90° counter-clockwise", "value": 270}]),
                help_txt="Use if the shots were exported sideways (fibers "
                         "must end up horizontal, like the calibration). "
                         "Check the result in the Exploration tab.",
                width=3),
        labeled("Rotate calibration image",
                dcc.Dropdown(
                    id="c-calibrot", clearable=False,
                    value=int(p.get("CALIB_ROTATION", 0) or 0),
                    options=[{"label": "None", "value": 0},
                             {"label": "90° clockwise", "value": 90},
                             {"label": "180°", "value": 180},
                             {"label": "90° counter-clockwise", "value": 270}]),
                help_txt="Independent from the shots — only if the HgAr "
                         "image itself is rotated.",
                width=3),
        dbc.Col([
            dbc.Checklist(
                id="c-switches",
                options=[{"label": " Background subtraction", "value": "bg"},
                         {"label": " Relative intensity calibration",
                          "value": "int"},
                         {"label": " ND filter correction (from shotbook)",
                          "value": "nd"}],
                value=(["bg"] if p["SUBTRACT_BG"] else [])
                + (["int"] if p["USE_INT_CALIBRATION"] else [])
                + (["nd"] if p.get("USE_ND_CORRECTION") else []),
                switch=True, className="mt-4"),
        ], width=3),
    ])


def layout():
    return html.Div([
        page_header(
            "Calibration",
            "Fiber positions, tilt, and pixel → wavelength conversion.",
            "Step 2. The default values are the validated ones from "
            "your pipeline: only change them if you know why. Calibration "
            "takes a few seconds; then check that the RMS is low (< 0.3 nm) "
            "and that the green lines fall on the spectral peaks."),
        card("Pipeline parameters", [
            _params_form(),
            dbc.Button([html.I(className="bi bi-play-fill me-1"),
                        "Run calibration"],
                       id="c-run", color="primary", size="lg"),
            dcc.Loading(html.Div(id="c-result", className="mt-3"),
                        type="default"),
        ], icon="bi-sliders"),
        _nd_card(),
        html.Div(id="c-diagnostics"),
    ])


# ─────────────────────────── ND filter correction ───────────────────────────
def _nd_card():
    return card("ND filter correction", [
        html.Small(
            "Shots may have been recorded with different neutral-density "
            "filters. The optical density OD is read from the shotbook (you "
            "choose which ND column below — campaigns differ). When the "
            "switch above is on, every spectrum is "
            "divided by the filter transmission T(λ) so that shots with "
            "different NDs become directly comparable. For best accuracy, "
            "give below the manufacturer's Excel file (transmission % vs "
            "wavelength nm) for each OD value used during the experiment: "
            "the curve is interpolated onto the calibration wavelength "
            "axis. If no file is given for a value, the flat theoretical "
            "transmission 10^(−OD) is applied instead (a warning is shown). "
            "The correction is applied on the fly — the extraction cache "
            "stays valid.",
            className="text-muted d-block mb-2"),
        dbc.Button([html.I(className="bi bi-arrow-repeat me-1"),
                    "Detect the ND values in the shotbook"],
                   id="c-nd-refresh", color="secondary", outline=True,
                   size="sm", className="mb-2"),
        dcc.Loading(html.Div(id="c-nd-selector")),
        dcc.Loading(html.Div(id="c-nd-body")),
        html.Hr(),
        html.Small(
            "Auto-detect the transmission files: point to a folder and the "
            "application reads the OD written inside each datasheet (e.g. "
            "'…OD: 1.0') to match it to the right ND value. The file names "
            "do not matter — only their content. This fills the fields "
            "above; review them, then save.",
            className="text-muted d-block mb-2"),
        dbc.Row([
            dbc.Col(dbc.Input(
                id="c-nd-folder",
                placeholder="folder containing the ND transmission Excel "
                            "files (empty = the images folder)"), width=9),
            dbc.Col(dbc.Button([html.I(className="bi bi-search me-1"),
                                "Scan folder for ND files"],
                               id="c-nd-scan", color="secondary",
                               outline=True, size="sm"), width=3),
        ], className="mb-1"),
        html.Div(id="c-nd-scan-msg"),
        html.Div(id="c-nd-status"),
    ], icon="bi-brightness-alt-high")


def _nd_column_selector():
    """Dropdown to choose which shotbook column holds the side-SRS ND, with
    each column's value distribution shown to help identify the right one."""
    from core import metadata as md
    from collections import Counter
    s = SESSION
    try:
        cols = md.nd_columns(s.excel_path)
    except Exception:
        cols = []
    if not cols:
        return html.Div()
    current = s.nd_column or md.auto_nd_column(cols)
    options = []
    for h, letter in cols:
        try:
            t = md.load_final_table(s.excel_path, nd_column=h)
            cc = Counter(f"{m['nd']:g}" if m['nd'] is not None else "—"
                         for m in t.values())
            dist = ", ".join(f"{k}:{v}" for k, v in sorted(cc.items()))
        except Exception:
            dist = ""
        options.append({"label": f"{h}  ({dist})", "value": h})
    return labeled(
        "ND column in the shotbook (the side-scattering spectrometer)",
        dcc.Dropdown(id="c-nd-column", options=options, value=current,
                     clearable=False),
        help_txt="Different diagnostics have their own ND column. Pick the "
                 "one for the multi-fiber side-scattering spectrometer; the "
                 "value counts (e.g. '1:24, 2:36') help you recognise it.")


def _nd_od_rows():
    """One file-path input per distinct OD value found in the chosen column,
    plus the Save button."""
    from core import ndfilters as ndf
    s = SESSION
    if not s.metadata:
        ok, msg = analysis.load_metadata()
        if not ok:
            return guard_alert(f"Cannot read the shotbook: {msg}")
    od_keys = ndf.od_values_in_metadata(s.metadata)
    if not od_keys:
        return dbc.Alert(
            "No non-zero ND value found in the selected column — nothing to "
            "correct (try another ND column above if this looks wrong).",
            color="light", className="border py-2")
    counts = {}
    for m in s.metadata.values():
        try:
            od = float(m.get("nd"))
        except (TypeError, ValueError):
            continue
        if od != 0.0:
            counts[f"{od:g}"] = counts.get(f"{od:g}", 0) + 1
    rows = []
    for key in od_keys:
        rows.append(dbc.Row([
            dbc.Col(html.Small([html.B(f"ND {key}"),
                                f" — {counts.get(key, 0)} shots"]),
                    width=2, className="mt-2"),
            dbc.Col(dbc.Input(
                id={"type": "nd-file", "index": key},
                value=s.nd_files.get(key, ""),
                placeholder="full path to the transmission Excel file "
                            "(empty = theoretical 10^-OD)"), width=10),
        ], className="mb-1"))
    rows.append(dbc.Button([html.I(className="bi bi-check2 me-1"),
                            "Save & check the ND files"],
                           id="c-nd-apply", color="primary", size="sm",
                           className="mt-2"))
    return html.Div(rows)


@callback(Output("c-nd-body", "children", allow_duplicate=True),
          Input("c-nd-column", "value"),
          prevent_initial_call=True)
def nd_change_column(col):
    """User picked a different ND column: persist it, re-read the shotbook,
    rebuild the OD rows (the selector itself is not re-rendered)."""
    s = SESSION
    if col == (s.nd_column or None):
        from dash import no_update
        return no_update
    s.nd_column = col or None
    s.save_config()
    analysis.load_metadata()
    from core import ndfilters as ndf
    ndf.clear_cache()
    return _nd_od_rows()


@callback(Output("c-nd-selector", "children"),
          Output("c-nd-body", "children"),
          Input("c-nd-refresh", "n_clicks"),
          prevent_initial_call=True)
def nd_refresh(_):
    return _nd_column_selector(), _nd_od_rows()


@callback(Output("c-nd-body", "children", allow_duplicate=True),
          Output("c-nd-scan-msg", "children"),
          Input("c-nd-scan", "n_clicks"),
          State("c-nd-folder", "value"),
          prevent_initial_call=True)
def nd_scan(_, folder):
    from core import ndfilters as ndf
    s = SESSION
    if not s.metadata:
        ok, msg = analysis.load_metadata()
        if not ok:
            return dash.no_update, guard_alert(f"Cannot read Final.xlsx: {msg}")
    folder = (folder or "").strip().strip('"')
    if not folder:
        folder = s.images_dir or ""
    if not folder:
        return dash.no_update, guard_alert(
            "No folder given and no images folder set — type the path to "
            "the folder that contains your ND transmission files.")
    wanted = ndf.od_values_in_metadata(s.metadata)
    found, reports = ndf.scan_nd_folder(folder, wanted_od_keys=wanted)
    # merge into the registry: keep existing manual entries the scan didn't fill
    s.nd_files = {**s.nd_files, **found}
    # build the report block
    colors = {"ok": "success", "warn": "warning", "info": "secondary"}
    items = [html.Li([dbc.Badge(lvl, color=colors.get(lvl, "secondary"),
                                className="me-2"), msg])
             for lvl, msg in reports]
    missing = [k for k in wanted if k not in s.nd_files]
    if missing:
        items.append(html.Li([dbc.Badge("warn", color="warning",
                                         className="me-2"),
                              f"No file found for OD {', '.join(missing)} — "
                              f"the theoretical 10^-OD will be used for "
                              f"those (or fill the field above by hand)."]))
    msg_block = dbc.Alert(
        [html.B(f"{len(found)} file(s) matched. "),
         html.Ul(items, className="mb-0 mt-1")],
        color="success" if found and not missing else "warning",
        className="py-2")
    # regenerate the input rows, now pre-filled from the updated registry
    return _nd_od_rows(), msg_block


@callback(Output("c-nd-status", "children"),
          Input("c-nd-apply", "n_clicks"),
          State({"type": "nd-file", "index": dash.ALL}, "value"),
          State({"type": "nd-file", "index": dash.ALL}, "id"),
          prevent_initial_call=True)
def nd_apply(_, values, ids):
    from core import ndfilters as ndf
    s = SESSION
    nd_files = {}
    for v, i in zip(values, ids):
        v = (v or "").strip().strip('"')
        if v:
            nd_files[str(i["index"])] = v
    s.nd_files = nd_files
    s.save_config()
    ndf.clear_cache()
    rows = analysis.nd_status_table()
    if not rows:
        return dbc.Alert("No ND value to correct.", color="light",
                         className="border py-2")
    body = []
    any_theory = False
    for r in rows:
        if r["source"].startswith("theory"):
            any_theory = True
        badge_color = {"file": "success", "file+clamp": "warning"}.get(
            r["source"], "secondary")
        body.append(html.Tr([
            html.Td(f"ND {r['od']}"),
            html.Td(f"{r['n_shots']}"),
            html.Td(html.Small(r["path"] or "—")),
            html.Td(dbc.Badge(r["source"], color=badge_color)),
            html.Td(html.Small(r["detail"])),
        ]))
    table = dbc.Table([
        html.Thead(html.Tr([html.Th(h) for h in
                            ["OD", "Shots", "File", "Correction",
                             "Curve coverage"]])),
        html.Tbody(body)], size="sm", hover=True, className="mb-1 mt-2")
    notes = [table]
    if any_theory:
        notes.append(dbc.Alert(
            "At least one OD value has no measured curve: the flat "
            "theoretical transmission 10^(−OD) will be used for those "
            "shots. Supply the manufacturer's file for full accuracy.",
            color="warning", className="py-2"))
    if any(r["source"] == "file+clamp" for r in rows):
        notes.append(dbc.Alert(
            "'file+clamp': the calibration wavelength axis extends beyond "
            "the measured curve; the edge transmission values are held "
            "constant outside the measured range.",
            color="info", className="py-2"))
    if not SESSION.params.get("USE_ND_CORRECTION"):
        notes.append(dbc.Alert(
            "Files saved — remember to turn on the 'ND filter correction' "
            "switch above and re-run the calibration for the correction to "
            "be applied.", color="info", className="py-2"))
    return html.Div(notes)


# ────────────────────────────── Calibration run ──────────────────────────────
@callback(
    Output("c-result", "children"), Output("c-diagnostics", "children"),
    Input("c-run", "n_clicks"),
    State("c-nfibers", "value"), State("c-halfwidth", "value"),
    State("c-wlmethod", "value"), State("c-wlshift", "value"),
    State("c-bgfilter", "value"), State("c-peakprom", "value"),
    State("c-bggap", "value"), State("c-switches", "value"),
    State("c-fibermode", "value"), State("c-nscience", "value"),
    State("c-shotrot", "value"), State("c-calibrot", "value"),
    prevent_initial_call=True)
def run_calibration(_, nf, hw, wlm, wls, bgf, prom, gap, switches,
                    fibermode, nscience, shotrot, calibrot):
    s = SESSION
    if not s.image_dict:
        s.scan_images()
    s.params.update(N_FIBERS=int(nf or 80), HALF_WIDTH=int(hw or 6),
                    WL_METHOD=wlm or "auto", WL_SHIFT_NM=float(wls or 0),
                    BG_FILTER_SIZE=int(bgf or 55),
                    PEAK_MIN_PROM=float(prom or 50),
                    BG_COLUMN_GAP=int(gap or 4),
                    SUBTRACT_BG="bg" in (switches or []),
                    USE_INT_CALIBRATION="int" in (switches or []),
                    USE_ND_CORRECTION="nd" in (switches or []),
                    FIBER_MODE=fibermode or "manual",
                    FIBER_AUTO_N_SCIENCE=int(nscience or 5),
                    SHOT_ROTATION=int(shotrot or 0),
                    CALIB_ROTATION=int(calibrot or 0))
    s.save_config()
    ok, msg = analysis.run_calibration()
    if not ok:
        return guard_alert(msg, "danger"), None

    summ = analysis.calib_summary()
    badges = dbc.Row([
        dbc.Col(dbc.Card(dbc.CardBody([
            html.Div(v, className="fs-5 fw-bold"),
            html.Small(k, className="text-muted")]),
            className="text-center shadow-sm"), width="auto")
        for k, v in [
            ("Fibers", f"{summ['n_fibers']}"),
            ("Period (px)", f"{summ['fiber_period']:.1f}"),
            ("Spectral range", f"{summ['wl_min']:.1f} – {summ['wl_max']:.1f} nm"),
            ("Dispersion", f"{summ['dispersion']:.4f} nm/px"),
            ("Calibration RMS", f"{summ['rms']:.3f} nm"),
            ("Matched lines", f"{summ['n_pairs']}"),
        ]], className="g-2 mb-2")
    rms_ok = summ["rms"] < 0.3
    verdict = dbc.Alert(
        ("Reliable calibration: residuals are small and homogeneous."
         if rms_ok else
         "High RMS (> 0.3 nm): visually check the line matching below "
         "before going further."),
        color="success" if rms_ok else "warning", className="py-2")

    # ND correction status (when enabled)
    nd_alert = html.Div()
    if s.params.get("USE_ND_CORRECTION"):
        try:
            rows = analysis.nd_status_table()
            n_theory = sum(1 for r in rows
                           if r["source"].startswith("theory"))
            nd_alert = dbc.Alert(
                [html.B("ND correction is ON. "),
                 f"{len(rows)} OD value(s) in the shotbook — "
                 + (f"{n_theory} without a measured curve (theoretical "
                    f"10^-OD applied). " if n_theory else
                    "all with a measured transmission curve. ")
                 + "Details in the 'ND filter correction' card."],
                color="warning" if n_theory else "success",
                className="py-2")
        except Exception as e:
            nd_alert = guard_alert(f"ND correction is ON but its status "
                                   f"could not be checked: {e}")

    # Detected lines vs catalogue table (equivalent of cell 5)
    table = None
    try:
        arr_calib = sf.load_image(s.hgar_path)
        detected = sf.auto_detect_wl_lines(
            arr_calib, s.calib["wl_axis"],
            mean_spectrum=s.calib.get("mean_calib_spectrum"))
        rows = []
        for px, wl_cat, wl_meas, diff in detected:
            sym = sf.HGAR_LINES.get(wl_cat, "?")
            used = any(abs(p[1] - wl_cat) < 0.01 for p in s.calib["wl_pairs"])
            flag = ("OK" if diff < 0.5 else
                    "check" if diff < 1.5 else "misidentified?")
            rows.append(html.Tr([
                html.Td(px), html.Td(f"{wl_cat:.2f}"),
                html.Td(f"{wl_meas:.2f}"), html.Td(f"{diff:.3f}"),
                html.Td(sym),
                html.Td(dbc.Badge("used" if used else "no",
                                  color="success" if used else "secondary")),
                html.Td(dbc.Badge(flag, color={"OK": "success",
                                               "check": "warning"}
                                  .get(flag, "danger"))),
            ]))
        table = dbc.Table([
            html.Thead(html.Tr([html.Th(h) for h in
                                ["Pixel", "λ catalogue (nm)", "λ measured (nm)",
                                 "Δ (nm)", "Elem.", "Calibration", "Status"]])),
            html.Tbody(rows)], bordered=False, hover=True, size="sm",
            className="mb-0")
    except Exception as e:
        table = guard_alert(f"Line table unavailable: {e}")

    # Diagnostic figures
    arr_small, factor = analysis.load_image_preview(s.hgar_path)
    fy = s.calib.get("fiber_y_ref", getattr(sf, "FIBER_Y_MANUAL", None))
    figs = html.Div([
        dbc.Row([
            dbc.Col(graph(plotting.fig_image(
                arr_small, factor, fiber_y=fy,
                title="HgAr image (derotated at extraction) + fiber positions")),
                md=6),
            dbc.Col(graph(plotting.fig_calib_spectrum(s.calib)),
                    md=6),
        ]),
        graph(plotting.fig_calib_residuals(s.calib)),
        card("Auto-detected HgAr lines", table, icon="bi-list-check"),
        (card("Intensity calibration factors",
              graph(plotting.fig_intensity_calib(
                  s.int_calib_factors, s.calib["wl_axis"])),
              icon="bi-bar-chart")
         if s.params["USE_INT_CALIBRATION"] else html.Div()),
        (_fiber_detection_section(s) if s.params.get("FIBER_MODE") == "auto"
         else html.Div()),
        dbc.Accordion([dbc.AccordionItem(
            html.Pre(s.calib_log, style={"fontSize": "0.75rem",
                                         "maxHeight": "300px",
                                         "overflow": "auto"}),
            title="Technical calibration log")],
            start_collapsed=True),
        dbc.Button("Next step: Absolute calibration →", href="/absolute",
                   color="primary", className="mt-3"),
    ])
    return html.Div([badges, verdict, nd_alert]), figs


def _flagged_indices(conf):
    return [i for i, c in enumerate(conf)
            if not str(c).startswith("directe")]


def _fiber_detection_section(s):
    from core import analysis as an
    fa = s.fiber_auto
    if fa is None:
        return html.Div()
    pos = an.effective_fiber_positions()
    conf = fa["conf"]
    flagged = _flagged_indices(conf)
    n_dir = len(conf) - len(flagged)
    manual = an._FIBER_Y_PIPELINE if len(an._FIBER_Y_PIPELINE) == len(pos) \
        else None
    figs = []
    diag = an.get_fiber_diag()
    if diag is not None:
        figs.append(graph(plotting.fig_fiber_detection(
            diag, pos, conf, manual=manual)))
        figs.append(graph(plotting.fig_fiber_zooms(
            diag["profile"], pos, conf, flagged)))
    else:
        figs.append(guard_alert(
            "Diagnostics unavailable (older detection) — click "
            "'Re-detect fibers' to regenerate them.", "info"))
    # Overlay on a real image, user's choice
    keys = sorted(SESSION.image_dict)
    figs.append(html.Div([
        html.Hr(),
        html.H6("Check on a real image"),
        html.Small("Pick a shot: the 80 detected positions are overlaid on "
                   "it (green = direct; coloured + dashed = to check). Zoom "
                   "with the mouse on a flagged fiber to read its exact "
                   "position on hover, then correct it in the table below.",
                   className="text-muted d-block mb-2"),
        dbc.Row([
            dbc.Col(dcc.Dropdown(
                id="c-fib-shot",
                options=([{"label": "HgAr image (calibration)",
                           "value": "__hgar__"}]
                         + [{"label": k, "value": k} for k in keys]),
                value=keys[len(keys) // 2] if keys else "__hgar__",
                clearable=False), width=4),
            dbc.Col(dbc.Button("Show image + fibers",
                               id="c-fib-overlay-btn", color="primary",
                               size="sm", className="mt-1"), width=3),
        ]),
        dcc.Loading(html.Div(id="c-fib-overlay")),
    ]))
    if manual is not None:
        figs.append(graph(plotting.fig_fiber_compare(manual, pos, conf)))
    # targeted manual corrections
    ov = fa.get("overrides") or {}
    rows = []
    for i in flagged:
        rows.append(dbc.Row([
            dbc.Col(html.Small(f"Fiber {i + 1} — {conf[i]}"), width=5),
            dbc.Col(dbc.Input(
                id={"type": "fib-ov", "index": i}, type="number", step=0.1,
                value=ov.get(str(i), round(float(pos[i]), 2))), width=3),
            dbc.Col(html.Small(f"detected at {fa['positions'][i]:.2f} px",
                               className="text-muted"), width=4),
        ], className="mb-1"))
    corr = html.Div([
        html.Small("Only correct here the flagged fibers (zooms above). "
                   "Any correction invalidates the cache and requires "
                   "re-running the calibration.",
                   className="text-muted d-block mb-2"),
        *rows,
        dbc.ButtonGroup([
            dbc.Button("Apply corrections", id="c-fib-apply",
                       color="primary", size="sm"),
            dbc.Button("Re-detect fibers", id="c-fib-redetect",
                       color="warning", outline=True, size="sm"),
        ]),
        html.Div(id="c-fib-msg", className="mt-2"),
    ]) if flagged else html.Div([
        dbc.ButtonGroup([dbc.Button("Re-detect fibers",
                                    id="c-fib-redetect", color="warning",
                                    outline=True, size="sm")]),
        html.Div(id="c-fib-msg", className="mt-2"),
        html.Div(dbc.Input(id={"type": "fib-ov", "index": -1},
                           style={"display": "none"})),
    ])
    badge = dbc.Alert(
        [html.B(f"{n_dir}/{len(conf)} fibers measured directly. "),
         (f"{len(flagged)} buried/interpolated fibers to check "
          f"(fibers {', '.join(str(i + 1) for i in flagged)}): review the "
          f"zooms and correct if needed."
          if flagged else "No doubtful fiber.")],
        color="success" if not flagged else "warning", className="py-2")
    return card("Automatic fiber detection", [badge] + figs + [corr],
                icon="bi-bullseye")


@callback(Output("c-fib-msg", "children"),
          Input("c-fib-apply", "n_clicks"),
          State({"type": "fib-ov", "index": dash.ALL}, "value"),
          State({"type": "fib-ov", "index": dash.ALL}, "id"),
          prevent_initial_call=True)
def apply_overrides(_, values, ids):
    from core import analysis as an
    fa = SESSION.fiber_auto
    if fa is None:
        return guard_alert("No detection in memory.")
    ov = {}
    for v, i in zip(values, ids):
        idx = i["index"]
        if idx < 0 or v is None:
            continue
        if abs(float(v) - fa["positions"][idx]) > 1e-6:
            ov[str(idx)] = float(v)
    an.set_fiber_overrides(ov)
    return dbc.Alert(
        f"{len(ov)} correction(s) saved. Re-run the calibration to take "
        f"them into account (the cache will be regenerated automatically).",
        color="success", className="py-2")


@callback(Output("c-fib-msg", "children", allow_duplicate=True),
          Input("c-fib-redetect", "n_clicks"),
          State("c-nscience", "value"),
          prevent_initial_call=True)
def redetect(_, nscience):
    from core import analysis as an
    ok, msg = an.run_fiber_detection(int(nscience or 5))
    return dbc.Alert(msg + (" Re-run the calibration." if ok else ""),
                     color="success" if ok else "danger", className="py-2")


@callback(Output("c-fib-overlay", "children"),
          Input("c-fib-overlay-btn", "n_clicks"),
          State("c-fib-shot", "value"),
          prevent_initial_call=True)
def fiber_overlay(_, shot):
    from core import analysis as an
    import numpy as _np
    pos = an.effective_fiber_positions()
    fa = SESSION.fiber_auto
    if pos is None or fa is None:
        return guard_alert("No detection available (manual mode or "
                           "detection not run).")
    try:
        if shot == "__hgar__" or shot not in SESSION.image_dict:
            arr_raw = sf.load_image(SESSION.hgar_path).astype(float)
            arr = sf._rotate_image(arr_raw,
                                   sf._measure_image_rotation(arr_raw))
            name = "HgAr"
        else:
            arr = an.load_shot_derotated(shot)
            name = shot
    except Exception as e:
        return guard_alert(f"Could not load: {e}", "danger")
    if arr.shape[0] < int(_np.max(pos)) + 5:
        return guard_alert(
            f"{name}: dimensions incompatible with the calibration.")
    return graph(plotting.fig_shot_fibers_overlay(
        arr, pos, fa["conf"],
        title=f"{name} (derotated) — 80 detected fibers overlaid"))
