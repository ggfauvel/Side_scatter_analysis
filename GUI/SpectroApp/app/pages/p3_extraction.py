"""Page 3 — Extraction: process one image or the whole campaign."""
from __future__ import annotations

from pathlib import Path

import dash
import dash_bootstrap_components as dbc
from dash import Input, Output, State, dcc, html, no_update, ALL
from core.uistate import callback

from app.components import card, guard_alert, labeled, page_header
from core import analysis, jobs
from core.session import SESSION


def layout():
    s = SESSION
    n_img = len(s.image_dict)
    n_cached = len(analysis.cached_shots()) if s.calib is not None else 0
    return html.Div([
        page_header(
            "Spectrum extraction",
            "Turns every detector image into 80 calibrated spectra.",
            "Step 4. The full processing runs in the background: you "
            "can keep using the application meanwhile. Already processed "
            "images (cache) are skipped automatically; the cache is "
            "invalidated if you change the calibration."),
        dbc.Row([
            dbc.Col(card("Status", [
                html.P([dbc.Badge(f"{n_img} images", color="primary",
                                  className="me-2"),
                        dbc.Badge(f"{n_cached} cached", color="success",
                                  className="me-2"),
                        dbc.Badge("calibration: "
                                  + (s.calib_hash() if s.calib is not None
                                     else "missing"),
                                  color="secondary")]),
                html.Small("The cache stores raw spectra; the intensity "
                           "calibration is applied on the fly, as in your "
                           "original pipeline.",
                           className="text-muted"),
            ], icon="bi-hdd"), md=5),
            dbc.Col(card("Start a run", [
                labeled("Images to process",
                        dcc.Dropdown(id="e-scope", clearable=False, value="all",
                                     options=[
                                         {"label": "All images in the folder",
                                          "value": "all"},
                                         {"label": "Only images from the defined groups",
                                          "value": "groups"},
                                         {"label": "Number range…",
                                          "value": "range"}])),
                dbc.Row([
                    labeled("From shot #", dbc.Input(id="e-from", type="number",
                                                    value=0, min=0), width=4),
                    labeled("to shot #", dbc.Input(id="e-to", type="number",
                                                    value=546, min=0), width=4),
                    dbc.Col(dbc.Checklist(
                        id="e-csv", switch=True,
                        options=[{"label": " CSV export per image",
                                  "value": "csv"},
                                 {"label": " Ignore the cache (re-extract everything)",
                                  "value": "force"}],
                        value=["csv"], className="mt-4"), width=4),
                ]),
                dbc.ButtonGroup([
                    dbc.Button([html.I(className="bi bi-play-fill me-1"),
                                "Start"], id="e-start", color="primary"),
                    dbc.Button([html.I(className="bi bi-stop-fill me-1"),
                                "Stop"], id="e-cancel", color="danger",
                               outline=True),
                ]),
                html.Div(id="e-start-msg", className="mt-2"),
            ], icon="bi-cpu"), md=7),
        ]),
        card("Cache management", [
            html.Small("The cache is organised per calibration (one folder "
                       "per parameter fingerprint). Clear it to start from "
                       "scratch — for instance after replacing wrong images "
                       "with the right ones: same file names = same cache "
                       "entries, so a re-extraction is needed (or tick "
                       "'Ignore the cache').",
                       className="text-muted d-block mb-2"),
            html.Div(id="e-cache-list"),
            dbc.ButtonGroup([
                dbc.Button("Clear the current cache", id="e-clear-current",
                           color="danger", outline=True, size="sm"),
                dbc.Button("Delete older caches", id="e-clear-others",
                           color="warning", outline=True, size="sm"),
                dbc.Button("Delete everything", id="e-clear-all",
                           color="danger", size="sm"),
                dbc.Button("Refresh", id="e-cache-refresh",
                           color="secondary", outline=True, size="sm"),
            ], className="mt-2"),
            html.Div(id="e-cache-msg", className="mt-2"),
        ], icon="bi-trash3"),
        card("Progress", [
            dcc.Interval(id="e-interval", interval=800),
            dbc.Progress(id="e-progress", value=0, striped=True, animated=True,
                         style={"height": "22px"}),
            html.Div(id="e-progress-txt", className="mt-2"),
        ], icon="bi-hourglass-split"),
    ])


@callback(Output("e-start-msg", "children"),
          Input("e-start", "n_clicks"),
          State("e-scope", "value"), State("e-from", "value"),
          State("e-to", "value"), State("e-csv", "value"),
          prevent_initial_call=True)
def start(_, scope, n_from, n_to, csv_opt):
    s = SESSION
    if s.calib is None:
        return guard_alert("Run the calibration first (step 2).")
    if not s.image_dict:
        s.scan_images()
    keys = sorted(s.image_dict)
    if scope == "range":
        keys = [k for k in keys
                if (n_from or 0) <= analysis.shot_num(k) <= (n_to or 10**9)]
    elif scope == "groups":
        from core.campaign import load_groups
        wanted = {sh for g in load_groups(s.ensure_workspace())
                  for sh in g["shots"]}
        keys = [k for k in keys if k in wanted]
    ok, msg = jobs.start_batch(keys, export_csv="csv" in (csv_opt or []),
                               ignore_cache="force" in (csv_opt or []))
    return dbc.Alert(msg, color="success" if ok else "warning", className="py-2")


@callback(Output("e-progress", "value"), Output("e-progress", "label"),
          Output("e-progress-txt", "children"),
          Input("e-interval", "n_intervals"))
def progress(_):
    p = jobs.PROGRESS
    if p["total"] == 0:
        return 0, "", html.Small("No run started.",
                                 className="text-muted")
    pct = int(100 * p["done"] / p["total"])
    txt = [html.Span(f"{p['done']} / {p['total']} images — "
                     f"{p['skipped']} already cached. ")]
    if p["running"]:
        txt.append(html.Span(f"Processing: {p['current']}",
                             className="text-primary"))
    elif p["finished_msg"]:
        txt.append(html.Span(p["finished_msg"], className="fw-semibold"))
    if p["errors"]:
        # Group identical errors (ignoring the shot name) for readability
        from collections import Counter
        import re as _re
        patterns = Counter(_re.sub(r"shot\d+", "shotNNN", e)
                           for e in p["errors"])
        items = [html.Li(f"{msg}  (×{n})" if n > 1 else msg)
                 for msg, n in patterns.most_common(6)]
        txt.append(dbc.Alert([html.B(f"{len(p['errors'])} errors: "),
                              html.Ul(items)],
                             color="warning", className="mt-2 py-2"))
    return pct, f"{pct}%", html.Div(txt)


@callback(Output("e-start-msg", "children", allow_duplicate=True),
          Input("e-cancel", "n_clicks"), prevent_initial_call=True)
def cancel(_):
    jobs.cancel()
    return dbc.Alert("Stop requested — processing halts after the "
                     "current image.", color="info", className="py-2")


@callback(Output("e-cache-list", "children"),
          Input("e-cache-refresh", "n_clicks"),
          Input("e-cache-msg", "children"))
def cache_list(_, __):
    inv = jobs.cache_inventory()
    if not inv:
        return html.Small("No cache yet.", className="text-muted")
    rows = []
    for item in inv:
        m = item.get("meta") or {}
        badges = []
        if item["in_use"]:
            badges.append(dbc.Badge("in use", color="primary",
                                    className="ms-2"))
        elif item["current"]:
            badges.append(dbc.Badge("current calibration", color="success",
                                    className="ms-2"))
        if m and not item["same_campaign"]:
            badges.append(dbc.Badge("other campaign", color="warning",
                                    className="ms-2"))
        origin = " · ".join(x for x in [
            m.get("hgar", ""), Path(m["images_dir"]).name
            if m.get("images_dir") else "", m.get("created", "")] if x)
        rows.append(html.Tr([
            html.Td([item["name"]] + badges
                    + ([html.Br(), html.Small(origin,
                                              className="text-muted")]
                       if origin else
                       [html.Br(), html.Small("no provenance recorded "
                                              "(cache older than v30)",
                                              className="text-muted")])),
            html.Td(f"{item['n']} spectra"),
            html.Td(f"{item['size_mb']:.0f} MB"),
            html.Td(dbc.Button("Use this one", size="sm", outline=True,
                               color="primary",
                               id={"type": "e-use-cache",
                                   "path": str(item["path"])})
                    if item["n"] and not item["in_use"] else ""),
        ]))
    return dbc.Table([html.Thead(html.Tr([html.Th("Folder"), html.Th("Content"),
                                          html.Th("Size"), html.Th("")])),
                      html.Tbody(rows)], size="sm", className="mb-0")


@callback(Output("e-cache-msg", "children", allow_duplicate=True),
          Input({"type": "e-use-cache", "path": ALL}, "n_clicks"),
          prevent_initial_call=True)
def use_cache(clicks):
    """Adopter un cache existant plutot que de tout re-extraire."""
    from dash import ctx
    if not ctx.triggered_id or not any(c for c in (clicks or [])):
        return dash.no_update
    msg = jobs.adopt_cache(ctx.triggered_id["path"])
    return dbc.Alert(msg, color="info", className="py-2")


@callback(Output("e-cache-msg", "children"),
          Input("e-clear-current", "n_clicks"),
          Input("e-clear-others", "n_clicks"),
          Input("e-clear-all", "n_clicks"),
          prevent_initial_call=True)
def clear_cache(n1, n2, n3):
    from dash import ctx
    which = {"e-clear-current": "current", "e-clear-others": "others",
             "e-clear-all": "all"}[ctx.triggered_id]
    msg = jobs.clear_cache(which)
    return dbc.Alert(msg, color="info", className="py-2")
