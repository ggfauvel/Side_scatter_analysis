"""
app/main.py — Dash application: shell, side navigation, routing.
"""
from __future__ import annotations

from pathlib import Path

import dash
import dash_bootstrap_components as dbc
from dash import Input, Output, State, dcc, html
from flask import abort, send_file

from core.session import SESSION
from core import uistate

app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.FLATLY, dbc.icons.BOOTSTRAP],
    suppress_callback_exceptions=True,
    title="Spectro Multi-Fiber",
)
server = app.server

# Pages (importing them registers the callbacks)
from app.pages import (  # noqa: E402
    p1_data, p2_calibration, p3_extraction, p4_explore,
    p5_groups, p6_angular, p7_correlations, p8_exports, p9_abscal,
)

NAV = [
    ("/", "bi-folder2-open", "1. Data", p1_data),
    ("/calibration", "bi-sliders", "2. Calibration", p2_calibration),
    ("/absolute", "bi-rulers", "3. Absolute calibration", p9_abscal),
    ("/extraction", "bi-cpu", "4. Extraction", p3_extraction),
    ("/exploration", "bi-search", "5. Exploration", p4_explore),
    ("/groupes", "bi-collection", "6. Groups & comparisons", p5_groups),
    ("/angulaire", "bi-globe2", "7. Angular maps & 3D", p6_angular),
    ("/correlations", "bi-graph-up", "8. Laser correlations", p7_correlations),
    ("/exports", "bi-box-arrow-down", "9. Exports & history", p8_exports),
]

sidebar = html.Div(
    [
        html.Div([
            html.I(className="bi bi-stars me-2", style={"fontSize": "1.4rem"}),
            html.Span("Spectro", style={"fontWeight": 700, "fontSize": "1.25rem"}),
            html.Span(" Multi-Fiber", style={"fontWeight": 300,
                                              "fontSize": "1.05rem"}),
        ], className="px-3 py-3 text-white"),
        html.Hr(className="text-white-50 my-0"),
        dbc.Nav(
            [dbc.NavLink([html.I(className=f"bi {ico} me-2"), lab],
                         href=href, active="exact", className="text-white-75")
             for href, ico, lab, _ in NAV],
            vertical=True, pills=True, className="p-2",
        ),
        html.Div([
            html.Div(id="sidebar-status"),
            dbc.Button([html.I(className="bi bi-arrow-counterclockwise me-1"),
                        "Clear the displayed views"],
                       id="reset-views", size="sm", color="light",
                       outline=True, className="mt-2 w-100"),
        ], className="px-3 mt-auto pb-3"),
    ],
    style={
        "position": "fixed", "top": 0, "left": 0, "bottom": 0, "width": "17rem",
        "background": "linear-gradient(180deg,#1d3557 0%,#274b73 100%)",
        "display": "flex", "flexDirection": "column", "overflowY": "auto",
    },
)

content = html.Div(id="page-content",
                   style={"marginLeft": "17rem", "padding": "1.4rem 2rem"})

app.layout = html.Div([
    dcc.Location(id="url"),
    dcc.Interval(id="status-interval", interval=3000),
    sidebar,
    content,
])


@app.callback(Output("page-content", "children"), Input("url", "pathname"))
def route(pathname):
    for href, _, _, mod in NAV:
        if pathname == href:
            # Le layout est reconstruit (les listes deroulantes restent a
            # jour), puis l'etat memorise — reglages ET graphiques deja
            # calcules — est reinjecte par-dessus. Changer de page ne fait
            # donc plus perdre le travail affiche.
            return uistate.restore(mod.layout())
    return dbc.Alert("Unknown page.", color="warning")


@app.callback(Output("page-content", "children", allow_duplicate=True),
              Input("reset-views", "n_clicks"), State("url", "pathname"),
              prevent_initial_call=True)
def reset_views(n, pathname):
    """Vide les resultats memorises et reaffiche la page courante."""
    if not n:
        return dash.no_update
    uistate.forget()
    for href, _, _, mod in NAV:
        if pathname == href:
            return mod.layout()
    return dash.no_update


@app.callback(Output("sidebar-status", "children"),
              Input("status-interval", "n_intervals"))
def sidebar_status(_):
    s = SESSION
    def dot(ok, label):
        color = "#7CFC90" if ok else "#ffb3b3"
        return html.Div([
            html.Span("●", style={"color": color, "marginRight": "6px"}),
            html.Small(label, className="text-white-75"),
        ])
    from core import jobs
    items = [
        dot(bool(s.image_dict), f"Images: {len(s.image_dict)}"),
        dot(s.calib is not None, "Calibration"
            + (f" ({s.calib_hash()})" if s.calib is not None else "")),
        dot(bool(s.energy_table), f"Energies: {len(s.energy_table)}"),
    ]
    if jobs.PROGRESS["running"]:
        items.append(html.Div([
            html.Span("⟳ ", style={"color": "#ffd166"}),
            html.Small(f"Batch {jobs.PROGRESS['done']}/{jobs.PROGRESS['total']}",
                       className="text-white-75"),
        ]))
    return items


# ── Workspace file download ─────────────────────────────────
@server.route("/download/<path:relpath>")
def download(relpath):
    ws = Path(SESSION.ensure_workspace()).resolve()
    target = (ws / relpath).resolve()
    if not str(target).startswith(str(ws)) or not target.is_file():
        abort(404)
    return send_file(target, as_attachment=True)
