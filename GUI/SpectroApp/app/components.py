"""app/components.py — Petits composants UI reutilisables."""
from __future__ import annotations

import dash_bootstrap_components as dbc
from dash import dcc, html

# Barre d'outils Plotly : édition des titres/légendes au clic, export PNG HD
GRAPH_CONFIG = {
    "editable": True,
    "edits": {"titleText": True, "axisTitleText": True, "legendText": True,
              "annotationText": True, "colorbarTitleText": True},
    "displaylogo": False,
    "toImageButtonOptions": {"format": "png", "scale": 3},
    "modeBarButtonsToAdd": ["toggleSpikelines"],
}


def graph(fig, **kwargs):
    """Standard dcc.Graph of the application: applies the global display
    preferences (font size, theme) and enables click-to-edit (title, axis
    titles, legends, annotations)."""
    from core.plotting import apply_prefs
    return dcc.Graph(figure=apply_prefs(fig), config=GRAPH_CONFIG, **kwargs)


def page_header(title, subtitle=None, help_text=None):
    kids = [html.H3(title, className="mb-1")]
    if subtitle:
        kids.append(html.P(subtitle, className="text-muted mb-1"))
    if help_text:
        kids.append(dbc.Alert([html.I(className="bi bi-info-circle me-2"),
                               help_text],
                              color="light", className="border py-2 my-2",
                              style={"fontSize": "0.9rem"}))
    return html.Div(kids, className="mb-3")


def card(title, body, icon=None, color=None):
    head = []
    if icon:
        head.append(html.I(className=f"bi {icon} me-2"))
    head.append(title)
    return dbc.Card([
        dbc.CardHeader(head, className=f"fw-semibold"
                       + (f" text-{color}" if color else "")),
        dbc.CardBody(body),
    ], className="mb-3 shadow-sm")


def labeled(label, control, help_txt=None, width=None):
    kids = [dbc.Label(label, className="fw-semibold mb-1",
                      style={"fontSize": "0.85rem"}), control]
    if help_txt:
        kids.append(html.Small(help_txt, className="text-muted"))
    div = html.Div(kids, className="mb-2")
    return dbc.Col(div, width=width) if width else div


def status_badge(ok, text_ok, text_ko):
    return dbc.Badge(text_ok if ok else text_ko,
                     color="success" if ok else "danger", className="me-1")


def guard_alert(msg, color="warning"):
    return dbc.Alert([html.I(className="bi bi-exclamation-triangle me-2"), msg],
                     color=color)


def units_radio(radio_id):
    """Inline ADU / µJ-per-nm switch for spectra plots. The energy option is
    only enabled once an absolute calibration exists. Reflects and updates the
    global SESSION.display_units so the choice carries across pages."""
    import dash_bootstrap_components as dbc
    from core import analysis
    from core.session import SESSION
    ready = analysis.display_units_available()
    value = SESSION.display_units if ready else "adu"
    ctrl = dbc.RadioItems(
        id=radio_id, inline=True, value=value,
        options=[{"label": " ADU", "value": "adu"},
                 {"label": " J/nm", "value": "uJ", "disabled": not ready}],
    )
    hint = (None if ready else
            dbc.FormText("Do the absolute calibration (step 3) to enable "
                         "physical units.", className="text-muted"))
    return html.Div([html.Small("Units: ", className="me-2 text-muted"),
                     ctrl, hint], className="d-flex align-items-center gap-2 "
                                            "mb-2 flex-wrap")
