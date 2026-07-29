"""Page 6 — Angular maps & 3D (replaces the matplotlib view + GIF)."""
from __future__ import annotations

import numpy as np
import dash
import dash_bootstrap_components as dbc
from dash import Input, Output, State, dcc, html
from core.uistate import callback

from app.components import (card, guard_alert, labeled, page_header,
                            graph, units_radio)
from core import analysis, plotting
from core import spectro_functions as sf
from core.session import SESSION


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


def _cfg_options():
    from core import angles as ang
    opts = [{"label": "Automatic (Final.xlsx, else pipeline shot ranges)",
             "value": "auto"}]
    for c in ang.known_configs(SESSION.angle_registry):
        suffix = "" if c.lower() in sf.FIBER_CONFIGS else " (imported from Excel)"
        opts.append({"label": c + suffix, "value": c})
    return opts


# Direction du faisceau : imposee par l'experience, jamais un reglage.
# Ce sont les valeurs d'origine, et elles etaient correctes. Dans la
# conversion du pipeline, y = cos(theta) : l'axe polaire est y, et l'angle a
# la retrodiffusion (+y) vaut exactement theta. Mes deplacements successifs de
# cette fleche (v29 vers x, v33 vers z) ne corrigeaient rien — ils
# masquaient le fait que les coordonnees, elles, etaient calculees avec une
# mauvaise formule.
LASER_ORIGIN = (0.0, 1.0, 0.0)
LASER_DIR = (0.0, -1.0, 0.0)


def _resolve_cfg(img, cfg_choice):
    """Returns (config, source_str | None)."""
    if cfg_choice and cfg_choice != "auto":
        return cfg_choice, "manual choice"
    return analysis.resolve_config(img)


def _parse_range(txt):
    toks = [t for t in str(txt or "").replace(";", ",").split(",")
            if t.strip()]
    if len(toks) == 2:
        try:
            a, b = float(toks[0]), float(toks[1])
            return (min(a, b), max(a, b))
        except ValueError:
            pass
    return None


def _shot_e2w(img):
    """2ω energy (J) of a shot, from the Excel shotbook. None if unknown."""
    sn = analysis.shot_num(img)
    if sn is None:
        return None
    if not SESSION.metadata:
        analysis.load_metadata()          # silencieux : absent = None
    m = SESSION.metadata.get(sn) or {}
    return m.get("e2w")


def _collected_energy(sp, wl_axis, wl_range=None):
    """Integrale spectrale par fibre puis somme sur les fibres.

    Contrairement a compute_spectral_area (somme brute des echantillons), on
    integre ici sur λ (trapezes) : en unites µJ/nm cela donne bien des µJ.
    Retourne (total, n_fibres_valides, par_fibre).
    """
    wl = np.asarray(wl_axis, float)
    y = np.asarray(sp, float)
    if wl_range is not None:
        m = (wl >= wl_range[0]) & (wl <= wl_range[1])
        wl, y = wl[m], y[:, m]
    if wl.size < 2:
        return np.nan, 0, np.full(y.shape[0], np.nan)
    trapz = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
    # Une fibre dont le spectre contient un NaN (p.ex. sans facteur de
    # calibration absolue) est exclue plutot que comblee : on ne fabrique
    # pas d'energie.
    per_fiber = trapz(y, wl, axis=1)
    ok = np.isfinite(per_fiber)
    return float(per_fiber[ok].sum()), int(ok.sum()), per_fiber


def _parse_ints(txt):
    out = []
    for t in str(txt or "").replace(";", ",").split(","):
        t = t.strip()
        if t:
            try:
                out.append(int(t))
            except ValueError:
                pass
    return out


def layout():
    default_img = _first_usable()
    return html.Div([
        page_header(
            "Angular maps & 3D view",
            "Angular distribution of the signal on the collection sphere.",
            "Step 7. The 3D view is directly mouse-controlled "
            "(rotation, zoom) — it replaces the notebook GIF. Displayed "
            "indices are extraction indices; the physical beam inversion is "
            "applied in the angle↔fiber mapping, as in your pipeline."),
        card("Selection", dbc.Row([
            labeled("Image", dcc.Dropdown(id="a-image", options=_img_options(),
                                          value=default_img), width=3),
            labeled("Angular configuration",
                    dcc.Dropdown(id="a-config", options=_cfg_options(),
                                 value="auto", clearable=False), width=3),
            labeled("λ range (nm, empty = max/full spectrum)",
                    dbc.Input(id="a-wlrange", placeholder="e.g. 725,875"),
                    width=3),
        ]), icon="bi-crosshair"),
        dbc.Tabs([
            dbc.Tab([
                dbc.Row([
                    labeled("Metric", dcc.Dropdown(
                        id="a2-metric", clearable=False, value="area",
                        options=[{"label": "Spectral area (sum)",
                                  "value": "area"},
                                 {"label": "Spectral centroid (nm)",
                                  "value": "centroid"}]), width=3),
                    labeled("RBF smoothing", dbc.Input(id="a2-smooth",
                                                     type="number", value=2.0,
                                                     step=0.5), width=2),
                    labeled("Coverage (°, empty = no mask)",
                            dbc.Input(id="a2-cov", type="number", value=100),
                            width=2),
                    dbc.Col(dbc.Button("Show", id="a2-run",
                                       color="primary", className="mt-4"),
                            width=2),
                ]),
                units_radio("a2-units"),
                dcc.Loading(html.Div(id="a2-out")),
            ], label="2D map (θ, φ)", tab_id="a-2d"),
            dbc.Tab([
                dbc.Row([
                    labeled("RBF smoothing", dbc.Input(id="a3-smooth",
                                                     type="number", value=0.5,
                                                     step=0.5), width=2),
                    labeled("Coverage (°)", dbc.Input(id="a3-cov",
                                                        type="number",
                                                        value=15.0), width=2),
                    labeled("Fibers to exclude, 1–80 (e.g. 1,2,3)",
                            dbc.Input(id="a3-excl"), width=2),
                    labeled("Geometry", dcc.Dropdown(
                        id="a3-geom", clearable=False, value="auto",
                        options=[
                            {"label": "x, y, z from the file (if available)",
                             "value": "auto"},
                            {"label": "Recomputed from φ, θ (pipeline "
                                      "convention)", "value": "angles"}]),
                            width=3),
                ]),
                # Convention de signe : quel angle inverser, pour quel
                # quadrant. Les options sont les quadrants REELLEMENT
                # presents dans la configuration choisie — rien d'invente,
                # et donc rien a changer dans le code pour une campagne qui
                # arriverait avec d'autres quadrants.
                dbc.Row([
                    labeled("Negate θ for the quadrant(s)",
                            dcc.Dropdown(id="a3-neg-theta", multi=True,
                                         options=[], value=[],
                                         placeholder="none"),
                            "Mounting convention (the pipeline rule, now "
                            "yours to set). Negating θ mirrors x and y, so "
                            "the quadrant moves to the other side of the "
                            "axis; z is unchanged, since cos(−θ) = cos(θ).",
                            width=4),
                    labeled("Negate φ for the quadrant(s)",
                            dcc.Dropdown(id="a3-neg-phi", multi=True,
                                         options=[], value=[],
                                         placeholder="none"),
                            "Same, for the azimuth: mirrors y only.",
                            width=4),
                ]),
                html.Div(id="a3-sign-note", className="small text-muted mb-2"),
                dbc.Row([
                    labeled("Fiber order", dcc.Dropdown(
                        id="a3-numbering", clearable=False, value="",
                        options=[
                            {"label": "as imported", "value": ""},
                            {"label": "reversed — fiber 1 of the file is the "
                                      "last detector trace",
                             "value": "physical"},
                            {"label": "not reversed", "value": "extraction"}]),
                        "A property of the mounting, not of the files: "
                        "November sends the fibers back reversed, May does "
                        "not. If the intensity gradient runs the wrong way "
                        "along each arm, it is this.", width=5),
                ]),

                dbc.Row([
                    labeled("Extrapolate the collected energy over the sphere",
                            dbc.Checklist(
                                id="a3-extrap", switch=True, inline=True,
                                value=[],
                                options=[
                                    {"label": " horizontally (φ → 360°)",
                                     "value": "phi"},
                                    {"label": " vertically (θ → 0–180°)",
                                     "value": "theta"}]),
                            help_txt="Both together give the 4π energy. Each "
                                     "fiber sees Ω = A_core/R²; outside the "
                                     "measured domain the radiant intensity "
                                     "is held at its nearest measured value.",
                            width=8),
                    dbc.Col(dbc.Checklist(
                        id="a3-opts", switch=True,
                        options=[{"label": " Log scale", "value": "log"},
                                 {"label": " Laser vector", "value": "laser"}],
                        value=["laser"], className="mt-4"), width=2),
                    dbc.Col(dbc.Button("Show", id="a3-run",
                                       color="primary", className="mt-4"),
                            width=1),
                ]),
                units_radio("a3-units"),
                dcc.Loading(html.Div(id="a3-out")),
            ], label="Interactive 3D sphere", tab_id="a-3d"),
            dbc.Tab([
                dbc.Row([
                    labeled("Target φ (°)", dbc.Input(id="ac-phi", type="number",
                                                     value=-10), width=2),
                    labeled("φ tolerance (°)", dbc.Input(id="ac-tol",
                                                         type="number",
                                                         value=1.0), width=2),
                    labeled("λ range (nm)", dbc.Input(id="ac-wl",
                                                      value="600,1000"),
                            width=2),
                    labeled("Background percentile", dbc.Input(id="ac-bg",
                                                         type="number",
                                                         value=10), width=2),
                    labeled("SG smoothing (nm)", dbc.Input(id="ac-sg",
                                                         type="number",
                                                         value=5.0), width=2),
                    dbc.Col(dbc.Button("Compute", id="ac-run",
                                       color="primary", className="mt-4"),
                            width=2),
                ]),
                dcc.Loading(html.Div(id="ac-out")),
            ], label="Centroid vs θ (fixed φ) + export", tab_id="a-cent"),
        ], className="mt-2"),
    ])


@callback(Output("a2-out", "children"),
          Input("a2-run", "n_clicks"),
          State("a-image", "value"), State("a-config", "value"),
          State("a-wlrange", "value"), State("a2-metric", "value"),
          State("a2-smooth", "value"), State("a2-cov", "value"),
          State("a2-units", "value"),
          prevent_initial_call=True)
def map2d(_, img, cfg_choice, wl_txt, metric, smooth, cov, units):
    if SESSION.calib is None:
        return guard_alert("Run the calibration first (step 2).")
    if not img:
        return guard_alert("Pick an image.")
    cfg, cfg_src = _resolve_cfg(img, cfg_choice)
    if cfg is None:
        return guard_alert(
            f"Undetermined angular configuration for {img} "
            f"({cfg_src}) — pick it manually.")
    bad = SESSION.unusable_images().get(img)
    if bad:
        return guard_alert(f"{img}: {bad}", "danger")
    try:
        sp = analysis.get_spectra(img)
    except Exception as e:
        return guard_alert(f"{img} cannot be extracted: {e}", "danger")
    to_uj = units == "uJ" and analysis.abs_cal_ready()
    wl_axis = SESSION.calib["wl_axis"]
    wl_range = _parse_range(wl_txt)
    phis, thetas = analysis.fiber_angles(cfg)
    if metric == "area":
        sp_a = analysis.to_absolute_energy(sp) if to_uj else sp
        values = sf.compute_spectral_area(sp_a, wl_axis, wl_range=wl_range)
        wl_str = (f"{wl_range[0]:.0f}–{wl_range[1]:.0f} nm" if wl_range
                  else "full spectrum")
        label = f"Signal sum ({wl_str}) [{'J' if to_uj else 'ADU'}]"
    else:
        wr = wl_range or (725, 875)
        values = sf.compute_spectral_centroid(sp, wl_axis, wl_range=wr,
                                              bg_percentile=10,
                                              savgol_window_nm=5.0)
        label = f"Spectral centroid {wr[0]:.0f}–{wr[1]:.0f} nm [nm]"
    fig = plotting.fig_angular_map(
        phis, thetas, np.asarray(values, float), label=label,
        title=f"{img} ({cfg}, source: {cfg_src})",
        smoothing=float(smooth or 2.0),
        coverage_deg=(float(cov) if cov not in (None, "") else None))
    SESSION.log_history("angular_map_2d",
                        {"image": img, "config": cfg, "metric": metric,
                         "wl_range": wl_range})
    v = np.asarray(values, float)
    ok = np.isfinite(v)
    stats = html.P([
        dbc.Badge(f"{ok.sum()}/80 valid fibers", color="primary",
                  className="me-2"),
        dbc.Badge(f"median: {np.median(v[ok]):.4g}", color="secondary",
                  className="me-2"),
        dbc.Badge(f"min: {np.nanmin(v):.4g} "
                  f"(fiber {int(np.nanargmin(v)) + 1})",
                  color="secondary", className="me-2"),
        dbc.Badge(f"max: {np.nanmax(v):.4g} "
                  f"(fiber {int(np.nanargmax(v)) + 1})",
                  color="secondary"),
    ])
    return html.Div([stats, graph(fig)])


@callback(Output("a3-neg-theta", "options"), Output("a3-neg-phi", "options"),
          Output("a3-neg-theta", "value"), Output("a3-neg-phi", "value"),
          Output("a3-sign-note", "children"),
          Input("a-config", "value"), Input("a-image", "value"))
def sign_options(cfg_choice, img):
    """Propose les quadrants effectivement detectes dans la configuration.

    Une configuration codee en dur dans le pipeline garde sa convention
    validee : on le dit plutot que d'offrir un reglage sans effet.
    """
    from core import angles as ang
    cfg, _src = _resolve_cfg(img, cfg_choice)
    if cfg is None:
        return [], [], [], [], "Pick a configuration to see its quadrants."
    # Ordre des arguments : quadrants_of(registry, name) — l'inverser
    # renvoyait silencieusement une liste vide (menus toujours vides).
    try:
        quads = ang.quadrants_of(SESSION.angle_registry, cfg)
    except Exception:
        quads = []
    if not quads:
        return [], [], [], [], ("'%s' carries no quadrant information "
                                "(pipeline configuration, or file imported "
                                "before this version): its validated sign "
                                "convention is kept." % cfg)
    opts = [{"label": q.capitalize(), "value": q} for q in quads]
    # Les menus doivent AFFICHER la regle de la configuration, pas partir
    # vides : sinon le trace utilisait « aucune inversion » alors que la
    # configuration en demandait une, et bouger le menu semblait sans effet
    # puisqu'on comparait a un etat deja faux.
    rules = ang.entry(SESSION.angle_registry.get(cfg, {})).get("sign_rules") \
        or {}
    v_th = [q for q in (rules.get("theta") or []) if q in quads]
    v_ph = [q for q in (rules.get("phi") or []) if q in quads]
    note = "Quadrants detected in '%s': %s." % (cfg, ", ".join(quads))
    if v_th or v_ph:
        note += " Rule from the configuration file, shown in the menus."
    return opts, opts, v_th, v_ph, note


@callback(Output("a3-out", "children"),
          Input("a3-run", "n_clicks"),
          State("a-image", "value"), State("a-config", "value"),
          State("a-wlrange", "value"), State("a3-smooth", "value"),
          State("a3-cov", "value"), State("a3-excl", "value"),
          State("a3-opts", "value"), State("a3-units", "value"),
          State("a3-geom", "value"), State("a3-extrap", "value"),
          State("a3-neg-theta", "value"), State("a3-neg-phi", "value"),
          State("a3-numbering", "value"),
          prevent_initial_call=True)
def sphere(_, img, cfg_choice, wl_txt, smooth, cov, excl_txt, opts, units,
           geom, extrap, neg_theta, neg_phi, numb):
    if SESSION.calib is None:
        return guard_alert("Run the calibration first (step 2).")
    if not img:
        return guard_alert("Pick an image.")
    cfg, cfg_src = _resolve_cfg(img, cfg_choice)
    if cfg is None:
        return guard_alert(
            f"Undetermined angular configuration for {img} "
            f"({cfg_src}) — pick it manually.")
    bad = SESSION.unusable_images().get(img)
    if bad:
        return guard_alert(f"{img}: {bad}", "danger")
    try:
        sp = analysis.get_spectra(img)
    except Exception as e:
        return guard_alert(f"{img} cannot be extracted: {e}", "danger")
    to_uj = units == "uJ" and analysis.abs_cal_ready()
    if to_uj:
        sp = analysis.to_absolute_energy(sp)
    unit_label = "J/nm" if to_uj else "ADU"
    sign_rules = {"theta": list(neg_theta or []), "phi": list(neg_phi or [])}
    from core import angles as _ang
    phis, thetas = _ang.get_fiber_angles_any(
        cfg, SESSION.angle_registry, sign_rules=sign_rules,
        numbering=(numb or None)) if cfg in SESSION.angle_registry \
        else analysis.fiber_angles(cfg, sign_rules=sign_rules)
    excl = [f - 1 for f in _parse_ints(excl_txt)]   # 1-80 -> internal
    wl_range = _parse_range(wl_txt)

    # Geometrie : coordonnees derivees des angles SIGNES, pour que la sphere
    # et la carte 2D racontent la meme chose.
    xyz, geom_note = None, "recomputed from (φ, θ), pipeline convention"
    if (geom or "auto") == "auto":
        from core import angles as ang
        try:
            xyz = ang.get_fiber_xyz_any(cfg, SESSION.angle_registry,
                                        sign_rules=sign_rules,
                                        numbering=(numb or None))
        except Exception:
            xyz = None
        if xyz is not None:
            geom_note = "x, y, z recomputed from the reference angles"
    if xyz is None:
        # Ne jamais laisser fig_sphere retomber sur sa formule interne, qui
        # prend z pour axe polaire : c'est exactement ce qui inversait
        # l'ordre des ports.
        from core import angles as ang
        _ent = ang.entry(SESSION.angle_registry.get(cfg, {}))
        cx, cy, cz = ang.xyz_from_angles(phis, thetas,
                                         _ent.get("polar_axis"))
        xyz = np.vstack([cx, cy, cz])
        geom_note = "x, y, z recomputed from (φ, θ) about the beam axis"
    inv = [f"θ<0 on {', '.join(sign_rules['theta'])}"] if sign_rules["theta"] else []
    if sign_rules["phi"]:
        inv.append(f"φ<0 on {', '.join(sign_rules['phi'])}")
    if inv:
        geom_note += " — sign convention: " + " ; ".join(inv)
    geom_note += (" — pipeline conversion (sf._fiber_angles_to_xyz): "
                  "backscatter at +y, and the angle to it is exactly θ")
    try:
        from core import angles as _a
        _e = _a.entry(SESSION.angle_registry.get(cfg, {}))
        if _e.get("fibres"):
            geom_note += (" — fiber order: "
                          + ("reversed" if str(numb or _e["numbering"])
                             .startswith("phys") else "not reversed"))
    except Exception:
        pass

    fig = plotting.fig_sphere(
        sp, phis, thetas, SESSION.calib["wl_axis"],
        wl_range=wl_range,
        smoothing=float(smooth if smooth is not None else 0.5),
        coverage_deg=float(cov if cov is not None else 15.0),
        log_scale="log" in (opts or []),
        exclude_fibers=excl,
        show_laser="laser" in (opts or []),
        laser_origin=LASER_ORIGIN, laser_dir=LASER_DIR,
        unit_label=unit_label, xyz=xyz,
        title=f"{img} ({cfg}, {cfg_src}) — drag to rotate, "
              f"scroll to zoom")

    # ── Bilan d'energie ────────────────────────────────────────────────
    sp_e = sp.copy()
    if excl:
        for i in excl:
            if 0 <= i < sp_e.shape[0]:
                sp_e[i] = np.nan
    total, n_ok, per_fiber = _collected_energy(
        sp_e, SESSION.calib["wl_axis"], wl_range)
    e2w = _shot_e2w(img)
    unit = "J" if to_uj else "ADU·nm"
    fmt = "{:.4e}" if to_uj else "{:.4g}"
    wl_str = (f"{wl_range[0]:.0f}–{wl_range[1]:.0f} nm" if wl_range
              else "full spectrum")
    rows = [
        dbc.Badge(f"Collected by the fibers: {fmt.format(total)} {unit}",
                  color="primary", className="me-2"),
        dbc.Badge(f"{n_ok} fibers summed ({wl_str})", color="secondary",
                  className="me-2"),
        dbc.Badge(f"E_2ω = {e2w:.4e} J" if e2w is not None
                  else "E_2ω unknown (shotbook)",
                  color="success" if e2w is not None else "warning",
                  className="me-2"),
    ]

    # ── Extrapolation par angle solide ─────────────────────────────────
    ext = extrap or []
    note_ext = None
    res = analysis.extrapolate_sphere_energy(
        per_fiber, phis, thetas, extend_phi="phi" in ext,
        extend_theta="theta" in ext)
    if res.get("error"):
        note_ext = html.Small(res["error"], className="text-warning")
    else:
        E_ext = res["energy_extrapolated"]
        om_dom = res["solid_angle_sr"]
        label = ("4π" if res["is_4pi"] else
                 "φ-extended" if res["extend_phi"] else
                 "θ-extended" if res["extend_theta"] else
                 "measured cone")
        rows.append(html.Br())
        rows.append(dbc.Badge(f"Extrapolated ({label}): "
                              f"{fmt.format(E_ext)} {unit} "
                              f"over {om_dom:.3f} sr", color="info",
                              className="me-2"))
        if e2w and E_ext and to_uj and e2w > 0:
            rows.append(dbc.Badge(
                f"E_{label} / E_2ω = {E_ext / e2w:.3e}",
                color="dark", className="me-2"))
        if res.get("n_negative_fibers"):
            rows.append(dbc.Badge(
                f"{res['n_negative_fibers']} fibers had negative energy "
                f"(background over-subtraction) — clipped to 0 before "
                f"extrapolation", color="warning", className="me-2"))
        ph0, ph1 = res["phi_range"]
        th0, th1 = res["theta_range"]
        note_ext = html.Small(
            f"Each fiber subtends Ω = A_core/R² = {res['omega_sr']:.3e} sr "
            f"(core {analysis.abs_cal_params()['core_um']:.0f} µm at "
            f"{analysis.abs_cal_params()['sphere_radius_cm']:.0f} cm). The "
            f"fibers cover φ ∈ [{ph0:.0f}, {ph1:.0f}]°, θ ∈ [{th0:.0f}, "
            f"{th1:.0f}]°, i.e. {res['solid_angle_measured_sr']:.3f} sr. "
            f"Outside that domain the radiant intensity is held at its "
            f"nearest measured value — this is an assumption of flat "
            f"emission in the extended direction, not a measurement.",
            className="text-muted")

    badges = html.P(rows + [html.Br(), note_ext, html.Br(),
                            html.Small(f"Geometry: {geom_note}.",
                                       className="text-muted")],
                    className="mb-2")
    SESSION.log_history("sphere_3d", {
        "image": img, "config": cfg, "units": "uJ" if to_uj else "adu",
        "collected": total, "e2w": e2w,
        "extrapolated": res.get("energy_extrapolated"),
        "extend": list(ext)})
    # uirevision : plotly conserve la camera de l'utilisateur quand la figure
    # est renvoyee au navigateur, au lieu de revenir a la vue initiale.
    fig.update_layout(uirevision="sphere3d",
                      scene=dict(uirevision="sphere3d"))
    return html.Div([badges, graph(fig, id="a3-graph")])


@callback(Output("a3-graph", "figure", allow_duplicate=True),
          Input("a3-graph", "relayoutData"),
          State("a3-graph", "figure"),
          prevent_initial_call=True)
def keep_camera(relayout, fig):
    """Recopie l'orientation courante dans la figure elle-meme.

    Le bouton « telecharger » de Plotly re-dessine la figure hors ecran a
    partir de la mise en page DECLAREE, pas de l'etat interactif : sans cela
    l'image exportee garde toujours l'angle de vue initial, quelle que soit
    la rotation appliquee a l'ecran. On ecrit donc la camera dans la figure a
    chaque rotation ; `uirevision` empeche l'aller-retour de faire sauter la
    vue.
    """
    if not fig or not isinstance(relayout, dict):
        return dash.no_update
    cam = relayout.get("scene.camera")
    if cam is None:
        return dash.no_update
    fig.setdefault("layout", {}).setdefault("scene", {})["camera"] = cam
    return fig


@callback(Output("ac-out", "children"),
          Input("ac-run", "n_clicks"),
          State("a-image", "value"), State("a-config", "value"),
          State("ac-phi", "value"), State("ac-tol", "value"),
          State("ac-wl", "value"), State("ac-bg", "value"),
          State("ac-sg", "value"),
          prevent_initial_call=True)
def centroid_profile(_, img, cfg_choice, phi, tol, wl_txt, bg_pct, sg):
    if SESSION.calib is None:
        return guard_alert("Run the calibration first (step 2).")
    if not img:
        return guard_alert("Pick an image.")
    wl_range = _parse_range(wl_txt) or (600, 1000)
    cfg, _src = _resolve_cfg(img, cfg_choice)
    try:
        res = analysis.centroid_theta_profile(
            img, float(phi or 0), float(tol or 1.0), wl_range=wl_range,
            bg_percentile=int(bg_pct or 10), sg_window_nm=float(sg or 5.0),
            config_name=cfg)
    except ValueError as e:
        return guard_alert(str(e))
    # Exports identical to the notebook (npz + txt) in the workspace
    out_dir = SESSION.outputs_dir() / "theta_profiles"
    npz, txt = analysis.export_centroid_npz_txt(res, out_dir)
    rel_npz = npz.relative_to(SESSION.ensure_workspace()).as_posix()
    rel_txt = txt.relative_to(SESSION.ensure_workspace()).as_posix()
    SESSION.log_history("centroid_theta_profile", {
        "image": img, "config": res["config"], "phi": res["phi_target"],
        "wl_range": list(wl_range), "export": npz.name})
    tbl = dbc.Table([
        html.Thead(html.Tr([html.Th(h) for h in
                            ["Fiber (extr., 1–80)", "Fiber (phys., 1–80)",
                             "θ (°)", "λ_c (nm)", "λ_c − min (nm)"]])),
        html.Tbody([html.Tr([html.Td(int(a) + 1), html.Td(int(b) + 1),
                             html.Td(f"{c:.3f}"), html.Td(f"{d:.4f}"),
                             html.Td(f"{e:.4f}")])
                    for a, b, c, d, e in zip(res["fiber_idx"],
                                             res["phys_fiber"], res["theta"],
                                             res["centroid"],
                                             res["lambda_c_norm"])]),
    ], size="sm", hover=True, striped=True)
    return html.Div([
        html.P([
            dbc.Badge(f"{len(res['theta'])} fibers at φ = "
                      f"{res['phi_target']}±{res['phi_tol']}°",
                      color="primary", className="me-2"),
            dbc.Badge(f"λ_c,min = {res['lambda_c_min']:.3f} nm",
                      color="secondary", className="me-2"),
            dbc.Badge(f"max−min spread = "
                      f"{float(np.max(res['lambda_c_norm'])):.3f} nm",
                      color="secondary"),
        ]),
        dbc.Row([
            dbc.Col(graph(plotting.fig_centroid_theta(res)), md=6),
            dbc.Col(graph(plotting.fig_selected_spectra(res)),
                    md=6),
        ]),
        dbc.Alert([
            html.I(className="bi bi-download me-2"),
            "Exported files (formats identical to the original pipeline): ",
            html.A(npz.name, href=f"/download/{rel_npz}", className="me-3"),
            html.A(txt.name, href=f"/download/{rel_txt}"),
            html.Small(" — in these files, fiber_idx and phys_fiber keep "
                       "the 0–79 convention (compatibility with the pipeline "
                       "and your existing tools).", className="text-muted"),
        ], color="light", className="border"),
        dbc.Accordion([dbc.AccordionItem(tbl, title="Centroid table")],
                      start_collapsed=True),
    ])
