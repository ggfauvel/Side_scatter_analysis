"""Page 5 — Groups & comparisons (Si %, pulse profiles)."""
from __future__ import annotations

import numpy as np
import dash_bootstrap_components as dbc
from dash import Input, Output, State, dcc, html, no_update
from core.uistate import callback

from app.components import (card, guard_alert, labeled, page_header,
                            graph, units_radio)
from core import analysis, plotting
from core import spectro_functions as sf
from core.campaign import DEFAULT_GROUPS, PROFILE_ORDER, load_groups, save_groups
from core.session import SESSION


def _gfac(units):
    """(calibration matrix, unit label) for the chosen display units.

    The matrix is (n_fibers, n_wavelengths): the conversion depends on λ as
    well as on the fiber, so a single scalar per fiber would flatten the
    spectral response. Returns (None, 'ADU') in ADU mode.

    UNITS — the trap this page fell into. `analysis.abs_cal_matrix()` returns
    µJ/(ADU·nm), and every other page converts with the ×1e-6 that
    `analysis.to_absolute_energy()` applies before plotting J/nm. This page
    used the matrix raw while labelling the axis J/nm, so every value here was
    10⁶ too large: a correct spectrum peaking at 70 pJ/nm was drawn as "70µ".
    The conversion below is what makes this page read the same as pages 5 and
    7 (ticks in "p", not in "µ").
    """
    if units == "uJ" and analysis.abs_cal_ready():
        C = analysis.abs_cal_matrix()
        if C is not None:
            return np.asarray(C, float) * 1e-6, "J"
    return None, "ADU"


def _groups():
    return load_groups(SESSION.ensure_workspace())


def _group_options():
    return [{"label": g["name"], "value": g["name"]} for g in _groups()]


def _x_axis(wl_range=None):
    wl = SESSION.calib["wl_axis"]
    if SESSION.params["USE_WL_AXIS"]:
        x, xlab = wl, "Wavelength (nm)"
    else:
        x, xlab = np.arange(len(wl)), "Pixel"
    if wl_range:
        m = (wl >= wl_range[0]) & (wl <= wl_range[1])
    else:
        m = np.ones(len(wl), bool)
    return x, xlab, m


def layout():
    return html.Div([
        page_header(
            "Groups & comparisons",
            "Define groups of shots (experimental condition) and "
            "compare them.",
            "Step 6. Your campaign groups (Si %, pulse profile) are "
            "pre-loaded; you can edit them, add new ones, or restore the "
            "originals. Comparisons use the cache: missing images are "
            "extracted on the fly."),
        card("Automatic groups from Final.xlsx", [
            html.Small("Builds groups from the Target / Pulse Profile / "
                       "2w E columns of the Excel file: one group per "
                       "(target, profile) combination, with an optional "
                       "energy filter. The assignment is shown BEFORE "
                       "creation for verification — nothing is applied "
                       "without your agreement.", className="text-muted d-block mb-2"),
            dbc.Row([
                labeled("Central 2ω energy (J, empty = all energies)",
                        dbc.Input(id="mg-ecenter", type="number",
                                  placeholder="e.g. 600"), width=3),
                labeled("Tolerance (± %)",
                        dbc.Input(id="mg-tol", type="number", value=10,
                                  min=1, max=100), width=2),
                labeled("Min. group size",
                        dbc.Input(id="mg-minsize", type="number", value=2,
                                  min=1), width=2),
                dbc.Col(dbc.Checklist(
                    id="mg-onlyavail", switch=True,
                    options=[{"label": " Only shots present in the folder",
                              "value": "avail"}],
                    value=["avail"], className="mt-4"), width=3),
                dbc.Col(dbc.Button("Analyse", id="mg-analyze",
                                   color="primary", className="mt-4"),
                        width=2),
            ]),
            dcc.Store(id="mg-proposal"),
            dcc.Loading(html.Div(id="mg-preview")),
            dbc.ButtonGroup([
                dbc.Button("Replace all groups with this proposal",
                           id="mg-replace", color="success"),
                dbc.Button("Add to the existing groups", id="mg-append",
                           color="secondary", outline=True),
            ], className="mt-2"),
            html.Div(id="mg-apply-msg", className="mt-2"),
        ], icon="bi-magic"),
        dbc.Accordion([dbc.AccordionItem([
            dbc.Row([
                labeled("Group to edit",
                        dcc.Dropdown(id="g-edit-select",
                                     options=_group_options(),
                                     placeholder="choose…"), width=3),
                labeled("Name", dbc.Input(id="g-edit-name"), width=3),
                labeled("Colour", dbc.Input(id="g-edit-color", type="color",
                                             value="#ff6347",
                                             style={"height": "38px"}), width=1),
                labeled("Si (%)", dbc.Input(id="g-edit-si", type="number",
                                            value=0), width=1),
                labeled("Profile", dcc.Dropdown(
                    id="g-edit-profile",
                    options=[{"label": p, "value": p} for p in PROFILE_ORDER],
                    value="10/90"), width=2),
            ]),
            labeled("Shots of the group (comma- or space-separated; "
                    "'shot431' or '431')",
                    dbc.Textarea(id="g-edit-shots", rows=2)),
            dbc.ButtonGroup([
                dbc.Button("Save this group", id="g-save",
                           color="primary"),
                dbc.Button("New group", id="g-new", color="secondary",
                           outline=True),
                dbc.Button("Delete this group", id="g-delete", color="danger",
                           outline=True),
                dbc.Button("Restore the campaign groups", id="g-reset",
                           color="warning", outline=True),
            ]),
            html.Div(id="g-edit-msg", className="mt-2"),
        ], title="Edit the groups")], start_collapsed=True,
            className="mb-3"),
        card("Comparison", [
            dbc.Row([
                labeled("Groups to compare",
                        dcc.Dropdown(id="g-compare", multi=True,
                                     options=_group_options(),
                                     value=[g["name"] for g in _groups()[:3]]),
                        width=5),
                labeled("Fiber (1–80)",
                        dbc.Input(id="g-fiber", type="number", value=64,
                                  min=1, max=80), width=2),
                labeled("λ range (nm, empty = all)",
                        dbc.Input(id="g-wlrange", placeholder="e.g. 700,900"),
                        width=2),
                labeled("Spread",
                        dcc.Dropdown(id="g-stdtype", clearable=False,
                                     value="std",
                                     options=[{"label": "standard deviation (std)",
                                               "value": "std"},
                                              {"label": "standard error of the mean (sem)",
                                               "value": "sem"}]), width=3),
            ]),
            dbc.Row([
                labeled("Outlier tolerance (± % around the group median)",
                        dbc.Input(id="g-outlier-tol", type="number", value=40,
                                  min=5, max=300, step=5),
                        "A shot whose signal departs from its group by more "
                        "than this is named as aberrant. Lower it to be "
                        "stricter, raise it for genuinely spread groups.",
                        width=4),
            ]),
            dbc.Button([html.I(className="bi bi-bar-chart-line me-1"),
                        "Compare"], id="g-run", color="primary"),
        ], icon="bi-collection"),
        units_radio("g-units"),
        dbc.Checklist(
            id="g-3dopts", switch=True, inline=True, value=[],
            options=[{"label": " 3D view: hide the fibers flagged as "
                               "outliers by the absolute calibration",
                      "value": "drop_outliers"}],
            className="mb-2"),
        dcc.Loading(html.Div(id="g-results")),
        card("Spectral-integral map: Si % × pulse profile", [
            html.Small("Reproduces the campaign summary map: integral of the "
                       "mean spectrum of the chosen fiber for each group, "
                       "positioned by (Si %, profile), with RBF "
                       "interpolation.", className="text-muted d-block mb-2"),
            dbc.Row([
                labeled("Fiber (1–80)",
                        dbc.Input(id="g-map-fiber", type="number",
                                  value=63, min=1, max=80), width=2),
                labeled("λ range (nm, empty = all)",
                        dbc.Input(id="g-map-wl", placeholder="e.g. 700,900"),
                        width=2),
                dbc.Col(dbc.Button("Compute the map", id="g-map-run",
                                   color="primary", className="mt-4"),
                        width=3),
            ]),
            dcc.Loading(html.Div(id="g-map-out")),
        ], icon="bi-grid-3x3-gap"),
    ])


# ── Group editing ──────────────────────────────────────────────────────
@callback(Output("g-edit-name", "value"), Output("g-edit-color", "value"),
          Output("g-edit-si", "value"), Output("g-edit-profile", "value"),
          Output("g-edit-shots", "value"),
          Input("g-edit-select", "value"), prevent_initial_call=True)
def fill_editor(name):
    for g in _groups():
        if g["name"] == name:
            return (g["name"], g.get("color", "#888888"),
                    g.get("si_pct", 0), g.get("profile", "10/90"),
                    ", ".join(g["shots"]))
    return no_update, no_update, no_update, no_update, no_update


def _parse_shots(txt):
    out = []
    for tok in str(txt or "").replace(",", " ").replace(";", " ").split():
        tok = tok.strip()
        if not tok:
            continue
        num = analysis.shot_num(tok)
        if num is not None:
            out.append(f"shot{num:03d}")
    return sorted(set(out), key=lambda k: analysis.shot_num(k))


@callback(Output("g-edit-msg", "children"),
          Output("g-edit-select", "options"),
          Output("g-compare", "options"),
          Input("g-save", "n_clicks"), Input("g-delete", "n_clicks"),
          Input("g-reset", "n_clicks"), Input("g-new", "n_clicks"),
          State("g-edit-select", "value"), State("g-edit-name", "value"),
          State("g-edit-color", "value"), State("g-edit-si", "value"),
          State("g-edit-profile", "value"), State("g-edit-shots", "value"),
          prevent_initial_call=True)
def edit_groups(n_save, n_del, n_reset, n_new, selected, name, color, si,
                profile, shots_txt):
    from dash import ctx
    ws = SESSION.ensure_workspace()
    groups = _groups()
    trig = ctx.triggered_id
    msg = None
    if trig == "g-reset":
        groups = [dict(g) for g in DEFAULT_GROUPS]
        save_groups(ws, groups)
        msg = dbc.Alert("Campaign groups restored.", color="success",
                        className="py-2")
    elif trig == "g-delete" and selected:
        groups = [g for g in groups if g["name"] != selected]
        save_groups(ws, groups)
        msg = dbc.Alert(f"Group '{selected}' deleted.", color="info",
                        className="py-2")
    elif trig in ("g-save", "g-new"):
        if not name:
            msg = guard_alert("Give the group a name.")
        else:
            shots = _parse_shots(shots_txt)
            if not shots:
                msg = guard_alert("No valid shot in the list.")
            else:
                new_g = {"name": name, "color": color or "#888888",
                         "si_pct": si if si is not None else 0,
                         "profile": profile or "10/90", "shots": shots}
                replaced = False
                target = selected if trig == "g-save" and selected else name
                for i, g in enumerate(groups):
                    if g["name"] == target:
                        groups[i] = new_g
                        replaced = True
                        break
                if not replaced:
                    groups.append(new_g)
                save_groups(ws, groups)
                missing = [sh for sh in shots
                           if sh not in SESSION.image_dict]
                extra = (f" Warning: {len(missing)} shots missing from "
                         f"the images folder ({', '.join(missing[:5])}…)."
                         if missing else "")
                msg = dbc.Alert(f"Group '{name}' saved "
                                f"({len(shots)} shots).{extra}",
                                color="success", className="py-2")
    opts = [{"label": g["name"], "value": g["name"]} for g in _groups()]
    return msg, opts, opts


# ── Comparison ──────────────────────────────────────────────────────────────
def _parse_wl_range(txt):
    toks = [t for t in str(txt or "").replace(";", ",").split(",")
            if t.strip()]
    if len(toks) == 2:
        try:
            a, b = float(toks[0]), float(toks[1])
            return (min(a, b), max(a, b))
        except ValueError:
            pass
    return None


def _ratio_and_z(values):
    """(rapport a la mediane, z robuste) pour une serie de mesures.

    Le rapport est le critere lisible ("ce tir vaut 0,35 fois les autres") et
    reste valable meme quand le groupe est tres disperse. Le z robuste (ecart
    a la mediane rapporte au MAD) attrape a l'inverse les ecarts modestes dans
    un groupe tres serre. Les deux sont renvoyes : le controle en aval declare
    aberrant si l'un OU l'autre depasse son seuil.

    Les valeurs sont ramenees en positif (|v|) avant le rapport : une
    integrale peut etre negative si le fond a ete sur-soustrait, et un rapport
    calcule sur des nombres negatifs inverserait le sens de « trop faible ».
    """
    v = np.asarray(values, float)
    ratio = np.full(v.shape, np.nan)
    z = np.full(v.shape, np.nan)
    fin = np.isfinite(v)
    if fin.sum() < 2:
        return ratio, z
    a = np.abs(v)
    med = float(np.median(a[fin]))
    if med > 0:
        ratio[fin] = a[fin] / med
    mad = float(np.median(np.abs(a[fin] - med)))
    if mad > 0:
        z[fin] = 0.6745 * (a[fin] - med) / mad
    else:
        sd = float(np.std(a[fin]))
        if sd > 0:
            z[fin] = (a[fin] - float(np.mean(a[fin]))) / sd
    return ratio, z


def _outlier_scan(shots, per_fiber, per_all, tol_pct=40.0, z_thresh=4.0):
    """Controle des tirs aberrants d'un groupe, sur DEUX mesures.

    per_fiber : integrale du spectre de la fibre affichee, tir par tir.
    per_all   : integrale sommee sur les 80 fibres, tir par tir. C'est l'ajout
                important : un tir globalement faible etait invisible tant que
                la fibre affichee etait bruitee ou peu eclairee, alors que le
                signal total le designe sans ambiguite.

    Retourne (lignes, aberrants) ou `lignes` decrit CHAQUE tir (affiche tel
    quel dans un tableau, pour que l'absence d'alerte soit verifiable et non
    un silence) et `aberrants` la liste [(shot, motif), …].
    """
    names = list(shots)
    r_f, z_f = _ratio_and_z(per_fiber)
    r_a, z_a = _ratio_and_z(per_all)
    lo, hi = 1.0 - tol_pct / 100.0, 1.0 + tol_pct / 100.0
    rows, bad = [], []
    for i, name in enumerate(names):
        reasons = []
        for lab, r, z in (("the displayed fiber", r_f[i], z_f[i]),
                          ("the total over the 80 fibers", r_a[i], z_a[i])):
            if np.isfinite(r) and (r <= lo or r >= hi):
                reasons.append(f"×{r:.2f} the group median on {lab}")
            elif np.isfinite(z) and abs(z) > z_thresh:
                reasons.append(f"{z:+.1f}σ (robust) on {lab}")
        # severite = plus grand ecart relatif a 1 parmi les rapports connus
        sev = max([abs(r - 1.0) for r in (r_f[i], r_a[i]) if np.isfinite(r)]
                  or [0.0])
        rows.append({
            "shot": name,
            "i_fiber": float(per_fiber[i]) if i < len(per_fiber) else np.nan,
            "r_fiber": float(r_f[i]) if i < len(r_f) else np.nan,
            "i_all": float(per_all[i]) if i < len(per_all) else np.nan,
            "r_all": float(r_a[i]) if i < len(r_a) else np.nan,
            "bad": bool(reasons),
            "sev": sev,
            "why": " ; ".join(reasons),
        })
        if reasons:
            bad.append((name, " ; ".join(reasons), sev))
    bad.sort(key=lambda t: -t[2])          # les plus ecartes en tete
    return rows, [(n, w) for n, w, _ in bad]


def _outlier_section(blocks, aunit, fiber, tol_pct):
    """Bandeau + tableau du controle des tirs aberrants.

    Deux exigences, apprises du fait que l'alerte precedente passait
    inapercue : elle est placee AVANT les graphiques (donc au-dessus du pli
    de la page), et le tableau tir par tir est affiche MEME quand rien n'est
    signale. « Pas d'alerte » devient ainsi une information verifiable — on
    voit les rapports a la mediane — au lieu d'un silence qu'on ne sait pas
    interpreter.
    """
    if not blocks:
        return []
    flagged = [(gname, bad) for gname, _c, _r, bad in blocks if bad]
    out = []
    for gname, bad in flagged:
        names = ", ".join(sh for sh, _ in bad)
        out.append(dbc.Alert([
            html.H6([html.I(className="bi bi-exclamation-octagon-fill me-2"),
                     f"Group “{gname}” — aberrant shot(s): {names}"],
                    className="alert-heading mb-2"),
            html.Ul([html.Li([html.B(sh), f" — {why}"]) for sh, why in bad],
                    className="mb-2", style={"fontSize": "0.88rem"}),
            html.Small("Remove them from the group above (or raise the "
                       "tolerance) and run the comparison again — they are "
                       "what inflates the ± band on the mean spectrum.",
                       className="text-muted"),
        ], color="danger"))
    if not flagged:
        out.append(dbc.Alert(
            [html.I(className="bi bi-check-circle me-2"),
             f"Outlier check run on {sum(len(r) for _g, _c, r, _b in blocks)} "
             f"shots: none departs from its group by more than "
             f"{tol_pct:.0f} %."], color="success", className="py-2"))

    head = html.Thead(html.Tr([
        html.Th("Group"), html.Th("Shot"),
        html.Th(f"Integral, fiber {fiber + 1} ({aunit})"),
        html.Th("× median"),
        html.Th(f"Total, 80 fibers ({aunit})"),
        html.Th("× median"), html.Th("Verdict")]))
    body = []
    for gname, color, rows, _bad in blocks:
        for k, r in enumerate(rows):
            def num(v):
                return "—" if not np.isfinite(v) else f"{v:.3e}"

            def rat(v):
                return "—" if not np.isfinite(v) else f"{v:.2f}"
            body.append(html.Tr([
                html.Td([html.Span("■ ", style={"color": color}), gname]
                        if k == 0 else ""),
                html.Td(html.B(r["shot"]) if r["bad"] else r["shot"]),
                html.Td(num(r["i_fiber"])), html.Td(rat(r["r_fiber"])),
                html.Td(num(r["i_all"])), html.Td(rat(r["r_all"])),
                html.Td("ABERRANT" if r["bad"] else "ok"),
            ], className="table-danger" if r["bad"] else None))
    out.append(dbc.Accordion([dbc.AccordionItem(
        [html.Small("Two measurements per shot: the integral of the displayed "
                    "fiber, and the integral summed over the 80 fibers. The "
                    "second is what catches a globally weak shot when the "
                    "displayed fiber happens to be noisy. A shot is flagged "
                    f"when either departs from its group median by more than "
                    f"{tol_pct:.0f} %, or by more than 4σ (robust) when the "
                    "group is otherwise very tight.",
                    className="text-muted d-block mb-2"),
         dbc.Table([head, html.Tbody(body)], size="sm", hover=True,
                   striped=False)],
        title="Outlier check — value of every shot")],
        start_collapsed=not flagged, className="mb-3"))
    return out


@callback(Output("g-results", "children"),
          Input("g-run", "n_clicks"),
          State("g-compare", "value"), State("g-fiber", "value"),
          State("g-wlrange", "value"), State("g-stdtype", "value"),
          State("g-units", "value"), State("g-3dopts", "value"),
          State("g-outlier-tol", "value"),
          prevent_initial_call=True)
def compare(_, names, fiber, wl_txt, std_type, units, opts3d, otol):
    if SESSION.calib is None:
        return guard_alert("Run the calibration first (step 2).")
    if not names:
        return guard_alert("Select at least one group.")
    fiber = int(fiber or 1) - 1   # display 1-80 -> internal
    wl_range = _parse_wl_range(wl_txt)
    x, xlab, m = _x_axis(wl_range)
    groups = [g for g in _groups() if g["name"] in names]
    gfac, aunit = _gfac(units)

    wl_m = SESSION.calib["wl_axis"][m]
    trapz = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
    groups_data, groups_stats, all_missing = [], [], []
    integ_rows, outlier_blocks = [], []
    stacks = {}          # cache : reutilise pour la vue 3D plus bas
    unreadable = []
    for g in groups:
        specs, missing, present = analysis.group_fiber_spectra(
            g["shots"], fiber)
        all_missing += [(g["name"], sh) for sh in missing]
        unreadable += [(g["name"], sh, why) for sh, why in
                       getattr(analysis.group_fiber_spectra,
                               "last_unreadable", [])]
        specs_m = [sp[m] for sp in specs]
        if gfac is not None:
            cf = gfac[fiber][m] if fiber < gfac.shape[0] else np.nan
            specs_m = [sp * cf for sp in specs_m]
        groups_data.append({"label": g["name"], "color": g["color"],
                            "spectra": specs_m})
        mean_sp, dev = analysis.mean_std(specs_m, kind=std_type)
        groups_stats.append({"label": g["name"], "color": g["color"],
                             "mean": mean_sp, "dev": dev})

        # ── Integrale du spectre de la fibre affichee, tir par tir ────────
        per_shot = np.array([trapz(np.nan_to_num(sp), wl_m)
                             for sp in specs_m], float)
        finite = per_shot[np.isfinite(per_shot)]
        if finite.size:
            integ_rows.append((g["name"], len(present), float(finite.mean()),
                               float(finite.std())))

        # ── Signal TOTAL de chaque tir (somme des 80 fibres) ──────────────
        # `group_stack_all_fibers` saute les memes shots manquants, dans le
        # meme ordre : la pile est donc alignee sur `present`.
        stack = analysis.group_stack_all_fibers(g["shots"], m)
        stacks[g["name"]] = stack
        if stack is not None and stack.shape[0] == len(present):
            per_all = np.array(
                [np.nansum(trapz(np.nan_to_num(sh), wl_m, axis=-1))
                 for sh in stack], float)
        else:
            per_all = np.full(len(present), np.nan)

        if present:
            rows_o, bad = _outlier_scan(present, per_shot, per_all,
                                        tol_pct=float(otol or 40))
            outlier_blocks.append((g["name"], g.get("color", "#888888"),
                                   rows_o, bad))
    SESSION.log_history("group_comparison", {
        "groups": names, "fiber": fiber, "wl_range": wl_range,
        "spread": std_type})

    ylab = analysis.ENERGY_LABEL if gfac is not None else "Intensity (ADU)"
    f_ind = plotting.fig_groups_individual(
        x[m], xlab, groups_data, title=f"Individual spectra — fiber {fiber+1}")
    f_ms = plotting.fig_groups_mean_std(
        x[m], xlab, groups_stats, title=f"Mean ± {std_type} — fiber {fiber+1}")
    f_ind.update_yaxes(title=ylab); f_ms.update_yaxes(title=ylab)

    # ── Controle des tirs aberrants, EN HAUT et TOUJOURS affiche ───────────
    out = []
    if unreadable:
        # En tete, avant tout le reste : un shot ecarte change la moyenne du
        # groupe, l'utilisateur doit le savoir avant de lire les courbes.
        out.append(dbc.Alert([
            html.H6([html.I(className="bi bi-file-earmark-x me-2"),
                     f"{len(unreadable)} shot(s) excluded — unusable file"],
                    className="alert-heading mb-2"),
            html.Ul([html.Li([html.B(sh), f" ({gname}) — {why}"])
                     for gname, sh, why in unreadable[:8]],
                    className="mb-1", style={"fontSize": "0.85rem"}),
            html.Small("They are absent from the curves and from the "
                       "averages below.", className="text-muted"),
        ], color="danger"))
    out += _outlier_section(outlier_blocks, aunit, fiber, float(otol or 40))
    out.append(dbc.Row([dbc.Col(graph(f_ind), md=6),
                        dbc.Col(graph(f_ms), md=6)]))

    if integ_rows:
        out.append(card("Integral of the group spectra (mean ± std)", [
            html.Small("Integral over the shown wavelength range of each "
                       "spectrum in the group, then averaged. The ± is the "
                       "spread across the group's spectra, not a fit error.",
                       className="text-muted d-block mb-2"),
            dbc.Table([
                html.Thead(html.Tr([html.Th("Group"), html.Th("N spectra"),
                                    html.Th(f"Integral ({aunit})")])),
                html.Tbody([html.Tr(
                    [html.Td(n), html.Td(k),
                     html.Td(f"{a:.4e} ± {s:.2e}"
                             + (f"  ({s / abs(a) * 100:.0f} %)"
                                if a else ""))])
                    for n, k, a, s in integ_rows])],
                size="sm", hover=True)], icon="bi-rulers"))
    if all_missing:
        out.append(guard_alert(
            f"{len(all_missing)} shots missing from the images folder: "
            + ", ".join(f"{g}:{s}" for g, s in all_missing[:8]) + "…"))

    # 3D surfaces — all fibers
    drop = []
    if gfac is not None and "drop_outliers" in (opts3d or []):
        drop = [int(f) - 1 for f in (SESSION.abs_cal.get("outliers") or [])]
    stats3d = []
    for g in groups:
        stack = stacks.get(g["name"])
        if stack is None:
            stats3d.append({"label": g["name"], "color": g["color"],
                            "mean": None, "dev": None})
            continue
        if gfac is not None:
            n = min(gfac.shape[0], stack.shape[1])
            cf = np.full((stack.shape[1], stack.shape[2]), np.nan)
            cf[:n] = gfac[:n][:, m]
            stack = stack * cf[None, :, :]
        mean3d = np.nanmean(stack, axis=0)
        dev3d = np.nanstd(stack, axis=0)
        for i in drop:
            if 0 <= i < mean3d.shape[0]:
                mean3d[i] = np.nan
                dev3d[i] = np.nan
        stats3d.append({"label": g["name"], "color": g["color"],
                        "mean": mean3d, "dev": dev3d})
    body3d = [graph(plotting.fig_groups_3d(
        x[m], stats3d, z_title=ylab,
        overlay_title="Overlaid groups — rotate the view with the mouse"))]
    body3d.append(_gfac_note(gfac, drop))
    out.append(card(
        "3D view — mean spectrum per fiber (surface = mean, envelopes = ±1 std)",
        body3d, icon="bi-badge-3d"))
    return html.Div(out)


def _gfac_note(gfac, dropped):
    """Explique pourquoi la vue 3D change de FORME entre ADU et µJ/nm.

    La conversion est un facteur PAR FIBRE (g_i). Sur un trace fibre a fibre
    elle n'est qu'un changement d'echelle — la courbe est identique. Sur la
    vue 3D, qui compare les 80 fibres entre elles, chaque fibre est multipliee
    par un facteur different : le relief change reellement, et les fibres de
    faible rendement (g grand) ressortent."""
    if gfac is None:
        return html.Small(
            "In ADU the fibers are not corrected for their throughput: the "
            "relief mixes the physical signal and the response of each fiber.",
            className="text-muted")
    g = np.asarray(gfac, float)
    per_fiber = np.nanmedian(np.where(g > 0, g, np.nan), axis=1)
    fin = per_fiber[np.isfinite(per_fiber)]
    if fin.size == 0:
        return None
    ratio = (float(np.nanmax(fin) / np.nanmin(fin))
             if np.nanmin(fin) > 0 else float("nan"))
    kids = [
        html.Small([
            html.B("Why does this surface differ from the ADU one? "),
            "The ADU → J/nm conversion is a factor ", html.I("per fiber and "
            "per wavelength"), f" (median across fibers between "
            f"{np.nanmin(fin):.3g} and {np.nanmax(fin):.3g} J/(ADU·nm), a "
            f"ratio of {ratio:.1f}×). On a single-fiber plot over a narrow "
            "band it acts almost as a pure change of scale, so the curve keeps "
            "its shape — which is why the fiber-to-fiber comparisons look "
            "unchanged. Here the 80 fibers are compared with each other over "
            "the whole band, so each one is rescaled differently and at each "
            "wavelength: the relief genuinely changes, and low-throughput "
            "fibers become dominant.",
        ], className="text-muted"),
    ]
    if dropped:
        kids.append(html.Br())
        kids.append(html.Small(
            f"{len(dropped)} fibers flagged by the absolute calibration are "
            f"hidden: " + ", ".join(str(i + 1) for i in dropped[:12])
            + ("…" if len(dropped) > 12 else ""), className="text-warning"))
    return html.Div(kids, className="mt-2")


# ── Si × profile map ────────────────────────────────────────────────────────
@callback(Output("g-map-out", "children"),
          Input("g-map-run", "n_clicks"),
          State("g-map-fiber", "value"), State("g-map-wl", "value"),
          State("g-units", "value"),
          prevent_initial_call=True)
def area_map(_, fiber, wl_txt, units):
    if SESSION.calib is None:
        return guard_alert("Run the calibration first (step 2).")
    fiber = int(fiber or 1) - 1
    wl_range = _parse_wl_range(wl_txt)
    wl_axis = SESSION.calib["wl_axis"]
    gfac, aunit = _gfac(units)
    results, rows = {}, []
    for g in _groups():
        si, prof = g.get("si_pct"), g.get("profile")
        if si is None or prof not in PROFILE_ORDER:
            continue
        specs, _, _ = analysis.group_fiber_spectra(g["shots"], fiber)
        if not specs:
            rows.append((g["name"], 0, None))
            continue
        mean_sp = np.nanmean(np.vstack(specs), axis=0)
        if gfac is not None and fiber < gfac.shape[0]:
            mean_sp = mean_sp * gfac[fiber]
        area = sf.compute_spectral_area(mean_sp[np.newaxis, :], wl_axis,
                                        wl_range=wl_range)[0]
        results[(si, prof)] = float(area)
        rows.append((g["name"], len(specs), area))
    if not results:
        return guard_alert("No usable group (missing images?).")
    present = set(results)
    missing = [(si, prof) for prof in PROFILE_ORDER for si in (0, 5, 15)
               if (si, prof) not in present]
    fig = plotting.fig_area_map_si_profile(results, PROFILE_ORDER,
                                           missing=missing)
    table = dbc.Table([
        html.Thead(html.Tr([html.Th("Group"), html.Th("N"),
                            html.Th(f"Mean integral ({aunit})")])),
        html.Tbody([html.Tr([html.Td(n), html.Td(k),
                             html.Td("N/A" if a is None else f"{a:.3e}")])
                    for n, k, a in rows])], size="sm", hover=True)
    SESSION.log_history("si_profile_map", {"fiber": fiber,
                                           "wl_range": wl_range})
    return html.Div([graph(fig), table,
                     html.Small("The RBF interpolation between 8 points is "
                                "a visual guide, not a prediction: do not "
                                "use it to extrapolate.",
                                className="text-muted")])


# ── Automatic groups ─────────────────────────────────────────────────────
@callback(Output("mg-preview", "children"), Output("mg-proposal", "data"),
          Input("mg-analyze", "n_clicks"),
          State("mg-ecenter", "value"), State("mg-tol", "value"),
          State("mg-minsize", "value"), State("mg-onlyavail", "value"),
          prevent_initial_call=True)
def auto_analyze(_, e_center, tol, min_size, only_avail):
    from core import metadata as md
    s = SESSION
    if not s.metadata:
        ok, msg = analysis.load_metadata()
        if not ok:
            return guard_alert(msg), None
    if not s.image_dict:
        s.scan_images()
    groups, assign, excluded = md.build_auto_groups(
        s.metadata,
        available_shots=set(s.image_dict),
        e_center=float(e_center) if e_center not in (None, "") else None,
        tol_pct=float(tol or 10), min_size=int(min_size or 1),
        only_available="avail" in (only_avail or []))
    if not groups:
        return guard_alert("No group can be built with these criteria — "
                           "see the exclusions below."
                           if excluded else "Final.xlsx contains no usable "
                           "row."), None
    head = [html.Tr([html.Th(h) for h in
                     ["Proposed group", "Si %", "Profile", "N shots",
                      "E2ω (J) min–max", "Shots"]])]
    rows = []
    for g in groups:
        es = [s.metadata[analysis.shot_num(sh)]["e2w"] for sh in g["shots"]
              if s.metadata.get(analysis.shot_num(sh), {}).get("e2w") is not None]
        erange = f"{min(es):.0f}–{max(es):.0f}" if es else "N/A"
        shots_disp = ", ".join(str(analysis.shot_num(sh)) for sh in g["shots"])
        rows.append(html.Tr([
            html.Td([html.Span("■ ", style={"color": g["color"]}), g["name"]]),
            html.Td("N/A" if g["si_pct"] is None else f"{g['si_pct']}"),
            html.Td(g["profile"]), html.Td(len(g["shots"])),
            html.Td(erange),
            html.Td(html.Small(shots_disp, style={"fontSize": "0.72rem"})),
        ]))
    out = [html.P([dbc.Badge(f"{len(groups)} proposed groups",
                             color="primary", className="me-2"),
                   dbc.Badge(f"{len(assign)} shots assigned", color="success",
                             className="me-2"),
                   dbc.Badge(f"{len(excluded)} excluded", color="secondary")]),
           dbc.Table(head + [html.Tbody(rows)], size="sm", hover=True)]
    if excluded:
        exc_rows = [html.Li(f"shot {sh} ({t} / {p}"
                            + (f", {e:.0f} J" if e is not None else "")
                            + f"): {r}")
                    for sh, t, p, e, r in excluded[:400]]
        out.append(dbc.Accordion([dbc.AccordionItem(
            html.Ul(exc_rows, style={"fontSize": "0.8rem",
                                     "maxHeight": "260px",
                                     "overflow": "auto"}),
            title=f"{len(excluded)} shots excluded (detailed reasons)")],
            start_collapsed=True))
    return html.Div(out), groups


@callback(Output("mg-apply-msg", "children"),
          Output("g-edit-select", "options", allow_duplicate=True),
          Output("g-compare", "options", allow_duplicate=True),
          Output("g-compare", "value"),
          Input("mg-replace", "n_clicks"), Input("mg-append", "n_clicks"),
          State("mg-proposal", "data"),
          prevent_initial_call=True)
def auto_apply(n_rep, n_app, proposal):
    from dash import ctx, no_update
    if not proposal:
        return (guard_alert("Run 'Analyse' first."),
                no_update, no_update, no_update)
    ws = SESSION.ensure_workspace()
    if ctx.triggered_id == "mg-replace":
        groups = list(proposal)
        verb = "replaced"
    else:
        groups = _groups()
        existing = {g["name"] for g in groups}
        added = [g for g in proposal if g["name"] not in existing]
        groups += added
        verb = f"extended ({len(added)} added)"
    save_groups(ws, groups)
    SESSION.log_history("auto_groups", {"n": len(proposal), "mode": verb})
    opts = [{"label": g["name"], "value": g["name"]} for g in groups]
    return (dbc.Alert(f"Groups {verb} — {len(groups)} groups in total.",
                      color="success", className="py-2"),
            opts, opts, [g["name"] for g in groups[:3]])
