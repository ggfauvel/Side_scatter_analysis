"""Page 1 — Data: folder/file selection + validation."""
from __future__ import annotations

import re
from pathlib import Path

import dash_bootstrap_components as dbc
from dash import Input, Output, State, dcc, html, no_update
from core.uistate import callback

from app.components import card, guard_alert, labeled, page_header
from core import analysis
from core.session import SESSION

_SHOT_RE = re.compile(r"shot\s*_?\s*0*(\d+)\.tiff?$", re.IGNORECASE)


def _tif_candidates(folder):
    """TIFF du dossier qui ne sont PAS des images 'shotN' -> candidats HgAr."""
    d = Path(folder)
    if not d.is_dir():
        return []
    return sorted(f.name for f in d.iterdir()
                  if f.suffix.lower() in (".tif", ".tiff")
                  and not _SHOT_RE.search(f.name))


def _xlsx_candidates(folder):
    d = Path(folder)
    if not d.is_dir():
        return []
    return sorted(f.name for f in d.iterdir() if f.suffix.lower() == ".xlsx")


def layout():
    s = SESSION
    return html.Div([
        page_header(
            "Input data",
            "Tell the application where your files are; it detects the rest.",
            "Step 1. Enter the folder containing the shotNNN.tif images, "
            "then click 'Scan the folder'. The HgAr image, the energy Excel "
            "file and the pulse CSVs are detected automatically when they "
            "are in the same folder; you can also point to them manually."),
        dbc.Row([
            dbc.Col(card("Science images folder", [
                labeled("Folder path (copy-paste from your file explorer)",
                        dbc.Input(id="d-images-dir", value=s.images_dir,
                                  placeholder=r"e.g. D:\campaign\images or /home/user/images")),
                dbc.Button([html.I(className="bi bi-search me-2"),
                            "Scan the folder"],
                           id="d-scan", color="primary", className="mt-1"),
                html.Div(id="d-scan-result", className="mt-3"),
            ], icon="bi-images"), md=6),
            dbc.Col(card("Associated files", [
                labeled("HgAr calibration image",
                        dcc.Dropdown(id="d-hgar", placeholder="detected after scanning…",
                                     options=[], value=None),
                        "HgAr lamp for the pixel → nm calibration."),
                labeled("…or manual path to the HgAr image",
                        dbc.Input(id="d-hgar-manual", value=s.hgar_path,
                                  placeholder="full path if elsewhere")),
                labeled("Energy Excel file (Final.xlsx)",
                        dcc.Dropdown(id="d-excel", options=[], value=None,
                                     placeholder="detected after scanning…"),
                        "Column C = shot number, column F = 2ω energy (J)."),
                labeled("…or manual path to the Excel file",
                        dbc.Input(id="d-excel-manual", value=s.excel_path)),
                labeled("Pulse CSV folder (oscilloscope)",
                        dbc.Input(id="d-pulses-dir", value=s.pulses_dir,
                                  placeholder="empty = same folder as the images"),
                        "Accepted names: 'shot 431.csv', 'shot_431.csv', 'shot431.csv'."),
                labeled("Workspace folder (cache, exports, history)",
                        dbc.Input(id="d-workspace", value=s.workspace,
                                  placeholder="empty = the application's 'workspace' folder")),
            ], icon="bi-paperclip"), md=6),
        ]),
        _angles_card(),
        dbc.Button([html.I(className="bi bi-check2-circle me-2"),
                    "Validate and save the configuration"],
                   id="d-validate", color="success", size="lg"),
        html.Div(id="d-validate-result", className="mt-3"),
    ])


# ── Angular configuration (φ, θ) ────────────────────────────────────────────
# Principe, apres les deboires de la campagne Fauvel : le fichier de campagne
# n'est cru QUE sur trois informations (quadrant, bras, port) et les angles
# viennent toujours du fichier de reference, qui est le seul format stable
# d'une campagne a l'autre. Il n'y a donc plus rien a regler ici dans le cas
# normal : deux chemins, un bouton.

_MAP_COLS = [("fiber", "Fiber number"), ("quadrant", "Quadrant (alpha/delta)"),
             ("arm", "Arm (letter)"), ("port", "Port (number)")]


def _angles_card():
    s = SESSION
    return card("Angular configuration (φ, θ) of the fibers", [
        html.Small([
            "Two files. The ", html.B("angle reference"),
            " (…EstimateStructureCoordinates.xlsx) gives φ for each arm and θ "
            "for each port; its layout does not change between campaigns, so "
            "it is the only source of angles. The ",
            html.B("campaign fiber map"),
            " only says, for each fiber, its quadrant, its arm and its port — "
            "the three things a campaign file cannot get wrong. φ and θ are "
            "then looked up in the reference and x, y, z recomputed from "
            "them. Any φ, θ, x, y, z columns present in the campaign file are ",
            html.B("ignored"), " (they were wrong on fiber_config_Fauvel_b: θ "
            "written as 180−θ, which flips the sphere upside down); the "
            "import report tells you by how much they differed. Columns are "
            "recognised by name and, failing that, by content, so a future "
            "campaign can rename or move them without any code change.",
        ], className="text-muted d-block mb-3"),
        dbc.Row([
            labeled("Angle reference file (.xlsx) — required",
                    dbc.Input(id="d-structure", value=s.structure_path,
                              placeholder="…/251102_EstimateStructureCoordinates.xlsx"),
                    "Shared by every configuration of the campaign.",
                    width=6),
            labeled("Campaign fiber map(s) (.xlsx) — one path per line",
                    dbc.Textarea(id="d-map-path", rows=3,
                                 value="\n".join(s.flat_angle_paths),
                                 placeholder="…/fiber_config_Fauvel_b.xlsx"),
                    "A campaign can have several configurations — November "
                    "has three. Put one file per line: each becomes its own "
                    "angular configuration, named after the file.", width=6),
        ]),
        dbc.Row([
            labeled("Fiber numbering in the campaign file", dcc.Dropdown(
                id="d-map-numbering", clearable=False, value="extraction",
                options=[{"label": "Not reversed — the file's fiber 1 is the "
                                   "first detector trace",
                          "value": "extraction"},
                         {"label": "Reversed — the file's fiber 1 is the last "
                                   "detector trace", "value": "physical"}]),
                "A property of the mounting, not of the file: November sends "
                "the fibers back reversed, May does not. It can also be "
                "flipped live on the 3D view (page 7). The test: the maximum "
                "of the signal must land near backscatter.", width=6),
        ]),
        dbc.Button([html.I(className="bi bi-box-arrow-in-down me-1"),
                    "Read the files and build the configuration"],
                   id="d-map-import", color="primary"),
        dcc.Loading(html.Div(id="d-map-result", className="mt-2")),
        dbc.Accordion([dbc.AccordionItem([
            html.Small("Only if the automatic reading picked the wrong "
                       "columns — the import report above always says which "
                       "ones it used. Give the column NUMBER (1 = first "
                       "column of the sheet); leave empty for automatic.",
                       className="text-muted d-block mb-2"),
            dbc.Row([labeled("Sheet", dbc.Input(id="d-map-sheet",
                                                placeholder="automatic"),
                             width=3)]),
            dbc.Row([labeled(lab, dbc.Input(id=f"d-map-col-{f}", type="number",
                                            min=1, placeholder="auto"),
                             width=3) for f, lab in _MAP_COLS]),
        ], title="Manual override (not needed in the normal case)")],
            start_collapsed=True, className="mt-3"),
    ], icon="bi-compass")


def _report_list(report):
    icon = {"ok": ("bi-check-circle text-success", ""),
            "info": ("bi-info-circle text-secondary", ""),
            "warn": ("bi-exclamation-triangle text-warning", "text-warning")}
    return html.Ul([
        html.Li([html.I(className=f"bi {icon.get(lvl, icon['info'])[0]} me-2"),
                 html.Span(msg, className=icon.get(lvl, icon["info"])[1])])
        for lvl, msg in report], className="mb-1",
        style={"fontSize": "0.85rem", "listStyle": "none", "paddingLeft": 0})


@callback(
    Output("d-map-result", "children"),
    Input("d-map-import", "n_clicks"),
    State("d-structure", "value"), State("d-map-path", "value"),
    State("d-map-sheet", "value"),
    State("d-map-numbering", "value"),
    *[State(f"d-map-col-{f}", "value") for f, _ in _MAP_COLS],
    prevent_initial_call=True)
def map_import(_, structure, paths_txt, sheet, numbering, *cols):
    from core import angles as ang
    s = SESSION
    if not structure or not Path(structure).exists():
        return guard_alert("Angle reference file not found — without it the "
                           "angles cannot be resolved.")
    paths = [ln.strip().strip('"') for ln in str(paths_txt or "").splitlines()]
    paths = [p for p in paths if p]
    if not paths:
        return guard_alert("No campaign fiber map given.")
    missing = [p for p in paths if not Path(p).exists()]
    if missing:
        return guard_alert("File(s) not found: " + ", ".join(missing))
    col_map = {f: int(c) - 1 for (f, _), c in zip(_MAP_COLS, cols)
               if c not in (None, "")}
    try:
        struct = ang.load_structure_tables(structure)
    except Exception as e:
        return guard_alert(f"Angle reference file unreadable: {e}")

    s.load_angle_registry()
    blocks, n_ok, warn = [], 0, False
    for path in paths:
        cfg_name = Path(path).stem
        try:
            ent, report = ang.build_config_from_map(
                struct, path, sheet=(sheet or None) or None, col_map=col_map,
                numbering=numbering or "extraction")
        except Exception as e:
            warn = True
            blocks.append(dbc.Alert(f"{cfg_name}: import failed — {e}",
                                    color="danger", className="py-2 mb-2"))
            continue
        summ = ang.flat_table_summary(ent)
        if not summ.get("n"):
            warn = True
            blocks.append(dbc.Alert(
                f"{cfg_name}: no fiber resolved — check the columns under "
                f"'Manual override'.", color="danger", className="py-2 mb-2"))
            continue
        s.angle_registry[cfg_name] = ent
        n_ok += 1
        warn = warn or any(lvl == "warn" for lvl, _ in report)
        blocks.append(html.Div([
            html.P([dbc.Badge(f"'{cfg_name}'", color="success",
                              className="me-2"),
                    dbc.Badge(f"{summ['n']} fibers", color="primary",
                              className="me-2"),
                    dbc.Badge(f"φ {summ['phi_min']:.1f}…{summ['phi_max']:.1f}°",
                              color="secondary", className="me-2"),
                    dbc.Badge(f"θ {summ['theta_min']:.1f}…"
                              f"{summ['theta_max']:.1f}°",
                              color="secondary")], className="mb-1"),
            _report_list(report),
        ], className="mb-3"))

    ang.save_registry(s.ensure_workspace(), s.angle_registry)
    s.structure_path = structure
    s.flat_angle_paths = paths
    s.flat_angle_path = paths[0]
    s.save_config()
    return dbc.Alert(
        [html.H6(f"{n_ok}/{len(paths)} configuration(s) imported",
                 className="alert-heading mb-2")] + blocks
        + [html.Small("Available on page 7 (Angular maps & 3D) in the "
                      "configuration dropdown.", className="text-muted")],
        color="warning" if warn else "success")


@callback(
    Output("d-scan-result", "children"),
    Output("d-hgar", "options"), Output("d-hgar", "value"),
    Output("d-excel", "options"), Output("d-excel", "value"),
    Input("d-scan", "n_clicks"),
    State("d-images-dir", "value"),
    prevent_initial_call=True)
def scan(_, folder):
    if not folder or not Path(folder).is_dir():
        return (guard_alert("Folder not found. Check the path."),
                [], None, [], None)
    SESSION.images_dir = folder
    info = SESSION.scan_images()
    tifs = _tif_candidates(folder)
    xls = _xlsx_candidates(folder)
    body = [
        html.P([dbc.Badge(f"{info['n']} images", color="primary",
                          className="me-2"),
                html.Span(f"from {info['first']} to {info['last']}"
                          if info["n"] else "no shotNNN.tif image found")]),
    ]
    if info["duplicates"]:
        body.append(guard_alert(
            f"{len(info['duplicates'])} duplicates ignored (same shot number): "
            + ", ".join(info["duplicates"][:5]) + "…"))
    if info["n"] == 0:
        body.append(html.Small(
            "Expected: .tif files whose name contains 'shot' followed by the "
            "number (prefixes tolerated).", className="text-muted"))
    color = "success" if info["n"] else "warning"
    res = dbc.Alert(body, color=color)
    hgar_val = tifs[0] if len(tifs) == 1 else None
    xl_val = xls[0] if len(xls) == 1 else None
    return (res, [{"label": t, "value": t} for t in tifs], hgar_val,
            [{"label": x, "value": x} for x in xls], xl_val)


@callback(
    Output("d-validate-result", "children"),
    Input("d-validate", "n_clicks"),
    State("d-images-dir", "value"), State("d-hgar", "value"),
    State("d-hgar-manual", "value"), State("d-excel", "value"),
    State("d-excel-manual", "value"), State("d-pulses-dir", "value"),
    State("d-workspace", "value"), State("d-structure", "value"),
    prevent_initial_call=True)
def validate(_, images_dir, hgar_dd, hgar_man, excel_dd, excel_man,
             pulses_dir, workspace, structure):
    s = SESSION
    # Trois niveaux, et non deux : un probleme qui n'empeche pas d'avancer ne
    # doit pas bloquer le bouton « etape suivante ». Deux images re-exportees
    # sur 222 sont ecartees proprement par l'extraction — c'est a signaler,
    # pas a traiter comme une configuration invalide.
    issues, warns, oks = [], [], []
    # ── Changement de campagne ────────────────────────────────────────────
    # Les chemins sont poses AVANT tout calcul pour que l'empreinte reflete
    # ce que l'utilisateur vient de saisir. Si elle a change, tout ce qui a
    # ete calcule pour la campagne precedente est efface : sinon la
    # calibration, la table d'energies et la calibration absolue de l'ancienne
    # campagne restaient en place et melangeaient les deux, au point qu'il
    # fallait redemarrer l'application.
    _hg_probe = (hgar_man or "").strip() or (
        str(Path(images_dir) / hgar_dd) if (images_dir and hgar_dd) else "")
    _xl_probe = (excel_man or "").strip() or (
        str(Path(images_dir) / excel_dd) if (images_dir and excel_dd) else "")
    prev_id = s.campaign_id
    s.images_dir, s.hgar_path, s.excel_path = (images_dir or "", _hg_probe,
                                               _xl_probe)
    new_id = s.campaign_key()
    if prev_id and new_id != prev_id:
        cleared = s.reset_computed_state()
        warns.append(
            "Campaign change detected — cleared: "
            + (", ".join(cleared) if cleared else "nothing was computed yet")
            + ". Redo the calibration (step 2) before anything else; the "
              "extraction cache of the other campaign is kept on disk and "
              "will be reused if you come back to it.")
    s.campaign_id = new_id

    if not images_dir or not Path(images_dir).is_dir():
        issues.append("Images folder not found.")
    else:
        s.images_dir = images_dir
        info = s.scan_images()
        (oks if info["n"] else issues).append(
            f"{info['n']} science images detected."
            if info["n"] else "No science image detected in the folder.")
    # HgAr: manual field takes priority when filled
    hgar = (hgar_man or "").strip()
    if not hgar and hgar_dd:
        hgar = str(Path(images_dir) / hgar_dd)
    if hgar and Path(hgar).exists():
        s.hgar_path = hgar
        oks.append(f"HgAr image: {Path(hgar).name}")
    else:
        issues.append("HgAr image missing or not found "
                      "(required for the calibration).")
    # Geometric consistency: science images vs HgAr image
    if s.image_dict and s.hgar_path and Path(s.hgar_path).exists():
        sizes = s.probe_image_sizes()
        hg = s.hgar_size()
        size_desc = "; ".join(f"{n[0]}×{n[1]} px: {len(ks)} images"
                               for n, ks in sizes.items())
        matching = {sz: ks for sz, ks in sizes.items() if sz == hg}
        mismatching = {sz: ks for sz, ks in sizes.items() if sz != hg}
        if mismatching and not matching:
            issues.append(
                f"Dimension mismatch: the HgAr image is "
                f"{hg[0]}×{hg[1]} px but your science images are "
                f"{size_desc}. The pixel→nm calibration cannot be applied "
                f"(different binning or sensor). Use an HgAr image acquired "
                f"in the same conditions as the shots.")
        elif mismatching:
            n_bad = sum(len(ks) for ks in mismatching.values())
            ex = next(iter(mismatching.values()))[:3]
            oks.append(
                f"Warning: {n_bad} images have dimensions different from "
                f"the HgAr image ({size_desc}) — they will be rejected at "
                f"extraction with an explicit message (e.g. {', '.join(ex)}).")
        else:
            oks.append(f"Consistent dimensions: images and HgAr are "
                       f"{hg[0]}×{hg[1]} px.")
        # Format : meme taille ne veut pas dire meme nature d'image.
        modes = s.probe_image_formats()
        bad_modes = {m: ks for m, ks in modes.items()
                     if analysis.image_format_problem(m)}
        if bad_modes:
            n_bad = sum(len(ks) for ks in bad_modes.values())
            for m, ks in bad_modes.items():
                warns.append(
                    f"{len(ks)} image(s) in '{m}' format ("
                    + ", ".join(ks[:6]) + ("…" if len(ks) > 6 else "")
                    + f"): {analysis.image_format_problem(m)}")
            warns.append(f"These {n_bad} image(s) will be skipped at "
                         f"extraction and excluded from the group averages. "
                         f"Everything else can proceed.")
        elif len(modes) > 1:
            warns.append("Several TIFF formats in the folder ("
                         + ", ".join(f"{m}: {len(ks)}"
                                     for m, ks in modes.items()) + ").")
        elif modes:
            oks.append(f"Uniform image format ({next(iter(modes))}).")
    excel = (excel_man or "").strip()
    if not excel and excel_dd:
        excel = str(Path(images_dir) / excel_dd)
    if excel and Path(excel).exists():
        s.excel_path = excel
        ok, msg = analysis.load_energy()
        (oks if ok else issues).append(f"Excel: {msg}")
        ok2, msg2 = analysis.load_metadata()
        (oks if ok2 else issues).append(f"Campaign metadata: {msg2}")
    else:
        oks.append("Excel not set — energy/power correlations will be "
                   "unavailable (everything else works).")
    s.pulses_dir = (pulses_dir or "").strip() or images_dir or ""
    n_pulse = len(s.list_pulse_shots())
    oks.append(f"Pulse CSVs detected: {n_pulse}.")
    if workspace:
        s.workspace = workspace
    s.ensure_workspace()
    # Configurations angulaires depuis les fichiers Excel (optionnel)
    s.structure_path = (structure or "").strip()
    if s.structure_path:
        from core import angles as ang
        if not Path(s.structure_path).exists():
            issues.append("Structure coordinates file not found.")
        else:
            pos_files = ang.scan_position_files(s.images_dir)
            if not pos_files:
                oks.append("No SidescatterFibrePos_*.xlsx file in the "
                           "images folder — pipeline angles only.")
            else:
                try:
                    struct = ang.load_structure_tables(s.structure_path)
                    registry, reports = {}, []
                    for name, f in pos_files.items():
                        ent, iss = ang.build_config_from_files(struct, f)
                        registry[name] = ent
                        status, msg = ang.compare_to_builtin(
                            name, ang.signed_fibres(ent))
                        reports.append((status, msg))
                        for i in iss[:3]:
                            reports.append(("info", f"{name}: {i}"))
                    s.angle_registry = registry
                    ang.save_registry(s.ensure_workspace(), registry)
                    s.angle_report = reports
                    n_id = sum(1 for st, _ in reports if st == "identique")
                    n_new = sum(1 for st, _ in reports if st == "nouveau")
                    n_bad = sum(1 for st, _ in reports if st == "ecart")
                    summary = (f"{len(registry)} angular configurations "
                               f"imported ({n_id} identical to the pipeline, "
                               f"{n_new} new"
                               + (f", {n_bad} DISCREPANT — check!"
                                  if n_bad else "") + ").")
                    (issues if n_bad else oks).append(summary)
                    for st, msg in reports:
                        if st in ("ecart",):
                            issues.append(msg)
                        elif st in ("identique", "nouveau"):
                            oks.append(msg)
                except Exception as e:
                    issues.append(f"Could not import the angular "
                                  f"configurations: {e}")
            # Plan de fibres de campagne : reconstruit a l'identique a chaque
            # validation, pour qu'un redemarrage ne laisse pas une config
            # angulaire perimee dans le registre.
            maps = [p for p in (s.flat_angle_paths or []) if Path(p).exists()]
            if maps:
                from core import angles as ang
                s.load_angle_registry()
                try:
                    struct = ang.load_structure_tables(s.structure_path)
                except Exception as e:
                    struct = None
                    issues.append(f"Angle reference file unreadable: {e}")
                for mp in (maps if struct else []):
                    nm = Path(mp).stem
                    try:
                        ent, rep = ang.build_config_from_map(struct, mp)
                    except Exception as e:
                        issues.append(f"Fiber map '{nm}' unreadable: {e}")
                        continue
                    s.angle_registry[nm] = ent
                    n_res = len(ent["fibres"])
                    (oks if n_res else issues).append(
                        f"Angular configuration '{nm}': {n_res} fibers "
                        f"resolved through the angle reference file.")
                    for lvl, msg in rep:
                        if lvl == "warn":
                            warns.append(f"{nm}: {msg}")
                if struct:
                    ang.save_registry(s.ensure_workspace(), s.angle_registry)
    elif s.flat_angle_paths:
        issues.append("A campaign fiber map is configured but no angle "
                      "reference file: the angles cannot be resolved.")
    s.save_config()
    items = ([html.Li(m) for m in oks]
             + [html.Li(m, className="text-warning") for m in warns]
             + [html.Li(m, className="text-danger") for m in issues])
    color = "danger" if issues else ("warning" if warns else "success")
    head = None
    if issues:
        head = html.P(html.B("Blocking: fix the points in red before "
                             "continuing."), className="mb-2")
    elif warns:
        head = html.P(html.B("Configuration usable — the points in orange "
                             "are warnings, not blockers."), className="mb-2")
    nxt = dbc.Button("Next step: Calibration →", href="/calibration",
                     color="primary", className="mt-2") if not issues else None
    return dbc.Alert([head, html.Ul(items, className="mb-1"), nxt],
                     color=color)
