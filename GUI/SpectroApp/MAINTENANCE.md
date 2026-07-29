# MAINTENANCE.md — Technical documentation

Audience: the (scientist) person who will need to modify or extend the application.

## Guiding principle

**`core/spectro_functions.py` is the scientific ground truth and has NOT been modified** (byte-for-byte copy of the original file). The whole application is an orchestration and display layer around it. If you improve the pipeline, replace that file and re-run the non-regression test.

## Architecture

```
SpectroApp/
├── run.py                  # starts the server (127.0.0.1:8050) + opens the browser
├── Lancer_*.bat/.command/.sh   # launchers: create .venv on first run then execute run.py
├── core/
│   ├── spectro_functions.py    # ORIGINAL PIPELINE, UNCHANGED
│   ├── session.py          # global state (paths, parameters, calibration, tables, nd_files)
│   ├── analysis.py         # orchestration: same sf calls as the notebook
│   ├── plotting.py         # Plotly figures built on the same numpy arrays
│   ├── ndfilters.py        # ND filter transmission parsing + correction (new)
│   ├── metadata.py         # Final.xlsx reading (target/profile/energy/ND/fiberpos)
│   ├── angles.py           # (phi, theta) reconstruction from Excel
│   ├── jobs.py             # extraction batch in a thread + progress
│   └── campaign.py         # default groups (campaign metadata) + persistence
├── app/
│   ├── main.py             # Dash app, sidebar, routing, /download route
│   ├── components.py       # small UI components
│   └── pages/p1..p8_*.py   # one page = one module (layout() + @callback callbacks)
└── tests/test_regression.py    # numeric app <-> notebook comparison
```

## Decisions and invariants (do not break without thinking)

1. **Server singleton session** (`core/session.py`). The app is local single-user: the state (including the calibration object, large and non-JSON) lives on the server side. Do not expose on a shared network without re-architecting (per-session stores, locks).
2. **Cache invalidated by hash** (`Session.calib_hash`). The notebook `.npy` did not know which calibration produced them; here the cache folder is suffixed with an md5 hash of (HgAr file, parameters, manual pairs, SUBTRACT_BG...). Changing the list of hashed parameters = invalidating or reusing cache: think before doing it.
3. **The cache stores the RAW spectra** (before intensity calibration), applied on the fly — same convention as the notebook (cell 10). The **ND correction is likewise applied on the fly, after the cache** (see below), so it never invalidates the cache.
4. **Exact reproduction**: `analysis.poly_fit_ci` and `analysis.exp_fit_ci` are line-by-line replicas of the helpers defined in the notebook (same CI formulas). `BEAM_AREA_CM2` reproduces cell 65. The `.npz`/`.txt` exports of `export_centroid_npz_txt` are in the exact format of cell 31.
5. **Deliberate deviation #1**: the "random demo data" fallback of the area-vs-energy cells has been removed. The app shows the list of skipped shots and never plots invented data.
6. **Deliberate deviation #2**: the 2w energy comes exclusively from `Final.xlsx` (via `sf.load_energy_table`, columns C/F), plus the hard-coded dictionaries. Check the Excel if a point differs from an old plot.
7. **Preserved quirk**: `sf.load_pulse_profile` reads column index 2 of the CSV (Channel 2) whereas its docstring says "col 1". This is the original code's behaviour, validated on your 4-channel CSVs; kept as-is.
8. **Detector image display**: downsampled (`analysis.load_image_preview`) ONLY for display. Every computation uses the full-resolution image.
9. **Physical beam inversion**: handled exclusively by `sf.get_fiber_angles`/`sf.physical_fiber_index`, as in the pipeline. Never re-invert elsewhere.

## Adding an analysis (recipe)

1. Write the computation function in `core/analysis.py` calling `sf.*` (never re-code maths if `sf` can do it).
2. Write the figure builder in `core/plotting.py` (inputs = numpy arrays, output = `go.Figure`).
3. Create/extend a page in `app/pages/`: `layout()` + callbacks decorated with `@callback` (IDs prefixed by page to avoid collisions).
4. Add a case to the non-regression test if the analysis produces publishable numbers.

## Tests

```bash
python -m tests.test_regression <data_folder>
```
The folder must contain the sample HgAr image, `shot431.tif`, `Final.xlsx`, `shot_498.csv`. Tolerance: 1e-9/1e-10 relative. Any divergence = scientific regression, to be handled before delivering.

## v2 additions (user feedback)

- **Fibers 1-80 in display**: the constant `plotting.FIBER_DISPLAY_OFFSET` centralises the convention. Internally, EVERYTHING stays 0-79 (numpy indices); the conversion happens at the UI boundaries (`int(input)-1` in the callbacks). The `.npz`/`.txt` exports stay 0-79 (pipeline compatibility) — change possible but to be decided explicitly.
- **Cache management** (`jobs.cache_inventory/clear_cache`, Extraction page) + "Ignore the cache" option (`start_batch(ignore_cache=True)`).
- **Background SNR map** (`jobs.start_snr`, `SNR_PROGRESS`): the old version computed in the callback (silent and slow on 500+ images); now button + progress bar + listed errors.
- **Editable graphs**: `components.graph()` replaces `dcc.Graph` everywhere (Plotly `editable` config + global preferences `PLOT_FONT_SIZE`/`PLOT_TEMPLATE` applied by `plotting.apply_prefs`).
- **Automatic groups** (`core/metadata.py`): Final.xlsx read by HEADERS (fallback to letters C/D/F/G/BH), target/profile normalisation (spaces, "10 90"->"10/90"), Si % parsing, energy filter, motivated exclusions. The reference energy for the computations stays `sf.load_energy_table` (original path).
- **Angles from Excel** (`core/angles.py`): (phi, theta) reconstruction per physical fiber from structure + positions; VALIDATED identical to the 3 hard-coded configs. Sign rule: theta->-theta for the **Alpha** quadrant (the oral description said Delta; the validated pipeline values match Alpha — explicit `NEGATE_QUADRANT` parameter). Arithmetic extrapolation of ports outside the table ONLY if the table is strictly arithmetic (case of port 14 of Config3_e: reproduces the pipeline value), flagged in the report. Registry persisted in `workspace/angles.json`. Per-shot resolution: `analysis.resolve_config` (Final.xlsx first, fallback to pipeline ranges, disagreements flagged).

### v2.1 fixes
- **SNR**: the polling interval is disabled as soon as the figure is rendered (otherwise the callback re-rendered the map every 0.9 s -> unusable page). Reusable pattern: `Output(interval, "disabled")` set to True at final render, reset to False by the launch button.
- **Correlations**: source selector (`r-source`) — "categories" builds the shot sets directly from `SESSION.metadata` (target x profile, all energies), without going through the saved groups; colours from `metadata.PALETTE`.

### Automatic fiber detection (v3)
`core/fiberdetect.py` — multi-layer algorithm, calibrated against the 80 manual positions of the campaign:
1. lit band + period (mode of gaps between strong peaks);
2. over-complete candidates = intensity maxima (2 smoothings) union curvature maxima (shoulders) union vote maxima;
3. votes = multi-channel persistence: peaks per HgAr line column (sub-pixel) + per derotated science image;
4. exact selection of N fibers by dynamic programming (gaps in [0.70, 1.30]xperiod, max weight);
5. precision: multi-channel guided centroid (median, SNR gate); fibers with <3 channels -> "2 exponential flanks + gaussian" fit (neighbours captured explicitly); fallback to curvature apex, then grid.

Integration: `analysis.run_fiber_detection()` (persisted in `workspace/fiber_auto.json`, invalidated if the HgAr image changes); `run_calibration()` injects the positions via `sf.FIBER_Y_MANUAL` (the pipeline file stays intact; pristine copy `_FIBER_Y_PIPELINE` restored in manual mode). The cache hash includes the mode + the effective positions (corrections included).

## v4 additions (this round)

All four features are additive and leave `spectro_functions.py` untouched.

### 1. ND filter correction (`core/ndfilters.py`, Calibration page)
Different shots may have been recorded with different ND filters in front of the side-SRS spectrometer. The OD is read from column `side-SRS ND` of Final.xlsx (letter AD; `metadata._COLS["nd"]`, exposed as the `nd` field per shot).
- `parse_nd_file(path)` scans the first sheet for a "Wavelength (nm)"/"Transmission (%)" header pair (robust to leading metadata rows/columns), returns (wl_nm, T_fraction), converting % to fraction. Validated on the Thorlabs NEx format (651 points, 200-850 nm).
- `transmission_on_axis(od, wl_axis, nd_files)` returns (T, source): interpolates the measured curve onto the calibration axis when a file is registered for that OD ('file', or 'file+clamp' when the axis extends beyond the measured range — edge values held constant), otherwise the flat theoretical 10^(-OD) ('theory:10^-OD'). `correction_factor` returns 1/T.
- `analysis.apply_nd_correction(spectra, shot_key)` is called inside `analysis.get_spectra` **after** the intensity calibration — on the fly, like the intensity calibration, so the cache stays valid. No-op when `USE_ND_CORRECTION` is off, OD is 0/None, or the shot is unknown.
- `analysis.nd_status_table()` drives the Calibration-page status table (one row per OD: shot count, file, actual source). The registry `SESSION.nd_files` ({od_str: path}) is persisted in `config.json`.
- **Auto-detection of ND files**: `ndfilters.od_in_file(path)` reads the OD from the datasheet's own header text (regex `\bOD\s*[:=]?\s*(number)` scanned over the first 40 rows, so a stray 'OD' in the numeric data can't be mistaken). `ndfilters.scan_nd_folder(folder, wanted_od_keys)` walks a folder, keeps every .xlsx/.xls that both parses as a transmission curve AND exposes an OD, and returns ({od_key: path}, reports). File names are irrelevant — the association is by the OD written inside the file. The Calibration-page callback `nd_scan` merges the result into `SESSION.nd_files` and regenerates the pre-filled input rows; the user still reviews and clicks "Save & check". Clashes (two files, same OD) and ODs unused in the shotbook are reported, not silently applied.
- **Convention decision**: ND 2 means a single OD-2 filter (signal divided by ~100), not stacked filters. One OD value -> one file.
- Parsed curves are cached in memory (`ndfilters._CURVE_CACHE`); `clear_cache()` is called by the Calibration callback when the files change.

### 2. Super-Gaussian focal spot (`analysis.beam_area_cm2`, Laser correlations page)
`beam_area_cm2(sg_n, sg_w_um)` computes the EXACT effective area A_eff = (pi/n)*w^2*Gamma(1/n) for I(r) = I0*exp(-(r/w)^(2n)). Defaults (n=8, w=111 um) reproduce the constant `BEAM_AREA_CM2` bit-for-bit (verified). `correlation_dataset(..., beam_area=...)` takes the area as a parameter (default = the constant, so existing behaviour is unchanged). The page exposes inputs `r-sgn`/`r-sgw`, shows A_eff live (`show_beam_area`), and logs it in the history when the x-axis is intensity.

### 3. Group pulse profiles (`analysis.group_pulse_profiles`, `plotting.fig_group_pulses`/`fig_group_pulse_meanstd`)
Loads the windowed pulse of every shot in a group via `sf.load_pulse_profile` (same windowing as the power computation), aligns them on the rising edge (t=0 at the start of the clean window), normalises each to its own peak, resamples onto a common grid, and returns mean/std with `n_valid` per bin. **Decision**: rising-edge alignment (uses the existing `frac_rise` threshold), peak normalisation. The page ("Pulse profiles" card) has a mode selector (single / category / group); category and group modes show the overlay + mean+/-std side by side, with a consistency badge (mean sigma over the bins where at least 80% of pulses contribute) and an accordion of skipped shots.

### 4. English interface
Full UI translation (all pages, plotting labels, and user-facing messages in `core/`). Internal status codes were preserved where other code depends on them — notably the angles report codes `"identique"/"nouveau"/"ecart"/"info"` (tested in `p1_data.validate`) and the fiber confidence labels emitted by `fiberdetect` (`"directe"/"enfouie"/...`, tested by `p2._flagged_indices` via `startswith("directe")` and coloured by `plotting._conf_color`). Only the human-readable messages around those codes were translated. If you ever translate the confidence labels themselves, update `_flagged_indices`, `_conf_color`, and any persisted `fiber_auto.json` in existing workspaces together.

## v5 additions (multi-campaign robustness)

The app must serve every future campaign, not just the first. Two campaign-specific assumptions were removed.

### Image rotation + grayscale (`analysis` load hook)
Some campaigns export shots (or the calibration) sideways, or as RGB/RGBA instead of a 2-D 16-bit array. Rather than editing `spectro_functions.py`, `analysis` rebinds `sf.load_image` at import time to a wrapper (`_load_image_normalized`) that:
- collapses multi-channel arrays to grayscale (mean of RGB, alpha dropped);
- pre-rotates by a user-chosen multiple of 90°.
Because internal `sf.*` functions call `load_image` by its module-global name, the single rebind covers every path (calibration, extraction, previews, fiber detection). The wrapper distinguishes calibration from shots **by path** (`SESSION.hgar_path` → `CALIB_ROTATION`, everything else → `SHOT_ROTATION`), since the two are sometimes exported in different orientations. Rotations are degrees clockwise (0/90/180/270), stored in params and therefore part of `calib_hash` → changing a rotation invalidates the cache. On the first campaign (2-D images, rotation 0) the wrapper is a strict identity, so no validated result changes. UI: two dropdowns on the Calibration page; `run_calibration` persists them. Caveat: `np.rot90` by 90/270 swaps H and W — harmless for square detectors; if a future campaign has non-square images with different calib/shot rotations, `check_image_compat` may need revisiting.

### ND read from the shotbook, column chosen by the user (`metadata`, Calibration page)
The old code hard-coded the ND column as letter AD / header "side-SRS ND" — which in a new shotbook lands on a completely different diagnostic ("Back SOP ND"). Now:
- `metadata._find_header_row` locates the header row (first row containing a "shot" column) instead of assuming row 1.
- ND has NO letter fallback. `metadata.nd_columns(excel_path)` lists every column whose header contains "ND" as a word; `metadata.auto_nd_column` picks a sensible default (exact "side-SRS ND" → any "SSRS" side-scattering column → a non-SOP/SBS/GOI SRS column → first). `load_final_table(excel_path, nd_column=...)` reads the chosen column (header text), `None` = auto.
- `SESSION.nd_column` (persisted) holds the choice; `analysis.load_metadata()` passes it through.
- UI: the ND card shows a dropdown of all ND columns **with each column's value distribution** (e.g. "1:24, 2:36") so the user can recognise the right one; `nd_change_column` persists the choice, re-reads the shotbook and rebuilds the OD rows. The other metadata columns (shot/profile/energy/target) are still matched by header name and happen to be stable across the known shotbooks; if a future shotbook renames them, extend `_COLS`.

Note: `USE_ND_CORRECTION` is now excluded from `calib_hash` (it's applied on the fly after the cache, so toggling it must not invalidate the raw cache); `SHOT_ROTATION`/`CALIB_ROTATION` are included (they change the raw extraction).

## v6 additions (absolute / physical calibration)

New page 8 "Absolute calibration" (`app/pages/p9_abscal.py`) + `core/abscal.py` (readers) + `analysis.build_absolute_calibration`. Converts detector ADU spectra to physical spectral energy (µJ/nm) per fiber. Three mandatory inputs, none tied to a specific campaign:
- **after-fiber lamp image** (.tiff, science detector) — extracted through the normal pipeline (so it inherits the shot rotation + grayscale hook), giving the per-fiber ADU shape on the science wavelength axis. Usually filtered (e.g. 650 nm high-pass), so only characterised above the cut.
- **before-fiber spectrum** (.spf2) — a single reference spectrum of the same lamp with no filter, spanning the full range. `abscal.read_spf2` reverse-engineers the binary (Ocean-Optics-style): float32 LE wavelength array (located by scanning for a monotone optical-range run) immediately followed by a float32 LE intensity array of the same length. Validated on a 3648-px file (197–1026 nm).
- **per-fiber power** (.txt) — `abscal.read_power_txt` parses `N / value` lines → {fiber: µW}.

Method (confirmed with the user), per fiber i: `I_i = ∫_cut^∞ A_i dλ` (background removed via `abscal.integrate_without_background`, same idea as the spectra noise handling); band fraction `f = ∫_cut^∞ B / ∫_all B` from the unfiltered spectrum (the power meter reads the full band, no filter, so only `P_i·f` falls in the characterised band); `κ_i = P_i·f / I_i` [µW/(ADU·nm)]; energy factor `g_i = κ_i · t_after[s]` [µJ/(ADU·nm)] → `E_i(λ) = g_i · S(λ)`. The fiber transfer function `τ_i(λ) = (A_i/t_after)/(B/t_before)` uses BOTH integration times (masked below the cut). Integration times auto-read from the TIFF (`abscal.tiff_exposure_ms`, PVCAM `expTime=` tag) and always user-editable; filter cut + which file carries it are user inputs too.

Storage: `SESSION.abs_cal` (scalars + per-fiber vectors) is persisted in `config.json` (~9 kB); the heavy per-λ arrays (`SESSION.abs_cal_arrays`: A, B, τ) are memory-only and recomputed on demand by `analysis.abs_cal_rebuild_arrays`. `analysis.to_absolute_energy(spectra)` applies the factors. Plots: `plotting.fig_abs_*` (before-fiber + shaded excluded band, after-fiber spectra, transfer function, per-fiber factor, power-vs-ADU consistency scatter, calibrated µJ/nm demo). The power-vs-ADU scatter is the key sanity check: outliers flag a mis-detected fiber position or a dark/saturated fiber. Placed at step 3 (right after the wavelength calibration, before extraction and analysis): it is a per-campaign calibration that reuses the step-2 wavelength axis and fiber positions and does not depend on science extraction. Nav renumbered accordingly (Extraction 4, ..., Exports 9).

## v7 additions (physical units everywhere + abscal robustness)

- **Display-units switch.** `SESSION.display_units` ("adu"/"uJ", persisted) + `analysis.to_display_units(spectra) -> (array, ylabel)` and `analysis.current_units()` (falls back to "adu" unless a calibration exists). A shared `components.units_radio(id)` renders an inline ADU / µJ-per-nm control (energy disabled until `abs_cal_ready()`). Each analysis page adds one radio and threads its value into the relevant plotting callbacks, converting the ADU spectra with `to_absolute_energy` (per-fiber ×g_i) before computing/plotting and relabelling the axis:
  - Exploration (`p4`): all-80 heatmap (colorbar), single fiber (+area unit), fiber comparison, temporal evolution.
  - Groups (`p5`): individual + mean±std spectra, 3D surfaces (stack ×g[None,:,None]), area tables, Si×profile map.
  - Angular (`p6`): the "area" metric map and the 3D sphere colour (centroid is normalised, so untouched).
  - Correlations (`p7`): `analysis.correlation_dataset(..., units=)` converts before `compute_spectral_area`, so the y-axis becomes spectral area in µJ.
  The centroid / SNR / detector-image views stay unit-agnostic by design. The conversion is a pure per-fiber scalar multiply, so it never changes the shape of any result, only its scale + label.
- **Absolute-calibration robustness.** The transfer function τ is now gated where the SMOOTHED before-fiber signal drops below 8% of its max (`scipy.ndimage.uniform_filter1d`, size 25), removing the division-by-~0 blow-up beyond the lamp's useful range (was dominating the plot past ~850 nm). Outlier fibers (robust MAD, |z|>5) are flagged in `SESSION.abs_cal["outliers"]`, drawn red in `plotting.fig_abs_factor`, and listed in the build message — usually a mis-detected fiber position or a dark/saturated fiber (the power-vs-ADU scatter is the diagnostic).

## Known limitations (honest)

- Flask development server: perfect locally, not hardened for network exposure.
- One extraction batch at a time (`core/jobs.py`) — enough for the use case, simple to reason about.
- The SNR map reloads all cached spectra: on 546 images expect ~1 GB of transient RAM. If that is a problem, stream in batches in `p4_explore.tab_snr`.
- Plotly figures PNG export goes through the browser (camera button); server-side PNG export (kaleido) was not included to keep the install light. The "publication figures" mode (original matplotlib) covers the need for paper figures.
