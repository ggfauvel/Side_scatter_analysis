# Spectro Multi-Fiber — Analysis application

Local application for analysing multi-fiber spectra (80 fibers): HgAr calibration, extraction, group comparisons, 3D angular maps, laser correlations. No programming knowledge required.

## Installation (once)

1. **Install Python** (version 3.10 or newer) if not already done: https://www.python.org/downloads/
   - **Windows**: during installation, be sure to tick the **"Add Python to PATH"** box.
2. Unzip the `SpectroApp` folder wherever you like (e.g. Documents).
3. Double-click the launcher for your system:
   - **Windows**: `Lancer_Windows.bat`
   - **macOS**: `Lancer_Mac.command` (if macOS blocks it: right-click -> Open)
   - **Linux**: `Lancer_Linux.sh`

The first launch automatically installs the required libraries (a few minutes, internet connection needed). Subsequent launches are immediate.

## Usage

The launcher opens a black window (the "engine" — do not close it while using the app) then your browser on the application. Follow the steps in the left-hand menu, in order the first time:

1. **Data** — point to the folder containing your `shotNNN.tif` images. The HgAr image, the energy Excel file and the pulse CSVs are detected automatically when they are in the same folder. Validate.
2. **Calibration** — click "Run calibration" (~5 s). Check that the RMS is low (< 0.3 nm) and that the green lines fall on the spectral peaks. This page also hosts the optional **ND filter correction** (see below).
3. **Absolute calibration** *(optional, before the analysis)* — turn the ADU spectra into physical energy (uJ/nm) per fiber. Needs three files (all mandatory): the after-fiber lamp image (spectral shape through the fibers), the before-fiber spectrum (.spf2, used to correct for the filtered band) and the per-fiber power meter file (.txt, absolute scale). Integration times and the filter cut are auto-detected when possible and always editable. Produces the per-fiber ADU->energy factor, the fiber transfer function, and control plots to spot bad fibers. It reuses the wavelength axis and fiber positions from step 2, so run the calibration first. Once it is built, the analysis pages (Exploration, Groups, Angular maps, Laser correlations) gain an **ADU / µJ/nm** switch so every spectrum, area and map can be shown in physical units.
4. **Extraction** — run the processing of all images. It runs in the background (progress bar); you can explore meanwhile. Count ~2 s per image.
5. **Exploration** — free inspection: detector image, 80 spectra, a single fiber, fiber comparison, temporal evolution, SNR map.
6. **Groups & comparisons** — your campaign groups (Si %, pulse profile) are pre-loaded and editable. 2D comparisons, mean +/- standard deviation, rotatable 3D views, Si x profile map.
7. **Angular maps & 3D** — (theta, phi) map, mouse-controlled 3D sphere (replaces the GIF), centroid-vs-theta profile with `.npz`/`.txt` export in the usual formats.
8. **Laser correlations** — spectral area vs 2w energy, peak power or peak intensity, with fits (polynomial or exponential) and confidence bands. Three shot sources: the **campaign categories** (target x profile read from Final.xlsx, ALL energies — recommended to study the effect of E/P/I at fixed condition), the saved groups as-is, or a manual list. This page also lets you set the **super-Gaussian focal spot** and inspect **pulse profiles per group** (see below).
9. **Exports & history** — every produced file (one-click download), "publication" figures with the original matplotlib rendering, log of your analyses.

### What's new

### Working with any campaign

The application is not tied to the first campaign's files or image layout:

- **Image rotation.** Some campaigns export the shots (or the calibration image) sideways. On the Calibration page, "Rotate science images" and "Rotate calibration image" (None / 90° / 180° / 270°) bring them into the orientation the pipeline expects — fibers horizontal, like the calibration. The two are independent because shots and calibration are sometimes exported differently. Check the result in the Exploration tab; changing a rotation re-extracts the affected images (the cache is keyed on it). Multi-channel images (RGB/RGBA exported by some tools) are converted to grayscale automatically.
- **ND read from the shotbook.** The OD is read from your shotbook, whatever its name — not from a fixed "Final.xlsx". Because different diagnostics each have their own ND column (SRS ND, side-SRS ND, Resolved-SSRS ND, Back SOP ND…), the ND card shows a "ND column" dropdown listing every ND column with its value distribution (e.g. "1:24, 2:36"), so you can pick the one for the multi-fiber side-scattering spectrometer. A sensible default is selected automatically; change it if your setup differs. The choice is saved.

- **ND filter correction** (Calibration page). Shots recorded with different neutral-density filters can be made directly comparable. Turn on the "ND filter correction" switch, then in the "ND filter correction" card pick the ND column (above) and click "Detect the ND values in the shotbook". To attach the transmission curves, either point to a folder and click "Scan folder for ND files" — the application reads the OD written inside each datasheet (e.g. a cell "…OD: 1.0") and matches every file to the right ND value automatically, whatever the file is named — or paste each file path by hand. The curve is interpolated onto the calibration wavelength axis and every spectrum is divided by T(lambda). If no file is provided for a value, the flat theoretical transmission 10^(-OD) is used instead (clearly flagged). The correction is applied on the fly — the extraction cache stays valid, so you can toggle it without re-extracting.
- **Super-Gaussian focal spot** (Laser correlations page). Peak intensity divides peak power by the effective area of the focal spot, A_eff = (pi/n)*w^2*Gamma(1/n) for the profile I(r) = I0*exp(-(r/w)^(2n)). The order *n* and radius *w* (um) are now adjustable inputs (defaults n = 8, w = 111 um reproduce the original pipeline value), because the focal spot may change between campaigns. The effective area is shown live.
- **Group pulse profiles** (Laser correlations page, "Pulse profiles" card). Beyond a single pulse, you can now load every pulse of a category or a saved group, aligned on their rising edge and normalised to their own peak, shown overlaid next to their mean +/- standard-deviation envelope — a quick check of pulse-to-pulse consistency within a group.
- **English interface** — the whole interface is now in English for the scientific community.

### Graph tips
- **Zoom**: click-drag. **Reset**: double-click. **Export to PNG**: camera icon (high resolution). **Hide a curve**: click its legend entry.
- **Customisation**: click directly on a title, an axis title or a legend text to edit it. Global font size and theme: Exports page -> "Graph display preferences". Group colour: the group editor.
- The 3D views rotate with the mouse and zoom with the wheel.

### Fiber numbering
Fibers are numbered **1 to 80** everywhere in the interface. The only intentional exception: in the `.npz`/`.txt` export files, the `fiber_idx` and `phys_fiber` columns keep the 0-79 convention of the original pipeline, to stay compatible with your existing tools (this is written inside the files themselves).

### Automatic groups (generalisation to future campaigns)
Groups page -> "Automatic groups from Final.xlsx": the application reads, for each shot, the target, the pulse profile and the 2w energy, proposes one group per (target x profile) combination — with an optional "central energy +/- tolerance %" filter — and shows the full assignment (including every exclusion and its reason) BEFORE you validate. You can create several sets: e.g. "@ 600 J +/- 15 %" groups to compare at fixed energy, and all-energy groups for the area-vs-E2w correlations.

### Automatic fiber positions (generalisation)
Calibration page -> "Fiber positions": the **Manual** mode uses the validated pipeline table (current campaign, bit-identical results); the **Automatic detection** mode finds the 80 fibers on the derotated HgAr image, reinforced by a few shot images. A single computation per campaign (~30 s, persisted). The detection classifies each fiber: *direct* (measured on >=3 channels, sub-pixel), *buried* (fitted through the flanks of bright neighbours) or *grid* (interpolated). Non-direct fibers are flagged with a dedicated zoom and a correction field: you only check 2-4 fibers instead of pointing all 80. Any correction automatically invalidates the cache.

### Automatic fiber angles (generalisation)
Data page: provide the structure coordinates file (`...EstimateStructureCoordinates.xlsx`) and place the `SidescatterFibrePos_<Config>.xlsx` files in the images folder. The application builds the (phi, theta) angles of each configuration and **compares them to the pipeline values when they exist** (report shown: identical / new / discrepancy). The configuration of each shot is read from the `side-SRS fibrePos` column of Final.xlsx. For a new campaign: simply provide the new Excel files in the same format.

## Where do the files go?

In the **workspace folder** (by default `SpectroApp/workspace`, editable on the Data step):
- `cache_XXXXXXXXXX/`: extracted spectra (`.npy`). The `XXXXXXXXXX` code identifies the calibration: if you change a calibration parameter, a new cache is created and the old one is never reused by mistake.
- `outputs/`: spectra CSVs, `.npz`/`.txt` exports, publication figures.
- `groups.json`: your group definitions.
- `history.jsonl`: log of the analyses.

The ND filter file paths you provide are stored in `config.json` (key `nd_files`), so you set them once.

## Frequently asked questions

**The application says an image/energy/CSV is missing.** This is intentional: the application never invents data. The message lists exactly what is missing; complete the folder and re-run.

**Can I close the browser?** Yes, reopen http://127.0.0.1:8050. It is the black window that must stay open.

**Can two people use it at the same time?** No — it is a personal local tool, one session at a time.

**The colours/fonts differ from the notebook.** The interactive graphs use a different rendering engine; **the plotted data are identical** (verified by automatic tests). For figures with a strictly identical matplotlib rendering: Exports page -> "Publication figures".
