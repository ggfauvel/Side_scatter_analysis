# Spectro Multi-Fibres

Analysis of 80-fiber scattering spectra: turns raw detector images into
calibrated spectra, then into angular maps of the scattered light on the
collection sphere.

**[→ Read the user guide (PDF)](SpectroMultiFibres_UserGuide.pdf)** — step by
step, no programming needed.

---

## Install

Python 3.10 or later.

```bash
pip install dash dash-bootstrap-components plotly numpy scipy \
            pillow tifffile openpyxl matplotlib
```

## Run

```bash
python run.py
```

The browser opens on <http://127.0.0.1:8050>. Everything runs locally: no data
leaves your machine. Closing the terminal stops the application.

> Keep the terminal window visible — if something goes wrong, the full error
> message appears there.

## What you need

| File | What it is |
|---|---|
| `shotNNN.tif` | One detector image per shot, **16-bit greyscale** |
| HgAr image | Mercury-argon lamp, acquired in the same conditions as the shots |
| Excel shotbook | Laser energies and campaign metadata |
| Angle reference file | φ per arm, θ per port — the same file for every campaign |
| Fiber map(s) | Quadrant, arm and port of each fiber — one file per configuration |
| Pulse CSVs | Optional, for peak power and peak intensity |

Images re-exported in colour or in 8 bits are detected and refused: they look
identical in a file browser but their values are capped at 255 instead of
65535.

## The nine pages

1. **Input data** — point to your files; the application checks them
2. **Calibration** — fiber positions, tilt, pixel → wavelength
3. **Absolute calibration** — ADU → J/nm *(optional)*
4. **Extraction** — every image into 80 spectra, in the background
5. **Exploration** — one image, one spectrum, one fiber
6. **Groups & comparisons** — compare conditions, find aberrant shots
7. **Angular maps & 3D** — where the light goes on the collection sphere
8. **Laser correlations** — spectral area vs energy, peak power, peak intensity
9. **Exports & history** — files produced and the trace of your analyses

## Design principle

The application does not guess. When information is missing or two files
disagree, it says so and names what it found, rather than filling in a
plausible default. Warnings that do not block are orange; only genuinely
blocking problems are red.

If you find it choosing silently on your behalf, that is a bug worth
reporting.

## Reporting a problem

Include what you did, what you expected, what happened, and the contents of
the terminal window. For anything angular, add the import report from page 1
and the convention line printed under the 3D figure.
