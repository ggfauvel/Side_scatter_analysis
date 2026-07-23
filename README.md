# ABSOLUTE-FIBER-CALIBRATION.py

Absolute spectral calibration pipeline for the ELI-MAY side-scatter (SRS/TPD) fiber-coupled spectrometer camera. Converts raw integrated camera counts per fiber/wavelength bin into an absolute spectral energy density (J/nm) at the fiber entrance, traceable to a NIST/manufacturer-calibrated power meter through an integrating-sphere IR-LED transfer standard.

## Physical model

The calibration chain is:

```
S120C power meter reading (spectral setting) --[responsivity correction]--> true optical power on power-meter aperture
    --[inverse-square distance + aperture-area scaling]--> irradiance / power entering each fiber core
    --[longpass filter transmission]--> power in the passband reaching the camera
    --[IR-LED spectral shape after longpass]--> spectral power density P(λ) [W/nm] incident per fiber
    --[divide by measured camera counts/s]--> calibration_J_per_nm_per_count[fiber, λ]
```

Per fiber `i` and wavelength bin `j`:

```
calibration_J_per_nm_per_count[i, j] = P_i(λ_j) / (counts_i(λ_j) / t_exp)
```

where `P_i(λ)` is built from the power-meter-measured power entering fiber `i`, scaled by the longpass-filtered fraction of the sphere-measured IR-LED spectrum, and normalized so `∫P_i(λ)dλ` equals the passband power.

The raw ratio is noisy where the LED spectrum is weak (large λ), so the pipeline saves a smoothed version by default (`calibration_J_per_nm_per_count`) alongside the raw point-by-point curve (`calibration_J_per_nm_per_count_raw`) for diagnostics.

**Downstream usage:**
```python
spectral_energy_J_per_nm = extracted_counts * calibration_J_per_nm_per_count
spectral_power_W_per_nm  = spectral_energy_J_per_nm / exposure_time_s
```

### Key corrections applied

- **Power-meter spectral responsivity**: the S120C is read at a single wavelength setting (`powermeter_selected_wavelength_nm`); the true power is corrected by the ratio of responsivity at that setting to the LED-spectrum-weighted effective responsivity, using the measured A/W sensitivity curve.
- **Sphere geometry**: inverse-square correction for the power-meter aperture plane being offset from the fiber-entrance plane on the integrating sphere (`powermeter_sensor_closer_to_led_by_cm`), plus area scaling from power-meter aperture to fiber core.
- **NA solid-angle correction**: optional (`apply_na_solid_angle_correction`), off by default — a centered point-like LED sits inside the fiber acceptance cone, so no NA² factor is needed unless modeling a diffuse/Lambertian source.
- **Longpass filter**: hard cutoff at `longpass_cutoff_nm` (default step function) or a measured transmission curve (`longpass_filter_curve_csv`).
- **Hard physical zero-energy cutoff**: below `zero_energy_below_wavelength_nm` (default 660 nm, vs. a 650 nm longpass), calibration coefficients are forced to exactly zero — enforced after every smoothing/replacement step so no interpolation or averaging can reintroduce nonzero response in a region with no physically transmitted light.
- **Optional fiber attenuation model** (off by default): predictive fiber-length attenuation of the LED spectrum; normally left off because the empirical camera-count calibration already includes the full fiber+spectrometer+camera chain.

## Pipeline steps (`build_absolute_calibration`)

1. Load calibration (IR-LED-on) and background (IR-LED-off) camera images; subtract background; optionally rotate 90° CCW to match the fiber-reference orientation.
2. Integrate each fiber's trace column-by-column in pixel space using fractional (sub-pixel) boundaries derived from the fiber reference file (`reference_y_lines`, `compute_integration_boundaries`, `fractional_column_integral`), with configurable integration width (`integration_width_calibration`) relative to inter-fiber spacing.
3. Map the pixel x-axis to a wavelength axis via the Hg-lamp calibration produced upstream by `HG-FOAM-CALIBRATION.py`.
4. Load and normalize the true IR-LED spectrum measured on-sphere (`.spf2` binary reader or generic 2-column CSV/TXT).
5. Compute the power-meter spectral correction factor and the true optical power entering each fiber (`compute_powermeter_spectral_correction`, `compute_power_entering_fibers`).
6. Apply the longpass transmission and the physical zero-energy cutoff to get the passband-filtered, normalized LED spectral density.
7. Compute the raw per-fiber, per-wavelength calibration coefficient as measured power spectral density divided by measured counts/s.
8. Smooth the saved calibration curves (per fiber) using one of: `spline` (weighted `UnivariateSpline`, weighted by LED density and counts, with a two-pass sigma-clipped refit), `moving_average`, `gaussian`, `savgol`, `median_filter`, or `none`. Smoothing is done in log10 space by default (`calibration_smoothing_fit_in_log_space`) since calibration noise is multiplicative.
9. Optionally replace manually flagged bad fibers (`weird_fiber_ids`) with the mean/median curve of the remaining fibers.
10. Optionally collapse all fibers to a single common average curve (`use_common_average_curve_for_all_fibers`).
11. Re-apply the hard zero-energy cutoff as a final step (idempotent safety net).
12. Assemble power/counts summary tables and metadata.

## Directory layout expected

Run from `C:\Users\marti\OneDrive\Desktop\GenF\ELI-SIDESCATTER-INTEGRATED` (or update `ROOT_DIR`). Required inputs:

```
Data/ELI-MAY/Calib_IRlight_650nmhigpass_ND2A_10sacq_100umslit(1).tiff   # IR-LED-on camera image
Data/ELI-MAY/background_reference_10s_20260528.tiff                     # IR-LED-off background
Fiber-References/fiber_reference_ELI-MAY.csv                            # per-fiber (slope, intercept) trace fit, fiber_id column
Calibration/hg_wavelength_calibration_axis_ELI-MAY_refELI-MAY.csv       # from HG-FOAM-CALIBRATION.py
Calibration/Calorimeter-Sensitivity.csv                                 # power-meter responsivity, A/W vs nm
Calibration/power calibration.txt                                      # fiber_id / power_reading pairs (µW by default)
Calibration/fiber-on-sphere.spf2                                        # true IR-LED spectrum measured on the sphere
```

Exposure time is parsed from the calibration image filename (`..._10sacq_...`) unless `calibration_exposure_time_s` is set explicitly.

## Outputs (written to `Calibration/`)

| File | Content |
|---|---|
| `absolute_fiber_spectral_calibration_ELI-MAY.npz` | Main arrays: `calibration_J_per_nm_per_count[fiber, λ]` (smoothed) and `_raw` variant, wavelength axis, bin widths, power/counts arrays |
| `absolute_fiber_spectral_calibration_ELI-MAY_long.csv` | Long-format table of the above |
| `absolute_fiber_power_summary_ELI-MAY.csv` | Per-fiber power-meter readings, corrections, and power entering each fiber |
| `absolute_calibration_metadata_ELI-MAY.json` | Run metadata: correction factors, smoothing settings, cutoffs |
| `absolute_calibration_summary_ELI-MAY.png` | Selected fiber calibration curves + normalized LED-after-longpass spectrum |
| `absolute_calibration_map_ELI-MAY.png` | 2D map of calibration coefficient vs. fiber × wavelength |
| `absolute_calibration_power_vs_fiber_paper_style_ELI-MAY.png` | Power entering each fiber vs. fiber index |
| `absolute_calibration_fiber_power_triangulated_ports_ELI-MAY.png` | 3D triangulated-sphere plot of power vs. fiber port angle (φ, θ) |
| `absolute_calibration_fiber_raw_counts_triangulated_ports_ELI-MAY.png` | Same 3D sphere, colored by raw uncalibrated counts |
| `absolute_calibration_image_fiber_traces_integration_regions_ELI-MAY.png` | Raw image overlay with fiber traces and integration bands, for verifying trace alignment |
| `absolute_calibration_curves_fibers_*_ELI-MAY.png` (×4) | 5×4 grids of raw + smoothed calibration curves for all 80 fibers |
| `absolute_calibration_weird_fiber_replacements_ELI-MAY.csv` | Log of which fibers were replaced by the average curve |
| `absolute_calibration_common_average_curve_ELI-MAY.csv` / `_status_ELI-MAY.csv` | Common-average-curve outputs, if enabled |

## Configuration

All parameters are module-level constants near the top of the script (no CLI/argparse). Notable groups:

- **Paths & I/O** — image/reference/calibration file locations, output prefixes.
- **Geometry** — `sphere_radius_cm`, `powermeter_sensor_closer_to_led_by_cm`, `powermeter_input_aperture_diameter_mm`, `fiber_core_diameter_um`, `fiber_numerical_aperture`.
- **Smoothing** — `calibration_smoothing_method` (`spline` / `moving_average` / `gaussian` / `savgol` / `median_filter` / `none`), method-specific window/order parameters, log-space fitting, outlier clipping.
- **Bad-fiber handling** — `weird_fiber_ids`, average-curve replacement, optional single common curve for all fibers.
- **Plot style** — a single `PLOT_STYLE` dict controlling fonts, line widths, colormaps, and export settings for publication-style figures, plus separate config blocks for the 3D sphere plots and the 2D image-overlay diagnostic.

Edit constants directly and rerun; there is no separate config file.

## Dependencies

```
numpy, pandas, matplotlib, Pillow, scipy
```

`scipy.interpolate` (`interp1d`, `UnivariateSpline`, `griddata`), `scipy.ndimage` (`gaussian_filter1d`, `median_filter`), `scipy.signal.savgol_filter`.

## Usage

```bash
python ABSOLUTE-FIBER-CALIBRATION.py
```

Runs `build_absolute_calibration()` → `save_outputs()` → `make_plots()`, prints the power-meter spectral correction factor, the longpass fraction of LED power on the camera axis, and the median power entering a fiber after the longpass filter, then displays all figures interactively (`plt.show()`).

## Notes / caveats

- The `.spf2` reader (`read_spf2_spectrum`) is a heuristic binary parser: it scans for the longest monotonically increasing float32 block in a plausible wavelength range (100–2000 nm) and treats the next equal-length block as the signal. Verify against the instrument software if the sphere reference file format changes.
- Fiber integration uses fractional pixel overlap (`fractional_column_integral`), not nearest-pixel rounding, so integration boundaries derived from `integration_width_calibration` are sub-pixel accurate.
- `apply_na_solid_angle_correction=False` is the physically correct default for the current centered point-source LED geometry; only enable for a diffuse/Lambertian source model.
- Downstream analysis scripts must read the **smoothed** arrays (`calibration_J_per_nm_per_count`), not `_raw`, unless specifically diagnosing calibration noise.
