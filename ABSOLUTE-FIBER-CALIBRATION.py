r"""
ABSOLUTE-FIBER-CALIBRATION.py

Purpose
-------
Build an absolute spectral calibration for the ELI-MAY side-SRS fiber camera.
The output converts extracted camera counts in each wavelength bin into spectral
energy density at the fiber entrance, in J/nm.

Run this script from:
    C:\Users\marti\OneDrive\Desktop\GenF\ELI-SIDESCATTER-INTEGRATED

Expected project structure:
    Data/ELI-MAY/Calib_IRlight_650nmhigpass_ND2A_10sacq_100umslit(1).tiff
    Data/ELI-MAY/background_reference_10s_20260528.tiff
    Fiber-References/fiber_reference_ELI-MAY.csv
    Calibration/hg_wavelength_calibration_axis_ELI-MAY_refELI-MAY.csv
    Calibration/Calorimeter-Sensitivity.csv
    Calibration/power calibration.txt
    Calibration/fiber-on-sphere.spf2

Outputs:
    Calibration/absolute_fiber_spectral_calibration_ELI-MAY.npz
    Calibration/absolute_fiber_spectral_calibration_ELI-MAY_long.csv
    Calibration/absolute_fiber_power_summary_ELI-MAY.csv
    Calibration/absolute_calibration_metadata_ELI-MAY.json
    Calibration/absolute_calibration_summary_ELI-MAY.png
    Calibration/absolute_calibration_map_ELI-MAY.png

Main quantity saved:
    calibration_J_per_nm_per_count[fiber_id, wavelength_index]

By default this is a spline-smoothed calibration curve, not the raw noisy
point-by-point division. The raw calibration is also saved for diagnostics as
calibration_J_per_nm_per_count_raw.

Usage in plotting:
    spectral_energy_J_per_nm = extracted_counts * calibration_J_per_nm_per_count
    spectral_power_W_per_nm  = spectral_energy_J_per_nm / exposure_time_s
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm, colors, tri as mtri
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.ticker import AutoMinorLocator, LogLocator, NullFormatter
from PIL import Image
from scipy.interpolate import interp1d, UnivariateSpline, griddata
from scipy.ndimage import gaussian_filter1d, median_filter
from scipy.signal import savgol_filter


# ============================================================
# PROJECT PATHS
# ============================================================

ROOT_DIR = Path(__file__).resolve().parent
DATA_ROOT = ROOT_DIR / "Data"
DATA_MAY_DIR = DATA_ROOT / "ELI-MAY"
CALIBRATION_FOLDER = ROOT_DIR / "Calibration"
FIBER_REFERENCE_FOLDER = ROOT_DIR / "Fiber-References"

CALIBRATION_FOLDER.mkdir(parents=True, exist_ok=True)
FIBER_REFERENCE_FOLDER.mkdir(parents=True, exist_ok=True)


# ============================================================
# USER PARAMETERS
# ============================================================

# Camera image acquired with IR LED ON.
calibration_image_path = DATA_MAY_DIR / "Calib_IRlight_650nmhigpass_ND2A_10sacq_100umslit(1).tiff"

# Camera image acquired with IR LED OFF.
background_image_path = DATA_MAY_DIR / "background_reference_10s_20260528.tiff"
subtract_background_image = True
background_scale_factor = 1.0
clip_negative_after_background_subtraction = True

# Exposure time. If None, it is extracted from the calibration image filename.
# Example: "..._10sacq_..." -> 10 s.
calibration_exposure_time_s = None

# Fiber reference and integration.
reference_file = FIBER_REFERENCE_FOLDER / "fiber_reference_ELI-MAY.csv"
integration_width_calibration = 0.95
integrated_x_step_px = 1
expected_number_of_fibers = 80

# Must match the image orientation used to build the fiber reference.
# Keep False if fiber_reference_ELI-MAY.csv was made on the raw image orientation.
# Set True if the May images must be np.rot90(img, k=1) before extraction.
rotate_ccw_90_image = True

# Wavelength calibration made by HG-FOAM-CALIBRATION.py.
wavelength_calibration_axis_csv = CALIBRATION_FOLDER / "hg_wavelength_calibration_axis_ELI-MAY_refELI-MAY.csv"
wavelength_calibration_axis_npz = CALIBRATION_FOLDER / "hg_wavelength_calibration_ELI-MAY_refELI-MAY.npz"

# Power meter responsivity curve, in A/W.
powermeter_responsivity_csv = CALIBRATION_FOLDER / "Calorimeter-Sensitivity.csv"

# Power meter readings at the fiber ports.
power_meter_readings_file = CALIBRATION_FOLDER / "power calibration.txt"
power_file_units = "uW"
power_file_fiber_ids_are_one_based = True

# IR LED true spectrum measured by the fiber spectrometer.
ir_led_true_spectrum_file = CALIBRATION_FOLDER / "fiber-on-sphere.spf2"

# Power meter wavelength setting used during the measurements.
powermeter_selected_wavelength_nm = 750.0

# Geometry.
sphere_radius_cm = 44.0
# Positive value means the S120C detector/aperture plane was this much CLOSER
# to the LED than the fiber entrance plane. The irradiance at the fiber plane is
# then smaller by ((R - offset) / R)^2.
powermeter_sensor_closer_to_led_by_cm = 4.5

# S120C input aperture diameter. The spec sheet gives input aperture = 9.5 mm.
powermeter_input_aperture_diameter_mm = 9.5

# Fiber core diameter used in the experiment.
fiber_core_diameter_um = 100.0

# Fiber NA. For a centered point-like LED this is normally only a check, not a
# multiplicative correction, because the source is inside the acceptance cone.
fiber_numerical_aperture = 0.22
apply_na_solid_angle_correction = False
# If True, assumes a Lambertian/diffuse field at the fiber face and multiplies
# by NA^2. Do NOT enable for the direct centered LED geometry unless this is the
# angular model you want.

# Longpass filter model between fiber output and spectrometer/camera.
# If you have a measured filter transmission curve, set longpass_filter_curve_csv.
longpass_cutoff_nm = 650.0
longpass_filter_curve_csv = None
# CSV format for filter curve, if used: first numeric column wavelength [nm],
# second numeric column transmission in either 0..1 or 0..100.

# Physical hard cutoff for the final absolute calibration.
# Any camera counts measured below this wavelength are forced to calibrate to
# zero energy, because they cannot correspond to a physically transmitted signal
# through the 650 nm longpass setup.
force_zero_energy_below_wavelength = True
zero_energy_below_wavelength_nm = 660.0

# Optional fiber attenuation curve. Usually NOT applied here, because this
# absolute calibration should empirically include the fiber + spectrometer +
# camera chain through the measured camera counts. Set True only if you really
# want to predict spectral reshaping by fiber attenuation from a known length.
apply_fiber_attenuation_curve_to_led_spectrum = False
fiber_attenuation_csv = CALIBRATION_FOLDER / "Fiber-Attenuation.csv"
fiber_length_m = 10.0

# Output files.
output_prefix = "absolute_fiber_spectral_calibration_ELI-MAY"
output_npz = CALIBRATION_FOLDER / f"{output_prefix}.npz"
output_long_csv = CALIBRATION_FOLDER / f"{output_prefix}_long.csv"
output_power_summary_csv = CALIBRATION_FOLDER / "absolute_fiber_power_summary_ELI-MAY.csv"
output_metadata_json = CALIBRATION_FOLDER / "absolute_calibration_metadata_ELI-MAY.json"
output_summary_plot_png = CALIBRATION_FOLDER / "absolute_calibration_summary_ELI-MAY.png"
output_map_plot_png = CALIBRATION_FOLDER / "absolute_calibration_map_ELI-MAY.png"

# Plot options.
selected_fibers_to_plot = [0, 9, 20, 30, 38, 50, 60, 70, 79]
plot_wavelength_min_nm = 650.0
plot_wavelength_max_nm = 950.0
plot_raw_calibration_curves_under_spline = True
# If True, selected-fiber plots show the raw noisy point-by-point calibration
# as faint dotted curves behind the spline-smoothed calibration.

# Smoothing of the saved absolute calibration.
# This is recommended because the IR LED spectrum becomes very weak at large
# wavelengths, so the raw ratio (known spectral power)/(camera counts/s) becomes
# noisy. The plotting code should read the smoothed arrays.
smooth_saved_absolute_calibration = True

calibration_smoothing_method = "moving_average"
# Options:
#   "none"          = save raw point-by-point calibration curves.
#   "spline"        = weighted smoothing spline, best for smooth transmission trends.
#   "moving_average"= rolling mean in wavelength.
#   "gaussian"      = Gaussian filter in wavelength.
#   "savgol"        = Savitzky-Golay polynomial smoothing.
#   "median_filter" = rolling median filter, useful against isolated spikes.

calibration_smoothing_fit_in_log_space = True
# True keeps the calibration positive and smooths multiplicative noise.
# Backward-compatible alias used by the old spline code:
calibration_spline_fit_in_log_space = calibration_smoothing_fit_in_log_space

calibration_smoothing_min_points = 30
# Minimum finite positive wavelength points needed for any smoothing method.
# Backward-compatible alias used by the old spline code:
calibration_spline_min_points = calibration_smoothing_min_points

# Spline-specific parameters.
calibration_spline_smoothing_factor = 0.05
# Larger = smoother. Good range to try: 0.05, 0.1, 0.35, 1.0, 3.0.
# The code uses s = factor * N * variance(log10(calibration)) for each fiber.

calibration_spline_order = 3
# 3 = cubic spline. Use 2 if a fiber has too few good points.

calibration_spline_outlier_clip_sigma = 5.0
# After a first spline pass, points with residuals above this robust sigma are
# ignored in a second pass. Set None to disable.

calibration_spline_weight_by_led_and_counts = True
# Higher weight where the LED spectral density and camera counts are large;
# lower weight where the LED is weak and the calibration ratio is noisy.

calibration_spline_min_relative_weight = 0.05
# Keep a small nonzero weight in weak LED regions so the spline still follows
# broad trends but does not chase noise.

calibration_spline_max_relative_weight = 1.0

# Non-spline smoothing parameters. Wavelength windows are converted internally
# to an odd number of camera pixels using the calibrated wavelength spacing.
calibration_moving_average_window_nm = 10.0
calibration_gaussian_sigma_nm = 4.0
calibration_savgol_window_nm = 15.0
calibration_savgol_polyorder = 3
calibration_median_filter_window_nm = 8.0

# If True, non-spline methods first reject strong outliers relative to a
# preliminary median-filter trend before applying the final smoothing method.
calibration_non_spline_outlier_clip_sigma = 6.0
# Set None to disable.


# ============================================================
# MANUAL REPLACEMENT OF BAD / WEIRD FIBER CURVES
# ============================================================

replace_weird_fiber_curves_with_average = True
# True  = every fiber listed in weird_fiber_ids is NOT saved with its own
#         spline-smoothed calibration curve. Instead, it receives the average
#         smoothed calibration curve computed from all non-weird fibers.
# False = keep every fiber's own fitted curve.
#0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 13, 9, 38, 77, 79
weird_fiber_ids = [0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 13, 9, 38, 77, 79]
# Fiber IDs whose smoothed calibration/transmission curve looks wrong.
# Internal convention is Python fiber_id = 0..79 by default.
# Example:
#     weird_fiber_ids = [7, 13, 38]

weird_fiber_ids_are_one_based = False
# False = weird_fiber_ids uses Python fiber IDs 0..79.
# True  = weird_fiber_ids uses labels 1..80 and is converted to 0..79.

average_curve_statistic_for_weird_fibers = "mean"
# "mean" or "median". The replacement curve is computed wavelength-by-wavelength
# from all good fibers, excluding weird_fiber_ids.

output_weird_fiber_replacement_csv = CALIBRATION_FOLDER / "absolute_calibration_weird_fiber_replacements_ELI-MAY.csv"


# ============================================================
# OPTIONAL COMMON AVERAGE CURVE FOR ALL FIBERS
# ============================================================

use_common_average_curve_for_all_fibers = False
# False = save one calibration/transmission curve per fiber.
# True  = compute one common average curve and save that same curve for all
#         fibers. This is useful if the fiber-to-fiber differences look mostly
#         like calibration noise and you want only one global spectral response.

common_average_curve_statistic = "mean"
# "mean" or "median". The common curve is computed wavelength-by-wavelength.

common_average_exclude_weird_fibers = True
# True = if weird_fiber_ids is non-empty, those fibers are excluded from the
#        common average curve. The final common curve is still assigned to all
#        fibers, including the weird ones.

output_common_average_curve_csv = CALIBRATION_FOLDER / "absolute_calibration_common_average_curve_ELI-MAY.csv"
output_common_average_status_csv = CALIBRATION_FOLDER / "absolute_calibration_common_average_status_ELI-MAY.csv"


# ============================================================
# ALL-FIBER RAW + SMOOTHED CURVE DIAGNOSTIC PLOTS
# ============================================================

plot_all_fiber_raw_and_smoothed_curve_grids = True
# True = save/show four figures. Each figure contains 20 fibers arranged as
#        5 columns x 4 rows. Each subplot shows raw and spline-smoothed curves.

all_fiber_curve_grid_ncols = 5
all_fiber_curve_grid_nrows = 4
all_fiber_curve_figsize = (15.5, 10.5)
all_fiber_curve_yscale = "log"
# "linear" or "log"

all_fiber_curve_raw_alpha = 0.35
all_fiber_curve_raw_linewidth = 0.65
all_fiber_curve_smooth_linewidth = 1.2
all_fiber_curve_grid_alpha = 0.22
all_fiber_curve_show_replaced_label = True

output_all_fiber_curve_plot_template = "absolute_calibration_curves_fibers_{start:02d}_{end:02d}_ELI-MAY.png"


# ============================================================
# SCIENTIFIC-ARTICLE PLOT STYLE OPTIONS
# ============================================================

# The plotting style below is intentionally structured like the style block in
# CALIBRATION-ND-FABS-SRS.py: one place controls fonts, line widths, ticks,
# legends, grids, labels, colormaps, and save settings.
use_scientific_article_plot_style = True

PLOT_STYLE = {
    # General figure/export.
    "figsize_summary": (6.4, 4.2),
    "figsize_map": (6.6, 4.8),
    "figsize_power": (6.4, 4.0),
    "figsize_all_fiber_grid": all_fiber_curve_figsize,
    "figsize_sphere": (7.2, 6.4),
    "figsize_image_overlay": (10.5, 7.8),
    "dpi": 100,
    "save_dpi": 350,
    "transparent": False,
    "bbox_inches": "tight",

    # Fonts and text.
    "font_family": "DejaVu Sans",
    "font_size": 10,
    "title_size": 11,
    "label_size": 10,
    "tick_size": 8.5,
    "legend_size": 7.5,
    "colorbar_label_size": 9,
    "colorbar_tick_size": 8,
    "subplot_title_size": 8,
    "annotation_size": 6.5,
    "title_weight": "normal",
    "label_weight": "normal",
    "title_pad": 8,
    "label_pad": 4,

    # Axes, ticks, and grid.
    "show_grid": False,
    "grid_alpha": 0.18,
    "grid_linewidth": 0.55,
    "minor_grid_alpha": 0.10,
    "minor_grid_linewidth": 0.45,
    "spine_linewidth": 1.0,
    "tick_width": 0.95,
    "tick_length": 4.5,
    "minor_tick_length": 2.5,
    "tick_direction": "in",
    "ticks_top": True,
    "ticks_right": True,

    # Lines and markers.
    "curve_linewidth": 1.65,
    "raw_linewidth": 0.65,
    "smooth_linewidth": 1.35,
    "power_linewidth": 1.45,
    "curve_alpha": 0.95,
    "raw_alpha": 0.28,
    "marker_size": 3.8,
    "raw_linestyle": ":",
    "smooth_linestyle": "-",

    # Fiber-index colormap for all line plots.
    "fiber_line_colormap": "turbo",
    "fiber_line_color_min": 0,
    "fiber_line_color_max": expected_number_of_fibers - 1,
    "raw_curve_color": "0.55",
    "led_curve_color": "black",
    "led_curve_linestyle": "--",
    "map_colormap": "turbo",
    "power_colormap": "turbo",

    # Legends.
    "legend_frame": True,
    "legend_frame_alpha": 0.96,
    "legend_edge_color": "0.70",
    "legend_face_color": "white",
    "legend_linewidth": 0.8,
    "legend_location": "best",
    "legend_ncol": 1,

    # Colorbars.
    "colorbar_pad": 0.025,
    "colorbar_fraction": 0.045,
    "colorbar_aspect": 26,

    # Manual plot text.
    "summary_title": "Absolute spectral calibration",
    "map_title": "Absolute calibration map",
    "power_title": "Power entering each fiber",
    "sphere_title": "Power at the entrance of each fiber — linear interpolation between fibers",
    "sphere_counts_title": "Integrated raw IR-LED camera counts — linear interpolation between fibers",
    "image_overlay_title": "IR-LED calibration image: fiber traces and integration regions",
    "all_fiber_grid_title": "Absolute calibration curves",
    "x_label_wavelength": "Wavelength (nm)",
    "x_label_fiber_id": "Fiber ID",
    "y_label_calibration": r"Calibration (J nm$^{-1}$ count$^{-1}$)",
    "y_label_led": "Normalized LED spectrum after longpass",
    "y_label_power_w": "Power at fiber entrance (W)",
    "y_label_power_nw": "Power at fiber entrance (nW)",
    "y_label_power_after_lp_nw": "Power entering fiber after longpass (nW)",
    "colorbar_calibration": r"J nm$^{-1}$ count$^{-1}$",
    "colorbar_power_w": "Power at fiber entrance (W)",
    "colorbar_camera_counts": "Integrated camera counts",
    "sphere_x_label": "x (cm)",
    "sphere_y_label": "y (cm)",
    "sphere_z_label": "z (cm)",
    "x_label_pixel": "x pixel",
    "y_label_pixel": "y pixel",
}

# Backward-compatible all-fiber plot size now follows the article-style dict.
all_fiber_curve_figsize = PLOT_STYLE["figsize_all_fiber_grid"]

# Optional scientific-style 3D plot of the calibration powers on the measured quarter sphere.
plot_fiber_power_sphere_3d = True
# True = add a 3D map using the same angle convention as the main plotting code:
#   theta = 0 deg is the laser axis,
#   theta = 90 deg and phi = 0 deg is the right side of the sphere,
#   theta = 90 deg and phi = 90 deg is the top of the sphere.
#
# IMPORTANT:
#   This plot intentionally shows only the measured quarter sphere containing
#   the fibers: 0 <= phi <= 90 deg and 0 <= theta <= 90 deg.
#   It does not mirror/reconstruct the other quarters of the forward half-sphere.

fiber_power_sphere_config_name = "28052026"
fiber_power_sphere_radius_cm = sphere_radius_cm
fiber_power_sphere_interpolation_mode = "fiber_triangulation"
# "fiber_triangulation" = default. Do NOT generate a regular quarter-sphere
# grid. The plot uses only the measured fiber-port positions and linearly
# interpolates inside the Delaunay triangles connecting those points, as in the
# main 3D energy-map diagnostic.
#
# "regular_grid" = optional old/debug behavior: interpolate onto a regular
# phi-theta domain. Keep this disabled unless you explicitly want a filled
# quarter/half-sphere patch.

fiber_power_sphere_phi_grid_points = 121
fiber_power_sphere_theta_grid_points = 81
fiber_power_sphere_interpolation_method = "linear"
# Used only if fiber_power_sphere_interpolation_mode = "regular_grid".

fiber_power_sphere_plot_reconstructed_half_sphere = False
# Used only if fiber_power_sphere_interpolation_mode = "regular_grid".
# False = measured quarter domain. True = optional mirrored half-sphere.

fiber_power_sphere_phi_min_deg = 0.0
fiber_power_sphere_phi_max_deg = 90.0
fiber_power_sphere_theta_min_deg = 0.0
fiber_power_sphere_theta_max_deg = 90.0
# Used only if fiber_power_sphere_interpolation_mode = "regular_grid".

fiber_power_sphere_symmetry_mode = "mirror_phi0_phi90"
# Used only for the optional regular-grid reconstructed half-sphere mode.
# "mirror_phi0_phi90" respects phi=90 deg as a vertical symmetry plane.

fiber_power_sphere_triangulation_subdivisions = 2
# 0 = triangles directly between fibers only.
# 1, 2, 3 = subdivide those same triangles to make color interpolation smoother.
# This still does NOT create a quarter sphere: all points remain inside the
# convex hull of the measured fiber ports.

fiber_power_sphere_mask_large_triangles = False
fiber_power_sphere_max_triangle_edge_deg = 25.0
# Optional diagnostic cleanup for the triangulated plot. If True, triangles with
# any edge larger than this distance in (phi, theta) space are hidden.

fiber_power_sphere_power_column = "power_entering_fiber_total_W"
# Options usually available in power_df:
#   "power_entering_fiber_total_W"           = power at the fiber entrance [W]
#   "power_entering_fiber_after_longpass_W"  = power after the ideal/measured LP passband [W]
#   "power_true_W_on_powermeter_aperture"    = corrected S120C aperture power [W]

fiber_power_sphere_use_log_color = False
fiber_power_sphere_color_percentiles = [2, 98]
fiber_power_sphere_colormap = PLOT_STYLE["power_colormap"]
fiber_power_sphere_surface_alpha = 0.8
fiber_power_sphere_surface_edgecolor = "none"
fiber_power_sphere_surface_linewidth = 0.0
fiber_power_sphere_surface_antialiased = False
fiber_power_sphere_show_ports = True
fiber_power_sphere_port_marker_size = 28
fiber_power_sphere_show_port_labels = True
fiber_power_sphere_port_label_size = 5.5
fiber_power_sphere_draw_laser_axis = True
fiber_power_sphere_laser_axis_color = "black"
fiber_power_sphere_laser_axis_linewidth = 1.2
fiber_power_sphere_laser_axis_label = r"laser axis, $\theta=0^\circ$"
fiber_power_sphere_view_elev = 22
fiber_power_sphere_view_azim = -42
fiber_power_sphere_axis_grid = False
fiber_power_sphere_axis_panes = False
fiber_power_sphere_add_axis_point = True
fiber_power_sphere_axis_point_theta_window_deg = 8.0

output_fiber_power_sphere_png = CALIBRATION_FOLDER / "absolute_calibration_fiber_power_triangulated_ports_ELI-MAY.png"
output_fiber_power_vs_fiber_png = CALIBRATION_FOLDER / "absolute_calibration_power_vs_fiber_paper_style_ELI-MAY.png"

# Optional second 3D plot: same interpolation/geometry as the power plot, but
# colored by the raw uncalibrated camera counts integrated over the IR-LED
# calibration spectrum for each fiber.
plot_fiber_counts_sphere_3d = True

# "raw_signal_image" integrates the calibration image before background subtraction.
# "background_subtracted_image" uses the background-subtracted image that is used
# internally to build the absolute calibration.
fiber_counts_sphere_counts_source = "raw_signal_image"
# Options: "raw_signal_image", "background_subtracted_image"

fiber_counts_sphere_use_log_color = False
fiber_counts_sphere_color_percentiles = [2, 98]
fiber_counts_sphere_colormap = PLOT_STYLE["map_colormap"]
fiber_counts_sphere_surface_alpha = fiber_power_sphere_surface_alpha
fiber_counts_sphere_port_marker_size = fiber_power_sphere_port_marker_size
fiber_counts_sphere_show_ports = fiber_power_sphere_show_ports
fiber_counts_sphere_show_port_labels = fiber_power_sphere_show_port_labels
fiber_counts_sphere_port_label_size = fiber_power_sphere_port_label_size

output_fiber_counts_sphere_png = CALIBRATION_FOLDER / "absolute_calibration_fiber_raw_counts_triangulated_ports_ELI-MAY.png"


# Optional 2D diagnostic plot: raw image with fiber traces and integration
# regions overlaid. This is useful to verify that the reference lines and
# non-overlapping integration bands match the actual IR-LED calibration image.
plot_calibration_image_with_fiber_overlay = True

# Image used as background for the overlay.
#   "raw_signal_image"             = original IR-LED image before background subtraction.
#   "background_subtracted_image"  = processed image used to build the calibration.
calibration_image_overlay_source = "raw_signal_image"
# Options: "raw_signal_image", "background_subtracted_image"

# Display scaling for the image. Percentile clipping makes the faint traces
# visible without being dominated by hot pixels or saturated regions.
calibration_image_overlay_use_log_image = False
calibration_image_overlay_percentiles = [1, 99.7]
calibration_image_overlay_cmap = "gray"

# Overlay appearance.
calibration_image_overlay_x_step_px = 8
calibration_image_overlay_show_center_traces = True
calibration_image_overlay_show_integration_boundaries = True
calibration_image_overlay_show_integration_fill = True
calibration_image_overlay_center_linewidth = 0.65
calibration_image_overlay_boundary_linewidth = 0.45
calibration_image_overlay_boundary_linestyle = "--"
calibration_image_overlay_fill_alpha = 0.11
calibration_image_overlay_center_alpha = 0.95
calibration_image_overlay_boundary_alpha = 0.85

# Fiber labels on the image. Set interval to None or <=0 to disable labels.
calibration_image_overlay_label_every_n_fibers = 5
calibration_image_overlay_label_x_fraction = 0.985
calibration_image_overlay_label_fontsize = 5.5
calibration_image_overlay_label_color = "white"

output_calibration_image_overlay_png = CALIBRATION_FOLDER / "absolute_calibration_image_fiber_traces_integration_regions_ELI-MAY.png"



# ============================================================
# BASIC UTILITIES
# ============================================================

def resolve_existing_path(path: Path) -> Path:
    path = Path(path)
    if path.exists():
        return path

    # Try common TIFF variants for images.
    if path.suffix.lower() in {".tif", ".tiff", ""}:
        for suffix in [".tif", ".tiff", ".TIF", ".TIFF"]:
            candidate = path.with_suffix(suffix)
            if candidate.exists():
                return candidate

    raise FileNotFoundError(f"File not found:\n  {path}")


def first_existing_path(candidates: list[Path]) -> Path:
    for candidate in candidates:
        if Path(candidate).exists():
            return Path(candidate)
    return Path(candidates[0])


def infer_exposure_seconds_from_filename(path: Path) -> float:
    name = Path(path).name
    patterns = [
        r"(?P<t>\d+(?:\.\d+)?)\s*s\s*acq",
        r"(?P<t>\d+(?:\.\d+)?)\s*s",
    ]
    for pattern in patterns:
        match = re.search(pattern, name, flags=re.IGNORECASE)
        if match:
            return float(match.group("t"))
    raise RuntimeError(
        "Could not infer exposure time from filename. Set "
        "calibration_exposure_time_s manually."
    )


def load_image_as_float(path: Path) -> np.ndarray:
    path = resolve_existing_path(path)
    img = np.array(Image.open(path)).astype(np.float32)
    if img.ndim == 3:
        img = img.mean(axis=2)
    return img


def subtract_background(signal_img: np.ndarray, background_img: np.ndarray) -> np.ndarray:
    signal_img = np.asarray(signal_img, dtype=np.float32)
    background_img = np.asarray(background_img, dtype=np.float32)

    if signal_img.shape != background_img.shape:
        raise ValueError(
            "Signal and background images do not have the same shape:\n"
            f"  signal     : {signal_img.shape}\n"
            f"  background : {background_img.shape}"
        )

    corrected = signal_img - float(background_scale_factor) * background_img
    if clip_negative_after_background_subtraction:
        corrected = np.where(np.isfinite(corrected), corrected, 0.0)
        corrected[corrected < 0] = 0.0
    return corrected.astype(np.float32)


def read_two_column_numeric_file(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Read a simple two-column numeric CSV/TXT file."""
    path = resolve_existing_path(path)

    # Try pandas first. This accepts headers such as "x, y".
    try:
        df = pd.read_csv(path)
        numeric_cols = []
        for col in df.columns:
            vals = pd.to_numeric(df[col], errors="coerce")
            if np.sum(np.isfinite(vals)) > 5:
                numeric_cols.append(vals.to_numpy(float))
        if len(numeric_cols) >= 2:
            x = numeric_cols[0]
            y = numeric_cols[1]
            valid = np.isfinite(x) & np.isfinite(y)
            return x[valid], y[valid]
    except Exception:
        pass

    # Fallback: extract numeric pairs line by line.
    rows = []
    for line in path.read_text(errors="ignore").splitlines():
        nums = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", line)
        if len(nums) >= 2:
            rows.append((float(nums[0]), float(nums[1])))
    if len(rows) < 2:
        raise RuntimeError(f"Could not read two numeric columns from {path}")
    arr = np.asarray(rows, dtype=float)
    return arr[:, 0], arr[:, 1]


def read_spf2_spectrum(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """
    Minimal reader for the uploaded Thorlabs/CCS .spf2 spectrum.

    The file contains a float32 wavelength axis followed by a float32 signal
    array of the same length. This function finds the longest monotonically
    increasing float32 block in the physical wavelength range and takes the next
    equal-length block as the spectrum.
    """
    path = resolve_existing_path(path)
    raw = path.read_bytes()
    arr = np.frombuffer(raw, dtype="<f4")

    good = np.isfinite(arr) & (arr > 100.0) & (arr < 2000.0)
    best_len = 0
    best_i0 = None
    best_i1 = None

    i = 0
    n = len(arr)
    while i < n:
        if not good[i]:
            i += 1
            continue
        j = i + 1
        while (
            j < n
            and good[j]
            and arr[j] > arr[j - 1]
            and 0.0 < arr[j] - arr[j - 1] < 5.0
        ):
            j += 1
        if j - i > best_len:
            best_len = j - i
            best_i0 = i
            best_i1 = j
        i = max(j, i + 1)

    if best_i0 is None or best_len < 100:
        raise RuntimeError(f"Could not find a wavelength axis in SPF2 file: {path}")

    wavelength_nm = arr[best_i0:best_i1].astype(float)
    signal_start = best_i1
    signal_stop = signal_start + best_len
    if signal_stop > len(arr):
        raise RuntimeError("SPF2 file ended before the expected spectrum array.")

    signal = arr[signal_start:signal_stop].astype(float)

    valid = np.isfinite(wavelength_nm) & np.isfinite(signal)
    wavelength_nm = wavelength_nm[valid]
    signal = signal[valid]

    order = np.argsort(wavelength_nm)
    wavelength_nm = wavelength_nm[order]
    signal = signal[order]

    # Remove small negative spectrometer noise before using it as a spectral density.
    signal = signal - np.nanpercentile(signal, 1.0)
    signal[signal < 0] = 0.0

    return wavelength_nm, signal


def load_true_led_spectrum(path: Path) -> tuple[np.ndarray, np.ndarray]:
    path = Path(path)
    if path.suffix.lower() == ".spf2":
        return read_spf2_spectrum(path)
    return read_two_column_numeric_file(path)


def parse_power_meter_readings(path: Path) -> pd.DataFrame:
    path = resolve_existing_path(path)
    text = path.read_text(errors="ignore")
    rows = []
    for line in text.splitlines():
        match = re.search(
            r"^\s*(?P<fiber>\d+)\s*[/,;\t ]+\s*(?P<power>[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)",
            line,
        )
        if match:
            fiber_label = int(match.group("fiber"))
            power_value = float(match.group("power"))
            fiber_id = fiber_label - 1 if power_file_fiber_ids_are_one_based else fiber_label
            rows.append(
                {
                    "fiber_label_in_file": fiber_label,
                    "fiber_id": fiber_id,
                    "power_reading_uW": power_value,
                }
            )

    if len(rows) == 0:
        raise RuntimeError(f"Could not parse any fiber/power rows from {path}")

    df = pd.DataFrame(rows).sort_values("fiber_id").reset_index(drop=True)
    if len(df) != expected_number_of_fibers:
        print(
            f"WARNING: parsed {len(df)} power readings, expected "
            f"{expected_number_of_fibers}."
        )
    return df


# ============================================================
# IMAGE INTEGRATION
# ============================================================

def reference_y_lines(ref_df: pd.DataFrame, x_sample: np.ndarray) -> np.ndarray:
    ref_df = ref_df.sort_values("fiber_id").reset_index(drop=True)
    y_centers = np.zeros((len(ref_df), len(x_sample)), dtype=float)
    for i, row in ref_df.iterrows():
        y_centers[i, :] = row["slope_px_per_px"] * x_sample + row["intercept_px"]
    return y_centers


def compute_integration_boundaries(
    ref_df: pd.DataFrame,
    x_sample: np.ndarray,
    ny: int,
    integration_width: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not (0 < integration_width <= 1):
        raise ValueError("integration_width must be > 0 and <= 1.")

    y_centers = reference_y_lines(ref_df, x_sample)
    internal_boundaries = 0.5 * (y_centers[:-1, :] + y_centers[1:, :])

    y_top_full = np.zeros_like(y_centers)
    y_bottom_full = np.zeros_like(y_centers)

    y_top_full[0, :] = y_centers[0, :] - 0.5 * (y_centers[1, :] - y_centers[0, :])
    y_bottom_full[-1, :] = y_centers[-1, :] + 0.5 * (y_centers[-1, :] - y_centers[-2, :])

    for i in range(len(ref_df)):
        if i > 0:
            y_top_full[i, :] = internal_boundaries[i - 1, :]
        if i < len(ref_df) - 1:
            y_bottom_full[i, :] = internal_boundaries[i, :]

    y_top = y_centers + integration_width * (y_top_full - y_centers)
    y_bottom = y_centers + integration_width * (y_bottom_full - y_centers)

    y_min = np.minimum(y_top, y_bottom)
    y_max = np.maximum(y_top, y_bottom)

    y_top = np.clip(y_min, 0, ny - 1)
    y_bottom = np.clip(y_max, 0, ny - 1)

    return y_centers, y_top, y_bottom


def fractional_column_integral(img_column: np.ndarray, y_min_float: float, y_max_float: float) -> float:
    img_column = np.asarray(img_column, dtype=float)
    ny = img_column.size

    y0 = float(min(y_min_float, y_max_float))
    y1 = float(max(y_min_float, y_max_float))

    y0 = max(-0.5, y0)
    y1 = min(ny - 0.5, y1)

    if not np.isfinite(y0) or not np.isfinite(y1) or y1 <= y0:
        return np.nan

    k0 = int(np.floor(y0 + 0.5))
    k1 = int(np.floor(y1 + 0.5))
    k0 = max(0, k0)
    k1 = min(ny - 1, k1)

    value = 0.0
    total_weight = 0.0
    for k in range(k0, k1 + 1):
        pixel_low = k - 0.5
        pixel_high = k + 0.5
        overlap = min(y1, pixel_high) - max(y0, pixel_low)
        if overlap <= 0:
            continue
        pix = float(img_column[k])
        if not np.isfinite(pix):
            continue
        value += overlap * pix
        total_weight += overlap

    if total_weight <= 0:
        return np.nan

    return value


def integrate_image_in_pixel_space(
    img: np.ndarray,
    ref_df: pd.DataFrame,
    integration_width: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    ny, nx = img.shape
    x_sample = np.arange(0, nx, integrated_x_step_px, dtype=int)

    _, y_top, y_bottom = compute_integration_boundaries(
        ref_df=ref_df,
        x_sample=x_sample.astype(float),
        ny=ny,
        integration_width=integration_width,
    )

    integrated = np.full((len(ref_df), len(x_sample)), np.nan, dtype=float)
    for i in range(len(ref_df)):
        for j, x in enumerate(x_sample):
            integrated[i, j] = fractional_column_integral(
                img_column=img[:, int(x)],
                y_min_float=y_top[i, j],
                y_max_float=y_bottom[i, j],
            )

    return integrated, x_sample, y_top, y_bottom


# ============================================================
# WAVELENGTH / SPECTRAL MODELS
# ============================================================

def find_first_existing_wavelength_file() -> Path:
    candidates = [
        wavelength_calibration_axis_csv,
        CALIBRATION_FOLDER / "hg_wavelength_calibration_axis_*ELI-MAY*.csv",
        CALIBRATION_FOLDER / "hg_wavelength_calibration_axis_*260528*.csv",
    ]

    if wavelength_calibration_axis_csv.exists():
        return wavelength_calibration_axis_csv

    for pattern in [
        "hg_wavelength_calibration_axis_ELI-MAY*.csv",
        "hg_wavelength_calibration_axis_*260528*.csv",
    ]:
        matches = sorted(CALIBRATION_FOLDER.glob(pattern))
        if matches:
            return matches[0]

    if wavelength_calibration_axis_npz.exists():
        return wavelength_calibration_axis_npz

    for pattern in [
        "hg_wavelength_calibration_ELI-MAY*.npz",
        "hg_wavelength_calibration_*260528*.npz",
    ]:
        matches = sorted(CALIBRATION_FOLDER.glob(pattern))
        if matches:
            return matches[0]

    raise FileNotFoundError(
        "Could not find May wavelength calibration axis in Calibration/. "
        "Run HG-FOAM-CALIBRATION.py first."
    )


def load_wavelength_axis_for_x_sample(x_sample: np.ndarray) -> tuple[np.ndarray, Path]:
    path = find_first_existing_wavelength_file()
    x_sample = np.asarray(x_sample, dtype=float)

    if path.suffix.lower() == ".csv":
        df = pd.read_csv(path)
        lower_cols = {str(c).strip().lower(): c for c in df.columns}
        wavelength_col = None
        for name in [
            "wavelength_hg_calibrated_nm",
            "wavelengths_hg_calibrated_nm",
            "calibrated_wavelength_nm",
            "wavelength_nm",
            "wavelength",
        ]:
            if name in lower_cols:
                wavelength_col = lower_cols[name]
                break
        if wavelength_col is None:
            raise KeyError(f"Could not find wavelength column in {path}")

        x_col = None
        for name in ["x_pixel", "pixel", "x", "pixel_index"]:
            if name in lower_cols:
                x_col = lower_cols[name]
                break

        cal_wl = pd.to_numeric(df[wavelength_col], errors="coerce").to_numpy(float)
        if x_col is None:
            cal_x = np.arange(len(cal_wl), dtype=float)
        else:
            cal_x = pd.to_numeric(df[x_col], errors="coerce").to_numpy(float)

    elif path.suffix.lower() == ".npz":
        npz = np.load(path)
        wl_key = None
        for key in [
            "wavelengths_hg_calibrated_nm",
            "wavelength_hg_calibrated_nm",
            "wavelength_nm",
            "wavelengths_nm",
        ]:
            if key in npz:
                wl_key = key
                break
        if wl_key is None:
            raise KeyError(f"Could not find wavelength array in {path}")
        cal_wl = np.asarray(npz[wl_key], dtype=float)
        if "x_sample" in npz:
            cal_x = np.asarray(npz["x_sample"], dtype=float)
        elif "x_pixel" in npz:
            cal_x = np.asarray(npz["x_pixel"], dtype=float)
        else:
            cal_x = np.arange(len(cal_wl), dtype=float)
    else:
        raise ValueError(f"Unsupported wavelength calibration file: {path}")

    valid = np.isfinite(cal_x) & np.isfinite(cal_wl)
    if np.sum(valid) < 2:
        raise RuntimeError(f"Not enough finite wavelength calibration points in {path}")

    order = np.argsort(cal_x[valid])
    cal_x = cal_x[valid][order]
    cal_wl = cal_wl[valid][order]

    wavelengths_nm = np.interp(x_sample, cal_x, cal_wl, left=cal_wl[0], right=cal_wl[-1])
    return wavelengths_nm, path


def wavelength_bin_widths(wavelengths_nm: np.ndarray) -> np.ndarray:
    wavelengths_nm = np.asarray(wavelengths_nm, dtype=float)
    edges = np.empty(len(wavelengths_nm) + 1, dtype=float)
    edges[1:-1] = 0.5 * (wavelengths_nm[:-1] + wavelengths_nm[1:])
    edges[0] = wavelengths_nm[0] - 0.5 * (wavelengths_nm[1] - wavelengths_nm[0])
    edges[-1] = wavelengths_nm[-1] + 0.5 * (wavelengths_nm[-1] - wavelengths_nm[-2])
    return np.diff(edges)


def normalize_spectral_density(wavelength_nm: np.ndarray, signal: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    wavelength_nm = np.asarray(wavelength_nm, dtype=float)
    signal = np.asarray(signal, dtype=float)
    valid = np.isfinite(wavelength_nm) & np.isfinite(signal) & (signal >= 0)
    wavelength_nm = wavelength_nm[valid]
    signal = signal[valid]
    order = np.argsort(wavelength_nm)
    wavelength_nm = wavelength_nm[order]
    signal = signal[order]
    area = np.trapezoid(signal, wavelength_nm)
    if not np.isfinite(area) or area <= 0:
        raise RuntimeError("Cannot normalize spectral density: integral <= 0.")
    return wavelength_nm, signal / area


def interpolate_density_to_axis(
    source_wavelength_nm: np.ndarray,
    source_density_per_nm: np.ndarray,
    target_wavelength_nm: np.ndarray,
) -> np.ndarray:
    interp = interp1d(
        source_wavelength_nm,
        source_density_per_nm,
        kind="linear",
        bounds_error=False,
        fill_value=0.0,
    )
    density = interp(target_wavelength_nm)
    density[~np.isfinite(density)] = 0.0
    density[density < 0] = 0.0
    return density


def load_longpass_transmission(target_wavelength_nm: np.ndarray) -> np.ndarray:
    target_wavelength_nm = np.asarray(target_wavelength_nm, dtype=float)
    if longpass_filter_curve_csv is None:
        return (target_wavelength_nm >= float(longpass_cutoff_nm)).astype(float)

    wl, tr = read_two_column_numeric_file(Path(longpass_filter_curve_csv))
    tr = np.asarray(tr, dtype=float)
    if np.nanmax(tr) > 1.5:
        tr = tr / 100.0
    tr = np.clip(tr, 0.0, 1.0)
    order = np.argsort(wl)
    return np.interp(target_wavelength_nm, wl[order], tr[order], left=0.0, right=tr[order][-1])


def physical_energy_wavelength_mask(wavelength_nm: np.ndarray) -> np.ndarray:
    """
    Mask of wavelengths where calibrated energy is physically allowed.

    Below zero_energy_below_wavelength_nm, the saved calibration coefficients
    are forced to exactly zero. Therefore any measured camera counts there give
    spectral_energy_J_per_nm = 0.
    """
    wavelength_nm = np.asarray(wavelength_nm, dtype=float)
    if not force_zero_energy_below_wavelength:
        return np.ones_like(wavelength_nm, dtype=bool)
    return np.isfinite(wavelength_nm) & (wavelength_nm >= float(zero_energy_below_wavelength_nm))


def enforce_zero_energy_cutoff_on_spectral_arrays(
    wavelength_nm: np.ndarray,
    arrays_2d: list[np.ndarray] | None = None,
    arrays_1d: list[np.ndarray] | None = None,
) -> dict[str, Any]:
    """
    Force spectral calibration/power arrays to zero below the physical cutoff.

    This is applied after smoothing/replacement/common-curve operations so no
    extrapolation or averaging can reintroduce nonzero calibration values below
    the cutoff.
    """
    wavelength_nm = np.asarray(wavelength_nm, dtype=float)
    allowed = physical_energy_wavelength_mask(wavelength_nm)
    blocked = ~allowed

    if arrays_2d is not None:
        for arr in arrays_2d:
            if arr is None:
                continue
            arr = np.asarray(arr)
            if arr.ndim != 2 or arr.shape[1] != wavelength_nm.size:
                raise ValueError(
                    "2D spectral array has incompatible shape for cutoff: "
                    f"array shape={arr.shape}, wavelength size={wavelength_nm.size}"
                )
            arr[:, blocked] = 0.0

    if arrays_1d is not None:
        for arr in arrays_1d:
            if arr is None:
                continue
            arr = np.asarray(arr)
            if arr.ndim != 1 or arr.shape[0] != wavelength_nm.size:
                raise ValueError(
                    "1D spectral array has incompatible shape for cutoff: "
                    f"array shape={arr.shape}, wavelength size={wavelength_nm.size}"
                )
            arr[blocked] = 0.0

    return {
        "physical_energy_allowed_wavelength_mask": allowed,
        "zero_energy_below_wavelength_nm": float(zero_energy_below_wavelength_nm),
        "force_zero_energy_below_wavelength": bool(force_zero_energy_below_wavelength),
        "n_zeroed_wavelength_bins": int(np.sum(blocked)),
    }


def apply_optional_fiber_attenuation(wavelength_nm: np.ndarray, density_per_nm: np.ndarray) -> np.ndarray:
    if not apply_fiber_attenuation_curve_to_led_spectrum:
        return density_per_nm.copy()

    wl_att, att_db_per_km = read_two_column_numeric_file(fiber_attenuation_csv)
    order = np.argsort(wl_att)
    wl_att = wl_att[order]
    att_db_per_km = att_db_per_km[order]
    att_interp = np.interp(wavelength_nm, wl_att, att_db_per_km, left=att_db_per_km[0], right=att_db_per_km[-1])
    transmission = 10.0 ** (-(att_interp * float(fiber_length_m) / 1000.0) / 10.0)
    out = density_per_nm * transmission
    area = np.trapezoid(out, wavelength_nm)
    if area > 0:
        out = out / area
    return out


# ============================================================
# ABSOLUTE POWER MODEL
# ============================================================

def compute_powermeter_spectral_correction(
    led_wavelength_nm: np.ndarray,
    led_density_per_nm: np.ndarray,
) -> dict:
    resp_wl, resp_aw = read_two_column_numeric_file(powermeter_responsivity_csv)
    order = np.argsort(resp_wl)
    resp_wl = resp_wl[order]
    resp_aw = resp_aw[order]

    # Use only wavelengths covered by the responsivity curve.
    wl_min = max(np.nanmin(led_wavelength_nm), np.nanmin(resp_wl))
    wl_max = min(np.nanmax(led_wavelength_nm), np.nanmax(resp_wl))
    grid = np.linspace(wl_min, wl_max, 5000)

    led_density = interpolate_density_to_axis(led_wavelength_nm, led_density_per_nm, grid)
    density_area = np.trapezoid(led_density, grid)
    if density_area <= 0:
        raise RuntimeError("LED spectrum has no overlap with the powermeter sensitivity curve.")
    led_density = led_density / density_area

    resp = np.interp(grid, resp_wl, resp_aw)
    r_eff = float(np.trapezoid(led_density * resp, grid))
    r_set = float(np.interp(powermeter_selected_wavelength_nm, resp_wl, resp_aw))
    correction = r_set / r_eff

    return {
        "responsivity_at_selected_wavelength_A_per_W": r_set,
        "spectrum_weighted_responsivity_A_per_W": r_eff,
        "powermeter_true_power_correction_factor": correction,
        "responsivity_curve_min_nm": float(np.nanmin(resp_wl)),
        "responsivity_curve_max_nm": float(np.nanmax(resp_wl)),
        "overlap_min_nm": float(wl_min),
        "overlap_max_nm": float(wl_max),
    }


def compute_power_entering_fibers(power_df: pd.DataFrame, spectral_correction_factor: float) -> pd.DataFrame:
    out = power_df.copy()

    if str(power_file_units).lower() in {"uw", "µw", "micro-w", "microwatt", "microwatts"}:
        scale_to_w = 1e-6
    elif str(power_file_units).lower() in {"mw", "milliwatt", "milliwatts"}:
        scale_to_w = 1e-3
    elif str(power_file_units).lower() in {"w", "watt", "watts"}:
        scale_to_w = 1.0
    else:
        raise ValueError(f"Unsupported power_file_units={power_file_units!r}")

    out["power_reading_W_at_setting"] = out["power_reading_uW"].to_numpy(float) * scale_to_w
    out["power_true_W_on_powermeter_aperture"] = (
        out["power_reading_W_at_setting"] * float(spectral_correction_factor)
    )

    r_fiber_cm = float(sphere_radius_cm)
    r_pm_cm = r_fiber_cm - float(powermeter_sensor_closer_to_led_by_cm)
    if r_pm_cm <= 0:
        raise ValueError("powermeter_sensor_closer_to_led_by_cm is too large.")
    distance_correction = (r_pm_cm / r_fiber_cm) ** 2

    area_pm_cm2 = np.pi * (0.1 * float(powermeter_input_aperture_diameter_mm) / 2.0) ** 2
    area_core_cm2 = np.pi * ((float(fiber_core_diameter_um) * 1e-4) / 2.0) ** 2

    if apply_na_solid_angle_correction:
        na_factor = float(fiber_numerical_aperture) ** 2
    else:
        na_factor = 1.0

    out["distance_correction_factor_to_fiber_plane"] = distance_correction
    out["powermeter_aperture_area_cm2"] = area_pm_cm2
    out["fiber_core_area_cm2"] = area_core_cm2
    out["fiber_area_over_powermeter_area"] = area_core_cm2 / area_pm_cm2
    out["na_acceptance_factor_applied"] = na_factor

    out["irradiance_true_W_cm2_at_fiber_plane"] = (
        out["power_true_W_on_powermeter_aperture"] / area_pm_cm2 * distance_correction
    )
    out["power_entering_fiber_total_W"] = (
        out["irradiance_true_W_cm2_at_fiber_plane"] * area_core_cm2 * na_factor
    )

    return out


# ============================================================
# CALIBRATION BUILD
# ============================================================


def robust_sigma_mad(y):
    """Robust scatter estimate using MAD, ignoring non-finite values."""
    y = np.asarray(y, dtype=float)
    y = y[np.isfinite(y)]
    if y.size == 0:
        return np.nan
    med = float(np.nanmedian(y))
    mad = float(np.nanmedian(np.abs(y - med)))
    if not np.isfinite(mad) or mad <= 0:
        sig = float(np.nanstd(y))
    else:
        sig = 1.4826 * mad
    return sig


def _smoothing_method_name() -> str:
    """Return normalized smoothing method name."""
    method = str(calibration_smoothing_method).lower().strip().replace("-", "_")
    aliases = {
        "off": "none",
        "raw": "none",
        "no": "none",
        "moving": "moving_average",
        "boxcar": "moving_average",
        "gauss": "gaussian",
        "savitzky_golay": "savgol",
        "sgolay": "savgol",
        "median": "median_filter",
    }
    method = aliases.get(method, method)
    allowed = {"none", "spline", "moving_average", "gaussian", "savgol", "median_filter"}
    if method not in allowed:
        raise ValueError(
            "calibration_smoothing_method must be one of: "
            "'none', 'spline', 'moving_average', 'gaussian', 'savgol', 'median_filter'."
        )
    if not smooth_saved_absolute_calibration:
        return "none"
    return method


def _odd_window_points_from_nm(wavelength_nm, window_nm, minimum=3) -> int:
    """Convert a wavelength window [nm] into an odd number of samples."""
    wavelength_nm = np.asarray(wavelength_nm, dtype=float)
    finite = np.isfinite(wavelength_nm)
    if np.sum(finite) < 2:
        return int(max(3, minimum)) | 1
    dw = np.nanmedian(np.abs(np.diff(wavelength_nm[finite])))
    if not np.isfinite(dw) or dw <= 0:
        dw = 1.0
    n = int(np.ceil(float(window_nm) / dw))
    n = max(int(minimum), n)
    if n % 2 == 0:
        n += 1
    return n


def _fill_nonfinite_by_interpolation(y):
    """Fill non-finite samples by linear interpolation, then edge values."""
    y = np.asarray(y, dtype=float)
    out = y.copy()
    idx = np.arange(len(out), dtype=float)
    finite = np.isfinite(out)
    if np.sum(finite) == 0:
        return np.zeros_like(out, dtype=float)
    if np.sum(finite) == 1:
        out[~finite] = out[finite][0]
        return out
    out[~finite] = np.interp(idx[~finite], idx[finite], out[finite])
    return out


def _nan_moving_average(y, window_points: int) -> np.ndarray:
    """NaN-aware moving average."""
    y = np.asarray(y, dtype=float)
    window_points = int(window_points)
    if window_points <= 1:
        return y.copy()
    if window_points % 2 == 0:
        window_points += 1
    kernel = np.ones(window_points, dtype=float)
    valid = np.isfinite(y).astype(float)
    y0 = np.where(np.isfinite(y), y, 0.0)
    num = np.convolve(y0, kernel, mode="same")
    den = np.convolve(valid, kernel, mode="same")
    out = np.full_like(y, np.nan, dtype=float)
    good = den > 0
    out[good] = num[good] / den[good]
    return out


def _preclip_outliers_for_non_spline(x, y):
    """Remove isolated strong outliers before simple filters."""
    if calibration_non_spline_outlier_clip_sigma is None:
        return x, y
    if len(y) < max(7, int(calibration_smoothing_min_points)):
        return x, y

    window = _odd_window_points_from_nm(
        x,
        calibration_median_filter_window_nm,
        minimum=7,
    )
    window = min(window, len(y) if len(y) % 2 == 1 else len(y) - 1)
    if window < 3:
        return x, y

    y_filled = _fill_nonfinite_by_interpolation(y)
    trend = median_filter(y_filled, size=window, mode="nearest")
    residual = y - trend
    sig = robust_sigma_mad(residual)
    if not np.isfinite(sig) or sig <= 0:
        return x, y
    keep = np.abs(residual) <= float(calibration_non_spline_outlier_clip_sigma) * sig
    if np.sum(keep) < max(5, int(calibration_smoothing_min_points)):
        return x, y
    return x[keep], y[keep]


def _smooth_working_curve_non_spline(x, y, method):
    """Apply non-spline smoothing to already transformed y values."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    y = _fill_nonfinite_by_interpolation(y)

    if method == "moving_average":
        window = _odd_window_points_from_nm(x, calibration_moving_average_window_nm, minimum=3)
        return _nan_moving_average(y, window)

    if method == "gaussian":
        finite = np.isfinite(x)
        if np.sum(finite) >= 2:
            dx = np.nanmedian(np.abs(np.diff(x[finite])))
        else:
            dx = 1.0
        if not np.isfinite(dx) or dx <= 0:
            dx = 1.0
        sigma_points = max(float(calibration_gaussian_sigma_nm) / dx, 0.01)
        return gaussian_filter1d(y, sigma=sigma_points, mode="nearest")

    if method == "savgol":
        window = _odd_window_points_from_nm(
            x,
            calibration_savgol_window_nm,
            minimum=int(calibration_savgol_polyorder) + 3,
        )
        max_window = len(y) if len(y) % 2 == 1 else len(y) - 1
        window = min(window, max_window)
        poly = int(calibration_savgol_polyorder)
        poly = max(0, min(poly, window - 2))
        if window < poly + 2 or window < 3:
            return y.copy()
        return savgol_filter(y, window_length=window, polyorder=poly, mode="interp")

    if method == "median_filter":
        window = _odd_window_points_from_nm(x, calibration_median_filter_window_nm, minimum=3)
        return median_filter(y, size=window, mode="nearest")

    raise ValueError(f"Unsupported non-spline smoothing method: {method!r}")


def spline_smooth_one_calibration_curve(
    wavelength_nm,
    calibration_raw,
    counts_per_s_curve=None,
    led_density_curve=None,
):
    """
    Smooth one fiber calibration curve.

    Supported methods are selected by calibration_smoothing_method:
        none, spline, moving_average, gaussian, savgol, median_filter.

    By default smoothing is done in log10(calibration), which is better for
    absolute calibration coefficients because noise is mostly multiplicative and
    the coefficient must remain positive.
    """
    method = _smoothing_method_name()
    wavelength_nm = np.asarray(wavelength_nm, dtype=float)
    calibration_raw = np.asarray(calibration_raw, dtype=float)

    valid = (
        np.isfinite(wavelength_nm)
        & np.isfinite(calibration_raw)
        & (calibration_raw > 0)
    )

    n_valid = int(np.sum(valid))
    if method == "none":
        return calibration_raw.copy(), {
            "smoothing_method": method,
            "smoothing_success": True,
            "smoothing_reason": "smoothing disabled or method='none'",
            "n_points_initial": n_valid,
            "n_points_used": n_valid,
            "spline_success": False,
            "spline_reason": "not used",
            "spline_order_used": np.nan,
            "spline_smoothing_s": np.nan,
        }

    if n_valid < calibration_smoothing_min_points:
        return calibration_raw.copy(), {
            "smoothing_method": method,
            "smoothing_success": False,
            "smoothing_reason": "not enough finite positive calibration points",
            "n_points_initial": n_valid,
            "n_points_used": n_valid,
            "spline_success": False,
            "spline_reason": "not enough finite positive calibration points",
            "spline_order_used": np.nan,
            "spline_smoothing_s": np.nan,
        }

    x = wavelength_nm[valid].astype(float)
    y_raw = calibration_raw[valid].astype(float)

    if calibration_smoothing_fit_in_log_space:
        y = np.log10(y_raw)
    else:
        y = y_raw.copy()

    order = np.argsort(x)
    x = x[order]
    y = y[order]

    x_unique, unique_indices = np.unique(x, return_index=True)
    x = x_unique
    y = y[unique_indices]

    if x.size < calibration_smoothing_min_points:
        return calibration_raw.copy(), {
            "smoothing_method": method,
            "smoothing_success": False,
            "smoothing_reason": "not enough unique wavelength points",
            "n_points_initial": n_valid,
            "n_points_used": int(x.size),
            "spline_success": False,
            "spline_reason": "not enough unique wavelength points",
            "spline_order_used": np.nan,
            "spline_smoothing_s": np.nan,
        }

    weights = None
    if method == "spline" and calibration_spline_weight_by_led_and_counts:
        weight_components = []

        if led_density_curve is not None:
            led = np.asarray(led_density_curve, dtype=float)[valid][order][unique_indices]
            if np.nanmax(led) > 0:
                led_rel = led / np.nanmax(led)
                weight_components.append(np.sqrt(np.clip(led_rel, 0.0, np.inf)))

        if counts_per_s_curve is not None:
            cps = np.asarray(counts_per_s_curve, dtype=float)[valid][order][unique_indices]
            cps = np.where(np.isfinite(cps) & (cps > 0), cps, np.nan)
            if np.nanmax(cps) > 0:
                cps_rel = cps / np.nanmax(cps)
                weight_components.append(np.sqrt(np.clip(cps_rel, 0.0, np.inf)))

        if weight_components:
            weights = np.ones_like(x, dtype=float)
            for comp in weight_components:
                weights *= comp
            weights = np.clip(
                weights,
                float(calibration_spline_min_relative_weight),
                float(calibration_spline_max_relative_weight),
            )
            weights[~np.isfinite(weights)] = float(calibration_spline_min_relative_weight)

    try:
        if method == "spline":
            k = int(calibration_spline_order)
            k = max(1, min(k, 5, x.size - 1))

            var_y = float(np.nanvar(y))
            if not np.isfinite(var_y) or var_y <= 0:
                var_y = 1.0

            s_value = float(calibration_spline_smoothing_factor) * float(x.size) * var_y
            spline1 = UnivariateSpline(x, y, w=weights, k=k, s=s_value, ext=3)
            x_fit = x
            y_fit = y
            weights_fit = weights

            if calibration_spline_outlier_clip_sigma is not None:
                residual = y - spline1(x)
                sig = robust_sigma_mad(residual)
                if np.isfinite(sig) and sig > 0:
                    keep = np.abs(residual) <= float(calibration_spline_outlier_clip_sigma) * sig
                    if np.sum(keep) >= max(calibration_smoothing_min_points, k + 1):
                        x_fit = x[keep]
                        y_fit = y[keep]
                        weights_fit = None if weights is None else weights[keep]

            k2 = max(1, min(k, x_fit.size - 1))
            var_y2 = float(np.nanvar(y_fit))
            if not np.isfinite(var_y2) or var_y2 <= 0:
                var_y2 = var_y
            s_value2 = float(calibration_spline_smoothing_factor) * float(x_fit.size) * var_y2
            spline2 = UnivariateSpline(x_fit, y_fit, w=weights_fit, k=k2, s=s_value2, ext=3)
            y_smooth_full = spline2(wavelength_nm)
            n_used = int(x_fit.size)
            spline_success = True
            spline_reason = ""
            spline_order_used = int(k2)
            spline_smoothing_s = float(s_value2)

        else:
            x_fit, y_fit = _preclip_outliers_for_non_spline(x, y)
            y_smooth_fit = _smooth_working_curve_non_spline(x_fit, y_fit, method)
            y_smooth_full = np.interp(
                wavelength_nm,
                x_fit,
                y_smooth_fit,
                left=float(y_smooth_fit[0]),
                right=float(y_smooth_fit[-1]),
            )
            n_used = int(x_fit.size)
            spline_success = False
            spline_reason = "not used"
            spline_order_used = np.nan
            spline_smoothing_s = np.nan

        if calibration_smoothing_fit_in_log_space:
            smoothed = 10.0 ** y_smooth_full
        else:
            smoothed = y_smooth_full
            smoothed[smoothed <= 0] = np.nan

        smoothed[~np.isfinite(wavelength_nm)] = np.nan
        smoothed[~np.isfinite(smoothed)] = np.nan

        return smoothed, {
            "smoothing_method": method,
            "smoothing_success": True,
            "smoothing_reason": "",
            "n_points_initial": n_valid,
            "n_points_used": n_used,
            "spline_success": spline_success,
            "spline_reason": spline_reason,
            "spline_order_used": spline_order_used,
            "spline_smoothing_s": spline_smoothing_s,
        }

    except Exception as exc:
        return calibration_raw.copy(), {
            "smoothing_method": method,
            "smoothing_success": False,
            "smoothing_reason": str(exc),
            "n_points_initial": n_valid,
            "n_points_used": int(x.size),
            "spline_success": False,
            "spline_reason": str(exc) if method == "spline" else "not used",
            "spline_order_used": np.nan,
            "spline_smoothing_s": np.nan,
        }


def spline_smooth_absolute_calibration_curves(
    wavelength_nm,
    calibration_raw_J_per_nm_per_count,
    counts_per_s,
    led_after_lp_density_per_nm,
    delta_lambda_nm,
):
    """Smooth the absolute calibration curves for all fibers before saving."""
    calibration_raw = np.asarray(calibration_raw_J_per_nm_per_count, dtype=float)
    method = _smoothing_method_name()

    if method == "none":
        info_df = pd.DataFrame(
            {
                "fiber_id": np.arange(calibration_raw.shape[0], dtype=int),
                "smoothing_method": method,
                "smoothing_success": True,
                "smoothing_reason": "smoothing disabled or method='none'",
                "n_points_initial": np.sum(np.isfinite(calibration_raw) & (calibration_raw > 0), axis=1),
                "n_points_used": np.sum(np.isfinite(calibration_raw) & (calibration_raw > 0), axis=1),
                "spline_success": False,
                "spline_reason": "not used",
                "spline_order_used": np.nan,
                "spline_smoothing_s": np.nan,
            }
        )
        return calibration_raw.copy(), calibration_raw * delta_lambda_nm[None, :], info_df

    smoothed = np.full_like(calibration_raw, np.nan, dtype=float)
    rows = []

    for fiber_id in range(calibration_raw.shape[0]):
        curve_smooth, info = spline_smooth_one_calibration_curve(
            wavelength_nm=wavelength_nm,
            calibration_raw=calibration_raw[fiber_id, :],
            counts_per_s_curve=counts_per_s[fiber_id, :],
            led_density_curve=led_after_lp_density_per_nm,
        )
        smoothed[fiber_id, :] = curve_smooth
        info = dict(info)
        info["fiber_id"] = int(fiber_id)
        rows.append(info)

    info_df = pd.DataFrame(rows)
    bin_smoothed = smoothed * delta_lambda_nm[None, :]

    return smoothed, bin_smoothed, info_df


# Backward-compatible alias name: the function now supports all smoothing methods.
smooth_absolute_calibration_curves = spline_smooth_absolute_calibration_curves


def normalize_weird_fiber_ids(n_fibers: int) -> list[int]:
    """Return sorted valid internal fiber IDs from weird_fiber_ids."""
    ids = []
    for value in weird_fiber_ids:
        try:
            fid = int(value)
        except Exception:
            continue
        if weird_fiber_ids_are_one_based:
            fid -= 1
        if 0 <= fid < int(n_fibers):
            ids.append(fid)
        else:
            print(f"WARNING: weird fiber id {value!r} is outside valid range and was ignored.")
    return sorted(set(ids))


def nan_average_curve(curves: np.ndarray, statistic: str) -> np.ndarray:
    """Average curves wavelength-by-wavelength while ignoring NaNs."""
    curves = np.asarray(curves, dtype=float)
    if curves.size == 0:
        return np.full(curves.shape[-1], np.nan, dtype=float)

    statistic = str(statistic).lower().strip()
    with np.errstate(invalid="ignore"):
        if statistic == "median":
            return np.nanmedian(curves, axis=0)
        if statistic == "mean":
            return np.nanmean(curves, axis=0)
    raise ValueError("average_curve_statistic_for_weird_fibers must be 'mean' or 'median'.")


def replace_bad_fiber_curves_with_average_curve(
    wavelength_nm: np.ndarray,
    calibration_J_per_nm_per_count: np.ndarray,
    delta_lambda_nm: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame, np.ndarray, list[int]]:
    """
    Replace manually selected bad fiber calibration curves by the average curve.

    The replacement is applied only to the saved smoothed calibration arrays.
    Raw point-by-point curves are kept unchanged for diagnostics.
    """
    cal = np.asarray(calibration_J_per_nm_per_count, dtype=float).copy()
    delta_lambda_nm = np.asarray(delta_lambda_nm, dtype=float)
    n_fibers = cal.shape[0]
    bad_ids = normalize_weird_fiber_ids(n_fibers)

    rows = []
    replaced = np.zeros(n_fibers, dtype=bool)

    if (not replace_weird_fiber_curves_with_average) or len(bad_ids) == 0:
        for fid in range(n_fibers):
            rows.append(
                {
                    "fiber_id": int(fid),
                    "was_replaced_by_average_curve": False,
                    "replacement_reason": "disabled or not selected",
                    "average_curve_statistic": str(average_curve_statistic_for_weird_fibers),
                    "n_good_fibers_used_for_average": int(n_fibers),
                }
            )
        return cal, cal * delta_lambda_nm[None, :], pd.DataFrame(rows), replaced, bad_ids

    good_ids = [fid for fid in range(n_fibers) if fid not in bad_ids]
    if len(good_ids) == 0:
        raise RuntimeError(
            "All fibers were listed in weird_fiber_ids. Cannot build an average replacement curve."
        )

    good_curves = cal[good_ids, :]
    average_curve = nan_average_curve(good_curves, average_curve_statistic_for_weird_fibers)

    if not np.any(np.isfinite(average_curve) & (average_curve > 0)):
        raise RuntimeError("Average replacement curve is invalid: no finite positive values.")

    for fid in range(n_fibers):
        is_bad = fid in bad_ids
        if is_bad:
            cal[fid, :] = average_curve
            replaced[fid] = True

        rows.append(
            {
                "fiber_id": int(fid),
                "was_replaced_by_average_curve": bool(is_bad),
                "replacement_reason": "manual weird_fiber_ids" if is_bad else "kept own spline curve",
                "average_curve_statistic": str(average_curve_statistic_for_weird_fibers),
                "n_good_fibers_used_for_average": int(len(good_ids)),
            }
        )

    return cal, cal * delta_lambda_nm[None, :], pd.DataFrame(rows), replaced, bad_ids


def apply_common_average_curve_to_all_fibers(
    calibration_J_per_nm_per_count: np.ndarray,
    delta_lambda_nm: np.ndarray,
    weird_fiber_ids_internal: list[int] | None = None,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame, np.ndarray, np.ndarray]:
    """
    Optionally replace every saved fiber curve by one common average curve.

    This is applied after individual smoothing and after optional weird-fiber
    handling. Raw point-by-point curves remain unchanged for diagnostics.
    """
    cal_in = np.asarray(calibration_J_per_nm_per_count, dtype=float)
    cal = cal_in.copy()
    delta_lambda_nm = np.asarray(delta_lambda_nm, dtype=float)
    n_fibers = cal.shape[0]

    common_mask = np.zeros(n_fibers, dtype=bool)
    rows = []

    if not use_common_average_curve_for_all_fibers:
        for fid in range(n_fibers):
            rows.append(
                {
                    "fiber_id": int(fid),
                    "uses_common_average_curve": False,
                    "used_to_build_common_average_curve": False,
                    "common_average_curve_statistic": str(common_average_curve_statistic),
                    "n_fibers_used_for_common_average": 0,
                    "reason": "common average disabled",
                }
            )
        common_curve = np.full(cal.shape[1], np.nan, dtype=float)
        return cal, cal * delta_lambda_nm[None, :], pd.DataFrame(rows), common_mask, common_curve

    if weird_fiber_ids_internal is None:
        weird_fiber_ids_internal = normalize_weird_fiber_ids(n_fibers)

    excluded = set(int(v) for v in weird_fiber_ids_internal) if common_average_exclude_weird_fibers else set()
    good_ids = [fid for fid in range(n_fibers) if fid not in excluded]

    if len(good_ids) == 0:
        raise RuntimeError(
            "No fibers are available to build the common average curve. "
            "Disable common_average_exclude_weird_fibers or remove some weird_fiber_ids."
        )

    common_curve = nan_average_curve(cal[good_ids, :], common_average_curve_statistic)
    if not np.any(np.isfinite(common_curve) & (common_curve > 0)):
        raise RuntimeError("Common average curve is invalid: no finite positive values.")

    for fid in range(n_fibers):
        cal[fid, :] = common_curve
        common_mask[fid] = True
        rows.append(
            {
                "fiber_id": int(fid),
                "uses_common_average_curve": True,
                "used_to_build_common_average_curve": bool(fid in good_ids),
                "common_average_curve_statistic": str(common_average_curve_statistic),
                "n_fibers_used_for_common_average": int(len(good_ids)),
                "reason": "common curve assigned to all fibers",
            }
        )

    return cal, cal * delta_lambda_nm[None, :], pd.DataFrame(rows), common_mask, common_curve


def build_absolute_calibration() -> dict:
    exposure_s = (
        float(calibration_exposure_time_s)
        if calibration_exposure_time_s is not None
        else infer_exposure_seconds_from_filename(calibration_image_path)
    )

    # Load and process calibration image.
    signal_img = load_image_as_float(calibration_image_path)
    if subtract_background_image:
        bg_img = load_image_as_float(background_image_path)
        img = subtract_background(signal_img, bg_img)
    else:
        img = signal_img

    # Keep a raw, uncalibrated, non-background-subtracted version of the IR-LED
    # calibration image for the optional raw-counts 3D diagnostic plot.
    raw_signal_img_for_counts = signal_img.copy()

    if rotate_ccw_90_image:
        img = np.rot90(img, k=1)
        raw_signal_img_for_counts = np.rot90(raw_signal_img_for_counts, k=1)

    # Load fiber reference and integrate image.
    ref_df = pd.read_csv(resolve_existing_path(reference_file)).sort_values("fiber_id").reset_index(drop=True)
    integrated_counts, x_sample, _, _ = integrate_image_in_pixel_space(
        img=img,
        ref_df=ref_df,
        integration_width=integration_width_calibration,
    )
    counts_per_s = integrated_counts / exposure_s

    # Same integration, but on the raw IR-LED calibration image before background
    # subtraction. This is used only for the optional 3D raw-counts sphere plot.
    integrated_counts_raw_signal, _, _, _ = integrate_image_in_pixel_space(
        img=raw_signal_img_for_counts,
        ref_df=ref_df,
        integration_width=integration_width_calibration,
    )
    counts_per_s_raw_signal = integrated_counts_raw_signal / exposure_s

    # Load wavelength axis and bin widths.
    wavelengths_nm, wavelength_source = load_wavelength_axis_for_x_sample(x_sample)
    delta_lambda_nm = wavelength_bin_widths(wavelengths_nm)

    # Load and normalize LED spectrum.
    led_wl_raw, led_signal_raw = load_true_led_spectrum(ir_led_true_spectrum_file)
    led_wl_raw, led_density_raw = normalize_spectral_density(led_wl_raw, led_signal_raw)
    led_density_raw = apply_optional_fiber_attenuation(led_wl_raw, led_density_raw)

    # Power meter spectral correction.
    pm_correction = compute_powermeter_spectral_correction(led_wl_raw, led_density_raw)

    # Power entering each fiber.
    power_df = parse_power_meter_readings(power_meter_readings_file)
    power_df = compute_power_entering_fibers(
        power_df=power_df,
        spectral_correction_factor=pm_correction["powermeter_true_power_correction_factor"],
    )

    # Longpass-filtered LED spectral density on the camera wavelength axis.
    led_density_on_camera_axis = interpolate_density_to_axis(led_wl_raw, led_density_raw, wavelengths_nm)
    longpass_transmission = load_longpass_transmission(wavelengths_nm)

    # Hard physical cutoff: below zero_energy_below_wavelength_nm, the true
    # transmitted calibration light is defined as zero. Counts measured by the
    # camera below this cutoff are therefore never converted to nonzero energy.
    physical_energy_allowed_mask = physical_energy_wavelength_mask(wavelengths_nm)
    if force_zero_energy_below_wavelength:
        longpass_transmission = np.asarray(longpass_transmission, dtype=float).copy()
        longpass_transmission[~physical_energy_allowed_mask] = 0.0

    led_after_lp_unnorm = led_density_on_camera_axis * longpass_transmission
    led_density_area_on_camera_axis = float(np.sum(led_density_on_camera_axis * delta_lambda_nm))
    longpass_fraction = float(np.sum(led_after_lp_unnorm * delta_lambda_nm))

    if longpass_fraction <= 0:
        raise RuntimeError("Longpass-filtered LED spectrum has zero integral on the camera wavelength axis.")

    led_after_lp_density_per_nm = led_after_lp_unnorm / longpass_fraction

    # Convert power into the passband and distribute it spectrally.
    power_lookup = power_df.set_index("fiber_id")["power_entering_fiber_total_W"].to_dict()
    power_after_lp_lookup = {
        fid: float(power_lookup[fid]) * longpass_fraction
        for fid in power_lookup
    }

    n_fibers, n_wl = counts_per_s.shape
    power_spectral_W_per_nm = np.full_like(counts_per_s, np.nan, dtype=float)
    power_bin_W = np.full_like(counts_per_s, np.nan, dtype=float)

    # Raw point-by-point calibration. This is physically direct but noisy where
    # the IR LED spectrum is weak. It is kept for diagnostics only.
    calibration_J_per_nm_per_count_raw = np.full_like(counts_per_s, np.nan, dtype=float)
    calibration_J_per_count_bin_raw = np.full_like(counts_per_s, np.nan, dtype=float)

    for fiber_id in range(n_fibers):
        p_lp = power_after_lp_lookup.get(fiber_id, np.nan)
        if not np.isfinite(p_lp):
            continue

        power_spectral_W_per_nm[fiber_id, :] = p_lp * led_after_lp_density_per_nm
        power_bin_W[fiber_id, :] = power_spectral_W_per_nm[fiber_id, :] * delta_lambda_nm

        cps = counts_per_s[fiber_id, :]
        valid = np.isfinite(cps) & (cps > 0)
        calibration_J_per_nm_per_count_raw[fiber_id, valid] = power_spectral_W_per_nm[fiber_id, valid] / cps[valid]
        calibration_J_per_count_bin_raw[fiber_id, valid] = power_bin_W[fiber_id, valid] / cps[valid]

    # Saved calibration: spline-smoothed by default.
    calibration_J_per_nm_per_count, calibration_J_per_count_bin, spline_info_df = (
        spline_smooth_absolute_calibration_curves(
            wavelength_nm=wavelengths_nm,
            calibration_raw_J_per_nm_per_count=calibration_J_per_nm_per_count_raw,
            counts_per_s=counts_per_s,
            led_after_lp_density_per_nm=led_after_lp_density_per_nm,
            delta_lambda_nm=delta_lambda_nm,
        )
    )

    # Optional manual replacement of weird/bad fiber curves.
    # This is applied only to the saved smoothed calibration arrays. Raw curves
    # are kept unchanged for diagnostics and for the 5x4 plots.
    (
        calibration_J_per_nm_per_count,
        calibration_J_per_count_bin,
        weird_fiber_replacement_df,
        weird_fiber_replaced_mask,
        weird_fiber_ids_internal,
    ) = replace_bad_fiber_curves_with_average_curve(
        wavelength_nm=wavelengths_nm,
        calibration_J_per_nm_per_count=calibration_J_per_nm_per_count,
        delta_lambda_nm=delta_lambda_nm,
    )

    # Optional: use a single common average calibration/transmission curve for
    # every fiber. This is applied only to the saved smoothed arrays.
    (
        calibration_J_per_nm_per_count,
        calibration_J_per_count_bin,
        common_average_curve_df,
        common_average_curve_mask,
        common_average_curve,
    ) = apply_common_average_curve_to_all_fibers(
        calibration_J_per_nm_per_count=calibration_J_per_nm_per_count,
        delta_lambda_nm=delta_lambda_nm,
        weird_fiber_ids_internal=weird_fiber_ids_internal,
    )

    # Final hard cutoff applied after smoothing, weird-fiber replacement, and
    # optional common-average replacement. This guarantees that calibrated
    # energy is exactly zero below the physical cutoff, no matter what camera
    # counts are measured there.
    zero_cutoff_info = enforce_zero_energy_cutoff_on_spectral_arrays(
        wavelength_nm=wavelengths_nm,
        arrays_2d=[
            power_spectral_W_per_nm,
            power_bin_W,
            calibration_J_per_nm_per_count_raw,
            calibration_J_per_count_bin_raw,
            calibration_J_per_nm_per_count,
            calibration_J_per_count_bin,
        ],
        arrays_1d=[
            led_after_lp_density_per_nm,
            longpass_transmission,
            common_average_curve if np.asarray(common_average_curve).size == len(wavelengths_nm) else None,
        ],
    )

    # Keep bin calibration exactly consistent with per-nm calibration after the
    # cutoff has been applied.
    calibration_J_per_count_bin = calibration_J_per_nm_per_count * delta_lambda_nm[None, :]
    calibration_J_per_count_bin_raw = calibration_J_per_nm_per_count_raw * delta_lambda_nm[None, :]
    power_bin_W = power_spectral_W_per_nm * delta_lambda_nm[None, :]

    # Update power summary with passband power.
    power_df["longpass_fraction_of_led_power_on_camera_axis"] = longpass_fraction
    power_df["power_entering_fiber_after_longpass_W"] = (
        power_df["power_entering_fiber_total_W"] * longpass_fraction
    )
    power_df["camera_counts_total"] = np.nansum(integrated_counts, axis=1)
    power_df["camera_counts_per_s_total"] = np.nansum(counts_per_s, axis=1)
    power_df["camera_counts_total_raw_signal"] = np.nansum(integrated_counts_raw_signal, axis=1)
    power_df["camera_counts_per_s_total_raw_signal"] = np.nansum(counts_per_s_raw_signal, axis=1)

    return {
        "exposure_s": exposure_s,
        "img": img,
        "raw_signal_img": raw_signal_img_for_counts,
        "ref_df": ref_df,
        "integrated_counts": integrated_counts,
        "counts_per_s": counts_per_s,
        "integrated_counts_raw_signal": integrated_counts_raw_signal,
        "counts_per_s_raw_signal": counts_per_s_raw_signal,
        "x_sample": x_sample,
        "wavelengths_nm": wavelengths_nm,
        "delta_lambda_nm": delta_lambda_nm,
        "wavelength_source": wavelength_source,
        "led_wl_raw": led_wl_raw,
        "led_density_raw": led_density_raw,
        "led_density_on_camera_axis": led_density_on_camera_axis,
        "longpass_transmission": longpass_transmission,
        "longpass_fraction": longpass_fraction,
        "led_density_area_on_camera_axis": led_density_area_on_camera_axis,
        "led_after_lp_density_per_nm": led_after_lp_density_per_nm,
        "pm_correction": pm_correction,
        "power_df": power_df,
        "power_spectral_W_per_nm": power_spectral_W_per_nm,
        "power_bin_W": power_bin_W,
        "calibration_J_per_nm_per_count": calibration_J_per_nm_per_count,
        "calibration_J_per_count_bin": calibration_J_per_count_bin,
        "calibration_J_per_nm_per_count_raw": calibration_J_per_nm_per_count_raw,
        "calibration_J_per_count_bin_raw": calibration_J_per_count_bin_raw,
        "spline_info_df": spline_info_df,
        "weird_fiber_replacement_df": weird_fiber_replacement_df,
        "weird_fiber_replaced_mask": weird_fiber_replaced_mask,
        "weird_fiber_ids_internal": weird_fiber_ids_internal,
        "common_average_curve_df": common_average_curve_df,
        "common_average_curve_mask": common_average_curve_mask,
        "common_average_curve": common_average_curve,
        **zero_cutoff_info,
    }


def save_outputs(result: dict) -> None:
    wavelengths_nm = result["wavelengths_nm"]
    delta_lambda_nm = result["delta_lambda_nm"]
    fiber_ids = np.arange(result["integrated_counts"].shape[0], dtype=int)

    # NPZ for fast loading by plotting code.
    np.savez_compressed(
        output_npz,
        fiber_id=fiber_ids,
        wavelength_nm=wavelengths_nm,
        delta_lambda_nm=delta_lambda_nm,
        calibration_J_per_nm_per_count=result["calibration_J_per_nm_per_count"],
        calibration_J_per_count_bin=result["calibration_J_per_count_bin"],
        calibration_J_per_nm_per_count_raw=result["calibration_J_per_nm_per_count_raw"],
        calibration_J_per_count_bin_raw=result["calibration_J_per_count_bin_raw"],
        weird_fiber_replaced_mask=np.asarray(result.get("weird_fiber_replaced_mask", np.zeros(len(fiber_ids), dtype=bool)), dtype=bool),
        weird_fiber_ids_internal=np.asarray(result.get("weird_fiber_ids_internal", []), dtype=int),
        common_average_curve_mask=np.asarray(result.get("common_average_curve_mask", np.zeros(len(fiber_ids), dtype=bool)), dtype=bool),
        use_common_average_curve_for_all_fibers=bool(use_common_average_curve_for_all_fibers),
        smoothing_enabled=bool(smooth_saved_absolute_calibration),
        smoothing_method=str(_smoothing_method_name()),
        smoothing_fit_in_log_space=bool(calibration_smoothing_fit_in_log_space),
        smoothing_factor=float(calibration_spline_smoothing_factor),
        power_spectral_W_per_nm=result["power_spectral_W_per_nm"],
        power_bin_W=result["power_bin_W"],
        camera_counts=result["integrated_counts"],
        camera_counts_per_s=result["counts_per_s"],
        camera_counts_raw_signal=result.get("integrated_counts_raw_signal", result["integrated_counts"]),
        camera_counts_per_s_raw_signal=result.get("counts_per_s_raw_signal", result["counts_per_s"]),
        led_after_longpass_density_per_nm=result["led_after_lp_density_per_nm"],
        longpass_transmission=result["longpass_transmission"],
        force_zero_energy_below_wavelength=bool(result.get("force_zero_energy_below_wavelength", False)),
        zero_energy_below_wavelength_nm=float(result.get("zero_energy_below_wavelength_nm", np.nan)),
        physical_energy_allowed_wavelength_mask=np.asarray(
            result.get("physical_energy_allowed_wavelength_mask", np.ones_like(wavelengths_nm, dtype=bool)),
            dtype=bool,
        ),
        n_zeroed_wavelength_bins=int(result.get("n_zeroed_wavelength_bins", 0)),
    )

    # Long CSV for transparency.
    rows = []
    for fiber_id in fiber_ids:
        for j, wl in enumerate(wavelengths_nm):
            rows.append(
                {
                    "fiber_id": int(fiber_id),
                    "wavelength_nm": float(wl),
                    "delta_lambda_nm": float(delta_lambda_nm[j]),
                    "physical_energy_allowed": bool(result.get("physical_energy_allowed_wavelength_mask", np.ones_like(wavelengths_nm, dtype=bool))[j]),
                    "camera_counts": float(result["integrated_counts"][fiber_id, j]),
                    "camera_counts_per_s": float(result["counts_per_s"][fiber_id, j]),
                    "power_spectral_W_per_nm_at_fiber_input": float(result["power_spectral_W_per_nm"][fiber_id, j]),
                    "power_bin_W_at_fiber_input": float(result["power_bin_W"][fiber_id, j]),
                    "calibration_J_per_nm_per_count": float(result["calibration_J_per_nm_per_count"][fiber_id, j]),
                    "calibration_J_per_count_bin": float(result["calibration_J_per_count_bin"][fiber_id, j]),
                    "calibration_J_per_nm_per_count_raw": float(result["calibration_J_per_nm_per_count_raw"][fiber_id, j]),
                    "calibration_J_per_count_bin_raw": float(result["calibration_J_per_count_bin_raw"][fiber_id, j]),
                }
            )
    pd.DataFrame(rows).to_csv(output_long_csv, index=False)

    result["power_df"].to_csv(output_power_summary_csv, index=False)

    spline_info_csv = CALIBRATION_FOLDER / "absolute_calibration_spline_fit_info_ELI-MAY.csv"
    result["spline_info_df"].to_csv(spline_info_csv, index=False)

    replacement_df = result.get("weird_fiber_replacement_df", pd.DataFrame())
    if replacement_df is not None and len(replacement_df) > 0:
        replacement_df.to_csv(output_weird_fiber_replacement_csv, index=False)

    common_average_df = result.get("common_average_curve_df", pd.DataFrame())
    if common_average_df is not None and len(common_average_df) > 0:
        common_average_df.to_csv(output_common_average_status_csv, index=False)

    common_average_curve = np.asarray(result.get("common_average_curve", []), dtype=float)
    if common_average_curve.size == len(wavelengths_nm) and np.any(np.isfinite(common_average_curve)):
        pd.DataFrame(
            {
                "wavelength_nm": wavelengths_nm,
                "common_calibration_J_per_nm_per_count": common_average_curve,
                "common_calibration_J_per_count_bin": common_average_curve * delta_lambda_nm,
            }
        ).to_csv(output_common_average_curve_csv, index=False)

    metadata = {
        "calibration_image_path": str(calibration_image_path),
        "background_image_path": str(background_image_path),
        "reference_file": str(reference_file),
        "wavelength_source": str(result["wavelength_source"]),
        "exposure_s": float(result["exposure_s"]),
        "integration_width_calibration": float(integration_width_calibration),
        "powermeter_selected_wavelength_nm": float(powermeter_selected_wavelength_nm),
        "powermeter_responsivity_csv": str(powermeter_responsivity_csv),
        "ir_led_true_spectrum_file": str(ir_led_true_spectrum_file),
        "sphere_radius_cm": float(sphere_radius_cm),
        "powermeter_sensor_closer_to_led_by_cm": float(powermeter_sensor_closer_to_led_by_cm),
        "powermeter_input_aperture_diameter_mm": float(powermeter_input_aperture_diameter_mm),
        "fiber_core_diameter_um": float(fiber_core_diameter_um),
        "fiber_numerical_aperture": float(fiber_numerical_aperture),
        "apply_na_solid_angle_correction": bool(apply_na_solid_angle_correction),
        "longpass_cutoff_nm": float(longpass_cutoff_nm),
        "longpass_filter_curve_csv": None if longpass_filter_curve_csv is None else str(longpass_filter_curve_csv),
        "force_zero_energy_below_wavelength": bool(result.get("force_zero_energy_below_wavelength", False)),
        "zero_energy_below_wavelength_nm": float(result.get("zero_energy_below_wavelength_nm", np.nan)),
        "n_zeroed_wavelength_bins": int(result.get("n_zeroed_wavelength_bins", 0)),
        "smooth_saved_absolute_calibration": bool(smooth_saved_absolute_calibration),
        "calibration_smoothing_method": str(_smoothing_method_name()),
        "calibration_smoothing_fit_in_log_space": bool(calibration_smoothing_fit_in_log_space),
        "calibration_smoothing_min_points": int(calibration_smoothing_min_points),
        "calibration_spline_fit_in_log_space": bool(calibration_spline_fit_in_log_space),
        "calibration_spline_smoothing_factor": float(calibration_spline_smoothing_factor),
        "calibration_spline_order": int(calibration_spline_order),
        "calibration_spline_min_points": int(calibration_spline_min_points),
        "calibration_spline_outlier_clip_sigma": None if calibration_spline_outlier_clip_sigma is None else float(calibration_spline_outlier_clip_sigma),
        "calibration_spline_weight_by_led_and_counts": bool(calibration_spline_weight_by_led_and_counts),
        "calibration_spline_min_relative_weight": float(calibration_spline_min_relative_weight),
        "calibration_moving_average_window_nm": float(calibration_moving_average_window_nm),
        "calibration_gaussian_sigma_nm": float(calibration_gaussian_sigma_nm),
        "calibration_savgol_window_nm": float(calibration_savgol_window_nm),
        "calibration_savgol_polyorder": int(calibration_savgol_polyorder),
        "calibration_median_filter_window_nm": float(calibration_median_filter_window_nm),
        "calibration_non_spline_outlier_clip_sigma": None if calibration_non_spline_outlier_clip_sigma is None else float(calibration_non_spline_outlier_clip_sigma),
        "replace_weird_fiber_curves_with_average": bool(replace_weird_fiber_curves_with_average),
        "weird_fiber_ids_user_input": [int(v) for v in weird_fiber_ids],
        "weird_fiber_ids_are_one_based": bool(weird_fiber_ids_are_one_based),
        "weird_fiber_ids_internal": [int(v) for v in result.get("weird_fiber_ids_internal", [])],
        "average_curve_statistic_for_weird_fibers": str(average_curve_statistic_for_weird_fibers),
        "use_common_average_curve_for_all_fibers": bool(use_common_average_curve_for_all_fibers),
        "common_average_curve_statistic": str(common_average_curve_statistic),
        "common_average_exclude_weird_fibers": bool(common_average_exclude_weird_fibers),
        "longpass_fraction_of_led_power_on_camera_axis": float(result["longpass_fraction"]),
        "led_density_area_on_camera_axis_before_longpass": float(result["led_density_area_on_camera_axis"]),
        **result["pm_correction"],
    }
    output_metadata_json.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print("Saved absolute calibration files:")
    print(f"  {output_npz.resolve()}")
    print(f"  {output_long_csv.resolve()}")
    print(f"  {output_power_summary_csv.resolve()}")
    print(f"  {spline_info_csv.resolve()}")
    if replacement_df is not None and len(replacement_df) > 0:
        print(f"  {output_weird_fiber_replacement_csv.resolve()}")
    if common_average_df is not None and len(common_average_df) > 0:
        print(f"  {output_common_average_status_csv.resolve()}")
    if common_average_curve.size == len(wavelengths_nm) and np.any(np.isfinite(common_average_curve)):
        print(f"  {output_common_average_curve_csv.resolve()}")
    print(f"  {output_metadata_json.resolve()}")



def apply_scientific_article_style():
    """Apply compact scientific-article style to matplotlib."""
    if not use_scientific_article_plot_style:
        return
    plt.rcParams.update(
        {
            "font.family": PLOT_STYLE["font_family"],
            "font.size": PLOT_STYLE["font_size"],
            "axes.titlesize": PLOT_STYLE["title_size"],
            "axes.labelsize": PLOT_STYLE["label_size"],
            "axes.titleweight": PLOT_STYLE["title_weight"],
            "axes.labelweight": PLOT_STYLE["label_weight"],
            "xtick.labelsize": PLOT_STYLE["tick_size"],
            "ytick.labelsize": PLOT_STYLE["tick_size"],
            "legend.fontsize": PLOT_STYLE["legend_size"],
            "axes.linewidth": PLOT_STYLE["spine_linewidth"],
            "xtick.major.width": PLOT_STYLE["tick_width"],
            "ytick.major.width": PLOT_STYLE["tick_width"],
            "xtick.minor.width": PLOT_STYLE["tick_width"],
            "ytick.minor.width": PLOT_STYLE["tick_width"],
            "xtick.major.size": PLOT_STYLE["tick_length"],
            "ytick.major.size": PLOT_STYLE["tick_length"],
            "xtick.minor.size": PLOT_STYLE["minor_tick_length"],
            "ytick.minor.size": PLOT_STYLE["minor_tick_length"],
            "figure.dpi": PLOT_STYLE["dpi"],
            "savefig.dpi": PLOT_STYLE["save_dpi"],
        }
    )


def setup_paper_axis(ax, title=None, xlabel=None, ylabel=None, yscale="linear", xlim=None, ylim=None):
    """Style one 2D axis with article-like ticks and optional grid."""
    if title is not None:
        ax.set_title(str(title), pad=PLOT_STYLE["title_pad"])
    if xlabel is not None:
        ax.set_xlabel(str(xlabel), labelpad=PLOT_STYLE["label_pad"])
    if ylabel is not None:
        ax.set_ylabel(str(ylabel), labelpad=PLOT_STYLE["label_pad"])
    if xlim is not None:
        ax.set_xlim(*xlim)
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.set_yscale(yscale)

    ax.tick_params(
        direction=PLOT_STYLE["tick_direction"],
        which="both",
        top=bool(PLOT_STYLE["ticks_top"]),
        right=bool(PLOT_STYLE["ticks_right"]),
    )
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    if str(yscale).lower() == "log":
        ax.yaxis.set_major_locator(LogLocator(base=10))
        ax.yaxis.set_minor_locator(LogLocator(base=10, subs=np.arange(2, 10) * 0.1))
        ax.yaxis.set_minor_formatter(NullFormatter())
    else:
        ax.yaxis.set_minor_locator(AutoMinorLocator())

    if PLOT_STYLE["show_grid"]:
        ax.grid(True, which="major", alpha=PLOT_STYLE["grid_alpha"], linewidth=PLOT_STYLE["grid_linewidth"])
        ax.grid(True, which="minor", alpha=PLOT_STYLE["minor_grid_alpha"], linewidth=PLOT_STYLE["minor_grid_linewidth"])
    else:
        ax.grid(False)

    for spine in ax.spines.values():
        spine.set_linewidth(PLOT_STYLE["spine_linewidth"])


def add_paper_legend(ax, loc=None, ncol=None):
    """Add an opaque, publication-style legend if labelled artists exist."""
    handles, labels = ax.get_legend_handles_labels()
    if not handles:
        return None
    legend = ax.legend(
        loc=PLOT_STYLE["legend_location"] if loc is None else loc,
        frameon=PLOT_STYLE["legend_frame"],
        framealpha=PLOT_STYLE["legend_frame_alpha"],
        ncol=PLOT_STYLE["legend_ncol"] if ncol is None else ncol,
    )
    if legend is not None and PLOT_STYLE["legend_frame"]:
        frame = legend.get_frame()
        frame.set_edgecolor(PLOT_STYLE["legend_edge_color"])
        frame.set_facecolor(PLOT_STYLE["legend_face_color"])
        frame.set_linewidth(PLOT_STYLE["legend_linewidth"])
    return legend


def add_paper_colorbar(fig, ax, mappable, label):
    """Attach a controlled colorbar to an axis."""
    cbar = fig.colorbar(
        mappable,
        ax=ax,
        pad=PLOT_STYLE["colorbar_pad"],
        fraction=PLOT_STYLE["colorbar_fraction"],
        aspect=PLOT_STYLE["colorbar_aspect"],
    )
    cbar.set_label(label, fontsize=PLOT_STYLE["colorbar_label_size"])
    cbar.ax.tick_params(labelsize=PLOT_STYLE["colorbar_tick_size"])
    cbar.outline.set_linewidth(PLOT_STYLE["spine_linewidth"])
    return cbar


def save_paper_figure(fig, path):
    """Save a figure using the global article-style export settings."""
    fig.tight_layout()
    fig.savefig(
        path,
        dpi=PLOT_STYLE["save_dpi"],
        transparent=PLOT_STYLE["transparent"],
        bbox_inches=PLOT_STYLE["bbox_inches"],
    )


def fiber_color_mapper(n_fibers=expected_number_of_fibers):
    """Return colormap and normalization for fiber-index dependent line colors."""
    cmap = cm.get_cmap(PLOT_STYLE["fiber_line_colormap"])
    vmin = float(PLOT_STYLE["fiber_line_color_min"])
    vmax = float(PLOT_STYLE["fiber_line_color_max"])
    if not np.isfinite(vmin):
        vmin = 0.0
    if not np.isfinite(vmax) or vmax <= vmin:
        vmax = max(float(n_fibers - 1), vmin + 1.0)
    norm = colors.Normalize(vmin=vmin, vmax=vmax)
    return cmap, norm


def color_for_fiber(fiber_id, n_fibers=expected_number_of_fibers):
    """Color selected from a colormap according to fiber index."""
    cmap, norm = fiber_color_mapper(n_fibers=n_fibers)
    return cmap(norm(float(fiber_id)))


def make_calibration_fibre_config_28052026():
    """May/foam sphere fiber-port angles: fiber_id -> (phi_deg, theta_deg)."""
    return {
         0: (  0,  84.26),    1: (  0,  78.52),    2: (  0,  72.78),
         3: (  0,  67.04),    4: (  0,  61.30),    5: (  0,  55.56),
         6: (  0,  49.82),    7: (  0,  44.08),    8: (  0,  38.34),
         9: (  0,  32.60),   10: (  0,  26.86),   11: (  0,  21.12),
        12: ( 10,  84.26),   13: ( 10,  78.52),   14: ( 10,  72.78),
        15: ( 10,  67.04),   16: ( 10,  61.30),   17: ( 10,  49.82),
        18: ( 10,  44.08),   19: ( 10,  38.34),   20: ( 10,  32.60),
        21: ( 10,  26.86),   22: ( 10,  21.12),
        23: ( 20,  84.26),   24: ( 20,  72.78),   25: ( 20,  67.04),
        26: ( 20,  55.56),   27: ( 20,  49.82),   28: ( 20,  38.34),
        29: ( 20,  32.60),   30: ( 20,  21.12),
        31: ( 30,  84.26),   32: ( 30,  72.78),   33: ( 30,  61.30),
        34: ( 30,  44.08),   35: ( 30,  32.60),   36: ( 30,  21.12),
        37: ( 40,  84.26),   38: ( 40,  72.78),   39: ( 40,  61.30),
        40: ( 40,  44.08),   41: ( 40,  32.60),   42: ( 40,  21.12),
        43: ( 50,  84.26),   44: ( 50,  72.78),   45: ( 50,  61.30),
        46: ( 50,  44.08),   47: ( 50,  32.60),   48: ( 50,  21.12),
        49: ( 60,  84.26),   50: ( 60,  72.78),   51: ( 60,  67.04),
        52: ( 60,  55.56),   53: ( 60,  49.82),   54: ( 60,  38.34),
        55: ( 60,  32.60),   56: ( 60,  21.12),
        57: ( 70,  84.26),   58: ( 70,  78.52),   59: ( 70,  72.78),
        60: ( 70,  67.04),   61: ( 70,  61.30),   62: ( 70,  49.82),
        63: ( 70,  44.08),   64: ( 70,  38.34),   65: ( 70,  32.60),
        66: ( 70,  26.86),   67: ( 70,  21.12),
        68: ( 80,  84.26),   69: ( 80,  78.52),   70: ( 80,  72.78),
        71: ( 80,  67.04),   72: ( 80,  61.30),   73: ( 80,  55.56),
        74: ( 80,  49.82),   75: ( 80,  44.08),   76: ( 80,  38.34),
        77: ( 80,  32.60),   78: ( 80,  26.86),   79: ( 80,  21.12),
    }


def get_fiber_power_sphere_config():
    """Return the fiber-port angle dictionary used for the 3D power sphere."""
    name = str(fiber_power_sphere_config_name).strip().lower().replace("_", "-")
    if name in {"28052026", "config-28052026", "fauvel-b", "may", "eli-may"}:
        return make_calibration_fibre_config_28052026()
    raise ValueError(
        "Unsupported fiber_power_sphere_config_name. Currently available: '28052026'."
    )


def sphere_xyz_from_phi_theta(phi_deg, theta_deg, radius_cm):
    """Convert physical sphere angles to Cartesian coordinates.

    Convention:
        theta = 0 deg is the laser axis (+x),
        theta = 90 deg, phi = 0 deg is +y,
        theta = 90 deg, phi = 90 deg is +z.
    """
    phi = np.deg2rad(np.asarray(phi_deg, dtype=float))
    theta = np.deg2rad(np.asarray(theta_deg, dtype=float))
    r = float(radius_cm)
    x = r * np.cos(theta)
    y = r * np.sin(theta) * np.cos(phi)
    z = r * np.sin(theta) * np.sin(phi)
    return x, y, z


def fold_phi_to_measured_quarter_deg(phi_deg):
    """Mirror any phi onto the measured 0..90 deg quadrant.

    This respects both phi=0 and phi=90 deg as symmetry planes. In particular,
    phi=100 deg maps to 80 deg, not to 10 deg.
    """
    phi = np.asarray(phi_deg, dtype=float)
    phi360 = np.mod(phi, 360.0)
    phi180 = np.mod(phi360, 180.0)
    return np.where(phi180 <= 90.0, phi180, 180.0 - phi180)


def interpolate_quantity_on_power_sphere(phi_points, theta_points, value_points, phi_grid, theta_grid):
    """Interpolate measured fiber values over the requested 3D sphere domain.

    Default behavior is the measured quarter sphere only. No symmetry folding is
    applied unless fiber_power_sphere_plot_reconstructed_half_sphere=True.
    """
    phi_points = np.asarray(phi_points, dtype=float)
    theta_points = np.asarray(theta_points, dtype=float)
    value_points = np.asarray(value_points, dtype=float)

    valid = np.isfinite(phi_points) & np.isfinite(theta_points) & np.isfinite(value_points) & (value_points > 0)
    phi_points = phi_points[valid]
    theta_points = theta_points[valid]
    value_points = value_points[valid]

    if value_points.size < 3:
        raise RuntimeError("Not enough finite positive fiber powers to build the 3D sphere map.")

    # Add a synthetic theta=0 axis value when no port is exactly on the laser axis.
    # At theta=0 all phi values represent the same point, so multiple phi samples
    # stabilize interpolation near the pole.
    if fiber_power_sphere_add_axis_point:
        theta_min = float(np.nanmin(theta_points))
        near_axis = theta_points <= theta_min + float(fiber_power_sphere_axis_point_theta_window_deg)
        if np.any(near_axis):
            axis_value = float(np.nanmedian(value_points[near_axis]))
            axis_phi = np.linspace(0.0, 90.0, 10)
            phi_points = np.concatenate([phi_points, axis_phi])
            theta_points = np.concatenate([theta_points, np.zeros_like(axis_phi)])
            value_points = np.concatenate([value_points, np.full_like(axis_phi, axis_value, dtype=float)])

    if bool(fiber_power_sphere_plot_reconstructed_half_sphere):
        # Optional old/debug behavior: fold the full half-sphere onto the measured
        # 0..90 deg quarter using the declared symmetry planes.
        query_phi = fold_phi_to_measured_quarter_deg(phi_grid)
    else:
        # Default calibration diagnostic: plot only the measured quarter sphere.
        # The query points already lie in 0..90 deg, so no mirroring is applied.
        query_phi = np.asarray(phi_grid, dtype=float)

    points = np.column_stack([phi_points, theta_points])
    query = np.column_stack([query_phi.ravel(), theta_grid.ravel()])

    method = str(fiber_power_sphere_interpolation_method).lower().strip()
    if method not in {"linear", "nearest", "cubic"}:
        method = "linear"

    interp = griddata(points, value_points, query, method=method)
    interp = np.asarray(interp, dtype=float).reshape(phi_grid.shape)

    # Fill outside the convex hull with nearest-neighbour values to avoid holes
    # near the pole and near sparse boundaries.
    nearest = griddata(points, value_points, query, method="nearest")
    nearest = np.asarray(nearest, dtype=float).reshape(phi_grid.shape)
    interp[~np.isfinite(interp)] = nearest[~np.isfinite(interp)]

    return interp


def set_3d_axes_equal(ax, radius_cm):
    """Set equal 3D axis limits for either the measured quarter or full half-sphere."""
    r = float(radius_cm)
    ax.set_xlim(0.0, r)

    if bool(fiber_power_sphere_plot_reconstructed_half_sphere):
        ax.set_ylim(-r, r)
        ax.set_zlim(-r, r)
        aspect = (1.0, 2.0, 2.0)
    else:
        # Measured quarter sphere: 0 <= phi <= 90 deg, so y and z are positive.
        ax.set_ylim(0.0, r)
        ax.set_zlim(0.0, r)
        aspect = (1.0, 1.0, 1.0)

    try:
        ax.set_box_aspect(aspect)
    except Exception:
        pass


def clean_3d_axis_grid(ax):
    """Remove grey 3D grid/pane lines for a cleaner article-style sphere plot."""
    ax.grid(bool(fiber_power_sphere_axis_grid))
    if not fiber_power_sphere_axis_grid:
        for axis in [ax.xaxis, ax.yaxis, ax.zaxis]:
            try:
                axis._axinfo["grid"]["linewidth"] = 0.0
            except Exception:
                pass
    if not fiber_power_sphere_axis_panes:
        for axis in [ax.xaxis, ax.yaxis, ax.zaxis]:
            try:
                axis.pane.fill = False
                axis.pane.set_edgecolor("white")
            except Exception:
                pass



def _triangle_edge_lengths_in_angle_space(triangulation):
    """Return the maximum edge length of each triangle in (phi, theta) degrees."""
    xy = np.column_stack([triangulation.x, triangulation.y])
    tris = triangulation.triangles
    max_edges = np.zeros(len(tris), dtype=float)
    for i, tri in enumerate(tris):
        pts = xy[tri]
        d01 = np.linalg.norm(pts[0] - pts[1])
        d12 = np.linalg.norm(pts[1] - pts[2])
        d20 = np.linalg.norm(pts[2] - pts[0])
        max_edges[i] = max(d01, d12, d20)
    return max_edges


def _build_fiber_triangulated_surface(
    phi_deg,
    theta_deg,
    values,
    *,
    use_log_color=None,
    color_percentiles=None,
    colormap_name=None,
    surface_alpha=None,
):
    """Build a linearly interpolated triangular 3D surface from fiber positions only.

    No regular quarter-sphere grid is generated. The triangulation is done in
    measured (phi, theta) coordinates, and the plotted surface is the same
    spherical patch restricted to the convex hull of the measured fiber ports.
    The value can be either physical power [W] or raw uncalibrated camera counts.
    """
    phi_deg = np.asarray(phi_deg, dtype=float)
    theta_deg = np.asarray(theta_deg, dtype=float)
    values = np.asarray(values, dtype=float)

    valid = np.isfinite(phi_deg) & np.isfinite(theta_deg) & np.isfinite(values) & (values > 0)
    phi_deg = phi_deg[valid]
    theta_deg = theta_deg[valid]
    values = values[valid]

    if values.size < 3:
        raise RuntimeError("Not enough finite positive fiber powers for triangulated 3D plot.")

    triang = mtri.Triangulation(phi_deg, theta_deg)

    if bool(fiber_power_sphere_mask_large_triangles):
        max_edges = _triangle_edge_lengths_in_angle_space(triang)
        triang.set_mask(max_edges > float(fiber_power_sphere_max_triangle_edge_deg))

    subdiv = int(fiber_power_sphere_triangulation_subdivisions)
    subdiv = max(0, min(subdiv, 5))

    if subdiv > 0:
        interpolator = mtri.LinearTriInterpolator(triang, values)
        refiner = mtri.UniformTriRefiner(triang)
        tri_plot = refiner.refine_triangulation(subdiv=subdiv)
        value_plot = interpolator(tri_plot.x, tri_plot.y)
        value_plot = np.asarray(value_plot.filled(np.nan), dtype=float)
    else:
        tri_plot = triang
        value_plot = values.copy()

    finite_vertex = np.isfinite(value_plot) & (value_plot > 0)
    if np.sum(finite_vertex) < 3:
        raise RuntimeError("Triangulated 3D plot has too few finite interpolated vertices.")

    x, y, z = sphere_xyz_from_phi_theta(
        tri_plot.x,
        tri_plot.y,
        fiber_power_sphere_radius_cm,
    )

    triangles = np.asarray(tri_plot.triangles, dtype=int)
    if tri_plot.mask is not None:
        triangles = triangles[~np.asarray(tri_plot.mask, dtype=bool)]

    # Remove triangles touching invalid interpolated vertices.
    tri_valid = np.all(finite_vertex[triangles], axis=1)
    triangles = triangles[tri_valid]

    if len(triangles) == 0:
        raise RuntimeError("No valid triangles remain for the 3D fiber interpolation plot.")

    triangle_values = np.nanmean(value_plot[triangles], axis=1)
    positive = triangle_values[np.isfinite(triangle_values) & (triangle_values > 0)]
    if positive.size == 0:
        raise RuntimeError("No finite positive triangle values for the 3D fiber interpolation plot.")

    if color_percentiles is None:
        color_percentiles = fiber_power_sphere_color_percentiles
    if use_log_color is None:
        use_log_color = fiber_power_sphere_use_log_color
    if colormap_name is None:
        colormap_name = fiber_power_sphere_colormap
    if surface_alpha is None:
        surface_alpha = fiber_power_sphere_surface_alpha

    vmin, vmax = np.nanpercentile(positive, color_percentiles)
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
        vmin = float(np.nanmin(positive))
        vmax = float(np.nanmax(positive))
        if vmax <= vmin:
            vmax = vmin * 1.01

    if bool(use_log_color) and vmin > 0:
        norm = colors.LogNorm(vmin=vmin, vmax=vmax)
    else:
        norm = colors.Normalize(vmin=vmin, vmax=vmax)

    cmap = cm.get_cmap(colormap_name)
    facecolors = cmap(norm(np.clip(triangle_values, vmin, vmax)))
    facecolors[:, -1] = float(surface_alpha)

    vertices = [
        [(float(x[j]), float(y[j]), float(z[j])) for j in tri]
        for tri in triangles
    ]

    collection = Poly3DCollection(
        vertices,
        facecolors=facecolors,
        edgecolors=fiber_power_sphere_surface_edgecolor,
        linewidths=float(fiber_power_sphere_surface_linewidth),
        antialiaseds=bool(fiber_power_sphere_surface_antialiased),
    )

    return collection, norm, cmap

def make_fiber_power_sphere_plot(result: dict) -> Path | None:
    """Plot measured fiber powers as a 3D surface interpolated only between fiber positions."""
    if not plot_fiber_power_sphere_3d:
        return None

    power_df = result.get("power_df", None)
    if power_df is None or len(power_df) == 0:
        return None

    power_col = str(fiber_power_sphere_power_column)
    if power_col not in power_df.columns:
        raise KeyError(
            f"Cannot make 3D power sphere: column {power_col!r} is missing from power_df. "
            f"Available columns: {list(power_df.columns)}"
        )

    config = get_fiber_power_sphere_config()
    rows = []
    for _, row in power_df.iterrows():
        fid = int(row["fiber_id"])
        if fid not in config:
            continue
        phi, theta = config[fid]
        value = float(row[power_col])
        if not np.isfinite(value) or value <= 0:
            continue
        rows.append({"fiber_id": fid, "phi": float(phi), "theta": float(theta), "power_W": value})

    if len(rows) < 3:
        raise RuntimeError("Not enough valid fiber powers with angle metadata to draw the 3D fiber-position plot.")

    sphere_df = pd.DataFrame(rows).sort_values("fiber_id").reset_index(drop=True)

    # Fiber-port Cartesian coordinates. These are the actual measured positions.
    xp, yp, zp = sphere_xyz_from_phi_theta(
        sphere_df["phi"].to_numpy(float),
        sphere_df["theta"].to_numpy(float),
        fiber_power_sphere_radius_cm,
    )

    mode = str(fiber_power_sphere_interpolation_mode).lower().strip().replace("-", "_")

    fig = plt.figure(figsize=PLOT_STYLE["figsize_sphere"], dpi=PLOT_STYLE["dpi"])
    ax = fig.add_subplot(111, projection="3d")

    if mode == "fiber_triangulation":
        # This is the requested behavior: no generated quarter/half sphere.
        # The colored surface is only the Delaunay triangulation of the measured
        # fiber-port positions, with linear interpolation inside those triangles.
        collection, norm, cmap = _build_fiber_triangulated_surface(
            phi_deg=sphere_df["phi"].to_numpy(float),
            theta_deg=sphere_df["theta"].to_numpy(float),
            values=sphere_df["power_W"].to_numpy(float),
        )
        ax.add_collection3d(collection)

    elif mode == "regular_grid":
        # Optional old/debug behavior kept only as a switch. This creates a
        # regular phi-theta patch and should normally remain disabled.
        if bool(fiber_power_sphere_plot_reconstructed_half_sphere):
            phi_min = 0.0
            phi_max = 360.0
        else:
            phi_min = float(fiber_power_sphere_phi_min_deg)
            phi_max = float(fiber_power_sphere_phi_max_deg)

        theta_min = float(fiber_power_sphere_theta_min_deg)
        theta_max = float(fiber_power_sphere_theta_max_deg)

        phi_grid_1d = np.linspace(phi_min, phi_max, int(fiber_power_sphere_phi_grid_points))
        theta_grid_1d = np.linspace(theta_min, theta_max, int(fiber_power_sphere_theta_grid_points))
        phi_grid, theta_grid = np.meshgrid(phi_grid_1d, theta_grid_1d)

        power_grid = interpolate_quantity_on_power_sphere(
            phi_points=sphere_df["phi"].to_numpy(float),
            theta_points=sphere_df["theta"].to_numpy(float),
            value_points=sphere_df["power_W"].to_numpy(float),
            phi_grid=phi_grid,
            theta_grid=theta_grid,
        )

        x, y, z = sphere_xyz_from_phi_theta(phi_grid, theta_grid, fiber_power_sphere_radius_cm)
        positive = power_grid[np.isfinite(power_grid) & (power_grid > 0)]
        if positive.size == 0:
            raise RuntimeError("3D power sphere has no finite positive values.")

        vmin, vmax = np.nanpercentile(positive, fiber_power_sphere_color_percentiles)
        if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
            vmin = float(np.nanmin(positive))
            vmax = float(np.nanmax(positive))
            if vmax <= vmin:
                vmax = vmin * 1.01

        if fiber_power_sphere_use_log_color and vmin > 0:
            norm = colors.LogNorm(vmin=vmin, vmax=vmax)
        else:
            norm = colors.Normalize(vmin=vmin, vmax=vmax)

        cmap = cm.get_cmap(fiber_power_sphere_colormap)
        facecolors = cmap(norm(np.clip(power_grid, vmin, vmax)))
        facecolors[..., -1] = float(fiber_power_sphere_surface_alpha)
        ax.plot_surface(
            x,
            y,
            z,
            facecolors=facecolors,
            rstride=1,
            cstride=1,
            linewidth=float(fiber_power_sphere_surface_linewidth),
            edgecolor=fiber_power_sphere_surface_edgecolor,
            antialiased=bool(fiber_power_sphere_surface_antialiased),
            shade=False,
        )
    else:
        raise ValueError(
            "fiber_power_sphere_interpolation_mode must be 'fiber_triangulation' or 'regular_grid'."
        )

    if fiber_power_sphere_show_ports:
        ax.scatter(
            xp,
            yp,
            zp,
            c=sphere_df["power_W"].to_numpy(float),
            cmap=cmap,
            norm=norm,
            s=float(fiber_power_sphere_port_marker_size),
            edgecolors="black",
            linewidths=0.35,
            depthshade=False,
        )

    if fiber_power_sphere_show_port_labels:
        for _, row in sphere_df.iterrows():
            xx, yy, zz = sphere_xyz_from_phi_theta(row["phi"], row["theta"], fiber_power_sphere_radius_cm)
            ax.text(xx, yy, zz, f"{int(row['fiber_id'])}", fontsize=fiber_power_sphere_port_label_size)

    if fiber_power_sphere_draw_laser_axis:
        r = float(fiber_power_sphere_radius_cm)
        ax.plot(
            [r, 0.0],
            [0.0, 0.0],
            [0.0, 0.0],
            color=fiber_power_sphere_laser_axis_color,
            linewidth=float(fiber_power_sphere_laser_axis_linewidth),
            label=fiber_power_sphere_laser_axis_label,
        )
        add_paper_legend(ax, loc="upper left")

    ax.set_title(PLOT_STYLE["sphere_title"], pad=PLOT_STYLE["title_pad"])
    ax.set_xlabel(PLOT_STYLE["sphere_x_label"], labelpad=PLOT_STYLE["label_pad"])
    ax.set_ylabel(PLOT_STYLE["sphere_y_label"], labelpad=PLOT_STYLE["label_pad"])
    ax.set_zlabel(PLOT_STYLE["sphere_z_label"], labelpad=PLOT_STYLE["label_pad"])
    ax.view_init(elev=float(fiber_power_sphere_view_elev), azim=float(fiber_power_sphere_view_azim))
    set_3d_axes_equal(ax, fiber_power_sphere_radius_cm)
    clean_3d_axis_grid(ax)

    mappable = cm.ScalarMappable(norm=norm, cmap=cmap)
    mappable.set_array([])
    cbar = fig.colorbar(mappable, ax=ax, pad=0.02, fraction=0.04, shrink=0.80, aspect=24)
    cbar.set_label(PLOT_STYLE["colorbar_power_w"], fontsize=PLOT_STYLE["colorbar_label_size"])
    cbar.ax.tick_params(labelsize=PLOT_STYLE["colorbar_tick_size"])

    save_paper_figure(fig, output_fiber_power_sphere_png)
    return output_fiber_power_sphere_png



def make_fiber_counts_sphere_plot(result: dict) -> Path | None:
    """Plot raw uncalibrated IR-LED calibration-image counts on the 3D fiber surface.

    This uses exactly the same geometry/interpolation logic as the power sphere
    plot, but the color is the wavelength-integrated camera counts measured for
    each fiber in the calibration image, before any absolute power calibration.
    """
    if not plot_fiber_counts_sphere_3d:
        return None

    source = str(fiber_counts_sphere_counts_source).lower().strip().replace("-", "_")
    if source in {"raw", "raw_signal", "raw_signal_image", "raw_image"}:
        counts_matrix = np.asarray(result.get("integrated_counts_raw_signal"), dtype=float)
        title_suffix = "raw image"
    elif source in {"background_subtracted", "background_subtracted_image", "bg_subtracted", "processed"}:
        counts_matrix = np.asarray(result.get("integrated_counts"), dtype=float)
        title_suffix = "background subtracted"
    else:
        raise ValueError(
            "fiber_counts_sphere_counts_source must be 'raw_signal_image' or "
            "'background_subtracted_image'."
        )

    if counts_matrix.ndim != 2:
        raise RuntimeError("Cannot make raw-counts sphere: integrated count array is not 2D.")

    counts_total = np.nansum(counts_matrix, axis=1)

    config = get_fiber_power_sphere_config()
    rows = []
    for fid, value in enumerate(counts_total):
        if fid not in config:
            continue
        phi, theta = config[fid]
        value = float(value)
        if not np.isfinite(value) or value <= 0:
            continue
        rows.append({"fiber_id": int(fid), "phi": float(phi), "theta": float(theta), "counts": value})

    if len(rows) < 3:
        raise RuntimeError("Not enough valid raw fiber counts with angle metadata to draw the 3D count plot.")

    sphere_df = pd.DataFrame(rows).sort_values("fiber_id").reset_index(drop=True)

    xp, yp, zp = sphere_xyz_from_phi_theta(
        sphere_df["phi"].to_numpy(float),
        sphere_df["theta"].to_numpy(float),
        fiber_power_sphere_radius_cm,
    )

    mode = str(fiber_power_sphere_interpolation_mode).lower().strip().replace("-", "_")

    fig = plt.figure(figsize=PLOT_STYLE["figsize_sphere"], dpi=PLOT_STYLE["dpi"])
    ax = fig.add_subplot(111, projection="3d")

    if mode == "fiber_triangulation":
        collection, norm, cmap = _build_fiber_triangulated_surface(
            phi_deg=sphere_df["phi"].to_numpy(float),
            theta_deg=sphere_df["theta"].to_numpy(float),
            values=sphere_df["counts"].to_numpy(float),
            use_log_color=fiber_counts_sphere_use_log_color,
            color_percentiles=fiber_counts_sphere_color_percentiles,
            colormap_name=fiber_counts_sphere_colormap,
            surface_alpha=fiber_counts_sphere_surface_alpha,
        )
        ax.add_collection3d(collection)

    elif mode == "regular_grid":
        if bool(fiber_power_sphere_plot_reconstructed_half_sphere):
            phi_min = 0.0
            phi_max = 360.0
        else:
            phi_min = float(fiber_power_sphere_phi_min_deg)
            phi_max = float(fiber_power_sphere_phi_max_deg)

        theta_min = float(fiber_power_sphere_theta_min_deg)
        theta_max = float(fiber_power_sphere_theta_max_deg)

        phi_grid_1d = np.linspace(phi_min, phi_max, int(fiber_power_sphere_phi_grid_points))
        theta_grid_1d = np.linspace(theta_min, theta_max, int(fiber_power_sphere_theta_grid_points))
        phi_grid, theta_grid = np.meshgrid(phi_grid_1d, theta_grid_1d)

        counts_grid = interpolate_quantity_on_power_sphere(
            phi_points=sphere_df["phi"].to_numpy(float),
            theta_points=sphere_df["theta"].to_numpy(float),
            value_points=sphere_df["counts"].to_numpy(float),
            phi_grid=phi_grid,
            theta_grid=theta_grid,
        )

        x, y, z = sphere_xyz_from_phi_theta(phi_grid, theta_grid, fiber_power_sphere_radius_cm)
        positive = counts_grid[np.isfinite(counts_grid) & (counts_grid > 0)]
        if positive.size == 0:
            raise RuntimeError("3D raw-counts sphere has no finite positive values.")

        vmin, vmax = np.nanpercentile(positive, fiber_counts_sphere_color_percentiles)
        if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
            vmin = float(np.nanmin(positive))
            vmax = float(np.nanmax(positive))
            if vmax <= vmin:
                vmax = vmin * 1.01

        if bool(fiber_counts_sphere_use_log_color) and vmin > 0:
            norm = colors.LogNorm(vmin=vmin, vmax=vmax)
        else:
            norm = colors.Normalize(vmin=vmin, vmax=vmax)

        cmap = cm.get_cmap(fiber_counts_sphere_colormap)
        facecolors = cmap(norm(np.clip(counts_grid, vmin, vmax)))
        facecolors[..., -1] = float(fiber_counts_sphere_surface_alpha)
        ax.plot_surface(
            x,
            y,
            z,
            facecolors=facecolors,
            rstride=1,
            cstride=1,
            linewidth=float(fiber_power_sphere_surface_linewidth),
            edgecolor=fiber_power_sphere_surface_edgecolor,
            antialiased=bool(fiber_power_sphere_surface_antialiased),
            shade=False,
        )
    else:
        raise ValueError(
            "fiber_power_sphere_interpolation_mode must be 'fiber_triangulation' or 'regular_grid'."
        )

    if fiber_counts_sphere_show_ports:
        ax.scatter(
            xp,
            yp,
            zp,
            c=sphere_df["counts"].to_numpy(float),
            cmap=cmap,
            norm=norm,
            s=float(fiber_counts_sphere_port_marker_size),
            edgecolors="black",
            linewidths=0.35,
            depthshade=False,
        )

    if fiber_counts_sphere_show_port_labels:
        for _, row in sphere_df.iterrows():
            xx, yy, zz = sphere_xyz_from_phi_theta(row["phi"], row["theta"], fiber_power_sphere_radius_cm)
            ax.text(xx, yy, zz, f"{int(row['fiber_id'])}", fontsize=fiber_counts_sphere_port_label_size)

    if fiber_power_sphere_draw_laser_axis:
        r = float(fiber_power_sphere_radius_cm)
        ax.plot(
            [r, 0.0],
            [0.0, 0.0],
            [0.0, 0.0],
            color=fiber_power_sphere_laser_axis_color,
            linewidth=float(fiber_power_sphere_laser_axis_linewidth),
            label=fiber_power_sphere_laser_axis_label,
        )
        add_paper_legend(ax, loc="upper left")

    ax.set_title(f"{PLOT_STYLE['sphere_counts_title']} ({title_suffix})", pad=PLOT_STYLE["title_pad"])
    ax.set_xlabel(PLOT_STYLE["sphere_x_label"], labelpad=PLOT_STYLE["label_pad"])
    ax.set_ylabel(PLOT_STYLE["sphere_y_label"], labelpad=PLOT_STYLE["label_pad"])
    ax.set_zlabel(PLOT_STYLE["sphere_z_label"], labelpad=PLOT_STYLE["label_pad"])
    ax.view_init(elev=float(fiber_power_sphere_view_elev), azim=float(fiber_power_sphere_view_azim))
    set_3d_axes_equal(ax, fiber_power_sphere_radius_cm)
    clean_3d_axis_grid(ax)

    mappable = cm.ScalarMappable(norm=norm, cmap=cmap)
    mappable.set_array([])
    cbar = fig.colorbar(mappable, ax=ax, pad=0.02, fraction=0.04, shrink=0.80, aspect=24)
    cbar.set_label(PLOT_STYLE["colorbar_camera_counts"], fontsize=PLOT_STYLE["colorbar_label_size"])
    cbar.ax.tick_params(labelsize=PLOT_STYLE["colorbar_tick_size"])

    save_paper_figure(fig, output_fiber_counts_sphere_png)
    return output_fiber_counts_sphere_png


def _image_display_limits(img: np.ndarray, percentiles: list[float] | tuple[float, float]) -> tuple[float, float]:
    """Return robust image display limits from finite pixels."""
    arr = np.asarray(img, dtype=float)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return 0.0, 1.0
    p0, p1 = percentiles
    vmin, vmax = np.nanpercentile(finite, [float(p0), float(p1)])
    if not np.isfinite(vmin):
        vmin = float(np.nanmin(finite))
    if not np.isfinite(vmax):
        vmax = float(np.nanmax(finite))
    if vmax <= vmin:
        vmax = vmin + 1.0
    return float(vmin), float(vmax)


def make_calibration_image_fiber_overlay_plot(result: dict) -> Path | None:
    """Plot the calibration image with fiber center traces and integration regions.

    The center traces come from the same reference CSV used for extraction. The
    two integration boundaries are recomputed with integration_width_calibration,
    so this diagnostic shows exactly what parts of the image contribute to each
    extracted fiber spectrum.
    """
    if not plot_calibration_image_with_fiber_overlay:
        return None

    source = str(calibration_image_overlay_source).lower().strip().replace("-", "_")
    if source in {"raw", "raw_signal", "raw_signal_image", "raw_image"}:
        img = np.asarray(result.get("raw_signal_img"), dtype=float)
        title_suffix = "raw image"
    elif source in {"background_subtracted", "background_subtracted_image", "bg_subtracted", "processed"}:
        img = np.asarray(result.get("img"), dtype=float)
        title_suffix = "background subtracted"
    else:
        raise ValueError(
            "calibration_image_overlay_source must be 'raw_signal_image' or "
            "'background_subtracted_image'."
        )

    if img.ndim != 2:
        raise RuntimeError("Calibration image overlay requires a 2D image array.")

    ref_df = result["ref_df"].sort_values("fiber_id").reset_index(drop=True)
    ny, nx = img.shape
    step = max(1, int(calibration_image_overlay_x_step_px))
    x_plot = np.arange(0, nx, step, dtype=int)
    if x_plot[-1] != nx - 1:
        x_plot = np.append(x_plot, nx - 1)

    y_centers, y_top, y_bottom = compute_integration_boundaries(
        ref_df=ref_df,
        x_sample=x_plot.astype(float),
        ny=ny,
        integration_width=float(integration_width_calibration),
    )

    display_img = img.copy()
    if calibration_image_overlay_use_log_image:
        finite = display_img[np.isfinite(display_img)]
        offset = 0.0
        if finite.size and np.nanmin(finite) <= 0:
            offset = -float(np.nanmin(finite)) + 1.0
        display_img = np.log10(np.clip(display_img + offset, 1e-12, None))

    vmin, vmax = _image_display_limits(display_img, calibration_image_overlay_percentiles)

    fig, ax = plt.subplots(
        figsize=PLOT_STYLE["figsize_image_overlay"],
        dpi=PLOT_STYLE["dpi"],
    )
    ax.imshow(
        display_img,
        origin="upper",
        cmap=calibration_image_overlay_cmap,
        vmin=vmin,
        vmax=vmax,
        interpolation="nearest",
        aspect="auto",
    )

    n_fibers = len(ref_df)
    for i, row in ref_df.iterrows():
        fiber_id = int(row["fiber_id"])
        color = color_for_fiber(fiber_id, n_fibers=max(n_fibers, expected_number_of_fibers))

        y0 = y_top[i, :]
        y1 = y_bottom[i, :]
        yc = y_centers[i, :]

        if calibration_image_overlay_show_integration_fill:
            ax.fill_between(
                x_plot,
                y0,
                y1,
                color=color,
                alpha=float(calibration_image_overlay_fill_alpha),
                linewidth=0.0,
                zorder=2,
            )

        if calibration_image_overlay_show_integration_boundaries:
            ax.plot(
                x_plot,
                y0,
                color=color,
                linewidth=float(calibration_image_overlay_boundary_linewidth),
                linestyle=calibration_image_overlay_boundary_linestyle,
                alpha=float(calibration_image_overlay_boundary_alpha),
                zorder=3,
            )
            ax.plot(
                x_plot,
                y1,
                color=color,
                linewidth=float(calibration_image_overlay_boundary_linewidth),
                linestyle=calibration_image_overlay_boundary_linestyle,
                alpha=float(calibration_image_overlay_boundary_alpha),
                zorder=3,
            )

        if calibration_image_overlay_show_center_traces:
            ax.plot(
                x_plot,
                yc,
                color=color,
                linewidth=float(calibration_image_overlay_center_linewidth),
                alpha=float(calibration_image_overlay_center_alpha),
                zorder=4,
            )

        label_every = calibration_image_overlay_label_every_n_fibers
        if label_every is not None and int(label_every) > 0 and (fiber_id % int(label_every) == 0):
            x_label = int(np.clip(float(calibration_image_overlay_label_x_fraction) * (nx - 1), 0, nx - 1))
            y_label = float(np.interp(x_label, x_plot.astype(float), yc))
            if np.isfinite(y_label):
                ax.text(
                    x_label,
                    y_label,
                    str(fiber_id),
                    color=calibration_image_overlay_label_color,
                    fontsize=float(calibration_image_overlay_label_fontsize),
                    ha="right",
                    va="center",
                    zorder=5,
                    bbox=dict(facecolor="black", edgecolor="none", alpha=0.35, pad=0.8),
                )

    setup_paper_axis(
        ax=ax,
        title=f"{PLOT_STYLE['image_overlay_title']} ({title_suffix})",
        xlabel=PLOT_STYLE["x_label_pixel"],
        ylabel=PLOT_STYLE["y_label_pixel"],
        xlim=(0, nx - 1),
        ylim=(ny - 1, 0),
    )
    save_paper_figure(fig, output_calibration_image_overlay_png)
    return output_calibration_image_overlay_png


def edges_from_centers(x):
    """Return bin edges for a 1D array of bin centers, for pcolormesh."""
    x = np.asarray(x, dtype=float)

    if x.ndim != 1 or x.size < 2:
        raise ValueError("edges_from_centers needs at least two finite 1D centers.")

    if not np.all(np.isfinite(x)):
        raise ValueError("Cannot compute edges: center array contains non-finite values.")

    mid = 0.5 * (x[:-1] + x[1:])
    first = x[0] - 0.5 * (x[1] - x[0])
    last = x[-1] + 0.5 * (x[-1] - x[-2])
    return np.concatenate([[first], mid, [last]])




def smoothing_description_text() -> str:
    """Human-readable description for titles."""
    method = _smoothing_method_name()
    if method == "none":
        base = "raw"
    else:
        base = f"{method}-smoothed"
    if use_common_average_curve_for_all_fibers:
        base += ", common average for all fibers"
    elif replace_weird_fiber_curves_with_average and len(weird_fiber_ids) > 0:
        base += ", weird fibers replaced"
    return base


def make_all_fiber_raw_and_smoothed_curve_grid_plots(result: dict) -> list[Path]:
    """Save four 5x4 figures showing raw and processed curves for all fibers."""
    if not plot_all_fiber_raw_and_smoothed_curve_grids:
        return []

    wl = np.asarray(result["wavelengths_nm"], dtype=float)
    cal = np.asarray(result["calibration_J_per_nm_per_count"], dtype=float)
    cal_raw = np.asarray(result.get("calibration_J_per_nm_per_count_raw", np.full_like(cal, np.nan)), dtype=float)
    replaced_mask = np.asarray(
        result.get("weird_fiber_replaced_mask", np.zeros(cal.shape[0], dtype=bool)),
        dtype=bool,
    )
    common_mask = np.asarray(
        result.get("common_average_curve_mask", np.zeros(cal.shape[0], dtype=bool)),
        dtype=bool,
    )

    mask = np.isfinite(wl) & (wl >= plot_wavelength_min_nm) & (wl <= plot_wavelength_max_nm)
    if np.sum(mask) < 2:
        raise RuntimeError("Not enough wavelength points in plot range for all-fiber curve grids.")

    n_fibers = cal.shape[0]
    per_fig = int(all_fiber_curve_grid_ncols) * int(all_fiber_curve_grid_nrows)
    saved_paths = []

    for start in range(0, n_fibers, per_fig):
        end = min(start + per_fig, n_fibers)
        fig, axes = plt.subplots(
            int(all_fiber_curve_grid_nrows),
            int(all_fiber_curve_grid_ncols),
            figsize=PLOT_STYLE["figsize_all_fiber_grid"],
            dpi=PLOT_STYLE["dpi"],
            sharex=True,
            sharey=True,
        )
        axes = np.asarray(axes).ravel()

        for ax_index, ax in enumerate(axes):
            fiber_id = start + ax_index
            if fiber_id >= end:
                ax.axis("off")
                continue

            fiber_color = color_for_fiber(fiber_id, n_fibers=n_fibers)
            has_raw = np.any(np.isfinite(cal_raw[fiber_id, mask]) & (cal_raw[fiber_id, mask] > 0))
            has_smooth = np.any(np.isfinite(cal[fiber_id, mask]) & (cal[fiber_id, mask] > 0))

            if has_raw:
                ax.plot(
                    wl[mask],
                    cal_raw[fiber_id, mask],
                    color=PLOT_STYLE["raw_curve_color"],
                    linestyle=PLOT_STYLE["raw_linestyle"],
                    linewidth=float(PLOT_STYLE["raw_linewidth"]),
                    alpha=float(PLOT_STYLE["raw_alpha"]),
                    label="raw" if ax_index == 0 else None,
                )

            if has_smooth:
                ax.plot(
                    wl[mask],
                    cal[fiber_id, mask],
                    color=fiber_color,
                    linestyle=PLOT_STYLE["smooth_linestyle"],
                    linewidth=float(PLOT_STYLE["smooth_linewidth"]),
                    alpha=float(PLOT_STYLE["curve_alpha"]),
                    label="processed" if ax_index == 0 else None,
                )

            title = f"F{fiber_id}"
            if bool(common_mask[fiber_id]):
                title += " common"
            elif bool(replaced_mask[fiber_id]) and all_fiber_curve_show_replaced_label:
                title += " avg"
            setup_paper_axis(
                ax,
                title=title,
                xlabel=None,
                ylabel=None,
                yscale=all_fiber_curve_yscale,
                xlim=(plot_wavelength_min_nm, plot_wavelength_max_nm),
            )
            ax.title.set_fontsize(PLOT_STYLE["subplot_title_size"])

        fig.supxlabel(PLOT_STYLE["x_label_wavelength"], fontsize=PLOT_STYLE["label_size"])
        fig.supylabel(PLOT_STYLE["y_label_calibration"], fontsize=PLOT_STYLE["label_size"])
        fig.suptitle(
            f"{PLOT_STYLE['all_fiber_grid_title']}: fibers {start}-{end - 1} "
            "(raw dotted, processed solid)",
            fontsize=PLOT_STYLE["title_size"],
            fontweight=PLOT_STYLE["title_weight"],
        )

        handles, labels = axes[0].get_legend_handles_labels()
        if handles:
            fig.legend(
                handles,
                labels,
                loc="upper right",
                fontsize=PLOT_STYLE["legend_size"],
                frameon=PLOT_STYLE["legend_frame"],
                framealpha=PLOT_STYLE["legend_frame_alpha"],
            )

        fig.tight_layout(rect=[0.02, 0.02, 0.985, 0.94])
        out_path = CALIBRATION_FOLDER / output_all_fiber_curve_plot_template.format(
            start=int(start),
            end=int(end - 1),
        )
        fig.savefig(
            out_path,
            dpi=PLOT_STYLE["save_dpi"],
            transparent=PLOT_STYLE["transparent"],
            bbox_inches=PLOT_STYLE["bbox_inches"],
        )
        saved_paths.append(out_path)

    return saved_paths


def make_plots(result: dict) -> None:
    apply_scientific_article_style()

    wl = result["wavelengths_nm"]
    cal = result["calibration_J_per_nm_per_count"]
    cal_raw = result.get("calibration_J_per_nm_per_count_raw", None)
    led_lp = result["led_after_lp_density_per_nm"]
    power_df = result["power_df"]

    mask = np.isfinite(wl) & (wl >= plot_wavelength_min_nm) & (wl <= plot_wavelength_max_nm)
    if np.sum(mask) < 2:
        raise RuntimeError("Not enough wavelength points in the requested plot range.")

    # ------------------------------------------------------------
    # Selected calibration curves + LED spectrum.
    # ------------------------------------------------------------
    fig, ax1 = plt.subplots(figsize=PLOT_STYLE["figsize_summary"], dpi=PLOT_STYLE["dpi"])
    n_fibers = cal.shape[0]
    for fiber_id in selected_fibers_to_plot:
        if 0 <= fiber_id < n_fibers:
            if plot_raw_calibration_curves_under_spline and cal_raw is not None:
                ax1.plot(
                    wl[mask],
                    cal_raw[fiber_id, mask],
                    linewidth=PLOT_STYLE["raw_linewidth"],
                    linestyle=PLOT_STYLE["raw_linestyle"],
                    alpha=PLOT_STYLE["raw_alpha"],
                    color=PLOT_STYLE["raw_curve_color"],
                )
            ax1.plot(
                wl[mask],
                cal[fiber_id, mask],
                linewidth=PLOT_STYLE["curve_linewidth"],
                alpha=PLOT_STYLE["curve_alpha"],
                color=color_for_fiber(fiber_id, n_fibers=n_fibers),
                label=f"fiber {fiber_id}",
            )

    setup_paper_axis(
        ax=ax1,
        title=f"{PLOT_STYLE['summary_title']} ({smoothing_description_text()})",
        xlabel=PLOT_STYLE["x_label_wavelength"],
        ylabel=PLOT_STYLE["y_label_calibration"],
        yscale="log",
        xlim=(plot_wavelength_min_nm, plot_wavelength_max_nm),
    )

    ax2 = ax1.twinx()
    if np.nanmax(led_lp[mask]) > 0:
        ax2.plot(
            wl[mask],
            led_lp[mask] / np.nanmax(led_lp[mask]),
            color=PLOT_STYLE["led_curve_color"],
            linestyle=PLOT_STYLE["led_curve_linestyle"],
            linewidth=1.25,
            label="LED after LP",
        )
    ax2.set_ylabel(PLOT_STYLE["y_label_led"], fontsize=PLOT_STYLE["label_size"])
    ax2.tick_params(
        direction=PLOT_STYLE["tick_direction"],
        which="both",
        top=bool(PLOT_STYLE["ticks_top"]),
        right=bool(PLOT_STYLE["ticks_right"]),
        labelsize=PLOT_STYLE["tick_size"],
    )
    ax2.yaxis.set_minor_locator(AutoMinorLocator())

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    if lines1 or lines2:
        legend = ax1.legend(
            lines1 + lines2,
            labels1 + labels2,
            fontsize=PLOT_STYLE["legend_size"],
            frameon=PLOT_STYLE["legend_frame"],
            framealpha=PLOT_STYLE["legend_frame_alpha"],
            loc=PLOT_STYLE["legend_location"],
        )
        if legend is not None:
            legend.get_frame().set_edgecolor(PLOT_STYLE["legend_edge_color"])
            legend.get_frame().set_facecolor(PLOT_STYLE["legend_face_color"])
            legend.get_frame().set_linewidth(PLOT_STYLE["legend_linewidth"])

    save_paper_figure(fig, output_summary_plot_png)

    # ------------------------------------------------------------
    # Map of calibration coefficient.
    # ------------------------------------------------------------
    fig, ax = plt.subplots(figsize=PLOT_STYLE["figsize_map"], dpi=PLOT_STYLE["dpi"])
    plot_data = np.ma.masked_invalid(cal[:, mask])
    positive = plot_data.compressed()
    positive = positive[positive > 0]
    if positive.size > 0:
        vmin, vmax = np.nanpercentile(positive, [2, 98])
    else:
        vmin, vmax = None, None
    x_edges = edges_from_centers(wl[mask])
    y_edges = np.arange(cal.shape[0] + 1, dtype=float) - 0.5
    im = ax.pcolormesh(
        x_edges,
        y_edges,
        plot_data,
        shading="flat",
        cmap=PLOT_STYLE["map_colormap"],
        vmin=vmin,
        vmax=vmax,
    )
    add_paper_colorbar(fig, ax, im, PLOT_STYLE["colorbar_calibration"])
    setup_paper_axis(
        ax=ax,
        title=f"{PLOT_STYLE['map_title']} ({smoothing_description_text()})",
        xlabel=PLOT_STYLE["x_label_wavelength"],
        ylabel=PLOT_STYLE["x_label_fiber_id"],
        xlim=(plot_wavelength_min_nm, plot_wavelength_max_nm),
    )
    save_paper_figure(fig, output_map_plot_png)

    # ------------------------------------------------------------
    # Power summary plot: fiber-index colored curve.
    # ------------------------------------------------------------
    fig, ax = plt.subplots(figsize=PLOT_STYLE["figsize_power"], dpi=PLOT_STYLE["dpi"])
    fiber_ids = power_df["fiber_id"].to_numpy(int)
    power_W = power_df["power_entering_fiber_total_W"].to_numpy(float)
    for i0, i1 in zip(range(len(fiber_ids) - 1), range(1, len(fiber_ids))):
        ax.plot(
            fiber_ids[[i0, i1]],
            power_W[[i0, i1]],
            color=color_for_fiber(fiber_ids[i0], n_fibers=n_fibers),
            linewidth=PLOT_STYLE["power_linewidth"],
            alpha=PLOT_STYLE["curve_alpha"],
        )
    ax.scatter(
        fiber_ids,
        power_W,
        c=fiber_ids,
        cmap=cm.get_cmap(PLOT_STYLE["fiber_line_colormap"]),
        norm=colors.Normalize(PLOT_STYLE["fiber_line_color_min"], PLOT_STYLE["fiber_line_color_max"]),
        s=PLOT_STYLE["marker_size"] ** 2,
        edgecolors="black",
        linewidths=0.25,
        zorder=3,
    )
    setup_paper_axis(
        ax=ax,
        title=PLOT_STYLE["power_title"],
        xlabel=PLOT_STYLE["x_label_fiber_id"],
        ylabel=PLOT_STYLE["y_label_power_w"],
        yscale="log" if np.all(power_W[np.isfinite(power_W)] > 0) else "linear",
    )
    save_paper_figure(fig, output_fiber_power_vs_fiber_png)

    # ------------------------------------------------------------
    # 3D power sphere matching the main plotting-code geometry.
    # ------------------------------------------------------------
    sphere_plot_path = make_fiber_power_sphere_plot(result)

    # ------------------------------------------------------------
    # Second 3D sphere: raw uncalibrated IR-LED camera counts.
    # ------------------------------------------------------------
    counts_sphere_plot_path = make_fiber_counts_sphere_plot(result)

    # ------------------------------------------------------------
    # 2D raw-image overlay: fiber traces + integration regions.
    # ------------------------------------------------------------
    image_overlay_plot_path = make_calibration_image_fiber_overlay_plot(result)

    all_fiber_curve_paths = make_all_fiber_raw_and_smoothed_curve_grid_plots(result)

    plt.show()

    print("Saved plots:")
    print(f"  {output_summary_plot_png.resolve()}")
    print(f"  {output_map_plot_png.resolve()}")
    print(f"  {output_fiber_power_vs_fiber_png.resolve()}")
    if sphere_plot_path is not None:
        print(f"  {sphere_plot_path.resolve()}")
    if counts_sphere_plot_path is not None:
        print(f"  {counts_sphere_plot_path.resolve()}")
    if image_overlay_plot_path is not None:
        print(f"  {image_overlay_plot_path.resolve()}")
    for path in all_fiber_curve_paths:
        print(f"  {path.resolve()}")


def main() -> None:
    result = build_absolute_calibration()
    save_outputs(result)
    make_plots(result)

    print("\nKey factors:")
    print(
        "  Power-meter spectral correction factor = "
        f"{result['pm_correction']['powermeter_true_power_correction_factor']:.6g}"
    )
    print(
        "  Longpass fraction of LED power on camera axis = "
        f"{result['longpass_fraction']:.6g}"
    )
    print(
        "  Median power entering fiber after longpass = "
        f"{1e9 * np.nanmedian(result['power_df']['power_entering_fiber_after_longpass_W']):.6g} nW"
    )


if __name__ == "__main__":
    main()
