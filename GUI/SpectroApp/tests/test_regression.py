"""
tests/test_regression.py — Non-regression numerique.

Compares the application outputs (via core.analysis, i.e. the REAL code
path of the app) to reference values obtained by running the notebook
pipeline on the sample data. If this test passes, the application
reproduces the notebook identically on that data.

Usage:  python -m tests.test_regression <data_folder>
The folder must contain the HgAr image, shot431.tif, Final.xlsx, shot_498.csv.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REF = {
    "wl_first": 637.5673036994363,
    "wl_last": 958.1586241090433,
    "wl_rms": 0.17909975167474407,
    "n_wl_pairs": 18,
    "shot431_angle": 0.9198795317549184,
    "shot431_area_f62": 6646797.159530856,
    "shot431_area_mean": 3054903.8681229777,
    "shot431_centroid_f62": 781.2014661731632,
    "e431": 627.93594,
    "p498": 283385676388.7677,
}


def main(data_dir: str):
    data = Path(data_dir)
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from core import analysis
    from core import spectro_functions as sf
    from core.session import SESSION

    hgar = data / "251119_HgArlines_noND_aqc5s_WLcent800_150grating.tif"
    assert hgar.exists(), f"HgAr image missing: {hgar}"
    SESSION.images_dir = str(data)
    SESSION.hgar_path = str(hgar)
    SESSION.excel_path = str(data / "Final.xlsx")
    SESSION.pulses_dir = str(data)
    SESSION.workspace = str(data / "_test_workspace")
    SESSION.scan_images()

    ok, msg = analysis.run_calibration()
    assert ok, msg
    c = SESSION.calib
    checks = []

    def check(name, got, ref, rtol=1e-9, atol=1e-9):
        good = np.isclose(got, ref, rtol=rtol, atol=atol)
        checks.append((name, got, ref, bool(good)))
        return good

    check("wl_first", float(c["wl_axis"][0]), REF["wl_first"])
    check("wl_last", float(c["wl_axis"][-1]), REF["wl_last"])
    check("wl_rms", float(np.sqrt(np.mean(c["wl_residuals"] ** 2))),
          REF["wl_rms"])
    check("n_wl_pairs", len(c["wl_pairs"]), REF["n_wl_pairs"])

    sp = analysis.get_spectra("shot431", use_cache=False)
    areas = sf.compute_spectral_area(sp, c["wl_axis"])
    cents = sf.compute_spectral_centroid(sp, c["wl_axis"], wl_range=(725, 875),
                                         bg_percentile=10, savgol_window_nm=5.0)
    check("shot431_area_f62", float(areas[62]), REF["shot431_area_f62"],
          rtol=1e-10)
    check("shot431_area_mean", float(np.nanmean(areas)),
          REF["shot431_area_mean"], rtol=1e-10)
    check("shot431_centroid_f62", float(cents[62]),
          REF["shot431_centroid_f62"], rtol=1e-10)

    ok, msg = analysis.load_energy()
    assert ok, msg
    check("e431", SESSION.energy_table[431], REF["e431"])
    P, err = analysis.shot_power("shot498")
    assert err is None, err
    check("p498", P, REF["p498"], rtol=1e-10)

    # Angles : reconstruction Excel vs valeurs codees du pipeline (si fichiers presents)
    struct_f = data / "251102_EstimateStructureCoordinates.xlsx"
    if struct_f.exists():
        from core import angles as ang
        struct = ang.load_structure_tables(struct_f)
        for cfg_name in ("Config3_d", "Config3_e", "Config3_f"):
            pos_f = data / f"SidescatterFibrePos_{cfg_name}.xlsx"
            if not pos_f.exists():
                continue
            fibres, _iss = ang.build_config_from_files(struct, pos_f)
            ref = sf.FIBER_CONFIGS[cfg_name.lower()]["fibres"]
            same_keys = set(ref) == set(fibres)
            dmax = max(max(abs(ref[k][0] - fibres[k][0]),
                           abs(ref[k][1] - fibres[k][1]))
                       for k in ref) if same_keys else 999.0
            check(f"angles_{cfg_name}", dmax if same_keys else 999.0, 0.0,
                  atol=1e-9)

    # Detection automatique des fibres vs positions manuelles du pipeline
    # (tolerances : le pointage manuel a lui-meme ~0.6 px de bruit ; les
    #  fibres quasi invisibles peuvent differer de 3-4 px, information limite)
    from core import analysis as an
    SESSION.params["FIBER_MODE"] = "manual"   # ne pas polluer les tests exacts
    ok, msg = an.run_fiber_detection(5)
    assert ok, msg
    pos = np.sort(np.array(SESSION.fiber_auto["positions"], float))
    ref_pos = an._FIBER_Y_PIPELINE
    d = pos - ref_pos
    check("fibres_n", len(pos), 80)
    check("fibres_rms_px", float(np.sqrt(np.mean(d ** 2))), 0.0, atol=1.6)
    check("fibres_max_px", float(np.abs(d).max()), 0.0, atol=4.5)
    check("fibres_ok2px", int((np.abs(d) <= 2).sum()), 80, atol=8)

    print(f"{'test':<24}{'obtenu':>22}{'reference':>22}  statut")
    print("-" * 78)
    n_bad = 0
    for name, got, ref, good in checks:
        print(f"{name:<24}{got:>22.10g}{ref:>22.10g}  "
              f"{'OK' if good else 'MISMATCH!'}")
        n_bad += 0 if good else 1
    print("-" * 78)
    if n_bad:
        print(f"FAILURE: {n_bad} mismatch(es) — the application does not "
              f"reproduce the pipeline.")
        sys.exit(1)
    print("SUCCESS: the application reproduces the notebook pipeline "
          "identically on that data.")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "../data")
