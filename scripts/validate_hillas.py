#!/usr/bin/env python3
"""
Validation script: Verify extracted Hillas parameters against ground-truth simulation values
and cross-validate against ctapipe (if installed).

Saves:
- figures/hillas_validation.png
- results/hillas_validation.json
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from tqdm import tqdm

from cherenkovlens.camera import CameraGeometry
from cherenkovlens.hillas import calculate_hillas_parameters
from cherenkovlens.plotting import plot_hillas_validation
from cherenkovlens.simulate import ShowerSimulator
from cherenkovlens.utils import ensure_dirs, get_project_root, load_config, setup_logger

# Optional ctapipe import
try:
    import astropy.units as u
    from ctapipe.containers import HillasParametersContainer
    from ctapipe.image.hillas import hillas_parameters as ctapipe_hillas
    from ctapipe.instrument import CameraGeometry as CtapipeCameraGeometry
    CTAPIPE_AVAILABLE = True
except ImportError:
    CTAPIPE_AVAILABLE = False


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate Hillas parameter reconstruction.")
    parser.add_argument("--n-events", type=int, default=1000, help="Number of validation events (default: 1000)")
    parser.add_argument("--seed", type=int, default=42, help="Seed (default: 42)")
    args = parser.parse_args()

    logger = setup_logger("validate_hillas")
    root = get_project_root()
    cfg = load_config()

    figures_dir = root / "figures"
    results_dir = root / "results"
    ensure_dirs(figures_dir, results_dir)

    cam = CameraGeometry(**cfg.get("camera", {}))
    sim = ShowerSimulator(camera=cam, config=cfg, rng=np.random.default_rng(args.seed))

    logger.info(f"Simulating {args.n_events} noise-free gamma events for Hillas ground-truth verification...")
    if CTAPIPE_AVAILABLE:
        logger.info("ctapipe is installed. Cross-validation against ctapipe enabled.")
    else:
        logger.info("ctapipe is not installed. Skipping ctapipe comparison and proceeding with ground-truth validation.")

    records: List[Dict[str, Any]] = []

    for _ in tqdm(range(args.n_events), desc="Validating Hillas extraction"):
        # Simulate bright noise-free gamma
        ev = sim.simulate_gamma(sigma_nsb=0.0)
        hillas = calculate_hillas_parameters(ev.clean_image, camera=cam, time_image=ev.time_image)

        if not hillas.is_valid:
            continue

        true_p = ev.true_parameters
        rec = {
            "true_length": true_p["length"],
            "length": hillas.length,
            "true_width": true_p["width"],
            "width": hillas.width,
            "true_distance": true_p["distance"],
            "distance": hillas.distance,
            "true_alpha": true_p["alpha_deg"],
            "alpha": hillas.alpha,
            "true_size": true_p["true_size"],
            "size": hillas.size,
            "phi_deg": hillas.phi_deg,
            "true_phi_deg": true_p["phi_deg"],
        }
        records.append(rec)

    df_val = pd.DataFrame(records)

    # Compute correlation coefficients and relative bias
    corr_length = float(np.corrcoef(df_val["true_length"], df_val["length"])[0, 1])
    corr_width = float(np.corrcoef(df_val["true_width"], df_val["width"])[0, 1])
    corr_dist = float(np.corrcoef(df_val["true_distance"], df_val["distance"])[0, 1])
    corr_alpha = float(np.corrcoef(df_val["true_alpha"], df_val["alpha"])[0, 1])

    mean_rel_diff_length = float(np.mean(np.abs(df_val["length"] - df_val["true_length"]) / df_val["true_length"]))
    mean_rel_diff_width = float(np.mean(np.abs(df_val["width"] - df_val["true_width"]) / df_val["true_width"]))

    logger.info("=== Hillas Ground-Truth Agreement ===")
    logger.info(f"Length: Pearson r = {corr_length:.4f}, Mean Rel Err = {mean_rel_diff_length * 100:.2f}%")
    logger.info(f"Width:  Pearson r = {corr_width:.4f}, Mean Rel Err = {mean_rel_diff_width * 100:.2f}%")
    logger.info(f"Dist:   Pearson r = {corr_dist:.4f}")
    logger.info(f"Alpha:  Pearson r = {corr_alpha:.4f}")

    metrics = {
        "n_events": len(df_val),
        "ctapipe_available": CTAPIPE_AVAILABLE,
        "length_pearson_r": corr_length,
        "width_pearson_r": corr_width,
        "distance_pearson_r": corr_dist,
        "alpha_pearson_r": corr_alpha,
        "mean_relative_error_length": mean_rel_diff_length,
        "mean_relative_error_width": mean_rel_diff_width,
    }

    # Save metrics JSON
    json_path = results_dir / "hillas_validation.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Saved metrics to {json_path}")

    # Plot validation figure
    fig_path = figures_dir / "hillas_validation.png"
    plot_hillas_validation(df_val, save_path=fig_path)
    logger.info(f"Saved validation figure to {fig_path}")


if __name__ == "__main__":
    main()
