#!/usr/bin/env python3
"""
Study script: Comparative performance of Two-Level Tail-Cuts vs Naive Single-Threshold Cleaning.

Measures:
1. Signal pixel retention: fraction of true shower pixels successfully preserved.
2. Noise leakage: average count of spurious NSB noise pixels surviving cleaning.
3. Size reconstruction bias: percentage bias between cleaned Size and ground-truth Size.

Saves:
- figures/cleaning_study.png
- results/cleaning_study.json
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict

import numpy as np
from tqdm import tqdm

from cherenkovlens.camera import CameraGeometry
from cherenkovlens.cleaning import naive_clean, tailcuts_clean
from cherenkovlens.plotting import plot_cleaning_comparison_study
from cherenkovlens.simulate import ShowerSimulator
from cherenkovlens.utils import ensure_dirs, get_project_root, load_config, setup_logger


def main() -> None:
    parser = argparse.ArgumentParser(description="Run image cleaning comparison study.")
    parser.add_argument("--n-events", type=int, default=2000, help="Number of simulated gamma events (default: 2000)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    args = parser.parse_args()

    logger = setup_logger("cleaning_study")
    root = get_project_root()
    cfg = load_config()

    figures_dir = root / "figures"
    results_dir = root / "results"
    ensure_dirs(figures_dir, results_dir)

    cam = CameraGeometry(**cfg.get("camera", {}))
    sim = ShowerSimulator(camera=cam, config=cfg, rng=np.random.default_rng(args.seed))

    sig_nsb = cfg.get("noise", {}).get("sigma_nsb", 4.5)
    core_sig = cfg.get("cleaning", {}).get("tailcuts", {}).get("core_sigma", 6.0)
    bound_sig = cfg.get("cleaning", {}).get("tailcuts", {}).get("boundary_sigma", 3.0)
    naive_thresh = cfg.get("cleaning", {}).get("naive", {}).get("threshold_sigma", 4.5) * sig_nsb

    logger.info(f"Running cleaning study on {args.n_events} simulated gamma showers...")
    logger.info(f"Tail-cuts thresholds: Core={core_sig}*sigma ({core_sig*sig_nsb:.1f} pe), Boundary={bound_sig}*sigma ({bound_sig*sig_nsb:.1f} pe)")
    logger.info(f"Naive threshold: {naive_thresh:.1f} pe")

    naive_retention = []
    tailcuts_retention = []
    naive_noise_leak = []
    tailcuts_noise_leak = []
    naive_size_bias = []
    tailcuts_size_bias = []

    # Store one representative event for visualization
    sample_raw = None
    sample_naive = None
    sample_tailcuts = None

    for i in tqdm(range(args.n_events), desc="Evaluating cleaning methods"):
        ev = sim.simulate_gamma(sigma_nsb=sig_nsb)

        # Ground truth signal mask: pixels with true Cherenkov signal >= 2.0 pe
        true_signal_mask = ev.clean_image >= 2.0
        n_true_signal = np.sum(true_signal_mask)
        true_size = np.sum(ev.clean_image)

        if n_true_signal < 3 or true_size < 15.0:
            continue

        # 1. Naive Cleaning
        naive_res = naive_clean(ev.image, threshold=naive_thresh)
        naive_kept_mask = naive_res.mask

        # 2. Tail-Cuts Cleaning
        tc_res = tailcuts_clean(
            ev.image,
            sigma_nsb=sig_nsb,
            core_sigma=core_sig,
            boundary_sigma=bound_sig,
            min_island_size=2,
        )
        tc_kept_mask = tc_res.mask

        # Metrics for Naive
        n_naive_signal_kept = np.sum(naive_kept_mask & true_signal_mask)
        n_naive_noise_kept = np.sum(naive_kept_mask & ~true_signal_mask)
        naive_retention.append(n_naive_signal_kept / n_true_signal)
        naive_noise_leak.append(n_naive_noise_kept)
        naive_size = np.sum(naive_res.cleaned_image)
        naive_size_bias.append((naive_size - true_size) / true_size * 100.0)

        # Metrics for Tail-Cuts
        n_tc_signal_kept = np.sum(tc_kept_mask & true_signal_mask)
        n_tc_noise_kept = np.sum(tc_kept_mask & ~true_signal_mask)
        tailcuts_retention.append(n_tc_signal_kept / n_true_signal)
        tailcuts_noise_leak.append(n_tc_noise_kept)
        tc_size = np.sum(tc_res.cleaned_image)
        tailcuts_size_bias.append((tc_size - true_size) / true_size * 100.0)

        # Pick a visually prominent representative event
        if sample_raw is None and 400.0 < true_size < 1200.0:
            sample_raw = ev.image
            sample_naive = naive_res.cleaned_image
            sample_tailcuts = tc_res.cleaned_image

    summary: Dict[str, Any] = {
        "n_events_evaluated": len(naive_retention),
        "naive_signal_retention": float(np.mean(naive_retention)),
        "tailcuts_signal_retention": float(np.mean(tailcuts_retention)),
        "naive_noise_leakage_mean": float(np.mean(naive_noise_leak)),
        "tailcuts_noise_leakage_mean": float(np.mean(tailcuts_noise_leak)),
        "naive_size_bias_pct": float(np.mean(naive_size_bias)),
        "tailcuts_size_bias_pct": float(np.mean(tailcuts_size_bias)),
        "noise_rejection_gain_factor": float(np.mean(naive_noise_leak) / max(1e-4, np.mean(tailcuts_noise_leak))),
    }

    logger.info("=== Cleaning Study Results ===")
    logger.info(f"Signal Retention: Naive = {summary['naive_signal_retention']*100:.2f}%, Tail-cuts = {summary['tailcuts_signal_retention']*100:.2f}%")
    logger.info(f"Noise Leakage (pixels/ev): Naive = {summary['naive_noise_leakage_mean']:.2f}, Tail-cuts = {summary['tailcuts_noise_leakage_mean']:.2f}")
    logger.info(f"Noise Rejection Gain Factor: {summary['noise_rejection_gain_factor']:.2f}x fewer false noise pixels")
    logger.info(f"Size Reconstruction Bias: Naive = {summary['naive_size_bias_pct']:+.2f}%, Tail-cuts = {summary['tailcuts_size_bias_pct']:+.2f}%")

    # Save summary JSON
    json_path = results_dir / "cleaning_study.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Saved metrics to {json_path}")

    # Plot headline comparison
    fig_path = figures_dir / "cleaning_study.png"
    if sample_raw is None:
        sample_raw = ev.image
        sample_naive = naive_res.cleaned_image
        sample_tailcuts = tc_res.cleaned_image

    plot_cleaning_comparison_study(sample_raw, sample_naive, sample_tailcuts, summary, save_path=fig_path)
    logger.info(f"Saved headline comparison figure to {fig_path}")


if __name__ == "__main__":
    main()
