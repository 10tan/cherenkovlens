#!/usr/bin/env python3
"""
Noise Robustness Study: Assess IACT gamma/hadron classification under elevated
Night Sky Background (NSB) photon noise (e.g. 1x, 2x, 4x, 8x nominal, mimicking moonlit observations).

Compares:
1. Standard analysis chain with Two-Level Tail-Cuts cleaning.
2. Naive analysis chain with single-threshold cleaning.

Saves:
- figures/noise_robustness.png
- results/noise_study.csv
"""

import argparse
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from tqdm import tqdm

from cherenkovlens.camera import CameraGeometry
from cherenkovlens.classify import (
    compute_q_factor_curve,
    prepare_features,
    train_and_evaluate_classifier,
)
from cherenkovlens.cleaning import naive_clean, tailcuts_clean
from cherenkovlens.hillas import calculate_hillas_parameters
from cherenkovlens.plotting import plot_noise_robustness_study
from cherenkovlens.simulate import ShowerSimulator
from cherenkovlens.utils import ensure_dirs, get_project_root, load_config, setup_logger


def simulate_and_extract_for_noise(
    sim: ShowerSimulator,
    camera: CameraGeometry,
    sigma_nsb: float,
    n_events: int = 4000,
    use_tailcuts: bool = True,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate events under specified noise level and extract features using chosen cleaning method."""
    rng = np.random.default_rng(seed)
    records = []

    for i in range(n_events):
        label = i % 2
        ev = sim.simulate_event(label, sigma_nsb=sigma_nsb)

        if use_tailcuts:
            res = tailcuts_clean(ev.image, sigma_nsb=sigma_nsb, core_sigma=6.0, boundary_sigma=3.0, min_island_size=2)
        else:
            res = naive_clean(ev.image, threshold=4.5 * sigma_nsb, min_island_size=1)

        hillas = calculate_hillas_parameters(res.cleaned_image, camera=camera, time_image=ev.time_image)
        if hillas.is_valid:
            row = hillas.to_feature_dict()
            row["label"] = label
            records.append(row)

    return pd.DataFrame(records)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run NSB noise robustness study.")
    parser.add_argument("--events-per-level", type=int, default=3000, help="Events per NSB condition (default: 3000)")
    parser.add_argument("--seed", type=int, default=42, help="Seed (default: 42)")
    args = parser.parse_args()

    logger = setup_logger("noise_study")
    root = get_project_root()
    cfg = load_config()

    figures_dir = root / "figures"
    results_dir = root / "results"
    ensure_dirs(figures_dir, results_dir)

    cam = CameraGeometry(**cfg.get("camera", {}))
    sim = ShowerSimulator(camera=cam, config=cfg, rng=np.random.default_rng(args.seed))

    nominal_sigma = cfg.get("noise", {}).get("sigma_nsb", 4.5)
    nsb_multipliers = [1.0, 2.0, 4.0, 8.0]

    study_records: List[Dict[str, Any]] = []

    for mult in tqdm(nsb_multipliers, desc="Testing NSB levels"):
        current_sigma = nominal_sigma * mult
        logger.info(f"--- NSB Multiplier: {mult}x (sigma_nsb = {current_sigma:.2f} pe) ---")

        # 1. With Tail-Cuts Cleaning
        df_tc = simulate_and_extract_for_noise(
            sim, cam, sigma_nsb=current_sigma, n_events=args.events_per_level, use_tailcuts=True, seed=args.seed + int(mult * 10)
        )
        _, res_tc = train_and_evaluate_classifier(df_tc, model_type="random_forest", random_state=args.seed)

        # 2. With Naive Cleaning
        df_naive = simulate_and_extract_for_noise(
            sim, cam, sigma_nsb=current_sigma, n_events=args.events_per_level, use_tailcuts=False, seed=args.seed + int(mult * 10)
        )
        _, res_naive = train_and_evaluate_classifier(df_naive, model_type="random_forest", random_state=args.seed)

        logger.info(
            f"[{mult}x NSB] Tail-Cuts: AUC={res_tc.auc:.4f}, Q={res_tc.best_q_factor:.2f} | "
            f"Naive: AUC={res_naive.auc:.4f}, Q={res_naive.best_q_factor:.2f}"
        )

        study_records.append(
            {
                "nsb_mult": mult,
                "sigma_nsb": current_sigma,
                "n_survived_tailcuts": len(df_tc),
                "n_survived_naive": len(df_naive),
                "auc_tailcuts": res_tc.auc,
                "auc_naive": res_naive.auc,
                "q_tailcuts": res_tc.best_q_factor,
                "q_naive": res_naive.best_q_factor,
                "auc_delta": res_tc.auc - res_naive.auc,
                "q_delta": res_tc.best_q_factor - res_naive.q_factors[int(np.argmax(res_naive.q_factors))],
            }
        )

    df_results = pd.DataFrame(study_records)

    # Save CSV
    csv_path = results_dir / "noise_study.csv"
    df_results.to_csv(csv_path, index=False)
    logger.info(f"Saved noise study metrics to {csv_path}")

    # Plot Noise Robustness Figure
    fig_path = figures_dir / "noise_robustness.png"
    plot_noise_robustness_study(df_results, save_path=fig_path)
    logger.info(f"Saved noise study figure to {fig_path}")


if __name__ == "__main__":
    main()
