#!/usr/bin/env python3
"""
Script to generate the primary CherenkovLens dataset (default: 40,000 events).
"""

import argparse
import sys
from pathlib import Path

from cherenkovlens.dataset import generate_dataset
from cherenkovlens.utils import ensure_dirs, get_project_root, load_config, setup_logger


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic IACT Cherenkov dataset.")
    parser.add_argument("--n-events", type=int, default=40000, help="Total events to generate (default: 40,000)")
    parser.add_argument("--gamma-fraction", type=float, default=0.5, help="Gamma event fraction (default: 0.5)")
    parser.add_argument("--seed", type=int, default=42, help="Master seed (default: 42)")
    parser.add_argument("--n-jobs", type=int, default=-1, help="Parallel worker processes (-1 for all CPUs)")
    parser.add_argument("--output", type=str, default=None, help="Output file path (default: data/dataset.csv)")
    args = parser.parse_args()

    logger = setup_logger("generate_dataset")
    root = get_project_root()
    cfg = load_config()

    output_path = Path(args.output) if args.output else root / "data" / "dataset.csv"
    ensure_dirs(output_path.parent)

    logger.info(f"Generating {args.n_events} events (gamma_frac={args.gamma_fraction}, seed={args.seed})...")
    df = generate_dataset(
        n_events=args.n_events,
        gamma_fraction=args.gamma_fraction,
        config=cfg,
        output_path=output_path,
        n_jobs=args.n_jobs,
        seed=args.seed,
        show_progress=True,
    )

    n_gammas = int(df["label"].sum())
    n_protons = int(len(df) - n_gammas)
    logger.info(f"Dataset successfully created and saved to {output_path}")
    logger.info(f"Total surviving events: {len(df)} (Gammas: {n_gammas}, Protons: {n_protons})")
    logger.info(f"Survival efficiency: {len(df) / args.n_events * 100:.2f}%")


if __name__ == "__main__":
    main()
