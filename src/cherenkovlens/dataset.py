"""
Dataset generation pipeline for IACT Cherenkov events.

Simulates batches of gamma and proton showers, runs image cleaning,
extracts Hillas and arrival-time features, and packages the data into
pandas DataFrames exported to Parquet / CSV.
"""

from __future__ import annotations

import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

from cherenkovlens.camera import CameraGeometry
from cherenkovlens.cleaning import tailcuts_clean
from cherenkovlens.hillas import calculate_hillas_parameters
from cherenkovlens.simulate import ShowerSimulator
from cherenkovlens.utils import load_config, set_seed


def _generate_chunk(
    chunk_seed: int,
    n_gammas: int,
    n_protons: int,
    camera_kwargs: Dict[str, Any],
    sim_config: Dict[str, Any],
    sigma_nsb: float,
    core_sigma: float,
    boundary_sigma: float,
    min_island_size: int,
) -> List[Dict[str, Any]]:
    """Worker function to simulate and extract features for a single chunk."""
    rng = np.random.default_rng(chunk_seed)
    camera = CameraGeometry(**camera_kwargs)
    sim = ShowerSimulator(camera=camera, config=sim_config, rng=rng)

    labels = [1] * n_gammas + [0] * n_protons
    rng.shuffle(labels)

    records = []
    for label in labels:
        ev = sim.simulate_event(label, sigma_nsb=sigma_nsb)
        clean_res = tailcuts_clean(
            ev.image,
            sigma_nsb=sigma_nsb,
            core_sigma=core_sigma,
            boundary_sigma=boundary_sigma,
            min_island_size=min_island_size,
        )
        hillas = calculate_hillas_parameters(
            clean_res.cleaned_image,
            camera=camera,
            time_image=ev.time_image,
        )

        if hillas.is_valid:
            row = hillas.to_feature_dict()
            row["label"] = label
            row["n_islands"] = clean_res.n_islands
            records.append(row)

    return records


def generate_dataset(
    n_events: int = 40000,
    gamma_fraction: float = 0.5,
    config: Optional[Dict[str, Any]] = None,
    sigma_nsb: Optional[float] = None,
    output_path: Optional[str | Path] = None,
    n_jobs: int = -1,
    seed: int = 42,
    show_progress: bool = True,
) -> pd.DataFrame:
    """
    Generate a balanced or stratified dataset of IACT events with extracted Hillas features.

    Parameters
    ----------
    n_events : int, default=40000
        Total events to simulate.
    gamma_fraction : float, default=0.5
        Proportion of gamma-ray events (label=1).
    config : dict, optional
        Configuration dictionary. If None, default config is loaded.
    sigma_nsb : float, optional
        NSB noise standard deviation in photo-electrons.
    output_path : str or Path, optional
        File path to save the generated dataset (supports .csv and .parquet).
    n_jobs : int, default=-1
        Number of parallel worker processes (-1 for all CPUs).
    seed : int, default=42
        Master seed for reproducible dataset generation.
    show_progress : bool, default=True
        Whether to display a tqdm progress bar.

    Returns
    -------
    pd.DataFrame
        DataFrame with extracted Hillas and timing features and class labels.
    """
    if config is None:
        config = load_config()

    cam_cfg = config.get("camera", {})
    clean_cfg = config.get("cleaning", {}).get("tailcuts", {})
    noise_cfg = config.get("noise", {})

    sig_nsb = sigma_nsb if sigma_nsb is not None else noise_cfg.get("sigma_nsb", 4.5)
    core_sig = clean_cfg.get("core_sigma", 6.0)
    bound_sig = clean_cfg.get("boundary_sigma", 3.0)
    min_island = clean_cfg.get("min_island_size", 2)

    n_gammas_total = int(round(n_events * gamma_fraction))
    n_protons_total = n_events - n_gammas_total

    if n_jobs < 1:
        n_workers = max(1, mp.cpu_count())
    else:
        n_workers = n_jobs

    # Divide events among chunks
    chunk_size = max(500, n_events // (n_workers * 4))
    n_chunks = max(1, int(np.ceil(n_events / chunk_size)))

    chunk_gammas = [n_gammas_total // n_chunks] * n_chunks
    for i in range(n_gammas_total % n_chunks):
        chunk_gammas[i] += 1

    chunk_protons = [n_protons_total // n_chunks] * n_chunks
    for i in range(n_protons_total % n_chunks):
        chunk_protons[i] += 1

    # Deterministic child seeds derived from master seed
    master_rng = np.random.default_rng(seed)
    chunk_seeds = master_rng.integers(1, 2**31 - 1, size=n_chunks).tolist()

    camera_kwargs = {
        "n_pixels_side": cam_cfg.get("n_pixels_side", 34),
        "pixel_size_deg": cam_cfg.get("pixel_size_deg", 0.1),
        "geometry_type": cam_cfg.get("geometry_type", "square"),
    }

    all_records: List[Dict[str, Any]] = []

    if n_workers == 1 or n_chunks == 1:
        # Sequential execution
        iterator = range(n_chunks)
        if show_progress:
            iterator = tqdm(iterator, desc="Simulating Cherenkov events")
        for i in iterator:
            chunk_rec = _generate_chunk(
                chunk_seed=chunk_seeds[i],
                n_gammas=chunk_gammas[i],
                n_protons=chunk_protons[i],
                camera_kwargs=camera_kwargs,
                sim_config=config,
                sigma_nsb=sig_nsb,
                core_sigma=core_sig,
                boundary_sigma=bound_sig,
                min_island_size=min_island,
            )
            all_records.extend(chunk_rec)
    else:
        # Parallel execution with ProcessPoolExecutor
        tasks = []
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            for i in range(n_chunks):
                t = executor.submit(
                    _generate_chunk,
                    chunk_seed=chunk_seeds[i],
                    n_gammas=chunk_gammas[i],
                    n_protons=chunk_protons[i],
                    camera_kwargs=camera_kwargs,
                    sim_config=config,
                    sigma_nsb=sig_nsb,
                    core_sigma=core_sig,
                    boundary_sigma=bound_sig,
                    min_island_size=min_island,
                )
                tasks.append(t)

            iterator = tasks
            if show_progress:
                iterator = tqdm(tasks, desc="Simulating Cherenkov events (parallel)")
            for t in iterator:
                all_records.extend(t.result())

    df = pd.DataFrame(all_records)

    # Save to disk if requested
    if output_path is not None:
        out_p = Path(output_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        if out_p.suffix == ".parquet":
            df.to_parquet(out_p, index=False)
        else:
            df.to_csv(out_p, index=False)

    return df
