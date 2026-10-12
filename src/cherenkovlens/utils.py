"""
CherenkovLens utilities: configuration loader, reproducible seeding, logging, and path resolution.
"""

from __future__ import annotations

import logging
import os
import random
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import yaml


def get_project_root() -> Path:
    """Return the absolute path to the project root directory."""
    # Current file is in src/cherenkovlens/utils.py
    return Path(__file__).resolve().parent.parent.parent


def load_config(config_path: Optional[str | Path] = None) -> Dict[str, Any]:
    """
    Load YAML configuration file.

    Parameters
    ----------
    config_path : str or Path, optional
        Path to config YAML. If None, defaults to config/default.yaml in project root.

    Returns
    -------
    dict
        Parsed configuration dictionary.
    """
    if config_path is None:
        config_path = get_project_root() / "config" / "default.yaml"
    else:
        config_path = Path(config_path)

    if not config_path.is_file():
        raise FileNotFoundError(f"Configuration file not found at: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    return config


def set_seed(seed: int = 42) -> np.random.Generator:
    """
    Set seed across Python and NumPy, and return an independent NumPy Generator.

    Parameters
    ----------
    seed : int
        Seed for pseudo-random number generators.

    Returns
    -------
    np.random.Generator
        Initialized NumPy random generator.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    return np.random.default_rng(seed)


def setup_logger(name: str = "cherenkovlens", level: int = logging.INFO) -> logging.Logger:
    """
    Set up and return a configured logger with clean formatting.

    Parameters
    ----------
    name : str
        Logger name.
    level : int
        Logging level (default: logging.INFO).

    Returns
    -------
    logging.Logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Avoid adding duplicate handlers if logger was already created
    if not logger.handlers:
        ch = logging.StreamHandler()
        ch.setLevel(level)
        formatter = logging.Formatter(
            fmt="[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        ch.setFormatter(formatter)
        logger.addHandler(ch)

    return logger


def ensure_dirs(*dirs: Path | str) -> None:
    """Ensure that the specified directories exist."""
    for d in dirs:
        Path(d).mkdir(parents=True, exist_ok=True)
