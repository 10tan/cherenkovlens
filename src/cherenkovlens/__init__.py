"""
CherenkovLens: Simulation, cleaning, feature extraction, and classification pipeline
for Imaging Atmospheric Cherenkov Telescopes (IACTs), inspired by MACE and HAGAR.
"""

__version__ = "0.1.0"

from cherenkovlens.camera import CameraGeometry
from cherenkovlens.simulate import Event, ShowerSimulator
from cherenkovlens.cleaning import tailcuts_clean, naive_clean
from cherenkovlens.hillas import calculate_hillas_parameters, HillasParameters
from cherenkovlens.utils import load_config, set_seed

__all__ = [
    "CameraGeometry",
    "ShowerSimulator",
    "Event",
    "tailcuts_clean",
    "naive_clean",
    "calculate_hillas_parameters",
    "HillasParameters",
    "load_config",
    "set_seed",
]
