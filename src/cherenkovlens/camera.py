"""
Camera geometry representations for IACT telescopes.

NOTE ON REAL HARDWARE GEOMETRY:
Real IACTs like the Major Atmospheric Cerenkov Experiment (MACE) at Hanle, Ladakh
(altitude ~4270m a.s.l.) utilize a dedicated camera consisting of 1088 photomultiplier
tubes (PMTs) arranged in a graded layout: an inner region (0.1 deg PMTs) and an outer
region (0.2 deg PMTs), spanning an overall Field of View (FoV) of approximately 4.0 degrees.
The High Altitude GAmma Ray (HAGAR) observatory is a distributed 7-telescope wavefront
sampling array rather than a single large imaging reflector.

In this package, CameraGeometry provides:
1. Default: A 34 x 34 square pixel grid (1156 pixels, closely matching MACE's 1088 channel count)
   with uniform pixel angular size (default 0.1 deg).
2. Hexagonal grid variant: Hexagonal pixel tiling with explicit 6-neighbour adjacency lookup.

Users intending to analyze physical MACE or HAGAR archival FITS data should consult:
- Koul et al., "The MACE Gamma-Ray Telescope at Hanle, Ladakh" (Astroparticle Physics)
- Chitnis et al., "The HAGAR Telescope Array at Hanle"
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional, Set, Tuple

import numpy as np


@dataclass
class CameraGeometry:
    """
    Representation of the IACT camera pixel layout in degrees.

    Parameters
    ----------
    n_pixels_side : int, default=34
        Number of pixels per side for square grid (total pixels = n_pixels_side**2).
    pixel_size_deg : float, default=0.1
        Angular diameter/pitch of each pixel in degrees.
    geometry_type : {'square', 'hexagonal'}, default='square'
        Grid type. 'square' produces a 2D matrix; 'hexagonal' produces hexagonal tiling.
    """

    n_pixels_side: int = 34
    pixel_size_deg: float = 0.1
    geometry_type: Literal["square", "hexagonal"] = "square"

    pix_x: np.ndarray = field(init=False)
    pix_y: np.ndarray = field(init=False)
    pix_x_grid: Optional[np.ndarray] = field(init=False, default=None)
    pix_y_grid: Optional[np.ndarray] = field(init=False, default=None)
    neighbors: Dict[int, List[int]] = field(init=False, default_factory=dict)
    n_pixels: int = field(init=False)

    def __post_init__(self) -> None:
        if self.geometry_type == "square":
            self._init_square_geometry()
        elif self.geometry_type == "hexagonal":
            self._init_hexagonal_geometry()
        else:
            raise ValueError(f"Unsupported geometry_type: '{self.geometry_type}'. Must be 'square' or 'hexagonal'.")

    def _init_square_geometry(self) -> None:
        """Initialize square grid geometry and coordinates centered at (0, 0)."""
        N = self.n_pixels_side
        self.n_pixels = N * N

        # Coordinate centered at (0, 0) in degrees
        offset = (N - 1) / 2.0
        coords_1d = (np.arange(N) - offset) * self.pixel_size_deg

        # Grid coordinates (X increases with column, Y increases with row)
        grid_x, grid_y = np.meshgrid(coords_1d, coords_1d)
        self.pix_x_grid = grid_x
        self.pix_y_grid = grid_y

        self.pix_x = grid_x.ravel()
        self.pix_y = grid_y.ravel()

        # Build 8-neighbour adjacency dictionary
        self.neighbors = {}
        for r in range(N):
            for c in range(N):
                idx = r * N + c
                neigh_list = []
                for dr in (-1, 0, 1):
                    for dc in (-1, 0, 1):
                        if dr == 0 and dc == 0:
                            continue
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < N and 0 <= nc < N:
                            neigh_list.append(nr * N + nc)
                self.neighbors[idx] = neigh_list

    def _init_hexagonal_geometry(self) -> None:
        """
        Initialize hexagonal pixel geometry with axial coordinates.
        Generates approximately the target number of pixels arranged in a hexagonal boundary.
        """
        # Determine radius R such that 3*R*(R-1) + 1 approx n_pixels_side^2
        target_pixels = self.n_pixels_side ** 2
        # 3*R^2 - 3*R + (1 - target) = 0
        r_approx = int(np.round((3.0 + np.sqrt(9.0 - 12.0 * (1.0 - target_pixels))) / 6.0))
        radius = max(2, r_approx)

        axial_coords: List[Tuple[int, int]] = []
        for q in range(-radius, radius + 1):
            r1 = max(-radius, -q - radius)
            r2 = min(radius, -q + radius)
            for r in range(r1, r2 + 1):
                axial_coords.append((q, r))

        self.n_pixels = len(axial_coords)
        coord_to_idx = {coord: idx for idx, coord in enumerate(axial_coords)}

        # Cartesian coordinates: flat-topped hexagon basis
        # x = s * (sqrt(3)*q + sqrt(3)/2 * r), y = s * (3/2 * r)
        s = self.pixel_size_deg / np.sqrt(3)
        xs = [s * (np.sqrt(3) * q + (np.sqrt(3) / 2.0) * r) for q, r in axial_coords]
        ys = [s * (1.5 * r) for q, r in axial_coords]

        self.pix_x = np.array(xs, dtype=np.float64)
        self.pix_y = np.array(ys, dtype=np.float64)
        self.pix_x_grid = None
        self.pix_y_grid = None

        # 6 axial directions
        hex_dirs = [(1, 0), (1, -1), (0, -1), (-1, 0), (-1, 1), (0, 1)]
        self.neighbors = {}
        for idx, (q, r) in enumerate(axial_coords):
            n_list = []
            for dq, dr in hex_dirs:
                n_coord = (q + dq, r + dr)
                if n_coord in coord_to_idx:
                    n_list.append(coord_to_idx[n_coord])
            self.neighbors[idx] = n_list

    @property
    def fov_diameter_deg(self) -> float:
        """Total field of view diameter across the diagonal or boundary (deg)."""
        r_max = np.max(np.sqrt(self.pix_x**2 + self.pix_y**2))
        return float(2.0 * r_max)

    def image_to_grid(self, image_flat: np.ndarray) -> np.ndarray:
        """Reshape a 1D flat pixel array into a 2D grid (square geometry only)."""
        if self.geometry_type != "square":
            raise ValueError("image_to_grid is only available for square geometry.")
        return image_flat.reshape((self.n_pixels_side, self.n_pixels_side))

    def grid_to_flat(self, image_grid: np.ndarray) -> np.ndarray:
        """Flatten a 2D image grid into a 1D pixel array."""
        return image_grid.ravel()
