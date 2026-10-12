"""
Image cleaning algorithms for IACT cameras.

Includes:
1. Two-level tail-cuts cleaning (core + boundary connected component island filter)
2. Naive single-threshold cleaning (for baseline comparison and study)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterator, List, Optional, Tuple, Union

import numpy as np
from scipy import ndimage


@dataclass
class CleaningResult:
    """
    Container for image cleaning outputs.

    Allows tuple unpacking: `cleaned_img, mask, n_islands = result`
    as well as attribute access.

    Attributes
    ----------
    cleaned_image : np.ndarray
        Cleaned pixel intensity array (pixels failing cleaning set to 0.0).
    mask : np.ndarray
        Boolean mask of surviving Cherenkov shower pixels (True = kept).
    n_islands : int
        Number of surviving connected component islands.
    core_mask : np.ndarray
        Boolean mask of pixels exceeding the core threshold.
    boundary_mask : np.ndarray
        Boolean mask of pixels exceeding the boundary threshold.
    """

    cleaned_image: np.ndarray
    mask: np.ndarray
    n_islands: int
    core_mask: np.ndarray
    boundary_mask: np.ndarray

    def __iter__(self) -> Iterator[Union[np.ndarray, int]]:
        """Enable unpacking as (cleaned_image, mask, n_islands)."""
        yield self.cleaned_image
        yield self.mask
        yield self.n_islands


def tailcuts_clean(
    image: np.ndarray,
    sigma_nsb: float = 4.5,
    core_sigma: float = 6.0,
    boundary_sigma: float = 3.0,
    connectivity: int = 8,
    min_island_size: int = 2,
    subtract_baseline: bool = True,
    mu_nsb: float = 0.0,
    neighbors: Optional[Dict[int, List[int]]] = None,
) -> CleaningResult:
    """
    Clean an IACT camera image using the two-level tail-cuts algorithm.

    Physical & Methodological Rationale:
    -----------------------------------
    IACT images are dominated by Poisson-like Night Sky Background (NSB) photon
    noise and electronic pedestal noise. Tail-cuts cleaning isolates coherent
    Cherenkov shower cascades through a dual-threshold spatial clustering:
    1. Core threshold (core_sigma * sigma_nsb): Identifies high-confidence shower core pixels.
    2. Boundary threshold (boundary_sigma * sigma_nsb): Identifies fainter shower boundary pixels.
    3. Connected component labeling: Only boundary-candidate pixels connected to at least
       one core pixel within a contiguous island are preserved; isolated noise spikes are purged.
    4. Baseline Subtraction: Subtracting mu_nsb before thresholding centers Gaussian/Poisson
       noise fluctuations symmetrically around 0 pe, avoiding systematic bias from varying night-sky
       illumination or pedestal levels across telescope runs.

    Parameters
    ----------
    image : np.ndarray
        Raw observed pixel intensity image (pe), 2D or 1D.
    sigma_nsb : float, default=4.5
        NSB noise standard deviation in photo-electrons (pe).
    core_sigma : float, default=6.0
        Core threshold multiplier (pe = core_sigma * sigma_nsb).
    boundary_sigma : float, default=3.0
        Boundary candidate threshold multiplier (pe = boundary_sigma * sigma_nsb).
    connectivity : int, default=8
        Pixel connectivity for 2D images (4 or 8).
    min_island_size : int, default=2
        Minimum number of pixels an island must contain to survive.
    subtract_baseline : bool, default=True
        Whether to subtract mu_nsb from the raw image before thresholding.
    mu_nsb : float, default=0.0
        Pedestal baseline mean in photo-electrons (pe).
    neighbors : dict, optional
        Adjacency list mapping pixel index to neighbor indices (used for 1D or hex grids).

    Returns
    -------
    CleaningResult
        Object containing cleaned_image, mask, n_islands, core_mask, and boundary_mask.
    """
    if boundary_sigma > core_sigma:
        raise ValueError(f"boundary_sigma ({boundary_sigma}) cannot exceed core_sigma ({core_sigma}).")

    # 1. Baseline subtraction
    img_sub = (image - mu_nsb) if subtract_baseline else image.copy()

    core_thresh = core_sigma * sigma_nsb
    boundary_thresh = boundary_sigma * sigma_nsb

    core_mask = img_sub >= core_thresh
    boundary_mask = img_sub >= boundary_thresh

    # Quick exit if no core pixels exist
    if not np.any(core_mask):
        zero_img = np.zeros_like(img_sub)
        false_mask = np.zeros_like(img_sub, dtype=bool)
        return CleaningResult(
            cleaned_image=zero_img,
            mask=false_mask,
            n_islands=0,
            core_mask=core_mask,
            boundary_mask=boundary_mask,
        )

    # 2. 2D grid path (standard square camera)
    if image.ndim == 2:
        if connectivity == 8:
            structure = np.ones((3, 3), dtype=int)
        elif connectivity == 4:
            structure = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=int)
        else:
            raise ValueError(f"Unsupported connectivity {connectivity} for 2D images. Use 4 or 8.")

        labeled_array, num_features = ndimage.label(boundary_mask, structure=structure)

        if num_features == 0:
            surviving_mask = np.zeros_like(boundary_mask, dtype=bool)
            n_islands = 0
        else:
            # Find labels containing at least one core pixel
            labels_with_core = np.unique(labeled_array[core_mask])
            labels_with_core = labels_with_core[labels_with_core > 0]  # ignore 0 (background)

            # Check min_island_size
            valid_labels = []
            for lab in labels_with_core:
                island_size = np.sum(labeled_array == lab)
                if island_size >= min_island_size:
                    valid_labels.append(lab)

            if valid_labels:
                surviving_mask = np.isin(labeled_array, valid_labels)
                n_islands = len(valid_labels)
            else:
                surviving_mask = np.zeros_like(boundary_mask, dtype=bool)
                n_islands = 0

    # 3. 1D graph-based path (hexagonal or arbitrary geometry)
    else:
        flat_boundary = boundary_mask.ravel()
        flat_core = core_mask.ravel()
        n_pixels = len(flat_boundary)

        if neighbors is None:
            # Fallback: assume 1D linear array with 2 neighbors
            neighbors = {i: [j for j in (i - 1, i + 1) if 0 <= j < n_pixels] for i in range(n_pixels)}

        visited = np.zeros(n_pixels, dtype=bool)
        surviving_mask = np.zeros(n_pixels, dtype=bool)
        n_islands = 0

        # Breadth-first search / connected components on graph of boundary-candidate pixels
        for i in range(n_pixels):
            if flat_boundary[i] and not visited[i]:
                # Start new component
                component = []
                queue = [i]
                visited[i] = True
                has_core = False

                while queue:
                    curr = queue.pop(0)
                    component.append(curr)
                    if flat_core[curr]:
                        has_core = True
                    for nbr in neighbors.get(curr, []):
                        if flat_boundary[nbr] and not visited[nbr]:
                            visited[nbr] = True
                            queue.append(nbr)

                if has_core and len(component) >= min_island_size:
                    surviving_mask[component] = True
                    n_islands += 1

        surviving_mask = surviving_mask.reshape(image.shape)

    # 4. Set unkept pixels to exactly 0.0, keep positive signal in surviving pixels
    cleaned_image = np.where(surviving_mask, img_sub, 0.0)

    return CleaningResult(
        cleaned_image=cleaned_image,
        mask=surviving_mask,
        n_islands=n_islands,
        core_mask=core_mask,
        boundary_mask=boundary_mask,
    )


def naive_clean(
    image: np.ndarray,
    threshold: float,
    subtract_baseline: bool = True,
    mu_nsb: float = 0.0,
    min_island_size: int = 1,
) -> CleaningResult:
    """
    Clean an IACT camera image using a naive single hard threshold.

    Parameters
    ----------
    image : np.ndarray
        Raw observed pixel intensity image (pe).
    threshold : float
        Single hard intensity threshold in photo-electrons.
    subtract_baseline : bool, default=True
        Whether to subtract mu_nsb before thresholding.
    mu_nsb : float, default=0.0
        Pedestal baseline mean in photo-electrons (pe).
    min_island_size : int, default=1
        Minimum island size filter.

    Returns
    -------
    CleaningResult
    """
    img_sub = (image - mu_nsb) if subtract_baseline else image.copy()
    mask = img_sub >= threshold

    if image.ndim == 2:
        labeled_array, num_features = ndimage.label(mask, structure=np.ones((3, 3), dtype=int))
        if min_island_size > 1 and num_features > 0:
            valid_labels = []
            for lab in range(1, num_features + 1):
                if np.sum(labeled_array == lab) >= min_island_size:
                    valid_labels.append(lab)
            if valid_labels:
                mask = np.isin(labeled_array, valid_labels)
                n_islands = len(valid_labels)
            else:
                mask = np.zeros_like(mask, dtype=bool)
                n_islands = 0
        else:
            n_islands = num_features
    else:
        n_islands = 1 if np.any(mask) else 0

    cleaned_image = np.where(mask, img_sub, 0.0)

    return CleaningResult(
        cleaned_image=cleaned_image,
        mask=mask,
        n_islands=n_islands,
        core_mask=mask,
        boundary_mask=mask,
    )
