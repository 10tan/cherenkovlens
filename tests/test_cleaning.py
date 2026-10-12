"""
Unit tests for tailcuts_clean and naive_clean algorithms.
"""

import numpy as np
import pytest

from cherenkovlens.cleaning import CleaningResult, naive_clean, tailcuts_clean


def test_tailcuts_keeps_core_connected_island_and_removes_noise():
    # Construct a 10x10 synthetic image with:
    # 1. An isolated boundary pixel (val = 15, boundary threshold is 13.5)
    # 2. A core pixel (val = 30, core threshold is 27) with an adjacent boundary pixel (val = 16)
    sigma_nsb = 4.5
    core_thresh = 6.0 * sigma_nsb       # 27.0
    boundary_thresh = 3.0 * sigma_nsb   # 13.5

    image = np.zeros((10, 10), dtype=float)

    # Isolated noise spike exceeding boundary threshold
    image[1, 1] = 18.0

    # Real shower island: core + boundary neighbor
    image[5, 5] = 35.0  # core pixel
    image[5, 6] = 16.0  # boundary neighbor
    image[6, 5] = 15.0  # boundary neighbor

    res = tailcuts_clean(image, sigma_nsb=sigma_nsb, core_sigma=6.0, boundary_sigma=3.0, min_island_size=2)
    assert isinstance(res, CleaningResult)
    cleaned_img, mask, n_islands = res

    # Isolated noise spike should be removed (set to 0.0)
    assert mask[1, 1] is False or mask[1, 1] == 0
    assert cleaned_img[1, 1] == 0.0

    # Connected island pixels should be kept
    assert mask[5, 5] == True
    assert mask[5, 6] == True
    assert mask[6, 5] == True
    assert cleaned_img[5, 5] == 35.0
    assert cleaned_img[5, 6] == 16.0
    assert n_islands == 1


def test_tailcuts_no_core_returns_empty():
    sigma_nsb = 4.5
    image = np.full((10, 10), 10.0)  # all below boundary threshold
    res = tailcuts_clean(image, sigma_nsb=sigma_nsb, core_sigma=6.0, boundary_sigma=3.0)

    assert res.n_islands == 0
    assert np.all(res.cleaned_image == 0.0)
    assert np.all(~res.mask)


def test_naive_clean_vs_tailcuts():
    image = np.zeros((10, 10), dtype=float)
    image[2, 2] = 20.0  # isolated pixel
    image[7, 7] = 35.0  # core

    naive_res = naive_clean(image, threshold=15.0)
    assert naive_res.mask[2, 2] == True  # naive keeps isolated pixel above threshold
    assert naive_res.mask[7, 7] == True

    tailcuts_res = tailcuts_clean(image, sigma_nsb=4.5, core_sigma=6.0, boundary_sigma=3.0, min_island_size=2)
    assert tailcuts_res.mask[2, 2] == False  # tailcuts eliminates it
    assert tailcuts_res.mask[7, 7] == False or tailcuts_res.n_islands == 0  # isolated core discarded if min_island_size=2
