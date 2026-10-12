"""
Unit tests for CameraGeometry and ShowerSimulator.
"""

import numpy as np
import pytest

from cherenkovlens.camera import CameraGeometry
from cherenkovlens.simulate import Event, ShowerSimulator


def test_camera_geometry_square():
    cam = CameraGeometry(n_pixels_side=34, pixel_size_deg=0.1, geometry_type="square")
    assert cam.n_pixels == 34 * 34
    assert cam.pix_x.shape == (1156,)
    assert cam.pix_y.shape == (1156,)
    assert cam.pix_x_grid is not None
    assert cam.pix_x_grid.shape == (34, 34)
    # Check center symmetry
    assert np.isclose(np.mean(cam.pix_x), 0.0)
    assert np.isclose(np.mean(cam.pix_y), 0.0)
    # Check neighbor connectivity (corner pixel has 3 neighbors, center pixel has 8)
    assert len(cam.neighbors[0]) == 3
    center_idx = 17 * 34 + 17
    assert len(cam.neighbors[center_idx]) == 8


def test_camera_geometry_hexagonal():
    cam = CameraGeometry(n_pixels_side=20, pixel_size_deg=0.1, geometry_type="hexagonal")
    assert cam.n_pixels > 0
    assert len(cam.pix_x) == cam.n_pixels
    assert len(cam.pix_y) == cam.n_pixels
    # Every internal hex pixel has at most 6 neighbors
    for nbrs in cam.neighbors.values():
        assert len(nbrs) <= 6


def test_simulate_gamma_shape_and_label():
    rng = np.random.default_rng(123)
    cam = CameraGeometry(n_pixels_side=34)
    sim = ShowerSimulator(camera=cam, rng=rng)

    event = sim.simulate_gamma(amplitude=500.0, x0=0.5, y0=0.5)
    assert isinstance(event, Event)
    assert event.label == 1
    assert event.image.shape == (34, 34)
    assert event.time_image.shape == (34, 34)
    assert event.clean_image.shape == (34, 34)
    assert "length" in event.true_parameters
    assert "width" in event.true_parameters
    assert "alpha_deg" in event.true_parameters
    assert event.true_parameters["true_size"] > 0


def test_simulate_proton_shape_and_label():
    rng = np.random.default_rng(456)
    cam = CameraGeometry(n_pixels_side=34)
    sim = ShowerSimulator(camera=cam, rng=rng)

    event = sim.simulate_proton(amplitude=800.0)
    assert isinstance(event, Event)
    assert event.label == 0
    assert event.image.shape == (34, 34)
    assert event.time_image.shape == (34, 34)
    assert event.clean_image.shape == (34, 34)
    assert event.true_parameters["type"] == "proton"


def test_simulate_muon_ring():
    rng = np.random.default_rng(789)
    cam = CameraGeometry(n_pixels_side=34)
    sim = ShowerSimulator(camera=cam, rng=rng)

    event = sim.simulate_proton(amplitude=600.0, force_muon_ring=True)
    assert event.label == 0
    assert event.true_parameters["is_muon_ring"] is True
    assert "radius" in event.true_parameters


def test_simulate_reproducibility():
    cam = CameraGeometry(n_pixels_side=34)
    sim1 = ShowerSimulator(camera=cam, rng=np.random.default_rng(42))
    sim2 = ShowerSimulator(camera=cam, rng=np.random.default_rng(42))

    ev1 = sim1.simulate_gamma(amplitude=300.0, x0=0.3, y0=0.3)
    ev2 = sim2.simulate_gamma(amplitude=300.0, x0=0.3, y0=0.3)

    np.testing.assert_allclose(ev1.image, ev2.image)
    np.testing.assert_allclose(ev1.time_image, ev2.time_image)
