"""
Unit tests for Hillas parameter calculation and analytical eigensolver.
"""

import numpy as np
import pytest

from cherenkovlens.camera import CameraGeometry
from cherenkovlens.hillas import (
    HillasParameters,
    calculate_hillas_parameters,
    solve_2x2_eigenvalues_analytical,
)
from cherenkovlens.simulate import ShowerSimulator


def test_analytical_eigensolver_matches_numpy_eigh():
    """Verify closed-form analytical eigenvalues against np.linalg.eigh across various tensors."""
    rng = np.random.default_rng(42)

    for _ in range(50):
        # Generate random positive semi-definite 2x2 covariance matrix
        A = rng.normal(0.0, 1.0, size=(2, 2))
        cov = A @ A.T  # symmetric positive semi-definite
        sxx = float(cov[0, 0])
        syy = float(cov[1, 1])
        sxy = float(cov[0, 1])

        lam1_analyt, lam2_analyt, phi_analyt = solve_2x2_eigenvalues_analytical(sxx, syy, sxy)

        # NumPy eigh returns eigenvalues in ascending order
        np_w, np_v = np.linalg.eigh(cov)
        lam1_np = np_w[1]  # largest
        lam2_np = np_w[0]  # smallest

        np.testing.assert_allclose(lam1_analyt, lam1_np, rtol=1e-7, atol=1e-7)
        np.testing.assert_allclose(lam2_analyt, lam2_np, rtol=1e-7, atol=1e-7)

        # Check that vector (cos(phi), sin(phi)) is an eigenvector for lam1
        u = np.array([np.cos(phi_analyt), np.sin(phi_analyt)])
        cov_u = cov @ u
        expected_cov_u = lam1_analyt * u
        np.testing.assert_allclose(cov_u, expected_cov_u, rtol=1e-6, atol=1e-6)


def test_hillas_recovery_on_noise_free_gamma():
    """Test that a clean, noise-free gamma event yields Hillas parameters near ground truth."""
    cam = CameraGeometry(n_pixels_side=34, pixel_size_deg=0.1)
    sim = ShowerSimulator(camera=cam, rng=np.random.default_rng(99))

    true_x0 = 0.5
    true_y0 = 0.5
    true_length = 0.25
    true_width = 0.08

    # Simulate gamma event with 0 noise
    event = sim.simulate_gamma(
        amplitude=2500.0,
        x0=true_x0,
        y0=true_y0,
        length=true_length,
        width=true_width,
        sigma_nsb=0.0,
    )

    params = calculate_hillas_parameters(event.clean_image, camera=cam, time_image=event.time_image)

    assert params.is_valid is True
    # Centroid should be within half a pixel size (< 0.05 deg)
    assert np.isclose(params.x_mean, true_x0, atol=0.04)
    assert np.isclose(params.y_mean, true_y0, atol=0.04)

    # Length and Width recovery should be within ~15% due to pixel discretization
    assert np.isclose(params.length, true_length, rtol=0.20)
    assert np.isclose(params.width, true_width, rtol=0.25)

    # Alpha recovery
    assert np.isclose(params.alpha, event.true_parameters["alpha_deg"], atol=15.0)


def test_hillas_edge_cases_never_crash():
    """Check empty image, single pixel, and fewer than 3 pixels return invalid without error."""
    cam = CameraGeometry(n_pixels_side=34)

    # 1. Empty image
    empty_img = np.zeros((34, 34))
    res_empty = calculate_hillas_parameters(empty_img, camera=cam)
    assert res_empty.is_valid is False
    assert np.isnan(res_empty.length)
    assert res_empty.size == 0.0

    # 2. Single pixel image
    single_pix = np.zeros((34, 34))
    single_pix[10, 10] = 50.0
    res_single = calculate_hillas_parameters(single_pix, camera=cam)
    assert res_single.is_valid is False
    assert np.isnan(res_single.width)

    # 3. Two pixels image
    two_pix = np.zeros((34, 34))
    two_pix[10, 10] = 50.0
    two_pix[10, 11] = 40.0
    res_two = calculate_hillas_parameters(two_pix, camera=cam, min_pixels=3)
    assert res_two.is_valid is False
    assert res_two.n_pixels == 2
