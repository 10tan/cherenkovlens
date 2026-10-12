"""
Hillas parameter and timing feature extraction for IACT camera images.

Extracts standard geometrical Hillas parameters (Size, Centroid, Length, Width,
Orientation phi, Distance, Alpha, Miss, Concentration) and arrival-time features
(timing gradient along major axis, timing RMS residual) from cleaned images.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np

from cherenkovlens.camera import CameraGeometry


@dataclass
class HillasParameters:
    """
    Extracted Hillas and timing parameters for an IACT event.

    Attributes
    ----------
    size : float
        Total Cherenkov light intensity (photo-electrons).
    x_mean : float
        Centroid X coordinate in camera FoV (deg).
    y_mean : float
        Centroid Y coordinate in camera FoV (deg).
    length : float
        RMS spread along major axis (deg).
    width : float
        RMS spread along minor axis (deg).
    phi : float
        Major axis orientation angle in radians, in [-pi/2, pi/2].
    phi_deg : float
        Major axis orientation angle in degrees, in [-90, 90].
    distance : float
        Distance from camera center (0, 0) to centroid (deg).
    alpha : float
        Angle between major axis and radius vector from camera center (deg), in [0, 90].
    miss : float
        Perpendicular distance from camera center to major axis (deg).
    concentration : float
        Fraction of total intensity contained in the two brightest pixels.
    n_pixels : int
        Number of surviving pixels in the cleaned shower image.
    time_gradient : float
        Arrival-time gradient along the shower major axis (ns/deg).
    time_rms : float
        Weighted RMS of arrival-time residuals around linear fit (ns).
    is_valid : bool
        Flag indicating whether parameter extraction succeeded without degeneracy.
    """

    size: float
    x_mean: float
    y_mean: float
    length: float
    width: float
    phi: float
    phi_deg: float
    distance: float
    alpha: float
    miss: float
    concentration: float
    n_pixels: int
    time_gradient: float
    time_rms: float
    is_valid: bool

    def to_dict(self) -> Dict[str, Any]:
        """Convert parameters to a standard dictionary."""
        return asdict(self)

    def to_feature_dict(self) -> Dict[str, float]:
        """Return feature dictionary suitable for ML models (excluding meta-flags)."""
        return {
            "size": self.size,
            "length": self.length,
            "width": self.width,
            "wl_ratio": self.width / max(1e-4, self.length) if np.isfinite(self.width) and np.isfinite(self.length) else np.nan,
            "distance": self.distance,
            "alpha": self.alpha,
            "miss": self.miss,
            "concentration": self.concentration,
            "n_pixels": float(self.n_pixels),
            "time_gradient": np.abs(self.time_gradient) if np.isfinite(self.time_gradient) else np.nan,
            "time_rms": self.time_rms,
        }


def solve_2x2_eigenvalues_analytical(
    sxx: float,
    syy: float,
    sxy: float,
) -> Tuple[float, float, float]:
    """
    Closed-form analytical solution for eigenvalues and orientation angle of a 2x2 symmetric covariance matrix.

    Covariance Matrix:
        Sigma = [[sxx, sxy],
                 [sxy, syy]]

    Trace: T = sxx + syy
    Determinant: D = sxx * syy - sxy^2
    Discriminant: Delta = sqrt((sxx - syy)^2 + 4 * sxy^2)

    Eigenvalues:
        lambda_1 = (T + Delta) / 2  (major eigenvalue >= lambda_2)
        lambda_2 = (T - Delta) / 2  (minor eigenvalue)

    Orientation angle phi of major eigenvector:
        tan(2*phi) = 2*sxy / (sxx - syy)
        phi = 0.5 * atan2(2*sxy, sxx - syy) in [-pi/2, pi/2]

    Parameters
    ----------
    sxx, syy, sxy : float
        Central second moments.

    Returns
    -------
    lambda_1 : float
        Major eigenvalue.
    lambda_2 : float
        Minor eigenvalue.
    phi : float
        Major axis orientation angle in radians.
    """
    trace = sxx + syy
    diff = sxx - syy
    delta = np.sqrt(max(0.0, diff**2 + 4.0 * (sxy**2)))

    lambda_1 = 0.5 * (trace + delta)
    lambda_2 = 0.5 * (trace - delta)

    # Orientation angle in [-pi/2, pi/2]
    phi = 0.5 * np.arctan2(2.0 * sxy, diff)

    return float(lambda_1), float(lambda_2), float(phi)


def calculate_hillas_parameters(
    cleaned_image: np.ndarray,
    camera: CameraGeometry,
    time_image: Optional[np.ndarray] = None,
    min_pixels: int = 3,
    min_size: float = 15.0,
) -> HillasParameters:
    """
    Calculate Hillas parameters and arrival-time features from a cleaned IACT image.

    Parameters
    ----------
    cleaned_image : np.ndarray
        Cleaned image (intensities in pe, unkept pixels are 0.0). Can be 2D or 1D.
    camera : CameraGeometry
        Camera geometry providing pixel coordinates in degrees.
    time_image : np.ndarray, optional
        Arrival-time image (ns). If None, timing features will be 0.0.
    min_pixels : int, default=3
        Minimum number of surviving pixels required for reliable moment extraction.
    min_size : float, default=15.0
        Minimum total size (pe) required for reliable extraction.

    Returns
    -------
    HillasParameters
        Calculated Hillas and timing features, with is_valid=False on degeneracies.
    """
    # Flatten inputs
    img_flat = cleaned_image.ravel()
    t_flat = time_image.ravel() if time_image is not None else None

    # Filter surviving pixels with strictly positive intensity
    mask = img_flat > 0.0
    n_pix = int(np.sum(mask))

    def _invalid_result(n_surviving: int, sz: float = 0.0) -> HillasParameters:
        return HillasParameters(
            size=sz,
            x_mean=np.nan,
            y_mean=np.nan,
            length=np.nan,
            width=np.nan,
            phi=np.nan,
            phi_deg=np.nan,
            distance=np.nan,
            alpha=np.nan,
            miss=np.nan,
            concentration=np.nan,
            n_pixels=n_surviving,
            time_gradient=np.nan,
            time_rms=np.nan,
            is_valid=False,
        )

    # Edge cases: empty image, single-pixel, or fewer than min_pixels
    if n_pix < min_pixels:
        return _invalid_result(n_pix, float(np.sum(img_flat)))

    w = img_flat[mask].astype(np.float64)
    size = float(np.sum(w))

    if size < min_size or not np.isfinite(size):
        return _invalid_result(n_pix, size)

    x = camera.pix_x[mask]
    y = camera.pix_y[mask]

    # 1. Centroid (first moments)
    x_mean = float(np.sum(w * x) / size)
    y_mean = float(np.sum(w * y) / size)

    # 2. Central second moments
    dx = x - x_mean
    dy = y - y_mean
    sxx = float(np.sum(w * (dx**2)) / size)
    syy = float(np.sum(w * (dy**2)) / size)
    sxy = float(np.sum(w * dx * dy) / size)

    # 3. Analytical eigensolution
    lambda_1, lambda_2, phi = solve_2x2_eigenvalues_analytical(sxx, syy, sxy)

    length = float(np.sqrt(max(0.0, lambda_1)))
    width = float(np.sqrt(max(0.0, lambda_2)))

    # 4. Geometrical features: Distance, Alpha, Miss
    distance = float(np.sqrt(x_mean**2 + y_mean**2))

    # Unit vector along major axis
    cos_phi = np.cos(phi)
    sin_phi = np.sin(phi)

    # Alpha angle: angle between major axis u=(cos_phi, sin_phi) and radius vector c=(x_mean, y_mean)
    if distance > 1e-5:
        cos_alpha = np.abs(x_mean * cos_phi + y_mean * sin_phi) / distance
        cos_alpha = np.clip(cos_alpha, 0.0, 1.0)
        alpha_rad = float(np.arccos(cos_alpha))
        alpha_deg = float(np.degrees(alpha_rad))
    else:
        # Centroid at camera center: alpha is undefined / isotropic
        alpha_rad = 0.0
        alpha_deg = 0.0

    # Miss: perpendicular distance from camera center (0,0) to major axis
    miss = float(np.abs(x_mean * sin_phi - y_mean * cos_phi))

    # 5. Concentration: fraction of light in 2 brightest pixels
    if len(w) >= 2:
        top_2_sum = float(np.sum(np.partition(w, -2)[-2:]))
        concentration = float(top_2_sum / size)
    else:
        concentration = 1.0

    # 6. Timing channel analysis
    if t_flat is not None:
        t = t_flat[mask]
        # Coordinate along major axis
        proj_p = dx * cos_phi + dy * sin_phi
        var_p = float(np.sum(w * (proj_p**2)) / size)

        if var_p > 1e-6 and len(t) >= min_pixels:
            t_mean = float(np.sum(w * t) / size)
            cov_pt = float(np.sum(w * proj_p * (t - t_mean)) / size)
            time_grad = float(cov_pt / var_p)

            residuals = t - (time_grad * proj_p + t_mean)
            time_rms = float(np.sqrt(np.sum(w * (residuals**2)) / size))
        else:
            time_grad = 0.0
            time_rms = 0.0
    else:
        time_grad = 0.0
        time_rms = 0.0

    return HillasParameters(
        size=size,
        x_mean=x_mean,
        y_mean=y_mean,
        length=length,
        width=width,
        phi=phi,
        phi_deg=float(np.degrees(phi)),
        distance=distance,
        alpha=alpha_deg,
        miss=miss,
        concentration=concentration,
        n_pixels=n_pix,
        time_gradient=time_grad,
        time_rms=time_rms,
        is_valid=True,
    )
