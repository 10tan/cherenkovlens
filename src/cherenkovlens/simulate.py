"""
Simulation module for Imaging Atmospheric Cherenkov Telescope (IACT) events.

Generates synthetic Cherenkov camera images of:
1. Gamma-ray showers (compact, elliptical, oriented toward camera center for on-axis sources)
2. Proton / hadronic showers (irregular, multi-blob, diffuse, uncorrelated orientation, with muon-ring variants)
3. Night Sky Background (NSB) photo-electron noise (Gaussian or Poisson)
4. Cherenkov arrival time profiles (coherent gradient for gammas; turbulent/incoherent for hadrons)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np

from cherenkovlens.camera import CameraGeometry


@dataclass
class Event:
    """
    Lightweight container for a simulated IACT camera event.

    Attributes
    ----------
    image : np.ndarray
        Observed pixel intensity array (pe), including Cherenkov signal and NSB noise.
        Shape: (N, N) for square geometry, or (N_pix,) flat.
    time_image : np.ndarray
        Pixel arrival times in nanoseconds (ns).
        Shape: (N, N) for square geometry, or (N_pix,) flat.
    label : int
        Class label: 1 for gamma-ray, 0 for proton (hadron).
    true_parameters : dict
        Ground-truth generation parameters (amplitude, centroid, length, width, phi, alpha, etc.).
    clean_image : np.ndarray
        Noise-free true Cherenkov signal image before NSB addition (pe).
    """

    image: np.ndarray
    time_image: np.ndarray
    label: int
    true_parameters: Dict[str, Any]
    clean_image: np.ndarray


class ShowerSimulator:
    """
    IACT event generator for gamma-ray and hadronic air showers with NSB noise.

    Parameters
    ----------
    camera : CameraGeometry, optional
        Camera geometry model. If None, default square 34x34 camera is used.
    config : dict, optional
        Simulation configuration dictionary. If None, default settings are used.
    rng : np.random.Generator, optional
        NumPy random number generator for reproducible stochastic generation.
    """

    def __init__(
        self,
        camera: Optional[CameraGeometry] = None,
        config: Optional[Dict[str, Any]] = None,
        rng: Optional[np.random.Generator] = None,
    ) -> None:
        self.camera = camera if camera is not None else CameraGeometry()
        self.config = config or {}
        self.rng = rng if rng is not None else np.random.default_rng(42)

        # Cache config subsections with defaults
        sim_cfg = self.config.get("simulation", {})
        self.gamma_cfg = sim_cfg.get("gamma", {})
        self.proton_cfg = sim_cfg.get("proton", {})
        self.noise_cfg = self.config.get("noise", {})

    def add_nsb_noise(
        self,
        signal_image: np.ndarray,
        sigma_nsb: Optional[float] = None,
        noise_type: Optional[str] = None,
        mu_nsb: Optional[float] = None,
    ) -> np.ndarray:
        """
        Add independent Night Sky Background (NSB) noise to each pixel.

        Parameters
        ----------
        signal_image : np.ndarray
            Pure signal image in photo-electrons.
        sigma_nsb : float, optional
            Standard deviation of NSB fluctuations per pixel (pe).
        noise_type : {'gaussian', 'poisson'}, optional
            Noise probability model.
        mu_nsb : float, optional
            Mean pedestal / baseline level (pe).

        Returns
        -------
        np.ndarray
            Observed image (signal + noise).
        """
        sig = sigma_nsb if sigma_nsb is not None else self.noise_cfg.get("sigma_nsb", 4.5)
        ntype = noise_type or self.noise_cfg.get("type", "gaussian")
        mu = mu_nsb if mu_nsb is not None else self.noise_cfg.get("mu_nsb", 0.0)

        if ntype == "poisson":
            # In Poisson statistics with mean N_pe, variance = mean = sig^2
            # For NSB with variance sigma_nsb^2, the mean Poisson counts is sig^2.
            lam = max(1.0, float(sig**2))
            noise = self.rng.poisson(lam=lam, size=signal_image.shape).astype(np.float64) - lam + mu
        else:  # gaussian
            noise = self.rng.normal(loc=mu, scale=sig, size=signal_image.shape)

        return signal_image + noise

    def simulate_gamma(
        self,
        amplitude: Optional[float] = None,
        x0: Optional[float] = None,
        y0: Optional[float] = None,
        length: Optional[float] = None,
        width: Optional[float] = None,
        phi: Optional[float] = None,
        sigma_nsb: Optional[float] = None,
    ) -> Event:
        """
        Simulate an electromagnetic (gamma-ray) air shower event.

        Physics Model:
        - Rotated 2D elliptical Gaussian intensity profile.
        - Orientation vector biased towards camera center (0, 0), reproducing
          the on-axis gamma signature observed by IACTs like MACE/HAGAR.
        - Linear arrival-time gradient along the shower major axis.
        - Poisson photo-electron fluctuation of Cherenkov light.
        """
        cfg = self.gamma_cfg

        # 1. Amplitude (pe) - log-uniform mimicking power-law energy spectrum
        if amplitude is None:
            a_min = cfg.get("amp_min", 60.0)
            a_max = cfg.get("amp_max", 3500.0)
            amplitude = float(np.exp(self.rng.uniform(np.log(a_min), np.log(a_max))))

        # 2. Centroid (x0, y0) in camera coordinates (degrees)
        r_max = cfg.get("centroid_r_max_deg", 1.1)
        if x0 is None or y0 is None:
            r = np.sqrt(self.rng.uniform(0.04, r_max**2))  # avoid exact (0,0)
            theta = self.rng.uniform(0.0, 2.0 * np.pi)
            x0 = r * np.cos(theta) if x0 is None else x0
            y0 = r * np.sin(theta) if y0 is None else y0

        # 3. Shape: Length (sx) and Width (sy) in degrees
        l_min, l_max = cfg.get("length_min_deg", 0.12), cfg.get("length_max_deg", 0.38)
        w_min, w_max = cfg.get("width_min_deg", 0.04), cfg.get("width_max_deg", 0.14)
        if length is None:
            length = float(self.rng.uniform(l_min, l_max))
        if width is None:
            # Gammas are compact ellipses: width is always strictly smaller than length
            width = float(self.rng.uniform(w_min, min(w_max, length * 0.55)))

        # 4. Orientation: biased toward camera center with small angular scatter
        dist = np.sqrt(x0**2 + y0**2)
        pointing_angle = np.arctan2(y0, x0)  # vector from (0,0) to centroid
        if phi is None:
            alpha_scatter = np.radians(cfg.get("alpha_scatter_deg", 10.0))
            # Shower points towards center (or away, symmetric modulo pi)
            phi = float(pointing_angle + self.rng.normal(0.0, alpha_scatter))

        # Normalize phi to [-pi/2, pi/2] for major axis convention
        phi = float((phi + np.pi / 2.0) % np.pi - np.pi / 2.0)

        # True alpha angle (between major axis and radius vector from camera center)
        # Vector along major axis: u = (cos(phi), sin(phi))
        # Vector to centroid: c = (x0, y0)
        cos_alpha = np.abs(x0 * np.cos(phi) + y0 * np.sin(phi)) / max(1e-7, dist)
        true_alpha_deg = float(np.degrees(np.arccos(np.clip(cos_alpha, 0.0, 1.0))))

        # 5. Compute 2D Elliptical Gaussian Intensity:
        # I(x, y) = A * exp( - [a(x - x0)^2 + 2b(x - x0)(y - y0) + c(y - y0)^2] )
        sx2 = max(1e-6, length**2)
        sy2 = max(1e-6, width**2)
        cos_phi = np.cos(phi)
        sin_phi = np.sin(phi)
        sin_2phi = np.sin(2.0 * phi)

        dx = self.camera.pix_x - x0
        dy = self.camera.pix_y - y0

        # Rotate coordinates onto shower principal axes:
        # x_rot is along major axis (phi), y_rot is along minor axis (transverse)
        x_rot = dx * cos_phi + dy * sin_phi
        y_rot = -dx * sin_phi + dy * cos_phi
        exponent = (x_rot**2) / (2.0 * sx2) + (y_rot**2) / (2.0 * sy2)

        # Truncate exponential to prevent underflow warnings
        clean_intensity = amplitude * np.exp(-np.clip(exponent, 0.0, 30.0))

        # Poisson fluctuation on physical Cherenkov photons
        # (signal in pe has Poisson counting statistics)
        clean_pe = self.rng.poisson(lam=np.clip(clean_intensity, 0.0, 1e7)).astype(np.float64)

        # 6. Timing channel:
        # Cherenkov light propagates down the atmosphere, yielding a smooth gradient along major axis
        grad_mean = cfg.get("time_gradient_mean", 1.6)
        grad_std = cfg.get("time_gradient_std", 0.4)
        time_gradient = float(self.rng.normal(grad_mean, grad_std))
        jitter_ns = cfg.get("time_jitter_ns", 0.75)

        # Coordinate projected along major axis:
        proj_long = dx * cos_phi + dy * sin_phi
        t_base = 20.0  # reference arrival time in ns
        time_signal = t_base + time_gradient * proj_long + self.rng.normal(0.0, jitter_ns, size=clean_pe.shape)

        # Pixels with negligible signal receive random background timing
        noise_time = self.rng.uniform(10.0, 30.0, size=clean_pe.shape)
        weights = 1.0 / (1.0 + np.exp(- (clean_pe - 2.0)))
        time_channel = weights * time_signal + (1.0 - weights) * noise_time

        # 7. Add NSB noise to intensity
        obs_intensity = self.add_nsb_noise(clean_pe, sigma_nsb=sigma_nsb)

        # Shape outputs
        if self.camera.geometry_type == "square":
            img = self.camera.image_to_grid(obs_intensity)
            t_img = self.camera.image_to_grid(time_channel)
            clean_img = self.camera.image_to_grid(clean_pe)
        else:
            img = obs_intensity
            t_img = time_channel
            clean_img = clean_pe

        true_params = {
            "type": "gamma",
            "amplitude": amplitude,
            "x0": float(x0),
            "y0": float(y0),
            "length": float(length),
            "width": float(width),
            "phi_rad": float(phi),
            "phi_deg": float(np.degrees(phi)),
            "distance": float(dist),
            "alpha_deg": float(true_alpha_deg),
            "time_gradient": float(time_gradient),
            "true_size": float(np.sum(clean_pe)),
        }

        return Event(
            image=img,
            time_image=t_img,
            label=1,
            true_parameters=true_params,
            clean_image=clean_img,
        )

    def simulate_proton(
        self,
        amplitude: Optional[float] = None,
        x0: Optional[float] = None,
        y0: Optional[float] = None,
        phi: Optional[float] = None,
        sigma_nsb: Optional[float] = None,
        force_muon_ring: bool = False,
    ) -> Event:
        """
        Simulate a hadronic (proton) air shower event.

        Physics Model:
        - Hadronic cascades generate multiple secondary electromagnetic sub-cascades
          (from neutral pion decay) leading to multi-blob, irregular, clumpy images.
        - Shower orientations are isotropic / uncorrelated with camera center.
        - Occasional muon ring-like or compact hadronic core is simulated.
        - Arrival times exhibit larger incoherent jitter across disjoint sub-cascades.
        """
        cfg = self.proton_cfg

        if amplitude is None:
            a_min = cfg.get("amp_min", 70.0)
            a_max = cfg.get("amp_max", 4500.0)
            amplitude = float(np.exp(self.rng.uniform(np.log(a_min), np.log(a_max))))

        # Check for muon ring variant
        muon_frac = cfg.get("muon_ring_fraction", 0.08)
        is_muon_ring = force_muon_ring or (self.rng.uniform(0.0, 1.0) < muon_frac)

        if is_muon_ring:
            return self._simulate_muon_ring(amplitude, x0=x0, y0=y0, sigma_nsb=sigma_nsb)

        # Standard multi-blob hadronic shower
        r_max = cfg.get("centroid_r_max_deg", 1.2)
        if x0 is not None and y0 is not None:
            x_center = float(x0)
            y_center = float(y0)
        else:
            r_core = np.sqrt(self.rng.uniform(0.02, r_max**2))
            theta_core = self.rng.uniform(0.0, 2.0 * np.pi)
            x_center = float(r_core * np.cos(theta_core))
            y_center = float(r_core * np.sin(theta_core))

        n_blobs_min = cfg.get("n_blobs_min", 2)
        n_blobs_max = cfg.get("n_blobs_max", 6)
        n_blobs = int(self.rng.integers(n_blobs_min, n_blobs_max + 1))

        # Split total amplitude randomly among sub-blobs
        blob_fractions = self.rng.dirichlet(np.ones(n_blobs))
        blob_spread = cfg.get("blob_spread_deg", 0.45)
        blob_size_min = cfg.get("blob_size_min_deg", 0.06)
        blob_size_max = cfg.get("blob_size_max_deg", 0.24)

        clean_intensity = np.zeros(self.camera.n_pixels, dtype=np.float64)
        time_channel = np.zeros(self.camera.n_pixels, dtype=np.float64)
        t_base = 20.0
        jitter_ns = cfg.get("time_jitter_ns", 1.35)

        for i in range(n_blobs):
            b_amp = amplitude * blob_fractions[i]
            bx = x_center + self.rng.normal(0.0, blob_spread)
            by = y_center + self.rng.normal(0.0, blob_spread)
            bsx = self.rng.uniform(blob_size_min, blob_size_max)
            bsy = self.rng.uniform(blob_size_min, bsx)
            b_phi = self.rng.uniform(0.0, np.pi) if phi is None else (phi + self.rng.normal(0.0, 0.4))

            # Rotation parameters for blob
            cos_p, sin_p = np.cos(b_phi), np.sin(b_phi)
            dx = self.camera.pix_x - bx
            dy = self.camera.pix_y - by
            dx_rot = dx * cos_p + dy * sin_p
            dy_rot = -dx * cos_p + dy * sin_p

            blob_profile = b_amp * np.exp(-0.5 * ((dx_rot / bsx)**2 + (dy_rot / bsy)**2))
            clean_intensity += blob_profile

            # Blob-specific timing offset and incoherent jitter
            blob_t_offset = self.rng.uniform(-2.0, 2.0)
            time_channel += blob_profile * (t_base + blob_t_offset + self.rng.normal(0.0, jitter_ns, size=dx.shape))

        # Normalize timing channel weighted by intensity
        tot_weights = np.maximum(clean_intensity, 1e-6)
        time_channel = time_channel / tot_weights

        # Background timing for pixels without signal
        noise_time = self.rng.uniform(10.0, 30.0, size=clean_intensity.shape)
        w_sig = 1.0 / (1.0 + np.exp(-(clean_intensity - 2.0)))
        time_channel = w_sig * time_channel + (1.0 - w_sig) * noise_time

        # Poisson fluctuation on physical Cherenkov photons
        clean_pe = self.rng.poisson(lam=np.clip(clean_intensity, 0.0, 1e7)).astype(np.float64)
        obs_intensity = self.add_nsb_noise(clean_pe, sigma_nsb=sigma_nsb)

        if self.camera.geometry_type == "square":
            img = self.camera.image_to_grid(obs_intensity)
            t_img = self.camera.image_to_grid(time_channel)
            clean_img = self.camera.image_to_grid(clean_pe)
        else:
            img = obs_intensity
            t_img = time_channel
            clean_img = clean_pe

        true_params = {
            "type": "proton",
            "is_muon_ring": False,
            "amplitude": amplitude,
            "x0": float(x_center),
            "y0": float(y_center),
            "n_blobs": n_blobs,
            "true_size": float(np.sum(clean_pe)),
        }

        return Event(
            image=img,
            time_image=t_img,
            label=0,
            true_parameters=true_params,
            clean_image=clean_img,
        )

    def _simulate_muon_ring(
        self,
        amplitude: float,
        x0: Optional[float] = None,
        y0: Optional[float] = None,
        sigma_nsb: Optional[float] = None,
    ) -> Event:
        """Simulate a local single-muon ring/arc event."""
        # Ring center in camera
        if x0 is not None and y0 is not None:
            ring_x0 = float(x0)
            ring_y0 = float(y0)
        else:
            r_core = self.rng.uniform(0.0, 0.8)
            theta_core = self.rng.uniform(0.0, 2.0 * np.pi)
            ring_x0 = float(r_core * np.cos(theta_core))
            ring_y0 = float(r_core * np.sin(theta_core))

        # Muon Cherenkov characteristic ring radius in air is ~1.0-1.3 deg
        radius = float(self.rng.uniform(0.7, 1.3))
        ring_width = float(self.rng.uniform(0.04, 0.08))

        dx = self.camera.pix_x - ring_x0
        dy = self.camera.pix_y - ring_y0
        r_pix = np.sqrt(dx**2 + dy**2)

        # Ring profile
        ring_profile = amplitude * np.exp(-0.5 * ((r_pix - radius) / ring_width)**2)

        # Often only a partial arc is intercepted by the reflector
        arc_angle_span = self.rng.uniform(np.pi * 0.8, 2.0 * np.pi)
        arc_center_angle = self.rng.uniform(0.0, 2.0 * np.pi)
        phi_pix = np.arctan2(dy, dx)
        d_angle = np.abs(np.arctan2(np.sin(phi_pix - arc_center_angle), np.cos(phi_pix - arc_center_angle)))
        arc_mask = np.exp(-0.5 * np.maximum(0.0, d_angle - arc_angle_span / 2.0)**2 / (0.3**2))

        clean_intensity = ring_profile * arc_mask
        clean_pe = self.rng.poisson(lam=np.clip(clean_intensity, 0.0, 1e7)).astype(np.float64)

        # Muon timing is virtually instantaneous across the ring
        t_base = 20.0
        time_channel = t_base + self.rng.normal(0.0, 0.5, size=clean_pe.shape)

        obs_intensity = self.add_nsb_noise(clean_pe, sigma_nsb=sigma_nsb)

        if self.camera.geometry_type == "square":
            img = self.camera.image_to_grid(obs_intensity)
            t_img = self.camera.image_to_grid(time_channel)
            clean_img = self.camera.image_to_grid(clean_pe)
        else:
            img = obs_intensity
            t_img = time_channel
            clean_img = clean_pe

        true_params = {
            "type": "proton",
            "is_muon_ring": True,
            "amplitude": amplitude,
            "x0": float(ring_x0),
            "y0": float(ring_y0),
            "radius": radius,
            "true_size": float(np.sum(clean_pe)),
        }

        return Event(
            image=img,
            time_image=t_img,
            label=0,
            true_parameters=true_params,
            clean_image=clean_img,
        )

    def simulate_event(self, label: int, **kwargs: Any) -> Event:
        """
        Simulate an event by label (1 = gamma, 0 = proton).

        Parameters
        ----------
        label : int
            1 for gamma, 0 for proton.
        **kwargs : Any
            Additional parameters passed to simulate_gamma or simulate_proton.

        Returns
        -------
        Event
        """
        if label == 1:
            return self.simulate_gamma(**kwargs)
        elif label == 0:
            return self.simulate_proton(**kwargs)
        else:
            raise ValueError(f"Invalid label: {label}. Must be 1 (gamma) or 0 (proton).")
