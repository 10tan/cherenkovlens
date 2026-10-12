"""
CherenkovLens Interactive Streamlit Application.

Interactive simulation, two-level tail-cuts image cleaning,
Hillas moment reconstruction, and real-time machine learning inference
for Imaging Atmospheric Cherenkov Telescope (IACT) data.
Modelled on MACE (Hanle, Ladakh) and HAGAR analysis workflows.
"""

from __future__ import annotations

import os
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import pandas as pd
import streamlit as st

from cherenkovlens.camera import CameraGeometry
from cherenkovlens.cleaning import tailcuts_clean
from cherenkovlens.hillas import calculate_hillas_parameters
from cherenkovlens.plotting import plot_live_event_quad
from cherenkovlens.simulate import ShowerSimulator
from cherenkovlens.utils import get_project_root, load_config


@st.cache_resource
def get_camera_and_config():
    root = get_project_root()
    cfg = load_config(root / "config" / "default.yaml")
    cam = CameraGeometry(**cfg.get("camera", {}))
    return cam, cfg


@st.cache_resource
def load_trained_model():
    root = get_project_root()
    model_path = root / "models" / "classifier_rf.joblib"
    if model_path.exists():
        try:
            return joblib.load(model_path)
        except Exception:
            return None
    return None


def main():
    st.set_page_config(
        page_title="CherenkovLens: IACT Analysis Studio",
        page_icon="🔭",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.markdown(
        """
        <style>
        .main-header {
            font-size: 2.1rem;
            font-weight: 700;
            color: #1e3d59;
            margin-bottom: 0.2rem;
        }
        .sub-header {
            font-size: 1.05rem;
            color: #555;
            margin-bottom: 1.5rem;
        }
        .metric-card {
            background-color: #f8f9fa;
            border-radius: 8px;
            padding: 12px;
            border-left: 4px solid #1f77b4;
            margin-bottom: 10px;
        }
        </style>
        """,
        unsafe_allow_style_format=True,
    )

    st.markdown("<div class='main-header'>🔭 CherenkovLens: IACT Analysis Studio</div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='sub-header'>Interactive Simulation, Tail-Cuts Cleaning, Hillas Moment Extraction, "
        "and Gamma/Hadron Classification (Modelled on MACE Hanle & HAGAR)</div>",
        unsafe_allow_html=True,
    )

    cam, cfg = get_camera_and_config()
    model = load_trained_model()

    # --- SIDEBAR CONTROLS ---
    st.sidebar.header("Shower & Camera Controls")

    particle_type = st.sidebar.radio(
        "Particle Nature",
        options=["Gamma-ray (Signal)", "Proton / Hadron (Background)"],
        index=0,
    )
    is_gamma = particle_type.startswith("Gamma")

    seed_val = st.sidebar.number_input("Random Seed", min_value=0, max_value=999999, value=42, step=1)
    rng = np.random.default_rng(int(seed_val))
    sim = ShowerSimulator(camera=cam, config=cfg, rng=rng)

    st.sidebar.subheader("Physics Parameters")
    amp = st.sidebar.slider("Shower Amplitude / Energy (pe)", min_value=80.0, max_value=3500.0, value=750.0, step=50.0)
    x0 = st.sidebar.slider("Centroid X (deg)", min_value=-1.0, max_value=1.0, value=0.45, step=0.05)
    y0 = st.sidebar.slider("Centroid Y (deg)", min_value=-1.0, max_value=1.0, value=0.45, step=0.05)

    if is_gamma:
        length = st.sidebar.slider("Shower Length (deg)", min_value=0.10, max_value=0.40, value=0.25, step=0.01)
        width = st.sidebar.slider("Shower Width (deg)", min_value=0.03, max_value=0.18, value=0.08, step=0.01)
        phi_deg = st.sidebar.slider("Major Axis Angle phi (deg)", min_value=-90.0, max_value=90.0, value=45.0, step=2.0)
    else:
        force_muon = st.sidebar.checkbox("Simulate Single Muon Ring Variant", value=False)

    st.sidebar.subheader("Noise & Cleaning Cuts")
    nsb_sigma = st.sidebar.slider("NSB Noise sigma (pe/pixel)", min_value=0.0, max_value=20.0, value=4.5, step=0.5)
    core_sig = st.sidebar.slider("Tail-Cuts Core Threshold (sigma)", min_value=3.0, max_value=10.0, value=6.0, step=0.5)
    bnd_sig = st.sidebar.slider("Tail-Cuts Boundary Threshold (sigma)", min_value=1.5, max_value=6.0, value=3.0, step=0.5)

    # --- SIMULATE EVENT ---
    if is_gamma:
        event = sim.simulate_gamma(
            amplitude=amp,
            x0=x0,
            y0=y0,
            length=length,
            width=width,
            phi=np.radians(phi_deg),
            sigma_nsb=nsb_sigma,
        )
    else:
        event = sim.simulate_proton(
            amplitude=amp,
            sigma_nsb=nsb_sigma,
            force_muon_ring=force_muon,
        )

    # --- CLEAN EVENT ---
    clean_res = tailcuts_clean(
        event.image,
        sigma_nsb=nsb_sigma if nsb_sigma > 0 else 1e-4,
        core_sigma=core_sig,
        boundary_sigma=bnd_sig,
        min_island_size=2,
    )

    # --- HILLAS EXTRACTION ---
    hillas = calculate_hillas_parameters(
        clean_res.cleaned_image,
        camera=cam,
        time_image=event.time_image,
    )

    # --- MAIN DISPLAY TABS ---
    tab1, tab2, tab3 = st.tabs(["🖼️ Live Event & Hillas Extraction", "🤖 Classifier Inference", "📖 Telescope & Physics Context"])

    with tab1:
        st.subheader("Event Camera Displays")

        # Visualizer: 3 Camera panels (Raw, Cleaned, Hillas Overlay)
        fig = plot_live_event_quad(
            raw_img=event.image,
            clean_img=clean_res.cleaned_image,
            hillas_obj=hillas,
            label=event.label,
            pixel_size_deg=cam.pixel_size_deg,
        )
        st.pyplot(fig)
        plt.close(fig)

        st.markdown("---")
        st.subheader("Extracted Parameter Table")

        if hillas.is_valid:
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Size (Cherenkov Light)", f"{hillas.size:.1f} pe")
            col2.metric("Length", f"{hillas.length:.3f}°")
            col3.metric("Width", f"{hillas.width:.3f}°")
            col4.metric("Alpha Angle", f"{hillas.alpha:.1f}°")

            col5, col6, col7, col8 = st.columns(4)
            col5.metric("Centroid Distance", f"{hillas.distance:.3f}°")
            col6.metric("Miss Parameter", f"{hillas.miss:.3f}°")
            col7.metric("Surviving Pixels", f"{hillas.n_pixels} pix")
            col8.metric("Timing Gradient", f"{abs(hillas.time_gradient):.2f} ns/°")

            # Detailed table
            params_df = pd.DataFrame([
                {"Parameter": "Size", "Value": f"{hillas.size:.1f}", "Unit": "pe", "Physics Description": "Total reconstructed Cherenkov light intensity"},
                {"Parameter": "Centroid (x, y)", "Value": f"({hillas.x_mean:.3f}, {hillas.y_mean:.3f})", "Unit": "deg", "Physics Description": "Center of gravity of the shower in camera FoV"},
                {"Parameter": "Length", "Value": f"{hillas.length:.3f}", "Unit": "deg", "Physics Description": "RMS spread along major axis (longitudinal development)"},
                {"Parameter": "Width", "Value": f"{hillas.width:.3f}", "Unit": "deg", "Physics Description": "RMS spread along minor axis (lateral spread)"},
                {"Parameter": "Width / Length", "Value": f"{hillas.width / max(1e-4, hillas.length):.3f}", "Unit": "ratio", "Physics Description": "Shower compactness ratio"},
                {"Parameter": "Distance", "Value": f"{hillas.distance:.3f}", "Unit": "deg", "Physics Description": "Distance from camera center (pointing direction) to centroid"},
                {"Parameter": "Alpha Angle", "Value": f"{hillas.alpha:.1f}", "Unit": "deg", "Physics Description": "Angle between major axis and camera-center radius vector"},
                {"Parameter": "Miss", "Value": f"{hillas.miss:.3f}", "Unit": "deg", "Physics Description": "Perpendicular distance from camera center to major axis line"},
                {"Parameter": "Concentration", "Value": f"{hillas.concentration:.3f}", "Unit": "fraction", "Physics Description": "Fraction of total light in the two brightest pixels"},
                {"Parameter": "Surviving Pixels", "Value": f"{hillas.n_pixels}", "Unit": "pixels", "Physics Description": "Number of pixels surviving tail-cuts cleaning"},
                {"Parameter": "Arrival-Time Gradient", "Value": f"{hillas.time_gradient:.3f}", "Unit": "ns/deg", "Physics Description": "Linear propagation slope along major axis"},
                {"Parameter": "Arrival-Time RMS", "Value": f"{hillas.time_rms:.3f}", "Unit": "ns", "Physics Description": "Incoherent timing fluctuation around linear fit"},
            ])
            st.table(params_df)
        else:
            st.warning(
                f"Shower cleaning yielded insufficient surviving pixels ({hillas.n_pixels} pixels). "
                "Hillas reconstruction requires at least 3 contiguous pixels above threshold."
            )

    with tab2:
        st.subheader("Machine Learning Gamma/Hadron Discrimination")

        if model is None:
            st.info(
                "Trained classifier model (`models/classifier_rf.joblib`) not found. "
                "Run `python3 scripts/train_classifier.py` to train and save the Random Forest model."
            )
        elif not hillas.is_valid:
            st.warning("Cannot run classification: Event did not survive image cleaning.")
        else:
            feat_dict = hillas.to_feature_dict()
            feature_names = [
                "size", "length", "width", "wl_ratio", "distance",
                "alpha", "miss", "concentration", "n_pixels",
                "time_gradient", "time_rms"
            ]
            X_input = np.array([[feat_dict.get(k, np.nan) for k in feature_names]])

            prob_gamma = float(model.predict_proba(X_input)[0, 1])
            is_pred_gamma = prob_gamma >= 0.5

            col_pred1, col_pred2 = st.columns([1, 2])
            with col_pred1:
                st.metric("Predicted Gamma Probability P(γ)", f"{prob_gamma * 100:.1f}%")
                if is_pred_gamma:
                    st.success("Classification: **GAMMA-RAY** (Signal Candidate)")
                else:
                    st.error("Classification: **HADRON / PROTON** (Cosmic Ray Background)")

                st.progress(prob_gamma)
                st.caption("Standard Threshold = 50.0% | Optimized Q-factor Cut ~ 70-80%")

            with col_pred2:
                st.markdown("#### Morphological Signatures for Current Event")
                st.write(
                    f"- **Alpha Angle**: `{hillas.alpha:.1f}°` "
                    f"({'Strongly on-axis (Gamma-like)' if hillas.alpha < 20 else 'Diffuse / Off-axis (Hadron-like)'})"
                )
                st.write(
                    f"- **Width / Length Ratio**: `{hillas.width / max(1e-4, hillas.length):.3f}` "
                    f"({'Compact ellipse (Gamma-like)' if (hillas.width / max(1e-4, hillas.length)) < 0.4 else 'Broad / Clumpy (Hadron-like)'})"
                )
                st.write(
                    f"- **Timing RMS**: `{hillas.time_rms:.2f} ns` "
                    f"({'Coherent Cherenkov wavefront' if hillas.time_rms < 1.0 else 'Incoherent hadronic sub-cascades'})"
                )

    with tab3:
        st.subheader("Observatory Context: Hanle, Ladakh & The MACE / HAGAR Experiments")
        st.markdown(
            """
            ### The Major Atmospheric Cerenkov Experiment (MACE)
            - **Location**: Hanle, Ladakh, India (32°46′N, 78°58′E)
            - **Altitude**: ~4,270 metres above sea level (one of the highest astronomical sites on Earth).
            - **Telescope Aperture**: 21-metre diameter tessellated quasi-parabolic reflector (~356 m² area).
            - **Camera**: 1088 photomultiplier tubes arranged in a graded layout (0.1° inner, 0.2° outer) with ~4.0° field of view.
            - **Energy Threshold**: Down to ~20–30 GeV due to extreme altitude.

            ### Why Gamma / Hadron Separation Matters
            IACTs detect brief (~few nanosecond) flashes of blue Cherenkov radiation generated in the upper atmosphere.
            However, cosmic-ray hadrons (protons, helium nuclei) outnumber astronomical gamma rays by **1,000 : 1** or more!
            Extracting Hillas parameters and optimizing the **Telescope Quality Factor** $Q = \\epsilon_\\gamma / \\sqrt{\\epsilon_p}$
            is the cornerstone of discovering distant TeV blazars, pulsar wind nebulae, and gamma-ray bursts.
            """
        )


if __name__ == "__main__":
    main()
