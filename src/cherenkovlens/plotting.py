"""
Publication-quality plotting functions for CherenkovLens.

Styles and visualizes:
- Raw, naive, and tail-cuts cleaned event displays
- Hillas ellipse overlays on camera geometry
- Cleaning efficiency and noise leakage studies
- Hillas parameter validation against simulation ground truth
- Classifier ROC curves and Quality Factor Q(theta) optimization curves
- Morphological parameter distributions (Alpha, Width vs Length)
- NSB Noise robustness benchmarks
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import pandas as pd

# Set global publication styling
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.titlesize": 14,
    "axes.grid": True,
    "grid.alpha": 0.35,
    "grid.linestyle": ":",
})

COLOR_GAMMA = "#1f77b4"     # Blue
COLOR_PROTON = "#d62728"    # Red
COLOR_TAILCUTS = "#2ca02c"  # Green
COLOR_NAIVE = "#ff7f0e"     # Orange


def plot_cleaning_comparison_study(
    raw_img: np.ndarray,
    naive_img: np.ndarray,
    tailcuts_img: np.ndarray,
    study_summary: Dict[str, Any],
    save_path: Optional[str | Path] = None,
) -> plt.Figure:
    """
    Produce a headline figure comparing Raw, Naive, and Tail-cuts cleaning.
    Top row: Event camera displays.
    Bottom row: Summary metrics (Signal Retention, Noise Leakage, Size Bias).
    """
    fig = plt.figure(figsize=(14, 9), dpi=300)
    gs = fig.add_gridspec(2, 3, height_ratios=[1.1, 1.0], hspace=0.32, wspace=0.28)

    # Top Left: Raw Noisy Image
    ax0 = fig.add_subplot(gs[0, 0])
    vmax = max(10.0, np.percentile(raw_img, 99.5))
    im0 = ax0.imshow(raw_img, origin="lower", cmap="magma", vmin=0, vmax=vmax)
    ax0.set_title("1. Raw Image (Signal + NSB)", fontweight="bold")
    ax0.set_xlabel("Pixel X")
    ax0.set_ylabel("Pixel Y")
    fig.colorbar(im0, ax=ax0, shrink=0.75, label="Intensity (pe)")

    # Top Middle: Naive Cleaning
    ax1 = fig.add_subplot(gs[0, 1])
    im1 = ax1.imshow(naive_img, origin="lower", cmap="magma", vmin=0, vmax=vmax)
    ax1.set_title("2. Naive Single-Threshold Cleaning", fontweight="bold")
    ax1.set_xlabel("Pixel X")
    fig.colorbar(im1, ax=ax1, shrink=0.75, label="Intensity (pe)")

    # Top Right: Tail-cuts Cleaning
    ax2 = fig.add_subplot(gs[0, 2])
    im2 = ax2.imshow(tailcuts_img, origin="lower", cmap="magma", vmin=0, vmax=vmax)
    ax2.set_title("3. Tail-Cuts Cleaning (6/3 Core/Bnd)", fontweight="bold")
    ax2.set_xlabel("Pixel X")
    fig.colorbar(im2, ax=ax2, shrink=0.75, label="Intensity (pe)")

    # Bottom Left: Signal Retention Rate (%)
    ax3 = fig.add_subplot(gs[1, 0])
    methods = ["Naive", "Tail-Cuts"]
    sig_ret = [study_summary["naive_signal_retention"] * 100, study_summary["tailcuts_signal_retention"] * 100]
    bars3 = ax3.bar(methods, sig_ret, color=[COLOR_NAIVE, COLOR_TAILCUTS], width=0.55, edgecolor="black", alpha=0.85)
    ax3.set_ylabel("Signal Retention (%)")
    ax3.set_ylim(0, 105)
    ax3.set_title("Signal Pixel Recovery Rate", fontweight="bold")
    for bar in bars3:
        h = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width() / 2.0, h + 2.0, f"{h:.1f}%", ha="center", va="bottom", fontweight="bold")

    # Bottom Middle: Noise Pixels Leaking Through
    ax4 = fig.add_subplot(gs[1, 1])
    noise_leak = [study_summary["naive_noise_leakage_mean"], study_summary["tailcuts_noise_leakage_mean"]]
    bars4 = ax4.bar(methods, noise_leak, color=[COLOR_NAIVE, COLOR_TAILCUTS], width=0.55, edgecolor="black", alpha=0.85)
    ax4.set_ylabel("Noise Pixels / Event")
    ax4.set_title("NSB Noise Leakage (Lower is Better)", fontweight="bold")
    for bar in bars4:
        h = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width() / 2.0, h + max(0.5, h * 0.05), f"{h:.2f}", ha="center", va="bottom", fontweight="bold")

    # Bottom Right: Size Bias (%)
    ax5 = fig.add_subplot(gs[1, 2])
    size_bias = [study_summary["naive_size_bias_pct"], study_summary["tailcuts_size_bias_pct"]]
    bars5 = ax5.bar(methods, size_bias, color=[COLOR_NAIVE, COLOR_TAILCUTS], width=0.55, edgecolor="black", alpha=0.85)
    ax5.axhline(0, color="gray", linestyle="--", linewidth=1.2)
    ax5.set_ylabel("Size Relative Bias (%)")
    ax5.set_title("Total Size Reconstruction Bias", fontweight="bold")
    for bar in bars5:
        h = bar.get_height()
        y_pos = h + 1.5 if h >= 0 else h - 4.5
        ax5.text(bar.get_x() + bar.get_width() / 2.0, y_pos, f"{h:+.1f}%", ha="center", va="bottom", fontweight="bold")

    fig.suptitle("IACT Image Cleaning Comparison: Tail-Cuts vs Naive Thresholding", fontsize=15, fontweight="bold", y=0.98)

    if save_path:
        p = Path(save_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(p, bbox_inches="tight")

    return fig


def plot_hillas_validation(
    df_val: pd.DataFrame,
    save_path: Optional[str | Path] = None,
) -> plt.Figure:
    """
    Plot validation of reconstructed Hillas parameters against true simulation values.
    Shows 4 panels: Length, Width, Centroid Distance, and Alpha angle.
    """
    fig, axes = plt.subplots(2, 2, figsize=(11, 10), dpi=300)
    axes = axes.flatten()

    params_to_plot = [
        ("true_length", "length", "Length (deg)", axes[0]),
        ("true_width", "width", "Width (deg)", axes[1]),
        ("true_distance", "distance", "Distance (deg)", axes[2]),
        ("true_alpha", "alpha", "Alpha Angle (deg)", axes[3]),
    ]

    for true_col, meas_col, label, ax in params_to_plot:
        v_true = df_val[true_col].to_numpy()
        v_meas = df_val[meas_col].to_numpy()

        # Scatter
        ax.scatter(v_true, v_meas, alpha=0.45, color="#2b5c8f", edgecolors="none", s=25, label="Simulated events")
        
        # 1-to-1 diagonal
        min_v = min(np.min(v_true), np.min(v_meas))
        max_v = max(np.max(v_true), np.max(v_meas))
        ax.plot([min_v, max_v], [min_v, max_v], "r--", linewidth=1.8, label="Ideal 1:1 line")

        # Correlation coefficient
        valid = np.isfinite(v_true) & np.isfinite(v_meas)
        if np.sum(valid) > 2:
            r = np.corrcoef(v_true[valid], v_meas[valid])[0, 1]
            ax.text(0.05, 0.90, f"Pearson r = {r:.3f}", transform=ax.transAxes,
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.85),
                    fontweight="bold")

        ax.set_xlabel(f"True {label}")
        ax.set_ylabel(f"Extracted {label}")
        ax.set_title(f"Hillas {label.split('(')[0].strip()}", fontweight="bold")
        ax.legend(loc="lower right")

    fig.suptitle("Hillas Parameter Validation: Reconstructed vs True Simulation Values", fontsize=14, fontweight="bold", y=0.99)
    plt.tight_layout()

    if save_path:
        p = Path(save_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(p, bbox_inches="tight")

    return fig


def plot_roc_curve(
    results_dict: Dict[str, Any],
    save_path: Optional[str | Path] = None,
) -> plt.Figure:
    """Plot publication-quality ROC curve for one or multiple trained models."""
    fig, ax = plt.subplots(figsize=(7, 6), dpi=300)

    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#9467bd"]
    for idx, (m_name, res) in enumerate(results_dict.items()):
        c = colors[idx % len(colors)]
        ax.plot(res["fpr"], res["tpr"], color=c, lw=2.2, label=f"{m_name} (AUC = {res['auc']:.4f})")

    ax.plot([0, 1], [0, 1], "k--", lw=1.2, alpha=0.7, label="Chance (AUC = 0.5000)")
    ax.set_xlim([-0.01, 1.01])
    ax.set_ylim([-0.01, 1.01])
    ax.set_xlabel("False Positive Rate (Proton Survival Fraction)", fontweight="bold")
    ax.set_ylabel("True Positive Rate (Gamma Efficiency)", fontweight="bold")
    ax.set_title("IACT Gamma / Hadron Classification ROC Curves", fontweight="bold")
    ax.legend(loc="lower right", frameon=True)

    if save_path:
        p = Path(save_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(p, bbox_inches="tight")

    return fig


def plot_q_factor_curve(
    results_dict: Dict[str, Any],
    save_path: Optional[str | Path] = None,
) -> plt.Figure:
    """
    Plot Quality Factor Q(theta) vs classifier threshold.
    Marks the peak Q-factor and optimal selection cut.
    """
    fig, ax = plt.subplots(figsize=(8, 6), dpi=300)

    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]
    for idx, (m_name, res) in enumerate(results_dict.items()):
        c = colors[idx % len(colors)]
        threshs = res["q_thresholds"]
        q_vals = res["q_factors"]
        best_q = res["best_q_factor"]
        best_th = res["best_threshold"]

        ax.plot(threshs, q_vals, color=c, lw=2.2, label=f"{m_name} (Peak Q = {best_q:.2f})")
        ax.axvline(best_th, color=c, linestyle=":", alpha=0.6)
        ax.scatter([best_th], [best_q], color=c, s=70, zorder=5)
        ax.annotate(
            f"Cut = {best_th:.2f}\nQ = {best_q:.2f}",
            xy=(best_th, best_q),
            xytext=(best_th + 0.05, best_q * 0.92),
            arrowprops=dict(arrowstyle="->", color=c, lw=1.2),
            fontweight="bold",
            fontsize=9,
        )

    ax.set_xlabel("Classifier Gamma Probability Cut", fontweight="bold")
    ax.set_ylabel(r"Quality Factor $Q = \epsilon_\gamma / \sqrt{\epsilon_p}$", fontweight="bold")
    ax.set_title("Telescope Quality Factor Optimization", fontweight="bold")
    ax.legend(loc="upper left", frameon=True)
    ax.set_xlim([0.0, 1.0])

    if save_path:
        p = Path(save_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(p, bbox_inches="tight")

    return fig


def plot_alpha_distribution(
    df: pd.DataFrame,
    save_path: Optional[str | Path] = None,
) -> plt.Figure:
    """Plot Alpha angle distributions for gammas vs hadrons."""
    fig, ax = plt.subplots(figsize=(7.5, 5.5), dpi=300)

    gamma_alpha = df[df["label"] == 1]["alpha"].dropna()
    hadron_alpha = df[df["label"] == 0]["alpha"].dropna()

    bins = np.linspace(0, 90, 31)
    ax.hist(gamma_alpha, bins=bins, density=True, alpha=0.65, color=COLOR_GAMMA, label=r"$\gamma$-ray Showers (On-axis Pointing)", edgecolor="black")
    ax.hist(hadron_alpha, bins=bins, density=True, alpha=0.55, color=COLOR_PROTON, label="Hadron / Proton Showers (Isotropic)", edgecolor="black")

    ax.set_xlabel(r"Alpha Angle $\alpha$ (degrees)", fontweight="bold")
    ax.set_ylabel("Probability Density", fontweight="bold")
    ax.set_title(r"Distribution of Hillas $\alpha$ Parameter", fontweight="bold")
    ax.set_xlim([0, 90])
    ax.legend(loc="upper right", frameon=True)

    if save_path:
        p = Path(save_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(p, bbox_inches="tight")

    return fig


def plot_width_length_scatter(
    df: pd.DataFrame,
    sample_size: int = 2500,
    save_path: Optional[str | Path] = None,
) -> plt.Figure:
    """Scatter plot of Width vs Length colored by class."""
    fig, ax = plt.subplots(figsize=(7.5, 6), dpi=300)

    sample = df.sample(min(len(df), sample_size), random_state=42)
    gammas = sample[sample["label"] == 1]
    protons = sample[sample["label"] == 0]

    ax.scatter(protons["length"], protons["width"], color=COLOR_PROTON, alpha=0.45, s=20, label="Proton / Hadron")
    ax.scatter(gammas["length"], gammas["width"], color=COLOR_GAMMA, alpha=0.55, s=20, label=r"$\gamma$-ray")

    ax.set_xlabel("Hillas Length (deg)", fontweight="bold")
    ax.set_ylabel("Hillas Width (deg)", fontweight="bold")
    ax.set_title("Shower Morphology: Width vs Length", fontweight="bold")
    ax.legend(loc="upper left", frameon=True)

    if save_path:
        p = Path(save_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(p, bbox_inches="tight")

    return fig


def plot_feature_importance(
    feat_imp: Dict[str, float],
    save_path: Optional[str | Path] = None,
) -> plt.Figure:
    """Plot feature importance ranking bar chart."""
    fig, ax = plt.subplots(figsize=(8, 5.5), dpi=300)

    sorted_items = sorted(feat_imp.items(), key=lambda x: x[1])
    names = [x[0] for x in sorted_items]
    vals = [x[1] for x in sorted_items]

    y_pos = np.arange(len(names))
    ax.barh(y_pos, vals, color="#2b5c8f", edgecolor="black", alpha=0.85)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, fontweight="bold")
    ax.set_xlabel("Relative Importance", fontweight="bold")
    ax.set_title("Classifier Feature Importances", fontweight="bold")

    if save_path:
        p = Path(save_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(p, bbox_inches="tight")

    return fig


def plot_noise_robustness_study(
    df_noise: pd.DataFrame,
    save_path: Optional[str | Path] = None,
) -> plt.Figure:
    """
    Plot classifier AUC and Q-factor degradation across NSB multipliers
    with and without tail-cuts cleaning.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5), dpi=300)

    nsb_levels = df_noise["nsb_mult"]

    # Left: AUC vs NSB Level
    ax1.plot(nsb_levels, df_noise["auc_tailcuts"], "o-", color=COLOR_TAILCUTS, lw=2.2, label="With Tail-Cuts Cleaning")
    ax1.plot(nsb_levels, df_noise["auc_naive"], "s--", color=COLOR_NAIVE, lw=2.2, label="Naive / No Tail-Cuts")
    ax1.set_xlabel("NSB Noise Level Multiplier (× Nominal)", fontweight="bold")
    ax1.set_ylabel("ROC AUC", fontweight="bold")
    ax1.set_title("Classification AUC vs NSB Noise Level", fontweight="bold")
    ax1.set_xticks(nsb_levels)
    ax1.legend(loc="lower left", frameon=True)
    ax1.set_ylim([0.45, 1.0])

    # Right: Q-factor vs NSB Level
    ax2.plot(nsb_levels, df_noise["q_tailcuts"], "o-", color=COLOR_TAILCUTS, lw=2.2, label="With Tail-Cuts Cleaning")
    ax2.plot(nsb_levels, df_noise["q_naive"], "s--", color=COLOR_NAIVE, lw=2.2, label="Naive / No Tail-Cuts")
    ax2.set_xlabel("NSB Noise Level Multiplier (× Nominal)", fontweight="bold")
    ax2.set_ylabel("Best Quality Factor (Q)", fontweight="bold")
    ax2.set_title("Quality Factor (Q) vs NSB Noise Level", fontweight="bold")
    ax2.set_xticks(nsb_levels)
    ax2.legend(loc="upper right", frameon=True)

    fig.suptitle("Night Sky Background (NSB) Noise Robustness Study", fontsize=14, fontweight="bold", y=0.98)
    plt.tight_layout()

    if save_path:
        p = Path(save_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(p, bbox_inches="tight")

    return fig


def plot_live_event_quad(
    raw_img: np.ndarray,
    clean_img: np.ndarray,
    hillas_obj: Any,
    label: int,
    pixel_size_deg: float = 0.1,
    save_path: Optional[str | Path] = None,
) -> plt.Figure:
    """
    Generate a 3-panel live visualization:
    1. Raw noisy camera image
    2. Cleaned camera image
    3. Cleaned image with Hillas ellipse & major axis vector overlay
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), dpi=200)

    title_kind = "Gamma-ray" if label == 1 else "Hadron / Proton"
    fig.suptitle(f"Event Display: {title_kind} Shower", fontsize=14, fontweight="bold", y=1.02)

    vmax = max(10.0, np.percentile(raw_img, 99.5))

    # Panel 1: Raw
    axes[0].imshow(raw_img, origin="lower", cmap="viridis", vmin=0, vmax=vmax)
    axes[0].set_title("1. Raw Image + NSB Noise", fontweight="bold")
    axes[0].set_xlabel("Pixel X")
    axes[0].set_ylabel("Pixel Y")

    # Panel 2: Cleaned
    axes[1].imshow(clean_img, origin="lower", cmap="viridis", vmin=0, vmax=vmax)
    axes[1].set_title("2. Tail-Cuts Cleaned", fontweight="bold")
    axes[1].set_xlabel("Pixel X")

    # Panel 3: Cleaned with Hillas overlay
    axes[2].imshow(clean_img, origin="lower", cmap="viridis", vmin=0, vmax=vmax)
    axes[2].set_title("3. Hillas Ellipse Reconstruction", fontweight="bold")
    axes[2].set_xlabel("Pixel X")

    if hillas_obj is not None and hillas_obj.is_valid:
        # Convert physical degrees back to pixel coordinates for overlay
        # x_deg = (col - (N-1)/2) * pixel_size_deg => col = x_deg / pixel_size_deg + (N-1)/2
        N = raw_img.shape[0]
        offset = (N - 1) / 2.0
        col_c = hillas_obj.x_mean / pixel_size_deg + offset
        row_c = hillas_obj.y_mean / pixel_size_deg + offset

        len_pix = hillas_obj.length / pixel_size_deg
        wid_pix = hillas_obj.width / pixel_size_deg
        phi_deg = hillas_obj.phi_deg

        # Matplotlib Ellipse: width is along x before angle rotation (deg)
        # Note: angle in degrees anti-clockwise
        ellipse = patches.Ellipse(
            (col_c, row_c),
            width=2.0 * len_pix,
            height=2.0 * wid_pix,
            angle=phi_deg,
            fill=False,
            edgecolor="red",
            linewidth=2.2,
            linestyle="-",
            label="Hillas 1-sigma Ellipse",
        )
        axes[2].add_patch(ellipse)

        # Plot centroid
        axes[2].plot(col_c, row_c, "r+", markersize=10, markeredgewidth=2.2, label="Centroid")

        # Plot major axis vector
        dx_vec = len_pix * np.cos(np.radians(phi_deg))
        dy_vec = len_pix * np.sin(np.radians(phi_deg))
        axes[2].plot([col_c - dx_vec, col_c + dx_vec], [row_c - dy_vec, row_c + dy_vec], "r-", lw=1.8, label="Major Axis")

        # Plot line connecting camera center to centroid (for Alpha angle)
        axes[2].plot([offset, col_c], [offset, row_c], "c--", lw=1.5, label="Center-to-Centroid")
        axes[2].plot(offset, offset, "co", markersize=6, label="Camera Center")
        axes[2].legend(loc="upper right", fontsize=8, framealpha=0.8)

    plt.tight_layout()
    if save_path:
        p = Path(save_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(p, bbox_inches="tight")

    return fig
