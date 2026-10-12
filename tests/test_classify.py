"""
Unit tests for classifier training, Q-factor computation, and evaluation.
"""

import numpy as np
import pandas as pd
import pytest

from cherenkovlens.classify import (
    FEATURE_NAMES,
    compute_q_factor_curve,
    prepare_features,
    train_and_evaluate_classifier,
)
from cherenkovlens.dataset import generate_dataset


def test_q_factor_computation():
    # Synthetic test: perfect discriminator vs random discriminator
    y_true = np.array([1, 1, 1, 1, 0, 0, 0, 0])
    y_prob_perfect = np.array([0.9, 0.8, 0.85, 0.95, 0.1, 0.2, 0.15, 0.05])

    thresh, q_arr, best_q, best_th, best_eg, best_ep = compute_q_factor_curve(y_true, y_prob_perfect)
    assert best_q > 1.5
    assert best_eg == 1.0


def test_prepare_features_no_leakage():
    # Ensure any non-Hillas columns like 'true_size', 'amplitude', etc. are excluded
    df = pd.DataFrame(
        {
            "size": [100.0, 200.0],
            "length": [0.2, 0.3],
            "width": [0.05, 0.1],
            "wl_ratio": [0.25, 0.33],
            "distance": [0.5, 0.6],
            "alpha": [10.0, 45.0],
            "miss": [0.1, 0.2],
            "concentration": [0.3, 0.4],
            "n_pixels": [15, 25],
            "time_gradient": [1.2, 0.1],
            "time_rms": [0.3, 1.5],
            "label": [1, 0],
            "amplitude": [1500.0, 2000.0],  # Ground truth - must be dropped!
            "true_size": [120.0, 180.0],    # Ground truth - must be dropped!
        }
    )

    X, y, feature_names = prepare_features(df)
    assert "amplitude" not in feature_names
    assert "true_size" not in feature_names
    for f in feature_names:
        assert f in FEATURE_NAMES


def test_classifier_beats_chance_on_small_dataset():
    # Generate 300 events
    df = generate_dataset(n_events=300, gamma_fraction=0.5, seed=123, show_progress=False, n_jobs=1)
    assert len(df) > 150  # survived cleaning

    pipeline, results = train_and_evaluate_classifier(
        df,
        model_type="random_forest",
        test_size=0.25,
        random_state=42,
    )

    # Physical discrimination should easily beat chance (AUC >> 0.75, typically > 0.85)
    assert results.auc > 0.75
    assert results.best_q_factor > 1.2
