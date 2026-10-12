"""
Gamma / Hadron classification models, evaluation, and Q-factor optimization.

Features used:
- Hillas parameters: size, length, width, wl_ratio, distance, alpha, miss, concentration, n_pixels
- Arrival-time features: time_gradient, time_rms

Strictly avoids leakage of ground-truth simulation variables.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    auc,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

FEATURE_NAMES = [
    "size",
    "length",
    "width",
    "wl_ratio",
    "distance",
    "alpha",
    "miss",
    "concentration",
    "n_pixels",
    "time_gradient",
    "time_rms",
]


@dataclass
class ClassificationResults:
    """Evaluation metrics for a trained classifier."""

    model_name: str
    auc: float
    accuracy: float
    f1: float
    precision: float
    recall: float
    best_q_factor: float
    best_threshold: float
    gamma_efficiency_at_best_q: float
    proton_survival_at_best_q: float
    fpr: List[float]
    tpr: List[float]
    roc_thresholds: List[float]
    q_thresholds: List[float]
    q_factors: List[float]
    confusion_matrix: List[List[int]]
    feature_importances: Dict[str, float]


def compute_q_factor_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    thresholds: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, float, float, float, float]:
    """
    Compute the IACT telescope Quality Factor Q(theta) across classifier thresholds.

    Definition:
        Q(theta) = epsilon_gamma(theta) / sqrt(epsilon_proton(theta))

    Where:
        epsilon_gamma = True Positive Rate (gamma efficiency)
        epsilon_proton = False Positive Rate (hadron / proton survival fraction)

    A higher Q-factor boosts signal significance by a factor of Q:
        Significance_after = Q * Significance_before

    Parameters
    ----------
    y_true : np.ndarray
        True binary labels (1=gamma, 0=proton).
    y_prob : np.ndarray
        Predicted gamma probabilities.
    thresholds : np.ndarray, optional
        Array of threshold values in (0, 1). Default: 100 points between 0.01 and 0.99.

    Returns
    -------
    thresh_array : np.ndarray
    q_factors : np.ndarray
    best_q : float
    best_thresh : float
    best_eps_gamma : float
    best_eps_proton : float
    """
    if thresholds is None:
        thresholds = np.linspace(0.01, 0.99, 100)

    n_gamma = np.sum(y_true == 1)
    n_proton = np.sum(y_true == 0)

    q_vals = []
    eps_g_list = []
    eps_p_list = []

    for th in thresholds:
        pred_gamma = y_prob >= th
        tp = np.sum((pred_gamma) & (y_true == 1))
        fp = np.sum((pred_gamma) & (y_true == 0))

        eps_gamma = tp / max(1, n_gamma)
        eps_proton = fp / max(1, n_proton)

        eps_g_list.append(eps_gamma)
        eps_p_list.append(eps_proton)

        # Regularize to prevent division by zero
        if eps_proton > 0:
            q = eps_gamma / np.sqrt(eps_proton)
        else:
            q = 0.0
        q_vals.append(q)

    q_array = np.array(q_vals)
    best_idx = int(np.argmax(q_array))
    best_q = float(q_array[best_idx])
    best_thresh = float(thresholds[best_idx])
    best_eps_gamma = float(eps_g_list[best_idx])
    best_eps_proton = float(eps_p_list[best_idx])

    return thresholds, q_array, best_q, best_thresh, best_eps_gamma, best_eps_proton


def prepare_features(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Extract strictly vetted features and labels from DataFrame.
    Guarantees no leakage of simulation parameters.
    """
    available_features = [col for col in FEATURE_NAMES if col in df.columns]
    X = df[available_features].to_numpy(dtype=np.float64)
    y = df["label"].to_numpy(dtype=np.int64)

    return X, y, available_features


def train_and_evaluate_classifier(
    df: pd.DataFrame,
    model_type: str = "random_forest",
    test_size: float = 0.2,
    val_size: float = 0.1,
    random_state: int = 42,
    save_model_path: Optional[str | Path] = None,
) -> Tuple[Pipeline, ClassificationResults]:
    """
    Train a gamma/hadron classifier with stratified splitting, imputing, and evaluation.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset DataFrame.
    model_type : {'random_forest', 'gradient_boosting', 'logistic_regression'}
        Type of classifier to train.
    test_size : float, default=0.2
        Test split proportion.
    val_size : float, default=0.1
        Validation split proportion.
    random_state : int, default=42
        Seed for reproducibility.
    save_model_path : str or Path, optional
        Path to save the serialized trained pipeline.

    Returns
    -------
    pipeline : Pipeline
    results : ClassificationResults
    """
    X, y, feature_names = prepare_features(df)

    # Stratified Train-Val-Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # Define base estimator
    if model_type == "random_forest":
        clf = RandomForestClassifier(
            n_estimators=140,
            max_depth=12,
            min_samples_split=6,
            random_state=random_state,
            n_jobs=-1,
        )
    elif model_type == "gradient_boosting":
        clf = HistGradientBoostingClassifier(
            max_iter=150,
            max_depth=6,
            learning_rate=0.08,
            random_state=random_state,
        )
    elif model_type == "logistic_regression":
        clf = LogisticRegression(max_iter=1000, random_state=random_state)
    else:
        raise ValueError(f"Unsupported model_type: '{model_type}'")

    pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("classifier", clf),
        ]
    )

    pipeline.fit(X_train, y_train)

    # Predictions on test set
    y_prob = pipeline.predict_proba(X_test)[:, 1]
    y_pred_default = (y_prob >= 0.5).astype(int)

    # Standard metrics
    roc_auc = float(roc_auc_score(y_test, y_prob))
    acc = float(accuracy_score(y_test, y_pred_default))
    f1 = float(f1_score(y_test, y_pred_default))
    prec = float(precision_score(y_test, y_pred_default, zero_division=0))
    rec = float(recall_score(y_test, y_pred_default))

    # ROC Curve
    fpr, tpr, roc_thresh = roc_curve(y_test, y_prob)

    # Q-factor Curve
    q_threshs, q_vals, best_q, best_th, best_eg, best_ep = compute_q_factor_curve(y_test, y_prob)

    # Confusion matrix at best Q-factor threshold
    y_pred_best = (y_prob >= best_th).astype(int)
    cm = confusion_matrix(y_test, y_pred_best).tolist()

    # Feature importances
    feat_imp = {}
    if hasattr(clf, "feature_importances_"):
        for name, imp in zip(feature_names, clf.feature_importances_):
            feat_imp[name] = float(imp)
    else:
        # Fallback to permutation importance
        perm_res = permutation_importance(pipeline, X_test, y_test, n_repeats=5, random_state=random_state)
        for name, imp in zip(feature_names, perm_res.importances_mean):
            feat_imp[name] = float(imp)

    results = ClassificationResults(
        model_name=model_type,
        auc=roc_auc,
        accuracy=acc,
        f1=f1,
        precision=prec,
        recall=rec,
        best_q_factor=best_q,
        best_threshold=best_th,
        gamma_efficiency_at_best_q=best_eg,
        proton_survival_at_best_q=best_ep,
        fpr=fpr.tolist(),
        tpr=tpr.tolist(),
        roc_thresholds=roc_thresh.tolist(),
        q_thresholds=q_threshs.tolist(),
        q_factors=q_vals.tolist(),
        confusion_matrix=cm,
        feature_importances=feat_imp,
    )

    if save_model_path is not None:
        p = Path(save_model_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(pipeline, p)

    return pipeline, results


def save_metrics_to_json(metrics: Dict[str, Any], filepath: str | Path) -> None:
    """Save metrics dictionary to formatted JSON."""
    p = Path(filepath)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
