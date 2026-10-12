#!/usr/bin/env python3
"""
Training and evaluation script for IACT Gamma/Hadron Classifiers.

Trains:
- Random Forest Classifier (primary)
- Histogram Gradient Boosting Classifier
- Logistic Regression (linear baseline)

Evaluates:
- ROC Curves and AUC
- Quality Factor Q(theta) = epsilon_gamma / sqrt(epsilon_proton)
- Morphological parameter distributions (Alpha, Width vs Length)
- Feature Importances

Saves:
- models/classifier_rf.joblib
- results/metrics.json
- figures/roc_curve.png
- figures/q_factor_curve.png
- figures/alpha_distribution.png
- figures/width_length_scatter.png
- figures/feature_importance.png
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict

import pandas as pd

from cherenkovlens.classify import (
    save_metrics_to_json,
    train_and_evaluate_classifier,
)
from cherenkovlens.dataset import generate_dataset
from cherenkovlens.plotting import (
    plot_alpha_distribution,
    plot_feature_importance,
    plot_q_factor_curve,
    plot_roc_curve,
    plot_width_length_scatter,
)
from cherenkovlens.utils import ensure_dirs, get_project_root, load_config, setup_logger


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and evaluate Gamma/Hadron classifiers.")
    parser.add_argument("--data-path", type=str, default=None, help="Path to input dataset (default: data/dataset.csv)")
    parser.add_argument("--seed", type=int, default=42, help="Seed (default: 42)")
    args = parser.parse_args()

    logger = setup_logger("train_classifier")
    root = get_project_root()
    cfg = load_config()

    figures_dir = root / "figures"
    results_dir = root / "results"
    models_dir = root / "models"
    data_dir = root / "data"
    ensure_dirs(figures_dir, results_dir, models_dir, data_dir)

    data_file = Path(args.data_path) if args.data_path else data_dir / "dataset.csv"
    if not data_file.exists():
        logger.info(f"Dataset not found at {data_file}. Generating dataset now...")
        df = generate_dataset(
            n_events=cfg.get("dataset", {}).get("n_events", 40000),
            gamma_fraction=0.5,
            config=cfg,
            output_path=data_file,
            seed=args.seed,
            show_progress=True,
        )
    else:
        logger.info(f"Loading existing dataset from {data_file}...")
        df = pd.read_csv(data_file)

    logger.info(f"Loaded {len(df)} events ({df['label'].sum()} gammas, {len(df) - df['label'].sum()} protons).")

    # Plot baseline morphological distributions
    logger.info("Plotting parameter distributions (Alpha, Width vs Length)...")
    plot_alpha_distribution(df, save_path=figures_dir / "alpha_distribution.png")
    plot_width_length_scatter(df, save_path=figures_dir / "width_length_scatter.png")

    models_to_train = ["random_forest", "gradient_boosting", "logistic_regression"]
    all_results: Dict[str, Any] = {}
    best_rf_pipeline = None

    for m_type in models_to_train:
        logger.info(f"Training {m_type}...")
        save_path = models_dir / "classifier_rf.joblib" if m_type == "random_forest" else None
        pipeline, res = train_and_evaluate_classifier(
            df=df,
            model_type=m_type,
            test_size=0.2,
            random_state=args.seed,
            save_model_path=save_path,
        )
        if m_type == "random_forest":
            best_rf_pipeline = pipeline

        all_results[m_type] = {
            "model_name": res.model_name,
            "auc": res.auc,
            "accuracy": res.accuracy,
            "f1": res.f1,
            "precision": res.precision,
            "recall": res.recall,
            "best_q_factor": res.best_q_factor,
            "best_threshold": res.best_threshold,
            "gamma_efficiency_at_best_q": res.gamma_efficiency_at_best_q,
            "proton_survival_at_best_q": res.proton_survival_at_best_q,
            "fpr": res.fpr,
            "tpr": res.tpr,
            "q_thresholds": res.q_thresholds,
            "q_factors": res.q_factors,
            "confusion_matrix": res.confusion_matrix,
            "feature_importances": res.feature_importances,
        }

        logger.info(
            f"[{m_type}] AUC = {res.auc:.4f} | Peak Q = {res.best_q_factor:.2f} "
            f"(at cut={res.best_threshold:.2f}, eps_g={res.gamma_efficiency_at_best_q*100:.1f}%, eps_p={res.proton_survival_at_best_q*100:.2f}%)"
        )

    # Plot combined ROC and Q-factor curves
    logger.info("Generating ROC and Q-factor comparison curves...")
    plot_roc_curve(all_results, save_path=figures_dir / "roc_curve.png")
    plot_q_factor_curve(all_results, save_path=figures_dir / "q_factor_curve.png")

    # Plot Random Forest Feature Importance
    if "random_forest" in all_results:
        plot_feature_importance(
            all_results["random_forest"]["feature_importances"],
            save_path=figures_dir / "feature_importance.png",
        )

    # Save summary metrics to results/metrics.json
    metrics_summary = {
        m_type: {
            "auc": all_results[m_type]["auc"],
            "accuracy": all_results[m_type]["accuracy"],
            "f1": all_results[m_type]["f1"],
            "precision": all_results[m_type]["precision"],
            "recall": all_results[m_type]["recall"],
            "best_q_factor": all_results[m_type]["best_q_factor"],
            "best_threshold": all_results[m_type]["best_threshold"],
            "gamma_efficiency_at_best_q": all_results[m_type]["gamma_efficiency_at_best_q"],
            "proton_survival_at_best_q": all_results[m_type]["proton_survival_at_best_q"],
            "confusion_matrix": all_results[m_type]["confusion_matrix"],
            "feature_importances": all_results[m_type]["feature_importances"],
        }
        for m_type in models_to_train
    }

    metrics_file = results_dir / "metrics.json"
    save_metrics_to_json(metrics_summary, metrics_file)
    logger.info(f"Saved metrics summary to {metrics_file}")
    logger.info(f"Saved primary model pipeline to {models_dir / 'classifier_rf.joblib'}")


if __name__ == "__main__":
    main()
