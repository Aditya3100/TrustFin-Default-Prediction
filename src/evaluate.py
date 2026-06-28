"""
evaluate.py
-----------
Model evaluation module for TrustFin Bank Loan Default Risk project.

Metrics computed
----------------
- Accuracy
- Precision
- Recall
- F1 Score (macro + weighted)
- ROC-AUC
- Confusion matrix (saved as plot)

Outputs
-------
reports/metrics.json          – all numeric metrics
reports/confusion_matrix.png  – confusion matrix heatmap
reports/roc_curve.png         – ROC curve
reports/feature_importance.png – top-30 feature importances (if available)

Why accuracy alone is not enough
---------------------------------
The target variable is severely imbalanced (~8 % default vs 92 % non-default).
A trivial classifier that always predicts "no default" achieves ~92 % accuracy
while catching zero actual defaulters.  This is catastrophically bad for a
bank.  Precision and Recall capture the trade-off between false positives
(declining creditworthy customers) and false negatives (approving borrowers
who will default).  ROC-AUC measures the model's ability to discriminate
across all possible decision thresholds, making it the primary metric.
"""

import argparse
import json
import logging
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

REPORTS_DIR = Path("reports")
MODELS_DIR  = Path("models")


# ---------------------------------------------------------------------------
# Metric computation
# ---------------------------------------------------------------------------

def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
) -> Dict[str, float]:
    """
    Compute all classification metrics.

    Parameters
    ----------
    y_true    : ground-truth labels
    y_pred    : predicted labels (0/1)
    y_prob    : predicted probabilities for the positive class
    threshold : decision threshold used for binary prediction

    Returns
    -------
    dict of metric names → float values
    """
    metrics = {
        "accuracy":          round(accuracy_score(y_true, y_pred), 4),
        "precision_macro":   round(precision_score(y_true, y_pred, average="macro",    zero_division=0), 4),
        "precision_weighted":round(precision_score(y_true, y_pred, average="weighted", zero_division=0), 4),
        "recall_macro":      round(recall_score(   y_true, y_pred, average="macro",    zero_division=0), 4),
        "recall_weighted":   round(recall_score(   y_true, y_pred, average="weighted", zero_division=0), 4),
        "f1_macro":          round(f1_score(        y_true, y_pred, average="macro",   zero_division=0), 4),
        "f1_weighted":       round(f1_score(        y_true, y_pred, average="weighted",zero_division=0), 4),
        "roc_auc":           round(roc_auc_score(y_true, y_prob), 4),
        "threshold_used":    threshold,
    }
    logger.info("Metrics: %s", json.dumps(metrics, indent=2))
    return metrics


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------

def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    out_path: Path,
    labels: List[str] = ["Non-Default", "Default"],
) -> None:
    """Save a styled confusion-matrix heatmap."""
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=labels, yticklabels=labels,
        linewidths=0.5, linecolor="grey", ax=ax,
    )
    ax.set_xlabel("Predicted Label", fontsize=12)
    ax.set_ylabel("True Label",      fontsize=12)
    ax.set_title("Confusion Matrix – Loan Default Prediction", fontsize=13, fontweight="bold")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.info("Saved confusion matrix → %s", out_path)


def plot_roc_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    out_path: Path,
) -> None:
    """Save a ROC curve plot."""
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc_score = roc_auc_score(y_true, y_prob)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(fpr, tpr, color="#1a73e8", lw=2, label=f"ROC Curve (AUC = {auc_score:.4f})")
    ax.plot([0, 1], [0, 1], color="grey", lw=1, linestyle="--", label="Random Classifier")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate",  fontsize=12)
    ax.set_title("ROC Curve – Loan Default Prediction", fontsize=13, fontweight="bold")
    ax.legend(loc="lower right", fontsize=11)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.info("Saved ROC curve → %s", out_path)


def plot_feature_importance(
    model: Any,
    feature_names: List[str],
    out_path: Path,
    top_n: int = 30,
) -> None:
    """Save a top-N feature importance bar chart (works for XGBoost / LightGBM)."""
    importances: Optional[np.ndarray] = None

    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    elif hasattr(model, "coef_"):
        importances = np.abs(model.coef_[0])

    if importances is None:
        logger.warning("Model does not expose feature importances; skipping plot.")
        return

    indices = np.argsort(importances)[::-1][:top_n]
    top_features = [feature_names[i] for i in indices]
    top_values   = importances[indices]

    fig, ax = plt.subplots(figsize=(10, 8))
    y_pos = np.arange(len(top_features))
    ax.barh(y_pos, top_values[::-1], color="#1a73e8", edgecolor="white")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(top_features[::-1], fontsize=9)
    ax.set_xlabel("Importance Score", fontsize=11)
    ax.set_title(f"Top {top_n} Feature Importances", fontsize=13, fontweight="bold")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.info("Saved feature importance → %s", out_path)


# ---------------------------------------------------------------------------
# Main evaluation runner
# ---------------------------------------------------------------------------

def evaluate(
    X_test: pd.DataFrame,
    y_test: pd.Series,
    model: Any,
    feature_list: List[str],
    reports_dir: Path = REPORTS_DIR,
    threshold: float = 0.5,
) -> Dict[str, float]:
    """
    Run full evaluation pipeline.

    Parameters
    ----------
    X_test       : test features
    y_test       : test labels
    model        : fitted classifier (must expose predict_proba)
    feature_list : column names matching the model's training features
    reports_dir  : directory to write artefacts
    threshold    : decision threshold for binary classification

    Returns
    -------
    dict of metrics
    """
    reports_dir.mkdir(parents=True, exist_ok=True)

    # Align columns
    X_test = X_test.reindex(columns=feature_list, fill_value=0)

    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= threshold).astype(int)

    metrics = compute_classification_metrics(
        y_true=y_test.values,
        y_pred=y_pred,
        y_prob=y_prob,
        threshold=threshold,
    )

    # Persist metrics
    metrics_path = reports_dir / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info("Saved metrics → %s", metrics_path)

    # Plots
    plot_confusion_matrix(y_test.values, y_pred, reports_dir / "confusion_matrix.png")
    plot_roc_curve(y_test.values, y_prob, reports_dir / "roc_curve.png")
    plot_feature_importance(model, feature_list, reports_dir / "feature_importance.png")

    return metrics


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate TrustFin loan-default model.")
    parser.add_argument("--models-dir",  type=Path, default=Path("models"),  help="Path to saved artefacts")
    parser.add_argument("--reports-dir", type=Path, default=Path("reports"), help="Output directory for reports")
    parser.add_argument("--data-path",   type=Path, required=True,           help="Path to processed test CSV")
    parser.add_argument("--threshold",   type=float, default=0.5,            help="Decision threshold")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    # Load artefacts
    with open(args.models_dir / "model.pkl", "rb") as f:
        model = pickle.load(f)
    with open(args.models_dir / "feature_list.pkl", "rb") as f:
        feature_list = pickle.load(f)

    # Load test data
    df_test = pd.read_csv(args.data_path)
    X_test = df_test.drop(columns=["TARGET", "SK_ID_CURR"], errors="ignore")
    y_test = df_test["TARGET"]

    metrics = evaluate(X_test, y_test, model, feature_list,
                        reports_dir=args.reports_dir, threshold=args.threshold)
    print(json.dumps(metrics, indent=2))