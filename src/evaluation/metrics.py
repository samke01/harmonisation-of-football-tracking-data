"""Evaluation metrics for downstream classification and regression tasks."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)


def _evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
    }
    labels = np.unique(y_true)
    if len(labels) == 2 and np.issubdtype(np.asarray(y_true).dtype, np.number):
        metrics["positive_rate_true"] = float(np.mean(y_true))
        metrics["positive_rate_pred"] = float(np.mean(y_pred))
    return metrics


def _evaluate_regression(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
        "target_mean": float(np.mean(y_true)),
        "target_std": float(np.std(y_true)),
    }
