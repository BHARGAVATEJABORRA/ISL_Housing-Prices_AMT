"""Metric helpers for regression and classification."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable

import numpy as np
from sklearn import metrics


@dataclass
class RegressionMetrics:
    rmse: float
    rmsle: float
    mae: float
    r2: float


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> RegressionMetrics:
    """Compute common regression metrics."""

    rmse = metrics.mean_squared_error(y_true, y_pred, squared=False)
    rmsle = metrics.mean_squared_log_error(
        np.clip(y_true, a_min=0, a_max=None),
        np.clip(y_pred, a_min=0, a_max=None),
    ) ** 0.5
    mae = metrics.mean_absolute_error(y_true, y_pred)
    r2 = metrics.r2_score(y_true, y_pred)
    return RegressionMetrics(rmse=rmse, rmsle=rmsle, mae=mae, r2=r2)


@dataclass
class ClassificationMetrics:
    accuracy: float
    macro_f1: float
    confusion_matrix: np.ndarray
    labels: Iterable[str]


def classification_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, labels: Iterable[str]
) -> ClassificationMetrics:
    """Compute classification metrics."""

    accuracy = metrics.accuracy_score(y_true, y_pred)
    macro_f1 = metrics.f1_score(y_true, y_pred, average="macro")
    confusion = metrics.confusion_matrix(y_true, y_pred, labels=labels)
    return ClassificationMetrics(
        accuracy=accuracy, macro_f1=macro_f1, confusion_matrix=confusion, labels=labels
    )


def to_dict(metrics_obj) -> Dict[str, float]:  # pragma: no cover - convenience
    """Convert a dataclass metrics object to dictionary."""

    return {field: getattr(metrics_obj, field) for field in metrics_obj.__dataclass_fields__}
