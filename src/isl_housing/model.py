"""Model training utilities."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.base import clone
from sklearn.linear_model import Ridge, RidgeClassifier
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline

from .data import load_raw
from .features import FeatureConfig, build_preprocessor
from .metrics import (
    ClassificationMetrics,
    RegressionMetrics,
    classification_metrics,
    regression_metrics,
)
from .utils import safe_expm1, safe_log1p


@dataclass
class TrainConfig:
    data_path: Path
    target: str
    log_target: bool = True
    rare_threshold: float = 0.01
    scale_numeric: bool = True
    model_type: str = "xgboost"
    model_params: Optional[Dict[str, Any]] = None
    cv_folds: int = 5
    random_state: int = 42
    artifact_dir: Path = Path("models")
    conformal_alpha: float = 0.1
    price_bins: Optional[List[float]] = None


@dataclass
class TrainResult:
    pipeline: Pipeline
    regression_metrics: RegressionMetrics
    classification_metrics: Optional[ClassificationMetrics]
    conformal_quantile: Optional[float]


def _get_regressor(model_type: str, params: Optional[Dict[str, Any]] = None):
    params = params or {}
    model_type = model_type.lower()
    if model_type == "xgboost":
        try:
            from xgboost import XGBRegressor

            return XGBRegressor(
                random_state=42,
                objective="reg:squarederror",
                **params,
            )
        except ImportError:  # pragma: no cover - fallback when dependency missing
            return Ridge(random_state=42)
    if model_type == "ridge":
        return Ridge(random_state=42, **params)
    if model_type == "lightgbm":
        try:
            from lightgbm import LGBMRegressor

            return LGBMRegressor(random_state=42, **params)
        except ImportError:  # pragma: no cover
            return Ridge(random_state=42)
    if model_type == "catboost":
        try:
            from catboost import CatBoostRegressor

            return CatBoostRegressor(verbose=False, random_seed=42, **params)
        except ImportError:  # pragma: no cover
            return Ridge(random_state=42)
    raise ValueError(f"Unsupported model type: {model_type}")


def _get_classifier(params: Optional[Dict[str, Any]] = None):
    params = params or {}
    try:
        from lightgbm import LGBMClassifier

        return LGBMClassifier(random_state=42, **params)
    except ImportError:  # pragma: no cover
        return RidgeClassifier(random_state=42, **params)


def _bin_targets(y: np.ndarray, bins: List[float]) -> np.ndarray:
    labels = [f"bin_{i}" for i in range(1, len(bins))]
    return pd.cut(y, bins=bins, labels=labels, include_lowest=True)


def train(config: TrainConfig) -> TrainResult:
    """Train the regression (and optional classification) pipeline."""

    df = load_raw(config.data_path, target=config.target)
    target_col = config.target.strip().lower()
    y = df[target_col].values.astype(float)
    X = df.drop(columns=[target_col])

    if config.log_target:
        y_transformed = safe_log1p(y)
    else:
        y_transformed = y

    feature_config = FeatureConfig(
        rare_threshold=config.rare_threshold, scale_numeric=config.scale_numeric
    )
    preprocessor = build_preprocessor(X, feature_config)
    regressor = _get_regressor(config.model_type, config.model_params)
    base_pipeline = Pipeline(
        steps=[("preprocessor", preprocessor), ("model", regressor)]
    )

    cv = KFold(n_splits=config.cv_folds, shuffle=True, random_state=config.random_state)
    oof_predictions = np.zeros_like(y_transformed, dtype=float)

    for train_idx, val_idx in cv.split(X, y_transformed):
        cv_pipeline = clone(base_pipeline)
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train = y_transformed[train_idx]
        cv_pipeline.fit(X_train, y_train)
        preds = cv_pipeline.predict(X_val)
        oof_predictions[val_idx] = preds

    final_pipeline = clone(base_pipeline)
    final_pipeline.fit(X, y_transformed)

    if config.log_target:
        predictions = safe_expm1(oof_predictions)
    else:
        predictions = oof_predictions

    reg_metrics = regression_metrics(y, predictions)

    conformal_quantile: Optional[float] = None
    if config.conformal_alpha and 0 < config.conformal_alpha < 1:
        residuals = np.abs(predictions - y)
        conformal_quantile = float(np.quantile(residuals, 1 - config.conformal_alpha))

    clf_metrics: Optional[ClassificationMetrics] = None
    if config.price_bins and len(config.price_bins) >= 2:
        binned = _bin_targets(y, config.price_bins)
        features_transformed = final_pipeline.named_steps["preprocessor"].transform(X)
        classifier = _get_classifier()
        classifier.fit(features_transformed, binned)
        preds = classifier.predict(features_transformed)
        clf_metrics = classification_metrics(
            binned.astype(str), preds.astype(str), labels=np.unique(binned.astype(str))
        )

    artifact_dir = config.artifact_dir
    artifact_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(final_pipeline, artifact_dir / "model.joblib")

    metadata: Dict[str, Any] = {
        "task": "regression",
        "metrics": {
            "rmse": float(reg_metrics.rmse),
            "rmsle": float(reg_metrics.rmsle),
            "mae": float(reg_metrics.mae),
            "r2": float(reg_metrics.r2),
        },
        "conformal_quantile": conformal_quantile,
        "log_target": config.log_target,
    }
    if clf_metrics:
        metadata["classification"] = {
            "accuracy": float(clf_metrics.accuracy),
            "macro_f1": float(clf_metrics.macro_f1),
        }

    with open(artifact_dir / "metadata.yaml", "w", encoding="utf-8") as fh:
        yaml.safe_dump(metadata, fh)

    return TrainResult(
        pipeline=final_pipeline,
        regression_metrics=reg_metrics,
        classification_metrics=clf_metrics,
        conformal_quantile=conformal_quantile,
    )
