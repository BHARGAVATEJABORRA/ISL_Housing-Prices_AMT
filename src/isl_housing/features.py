"""Feature engineering pipelines."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .utils import bucket_rare_categories


@dataclass
class FeatureConfig:
    """Configuration for feature engineering."""

    rare_threshold: float = 0.01
    scale_numeric: bool = True


class RareCategoryBucketer:
    """Simple transformer to bucket rare categories."""

    def __init__(self, threshold: float = 0.01):
        self.threshold = threshold
        self.columns: List[str] = []

    def fit(self, X: pd.DataFrame, y=None):  # type: ignore[override]
        self.columns = list(X.columns)
        return self

    def transform(self, X: pd.DataFrame):  # type: ignore[override]
        X = X.copy()
        for column in self.columns:
            X[column] = bucket_rare_categories(X[column], self.threshold)
        return X

    def get_feature_names_out(self, input_features=None):  # pragma: no cover
        return np.array(self.columns)


def build_preprocessor(df: pd.DataFrame, config: FeatureConfig) -> ColumnTransformer:
    """Construct the preprocessing pipeline."""

    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    categorical_cols = df.select_dtypes(exclude=["number"]).columns.tolist()

    numeric_pipeline_steps = [("imputer", SimpleImputer(strategy="median"))]
    if config.scale_numeric:
        numeric_pipeline_steps.append(("scaler", StandardScaler()))

    numeric_pipeline = Pipeline(steps=numeric_pipeline_steps)

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("bucketer", RareCategoryBucketer(threshold=config.rare_threshold)),
            (
                "encoder",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            ),
        ]
    )

    transformers = []
    if numeric_cols:
        transformers.append(("numeric", numeric_pipeline, numeric_cols))
    if categorical_cols:
        transformers.append(("categorical", categorical_pipeline, categorical_cols))

    if not transformers:
        raise ValueError("No features available for preprocessing.")

    return ColumnTransformer(transformers=transformers)
