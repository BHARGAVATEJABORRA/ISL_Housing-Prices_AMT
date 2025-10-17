"""Utility helpers for the ISL housing project."""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd


def bucket_rare_categories(
    series: pd.Series, threshold: float, other_label: str = "__rare__"
) -> pd.Series:
    """Replace infrequent categories with a shared label.

    Args:
        series: Input categorical series.
        threshold: Minimum frequency proportion to keep original category.
        other_label: Replacement label for rare categories.

    Returns:
        Series with rare categories replaced.
    """

    if threshold <= 0:
        return series

    value_counts = series.value_counts(normalize=True, dropna=False)
    rare_values: Iterable = value_counts[value_counts < threshold].index
    if not len(rare_values):
        return series
    return series.where(~series.isin(rare_values), other_label)


def normalise_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of the dataframe with normalised column names."""
    df = df.copy()
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_", regex=False)
    return df


def safe_log1p(values: np.ndarray) -> np.ndarray:
    """Apply log1p safely to an array."""
    return np.log1p(np.clip(values, a_min=0, a_max=None))


def safe_expm1(values: np.ndarray) -> np.ndarray:
    """Apply expm1 safely to an array."""
    return np.expm1(values)
