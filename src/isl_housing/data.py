"""Data loading utilities."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd

from .utils import normalise_columns


class DataLoadError(RuntimeError):
    """Custom error when loading the housing dataset fails."""


def load_raw(path: str | Path, target: Optional[str] = None) -> pd.DataFrame:
    """Load the raw dataset, handling Excel and CSV formats.

    Args:
        path: Path to the raw dataset (CSV or XLS/XLSX).
        target: Optional target column to validate.

    Returns:
        DataFrame with normalised column names.
    """

    path = Path(path)
    if not path.exists():
        raise DataLoadError(f"Dataset not found at {path}.")

    suffix = path.suffix.lower()
    try:
        if suffix in {".xls", ".xlsx"}:
            try:
                df = pd.read_excel(path)
            except ImportError as exc:
                raise DataLoadError(
                    "Excel support requires the 'xlrd' optional dependency. "
                    "Install it via 'pip install xlrd' or convert the file to CSV."
                ) from exc
        elif suffix == ".csv":
            df = pd.read_csv(path)
        else:
            raise DataLoadError(f"Unsupported file extension: {suffix}")
    except Exception as exc:  # pragma: no cover - surfacing informative errors
        raise DataLoadError(f"Failed to load dataset: {exc}") from exc

    df = normalise_columns(df)
    if target:
        target_lower = target.strip().lower()
        if target_lower not in df.columns:
            raise DataLoadError(
                f"Target column '{target}' not found after normalisation."
            )
    return df
