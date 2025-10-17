"""Streamlit app for interactive predictions."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from isl_housing.utils import normalise_columns

st.set_page_config(page_title="ISL Housing Predictor")


@st.cache_resource
def load_artifacts(artifact_dir: Path) -> tuple[Any, Dict[str, Any]]:
    model_path = artifact_dir / "model.joblib"
    metadata_path = artifact_dir / "metadata.yaml"
    if not model_path.exists():
        raise FileNotFoundError("Model artifacts not found. Run training first.")
    pipeline = joblib.load(model_path)
    metadata: Dict[str, Any] = {}
    if metadata_path.exists():
        with open(metadata_path, "r", encoding="utf-8") as fh:
            metadata = yaml.safe_load(fh) or {}
    return pipeline, metadata


def main() -> None:
    st.title("ISL Housing Price Prediction")
    artifact_dir = Path(st.sidebar.text_input("Artifact directory", "models"))

    try:
        pipeline, metadata = load_artifacts(artifact_dir)
    except FileNotFoundError as exc:
        st.warning(str(exc))
        return

    st.sidebar.markdown("### Metadata")
    st.sidebar.json(metadata)

    st.subheader("Single prediction")
    sample_input = st.text_area(
        "Enter JSON for a single row",
        value='{"Gr_Liv_Area": 1500, "Overall_Qual": 7}',
    )

    if st.button("Predict from JSON"):
        try:
            parsed = json.loads(sample_input)
            if not isinstance(parsed, dict):
                raise ValueError("JSON must represent an object")
            data = normalise_columns(pd.DataFrame([parsed]))
        except Exception as exc:  # pragma: no cover - user input
            st.error(f"Invalid JSON: {exc}")
        else:
            preds = pipeline.predict(data)
            if metadata.get("log_target", True):
                preds = np.expm1(preds)
            st.write({"prediction": float(preds[0])})

    st.subheader("Batch prediction via CSV upload")
    uploaded = st.file_uploader("Upload CSV", type=["csv"])
    if uploaded is not None:
        df = normalise_columns(pd.read_csv(uploaded))
        preds = pipeline.predict(df)
        if metadata.get("log_target", True):
            preds = np.expm1(preds)
        conformal_q = metadata.get("conformal_quantile")
        results = pd.DataFrame({"prediction": preds})
        if conformal_q is not None:
            results["lower"] = np.clip(preds - conformal_q, a_min=0, a_max=None)
            results["upper"] = preds + conformal_q
        st.dataframe(results)
        st.download_button(
            "Download predictions",
            results.to_csv(index=False).encode("utf-8"),
            file_name="predictions.csv",
            mime="text/csv",
        )


if __name__ == "__main__":  # pragma: no cover
    main()
