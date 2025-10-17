"""Command line interface for training and inference."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

import joblib
import numpy as np
import pandas as pd
import yaml

from .model import TrainConfig, train
from .utils import normalise_columns, safe_expm1


def _load_config(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def _config_from_yaml(cfg: Dict[str, Any]) -> TrainConfig:
    data_cfg = cfg.get("data", {})
    preprocessing_cfg = cfg.get("preprocessing", {})
    model_cfg = cfg.get("model", {})
    cv_cfg = cfg.get("cv", {})
    output_cfg = cfg.get("output", {})
    classification_cfg = cfg.get("classification", {})

    return TrainConfig(
        data_path=Path(data_cfg.get("path", "data/Housing_data.xls")),
        target=data_cfg.get("target", "SalePrice"),
        log_target=data_cfg.get("log_target", True),
        rare_threshold=preprocessing_cfg.get("rare_threshold", 0.01),
        scale_numeric=preprocessing_cfg.get("scale_numeric", True),
        model_type=model_cfg.get("type", "xgboost"),
        model_params=model_cfg.get("params"),
        cv_folds=cv_cfg.get("folds", 5),
        random_state=cv_cfg.get("random_state", 42),
        artifact_dir=Path(output_cfg.get("artifact_dir", "models")),
        conformal_alpha=output_cfg.get("conformal_alpha", 0.1),
        price_bins=classification_cfg.get("price_bins"),
    )


def cmd_train(args: argparse.Namespace) -> None:
    cfg = _load_config(Path(args.config))
    train_config = _config_from_yaml(cfg)
    result = train(train_config)
    print("Training complete")
    print(
        json.dumps(
            {
                "rmse": result.regression_metrics.rmse,
                "rmsle": result.regression_metrics.rmsle,
                "mae": result.regression_metrics.mae,
                "r2": result.regression_metrics.r2,
                "conformal_quantile": result.conformal_quantile,
            },
            indent=2,
        )
    )


def cmd_predict(args: argparse.Namespace) -> None:
    cfg = _load_config(Path(args.config))
    output_cfg = cfg.get("output", {})
    artifact_dir = Path(output_cfg.get("artifact_dir", "models"))
    model_path = artifact_dir / "model.joblib"
    metadata_path = artifact_dir / "metadata.yaml"

    if not model_path.exists():
        raise FileNotFoundError("Trained model not found. Run the train command first.")

    pipeline = joblib.load(model_path)
    metadata: Dict[str, Any] = {}
    if metadata_path.exists():
        with open(metadata_path, "r", encoding="utf-8") as fh:
            metadata = yaml.safe_load(fh) or {}

    df = normalise_columns(pd.read_csv(args.input))
    preds = pipeline.predict(df)

    if metadata.get("log_target", True):
        preds = safe_expm1(preds)

    result = pd.DataFrame({"prediction": preds})

    conformal_q = metadata.get("conformal_quantile")
    if conformal_q is not None:
        result["lower"] = np.clip(preds - conformal_q, a_min=0, a_max=None)
        result["upper"] = preds + conformal_q

    result.to_csv(args.output, index=False)
    print(f"Predictions written to {args.output}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="ISL housing ML utilities")
    parent = argparse.ArgumentParser(add_help=False)
    parent.add_argument(
        "--config", default="configs/config.yaml", help="Path to config file"
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser(
        "train", help="Train the model", parents=[parent]
    )
    train_parser.set_defaults(func=cmd_train)

    predict_parser = subparsers.add_parser(
        "predict", help="Run batch predictions", parents=[parent]
    )
    predict_parser.add_argument("--input", required=True, help="Input CSV features")
    predict_parser.add_argument("--output", required=True, help="Output CSV path")
    predict_parser.set_defaults(func=cmd_predict)

    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":  # pragma: no cover
    main()
