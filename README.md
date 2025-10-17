# ISL Housing Prices – Modern ML Application

This repository upgrades the classic Introduction to Statistical Learning housing assignment into a production-oriented machine learning application. It includes configuration-driven training pipelines, automated testing, reproducible environments, and lightweight serving layers for both API and UI usage.

## Quickstart

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export PYTHONPATH=src  # ensure package imports resolve
cp .env.example .env  # optional; configure MLFLOW_TRACKING_URI if using MLflow
```

Place the **Housing_data.xls** file in the `data/` directory (already present in this repository). Reading Excel files requires the optional `xlrd` dependency. Install it with `pip install xlrd` or convert the file to CSV if preferred.

## Training workflow

```bash
python -m isl_housing.cli train --config configs/config.yaml
```

The training command performs the following:

- Loads the dataset using safe helpers that normalise column names and validate the configured target.
- Builds a scikit-learn pipeline with rare-category bucketing, imputers, scalers, and an XGBoost regressor (Ridge fallback when XGBoost is unavailable).
- Runs 5-fold cross-validation with deterministic seeding and optional log-transformation of the target.
- Computes regression metrics and conformal prediction intervals, saving a fitted pipeline plus metadata to the `models/` directory.

After training, you can perform batch prediction:

```bash
python -m isl_housing.cli predict --config configs/config.yaml --input my_features.csv --output preds.csv
```

## API service

```bash
uvicorn api.main:app --reload --port 8000
```

Example request:

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"rows": [{"Gr_Liv_Area": 1500, "Overall_Qual": 7}]}'
```

The response includes predictions and, when available, lower/upper bounds from conformal intervals.

## Streamlit app

```bash
streamlit run app/streamlit_app.py
```

Upload a CSV without the target column or use the manual input form to obtain predictions and intervals. The app also surfaces model metadata to help understand which artifacts are being used. Add screenshots of the UI to `reports/figures/` once generated.

## Repository structure

```
.
├── src/isl_housing/          # Core Python package
├── api/main.py               # FastAPI prediction endpoint
├── app/streamlit_app.py      # Streamlit UI for interactive use
├── configs/config.yaml       # Training and inference configuration
├── tests/                    # Lightweight smoke tests
├── .github/workflows/ci.yml  # GitHub Actions CI pipeline
├── Dockerfile.api            # API container image
├── Dockerfile.app            # Streamlit container image
├── docker-compose.yml        # Optional local orchestration
├── Makefile                  # Common developer tasks
└── requirements.txt          # Python dependencies
```

## Continuous integration

GitHub Actions runs linting (Black, isort, Flake8) and the pytest suite on every push or pull request. Use `make lint` and `make test` locally before opening a PR.

## Results tracking

Populate the table below with evaluation outcomes from your experiments. Metrics refer to the original SalePrice scale.

| Run | Model | RMSE | MAE | R² | Notes |
|-----|-------|------|-----|----|-------|
| TBD | TBD   | TBD  | TBD | TBD| Fill in after training |

## License

This project is distributed under the terms of the MIT License. See `LICENSE` for details.
