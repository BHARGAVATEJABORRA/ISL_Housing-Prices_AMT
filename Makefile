PYTHON=python
PIP=pip
CONFIG=configs/config.yaml

.PHONY: install format lint test train api app

install:
$(PIP) install -r requirements.txt

format:
pre-commit run --all-files

lint:
black --check src app api tests
isort --check-only src app api tests
flake8 src app api tests

test:
pytest -q

train:
$(PYTHON) -m isl_housing.cli train --config $(CONFIG)

api:
uvicorn api.main:app --reload

app:
streamlit run app/streamlit_app.py
