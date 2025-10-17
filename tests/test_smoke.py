"""Basic smoke tests."""
from __future__ import annotations

import importlib

import pytest


def test_package_importable():
    try:
        module = importlib.import_module("isl_housing")
    except ModuleNotFoundError as exc:  # pragma: no cover - optional deps
        if exc.name == "pandas":
            pytest.skip("pandas not installed")
        raise
    assert module is not None
