"""Tests for DataLoader using synthetic data (no network)."""
import numpy as np
import pandas as pd
import pytest

from src.data_loader import DataLoader


def _make_loader_with_fake_data():
    idx = pd.date_range("2024-01-01", periods=50, freq="D")
    rng = np.random.default_rng(0)
    a = pd.Series(100 + rng.standard_normal(50).cumsum(), index=idx, name="A")
    b = pd.Series(50 + rng.standard_normal(50).cumsum(), index=idx, name="B")
    # inject some NaNs
    a.iloc[5:8] = np.nan
    b.iloc[20] = np.nan
    dl = DataLoader(tickers={"A": "A", "B": "B"}, start_date="2024-01-01", end_date="2024-03-01")
    dl.raw = {"A": a, "B": b}
    return dl


def test_merge_outer_join_and_columns():
    dl = _make_loader_with_fake_data()
    merged = dl.merge_data()
    assert list(merged.columns) == ["A", "B"]
    assert merged.index.is_monotonic_increasing
    assert merged.shape == (50, 2)


def test_impute_removes_all_nans():
    dl = _make_loader_with_fake_data()
    dl.merge_data()
    out = dl.impute_data()
    assert not out.isna().any().any()
    assert out.shape == (50, 2)


def test_missing_test_returns_pvalues():
    dl = _make_loader_with_fake_data()
    dl.merge_data()
    res = dl.missing_test()
    assert "per_column" in res and "mcar_plausible" in res
    assert set(res["per_column"].keys()) == {"A", "B"}


def test_merge_without_data_raises():
    dl = DataLoader(tickers={}, start_date="2024-01-01", end_date="2024-02-01")
    with pytest.raises(ValueError):
        dl.merge_data()
