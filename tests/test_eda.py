import numpy as np
import pandas as pd
import pytest

from src.eda import EDA


@pytest.fixture
def df():
    idx = pd.date_range("2024-01-01", periods=120, freq="D")
    rng = np.random.default_rng(2)
    a = 100 + rng.standard_normal(120).cumsum()
    b = a + rng.standard_normal(120) * 0.1  # highly correlated
    c = 50 + rng.standard_normal(120).cumsum()
    return pd.DataFrame({"A": a, "B": b, "C": c}, index=idx)


def test_correlation_matrix_structure(df):
    eda = EDA(df)
    res = eda.correlation_matrix(max_lag=3)
    assert res["corr"].shape == (3, 3)
    assert abs(res["corr"].loc["A", "B"]) > 0.5  # A and B correlated
    assert all(len(v) == 3 for v in res["cross_corr"].values())


def test_dtw_matrix_symmetric_zero_diag(df):
    eda = EDA(df)
    D = eda.dtw_distance_matrix()
    assert D.shape == (3, 3)
    np.testing.assert_array_almost_equal(D.values, D.values.T)
    np.testing.assert_array_almost_equal(np.diag(D.values), np.zeros(3))
    # A-B much closer than A-C
    assert D.loc["A", "B"] < D.loc["A", "C"]


def test_seasonality_keys(df):
    eda = EDA(df)
    out = eda.seasonality_tracker(period=7)
    assert set(out.keys()) == {"A", "B", "C"}
    assert all({"trend", "seasonal", "resid"} <= set(v.keys()) for v in out.values())
