import numpy as np
import pandas as pd
import pytest

from src.preprocessing import Preprocessing


@pytest.fixture
def df():
    idx = pd.date_range("2024-01-01", periods=200, freq="D")
    rng = np.random.default_rng(1)
    a = 100 + rng.standard_normal(200).cumsum()
    b = 50 + rng.standard_normal(200).cumsum()
    return pd.DataFrame({"A": a, "B": b}, index=idx)


def test_transform_shapes(df):
    pp = Preprocessing(df)
    out = pp.transform_data(ma_window=10)
    assert out["log_returns"].shape == (199, 2)
    assert out["moving_average"].shape == df.shape
    assert out["scaled"].shape == df.shape
    # robust-scaled median ~ 0
    assert abs(out["scaled"].median().mean()) < 0.5


def test_filters_preserve_length(df):
    pp = Preprocessing(df)
    for fn in [
        pp.apply_kalman_filter,
        pp.apply_butterworth_filter,
        pp.apply_savgol_filter,
        pp.apply_moving_average,
        pp.apply_ta_lib_filter,
    ]:
        out = fn("A")
        assert len(out) == 200, fn.__name__
        assert out.notna().sum() > 100, fn.__name__


def test_apply_all_filters_columns(df):
    pp = Preprocessing(df)
    out = pp.apply_all_filters("A")
    assert out.shape[1] == 6
