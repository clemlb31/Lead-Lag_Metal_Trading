"""Validate LeadLagDTW on synthetic shifted signals (ground truth available)."""
import numpy as np
import pandas as pd
import pytest

from src.lead_lag_dtw import LeadLagDTW


@pytest.fixture
def shifted_pair():
    n, k = 300, 7  # k = true lag (A leads B by k steps)
    t = np.arange(n)
    a = np.sin(2 * np.pi * t / 40) + 0.05 * np.random.default_rng(0).standard_normal(n)
    b = np.sin(2 * np.pi * (t - k) / 40) + 0.05 * np.random.default_rng(1).standard_normal(n)
    return pd.DataFrame({"A": a, "B": b}), k


def test_recovers_known_lag(shifted_pair):
    df, k = shifted_pair
    ll = LeadLagDTW(df, sakoe_chiba_radius=20)
    res = ll.identify_lead_lag()
    L = res["lag"]
    assert L.shape == (2, 2)
    # A should lead B → L["A","B"] > 0 and ≈ k
    assert L.loc["A", "B"] > 0
    assert abs(L.loc["A", "B"] - k) <= 3, f"recovered {L.loc['A','B']}, expected ~{k}"
    # antisymmetry
    np.testing.assert_almost_equal(L.loc["A", "B"], -L.loc["B", "A"])


def test_distance_matrix_properties(shifted_pair):
    df, _ = shifted_pair
    ll = LeadLagDTW(df)
    res = ll.identify_lead_lag()
    D = res["distance"]
    np.testing.assert_array_almost_equal(D.values, D.values.T)
    np.testing.assert_array_almost_equal(np.diag(D.values), np.zeros(2))


def test_forecast_and_validate(shifted_pair):
    df, k = shifted_pair
    ll = LeadLagDTW(df, sakoe_chiba_radius=20)
    ll.identify_lead_lag()
    fc = ll.forecast("A", "B")
    metrics = ll.validate(fc, df["B"])
    assert metrics["n"] > 100
    assert metrics["dir_acc"] > 0.6  # naive shift should beat random
