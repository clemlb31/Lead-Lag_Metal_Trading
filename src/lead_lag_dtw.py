"""LeadLagDTW — Dynamic Time Warping for lead/lag detection and forecasting.

Sign convention
---------------
Given a warping path of pairs (i, j) aligning series1[i] with series2[j]:
    lag = mean(j - i)
    lag > 0  →  series1 LEADS series2 by ``lag`` steps
                (to match index i of series1, DTW had to look further
                 ahead in series2 → series2 is delayed)
    lag < 0  →  series2 leads series1
This matches MDPI 2022 (sensors-22-06884) and tslearn conventions.

References
----------
- MDPI 2022, "Multi-Dimensional DTW to Identify Time-Varying Lead-Lag":
  https://www.mdpi.com/1424-8220/22/18/6884
- dtaidistance: https://dtaidistance.readthedocs.io
- Sakoe-Chiba band: classical global constraint, radius r limits |i-j| <= r.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np
import pandas as pd


@dataclass
class LeadLagDTW:
    data: pd.DataFrame
    sakoe_chiba_radius: Optional[int] = None
    distance_matrix_: Optional[pd.DataFrame] = field(default=None, init=False)
    lag_matrix_: Optional[pd.DataFrame] = field(default=None, init=False)

    # ------------------------------------------------------------------ core
    def _standardise(self, s: np.ndarray) -> np.ndarray:
        s = np.asarray(s, dtype=float)
        return (s - s.mean()) / (s.std() + 1e-12)

    def compute_dtw(self, series1, series2) -> Tuple[float, list]:
        """Return (DTW distance, optimal warping path as list of (i, j))."""
        from dtaidistance import dtw

        a = self._standardise(np.asarray(series1))
        b = self._standardise(np.asarray(series2))
        kwargs = {}
        if self.sakoe_chiba_radius is not None:
            kwargs["window"] = int(self.sakoe_chiba_radius)
        d = dtw.distance(a, b, **kwargs)
        path = dtw.warping_path(a, b, **kwargs)
        return float(d), path

    @staticmethod
    def _path_lag(path: list) -> float:
        """Mean signed deviation (j - i) of the warping path from the diagonal."""
        arr = np.asarray(path, dtype=float)
        return float(np.mean(arr[:, 1] - arr[:, 0]))

    # ------------------------------------------------------ pairwise matrices
    def identify_lead_lag(self) -> dict:
        """Compute n×n DTW distance and lead-lag matrices over all asset pairs."""
        cols = list(self.data.columns)
        n = len(cols)
        D = np.zeros((n, n))
        L = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                d, path = self.compute_dtw(self.data[cols[i]], self.data[cols[j]])
                lag = self._path_lag(path)
                D[i, j] = D[j, i] = d
                L[i, j] = lag           # cols[i] leads cols[j] if > 0
                L[j, i] = -lag
        self.distance_matrix_ = pd.DataFrame(D, index=cols, columns=cols)
        self.lag_matrix_ = pd.DataFrame(L, index=cols, columns=cols)
        return {"distance": self.distance_matrix_, "lag": self.lag_matrix_}

    # ------------------------------------------------------------ forecasting
    def forecast(self, leader: str, follower: str, lag: Optional[int] = None) -> pd.Series:
        """Forecast follower at t using leader at t - lag (naive shift model)."""
        if lag is None:
            if self.lag_matrix_ is None:
                self.identify_lead_lag()
            lag = int(round(self.lag_matrix_.loc[leader, follower]))
        lag = max(int(lag), 0)
        return self.data[leader].shift(lag).rename(f"{follower}_hat")

    @staticmethod
    def validate(forecast: pd.Series, actual: pd.Series) -> dict:
        """MSE and directional accuracy on log-returns."""
        df = pd.concat([forecast, actual], axis=1).dropna()
        df.columns = ["yhat", "y"]
        if df.empty:
            return {"mse": np.nan, "dir_acc": np.nan, "n": 0}
        mse = float(((df["y"] - df["yhat"]) ** 2).mean())
        dy = df["y"].diff().dropna()
        dyhat = df["yhat"].diff().dropna()
        common = dy.index.intersection(dyhat.index)
        dir_acc = float((np.sign(dy.loc[common]) == np.sign(dyhat.loc[common])).mean())
        return {"mse": mse, "dir_acc": dir_acc, "n": int(len(df))}
