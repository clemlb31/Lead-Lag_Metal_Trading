"""EDA — interactive plots, correlation, DTW clustermap, seasonality.

References
----------
- Bokeh user guide: https://docs.bokeh.org
- dtaidistance DTW: https://dtaidistance.readthedocs.io
- statsmodels STL: https://www.statsmodels.org/dev/generated/statsmodels.tsa.seasonal.STL.html
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd


@dataclass
class EDA:
    data: pd.DataFrame  # cleaned, imputed price (or scaled) frame

    # ------------------------------------------------------------- bokeh plot
    def plot_timeseries(self):
        """Linked-pan Bokeh figure: one panel per series."""
        from bokeh.layouts import column
        from bokeh.plotting import figure

        figs = []
        x_range = None
        for col in self.data.columns:
            p = figure(
                x_axis_type="datetime",
                title=col,
                height=180,
                width=900,
                x_range=x_range,
            )
            p.line(self.data.index, self.data[col].values, line_width=1.2)
            x_range = x_range or p.x_range
            figs.append(p)
        return column(*figs)

    # ----------------------------------------------------------- correlations
    def correlation_matrix(self, max_lag: int = 5) -> dict:
        """Pearson correlation + per-pair cross-correlations for lags 1..max_lag."""
        ret = np.log(self.data).diff().dropna()
        corr = ret.corr()
        cross = {}
        cols = ret.columns
        for i, a in enumerate(cols):
            for b in cols[i + 1 :]:
                xs = []
                for lag in range(1, max_lag + 1):
                    xs.append(ret[a].corr(ret[b].shift(lag)))
                cross[(a, b)] = xs
        return {"corr": corr, "cross_corr": cross}

    # ------------------------------------------------------------- dtw matrix
    def dtw_distance_matrix(self) -> pd.DataFrame:
        from dtaidistance import dtw

        ret = np.log(self.data).diff().dropna()
        # standardise per series so DTW compares shapes, not levels
        z = (ret - ret.mean()) / ret.std()
        cols = list(z.columns)
        n = len(cols)
        D = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                d = dtw.distance(z[cols[i]].values, z[cols[j]].values)
                D[i, j] = D[j, i] = d
        return pd.DataFrame(D, index=cols, columns=cols)

    def dtw_clustermap(self):
        import seaborn as sns

        D = self.dtw_distance_matrix()
        g = sns.clustermap(D, cmap="viridis", annot=True, fmt=".1f")
        return g, D

    # ----------------------------------------------------------- seasonality
    def seasonality_tracker(self, period: int = 5) -> dict:
        """STL decomposition per series. Default period=5 (weekly on daily)."""
        from statsmodels.tsa.seasonal import STL

        out = {}
        for col in self.data.columns:
            s = self.data[col].dropna()
            if len(s) < 2 * period + 1:
                continue
            stl = STL(s, period=period, robust=True).fit()
            out[col] = {"trend": stl.trend, "seasonal": stl.seasonal, "resid": stl.resid}
        return out
