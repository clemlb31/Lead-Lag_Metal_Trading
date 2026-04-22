"""DataLoader — fetch, merge, diagnose and impute Yahoo Finance time series.

References
----------
- yfinance docs: https://github.com/ranaroussi/yfinance
- Little's MCAR test (Little, 1988): EM-based likelihood-ratio chi-square test.
  See https://github.com/RianneSchouten/pyampute and
  https://medium.com/@tarangds/understanding-littles-mcar-test-a-key-tool-in-missing-data-analysis-47fd70698149
- Forward-fill rationale for prices: prices are *stocks* (last observed value
  remains valid until a new trade), unlike returns which are flows.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


@dataclass
class DataLoader:
    """Download, merge, diagnose and impute multi-asset Yahoo Finance data.

    Parameters
    ----------
    tickers : dict[str, str]
        Mapping ``{display_name: yahoo_ticker}``.
    start_date, end_date : str
        ISO dates ``YYYY-MM-DD``.
    freq : {"1d", "1h"}
        Sampling frequency. Yahoo limits hourly history to ~730 days.
    price_col : str
        Which OHLC column to keep (default ``"Close"``).
    """

    tickers: Dict[str, str]
    start_date: str
    end_date: str
    freq: str = "1d"
    price_col: str = "Close"
    raw: Dict[str, pd.Series] = field(default_factory=dict, init=False)
    merged: Optional[pd.DataFrame] = field(default=None, init=False)

    # ------------------------------------------------------------------ fetch
    def fetch_data(self) -> Dict[str, pd.Series]:
        """Download each ticker individually; tolerate per-ticker failures."""
        import yfinance as yf

        out: Dict[str, pd.Series] = {}
        for name, sym in self.tickers.items():
            try:
                df = yf.download(
                    sym,
                    start=self.start_date,
                    end=self.end_date,
                    interval=self.freq,
                    progress=False,
                    auto_adjust=False,
                )
                if df is None or df.empty:
                    print(f"[WARN] empty data for {name} ({sym})")
                    continue
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)
                s = df[self.price_col].rename(name)
                # ensure tz-naive UTC index for safe joins
                if s.index.tz is not None:
                    s.index = s.index.tz_convert("UTC").tz_localize(None)
                out[name] = s
            except Exception as e:  # noqa: BLE001
                print(f"[ERROR] {name} ({sym}): {e}")
        self.raw = out
        return out

    # ------------------------------------------------------------------ merge
    def merge_data(self, data_dict: Optional[Dict[str, pd.Series]] = None) -> pd.DataFrame:
        """Outer-join all series on the datetime index."""
        d = data_dict if data_dict is not None else self.raw
        if not d:
            raise ValueError("No data to merge — call fetch_data() first.")
        merged = pd.concat(d.values(), axis=1, join="outer").sort_index()
        merged.columns = list(d.keys())
        self.merged = merged
        return merged

    # ----------------------------------------------------------- missing EDA
    def eda_missing(self, merged_data: Optional[pd.DataFrame] = None):
        """Visual diagnostic of missingness via missingno."""
        import matplotlib.pyplot as plt
        import missingno as msno

        df = merged_data if merged_data is not None else self.merged
        if df is None:
            raise ValueError("merged data missing")
        fig, axes = plt.subplots(1, 2, figsize=(14, 4))
        msno.matrix(df, ax=axes[0], sparkline=False)
        axes[0].set_title("Missingness matrix")
        msno.heatmap(df, ax=axes[1])
        axes[1].set_title("Missingness correlation")
        plt.tight_layout()
        return fig

    # --------------------------------------------------------- MCAR-ish test
    def missing_test(self, merged_data: Optional[pd.DataFrame] = None) -> dict:
        """Lightweight Little-style MCAR diagnostic.

        We group rows by their missingness *pattern* and, for each fully
        observed column, compare the per-pattern means via a one-way ANOVA
        (chi-square / F). If the null (equal means across patterns) is
        rejected, the missingness is unlikely to be MCAR. This is a simplified
        proxy of Little (1988): the full EM-LR test requires multivariate
        normality which financial returns rarely satisfy.

        Returns
        -------
        dict with per-column F-statistic and p-value, plus a boolean
        ``mcar_plausible`` (all p > 0.05).
        """
        from scipy import stats

        df = merged_data if merged_data is not None else self.merged
        if df is None:
            raise ValueError("merged data missing")
        patterns = df.isna().apply(lambda r: tuple(r.values), axis=1)
        results = {}
        for col in df.columns:
            obs = df[col].dropna()
            grp_labels = patterns.loc[obs.index]
            groups = [obs.loc[grp_labels == p].values for p in grp_labels.unique()]
            groups = [g for g in groups if len(g) > 1]
            if len(groups) < 2:
                results[col] = {"F": np.nan, "p": np.nan}
                continue
            F, p = stats.f_oneway(*groups)
            results[col] = {"F": float(F), "p": float(p)}
        ps = [r["p"] for r in results.values() if not np.isnan(r["p"])]
        return {
            "per_column": results,
            "mcar_plausible": bool(ps) and all(p > 0.05 for p in ps),
        }

    # ---------------------------------------------------------------- impute
    def impute_data(self, merged_data: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """Forward-fill then back-fill (head). Justification: prices = stocks."""
        df = merged_data if merged_data is not None else self.merged
        if df is None:
            raise ValueError("merged data missing")
        out = df.ffill().bfill()
        if out.isna().any().any():
            raise RuntimeError("NaNs remain after ffill/bfill — check input.")
        self.merged = out
        return out
