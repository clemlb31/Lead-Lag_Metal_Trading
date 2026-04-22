"""Preprocessing — feature engineering, scaling, filtering, anomaly detection.

Filter choices and rationale
----------------------------
- **Kalman (local level)**  : optimal under random-walk + Gaussian noise; no
  window to choose. Ref: pykalman docs, Quantrocket Lecture 45.
- **Butterworth low-pass**  : flat passband, removes HF noise without ripple.
  Standard in DSP. Cutoff = 0.1 (Nyquist units) by default → keeps ~10-day
  cycles on daily data.
- **Savitzky–Golay**        : local polynomial fit; preserves peaks better
  than MA. Good for keeping price features intact.
- **Simple moving average** : baseline / sanity check.
- **TA-Lib EMA**            : exponentially weighted, causal (no leak),
  industry standard for trading.

Anomaly detection
-----------------
- **MarkovRegression** (statsmodels): regime shifts (calm vs crisis).
- **Prophet**: trend + seasonality decomposition with built-in outlier flag.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd


@dataclass
class Preprocessing:
    data: pd.DataFrame
    scaled: Optional[pd.DataFrame] = field(default=None, init=False)
    log_returns: Optional[pd.DataFrame] = field(default=None, init=False)

    # ------------------------------------------------------------ transforms
    def transform_data(self, ma_window: int = 20) -> dict:
        """Compute log-returns, moving average, and a robust-scaled view."""
        from sklearn.preprocessing import RobustScaler

        prices = self.data.astype(float)
        log_ret = np.log(prices / prices.shift(1)).dropna()
        ma = prices.rolling(ma_window, min_periods=1).mean()
        scaler = RobustScaler()
        scaled = pd.DataFrame(
            scaler.fit_transform(prices.values),
            index=prices.index,
            columns=prices.columns,
        )
        self.log_returns = log_ret
        self.scaled = scaled
        return {"log_returns": log_ret, "moving_average": ma, "scaled": scaled}

    # ---------------------------------------------------------------- filters
    def apply_kalman_filter(self, col: str) -> pd.Series:
        """Local level (random walk + noise) Kalman smoother."""
        from pykalman import KalmanFilter

        s = self.data[col].dropna().astype(float)
        kf = KalmanFilter(
            transition_matrices=[1],
            observation_matrices=[1],
            initial_state_mean=s.iloc[0],
            initial_state_covariance=1.0,
            observation_covariance=1.0,
            transition_covariance=0.01,
        )
        state_means, _ = kf.smooth(s.values)
        return pd.Series(state_means.ravel(), index=s.index, name=f"{col}_kalman")

    def apply_butterworth_filter(self, col: str, cutoff: float = 0.1, order: int = 3) -> pd.Series:
        from scipy.signal import butter, filtfilt

        s = self.data[col].dropna().astype(float)
        b, a = butter(order, cutoff, btype="low")
        y = filtfilt(b, a, s.values)
        return pd.Series(y, index=s.index, name=f"{col}_butter")

    def apply_savgol_filter(self, col: str, window: int = 21, poly: int = 3) -> pd.Series:
        from scipy.signal import savgol_filter

        s = self.data[col].dropna().astype(float)
        if window % 2 == 0:
            window += 1
        y = savgol_filter(s.values, window_length=window, polyorder=poly)
        return pd.Series(y, index=s.index, name=f"{col}_savgol")

    def apply_moving_average(self, col: str, window: int = 20) -> pd.Series:
        return self.data[col].rolling(window, min_periods=1).mean().rename(f"{col}_ma")

    def apply_ta_lib_filter(self, col: str, window: int = 20) -> pd.Series:
        """EMA via TA-Lib if available, else pandas EWM fallback."""
        s = self.data[col].dropna().astype(float)
        try:
            import talib  # type: ignore

            y = talib.EMA(s.values, timeperiod=window)
            return pd.Series(y, index=s.index, name=f"{col}_ema")
        except ImportError:
            return s.ewm(span=window, adjust=False).mean().rename(f"{col}_ema")

    def apply_all_filters(self, col: str) -> pd.DataFrame:
        return pd.concat(
            [
                self.data[col].rename(f"{col}_raw"),
                self.apply_kalman_filter(col),
                self.apply_butterworth_filter(col),
                self.apply_savgol_filter(col),
                self.apply_moving_average(col),
                self.apply_ta_lib_filter(col),
            ],
            axis=1,
        )

    # ------------------------------------------------------- anomaly detection
    def detect_regimes_markov(self, col: str, k_regimes: int = 2) -> pd.Series:
        """Markov-switching mean model on log-returns → most likely regime."""
        from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression

        ret = np.log(self.data[col]).diff().dropna()
        model = MarkovRegression(ret, k_regimes=k_regimes, trend="c", switching_variance=True)
        res = model.fit(disp=False)
        regimes = res.smoothed_marginal_probabilities.idxmax(axis=1)
        return regimes.rename(f"{col}_regime")

    def detect_outliers_prophet(self, col: str, interval_width: float = 0.99) -> pd.DataFrame:
        from prophet import Prophet

        s = self.data[col].dropna()
        df = pd.DataFrame({"ds": s.index, "y": s.values})
        m = Prophet(interval_width=interval_width, daily_seasonality=False)
        m.fit(df)
        fc = m.predict(df[["ds"]])
        merged = df.merge(fc[["ds", "yhat_lower", "yhat_upper"]], on="ds")
        merged["outlier"] = (merged["y"] < merged["yhat_lower"]) | (merged["y"] > merged["yhat_upper"])
        return merged.set_index("ds")
