import pandas as pd
import numpy as np
from typing import Literal


def compute_risk_scaler(
    opens: pd.Series,
    vol_lookback_days: int = 63,
    vol_target_ann: float = 0.25,
    max_gross_leverage: float = 1.0,
    rebalance: Literal["monthly", "days"] = "monthly",
    rebalance_days: int = 30,
) -> pd.Series:
    """
    Build a time series scaler based on recent annualized volatility.
    - Compute daily close-to-close simple returns from minute opens via daily last.
    - Rolling stdev over `vol_lookback_days` → annualize with sqrt(365).
    - scaler = min(vol_target_ann / sigma, max_gross_leverage).
    - Recompute on risk rebalance dates only, forward-fill between dates to minute index.
    """
    if opens.empty:
        return pd.Series(dtype=float)

    # Daily close prices (use daily last open as proxy for daily close-to-close base)
    daily_price = opens.resample("1D").last().dropna()
    daily_ret = daily_price.pct_change().dropna()

    rolling_sigma = daily_ret.rolling(window=vol_lookback_days, min_periods=vol_lookback_days).std(ddof=0)
    sigma_ann = rolling_sigma * np.sqrt(365.0)

    raw_scaler = (vol_target_ann / sigma_ann).clip(upper=max_gross_leverage)
    raw_scaler = raw_scaler.fillna(1.0)

    # Select rebalance dates
    if rebalance == "monthly":
        dates = raw_scaler.resample("MS").last().index  # month starts
    else:
        # every N days from the first available date
        start = raw_scaler.index.min()
        dates = pd.date_range(start=start, end=raw_scaler.index.max(), freq=f"{int(rebalance_days)}D")

    # Intersect with available scaler index and forward-fill between rebalances
    rebal_points = raw_scaler.reindex(dates).dropna()
    if rebal_points.empty:
        # Fallback to a constant scaler of 1.0
        scaler_daily = pd.Series(1.0, index=daily_price.index)
    else:
        scaler_daily = rebal_points.reindex(daily_price.index, method="ffill").fillna(1.0)

    # Map to minute index and forward-fill
    scaler_minute = scaler_daily.reindex(opens.index, method="ffill").fillna(1.0)
    scaler_minute.name = "risk_scaler"
    return scaler_minute
