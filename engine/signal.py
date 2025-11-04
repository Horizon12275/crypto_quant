import pandas as pd
import numpy as np


def _rolling_zscore(x: pd.Series, window: int) -> pd.Series:
    mean = x.rolling(window=window, min_periods=window).mean()
    std = x.rolling(window=window, min_periods=window).std(ddof=0)
    z = (x - mean) / std
    return z


def _rolling_percentile_signal(x: pd.Series, window: int) -> pd.Series:
    """
    Compute rolling percentile of the latest value within each window, mapped to [-1, 1].
    Uses average-rank interpolation for ties.
    """
    if window <= 1:
        return pd.Series(index=x.index, dtype=float)

    def last_value_avg_percentile(arr: np.ndarray) -> float:
        if arr.size == 0:
            return np.nan
        last = arr[-1]
        if np.isnan(last):
            return np.nan
        valid = arr[~np.isnan(arr)]
        n = valid.size
        if n == 0:
            return np.nan
        num_less = np.sum(valid < last)
        num_equal = np.sum(valid == last)
        percentile = (num_less + 0.5 * num_equal) / n  # in [0,1]
        return 2.0 * percentile - 1.0  # map to [-1, 1]

    return x.rolling(window=window, min_periods=window).apply(last_value_avg_percentile, raw=True)


def map_factor_to_target_notional(
    factor: pd.Series,
    capital: float,
    mapper: str = "zscore",
    zscore_window: int = 100,
    clip_abs: float = 1.0,
    allow_short: bool = True,
    long_leverage: float = 1.0,
    short_leverage: float = 1.0,
) -> pd.Series:
    """
    Map factor values to target dollar notional exposures.
    - mapper 'zscore': rolling z-score, clipped to [-clip_abs, clip_abs]
    - mapper 'sign': sign(factor)
    Apply leverage: s>=0 scaled by long_leverage, s<0 scaled by short_leverage.
    If short is not allowed, negatives are set to 0.
    """
    f = factor.copy().astype(float)

    if mapper == "zscore":
        s = _rolling_zscore(f, zscore_window)
        s = s.clip(lower=-clip_abs, upper=clip_abs)
    elif mapper == "sign":
        s = np.sign(f)
        s = s.clip(lower=-clip_abs, upper=clip_abs)
    elif mapper == "percentile":
        # Use zscore_window as the rolling window length for percentile as well
        s = _rolling_percentile_signal(f, zscore_window)
        s = s.clip(lower=-clip_abs, upper=clip_abs)
    else:
        raise ValueError(f"Unknown mapper: {mapper}")

    if not allow_short:
        s = s.clip(lower=0.0)

    # Apply leverage asymmetrically (clearer formulation)
    s_pos = s.clip(lower=0) * float(long_leverage)
    s_neg = s.clip(upper=0) * float(short_leverage)
    s_levered = s_pos + s_neg

    target_notional = capital * s_levered.fillna(0.0)
    target_notional.name = "target_notional"
    return target_notional 