import pandas as pd
import numpy as np


def _rolling_zscore(x: pd.Series, window: int) -> pd.Series:
    mean = x.rolling(window=window, min_periods=window).mean()
    std = x.rolling(window=window, min_periods=window).std(ddof=0)
    z = (x - mean) / std
    return z


def _percent_rank_last(arr: np.ndarray) -> float:
    n = arr.size
    if n <= 1:
        return np.nan
    sorted_arr = np.sort(arr)
    idx = np.searchsorted(sorted_arr, arr[-1], side="right") - 1
    return idx / (n - 1)


def _rolling_percentile(x: pd.Series, window: int) -> pd.Series:
    return x.rolling(window=window, min_periods=window).apply(_percent_rank_last, raw=True)


def map_factor_to_target_notional(
    factor: pd.Series,
    capital: float,
    mapper: str = "zscore",
    zscore_window: int = 100,
    clip_abs: float = 1.0,
    allow_short: bool = True,
    long_leverage: float = 1.0,
    short_leverage: float = 1.0,
    trade_mode: str = "continuous",
    entry_long_threshold: float = 0.9,
    entry_short_threshold: float = -0.9,
) -> pd.Series:
    """
    Legacy: returns dollar target notionals using a fixed 'capital'. Prefer map_factor_to_target_fraction for
    dynamic equity sizing.
    """
    f = factor.copy().astype(float)

    if mapper == "zscore":
        s = _rolling_zscore(f, zscore_window)
        s = s.clip(lower=-clip_abs, upper=clip_abs)
    elif mapper == "sign":
        s = np.sign(f)
        s = s.clip(lower=-clip_abs, upper=clip_abs)
    elif mapper == "percentile":
        p = _rolling_percentile(f, zscore_window)
        s = (2.0 * p - 1.0).clip(lower=-clip_abs, upper=clip_abs)
    else:
        raise ValueError(f"Unknown mapper: {mapper}")

    if not allow_short:
        s = s.clip(lower=0.0)

    if trade_mode == "continuous":
        s_pos = s.clip(lower=0) * float(long_leverage)
        s_neg = s.clip(upper=0) * float(short_leverage)
        s_levered = s_pos + s_neg
        target_notional = capital * s_levered.fillna(0.0)
    elif trade_mode == "threshold":
        st = pd.Series(index=s.index, dtype=float)
        st[(s >= float(entry_long_threshold))] = float(long_leverage)
        st[(s <= float(entry_short_threshold))] = -float(short_leverage)
        if not allow_short:
            st[st < 0] = np.nan
        target_notional = capital * st
        target_notional.name = "target_notional"
    else:
        raise ValueError(f"Unknown trade_mode: {trade_mode}")

    target_notional.name = "target_notional"
    return target_notional


def map_factor_to_target_fraction(
    factor: pd.Series,
    mapper: str = "zscore",
    zscore_window: int = 100,
    clip_abs: float = 1.0,
    allow_short: bool = True,
    long_leverage: float = 1.0,
    short_leverage: float = 1.0,
    trade_mode: str = "continuous",
    entry_long_threshold: float = 0.9,
    entry_short_threshold: float = -0.9,
) -> pd.Series:
    """
    Return target exposure fraction series in [-inf, inf], which will be multiplied by current equity at trade time.
    - continuous: rolling-zscore/sign/percentile normalized and levered asymmetrically
    - threshold: returns +/- leverage when thresholds are crossed; NaN means hold existing
    """
    f = factor.copy().astype(float)

    if mapper == "zscore":
        s = _rolling_zscore(f, zscore_window)
        s = s.clip(lower=-clip_abs, upper=clip_abs)
    elif mapper == "sign":
        s = np.sign(f)
        s = s.clip(lower=-clip_abs, upper=clip_abs)
    elif mapper == "percentile":
        p = _rolling_percentile(f, zscore_window)
        s = (2.0 * p - 1.0).clip(lower=-clip_abs, upper=clip_abs)
    else:
        raise ValueError(f"Unknown mapper: {mapper}")

    # if not allow_short:
    #     s = s.clip(lower=0.0)

    if trade_mode == "continuous":
        s_pos = s.clip(lower=0) * float(long_leverage)
        s_neg = s.clip(upper=0) * float(short_leverage)
        frac = (s_pos + s_neg).fillna(0.0)
        if not allow_short:
            frac[frac < 0] = 0

    elif trade_mode == "threshold":
        frac = pd.Series(data=np.nan, index=s.index, dtype=float)
        frac[(s >= float(entry_long_threshold))] = float(long_leverage)
        frac[(s <= float(entry_short_threshold))] = -float(short_leverage)
        if not allow_short:
            frac[frac < 0] = 0
    else:
        raise ValueError(f"Unknown trade_mode: {trade_mode}")

    frac.name = "target_fraction"
    return frac 