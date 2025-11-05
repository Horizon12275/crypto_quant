import numpy as np
from typing import Callable


def reversal(window_ohlcv: np.ndarray) -> float:
    """
    Simple momentum factor: ratio of last close to average close over the window minus 1.
    Input shape: (lookback, 5 or 6) with columns [open, high, low, close, volume, ...].
    Returns a scalar float.
    """
    wind = 240
    close = window_ohlcv[:, 3]
    if close.size == 0:
        return np.nan
    return -float((close[-1] / close[-wind]) - 1.0)

def illiq(window_ohlcv: np.ndarray) -> float:
    wind = 240
    close = window_ohlcv[:, 3]
    if close.size < wind + 1:
        return np.nan

    ret = close[-wind:] / close[-wind-1:-1] - 1.0
    volume = window_ohlcv[-wind:, 4]
    # 兼容volume=0的情况，避免除以零
    illiq = np.empty_like(ret)
    illiq[:] = np.nan
    valid = volume != 0
    illiq[valid] = np.abs(ret[valid]) / volume[valid]

    return np.nansum(illiq)

def register(registry) -> None:
    registry.register("reversal", reversal)
    registry.register("illiq", illiq)