import numpy as np
from typing import Callable


def example_momo(window_ohlcv: np.ndarray) -> float:
    """
    Simple momentum factor: ratio of last close to average close over the window minus 1.
    Input shape: (lookback, 5 or 6) with columns [open, high, low, close, volume, ...].
    Returns a scalar float.
    """
    close = window_ohlcv[:, 3]
    if close.size == 0:
        return np.nan
    # mean_close = np.mean(close)
    # if mean_close == 0:
    #     return 0.0
    return -float((close[-1] / close[-180]) - 1.0)


def register(registry) -> None:
    registry.register("example_momo_percentile", example_momo) 