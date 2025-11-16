import numpy as np
from typing import Callable

def volume_skewness(window_ohlcv: np.ndarray) -> float:
    wind = 1440
    volume = window_ohlcv[-wind:, 4]
    if volume.size == 0:
        return np.nan
    mean = np.mean(volume)
    std = np.std(volume)
    if std == 0:
        return 0.0
    skew = np.mean(((volume - mean) / std) ** 3)
    return -skew

def volume_peak_count(window_ohlcv: np.ndarray, peak_threshold: float = 2.0) -> float:
    wind = 1440
    volume = window_ohlcv[-wind:, 4]
    if volume.size == 0:
        return np.nan
    mean = np.mean(volume)
    std = np.std(volume)
    if std == 0:
        return 0.0
    peaks = volume > (mean + peak_threshold * std)
    return float(np.sum(peaks))

def volume_vol_vol(window_ohlcv: np.ndarray, n: int = 10, win_size: int = 360) -> float:
    volume = window_ohlcv[:, 4]
    total_length = n * win_size
    if volume.size < total_length:
        return np.nan
    std_list = []
    for i in range(n):
        start = -total_length + i * win_size - 1
        end = start + win_size
        vol_win = volume[start:end] if start != 0 else volume[:end]
        # if vol_win.size < win_size:
        #     return np.nan
        std_list.append(np.std(vol_win))
    if len(std_list) < 2:
        return np.nan
    return np.std(std_list)


def register(registry) -> None:
    registry.register("volume_peak_count", volume_peak_count)
    registry.register("volume_skewness", volume_skewness)
    registry.register("volume_vol_vol", volume_vol_vol)