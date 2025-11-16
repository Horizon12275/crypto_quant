import numpy as np
from typing import Callable


def reversal(window_ohlcv: np.ndarray) -> float:
    """
    Simple momentum factor: ratio of last close to average close over the window minus 1.
    Input shape: (lookback, 5 or 6) with columns [open, high, low, close, volume, ...].
    Returns a scalar float.
    """
    wind = 720
    close = window_ohlcv[:, 3]
    if close.size == 0:
        return np.nan
    return -float((close[-1] / close[-wind]) - 1.0)

def volatility(window_ohlcv: np.ndarray) -> float:
    wind = 720
    close = window_ohlcv[:, 3]
    ret = close[-wind:] / close[-wind-1:-1] - 1.0
    if close.size == 0:
        return np.nan
    return -np.std(ret)

def illiq(window_ohlcv: np.ndarray) -> float:
    wind = 720
    close = window_ohlcv[:, 3]
    if close.size < wind + 1:
        return np.nan

    ret = close[-wind:] / close[-wind-1:-1] - 1.0
    volume = window_ohlcv[-wind:, 4]
    # 兼容volume=0的情况，避免除以零
    illiq = np.zeros_like(ret)
    valid = volume != 0
    illiq[valid] = np.abs(ret[valid]) / volume[valid]

    return np.nansum(illiq)

def pvcorr(window_ohlcv: np.ndarray) -> float:
    wind = 1440
    close = window_ohlcv[-wind:, 3]
    volume = window_ohlcv[-wind:, 4]
    if close.size != volume.size:
        return np.nan
    return -np.corrcoef(close, volume)[0, 1]

def double_ma(window_ohlcv: np.ndarray) -> float:
    wind = 720
    close = window_ohlcv[:, 3]
    if close.size == 0:
        return np.nan
    return np.mean(close[-wind:]) / np.mean(close[-5*wind:])

def ret_skewness(window_ohlcv: np.ndarray) -> float:
    wind = 1440
    close = window_ohlcv[:, 3]
    if close.size == 0:
        return np.nan
    ret = close[-wind:] / close[-wind-1:-1] - 1.0
    if ret.size == 0:
        return np.nan
    mean = np.mean(ret)
    std = np.std(ret)
    if std == 0:
        return 0.0
    skew = np.mean(((ret - mean) / std) ** 3)
    return skew

def ret_kurt(window_ohlcv: np.ndarray) -> float:
    wind = 1440
    close = window_ohlcv[:, 3]
    if close.size == 0:
        return np.nan
    ret = close[-wind:] / close[-wind-1:-1] - 1.0
    if ret.size == 0:
        return np.nan
    mean = np.mean(ret)
    std = np.std(ret)
    if std == 0:
        return 0.0
    kurt = np.mean(((ret - mean) / std) ** 4)
    return -(kurt - 3.0)

def rsi(window_ohlcv: np.ndarray) -> float:
    wind = 720
    close = window_ohlcv[:, 3]
    if close.size < wind + 1:
        return np.nan
    delta = close[-wind-1:][1:] - close[-wind-1:][:-1]
    up = np.where(delta > 0, delta, 0)
    down = np.where(delta < 0, -delta, 0)
    sum_gain = np.sum(up)
    sum_loss = np.sum(down)
    
    return (100.0 * sum_gain / (sum_gain + sum_loss))

def volume_vol_vol(window_ohlcv: np.ndarray, n: int = 10, win_size: int = 360) -> float:
    """
    :param window_ohlcv: OHLCV数据, shape=(window, 5)
    :param n: 窗口数。
    :param win_size: 单个窗口的长度。
    :return: volume的波动率的波动率。
    """
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
    registry.register("reversal", reversal)
    registry.register("illiq", illiq)
    registry.register("volatility", volatility)
    registry.register("pvcorr", pvcorr)
    registry.register("double_ma", double_ma)
    registry.register("ret_skewness", ret_skewness)
    registry.register("rsi", rsi)
    registry.register("ret_kurt", ret_kurt)
    registry.register("volume_vol_vol", volume_vol_vol)