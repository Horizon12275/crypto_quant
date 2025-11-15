import pandas as pd
import numpy as np
from typing import Dict


def _align(factor: pd.Series, label: pd.Series) -> pd.DataFrame:
    df = pd.concat({"factor": factor, "label": label}, axis=1)
    return df.dropna()


def compute_ic_metrics(
    factor: pd.Series,
    label: pd.Series,
    rolling_window: int = 50,
) -> Dict[str, pd.Series]:
    """
    Compute Pearson and Spearman IC time series and their cumulative sums.
    """
    aligned = _align(factor, label)
    f = aligned["factor"]
    y = aligned["label"]

    pearson_ic = f.rolling(window=rolling_window, min_periods=rolling_window).corr(y)
    # Spearman via rank transformation then Pearson
    f_rank = f.rank(method="average")
    y_rank = y.rank(method="average")
    spearman_ic = f_rank.rolling(window=rolling_window, min_periods=rolling_window).corr(y_rank)

    ic_cumsum = pearson_ic.fillna(0.0).cumsum()

    return {
        "pearson_ic": pearson_ic.rename("pearson_ic"),
        "spearman_ic": spearman_ic.rename("spearman_ic"),
        "ic_cumsum": ic_cumsum.rename("ic_cumsum"),
    }


def _max_drawdown(series: pd.Series) -> float:
    if series.empty:
        return 0.0
    roll_max = series.cummax()
    drawdowns = series / roll_max - 1.0
    return float(drawdowns.min())


def compute_performance_summary(equity: pd.Series, initial_capital: float) -> Dict[str, float]:
    """
    Compute key metrics using daily equity: annualized return/vol, Sharpe, max drawdown, Calmar.
    Robust to equity drawdowns beyond -100% (caps CAGR at -100%) and non-finite returns.
    Annualization uses 365 days for crypto markets.
    """
    if equity is None or equity.empty:
        return {"annual_return": 0.0, "annual_vol": 0.0, "sharpe": 0.0, "max_drawdown": 0.0, "calmar": 0.0, "days": 0}

    daily = equity.resample("1D").last().dropna()
    if len(daily) < 2:
        return {"annual_return": 0.0, "annual_vol": 0.0, "sharpe": 0.0, "max_drawdown": 0.0, "calmar": 0.0, "days": len(daily)}

    daily_returns = daily.pct_change().replace([np.inf, -np.inf], np.nan).dropna()
    num_days = len(daily_returns)

    ratio = float(daily.iloc[-1] / daily.iloc[0]) if daily.iloc[0] != 0 else 0.0
    if ratio <= 0:
        annual_return = -1.0
    else:
        annual_return = ratio ** (365.0 / num_days) - 1.0

    daily_vol = float(daily_returns.std(ddof=0)) if num_days > 0 else 0.0
    annual_vol = daily_vol * (365.0 ** 0.5)

    sharpe = (annual_return / annual_vol) if annual_vol > 0 else 0.0

    max_dd = _max_drawdown(daily)
    calmar = (annual_return / abs(max_dd)) if max_dd < 0 else 0.0

    return {
        "annual_return": float(annual_return),
        "annual_vol": float(annual_vol),
        "sharpe": float(sharpe),
        "max_drawdown": float(max_dd),
        "calmar": float(calmar),
        "days": int(num_days),
    } 