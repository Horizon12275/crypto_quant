import pandas as pd
import numpy as np
from typing import Iterable, Dict, Any, Optional


def run_backtest(
    df: pd.DataFrame,
    eval_times: Iterable[pd.Timestamp],
    trade_delay_minutes: int,
    target_notional: Optional[pd.Series] = None,
    target_fraction: Optional[pd.Series] = None,
    initial_capital: float = 0.0,
    cost_bps: float = 0.0,
    stop_loss_pct: float = 0.0,
    min_trade_notional: float = 1e-6,
) -> Dict[str, Any]:
    """
    Single-asset backtest.
    - If target_fraction is provided, target notional at trade time = fraction * current equity before trade.
    - Else, use fixed-dollar target_notional.
    - Ignore micro trades where abs(delta_units)*price < min_trade_notional to avoid float noise.
    """
    opens = df["open"].astype(float)
    index = opens.index

    eval_times = pd.DatetimeIndex(eval_times)
    eval_times = eval_times.intersection(index)

    is_fraction = target_fraction is not None
    series = target_fraction if is_fraction else target_notional
    if series is None:
        raise ValueError("Provide either target_fraction or target_notional")
    series = series.reindex(eval_times)

    plan = {}
    for t_eval in eval_times:
        t_trade = t_eval + pd.to_timedelta(trade_delay_minutes, unit="min")
        if t_trade in index:
            val = series.loc[t_eval] if t_eval in series.index else np.nan
            plan[t_trade] = float(val) if pd.notna(val) else np.nan

    cash = float(initial_capital)
    units = 0.0

    entry_price = None
    entry_time = None

    equity = []
    pnl = []
    pos_units = []
    pos_notional = []

    trades = []

    prev_price = None

    for t in index:
        price = float(opens.loc[t])

        minute_pnl = units * (price - (prev_price if prev_price is not None else price))

        if stop_loss_pct and units != 0 and entry_price is not None:
            signed_ret_from_entry = (price / entry_price - 1.0) * (1 if units > 0 else -1)
            if signed_ret_from_entry <= -abs(float(stop_loss_pct)):
                delta_units = -units
                traded_notional = abs(delta_units) * price
                if traded_notional >= min_trade_notional:
                    cost = traded_notional * float(cost_bps) / 10000.0
                    cash -= (delta_units * price)
                    cash -= cost
                    trades.append({
                        "time": t,
                        "price": price,
                        "target_notional": 0.0,
                        "target_units": 0.0,
                        "delta_units": delta_units,
                        "traded_notional": traded_notional,
                        "cost": cost,
                        "reason": "stop_loss",
                    })
                units = 0.0
                entry_price = None
                entry_time = None

        if t in plan and pd.notna(plan[t]):
            desired = plan[t]
            if is_fraction:
                equity_before = cash + units * price
                target_notional_t = desired * equity_before
            else:
                target_notional_t = desired
            target_units = target_notional_t / price if price != 0 else 0.0
            delta_units = target_units - units
            traded_notional = abs(delta_units) * price
            if traded_notional >= min_trade_notional:
                cost = traded_notional * float(cost_bps) / 10000.0
                cash -= (delta_units * price)
                cash -= cost
                trades.append({
                    "time": t,
                    "price": price,
                    "target_notional": target_notional_t,
                    "target_units": target_units,
                    "delta_units": delta_units,
                    "traded_notional": traded_notional,
                    "cost": cost,
                })
                prev_units = units
                units = target_units
                if units == 0.0:
                    entry_price = None
                    entry_time = None
                elif prev_units == 0.0 or np.sign(prev_units) != np.sign(units):
                    entry_price = price
                    entry_time = t

        equity_t = cash + units * price
        equity.append(equity_t)
        pnl.append(minute_pnl)
        pos_units.append(units)
        pos_notional.append(units * price)

        prev_price = price

    equity_series = pd.Series(equity, index=index, name="equity")
    pnl_series = pd.Series(pnl, index=index, name="pnl")
    pos_units_series = pd.Series(pos_units, index=index, name="position_units")
    pos_notional_series = pd.Series(pos_notional, index=index, name="position_notional")

    returns = equity_series.pct_change().fillna(0.0)

    trades_df = pd.DataFrame(trades)
    if not trades_df.empty:
        trades_df.set_index("time", inplace=True)
        trades_df.index = pd.DatetimeIndex(trades_df.index)

    results = {
        "equity": equity_series,
        "returns": returns,
        "pnl": pnl_series,
        "position_units": pos_units_series,
        "position_notional": pos_notional_series,
        "trades": trades_df,
        "initial_capital": float(initial_capital),
        "buy_and_hold_equity": (float(initial_capital) / float(opens.iloc[0]) if not opens.empty and float(opens.iloc[0]) != 0 else 0.0) * opens,
        "buy_and_hold_pnl": ((float(initial_capital) / float(opens.iloc[0]) if not opens.empty and float(opens.iloc[0]) != 0 else 0.0) * opens).diff().fillna(0.0),
    }
    return results 