import pandas as pd
import numpy as np
from typing import Dict, Any, Iterable


def run_portfolio_backtest(
    df: pd.DataFrame,
    eval_times: Iterable[pd.Timestamp],
    trade_delay_minutes: int,
    targets_by_factor: Dict[str, pd.Series],
    initial_capital: float,
    cost_bps: float = 0.0,
) -> Dict[str, Any]:
    """
    Multi-sleeve backtest. `targets_by_factor` maps factor_name to target_notional Series (on minute index).
    Per-sleeve stop-loss is provided via Series.attrs["stop_loss_pct"] on each series (optional).
    """
    opens = df["open"].astype(float)
    index = opens.index

    eval_times = pd.DatetimeIndex(eval_times).intersection(index)

    # Build trade plan per sleeve: trade at eval_time + delay
    plans: Dict[str, Dict[pd.Timestamp, float]] = {}
    for fname, series in targets_by_factor.items():
        series = series.reindex(index)
        plan: Dict[pd.Timestamp, float] = {}
        for t_eval in eval_times:
            t_trade = t_eval + pd.to_timedelta(trade_delay_minutes, unit="min")
            if t_trade in index and t_eval in series.index and pd.notna(series.loc[t_eval]):
                plan[t_trade] = float(series.loc[t_eval])
        plans[fname] = plan

    cash = float(initial_capital)
    units: Dict[str, float] = {fname: 0.0 for fname in targets_by_factor.keys()}

    entries: Dict[str, float] = {fname: None for fname in targets_by_factor.keys()}

    equity = []
    pnl = []

    trades = []

    prev_price = None

    # Buy-and-hold baseline
    if not index.empty:
        bh_units = float(initial_capital) / float(opens.iloc[0]) if float(opens.iloc[0]) != 0 else 0.0
    else:
        bh_units = 0.0
    bh_equity = []
    bh_pnl = []
    bh_prev_price = None

    for t in index:
        price = float(opens.loc[t])

        # Aggregate minute pnl from price move
        if prev_price is not None:
            minute_pnl = sum(units[f] for f in units.keys()) * (price - prev_price)
        else:
            minute_pnl = 0.0

        # Stop-loss per sleeve
        for fname, u in list(units.items()):
            stop_pct = float(targets_by_factor[fname].attrs.get("stop_loss_pct", 0.0)) if hasattr(targets_by_factor[fname], "attrs") else 0.0
            if stop_pct and u != 0 and entries[fname] is not None:
                signed_ret = (price / entries[fname] - 1.0) * (1 if u > 0 else -1)
                if signed_ret <= -abs(stop_pct):
                    delta_units = -u
                    traded_notional = abs(delta_units) * price
                    cost = traded_notional * float(cost_bps) / 10000.0
                    cash -= (delta_units * price)
                    cash -= cost
                    trades.append({
                        "time": t,
                        "factor": fname,
                        "price": price,
                        "target_notional": 0.0,
                        "target_units": 0.0,
                        "delta_units": delta_units,
                        "traded_notional": traded_notional,
                        "cost": cost,
                        "reason": "stop_loss",
                    })
                    units[fname] = 0.0
                    entries[fname] = None

        # Scheduled trades
        for fname, plan in plans.items():
            if t in plan:
                target_notional_t = float(plan[t])
                target_units = target_notional_t / price if price != 0 else 0.0
                delta_units = target_units - units[fname]
                if delta_units != 0:
                    traded_notional = abs(delta_units) * price
                    cost = traded_notional * float(cost_bps) / 10000.0
                    cash -= (delta_units * price)
                    cash -= cost
                    trades.append({
                        "time": t,
                        "factor": fname,
                        "price": price,
                        "target_notional": target_notional_t,
                        "target_units": target_units,
                        "delta_units": delta_units,
                        "traded_notional": traded_notional,
                        "cost": cost,
                    })
                    prev_u = units[fname]
                    units[fname] = target_units
                    if units[fname] == 0.0:
                        entries[fname] = None
                    elif prev_u == 0.0 or np.sign(prev_u) != np.sign(units[fname]):
                        entries[fname] = price

        # Aggregate equity
        equity_t = cash + price * sum(units[f] for f in units.keys())
        equity.append(equity_t)
        pnl.append(minute_pnl)

        # Baseline
        if bh_prev_price is not None:
            bh_minute_pnl = bh_units * (price - bh_prev_price)
        else:
            bh_minute_pnl = 0.0
        bh_equity_t = bh_units * price
        bh_equity.append(bh_equity_t)
        bh_pnl.append(bh_minute_pnl)

        prev_price = price
        bh_prev_price = price

    equity_series = pd.Series(equity, index=index, name="equity")
    pnl_series = pd.Series(pnl, index=index, name="pnl")
    returns = equity_series.pct_change().fillna(0.0)

    trades_df = pd.DataFrame(trades)
    if not trades_df.empty:
        trades_df.set_index("time", inplace=True)
        trades_df.index = pd.DatetimeIndex(trades_df.index)

    results = {
        "equity": equity_series,
        "returns": returns,
        "pnl": pnl_series,
        "trades": trades_df,
        "initial_capital": float(initial_capital),
        "buy_and_hold_equity": pd.Series(bh_equity, index=index, name="buy_and_hold_equity"),
        "buy_and_hold_pnl": pd.Series(bh_pnl, index=index, name="buy_and_hold_pnl"),
    }
    return results
