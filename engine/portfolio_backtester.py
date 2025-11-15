import pandas as pd
import numpy as np
from typing import Dict, Any, Iterable, Optional


def run_portfolio_backtest(
    df: pd.DataFrame,
    eval_times: Iterable[pd.Timestamp],
    trade_delay_minutes: int,
    targets_by_factor: Dict[str, pd.Series],
    initial_capital: float,
    cost_bps: float = 0.0,
    weights_by_factor: Optional[Dict[str, float]] = None,
    min_trade_notional: float = 1e-6,
) -> Dict[str, Any]:
    """
    Multi-sleeve backtest. `targets_by_factor` maps factor_name to series values interpreted as FRACTIONS
    (exposure multipliers). At trade time: target_notional_i = fraction_i * (portfolio_equity_before * weight_i).
    Per-sleeve stop-loss via Series.attrs["stop_loss_pct"].
    Ignore micro trades where abs(delta_units)*price < min_trade_notional.
    """
    opens = df["open"].astype(float)
    index = opens.index

    eval_times = pd.DatetimeIndex(eval_times).intersection(index)

    if weights_by_factor is None or set(weights_by_factor.keys()) != set(targets_by_factor.keys()):
        n = len(targets_by_factor)
        weights_by_factor = {k: 1.0 / n for k in targets_by_factor.keys()} if n > 0 else {}

    plans: Dict[str, Dict[pd.Timestamp, float]] = {}
    for fname, series in targets_by_factor.items():
        series = series.reindex(eval_times)
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

    if not index.empty:
        bh_units = float(initial_capital) / float(opens.iloc[0]) if float(opens.iloc[0]) != 0 else 0.0
    else:
        bh_units = 0.0
    bh_equity = []
    bh_pnl = []
    bh_prev_price = None

    for t in index:
        price = float(opens.loc[t])

        port_units = sum(units.values())
        minute_pnl = port_units * (price - (prev_price if prev_price is not None else price))

        for fname, u in list(units.items()):
            stop_pct = float(targets_by_factor[fname].attrs.get("stop_loss_pct", 0.0)) if hasattr(targets_by_factor[fname], "attrs") else 0.0
            if stop_pct and u != 0 and entries[fname] is not None:
                signed_ret = (price / entries[fname] - 1.0) * (1 if u > 0 else -1)
                if signed_ret <= -abs(stop_pct):
                    delta_units = -u
                    traded_notional = abs(delta_units) * price
                    if traded_notional >= min_trade_notional:
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

        equity_before = cash + sum(units.values()) * price
        for fname, plan in plans.items():
            if t in plan:
                fraction = float(plan[t])
                weight = float(weights_by_factor.get(fname, 0.0))
                target_notional_t = fraction * equity_before * weight
                target_units = target_notional_t / price if price != 0 else 0.0
                delta_units = target_units - units[fname]
                traded_notional = abs(delta_units) * price
                if traded_notional >= min_trade_notional:
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

        equity_t = cash + sum(units.values()) * price
        equity.append(equity_t)
        pnl.append(minute_pnl)

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
