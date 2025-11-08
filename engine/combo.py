import os
import yaml
import pandas as pd
from typing import Dict, List, Tuple, Optional

from engine.factor_engine import compute_factors
from engine.signal import map_factor_to_target_notional
from engine.risk import compute_risk_scaler
from factors.registry import registry


DefSettings = Dict[str, object]


def _load_factor_config_from_report(base_dir: str, factor_name: str) -> Optional[dict]:
    """
    Try to load prior used config from reports/{factor_name}/backtest.yaml for parameters.
    Returns dict or None if not found.
    """
    path = os.path.join(base_dir, factor_name, "backtest.yaml")
    if not os.path.exists(path):
        return None
    with open(path, "r") as f:
        try:
            return yaml.safe_load(f)
        except Exception:
            return None


def _compute_eval_times(index: pd.DatetimeIndex, rebalance_minutes: int) -> pd.DatetimeIndex:
    if len(index) == 0:
        return index
    first = index[0]
    aligned_start = first + pd.Timedelta(minutes=(rebalance_minutes - (first.minute % rebalance_minutes)) % rebalance_minutes)
    return index[(index >= aligned_start) & (((index - aligned_start).asi8 // 60_000_000_000) % rebalance_minutes == 0)]


def build_combo_targets(
    df: pd.DataFrame,
    initial_capital: float,
    global_signals_cfg: dict,
    global_exec_cfg: dict,
    combo_cfg: dict,
    risk_cfg: dict,
    reports_dir: str,
) -> Tuple[pd.DatetimeIndex, Dict[str, pd.Series]]:
    """
    Return union eval_times and dict of {factor_name: target_notional_series} after applying shared risk scaler.
    Equal weights if combo_cfg.weights missing or empty.
    """
    factors: List[str] = combo_cfg.get("factors", []) or []
    weights: List[float] = combo_cfg.get("weights", []) or []
    n = len(factors)
    if n == 0:
        raise ValueError("combo.factors is empty")

    if not weights or len(weights) != n:
        weights = [1.0 / n] * n
    else:
        total = sum(weights)
        if total == 0:
            weights = [1.0 / n] * n
        else:
            weights = [w / total for w in weights]

    # Shared risk scaler
    scaler = compute_risk_scaler(
        opens=df["open"].astype(float),
        vol_lookback_days=int(risk_cfg.get("vol_lookback_days", 63)),
        vol_target_ann=float(risk_cfg.get("vol_target_ann", 0.25)),
        max_gross_leverage=float(risk_cfg.get("max_gross_leverage", 1.0)),
        rebalance=str(risk_cfg.get("rebalance", "monthly")),
        rebalance_days=int(risk_cfg.get("rebalance_days", 30)),
    )

    union_eval_times = pd.DatetimeIndex([])
    targets: Dict[str, pd.Series] = {}

    for name, weight in zip(factors, weights):
        factor_fn = registry.get(name)
        # Load per-factor params from report config if available
        report_cfg = _load_factor_config_from_report(reports_dir, name) or {}
        signals_cfg = report_cfg.get("signals", {})
        exec_cfg = report_cfg.get("execution", {})

        lookback = int(signals_cfg.get("lookback_minutes", global_signals_cfg.get("lookback_minutes", 720)))
        k_minutes = int(signals_cfg.get("k_minutes", global_signals_cfg.get("k_minutes", 60)))
        mapper = signals_cfg.get("mapper", global_signals_cfg.get("mapper", "zscore"))
        zscore_window = int(signals_cfg.get("zscore_window", global_signals_cfg.get("zscore_window", 100)))
        clip_abs = float(signals_cfg.get("clip_abs_signal", global_signals_cfg.get("clip_abs_signal", 1.0)))
        trade_mode = signals_cfg.get("trade_mode", global_signals_cfg.get("trade_mode", "continuous"))
        entry_long_threshold = float(signals_cfg.get("entry_long_threshold", global_signals_cfg.get("entry_long_threshold", 0.9)))
        entry_short_threshold = float(signals_cfg.get("entry_short_threshold", global_signals_cfg.get("entry_short_threshold", -0.9)))

        allow_short = bool(exec_cfg.get("allow_short", global_exec_cfg.get("allow_short", True)))
        long_leverage = float(exec_cfg.get("long_leverage", global_exec_cfg.get("long_leverage", 1.0)))
        short_leverage = float(exec_cfg.get("short_leverage", global_exec_cfg.get("short_leverage", 1.0)))
        rebalance_minutes = int(exec_cfg.get("rebalance_minutes", global_exec_cfg.get("rebalance_minutes", 720)))
        stop_loss_pct = float(exec_cfg.get("stop_loss_pct", global_exec_cfg.get("stop_loss_pct", 0.0)))

        eval_times = _compute_eval_times(df.index, rebalance_minutes)
        union_eval_times = union_eval_times.union(eval_times)

        factor_series = compute_factors(df=df, factor_fn=factor_fn, lookback_minutes=lookback, eval_times=eval_times)

        capital_i = float(initial_capital) * float(weight)
        tn = map_factor_to_target_notional(
            factor=factor_series,
            capital=capital_i,
            mapper=mapper,
            zscore_window=zscore_window,
            clip_abs=clip_abs,
            allow_short=allow_short,
            long_leverage=long_leverage,
            short_leverage=short_leverage,
            trade_mode=trade_mode,
            entry_long_threshold=entry_long_threshold,
            entry_short_threshold=entry_short_threshold,
        )
        # Apply risk scaler
        tn = tn.reindex(df.index).reindex(eval_times).reindex(df.index, method=None)
        tn = tn.reindex(df.index)  # ensure minute index
        tn = tn * scaler
        tn.name = f"target_notional_{name}"
        targets[name] = tn

        # Stash stop-loss for this factor (return separately via metadata? We'll let portfolio_backtester accept per-factor sl settings.)
        targets[name].attrs = {"stop_loss_pct": stop_loss_pct}

    union_eval_times = union_eval_times.intersection(df.index)
    return union_eval_times, targets
