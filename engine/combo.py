import os
import yaml
import pandas as pd
from typing import Dict, List, Tuple, Optional

from engine.factor_engine import compute_factors
from engine.signal import map_factor_to_target_fraction
from engine.risk import compute_risk_scaler
from engine.profile_loader import load_factor_profile
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
    freq = pd.Timedelta(minutes=int(rebalance_minutes))
    anchor = index[0].normalize()
    schedule = pd.date_range(start=anchor, end=index[-1], freq=freq, tz=index.tz)
    schedule = schedule[schedule >= index[0]]
    return index.intersection(schedule)


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
    Legacy builder kept for backward compatibility (reads from reports);
    retained as-is so older configs continue to work.
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

    # Shared base risk scaler: vol_target / recent_vol (no cap)
    base_scaler = compute_risk_scaler(
        opens=df["open"].astype(float),
        vol_lookback_days=int(risk_cfg.get("vol_lookback_days", 63)),
        vol_target_ann=float(risk_cfg.get("vol_target_ann", 0.25)),
        rebalance=str(risk_cfg.get("rebalance", "monthly")),
        rebalance_days=int(risk_cfg.get("rebalance_days", 30)),
    )

    union_eval_times = pd.DatetimeIndex([])
    targets: Dict[str, pd.Series] = {}

    for name, weight in zip(factors, weights):
        factor_fn = registry.get(name)
        report_cfg = _load_factor_config_from_report(reports_dir, name) or {}
        signals_cfg = report_cfg.get("signals", {})
        exec_cfg = report_cfg.get("execution", {})

        lookback = int(signals_cfg.get("lookback_minutes", global_signals_cfg.get("lookback_minutes", 720)))
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
        # Map without per-sleeve leverage to avoid double-counting; leverage enforced via risk scaler floor
        tn_unlevered = map_factor_to_target_fraction(
            factor=factor_series,
            capital=capital_i,
            mapper=mapper,
            zscore_window=zscore_window,
            clip_abs=clip_abs,
            allow_short=allow_short,
            long_leverage=1.0,
            short_leverage=1.0,
            trade_mode=trade_mode,
            entry_long_threshold=entry_long_threshold,
            entry_short_threshold=entry_short_threshold,
        )

        # Apply per-sleeve leverage floor: scaler = max(base_scaler, sleeve_leverage)
        leverage_floor = max(abs(long_leverage), abs(short_leverage))
        scaler_i = base_scaler.clip(lower=leverage_floor)

        tn = tn_unlevered.reindex(df.index)
        tn = tn * scaler_i
        tn.name = f"target_notional_{name}"
        targets[name] = tn
        targets[name].attrs = {"stop_loss_pct": stop_loss_pct}

    union_eval_times = union_eval_times.intersection(df.index)
    return union_eval_times, targets


def build_combo_targets_from_profiles(
    df: pd.DataFrame,
    initial_capital: float,
    sleeves: List[Dict],
    risk_cfg: Dict,
) -> Tuple[pd.DatetimeIndex, Dict[str, pd.Series], Dict[str, float]]:
    """
    Return (union_eval_times, fraction_by_factor, weights_by_factor).
    Fractions are risk-scaled and will be sized to equity*weight at trade time.
    """
    if not sleeves:
        raise ValueError("sleeves list is empty")

    weights = [float(s.get("weight", 1.0)) for s in sleeves]
    total = sum(weights)
    weights = [w / total for w in weights] if total != 0 else [1.0 / len(sleeves)] * len(sleeves)

    base_scaler = compute_risk_scaler(
        opens=df["open"].astype(float),
        vol_lookback_days=int(risk_cfg.get("vol_lookback_days", 63)),
        vol_target_ann=float(risk_cfg.get("vol_target_ann", 0.25)),
        rebalance=str(risk_cfg.get("rebalance", "monthly")),
        rebalance_days=int(risk_cfg.get("rebalance_days", 30)),
    )

    union_eval_times = pd.DatetimeIndex([])
    fractions: Dict[str, pd.Series] = {}
    weight_map: Dict[str, float] = {}

    for sleeve, weight in zip(sleeves, weights):
        prof_name = sleeve["profile"]
        prof = load_factor_profile(prof_name)

        lookback = int(sleeve.get("lookback_minutes", prof.get("lookback_minutes", 720)))
        mapper = sleeve.get("mapper", prof.get("mapper", "zscore"))
        zscore_window = int(sleeve.get("zscore_window", prof.get("zscore_window", 100)))
        clip_abs = float(sleeve.get("clip_abs_signal", prof.get("clip_abs_signal", 1.0)))
        trade_mode = sleeve.get("trade_mode", prof.get("trade_mode", "continuous"))
        entry_long_threshold = float(sleeve.get("entry_long_threshold", prof.get("entry_long_threshold", 0.9)))
        entry_short_threshold = float(sleeve.get("entry_short_threshold", prof.get("entry_short_threshold", -0.9)))
        allow_short = bool(sleeve.get("allow_short", prof.get("allow_short", True)))
        long_leverage = float(sleeve.get("long_leverage", prof.get("long_leverage", 1.0)))
        short_leverage = float(sleeve.get("short_leverage", prof.get("short_leverage", 1.0)))
        rebalance_minutes = int(sleeve.get("rebalance_minutes", prof.get("rebalance_minutes", 720)))
        stop_loss_pct = float(sleeve.get("stop_loss_pct", prof.get("stop_loss_pct", 0.0)))

        eval_times = _compute_eval_times(df.index, rebalance_minutes)
        union_eval_times = union_eval_times.union(eval_times)

        factor_fn = registry.get(prof_name)
        factor_series = compute_factors(df=df, factor_fn=factor_fn, lookback_minutes=lookback, eval_times=eval_times)

        frac = map_factor_to_target_fraction(
            factor=factor_series,
            mapper=mapper,
            zscore_window=zscore_window,
            clip_abs=clip_abs,
            allow_short=allow_short,
            long_leverage=1.0,  # leverage via scaler floor
            short_leverage=1.0,
            trade_mode=trade_mode,
            entry_long_threshold=entry_long_threshold,
            entry_short_threshold=entry_short_threshold,
        )

        leverage_floor = max(abs(long_leverage), abs(short_leverage))
        scaler_i = base_scaler.clip(lower=leverage_floor)
        frac_scaled = frac.reindex(df.index) * scaler_i
        frac_scaled.name = f"target_fraction_{prof_name}"
        frac_scaled.attrs = {"stop_loss_pct": stop_loss_pct}

        fractions[prof_name] = frac_scaled
        weight_map[prof_name] = weight

    union_eval_times = union_eval_times.intersection(df.index)
    return union_eval_times, fractions, weight_map
