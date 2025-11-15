import os
import argparse
import yaml
import pandas as pd

from factors.registry import register_builtin_factors, registry
from engine.data_loader import load_minute_data
from engine.profile_loader import load_factor_profile
from engine.factor_engine import compute_factors
from engine.signal import map_factor_to_target_notional, map_factor_to_target_fraction
from engine.backtester import run_backtest
from engine.portfolio_backtester import run_portfolio_backtest
from engine.combo import build_combo_targets_from_profiles
from engine.metrics import compute_ic_metrics, compute_performance_summary
from engine.report import generate_reports


def save_config(out_dir: str, cfg: dict) -> None:
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "backtest.yaml"), "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)


def _compute_eval_times(index: pd.DatetimeIndex, rebalance_minutes: int) -> pd.DatetimeIndex:
    if len(index) == 0:
        return index
    freq = pd.Timedelta(minutes=int(rebalance_minutes))
    anchor = index[0].normalize()
    schedule = pd.date_range(start=anchor, end=index[-1], freq=freq, tz=index.tz)
    schedule = schedule[schedule >= index[0]]
    return index.intersection(schedule)


def run_single(cfg: dict) -> None:
    data_cfg = cfg["data"]
    df = load_minute_data(
        csv_path=data_cfg["source_csv"],
        tz=data_cfg.get("timezone", "UTC"),
        start_ts=pd.Timestamp(data_cfg["start"], tz="UTC"),
        end_ts=pd.Timestamp(data_cfg["end"], tz="UTC"),
        forward_fill=bool(data_cfg.get("forward_fill", True)),
    )

    profile_name = cfg["factor_profile"]
    prof = load_factor_profile(profile_name)

    lookback = int(prof.get("lookback_minutes", 720))
    rebalance_minutes = int(prof.get("rebalance_minutes", 720))
    mapper = prof.get("mapper", "zscore")
    zscore_window = int(prof.get("zscore_window", 100))
    clip_abs = float(prof.get("clip_abs_signal", 1.0))
    trade_mode = prof.get("trade_mode", "continuous")
    entry_long_threshold = float(prof.get("entry_long_threshold", 0.9))
    entry_short_threshold = float(prof.get("entry_short_threshold", -0.9))
    allow_short = bool(prof.get("allow_short", True))
    long_leverage = float(prof.get("long_leverage", 1.0))
    short_leverage = float(prof.get("short_leverage", 1.0))
    stop_loss_pct = float(prof.get("stop_loss_pct", 0.0))

    exec_cfg = cfg["execution"]
    initial_capital = float(exec_cfg["initial_capital"])
    trade_delay = int(exec_cfg.get("trade_delay_minutes", 1))
    cost_bps = float(exec_cfg.get("cost_bps", 0.0))

    eval_times = _compute_eval_times(df.index, rebalance_minutes)

    factor_fn = registry.get(profile_name)
    factor_series = compute_factors(df=df, factor_fn=factor_fn, lookback_minutes=lookback, eval_times=eval_times)

    ic_metrics = {}
    if trade_mode != "threshold":
        k_minutes = int(prof.get("k_minutes", 60))
        from engine.labeler import build_labels
        labels = build_labels(df=df, eval_times=eval_times, k_minutes=k_minutes)
        ic_metrics = compute_ic_metrics(factor=factor_series, label=labels, rolling_window=50)

    target_fraction = map_factor_to_target_fraction(
        factor=factor_series,
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

    results = run_backtest(
        df=df,
        eval_times=eval_times,
        trade_delay_minutes=trade_delay,
        target_fraction=target_fraction,
        initial_capital=initial_capital,
        cost_bps=cost_bps,
        stop_loss_pct=stop_loss_pct,
    )

    summary = compute_performance_summary(results["equity"], initial_capital)

    out_dir = os.path.join(cfg["reporting"]["out_dir"], cfg["reporting"].get("folder", profile_name) or profile_name)
    save_config(out_dir, cfg)

    generate_reports(
        out_dir=out_dir,
        ic_metrics=ic_metrics,
        results=results,
        plots=cfg["reporting"].get("plots", []),
        write_results_core=bool(cfg["reporting"].get("write_results_core", False)),
        summary=summary,
        write_trades_csv=bool(cfg["reporting"].get("write_trades_csv", False)),
    )

    print("Backtest complete. Reports written to:", out_dir)


def run_portfolio(cfg: dict) -> None:
    data_cfg = cfg["data"]
    df = load_minute_data(
        csv_path=data_cfg["source_csv"],
        tz=data_cfg.get("timezone", "UTC"),
        start_ts=pd.Timestamp(data_cfg["start"], tz="UTC"),
        end_ts=pd.Timestamp(data_cfg["end"], tz="UTC"),
        forward_fill=bool(data_cfg.get("forward_fill", True)),
    )

    sleeves = cfg["sleeves"]
    risk_cfg = cfg.get("risk_control", {})

    exec_cfg = cfg["execution"]
    initial_capital = float(exec_cfg["initial_capital"])
    trade_delay = int(exec_cfg.get("trade_delay_minutes", 1))
    cost_bps = float(exec_cfg.get("cost_bps", 0.0))

    from engine.combo import build_combo_targets_from_profiles
    eval_times, fraction_by_factor, weights_by_factor = build_combo_targets_from_profiles(
        df=df,
        initial_capital=initial_capital,
        sleeves=sleeves,
        risk_cfg=risk_cfg,
    )

    results = run_portfolio_backtest(
        df=df,
        eval_times=eval_times,
        trade_delay_minutes=trade_delay,
        targets_by_factor=fraction_by_factor,
        initial_capital=initial_capital,
        cost_bps=cost_bps,
        weights_by_factor=weights_by_factor,
    )

    summary = compute_performance_summary(results["equity"], initial_capital)

    name = cfg.get("name", "portfolio")
    out_dir = os.path.join(cfg["reporting"]["out_dir"], cfg["reporting"].get("folder", name) or name)
    save_config(out_dir, cfg)

    generate_reports(
        out_dir=out_dir,
        ic_metrics={},
        results=results,
        plots=cfg["reporting"].get("plots", []),
        write_results_core=bool(cfg["reporting"].get("write_results_core", False)),
        summary=summary,
        write_trades_csv=bool(cfg["reporting"].get("write_trades_csv", False)),
    )

    print("Backtest complete. Reports written to:", out_dir)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to unified config yaml")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    register_builtin_factors()

    if "sleeves" in cfg and cfg["sleeves"]:
        run_portfolio(cfg)
    elif "factor_profile" in cfg and cfg["factor_profile"]:
        run_single(cfg)
    else:
        raise ValueError("Config must define either 'factor_profile' (single) or 'sleeves' (portfolio)")


if __name__ == "__main__":
    main()
