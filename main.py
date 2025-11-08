import os
import copy
import yaml
from datetime import datetime, timezone
import pandas as pd

from engine.data_loader import load_minute_data
from engine.factor_engine import compute_factors
from engine.labeler import build_labels
from engine.signal import map_factor_to_target_notional
from engine.backtester import run_backtest
from engine.portfolio_backtester import run_portfolio_backtest
from engine.metrics import compute_ic_metrics, compute_performance_summary
from engine.report import generate_reports
from engine.combo import build_combo_targets
from factors.registry import registry, register_builtin_factors


def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def ensure_out_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def save_config(out_dir: str, cfg: dict) -> None:
    try:
        subset = {
            k: cfg[k]
            for k in ["mode", "data", "signals", "execution", "combo", "risk_control", "reporting"]
            if k in cfg
        }
        with open(os.path.join(out_dir, "backtest.yaml"), "w") as f:
            yaml.safe_dump(subset, f, sort_keys=False)
    except Exception:
        # Best-effort; do not crash backtest on write failure
        pass


if __name__ == "__main__":
    cfg_path = os.path.join("config", "backtest.yaml")
    cfg = load_config(cfg_path)

    register_builtin_factors()

    data_cfg = cfg["data"]
    src_csv = data_cfg["source_csv"]

    df = load_minute_data(
        csv_path=src_csv,
        tz=data_cfg.get("timezone", "UTC"),
        start_ts=pd.Timestamp(data_cfg["start"], tz="UTC"),
        end_ts=pd.Timestamp(data_cfg["end"], tz="UTC"),
        forward_fill=bool(data_cfg.get("forward_fill", True)),
    )

    mode = cfg.get("mode", "single")

    base_out_dir = cfg["reporting"]["out_dir"]
    write_results_core = bool(cfg["reporting"].get("write_results_core", False))
    folder_override = cfg["reporting"].get("folder")

    if mode == "combo":
        signals_cfg = cfg.get("signals", {})
        exec_cfg = cfg.get("execution", {})
        combo_cfg = cfg.get("combo", {})
        risk_cfg = cfg.get("risk_control", {})

        eval_times, targets_by_factor = build_combo_targets(
            df=df,
            initial_capital=float(exec_cfg["initial_capital"]),
            global_signals_cfg=signals_cfg,
            global_exec_cfg=exec_cfg,
            combo_cfg=combo_cfg,
            risk_cfg=risk_cfg,
            reports_dir=base_out_dir,
        )

        results = run_portfolio_backtest(
            df=df,
            eval_times=eval_times,
            trade_delay_minutes=int(exec_cfg.get("trade_delay_minutes", 1)),
            targets_by_factor=targets_by_factor,
            initial_capital=float(exec_cfg["initial_capital"]),
            cost_bps=float(exec_cfg.get("cost_bps", 0.0)),
        )

        summary = compute_performance_summary(results["equity"], float(exec_cfg["initial_capital"]))

        combo_name = combo_cfg.get("name", "combo")
        out_subdir = folder_override if folder_override else combo_name
        out_dir = os.path.join(base_out_dir, out_subdir)
        ensure_out_dir(out_dir)
        save_config(out_dir, cfg)

        generate_reports(
            out_dir=out_dir,
            ic_metrics={},
            results=results,
            plots=cfg["reporting"].get("plots", []),
            write_results_core=write_results_core,
            summary=summary,
        )

        print("Backtest complete. Reports written to:", out_dir)

    else:
        signals_cfg = cfg["signals"]
        factor_name = signals_cfg["factor"]
        lookback = int(signals_cfg["lookback_minutes"])
        k_minutes = int(signals_cfg["k_minutes"])
        evaluate_on_rebalance_only = bool(signals_cfg.get("evaluate_on_rebalance_only", True))

        # Determine evaluation timestamps (rebalances or every minute)
        exec_cfg = cfg["execution"]
        rebalance_minutes = int(exec_cfg["rebalance_minutes"])
        trade_delay = int(exec_cfg.get("trade_delay_minutes", 1))

        if evaluate_on_rebalance_only:
            # Align timestamps to multiples of rebalance_minutes starting from first index minute
            minute_index = df.index
            first = minute_index[0]
            aligned_start = first + pd.Timedelta(minutes=(rebalance_minutes - (first.minute % rebalance_minutes)) % rebalance_minutes)
            eval_times = minute_index[(minute_index >= aligned_start) & (((minute_index - aligned_start).asi8 // 60_000_000_000) % rebalance_minutes == 0)]
        else:
            eval_times = df.index

        factor_fn = registry.get(factor_name)

        factor_series = compute_factors(
            df=df,
            factor_fn=factor_fn,
            lookback_minutes=lookback,
            eval_times=eval_times,
        )

        trade_mode = signals_cfg.get("trade_mode", "continuous")

        ic_metrics = {}
        if trade_mode != "threshold":
            labels = build_labels(
                df=df,
                eval_times=eval_times,
                k_minutes=k_minutes,
            )
            ic_metrics = compute_ic_metrics(
                factor=factor_series,
                label=labels,
                rolling_window=int(cfg["reporting"].get("ic_rolling_window", 50)),
            )

        targets = map_factor_to_target_notional(
            factor=factor_series,
            capital=float(exec_cfg["initial_capital"]),
            mapper=signals_cfg.get("mapper", "zscore"),
            zscore_window=int(signals_cfg.get("zscore_window", 100)),
            clip_abs=float(signals_cfg.get("clip_abs_signal", 1.0)),
            allow_short=bool(exec_cfg.get("allow_short", True)),
            long_leverage=float(exec_cfg.get("long_leverage", 1.0)),
            short_leverage=float(exec_cfg.get("short_leverage", 1.0)),
            trade_mode=trade_mode,
            entry_long_threshold=float(signals_cfg.get("entry_long_threshold", 0.9)),
            entry_short_threshold=float(signals_cfg.get("entry_short_threshold", -0.9)),
        )

        results = run_backtest(
            df=df,
            eval_times=eval_times,
            trade_delay_minutes=trade_delay,
            target_notional=targets,
            initial_capital=float(exec_cfg["initial_capital"]),
            cost_bps=float(exec_cfg.get("cost_bps", 0.0)),
            stop_loss_pct=float(exec_cfg.get("stop_loss_pct", 0.0)),
        )

        summary = compute_performance_summary(results["equity"], float(exec_cfg["initial_capital"]))

        out_subdir = folder_override if folder_override else factor_name
        out_dir = os.path.join(base_out_dir, out_subdir)
        ensure_out_dir(out_dir)
        save_config(out_dir, cfg)

        generate_reports(
            out_dir=out_dir,
            ic_metrics=ic_metrics,
            results=results,
            plots=cfg["reporting"].get("plots", []),
            write_results_core=write_results_core,
            summary=summary,
        )

        print("Backtest complete. Reports written to:", out_dir)