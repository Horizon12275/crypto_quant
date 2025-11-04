## Crypto Multi-Factor CTA Backtest (Single-Asset)

This repository contains a modular, config-driven backtesting framework for a single-asset, single-factor CTA strategy using 1-minute OHLCV data (OKX via CCXT). It is designed so the same factor and signal interfaces can be reused in live trading later.

### Directory structure
```
crypto_quant/
  crawler.py                # CCXT data crawler (OKX example)
  okx_BTCUSDT_1min_data.csv # Consolidated 1m OHLCV (UTC index) for quick start
  config/
    backtest.yaml           # Backtest configuration
  engine/
    __init__.py
    data_loader.py          # CSV loader, UTC index, forward-fill minute gaps
    factor_engine.py        # Rolling factor computation (numpy window)
    labeler.py              # Open-to-open forward return labels
    signal.py               # Factor -> target notional mapping (zscore/sign)
    backtester.py           # Execution simulator (rebalance, costs, equity)
    metrics.py              # Rolling IC (Pearson/Spearman), IC cumsum
    report.py               # PNG plots + CSV exports per factor
  factors/
    registry.py             # Factor registry (name -> callable)
    example_factors.py      # Example numpy factor: example_momo
  main.py                   # CLI runner to execute a backtest
  requirements.txt          # Python dependencies
  README.md                 # This file
```

### Data expectations
- CSV should include either `open_time` (ms since epoch) or `Datetime` (parseable to UTC), and columns: `open, high, low, close, volume`.
- Loader standardizes to UTC, sorts, de-duplicates, and (optionally) forward-fills minute gaps:
  - OHLC forward-filled
  - Inserted rows get `volume=0`

### Config (`config/backtest.yaml`)
- `data`:
  - `source_csv`: path to 1-minute data
  - `timezone`: canonical processing tz (use `UTC`)
  - `start`, `end`: backtest window (UTC timestamps)
  - `forward_fill`: whether to forward-fill missing minutes
- `signals`:
  - `factor`: factor name in registry (e.g., `example_momo`)
  - `lookback_minutes`: rolling window size for factor inputs
  - `k_minutes`: label horizon; return from open[t+1] to open[t+1+k]
  - `evaluate_on_rebalance_only`: compute factor only on rebalance timestamps
  - `mapper`: `zscore` or `sign` or `percentile`
  - `zscore_window`: rolling window for z-score
  - `clip_abs_signal`: clip final signal to [-clip, clip]
- `execution`:
  - `rebalance_minutes`: fixed interval rebalancing (e.g., 720)
  - `trade_delay_minutes`: trade at t + delay minutes (e.g., 1 → next open)
  - `initial_capital`: starting capital in dollars
  - `allow_short`: whether short exposure is allowed
  - `long_leverage`, `short_leverage`: leverage multipliers for signals >=0 and <0
  - `cost_bps`: combined trading cost in basis points (fees + slippage)
- `reporting`:
  - `out_dir`: base output directory (plots and CSVs)
  - `ic_rolling_window`: rolling window for IC
  - `plots`: choose from `rolling_ic`, `rolling_ic_cumsum`, `pnl`, `equity`

### Factor API
- Factors are pure numpy functions registered by name.
- Signature: a function that accepts a numpy array of shape `(lookback, 5+)` with columns `[open, high, low, close, volume, ...]` and returns a single `float`.

```python
# factors/example_factors.py
import numpy as np

def example_momo(window_ohlcv: np.ndarray) -> float:
    close = window_ohlcv[:, 3]
    mean_close = np.mean(close)
    return float((close[-1] / mean_close) - 1.0) if mean_close != 0 else 0.0
```

- Register factors in `factors/registry.py` (built-ins are auto-registered via `register_builtin_factors()`).

### Labeling
- For factor value computed at time t, the label is the simple return from `open[t+1]` to `open[t+1+k]` where `k` is `k_minutes`.

### Signal mapping and sizing
- Map factor to a target dollar notional using either:
  - `zscore`: rolling z-score of the factor (window = `zscore_window`), clipped to `[-clip_abs_signal, clip_abs_signal]`
  - `sign`: `-1, 0, +1` clipped similarly
- Apply asymmetric leverage:
  - `long_leverage` scales signals >= 0; `short_leverage` scales signals < 0
- If `allow_short=false`, negative signals are clipped to 0.

### Execution model
- Rebalance at fixed intervals (e.g., every 720 minutes).
- Compute signals at the evaluation time t; place orders at `t + trade_delay_minutes` at the open price.
- Apply a combined `cost_bps` on traded notional each rebalance.
- Track `equity`, `returns`, `pnl`, `position_units`, `position_notional`, and `trades`.

### Metrics and reporting
- Rolling IC (Pearson & Spearman) and cumulative IC (cumsum of Pearson IC) saved to CSV and PNG.
- PnL (cumulative) and net value (equity / initial capital) plots.
- Outputs are written to `reports/{factor_name}/` and include:
  - `rolling_ic.png`, `rolling_ic_cumsum.png`, `pnl.png`, `net_value.png`
  - `ic_metrics.csv`, `results_core.csv`

### Running a backtest
1. Install dependencies:
```bash
pip install -r requirements.txt
```
2. Adjust `config/backtest.yaml` if needed (dates, factor name, lookback, k, costs, etc.).
3. Run:
```bash
python main.py
```
4. Inspect outputs under `reports/{factor_name}/`.

### Notes on live compatibility
- Factor, labeling, and signal interfaces are identical for live.
- Replace the historical loader with a streaming market data adapter, and swap the backtester’s execution with a broker adapter (e.g., OKX via CCXT). The rebalance scheduling and risk constraints carry over.

### Extending
- Add new factors: implement numpy function in `factors/your_factor.py`, register in `factors/registry.py`, and reference by name in `config/backtest.yaml`.
- Add signal mappers or constraints in `engine/signal.py`.
- Add risk overlays (max position change, cooldowns) or alternative execution models in `engine/backtester.py`. 