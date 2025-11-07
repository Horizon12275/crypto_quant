## TODO
2025/11/06: 完善回测框架，以直接支持0-1突破信号型的因子（例如双均线，唐奇安通道）；完善风险控制模块，支持灵活的止损阈值设定；（可选）设置根据近期波动率灵活调整的仓位

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
    signal.py               # Factor -> target notional mapping (zscore/sign/threshold)
    backtester.py           # Execution simulator (rebalance, costs, equity, stop-loss)
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
  - `mapper`: `zscore` or `sign` to normalize factor to [-1, 1]
  - `zscore_window`: rolling window for z-score
  - `clip_abs_signal`: clip normalized signal to [-clip, clip]
  - `trade_mode`: `continuous` or `threshold`
    - `continuous`: position proportional to normalized signal (asymmetric leverage applied)
    - `threshold`: discrete entries only when normalized signal crosses thresholds
  - `entry_long_threshold`, `entry_short_threshold`: thresholds for `threshold` mode (e.g., 0.9, -0.9)
- `execution`:
  - `rebalance_minutes`: fixed interval rebalancing (e.g., 720)
  - `trade_delay_minutes`: trade at t + delay minutes (e.g., 1 → next open)
  - `initial_capital`: starting capital in dollars
  - `allow_short`: whether short exposure is allowed
  - `long_leverage`, `short_leverage`: leverage multipliers
  - `cost_bps`: combined trading cost in basis points (fees + slippage)
  - `stop_loss_pct`: per-trade stop-loss threshold; exit at next minute open when breached
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

### Trade modes
- `continuous` (original):
  - Normalize factor to `s` in [-1, 1], apply asymmetric leverage, target notional = `capital * s_levered`.
- `threshold` (CTA-style):
  - Normalize factor to `s` in [-1, 1] via mapper.
  - At evaluation (rebalance) times only:
    - If `s >= entry_long_threshold`: open/hold long at `long_leverage * capital` notional.
    - If `s <= entry_short_threshold`: open/hold short at `short_leverage * capital` notional.
    - Otherwise: no new instruction (hold existing position).

### Stop-loss
- When a position is opened (or flips side), the entry price is recorded.
- Each minute before applying scheduled trades, if the adverse return from entry reaches `-stop_loss_pct`, the position is closed immediately at that minute’s open (costs applied), and entry is reset.

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
2. Adjust `config/backtest.yaml` (dates, factor name, lookback, k, trade_mode, thresholds, costs, stop-loss).
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