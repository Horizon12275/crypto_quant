import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
from typing import Dict, List


def _save_plot(fig, out_dir: str, name: str) -> None:
    path = os.path.join(out_dir, f"{name}.png")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def generate_reports(
    out_dir: str,
    ic_metrics: Dict[str, pd.Series],
    results: Dict[str, pd.Series],
    plots: List[str],
    write_results_core: bool = False,
) -> None:
    os.makedirs(out_dir, exist_ok=True)

    # Save core series as CSV for convenience
    ic_df = pd.concat(ic_metrics.values(), axis=1)
    ic_df.to_csv(os.path.join(out_dir, "ic_metrics.csv"))

    if write_results_core:
        pd.DataFrame({
            "equity": results["equity"],
            "returns": results["returns"],
            "pnl": results["pnl"],
        }).to_csv(os.path.join(out_dir, "results_core.csv"))

    # rolling_ic plot
    if "rolling_ic" in plots and "pearson_ic" in ic_metrics:
        fig, ax = plt.subplots(figsize=(10, 4))
        ic_metrics["pearson_ic"].plot(ax=ax, color="tab:blue", label="Pearson IC")
        if "spearman_ic" in ic_metrics:
            ic_metrics["spearman_ic"].plot(ax=ax, color="tab:orange", alpha=0.6, label="Spearman IC")
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_title("Rolling IC")
        ax.legend()
        _save_plot(fig, out_dir, "rolling_ic")

    # cumulative rolling IC cumsum plot
    if "rolling_ic_cumsum" in plots and "ic_cumsum" in ic_metrics:
        fig, ax = plt.subplots(figsize=(10, 4))
        ic_metrics["ic_cumsum"].plot(ax=ax, color="tab:green", label="IC Cumulative Sum")
        ax.axhline(0, color="black", linewidth=0.8)
        # Annotate average IC (Pearson) if available
        if "pearson_ic" in ic_metrics:
            mean_ic = float(ic_metrics["pearson_ic"].dropna().mean())
            ax.text(
                0.01,
                0.95,
                f"Avg IC: {mean_ic:.4f}",
                transform=ax.transAxes,
                fontsize=10,
                verticalalignment="top",
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.6),
            )
        ax.set_title("Cumulative Rolling IC (cumsum)")
        ax.legend()
        _save_plot(fig, out_dir, "rolling_ic_cumsum")

    # PnL plot (cumulative pnl)
    if "pnl" in plots and "pnl" in results:
        fig, ax = plt.subplots(figsize=(10, 4))
        results["pnl"].cumsum().plot(ax=ax, color="tab:red", label="Cumulative PnL")
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_title("Cumulative PnL")
        ax.legend()
        _save_plot(fig, out_dir, "pnl")

    # Net value (equity normalized)
    if "equity" in plots and "equity" in results:
        fig, ax = plt.subplots(figsize=(10, 4))
        init_cap = float(results.get("initial_capital", results["equity"].iloc[0]))
        (results["equity"] / init_cap).rename("net_value").plot(ax=ax, color="tab:purple")
        ax.set_title("Net Value (Equity / Initial Capital)")
        _save_plot(fig, out_dir, "net_value") 

    # Turnover time series plot
    if "turnover" in plots and "trades" in results and isinstance(results["trades"], pd.DataFrame) and not results["trades"].empty:
        trades_df = results["trades"]
        equity = results.get("equity")
        if equity is not None:
            # Compute turnover at trade timestamps: traded_notional / equity
            eq_at_trades = equity.reindex(trades_df.index).astype(float)
            turnover = (trades_df["traded_notional"].astype(float) / eq_at_trades).rename("turnover")
            turnover = turnover.replace([pd.NA, pd.NaT], pd.NA).dropna()

            if not turnover.empty:
                fig, ax = plt.subplots(figsize=(10, 4))
                turnover.plot(ax=ax, color="tab:brown", label="Turnover")
                mean_turnover = float(turnover.mean())
                ax.axhline(mean_turnover, color="tab:gray", linestyle="--", linewidth=1.0, label=f"Mean: {mean_turnover:.4f}")
                ax.set_title("Turnover per Rebalance")
                ax.set_ylabel("Turnover (fraction of equity)")
                ax.legend()
                # Add text box with mean turnover
                ax.text(
                    0.01,
                    0.95,
                    f"Avg Turnover: {mean_turnover:.4f}",
                    transform=ax.transAxes,
                    fontsize=10,
                    verticalalignment="top",
                    bbox=dict(boxstyle="round", facecolor="white", alpha=0.6),
                )
                _save_plot(fig, out_dir, "turnover")