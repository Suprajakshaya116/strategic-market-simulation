import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def set_publication_style():
    """Apply clean, modern publication formatting to matplotlib plots."""
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "figure.titlesize": 12,
        "figure.dpi": 300,
        "lines.linewidth": 1.5,
        "axes.edgecolor": "#cccccc",
        "grid.color": "#ebebeb",
        "grid.linestyle": "--",
        "grid.alpha": 0.7
    })


def plot_market_trajectory(
    history: dict | pd.DataFrame,
    save_path: str = "analysis/figures/market_trajectory.png",
    title: str = "Market Dynamics Simulation Trajectory"
) -> str:
    """Generate comprehensive 4-panel market microstructure trajectory plot."""
    set_publication_style()
    df = pd.DataFrame(history) if isinstance(history, dict) else history.copy()
    steps = df["step"] if "step" in df else np.arange(len(df))

    fig, axes = plt.subplots(4, 1, figsize=(10, 10), sharex=True)
    fig.suptitle(title, fontsize=13, fontweight="bold", y=0.99)

    # 1. Price vs Fundamental Value
    ax1 = axes[0]
    ax1.plot(steps, df["price"], label="Price P_t", color="#1f77b4", linewidth=2.0)
    if "fundamental_value" in df:
        ax1.plot(steps, df["fundamental_value"], label="Fundamental V_t", color="#ff7f0e", linestyle="--", linewidth=1.5)
    ax1.set_ylabel("Price ($)")
    ax1.legend(loc="upper left")
    ax1.set_title("Price Formation & Fundamental Anchor")

    # 2. Spread & Liquidity
    ax2 = axes[1]
    color_spr = "#d62728"
    color_liq = "#2ca02c"
    ax2.plot(steps, df["spread"], label="Spread S_t", color=color_spr, linewidth=1.8)
    ax2.set_ylabel("Bid-Ask Spread", color=color_spr)
    ax2.tick_params(axis="y", labelcolor=color_spr)

    ax2_twin = ax2.twinx()
    ax2_twin.plot(steps, df["liquidity"], label="Liquidity L_t", color=color_liq, linestyle="-.", linewidth=1.8)
    ax2_twin.set_ylabel("Market Depth / Liquidity", color=color_liq)
    ax2_twin.tick_params(axis="y", labelcolor=color_liq)
    ax2.set_title("Spread Dynamics & Liquidity Depletion")

    # 3. Order Imbalance & Volume
    ax3 = axes[2]
    if "order_imbalance" in df:
        colors = np.where(df["order_imbalance"] >= 0, "#2ca02c", "#d62728")
        ax3.bar(steps, df["order_imbalance"], color=colors, alpha=0.6, width=0.8, label="Order Imbalance")
        ax3.axhline(0, color="gray", linestyle=":", linewidth=0.8)
        ax3.set_ylabel("Order Imbalance")
        ax3.set_title("Order Flow Pressure (Buyer vs Seller Dominance)")

    # 4. Agent Portfolio Values
    ax4 = axes[3]
    agent_cols = [c for c in df.columns if c.startswith("portfolio_value_") or c in ("rl_trader", "momentum", "value")]
    palette = ["#9467bd", "#8c564b", "#e377c2", "#17becf"]
    for i, col in enumerate(agent_cols):
        name = col.replace("portfolio_value_", "").upper()
        ax4.plot(steps, df[col], label=name, color=palette[i % len(palette)], linewidth=1.6)
    if not agent_cols and "rl_trader_portfolio_value" in df:
        ax4.plot(steps, df["rl_trader_portfolio_value"], label="RL Trader", color="#9467bd", linewidth=1.8)
    ax4.set_ylabel("Portfolio Value ($)")
    ax4.set_xlabel("Simulation Step (t)")
    ax4.set_title("Mark-to-Market Portfolio Trajectories")
    if agent_cols or "rl_trader_portfolio_value" in df:
        ax4.legend(loc="upper left")

    fig.subplots_adjust(top=0.94, hspace=0.35)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return save_path


def plot_drawdown_underwater(
    portfolio_values: np.ndarray,
    save_path: str = "analysis/figures/drawdown_underwater.png"
) -> str:
    """Generate underwater drawdown curve with peak and trough highlights."""
    set_publication_style()
    pv = np.asarray(portfolio_values, dtype=np.float64)
    peaks = np.maximum.accumulate(pv)
    drawdowns = (pv - peaks) / np.maximum(peaks, 1e-8) * 100.0

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(drawdowns, color="#d62728", linewidth=1.5, label="Drawdown (%)")
    ax.fill_between(range(len(drawdowns)), drawdowns, 0, color="#d62728", alpha=0.3)
    ax.axhline(0, color="black", linestyle="--", linewidth=0.8)

    max_dd_idx = int(np.argmin(drawdowns))
    max_dd_val = float(drawdowns[max_dd_idx])
    ax.scatter([max_dd_idx], [max_dd_val], color="darkred", zorder=5, label=f"Max Drawdown: {max_dd_val:.1f}%")
    ax.annotate(
        f"MDD: {max_dd_val:.1f}%\n(Step {max_dd_idx})",
        xy=(max_dd_idx, max_dd_val),
        xytext=(max_dd_idx + 2, max_dd_val + 5 if max_dd_val < -20 else max_dd_val - 5),
        arrowprops=dict(arrowstyle="->", color="darkred", lw=1.2)
    )

    ax.set_title("Portfolio Underwater Drawdown Profile", fontweight="bold")
    ax.set_xlabel("Timestep")
    ax.set_ylabel("Drawdown from Peak (%)")
    ax.legend(loc="lower left")

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return save_path


def plot_ablation_comparison(
    results_path: str = "experiments/results/controlled_experiments.json",
    save_dir: str = "analysis/figures"
) -> str:
    """Generate publication-ready 4-panel comparison bar charts across ablation variants."""
    set_publication_style()

    if not os.path.exists(results_path):
        print(f"File not found: {results_path}")
        return ""

    with open(results_path, "r") as f:
        data = json.load(f)

    df = pd.DataFrame(data)
    variants = ["rl_only", "stackelberg_only", "opponent_model_only", "proposed_full"]
    var_labels = ["RL Only\n(No GT)", "Stackelberg\nOnly", "Opponent Model\nOnly", "Proposed Model\n(Full MARL+GT)"]
    colors = ["#e74c3c", "#3498db", "#9b59b6", "#2ecc71"]

    # Filter to known variants present in df
    available_vars = [v for v in variants if v in df["variant"].values]
    sub_df = df[df["variant"].isin(available_vars)]

    grouped = sub_df.groupby("variant").agg({
        "cumulative_return": ["mean", "std"],
        "max_drawdown": ["mean", "std"],
        "mean_spread": ["mean", "std"],
        "win_rate": ["mean", "std"]
    }).reindex(available_vars)

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    fig.suptitle("Controlled Multi-Seed Ablation Study Comparison", fontsize=13, fontweight="bold", y=0.98)

    labels = [var_labels[variants.index(v)] for v in available_vars]

    # Panel 1: Cumulative Return (%)
    ax1 = axes[0, 0]
    means = grouped["cumulative_return"]["mean"] * 100
    stds = grouped["cumulative_return"]["std"] * 100
    ax1.bar(labels, means, yerr=stds, capsize=4, color=colors[:len(labels)], alpha=0.85, edgecolor="#333333")
    ax1.axhline(0, color="black", linestyle="--", linewidth=0.8)
    ax1.set_ylabel("Cumulative Return (%)")
    ax1.set_title("Return Performance")

    # Panel 2: Max Drawdown (%)
    ax2 = axes[0, 1]
    means_dd = grouped["max_drawdown"]["mean"] * 100
    stds_dd = grouped["max_drawdown"]["std"] * 100
    ax2.bar(labels, means_dd, yerr=stds_dd, capsize=4, color=colors[:len(labels)], alpha=0.85, edgecolor="#333333")
    ax2.set_ylabel("Max Drawdown (%)")
    ax2.set_title("Downside Risk / Max Drawdown")

    # Panel 3: Mean Spread (Log scale to display divergence vs regulation)
    ax3 = axes[1, 0]
    means_sp = grouped["mean_spread"]["mean"]
    stds_sp = grouped["mean_spread"]["std"]
    bars = ax3.bar(labels, means_sp, yerr=stds_sp, capsize=4, color=colors[:len(labels)], alpha=0.85, edgecolor="#333333")
    ax3.set_yscale("log")
    ax3.set_ylabel("Mean Spread (Log Scale)")
    ax3.set_title("Market Quality: Spread Regulation")

    # Add text labels on bars for spread
    for bar, val in zip(bars, means_sp):
        ax3.text(bar.get_x() + bar.get_width() / 2, val * 1.2, f"{val:.1f}", ha="center", va="bottom", fontsize=8, fontweight="bold")

    # Panel 4: Win Rate (%)
    ax4 = axes[1, 1]
    means_wr = grouped["win_rate"]["mean"] * 100
    stds_wr = grouped["win_rate"]["std"] * 100
    ax4.bar(labels, means_wr, yerr=stds_wr, capsize=4, color=colors[:len(labels)], alpha=0.85, edgecolor="#333333")
    ax4.set_ylabel("Win Rate (%)")
    ax4.set_title("Trade Execution Win Rate")

    plt.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    out_path = os.path.join(save_dir, "ablation_comparison.png")
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved ablation comparison figure to: {out_path}")
    return out_path


def generate_all_research_figures():
    """CLI runner to generate all research figures from available experiment results."""
    figures_dir = "analysis/figures"
    os.makedirs(figures_dir, exist_ok=True)

    print("Generating publication research figures in:", figures_dir)

    # 1. Ablation comparison from controlled experiments
    json_path = "experiments/results/controlled_experiments.json"
    if os.path.exists(json_path):
        plot_ablation_comparison(json_path, save_dir=figures_dir)

    # 2. Synthetic demonstration trajectory
    steps = 100
    np.random.seed(42)
    t = np.arange(steps)
    fundamental = 100.0 + np.cumsum(np.random.normal(0, 0.2, steps))
    price = fundamental + np.sin(t / 8.0) * 2.0 + np.random.normal(0, 0.1, steps)
    spread = 0.02 + np.abs(np.sin(t / 10.0)) * 0.05
    liquidity = 1.0 - spread * 5.0
    imbalance = np.random.uniform(-0.8, 0.8, steps)
    pv_rl = 100_000.0 * np.cumprod(1.0 + np.random.normal(0.0005, 0.003, steps))
    pv_mom = 100_000.0 * np.cumprod(1.0 + np.random.normal(0.0001, 0.004, steps))
    pv_val = 100_000.0 * np.cumprod(1.0 + np.random.normal(0.0002, 0.002, steps))

    demo_history = {
        "step": t,
        "price": price,
        "fundamental_value": fundamental,
        "spread": spread,
        "liquidity": liquidity,
        "order_imbalance": imbalance,
        "rl_trader": pv_rl,
        "momentum": pv_mom,
        "value": pv_val
    }

    traj_path = os.path.join(figures_dir, "market_trajectory.png")
    plot_market_trajectory(demo_history, save_path=traj_path)
    print(f"Saved market trajectory figure to: {traj_path}")

    # 3. Drawdown profile
    dd_path = os.path.join(figures_dir, "drawdown_underwater.png")
    plot_drawdown_underwater(pv_rl, save_path=dd_path)
    print(f"Saved drawdown profile figure to: {dd_path}")

    print("\nAll figures generated successfully in:", figures_dir)


if __name__ == "__main__":
    generate_all_research_figures()
