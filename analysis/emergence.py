import numpy as np
import pandas as pd
from typing import Dict, Any, List


def calculate_price_discovery_error(
    prices: np.ndarray,
    fundamental_values: np.ndarray
) -> Dict[str, float]:
    """Evaluate price discovery efficiency by tracking deviations from fundamental value.

    Computes:
      - Mean Absolute Mispricing (MAE)
      - Root Mean Squared Tracking Error (RMSE)
      - Maximum Fundamental Divergence
      - Autocorrelation of Mispricing (persistence of deviation)
    """
    p = np.asarray(prices, dtype=np.float64)
    f = np.asarray(fundamental_values, dtype=np.float64)
    if len(p) == 0 or len(f) == 0:
        return {}

    divergence = p - f
    rel_divergence = divergence / np.maximum(f, 1e-8)

    mae = float(np.mean(np.abs(rel_divergence)))
    rmse = float(np.sqrt(np.mean(rel_divergence ** 2)))
    max_div = float(np.max(np.abs(rel_divergence)))

    # Autocorrelation of divergence (higher = slower convergence to fundamentals)
    if len(divergence) > 2 and np.var(divergence) > 1e-12:
        autocorr = float(np.corrcoef(divergence[:-1], divergence[1:])[0, 1])
    else:
        autocorr = 0.0

    return {
        "price_discovery_mae": mae,
        "price_discovery_rmse": rmse,
        "max_fundamental_divergence": max_div,
        "mispricing_persistence": autocorr
    }


def calculate_spread_stabilization_index(
    spreads_regulated: np.ndarray,
    spreads_unconstrained: np.ndarray
) -> Dict[str, float]:
    """Evaluate how effectively the Stackelberg strategic leader dampens spread divergence.

    Compares spread trajectories between Stackelberg-regulated vs. unconstrained baselines.
    """
    sr = np.asarray(spreads_regulated, dtype=np.float64)
    su = np.asarray(spreads_unconstrained, dtype=np.float64)

    mean_sr, mean_su = np.mean(sr), np.mean(su)
    var_sr, var_su = np.var(sr), np.var(su)
    max_sr, max_su = np.max(sr), np.max(su)

    # Reduction percentages
    spread_reduction = float((mean_su - mean_sr) / np.maximum(mean_su, 1e-8))
    var_reduction = float((var_su - var_sr) / np.maximum(var_su, 1e-8))
    peak_mitigation = float((max_su - max_sr) / np.maximum(max_su, 1e-8))

    return {
        "mean_spread_regulated": float(mean_sr),
        "mean_spread_unconstrained": float(mean_su),
        "spread_reduction_ratio": spread_reduction,
        "spread_variance_reduction": var_reduction,
        "peak_spread_mitigation": peak_mitigation
    }


def calculate_herding_measure(agent_actions_history: Dict[str, List[str]]) -> float:
    """Calculate Lakonishok, Shleifer, and Vishny (LSV) herding index across trading agents.

    Measures excessive concentration of simultaneous BUY or SELL actions across participants.
    """
    if not agent_actions_history:
        return 0.0

    agents = list(agent_actions_history.keys())
    num_steps = len(agent_actions_history[agents[0]])
    if num_steps == 0:
        return 0.0

    p_buys = []
    for t in range(num_steps):
        actions_at_t = [str(agent_actions_history[a][t]).upper() for a in agents]
        active = [act for act in actions_at_t if act in ("BUY", "SELL")]
        if active:
            buys = sum(1 for act in active if act == "BUY")
            p_buys.append(buys / len(active))

    if not p_buys:
        return 0.0

    overall_p_buy = np.mean(p_buys)
    herding_index = float(np.mean(np.abs(np.array(p_buys) - overall_p_buy)))
    return herding_index


def evaluate_stackelberg_efficiency(experiments_data: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate controlled experiment runs to quantify Stackelberg strategic governance efficiency."""
    df = pd.DataFrame(experiments_data)
    if "variant" not in df.columns:
        return {}

    grouped = df.groupby("variant").agg({
        "cumulative_return": ["mean", "std"],
        "max_drawdown": ["mean", "std"],
        "mean_spread": ["mean", "std"],
        "mean_transaction_cost": ["mean", "std"],
        "win_rate": ["mean"]
    })

    summary = {}
    for var in df["variant"].unique():
        sub = df[df["variant"] == var]
        summary[var] = {
            "mean_return": float(sub["cumulative_return"].mean()),
            "mean_max_drawdown": float(sub["max_drawdown"].mean()),
            "mean_spread": float(sub["mean_spread"].mean()),
            "mean_tx_cost": float(sub["mean_transaction_cost"].mean()) if "mean_transaction_cost" in sub else 0.0,
            "mean_win_rate": float(sub["win_rate"].mean()) if "win_rate" in sub else 0.0,
        }

    return summary
