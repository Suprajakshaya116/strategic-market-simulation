import numpy as np
from typing import Dict, Any, Tuple


def calculate_cumulative_return(portfolio_values: np.ndarray) -> float:
    """Calculate cumulative return: (V_final - V_initial) / V_initial."""
    pv = np.asarray(portfolio_values, dtype=np.float64)
    if len(pv) < 2 or pv[0] <= 0:
        return 0.0
    return float((pv[-1] - pv[0]) / pv[0])


def calculate_sharpe_ratio(
    returns: np.ndarray,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252
) -> float:
    """Calculate annualized Sharpe Ratio: (mean(R) - Rf) / std(R) * sqrt(periods)."""
    rets = np.asarray(returns, dtype=np.float64)
    if len(rets) < 2:
        return 0.0
    excess = rets - risk_free_rate
    std = np.std(excess)
    if std < 1e-12:
        return 0.0
    return float((np.mean(excess) / std) * np.sqrt(periods_per_year))


def calculate_sortino_ratio(
    returns: np.ndarray,
    target_return: float = 0.0,
    periods_per_year: int = 252
) -> float:
    """Calculate Sortino Ratio using downside semi-deviation."""
    rets = np.asarray(returns, dtype=np.float64)
    if len(rets) < 2:
        return 0.0
    excess = rets - target_return
    downside = np.minimum(excess, 0.0)
    downside_std = np.sqrt(np.mean(downside ** 2))
    if downside_std < 1e-12:
        return 0.0
    return float((np.mean(excess) / downside_std) * np.sqrt(periods_per_year))


def calculate_max_drawdown(portfolio_values: np.ndarray) -> Tuple[float, int, int]:
    """Calculate Maximum Drawdown (MDD) percentage, peak index, and trough index."""
    pv = np.asarray(portfolio_values, dtype=np.float64)
    if len(pv) < 2:
        return 0.0, 0, 0

    peaks = np.maximum.accumulate(pv)
    drawdowns = (peaks - pv) / np.maximum(peaks, 1e-8)
    max_idx = int(np.argmax(drawdowns))
    max_dd = float(drawdowns[max_idx])

    if max_dd == 0:
        return 0.0, 0, 0

    peak_idx = int(np.argmax(pv[:max_idx + 1]))
    return max_dd, peak_idx, max_idx


def calculate_calmar_ratio(annualized_return: float, max_drawdown: float) -> float:
    """Calculate Calmar Ratio: Annualized Return / Maximum Drawdown."""
    if max_drawdown <= 1e-8:
        return 0.0
    return float(annualized_return / max_drawdown)


def calculate_win_rate(trade_pnls: np.ndarray) -> float:
    """Calculate percentage of trades with positive PnL."""
    pnls = np.asarray(trade_pnls, dtype=np.float64)
    if len(pnls) == 0:
        return 0.0
    wins = np.sum(pnls > 0)
    return float(wins / len(pnls))


def calculate_profit_factor(trade_pnls: np.ndarray) -> float:
    """Calculate Profit Factor: gross profits / gross losses."""
    pnls = np.asarray(trade_pnls, dtype=np.float64)
    if len(pnls) == 0:
        return 0.0
    gains = np.sum(pnls[pnls > 0])
    losses = np.abs(np.sum(pnls[pnls < 0]))
    if losses < 1e-12:
        return float(gains) if gains > 0 else 1.0
    return float(gains / losses)


def compute_agent_performance_summary(
    portfolio_values: np.ndarray,
    costs: Dict[str, float] | None = None
) -> Dict[str, Any]:
    """Compute comprehensive trading performance summary for an agent trajectory."""
    pv = np.asarray(portfolio_values, dtype=np.float64)
    if len(pv) < 2:
        return {}

    returns = np.diff(pv) / np.maximum(pv[:-1], 1e-8)
    cum_ret = calculate_cumulative_return(pv)
    ann_ret = float(np.mean(returns) * 252)
    sharpe = calculate_sharpe_ratio(returns)
    sortino = calculate_sortino_ratio(returns)
    max_dd, peak_i, trough_i = calculate_max_drawdown(pv)
    calmar = calculate_calmar_ratio(ann_ret, max_dd)

    summary = {
        "initial_value": float(pv[0]),
        "final_value": float(pv[-1]),
        "cumulative_return": cum_ret,
        "annualized_return": ann_ret,
        "sharpe_ratio": sharpe,
        "sortino_ratio": sortino,
        "max_drawdown": max_dd,
        "max_drawdown_peak_step": peak_i,
        "max_drawdown_trough_step": trough_i,
        "calmar_ratio": calmar,
        "daily_volatility": float(np.std(returns)),
        "annualized_volatility": float(np.std(returns) * np.sqrt(252)),
    }

    if costs:
        summary["transaction_cost"] = float(costs.get("transaction_cost", 0.0))
        summary["market_impact_cost"] = float(costs.get("market_impact_cost", 0.0))
        summary["total_trading_cost"] = summary["transaction_cost"] + summary["market_impact_cost"]

    return summary
