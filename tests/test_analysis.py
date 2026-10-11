import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pandas as pd

from analysis.market_metrics import (
    calculate_realized_volatility,
    calculate_effective_spread,
    calculate_kyles_lambda,
    calculate_amihud_illiquidity,
    calculate_variance_ratio,
    summarize_market_microstructure,
)
from analysis.performance import (
    calculate_cumulative_return,
    calculate_sharpe_ratio,
    calculate_sortino_ratio,
    calculate_max_drawdown,
    calculate_calmar_ratio,
    calculate_win_rate,
    calculate_profit_factor,
    compute_agent_performance_summary,
)
from analysis.emergence import (
    calculate_price_discovery_error,
    calculate_spread_stabilization_index,
    calculate_herding_measure,
    evaluate_stackelberg_efficiency,
)
from analysis.plots import (
    plot_market_trajectory,
    plot_drawdown_underwater,
    plot_ablation_comparison,
)


def test_market_metrics():
    prices = np.array([100.0, 102.0, 101.5, 103.0, 104.5, 104.0])
    vols = np.array([100.0, 150.0, 120.0, 180.0, 200.0])
    dp = np.diff(prices)
    oi = np.array([0.2, 0.4, -0.1, 0.5, 0.3])

    vol = calculate_realized_volatility(prices)
    assert vol > 0

    eff_spread = calculate_effective_spread(np.array([100.5, 99.5]), np.array([100.0, 100.0]))
    assert len(eff_spread) == 2

    k_lambda = calculate_kyles_lambda(dp, oi)
    assert isinstance(k_lambda, float)

    amihud = calculate_amihud_illiquidity(dp / 100.0, vols)
    assert amihud >= 0

    vr = calculate_variance_ratio(prices, lag=2)
    assert isinstance(vr, float)

    summary = summarize_market_microstructure({
        "price": prices,
        "spread": np.array([0.02, 0.025, 0.02, 0.03, 0.025, 0.02]),
        "liquidity": np.array([1.0, 0.9, 0.95, 0.85, 0.8, 0.85]),
        "volume": np.array([100, 150, 120, 180, 200, 160]),
        "order_imbalance": np.array([0.0, 0.2, 0.4, -0.1, 0.5, 0.3])
    })
    assert "realized_volatility" in summary
    assert "amihud_illiquidity" in summary
    print("PASS: test_market_metrics")


def test_performance_metrics():
    pv = np.array([100_000.0, 102_000.0, 101_000.0, 105_000.0, 103_000.0, 108_000.0])
    rets = np.diff(pv) / pv[:-1]

    cum_ret = calculate_cumulative_return(pv)
    assert abs(cum_ret - 0.08) < 1e-4

    sharpe = calculate_sharpe_ratio(rets)
    assert isinstance(sharpe, float)

    sortino = calculate_sortino_ratio(rets)
    assert isinstance(sortino, float)

    max_dd, peak_i, trough_i = calculate_max_drawdown(pv)
    assert max_dd > 0
    assert peak_i <= trough_i

    win_rate = calculate_win_rate(np.array([100.0, -50.0, 200.0, -10.0]))
    assert win_rate == 0.5

    pf = calculate_profit_factor(np.array([100.0, -50.0, 200.0, -10.0]))
    assert pf == 300.0 / 60.0

    summary = compute_agent_performance_summary(pv)
    assert "cumulative_return" in summary
    assert "max_drawdown" in summary
    print("PASS: test_performance_metrics")


def test_emergence_metrics():
    prices = np.array([100.0, 101.0, 102.0, 101.5])
    fundamentals = np.array([100.0, 100.5, 101.0, 101.5])

    err = calculate_price_discovery_error(prices, fundamentals)
    assert "price_discovery_mae" in err
    assert "price_discovery_rmse" in err

    stab = calculate_spread_stabilization_index(
        spreads_regulated=np.array([2.0, 2.1, 1.9, 2.0]),
        spreads_unconstrained=np.array([10.0, 50.0, 200.0, 1000.0])
    )
    assert stab["spread_reduction_ratio"] > 0.9

    actions = {
        "agent1": ["BUY", "BUY", "SELL"],
        "agent2": ["BUY", "BUY", "HOLD"],
        "agent3": ["BUY", "SELL", "SELL"]
    }
    herding = calculate_herding_measure(actions)
    assert isinstance(herding, float)
    print("PASS: test_emergence_metrics")


def test_plots():
    test_dir = "analysis/figures"
    os.makedirs(test_dir, exist_ok=True)

    history = {
        "step": list(range(10)),
        "price": [100.0 + i for i in range(10)],
        "fundamental_value": [100.0] * 10,
        "spread": [0.02 + 0.001 * i for i in range(10)],
        "liquidity": [1.0 - 0.01 * i for i in range(10)],
        "order_imbalance": [0.1 * (i % 3) for i in range(10)],
        "rl_trader": [100000.0 + i * 100 for i in range(10)]
    }

    out1 = plot_market_trajectory(history, save_path=f"{test_dir}/test_traj.png")
    assert os.path.exists(out1)

    out2 = plot_drawdown_underwater(np.array(history["rl_trader"]), save_path=f"{test_dir}/test_dd.png")
    assert os.path.exists(out2)

    # Clean up test images
    if os.path.exists(out1):
        os.remove(out1)
    if os.path.exists(out2):
        os.remove(out2)

    print("PASS: test_plots")


if __name__ == "__main__":
    print("\n===== RUNNING ANALYSIS SUITE UNIT TESTS =====\n")
    test_market_metrics()
    test_performance_metrics()
    test_emergence_metrics()
    test_plots()
    print("\n===== ALL ANALYSIS TESTS PASSED! =====\n")
