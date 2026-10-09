import math
from game_theory import (
    StackelbergGame,
    GameTheoryConfig,
    LeaderAction,
    LeaderDecision,
    calculate_leader_utility,
    calculate_follower_utility,
)


def test_follower_utility_calculation():
    config = GameTheoryConfig()
    state = {
        "price": 100.0,
        "spread": 0.02,
        "volatility": 0.01,
        "return": 0.02,
        "fundamental_value": 100.0,
    }

    u_buy_tight = calculate_follower_utility("momentum", "BUY", state, LeaderAction.TIGHT, config)
    u_buy_wide = calculate_follower_utility("momentum", "BUY", state, LeaderAction.WIDE, config)

    # TIGHT spread reduces transaction friction, increasing follower utility
    assert u_buy_tight > u_buy_wide


def test_leader_utility_calculation():
    config = GameTheoryConfig()
    state = {"liquidity": 1.0, "volatility": 0.01}

    u_tight = calculate_leader_utility(state, LeaderAction.TIGHT, 100.0, 0.0, 1.0, config)
    u_wide = calculate_leader_utility(state, LeaderAction.WIDE, 100.0, 0.0, 1.0, config)

    assert isinstance(u_tight, float)
    assert isinstance(u_wide, float)


def test_stackelberg_solver():
    config = GameTheoryConfig()
    game = StackelbergGame(config)

    state = {
        "price": 100.0,
        "return": 0.01,
        "volume": 500.0,
        "spread": 0.02,
        "liquidity": 1.0,
        "volatility": 0.01,
        "order_imbalance": 0.2,
        "fundamental_value": 105.0,
    }

    decision = game.solve(state)

    assert isinstance(decision, LeaderDecision)
    assert decision.selected_action in LeaderAction
    assert len(decision.candidate_evaluations) == 3

    # Candidate evaluations must remain accessible
    actions_evaluated = [c.leader_action for c in decision.candidate_evaluations]
    assert set(actions_evaluated) == {LeaderAction.TIGHT, LeaderAction.MEDIUM, LeaderAction.WIDE}

    # Selected action must match the maximum utility among candidate evaluations
    max_util = max(c.leader_utility for c in decision.candidate_evaluations)
    assert math.isclose(decision.leader_utility, max_util, abs_tol=1e-5)


def test_stackelberg_deterministic_behavior():
    config = GameTheoryConfig(seed=42)
    game1 = StackelbergGame(config)
    game2 = StackelbergGame(config)

    state = {
        "price": 100.0,
        "return": -0.01,
        "volume": 200.0,
        "spread": 0.03,
        "liquidity": 0.8,
        "volatility": 0.02,
        "order_imbalance": -0.4,
        "fundamental_value": 95.0,
    }

    dec1 = game1.solve(state)
    dec2 = game2.solve(state)

    assert dec1.selected_action == dec2.selected_action
    assert math.isclose(dec1.leader_utility, dec2.leader_utility, abs_tol=1e-5)
    assert dec1.predicted_follower_actions == dec2.predicted_follower_actions
