import math
from game_theory import (
    LeaderAction,
    CandidateEvaluation,
    LeaderDecision,
    StrategicObservation,
    GameTheoryConfig,
)


def test_leader_action_enum():
    assert LeaderAction.TIGHT.value == "TIGHT"
    assert LeaderAction.MEDIUM.value == "MEDIUM"
    assert LeaderAction.WIDE.value == "WIDE"
    assert LeaderAction.from_str("tight") == LeaderAction.TIGHT
    assert LeaderAction.from_str("WIDE") == LeaderAction.WIDE

    raised = False
    try:
        LeaderAction.from_str("INVALID_ACTION")
    except ValueError:
        raised = True
    assert raised, "Expected ValueError for invalid LeaderAction"


def test_candidate_evaluation_creation():
    cand = CandidateEvaluation(
        leader_action=LeaderAction.TIGHT,
        predicted_follower_actions={"momentum": "BUY", "value": "HOLD"},
        predicted_follower_utilities={"momentum": 0.05, "value": 0.0},
        expected_buy_pressure=100.0,
        expected_sell_pressure=0.0,
        expected_order_imbalance=1.0,
        leader_utility=1.2,
    )
    assert cand.leader_action == LeaderAction.TIGHT
    assert cand.predicted_follower_actions["momentum"] == "BUY"
    assert cand.leader_utility == 1.2


def test_strategic_observation_vector_and_validation():
    obs = StrategicObservation(
        leader_action="TIGHT",
        leader_utility=1.5,
        momentum_buy_probability=0.7,
        momentum_hold_probability=0.2,
        momentum_sell_probability=0.1,
        value_buy_probability=0.1,
        value_hold_probability=0.8,
        value_sell_probability=0.1,
        expected_buy_pressure=80.0,
        expected_sell_pressure=20.0,
        expected_order_imbalance=0.6,
    )

    vec = obs.to_feature_vector()
    assert len(vec) == 11
    assert vec[0] == 0.0  # TIGHT
    assert vec[1] == 1.5  # leader_utility
    assert math.isclose(vec[2], 0.7, abs_tol=1e-5)
    assert math.isclose(vec[8], 80.0, abs_tol=1e-5)
    assert math.isclose(vec[10], 0.6, abs_tol=1e-5)

    # Test invalid probability sum validation
    invalid_raised = False
    try:
        StrategicObservation(
            leader_action="MEDIUM",
            leader_utility=0.0,
            momentum_buy_probability=0.9,  # Sums to 1.2
            momentum_hold_probability=0.2,
            momentum_sell_probability=0.1,
            value_buy_probability=0.333,
            value_hold_probability=0.333,
            value_sell_probability=0.334,
            expected_buy_pressure=0.0,
            expected_sell_pressure=0.0,
            expected_order_imbalance=0.0,
        )
    except ValueError:
        invalid_raised = True
    assert invalid_raised, "Expected ValueError for invalid probability sum"


def test_game_theory_config_validation():
    config = GameTheoryConfig()
    config.validate()

    raised_liquidity = False
    try:
        invalid_cfg = GameTheoryConfig(leader_liquidity_weight=-1.0)
        invalid_cfg.validate()
    except ValueError:
        raised_liquidity = True
    assert raised_liquidity, "Expected ValueError for negative leader_liquidity_weight"

    raised_order_size = False
    try:
        invalid_cfg = GameTheoryConfig(expected_order_size=-10.0)
        invalid_cfg.validate()
    except ValueError:
        raised_order_size = True
    assert raised_order_size, "Expected ValueError for negative expected_order_size"
