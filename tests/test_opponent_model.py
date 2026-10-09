import math
from game_theory import OpponentModel, GameTheoryConfig, calculate_expected_order_pressure


def test_unseen_state_returns_uniform_prior():
    model = OpponentModel()
    state = {
        "price": 100.0,
        "return": 0.0,
        "volatility": 0.01,
        "fundamental_value": 100.0,
        "order_imbalance": 0.0,
    }

    pred = model.predict("momentum", state)
    assert math.isclose(pred["BUY"], 1.0 / 3.0, abs_tol=1e-5)
    assert math.isclose(pred["HOLD"], 1.0 / 3.0, abs_tol=1e-5)
    assert math.isclose(pred["SELL"], 1.0 / 3.0, abs_tol=1e-5)
    assert math.isclose(sum(pred.values()), 1.0, abs_tol=1e-5)


def test_update_and_probability_shift():
    model = OpponentModel(GameTheoryConfig(smoothing_parameter=1.0))
    state = {
        "price": 100.0,
        "return": 0.02,
        "volatility": 0.01,
        "fundamental_value": 100.0,
        "order_imbalance": 0.3,
    }

    # Record multiple BUY observations for momentum
    for _ in range(7):
        model.update("momentum", state, "BUY")

    pred = model.predict("momentum", state)

    # With 7 BUY, 0 HOLD, 0 SELL and alpha=1:
    # BUY count = 8, HOLD count = 1, SELL count = 1, total = 10
    assert pred["BUY"] > pred["HOLD"]
    assert pred["BUY"] > pred["SELL"]
    assert math.isclose(sum(pred.values()), 1.0, abs_tol=1e-5)


def test_multiple_opponents_isolation():
    model = OpponentModel()
    state = {
        "price": 100.0,
        "return": 0.02,
        "volatility": 0.01,
        "fundamental_value": 100.0,
        "order_imbalance": 0.0,
    }

    model.update("momentum", state, "BUY")
    model.update("value", state, "SELL")

    mom_pred = model.predict("momentum", state)
    val_pred = model.predict("value", state)

    assert mom_pred["BUY"] > mom_pred["SELL"]
    assert val_pred["SELL"] > val_pred["BUY"]


def test_reset_functionality():
    model = OpponentModel()
    state = {"price": 100.0, "return": 0.01, "volatility": 0.01, "fundamental_value": 100.0, "order_imbalance": 0.0}

    model.update("momentum", state, "BUY")
    assert model.total_observations["momentum"] == 1

    model.reset()
    assert model.total_observations["momentum"] == 0

    pred = model.predict("momentum", state)
    assert math.isclose(pred["BUY"], 1.0 / 3.0, abs_tol=1e-5)


def test_order_pressure_calculation():
    beliefs = {
        "momentum": {"BUY": 0.8, "HOLD": 0.1, "SELL": 0.1},
        "value": {"BUY": 0.1, "HOLD": 0.1, "SELL": 0.8},
    }

    buy_p, sell_p, imb = calculate_expected_order_pressure(beliefs, expected_quantity=100.0)

    assert math.isclose(buy_p, 90.0, abs_tol=1e-5)   # (0.8 + 0.1) * 100
    assert math.isclose(sell_p, 90.0, abs_tol=1e-5)  # (0.1 + 0.8) * 100
    assert math.isclose(imb, 0.0, abs_tol=1e-5)
