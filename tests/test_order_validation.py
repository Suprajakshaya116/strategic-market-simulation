from environment.market_env import MarketEnvironment


def test_invalid_sell_is_rejected():
    env = MarketEnvironment()
    env.reset()

    actions = {
        "momentum": "SELL",
        "value": "HOLD",
        "rl_trader": "HOLD",
    }

    _, _, _, info = env.step(actions)

    order = next(
        item for item in info["orders"]
        if item["agent_id"] == "momentum"
    )

    assert order["action"] == "SELL"
    assert order["status"] == "REJECTED"
    assert order["quantity"] == 0
    assert order["transaction_cost"] == 0.0
    assert order["market_impact_cost"] == 0.0

    assert "Insufficient inventory" in order["reason"]

    assert info["sell_volume"] == 0.0

    print("PASS: Invalid SELL was rejected correctly.")
    print("Order:", order)
    print("Sell volume:", info["sell_volume"])


def test_large_buy_is_rejected_by_order_validation():
    env = MarketEnvironment()
    env.reset()

    actions = {
        "momentum": ("BUY", 1_000_000),
        "value": "HOLD",
        "rl_trader": "HOLD",
    }

    try:
        env.step(actions)
    except ValueError as exc:
        assert "Invalid quantity" in str(exc)
        print("PASS: Oversized BUY was rejected by OrderManager.")
        print("Error:", exc)
    else:
        raise AssertionError(
            "Oversized BUY should have been rejected by OrderManager."
        )


if __name__ == "__main__":
    print("===== TESTING INVALID SELL =====")
    test_invalid_sell_is_rejected()

    print("\n===== TESTING LARGE BUY =====")
    test_large_buy_is_rejected_by_order_validation()

    print("\n===== ORDER VALIDATION TESTS PASSED =====")