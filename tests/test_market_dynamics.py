from environment.config import MarketConfig
from environment.market_env import MarketEnvironment
from environment.price_model import PriceModel


def make_deterministic_config(**overrides):
    values = {
        "drift": 0.0,
        "noise_scale": 0.0,
        "fundamental_reversion": 0.0,
        "initial_price": 100.0,
        "initial_fundamental_value": 100.0,
        "initial_cash": 100000.0,
        "initial_inventory": 0,
        "max_order_size": 100,
        "seed": 42,
    }

    values.update(overrides)
    return MarketConfig(**values)


# ---------------------------------------------------------
# TEST 1: PriceModel responds to BUY imbalance
# ---------------------------------------------------------

def test_buy_pressure_increases_price():
    model = PriceModel(
        drift=0.0,
        imbalance_impact=0.05,
        noise_scale=0.0,
        fundamental_reversion=0.0,
        seed=42,
    )

    price = model.update(
    100.0,
    100.0,
    1.0,
    0.01,
    )

    assert price > 100.0

    print("PASS: BUY imbalance increases price")


# ---------------------------------------------------------
# TEST 2: PriceModel responds to SELL imbalance
# ---------------------------------------------------------

def test_sell_pressure_decreases_price():
    model = PriceModel(
        drift=0.0,
        imbalance_impact=0.05,
        noise_scale=0.0,
        fundamental_reversion=0.0,
        seed=42,
    )

    price = model.update(
    100.0,
    100.0,
    -1.0,
    0.01,
)

    assert price < 100.0

    print("PASS: SELL imbalance decreases price")


# ---------------------------------------------------------
# TEST 3: Fundamental value reversion
# ---------------------------------------------------------

def test_fundamental_reversion():
    model = PriceModel(
        drift=0.0,
        imbalance_impact=0.0,
        noise_scale=0.0,
        fundamental_reversion=0.10,
        seed=42,
    )

    starting_price = 80.0
    fundamental = 100.0

    new_price = model.update(
    starting_price,
    fundamental,
    0.0,
    0.01,
)

    assert starting_price < new_price < fundamental

    print("PASS: Price moves toward fundamental value")


# ---------------------------------------------------------
# TEST 4: BUY pressure reduces liquidity
# ---------------------------------------------------------

def test_buy_pressure_reduces_liquidity():
    config = make_deterministic_config(
        liquidity_impact=0.10,
        liquidity_recovery=0.05,
    )

    env = MarketEnvironment(config=config)
    env.reset()

    initial_liquidity = env.liquidity

    actions = {
        "momentum": "BUY",
        "value": "HOLD",
        "rl_trader": "HOLD",
    }

    _, _, _, info = env.step(actions)

    assert info["buy_volume"] > 0
    assert env.liquidity < initial_liquidity

    print("PASS: BUY pressure reduces liquidity")


# ---------------------------------------------------------
# TEST 5: Liquidity recovers when pressure disappears
# ---------------------------------------------------------

def test_liquidity_recovers():
    config = make_deterministic_config(
        liquidity_impact=0.10,
        liquidity_recovery=0.05,
    )

    env = MarketEnvironment(config=config)
    env.reset()

    buy_actions = {
        "momentum": "BUY",
        "value": "HOLD",
        "rl_trader": "HOLD",
    }

    env.step(buy_actions)

    reduced_liquidity = env.liquidity

    hold_actions = {
        "momentum": "HOLD",
        "value": "HOLD",
        "rl_trader": "HOLD",
    }

    env.step(hold_actions)

    assert env.liquidity > reduced_liquidity

    print("PASS: Liquidity recovers when pressure disappears")


# ---------------------------------------------------------
# TEST 6: Spread responds to market pressure
# ---------------------------------------------------------

def test_spread_increases_with_pressure():
    config = make_deterministic_config(
        liquidity_impact=0.10,
        liquidity_recovery=0.05,
    )

    env = MarketEnvironment(config=config)
    env.reset()

    initial_spread = env.spread

    actions = {
        "momentum": "BUY",
        "value": "HOLD",
        "rl_trader": "HOLD",
    }

    env.step(actions)

    assert env.spread > initial_spread

    print("PASS: Spread increases under strong pressure")


# ---------------------------------------------------------
# TEST 7: Volatility responds to order imbalance
# ---------------------------------------------------------

def test_volatility_increases_with_pressure():
    config = make_deterministic_config(
        volatility_decay=0.90,
        volatility_imbalance_sensitivity=0.10,
    )

    env = MarketEnvironment(config=config)
    env.reset()

    initial_volatility = env.volatility

    actions = {
        "momentum": "BUY",
        "value": "HOLD",
        "rl_trader": "HOLD",
    }

    env.step(actions)

    assert env.volatility > initial_volatility

    print("PASS: Volatility increases under order pressure")


# ---------------------------------------------------------
# TEST 8: Zero executed volume creates no artificial pressure
# ---------------------------------------------------------

def test_zero_executed_volume_creates_no_pressure():
    config = make_deterministic_config(
        liquidity_impact=0.10,
        liquidity_recovery=0.05,
        volatility_decay=1.0,
        volatility_imbalance_sensitivity=0.10,
    )

    env = MarketEnvironment(config=config)
    env.reset()

    initial_price = env.price
    initial_liquidity = env.liquidity
    initial_spread = env.spread

    # Value agent attempts to SELL despite having zero inventory.
    # ExecutionEngine should reject it.
    actions = {
        "momentum": "HOLD",
        "value": "SELL",
        "rl_trader": "HOLD",
    }

    _, _, _, info = env.step(actions)

    assert info["sell_volume"] == 0.0
    assert info["buy_volume"] == 0.0
    assert info["order_imbalance"] == 0.0

    assert env.price == initial_price
    assert env.liquidity == initial_liquidity
    assert env.spread == initial_spread

    rejected = [
        order
        for order in info["orders"]
        if order["agent_id"] == "value"
    ]

    assert len(rejected) == 1
    assert rejected[0]["status"] == "REJECTED"
    assert rejected[0]["quantity"] == 0

    print("PASS: Rejected order creates zero market pressure")


# ---------------------------------------------------------
# RUN ALL TESTS
# ---------------------------------------------------------

if __name__ == "__main__":
    print("\n===== MARKET DYNAMICS TESTS =====\n")

    test_buy_pressure_increases_price()
    test_sell_pressure_decreases_price()
    test_fundamental_reversion()
    test_buy_pressure_reduces_liquidity()
    test_liquidity_recovers()
    test_spread_increases_with_pressure()
    test_volatility_increases_with_pressure()
    test_zero_executed_volume_creates_no_pressure()

    print("\n===== ALL MARKET DYNAMICS TESTS PASSED =====")