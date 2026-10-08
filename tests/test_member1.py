from environment.config import MarketConfig
from environment.market_env import MarketEnvironment
from agents.momentum_agent import MomentumAgent
from agents.value_agent import ValueAgent


config = MarketConfig(
    episode_length=10,
    seed=42
)

env = MarketEnvironment(config)

momentum = MomentumAgent()
value = ValueAgent()

states = env.reset()

print("===== RULE-BASED MARKET SIMULATION =====")

for step in range(10):

    momentum_action = momentum.act(states["momentum"])
    value_action = value.act(states["value"])

    actions = {
        "momentum": momentum_action,
        "value": value_action,
        "rl_trader": "HOLD",
    }

    states, rewards, done, info = env.step(actions)

    print(f"\n--- Step {step + 1} ---")
    print(f"Price:       {info['price']:.4f}")
    print(f"Momentum:    {momentum_action}")
    print(f"Value:       {value_action}")
    print(f"Buy volume:  {info['buy_volume']:.2f}")
    print(f"Sell volume: {info['sell_volume']:.2f}")
    print(f"Imbalance:   {info['order_imbalance']:.4f}")
    print(f"Spread:      {info['spread']:.4f}")
    print(f"Liquidity:   {info['liquidity']:.4f}")
    print(f"Rewards:     {rewards}")

    if done:
        break

print("\n===== SIMULATION COMPLETED =====")