from environment.config import MarketConfig
from environment.market_env import MarketEnvironment


config = MarketConfig(
    episode_length=3,
    seed=42
)

env = MarketEnvironment(config)

states = env.reset()

print("===== INITIAL PORTFOLIOS =====")

for agent_id, portfolio in env.portfolios.items():
    print(
        agent_id,
        "| Cash:", portfolio.cash,
        "| Inventory:", portfolio.inventory,
        "| Value:", portfolio.portfolio_value
    )


actions = {
    "momentum": ("BUY", 100),
    "value": "HOLD",
    "rl_trader": "HOLD",
}

states, rewards, done, info = env.step(actions)

print("\n===== AFTER BUY =====")

for agent_id, portfolio in env.portfolios.items():
    print(
        agent_id,
        "| Cash:", portfolio.cash,
        "| Inventory:", portfolio.inventory,
        "| Value:", portfolio.portfolio_value
    )

print("\nTransaction costs:")
print(info["transaction_costs"])

print("\nMarket impact costs:")
print(info["market_impact_costs"])

print("\nRewards:")
print(rewards)

actions = {
    "momentum": ("SELL", 100),
    "value": "HOLD",
    "rl_trader": "HOLD",
}

states, rewards, done, info = env.step(actions)

print("\n===== AFTER SELL =====")

for agent_id, portfolio in env.portfolios.items():
    print(
        agent_id,
        "| Cash:", portfolio.cash,
        "| Inventory:", portfolio.inventory,
        "| Value:", portfolio.portfolio_value
    )

print("\nTransaction costs:")
print(info["transaction_costs"])

print("\nMarket impact costs:")
print(info["market_impact_costs"])

print("\nRewards:")
print(rewards)