"""End-to-End Strategic Market Simulation Pipeline Demonstration (Member 2).

Demonstrates the required Section 26 integration pipeline:
1. Reset Market Environment
2. Stackelberg Game Leader Decision (TIGHT/MEDIUM/WIDE)
3. Follower Actions & Market Environment Step
4. Opponent Model Updates (Momentum & Value)
5. Strategic Observation Generation for Member 3's RL Policy
"""

from environment import MarketConfig, MarketEnvironment
from agents import MomentumAgent, ValueAgent, MarketMaker
from game_theory import StrategicLayer, GameTheoryConfig


def main():
    print("==================================================")
    print(" MEMBER 2 — STRATEGIC MARKET SIMULATION PIPELINE ")
    print("==================================================")

    # Initialize environment and agents
    env_config = MarketConfig(episode_length=5, seed=42)
    env = MarketEnvironment(config=env_config)

    gt_config = GameTheoryConfig(seed=42)
    market_maker = MarketMaker(config=gt_config)
    momentum_agent = MomentumAgent()
    value_agent = ValueAgent()

    # Step 1: Environment Reset
    all_states = env.reset()
    market_state = env.get_market_state()

    print("\nInitial Market State:")
    print(f"  Price: {market_state['price']:.2f}, Spread: {market_state['spread']:.4f}, Volatility: {market_state['volatility']:.4f}")

    for step in range(1, 6):
        print(f"\n--- TIMESTEP {step} ---")

        # Step 2: Stackelberg Leader Decision
        leader_decision = market_maker.act(market_state)
        print(f"Leader Decision: {leader_decision.selected_action.value} (Utility: {leader_decision.leader_utility:.4f})")
        print(f"  Predicted Follower Actions: {leader_decision.predicted_follower_actions}")

        # Step 3: Followers respond to state
        mom_action = momentum_agent.act(all_states["momentum"])
        val_action = value_agent.act(all_states["value"])

        actions = {
            "momentum": mom_action,
            "value": val_action,
            "rl_trader": "HOLD",
        }

        # Advance environment with leader market control
        all_states, rewards, done, info = env.step(actions, market_control=leader_decision)

        # Step 4: Opponent Model Update
        market_maker.update_opponent("momentum", market_state, mom_action)
        market_maker.update_opponent("value", market_state, val_action)

        # Step 5: Build Strategic Observation for Member 3 RL Policy
        strat_obs = market_maker.get_strategic_observation(market_state, leader_decision)
        feature_vec = strat_obs.to_feature_vector()

        print(f"  Observed Actions -> Momentum: {mom_action}, Value: {val_action}")
        print(f"  Next Price: {info['price']:.4f}, Effective Spread: {info['spread']:.4f}")
        print(f"  Strategic Observation Feature Vector (len {len(feature_vec)}):")
        print(f"    {[round(x, 4) for x in feature_vec]}")

        # Update market_state for next step
        market_state = env.get_market_state()

        if done:
            break

    print("\n==================================================")
    print(" PIPELINE EXECUTED SUCCESSFULLY ")
    print("==================================================")


if __name__ == "__main__":
    main()
