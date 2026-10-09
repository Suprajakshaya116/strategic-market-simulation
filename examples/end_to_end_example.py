"""End-to-End Strategic Market Simulation Pipeline Demonstration (Members 1, 2, and 3).

Demonstrates the complete integrated research pipeline:
1. Reset Market Environment (Member 1)
2. Stackelberg Game Leader Decision (TIGHT/MEDIUM/WIDE) (Member 2)
3. Opponent Model Beliefs & Strategic Observation Generation (Member 2)
4. PPO RL Agent Action Selection with 21-dim state space (Member 3)
5. Execution Engine clearing orders at bid/ask with market control (Member 1)
6. Opponent Model updates on observed actions (Member 2)
7. Transition storage and PPO policy update (Member 3)
"""

from environment import MarketConfig, MarketEnvironment
from agents import MomentumAgent, ValueAgent, MarketMaker
from game_theory import GameTheoryConfig
from rl.config import PPOConfig, StateEncoderConfig
from rl.ppo_agent import PPOAgent


def main():
    print("================================================================")
    print(" COMPLETE STRATEGIC TRADING MARL RESEARCH PIPELINE (MEMBERS 1, 2, 3) ")
    print("================================================================")

    # 1. Initialize environment and baseline agents (Member 1)
    env_config = MarketConfig(episode_length=5, seed=42)
    env = MarketEnvironment(config=env_config)
    momentum_agent = MomentumAgent()
    value_agent = ValueAgent()

    # 2. Initialize Game-Theoretic Strategic Layer (Member 2)
    gt_config = GameTheoryConfig(seed=42)
    market_maker = MarketMaker(config=gt_config)

    # 3. Initialize Reinforcement Learning Subsystem (Member 3)
    ppo_config = PPOConfig(rollout_length=4, batch_size=4, ppo_epochs=2, lr=1e-3, device="cpu", seed=42)
    encoder_config = StateEncoderConfig(
        initial_price=env_config.initial_price,
        initial_cash=env_config.initial_cash,
        max_order_size=env_config.max_order_size
    )
    rl_agent = PPOAgent(config=ppo_config, encoder_config=encoder_config, strategic_dim=11)

    # Reset environment
    all_states = env.reset(seed=42)
    market_state = env.get_market_state()

    print("\nInitial Market State:")
    print(f"  Price: {market_state['price']:.2f}, Spread: {market_state['spread']:.4f}, Volatility: {market_state['volatility']:.4f}")

    for step in range(1, 6):
        print(f"\n--- TIMESTEP {step} ---")

        # Step A: Stackelberg Leader Decision (Member 2)
        leader_decision = market_maker.act(market_state)
        print(f"Leader Decision: {leader_decision.selected_action.value} (Utility: {leader_decision.leader_utility:.4f})")
        print(f"  Predicted Follower Actions: {leader_decision.predicted_follower_actions}")

        # Step B: Build Strategic Observation for Member 3's RL Policy (Member 2)
        strat_obs = market_maker.get_strategic_observation(market_state, leader_decision)
        feature_vec = strat_obs.to_feature_vector()
        print(f"  Strategic Feature Vector (dim {len(feature_vec)}): {[round(x, 3) for x in feature_vec[:5]]}...")

        # Step C: RL Agent selects action using market state + strategic features (Member 3)
        action_idx, log_prob, val = rl_agent.select_action(
            state=all_states["rl_trader"],
            strategic_dict=strat_obs,
            deterministic=False
        )
        rl_env_action = rl_agent.action_space.to_env_action(action_idx)
        print(f"  RL Agent Intent: {rl_env_action} (Action Index: {action_idx})")

        # Step D: Rule-based followers select actions (Member 1)
        mom_action = momentum_agent.act(all_states["momentum"])
        val_action = value_agent.act(all_states["value"])

        actions = {
            "momentum": mom_action,
            "value": val_action,
            "rl_trader": rl_env_action,
        }

        # Step E: Advance Market Environment with Leader Market Control (Member 1)
        next_states, rewards, done, info = env.step(actions, market_control=leader_decision)

        # Step F: Update Opponent Model with actual observed actions (Member 2)
        market_maker.update_opponent("momentum", market_state, mom_action)
        market_maker.update_opponent("value", market_state, val_action)

        # Step G: Store transition in RL Rollout Buffer (Member 3)
        rl_agent.store_transition(
            state=all_states["rl_trader"],
            action=action_idx,
            log_prob=log_prob,
            reward=rewards["rl_trader"],
            value=val,
            done=done,
            strategic_dict=strat_obs
        )

        executed_summary = [f"{o.get('agent_id')}:{o.get('action')}:{o.get('status')}" for o in info.get("orders", [])]
        print(f"  Executed Fills: {executed_summary}")
        print(f"  Next Price: {info['price']:.4f}, Effective Spread: {info['spread']:.4f}")
        print(f"  RL Trader Reward: {rewards['rl_trader']:.4f}, Portfolio Value: {info['portfolio_values']['rl_trader']:.2f}")

        # Update market_state for next step
        market_state = env.get_market_state()
        all_states = next_states

        if done:
            break

    # Step H: Demonstrate PPO optimization update (Member 3)
    if len(rl_agent.buffer.states) >= rl_agent.config.rollout_length:
        update_metrics = rl_agent.update(
            last_state=all_states["rl_trader"],
            last_done=done,
            last_strategic_dict=strat_obs
        )
        print("\n--- PPO OPTIMIZATION UPDATE ---")
        print(f"  Total Loss: {update_metrics['total_loss']:.4f}")
        print(f"  Policy Loss: {update_metrics['policy_loss']:.4f}")
        print(f"  Value Loss: {update_metrics['value_loss']:.4f}")

    print("\n================================================================")
    print(" COMPLETE RESEARCH PIPELINE EXECUTED SUCCESSFULLY ")
    print("================================================================")


if __name__ == "__main__":
    main()

