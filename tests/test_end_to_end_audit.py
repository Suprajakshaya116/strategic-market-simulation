import math
from environment.config import MarketConfig
from environment.market_env import MarketEnvironment
from agents.momentum_agent import MomentumAgent
from agents.value_agent import ValueAgent
from agents.market_maker import MarketMaker
from game_theory import StrategicLayer, GameTheoryConfig, LeaderAction, StrategicObservation
from rl.config import PPOConfig, StateEncoderConfig
from rl.ppo_agent import PPOAgent


def test_end_to_end_full_pipeline_audit():
    """End-to-End Rigorous Integration Audit Test across Member 1, Member 2, and Member 3."""

    # 1. Initialize Market Environment (Member 1)
    env_config = MarketConfig(episode_length=10, seed=42)
    env = MarketEnvironment(config=env_config)

    # 2. Initialize Game-Theoretic Strategic Layer & MarketMaker (Member 2)
    gt_config = GameTheoryConfig(seed=42)
    market_maker = MarketMaker(config=gt_config)
    momentum_agent = MomentumAgent()
    value_agent = ValueAgent()

    # 3. Initialize PPO RL Agent (Member 3) with 11-dim Strategic Feature Dimension
    ppo_config = PPOConfig(rollout_length=32, batch_size=16, device="cpu", seed=42)
    encoder_config = StateEncoderConfig(
        initial_price=env_config.initial_price,
        initial_cash=env_config.initial_cash,
        max_order_size=env_config.max_order_size
    )
    rl_agent = PPOAgent(config=ppo_config, encoder_config=encoder_config, strategic_dim=11)

    # 4. Reset Environment
    states = env.reset()
    assert "rl_trader" in states
    assert "momentum" in states
    assert "value" in states

    total_rl_reward = 0.0

    # 5. Run Multi-Step Simulation Loop
    for step in range(5):
        market_state = env.get_market_state()

        # Step A: Leader Stackelberg Decision
        leader_decision = market_maker.act(market_state)
        assert leader_decision.selected_action in LeaderAction
        assert len(leader_decision.candidate_evaluations) == 3

        # Step B: Strategic Observation Generation for RL Policy
        strat_obs = market_maker.get_strategic_observation(market_state, leader_decision)
        assert isinstance(strat_obs, StrategicObservation)

        feature_vector = strat_obs.to_feature_vector()
        assert len(feature_vector) == 11
        assert all(isinstance(x, float) for x in feature_vector)

        # Step C: RL Agent Action Selection using Market State + Strategic Features
        action_idx, log_prob, val = rl_agent.select_action(
            state=states["rl_trader"],
            strategic_dict=strat_obs,
            deterministic=True
        )
        rl_env_action = rl_agent.action_space.to_env_action(action_idx)
        assert rl_env_action in ("BUY", "HOLD", "SELL")

        # Step D: Followers observe state and select actions
        mom_act = momentum_agent.act(states["momentum"])
        val_act = value_agent.act(states["value"])

        actions = {
            "momentum": mom_act,
            "value": val_act,
            "rl_trader": rl_env_action,
        }

        # Step E: Environment Step with Leader Market Control
        next_states, rewards, done, info = env.step(actions, market_control=leader_decision)

        # Step F: Update Opponent Model with Observed Actions
        market_maker.update_opponent("momentum", market_state, mom_act)
        market_maker.update_opponent("value", market_state, val_act)

        # Step G: Store RL Transition in Rollout Buffer
        rl_agent.store_transition(
            state=states["rl_trader"],
            action=action_idx,
            log_prob=log_prob,
            reward=rewards["rl_trader"],
            value=val,
            done=done,
            strategic_dict=strat_obs
        )

        # Assert Output Metrics & Integrations
        total_rl_reward += rewards["rl_trader"]
        assert info["market_control"] is not None
        assert info["market_control"]["leader_action"] == leader_decision.selected_action.value
        assert "price" in info
        assert "spread" in info

        states = next_states
        if done:
            break

    # 6. Verify Buffer & RL Update Step
    assert len(rl_agent.buffer.states) == 5
    print("End-to-End Full Pipeline Integration Audit Passed!")


if __name__ == "__main__":
    test_end_to_end_full_pipeline_audit()
