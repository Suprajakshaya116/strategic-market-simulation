import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from environment.config import MarketConfig
from environment.market_env import MarketEnvironment
from rl.config import MARLConfig, PPOConfig, StateEncoderConfig
from rl.marl import MAPPOAgent


def run_marl_experiment(total_timesteps: int = 2000):
    print("==================================================")
    print("     MULTI-AGENT MAPPO CTDE TRADING EXPERIMENT")
    print("==================================================")

    market_config = MarketConfig(episode_length=100, seed=42)
    env = MarketEnvironment(config=market_config)

    ppo_config = PPOConfig(
        lr=3e-4,
        rollout_length=256,
        batch_size=32,
        ppo_epochs=4,
        device="cpu",
        seed=42
    )

    marl_config = MARLConfig(
        agent_ids=env.agent_ids,
        target_agent_id="rl_trader",
        use_ctde=True,
        ppo_config=ppo_config
    )

    encoder_config = StateEncoderConfig(
        initial_price=market_config.initial_price,
        initial_cash=market_config.initial_cash,
        max_order_size=market_config.max_order_size
    )

    mppo_agent = MAPPOAgent(marl_config=marl_config, encoder_config=encoder_config)

    states = env.reset()
    ep_rewards = {aid: 0.0 for aid in env.agent_ids}
    step = 0

    while step < total_timesteps:
        actions, log_probs, global_val = mppo_agent.select_actions(states, deterministic=False)

        env_actions = {
            aid: mppo_agent.action_space.to_env_action(actions[aid])
            for aid in env.agent_ids
        }

        next_states, rewards, done, info = env.step(env_actions)

        mppo_agent.store_transitions(
            states=states,
            actions=actions,
            log_probs=log_probs,
            rewards=rewards,
            global_val=global_val,
            done=done
        )

        for aid in env.agent_ids:
            ep_rewards[aid] += rewards[aid]

        step += 1

        if len(mppo_agent.buffers[env.agent_ids[0]].states) >= ppo_config.rollout_length:
            update_metrics = mppo_agent.update(next_states, done)
            print(f"[MAPPO Step {step}/{total_timesteps}] Policy Loss: {update_metrics.get('policy_loss', 0.0):.4f} | Value Loss: {update_metrics.get('value_loss', 0.0):.4f}")

        if done:
            states = env.reset()
            ep_rewards = {aid: 0.0 for aid in env.agent_ids}
        else:
            states = next_states

    print("\n===== MAPPO EXPERIMENT COMPLETED =====")


if __name__ == "__main__":
    run_marl_experiment(total_timesteps=1000)
