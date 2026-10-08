import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from environment.config import MarketConfig
from environment.market_env import MarketEnvironment
from rl.config import PPOConfig, StateEncoderConfig
from rl.ppo_agent import PPOAgent
from rl.trainer import RLTrainer


def run_stackelberg_marl_experiment(total_timesteps: int = 1000):
    print("==================================================")
    print("     STACKELBERG STRATEGIC MARL EXPERIMENT")
    print("==================================================")

    market_config = MarketConfig(episode_length=100, seed=42)
    env = MarketEnvironment(config=market_config)

    ppo_config = PPOConfig(
        lr=3e-4,
        rollout_length=128,
        batch_size=32,
        ppo_epochs=3,
        device="cpu",
        seed=42
    )

    encoder_config = StateEncoderConfig(
        initial_price=market_config.initial_price,
        initial_cash=market_config.initial_cash,
        max_order_size=market_config.max_order_size
    )

    # Strategic dimension incorporates Leader action + opponent probabilities + expected response
    agent = PPOAgent(
        config=ppo_config,
        encoder_config=encoder_config,
        strategic_dim=5
    )

    trainer = RLTrainer(
        env=env,
        agent=agent,
        target_agent_id="rl_trader",
        enable_opponent_modeling=True
    )

    results = trainer.train(
        total_timesteps=total_timesteps,
        eval_interval=500,
        checkpoint_interval=1000,
        checkpoint_dir="checkpoints/stackelberg_marl"
    )

    print("\n===== STACKELBERG MARL EXPERIMENT COMPLETED =====")
    return results


if __name__ == "__main__":
    run_stackelberg_marl_experiment(total_timesteps=500)
