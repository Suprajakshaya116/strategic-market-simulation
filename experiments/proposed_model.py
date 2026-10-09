import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from environment.config import MarketConfig
from environment.market_env import MarketEnvironment
from game_theory import StrategicLayer, GameTheoryConfig
from rl.config import PPOConfig, StateEncoderConfig
from rl.ppo_agent import PPOAgent
from rl.trainer import RLTrainer


def run_proposed_model_experiment(total_timesteps: int = 2000):
    print("==================================================")
    print(" PROPOSED MODEL: STACKELBERG + OPPONENT MODELING + PPO")
    print("==================================================")

    market_config = MarketConfig(episode_length=100, seed=42)
    env = MarketEnvironment(config=market_config)

    gt_config = GameTheoryConfig(seed=42)
    strategic_layer = StrategicLayer(
        config=gt_config,
        use_stackelberg=True,
        use_opponent_model=True,
    )

    ppo_config = PPOConfig(
        lr=3e-4,
        rollout_length=256,
        batch_size=32,
        ppo_epochs=4,
        device="cpu",
        seed=42
    )

    encoder_config = StateEncoderConfig(
        initial_price=market_config.initial_price,
        initial_cash=market_config.initial_cash,
        max_order_size=market_config.max_order_size
    )

    # 11-dimensional feature vector from Member 2 StrategicObservation
    agent = PPOAgent(
        config=ppo_config,
        encoder_config=encoder_config,
        strategic_dim=11
    )

    trainer = RLTrainer(
        env=env,
        agent=agent,
        target_agent_id="rl_trader",
        enable_opponent_modeling=True,
        strategic_layer=strategic_layer
    )

    results = trainer.train(
        total_timesteps=total_timesteps,
        eval_interval=1000,
        checkpoint_interval=2000,
        checkpoint_dir="checkpoints/proposed_model"
    )

    print("\n===== PROPOSED MODEL EXPERIMENT COMPLETED =====")
    print(f"Total Timesteps: {results['total_timesteps']}")
    print(f"Episodes Completed: {results['episodes']}")
    final_eval = results["final_eval"]
    print(f"Final Mean Reward: {final_eval['mean_reward']:.2f}")
    print(f"Final Sharpe Ratio: {final_eval['sharpe_ratio']:.2f}")
    print(f"Final Win Rate: {final_eval['win_rate']*100:.1f}%")
    print(f"Final Max Drawdown: {final_eval['max_drawdown']*100:.1f}%")

    return results


if __name__ == "__main__":
    run_proposed_model_experiment(total_timesteps=2000)
