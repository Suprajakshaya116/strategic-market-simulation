"""Controlled Ablation and Baseline Experiment Runner.

Executes fair, matched, multi-seed controlled experiments comparing:
1. RL-Only Baseline (no game theory, no opponent model, strategic_dim=0)
2. Stackelberg-Only Variant (Stackelberg leader without empirical opponent modeling)
3. Opponent-Model-Only Variant (Opponent modeling enabled without Stackelberg leader optimization)
4. Proposed Full Model (Stackelberg Game + Opponent Modeling + PPO RL)

Saves machine-readable results to JSON and CSV.
"""

import os
import json
import csv
import argparse
import numpy as np
import pandas as pd
from typing import Dict, Any, List

from environment.config import MarketConfig
from environment.market_env import MarketEnvironment
from game_theory import StrategicLayer, GameTheoryConfig
from rl.config import PPOConfig, StateEncoderConfig
from rl.ppo_agent import PPOAgent
from rl.trainer import RLTrainer


VARIANTS = [
    "rl_only",
    "stackelberg_only",
    "opponent_model_only",
    "proposed_full",
]


def run_variant(
    variant: str,
    seed: int,
    total_timesteps: int = 1000,
    eval_episodes: int = 5,
    episode_length: int = 50,
) -> Dict[str, Any]:
    """Run single experimental variant with exact controlled environment configuration."""
    print(f"\n---> Running Variant: {variant} | Seed: {seed} | Steps: {total_timesteps}")

    market_config = MarketConfig(episode_length=episode_length, seed=seed)
    env = MarketEnvironment(config=market_config)

    gt_config = GameTheoryConfig(seed=seed)

    if variant == "rl_only":
        strategic_layer = None
        enable_opp = False
        strat_dim = 0
    elif variant == "stackelberg_only":
        strategic_layer = StrategicLayer(
            config=gt_config,
            use_stackelberg=True,
            use_opponent_model=False,
        )
        enable_opp = False
        strat_dim = 11
    elif variant == "opponent_model_only":
        strategic_layer = StrategicLayer(
            config=gt_config,
            use_stackelberg=False,
            use_opponent_model=True,
        )
        enable_opp = True
        strat_dim = 11
    elif variant == "proposed_full":
        strategic_layer = StrategicLayer(
            config=gt_config,
            use_stackelberg=True,
            use_opponent_model=True,
        )
        enable_opp = True
        strat_dim = 11
    else:
        raise ValueError(f"Unknown variant: {variant}")

    ppo_config = PPOConfig(
        lr=3e-4,
        rollout_length=128,
        batch_size=32,
        ppo_epochs=3,
        device="cpu",
        seed=seed,
    )

    encoder_config = StateEncoderConfig(
        initial_price=market_config.initial_price,
        initial_cash=market_config.initial_cash,
        max_order_size=market_config.max_order_size,
    )

    agent = PPOAgent(
        config=ppo_config,
        encoder_config=encoder_config,
        strategic_dim=strat_dim,
    )

    trainer = RLTrainer(
        env=env,
        agent=agent,
        target_agent_id="rl_trader",
        enable_opponent_modeling=enable_opp,
        strategic_layer=strategic_layer,
    )

    ckpt_dir = f"checkpoints/experiments/{variant}_s{seed}"
    train_results = trainer.train(
        total_timesteps=total_timesteps,
        eval_interval=max(total_timesteps // 2, 100),
        checkpoint_interval=total_timesteps,
        checkpoint_dir=ckpt_dir,
    )

    # Perform final matched evaluation
    eval_metrics = trainer.evaluator.evaluate_agent(
        ppo_agent=agent,
        num_episodes=eval_episodes,
        target_agent_id="rl_trader",
        seed=1000 + seed,
        deterministic=True,
        strategic_layer=strategic_layer,
        enable_opponent_modeling=enable_opp,
        freeze_opponent_model=True,
    )

    # Extract final loss if available
    summary = train_results.get("metrics", {})
    last_loss = summary.get("mean_policy_loss", 0.0) + summary.get("mean_value_loss", 0.0)

    record = {
        "variant": variant,
        "seed": seed,
        "total_timesteps": total_timesteps,
        "cumulative_return": eval_metrics["cumulative_return"],
        "mean_reward": eval_metrics["mean_reward"],
        "std_reward": eval_metrics["std_reward"],
        "sharpe_ratio": eval_metrics["sharpe_ratio"],
        "max_drawdown": eval_metrics["max_drawdown"],
        "win_rate": eval_metrics["win_rate"],
        "mean_transaction_cost": eval_metrics["mean_transaction_cost"],
        "mean_market_impact_cost": eval_metrics["mean_market_impact_cost"],
        "mean_trade_count": eval_metrics["mean_trade_count"],
        "mean_turnover": eval_metrics["mean_turnover"],
        "mean_spread": eval_metrics["mean_spread"],
        "mean_market_volatility": eval_metrics["mean_market_volatility"],
        "mean_liquidity": eval_metrics["mean_liquidity"],
        "mean_order_imbalance": eval_metrics["mean_order_imbalance"],
        "final_loss": float(last_loss),
    }

    print(
        f"  [Result] Return: {record['cumulative_return']*100:.2f}% | "
        f"Sharpe: {record['sharpe_ratio']:.2f} | "
        f"MaxDD: {record['max_drawdown']*100:.2f}% | "
        f"Trades: {record['mean_trade_count']:.1f} | "
        f"Spread: {record['mean_spread']:.4f}"
    )

    return record


def main():
    parser = argparse.ArgumentParser(description="Run Controlled Multi-Seed MARL Experiments")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 100, 2026], help="Random seeds to evaluate")
    parser.add_argument("--timesteps", type=int, default=1000, help="Timesteps per variant run")
    parser.add_argument("--eval_episodes", type=int, default=5, help="Number of eval episodes")
    parser.add_argument("--output_dir", type=str, default="experiments/results", help="Directory for output files")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    results = []

    print("================================================================")
    print(" STARTING CONTROLLED ABLATION EXPERIMENTS ACROSS VARIANTS & SEEDS")
    print(f" Variants: {VARIANTS}")
    print(f" Seeds: {args.seeds}")
    print(f" Timesteps per run: {args.timesteps}")
    print("================================================================")

    for variant in VARIANTS:
        for seed in args.seeds:
            record = run_variant(
                variant=variant,
                seed=seed,
                total_timesteps=args.timesteps,
                eval_episodes=args.eval_episodes,
            )
            results.append(record)

    # Save to JSON
    json_path = os.path.join(args.output_dir, "controlled_experiments.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[Saved JSON] -> {json_path}")

    # Save to CSV
    csv_path = os.path.join(args.output_dir, "controlled_experiments.csv")
    df = pd.DataFrame(results)
    df.to_csv(csv_path, index=False)
    print(f"[Saved CSV]  -> {csv_path}")

    # Summary Statistics by Variant
    print("\n================================================================")
    print(" AGGREGATE SUMMARY BY EXPERIMENTAL VARIANT (Mean ± Std)")
    print("================================================================")
    agg_df = df.groupby("variant").agg({
        "cumulative_return": ["mean", "std"],
        "sharpe_ratio": ["mean", "std"],
        "max_drawdown": ["mean", "std"],
        "win_rate": ["mean", "std"],
        "mean_trade_count": ["mean", "std"],
        "mean_spread": ["mean", "std"],
    }).reset_index()

    print(agg_df.to_string())

    summary_path = os.path.join(args.output_dir, "summary_by_variant.csv")
    agg_df.to_csv(summary_path, index=False)
    print(f"\n[Saved Summary Table] -> {summary_path}")


if __name__ == "__main__":
    main()
