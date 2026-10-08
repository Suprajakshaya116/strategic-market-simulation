import numpy as np
from typing import Dict, List, Any


class MetricsTracker:
    """Collects and summarizes training and evaluation metrics."""

    def __init__(self):
        self.reset()

    def reset(self) -> None:
        self.episode_rewards: List[float] = []
        self.episode_lengths: List[int] = []
        self.policy_losses: List[float] = []
        self.value_losses: List[float] = []
        self.entropy_values: List[float] = []
        self.approx_kls: List[float] = []
        self.clip_fracs: List[float] = []
        self.portfolio_values: List[float] = []

    def log_episode(self, reward: float, length: int, final_portfolio_value: float | None = None) -> None:
        self.episode_rewards.append(float(reward))
        self.episode_lengths.append(int(length))
        if final_portfolio_value is not None:
            self.portfolio_values.append(float(final_portfolio_value))

    def log_update(self, update_metrics: dict[str, float]) -> None:
        if "policy_loss" in update_metrics:
            self.policy_losses.append(update_metrics["policy_loss"])
        if "value_loss" in update_metrics:
            self.value_losses.append(update_metrics["value_loss"])
        if "entropy" in update_metrics:
            self.entropy_values.append(update_metrics["entropy"])
        if "approx_kl" in update_metrics:
            self.approx_kls.append(update_metrics["approx_kl"])
        if "clip_frac" in update_metrics:
            self.clip_fracs.append(update_metrics["clip_frac"])

    def get_summary(self) -> dict[str, float]:
        summary = {}
        if self.episode_rewards:
            summary["mean_episode_reward"] = float(np.mean(self.episode_rewards[-100:]))
            summary["std_episode_reward"] = float(np.std(self.episode_rewards[-100:]))
        if self.policy_losses:
            summary["mean_policy_loss"] = float(np.mean(self.policy_losses[-50:]))
        if self.value_losses:
            summary["mean_value_loss"] = float(np.mean(self.value_losses[-50:]))
        if self.entropy_values:
            summary["mean_entropy"] = float(np.mean(self.entropy_values[-50:]))
        if self.approx_kls:
            summary["mean_approx_kl"] = float(np.mean(self.approx_kls[-50:]))
        if self.clip_fracs:
            summary["mean_clip_frac"] = float(np.mean(self.clip_fracs[-50:]))
        if self.portfolio_values:
            summary["mean_portfolio_value"] = float(np.mean(self.portfolio_values[-100:]))
        return summary
