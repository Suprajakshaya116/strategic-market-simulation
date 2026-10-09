import numpy as np
import torch
from typing import Generator, Dict, Any
from .utils import to_tensor


class RolloutBuffer:
    """Rollout Buffer for storing trajectories and computing GAE(lambda) advantages.

    Stores:
      - states
      - actions
      - log_probs
      - rewards
      - values
      - dones
      - strategic_states (optional)
      - global_states (optional for CTDE)
    """

    def __init__(self, rollout_length: int, state_dim: int, device: str = "cpu"):
        self.rollout_length = rollout_length
        self.state_dim = state_dim
        self.device = device
        self.reset()

    def reset(self) -> None:
        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.values = []
        self.dones = []
        self.strategic_states = []
        self.global_states = []

        self.returns = None
        self.advantages = None
        self.ptr = 0

    def add(
        self,
        state: np.ndarray,
        action: int,
        log_prob: float,
        reward: float,
        value: float,
        done: bool,
        strategic_state: np.ndarray | None = None,
        global_state: np.ndarray | None = None
    ) -> None:
        """Add a single transition step to the buffer."""
        self.states.append(np.array(state, dtype=np.float32))
        self.actions.append(int(action))
        self.log_probs.append(float(log_prob))
        self.rewards.append(float(reward))
        self.values.append(float(value))
        self.dones.append(bool(done))
        if strategic_state is not None:
            self.strategic_states.append(np.array(strategic_state, dtype=np.float32))
        if global_state is not None:
            self.global_states.append(np.array(global_state, dtype=np.float32))
        self.ptr += 1

    def compute_returns_and_advantages(
        self,
        last_value: float,
        last_done: bool,
        gamma: float = 0.99,
        gae_lambda: float = 0.95
    ) -> None:
        """Compute Generalized Advantage Estimation (GAE-lambda) and TD-lambda returns."""
        num_steps = len(self.rewards)
        self.advantages = np.zeros(num_steps, dtype=np.float32)
        self.returns = np.zeros(num_steps, dtype=np.float32)

        last_gae_lam = 0.0
        for t in reversed(range(num_steps)):
            if t == num_steps - 1:
                next_non_terminal = 1.0 - float(last_done)
                next_val = float(last_value)
            else:
                next_non_terminal = 1.0 - float(self.dones[t])
                next_val = self.values[t + 1]

            delta = self.rewards[t] + gamma * next_val * next_non_terminal - self.values[t]
            last_gae_lam = delta + gamma * gae_lambda * next_non_terminal * last_gae_lam
            self.advantages[t] = last_gae_lam

        self.returns = self.advantages + np.array(self.values, dtype=np.float32)

    def get_batches(
        self,
        batch_size: int,
        normalize_advantages: bool = True
    ) -> Generator[Dict[str, torch.Tensor], None, None]:
        """Yield mini-batches of tensors for PPO optimization epochs."""
        num_samples = len(self.states)
        indices = np.arange(num_samples)
        np.random.shuffle(indices)

        advantages = self.advantages.copy()
        if normalize_advantages and num_samples > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        states_arr = np.array(self.states, dtype=np.float32)
        actions_arr = np.array(self.actions, dtype=np.int64)
        log_probs_arr = np.array(self.log_probs, dtype=np.float32)
        returns_arr = np.array(self.returns, dtype=np.float32)
        values_arr = np.array(self.values, dtype=np.float32)

        has_strategic = len(self.strategic_states) == num_samples
        strategic_arr = np.array(self.strategic_states, dtype=np.float32) if has_strategic else None

        has_global = len(self.global_states) == num_samples
        global_arr = np.array(self.global_states, dtype=np.float32) if has_global else None

        for start_idx in range(0, num_samples, batch_size):
            end_idx = min(start_idx + batch_size, num_samples)
            mb_idx = indices[start_idx:end_idx]

            batch = {
                "states": to_tensor(states_arr[mb_idx], device=self.device),
                "actions": to_tensor(actions_arr[mb_idx], device=self.device, dtype=torch.long),
                "log_probs": to_tensor(log_probs_arr[mb_idx], device=self.device),
                "returns": to_tensor(returns_arr[mb_idx], device=self.device),
                "values": to_tensor(values_arr[mb_idx], device=self.device),
                "advantages": to_tensor(advantages[mb_idx], device=self.device),
            }
            if has_strategic:
                batch["strategic_states"] = to_tensor(strategic_arr[mb_idx], device=self.device)
            if has_global:
                batch["global_states"] = to_tensor(global_arr[mb_idx], device=self.device)

            yield batch
