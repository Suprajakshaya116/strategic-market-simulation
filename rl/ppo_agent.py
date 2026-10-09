import numpy as np
import torch
import torch.optim as optim

from .config import PPOConfig, StateEncoderConfig
from .state_encoder import StateEncoder
from .action_space import DiscreteActionSpace
from .policy import PolicyNetwork
from .value_network import ValueNetwork
from .rollout_buffer import RolloutBuffer
from .ppo_loss import compute_ppo_loss
from .checkpoint import CheckpointManager
from .utils import set_seed, to_tensor


class PPOAgent:
    """Production/Research-grade Proximal Policy Optimization (PPO) Agent."""

    def __init__(
        self,
        config: PPOConfig | None = None,
        encoder_config: StateEncoderConfig | None = None,
        strategic_dim: int = 0
    ):
        self.config = config or PPOConfig()
        set_seed(self.config.seed)

        self.device = self.config.device
        self.strategic_dim = strategic_dim

        self.encoder = StateEncoder(config=encoder_config, strategic_dim=strategic_dim)
        self.action_space = DiscreteActionSpace(num_actions=3)

        state_dim = self.encoder.total_dim
        self.policy = PolicyNetwork(
            state_size=state_dim,
            action_size=3,
            hidden_dim=self.config.hidden_dim,
            use_layer_norm=self.config.use_layer_norm,
            use_orthogonal_init=self.config.use_orthogonal_init
        ).to(self.device)

        self.value_net = ValueNetwork(
            state_size=state_dim,
            hidden_dim=self.config.hidden_dim,
            use_layer_norm=self.config.use_layer_norm,
            use_orthogonal_init=self.config.use_orthogonal_init
        ).to(self.device)

        self.optimizer = optim.Adam([
            {"params": self.policy.parameters(), "lr": self.config.lr},
            {"params": self.value_net.parameters(), "lr": self.config.lr}
        ], eps=1e-5)

        self.buffer = RolloutBuffer(
            rollout_length=self.config.rollout_length,
            state_dim=state_dim,
            device=self.device
        )

        self.checkpoint_manager = CheckpointManager()
        self.step_count = 0

    def select_action(
        self,
        state: dict,
        strategic_dict: dict | None = None,
        update_encoder_stats: bool = True,
        deterministic: bool = False
    ) -> tuple[int, float, float]:
        """Encode state, forward policy/value networks, sample action and return (action, log_prob, value)."""
        state_vec = self.encoder.encode(
            state=state,
            strategic_dict=strategic_dict,
            update_stats=update_encoder_stats
        )
        state_tensor = to_tensor(state_vec, device=self.device).unsqueeze(0)

        with torch.no_grad():
            action_tensor, log_prob_tensor, _ = self.policy.act(
                state_tensor,
                deterministic=deterministic
            )
            value_tensor = self.value_net(state_tensor)

        action = action_tensor.squeeze(0).item()
        log_prob = log_prob_tensor.squeeze(0).item()
        value = value_tensor.squeeze(0).item()

        return action, log_prob, value

    def get_value(self, state: dict, strategic_dict: dict | None = None) -> float:
        """Compute state value estimate V(s)."""
        state_vec = self.encoder.encode(state, strategic_dict, update_stats=False)
        state_tensor = to_tensor(state_vec, device=self.device).unsqueeze(0)
        with torch.no_grad():
            val = self.value_net(state_tensor).squeeze(0).item()
        return val

    def store_transition(
        self,
        state: dict,
        action: int,
        log_prob: float,
        reward: float,
        value: float,
        done: bool,
        strategic_dict: dict | None = None
    ) -> None:
        state_vec = self.encoder.encode(state, strategic_dict, update_stats=False)
        strat_vec = self.encoder.encode_strategic(strategic_dict) if self.strategic_dim > 0 else None

        self.buffer.add(
            state=state_vec,
            action=action,
            log_prob=log_prob,
            reward=reward,
            value=value,
            done=done,
            strategic_state=strat_vec
        )
        self.step_count += 1

    def update(self, last_state: dict, last_done: bool, last_strategic_dict: dict | None = None) -> dict[str, float]:
        """Perform PPO optimization epochs over rollout buffer."""
        last_val = self.get_value(last_state, last_strategic_dict)
        self.buffer.compute_returns_and_advantages(
            last_value=last_val,
            last_done=last_done,
            gamma=self.config.gamma,
            gae_lambda=self.config.gae_lambda
        )

        metrics_list = []

        for epoch in range(self.config.ppo_epochs):
            for batch in self.buffer.get_batches(self.config.batch_size):
                states = batch["states"]
                actions = batch["actions"]
                old_log_probs = batch["log_probs"]
                returns = batch["returns"]
                old_values = batch["values"]
                advantages = batch["advantages"]

                # Forward passes
                dist = self.policy(states)
                new_log_probs = dist.log_prob(actions)
                entropy = dist.entropy()
                new_values = self.value_net(states)

                # PPO loss
                loss, loss_metrics = compute_ppo_loss(
                    new_log_probs=new_log_probs,
                    old_log_probs=old_log_probs,
                    advantages=advantages,
                    new_values=new_values,
                    old_values=old_values,
                    returns=returns,
                    entropy=entropy,
                    clip_eps=self.config.clip_eps,
                    c1_value_loss_coeff=self.config.c1_value_loss_coeff,
                    c2_entropy_coeff=self.config.c2_entropy_coeff,
                    clip_value_loss=self.config.clip_value_loss
                )

                # Optimization step
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    list(self.policy.parameters()) + list(self.value_net.parameters()),
                    self.config.max_grad_norm
                )
                self.optimizer.step()

                metrics_list.append(loss_metrics)

                # Early stopping on target KL if set
                if self.config.target_kl is not None and loss_metrics["approx_kl"] > 1.5 * self.config.target_kl:
                    break

        self.buffer.reset()

        # Compute average metrics across updates
        avg_metrics = {}
        if metrics_list:
            for k in metrics_list[0].keys():
                avg_metrics[k] = float(np.mean([m[k] for m in metrics_list]))

        return avg_metrics

    def save_checkpoint(self, filepath: str) -> str:
        return self.checkpoint_manager.save_checkpoint(
            filepath=filepath,
            policy_state_dict=self.policy.state_dict(),
            value_state_dict=self.value_net.state_dict(),
            optimizer_state_dict=self.optimizer.state_dict(),
            encoder_state_dict=self.encoder.get_state_dict(),
            step_count=self.step_count,
            config_dict={"ppo_config": self.config.__dict__}
        )

    def load_checkpoint(self, filepath: str) -> None:
        payload = self.checkpoint_manager.load_checkpoint(filepath, device=self.device)
        self.policy.load_state_dict(payload["policy"])
        self.value_net.load_state_dict(payload["value"])
        self.optimizer.load_state_dict(payload["optimizer"])
        self.encoder.load_state_dict(payload["encoder"])
        self.step_count = payload.get("step_count", 0)
