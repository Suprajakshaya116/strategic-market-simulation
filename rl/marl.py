import numpy as np
import torch
import torch.optim as optim

from .config import MARLConfig, PPOConfig, StateEncoderConfig
from .state_encoder import StateEncoder
from .action_space import DiscreteActionSpace
from .policy import PolicyNetwork
from .critic import CentralizedCritic
from .rollout_buffer import RolloutBuffer
from .ppo_loss import compute_ppo_loss
from .utils import set_seed, to_tensor


class MAPPOAgent:
    """Multi-Agent PPO (MAPPO) Controller with Centralized Training and Decentralized Execution (CTDE).

    Each trader agent possesses a decentralized PolicyNetwork receiving local state s_i,
    while a CentralizedCritic estimates V(S_global) during training using all agent states.
    """

    def __init__(
        self,
        marl_config: MARLConfig | None = None,
        encoder_config: StateEncoderConfig | None = None
    ):
        self.marl_config = marl_config or MARLConfig()
        self.ppo_config = self.marl_config.ppo_config
        set_seed(self.ppo_config.seed)

        self.device = self.ppo_config.device
        self.agent_ids = self.marl_config.agent_ids

        # Decentralized Encoders & Policies per agent
        self.encoders = {
            aid: StateEncoder(config=encoder_config) for aid in self.agent_ids
        }
        self.action_space = DiscreteActionSpace(num_actions=3)

        local_dim = self.encoders[self.agent_ids[0]].total_dim
        global_dim = local_dim * len(self.agent_ids)

        self.policies = {
            aid: PolicyNetwork(
                state_size=local_dim,
                action_size=3,
                hidden_dim=self.ppo_config.hidden_dim,
                use_layer_norm=self.ppo_config.use_layer_norm,
                use_orthogonal_init=self.ppo_config.use_orthogonal_init
            ).to(self.device)
            for aid in self.agent_ids
        }

        # Centralized Critic
        self.centralized_critic = CentralizedCritic(
            global_state_dim=global_dim,
            hidden_dim=self.ppo_config.hidden_dim * 2,
            use_layer_norm=self.ppo_config.use_layer_norm,
            use_orthogonal_init=self.ppo_config.use_orthogonal_init
        ).to(self.device)

        # Optimizers
        params = list(self.centralized_critic.parameters())
        for policy in self.policies.values():
            params.extend(list(policy.parameters()))

        self.optimizer = optim.Adam(params, lr=self.ppo_config.lr, eps=1e-5)

        # Buffers per agent
        self.buffers = {
            aid: RolloutBuffer(
                rollout_length=self.ppo_config.rollout_length,
                state_dim=local_dim,
                device=self.device
            )
            for aid in self.agent_ids
        }

    def encode_global_state(self, states: dict[str, dict]) -> np.ndarray:
        """Construct global state tensor by concatenating normalized local agent states."""
        local_vecs = [self.encoders[aid].encode(states[aid]) for aid in self.agent_ids]
        return np.concatenate(local_vecs, axis=0).astype(np.float32)

    def select_actions(
        self,
        states: dict[str, dict],
        deterministic: bool = False
    ) -> tuple[dict[str, int], dict[str, float], float]:
        """Select actions for all agents and return values for CTDE."""
        global_vec = self.encode_global_state(states)
        global_tensor = to_tensor(global_vec, device=self.device).unsqueeze(0)

        with torch.no_grad():
            global_val = self.centralized_critic(global_tensor).squeeze(0).item()

        actions = {}
        log_probs = {}

        for aid in self.agent_ids:
            local_vec = self.encoders[aid].encode(states[aid], update_stats=True)
            local_tensor = to_tensor(local_vec, device=self.device).unsqueeze(0)
            with torch.no_grad():
                act_tensor, lp_tensor, _ = self.policies[aid].act(local_tensor, deterministic=deterministic)
            actions[aid] = act_tensor.squeeze(0).item()
            log_probs[aid] = lp_tensor.squeeze(0).item()

        return actions, log_probs, global_val

    def store_transitions(
        self,
        states: dict[str, dict],
        actions: dict[str, int],
        log_probs: dict[str, float],
        rewards: dict[str, float],
        global_val: float,
        done: bool
    ) -> None:
        global_vec = self.encode_global_state(states)
        for aid in self.agent_ids:
            local_vec = self.encoders[aid].encode(states[aid], update_stats=False)
            self.buffers[aid].add(
                state=local_vec,
                action=actions[aid],
                log_prob=log_probs[aid],
                reward=rewards[aid],
                value=global_val,
                done=done,
                global_state=global_vec
            )

    def update(self, last_states: dict[str, dict], last_done: bool) -> dict[str, float]:
        """Perform MAPPO optimization epoch over rollouts."""
        global_vec = self.encode_global_state(last_states)
        global_tensor = to_tensor(global_vec, device=self.device).unsqueeze(0)
        with torch.no_grad():
            last_global_val = self.centralized_critic(global_tensor).squeeze(0).item()

        for aid in self.agent_ids:
            self.buffers[aid].compute_returns_and_advantages(
                last_value=last_global_val,
                last_done=last_done,
                gamma=self.ppo_config.gamma,
                gae_lambda=self.ppo_config.gae_lambda
            )

        metrics_list = []
        # Update optimization loop
        for epoch in range(self.ppo_config.ppo_epochs):
            for aid in self.agent_ids:
                for batch in self.buffers[aid].get_batches(self.ppo_config.batch_size):
                    states = batch["states"]
                    actions = batch["actions"]
                    old_log_probs = batch["log_probs"]
                    returns = batch["returns"]
                    old_values = batch["values"]
                    advantages = batch["advantages"]
                    global_states = batch["global_states"]

                    dist = self.policies[aid](states)
                    new_log_probs = dist.log_prob(actions)
                    entropy = dist.entropy()
                    new_global_vals = self.centralized_critic(global_states)

                    loss, loss_metrics = compute_ppo_loss(
                        new_log_probs=new_log_probs,
                        old_log_probs=old_log_probs,
                        advantages=advantages,
                        new_values=new_global_vals,
                        old_values=old_values,
                        returns=returns,
                        entropy=entropy,
                        clip_eps=self.ppo_config.clip_eps,
                        c1_value_loss_coeff=self.ppo_config.c1_value_loss_coeff,
                        c2_entropy_coeff=self.ppo_config.c2_entropy_coeff,
                        clip_value_loss=self.ppo_config.clip_value_loss
                    )

                    self.optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        list(self.centralized_critic.parameters()) + list(self.policies[aid].parameters()),
                        self.ppo_config.max_grad_norm
                    )
                    self.optimizer.step()
                    metrics_list.append(loss_metrics)

        for aid in self.agent_ids:
            self.buffers[aid].reset()

        avg_metrics = {}
        if metrics_list:
            for k in metrics_list[0].keys():
                avg_metrics[k] = float(np.mean([m[k] for m in metrics_list]))
        return avg_metrics
