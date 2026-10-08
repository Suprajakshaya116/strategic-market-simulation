import torch
import torch.nn as nn
from torch.distributions import Categorical
from .networks import FeatureExtractor, CategoricalActor


class PolicyNetwork(nn.Module):
    """Research-quality Policy Network for discrete trading actions (HOLD, BUY, SELL).

    Combines a deep feature extraction backbone with LayerNorm, non-linear activation,
    and an orthogonally-initialized Categorical policy head.

    Exposes:
      - forward(state) -> distribution
      - act(state, deterministic=False) -> (action, log_prob, entropy)
      - evaluate_actions(state, actions) -> (log_probs, entropy)
      - get_action_and_value(state) -> (action, log_prob, entropy, logits, action_probs)
    """

    def __init__(
        self,
        state_size: int = 10,
        action_size: int = 3,
        hidden_dim: int = 64,
        use_layer_norm: bool = True,
        use_orthogonal_init: bool = True
    ):
        super().__init__()
        self.state_size = state_size
        self.action_size = action_size

        self.feature_extractor = FeatureExtractor(
            input_dim=state_size,
            hidden_dim=hidden_dim,
            use_layer_norm=use_layer_norm,
            use_orthogonal_init=use_orthogonal_init,
            activation="tanh"
        )
        self.actor_head = CategoricalActor(
            hidden_dim=hidden_dim,
            action_dim=action_size,
            use_orthogonal_init=use_orthogonal_init
        )

    def forward(self, state: torch.Tensor) -> Categorical:
        """Returns action distribution for given state tensor."""
        features = self.feature_extractor(state)
        dist = self.actor_head(features)
        return dist

    def act(
        self,
        state: torch.Tensor,
        deterministic: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample or greedily select an action and return (action, log_prob, entropy)."""
        dist = self.forward(state)
        if deterministic:
            action = torch.argmax(dist.probs, dim=-1)
        else:
            action = dist.sample()
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        return action, log_prob, entropy

    def evaluate_actions(
        self,
        state: torch.Tensor,
        actions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Evaluate log probabilities and entropy of given actions under current policy."""
        dist = self.forward(state)
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()
        return log_probs, entropy

    def get_action_details(
        self,
        state: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        """Return logits, action probabilities, sampled action, log prob, and entropy."""
        features = self.feature_extractor(state)
        dist = self.actor_head(features)
        action = dist.sample()
        return {
            "logits": dist.logits,
            "action_probs": dist.probs,
            "action": action,
            "log_prob": dist.log_prob(action),
            "entropy": dist.entropy()
        }
