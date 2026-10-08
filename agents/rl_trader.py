from rl.ppo_agent import PPOAgent
from rl.action_space import DiscreteActionSpace


class RLTraderAgent:
    """Agent wrapper integrating trained PPOAgent with MarketEnvironment interface."""

    def __init__(self, ppo_agent: PPOAgent | None = None, deterministic: bool = True):
        self.ppo_agent = ppo_agent or PPOAgent()
        self.deterministic = deterministic
        self.action_space = DiscreteActionSpace(num_actions=3)

    def act(self, state: dict) -> str:
        """Receive state dictionary, forward through PPO policy, and return action string ("BUY", "HOLD", "SELL")."""
        action_idx, _, _ = self.ppo_agent.select_action(
            state=state,
            update_encoder_stats=False,
            deterministic=self.deterministic
        )
        return self.action_space.to_env_action(action_idx)
