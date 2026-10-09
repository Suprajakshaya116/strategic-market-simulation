export interface MarketState {
  price: number;
  return: number;
  volume: number;
  spread: number;
  liquidity: number;
  volatility: number;
  order_imbalance: number;
  fundamental_value: number;
  bid_price: number;
  ask_price: number;
}

export interface AgentState {
  agent_id: string;
  action: 'BUY' | 'SELL' | 'HOLD';
  action_status: 'EXECUTED' | 'REJECTED' | 'HOLD';
  quantity: number;
  fill_price: number;
  cash: number;
  inventory: number;
  portfolio_value: number;
  reward: number;
  cumulative_pnl: number;
  transaction_cost: number;
  market_impact_cost: number;
  signal_metric?: number | null;
  signal_label?: string | null;
}

export interface CandidateEvaluation {
  leader_action: 'TIGHT' | 'MEDIUM' | 'WIDE';
  predicted_follower_actions: Record<string, string>;
  predicted_follower_utilities: Record<string, number>;
  expected_buy_pressure: number;
  expected_sell_pressure: number;
  expected_order_imbalance: number;
  leader_utility: number;
}

export interface LeaderDecision {
  selected_action: 'TIGHT' | 'MEDIUM' | 'WIDE';
  leader_utility: number;
  spread_multiplier: number;
  liquidity_multiplier: number;
  explanation: string;
  candidate_evaluations: CandidateEvaluation[];
}

export interface OpponentBelief {
  agent_id: string;
  buy_prob: number;
  hold_prob: number;
  sell_prob: number;
  observation_count: number;
  state_bucket: string;
  last_observed_action?: string | null;
}

export interface SimulationFrame {
  step: number;
  episode: number;
  status: 'IDLE' | 'RUNNING' | 'PAUSED' | 'COMPLETED' | 'FAILED';
  timestamp: number;
  market_state: MarketState;
  leader_decision: LeaderDecision;
  agents: Record<string, AgentState>;
  opponent_beliefs: Record<string, OpponentBelief>;
  recent_orders: Array<{
    agent_id: string;
    action: string;
    quantity: number;
    fill_price: number;
    status: string;
    reason?: string;
    transaction_cost?: number;
    notional?: number;
  }>;
  encoded_state_dim: number;
  strategic_feature_dim: number;
  active_variant: string;
}

export interface HistoryPoint {
  step: number;
  price: number;
  fundamental_value: number;
  spread: number;
  liquidity: number;
  return: number;
  volatility: number;
  order_imbalance: number;
  leader_action: string;
  leader_utility: number;
  rl_pnl: number;
  rl_portfolio_value: number;
  rl_reward: number;
}

export interface EventLog {
  id: string;
  timestamp: string;
  step: number;
  category: 'MARKET' | 'STACKELBERG' | 'OPPONENT' | 'AGENT' | 'ORDER' | 'PPO';
  message: string;
  details?: Record<string, any>;
}

export interface TrainingStatus {
  is_training: boolean;
  current_step: number;
  total_timesteps: number;
  variant: string;
  seed: number;
  metrics: {
    policy_loss?: number;
    value_loss?: number;
    total_loss?: number;
    entropy?: number;
    approx_kl?: number;
    clip_frac?: number;
  };
  history: Array<{
    step: number;
    policy_loss: number;
    value_loss: number;
    total_loss: number;
    entropy: number;
    approx_kl: number;
    reward: number;
  }>;
}

export interface CheckpointInfo {
  path: string;
  filename: string;
  size_kb: number;
  modified: string;
}

export interface ExperimentRecord {
  variant: string;
  seed: number;
  total_timesteps: number;
  cumulative_return: number;
  mean_reward: number;
  std_reward: number;
  sharpe_ratio: number;
  max_drawdown: number;
  win_rate: number;
  mean_transaction_cost: number;
  mean_market_impact_cost: number;
  mean_trade_count: number;
  mean_turnover: number;
  mean_spread: number;
  mean_market_volatility: number;
  mean_liquidity: number;
  mean_order_imbalance: number;
  final_loss: number;
}
