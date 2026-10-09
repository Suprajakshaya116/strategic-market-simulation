import React from 'react';
import { Bot, Shield, TrendingUp, Zap, DollarSign, Wallet } from 'lucide-react';
import { AgentState, LeaderDecision } from '../types/simulation';

interface MultiAgentPanelProps {
  agents: Record<string, AgentState> | undefined;
  leaderDecision: LeaderDecision | undefined;
}

export const MultiAgentPanel: React.FC<MultiAgentPanelProps> = ({ agents, leaderDecision }) => {
  const rlAgent = agents?.rl_trader;
  const momAgent = agents?.momentum;
  const valAgent = agents?.value;

  const getActionBadge = (action: string | undefined, status: string | undefined) => {
    const act = action || 'HOLD';
    const stat = status || 'HOLD';

    let color = 'bg-slate-800 text-slate-300 border-slate-700';
    if (act === 'BUY') {
      color = stat === 'EXECUTED' ? 'bg-emerald-500/15 text-emerald-400 border-emerald-500/40' : 'bg-emerald-950/40 text-emerald-300 border-emerald-800';
    } else if (act === 'SELL') {
      color = stat === 'EXECUTED' ? 'bg-rose-500/15 text-rose-400 border-rose-500/40' : 'bg-rose-950/40 text-rose-300 border-rose-800';
    }

    return (
      <div className="flex items-center space-x-1.5 font-mono">
        <span className={`px-2 py-0.5 rounded text-xs font-bold border ${color}`}>
          {act}
        </span>
        <span className={`text-[11px] font-semibold ${
          stat === 'EXECUTED' ? 'text-emerald-400' : stat === 'REJECTED' ? 'text-rose-400' : 'text-slate-500'
        }`}>
          [{stat}]
        </span>
      </div>
    );
  };

  return (
    <div className="bg-slate-900 border border-slate-800 rounded-lg p-4">
      <div className="flex items-center justify-between mb-4">
        <div>
          <h2 className="text-sm font-semibold text-slate-200">Multi-Agent Market Participants</h2>
          <p className="text-xs text-slate-500 font-mono">Live Action Intent, Execution States & Portfolio Valuation</p>
        </div>
        <span className="text-xs font-mono text-indigo-400 bg-indigo-950/50 border border-indigo-800/50 px-2 py-0.5 rounded">
          4 Co-existing Entities
        </span>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-3">
        {/* RL Trader (Member 3) */}
        <div className="bg-slate-950 border border-indigo-900/40 rounded-lg p-3.5 relative overflow-hidden">
          <div className="absolute top-0 right-0 w-24 h-24 bg-indigo-500/5 rounded-full blur-xl pointer-events-none" />
          <div className="flex items-center justify-between mb-2">
            <div className="flex items-center space-x-2">
              <div className="w-6 h-6 rounded bg-indigo-500/20 text-indigo-400 flex items-center justify-center">
                <Zap className="w-3.5 h-3.5" />
              </div>
              <span className="text-xs font-bold text-slate-200">RL Trader (PPO)</span>
            </div>
            <span className="text-[10px] font-mono px-1.5 py-0.5 rounded bg-indigo-950 text-indigo-300 border border-indigo-800">
              Target Agent
            </span>
          </div>

          <div className="my-2 flex items-center justify-between">
            <span className="text-xs text-slate-400">Current Action:</span>
            {getActionBadge(rlAgent?.action, rlAgent?.action_status)}
          </div>

          <div className="space-y-1.5 pt-2 border-t border-slate-800/80 font-mono text-xs">
            <div className="flex justify-between">
              <span className="text-slate-500">Portfolio Value:</span>
              <span className="font-semibold text-slate-200">${(rlAgent?.portfolio_value || 100000).toFixed(2)}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-slate-500">Cumulative Return:</span>
              <span className={`font-semibold ${((rlAgent?.cumulative_pnl || 0) >= 0) ? 'text-emerald-400' : 'text-rose-400'}`}>
                {((rlAgent?.cumulative_pnl || 0) * 100).toFixed(2)}%
              </span>
            </div>
            <div className="flex justify-between">
              <span className="text-slate-500">Inventory / Cash:</span>
              <span className="text-slate-300">{rlAgent?.inventory || 0} shares / ${(rlAgent?.cash || 0).toFixed(0)}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-slate-500">Step Reward:</span>
              <span className="text-indigo-400 font-semibold">{rlAgent?.reward ? rlAgent.reward.toFixed(4) : '0.0000'}</span>
            </div>
          </div>
        </div>

        {/* Momentum Agent (Member 1) */}
        <div className="bg-slate-950 border border-slate-800 rounded-lg p-3.5">
          <div className="flex items-center justify-between mb-2">
            <div className="flex items-center space-x-2">
              <div className="w-6 h-6 rounded bg-amber-500/20 text-amber-400 flex items-center justify-center">
                <TrendingUp className="w-3.5 h-3.5" />
              </div>
              <span className="text-xs font-bold text-slate-200">Momentum Agent</span>
            </div>
            <span className="text-[10px] font-mono text-slate-400">Trend Follower</span>
          </div>

          <div className="my-2 flex items-center justify-between">
            <span className="text-xs text-slate-400">Current Action:</span>
            {getActionBadge(momAgent?.action, momAgent?.action_status)}
          </div>

          <div className="space-y-1.5 pt-2 border-t border-slate-800/80 font-mono text-xs">
            <div className="flex justify-between">
              <span className="text-slate-500">Signal (Return):</span>
              <span className="text-slate-300">{momAgent?.signal_metric ? (momAgent.signal_metric * 100).toFixed(2) + '%' : '0.00%'}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-slate-500">Portfolio Value:</span>
              <span className="font-semibold text-slate-200">${(momAgent?.portfolio_value || 100000).toFixed(2)}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-slate-500">Inventory / Cash:</span>
              <span className="text-slate-300">{momAgent?.inventory || 0} shares / ${(momAgent?.cash || 0).toFixed(0)}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-slate-500">Total Tx Fees:</span>
              <span className="text-slate-400">${(momAgent?.transaction_cost || 0).toFixed(2)}</span>
            </div>
          </div>
        </div>

        {/* Value Agent (Member 1) */}
        <div className="bg-slate-950 border border-slate-800 rounded-lg p-3.5">
          <div className="flex items-center justify-between mb-2">
            <div className="flex items-center space-x-2">
              <div className="w-6 h-6 rounded bg-blue-500/20 text-blue-400 flex items-center justify-center">
                <Bot className="w-3.5 h-3.5" />
              </div>
              <span className="text-xs font-bold text-slate-200">Value Agent</span>
            </div>
            <span className="text-[10px] font-mono text-slate-400">Mean Reversion</span>
          </div>

          <div className="my-2 flex items-center justify-between">
            <span className="text-xs text-slate-400">Current Action:</span>
            {getActionBadge(valAgent?.action, valAgent?.action_status)}
          </div>

          <div className="space-y-1.5 pt-2 border-t border-slate-800/80 font-mono text-xs">
            <div className="flex justify-between">
              <span className="text-slate-500">Signal (Mispricing):</span>
              <span className="text-slate-300">{valAgent?.signal_metric ? (valAgent.signal_metric * 100).toFixed(2) + '%' : '0.00%'}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-slate-500">Portfolio Value:</span>
              <span className="font-semibold text-slate-200">${(valAgent?.portfolio_value || 100000).toFixed(2)}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-slate-500">Inventory / Cash:</span>
              <span className="text-slate-300">{valAgent?.inventory || 0} shares / ${(valAgent?.cash || 0).toFixed(0)}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-slate-500">Total Tx Fees:</span>
              <span className="text-slate-400">${(valAgent?.transaction_cost || 0).toFixed(2)}</span>
            </div>
          </div>
        </div>

        {/* Strategic Market Maker (Member 2) */}
        <div className="bg-slate-950 border border-amber-900/40 rounded-lg p-3.5 relative overflow-hidden">
          <div className="absolute top-0 right-0 w-24 h-24 bg-amber-500/5 rounded-full blur-xl pointer-events-none" />
          <div className="flex items-center justify-between mb-2">
            <div className="flex items-center space-x-2">
              <div className="w-6 h-6 rounded bg-amber-500/20 text-amber-400 flex items-center justify-center">
                <Shield className="w-3.5 h-3.5" />
              </div>
              <span className="text-xs font-bold text-slate-200">Market Maker (Leader)</span>
            </div>
            <span className="text-[10px] font-mono px-1.5 py-0.5 rounded bg-amber-950 text-amber-300 border border-amber-800">
              Stackelberg
            </span>
          </div>

          <div className="my-2 flex items-center justify-between">
            <span className="text-xs text-slate-400">Control Decision:</span>
            <span className={`px-2 py-0.5 rounded text-xs font-mono font-bold border ${
              leaderDecision?.selected_action === 'TIGHT' ? 'bg-emerald-500/20 text-emerald-400 border-emerald-500/40' :
              leaderDecision?.selected_action === 'WIDE' ? 'bg-rose-500/20 text-rose-400 border-rose-500/40' :
              'bg-blue-500/20 text-blue-400 border-blue-500/40'
            }`}>
              {leaderDecision?.selected_action || 'MEDIUM'}
            </span>
          </div>

          <div className="space-y-1.5 pt-2 border-t border-slate-800/80 font-mono text-xs">
            <div className="flex justify-between">
              <span className="text-slate-500">Leader Utility:</span>
              <span className="font-semibold text-amber-400">{leaderDecision?.leader_utility.toFixed(4) || '0.0000'}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-slate-500">Spread Multiplier:</span>
              <span className="text-slate-300">{leaderDecision?.spread_multiplier.toFixed(2) || '1.00'}x</span>
            </div>
            <div className="flex justify-between">
              <span className="text-slate-500">Liquidity Multiplier:</span>
              <span className="text-slate-300">{leaderDecision?.liquidity_multiplier.toFixed(2) || '1.00'}x</span>
            </div>
            <div className="flex justify-between">
              <span className="text-slate-500">Role:</span>
              <span className="text-slate-400 text-[11px]">Liquidity Controller</span>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};
