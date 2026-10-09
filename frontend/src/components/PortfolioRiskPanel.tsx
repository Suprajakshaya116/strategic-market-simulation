import React from 'react';
import {
  ResponsiveContainer,
  LineChart,
  Line,
  XAxis,
  YAxis,
  Tooltip,
  CartesianGrid,
} from 'recharts';
import { ShieldAlert, TrendingDown, DollarSign, Wallet, ArrowUpRight, ArrowDownRight } from 'lucide-react';
import { AgentState, HistoryPoint } from '../types/simulation';

interface PortfolioRiskPanelProps {
  agents: Record<string, AgentState> | undefined;
  history: HistoryPoint[];
}

export const PortfolioRiskPanel: React.FC<PortfolioRiskPanelProps> = ({ agents, history }) => {
  const rlAgent = agents?.rl_trader;
  const momAgent = agents?.momentum;
  const valAgent = agents?.value;

  const rlVal = rlAgent?.portfolio_value ?? 100000;
  const rlCash = rlAgent?.cash ?? 100000;
  const rlInv = rlAgent?.inventory ?? 0;
  const rlTxCost = rlAgent?.transaction_cost ?? 0;
  const rlImpact = rlAgent?.market_impact_cost ?? 0;
  const rlReturn = rlAgent?.cumulative_pnl ?? 0;

  // Compute live drawdown from historical values
  const rlHist = history.map((h) => h.rl_portfolio_value || 100000);
  let peak = 100000;
  let maxDd = 0;
  for (const v of rlHist) {
    if (v > peak) peak = v;
    const dd = (peak - v) / (peak + 1e-8);
    if (dd > maxDd) maxDd = dd;
  }

  return (
    <div className="bg-slate-900 border border-slate-800 rounded-lg p-4 space-y-4">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-sm font-semibold text-slate-200">Portfolio & Quantitative Risk Monitoring</h2>
          <p className="text-xs text-slate-500 font-mono">
            Mark-to-Market Accounting, Slippage Costs, and Drawdown Analytics
          </p>
        </div>
        <div className="flex items-center space-x-2 font-mono text-xs">
          <span className="text-slate-400">Target Policy PnL:</span>
          <span className={`font-bold ${rlReturn >= 0 ? 'text-emerald-400' : 'text-rose-400'}`}>
            {(rlReturn * 100).toFixed(2)}%
          </span>
        </div>
      </div>

      {/* Risk Metrics Cards */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-3 font-mono text-xs">
        <div className="bg-slate-950 border border-slate-800 rounded-lg p-3">
          <span className="text-slate-500 block mb-1">Max Drawdown (Live):</span>
          <div className="text-lg font-bold text-rose-400">{(maxDd * 100).toFixed(2)}%</div>
          <span className="text-[11px] text-slate-500 mt-0.5 block">Peak to trough risk</span>
        </div>

        <div className="bg-slate-950 border border-slate-800 rounded-lg p-3">
          <span className="text-slate-500 block mb-1">Total Transaction Fees:</span>
          <div className="text-lg font-bold text-amber-400">${rlTxCost.toFixed(2)}</div>
          <span className="text-[11px] text-slate-500 mt-0.5 block">Brokerage & execution fees</span>
        </div>

        <div className="bg-slate-950 border border-slate-800 rounded-lg p-3">
          <span className="text-slate-500 block mb-1">Market Impact Slippage:</span>
          <div className="text-lg font-bold text-cyan-400">${rlImpact.toFixed(2)}</div>
          <span className="text-[11px] text-slate-500 mt-0.5 block">Order-book depth friction</span>
        </div>

        <div className="bg-slate-950 border border-slate-800 rounded-lg p-3">
          <span className="text-slate-500 block mb-1">Capital Deployment:</span>
          <div className="text-lg font-bold text-slate-200">
            {((Math.abs(rlInv * (history[history.length - 1]?.price || 100)) / rlVal) * 100).toFixed(1)}%
          </div>
          <span className="text-[11px] text-slate-500 mt-0.5 block">Active Inventory exposure</span>
        </div>
      </div>

      {/* Comparative Portfolio Curve */}
      <div className="bg-slate-950 border border-slate-800 rounded-lg p-3.5">
        <div className="flex items-center justify-between mb-2">
          <span className="text-xs font-semibold text-slate-300">Target Agent Equity Trajectory ($)</span>
          <span className="text-xs font-mono text-indigo-400">PPO Trader</span>
        </div>

        <div className="h-44 w-full">
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={history} margin={{ top: 5, right: 10, left: -10, bottom: 0 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#1f2937" vertical={false} />
              <XAxis dataKey="step" stroke="#6b7280" fontSize={11} tickLine={false} />
              <YAxis
                stroke="#6b7280"
                fontSize={11}
                tickLine={false}
                domain={['auto', 'auto']}
                tickFormatter={(val) => `$${(val / 1000).toFixed(1)}k`}
              />
              <Tooltip
                contentStyle={{ backgroundColor: '#0f172a', borderColor: '#334155', borderRadius: '6px', fontSize: '12px' }}
                formatter={(val: any) => [`$${Number(val).toFixed(2)}`, 'Portfolio Value']}
              />
              <Line
                type="monotone"
                dataKey="rl_portfolio_value"
                name="RL Trader"
                stroke="#818cf8"
                strokeWidth={2}
                dot={false}
                isAnimationActive={false}
              />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </div>
    </div>
  );
};
