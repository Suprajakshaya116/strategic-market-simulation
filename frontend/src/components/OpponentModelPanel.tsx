import React from 'react';
import { Users, BrainCircuit, Info, Eye } from 'lucide-react';
import { OpponentBelief, AgentState } from '../types/simulation';

interface OpponentModelPanelProps {
  beliefs: Record<string, OpponentBelief> | undefined;
  agents: Record<string, AgentState> | undefined;
}

export const OpponentModelPanel: React.FC<OpponentModelPanelProps> = ({ beliefs, agents }) => {
  const momBelief = beliefs?.momentum;
  const valBelief = beliefs?.value;

  const renderProbabilityBar = (
    label: string,
    belief: OpponentBelief | undefined,
    actualAction: string | undefined
  ) => {
    const buyProb = (belief?.buy_prob ?? 0.333) * 100;
    const holdProb = (belief?.hold_prob ?? 0.333) * 100;
    const sellProb = (belief?.sell_prob ?? 0.333) * 100;
    const obsCount = belief?.observation_count ?? 0;
    const isPrior = obsCount === 0;

    return (
      <div className="bg-slate-950 border border-slate-800 rounded-lg p-3.5 space-y-3">
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-2">
            <span className="text-xs font-bold text-slate-200">{label}</span>
            <span className={`text-[10px] font-mono px-1.5 py-0.5 rounded border ${
              isPrior
                ? 'bg-slate-900 text-slate-400 border-slate-700'
                : 'bg-indigo-950 text-indigo-300 border-indigo-800'
            }`}>
              {isPrior ? 'Laplace Prior' : `${obsCount} Observations`}
            </span>
          </div>

          <div className="flex items-center space-x-1.5 font-mono text-[11px]">
            <span className="text-slate-500">Observed Action:</span>
            <span className={`px-1.5 py-0.5 rounded font-bold ${
              actualAction === 'BUY' ? 'bg-emerald-500/20 text-emerald-400' :
              actualAction === 'SELL' ? 'bg-rose-500/20 text-rose-400' :
              'bg-slate-800 text-slate-400'
            }`}>
              {actualAction || 'HOLD'}
            </span>
          </div>
        </div>

        {/* State Bucket Context */}
        <div className="text-[11px] font-mono text-slate-400 flex items-center justify-between">
          <span className="text-slate-500">Market State Discretized Bin:</span>
          <span className="text-slate-300 bg-slate-900 px-1.5 py-0.5 rounded border border-slate-800">
            {belief?.state_bucket || 'Neutral Regime'}
          </span>
        </div>

        {/* Stacked Probability Bar */}
        <div className="space-y-1">
          <div className="w-full h-3.5 rounded-full overflow-hidden flex bg-slate-800 font-mono text-[10px] text-white">
            <div
              className="bg-emerald-500 flex items-center justify-center transition-all duration-300"
              style={{ width: `${buyProb}%` }}
              title={`P(BUY) = ${buyProb.toFixed(1)}%`}
            >
              {buyProb > 15 ? `${buyProb.toFixed(0)}%` : ''}
            </div>
            <div
              className="bg-slate-600 flex items-center justify-center transition-all duration-300"
              style={{ width: `${holdProb}%` }}
              title={`P(HOLD) = ${holdProb.toFixed(1)}%`}
            >
              {holdProb > 15 ? `${holdProb.toFixed(0)}%` : ''}
            </div>
            <div
              className="bg-rose-500 flex items-center justify-center transition-all duration-300"
              style={{ width: `${sellProb}%` }}
              title={`P(SELL) = ${sellProb.toFixed(1)}%`}
            >
              {sellProb > 15 ? `${sellProb.toFixed(0)}%` : ''}
            </div>
          </div>

          {/* Probability Legend & Exact Numbers */}
          <div className="flex justify-between text-[11px] font-mono pt-1 text-slate-400">
            <span className="flex items-center space-x-1">
              <span className="w-2 h-2 rounded-full bg-emerald-500" />
              <span>BUY: <strong className="text-emerald-400">{buyProb.toFixed(1)}%</strong></span>
            </span>
            <span className="flex items-center space-x-1">
              <span className="w-2 h-2 rounded-full bg-slate-500" />
              <span>HOLD: <strong className="text-slate-300">{holdProb.toFixed(1)}%</strong></span>
            </span>
            <span className="flex items-center space-x-1">
              <span className="w-2 h-2 rounded-full bg-rose-500" />
              <span>SELL: <strong className="text-rose-400">{sellProb.toFixed(1)}%</strong></span>
            </span>
          </div>
        </div>
      </div>
    );
  };

  return (
    <div className="bg-slate-900 border border-slate-800 rounded-lg p-4 space-y-4">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div className="flex items-center space-x-2.5">
          <div className="w-8 h-8 rounded bg-cyan-500/20 text-cyan-400 flex items-center justify-center">
            <BrainCircuit className="w-4 h-4" />
          </div>
          <div>
            <h2 className="text-sm font-semibold text-slate-200">Statistical Opponent Modeling</h2>
            <p className="text-xs text-slate-500 font-mono">
              Dirichlet / Laplace Belief Distributions: P(action | discretized_state)
            </p>
          </div>
        </div>
        <div className="flex items-center space-x-1.5 text-xs font-mono text-cyan-400 bg-cyan-950/40 px-2 py-0.5 rounded border border-cyan-800/50">
          <Eye className="w-3.5 h-3.5" />
          <span>Real-Time Ingestion</span>
        </div>
      </div>

      {/* Grid of modeled followers */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
        {renderProbabilityBar('Momentum Follower Model', momBelief, agents?.momentum?.action)}
        {renderProbabilityBar('Value Follower Model', valBelief, agents?.value?.action)}
      </div>

      {/* Mathematical Integration Context Note */}
      <div className="bg-slate-950 border border-slate-800 rounded-lg p-3 flex items-start space-x-3 text-xs font-mono text-slate-400 leading-relaxed">
        <Info className="w-4 h-4 text-cyan-400 shrink-0 mt-0.5" />
        <div>
          <span className="text-cyan-300 font-semibold">Game-Theoretic Coupling: </span>
          Opponent model beliefs P(a_i | s) modulate follower candidate utilities in the Stackelberg solver
          via Quantal Response Equilibrium: P(a | s, a_L) ∝ P(a | s) · exp(U_i(a, a_L) / τ).
          This prevents assuming guaranteed deterministic best responses under noisy market conditions.
        </div>
      </div>
    </div>
  );
};
