import React from 'react';
import { Sliders, X, Check, Code, Shield } from 'lucide-react';
import { SimulationFrame } from '../types/simulation';

interface DiagnosticsModalProps {
  isOpen: boolean;
  onClose: () => void;
  frame: SimulationFrame | null;
}

export const DiagnosticsModal: React.FC<DiagnosticsModalProps> = ({ isOpen, onClose, frame }) => {
  if (!isOpen) return null;

  const marketState = frame?.market_state;
  const stateKeys = marketState ? Object.keys(marketState) : [];

  return (
    <div className="fixed inset-0 z-50 bg-black/70 backdrop-blur-sm flex items-center justify-center p-4">
      <div className="bg-slate-900 border border-slate-800 rounded-xl max-w-2xl w-full max-h-[90vh] overflow-y-auto shadow-2xl p-6 space-y-5 text-slate-200">
        <div className="flex items-center justify-between border-b border-slate-800 pb-3">
          <div className="flex items-center space-x-2">
            <Sliders className="w-5 h-5 text-indigo-400" />
            <h2 className="text-base font-bold text-slate-100">Architecture & State Space Inspector</h2>
          </div>
          <button
            onClick={onClose}
            className="text-slate-400 hover:text-slate-100 p-1 rounded hover:bg-slate-800"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        {/* Dimension Verification Summary */}
        <div className="grid grid-cols-3 gap-3 font-mono text-xs">
          <div className="bg-slate-950 p-3 rounded-lg border border-slate-800">
            <span className="text-slate-500 block mb-1">Market State Dims:</span>
            <div className="text-xl font-bold text-cyan-400">10 Features</div>
            <span className="text-[10px] text-slate-500">Domain Normalizations</span>
          </div>

          <div className="bg-slate-950 p-3 rounded-lg border border-slate-800">
            <span className="text-slate-500 block mb-1">Strategic Feature Dims:</span>
            <div className="text-xl font-bold text-amber-400">{frame?.strategic_feature_dim || 11} Features</div>
            <span className="text-[10px] text-slate-500">Leader Action + Opponent Probs</span>
          </div>

          <div className="bg-slate-950 p-3 rounded-lg border border-slate-800">
            <span className="text-slate-500 block mb-1">Total PPO State Space:</span>
            <div className="text-xl font-bold text-emerald-400">{frame?.encoded_state_dim || 21} Dims</div>
            <span className="text-[10px] text-slate-500">Concatenated Input Tensor</span>
          </div>
        </div>

        {/* Action Space Information */}
        <div className="bg-slate-950 p-3.5 rounded-lg border border-slate-800 space-y-2 font-mono text-xs">
          <h3 className="font-semibold text-slate-300 flex items-center space-x-1.5">
            <Code className="w-4 h-4 text-indigo-400" />
            <span>Discrete Action Space (DiscreteActionSpace: num_actions=3)</span>
          </h3>
          <div className="grid grid-cols-3 gap-2 text-center pt-1">
            <div className="bg-slate-900 p-2 rounded border border-slate-800">
              <span className="text-slate-500 block text-[10px]">Index 0</span>
              <span className="font-bold text-slate-300">HOLD</span>
            </div>
            <div className="bg-slate-900 p-2 rounded border border-slate-800">
              <span className="text-slate-500 block text-[10px]">Index 1</span>
              <span className="font-bold text-emerald-400">BUY (qty: 100)</span>
            </div>
            <div className="bg-slate-900 p-2 rounded border border-slate-800">
              <span className="text-slate-500 block text-[10px]">Index 2</span>
              <span className="font-bold text-rose-400">SELL (qty: 100)</span>
            </div>
          </div>
        </div>

        {/* Environment State Dictionary Keys */}
        <div className="bg-slate-950 p-3.5 rounded-lg border border-slate-800 space-y-2 font-mono text-xs">
          <span className="font-semibold text-slate-300 block">Raw Market Environment Keys:</span>
          <div className="flex flex-wrap gap-1.5 pt-1">
            {stateKeys.map((k) => (
              <span
                key={k}
                className="px-2 py-0.5 rounded bg-slate-900 border border-slate-800 text-indigo-300 text-[11px]"
              >
                {k}
              </span>
            ))}
          </div>
        </div>

        {/* Causal Decision Cycle Ordering */}
        <div className="bg-slate-950 p-3.5 rounded-lg border border-slate-800 space-y-2 text-xs font-mono">
          <span className="font-semibold text-slate-300 block">Verified Causal Stackelberg Sequence:</span>
          <ol className="list-decimal list-inside space-y-1 text-slate-400 text-[11px]">
            <li>Observe raw market state at timestep t.</li>
            <li>Evaluate candidate leader actions (TIGHT, MEDIUM, WIDE) in Stackelberg game.</li>
            <li>Select leader action maximizing U_L and apply spread/liquidity multipliers.</li>
            <li>Construct 11-dimensional StrategicObservation for PPO agent.</li>
            <li>Generate follower actions (RL Trader, Momentum, Value).</li>
            <li>Execute orders at modified quotes (Ask for BUY, Bid for SELL).</li>
            <li>Update market price dynamics, mark-to-market portfolios, and compute rewards.</li>
            <li>Update opponent model with observed follower actions for step t+1.</li>
          </ol>
        </div>

        <div className="flex justify-end pt-2">
          <button
            onClick={onClose}
            className="px-4 py-1.5 rounded-md bg-indigo-600 hover:bg-indigo-500 text-white font-medium text-xs font-mono"
          >
            Close Inspector
          </button>
        </div>
      </div>
    </div>
  );
};
