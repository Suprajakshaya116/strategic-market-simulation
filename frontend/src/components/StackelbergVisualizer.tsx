import React from 'react';
import {
  ResponsiveContainer,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  CartesianGrid,
  Cell,
} from 'recharts';
import { ShieldCheck, Info, Check, ArrowRight } from 'lucide-react';
import { LeaderDecision, HistoryPoint } from '../types/simulation';

interface StackelbergVisualizerProps {
  decision: LeaderDecision | undefined;
  history: HistoryPoint[];
}

export const StackelbergVisualizer: React.FC<StackelbergVisualizerProps> = ({ decision, history }) => {
  const candidates = decision?.candidate_evaluations || [];
  const selectedAction = decision?.selected_action || 'MEDIUM';

  const chartData = candidates.map((c) => ({
    name: c.leader_action,
    utility: Number(c.leader_utility.toFixed(3)),
    buy_pressure: Number(c.expected_buy_pressure.toFixed(1)),
    sell_pressure: Number(c.expected_sell_pressure.toFixed(1)),
    imbalance: Number(c.expected_order_imbalance.toFixed(2)),
  }));

  const recentDecisions = history.slice(-6).reverse();

  return (
    <div className="bg-slate-900 border border-slate-800 rounded-lg p-4 space-y-4">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div className="flex items-center space-x-2.5">
          <div className="w-8 h-8 rounded bg-amber-500/20 text-amber-400 flex items-center justify-center">
            <ShieldCheck className="w-4 h-4" />
          </div>
          <div>
            <h2 className="text-sm font-semibold text-slate-200">Stackelberg Game Theory Strategic Layer</h2>
            <p className="text-xs text-slate-500 font-mono">
              Leader Optimization over Discrete Controls: a_L* = argmax U_L(a_L, B(a_L))
            </p>
          </div>
        </div>
        <div className="flex items-center space-x-2">
          <span className="text-xs text-slate-400 font-mono">Selected Leader Action:</span>
          <span className="text-xs px-2.5 py-1 rounded font-bold font-mono bg-amber-500/20 text-amber-300 border border-amber-500/40">
            {selectedAction}
          </span>
        </div>
      </div>

      {/* Dynamic Data-Grounded Decision Explanation Banner */}
      <div className="bg-slate-950 border border-slate-800 rounded-lg p-3 flex items-start space-x-3">
        <Info className="w-4 h-4 text-amber-400 shrink-0 mt-0.5" />
        <div className="text-xs font-mono text-slate-300 leading-relaxed">
          <span className="text-amber-400 font-semibold">Strategic Solved Rationale: </span>
          {decision?.explanation || 'Awaiting first decision cycle...'}
        </div>
      </div>

      {/* Candidate Evaluations Grid */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
        {candidates.map((cand) => {
          const isSelected = cand.leader_action === selectedAction;
          return (
            <div
              key={cand.leader_action}
              className={`rounded-lg p-3.5 border transition-all ${
                isSelected
                  ? 'bg-amber-950/20 border-amber-500/60 shadow-lg shadow-amber-950/30'
                  : 'bg-slate-950 border-slate-800/80 opacity-80'
              }`}
            >
              <div className="flex items-center justify-between mb-2 pb-2 border-b border-slate-800">
                <div className="flex items-center space-x-2 font-mono">
                  <span className={`text-xs font-bold ${isSelected ? 'text-amber-300' : 'text-slate-300'}`}>
                    {cand.leader_action}
                  </span>
                  <span className="text-[10px] text-slate-500">
                    {cand.leader_action === 'TIGHT' ? '(0.7x sp / 1.3x liq)' :
                     cand.leader_action === 'WIDE' ? '(1.4x sp / 0.7x liq)' :
                     '(1.0x sp / 1.0x liq)'}
                  </span>
                </div>
                {isSelected && (
                  <span className="flex items-center space-x-0.5 text-[10px] font-mono text-amber-400 font-bold bg-amber-500/20 px-1.5 py-0.5 rounded border border-amber-500/40">
                    <Check className="w-3 h-3" />
                    <span>OPTIMAL</span>
                  </span>
                )}
              </div>

              <div className="space-y-1.5 text-xs font-mono">
                <div className="flex justify-between">
                  <span className="text-slate-500">Leader Utility:</span>
                  <span className={`font-bold ${isSelected ? 'text-amber-300' : 'text-slate-300'}`}>
                    {cand.leader_utility.toFixed(4)}
                  </span>
                </div>
                <div className="flex justify-between">
                  <span className="text-slate-500">Exp. Buy Pressure:</span>
                  <span className="text-emerald-400">{cand.expected_buy_pressure.toFixed(1)}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-slate-500">Exp. Sell Pressure:</span>
                  <span className="text-rose-400">{cand.expected_sell_pressure.toFixed(1)}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-slate-500">Exp. Imbalance:</span>
                  <span className="text-slate-300">{cand.expected_order_imbalance.toFixed(2)}</span>
                </div>

                <div className="pt-2 border-t border-slate-800/60 text-[11px]">
                  <span className="text-slate-500 block mb-1">Follower Best Responses:</span>
                  <div className="flex justify-between text-slate-400">
                    <span>Mom: <span className="text-slate-200">{cand.predicted_follower_actions.momentum || 'HOLD'}</span></span>
                    <span>Val: <span className="text-slate-200">{cand.predicted_follower_actions.value || 'HOLD'}</span></span>
                  </div>
                </div>
              </div>
            </div>
          );
        })}
      </div>

      {/* Leader Utilities Comparison Chart & Decision History */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4 pt-2">
        {/* Candidate Utility Comparison Bar Chart */}
        <div className="bg-slate-950 border border-slate-800 rounded-lg p-3.5">
          <h3 className="text-xs font-semibold text-slate-300 mb-2">Candidate Leader Utilities Comparison U_L(a_L)</h3>
          <div className="h-44 w-full">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={chartData} margin={{ top: 5, right: 10, left: -20, bottom: 0 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#1f2937" vertical={false} />
                <XAxis dataKey="name" stroke="#6b7280" fontSize={11} tickLine={false} />
                <YAxis stroke="#6b7280" fontSize={11} tickLine={false} domain={['auto', 'auto']} />
                <Tooltip
                  contentStyle={{ backgroundColor: '#0f172a', borderColor: '#334155', borderRadius: '6px', fontSize: '12px' }}
                />
                <Bar dataKey="utility" name="Leader Utility" radius={[4, 4, 0, 0]}>
                  {chartData.map((entry, index) => (
                    <Cell
                      key={`cell-${index}`}
                      fill={entry.name === selectedAction ? '#f59e0b' : '#334155'}
                    />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Stackelberg Decision History Table */}
        <div className="bg-slate-950 border border-slate-800 rounded-lg p-3.5 overflow-hidden">
          <h3 className="text-xs font-semibold text-slate-300 mb-2">Recent Leader Decisions vs Market Outcomes</h3>
          <div className="overflow-x-auto">
            <table className="w-full text-left text-xs font-mono">
              <thead className="bg-slate-900 text-slate-400 border-b border-slate-800">
                <tr>
                  <th className="py-1.5 px-2">Step</th>
                  <th className="py-1.5 px-2">Decision</th>
                  <th className="py-1.5 px-2">Utility</th>
                  <th className="py-1.5 px-2">Effective Spread</th>
                  <th className="py-1.5 px-2">Price Outcome</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-800/40">
                {recentDecisions.length > 0 ? (
                  recentDecisions.map((h, i) => (
                    <tr key={i} className="hover:bg-slate-900/40">
                      <td className="py-1 px-2 text-slate-400">{h.step}</td>
                      <td className="py-1 px-2">
                        <span className={`px-1.5 py-0.2 rounded text-[10px] font-bold ${
                          h.leader_action === 'TIGHT' ? 'bg-emerald-500/20 text-emerald-400' :
                          h.leader_action === 'WIDE' ? 'bg-rose-500/20 text-rose-400' :
                          'bg-blue-500/20 text-blue-400'
                        }`}>
                          {h.leader_action}
                        </span>
                      </td>
                      <td className="py-1 px-2 text-amber-400">{h.leader_utility.toFixed(3)}</td>
                      <td className="py-1 px-2 text-slate-300">{h.spread.toFixed(4)}</td>
                      <td className="py-1 px-2 text-slate-200">${h.price.toFixed(2)}</td>
                    </tr>
                  ))
                ) : (
                  <tr>
                    <td colSpan={5} className="py-3 text-center text-slate-500">
                      No historical steps yet.
                    </td>
                  </tr>
                )}
              </tbody>
            </table>
          </div>
        </div>
      </div>
    </div>
  );
};
