import React, { useState, useEffect } from 'react';
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
import { Download, Database, CheckCircle2, AlertTriangle, TrendingUp, ShieldCheck } from 'lucide-react';
import { api } from '../api/client';
import { ExperimentRecord } from '../types/simulation';

export const ExperimentsPage: React.FC = () => {
  const [results, setResults] = useState<ExperimentRecord[]>([]);
  const [summary, setSummary] = useState<any[]>([]);
  const [selectedMetric, setSelectedMetric] = useState<string>('cumulative_return');

  useEffect(() => {
    Promise.all([api.getExperimentResults(), api.getExperimentSummary()])
      .then(([res, summ]) => {
        setResults(res);
        setSummary(summ);
      })
      .catch((e) => console.error('Failed to load experiment records:', e));
  }, []);

  // Compute aggregate stats per variant if summary is loaded or calculate dynamically
  const variants = ['rl_only', 'opponent_model_only', 'stackelberg_only', 'proposed_full'];

  const variantStats = variants.map((v) => {
    const matches = results.filter((r) => r.variant === v);
    if (matches.length === 0) return { variant: v, count: 0, mean_return: 0, mean_dd: 0, mean_spread: 0, mean_trades: 0 };
    const meanReturn = matches.reduce((acc, m) => acc + m.cumulative_return, 0) / matches.length;
    const meanDd = matches.reduce((acc, m) => acc + m.max_drawdown, 0) / matches.length;
    const meanSpread = matches.reduce((acc, m) => acc + m.mean_spread, 0) / matches.length;
    const meanTrades = matches.reduce((acc, m) => acc + m.mean_trade_count, 0) / matches.length;
    return {
      variant: v,
      count: matches.length,
      mean_return: Number((meanReturn * 100).toFixed(2)),
      mean_dd: Number((meanDd * 100).toFixed(2)),
      mean_spread: Number(meanSpread.toFixed(2)),
      mean_trades: Number(meanTrades.toFixed(1)),
    };
  });

  const exportCSV = () => {
    if (results.length === 0) return;
    const headers = Object.keys(results[0]).join(',');
    const rows = results.map((r) => Object.values(r).join(',')).join('\n');
    const blob = new Blob([`${headers}\n${rows}`], { type: 'text/csv' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'controlled_experiments_export.csv';
    a.click();
    URL.revokeObjectURL(url);
  };

  return (
    <div className="space-y-4">
      {/* Header and Download Button */}
      <div className="bg-slate-900 border border-slate-800 rounded-lg p-5 flex flex-wrap items-center justify-between gap-4">
        <div className="flex items-center space-x-3">
          <div className="w-10 h-10 rounded-lg bg-indigo-600/20 text-indigo-400 flex items-center justify-center border border-indigo-500/40">
            <Database className="w-5 h-5" />
          </div>
          <div>
            <h2 className="text-base font-bold text-slate-100">Multi-Seed Controlled Experiment Benchmark</h2>
            <p className="text-xs text-slate-400 font-mono">
              Fair Empirical Comparison across 4 Research Variants (Seeds 42, 100, 2026)
            </p>
          </div>
        </div>

        <button
          onClick={exportCSV}
          disabled={results.length === 0}
          className="flex items-center space-x-1.5 px-4 py-2 rounded-md bg-slate-800 hover:bg-slate-700 text-slate-200 text-xs font-mono font-medium border border-slate-700 transition-all disabled:opacity-40"
        >
          <Download className="w-4 h-4" />
          <span>Export Results CSV</span>
        </button>
      </div>

      {/* Aggregate Scientific Findings Summary */}
      <div className="bg-slate-900 border border-slate-800 rounded-lg p-4">
        <h3 className="text-sm font-semibold text-slate-200 mb-3">Controlled Variant Comparison (Aggregate Averages)</h3>
        <div className="grid grid-cols-1 md:grid-cols-4 gap-3">
          {variantStats.map((st) => (
            <div
              key={st.variant}
              className={`rounded-lg p-3.5 border ${
                st.variant === 'proposed_full'
                  ? 'bg-indigo-950/20 border-indigo-500/60'
                  : 'bg-slate-950 border-slate-800'
              }`}
            >
              <div className="flex items-center justify-between mb-2">
                <span className="text-xs font-bold font-mono text-slate-200 uppercase">
                  {st.variant.replace('_', ' ')}
                </span>
                {st.variant === 'proposed_full' && (
                  <span className="text-[10px] px-1.5 py-0.2 rounded bg-indigo-500/20 text-indigo-300 font-bold border border-indigo-500/40">
                    Proposed
                  </span>
                )}
              </div>

              <div className="space-y-1 font-mono text-xs">
                <div className="flex justify-between">
                  <span className="text-slate-500">Cumulative Return:</span>
                  <span className={`font-bold ${st.mean_return >= 0 ? 'text-emerald-400' : 'text-rose-400'}`}>
                    {st.mean_return}%
                  </span>
                </div>
                <div className="flex justify-between">
                  <span className="text-slate-500">Max Drawdown:</span>
                  <span className="text-rose-400">{st.mean_dd}%</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-slate-500">Market Spread:</span>
                  <span className={`font-bold ${st.mean_spread > 500 ? 'text-rose-400' : 'text-emerald-400'}`}>
                    {st.mean_spread}
                  </span>
                </div>
                <div className="flex justify-between">
                  <span className="text-slate-500">Executed Trades:</span>
                  <span className="text-slate-300">{st.mean_trades}</span>
                </div>
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Comparison Bar Charts */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        {/* Cumulative Return by Variant */}
        <div className="bg-slate-900 border border-slate-800 rounded-lg p-4">
          <h3 className="text-xs font-semibold text-slate-300 mb-2">Mean Cumulative Return (%) Across Seeds</h3>
          <div className="h-56 w-full">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={variantStats} margin={{ top: 5, right: 10, left: -20, bottom: 0 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#1f2937" vertical={false} />
                <XAxis dataKey="variant" stroke="#6b7280" fontSize={10} tickLine={false} tickFormatter={(v) => v.replace('_', ' ')} />
                <YAxis stroke="#6b7280" fontSize={11} tickLine={false} />
                <Tooltip
                  contentStyle={{ backgroundColor: '#0f172a', borderColor: '#334155', borderRadius: '6px', fontSize: '12px' }}
                />
                <Bar dataKey="mean_return" name="Return (%)" radius={[4, 4, 0, 0]}>
                  {variantStats.map((entry, index) => (
                    <Cell
                      key={`cell-${index}`}
                      fill={entry.mean_return >= -1 ? '#10b981' : '#f43f5e'}
                    />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Market Spread Regulation Bar Chart */}
        <div className="bg-slate-900 border border-slate-800 rounded-lg p-4">
          <h3 className="text-xs font-semibold text-slate-300 mb-2">Effective Market Spread (Quote Friction)</h3>
          <div className="h-56 w-full">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={variantStats} margin={{ top: 5, right: 10, left: -10, bottom: 0 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#1f2937" vertical={false} />
                <XAxis dataKey="variant" stroke="#6b7280" fontSize={10} tickLine={false} tickFormatter={(v) => v.replace('_', ' ')} />
                <YAxis stroke="#6b7280" fontSize={11} tickLine={false} />
                <Tooltip
                  contentStyle={{ backgroundColor: '#0f172a', borderColor: '#334155', borderRadius: '6px', fontSize: '12px' }}
                />
                <Bar dataKey="mean_spread" name="Mean Spread" fill="#818cf8" radius={[4, 4, 0, 0]}>
                  {variantStats.map((entry, index) => (
                    <Cell
                      key={`cell-${index}`}
                      fill={entry.mean_spread > 500 ? '#f43f5e' : '#10b981'}
                    />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>

      {/* Exhaustive Seed Records Table */}
      <div className="bg-slate-900 border border-slate-800 rounded-lg p-4">
        <h3 className="text-sm font-semibold text-slate-200 mb-3">Individual Matched Runs (Seeds 42, 100, 2026)</h3>
        <div className="overflow-x-auto">
          <table className="w-full text-left text-xs font-mono">
            <thead className="bg-slate-950 text-slate-400 border-b border-slate-800">
              <tr>
                <th className="py-2 px-3">Variant</th>
                <th className="py-2 px-3">Seed</th>
                <th className="py-2 px-3">Steps</th>
                <th className="py-2 px-3">Return (%)</th>
                <th className="py-2 px-3">Max DD (%)</th>
                <th className="py-2 px-3">Sharpe</th>
                <th className="py-2 px-3">Trades</th>
                <th className="py-2 px-3">Mean Spread</th>
                <th className="py-2 px-3">Win Rate</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-800/60">
              {results.length > 0 ? (
                results.map((r, i) => (
                  <tr key={i} className="hover:bg-slate-800/40">
                    <td className="py-2 px-3 font-semibold text-indigo-300">{r.variant}</td>
                    <td className="py-2 px-3 text-slate-400">{r.seed}</td>
                    <td className="py-2 px-3 text-slate-400">{r.total_timesteps}</td>
                    <td className={`py-2 px-3 font-bold ${r.cumulative_return >= 0 ? 'text-emerald-400' : 'text-rose-400'}`}>
                      {(r.cumulative_return * 100).toFixed(2)}%
                    </td>
                    <td className="py-2 px-3 text-rose-400">{(r.max_drawdown * 100).toFixed(2)}%</td>
                    <td className="py-2 px-3 text-slate-300">{r.sharpe_ratio.toFixed(2)}</td>
                    <td className="py-2 px-3 text-slate-200">{r.mean_trade_count}</td>
                    <td className="py-2 px-3 text-slate-300">{r.mean_spread.toFixed(2)}</td>
                    <td className="py-2 px-3 text-slate-300">{(r.win_rate * 100).toFixed(0)}%</td>
                  </tr>
                ))
              ) : (
                <tr>
                  <td colSpan={9} className="py-4 text-center text-slate-500">
                    No experiment records found in experiments/results/. Run controlled experiments to populate.
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
};
