import React, { useState, useEffect } from 'react';
import {
  ResponsiveContainer,
  LineChart,
  Line,
  XAxis,
  YAxis,
  Tooltip,
  CartesianGrid,
} from 'recharts';
import { Play, Square, Cpu, HardDrive, AlertCircle, RefreshCw } from 'lucide-react';
import { api } from '../api/client';
import { TrainingStatus, CheckpointInfo } from '../types/simulation';

export const TrainingPage: React.FC = () => {
  const [status, setStatus] = useState<TrainingStatus | null>(null);
  const [checkpoints, setCheckpoints] = useState<CheckpointInfo[]>([]);
  const [timesteps, setTimesteps] = useState<number>(1000);
  const [variant, setVariant] = useState<string>('proposed_full');
  const [seed, setSeed] = useState<number>(42);
  const [isLoading, setIsLoading] = useState<boolean>(false);

  const fetchTrainingData = async () => {
    try {
      const [st, ckpts] = await Promise.all([
        api.getTrainingStatus(),
        api.getCheckpoints(),
      ]);
      setStatus(st);
      setCheckpoints(ckpts);
    } catch (e) {
      console.error('Failed to fetch training telemetry:', e);
    }
  };

  useEffect(() => {
    fetchTrainingData();
    const interval = setInterval(() => {
      fetchTrainingData();
    }, 1500);
    return () => clearInterval(interval);
  }, []);

  const handleStart = async () => {
    setIsLoading(true);
    try {
      await api.startTraining({ timesteps, variant, seed });
      await fetchTrainingData();
    } finally {
      setIsLoading(false);
    }
  };

  const handleStop = async () => {
    setIsLoading(true);
    try {
      await api.stopTraining();
      await fetchTrainingData();
    } finally {
      setIsLoading(false);
    }
  };

  const isTraining = status?.is_training ?? false;
  const currentStep = status?.current_step ?? 0;
  const totalSteps = status?.total_timesteps ?? 1000;
  const progressPct = Math.min(100, Math.round((currentStep / Math.max(1, totalSteps)) * 100));
  const history = status?.history ?? [];

  return (
    <div className="space-y-4">
      {/* Top Banner and Controls */}
      <div className="bg-slate-900 border border-slate-800 rounded-lg p-5">
        <div className="flex flex-wrap items-center justify-between gap-4">
          <div className="flex items-center space-x-3">
            <div className="w-10 h-10 rounded-lg bg-indigo-600/20 text-indigo-400 flex items-center justify-center border border-indigo-500/40">
              <Cpu className="w-5 h-5 animate-pulse" />
            </div>
            <div>
              <h2 className="text-base font-bold text-slate-100">PPO Reinforcement Learning Training Monitor</h2>
              <p className="text-xs text-slate-400 font-mono">
                Generalized Advantage Estimation (GAE) & Clipped Surrogate Optimization
              </p>
            </div>
          </div>

          {/* Action Buttons */}
          <div className="flex items-center space-x-3 font-mono text-xs">
            {!isTraining ? (
              <div className="flex items-center space-x-2">
                <select
                  value={variant}
                  onChange={(e) => setVariant(e.target.value)}
                  aria-label="Training Variant"
                  className="bg-slate-950 border border-slate-800 rounded px-2.5 py-1.5 text-indigo-300"
                >
                  <option value="proposed_full">Proposed: Stackelberg + Opponent + PPO</option>
                  <option value="stackelberg_only">Stackelberg Leader Only</option>
                  <option value="opponent_model_only">Opponent Modeling Only</option>
                  <option value="rl_only">RL-Only Baseline</option>
                </select>

                <select
                  value={timesteps}
                  onChange={(e) => setTimesteps(Number(e.target.value))}
                  aria-label="Training Timesteps"
                  className="bg-slate-950 border border-slate-800 rounded px-2.5 py-1.5 text-slate-200"
                >
                  <option value={500}>500 Steps</option>
                  <option value={1000}>1,000 Steps</option>
                  <option value={2000}>2,000 Steps</option>
                  <option value={5000}>5,000 Steps</option>
                </select>

                <button
                  onClick={handleStart}
                  disabled={isLoading}
                  className="flex items-center space-x-1.5 px-4 py-1.5 rounded-md bg-indigo-600 hover:bg-indigo-500 text-white font-bold shadow transition-all"
                >
                  <Play className="w-3.5 h-3.5 fill-current" />
                  <span>Launch Training</span>
                </button>
              </div>
            ) : (
              <div className="flex items-center space-x-3">
                <div className="flex items-center space-x-2 px-3 py-1 rounded bg-amber-500/10 border border-amber-500/30 text-amber-400">
                  <span className="w-2 h-2 rounded-full bg-amber-400 animate-ping" />
                  <span>Optimizing Network Weights...</span>
                </div>
                <button
                  onClick={handleStop}
                  disabled={isLoading}
                  className="flex items-center space-x-1 px-3 py-1.5 rounded-md bg-rose-600 hover:bg-rose-500 text-white font-bold transition-all"
                >
                  <Square className="w-3.5 h-3.5 fill-current" />
                  <span>Interrupt</span>
                </button>
              </div>
            )}
          </div>
        </div>

        {/* Training Progress Bar */}
        <div className="mt-4 pt-3 border-t border-slate-800/80 font-mono text-xs">
          <div className="flex justify-between text-slate-400 mb-1.5">
            <span>
              Budget Progress: <strong className="text-slate-200">{currentStep}</strong> / {totalSteps} Environment Timesteps
            </span>
            <span className="text-indigo-400 font-bold">{progressPct}%</span>
          </div>
          <div className="w-full bg-slate-950 h-2.5 rounded-full overflow-hidden border border-slate-800">
            <div
              className="bg-indigo-500 h-full transition-all duration-300"
              style={{ width: `${progressPct}%` }}
            />
          </div>
        </div>
      </div>

      {/* Real-Time Telemetry Stats */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-3 font-mono text-xs">
        <div className="bg-slate-900 border border-slate-800 rounded-lg p-3">
          <span className="text-slate-500 block mb-1">Surrogate Policy Loss:</span>
          <div className="text-lg font-bold text-slate-100">
            {status?.metrics?.policy_loss !== undefined ? status.metrics.policy_loss.toFixed(4) : '--'}
          </div>
          <span className="text-[11px] text-slate-500">Clipped objective</span>
        </div>

        <div className="bg-slate-900 border border-slate-800 rounded-lg p-3">
          <span className="text-slate-500 block mb-1">Critic Value Loss:</span>
          <div className="text-lg font-bold text-slate-100 truncate">
            {status?.metrics?.value_loss !== undefined ? status.metrics.value_loss.toFixed(2) : '--'}
          </div>
          <span className="text-[11px] text-slate-500">GAE return MSE</span>
        </div>

        <div className="bg-slate-900 border border-slate-800 rounded-lg p-3">
          <span className="text-slate-500 block mb-1">Policy Entropy:</span>
          <div className="text-lg font-bold text-indigo-400">
            {status?.metrics?.entropy !== undefined ? status.metrics.entropy.toFixed(4) : '--'}
          </div>
          <span className="text-[11px] text-slate-500">Exploration bonus</span>
        </div>

        <div className="bg-slate-900 border border-slate-800 rounded-lg p-3">
          <span className="text-slate-500 block mb-1">Approx. KL Divergence:</span>
          <div className="text-lg font-bold text-emerald-400">
            {status?.metrics?.approx_kl !== undefined ? status.metrics.approx_kl.toFixed(5) : '--'}
          </div>
          <span className="text-[11px] text-slate-500">Trust region step</span>
        </div>
      </div>

      {/* Loss & Reward Curves */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        {/* Policy & Value Loss Chart */}
        <div className="bg-slate-900 border border-slate-800 rounded-lg p-4">
          <div className="flex items-center justify-between mb-3">
            <div>
              <h3 className="text-sm font-semibold text-slate-200">Optimization Loss Convergence</h3>
              <p className="text-xs text-slate-500 font-mono">Policy Loss (L_clip) & Value Loss (MSE)</p>
            </div>
          </div>

          <div className="h-56 w-full">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={history} margin={{ top: 5, right: 10, left: -20, bottom: 0 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#1f2937" vertical={false} />
                <XAxis dataKey="step" stroke="#4b5563" fontSize={11} tickLine={false} />
                <YAxis stroke="#4b5563" fontSize={11} tickLine={false} domain={['auto', 'auto']} />
                <Tooltip
                  contentStyle={{ backgroundColor: '#0f172a', borderColor: '#334155', borderRadius: '6px', fontSize: '12px' }}
                />
                <Line
                  type="monotone"
                  dataKey="policy_loss"
                  name="Policy Loss"
                  stroke="#818cf8"
                  strokeWidth={2}
                  dot={false}
                  isAnimationActive={false}
                />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Episode Reward Curve */}
        <div className="bg-slate-900 border border-slate-800 rounded-lg p-4">
          <div className="flex items-center justify-between mb-3">
            <div>
              <h3 className="text-sm font-semibold text-slate-200">Episode Returns & PnL Reward Over Time</h3>
              <p className="text-xs text-slate-500 font-mono">Gross valuation return minus risk and trading friction</p>
            </div>
          </div>

          <div className="h-56 w-full">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={history} margin={{ top: 5, right: 10, left: -20, bottom: 0 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#1f2937" vertical={false} />
                <XAxis dataKey="step" stroke="#4b5563" fontSize={11} tickLine={false} />
                <YAxis stroke="#4b5563" fontSize={11} tickLine={false} domain={['auto', 'auto']} />
                <Tooltip
                  contentStyle={{ backgroundColor: '#0f172a', borderColor: '#334155', borderRadius: '6px', fontSize: '12px' }}
                />
                <Line
                  type="monotone"
                  dataKey="reward"
                  name="Reward"
                  stroke="#10b981"
                  strokeWidth={2}
                  dot={false}
                  isAnimationActive={false}
                />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>

      {/* Model Checkpoint Browser */}
      <div className="bg-slate-900 border border-slate-800 rounded-lg p-4">
        <div className="flex items-center justify-between mb-3">
          <div className="flex items-center space-x-2">
            <HardDrive className="w-4 h-4 text-indigo-400" />
            <h3 className="text-sm font-semibold text-slate-200">Serialized Policy Checkpoints</h3>
            <span className="text-xs font-mono text-slate-500">({checkpoints.length} weights files found)</span>
          </div>
          <button
            onClick={fetchTrainingData}
            title="Refresh checkpoints"
            className="p-1 rounded bg-slate-800 hover:bg-slate-700 text-slate-300 border border-slate-700"
          >
            <RefreshCw className="w-3 h-3" />
          </button>
        </div>

        <div className="overflow-x-auto">
          <table className="w-full text-left text-xs font-mono">
            <thead className="bg-slate-950 text-slate-400 border-b border-slate-800">
              <tr>
                <th className="py-2 px-3">File Name</th>
                <th className="py-2 px-3">Relative Path</th>
                <th className="py-2 px-3">File Size</th>
                <th className="py-2 px-3">Last Modified</th>
                <th className="py-2 px-3">Status</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-800/60">
              {checkpoints.length > 0 ? (
                checkpoints.map((ck, i) => (
                  <tr key={i} className="hover:bg-slate-800/40">
                    <td className="py-2 px-3 font-semibold text-indigo-300">{ck.filename}</td>
                    <td className="py-2 px-3 text-slate-400">{ck.path}</td>
                    <td className="py-2 px-3 text-slate-300">{ck.size_kb} KB</td>
                    <td className="py-2 px-3 text-slate-400">{ck.modified}</td>
                    <td className="py-2 px-3 text-emerald-400 font-bold">Validated</td>
                  </tr>
                ))
              ) : (
                <tr>
                  <td colSpan={5} className="py-4 text-center text-slate-500">
                    No checkpoints serialized yet in checkpoints/ directory.
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
