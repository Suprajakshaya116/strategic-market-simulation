import React from 'react';
import { Activity, Cpu, Database, BarChart3, Radio } from 'lucide-react';

interface NavbarProps {
  activeTab: 'dashboard' | 'training' | 'experiments';
  setActiveTab: (tab: 'dashboard' | 'training' | 'experiments') => void;
  isConnected: boolean;
  activeVariant: string;
}

export const Navbar: React.FC<NavbarProps> = ({
  activeTab,
  setActiveTab,
  isConnected,
  activeVariant,
}) => {
  return (
    <header className="bg-slate-900/90 backdrop-blur border-b border-slate-800 sticky top-0 z-50 px-6 py-3 flex items-center justify-between">
      {/* Brand & Title */}
      <div className="flex items-center space-x-3">
        <div className="w-9 h-9 rounded-lg bg-indigo-600/20 border border-indigo-500/40 flex items-center justify-center text-indigo-400 font-bold">
          <Activity className="w-5 h-5 text-indigo-400 animate-pulse" />
        </div>
        <div>
          <div className="flex items-center space-x-2">
            <h1 className="text-lg font-bold tracking-tight text-slate-100 uppercase">
              Strategic Trading Lab
            </h1>
            <span className="text-xs px-2 py-0.5 rounded-full bg-slate-800 text-slate-400 border border-slate-700 font-mono">
              MARL + Stackelberg
            </span>
          </div>
          <p className="text-xs text-slate-400 font-mono">
            Game-Theoretic Multi-Agent Reinforcement Learning Research Terminal
          </p>
        </div>
      </div>

      {/* Navigation Tabs */}
      <nav className="flex items-center space-x-1 bg-slate-950 p-1 rounded-lg border border-slate-800">
        <button
          onClick={() => setActiveTab('dashboard')}
          className={`flex items-center space-x-2 px-3.5 py-1.5 rounded-md text-xs font-medium transition-all ${
            activeTab === 'dashboard'
              ? 'bg-indigo-600 text-white shadow-sm'
              : 'text-slate-400 hover:text-slate-200 hover:bg-slate-900'
          }`}
        >
          <BarChart3 className="w-3.5 h-3.5" />
          <span>Live Market Terminal</span>
        </button>

        <button
          onClick={() => setActiveTab('training')}
          className={`flex items-center space-x-2 px-3.5 py-1.5 rounded-md text-xs font-medium transition-all ${
            activeTab === 'training'
              ? 'bg-indigo-600 text-white shadow-sm'
              : 'text-slate-400 hover:text-slate-200 hover:bg-slate-900'
          }`}
        >
          <Cpu className="w-3.5 h-3.5" />
          <span>RL Training Monitor</span>
        </button>

        <button
          onClick={() => setActiveTab('experiments')}
          className={`flex items-center space-x-2 px-3.5 py-1.5 rounded-md text-xs font-medium transition-all ${
            activeTab === 'experiments'
              ? 'bg-indigo-600 text-white shadow-sm'
              : 'text-slate-400 hover:text-slate-200 hover:bg-slate-900'
          }`}
        >
          <Database className="w-3.5 h-3.5" />
          <span>Experiment Benchmarks</span>
        </button>
      </nav>

      {/* Connection & Configuration Status */}
      <div className="flex items-center space-x-3">
        <div className="flex items-center space-x-1.5 px-2.5 py-1 rounded-md bg-slate-950 border border-slate-800 text-xs font-mono">
          <span className="text-slate-500">Model:</span>
          <span className="text-indigo-400 font-semibold uppercase">{activeVariant.replace('_', ' ')}</span>
        </div>

        <div className="flex items-center space-x-2 px-3 py-1 rounded-full bg-slate-950 border border-slate-800 text-xs font-mono">
          <Radio
            className={`w-3.5 h-3.5 ${
              isConnected ? 'text-emerald-400 animate-pulse' : 'text-rose-500'
            }`}
          />
          <span className={isConnected ? 'text-emerald-400' : 'text-rose-500'}>
            {isConnected ? 'LIVE WS' : 'DISCONNECTED'}
          </span>
        </div>
      </div>
    </header>
  );
};
