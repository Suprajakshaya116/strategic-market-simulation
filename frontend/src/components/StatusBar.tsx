import React, { useState } from 'react';
import {
  Play,
  Pause,
  Square,
  StepForward,
  RotateCcw,
  Sliders,
  CheckCircle2,
  AlertCircle,
  Clock,
} from 'lucide-react';
import { api } from '../api/client';
import { SimulationFrame } from '../types/simulation';

interface StatusBarProps {
  frame: SimulationFrame | null;
  onRefresh: () => void;
  onOpenDiagnostics: () => void;
}

export const StatusBar: React.FC<StatusBarProps> = ({ frame, onRefresh, onOpenDiagnostics }) => {
  const [speed, setSpeed] = useState<number>(300);
  const [isActionPending, setIsActionPending] = useState<boolean>(false);

  const status = frame?.status || 'IDLE';
  const step = frame?.step || 0;
  const episode = frame?.episode || 1;
  const variant = frame?.active_variant || 'proposed_full';

  const handleStart = async () => {
    setIsActionPending(true);
    try {
      await api.start();
      onRefresh();
    } finally {
      setIsActionPending(false);
    }
  };

  const handlePause = async () => {
    setIsActionPending(true);
    try {
      await api.pause();
      onRefresh();
    } finally {
      setIsActionPending(false);
    }
  };

  const handleResume = async () => {
    setIsActionPending(true);
    try {
      await api.resume();
      onRefresh();
    } finally {
      setIsActionPending(false);
    }
  };

  const handleStop = async () => {
    setIsActionPending(true);
    try {
      await api.stop();
      onRefresh();
    } finally {
      setIsActionPending(false);
    }
  };

  const handleStep = async () => {
    setIsActionPending(true);
    try {
      await api.step();
      onRefresh();
    } finally {
      setIsActionPending(false);
    }
  };

  const handleReset = async (newVariant?: string) => {
    setIsActionPending(true);
    try {
      await api.reset({ variant: newVariant || variant, seed: 42 });
      onRefresh();
    } finally {
      setIsActionPending(false);
    }
  };

  const handleSpeedChange = async (newSpeed: number) => {
    setSpeed(newSpeed);
    await api.setSpeed(newSpeed);
  };

  const getStatusBadge = () => {
    switch (status) {
      case 'RUNNING':
        return (
          <span className="flex items-center space-x-1.5 px-2.5 py-1 rounded bg-emerald-500/10 border border-emerald-500/30 text-emerald-400 text-xs font-mono font-medium">
            <span className="w-2 h-2 rounded-full bg-emerald-400 animate-ping" />
            <span>RUNNING</span>
          </span>
        );
      case 'PAUSED':
        return (
          <span className="flex items-center space-x-1.5 px-2.5 py-1 rounded bg-amber-500/10 border border-amber-500/30 text-amber-400 text-xs font-mono font-medium">
            <Clock className="w-3.5 h-3.5" />
            <span>PAUSED</span>
          </span>
        );
      case 'COMPLETED':
        return (
          <span className="flex items-center space-x-1.5 px-2.5 py-1 rounded bg-blue-500/10 border border-blue-500/30 text-blue-400 text-xs font-mono font-medium">
            <CheckCircle2 className="w-3.5 h-3.5" />
            <span>COMPLETED</span>
          </span>
        );
      case 'FAILED':
        return (
          <span className="flex items-center space-x-1.5 px-2.5 py-1 rounded bg-rose-500/10 border border-rose-500/30 text-rose-400 text-xs font-mono font-medium">
            <AlertCircle className="w-3.5 h-3.5" />
            <span>FAILED</span>
          </span>
        );
      default:
        return (
          <span className="flex items-center space-x-1.5 px-2.5 py-1 rounded bg-slate-800 border border-slate-700 text-slate-400 text-xs font-mono font-medium">
            <span className="w-2 h-2 rounded-full bg-slate-500" />
            <span>IDLE</span>
          </span>
        );
    }
  };

  return (
    <div className="bg-slate-900 border-b border-slate-800 px-6 py-2.5 flex flex-wrap items-center justify-between gap-4">
      {/* Simulation Telemetry */}
      <div className="flex items-center space-x-5 font-mono text-xs">
        <div className="flex items-center space-x-2">
          <span className="text-slate-500 uppercase tracking-wider text-[11px]">Status:</span>
          {getStatusBadge()}
        </div>

        <div className="flex items-center space-x-1.5">
          <span className="text-slate-500">Step:</span>
          <span className="text-slate-200 font-semibold px-1.5 py-0.5 rounded bg-slate-950 border border-slate-800">
            {step}
          </span>
        </div>

        <div className="flex items-center space-x-1.5">
          <span className="text-slate-500">Episode:</span>
          <span className="text-slate-200 font-semibold px-1.5 py-0.5 rounded bg-slate-950 border border-slate-800">
            {episode}
          </span>
        </div>

        <div className="flex items-center space-x-1.5">
          <span className="text-slate-500">Variant:</span>
          <select
            value={variant}
            onChange={(e) => handleReset(e.target.value)}
            disabled={status === 'RUNNING' || isActionPending}
            aria-label="Experimental Variant"
            className="bg-slate-950 border border-slate-800 text-indigo-300 rounded px-2 py-1 text-xs focus:outline-none focus:border-indigo-500"
          >
            <option value="proposed_full">Proposed: Stackelberg + Opponent + PPO</option>
            <option value="stackelberg_only">Stackelberg Leader Only</option>
            <option value="opponent_model_only">Opponent Modeling Only</option>
            <option value="rl_only">RL-Only Baseline</option>
          </select>
        </div>
      </div>

      {/* Primary Simulation Controls */}
      <div className="flex items-center space-x-2">
        {status === 'IDLE' && (
          <button
            onClick={handleStart}
            disabled={isActionPending}
            className="flex items-center space-x-1 px-3 py-1.5 rounded-md bg-emerald-600 hover:bg-emerald-500 text-white text-xs font-semibold shadow-sm transition-all disabled:opacity-50"
          >
            <Play className="w-3.5 h-3.5 fill-current" />
            <span>Start</span>
          </button>
        )}

        {status === 'RUNNING' && (
          <button
            onClick={handlePause}
            disabled={isActionPending}
            className="flex items-center space-x-1 px-3 py-1.5 rounded-md bg-amber-600 hover:bg-amber-500 text-white text-xs font-semibold shadow-sm transition-all disabled:opacity-50"
          >
            <Pause className="w-3.5 h-3.5 fill-current" />
            <span>Pause</span>
          </button>
        )}

        {status === 'PAUSED' && (
          <button
            onClick={handleResume}
            disabled={isActionPending}
            className="flex items-center space-x-1 px-3 py-1.5 rounded-md bg-emerald-600 hover:bg-emerald-500 text-white text-xs font-semibold shadow-sm transition-all disabled:opacity-50"
          >
            <Play className="w-3.5 h-3.5 fill-current" />
            <span>Resume</span>
          </button>
        )}

        <button
          onClick={handleStep}
          disabled={status === 'RUNNING' || isActionPending}
          className="flex items-center space-x-1 px-2.5 py-1.5 rounded-md bg-slate-800 hover:bg-slate-700 text-slate-200 text-xs font-medium border border-slate-700 transition-all disabled:opacity-40"
        >
          <StepForward className="w-3.5 h-3.5" />
          <span>Step</span>
        </button>

        <button
          onClick={handleStop}
          disabled={status === 'IDLE' || isActionPending}
          className="flex items-center space-x-1 px-2.5 py-1.5 rounded-md bg-slate-800 hover:bg-rose-950 hover:text-rose-300 text-slate-300 text-xs font-medium border border-slate-700 transition-all disabled:opacity-40"
        >
          <Square className="w-3.5 h-3.5 fill-current" />
          <span>Stop</span>
        </button>

        <button
          onClick={() => handleReset()}
          disabled={status === 'RUNNING' || isActionPending}
          className="flex items-center space-x-1 px-2.5 py-1.5 rounded-md bg-slate-800 hover:bg-slate-700 text-slate-300 text-xs font-medium border border-slate-700 transition-all disabled:opacity-40"
        >
          <RotateCcw className="w-3.5 h-3.5" />
          <span>Reset</span>
        </button>

        <div className="h-4 w-px bg-slate-800 mx-1" />

        {/* Speed Selector */}
        <div className="flex items-center space-x-1 bg-slate-950 p-0.5 rounded border border-slate-800 text-[11px] font-mono">
          {[
            { label: '0.5x', ms: 600 },
            { label: '1x', ms: 300 },
            { label: '2x', ms: 150 },
            { label: '5x', ms: 50 },
          ].map((item) => (
            <button
              key={item.label}
              onClick={() => handleSpeedChange(item.ms)}
              className={`px-1.5 py-0.5 rounded transition-all ${
                speed === item.ms
                  ? 'bg-indigo-600 text-white font-bold'
                  : 'text-slate-400 hover:text-slate-200'
              }`}
            >
              {item.label}
            </button>
          ))}
        </div>

        <button
          onClick={onOpenDiagnostics}
          title="Open State & Architecture Inspector"
          className="p-1.5 rounded bg-slate-800 hover:bg-slate-700 text-slate-400 hover:text-indigo-300 border border-slate-700 transition-all"
        >
          <Sliders className="w-3.5 h-3.5" />
        </button>
      </div>
    </div>
  );
};
