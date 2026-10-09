import { SimulationFrame, HistoryPoint, EventLog, TrainingStatus, CheckpointInfo, ExperimentRecord } from '../types/simulation';

const API_BASE = '/api';

export const api = {
  // Simulation Endpoints
  async getState(): Promise<SimulationFrame> {
    const res = await fetch(`${API_BASE}/simulation/state`);
    if (!res.ok) throw new Error('Failed to fetch simulation state');
    return res.json();
  },

  async getHistory(): Promise<HistoryPoint[]> {
    const res = await fetch(`${API_BASE}/simulation/history`);
    if (!res.ok) throw new Error('Failed to fetch history');
    return res.json();
  },

  async getEvents(): Promise<EventLog[]> {
    const res = await fetch(`${API_BASE}/simulation/events`);
    if (!res.ok) throw new Error('Failed to fetch events');
    return res.json();
  },

  async start(): Promise<any> {
    const res = await fetch(`${API_BASE}/simulation/start`, { method: 'POST' });
    return res.json();
  },

  async pause(): Promise<any> {
    const res = await fetch(`${API_BASE}/simulation/pause`, { method: 'POST' });
    return res.json();
  },

  async resume(): Promise<any> {
    const res = await fetch(`${API_BASE}/simulation/resume`, { method: 'POST' });
    return res.json();
  },

  async stop(): Promise<any> {
    const res = await fetch(`${API_BASE}/simulation/stop`, { method: 'POST' });
    return res.json();
  },

  async step(): Promise<SimulationFrame> {
    const res = await fetch(`${API_BASE}/simulation/step`, { method: 'POST' });
    return res.json();
  },

  async setSpeed(delay_ms: number): Promise<any> {
    const res = await fetch(`${API_BASE}/simulation/speed`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ delay_ms }),
    });
    return res.json();
  },

  async reset(config?: any): Promise<any> {
    const res = await fetch(`${API_BASE}/simulation/reset`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(config || {}),
    });
    return res.json();
  },

  // Training Endpoints
  async startTraining(payload: { timesteps: number; variant: string; seed: number }): Promise<any> {
    const res = await fetch(`${API_BASE}/training/start`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });
    return res.json();
  },

  async getTrainingStatus(): Promise<TrainingStatus> {
    const res = await fetch(`${API_BASE}/training/status`);
    return res.json();
  },

  async stopTraining(): Promise<any> {
    const res = await fetch(`${API_BASE}/training/stop`, { method: 'POST' });
    return res.json();
  },

  async getCheckpoints(): Promise<CheckpointInfo[]> {
    const res = await fetch(`${API_BASE}/training/checkpoints`);
    return res.json();
  },

  // Experiment Comparison Endpoints
  async getExperimentResults(): Promise<ExperimentRecord[]> {
    const res = await fetch(`${API_BASE}/experiments/results`);
    return res.json();
  },

  async getExperimentSummary(): Promise<any[]> {
    const res = await fetch(`${API_BASE}/experiments/summary`);
    return res.json();
  },
};
