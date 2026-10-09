import React, { useState } from 'react';
import { Navbar } from './components/Navbar';
import { StatusBar } from './components/StatusBar';
import { DashboardPage } from './pages/DashboardPage';
import { TrainingPage } from './pages/TrainingPage';
import { ExperimentsPage } from './pages/ExperimentsPage';
import { DiagnosticsModal } from './components/DiagnosticsModal';
import { useSimulationSocket } from './api/useSimulationSocket';

export default function App() {
  const [activeTab, setActiveTab] = useState<'dashboard' | 'training' | 'experiments'>('dashboard');
  const [isDiagnosticsOpen, setIsDiagnosticsOpen] = useState<boolean>(false);

  const {
    isConnected,
    frame,
    history,
    events,
    refreshEvents,
    refreshState,
  } = useSimulationSocket();

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100 flex flex-col font-sans selection:bg-indigo-500 selection:text-white">
      {/* 1. Research Terminal Header */}
      <Navbar
        activeTab={activeTab}
        setActiveTab={setActiveTab}
        isConnected={isConnected}
        activeVariant={frame?.active_variant || 'proposed_full'}
      />

      {/* 2. Interactive Simulation Controller Bar */}
      <StatusBar
        frame={frame}
        onRefresh={() => {
          refreshState();
          refreshEvents();
        }}
        onOpenDiagnostics={() => setIsDiagnosticsOpen(true)}
      />

      {/* 3. Main Workspace Area */}
      <main className="flex-1 p-5 max-w-[1700px] w-full mx-auto">
        {activeTab === 'dashboard' && (
          <DashboardPage
            frame={frame}
            history={history}
            events={events}
            onRefreshEvents={refreshEvents}
          />
        )}

        {activeTab === 'training' && <TrainingPage />}

        {activeTab === 'experiments' && <ExperimentsPage />}
      </main>

      {/* 4. Architecture & State Inspector Modal */}
      <DiagnosticsModal
        isOpen={isDiagnosticsOpen}
        onClose={() => setIsDiagnosticsOpen(false)}
        frame={frame}
      />

      {/* Footer */}
      <footer className="border-t border-slate-800/80 bg-slate-950 px-6 py-3 text-xs font-mono text-slate-500 flex flex-wrap justify-between items-center gap-2">
        <div>
          Strategic Trading MARL Research Project — Members 1 (Environment), 2 (Game Theory), 3 (PPO RL)
        </div>
        <div className="flex items-center space-x-3">
          <span>Engine: Python 3.11</span>
          <span>•</span>
          <span>Causal Execution: Verified</span>
          <span>•</span>
          <span className="text-emerald-400">WebSocket Connected</span>
        </div>
      </footer>
    </div>
  );
}
