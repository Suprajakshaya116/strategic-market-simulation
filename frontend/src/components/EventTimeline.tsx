import React, { useState } from 'react';
import { ListFilter, Clock, CheckCircle2, AlertTriangle, Shield, Cpu, RefreshCw } from 'lucide-react';
import { EventLog } from '../types/simulation';

interface EventTimelineProps {
  events: EventLog[];
  onRefresh: () => void;
}

export const EventTimeline: React.FC<EventTimelineProps> = ({ events, onRefresh }) => {
  const [filter, setFilter] = useState<string>('ALL');
  const [selectedEvent, setSelectedEvent] = useState<EventLog | null>(null);

  const categories = ['ALL', 'STACKELBERG', 'ORDER', 'OPPONENT', 'PPO', 'MARKET'];

  const filteredEvents = events.filter((e) => {
    if (filter === 'ALL') return true;
    return e.category === filter;
  });

  const getCategoryBadge = (cat: string) => {
    switch (cat) {
      case 'STACKELBERG':
        return 'bg-amber-500/20 text-amber-300 border-amber-500/40';
      case 'ORDER':
        return 'bg-emerald-500/20 text-emerald-300 border-emerald-500/40';
      case 'OPPONENT':
        return 'bg-cyan-500/20 text-cyan-300 border-cyan-500/40';
      case 'PPO':
        return 'bg-indigo-500/20 text-indigo-300 border-indigo-500/40';
      case 'MARKET':
      default:
        return 'bg-slate-800 text-slate-300 border-slate-700';
    }
  };

  return (
    <div className="bg-slate-900 border border-slate-800 rounded-lg p-4 space-y-3">
      {/* Header and Filter bar */}
      <div className="flex flex-wrap items-center justify-between gap-2">
        <div className="flex items-center space-x-2">
          <Clock className="w-4 h-4 text-indigo-400" />
          <h2 className="text-sm font-semibold text-slate-200">Live Simulation Event Log & Telemetry</h2>
          <span className="text-xs font-mono text-slate-500">({events.length} events logged)</span>
        </div>

        <div className="flex items-center space-x-1.5 font-mono text-xs">
          <ListFilter className="w-3.5 h-3.5 text-slate-400" />
          <div className="flex items-center space-x-1 bg-slate-950 p-0.5 rounded border border-slate-800">
            {categories.map((c) => (
              <button
                key={c}
                onClick={() => setFilter(c)}
                className={`px-2 py-0.5 rounded text-[11px] transition-all ${
                  filter === c
                    ? 'bg-indigo-600 text-white font-bold'
                    : 'text-slate-400 hover:text-slate-200'
                }`}
              >
                {c}
              </button>
            ))}
          </div>

          <button
            onClick={onRefresh}
            title="Refresh logs"
            className="p-1 rounded bg-slate-800 hover:bg-slate-700 text-slate-300 border border-slate-700"
          >
            <RefreshCw className="w-3 h-3" />
          </button>
        </div>
      </div>

      {/* Event Stream Container */}
      <div className="bg-slate-950 border border-slate-800/80 rounded-lg p-2 h-56 overflow-y-auto font-mono text-xs space-y-1.5">
        {filteredEvents.length > 0 ? (
          filteredEvents.slice().reverse().map((ev) => (
            <div
              key={ev.id}
              onClick={() => setSelectedEvent(ev)}
              className="px-2.5 py-1.5 rounded hover:bg-slate-900/80 border border-transparent hover:border-slate-800 transition-all cursor-pointer flex items-center justify-between text-slate-300"
            >
              <div className="flex items-center space-x-2.5 truncate">
                <span className="text-[11px] text-slate-500">{ev.timestamp}</span>
                <span className="text-[11px] text-slate-400 bg-slate-900 px-1 rounded border border-slate-800">
                  s.{ev.step}
                </span>
                <span className={`text-[10px] px-1.5 py-0.2 rounded font-bold border ${getCategoryBadge(ev.category)}`}>
                  {ev.category}
                </span>
                <span className="truncate text-slate-300">{ev.message}</span>
              </div>
              {ev.details && Object.keys(ev.details).length > 0 && (
                <span className="text-[10px] text-indigo-400 shrink-0 ml-2">[details]</span>
              )}
            </div>
          ))
        ) : (
          <div className="h-full flex items-center justify-center text-slate-500">
            No events matching filter {filter}.
          </div>
        )}
      </div>

      {/* Selected Event Payload Inspector Modal */}
      {selectedEvent && (
        <div className="bg-slate-950 border border-slate-800 rounded-lg p-3 text-xs font-mono text-slate-300">
          <div className="flex items-center justify-between mb-1 text-slate-400 border-b border-slate-800 pb-1">
            <span>Event Payload Inspector: {selectedEvent.category}</span>
            <button
              onClick={() => setSelectedEvent(null)}
              className="text-slate-500 hover:text-slate-300 font-bold"
            >
              ✕ Close
            </button>
          </div>
          <p className="text-slate-200 mb-2">{selectedEvent.message}</p>
          <pre className="bg-slate-900 p-2 rounded text-[11px] overflow-x-auto text-emerald-400">
            {JSON.stringify(selectedEvent.details, null, 2)}
          </pre>
        </div>
      )}
    </div>
  );
};
