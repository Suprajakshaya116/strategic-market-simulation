import React from 'react';
import { MarketOverview } from '../components/MarketOverview';
import { MultiAgentPanel } from '../components/MultiAgentPanel';
import { StackelbergVisualizer } from '../components/StackelbergVisualizer';
import { OpponentModelPanel } from '../components/OpponentModelPanel';
import { PortfolioRiskPanel } from '../components/PortfolioRiskPanel';
import { EventTimeline } from '../components/EventTimeline';
import { SimulationFrame, HistoryPoint, EventLog } from '../types/simulation';

interface DashboardPageProps {
  frame: SimulationFrame | null;
  history: HistoryPoint[];
  events: EventLog[];
  onRefreshEvents: () => void;
}

export const DashboardPage: React.FC<DashboardPageProps> = ({
  frame,
  history,
  events,
  onRefreshEvents,
}) => {
  return (
    <div className="space-y-4">
      {/* 1. Live Market Overview & Charts */}
      <MarketOverview
        marketState={frame?.market_state}
        history={history}
        recentOrders={frame?.recent_orders || []}
      />

      {/* 2. Multi-Agent Participant Cards */}
      <MultiAgentPanel
        agents={frame?.agents}
        leaderDecision={frame?.leader_decision}
      />

      {/* 3. Core Game Theory: Stackelberg & Opponent Modeling */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        <StackelbergVisualizer
          decision={frame?.leader_decision}
          history={history}
        />
        <OpponentModelPanel
          beliefs={frame?.opponent_beliefs}
          agents={frame?.agents}
        />
      </div>

      {/* 4. Portfolio Risk & Live Event Telemetry */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        <PortfolioRiskPanel
          agents={frame?.agents}
          history={history}
        />
        <EventTimeline
          events={events}
          onRefresh={onRefreshEvents}
        />
      </div>
    </div>
  );
};
