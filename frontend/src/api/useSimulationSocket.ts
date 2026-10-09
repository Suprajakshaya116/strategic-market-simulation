import { useEffect, useRef, useState, useCallback } from 'react';
import { SimulationFrame, HistoryPoint, EventLog } from '../types/simulation';
import { api } from './client';

export function useSimulationSocket() {
  const [isConnected, setIsConnected] = useState<boolean>(false);
  const [frame, setFrame] = useState<SimulationFrame | null>(null);
  const [history, setHistory] = useState<HistoryPoint[]>([]);
  const [events, setEvents] = useState<EventLog[]>([]);
  const socketRef = useRef<WebSocket | null>(null);
  const reconnectTimeoutRef = useRef<any>(null);

  const fetchInitialData = useCallback(async () => {
    try {
      const [initialState, initialHistory, initialEvents] = await Promise.all([
        api.getState(),
        api.getHistory(),
        api.getEvents(),
      ]);
      setFrame(initialState);
      setHistory(initialHistory);
      setEvents(initialEvents);
    } catch (e) {
      console.warn('Initial data fetch waiting for backend...', e);
    }
  }, []);

  useEffect(() => {
    fetchInitialData();
  }, [fetchInitialData]);

  const connect = useCallback(() => {
    if (socketRef.current && (socketRef.current.readyState === WebSocket.OPEN || socketRef.current.readyState === WebSocket.CONNECTING)) {
      return;
    }

    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    // Use window.location.host so Vite proxy passes /ws through to FastAPI backend
    const wsUrl = `${protocol}//${window.location.host}/ws/simulation`;

    const ws = new WebSocket(wsUrl);
    socketRef.current = ws;

    ws.onopen = () => {
      setIsConnected(true);
      console.log('[WebSocket] Connected to simulation server');
    };

    ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        if (data.step !== undefined && data.market_state) {
          const newFrame = data as SimulationFrame;
          setFrame(newFrame);

          // Append to local history for smooth chart updates
          setHistory((prev) => {
            const point: HistoryPoint = {
              step: newFrame.step,
              price: newFrame.market_state.price,
              fundamental_value: newFrame.market_state.fundamental_value,
              spread: newFrame.market_state.spread,
              liquidity: newFrame.market_state.liquidity,
              return: newFrame.market_state.return,
              volatility: newFrame.market_state.volatility,
              order_imbalance: newFrame.market_state.order_imbalance,
              leader_action: newFrame.leader_decision.selected_action,
              leader_utility: newFrame.leader_decision.leader_utility,
              rl_pnl: newFrame.agents.rl_trader?.cumulative_pnl || 0,
              rl_portfolio_value: newFrame.agents.rl_trader?.portfolio_value || 100000,
              rl_reward: newFrame.agents.rl_trader?.reward || 0,
            };
            const updated = [...prev, point];
            return updated.length > 200 ? updated.slice(-200) : updated;
          });

          // Fetch fresh events
          api.getEvents().then(setEvents).catch(() => {});
        }
      } catch (err) {
        console.error('[WebSocket] Parse error:', err);
      }
    };

    ws.onclose = () => {
      setIsConnected(false);
      socketRef.current = null;
      reconnectTimeoutRef.current = setTimeout(() => {
        connect();
      }, 2000);
    };

    ws.onerror = (e) => {
      console.warn('[WebSocket] Connection error:', e);
      ws.close();
    };
  }, []);

  useEffect(() => {
    connect();
    return () => {
      if (reconnectTimeoutRef.current) clearTimeout(reconnectTimeoutRef.current);
      if (socketRef.current) socketRef.current.close();
    };
  }, [connect]);

  return {
    isConnected,
    frame,
    history,
    events,
    refreshEvents: () => api.getEvents().then(setEvents),
    refreshHistory: () => api.getHistory().then(setHistory),
    refreshState: () => api.getState().then(setFrame),
  };
}
