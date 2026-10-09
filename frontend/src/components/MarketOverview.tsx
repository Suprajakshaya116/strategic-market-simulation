import React from 'react';
import {
  ResponsiveContainer,
  LineChart,
  Line,
  XAxis,
  YAxis,
  Tooltip,
  CartesianGrid,
  ReferenceLine,
  AreaChart,
  Area,
} from 'recharts';
import { TrendingUp, Activity, DollarSign, Layers, ArrowUpRight, ArrowDownRight } from 'lucide-react';
import { MarketState, HistoryPoint } from '../types/simulation';

interface MarketOverviewProps {
  marketState: MarketState | undefined;
  history: HistoryPoint[];
  recentOrders: Array<any>;
}

export const MarketOverview: React.FC<MarketOverviewProps> = ({
  marketState,
  history,
  recentOrders,
}) => {
  const price = marketState?.price ?? 100.0;
  const fundamental = marketState?.fundamental_value ?? 100.0;
  const spread = marketState?.spread ?? 0.02;
  const liquidity = marketState?.liquidity ?? 100.0;
  const volatility = marketState?.volatility ?? 0.01;
  const imbalance = marketState?.order_imbalance ?? 0.0;
  const bid = marketState?.bid_price ?? 99.99;
  const ask = marketState?.ask_price ?? 100.01;
  const returnRate = marketState?.return ?? 0.0;

  const mispricing = ((price - fundamental) / (fundamental + 1e-8)) * 100;

  return (
    <div className="space-y-4">
      {/* Top Real-Time Stat Cards */}
      <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-3">
        {/* Mid Price & Quotes */}
        <div className="bg-slate-900 border border-slate-800 rounded-lg p-3">
          <div className="flex items-center justify-between text-slate-400 text-xs mb-1">
            <span className="flex items-center space-x-1">
              <DollarSign className="w-3.5 h-3.5 text-indigo-400" />
              <span>Mid Price</span>
            </span>
            <span className={`flex items-center text-[11px] font-mono ${returnRate >= 0 ? 'text-emerald-400' : 'text-rose-400'}`}>
              {returnRate >= 0 ? <ArrowUpRight className="w-3 h-3" /> : <ArrowDownRight className="w-3 h-3" />}
              {(returnRate * 100).toFixed(2)}%
            </span>
          </div>
          <div className="text-xl font-bold font-mono text-slate-100">${price.toFixed(2)}</div>
          <div className="text-[11px] font-mono text-slate-500 mt-1 flex justify-between">
            <span>Bid: <span className="text-emerald-400">${bid.toFixed(2)}</span></span>
            <span>Ask: <span className="text-rose-400">${ask.toFixed(2)}</span></span>
          </div>
        </div>

        {/* Fundamental Value & Mispricing */}
        <div className="bg-slate-900 border border-slate-800 rounded-lg p-3">
          <div className="flex items-center justify-between text-slate-400 text-xs mb-1">
            <span className="flex items-center space-x-1">
              <TrendingUp className="w-3.5 h-3.5 text-blue-400" />
              <span>Fundamental</span>
            </span>
            <span className={`text-[11px] font-mono ${mispricing >= 0 ? 'text-amber-400' : 'text-cyan-400'}`}>
              {mispricing > 0 ? `+${mispricing.toFixed(2)}%` : `${mispricing.toFixed(2)}%`}
            </span>
          </div>
          <div className="text-xl font-bold font-mono text-slate-100">${fundamental.toFixed(2)}</div>
          <div className="text-[11px] font-mono text-slate-500 mt-1">
            Reversion Anchor
          </div>
        </div>

        {/* Bid-Ask Spread */}
        <div className="bg-slate-900 border border-slate-800 rounded-lg p-3">
          <div className="flex items-center justify-between text-slate-400 text-xs mb-1">
            <span className="flex items-center space-x-1">
              <Layers className="w-3.5 h-3.5 text-amber-400" />
              <span>Effective Spread</span>
            </span>
            <span className="text-[11px] font-mono text-slate-400">
              {((spread / price) * 10000).toFixed(0)} bps
            </span>
          </div>
          <div className="text-xl font-bold font-mono text-slate-100">{spread.toFixed(4)}</div>
          <div className="text-[11px] font-mono text-slate-500 mt-1">
            Stackelberg quote friction
          </div>
        </div>

        {/* Depth & Liquidity */}
        <div className="bg-slate-900 border border-slate-800 rounded-lg p-3">
          <div className="flex items-center justify-between text-slate-400 text-xs mb-1">
            <span className="flex items-center space-x-1">
              <Activity className="w-3.5 h-3.5 text-emerald-400" />
              <span>Pool Liquidity</span>
            </span>
            <span className="text-[11px] font-mono text-emerald-400 font-medium">Active</span>
          </div>
          <div className="text-xl font-bold font-mono text-slate-100">{liquidity.toFixed(1)}</div>
          <div className="text-[11px] font-mono text-slate-500 mt-1">
            Market impact buffer
          </div>
        </div>

        {/* Order Imbalance */}
        <div className="bg-slate-900 border border-slate-800 rounded-lg p-3">
          <div className="flex items-center justify-between text-slate-400 text-xs mb-1">
            <span>Order Imbalance</span>
            <span className={`text-[11px] font-mono font-medium ${imbalance > 0 ? 'text-emerald-400' : imbalance < 0 ? 'text-rose-400' : 'text-slate-400'}`}>
              {imbalance > 0 ? 'BUY HEAVY' : imbalance < 0 ? 'SELL HEAVY' : 'BALANCED'}
            </span>
          </div>
          <div className="text-xl font-bold font-mono text-slate-100">
            {imbalance > 0 ? `+${imbalance.toFixed(3)}` : imbalance.toFixed(3)}
          </div>
          <div className="w-full bg-slate-800 h-1.5 rounded-full mt-2 overflow-hidden flex">
            <div
              className="bg-rose-500 h-full transition-all"
              style={{ width: `${Math.max(0, -imbalance) * 50}%` }}
            />
            <div className="w-0.5 bg-slate-600 h-full" />
            <div
              className="bg-emerald-500 h-full transition-all"
              style={{ width: `${Math.max(0, imbalance) * 50}%` }}
            />
          </div>
        </div>

        {/* Volatility */}
        <div className="bg-slate-900 border border-slate-800 rounded-lg p-3">
          <div className="flex items-center justify-between text-slate-400 text-xs mb-1">
            <span>Inst. Volatility</span>
            <span className="text-[11px] font-mono text-slate-400">Sigma</span>
          </div>
          <div className="text-xl font-bold font-mono text-slate-100">{(volatility * 100).toFixed(2)}%</div>
          <div className="text-[11px] font-mono text-slate-500 mt-1">
            Imbalance sensitivity
          </div>
        </div>
      </div>

      {/* Main Charts Row */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
        {/* Price & Fundamental Overlay Chart */}
        <div className="lg:col-span-2 bg-slate-900 border border-slate-800 rounded-lg p-4">
          <div className="flex items-center justify-between mb-3">
            <div>
              <h2 className="text-sm font-semibold text-slate-200">Price Dynamics & Fundamental Trajectory</h2>
              <p className="text-xs text-slate-500 font-mono">Simulated Mid-Price vs Reference Fundamental Value</p>
            </div>
            <div className="flex items-center space-x-3 text-xs font-mono">
              <div className="flex items-center space-x-1">
                <span className="w-3 h-0.5 bg-indigo-400 rounded-full" />
                <span className="text-slate-300">Market Price</span>
              </div>
              <div className="flex items-center space-x-1">
                <span className="w-3 h-0.5 bg-cyan-400 stroke-dashed rounded-full" />
                <span className="text-slate-400">Fundamental</span>
              </div>
            </div>
          </div>

          <div className="h-64 w-full">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={history} margin={{ top: 5, right: 10, left: -20, bottom: 0 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#1f2937" vertical={false} />
                <XAxis dataKey="step" stroke="#4b5563" fontSize={11} tickLine={false} />
                <YAxis
                  stroke="#4b5563"
                  fontSize={11}
                  domain={['auto', 'auto']}
                  tickLine={false}
                  tickFormatter={(val) => `$${val.toFixed(1)}`}
                />
                <Tooltip
                  contentStyle={{ backgroundColor: '#0f172a', borderColor: '#334155', borderRadius: '6px', fontSize: '12px' }}
                  labelStyle={{ color: '#94a3b8' }}
                />
                <Line
                  type="monotone"
                  dataKey="price"
                  name="Price"
                  stroke="#818cf8"
                  strokeWidth={2}
                  dot={false}
                  isAnimationActive={false}
                />
                <Line
                  type="monotone"
                  dataKey="fundamental_value"
                  name="Fundamental"
                  stroke="#22d3ee"
                  strokeWidth={1.5}
                  strokeDasharray="4 4"
                  dot={false}
                  isAnimationActive={false}
                />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Spread & Liquidity Chart */}
        <div className="bg-slate-900 border border-slate-800 rounded-lg p-4">
          <div className="flex items-center justify-between mb-3">
            <div>
              <h2 className="text-sm font-semibold text-slate-200">Spread & Liquidity Depth</h2>
              <p className="text-xs text-slate-500 font-mono">Leader Spread Multipliers & Pool Liquidity</p>
            </div>
            <div className="flex items-center space-x-2 text-xs font-mono">
              <span className="text-amber-400">Spread</span> / <span className="text-emerald-400">Liquidity</span>
            </div>
          </div>

          <div className="h-64 w-full">
            <ResponsiveContainer width="100%" height="100%">
              <AreaChart data={history} margin={{ top: 5, right: 10, left: -20, bottom: 0 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#1f2937" vertical={false} />
                <XAxis dataKey="step" stroke="#4b5563" fontSize={11} tickLine={false} />
                <YAxis yAxisId="left" stroke="#f59e0b" fontSize={11} tickLine={false} domain={['auto', 'auto']} />
                <YAxis yAxisId="right" orientation="right" stroke="#10b981" fontSize={11} tickLine={false} domain={['auto', 'auto']} />
                <Tooltip
                  contentStyle={{ backgroundColor: '#0f172a', borderColor: '#334155', borderRadius: '6px', fontSize: '12px' }}
                />
                <Area
                  yAxisId="left"
                  type="monotone"
                  dataKey="spread"
                  name="Spread"
                  stroke="#f59e0b"
                  fill="#f59e0b"
                  fillOpacity={0.15}
                  isAnimationActive={false}
                />
                <Line
                  yAxisId="right"
                  type="monotone"
                  dataKey="liquidity"
                  name="Liquidity"
                  stroke="#10b981"
                  strokeWidth={1.5}
                  dot={false}
                  isAnimationActive={false}
                />
              </AreaChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>

      {/* Order Imbalance & Recent Orders Table Row */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
        {/* Order Imbalance Over Time */}
        <div className="bg-slate-900 border border-slate-800 rounded-lg p-4">
          <div className="flex items-center justify-between mb-3">
            <div>
              <h2 className="text-sm font-semibold text-slate-200">Order Flow Imbalance</h2>
              <p className="text-xs text-slate-500 font-mono">Normalized Flow Imbalance: (Buy - Sell) / Vol</p>
            </div>
          </div>

          <div className="h-44 w-full">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={history} margin={{ top: 5, right: 10, left: -20, bottom: 0 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#1f2937" vertical={false} />
                <XAxis dataKey="step" stroke="#4b5563" fontSize={11} tickLine={false} />
                <YAxis domain={[-1, 1]} stroke="#4b5563" fontSize={11} tickLine={false} />
                <ReferenceLine y={0} stroke="#64748b" strokeDasharray="2 2" />
                <Tooltip
                  contentStyle={{ backgroundColor: '#0f172a', borderColor: '#334155', borderRadius: '6px', fontSize: '12px' }}
                />
                <Line
                  type="monotone"
                  dataKey="order_imbalance"
                  name="Imbalance"
                  stroke="#ec4899"
                  strokeWidth={1.5}
                  dot={false}
                  isAnimationActive={false}
                />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Live Execution Table */}
        <div className="lg:col-span-2 bg-slate-900 border border-slate-800 rounded-lg p-4">
          <div className="flex items-center justify-between mb-3">
            <div>
              <h2 className="text-sm font-semibold text-slate-200">Recent Order Clearing & Execution Fills</h2>
              <p className="text-xs text-slate-500 font-mono">ExecutionEngine Quote Allocations (Ask for BUY, Bid for SELL)</p>
            </div>
            <span className="text-xs font-mono text-slate-400">Step {marketState ? history[history.length - 1]?.step || 0 : 0}</span>
          </div>

          <div className="overflow-x-auto">
            <table className="w-full text-left text-xs font-mono">
              <thead className="bg-slate-950 text-slate-400 border-b border-slate-800">
                <tr>
                  <th className="py-2 px-3">Participant</th>
                  <th className="py-2 px-3">Action</th>
                  <th className="py-2 px-3">Status</th>
                  <th className="py-2 px-3">Qty</th>
                  <th className="py-2 px-3">Fill Quote</th>
                  <th className="py-2 px-3">Transaction Cost</th>
                  <th className="py-2 px-3">Detail</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-800/60">
                {recentOrders && recentOrders.length > 0 ? (
                  recentOrders.map((ord, idx) => (
                    <tr key={idx} className="hover:bg-slate-800/40">
                      <td className="py-2 px-3 font-semibold text-slate-200">{ord.agent_id}</td>
                      <td className="py-2 px-3">
                        <span className={`px-1.5 py-0.5 rounded text-[11px] font-bold ${
                          ord.action === 'BUY' ? 'bg-emerald-500/10 text-emerald-400 border border-emerald-500/30' :
                          ord.action === 'SELL' ? 'bg-rose-500/10 text-rose-400 border border-rose-500/30' :
                          'bg-slate-800 text-slate-400'
                        }`}>
                          {ord.action}
                        </span>
                      </td>
                      <td className="py-2 px-3">
                        <span className={`text-[11px] font-semibold ${
                          ord.status === 'EXECUTED' ? 'text-emerald-400' :
                          ord.status === 'REJECTED' ? 'text-rose-400' :
                          'text-slate-400'
                        }`}>
                          {ord.status}
                        </span>
                      </td>
                      <td className="py-2 px-3 text-slate-300">{ord.quantity}</td>
                      <td className="py-2 px-3 text-slate-200">${Number(ord.fill_price || price).toFixed(2)}</td>
                      <td className="py-2 px-3 text-slate-400">${Number(ord.transaction_cost || 0).toFixed(4)}</td>
                      <td className="py-2 px-3 text-slate-500 text-[11px]">
                        {ord.reason ? <span className="text-rose-400">{ord.reason}</span> : ord.status === 'EXECUTED' ? 'Filled at quote' : 'No trade'}
                      </td>
                    </tr>
                  ))
                ) : (
                  <tr>
                    <td colSpan={7} className="py-4 text-center text-slate-500">
                      No order executions recorded in current cycle.
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
