import asyncio
import time
import uuid
import datetime
from typing import Dict, Any, List, Optional, Set
from fastapi import WebSocket

from environment.config import MarketConfig
from environment.market_env import MarketEnvironment
from agents.momentum_agent import MomentumAgent
from agents.value_agent import ValueAgent
from agents.market_maker import MarketMaker
from game_theory import StrategicLayer, GameTheoryConfig, LeaderAction
from rl.config import PPOConfig, StateEncoderConfig
from rl.ppo_agent import PPOAgent


class SimulationService:
    """Live interactive driver connecting FastAPI/WebSocket to real market simulation modules."""

    def __init__(self):
        self.status = "IDLE"  # IDLE, RUNNING, PAUSED, COMPLETED, FAILED
        self.step_count = 0
        self.episode_count = 1
        self.delay_ms = 300
        self.active_variant = "proposed_full"
        self.seed = 42

        # WebSocket active clients
        self.active_connections: Set[WebSocket] = set()
        self._running_task: Optional[asyncio.Task] = None
        self._lock = asyncio.Lock()

        # Rolling history (max 300 points)
        self.history: List[Dict[str, Any]] = []
        self.events: List[Dict[str, Any]] = []
        self.max_history = 300
        self.max_events = 200

        # Subsystems
        self.env: Optional[MarketEnvironment] = None
        self.market_maker: Optional[MarketMaker] = None
        self.momentum_agent: Optional[MomentumAgent] = None
        self.value_agent: Optional[ValueAgent] = None
        self.rl_agent: Optional[PPOAgent] = None

        self._initialize_simulation()

    def _initialize_simulation(
        self,
        episode_length: int = 100,
        initial_price: float = 100.0,
        initial_spread: float = 0.02,
        initial_liquidity: float = 100.0,
        variant: str = "proposed_full",
        seed: int = 42
    ):
        self.seed = seed
        self.active_variant = variant
        self.step_count = 0

        market_config = MarketConfig(
            episode_length=episode_length,
            initial_price=initial_price,
            initial_spread=initial_spread,
            initial_liquidity=initial_liquidity,
            seed=seed
        )
        self.env = MarketEnvironment(config=market_config)

        # Strategic Layer Configuration based on variant
        gt_config = GameTheoryConfig(seed=seed)
        use_stackelberg = variant in ("stackelberg_only", "proposed_full")
        use_opponent = variant in ("opponent_model_only", "proposed_full")

        self.market_maker = MarketMaker(config=gt_config)
        self.market_maker.strategic_layer.use_stackelberg = use_stackelberg
        self.market_maker.strategic_layer.use_opponent_model = use_opponent

        self.momentum_agent = MomentumAgent()
        self.value_agent = ValueAgent()

        # RL Agent Configuration
        strat_dim = 11 if variant != "rl_only" else 0
        ppo_config = PPOConfig(
            rollout_length=64,
            batch_size=16,
            ppo_epochs=2,
            lr=3e-4,
            device="cpu",
            seed=seed
        )
        encoder_config = StateEncoderConfig(
            initial_price=market_config.initial_price,
            initial_cash=market_config.initial_cash,
            max_order_size=market_config.max_order_size
        )
        self.rl_agent = PPOAgent(
            config=ppo_config,
            encoder_config=encoder_config,
            strategic_dim=strat_dim
        )

        self.env.reset(seed=seed)
        self.history.clear()
        self.events.clear()
        self._record_event(
            "MARKET",
            f"Initialized simulation with variant: {variant}, price: {initial_price:.2f}, spread: {initial_spread:.4f}"
        )

    def _record_event(self, category: str, message: str, details: Optional[Dict[str, Any]] = None):
        ev = {
            "id": str(uuid.uuid4())[:8],
            "timestamp": datetime.datetime.now().strftime("%H:%M:%S.%f")[:-3],
            "step": self.step_count,
            "category": category,
            "message": message,
            "details": details or {}
        }
        self.events.append(ev)
        if len(self.events) > self.max_events:
            self.events.pop(0)

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.add(websocket)
        # Send current frame immediately upon connection
        frame = self.get_current_frame()
        await websocket.send_json(frame)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.discard(websocket)

    async def broadcast_frame(self, frame: Dict[str, Any]):
        disconnected = set()
        for conn in self.active_connections:
            try:
                await conn.send_json(frame)
            except Exception:
                disconnected.add(conn)
        for conn in disconnected:
            self.disconnect(conn)

    def get_current_frame(self) -> Dict[str, Any]:
        market_state = self.env.get_market_state()
        leader_decision = self.market_maker.act(market_state)

        # Candidate evaluations schema
        cands = []
        for c in leader_decision.candidate_evaluations:
            cands.append({
                "leader_action": c.leader_action.value,
                "predicted_follower_actions": c.predicted_follower_actions,
                "predicted_follower_utilities": c.predicted_follower_utilities,
                "expected_buy_pressure": float(c.expected_buy_pressure),
                "expected_sell_pressure": float(c.expected_sell_pressure),
                "expected_order_imbalance": float(c.expected_order_imbalance),
                "leader_utility": float(c.leader_utility)
            })

        explanation = (
            f"The {leader_decision.selected_action.value} market control was selected "
            f"because it achieved the optimal leader utility ({leader_decision.leader_utility:.3f}) "
            f"under expected order imbalance ({cands[0]['expected_order_imbalance']:.2f})."
        )

        strat_obs = self.market_maker.get_strategic_observation(market_state, leader_decision)

        # Multi-agent states
        rl_port = self.env.portfolios.get("rl_trader")
        mom_port = self.env.portfolios.get("momentum")
        val_port = self.env.portfolios.get("value")

        rl_initial = self.env.config.initial_cash
        rl_val = rl_port.portfolio_value if rl_port else rl_initial

        agents_data = {
            "rl_trader": {
                "agent_id": "rl_trader",
                "action": "HOLD",
                "action_status": "HOLD",
                "quantity": 0,
                "fill_price": market_state["price"],
                "cash": rl_port.cash if rl_port else 0.0,
                "inventory": rl_port.inventory if rl_port else 0,
                "portfolio_value": rl_val,
                "reward": 0.0,
                "cumulative_pnl": (rl_val - rl_initial) / (rl_initial + 1e-8),
                "transaction_cost": rl_port.total_transaction_cost if rl_port else 0.0,
                "market_impact_cost": rl_port.total_market_impact_cost if rl_port else 0.0,
                "signal_metric": None,
                "signal_label": "PPO Policy"
            },
            "momentum": {
                "agent_id": "momentum",
                "action": "HOLD",
                "action_status": "HOLD",
                "quantity": 0,
                "fill_price": market_state["price"],
                "cash": mom_port.cash if mom_port else 0.0,
                "inventory": mom_port.inventory if mom_port else 0,
                "portfolio_value": mom_port.portfolio_value if mom_port else 0.0,
                "reward": 0.0,
                "cumulative_pnl": (mom_port.portfolio_value - rl_initial) / rl_initial if mom_port else 0.0,
                "transaction_cost": mom_port.total_transaction_cost if mom_port else 0.0,
                "market_impact_cost": mom_port.total_market_impact_cost if mom_port else 0.0,
                "signal_metric": market_state["return"],
                "signal_label": "Price Return"
            },
            "value": {
                "agent_id": "value",
                "action": "HOLD",
                "action_status": "HOLD",
                "quantity": 0,
                "fill_price": market_state["price"],
                "cash": val_port.cash if val_port else 0.0,
                "inventory": val_port.inventory if val_port else 0,
                "portfolio_value": val_port.portfolio_value if val_port else 0.0,
                "reward": 0.0,
                "cumulative_pnl": (val_port.portfolio_value - rl_initial) / rl_initial if val_port else 0.0,
                "transaction_cost": val_port.total_transaction_cost if val_port else 0.0,
                "market_impact_cost": val_port.total_market_impact_cost if val_port else 0.0,
                "signal_metric": (market_state["price"] - market_state["fundamental_value"]) / (market_state["fundamental_value"] + 1e-8),
                "signal_label": "Mispricing"
            }
        }

        # Opponent model beliefs
        op_model = self.market_maker.strategic_layer.opponent_model
        if op_model is not None:
            mom_probs = op_model.predict("momentum", market_state)
            val_probs = op_model.predict("value", market_state)
            bucket_str = str(op_model._discretize_state(market_state))
            mom_cnt = sum(op_model.counts["momentum"][bucket_str].values()) if "momentum" in op_model.counts else 0
            val_cnt = sum(op_model.counts["value"][bucket_str].values()) if "value" in op_model.counts else 0
        else:
            mom_probs = {"BUY": 0.333, "HOLD": 0.333, "SELL": 0.333}
            val_probs = {"BUY": 0.333, "HOLD": 0.333, "SELL": 0.333}
            mom_cnt, val_cnt = 0, 0
            bucket_str = "Prior (Static)"

        opponent_beliefs = {
            "momentum": {
                "agent_id": "momentum",
                "buy_prob": float(mom_probs["BUY"]),
                "hold_prob": float(mom_probs["HOLD"]),
                "sell_prob": float(mom_probs["SELL"]),
                "observation_count": mom_cnt,
                "state_bucket": bucket_str,
                "last_observed_action": None
            },
            "value": {
                "agent_id": "value",
                "buy_prob": float(val_probs["BUY"]),
                "hold_prob": float(val_probs["HOLD"]),
                "sell_prob": float(val_probs["SELL"]),
                "observation_count": val_cnt,
                "state_bucket": bucket_str,
                "last_observed_action": None
            }
        }

        return {
            "step": self.step_count,
            "episode": self.episode_count,
            "status": self.status,
            "timestamp": time.time(),
            "market_state": {
                "price": float(market_state["price"]),
                "return": float(market_state["return"]),
                "volume": float(market_state["volume"]),
                "spread": float(market_state["spread"]),
                "liquidity": float(market_state["liquidity"]),
                "volatility": float(market_state["volatility"]),
                "order_imbalance": float(market_state["order_imbalance"]),
                "fundamental_value": float(market_state["fundamental_value"]),
                "bid_price": float(self.env.bid_price),
                "ask_price": float(self.env.ask_price)
            },
            "leader_decision": {
                "selected_action": leader_decision.selected_action.value,
                "leader_utility": float(leader_decision.leader_utility),
                "spread_multiplier": 0.7 if leader_decision.selected_action.value == "TIGHT" else (1.4 if leader_decision.selected_action.value == "WIDE" else 1.0),
                "liquidity_multiplier": 1.3 if leader_decision.selected_action.value == "TIGHT" else (0.7 if leader_decision.selected_action.value == "WIDE" else 1.0),
                "explanation": explanation,
                "candidate_evaluations": cands
            },
            "agents": agents_data,
            "opponent_beliefs": opponent_beliefs,
            "recent_orders": [],
            "encoded_state_dim": self.rl_agent.encoder.total_dim if self.rl_agent else 10,
            "strategic_feature_dim": len(strat_obs.to_feature_vector()),
            "active_variant": self.active_variant
        }

    async def step(self) -> Dict[str, Any]:
        """Perform exactly one full causal simulation step across all members."""
        if self.env is None:
            return self.get_current_frame()

        market_state = self.env.get_market_state()
        all_states = self.env.get_all_states()

        # 1. Stackelberg Leader Decision
        leader_decision = self.market_maker.act(market_state)
        self._record_event(
            "STACKELBERG",
            f"Leader selected {leader_decision.selected_action.value} (Utility: {leader_decision.leader_utility:.3f})",
            {"utility": leader_decision.leader_utility, "predicted": leader_decision.predicted_follower_actions}
        )

        # 2. Strategic Observation for RL Policy
        strat_obs = self.market_maker.get_strategic_observation(market_state, leader_decision)

        # 3. RL Trader Action Selection
        action_idx, log_prob, val = self.rl_agent.select_action(
            state=all_states["rl_trader"],
            strategic_dict=strat_obs if self.active_variant != "rl_only" else None,
            deterministic=True
        )
        rl_env_action = self.rl_agent.action_space.to_env_action(action_idx)

        # 4. Follower Action Selection
        mom_action = self.momentum_agent.act(all_states["momentum"])
        val_action = self.value_agent.act(all_states["value"])

        actions = {
            "momentum": mom_action,
            "value": val_action,
            "rl_trader": rl_env_action
        }

        # 5. Environment Step with Leader Market Control
        next_states, rewards, done, info = self.env.step(actions, market_control=leader_decision)
        self.step_count += 1

        # 6. Update Opponent Model with actual observed actions
        self.market_maker.update_opponent("momentum", market_state, mom_action)
        self.market_maker.update_opponent("value", market_state, val_action)
        self._record_event(
            "OPPONENT",
            f"Opponent updates: Momentum={mom_action}, Value={val_action}"
        )

        # 7. Store RL Transition in Rollout Buffer
        self.rl_agent.store_transition(
            state=all_states["rl_trader"],
            action=action_idx,
            log_prob=log_prob,
            reward=rewards["rl_trader"],
            value=val,
            done=done,
            strategic_dict=strat_obs if self.active_variant != "rl_only" else None
        )

        # 8. Check for PPO update
        if len(self.rl_agent.buffer.states) >= self.rl_agent.config.rollout_length:
            update_metrics = self.rl_agent.update(
                last_state=next_states["rl_trader"],
                last_done=done,
                last_strategic_dict=strat_obs if self.active_variant != "rl_only" else None
            )
            self._record_event(
                "PPO",
                f"PPO Optimization Update executed. Total Loss: {update_metrics.get('total_loss', 0.0):.4f}"
            )

        # Parse executed orders
        orders_info = info.get("orders", [])
        for ord_item in orders_info:
            if ord_item.get("status") == "EXECUTED":
                self._record_event(
                    "ORDER",
                    f"{ord_item.get('agent_id')} EXECUTED {ord_item.get('action')} @ {ord_item.get('fill_price'):.2f} (qty: {ord_item.get('quantity')})"
                )
            elif ord_item.get("status") == "REJECTED":
                self._record_event(
                    "ORDER",
                    f"{ord_item.get('agent_id')} REJECTED {ord_item.get('action')} - Reason: {ord_item.get('reason')}"
                )

        frame = self.get_current_frame()
        # Add order fill status to frame
        for ord_item in orders_info:
            aid = ord_item.get("agent_id")
            if aid in frame["agents"]:
                frame["agents"][aid]["action_status"] = ord_item.get("status", "HOLD")
                frame["agents"][aid]["quantity"] = ord_item.get("quantity", 0)
                frame["agents"][aid]["fill_price"] = ord_item.get("fill_price", frame["market_state"]["price"])
                frame["agents"][aid]["reward"] = rewards.get(aid, 0.0)

        frame["recent_orders"] = orders_info

        # Record into history
        history_entry = {
            "step": self.step_count,
            "price": frame["market_state"]["price"],
            "fundamental_value": frame["market_state"]["fundamental_value"],
            "spread": frame["market_state"]["spread"],
            "liquidity": frame["market_state"]["liquidity"],
            "return": frame["market_state"]["return"],
            "volatility": frame["market_state"]["volatility"],
            "order_imbalance": frame["market_state"]["order_imbalance"],
            "leader_action": leader_decision.selected_action.value,
            "leader_utility": float(leader_decision.leader_utility),
            "rl_pnl": frame["agents"]["rl_trader"]["cumulative_pnl"],
            "rl_portfolio_value": frame["agents"]["rl_trader"]["portfolio_value"],
            "rl_reward": rewards.get("rl_trader", 0.0)
        }
        self.history.append(history_entry)
        if len(self.history) > self.max_history:
            self.history.pop(0)

        # Handle episode termination
        if done:
            self.episode_count += 1
            self._record_event("MARKET", f"Episode {self.episode_count - 1} completed. Resetting market environment.")
            self.env.reset(seed=self.seed + self.episode_count)

        await self.broadcast_frame(frame)
        return frame

    async def _run_loop(self):
        try:
            while self.status == "RUNNING":
                await self.step()
                await asyncio.sleep(self.delay_ms / 1000.0)
        except asyncio.CancelledError:
            pass
        except Exception as e:
            self.status = "FAILED"
            self._record_event("MARKET", f"Simulation failed with error: {str(e)}")

    def start(self):
        if self.status != "RUNNING":
            self.status = "RUNNING"
            self._record_event("MARKET", "Simulation started.")
            self._running_task = asyncio.create_task(self._run_loop())

    def pause(self):
        if self.status == "RUNNING":
            self.status = "PAUSED"
            if self._running_task:
                self._running_task.cancel()
            self._record_event("MARKET", "Simulation paused.")

    def resume(self):
        if self.status == "PAUSED":
            self.status = "RUNNING"
            self._record_event("MARKET", "Simulation resumed.")
            self._running_task = asyncio.create_task(self._run_loop())

    def stop(self):
        self.status = "IDLE"
        if self._running_task:
            self._running_task.cancel()
        self._record_event("MARKET", "Simulation stopped.")

    def reset(self, config_data: Optional[Dict[str, Any]] = None):
        self.stop()
        cfg = config_data or {}
        self._initialize_simulation(
            episode_length=cfg.get("episode_length", 100),
            initial_price=cfg.get("initial_price", 100.0),
            initial_spread=cfg.get("initial_spread", 0.02),
            initial_liquidity=cfg.get("initial_liquidity", 100.0),
            variant=cfg.get("variant", self.active_variant),
            seed=cfg.get("seed", 42)
        )
        self.status = "IDLE"
        self._record_event("MARKET", "Simulation reset to initial state.")


# Singleton instance
sim_service = SimulationService()
