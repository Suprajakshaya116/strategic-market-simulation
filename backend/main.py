import os
import json
import pandas as pd
from typing import Dict, Any, Optional
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from backend.simulation_service import sim_service
from backend.training_service import training_service
from backend.schemas import SimulationConfigSchema


app = FastAPI(
    title="Strategic Trading MARL Research Lab API",
    description="Live simulation, Stackelberg Game Theory, and PPO trading telemetry backend.",
    version="1.0.0"
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
def health_check():
    return {
        "status": "healthy",
        "service": "Strategic Trading MARL API",
        "simulation_status": sim_service.status,
        "step": sim_service.step_count,
        "episode": sim_service.episode_count
    }


# ==============================================================================
# Simulation Endpoints
# ==============================================================================

@app.get("/api/simulation/state")
def get_simulation_state():
    return sim_service.get_current_frame()


@app.get("/api/simulation/history")
def get_simulation_history():
    return sim_service.history


@app.get("/api/simulation/events")
def get_simulation_events():
    return sim_service.events


@app.post("/api/simulation/start")
def start_simulation():
    sim_service.start()
    return {"status": "started", "simulation_status": sim_service.status}


@app.post("/api/simulation/pause")
def pause_simulation():
    sim_service.pause()
    return {"status": "paused", "simulation_status": sim_service.status}


@app.post("/api/simulation/resume")
def resume_simulation():
    sim_service.resume()
    return {"status": "resumed", "simulation_status": sim_service.status}


@app.post("/api/simulation/stop")
def stop_simulation():
    sim_service.stop()
    return {"status": "stopped", "simulation_status": sim_service.status}


@app.post("/api/simulation/step")
async def step_simulation():
    frame = await sim_service.step()
    return frame


class SpeedPayload(BaseModel):
    delay_ms: int


@app.post("/api/simulation/speed")
def set_simulation_speed(payload: SpeedPayload):
    sim_service.delay_ms = max(payload.delay_ms, 20)
    return {"delay_ms": sim_service.delay_ms}


@app.post("/api/simulation/reset")
def reset_simulation(config: Optional[SimulationConfigSchema] = None):
    cfg_dict = config.dict() if config else None
    sim_service.reset(cfg_dict)
    return {"status": "reset", "frame": sim_service.get_current_frame()}


# WebSocket streaming endpoint
@app.websocket("/ws/simulation")
async def websocket_simulation_endpoint(websocket: WebSocket):
    await sim_service.connect(websocket)
    try:
        while True:
            # Keep connection open, handle incoming ping or commands
            data = await websocket.receive_text()
            if data == "step":
                await sim_service.step()
            elif data == "ping":
                await websocket.send_text("pong")
    except WebSocketDisconnect:
        sim_service.disconnect(websocket)
    except Exception:
        sim_service.disconnect(websocket)


# ==============================================================================
# RL Training Endpoints
# ==============================================================================

class TrainStartPayload(BaseModel):
    timesteps: int = 1000
    variant: str = "proposed_full"
    seed: int = 42


@app.post("/api/training/start")
def start_training(payload: TrainStartPayload):
    return training_service.start_training(
        timesteps=payload.timesteps,
        variant=payload.variant,
        seed=payload.seed
    )


@app.get("/api/training/status")
def get_training_status():
    return training_service.get_status()


@app.post("/api/training/stop")
def stop_training():
    return training_service.stop_training()


@app.get("/api/training/checkpoints")
def get_training_checkpoints():
    return training_service.list_checkpoints()


# ==============================================================================
# Research Experiment Comparison Endpoints
# ==============================================================================

@app.get("/api/experiments/results")
def get_experiment_results():
    json_path = "experiments/results/controlled_experiments.json"
    if os.path.exists(json_path):
        with open(json_path, "r") as f:
            return json.load(f)
    return []


@app.get("/api/experiments/summary")
def get_experiment_summary():
    csv_path = "experiments/results/summary_by_variant.csv"
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        return df.to_dict(orient="records")
    return []
