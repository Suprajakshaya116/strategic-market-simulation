import asyncio
import pytest
from backend.main import (
    health_check,
    get_simulation_state,
    get_simulation_history,
    step_simulation,
    reset_simulation,
    get_experiment_results,
    get_training_status,
    get_training_checkpoints,
)
from backend.simulation_service import sim_service


def test_health_check_endpoint():
    data = health_check()
    assert data["status"] == "healthy"
    assert "simulation_status" in data


def test_simulation_state_endpoint():
    data = get_simulation_state()
    assert "market_state" in data
    assert "leader_decision" in data
    assert "agents" in data
    assert "opponent_beliefs" in data
    assert data["encoded_state_dim"] in (10, 21)


def test_simulation_step_and_reset_endpoints():
    # Step simulation
    step_data = asyncio.run(step_simulation())
    assert step_data["step"] >= 1
    assert "market_state" in step_data

    # History should contain entries
    history_data = get_simulation_history()
    assert len(history_data) >= 1

    # Reset simulation
    reset_data = reset_simulation()
    assert reset_data["status"] == "reset"
    assert reset_data["frame"]["step"] == 0


def test_experiment_endpoints():
    results = get_experiment_results()
    assert isinstance(results, list)


def test_training_endpoints():
    status = get_training_status()
    assert "is_training" in status
    assert "current_step" in status

    ckpts = get_training_checkpoints()
    assert isinstance(ckpts, list)
