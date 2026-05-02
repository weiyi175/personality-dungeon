"""Integration tests for api/server.py.

Tests that server correctly:
  1. Wraps core.GameEngine.step() output in ResponseEnvelope
  2. Maintains session contract (immutable session_id, incrementing tick)
  3. Preserves schema contract across all endpoints
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from api.schemas import API_VERSION, normalize_response_envelope
from api.server import app, SESSIONS


@pytest.fixture
def client():
    """FastAPI test client."""
    return TestClient(app)


@pytest.fixture(autouse=True)
def clear_sessions():
    """Clear all sessions before each test."""
    SESSIONS.clear()
    yield
    SESSIONS.clear()


# ===================================================================
# Test: Initialize Session
# ===================================================================


def test_initialize_session_creates_unique_session_id(client):
    """Initialize endpoint returns unique session_id."""
    response1 = client.post("/sessions/initialize", json={})
    response2 = client.post("/sessions/initialize", json={})
    
    assert response1.status_code == 200
    assert response2.status_code == 200
    
    data1 = response1.json()
    data2 = response2.json()
    
    assert "session_id" in data1
    assert "session_id" in data2
    assert data1["session_id"] != data2["session_id"]


def test_initialize_session_with_params(client):
    """Initialize endpoint accepts n_players and seed params."""
    response = client.post("/sessions/initialize", json={"n_players": 20, "seed": 42})
    
    assert response.status_code == 200
    assert "session_id" in response.json()


def test_initialize_session_creates_entry_in_sessions_store(client):
    """Initialize adds session to SESSIONS dict."""
    response = client.post("/sessions/initialize", json={})
    session_id = response.json()["session_id"]
    
    assert session_id in SESSIONS


# ===================================================================
# Test: Step Endpoint
# ===================================================================


def test_step_returns_response_envelope(client):
    """Step endpoint returns ResponseEnvelope-compatible JSON."""
    # Initialize session
    init_response = client.post("/sessions/initialize", json={})
    session_id = init_response.json()["session_id"]
    
    # Step
    step_response = client.post(
        f"/sessions/{session_id}/step",
        json={"action": "aggressive"},
    )
    
    assert step_response.status_code == 200
    data = step_response.json()
    
    # Contract checks
    assert data["api_version"] == API_VERSION
    assert data["kind"] == "step"
    assert data["ok"] is True
    assert data["session_id"] == session_id
    assert "tick" in data
    assert "state_hash" in data


def test_step_increments_tick(client):
    """Each step increments session tick."""
    init_response = client.post("/sessions/initialize", json={})
    session_id = init_response.json()["session_id"]
    
    step1 = client.post(f"/sessions/{session_id}/step", json={})
    tick1 = step1.json()["tick"]
    
    step2 = client.post(f"/sessions/{session_id}/step", json={})
    tick2 = step2.json()["tick"]
    
    step3 = client.post(f"/sessions/{session_id}/step", json={})
    tick3 = step3.json()["tick"]
    
    # Each step should increment tick
    assert tick1 == 0  # First step has tick=0
    assert tick2 == 1
    assert tick3 == 2


def test_step_with_missing_session_returns_404(client):
    """Step on non-existent session returns 404."""
    response = client.post("/sessions/nonexistent/step", json={})
    assert response.status_code == 404


def test_step_includes_world_state(client):
    """Step response includes world_state."""
    init_response = client.post("/sessions/initialize", json={})
    session_id = init_response.json()["session_id"]
    
    step_response = client.post(f"/sessions/{session_id}/step", json={})
    data = step_response.json()
    
    assert "world_state" in data
    assert isinstance(data["world_state"], dict)


def test_step_includes_result_object(client):
    """Step response includes result object with expected fields."""
    init_response = client.post("/sessions/initialize", json={})
    session_id = init_response.json()["session_id"]
    
    step_response = client.post(f"/sessions/{session_id}/step", json={})
    data = step_response.json()
    
    assert "result" in data
    result = data["result"]
    
    # Result should have these fields
    assert "selected_choice_id" in result
    assert "reward" in result
    assert "utility_delta" in result
    assert "risk_delta" in result
    assert "terminated" in result


# ===================================================================
# Test: Snapshot Endpoint
# ===================================================================


def test_snapshot_returns_envelope(client):
    """Snapshot endpoint returns ResponseEnvelope with kind=snapshot."""
    init_response = client.post("/sessions/initialize", json={})
    session_id = init_response.json()["session_id"]
    
    # Do a few steps first
    client.post(f"/sessions/{session_id}/step", json={})
    client.post(f"/sessions/{session_id}/step", json={})
    
    snapshot_response = client.get(f"/sessions/{session_id}/snapshot")
    
    assert snapshot_response.status_code == 200
    data = snapshot_response.json()
    
    # Contract checks
    assert data["api_version"] == API_VERSION
    assert data["kind"] == "snapshot"
    assert data["ok"] is True
    assert data["session_id"] == session_id
    assert data["tick"] == 2  # Should reflect current tick


def test_snapshot_does_not_increment_tick(client):
    """Snapshot should not increment tick counter."""
    init_response = client.post("/sessions/initialize", json={})
    session_id = init_response.json()["session_id"]
    
    client.post(f"/sessions/{session_id}/step", json={})
    
    snapshot1 = client.get(f"/sessions/{session_id}/snapshot")
    tick1 = snapshot1.json()["tick"]
    
    snapshot2 = client.get(f"/sessions/{session_id}/snapshot")
    tick2 = snapshot2.json()["tick"]
    
    assert tick1 == tick2 == 1


def test_snapshot_with_missing_session_returns_404(client):
    """Snapshot on non-existent session returns 404."""
    response = client.get("/sessions/nonexistent/snapshot")
    assert response.status_code == 404


# ===================================================================
# Test: Reset Endpoint
# ===================================================================


def test_reset_resets_tick_to_zero(client):
    """Reset endpoint sets tick back to 0."""
    init_response = client.post("/sessions/initialize", json={})
    session_id = init_response.json()["session_id"]
    
    # Do steps
    client.post(f"/sessions/{session_id}/step", json={})
    client.post(f"/sessions/{session_id}/step", json={})
    
    # Verify tick is 2
    snapshot = client.get(f"/sessions/{session_id}/snapshot")
    assert snapshot.json()["tick"] == 2
    
    # Reset
    reset_response = client.post(f"/sessions/{session_id}/reset")
    assert reset_response.status_code == 200
    
    # Verify tick is now 0
    snapshot = client.get(f"/sessions/{session_id}/snapshot")
    assert snapshot.json()["tick"] == 0


def test_reset_returns_envelope_with_kind_reset(client):
    """Reset endpoint returns ResponseEnvelope with kind=reset."""
    init_response = client.post("/sessions/initialize", json={})
    session_id = init_response.json()["session_id"]
    
    reset_response = client.post(f"/sessions/{session_id}/reset")
    data = reset_response.json()
    
    assert data["api_version"] == API_VERSION
    assert data["kind"] == "reset"
    assert data["ok"] is True


def test_reset_with_missing_session_returns_404(client):
    """Reset on non-existent session returns 404."""
    response = client.post("/sessions/nonexistent/reset")
    assert response.status_code == 404


# ===================================================================
# Test: State Hash Endpoint
# ===================================================================


def test_state_hash_returns_hash_string(client):
    """State hash endpoint returns deterministic hash."""
    init_response = client.post("/sessions/initialize", json={})
    session_id = init_response.json()["session_id"]
    
    hash_response = client.get(f"/sessions/{session_id}/state-hash")
    
    assert hash_response.status_code == 200
    data = hash_response.json()
    
    assert "state_hash" in data
    assert isinstance(data["state_hash"], str)
    assert len(data["state_hash"]) > 0


def test_state_hash_is_deterministic(client):
    """Same session should produce same hash until state changes."""
    init_response = client.post("/sessions/initialize", json={})
    session_id = init_response.json()["session_id"]
    
    hash1 = client.get(f"/sessions/{session_id}/state-hash").json()["state_hash"]
    hash2 = client.get(f"/sessions/{session_id}/state-hash").json()["state_hash"]
    
    assert hash1 == hash2


def test_state_hash_changes_after_step(client):
    """State hash should change after step (tick incremented)."""
    init_response = client.post("/sessions/initialize", json={})
    session_id = init_response.json()["session_id"]
    
    hash_before = client.get(f"/sessions/{session_id}/state-hash").json()["state_hash"]
    
    client.post(f"/sessions/{session_id}/step", json={})
    
    hash_after = client.get(f"/sessions/{session_id}/state-hash").json()["state_hash"]
    
    assert hash_before != hash_after


def test_state_hash_with_missing_session_returns_404(client):
    """State hash on non-existent session returns 404."""
    response = client.get("/sessions/nonexistent/state-hash")
    assert response.status_code == 404


# ===================================================================
# Test: Session Isolation
# ===================================================================


def test_multiple_sessions_are_isolated(client):
    """Multiple sessions maintain independent state."""
    init1 = client.post("/sessions/initialize", json={})
    session1 = init1.json()["session_id"]
    
    init2 = client.post("/sessions/initialize", json={})
    session2 = init2.json()["session_id"]
    
    # Step session 1 twice
    client.post(f"/sessions/{session1}/step", json={})
    client.post(f"/sessions/{session1}/step", json={})
    
    # Step session 2 once
    client.post(f"/sessions/{session2}/step", json={})
    
    # Check ticks are independent
    snap1 = client.get(f"/sessions/{session1}/snapshot").json()
    snap2 = client.get(f"/sessions/{session2}/snapshot").json()
    
    assert snap1["tick"] == 2
    assert snap2["tick"] == 1


# ===================================================================
# Test: Contract Consistency
# ===================================================================


def test_all_responses_have_api_version(client):
    """All endpoints return api_version field."""
    init_response = client.post("/sessions/initialize", json={})
    session_id = init_response.json()["session_id"]
    
    # Initialize returns session_id only, not envelope
    # But step, snapshot, reset should all have api_version
    step_response = client.post(f"/sessions/{session_id}/step", json={})
    snapshot_response = client.get(f"/sessions/{session_id}/snapshot")
    reset_response = client.post(f"/sessions/{session_id}/reset")
    
    assert step_response.json()["api_version"] == API_VERSION
    assert snapshot_response.json()["api_version"] == API_VERSION
    assert reset_response.json()["api_version"] == API_VERSION


def test_all_envelope_responses_have_required_fields(client):
    """All ResponseEnvelope responses have required fields."""
    init_response = client.post("/sessions/initialize", json={})
    session_id = init_response.json()["session_id"]
    
    for endpoint_func, path in [
        (client.post, f"/sessions/{session_id}/step"),
        (client.get, f"/sessions/{session_id}/snapshot"),
        (client.post, f"/sessions/{session_id}/reset"),
    ]:
        response = endpoint_func(path, json={}) if endpoint_func == client.post else endpoint_func(path)
        data = response.json()
        
        # Required envelope fields
        assert "api_version" in data, f"Missing api_version in {path}"
        assert "kind" in data, f"Missing kind in {path}"
        assert "ok" in data, f"Missing ok in {path}"
        assert "session_id" in data, f"Missing session_id in {path}"
        assert "tick" in data, f"Missing tick in {path}"
