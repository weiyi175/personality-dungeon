"""Integration tests for the Space-A event RL endpoints (P7-H).

POST /rl_sessions/{id}/apply-event and GET /rl_sessions/{id}/personality.
"""

from __future__ import annotations

import numpy as np
import pytest
from fastapi.testclient import TestClient

from api.server import app
from api.rl_session_manager import get_session_manager


@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture(autouse=True)
def clear_sessions():
    manager = get_session_manager()
    for sid in list(manager.sessions.keys()):
        manager.delete_session(sid)
    yield
    for sid in list(manager.sessions.keys()):
        manager.delete_session(sid)


def _init(client, events_enabled):
    r = client.post(
        "/rl_sessions/initialize",
        json={
            "n_players": 20,
            "n_rounds": 200,
            "burn_in": 10,
            "seed": 1,
            "space_a_events_enabled": events_enabled,
        },
    )
    assert r.status_code == 200
    return r.json()["session_id"]


def test_snapshot_carries_space_a_fields(client):
    sid = _init(client, events_enabled=True)
    snap = client.get(f"/rl_sessions/{sid}/snapshot").json()["snapshot"]
    assert len(snap["mean_personality"]) == 9
    assert snap["personality_displacement"] == pytest.approx(0.0)


def test_apply_event_explicit_displacement_moves_dv(client):
    sid = _init(client, events_enabled=True)
    disp = [0.02] * 9
    r = client.post(f"/rl_sessions/{sid}/apply-event", json={"displacement": disp})
    assert r.status_code == 200
    snap = r.json()["snapshot"]
    assert snap["personality_displacement"] > 0.0


def test_apply_event_designed_from_group(client):
    sid = _init(client, events_enabled=True)
    r = client.post(
        f"/rl_sessions/{sid}/apply-event",
        json={"group": "experiment", "intensity_scale": 1.0},
    )
    assert r.status_code == 200
    assert r.json()["snapshot"]["personality_displacement"] > 0.0


def test_apply_event_disabled_returns_409(client):
    sid = _init(client, events_enabled=False)
    r = client.post(f"/rl_sessions/{sid}/apply-event", json={"displacement": [0.01] * 9})
    assert r.status_code == 409


def test_apply_event_wrong_length_422(client):
    sid = _init(client, events_enabled=True)
    r = client.post(f"/rl_sessions/{sid}/apply-event", json={"displacement": [0.01] * 8})
    assert r.status_code == 422


def test_apply_event_unknown_session_404(client):
    r = client.post("/rl_sessions/ghost/apply-event", json={"displacement": [0.01] * 9})
    assert r.status_code == 404


def test_personality_endpoint_reports_both_spaces(client):
    sid = _init(client, events_enabled=True)
    body = client.get(f"/rl_sessions/{sid}/personality").json()
    assert len(body["mean_personality_space_a"]) == 9
    assert len(body["mean_personality_space_b"]) == 9
    assert "bifurcation" in body
    assert len(body["feature_names"]) == 9


def test_personality_endpoint_unknown_session_404(client):
    assert client.get("/rl_sessions/ghost/personality").status_code == 404


def test_control_vs_experiment_same_magnitude(client):
    """Both arms must apply equal force; only the direction differs."""
    sid_e = _init(client, events_enabled=True)
    sid_c = _init(client, events_enabled=True)
    # Designed-event displacement magnitude depends on proximity at the (shared,
    # seed-fixed) initial mean, so both arms should match in magnitude.
    e = client.post(f"/rl_sessions/{sid_e}/apply-event", json={"group": "experiment"})
    c = client.post(f"/rl_sessions/{sid_c}/apply-event", json={"group": "control", "seed": 0})
    dv_e = e.json()["snapshot"]["personality_displacement"]
    dv_c = c.json()["snapshot"]["personality_displacement"]
    assert dv_e == pytest.approx(dv_c, rel=1e-6)
