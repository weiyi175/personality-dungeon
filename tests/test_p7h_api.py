"""Integration tests for the P7-H API routes in api/server.py.

Covers the bifurcation, passive-event, A/B-test, player-test and survey
endpoints added for the real-human study. Each test gets fresh manager
singletons and a temp P7H_OUT_DIR so the restart-persistence writes triggered
on assign/end/submit do not touch the real study data directory.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

import api.ab_test_manager as ab_mod
import api.player_test_tracker as pt_mod
import api.survey_manager as sv_mod
from api import server
from api.server import app

PV = [0.2, -0.1, 0.3, 0.0, 0.1, -0.2, 0.15, 0.05, -0.1]


@pytest.fixture
def client(tmp_path, monkeypatch):
    # Fresh singletons so count-balance / trajectories don't leak across tests.
    monkeypatch.setattr(ab_mod, "_manager", None)
    monkeypatch.setattr(pt_mod, "_tracker", None)
    monkeypatch.setattr(sv_mod, "_survey", None)
    # Redirect persistence away from the real study directory.
    monkeypatch.setattr(server, "P7H_OUT_DIR", str(tmp_path))
    return TestClient(app)


# ── Bifurcation ───────────────────────────────────────────────────────────────


def test_bifurcation_detect_ok(client):
    r = client.post("/bifurcation/detect", json={"personality_vector": PV})
    assert r.status_code == 200
    body = r.json()
    assert len(body["feature_names"]) == 9
    assert "bifurcation" in body and "sensitive_direction" in body


def test_bifurcation_detect_rejects_wrong_length(client):
    r = client.post("/bifurcation/detect", json={"personality_vector": [0.0] * 8})
    assert r.status_code == 422


def test_bifurcation_event_control_is_reproducible(client):
    payload = {"personality_vector": PV, "group": "control", "seed": 7}
    a = client.post("/bifurcation/event", json=payload).json()
    b = client.post("/bifurcation/event", json=payload).json()
    assert a["direction"] == b["direction"]
    assert a["direction_mode"] == "random"


def test_bifurcation_sequence_validates_n_steps(client):
    r = client.post("/bifurcation/sequence", json={"personality_vector": PV, "n_steps": 0})
    assert r.status_code == 422


# ── Passive event choice ──────────────────────────────────────────────────────


def test_event_choose_unknown_event_id_404(client):
    r = client.post("/event/choose", json={"personality_vector": PV, "event_id": "nope"})
    assert r.status_code == 404


def test_event_choose_deterministic(client):
    payload = {"personality_vector": PV, "seed": 3, "deterministic_outcome": True}
    a = client.post("/event/choose", json=payload)
    b = client.post("/event/choose", json=payload)
    assert a.status_code == 200
    assert a.json() == b.json()


# ── A/B test ──────────────────────────────────────────────────────────────────


def test_ab_assign_then_summary_and_lookup(client):
    assign = client.post("/bifurcation/ab-test/assign", json={"session_id": "s1"}).json()
    assert assign["existing"] is False
    assert assign["group"] in ("control", "experiment")

    # idempotent
    again = client.post("/bifurcation/ab-test/assign", json={"session_id": "s1"}).json()
    assert again["existing"] is True

    rec = client.post(
        "/bifurcation/ab-test/record-step",
        json={
            "session_id": "s1",
            "personality_before": [0.0] * 9,
            "personality_after": [1.0] + [0.0] * 8,
            "proximity": 0.6,
            "step_index": 0,
        },
    )
    assert rec.status_code == 200

    # "summary" must resolve to the static route, not the {session_id} catch-all
    summary = client.get("/bifurcation/ab-test/summary").json()
    assert summary["total_sessions"] == 1

    session = client.get("/bifurcation/ab-test/s1").json()
    assert session["event_count"] == 1


def test_ab_get_unknown_session_404(client):
    assert client.get("/bifurcation/ab-test/ghost").status_code == 404


def test_ab_record_step_rejects_wrong_length(client):
    client.post("/bifurcation/ab-test/assign", json={"session_id": "s1"})
    r = client.post(
        "/bifurcation/ab-test/record-step",
        json={
            "session_id": "s1",
            "personality_before": [0.0] * 8,
            "personality_after": [0.0] * 9,
            "proximity": 0.1,
        },
    )
    assert r.status_code == 422


def test_ab_assign_persists_to_disk(client, tmp_path):
    client.post("/bifurcation/ab-test/assign", json={"session_id": "s1"})
    assert (tmp_path / "ab_test_sessions.json").exists()


# ── Player test ───────────────────────────────────────────────────────────────


def test_player_test_full_lifecycle(client):
    start = client.post(
        "/player-test/start",
        json={"session_id": "s1", "group": "experiment", "player_alias": "p"},
    ).json()
    assert start["ok"] is True

    step = client.post(
        "/player-test/step",
        json={
            "session_id": "s1",
            "action_text": "fight",
            "personality_before": [0.0] * 9,
            "personality_after": [0.5] * 9,
            "proximity_before": 0.1,
            "proximity_after": 0.4,
            "event_type": "shift",
        },
    )
    assert step.status_code == 200

    end = client.post("/player-test/end", json={"session_id": "s1"}).json()
    assert end["ok"] is True
    assert end["n_steps"] == 1

    summary = client.get("/player-test/summary").json()
    assert summary["total_sessions"] == 1
    assert summary["experiment"]["n_completed"] == 1


def test_player_test_step_wrong_length_422(client):
    client.post("/player-test/start", json={"session_id": "s1", "group": "control"})
    r = client.post(
        "/player-test/step",
        json={
            "session_id": "s1",
            "personality_before": [0.0] * 9,
            "personality_after": [0.0] * 7,
        },
    )
    assert r.status_code == 422


def test_player_test_end_unknown_session_404(client):
    assert client.post("/player-test/end", json={"session_id": "ghost"}).status_code == 404


def test_player_test_get_unknown_404(client):
    assert client.get("/player-test/ghost").status_code == 404


# ── Survey ────────────────────────────────────────────────────────────────────


def test_survey_questions_returns_three(client):
    body = client.get("/survey/questions").json()
    assert len(body["questions"]) == 3


def test_survey_submit_and_summary(client):
    r = client.post(
        "/survey/submit",
        json={
            "session_id": "s1",
            "group": "experiment",
            "q1_naturalness": 9,
            "q2_fun": 8,
            "q3_replay": 10,
        },
    )
    assert r.status_code == 200
    summary = client.get("/survey/summary").json()
    assert summary["total_responses"] == 1


def test_survey_submit_out_of_range_422(client):
    r = client.post(
        "/survey/submit",
        json={
            "session_id": "s1",
            "group": "control",
            "q1_naturalness": 11,
            "q2_fun": 5,
            "q3_replay": 5,
        },
    )
    assert r.status_code == 422
