from __future__ import annotations

import pytest

from api.schemas import API_VERSION, ResponseEnvelope, SchemaError, normalize_response_envelope


def _sample_response_payload() -> dict[str, object]:
    return {
        "api_version": API_VERSION,
        "kind": "step",
        "ok": True,
        "session_id": "session_20260429_0001",
        "tick": 12,
        "state_hash": "sha256:7ce1f2a8b0d3e1d1",
        "event": {
            "event_id": "evt_00091",
            "event_type": "choice",
            "title": "門後的回聲",
            "summary": "你聽見門後傳來微弱回聲，三種反應都可以選擇。",
            "tags": ["story", "decision"],
            "severity": 0.42,
            "phase": "active",
        },
        "choices": [
            {
                "choice_id": "c_aggressive",
                "label": "直接推門",
                "strategy": "aggressive",
                "enabled": True,
                "selected": False,
                "description": None,
                "preview": {
                    "expected_reward": 0.25,
                    "expected_risk": 0.58,
                    "confidence": 0.71,
                    "expected_personality_delta": None,
                },
            },
            {
                "choice_id": "c_defensive",
                "label": "先觀察再進入",
                "strategy": "defensive",
                "enabled": True,
                "selected": True,
                "description": None,
                "preview": {
                    "expected_reward": 0.12,
                    "expected_risk": 0.18,
                    "confidence": 0.83,
                    "expected_personality_delta": None,
                },
            },
            {
                "choice_id": "c_balanced",
                "label": "緩慢推進",
                "strategy": "balanced",
                "enabled": True,
                "selected": False,
                "description": None,
                "preview": {
                    "expected_reward": 0.19,
                    "expected_risk": 0.31,
                    "confidence": 0.77,
                    "expected_personality_delta": None,
                },
            },
        ],
        "personality_delta": {
            "vector": {
                "risk_aversion": 0.01,
                "suspicion": 0.02,
                "endurance": 0.0,
                "assertiveness": 0.01,
            },
            "norm": 0.024,
            "confidence": 0.88,
        },
        "emotion": {
            "valence": -0.08,
            "arousal": 0.41,
            "dominance": 0.36,
            "intensity": 0.29,
        },
        "world_state": {
            "scarcity": 0.34,
            "threat": 0.57,
            "noise": 0.22,
            "intel": 0.63,
        },
        "player_state": {
            "health": 0.82,
            "stamina": 0.59,
            "stress": 0.24,
            "utility": 1.87,
            "alive": True,
            "personality": {
                "impulsiveness": 0.18,
                "assertiveness": 0.11,
                "optimism": 0.07,
                "risk_aversion": 0.23,
                "suspicion": 0.19,
                "endurance": 0.14,
                "randomness": 0.25,
                "stability_seeking": 0.21,
                "curiosity": 0.29,
            },
            "memory_load": 0.31,
        },
        "result": {
            "selected_choice_id": "c_defensive",
            "reward": 0.12,
            "utility_delta": 0.04,
            "risk_delta": -0.03,
            "terminated": False,
            "termination_reason": None,
            "state_delta": None,
        },
        "warnings": [],
        "error": None,
        "extensions": {},
    }


def test_normalize_response_envelope_round_trip() -> None:
    payload = _sample_response_payload()

    envelope = normalize_response_envelope(payload)

    assert isinstance(envelope, ResponseEnvelope)
    assert envelope.to_dict() == payload


def test_missing_optional_sections_are_normalized_to_stable_defaults() -> None:
    payload = {
        "api_version": API_VERSION,
        "kind": "snapshot",
        "ok": True,
        "session_id": "session_minimal",
        "tick": 0,
        "state_hash": None,
        "warnings": [],
        "extensions": {},
    }

    envelope = normalize_response_envelope(payload)

    assert envelope.event is None
    assert envelope.choices == ()
    assert envelope.personality_delta.to_dict() == {}
    assert envelope.emotion.to_dict() == {}
    assert envelope.world_state.to_dict() == {}
    assert envelope.player_state.to_dict() == {}
    assert envelope.result is None
    assert envelope.error is None
    assert envelope.to_dict()["personality_delta"] == {}
    assert envelope.to_dict()["emotion"] == {}
    assert envelope.to_dict()["world_state"] == {}
    assert envelope.to_dict()["player_state"] == {}


def test_invalid_choice_strategy_is_rejected() -> None:
    payload = _sample_response_payload()
    payload["choices"] = [
        {
            "choice_id": "c_invalid",
            "label": "非法策略",
            "strategy": "stealth",
            "enabled": True,
            "selected": False,
        }
    ]

    with pytest.raises(SchemaError):
        normalize_response_envelope(payload)


def test_error_kind_requires_error_payload() -> None:
    payload = {
        "api_version": API_VERSION,
        "kind": "error",
        "ok": False,
        "session_id": "session_error",
        "tick": 1,
        "state_hash": None,
        "warnings": [],
        "extensions": {},
    }

    with pytest.raises(SchemaError):
        normalize_response_envelope(payload)