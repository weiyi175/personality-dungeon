from __future__ import annotations

import random
from pathlib import Path

from core.game_engine import GameEngine
from dungeon.dungeon_ai import DungeonAI
from dungeon.event_loader import EventLoader, EventSchemaError
from players.base_player import BasePlayer


class _FixedRng:
	def __init__(self, *, random_value: float, uniform_value: float = 0.0):
		self._random_value = float(random_value)
		self._uniform_value = float(uniform_value)

	def random(self) -> float:
		return self._random_value

	def uniform(self, _a: float, _b: float) -> float:
		return self._uniform_value

	def choices(self, population, weights=None, k=1):
		return [population[0]]

	def choice(self, seq):
		return seq[0]


def _event_json_path() -> Path:
	return Path(__file__).resolve().parents[1] / "docs" / "personality_dungeon_v1" / "02_event_templates_v1.json"


def test_event_loader_defaults_are_available_for_actions_without_local_schema() -> None:
	loader = EventLoader(_event_json_path())
	event = loader.get_event_template("resource_suspicious_chest")
	action = next(action for action in event["actions"] if action["name"] == "open_now")
	success_model = loader.resolve_success_model(action)
	state_effects = loader.resolve_state_effects(action)

	assert success_model["name"] == "linear_risk_complement"
	assert success_model["kwargs"] == {}
	assert state_effects["on_success"]["stress_delta"] == 0.0
	assert state_effects["on_failure"]["health_delta"] == 0.0


def test_event_loader_named_success_model_registry_path() -> None:
	loader = EventLoader(_event_json_path())
	action = {
		"weights": [0.0] * len(loader.dimensions_order),
		"base_risk": 0.2,
		"reward_effects": {
			"utility_delta": 0.0,
			"risk_delta": 0.0,
			"popularity_shift": {},
			"trait_deltas": {},
		},
		"failure_outcomes": [
			{
				"kind": "failure",
				"probability": 1.0,
				"utility_delta": 0.0,
				"risk_delta": 0.0,
				"popularity_shift": {},
				"trait_deltas": {},
			}
		],
		"success_model": "logistic",
		"success_model_kwargs": {"success_bias": 1.0, "steepness": 5.0},
	}

	resolved = loader.resolve_success_model(action)
	success_prob = loader.compute_success_prob(action, 0.2)

	assert resolved["name"] == "logistic"
	assert resolved["kwargs"]["steepness"] == 5.0
	assert 0.0 <= success_prob <= 1.0


def test_event_loader_rejects_unexpected_success_model_kwargs() -> None:
	loader = EventLoader(_event_json_path())
	action = {
		"weights": [0.0] * len(loader.dimensions_order),
		"base_risk": 0.2,
		"reward_effects": {
			"utility_delta": 0.0,
			"risk_delta": 0.0,
			"popularity_shift": {},
			"trait_deltas": {},
		},
		"failure_outcomes": [
			{
				"kind": "failure",
				"probability": 1.0,
				"utility_delta": 0.0,
				"risk_delta": 0.0,
				"popularity_shift": {},
				"trait_deltas": {},
			}
		],
		"success_model": "logistic",
		"success_model_kwargs": {"unexpected": 1.0},
	}

	try:
		loader.compute_success_prob(action, 0.2)
	except EventSchemaError as exc:
		assert "unexpected kwargs" in str(exc)
	else:
		raise AssertionError("expected EventSchemaError for unexpected success_model kwargs")


def test_event_loader_state_feedback_increases_risk() -> None:
	loader = EventLoader(_event_json_path())
	event = loader.get_event_template("internal_panic_wave")
	action = next(action for action in event["actions"] if action["name"] == "steady_breathing")

	base_risk = loader.compute_final_risk(action, {}, state={})
	state_risk = loader.compute_final_risk(
		action,
		{},
		state={"risk": 0.10, "risk_drift": 0.05, "stress": 0.5},
	)

	assert state_risk > base_risk


def test_event_loader_noise_scales_action_utility_jitter() -> None:
	loader = EventLoader(_event_json_path())
	event = loader.get_event_template("internal_panic_wave")
	action = next(action for action in event["actions"] if action["name"] == "flee_randomly")
	personality = {"randomness": 1.0}
	rng = _FixedRng(random_value=0.0, uniform_value=1.0)

	low_noise = loader.compute_action_utility(action, personality, state={"noise": 0.0}, rng=rng)
	high_noise = loader.compute_action_utility(action, personality, state={"noise": 2.0}, rng=rng)

	assert high_noise > low_noise


def test_event_loader_intel_improves_success_prob() -> None:
	loader = EventLoader(_event_json_path())
	event = loader.get_event_template("resource_suspicious_chest")
	action = next(action for action in event["actions"] if action["name"] == "inspect")
	base_prob = loader.compute_success_prob(action, 0.4, state={"intel": 0.0})
	intel_prob = loader.compute_success_prob(action, 0.4, state={"intel": 2.0})

	assert intel_prob > base_prob


def test_event_loader_process_turn_applies_success_updates() -> None:
	loader = EventLoader(_event_json_path())
	player = BasePlayer(
		["aggressive", "defensive", "balanced"],
		personality={"impulsiveness": 0.8, "ambition": 0.7, "caution": 0.1},
	)

	# Use observe: base_risk=0.18 with this personality gives final_risk≈0.17 < failure_threshold=0.42.
	result = loader.process_turn(
		player,
		event_id="threat_shadow_stalker",
		action_name="observe",
		rng=_FixedRng(random_value=0.0),
	)

	assert result["success"] is True
	assert result["utility_delta"] == 0.14
	assert player.personality["curiosity"] > 0.0
	assert player.state["stress"] < 0.0
	assert player.state["risk"] < 0.0
	assert player.state["last_event_id"] == "threat_shadow_stalker"


def test_failure_threshold_hard_gate_when_risk_exceeds_threshold() -> None:
	loader = EventLoader(_event_json_path())
	event = loader.get_event_template("threat_shadow_stalker")
	attack = next(a for a in event["actions"] if a["name"] == "attack")

	# failure_threshold=0.82; pass final_risk=0.90 which exceeds it
	prob = loader.compute_success_prob(attack, 0.90)

	assert prob == 0.0


# below threshold → normal model applies (non-zero success prob)
def test_failure_threshold_below_threshold_uses_model() -> None:
	loader = EventLoader(_event_json_path())
	event = loader.get_event_template("threat_shadow_stalker")
	retreat = next(a for a in event["actions"] if a["name"] == "retreat")

	# failure_threshold=0.48; pass final_risk=0.30 which is below
	prob = loader.compute_success_prob(retreat, 0.30)

	assert prob > 0.0


def test_health_penalty_increases_final_risk_when_health_is_low() -> None:
	loader = EventLoader(_event_json_path())
	event = loader.get_event_template("threat_shadow_stalker")
	observe = next(a for a in event["actions"] if a["name"] == "observe")

	risk_full_health = loader.compute_final_risk(observe, {}, state={"health": 1.0})
	risk_low_health = loader.compute_final_risk(observe, {}, state={"health": 0.0})

	assert risk_low_health > risk_full_health


def test_state_decay_reduces_stress_and_risk_drift_after_turn() -> None:
	loader = EventLoader(_event_json_path())
	player = BasePlayer(
		["aggressive", "defensive", "balanced"],
		personality={"caution": 0.8, "patience": 0.5},
	)
	# Set moderate state values; with caution=0.8 the observe action has
	# final_risk = 0.18 - 0.064 + 0 + 0.2(risk_drift) + 0.04(stress) = 0.356 < 0.42(threshold)
	player.state["stress"] = 0.4
	player.state["risk_drift"] = 0.2
	loader.process_turn(
		player,
		event_id="threat_shadow_stalker",
		action_name="observe",
		rng=_FixedRng(random_value=0.0),
	)

	# success path: delta applied then decay(0.92 / 0.90) brings both below initial values
	assert player.state["stress"] < 0.4
	assert player.state["risk_drift"] < 0.2


def test_health_regen_occurs_each_turn() -> None:
	loader = EventLoader(_event_json_path())
	player = BasePlayer(
		["aggressive", "defensive", "balanced"],
		personality={"caution": 0.8},
	)
	player.state["health"] = 0.5
	loader.process_turn(
		player,
		event_id="threat_shadow_stalker",
		action_name="observe",
		rng=_FixedRng(random_value=0.0),
	)

	# health regen = +0.02 per turn; success health_delta = 0.0 for observe
	assert player.state["health"] > 0.5


def test_game_engine_can_include_event_loader_reward_stream() -> None:
	loader = EventLoader(_event_json_path())
	player = BasePlayer(
		["aggressive", "defensive", "balanced"],
		rng=random.Random(1),
		personality={"impulsiveness": 0.8, "ambition": 0.7, "caution": 0.1},
	)
	player.update_weights({"aggressive": 100.0, "defensive": 1.0, "balanced": 1.0})
	dungeon = DungeonAI(
		payoff_mode="count_cycle",
		gamma=0.0,
		epsilon=0.0,
		base_reward=10.0,
		strategy_cycle=["aggressive", "defensive", "balanced"],
		event_loader=loader,
		event_rng=_FixedRng(random_value=0.0),
	)
	engine = GameEngine([player], dungeon)

	engine.step()

	assert player.last_strategy == "aggressive"
	# attack final_risk ≈ 0.938 (high impulsiveness+ambition) >= failure_threshold=0.74 → hard gate → failure.
	# counter_hit utility_delta = -0.28 → base_reward(10.0) + (-0.28) = 9.72
	assert player.last_reward == 9.72
	assert player.utility == 9.72