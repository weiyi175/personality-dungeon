from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from analysis.event_provenance_summary import summarize_event_provenance
from dungeon.event_loader import EventLoader
from simulation.personality_gate1 import seed_metrics
from simulation.run_simulation import SimConfig, _write_timeseries_csv, simulate

WORLD_DIMENSIONS = ("scarcity", "threat", "noise", "intel")
EVENT_FAMILIES = ("Threat", "Resource", "Uncertainty", "Navigation", "Internal")
STRATEGY_ORDER = ("aggressive", "defensive", "balanced")

INITIAL_WORLD_STATE = {
	"scarcity": 0.5,
	"threat": 0.5,
	"noise": 0.5,
	"intel": 0.5,
}

B_P = {
	"scarcity": (0.60, 0.20, -0.40),
	"threat": (0.60, -0.10, -0.30),
	"noise": (-0.25, -0.25, 0.40),
	"intel": (-0.30, -0.10, 0.45),
}

B_E = {
	"scarcity": (-0.10, 0.40, 0.10, -0.10, 0.10),
	"threat": (0.45, -0.15, 0.10, -0.05, 0.05),
	"noise": (0.05, -0.10, 0.35, -0.05, 0.30),
	"intel": (-0.20, 0.15, -0.20, 0.40, -0.05),
}

REWARD_TARGET = 0.27


@dataclass(frozen=True)
class W1CellConfig:
	condition: str
	lambda_world: float
	out_dir: Path
	events_json: Path
	players: int
	rounds: int
	selection_strength: float
	init_bias: float
	memory_kernel: int
	world_update_interval: int
	burn_in: int
	tail: int
	a: float
	b: float
	cross: float
	world_mode: str = "adaptive_world_w1"

	def world_updates_path(self, seed: int) -> Path:
		return self.out_dir / f"w1_step_world_{self.condition}_seed{int(seed)}.tsv"

	def out_csv_path(self, seed: int) -> Path:
		return self.out_dir / f"seed{int(seed)}.csv"

	def provenance_path(self, seed: int) -> Path:
		return self.out_dir / f"seed{int(seed)}_provenance.json"


def _clamp(value: float, lower: float, upper: float) -> float:
	return max(lower, min(upper, float(value)))


def _safe_mean(values: list[float]) -> float:
	if not values:
		return 0.0
	return float(sum(values) / float(len(values)))


def _normalize_positive_mapping(values: Mapping[str, float]) -> dict[str, float]:
	total = sum(max(0.0, float(value)) for value in values.values())
	if total <= 0.0:
		uniform = 1.0 / float(len(values) or 1)
		return {str(key): uniform for key in values}
	return {str(key): max(0.0, float(value)) / total for key, value in values.items()}


def _state_deviated(state: Mapping[str, float]) -> bool:
	return any(abs(float(state.get(name, 0.5)) - 0.5) > 1e-9 for name in WORLD_DIMENSIONS)


def _world_to_event_type_weights(state: Mapping[str, float]) -> dict[str, float]:
	scarcity = float(state["scarcity"])
	threat = float(state["threat"])
	noise = float(state["noise"])
	intel = float(state["intel"])
	return {
		"Threat": _clamp(1.0 + 0.80 * (threat - 0.5) + 0.30 * (noise - 0.5) - 0.20 * (intel - 0.5), 0.20, 3.00),
		"Resource": _clamp(1.0 - 0.80 * (scarcity - 0.5) + 0.35 * (intel - 0.5), 0.20, 3.00),
		"Uncertainty": _clamp(1.0 + 0.90 * (noise - 0.5) + 0.20 * (threat - 0.5), 0.20, 3.00),
		"Navigation": _clamp(1.0 + 0.75 * (intel - 0.5) - 0.30 * (noise - 0.5), 0.20, 3.00),
		"Internal": _clamp(1.0 + 0.55 * (noise - 0.5) + 0.25 * (scarcity - 0.5) - 0.20 * (intel - 0.5), 0.20, 3.00),
	}


def _world_to_risk_multipliers(state: Mapping[str, float]) -> dict[str, float]:
	scarcity = float(state["scarcity"])
	threat = float(state["threat"])
	noise = float(state["noise"])
	intel = float(state["intel"])
	return {
		"Threat": _clamp(1.0 + 0.30 * (threat - 0.5), 0.85, 1.15),
		"Resource": _clamp(1.0 + 0.15 * (scarcity - 0.5) - 0.10 * (intel - 0.5), 0.90, 1.10),
		"Uncertainty": _clamp(1.0 + 0.35 * (noise - 0.5), 0.85, 1.15),
		"Navigation": _clamp(1.0 - 0.20 * (intel - 0.5) + 0.10 * (noise - 0.5), 0.90, 1.10),
		"Internal": _clamp(1.0 + 0.20 * (noise - 0.5) + 0.10 * (scarcity - 0.5), 0.90, 1.10),
	}


def _world_to_reward_multipliers(state: Mapping[str, float]) -> dict[str, float]:
	scarcity = float(state["scarcity"])
	threat = float(state["threat"])
	noise = float(state["noise"])
	intel = float(state["intel"])
	return {
		"Threat": _clamp(1.0 + 0.20 * (threat - 0.5), 0.85, 1.15),
		"Resource": _clamp(1.0 - 0.30 * (scarcity - 0.5), 0.85, 1.15),
		"Uncertainty": _clamp(1.0 + 0.40 * (noise - 0.5), 0.80, 1.20),
		"Navigation": _clamp(1.0 + 0.30 * (intel - 0.5), 0.85, 1.15),
		"Internal": _clamp(1.0 + 0.40 * (noise - 0.5), 0.80, 1.20),
	}


def _world_to_trait_multipliers(state: Mapping[str, float]) -> dict[str, float]:
	scarcity = float(state["scarcity"])
	threat = float(state["threat"])
	noise = float(state["noise"])
	intel = float(state["intel"])
	return {
		"Threat": _clamp(1.0 + 0.20 * (threat - 0.5), 0.85, 1.15),
		"Resource": _clamp(1.0 - 0.15 * (scarcity - 0.5), 0.85, 1.15),
		"Uncertainty": _clamp(1.0 + 0.25 * (noise - 0.5), 0.80, 1.20),
		"Navigation": _clamp(1.0 + 0.20 * (intel - 0.5), 0.85, 1.15),
		"Internal": _clamp(1.0 + 0.20 * (noise - 0.5) + 0.10 * (scarcity - 0.5), 0.85, 1.15),
	}


def world_profile_from_state(state: Mapping[str, float]) -> dict[str, dict[str, float]]:
	return {
		"event_type_weights": _world_to_event_type_weights(state),
		"risk_multipliers": _world_to_risk_multipliers(state),
		"reward_multipliers": _world_to_reward_multipliers(state),
		"trait_multipliers": _world_to_trait_multipliers(state),
	}


def nonisomorphic_profile(profile: Mapping[str, Mapping[str, float]]) -> bool:
	for key in ("risk_multipliers", "reward_multipliers", "trait_multipliers"):
		for value in dict(profile.get(key, {})).values():
			if abs(float(value) - 1.0) > 1e-9:
				return True
	return False


class W1RoundAdapter:
	def __init__(self, *, config: W1CellConfig, seed: int):
		self.config = config
		self.seed = int(seed)
		self.state = dict(INITIAL_WORLD_STATE)
		self.window_rows: list[dict[str, Any]] = []
		self.event_counts: dict[str, int] = {family: 0 for family in EVENT_FAMILIES}
		self.world_update_rows: list[dict[str, Any]] = []
		self.window_index = 0

	def _apply_profile(self, loader: EventLoader) -> dict[str, dict[str, float]]:
		profile = world_profile_from_state(self.state)
		loader.set_event_type_weights(profile["event_type_weights"])
		loader.set_event_type_risk_multipliers(profile["risk_multipliers"])
		loader.set_event_type_reward_multipliers(profile["reward_multipliers"])
		loader.set_event_type_trait_delta_multipliers(profile["trait_multipliers"])
		return profile

	def _update_state(self) -> None:
		if str(self.config.condition) == "control" or float(self.config.lambda_world) == 0.0:
			self.state = dict(INITIAL_WORLD_STATE)
			return
		p_bar = [
			_safe_mean([float(row[f"p_{strategy}"]) for row in self.window_rows])
			for strategy in STRATEGY_ORDER
		]
		p_delta = [
			float(p_bar[idx]) - (1.0 / 3.0)
			for idx in range(len(STRATEGY_ORDER))
		]
		share_den = float(sum(self.event_counts.values()))
		if share_den <= 0.0:
			family_shares = [0.0 for _ in EVENT_FAMILIES]
		else:
			family_shares = [
				float(self.event_counts[family]) / share_den
				for family in EVENT_FAMILIES
			]
		reward_gap = _safe_mean([float(row.get("avg_reward") or 0.0) for row in self.window_rows]) - REWARD_TARGET
		for dim in WORLD_DIMENSIONS:
			delta_p = sum(B_P[dim][idx] * p_delta[idx] for idx in range(len(STRATEGY_ORDER)))
			delta_e = reward_gap * sum(B_E[dim][idx] * family_shares[idx] for idx in range(len(EVENT_FAMILIES)))
			self.state[dim] = _clamp(
				float(self.state[dim]) + float(self.config.lambda_world) * (delta_p + delta_e),
				0.0,
				1.0,
			)

	def _flush_window(self, *, round_index: int, dungeon: Any) -> None:
		if not self.window_rows:
			return
		self._update_state()
		profile = self._apply_profile(dungeon.event_loader)
		normalized_distribution = _normalize_positive_mapping(profile["event_type_weights"])
		dominant_event_type = ""
		if any(self.event_counts.values()):
			dominant_event_type = max(
				EVENT_FAMILIES,
				key=lambda family: (self.event_counts[family], family),
			)
		start_round = int(round_index + 1 - len(self.window_rows))
		end_round = int(round_index)
		self.world_update_rows.append(
			{
				"seed": self.seed,
				"window_index": self.window_index,
				"round": int(round_index + 1),
				"window_start_round": start_round,
				"window_end_round": end_round,
				"scarcity": float(self.state["scarcity"]),
				"threat": float(self.state["threat"]),
				"noise": float(self.state["noise"]),
				"intel": float(self.state["intel"]),
				"dominant_event_type": dominant_event_type,
				"a_new": float(dungeon.a),
				"b_new": float(dungeon.b),
				"event_distribution": json.dumps(normalized_distribution, sort_keys=True),
				"p_aggressive": _safe_mean([float(row["p_aggressive"]) for row in self.window_rows]),
				"p_defensive": _safe_mean([float(row["p_defensive"]) for row in self.window_rows]),
				"p_balanced": _safe_mean([float(row["p_balanced"]) for row in self.window_rows]),
				"mean_reward_window": _safe_mean([float(row.get("avg_reward") or 0.0) for row in self.window_rows]),
				"event_share_threat": float(self.event_counts["Threat"]) / float(sum(self.event_counts.values()) or 1),
				"event_share_resource": float(self.event_counts["Resource"]) / float(sum(self.event_counts.values()) or 1),
				"event_share_uncertainty": float(self.event_counts["Uncertainty"]) / float(sum(self.event_counts.values()) or 1),
				"event_share_navigation": float(self.event_counts["Navigation"]) / float(sum(self.event_counts.values()) or 1),
				"event_share_internal": float(self.event_counts["Internal"]) / float(sum(self.event_counts.values()) or 1),
				"state_deviated": bool(_state_deviated(self.state)),
				"risk_multipliers_json": json.dumps(profile["risk_multipliers"], sort_keys=True),
				"reward_multipliers_json": json.dumps(profile["reward_multipliers"], sort_keys=True),
				"trait_multipliers_json": json.dumps(profile["trait_multipliers"], sort_keys=True),
			}
		)
		self.window_rows = []
		self.event_counts = {family: 0 for family in EVENT_FAMILIES}
		self.window_index += 1

	def __call__(
		self,
		round_index: int,
		_cfg: SimConfig,
		_players: list[object],
		dungeon: Any,
		_step_records: list[dict[str, object]],
		row: dict[str, Any],
	) -> None:
		self.window_rows.append(
			{
				"p_aggressive": float(row.get("p_aggressive") or 0.0),
				"p_defensive": float(row.get("p_defensive") or 0.0),
				"p_balanced": float(row.get("p_balanced") or 0.0),
				"avg_reward": float(row.get("avg_reward") or 0.0),
			}
		)
		for event_type in json.loads(str(row.get("event_types_json") or "[]")):
			label = str(event_type)
			if label in self.event_counts:
				self.event_counts[label] = int(self.event_counts[label]) + 1
		if (round_index + 1) % int(self.config.world_update_interval) == 0:
			self._flush_window(round_index=round_index, dungeon=dungeon)


def run_w1_cell(config: W1CellConfig, seed: int) -> dict[str, Any]:
	config.out_dir.mkdir(parents=True, exist_ok=True)
	adapter = W1RoundAdapter(config=config, seed=int(seed))
	sim_cfg = SimConfig(
		n_players=int(config.players),
		n_rounds=int(config.rounds),
		seed=int(seed),
		payoff_mode="matrix_ab",
		popularity_mode="sampled",
		gamma=0.1,
		epsilon=0.0,
		a=float(config.a),
		b=float(config.b),
		matrix_cross_coupling=float(config.cross),
		init_bias=float(config.init_bias),
		evolution_mode="sampled",
		payoff_lag=1,
		selection_strength=float(config.selection_strength),
		enable_events=True,
		events_json=config.events_json,
		out_csv=config.out_csv_path(int(seed)),
		memory_kernel=int(config.memory_kernel),
	)
	strategy_space, rows = simulate(sim_cfg, round_callback=adapter)
	_write_timeseries_csv(sim_cfg.out_csv, strategy_space=strategy_space, rows=rows)
	metrics = seed_metrics(
		rows,
		burn_in=int(config.burn_in),
		tail=int(config.tail),
		eta=0.55,
		corr_threshold=0.09,
	)
	provenance = summarize_event_provenance(sim_cfg.out_csv, events_json=config.events_json)
	provenance_path = config.provenance_path(int(seed))
	provenance_path.write_text(json.dumps(provenance, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
	return {
		"seed": int(seed),
		"cycle_level": int(metrics["cycle_level"]),
		"stage3_score": float(metrics["stage3_score"]),
		"turn_strength": float(metrics["turn_strength"]),
		"env_gamma": float(metrics["env_gamma"]),
		"env_gamma_r2": float(metrics["env_gamma_r2"]),
		"env_gamma_n_peaks": int(metrics["env_gamma_n_peaks"]),
		"success_rate": float(metrics["success_rate"]),
		"out_csv": str(sim_cfg.out_csv),
		"provenance_json": str(provenance_path),
		"world_update_rows": list(adapter.world_update_rows),
		"world_updates_tsv": str(config.world_updates_path(int(seed))),
	}