from __future__ import annotations

import argparse
import csv
import json
import random
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

from analysis.event_provenance_summary import summarize_event_provenance
from analysis.metrics import average_reward, average_utility, strategy_distribution
from core.game_engine import GameEngine
from dungeon.dungeon_ai import DungeonAI
from dungeon.event_loader import EventLoader
from evolution.replicator_dynamics import replicator_step
from players.base_player import BasePlayer, DEFAULT_PERSONALITY_KEYS
from simulation.personality_gate0 import DEFAULT_FULL_EVENTS_JSON, projected_initial_weights, zero_personality
from simulation.personality_gate1 import seed_metrics
from simulation.run_simulation import _aggregate_event_records, _write_timeseries_csv


REPO_ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = "w2_episode"
STRATEGY_SPACE = ["aggressive", "defensive", "balanced"]
TAIL_LIFE_START = 3
DEFAULT_OUT_ROOT = REPO_ROOT / "outputs" / "w2_episode"
DEFAULT_LIFE_STEPS_TSV = REPO_ROOT / "outputs" / "w2_episode_life_steps.tsv"
DEFAULT_COMBINED_TSV = REPO_ROOT / "outputs" / "w2_episode_combined.tsv"
DEFAULT_DECISION_MD = REPO_ROOT / "outputs" / "w2_episode_decision.md"

DOMINANT_TEMPLATES = {
	"aggressive": {
		"impulsiveness": 0.18,
		"greed": 0.18,
		"ambition": 0.18,
	},
	"defensive": {
		"caution": 0.18,
		"stability_seeking": 0.18,
		"patience": 0.18,
	},
	"balanced": {
		"curiosity": 0.18,
		"optimism": 0.18,
		"persistence": 0.18,
	},
}

PERSONALITY_RISK_WEIGHTS = {
	"impulsiveness": 0.22,
	"caution": -0.25,
	"stability_seeking": -0.20,
	"fearfulness": 0.18,
}

TESTAMENT_WEIGHTS = {
	"util": 0.50,
	"dom": 0.35,
	"event": 0.15,
}

LIFE_STEP_FIELDNAMES = [
	"protocol",
	"condition",
	"seed",
	"life_index",
	"mean_personality_abs_shift",
	"mean_personality_l2_shift",
	"mean_personality_centroid_json",
	"ended_by_death",
	"n_deaths",
	"mean_life_rounds",
	"rounds_completed",
	"cycle_level",
	"mean_stage3_score",
	"mean_turn_strength",
	"mean_env_gamma",
	"level3_seed_count",
	"testament_alpha",
	"testament_applied",
	"dominant_strategy_last500",
	"mean_utility",
	"std_utility",
	"mean_success_rate",
	"mean_risk_final",
	"mean_threshold",
	"verdict",
	"out_csv",
	"provenance_json",
]

COMBINED_FIELDNAMES = [
	"protocol",
	"condition",
	"is_control",
	"n_seeds",
	"total_lives",
	"tail_life_start",
	"testament_alpha",
	"mean_stage3_score_all_lives",
	"mean_env_gamma_all_lives",
	"mean_stage3_score_tail_lives",
	"mean_env_gamma_tail_lives",
	"tail_level3_seed_count",
	"first_level3_life",
	"first_tail_level3_life",
	"tail_p_level_3",
	"mean_n_deaths",
	"mean_death_rate",
	"mean_life_rounds",
	"tail_mean_n_deaths",
	"tail_mean_death_rate",
	"tail_death_rate_band_ok",
	"tail_mean_rounds_completed",
	"testament_activation_rate",
	"control_tail_mean_env_gamma",
	"control_tail_mean_stage3_score",
	"tail_gamma_uplift_vs_control",
	"tail_stage3_uplift_vs_control",
	"tail_mean_personality_abs_shift",
	"tail_mean_personality_l2_shift",
	"tail_mean_personality_centroid_json",
	"tail_gamma_nonnegative",
	"tail_level3_pass",
	"full_pass",
	"verdict",
	"events_json",
	"selection_strength",
	"memory_kernel",
	"out_dir",
]

CellRunner = Callable[["W2CellConfig", int], Mapping[str, Any]]


@dataclass(frozen=True)
class W2CellConfig:
	condition: str
	testament_alpha: float
	total_lives: int
	rounds_per_life: int
	players: int
	events_json: Path
	selection_strength: float
	init_bias: float
	memory_kernel: int
	burn_in: int
	tail: int
	a: float
	b: float
	cross: float
	out_dir: Path

	def life_out_csv(self, seed: int, life_index: int) -> Path:
		return self.out_dir / f"seed{int(seed)}_life{int(life_index)}.csv"

	def provenance_path(self, seed: int, life_index: int) -> Path:
		return self.out_dir / f"seed{int(seed)}_life{int(life_index)}_provenance.json"

	@property
	def is_control(self) -> bool:
		return str(self.condition) == "control"


def _parse_seeds(spec: str) -> list[int]:
	parts = [part.strip() for part in str(spec).split(",") if part.strip()]
	if not parts:
		raise ValueError("--seeds cannot be empty")
	return [int(part) for part in parts]


def _clamp(value: float, lower: float, upper: float) -> float:
	return max(lower, min(upper, float(value)))


def _safe_mean(values: list[float]) -> float:
	if not values:
		return 0.0
	return float(sum(values) / float(len(values)))


def _yes_no(value: bool) -> str:
	return "yes" if bool(value) else "no"


def _format_float(value: float) -> str:
	return f"{float(value):.6f}"


def _write_tsv(path: Path, *, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
	path.parent.mkdir(parents=True, exist_ok=True)
	with path.open("w", newline="", encoding="utf-8") as handle:
		writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
		writer.writeheader()
		writer.writerows(rows)


def _personality_distribution_stats(personalities: list[Mapping[str, float]]) -> dict[str, Any]:
	if not personalities:
		return {
			"mean_personality_abs_shift": 0.0,
			"mean_personality_l2_shift": 0.0,
			"mean_personality_centroid": _zero_vector(),
		}
	centroid = _zero_vector()
	abs_shifts: list[float] = []
	l2_shifts: list[float] = []
	for personality in personalities:
		for key in DEFAULT_PERSONALITY_KEYS:
			centroid[key] += float(personality.get(key, 0.0))
		values = [float(personality.get(key, 0.0)) for key in DEFAULT_PERSONALITY_KEYS]
		abs_shifts.append(sum(abs(value) for value in values) / float(len(values) or 1))
		l2_shifts.append(sum(value * value for value in values) ** 0.5)
	den = float(len(personalities))
	centroid = {key: float(value) / den for key, value in centroid.items()}
	return {
		"mean_personality_abs_shift": _safe_mean(abs_shifts),
		"mean_personality_l2_shift": _safe_mean(l2_shifts),
		"mean_personality_centroid": centroid,
	}


def _mean_centroid_json(rows: list[dict[str, Any]]) -> str:
	if not rows:
		return json.dumps(_zero_vector(), sort_keys=True)
	acc = _zero_vector()
	count = 0
	for row in rows:
		text = str(row.get("mean_personality_centroid_json", "") or "").strip()
		if not text:
			continue
		parsed = json.loads(text)
		if not isinstance(parsed, dict):
			continue
		for key in DEFAULT_PERSONALITY_KEYS:
			acc[key] += float(parsed.get(key, 0.0))
		count += 1
	if count <= 0:
		return json.dumps(_zero_vector(), sort_keys=True)
	return json.dumps({key: acc[key] / float(count) for key in DEFAULT_PERSONALITY_KEYS}, sort_keys=True)


def _zero_vector() -> dict[str, float]:
	return {key: 0.0 for key in DEFAULT_PERSONALITY_KEYS}


def _copy_personality(personality: Mapping[str, float]) -> dict[str, float]:
	base = _zero_vector()
	for key in DEFAULT_PERSONALITY_KEYS:
		base[key] = _clamp(float(personality.get(key, 0.0)), -1.0, 1.0)
	return base


def _scale_vector(vector: Mapping[str, float], factor: float) -> dict[str, float]:
	return {key: float(value) * float(factor) for key, value in vector.items()}


def _add_vectors(*vectors: Mapping[str, float]) -> dict[str, float]:
	out = _zero_vector()
	for vector in vectors:
		for key, value in vector.items():
			if key in out:
				out[key] += float(value)
	return out


def _clip_vector(vector: Mapping[str, float], lower: float, upper: float) -> dict[str, float]:
	return {
		key: _clamp(float(vector.get(key, 0.0)), float(lower), float(upper))
		for key in DEFAULT_PERSONALITY_KEYS
	}


def _mean_player_weights(players: list[BasePlayer]) -> dict[str, float]:
	if not players:
		return {strategy: 0.0 for strategy in STRATEGY_SPACE}
	totals = {strategy: 0.0 for strategy in STRATEGY_SPACE}
	for player in players:
		for strategy in STRATEGY_SPACE:
			totals[strategy] += float(player.strategy_weights.get(strategy, 0.0))
	count = float(len(players))
	return {strategy: float(totals[strategy]) / count for strategy in STRATEGY_SPACE}


def _dominant_strategy(history: list[str]) -> str | None:
	if not history:
		return None
	window = history[-500:]
	counts = {strategy: 0 for strategy in STRATEGY_SPACE}
	for strategy in window:
		if strategy in counts:
			counts[strategy] += 1
	best = max(counts.values())
	if best <= 0:
		return None
	candidates = [strategy for strategy, count in counts.items() if count == best]
	return candidates[0]


def _normalize_template(template: Mapping[str, float]) -> dict[str, float]:
	den = sum(abs(float(value)) for value in template.values())
	if den <= 0.0:
		return _zero_vector()
	out = _zero_vector()
	for key, value in template.items():
		if key in out:
			out[key] = float(value) / den
	return out


def _player_personality(obj: object | Mapping[str, float]) -> Mapping[str, float]:
	if isinstance(obj, Mapping):
		return obj
	return getattr(obj, "personality")


def compute_personality_risk_delta(player_or_personality: object | Mapping[str, float]) -> float:
	personality = _player_personality(player_or_personality)
	return float(
		sum(
			float(personality.get(key, 0.0)) * float(weight)
			for key, weight in PERSONALITY_RISK_WEIGHTS.items()
		)
	)


def compute_death_threshold(player_or_personality: object | Mapping[str, float]) -> float:
	personality = _player_personality(player_or_personality)
	return float(
		1.0
		+ 0.15
		* (
			float(personality.get("caution", 0.0))
			+ float(personality.get("stability_seeking", 0.0))
			- float(personality.get("impulsiveness", 0.0))
			- float(personality.get("fearfulness", 0.0))
		)
	)


def _success_rate(player: BasePlayer) -> float:
	event_count = int(getattr(player, "w2_event_count", 0))
	if event_count <= 0:
		return 0.0
	return float(getattr(player, "w2_event_success_count", 0)) / float(event_count)


def _utility_reference(players: list[BasePlayer]) -> tuple[float, float]:
	utilities = [float(player.utility) for player in players]
	mean_utility = _safe_mean(utilities)
	std_utility = float(statistics.pstdev(utilities)) if len(utilities) >= 2 else 0.0
	return mean_utility, std_utility


def _next_personality_from_player(
	player: BasePlayer,
	*,
	alpha: float,
	mean_utility: float,
	std_utility: float,
) -> dict[str, float]:
	if float(alpha) == 0.0:
		return _copy_personality(player.personality)
	normalizer = max(float(std_utility), 1e-6)
	dominant = _dominant_strategy(list(getattr(player, "w2_strategy_history", [])))
	delta_dom = _zero_vector()
	if dominant is not None:
		delta_dom = _add_vectors(DOMINANT_TEMPLATES.get(dominant, {}))
	normalized_dom = _normalize_template(delta_dom)
	if dominant is None:
		delta_util = _zero_vector()
	else:
		z_util = _clamp((float(player.utility) - float(mean_utility)) / normalizer, -1.0, 1.0)
		delta_util = _scale_vector(normalized_dom, z_util)
	event_count = int(getattr(player, "w2_event_count", 0))
	if event_count <= 0:
		delta_event = _zero_vector()
	else:
		mean_event = _scale_vector(
			getattr(player, "w2_trait_delta_sum", _zero_vector()),
			1.0 / float(event_count),
		)
		delta_event = _scale_vector(mean_event, _success_rate(player))
	delta = _add_vectors(
		_scale_vector(delta_util, TESTAMENT_WEIGHTS["util"]),
		_scale_vector(delta_dom, TESTAMENT_WEIGHTS["dom"]),
		_scale_vector(delta_event, TESTAMENT_WEIGHTS["event"]),
	)
	delta = _clip_vector(delta, -0.25, 0.25)
	return {
		key: _clamp(float(player.personality.get(key, 0.0)) + float(alpha) * float(delta[key]), -1.0, 1.0)
		for key in DEFAULT_PERSONALITY_KEYS
	}


def apply_testament(players: list[BasePlayer], alpha: float) -> list[dict[str, float]]:
	mean_utility, std_utility = _utility_reference(players)
	return [
		_next_personality_from_player(
			player,
			alpha=float(alpha),
			mean_utility=float(mean_utility),
			std_utility=float(std_utility),
		)
		for player in players
	]


def reset_player_for_next_life(player: BasePlayer, next_personality: Mapping[str, float], *, init_bias: float) -> None:
	player.personality = _copy_personality(next_personality)
	player.state = {
		"risk": 0.0,
		"stress": 0.0,
		"noise": 0.0,
		"risk_drift": 0.0,
		"health": 1.0,
		"intel": 0.0,
	}
	player.utility = 0.0
	player.last_strategy = None
	player.last_reward = None
	player.strategy_biases = {strategy: 0.0 for strategy in player.strategy_space}
	player.update_weights(projected_initial_weights(player.personality, init_bias=float(init_bias)))
	setattr(player, "w2_strategy_history", [])
	setattr(player, "w2_trait_delta_sum", _zero_vector())
	setattr(player, "w2_event_count", 0)
	setattr(player, "w2_event_success_count", 0)
	setattr(player, "w2_active_rounds", 0)
	setattr(player, "w2_dead", False)


def _make_players(config: W2CellConfig, seed: int, personalities: list[dict[str, float]]) -> list[BasePlayer]:
	players = [
		BasePlayer(STRATEGY_SPACE, rng=random.Random(int(seed) + idx))
		for idx in range(int(config.players))
	]
	for player, personality in zip(players, personalities, strict=True):
		reset_player_for_next_life(player, personality, init_bias=float(config.init_bias))
	return players


def _life_metrics(rows: list[dict[str, Any]], *, burn_in: int, tail: int) -> dict[str, float | int]:
	if not rows:
		return {
			"cycle_level": 0,
			"stage3_score": 0.0,
			"turn_strength": 0.0,
			"env_gamma": 0.0,
			"env_gamma_r2": 0.0,
			"env_gamma_n_peaks": 0,
			"success_rate": 0.0,
		}
	return seed_metrics(rows, burn_in=int(burn_in), tail=int(tail), eta=0.55, corr_threshold=0.09)


def _dominant_strategy_for_life(players: list[BasePlayer]) -> str:
	counts = {strategy: 0 for strategy in STRATEGY_SPACE}
	for player in players:
		dominant = _dominant_strategy(list(getattr(player, "w2_strategy_history", [])))
		if dominant is not None:
			counts[dominant] += 1
	best = max(counts.values()) if counts else 0
	if best <= 0:
		return ""
	for strategy in STRATEGY_SPACE:
		if counts[strategy] == best:
			return strategy
	return ""


def _dominant_strategy_from_labels(labels: list[str | None]) -> str:
	counts = {strategy: 0 for strategy in STRATEGY_SPACE}
	for label in labels:
		if label in counts:
			counts[str(label)] += 1
	best = max(counts.values()) if counts else 0
	if best <= 0:
		return ""
	for strategy in STRATEGY_SPACE:
		if counts[strategy] == best:
			return strategy
	return ""


def run_life(
	config: W2CellConfig,
	*,
	seed: int,
	life_index: int,
	players: list[BasePlayer],
	starting_personalities: list[Mapping[str, float]],
) -> dict[str, Any]:
	event_loader = EventLoader(config.events_json)
	dungeon = DungeonAI(
		payoff_mode="matrix_ab",
		gamma=0.1,
		epsilon=0.0,
		a=float(config.a),
		b=float(config.b),
		matrix_cross_coupling=float(config.cross),
		memory_kernel=int(config.memory_kernel),
		strategy_cycle=STRATEGY_SPACE,
		event_loader=event_loader,
		event_rng=random.Random(int(seed) * 1000 + int(life_index)),
	)
	engine = GameEngine(players, dungeon, popularity_mode="sampled")
	alive_ids = list(range(len(players)))
	rows: list[dict[str, Any]] = []
	n_deaths = 0
	pending_next_personalities: list[dict[str, float] | None] = [None for _ in players]
	final_utilities: list[float | None] = [None for _ in players]
	final_risks: list[float | None] = [None for _ in players]
	final_thresholds: list[float | None] = [None for _ in players]
	final_rounds: list[float | None] = [None for _ in players]
	final_dominants: list[str | None] = [None for _ in players]
	for round_index in range(int(config.rounds_per_life)):
		if not alive_ids:
			break
		active_players = [players[idx] for idx in alive_ids]
		personality_snapshots = [_copy_personality(player.personality) for player in active_players]
		engine.players = active_players
		step_records = engine.step()
		mean_weights = _mean_player_weights(active_players)
		dist = strategy_distribution(active_players, STRATEGY_SPACE)
		avg_u = average_utility(active_players)
		avg_r = average_reward(active_players)
		alive_next: list[int] = []
		mean_utility, std_utility = _utility_reference(players)
		for player_idx, player, record, personality_snapshot in zip(alive_ids, active_players, step_records, personality_snapshots, strict=True):
			setattr(player, "w2_active_rounds", int(getattr(player, "w2_active_rounds", 0)) + 1)
			history = list(getattr(player, "w2_strategy_history", []))
			history.append(str(record.get("strategy", player.last_strategy or "")))
			setattr(player, "w2_strategy_history", history)
			event_result = record.get("event_result")
			if isinstance(event_result, dict):
				setattr(player, "w2_event_count", int(getattr(player, "w2_event_count", 0)) + 1)
				if bool(event_result.get("success")):
					setattr(player, "w2_event_success_count", int(getattr(player, "w2_event_success_count", 0)) + 1)
				trait_sum = _add_vectors(
					getattr(player, "w2_trait_delta_sum", _zero_vector()),
					dict(event_result.get("trait_deltas", {})),
				)
				setattr(player, "w2_trait_delta_sum", trait_sum)
			player.personality = personality_snapshot
			player.state["risk"] = float(player.state.get("risk", 0.0)) + compute_personality_risk_delta(player)
			threshold = compute_death_threshold(player)
			player.state["w2_death_threshold"] = threshold
			if float(player.state.get("risk", 0.0)) >= threshold:
				setattr(player, "w2_dead", True)
				n_deaths += 1
				final_utilities[int(player_idx)] = float(player.utility)
				final_risks[int(player_idx)] = float(player.state.get("risk", 0.0))
				final_thresholds[int(player_idx)] = float(threshold)
				final_rounds[int(player_idx)] = float(getattr(player, "w2_active_rounds", 0))
				final_dominants[int(player_idx)] = _dominant_strategy(list(getattr(player, "w2_strategy_history", [])))
				if not config.is_control:
					pending_next_personality = _next_personality_from_player(
						player,
						alpha=float(config.testament_alpha),
						mean_utility=float(mean_utility),
						std_utility=float(std_utility),
					)
					pending_next_personalities[int(player_idx)] = pending_next_personality
					if int(life_index) < int(config.total_lives):
						reset_player_for_next_life(player, pending_next_personality, init_bias=float(config.init_bias))
						setattr(player, "w2_dead", True)
			else:
				alive_next.append(int(player_idx))
		new_weights = replicator_step(active_players, STRATEGY_SPACE, selection_strength=float(config.selection_strength))
		for player in active_players:
			if not bool(getattr(player, "w2_dead", False)):
				player.update_weights(new_weights)
		row = {
			"round": int(round_index),
			"avg_reward": "" if avg_r is None else float(avg_r),
			"avg_utility": float(avg_u),
			"threshold_regime_hi": "",
			"threshold_state_value": "",
		}
		for strategy in STRATEGY_SPACE:
			row[f"p_{strategy}"] = float(dist[strategy])
		for strategy in STRATEGY_SPACE:
			row[f"w_{strategy}"] = float(mean_weights[strategy])
		row.update(_aggregate_event_records(step_records))
		rows.append(row)
		alive_ids = alive_next

	rounds_completed = len(rows)
	for idx, player in enumerate(players):
		if final_utilities[idx] is not None:
			continue
		final_utilities[idx] = float(player.utility)
		final_risks[idx] = float(player.state.get("risk", 0.0))
		final_thresholds[idx] = float(player.state.get("w2_death_threshold", compute_death_threshold(player)))
		final_rounds[idx] = float(getattr(player, "w2_active_rounds", 0))
		final_dominants[idx] = _dominant_strategy(list(getattr(player, "w2_strategy_history", [])))
	out_csv = config.life_out_csv(seed, life_index)
	_write_timeseries_csv(out_csv, strategy_space=STRATEGY_SPACE, rows=rows)
	provenance = summarize_event_provenance(out_csv, events_json=config.events_json)
	provenance_path = config.provenance_path(seed, life_index)
	provenance_path.write_text(json.dumps(provenance, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
	metrics = _life_metrics(rows, burn_in=min(int(config.burn_in), rounds_completed), tail=min(int(config.tail), rounds_completed or int(config.tail)))
	utilities = [float(value) for value in final_utilities if value is not None]
	mean_life_rounds = _safe_mean([float(value) for value in final_rounds if value is not None])
	mean_threshold = _safe_mean([float(value) for value in final_thresholds if value is not None])
	mean_risk_final = _safe_mean([float(value) for value in final_risks if value is not None])
	personality_stats = _personality_distribution_stats(list(starting_personalities))
	life_verdict = "control" if config.is_control else ("level3" if int(metrics["cycle_level"]) >= 3 else "non_level3")
	return {
		"protocol": PROTOCOL,
		"condition": str(config.condition),
		"seed": int(seed),
		"life_index": int(life_index),
		"mean_personality_abs_shift": _format_float(float(personality_stats["mean_personality_abs_shift"])),
		"mean_personality_l2_shift": _format_float(float(personality_stats["mean_personality_l2_shift"])),
		"mean_personality_centroid_json": json.dumps(personality_stats["mean_personality_centroid"], sort_keys=True),
		"ended_by_death": _yes_no(n_deaths > 0),
		"n_deaths": int(n_deaths),
		"mean_life_rounds": _format_float(mean_life_rounds),
		"rounds_completed": int(rounds_completed),
		"cycle_level": int(metrics["cycle_level"]),
		"mean_stage3_score": _format_float(float(metrics["stage3_score"])),
		"mean_turn_strength": _format_float(float(metrics["turn_strength"])),
		"mean_env_gamma": _format_float(float(metrics["env_gamma"])),
		"level3_seed_count": int(int(metrics["cycle_level"]) >= 3),
		"testament_alpha": _format_float(float(config.testament_alpha)),
		"testament_applied": _yes_no((not config.is_control) and life_index < int(config.total_lives)),
		"dominant_strategy_last500": _dominant_strategy_from_labels(final_dominants),
		"mean_utility": _format_float(_safe_mean(utilities)),
		"std_utility": _format_float(float(statistics.pstdev(utilities)) if len(utilities) >= 2 else 0.0),
		"mean_success_rate": _format_float(float(metrics["success_rate"])),
		"mean_risk_final": _format_float(mean_risk_final),
		"mean_threshold": _format_float(mean_threshold),
		"verdict": life_verdict,
		"out_csv": str(out_csv),
		"provenance_json": str(provenance_path),
		"next_personalities": pending_next_personalities,
	}


def run_w2_cell(config: W2CellConfig, seed: int) -> dict[str, Any]:
	initial_personalities = [zero_personality() for _ in range(int(config.players))]
	current_personalities = [_copy_personality(personality) for personality in initial_personalities]
	players = _make_players(config, seed, current_personalities)
	life_rows: list[dict[str, Any]] = []
	for life_index in range(1, int(config.total_lives) + 1):
		starting_personalities = [_copy_personality(personality) for personality in current_personalities]
		for player, personality in zip(players, current_personalities, strict=True):
			reset_player_for_next_life(player, personality, init_bias=float(config.init_bias))
			setattr(player, "w2_dead", False)
		life_result = run_life(
			config,
			seed=int(seed),
			life_index=int(life_index),
			players=players,
			starting_personalities=starting_personalities,
		)
		pending_next_personalities = list(life_result.pop("next_personalities"))
		life_rows.append(life_result)
		if life_index >= int(config.total_lives):
			continue
		if config.is_control:
			current_personalities = [_copy_personality(personality) for personality in initial_personalities]
		else:
			survivor_indices = [idx for idx, pending in enumerate(pending_next_personalities) if pending is None]
			boundary_next_personalities: list[dict[str, float] | None] = [None for _ in players]
			if survivor_indices:
				survivors = [players[idx] for idx in survivor_indices]
				mean_utility, std_utility = _utility_reference(survivors)
				for idx in survivor_indices:
					boundary_next_personalities[idx] = _next_personality_from_player(
						players[idx],
						alpha=float(config.testament_alpha),
						mean_utility=float(mean_utility),
						std_utility=float(std_utility),
					)
			current_personalities = [
				pending if pending is not None else boundary
				for pending, boundary in zip(pending_next_personalities, boundary_next_personalities, strict=True)
			]
	return {
		"seed": int(seed),
		"condition": str(config.condition),
		"life_rows": life_rows,
	}


def _build_cell_config(
	*,
	condition: str,
	testament_alpha: float,
	total_lives: int,
	rounds_per_life: int,
	players: int,
	events_json: Path,
	selection_strength: float,
	init_bias: float,
	memory_kernel: int,
	burn_in: int,
	tail: int,
	a: float,
	b: float,
	cross: float,
	out_root: Path,
) -> W2CellConfig:
	return W2CellConfig(
		condition=str(condition),
		testament_alpha=float(testament_alpha),
		total_lives=int(total_lives),
		rounds_per_life=int(rounds_per_life),
		players=int(players),
		events_json=Path(events_json),
		selection_strength=float(selection_strength),
		init_bias=float(init_bias),
		memory_kernel=int(memory_kernel),
		burn_in=int(burn_in),
		tail=int(tail),
		a=float(a),
		b=float(b),
		cross=float(cross),
		out_dir=out_root / str(condition),
	)


def _cell_summary(
	row_group: list[dict[str, Any]],
	*,
	control_row: dict[str, Any] | None,
	config: W2CellConfig,
	tail_life_start: int,
) -> dict[str, Any]:
	tail_rows = [row for row in row_group if int(row["life_index"]) >= int(tail_life_start)]
	if config.is_control and not tail_rows:
		tail_rows = list(row_group)
	tail_count = len(tail_rows)
	tail_level3_seed_count = sum(int(row["level3_seed_count"]) for row in tail_rows)
	first_level3_life = min((int(row["life_index"]) for row in row_group if int(row["level3_seed_count"]) > 0), default=None)
	first_tail_level3_life = min((int(row["life_index"]) for row in tail_rows if int(row["level3_seed_count"]) > 0), default=None)
	tail_p_level_3 = 0.0 if tail_count == 0 else float(tail_level3_seed_count) / float(tail_count)
	tail_stage3 = _safe_mean([float(row["mean_stage3_score"]) for row in tail_rows])
	tail_gamma = _safe_mean([float(row["mean_env_gamma"]) for row in tail_rows])
	tail_mean_n_deaths = _safe_mean([float(row["n_deaths"]) for row in tail_rows])
	tail_mean_death_rate = 0.0 if int(config.players) <= 0 else float(tail_mean_n_deaths) / float(config.players)
	tail_mean_rounds_completed = _safe_mean([float(row["rounds_completed"]) for row in tail_rows])
	tail_mean_personality_abs_shift = _safe_mean([float(row["mean_personality_abs_shift"]) for row in tail_rows])
	tail_mean_personality_l2_shift = _safe_mean([float(row["mean_personality_l2_shift"]) for row in tail_rows])
	mean_n_deaths = _safe_mean([float(row["n_deaths"]) for row in row_group])
	mean_death_rate = 0.0 if int(config.players) <= 0 else float(mean_n_deaths) / float(config.players)
	tail_death_rate_band_ok = 0.15 <= float(tail_mean_death_rate) <= 0.40
	control_tail_gamma = 0.0 if control_row is None else float(control_row["mean_env_gamma_tail_lives"])
	control_tail_stage3 = 0.0 if control_row is None else float(control_row["mean_stage3_score_tail_lives"])
	tail_level3_pass = tail_level3_seed_count >= 1
	tail_gamma_nonnegative = tail_gamma >= 0.0
	full_pass = tail_level3_pass and tail_gamma_nonnegative
	if config.is_control:
		verdict = "control"
	elif full_pass:
		verdict = "pass"
	elif tail_level3_seed_count == 0 and tail_gamma > control_tail_gamma and _safe_mean([float(row["n_deaths"]) for row in row_group]) > 0.0:
		verdict = "weak_positive"
	else:
		verdict = "fail"
	return {
		"protocol": PROTOCOL,
		"condition": str(config.condition),
		"is_control": _yes_no(config.is_control),
		"n_seeds": len({int(row["seed"]) for row in row_group}),
		"total_lives": int(config.total_lives),
		"tail_life_start": int(tail_life_start),
		"testament_alpha": _format_float(float(config.testament_alpha)),
		"mean_stage3_score_all_lives": _format_float(_safe_mean([float(row["mean_stage3_score"]) for row in row_group])),
		"mean_env_gamma_all_lives": _format_float(_safe_mean([float(row["mean_env_gamma"]) for row in row_group])),
		"mean_stage3_score_tail_lives": _format_float(tail_stage3),
		"mean_env_gamma_tail_lives": _format_float(tail_gamma),
		"tail_level3_seed_count": int(tail_level3_seed_count),
		"first_level3_life": "" if first_level3_life is None else int(first_level3_life),
		"first_tail_level3_life": "" if first_tail_level3_life is None else int(first_tail_level3_life),
		"tail_p_level_3": _format_float(tail_p_level_3),
		"mean_n_deaths": _format_float(mean_n_deaths),
		"mean_death_rate": _format_float(mean_death_rate),
		"mean_life_rounds": _format_float(_safe_mean([float(row["mean_life_rounds"]) for row in row_group])),
		"tail_mean_n_deaths": _format_float(tail_mean_n_deaths),
		"tail_mean_death_rate": _format_float(tail_mean_death_rate),
		"tail_death_rate_band_ok": _yes_no(tail_death_rate_band_ok),
		"tail_mean_rounds_completed": _format_float(tail_mean_rounds_completed),
		"testament_activation_rate": _format_float(
			_safe_mean([1.0 if row["testament_applied"] == "yes" else 0.0 for row in row_group])
		),
		"control_tail_mean_env_gamma": _format_float(control_tail_gamma),
		"control_tail_mean_stage3_score": _format_float(control_tail_stage3),
		"tail_gamma_uplift_vs_control": _format_float(tail_gamma - control_tail_gamma),
		"tail_stage3_uplift_vs_control": _format_float(tail_stage3 - control_tail_stage3),
		"tail_mean_personality_abs_shift": _format_float(tail_mean_personality_abs_shift),
		"tail_mean_personality_l2_shift": _format_float(tail_mean_personality_l2_shift),
		"tail_mean_personality_centroid_json": _mean_centroid_json(tail_rows),
		"tail_gamma_nonnegative": _yes_no(tail_gamma_nonnegative),
		"tail_level3_pass": _yes_no(tail_level3_pass),
		"full_pass": _yes_no(full_pass),
		"verdict": verdict,
		"events_json": str(config.events_json),
		"selection_strength": _format_float(float(config.selection_strength)),
		"memory_kernel": int(config.memory_kernel),
		"out_dir": str(config.out_dir),
	}


def _write_decision(
	path: Path,
	*,
	decision: str,
	control_row: Mapping[str, Any],
	combined_rows: list[dict[str, Any]],
	tail_life_start: int,
	tail_life_end: int,
) -> None:
	lines = [
		"# W2.1 Decision",
		"",
		f"- protocol: {PROTOCOL}",
		f"- decision: {decision}",
		f"- tail_life_window: {int(tail_life_start)}~{int(tail_life_end)}",
		f"- control_tail_mean_stage3_score: {control_row['mean_stage3_score_tail_lives']}",
		f"- control_tail_mean_env_gamma: {control_row['mean_env_gamma_tail_lives']}",
		f"- control_tail_level3_seed_count: {control_row['tail_level3_seed_count']}",
		"",
		"## Tail-Life Verdicts",
		"",
		"| Cell | first tail Level 3 life | tail Level 3 seeds | tail mean_stage3_score | tail mean_env_gamma | tail death rate | tail mean rounds | tail personality abs shift | verdict |",
		"|---|---:|---:|---:|---:|---:|---:|---:|---|",
	]
	for row in combined_rows:
		lines.append(
			f"| {row['condition']} | {row['first_tail_level3_life']} | {row['tail_level3_seed_count']} | {row['mean_stage3_score_tail_lives']} | {row['mean_env_gamma_tail_lives']} | {row['tail_mean_death_rate']} | {row['tail_mean_rounds_completed']} | {row['tail_mean_personality_abs_shift']} | {row['verdict']} |"
		)
	lines.extend(
		[
			"",
			"## Focus Metrics",
			"",
			f"- 後半段 life（{int(tail_life_start)}~{int(tail_life_end)}）的 mean_env_gamma 是否轉正：看 tail mean_env_gamma 與 tail_gamma_uplift_vs_control。",
			f"- 是否在 life {int(tail_life_start)}~{int(tail_life_end)} 首次出現 Level 3 seed：看 first tail Level 3 life 與 tail Level 3 seeds。",
			"- n_deaths 是否合理：看 tail mean death rate，理想區間為 15%~40%。",
			"- personality 是否偏離初值：看 tail personality abs shift 與 tail_mean_personality_centroid_json。",
			"",
			"## Mechanism Evidence",
			"",
		]
	)
	for row in combined_rows:
		lines.append(
			f"- {row['condition']}: mean_n_deaths={row['mean_n_deaths']} mean_death_rate={row['mean_death_rate']} tail_mean_n_deaths={row['tail_mean_n_deaths']} tail_mean_death_rate={row['tail_mean_death_rate']} tail_death_rate_band_ok={row['tail_death_rate_band_ok']} tail_mean_rounds={row['tail_mean_rounds_completed']} testament_activation_rate={row['testament_activation_rate']} tail_personality_abs_shift={row['tail_mean_personality_abs_shift']} tail_personality_centroid={row['tail_mean_personality_centroid_json']}"
		)
	lines.extend(
		[
			"",
			"## Gate",
			"",
			f"- pass: 至少 1 個 non-control cell 在 life {int(tail_life_start)}~{int(tail_life_end)} 出現 >=1 個 Level 3 seed，且 tail mean_env_gamma >= 0。",
			"- close_w2_1: 若後半段 life 完全沒有 Level 3 seed，則不再微調 alpha，直接結案 W2.1。",
		]
	)
	path.parent.mkdir(parents=True, exist_ok=True)
	path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _decide_overall(combined_rows: list[dict[str, Any]]) -> str:
	candidate_rows = [row for row in combined_rows if row["is_control"] == "no"]
	if any(row["full_pass"] == "yes" for row in candidate_rows):
		return "pass"
	if all(int(row["tail_level3_seed_count"]) == 0 for row in candidate_rows):
		return "close_w2_1"
	if any(row["verdict"] == "weak_positive" for row in candidate_rows):
		return "weak_positive"
	return "fail"


def run_w2_scout(
	*,
	seeds: list[int],
	out_root: Path,
	life_steps_tsv: Path,
	combined_tsv: Path,
	decision_md: Path,
	conditions: list[str] | None = None,
	events_json: Path = DEFAULT_FULL_EVENTS_JSON,
	players: int = 300,
	rounds_per_life: int = 3000,
	total_lives: int = 5,
	tail_life_start: int = TAIL_LIFE_START,
	selection_strength: float = 0.06,
	init_bias: float = 0.12,
	memory_kernel: int = 3,
	burn_in: int = 1000,
	tail: int = 1000,
	a: float = 1.0,
	b: float = 0.9,
	cross: float = 0.20,
	cell_runner: CellRunner | None = None,
) -> dict[str, Any]:
	runner = cell_runner or run_w2_cell
	selected_conditions = set(str(value) for value in conditions) if conditions is not None else None
	out_root.mkdir(parents=True, exist_ok=True)
	cell_configs = [
		_build_cell_config(
			condition="control",
			testament_alpha=0.0,
			total_lives=1,
			rounds_per_life=rounds_per_life,
			players=players,
			events_json=events_json,
			selection_strength=selection_strength,
			init_bias=init_bias,
			memory_kernel=memory_kernel,
			burn_in=burn_in,
			tail=tail,
			a=a,
			b=b,
			cross=cross,
			out_root=out_root,
		),
		_build_cell_config(
			condition="w2_base",
			testament_alpha=0.12,
			total_lives=total_lives,
			rounds_per_life=rounds_per_life,
			players=players,
			events_json=events_json,
			selection_strength=selection_strength,
			init_bias=init_bias,
			memory_kernel=memory_kernel,
			burn_in=burn_in,
			tail=tail,
			a=a,
			b=b,
			cross=cross,
			out_root=out_root,
		),
		_build_cell_config(
			condition="w2_strong",
			testament_alpha=0.22,
			total_lives=total_lives,
			rounds_per_life=rounds_per_life,
			players=players,
			events_json=events_json,
			selection_strength=selection_strength,
			init_bias=init_bias,
			memory_kernel=memory_kernel,
			burn_in=burn_in,
			tail=tail,
			a=a,
			b=b,
			cross=cross,
			out_root=out_root,
		),
	]
	if selected_conditions is not None:
		cell_configs = [config for config in cell_configs if config.condition in selected_conditions]
		if not cell_configs:
			raise ValueError("no W2 cells selected")
	by_condition: dict[str, list[dict[str, Any]]] = {config.condition: [] for config in cell_configs}
	for config in cell_configs:
		for seed in seeds:
			result = runner(config, int(seed))
			life_rows = list(result.get("life_rows", []))
			by_condition[config.condition].extend(life_rows)
	life_rows_flat = [row for config in cell_configs for row in by_condition[config.condition]]
	_write_tsv(life_steps_tsv, fieldnames=LIFE_STEP_FIELDNAMES, rows=life_rows_flat)
	control_config = next((config for config in cell_configs if config.condition == "control"), None)
	control_rows = by_condition.get("control", [])
	control_base = None
	combined_rows: list[dict[str, Any]] = []
	if control_config is not None:
		control_base = _cell_summary(control_rows, control_row=None, config=control_config, tail_life_start=int(tail_life_start))
		combined_rows.append(control_base)
	for config in cell_configs:
		if config.condition == "control":
			continue
		combined_rows.append(
			_cell_summary(
				by_condition[config.condition],
				control_row=control_base,
				config=config,
				tail_life_start=int(tail_life_start),
			)
		)
	if control_config is not None:
		combined_rows[0] = _cell_summary(control_rows, control_row=combined_rows[0], config=control_config, tail_life_start=int(tail_life_start))
	_write_tsv(combined_tsv, fieldnames=COMBINED_FIELDNAMES, rows=combined_rows)
	if len(combined_rows) == 1 and combined_rows[0]["is_control"] == "yes":
		decision = "control_only"
		control_row = combined_rows[0]
	else:
		decision = _decide_overall(combined_rows)
		control_row = next(row for row in combined_rows if row["is_control"] == "yes")
	_write_decision(
		decision_md,
		decision=decision,
		control_row=control_row,
		combined_rows=combined_rows,
		tail_life_start=int(tail_life_start),
		tail_life_end=int(total_lives),
	)
	return {
		"decision": decision,
		"life_rows": life_rows_flat,
		"combined_rows": combined_rows,
	}


def main(*, cell_runner: CellRunner | None = None) -> None:
	parser = argparse.ArgumentParser(description="W2.1 episode/life scout")
	parser.add_argument("--conditions", type=str, default="control,w2_base,w2_strong")
	parser.add_argument("--seeds", type=str, default="45,47,49")
	parser.add_argument("--players", type=int, default=300)
	parser.add_argument("--rounds-per-life", type=int, default=3000)
	parser.add_argument("--total-lives", type=int, default=5)
	parser.add_argument("--tail-life-start", type=int, default=TAIL_LIFE_START)
	parser.add_argument("--selection-strength", type=float, default=0.06)
	parser.add_argument("--init-bias", type=float, default=0.12)
	parser.add_argument("--memory-kernel", type=int, default=3)
	parser.add_argument("--burn-in", type=int, default=1000)
	parser.add_argument("--tail", type=int, default=1000)
	parser.add_argument("--a", type=float, default=1.0)
	parser.add_argument("--b", type=float, default=0.9)
	parser.add_argument("--cross", type=float, default=0.20)
	parser.add_argument("--events-json", type=Path, default=DEFAULT_FULL_EVENTS_JSON)
	parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
	parser.add_argument("--life-steps-tsv", type=Path, default=DEFAULT_LIFE_STEPS_TSV)
	parser.add_argument("--combined-tsv", type=Path, default=DEFAULT_COMBINED_TSV)
	parser.add_argument("--decision-md", type=Path, default=DEFAULT_DECISION_MD)
	args = parser.parse_args()
	run_w2_scout(
		conditions=[part.strip() for part in str(args.conditions).split(",") if part.strip()],
		seeds=_parse_seeds(args.seeds),
		out_root=args.out_root,
		life_steps_tsv=args.life_steps_tsv,
		combined_tsv=args.combined_tsv,
		decision_md=args.decision_md,
		events_json=args.events_json,
		players=int(args.players),
		rounds_per_life=int(args.rounds_per_life),
		total_lives=int(args.total_lives),
		tail_life_start=int(args.tail_life_start),
		selection_strength=float(args.selection_strength),
		init_bias=float(args.init_bias),
		memory_kernel=int(args.memory_kernel),
		burn_in=int(args.burn_in),
		tail=int(args.tail),
		a=float(args.a),
		b=float(args.b),
		cross=float(args.cross),
		cell_runner=cell_runner,
	)


if __name__ == "__main__":
	main()