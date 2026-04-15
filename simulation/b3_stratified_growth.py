from __future__ import annotations

import argparse
import csv
import json
from math import atan2, exp, log, pi, sqrt
import random
from pathlib import Path
from typing import Any, Mapping

from analysis.cycle_metrics import classify_cycle_level
from analysis.decay_rate import estimate_decay_gamma
from analysis.event_provenance_summary import summarize_event_provenance
from evolution.replicator_dynamics import sampled_growth_vector
from simulation.personality_coupling import personality_signal_k
from simulation.personality_gate0 import PROTOTYPES, sample_personality
from simulation.run_simulation import DEFAULT_EVENTS_JSON, SimConfig, _write_timeseries_csv, simulate


REPO_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_OUT_ROOT = REPO_ROOT / "outputs" / "b3_stratified_growth"
DEFAULT_SUMMARY_TSV = REPO_ROOT / "outputs" / "b3_stratified_growth_summary.tsv"
DEFAULT_COMBINED_TSV = REPO_ROOT / "outputs" / "b3_stratified_growth_combined.tsv"
DEFAULT_DECISION_MD = REPO_ROOT / "outputs" / "b3_stratified_growth_decision.md"

SUMMARY_FIELDNAMES = [
	"condition",
	"strata_mode",
	"personality_jitter",
	"phase_rebucket_interval",
	"n_strata",
	"selection_strength",
	"seed",
	"cycle_level",
	"stage3_score",
	"turn_strength",
	"env_gamma",
	"env_gamma_r2",
	"env_gamma_n_peaks",
	"control_env_gamma",
	"control_stage3_score",
	"gamma_uplift_ratio_vs_control_seed",
	"stage3_uplift_vs_control_seed",
	"has_level3_seed",
	"mean_inter_strata_cosine",
	"min_inter_strata_cosine",
	"max_inter_strata_cosine",
	"mean_growth_dispersion",
	"mean_active_strata",
	"mean_phase_rebucket_churn",
	"mean_within_stratum_phase_spread",
	"phase_occupancy_entropy",
	"out_csv",
	"provenance_json",
]

COMBINED_FIELDNAMES = [
	"condition",
	"strata_mode",
	"personality_jitter",
	"phase_rebucket_interval",
	"n_strata",
	"selection_strength",
	"is_control",
	"n_seeds",
	"mean_cycle_level",
	"mean_stage3_score",
	"mean_turn_strength",
	"mean_env_gamma",
	"level_counts_json",
	"p_level_3",
	"level3_seed_count",
	"control_mean_env_gamma",
	"control_mean_stage3_score",
	"gamma_uplift_ratio_vs_control",
	"stage3_uplift_vs_control",
	"mean_inter_strata_cosine",
	"min_inter_strata_cosine",
	"max_inter_strata_cosine",
	"mean_growth_dispersion",
	"mean_active_strata",
	"mean_phase_rebucket_churn",
	"mean_within_stratum_phase_spread",
	"phase_occupancy_entropy",
	"gamma_gate_pass",
	"short_scout_pass",
	"hard_stop_fail",
	"longer_confirm_candidate",
	"verdict",
	"representative_seed",
	"representative_simplex_png",
	"representative_phase_amplitude_png",
	"enable_events",
	"events_json",
	"memory_kernel",
	"players",
	"rounds",
	"out_dir",
]

DEFAULT_PERSONALITY_JITTER = 0.08
DEFAULT_PHASE_REBUCKET_INTERVAL = 100
STRATA_MODE_CHOICES = ("index", "personality", "phase")


def _parse_seeds(spec: str) -> list[int]:
	parts = [part.strip() for part in str(spec).split(",") if part.strip()]
	if not parts:
		raise ValueError("--seeds cannot be empty")
	return [int(part) for part in parts]


def _parse_int_list(spec: str) -> list[int]:
	parts = [part.strip() for part in str(spec).split(",") if part.strip()]
	if not parts:
		raise ValueError("int list cannot be empty")
	return [int(part) for part in parts]


def _parse_float_list(spec: str) -> list[float]:
	parts = [part.strip() for part in str(spec).split(",") if part.strip()]
	if not parts:
		raise ValueError("float list cannot be empty")
	return [float(part) for part in parts]


def _sample_b32_personalities(*, n_players: int, jitter: float, seed: int) -> list[dict[str, float]]:
	if int(n_players) % 3 != 0:
		raise ValueError("B3.2 personality strata requires n_players divisible by 3 for the fixed 1:1:1 world")
	per_group = int(n_players) // 3
	rng = random.Random(int(seed))
	personalities: list[dict[str, float]] = []
	for cohort in ("aggressive", "defensive", "balanced"):
		prototype = PROTOTYPES[cohort]
		for _ in range(per_group):
			personalities.append(sample_personality(prototype, jitter=float(jitter), rng=rng))
	return personalities


def _assign_personality_signal_strata(players: list[object], *, n_strata: int) -> None:
	strata = max(1, int(n_strata))
	indexed_signals = [
		(idx, float(personality_signal_k(getattr(player, "personality", {}))))
		for idx, player in enumerate(players)
	]
	indexed_signals.sort(key=lambda item: (item[1], item[0]))
	count = max(1, len(indexed_signals))
	for rank, (idx, signal_k) in enumerate(indexed_signals):
		bucket = min(strata - 1, int(rank * strata // count))
		setattr(players[idx], "stratum", bucket)
		setattr(players[idx], "b3_personality_signal_k", float(signal_k))


def _normalize_distribution(weights: Mapping[str, float], *, strategy_space: list[str]) -> dict[str, float]:
	raw = {strategy: max(0.0, float(weights.get(strategy, 0.0))) for strategy in strategy_space}
	total = float(sum(raw.values()))
	if total <= 1e-12:
		uniform = 1.0 / float(len(strategy_space)) if strategy_space else 0.0
		return {strategy: uniform for strategy in strategy_space}
	return {strategy: float(raw[strategy]) / total for strategy in strategy_space}


def _player_effective_distribution(player: object, strategy_space: list[str]) -> dict[str, float]:
	weights = getattr(player, "strategy_weights", {})
	biases = getattr(player, "strategy_biases", {})
	raw: dict[str, float] = {}
	for strategy in strategy_space:
		weight = max(1e-9, float(weights.get(strategy, 1.0)))
		bias = max(-6.0, min(6.0, float(biases.get(strategy, 0.0))))
		raw[strategy] = weight * exp(bias)
	return _normalize_distribution(raw, strategy_space=strategy_space)


def _phase_angle_from_distribution(distribution: Mapping[str, float]) -> float:
	p_aggressive = float(distribution.get("aggressive", 0.0))
	p_defensive = float(distribution.get("defensive", 0.0))
	p_balanced = float(distribution.get("balanced", 0.0))
	return float(atan2(sqrt(3.0) * (p_defensive - p_balanced), 2.0 * p_aggressive - p_defensive - p_balanced))


def _phase_bucket(angle: float, *, n_strata: int) -> int:
	strata = max(1, int(n_strata))
	if strata == 1:
		return 0
	width = (2.0 * pi) / float(strata)
	normalized = (float(angle) + pi) % (2.0 * pi)
	return min(strata - 1, int(normalized / width))


def _phase_distribution_from_player(player: object, strategy_space: list[str]) -> dict[str, float]:
	counts = getattr(player, "b3_phase_counts", None)
	if isinstance(counts, dict):
		return _normalize_distribution({strategy: float(counts.get(strategy, 0.0)) for strategy in strategy_space}, strategy_space=strategy_space)
	return _player_effective_distribution(player, strategy_space)


def _refresh_phase_angles(players: list[object], *, strategy_space: list[str]) -> list[float]:
	angles: list[float] = []
	for player in players:
		distribution = _phase_distribution_from_player(player, strategy_space)
		angle = _phase_angle_from_distribution(distribution)
		setattr(player, "b3_phase_angle", float(angle))
		angles.append(float(angle))
	return angles


def _initialize_phase_counts(players: list[object], *, strategy_space: list[str]) -> None:
	for player in players:
		distribution = _player_effective_distribution(player, strategy_space)
		setattr(player, "b3_phase_counts", {strategy: float(distribution[strategy]) for strategy in strategy_space})


def _update_phase_counts_from_last_strategy(players: list[object], *, strategy_space: list[str]) -> None:
	for player in players:
		counts = getattr(player, "b3_phase_counts", None)
		if not isinstance(counts, dict):
			counts = {strategy: 0.0 for strategy in strategy_space}
		strategy = getattr(player, "last_strategy", None)
		if strategy in counts:
			counts[str(strategy)] = float(counts[str(strategy)]) + 1.0
		setattr(player, "b3_phase_counts", counts)


def _assign_phase_signal_strata(players: list[object], *, strategy_space: list[str], n_strata: int) -> float:
	previous = [int(getattr(player, "stratum", 0)) for player in players]
	angles = _refresh_phase_angles(players, strategy_space=strategy_space)
	updated: list[int] = []
	for player, angle in zip(players, angles, strict=True):
		bucket = _phase_bucket(angle, n_strata=int(n_strata))
		setattr(player, "stratum", bucket)
		updated.append(bucket)
	if not players:
		return 0.0
	changed = sum(1 for old_bucket, new_bucket in zip(previous, updated, strict=True) if old_bucket != new_bucket)
	return float(changed) / float(len(players))


def _mean_within_stratum_phase_spread(players: list[object], *, n_strata: int) -> float:
	buckets: dict[int, list[float]] = {idx: [] for idx in range(max(1, int(n_strata)))}
	for player in players:
		bucket = int(getattr(player, "stratum", 0)) % max(1, int(n_strata))
		buckets[bucket].append(float(getattr(player, "b3_phase_angle", 0.0)))
	spreads = [max(values) - min(values) for values in buckets.values() if len(values) >= 2]
	return _safe_mean([float(value) for value in spreads]) if spreads else 0.0


def _phase_occupancy_entropy(players: list[object], *, n_strata: int) -> float:
	strata = max(1, int(n_strata))
	if strata <= 1 or not players:
		return 0.0
	counts = [0 for _ in range(strata)]
	for player in players:
		counts[int(getattr(player, "stratum", 0)) % strata] += 1
	total = float(sum(counts))
	if total <= 0.0:
		return 0.0
	entropy = 0.0
	for count in counts:
		if count <= 0:
			continue
		share = float(count) / total
		entropy -= share * log(share)
	return float(entropy / log(float(strata))) if strata > 1 else 0.0


def build_b3_player_setup_callback(
	*,
	strata_mode: str,
	n_strata: int,
	personality_jitter: float,
	personality_seed: int,
) -> Any:
	mode = str(strata_mode)
	if mode not in STRATA_MODE_CHOICES:
		raise ValueError(f"unsupported strata_mode: {mode}")
	if mode == "index":
		return None
	if mode == "phase":
		def _setup_phase(players: list[object], strategy_space: list[str], _cfg: SimConfig) -> None:
			_initialize_phase_counts(players, strategy_space=strategy_space)
			_assign_phase_signal_strata(players, strategy_space=strategy_space, n_strata=int(n_strata))

		return _setup_phase

	def _setup(players: list[object], _strategy_space: list[str], _cfg: SimConfig) -> None:
		personalities = _sample_b32_personalities(
			n_players=len(players),
			jitter=float(personality_jitter),
			seed=int(personality_seed),
		)
		if len(players) != len(personalities):
			raise RuntimeError("personality count mismatch in B3.2 setup")
		for player, personality in zip(players, personalities, strict=True):
			player.personality = dict(personality)
		_assign_personality_signal_strata(players, n_strata=int(n_strata))

	return _setup


def _format_float(value: float) -> str:
	return f"{float(value):.6f}"


def _format_ratio(value: float) -> str:
	if value == float("inf"):
		return "inf"
	if value == float("-inf"):
		return "-inf"
	return f"{float(value):.6f}"


def _yes_no(value: bool) -> str:
	return "yes" if bool(value) else "no"


def _safe_mean(values: list[float]) -> float:
	if not values:
		return 0.0
	return float(sum(values) / float(len(values)))


def _gamma_uplift_ratio(*, baseline_gamma: float, candidate_gamma: float) -> float:
	base = float(baseline_gamma)
	alt = float(candidate_gamma)
	if base == 0.0:
		if alt > 0.0:
			return float("inf")
		if alt < 0.0:
			return float("-inf")
		return 1.0
	return alt / base


def _write_tsv(path: Path, *, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
	path.parent.mkdir(parents=True, exist_ok=True)
	with path.open("w", newline="", encoding="utf-8") as handle:
		writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
		writer.writeheader()
		writer.writerows(rows)


def _seed_metrics(rows: list[dict[str, Any]], *, burn_in: int, tail: int, eta: float, corr_threshold: float) -> dict[str, float | int]:
	series_map = {
		"aggressive": [float(row["p_aggressive"]) for row in rows],
		"defensive": [float(row["p_defensive"]) for row in rows],
		"balanced": [float(row["p_balanced"]) for row in rows],
	}
	cycle = classify_cycle_level(
		series_map,
		burn_in=int(burn_in),
		tail=int(tail),
		amplitude_threshold=0.02,
		corr_threshold=float(corr_threshold),
		eta=float(eta),
		stage3_method="turning",
		phase_smoothing=1,
	)
	fit = estimate_decay_gamma(series_map, series_kind="p")
	return {
		"cycle_level": int(cycle.level),
		"stage3_score": float(cycle.stage3.score) if cycle.stage3 is not None else 0.0,
		"turn_strength": float(cycle.stage3.turn_strength) if cycle.stage3 is not None else 0.0,
		"env_gamma": float(fit.gamma) if fit is not None else 0.0,
		"env_gamma_r2": float(fit.r2) if fit is not None else 0.0,
		"env_gamma_n_peaks": int(fit.n_peaks) if fit is not None else 0,
	}


def _condition_name(*, strata_mode: str, n_strata: int, selection_strength: float) -> str:
	strength = f"{float(selection_strength):.2f}".replace("-", "m").replace(".", "p")
	if str(strata_mode) == "index":
		prefix = "strata"
	elif str(strata_mode) == "personality":
		prefix = "personality_strata"
	else:
		prefix = "phase_strata"
	return f"{prefix}{int(n_strata)}_k{strength}"


def _control_condition_name(*, strata_mode: str, selection_strength: float) -> str:
	return _condition_name(strata_mode=strata_mode, n_strata=1, selection_strength=selection_strength)


def _slice_tail(rows: list[dict[str, Any]], *, burn_in: int, tail: int) -> list[dict[str, Any]]:
	if not rows:
		return []
	begin = max(int(burn_in), len(rows) - int(tail))
	return rows[begin:]


def _simplex_xy(row: dict[str, Any]) -> tuple[float, float]:
	p_a = float(row["p_aggressive"])
	p_d = float(row["p_defensive"])
	p_b = float(row["p_balanced"])
	return (p_d - p_b, p_a - 0.5 * (p_d + p_b))


def _phase_angle_from_row(row: dict[str, Any]) -> float:
	p_a = float(row["p_aggressive"])
	p_d = float(row["p_defensive"])
	p_b = float(row["p_balanced"])
	return float(atan2(sqrt(3.0) * (p_d - p_b), 2.0 * p_a - p_d - p_b))


def _amplitude_from_row(row: dict[str, Any]) -> float:
	p_a = float(row["p_aggressive"])
	p_d = float(row["p_defensive"])
	p_b = float(row["p_balanced"])
	return float(((p_a - (1.0 / 3.0)) ** 2 + (p_d - (1.0 / 3.0)) ** 2 + (p_b - (1.0 / 3.0)) ** 2) ** 0.5)


def _write_simplex_plot(rows: list[dict[str, Any]], *, out_png: Path, title: str, burn_in: int, tail: int) -> None:
	try:
		import matplotlib.pyplot as plt
		from matplotlib.collections import LineCollection
		import numpy as np
	except ImportError as exc:
		raise RuntimeError("matplotlib is required to write B3 simplex plots") from exc

	tail_rows = _slice_tail(rows, burn_in=int(burn_in), tail=int(tail))
	points = [_simplex_xy(row) for row in tail_rows]
	if len(points) < 2:
		return
	xs = np.array([pt[0] for pt in points], dtype=float)
	ys = np.array([pt[1] for pt in points], dtype=float)
	segments = np.stack(
		[
			np.column_stack([xs[:-1], ys[:-1]]),
			np.column_stack([xs[1:], ys[1:]]),
		],
		axis=1,
	)
	out_png.parent.mkdir(parents=True, exist_ok=True)
	fig, ax = plt.subplots(figsize=(6, 6), constrained_layout=True)
	line = LineCollection(segments, cmap="viridis", linewidths=2.0)
	line.set_array(np.linspace(0.0, 1.0, len(segments)))
	ax.add_collection(line)
	ax.scatter(xs[0], ys[0], color="#1f77b4", s=45, marker="o", label="start")
	ax.scatter(xs[-1], ys[-1], color="#d62728", s=55, marker="X", label="end")
	ax.axhline(0.0, color="#cccccc", linewidth=0.8)
	ax.axvline(0.0, color="#cccccc", linewidth=0.8)
	ax.set_title(title)
	ax.set_xlabel("p_defensive - p_balanced")
	ax.set_ylabel("p_aggressive - (p_defensive + p_balanced) / 2")
	ax.legend(loc="best")
	ax.set_aspect("equal", adjustable="box")
	fig.colorbar(line, ax=ax, fraction=0.046, pad=0.04, label="tail time")
	fig.savefig(out_png, dpi=160)
	plt.close(fig)


def _write_phase_amplitude_plot(rows: list[dict[str, Any]], *, out_png: Path, title: str, burn_in: int, tail: int) -> None:
	try:
		import matplotlib.pyplot as plt
		import numpy as np
	except ImportError as exc:
		raise RuntimeError("matplotlib is required to write B3 phase-amplitude plots") from exc

	tail_rows = _slice_tail(rows, burn_in=int(burn_in), tail=int(tail))
	if not tail_rows:
		return
	rounds = np.array([int(row["round"]) for row in tail_rows], dtype=float)
	phases = np.unwrap(np.array([_phase_angle_from_row(row) for row in tail_rows], dtype=float))
	amplitudes = np.array([_amplitude_from_row(row) for row in tail_rows], dtype=float)
	out_png.parent.mkdir(parents=True, exist_ok=True)
	fig, axes = plt.subplots(2, 1, figsize=(8, 5.5), sharex=True, constrained_layout=True)
	axes[0].plot(rounds, phases, color="#1f77b4", linewidth=1.8)
	axes[0].set_ylabel("phase angle")
	axes[0].set_title(title)
	axes[0].grid(alpha=0.25)
	axes[1].plot(rounds, amplitudes, color="#d62728", linewidth=1.8)
	axes[1].set_ylabel("amplitude")
	axes[1].set_xlabel("round")
	axes[1].grid(alpha=0.25)
	fig.savefig(out_png, dpi=160)
	plt.close(fig)


def _vec_dot(left: dict[str, float], right: dict[str, float]) -> float:
	return sum(float(left[key]) * float(right[key]) for key in left.keys())


def _vec_norm(vector: dict[str, float]) -> float:
	return float(_vec_dot(vector, vector) ** 0.5)


def _vec_distance(left: dict[str, float], right: dict[str, float]) -> float:
	return float(sum((float(left[key]) - float(right[key])) ** 2 for key in left.keys()) ** 0.5)


class _B3RoundDiagnostics:
	def __init__(self, *, strata_mode: str, phase_rebucket_interval: int) -> None:
		if int(phase_rebucket_interval) <= 0:
			raise ValueError("phase_rebucket_interval must be > 0")
		self.strata_mode = str(strata_mode)
		self.phase_rebucket_interval = int(phase_rebucket_interval)
		self.mean_pairwise_cosines: list[float] = []
		self.min_pairwise_cosines: list[float] = []
		self.max_pairwise_cosines: list[float] = []
		self.growth_dispersions: list[float] = []
		self.active_strata_counts: list[float] = []
		self.phase_rebucket_churns: list[float] = []
		self.within_stratum_phase_spreads: list[float] = []
		self.phase_occupancy_entropies: list[float] = []

	def callback(
		self,
		_round_index: int,
		cfg: SimConfig,
		players: list[object],
		_dungeon: object,
		_step_records: list[dict[str, object]],
		_row: dict[str, Any],
	) -> None:
		strategy_space = ["aggressive", "defensive", "balanced"]
		if self.strata_mode == "phase":
			_update_phase_counts_from_last_strategy(players, strategy_space=strategy_space)
			_refresh_phase_angles(players, strategy_space=strategy_space)
			phase_churn = 0.0
			if int(cfg.sampled_growth_n_strata) > 1 and ((int(_round_index) + 1) % int(self.phase_rebucket_interval) == 0):
				phase_churn = _assign_phase_signal_strata(players, strategy_space=strategy_space, n_strata=int(cfg.sampled_growth_n_strata))
				_refresh_phase_angles(players, strategy_space=strategy_space)
			self.phase_rebucket_churns.append(float(phase_churn))
			self.within_stratum_phase_spreads.append(
				_mean_within_stratum_phase_spread(players, n_strata=int(cfg.sampled_growth_n_strata))
			)
			self.phase_occupancy_entropies.append(
				_phase_occupancy_entropy(players, n_strata=int(cfg.sampled_growth_n_strata))
			)
		else:
			self.phase_rebucket_churns.append(0.0)
			self.within_stratum_phase_spreads.append(0.0)
			self.phase_occupancy_entropies.append(0.0)
		buckets: dict[int, list[object]] = {idx: [] for idx in range(int(cfg.sampled_growth_n_strata))}
		for player in players:
			bucket = int(getattr(player, "stratum", 0)) % int(cfg.sampled_growth_n_strata)
			buckets[bucket].append(player)
		bucket_growths = [
			sampled_growth_vector(bucket_players, strategy_space)
			for bucket_players in buckets.values()
			if bucket_players
		]
		if not bucket_growths:
			return
		global_growth = {
			strategy: sum(float(len(bucket_players)) * float(sampled_growth_vector(bucket_players, strategy_space)[strategy]) for bucket_players in buckets.values() if bucket_players) / float(sum(len(bucket_players) for bucket_players in buckets.values() if bucket_players))
			for strategy in strategy_space
		}
		pairwise_cosines: list[float] = []
		for idx, left in enumerate(bucket_growths):
			left_norm = _vec_norm(left)
			if left_norm <= 1e-12:
				continue
			for right in bucket_growths[idx + 1 :]:
				right_norm = _vec_norm(right)
				if right_norm <= 1e-12:
					continue
				pairwise_cosines.append(_vec_dot(left, right) / float(left_norm * right_norm))
		growth_dispersion = _safe_mean([_vec_distance(bucket_growth, global_growth) for bucket_growth in bucket_growths])
		self.mean_pairwise_cosines.append(_safe_mean(pairwise_cosines) if pairwise_cosines else 1.0)
		self.min_pairwise_cosines.append(min(pairwise_cosines) if pairwise_cosines else 1.0)
		self.max_pairwise_cosines.append(max(pairwise_cosines) if pairwise_cosines else 1.0)
		self.growth_dispersions.append(growth_dispersion)
		self.active_strata_counts.append(float(len(bucket_growths)))

	def finalize(self) -> dict[str, float]:
		return {
			"mean_inter_strata_cosine": _safe_mean(self.mean_pairwise_cosines),
			"min_inter_strata_cosine": min(self.min_pairwise_cosines) if self.min_pairwise_cosines else 1.0,
			"max_inter_strata_cosine": max(self.max_pairwise_cosines) if self.max_pairwise_cosines else 1.0,
			"mean_growth_dispersion": _safe_mean(self.growth_dispersions),
			"mean_active_strata": _safe_mean(self.active_strata_counts),
			"mean_phase_rebucket_churn": _safe_mean(self.phase_rebucket_churns),
			"mean_within_stratum_phase_spread": _safe_mean(self.within_stratum_phase_spreads),
			"phase_occupancy_entropy": _safe_mean(self.phase_occupancy_entropies),
		}


def build_b3_round_diagnostics(*, strata_mode: str, phase_rebucket_interval: int = DEFAULT_PHASE_REBUCKET_INTERVAL) -> _B3RoundDiagnostics:
	if str(strata_mode) not in STRATA_MODE_CHOICES:
		raise ValueError(f"unsupported strata_mode: {strata_mode}")
	if int(phase_rebucket_interval) <= 0:
		raise ValueError("phase_rebucket_interval must be > 0")
	return _B3RoundDiagnostics(strata_mode=str(strata_mode), phase_rebucket_interval=int(phase_rebucket_interval))


def _build_condition_summary(
	*,
	condition: str,
	strata_mode: str,
	personality_jitter: float,
	phase_rebucket_interval: int,
	n_strata: int,
	selection_strength: float,
	metrics_rows: list[dict[str, Any]],
	per_seed_rows: list[dict[str, Any]],
	out_dir: Path,
	players: int,
	rounds: int,
	memory_kernel: int,
	enable_events: bool,
	events_json: Path | None,
) -> dict[str, Any]:
	levels = [int(row["cycle_level"]) for row in metrics_rows]
	level_counts = {level: levels.count(level) for level in range(4)}
	den = float(len(metrics_rows)) or 1.0
	return {
		"condition": condition,
		"strata_mode": str(strata_mode),
		"personality_jitter": _format_float(personality_jitter),
		"phase_rebucket_interval": int(phase_rebucket_interval),
		"n_strata": int(n_strata),
		"selection_strength": _format_float(selection_strength),
		"is_control": _yes_no(int(n_strata) == 1),
		"n_seeds": len(metrics_rows),
		"mean_cycle_level": _format_float(_safe_mean([float(level) for level in levels])),
		"mean_stage3_score": _format_float(_safe_mean([float(row["stage3_score"]) for row in metrics_rows])),
		"mean_turn_strength": _format_float(_safe_mean([float(row["turn_strength"]) for row in metrics_rows])),
		"mean_env_gamma": _format_float(_safe_mean([float(row["env_gamma"]) for row in metrics_rows])),
		"level_counts_json": json.dumps(level_counts, sort_keys=True),
		"p_level_3": _format_float(sum(1 for level in levels if level >= 3) / den),
		"level3_seed_count": sum(1 for level in levels if level >= 3),
		"control_mean_env_gamma": "",
		"control_mean_stage3_score": "",
		"gamma_uplift_ratio_vs_control": "",
		"stage3_uplift_vs_control": "",
		"mean_inter_strata_cosine": _format_float(_safe_mean([float(row["mean_inter_strata_cosine"]) for row in per_seed_rows])),
		"min_inter_strata_cosine": _format_float(min(float(row["min_inter_strata_cosine"]) for row in per_seed_rows)),
		"max_inter_strata_cosine": _format_float(max(float(row["max_inter_strata_cosine"]) for row in per_seed_rows)),
		"mean_growth_dispersion": _format_float(_safe_mean([float(row["mean_growth_dispersion"]) for row in per_seed_rows])),
		"mean_active_strata": _format_float(_safe_mean([float(row["mean_active_strata"]) for row in per_seed_rows])),
		"mean_phase_rebucket_churn": _format_float(_safe_mean([float(row["mean_phase_rebucket_churn"]) for row in per_seed_rows])),
		"mean_within_stratum_phase_spread": _format_float(_safe_mean([float(row["mean_within_stratum_phase_spread"]) for row in per_seed_rows])),
		"phase_occupancy_entropy": _format_float(_safe_mean([float(row["phase_occupancy_entropy"]) for row in per_seed_rows])),
		"gamma_gate_pass": "",
		"short_scout_pass": "",
		"hard_stop_fail": "",
		"longer_confirm_candidate": "no",
		"verdict": "control" if int(n_strata) == 1 else "pending",
		"representative_seed": "",
		"representative_simplex_png": "",
		"representative_phase_amplitude_png": "",
		"enable_events": _yes_no(enable_events),
		"events_json": str(events_json) if events_json is not None else "",
		"memory_kernel": int(memory_kernel),
		"players": int(players),
		"rounds": int(rounds),
		"out_dir": str(out_dir),
	}


def _representative_seed_row(rows: list[dict[str, Any]]) -> dict[str, Any]:
	return max(
		rows,
		key=lambda row: (
			int(row["cycle_level"]),
			float(row["stage3_score"]),
			float(row["mean_growth_dispersion"]),
			-int(row["seed"]),
		),
	)


def _write_decision(
	path: Path,
	*,
	combined_rows: list[dict[str, Any]],
	recommended_candidates: list[dict[str, Any]],
	best_active_seed: dict[str, Any] | None = None,
	worst_active_seed: dict[str, Any] | None = None,
) -> None:
	mode = str(combined_rows[0]["strata_mode"]) if combined_rows else "index"
	lines = [
		"# B3 Short Scout Decision",
		"",
		"## Conditions",
		"",
	]
	for row in combined_rows:
		lines.append(
			f"- {row['condition']}: strata_mode={row['strata_mode']} n_strata={row['n_strata']} selection_strength={row['selection_strength']} level3_seed_count={row['level3_seed_count']} stage3_uplift={row['stage3_uplift_vs_control']} gamma_ratio={row['gamma_uplift_ratio_vs_control']} mean_inter_strata_cosine={row['mean_inter_strata_cosine']} mean_growth_dispersion={row['mean_growth_dispersion']} short_scout_pass={row['short_scout_pass']} hard_stop_fail={row['hard_stop_fail']} verdict={row['verdict']} simplex_png={row['representative_simplex_png']} phase_amp_png={row['representative_phase_amplitude_png']}"
		)
	lines.extend(["", "## Recommendation", ""])
	if not recommended_candidates:
		lines.append("- longer_confirm_candidate: none")
	else:
		for row in recommended_candidates:
			lines.append(
				f"- longer_confirm_candidate: {row['condition']} stage3_uplift={row['stage3_uplift_vs_control']} gamma_ratio={row['gamma_uplift_ratio_vs_control']} mean_inter_strata_cosine={row['mean_inter_strata_cosine']} representative_seed={row['representative_seed']}"
			)
	if best_active_seed is not None or worst_active_seed is not None:
		lines.extend(["", "## Diagnostics", ""])
		if best_active_seed is not None:
			lines.append(
				f"- best_active_seed: condition={best_active_seed['condition']} seed={best_active_seed['seed']} stage3_uplift={best_active_seed['stage3_uplift_vs_control_seed']} simplex_png={best_active_seed['simplex_png']} phase_amp_png={best_active_seed['phase_amp_png']}"
			)
		if worst_active_seed is not None:
			lines.append(
				f"- worst_active_seed: condition={worst_active_seed['condition']} seed={worst_active_seed['seed']} stage3_uplift={worst_active_seed['stage3_uplift_vs_control_seed']} simplex_png={worst_active_seed['simplex_png']} phase_amp_png={worst_active_seed['phase_amp_png']}"
			)
	lines.extend(
		[
			"",
			"## Stop Rule",
			"",
			"- 若 0/3 seeds 達 Level 3 且 mean_stage3_score uplift < 0.02，該 cell 直接記為 fail，不做 Longer Confirm。",
		]
	)
	if mode == "index":
		lines.append("- 若 B3.1 fail 且 mean_inter_strata_cosine 仍接近 1、mean_growth_dispersion 很低，代表 index-based strata 沒有抓到真實局部結構；下一步才允許升級到 B3.2 personality-based strata。")
	elif mode == "phase":
		lines.append("- 若 B3.3 fail，代表 phase-aware strata 仍不足以把局部差異轉成穩定 Level 3 uplift；下一步應轉入 B5 tangential drift，而不是回頭調 rebucket interval 或 sector offset。")
	else:
		lines.append("- 若 B3.2 fail，代表 static personality-based strata 仍不足以把局部差異轉成 Level 3 uplift；下一步才允許升級到 B3.3 phase-aware strata。")
	path.parent.mkdir(parents=True, exist_ok=True)
	path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_b3_scout(
	*,
	seeds: list[int],
	n_strata_values: list[int],
	selection_strengths: list[float],
	strata_mode: str = "index",
	personality_jitter: float = DEFAULT_PERSONALITY_JITTER,
	phase_rebucket_interval: int = DEFAULT_PHASE_REBUCKET_INTERVAL,
	out_root: Path,
	summary_tsv: Path,
	combined_tsv: Path,
	decision_md: Path,
	players: int = 300,
	rounds: int = 3000,
	burn_in: int = 1000,
	tail: int = 1000,
	memory_kernel: int = 3,
	init_bias: float = 0.12,
	a: float = 1.0,
	b: float = 0.9,
	cross: float = 0.20,
	eta: float = 0.55,
	corr_threshold: float = 0.09,
	enable_events: bool = True,
	events_json: Path | None = DEFAULT_EVENTS_JSON,
) -> dict[str, Any]:
	mode = str(strata_mode)
	if mode not in STRATA_MODE_CHOICES:
		raise ValueError(f"strata_mode must be one of {STRATA_MODE_CHOICES}")
	if not (float(personality_jitter) >= 0.0):
		raise ValueError("personality_jitter must be >= 0")
	if int(phase_rebucket_interval) <= 0:
		raise ValueError("phase_rebucket_interval must be > 0")
	if 1 not in {int(value) for value in n_strata_values}:
		raise ValueError("B3 short scout requires n_strata_values to include 1 for matched controls")
	if mode == "personality" and int(players) % 3 != 0:
		raise ValueError("B3.2 personality strata requires players divisible by 3")
	out_root.mkdir(parents=True, exist_ok=True)
	all_summary_rows: list[dict[str, Any]] = []
	all_combined_rows: list[dict[str, Any]] = []
	condition_seed_rows: dict[str, list[dict[str, Any]]] = {}
	condition_timeseries: dict[str, dict[int, list[dict[str, Any]]]] = {}

	for selection_strength in selection_strengths:
		for n_strata in n_strata_values:
			condition = _condition_name(strata_mode=mode, n_strata=int(n_strata), selection_strength=float(selection_strength))
			out_dir = out_root / condition
			out_dir.mkdir(parents=True, exist_ok=True)
			per_seed_rows: list[dict[str, Any]] = []
			metric_rows: list[dict[str, Any]] = []
			seed_rows_map: dict[int, list[dict[str, Any]]] = {}

			for seed in seeds:
				diagnostics = build_b3_round_diagnostics(
					strata_mode=mode,
					phase_rebucket_interval=int(phase_rebucket_interval),
				)
				setup_callback = build_b3_player_setup_callback(
					strata_mode=mode,
					n_strata=int(n_strata),
					personality_jitter=float(personality_jitter),
					personality_seed=int(seed),
				)
				cfg = SimConfig(
					n_players=int(players),
					n_rounds=int(rounds),
					seed=int(seed),
					payoff_mode="matrix_ab",
					popularity_mode="sampled",
					gamma=0.1,
					epsilon=0.0,
					a=float(a),
					b=float(b),
					matrix_cross_coupling=float(cross),
					init_bias=float(init_bias),
					evolution_mode="sampled",
					payoff_lag=1,
					selection_strength=float(selection_strength),
					enable_events=bool(enable_events),
					events_json=events_json if enable_events else None,
					out_csv=out_dir / f"seed{seed}.csv",
					memory_kernel=int(memory_kernel),
					sampled_growth_n_strata=int(n_strata),
				)
				strategy_space, rows = simulate(cfg, player_setup_callback=setup_callback, round_callback=diagnostics.callback)
				_write_timeseries_csv(cfg.out_csv, strategy_space=strategy_space, rows=rows)
				seed_rows_map[int(seed)] = rows
				seed_metric = _seed_metrics(rows, burn_in=int(burn_in), tail=int(tail), eta=float(eta), corr_threshold=float(corr_threshold))
				diag_stats = diagnostics.finalize()
				provenance = {
					"condition": condition,
					"strata_mode": mode,
					"personality_jitter": float(personality_jitter),
					"phase_rebucket_interval": int(phase_rebucket_interval),
					"n_strata": int(n_strata),
					"selection_strength": float(selection_strength),
					"config": {
						"players": int(players),
						"rounds": int(rounds),
						"burn_in": int(burn_in),
						"tail": int(tail),
						"memory_kernel": int(memory_kernel),
						"init_bias": float(init_bias),
						"a": float(a),
						"b": float(b),
						"matrix_cross_coupling": float(cross),
						"enable_events": bool(enable_events),
						"events_json": str(events_json) if enable_events and events_json is not None else None,
					},
					"round_diagnostics": diag_stats,
				}
				if enable_events and events_json is not None:
					provenance["event_provenance"] = summarize_event_provenance(cfg.out_csv, events_json=events_json)
				provenance_path = out_dir / f"seed{seed}_provenance.json"
				provenance_path.write_text(json.dumps(provenance, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

				per_seed_rows.append(
					{
						"condition": condition,
						"strata_mode": mode,
						"personality_jitter": _format_float(personality_jitter),
						"phase_rebucket_interval": int(phase_rebucket_interval),
						"n_strata": int(n_strata),
						"selection_strength": _format_float(selection_strength),
						"seed": int(seed),
						"cycle_level": int(seed_metric["cycle_level"]),
						"stage3_score": _format_float(seed_metric["stage3_score"]),
						"turn_strength": _format_float(seed_metric["turn_strength"]),
						"env_gamma": _format_float(seed_metric["env_gamma"]),
						"env_gamma_r2": _format_float(seed_metric["env_gamma_r2"]),
						"env_gamma_n_peaks": int(seed_metric["env_gamma_n_peaks"]),
						"control_env_gamma": "",
						"control_stage3_score": "",
						"gamma_uplift_ratio_vs_control_seed": "",
						"stage3_uplift_vs_control_seed": "",
						"has_level3_seed": _yes_no(int(seed_metric["cycle_level"]) >= 3),
						"mean_inter_strata_cosine": _format_float(diag_stats["mean_inter_strata_cosine"]),
						"min_inter_strata_cosine": _format_float(diag_stats["min_inter_strata_cosine"]),
						"max_inter_strata_cosine": _format_float(diag_stats["max_inter_strata_cosine"]),
						"mean_growth_dispersion": _format_float(diag_stats["mean_growth_dispersion"]),
						"mean_active_strata": _format_float(diag_stats["mean_active_strata"]),
						"mean_phase_rebucket_churn": _format_float(diag_stats["mean_phase_rebucket_churn"]),
						"mean_within_stratum_phase_spread": _format_float(diag_stats["mean_within_stratum_phase_spread"]),
						"phase_occupancy_entropy": _format_float(diag_stats["phase_occupancy_entropy"]),
						"out_csv": str(cfg.out_csv),
						"provenance_json": str(provenance_path),
					}
				)
				metric_rows.append(per_seed_rows[-1])

			condition_seed_rows[condition] = per_seed_rows
			condition_timeseries[condition] = seed_rows_map
			all_combined_rows.append(
				_build_condition_summary(
					condition=condition,
					strata_mode=mode,
					personality_jitter=float(personality_jitter),
					phase_rebucket_interval=int(phase_rebucket_interval),
					n_strata=int(n_strata),
					selection_strength=float(selection_strength),
					metrics_rows=metric_rows,
					per_seed_rows=per_seed_rows,
					out_dir=out_dir,
					players=int(players),
					rounds=int(rounds),
					memory_kernel=int(memory_kernel),
					enable_events=bool(enable_events),
					events_json=events_json if enable_events else None,
				)
			)

	for combined_row in all_combined_rows:
		condition = str(combined_row["condition"])
		selection_strength = float(combined_row["selection_strength"])
		is_control = str(combined_row["is_control"]) == "yes"
		control_condition = _control_condition_name(strata_mode=mode, selection_strength=selection_strength)
		control_row = next(row for row in all_combined_rows if str(row["condition"]) == control_condition)
		control_mean_env_gamma = float(control_row["mean_env_gamma"])
		control_mean_stage3_score = float(control_row["mean_stage3_score"])
		mean_env_gamma = float(combined_row["mean_env_gamma"])
		mean_stage3_score = float(combined_row["mean_stage3_score"])
		level3_seed_count = int(combined_row["level3_seed_count"])
		stage3_uplift = mean_stage3_score - control_mean_stage3_score
		gamma_ratio = _gamma_uplift_ratio(baseline_gamma=control_mean_env_gamma, candidate_gamma=mean_env_gamma)
		gamma_gate_pass = abs(mean_env_gamma) <= 5.0e-4
		short_scout_pass = (level3_seed_count >= 1) and gamma_gate_pass and (not is_control)
		hard_stop_fail = (level3_seed_count == 0) and (stage3_uplift < 0.02) and (not is_control)
		if is_control:
			verdict = "control"
		elif short_scout_pass:
			verdict = "pass"
		elif hard_stop_fail:
			verdict = "fail"
		else:
			verdict = "weak_positive"
		combined_row["control_mean_env_gamma"] = _format_float(control_mean_env_gamma)
		combined_row["control_mean_stage3_score"] = _format_float(control_mean_stage3_score)
		combined_row["gamma_uplift_ratio_vs_control"] = _format_ratio(1.0 if is_control else gamma_ratio)
		combined_row["stage3_uplift_vs_control"] = _format_float(0.0 if is_control else stage3_uplift)
		combined_row["gamma_gate_pass"] = _yes_no(True if is_control else gamma_gate_pass)
		combined_row["short_scout_pass"] = _yes_no(False if is_control else short_scout_pass)
		combined_row["hard_stop_fail"] = _yes_no(False if is_control else hard_stop_fail)
		combined_row["longer_confirm_candidate"] = _yes_no(verdict == "pass")
		combined_row["verdict"] = verdict

		representative_row = _representative_seed_row(condition_seed_rows[condition])
		representative_seed = int(representative_row["seed"])
		representative_png = Path(combined_row["out_dir"]) / f"seed{representative_seed}_simplex.png"
		representative_phase_png = Path(combined_row["out_dir"]) / f"seed{representative_seed}_phase_amplitude.png"
		_write_simplex_plot(
			condition_timeseries[condition][representative_seed],
			out_png=representative_png,
			title=f"{condition} seed={representative_seed}",
			burn_in=int(burn_in),
			tail=int(tail),
		)
		_write_phase_amplitude_plot(
			condition_timeseries[condition][representative_seed],
			out_png=representative_phase_png,
			title=f"{condition} seed={representative_seed}",
			burn_in=int(burn_in),
			tail=int(tail),
		)
		combined_row["representative_seed"] = representative_seed
		combined_row["representative_simplex_png"] = str(representative_png)
		combined_row["representative_phase_amplitude_png"] = str(representative_phase_png)

	for condition, per_seed_rows in condition_seed_rows.items():
		selection_strength = float(next(row for row in all_combined_rows if str(row["condition"]) == condition)["selection_strength"])
		control_rows = condition_seed_rows[_control_condition_name(strata_mode=mode, selection_strength=selection_strength)]
		control_by_seed = {int(row["seed"]): row for row in control_rows}
		for row in per_seed_rows:
			seed = int(row["seed"])
			control_seed_row = control_by_seed[seed]
			row["control_env_gamma"] = control_seed_row["env_gamma"]
			row["control_stage3_score"] = control_seed_row["stage3_score"]
			if str(row["condition"]) == _control_condition_name(strata_mode=mode, selection_strength=selection_strength):
				row["gamma_uplift_ratio_vs_control_seed"] = _format_ratio(1.0)
				row["stage3_uplift_vs_control_seed"] = _format_float(0.0)
			else:
				row["gamma_uplift_ratio_vs_control_seed"] = _format_ratio(
					_gamma_uplift_ratio(
						baseline_gamma=float(control_seed_row["env_gamma"]),
						candidate_gamma=float(row["env_gamma"]),
					)
				)
				row["stage3_uplift_vs_control_seed"] = _format_float(
					float(row["stage3_score"]) - float(control_seed_row["stage3_score"])
				)
		all_summary_rows.extend(per_seed_rows)

	recommended_candidates = [row for row in all_combined_rows if str(row["verdict"]) == "pass"]
	recommended_candidates.sort(
		key=lambda row: (
			float(row["stage3_uplift_vs_control"]),
			int(row["level3_seed_count"]),
			float(row["mean_growth_dispersion"]),
		),
		reverse=True,
	)
	active_seed_rows = [row for row in all_summary_rows if str(row["condition"]) != _control_condition_name(strata_mode=mode, selection_strength=float(row["selection_strength"]))]
	best_active_seed: dict[str, Any] | None = None
	worst_active_seed: dict[str, Any] | None = None
	if active_seed_rows:
		best_row = max(
			active_seed_rows,
			key=lambda row: (
				float(row["stage3_uplift_vs_control_seed"] or 0.0),
				float(row["stage3_score"]),
				float(row["mean_growth_dispersion"]),
				-int(row["seed"]),
			),
		)
		worst_row = min(
			active_seed_rows,
			key=lambda row: (
				float(row["stage3_uplift_vs_control_seed"] or 0.0),
				float(row["stage3_score"]),
				-float(row["mean_growth_dispersion"]),
				int(row["seed"]),
			),
		)
		for label, row in (("best", best_row), ("worst", worst_row)):
			condition = str(row["condition"])
			seed = int(row["seed"])
			out_dir = Path(next(item["out_dir"] for item in all_combined_rows if str(item["condition"]) == condition))
			simplex_png = out_dir / f"seed{seed}_{label}_diagnostic_simplex.png"
			phase_png = out_dir / f"seed{seed}_{label}_diagnostic_phase_amplitude.png"
			_write_simplex_plot(
				condition_timeseries[condition][seed],
				out_png=simplex_png,
				title=f"{condition} {label} seed={seed}",
				burn_in=int(burn_in),
				tail=int(tail),
			)
			_write_phase_amplitude_plot(
				condition_timeseries[condition][seed],
				out_png=phase_png,
				title=f"{condition} {label} seed={seed}",
				burn_in=int(burn_in),
				tail=int(tail),
			)
			payload = {
				"condition": condition,
				"seed": seed,
				"stage3_uplift_vs_control_seed": row["stage3_uplift_vs_control_seed"],
				"simplex_png": str(simplex_png),
				"phase_amp_png": str(phase_png),
			}
			if label == "best":
				best_active_seed = payload
			else:
				worst_active_seed = payload

	_write_tsv(summary_tsv, fieldnames=SUMMARY_FIELDNAMES, rows=all_summary_rows)
	_write_tsv(combined_tsv, fieldnames=COMBINED_FIELDNAMES, rows=all_combined_rows)
	_write_decision(
		decision_md,
		combined_rows=all_combined_rows,
		recommended_candidates=recommended_candidates,
		best_active_seed=best_active_seed,
		worst_active_seed=worst_active_seed,
	)

	return {
		"summary_tsv": str(summary_tsv),
		"combined_tsv": str(combined_tsv),
		"decision_md": str(decision_md),
		"recommended_candidates": [dict(row) for row in recommended_candidates],
	}


def main() -> None:
	parser = argparse.ArgumentParser(description="B3 stratified growth short-scout harness")
	parser.add_argument("--seeds", type=str, default="45,47,49")
	parser.add_argument("--n-strata", type=str, default="1,3,5,10")
	parser.add_argument("--selection-strengths", type=str, default="0.06")
	parser.add_argument("--strata-mode", type=str, choices=list(STRATA_MODE_CHOICES), default="index")
	parser.add_argument("--personality-jitter", type=float, default=DEFAULT_PERSONALITY_JITTER)
	parser.add_argument("--phase-rebucket-interval", type=int, default=DEFAULT_PHASE_REBUCKET_INTERVAL)
	parser.add_argument("--players", type=int, default=300)
	parser.add_argument("--rounds", type=int, default=3000)
	parser.add_argument("--burn-in", type=int, default=1000)
	parser.add_argument("--tail", type=int, default=1000)
	parser.add_argument("--memory-kernel", type=int, default=3)
	parser.add_argument("--init-bias", type=float, default=0.12)
	parser.add_argument("--a", type=float, default=1.0)
	parser.add_argument("--b", type=float, default=0.9)
	parser.add_argument("--matrix-cross-coupling", type=float, default=0.20)
	parser.add_argument("--eta", type=float, default=0.55)
	parser.add_argument("--corr-threshold", type=float, default=0.09)
	parser.add_argument("--disable-events", action="store_true")
	parser.add_argument("--events-json", type=Path, default=DEFAULT_EVENTS_JSON)
	parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
	parser.add_argument("--summary-tsv", type=Path, default=DEFAULT_SUMMARY_TSV)
	parser.add_argument("--combined-tsv", type=Path, default=DEFAULT_COMBINED_TSV)
	parser.add_argument("--decision-md", type=Path, default=DEFAULT_DECISION_MD)
	args = parser.parse_args()

	result = run_b3_scout(
		seeds=_parse_seeds(args.seeds),
		n_strata_values=_parse_int_list(args.n_strata),
		selection_strengths=_parse_float_list(args.selection_strengths),
		strata_mode=str(args.strata_mode),
		personality_jitter=float(args.personality_jitter),
		phase_rebucket_interval=int(args.phase_rebucket_interval),
		out_root=args.out_root,
		summary_tsv=args.summary_tsv,
		combined_tsv=args.combined_tsv,
		decision_md=args.decision_md,
		players=int(args.players),
		rounds=int(args.rounds),
		burn_in=int(args.burn_in),
		tail=int(args.tail),
		memory_kernel=int(args.memory_kernel),
		init_bias=float(args.init_bias),
		a=float(args.a),
		b=float(args.b),
		cross=float(args.matrix_cross_coupling),
		eta=float(args.eta),
		corr_threshold=float(args.corr_threshold),
		enable_events=not bool(args.disable_events),
		events_json=args.events_json,
	)
	print(f"summary_tsv={result['summary_tsv']}")
	print(f"combined_tsv={result['combined_tsv']}")
	print(f"decision_md={result['decision_md']}")


if __name__ == "__main__":
	main()