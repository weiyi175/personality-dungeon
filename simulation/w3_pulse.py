from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

from analysis.event_provenance_summary import summarize_event_provenance
from core.stackelberg import StackelbergCommitment, dominant_strategy_from_shares, ema_step, leader_prefers, select_best_commitment
from simulation.personality_gate0 import DEFAULT_FULL_EVENTS_JSON
from simulation.personality_gate1 import seed_metrics
from simulation.run_simulation import SimConfig, _write_timeseries_csv, simulate


REPO_ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = "w3_pulse"
DEFAULT_OUT_ROOT = REPO_ROOT / "outputs" / "w3_pulse"
DEFAULT_SUMMARY_TSV = REPO_ROOT / "outputs" / "w3_pulse_summary.tsv"
DEFAULT_COMBINED_TSV = REPO_ROOT / "outputs" / "w3_pulse_combined.tsv"
DEFAULT_DECISION_MD = REPO_ROOT / "outputs" / "w3_pulse_decision.md"

DEFAULT_EMA_ALPHA = 0.15
DEFAULT_PULSE_HORIZON = 120
DEFAULT_REFRACTORY_ROUNDS = 240

PULSE_STEP_FIELDNAMES = [
	"condition",
	"seed",
	"round",
	"p_aggressive",
	"p_defensive",
	"p_balanced",
	"p_hat_aggressive",
	"p_hat_defensive",
	"p_hat_balanced",
	"dominant_strategy_ema",
	"dominant_changed",
	"leader_action",
	"pulse_active",
	"pulse_started",
	"pulse_rounds_left",
	"refractory_rounds_left",
	"a",
	"b",
	"matrix_cross_coupling",
]

SUMMARY_FIELDNAMES = [
	"protocol",
	"condition",
	"leader_action",
	"a",
	"b",
	"matrix_cross_coupling",
	"seed",
	"cycle_level",
	"mean_stage3_score",
	"mean_turn_strength",
	"mean_env_gamma",
	"env_gamma_r2",
	"env_gamma_n_peaks",
	"mean_success_rate",
	"level3_seed_count",
	"has_level3_seed",
	"pulse_count",
	"pulse_active_share",
	"first_pulse_round",
	"dominant_transition_count",
	"pulse_steps_tsv",
	"out_csv",
	"provenance_json",
]

COMBINED_FIELDNAMES = [
	"protocol",
	"condition",
	"is_control",
	"leader_action",
	"n_seeds",
	"mean_stage3_score",
	"mean_turn_strength",
	"mean_env_gamma",
	"level_counts_json",
	"level3_seed_count",
	"control_mean_env_gamma",
	"control_mean_stage3_score",
	"gamma_uplift_ratio_vs_control",
	"stage3_uplift_vs_control",
	"leader_prefers_over_control",
	"pulse_activation_rate",
	"mean_pulse_count",
	"mean_pulse_active_share",
	"mean_first_pulse_round",
	"mean_dominant_transition_count",
	"is_best_policy",
	"promotion_gate_pass",
	"verdict",
	"events_json",
	"selection_strength",
	"memory_kernel",
	"ema_alpha",
	"pulse_horizon",
	"refractory_rounds",
	"out_dir",
]

CellRunner = Callable[["W3PulseCellConfig", int], Mapping[str, Any]]


@dataclass(frozen=True)
class W3PulseCellConfig:
	condition: str
	baseline_commitment: StackelbergCommitment
	pulse_commitment: StackelbergCommitment
	players: int
	rounds: int
	events_json: Path
	selection_strength: float
	init_bias: float
	memory_kernel: int
	burn_in: int
	tail: int
	ema_alpha: float
	pulse_horizon: int
	refractory_rounds: int
	out_dir: Path

	@property
	def is_control(self) -> bool:
		return self.condition == "control_pulse_policy"

	def out_csv(self, seed: int) -> Path:
		return self.out_dir / f"seed{int(seed)}.csv"

	def provenance_path(self, seed: int) -> Path:
		return self.out_dir / f"seed{int(seed)}_provenance.json"

	def pulse_steps_path(self, seed: int) -> Path:
		return self.out_dir / f"w3_pulse_steps_{self.condition}_seed{int(seed)}.tsv"


def _default_baseline_commitment() -> StackelbergCommitment:
	return StackelbergCommitment(
		condition="control_pulse_policy",
		leader_action="baseline_pulse_policy",
		a=1.00,
		b=0.90,
		matrix_cross_coupling=0.20,
		description="Baseline payoff geometry for W3.3 control",
	)


def _default_pulse_configs() -> list[W3PulseCellConfig]:
	baseline = _default_baseline_commitment()
	return [
		W3PulseCellConfig(
			condition="control_pulse_policy",
			baseline_commitment=baseline,
			pulse_commitment=baseline,
			players=300,
			rounds=3000,
			events_json=DEFAULT_FULL_EVENTS_JSON,
			selection_strength=0.06,
			init_bias=0.12,
			memory_kernel=3,
			burn_in=1000,
			tail=1000,
			ema_alpha=DEFAULT_EMA_ALPHA,
			pulse_horizon=DEFAULT_PULSE_HORIZON,
			refractory_rounds=DEFAULT_REFRACTORY_ROUNDS,
			out_dir=DEFAULT_OUT_ROOT / "control_pulse_policy",
		),
		W3PulseCellConfig(
			condition="w3_pulse_crossguard",
			baseline_commitment=baseline,
			pulse_commitment=StackelbergCommitment(
				condition="w3_pulse_crossguard",
				leader_action="cross_pulse",
				a=1.00,
				b=0.90,
				matrix_cross_coupling=0.35,
				description="Finite cross-only pulse triggered by dominant transition",
			),
			players=300,
			rounds=3000,
			events_json=DEFAULT_FULL_EVENTS_JSON,
			selection_strength=0.06,
			init_bias=0.12,
			memory_kernel=3,
			burn_in=1000,
			tail=1000,
			ema_alpha=DEFAULT_EMA_ALPHA,
			pulse_horizon=DEFAULT_PULSE_HORIZON,
			refractory_rounds=DEFAULT_REFRACTORY_ROUNDS,
			out_dir=DEFAULT_OUT_ROOT / "w3_pulse_crossguard",
		),
		W3PulseCellConfig(
			condition="w3_pulse_edgetilt",
			baseline_commitment=baseline,
			pulse_commitment=StackelbergCommitment(
				condition="w3_pulse_edgetilt",
				leader_action="edge_pulse",
				a=1.05,
				b=0.85,
				matrix_cross_coupling=0.20,
				description="Finite edge-only pulse triggered by dominant transition",
			),
			players=300,
			rounds=3000,
			events_json=DEFAULT_FULL_EVENTS_JSON,
			selection_strength=0.06,
			init_bias=0.12,
			memory_kernel=3,
			burn_in=1000,
			tail=1000,
			ema_alpha=DEFAULT_EMA_ALPHA,
			pulse_horizon=DEFAULT_PULSE_HORIZON,
			refractory_rounds=DEFAULT_REFRACTORY_ROUNDS,
			out_dir=DEFAULT_OUT_ROOT / "w3_pulse_edgetilt",
		),
		W3PulseCellConfig(
			condition="w3_pulse_commitpush",
			baseline_commitment=baseline,
			pulse_commitment=StackelbergCommitment(
				condition="w3_pulse_commitpush",
				leader_action="commit_pulse",
				a=1.05,
				b=0.85,
				matrix_cross_coupling=0.35,
				description="Finite edge+cross pulse triggered by dominant transition",
			),
			players=300,
			rounds=3000,
			events_json=DEFAULT_FULL_EVENTS_JSON,
			selection_strength=0.06,
			init_bias=0.12,
			memory_kernel=3,
			burn_in=1000,
			tail=1000,
			ema_alpha=DEFAULT_EMA_ALPHA,
			pulse_horizon=DEFAULT_PULSE_HORIZON,
			refractory_rounds=DEFAULT_REFRACTORY_ROUNDS,
			out_dir=DEFAULT_OUT_ROOT / "w3_pulse_commitpush",
		),
	]


def _parse_seeds(spec: str) -> list[int]:
	parts = [part.strip() for part in str(spec).split(",") if part.strip()]
	if not parts:
		raise ValueError("--seeds cannot be empty")
	return [int(part) for part in parts]


def _safe_mean(values: list[float]) -> float:
	if not values:
		return 0.0
	return float(sum(values) / float(len(values)))


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


def _write_tsv(path: Path, *, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
	path.parent.mkdir(parents=True, exist_ok=True)
	with path.open("w", newline="", encoding="utf-8") as handle:
		writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
		writer.writeheader()
		writer.writerows(rows)


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


def _level_counts(metrics_rows: list[dict[str, Any]]) -> dict[int, int]:
	levels = [int(row["cycle_level"]) for row in metrics_rows]
	return {level: levels.count(level) for level in range(4)}


def _apply_commitment(dungeon: Any, commitment: StackelbergCommitment) -> None:
	dungeon.a = float(commitment.a)
	dungeon.b = float(commitment.b)
	dungeon.matrix_cross_coupling = float(commitment.matrix_cross_coupling)


class W3PulseAdapter:
	def __init__(self, *, config: W3PulseCellConfig, seed: int):
		self.config = config
		self.seed = int(seed)
		self.last_dungeon: Any | None = None
		self.prev_dominant: str | None = None
		self.p_hat_aggressive: float | None = None
		self.p_hat_defensive: float | None = None
		self.p_hat_balanced: float | None = None
		self.pulse_rounds_left = 0
		self.refractory_rounds_left = 0
		self.pulse_count = 0
		self.first_pulse_round: int | None = None
		self.dominant_transition_count = 0
		self.step_rows: list[dict[str, Any]] = []

	def flush_final(self) -> None:
		return None

	def _advance_counters(self) -> None:
		if self.pulse_rounds_left > 0:
			self.pulse_rounds_left -= 1
			if self.pulse_rounds_left == 0 and int(self.config.refractory_rounds) > 0:
				self.refractory_rounds_left = int(self.config.refractory_rounds)
			return
		if self.refractory_rounds_left > 0:
			self.refractory_rounds_left -= 1

	def __call__(
		self,
		round_index: int,
		_cfg: SimConfig,
		_players: list[object],
		dungeon: Any,
		_step_records: list[dict[str, object]],
		row: dict[str, Any],
	) -> None:
		self.last_dungeon = dungeon
		self._advance_counters()
		p_aggressive = float(row.get("p_aggressive") or 0.0)
		p_defensive = float(row.get("p_defensive") or 0.0)
		p_balanced = float(row.get("p_balanced") or 0.0)
		self.p_hat_aggressive = ema_step(previous=self.p_hat_aggressive, current=p_aggressive, alpha=float(self.config.ema_alpha))
		self.p_hat_defensive = ema_step(previous=self.p_hat_defensive, current=p_defensive, alpha=float(self.config.ema_alpha))
		self.p_hat_balanced = ema_step(previous=self.p_hat_balanced, current=p_balanced, alpha=float(self.config.ema_alpha))
		dominant_strategy = dominant_strategy_from_shares(
			aggressive=float(self.p_hat_aggressive),
			defensive=float(self.p_hat_defensive),
			balanced=float(self.p_hat_balanced),
		)
		dominant_changed = self.prev_dominant is not None and dominant_strategy != self.prev_dominant
		if dominant_changed:
			self.dominant_transition_count += 1
		pulse_started = False
		if (
			not self.config.is_control
			and dominant_changed
			and self.pulse_rounds_left == 0
			and self.refractory_rounds_left == 0
		):
			self.pulse_rounds_left = int(self.config.pulse_horizon)
			self.pulse_count += 1
			pulse_started = True
			if self.first_pulse_round is None:
				self.first_pulse_round = int(round_index) + 1
		pulse_active = self.pulse_rounds_left > 0
		selected_commitment = self.config.pulse_commitment if pulse_active and not self.config.is_control else self.config.baseline_commitment
		_apply_commitment(dungeon, selected_commitment)
		self.step_rows.append(
			{
				"condition": self.config.condition,
				"seed": self.seed,
				"round": int(round_index) + 1,
				"p_aggressive": _format_float(p_aggressive),
				"p_defensive": _format_float(p_defensive),
				"p_balanced": _format_float(p_balanced),
				"p_hat_aggressive": _format_float(float(self.p_hat_aggressive)),
				"p_hat_defensive": _format_float(float(self.p_hat_defensive)),
				"p_hat_balanced": _format_float(float(self.p_hat_balanced)),
				"dominant_strategy_ema": dominant_strategy,
				"dominant_changed": _yes_no(dominant_changed),
				"leader_action": selected_commitment.leader_action,
				"pulse_active": _yes_no(pulse_active),
				"pulse_started": _yes_no(pulse_started),
				"pulse_rounds_left": int(self.pulse_rounds_left),
				"refractory_rounds_left": int(self.refractory_rounds_left),
				"a": _format_float(float(selected_commitment.a)),
				"b": _format_float(float(selected_commitment.b)),
				"matrix_cross_coupling": _format_float(float(selected_commitment.matrix_cross_coupling)),
			}
		)
		self.prev_dominant = dominant_strategy


def _build_cell_config(
	*,
	condition: str,
	players: int,
	rounds: int,
	events_json: Path,
	selection_strength: float,
	init_bias: float,
	memory_kernel: int,
	burn_in: int,
	tail: int,
	ema_alpha: float,
	pulse_horizon: int,
	refractory_rounds: int,
	out_root: Path,
) -> W3PulseCellConfig:
	default_map = {config.condition: config for config in _default_pulse_configs()}
	base = default_map[condition]
	return W3PulseCellConfig(
		condition=base.condition,
		baseline_commitment=base.baseline_commitment,
		pulse_commitment=base.pulse_commitment,
		players=int(players),
		rounds=int(rounds),
		events_json=events_json,
		selection_strength=float(selection_strength),
		init_bias=float(init_bias),
		memory_kernel=int(memory_kernel),
		burn_in=int(burn_in),
		tail=int(tail),
		ema_alpha=float(ema_alpha),
		pulse_horizon=int(pulse_horizon),
		refractory_rounds=int(refractory_rounds),
		out_dir=out_root / condition,
	)


def run_w3_pulse_cell(config: W3PulseCellConfig, seed: int) -> dict[str, Any]:
	config.out_dir.mkdir(parents=True, exist_ok=True)
	adapter = W3PulseAdapter(config=config, seed=int(seed))
	baseline = config.baseline_commitment
	cfg = SimConfig(
		n_players=int(config.players),
		n_rounds=int(config.rounds),
		seed=int(seed),
		payoff_mode="matrix_ab",
		popularity_mode="sampled",
		gamma=0.1,
		epsilon=0.0,
		a=float(baseline.a),
		b=float(baseline.b),
		matrix_cross_coupling=float(baseline.matrix_cross_coupling),
		init_bias=float(config.init_bias),
		evolution_mode="sampled",
		payoff_lag=1,
		selection_strength=float(config.selection_strength),
		enable_events=True,
		events_json=config.events_json,
		out_csv=config.out_csv(int(seed)),
		memory_kernel=int(config.memory_kernel),
	)
	strategy_space, rows = simulate(cfg, round_callback=adapter)
	adapter.flush_final()
	_write_timeseries_csv(cfg.out_csv, strategy_space=strategy_space, rows=rows)
	pulse_steps_path = config.pulse_steps_path(int(seed))
	_write_tsv(pulse_steps_path, fieldnames=PULSE_STEP_FIELDNAMES, rows=adapter.step_rows)
	seed_metric = seed_metrics(rows, burn_in=int(config.burn_in), tail=int(config.tail), eta=0.55, corr_threshold=0.09)
	provenance_path = config.provenance_path(int(seed))
	provenance_payload = {
		"protocol": PROTOCOL,
		"condition": config.condition,
		"baseline_commitment": {
			"leader_action": baseline.leader_action,
			"a": float(baseline.a),
			"b": float(baseline.b),
			"matrix_cross_coupling": float(baseline.matrix_cross_coupling),
		},
		"pulse_commitment": {
			"leader_action": config.pulse_commitment.leader_action,
			"a": float(config.pulse_commitment.a),
			"b": float(config.pulse_commitment.b),
			"matrix_cross_coupling": float(config.pulse_commitment.matrix_cross_coupling),
		},
		"ema_alpha": float(config.ema_alpha),
		"pulse_horizon": int(config.pulse_horizon),
		"refractory_rounds": int(config.refractory_rounds),
		"event_provenance": summarize_event_provenance(cfg.out_csv, events_json=config.events_json),
	}
	provenance_path.write_text(json.dumps(provenance_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
	cycle_level = int(seed_metric["cycle_level"])
	has_level3_seed = cycle_level >= 3
	pulse_active_share = _safe_mean([1.0 if row["pulse_active"] == "yes" else 0.0 for row in adapter.step_rows])
	first_pulse_round = "" if adapter.first_pulse_round is None else int(adapter.first_pulse_round)
	selected_commitment = config.pulse_commitment if not config.is_control else baseline
	return {
		"protocol": PROTOCOL,
		"condition": config.condition,
		"leader_action": selected_commitment.leader_action,
		"a": _format_float(float(selected_commitment.a)),
		"b": _format_float(float(selected_commitment.b)),
		"matrix_cross_coupling": _format_float(float(selected_commitment.matrix_cross_coupling)),
		"seed": int(seed),
		"cycle_level": cycle_level,
		"mean_stage3_score": _format_float(float(seed_metric["stage3_score"])),
		"mean_turn_strength": _format_float(float(seed_metric["turn_strength"])),
		"mean_env_gamma": _format_float(float(seed_metric["env_gamma"])),
		"env_gamma_r2": _format_float(float(seed_metric["env_gamma_r2"])),
		"env_gamma_n_peaks": int(seed_metric["env_gamma_n_peaks"]),
		"mean_success_rate": _format_float(float(seed_metric["success_rate"])),
		"level3_seed_count": 1 if has_level3_seed else 0,
		"has_level3_seed": _yes_no(has_level3_seed),
		"pulse_count": int(adapter.pulse_count),
		"pulse_active_share": _format_float(pulse_active_share),
		"first_pulse_round": first_pulse_round,
		"dominant_transition_count": int(adapter.dominant_transition_count),
		"pulse_steps_tsv": str(pulse_steps_path),
		"out_csv": str(cfg.out_csv),
		"provenance_json": str(provenance_path),
	}


def _cell_summary(rows: list[dict[str, Any]], *, control_row: Mapping[str, Any] | None, config: W3PulseCellConfig) -> dict[str, Any]:
	mean_stage3_score = _safe_mean([float(row["mean_stage3_score"]) for row in rows])
	mean_turn_strength = _safe_mean([float(row["mean_turn_strength"]) for row in rows])
	mean_env_gamma = _safe_mean([float(row["mean_env_gamma"]) for row in rows])
	level_counts = _level_counts(rows)
	level3_seed_count = sum(1 for row in rows if int(row["cycle_level"]) >= 3)
	control_mean_env_gamma = mean_env_gamma if control_row is None else float(control_row["mean_env_gamma"])
	control_mean_stage3_score = mean_stage3_score if control_row is None else float(control_row["mean_stage3_score"])
	leader_prefers_over_control = (
		False
		if config.is_control or control_row is None
		else leader_prefers(
			{
				"condition": config.condition,
				"level3_seed_count": level3_seed_count,
				"mean_stage3_score": mean_stage3_score,
				"mean_env_gamma": mean_env_gamma,
				"mean_turn_strength": mean_turn_strength,
			},
			{
				"condition": str(control_row["condition"]),
				"level3_seed_count": int(control_row["level3_seed_count"]),
				"mean_stage3_score": float(control_row["mean_stage3_score"]),
				"mean_env_gamma": float(control_row["mean_env_gamma"]),
				"mean_turn_strength": float(control_row["mean_turn_strength"]),
			},
		)
	)
	pulse_activation_rate = _safe_mean([1.0 if int(row["pulse_count"]) > 0 else 0.0 for row in rows])
	mean_pulse_count = _safe_mean([float(row["pulse_count"]) for row in rows])
	mean_pulse_active_share = _safe_mean([float(row["pulse_active_share"]) for row in rows])
	first_pulse_rounds = [float(row["first_pulse_round"]) for row in rows if str(row["first_pulse_round"]) not in {"", "None"}]
	mean_first_pulse_round = _safe_mean(first_pulse_rounds)
	mean_dominant_transition_count = _safe_mean([float(row["dominant_transition_count"]) for row in rows])
	promotion_gate_pass = (
		not config.is_control
		and level3_seed_count >= 1
		and mean_env_gamma >= 0.0
		and mean_stage3_score >= control_mean_stage3_score
		and pulse_activation_rate > 0.0
	)
	if config.is_control:
		verdict = "control"
	elif promotion_gate_pass:
		verdict = "pass"
	elif pulse_activation_rate > 0.0 and leader_prefers_over_control:
		verdict = "weak_positive"
	else:
		verdict = "fail"
	return {
		"protocol": PROTOCOL,
		"condition": config.condition,
		"is_control": _yes_no(config.is_control),
		"leader_action": config.pulse_commitment.leader_action if not config.is_control else config.baseline_commitment.leader_action,
		"n_seeds": len(rows),
		"mean_stage3_score": _format_float(mean_stage3_score),
		"mean_turn_strength": _format_float(mean_turn_strength),
		"mean_env_gamma": _format_float(mean_env_gamma),
		"level_counts_json": json.dumps(level_counts, sort_keys=True),
		"level3_seed_count": int(level3_seed_count),
		"control_mean_env_gamma": _format_float(control_mean_env_gamma),
		"control_mean_stage3_score": _format_float(control_mean_stage3_score),
		"gamma_uplift_ratio_vs_control": _format_ratio(
			_gamma_uplift_ratio(baseline_gamma=control_mean_env_gamma, candidate_gamma=mean_env_gamma)
		),
		"stage3_uplift_vs_control": _format_float(mean_stage3_score - control_mean_stage3_score),
		"leader_prefers_over_control": _yes_no(leader_prefers_over_control),
		"pulse_activation_rate": _format_float(pulse_activation_rate),
		"mean_pulse_count": _format_float(mean_pulse_count),
		"mean_pulse_active_share": _format_float(mean_pulse_active_share),
		"mean_first_pulse_round": _format_float(mean_first_pulse_round),
		"mean_dominant_transition_count": _format_float(mean_dominant_transition_count),
		"is_best_policy": "no",
		"promotion_gate_pass": _yes_no(promotion_gate_pass),
		"verdict": verdict,
		"events_json": str(config.events_json),
		"selection_strength": _format_float(float(config.selection_strength)),
		"memory_kernel": int(config.memory_kernel),
		"ema_alpha": _format_float(float(config.ema_alpha)),
		"pulse_horizon": int(config.pulse_horizon),
		"refractory_rounds": int(config.refractory_rounds),
		"out_dir": str(config.out_dir),
	}


def _write_decision(path: Path, *, decision: str, control_row: Mapping[str, Any], combined_rows: list[dict[str, Any]], best_condition: str | None) -> None:
	best_row = next((row for row in combined_rows if row["condition"] == best_condition), None)
	lines = [
		"# W3.3 Pulse Leader Policy Decision",
		"",
		f"- protocol: {PROTOCOL}",
		f"- decision: {decision}",
		f"- control_mean_stage3_score: {control_row['mean_stage3_score']}",
		f"- control_mean_env_gamma: {control_row['mean_env_gamma']}",
		f"- control_level3_seed_count: {control_row['level3_seed_count']}",
		f"- best_policy: {'' if best_row is None else best_row['condition']}",
		f"- best_leader_action: {'' if best_row is None else best_row['leader_action']}",
		"",
		"## Policy Verdicts",
		"",
		"| Cell | leader_action | Level 3 seeds | mean_stage3_score | mean_env_gamma | pulse_activation_rate | mean_pulse_count | leader_prefers_over_control | verdict |",
		"|---|---|---:|---:|---:|---:|---:|---|---|",
	]
	for row in combined_rows:
		lines.append(
			f"| {row['condition']} | {row['leader_action']} | {row['level3_seed_count']} | {row['mean_stage3_score']} | {row['mean_env_gamma']} | {row['pulse_activation_rate']} | {row['mean_pulse_count']} | {row['leader_prefers_over_control']} | {row['verdict']} |"
		)
	lines.extend(
		[
			"",
			"## Gate",
			"",
			"- pass: 至少 1 個 non-control pulse policy cell 同時滿足 level3_seed_count >= 1、mean_env_gamma >= 0、mean_stage3_score 不低於 control、且 pulse_activation_rate > 0。",
			"- close_w3_3: 若所有 non-control pulse policy cells 都沒有任何 Level 3 seed，則 W3.3 直接結案，不再做 ema_alpha、pulse_horizon、refractory_rounds 或 pulse action 排列微調。",
			"",
			"## Interpretation",
			"",
			"- W3.3 的 value 在於檢查 event-driven finite pulse 是否能比 W3.2 的 sticky state-feedback 更有效地打到 sampled plateau 的薄弱點。",
			"- pulse_activation_rate 若接近 0，表示 trigger / pulse mechanics 幾乎沒有實際介入；此時即使 aggregate 指標略好，也不應視為強證據。",
		]
	)
	path.parent.mkdir(parents=True, exist_ok=True)
	path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _decide_overall(combined_rows: list[dict[str, Any]]) -> str:
	candidate_rows = [row for row in combined_rows if row["is_control"] == "no"]
	if any(row["promotion_gate_pass"] == "yes" for row in candidate_rows):
		return "pass"
	if all(int(row["level3_seed_count"]) == 0 for row in candidate_rows):
		return "close_w3_3"
	if any(
		(
			int(row["level3_seed_count"]) >= 1 or row["leader_prefers_over_control"] == "yes"
		)
		and float(row["pulse_activation_rate"]) > 0.0
		for row in candidate_rows
	):
		return "weak_positive"
	return "fail"


def run_w3_pulse_scout(
	*,
	seeds: list[int],
	out_root: Path,
	summary_tsv: Path,
	combined_tsv: Path,
	decision_md: Path,
	conditions: list[str] | None = None,
	events_json: Path = DEFAULT_FULL_EVENTS_JSON,
	players: int = 300,
	rounds: int = 3000,
	selection_strength: float = 0.06,
	init_bias: float = 0.12,
	memory_kernel: int = 3,
	burn_in: int = 1000,
	tail: int = 1000,
	ema_alpha: float = DEFAULT_EMA_ALPHA,
	pulse_horizon: int = DEFAULT_PULSE_HORIZON,
	refractory_rounds: int = DEFAULT_REFRACTORY_ROUNDS,
	cell_runner: CellRunner | None = None,
) -> dict[str, Any]:
	runner = cell_runner or run_w3_pulse_cell
	selected_conditions = set(str(value) for value in conditions) if conditions is not None else None
	out_root.mkdir(parents=True, exist_ok=True)
	condition_order = [config.condition for config in _default_pulse_configs()]
	cell_configs = [
		_build_cell_config(
			condition=condition,
			players=players,
			rounds=rounds,
			events_json=events_json,
			selection_strength=selection_strength,
			init_bias=init_bias,
			memory_kernel=memory_kernel,
			burn_in=burn_in,
			tail=tail,
			ema_alpha=ema_alpha,
			pulse_horizon=pulse_horizon,
			refractory_rounds=refractory_rounds,
			out_root=out_root,
		)
		for condition in condition_order
	]
	if selected_conditions is not None:
		cell_configs = [config for config in cell_configs if config.condition in selected_conditions]
		if not cell_configs:
			raise ValueError("no W3 pulse cells selected")
	summary_rows: list[dict[str, Any]] = []
	by_condition: dict[str, list[dict[str, Any]]] = {config.condition: [] for config in cell_configs}
	for config in cell_configs:
		for seed in seeds:
			row = dict(runner(config, int(seed)))
			summary_rows.append(row)
			by_condition[config.condition].append(row)
	_write_tsv(summary_tsv, fieldnames=SUMMARY_FIELDNAMES, rows=summary_rows)
	control_config = next((config for config in cell_configs if config.is_control), None)
	control_rows = by_condition.get("control_pulse_policy", [])
	control_base = None
	combined_rows: list[dict[str, Any]] = []
	if control_config is not None:
		control_base = _cell_summary(control_rows, control_row=None, config=control_config)
		combined_rows.append(control_base)
	for config in cell_configs:
		if config.is_control:
			continue
		combined_rows.append(_cell_summary(by_condition[config.condition], control_row=control_base, config=config))
	if control_config is not None:
		combined_rows[0] = _cell_summary(control_rows, control_row=combined_rows[0], config=control_config)
	best_condition = select_best_commitment(combined_rows)
	for row in combined_rows:
		row["is_best_policy"] = _yes_no((best_condition is not None) and row["condition"] == best_condition)
	_write_tsv(combined_tsv, fieldnames=COMBINED_FIELDNAMES, rows=combined_rows)
	if len(combined_rows) == 1 and combined_rows[0]["is_control"] == "yes":
		decision = "control_only"
		control_row = combined_rows[0]
	else:
		decision = _decide_overall(combined_rows)
		control_row = next(row for row in combined_rows if row["is_control"] == "yes")
	_write_decision(decision_md, decision=decision, control_row=control_row, combined_rows=combined_rows, best_condition=best_condition)
	return {
		"decision": decision,
		"summary_rows": summary_rows,
		"combined_rows": combined_rows,
	}


def main(*, cell_runner: CellRunner | None = None) -> int:
	parser = argparse.ArgumentParser(description="W3.3 event-driven pulse leader policy scout")
	parser.add_argument("--conditions", type=str, default="control_pulse_policy,w3_pulse_crossguard,w3_pulse_edgetilt,w3_pulse_commitpush")
	parser.add_argument("--seeds", type=str, default="45,47,49")
	parser.add_argument("--players", type=int, default=300)
	parser.add_argument("--rounds", type=int, default=3000)
	parser.add_argument("--selection-strength", type=float, default=0.06)
	parser.add_argument("--init-bias", type=float, default=0.12)
	parser.add_argument("--memory-kernel", type=int, default=3)
	parser.add_argument("--burn-in", type=int, default=1000)
	parser.add_argument("--tail", type=int, default=1000)
	parser.add_argument("--ema-alpha", type=float, default=DEFAULT_EMA_ALPHA)
	parser.add_argument("--pulse-horizon", type=int, default=DEFAULT_PULSE_HORIZON)
	parser.add_argument("--refractory-rounds", type=int, default=DEFAULT_REFRACTORY_ROUNDS)
	parser.add_argument("--events-json", type=Path, default=DEFAULT_FULL_EVENTS_JSON)
	parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
	parser.add_argument("--summary-tsv", type=Path, default=DEFAULT_SUMMARY_TSV)
	parser.add_argument("--combined-tsv", type=Path, default=DEFAULT_COMBINED_TSV)
	parser.add_argument("--decision-md", type=Path, default=DEFAULT_DECISION_MD)
	args = parser.parse_args()
	run_w3_pulse_scout(
		conditions=[part.strip() for part in str(args.conditions).split(",") if part.strip()],
		seeds=_parse_seeds(args.seeds),
		out_root=args.out_root,
		summary_tsv=args.summary_tsv,
		combined_tsv=args.combined_tsv,
		decision_md=args.decision_md,
		events_json=args.events_json,
		players=int(args.players),
		rounds=int(args.rounds),
		selection_strength=float(args.selection_strength),
		init_bias=float(args.init_bias),
		memory_kernel=int(args.memory_kernel),
		burn_in=int(args.burn_in),
		tail=int(args.tail),
		ema_alpha=float(args.ema_alpha),
		pulse_horizon=int(args.pulse_horizon),
		refractory_rounds=int(args.refractory_rounds),
		cell_runner=cell_runner,
	)
	return 0


if __name__ == "__main__":
	raise SystemExit(main())