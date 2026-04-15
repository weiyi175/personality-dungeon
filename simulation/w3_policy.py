from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

from analysis.event_provenance_summary import summarize_event_provenance
from core.stackelberg import StackelbergCommitment, dominance_gap_from_shares, hysteresis_gate, leader_prefers, select_best_commitment
from simulation.personality_gate0 import DEFAULT_FULL_EVENTS_JSON
from simulation.personality_gate1 import seed_metrics
from simulation.run_simulation import SimConfig, _write_timeseries_csv, simulate


REPO_ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = "w3_policy"
DEFAULT_OUT_ROOT = REPO_ROOT / "outputs" / "w3_policy"
DEFAULT_SUMMARY_TSV = REPO_ROOT / "outputs" / "w3_policy_summary.tsv"
DEFAULT_COMBINED_TSV = REPO_ROOT / "outputs" / "w3_policy_combined.tsv"
DEFAULT_DECISION_MD = REPO_ROOT / "outputs" / "w3_policy_decision.md"

DEFAULT_POLICY_UPDATE_INTERVAL = 150
DEFAULT_THETA_LOW = 0.08
DEFAULT_THETA_HIGH = 0.12

POLICY_STEP_FIELDNAMES = [
	"condition",
	"seed",
	"window_index",
	"round",
	"dominant_strategy",
	"p_aggressive",
	"p_defensive",
	"p_balanced",
	"dominance_gap",
	"leader_action",
	"regime_active",
	"regime_switched",
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
	"policy_activation_share",
	"n_policy_switches",
	"mean_dominance_gap",
	"policy_steps_tsv",
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
	"policy_activation_rate",
	"mean_policy_switches",
	"mean_dominance_gap",
	"is_best_policy",
	"promotion_gate_pass",
	"verdict",
	"events_json",
	"selection_strength",
	"memory_kernel",
	"policy_update_interval",
	"theta_low",
	"theta_high",
	"out_dir",
]

CellRunner = Callable[["W3PolicyCellConfig", int], Mapping[str, Any]]


@dataclass(frozen=True)
class W3PolicyCellConfig:
	condition: str
	baseline_commitment: StackelbergCommitment
	active_commitment: StackelbergCommitment
	players: int
	rounds: int
	events_json: Path
	selection_strength: float
	init_bias: float
	memory_kernel: int
	burn_in: int
	tail: int
	policy_update_interval: int
	theta_low: float
	theta_high: float
	out_dir: Path

	@property
	def is_control(self) -> bool:
		return self.condition == "control_policy"

	def out_csv(self, seed: int) -> Path:
		return self.out_dir / f"seed{int(seed)}.csv"

	def provenance_path(self, seed: int) -> Path:
		return self.out_dir / f"seed{int(seed)}_provenance.json"

	def policy_steps_path(self, seed: int) -> Path:
		return self.out_dir / f"w3_policy_steps_{self.condition}_seed{int(seed)}.tsv"


def _default_baseline_commitment() -> StackelbergCommitment:
	return StackelbergCommitment(
		condition="control_policy",
		leader_action="baseline_policy",
		a=1.00,
		b=0.90,
		matrix_cross_coupling=0.20,
		description="Baseline payoff geometry for W3.2 control",
	)


def _default_policy_configs() -> list[W3PolicyCellConfig]:
	baseline = _default_baseline_commitment()
	return [
		W3PolicyCellConfig(
			condition="control_policy",
			baseline_commitment=baseline,
			active_commitment=baseline,
			players=300,
			rounds=3000,
			events_json=DEFAULT_FULL_EVENTS_JSON,
			selection_strength=0.06,
			init_bias=0.12,
			memory_kernel=3,
			burn_in=1000,
			tail=1000,
			policy_update_interval=DEFAULT_POLICY_UPDATE_INTERVAL,
			theta_low=DEFAULT_THETA_LOW,
			theta_high=DEFAULT_THETA_HIGH,
			out_dir=DEFAULT_OUT_ROOT / "control_policy",
		),
		W3PolicyCellConfig(
			condition="w3_policy_crossguard",
			baseline_commitment=baseline,
			active_commitment=StackelbergCommitment(
				condition="w3_policy_crossguard",
				leader_action="cross_guard_policy",
				a=1.00,
				b=0.90,
				matrix_cross_coupling=0.35,
				description="Activate stronger cross penalty when dominance gap stays high",
			),
			players=300,
			rounds=3000,
			events_json=DEFAULT_FULL_EVENTS_JSON,
			selection_strength=0.06,
			init_bias=0.12,
			memory_kernel=3,
			burn_in=1000,
			tail=1000,
			policy_update_interval=DEFAULT_POLICY_UPDATE_INTERVAL,
			theta_low=DEFAULT_THETA_LOW,
			theta_high=DEFAULT_THETA_HIGH,
			out_dir=DEFAULT_OUT_ROOT / "w3_policy_crossguard",
		),
		W3PolicyCellConfig(
			condition="w3_policy_edgetilt",
			baseline_commitment=baseline,
			active_commitment=StackelbergCommitment(
				condition="w3_policy_edgetilt",
				leader_action="edge_tilt_policy",
				a=1.05,
				b=0.85,
				matrix_cross_coupling=0.20,
				description="Activate sharper cyclic edge when dominance gap stays high",
			),
			players=300,
			rounds=3000,
			events_json=DEFAULT_FULL_EVENTS_JSON,
			selection_strength=0.06,
			init_bias=0.12,
			memory_kernel=3,
			burn_in=1000,
			tail=1000,
			policy_update_interval=DEFAULT_POLICY_UPDATE_INTERVAL,
			theta_low=DEFAULT_THETA_LOW,
			theta_high=DEFAULT_THETA_HIGH,
			out_dir=DEFAULT_OUT_ROOT / "w3_policy_edgetilt",
		),
		W3PolicyCellConfig(
			condition="w3_policy_commitpush",
			baseline_commitment=baseline,
			active_commitment=StackelbergCommitment(
				condition="w3_policy_commitpush",
				leader_action="commit_push_policy",
				a=1.05,
				b=0.85,
				matrix_cross_coupling=0.35,
				description="Activate stronger edge and cross penalty when dominance gap stays high",
			),
			players=300,
			rounds=3000,
			events_json=DEFAULT_FULL_EVENTS_JSON,
			selection_strength=0.06,
			init_bias=0.12,
			memory_kernel=3,
			burn_in=1000,
			tail=1000,
			policy_update_interval=DEFAULT_POLICY_UPDATE_INTERVAL,
			theta_low=DEFAULT_THETA_LOW,
			theta_high=DEFAULT_THETA_HIGH,
			out_dir=DEFAULT_OUT_ROOT / "w3_policy_commitpush",
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


class W3PolicyAdapter:
	def __init__(self, *, config: W3PolicyCellConfig, seed: int):
		self.config = config
		self.seed = int(seed)
		self.window_rows: list[dict[str, float]] = []
		self.policy_rows: list[dict[str, Any]] = []
		self.window_index = 0
		self.regime_active = False
		self.last_dungeon: Any | None = None

	def _flush_window(self, *, round_index: int, dungeon: Any) -> None:
		if not self.window_rows:
			return
		mean_p_agg = _safe_mean([float(row["p_aggressive"]) for row in self.window_rows])
		mean_p_def = _safe_mean([float(row["p_defensive"]) for row in self.window_rows])
		mean_p_bal = _safe_mean([float(row["p_balanced"]) for row in self.window_rows])
		dominance_gap = dominance_gap_from_shares(
			aggressive=mean_p_agg,
			defensive=mean_p_def,
			balanced=mean_p_bal,
		)
		previous_active = bool(self.regime_active)
		self.regime_active = False if self.config.is_control else hysteresis_gate(
			value=dominance_gap,
			current_active=previous_active,
			theta_low=float(self.config.theta_low),
			theta_high=float(self.config.theta_high),
		)
		selected_commitment = self.config.active_commitment if self.regime_active else self.config.baseline_commitment
		_apply_commitment(dungeon, selected_commitment)
		dominant_strategy = max(
			(
				("aggressive", mean_p_agg),
				("defensive", mean_p_def),
				("balanced", mean_p_bal),
			),
			key=lambda item: (item[1], item[0]),
		)[0]
		self.policy_rows.append(
			{
				"condition": self.config.condition,
				"seed": self.seed,
				"window_index": self.window_index,
				"round": int(round_index + 1),
				"dominant_strategy": dominant_strategy,
				"p_aggressive": _format_float(mean_p_agg),
				"p_defensive": _format_float(mean_p_def),
				"p_balanced": _format_float(mean_p_bal),
				"dominance_gap": _format_float(dominance_gap),
				"leader_action": selected_commitment.leader_action,
				"regime_active": _yes_no(self.regime_active),
				"regime_switched": _yes_no(previous_active != self.regime_active),
				"a": _format_float(float(selected_commitment.a)),
				"b": _format_float(float(selected_commitment.b)),
				"matrix_cross_coupling": _format_float(float(selected_commitment.matrix_cross_coupling)),
			}
		)
		self.window_rows = []
		self.window_index += 1

	def flush_final(self) -> None:
		if self.last_dungeon is None:
			return
		if self.window_rows:
			self._flush_window(round_index=int(self.config.rounds) - 1, dungeon=self.last_dungeon)

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
		self.window_rows.append(
			{
				"p_aggressive": float(row.get("p_aggressive") or 0.0),
				"p_defensive": float(row.get("p_defensive") or 0.0),
				"p_balanced": float(row.get("p_balanced") or 0.0),
			}
		)
		if (round_index + 1) % int(self.config.policy_update_interval) == 0:
			self._flush_window(round_index=round_index, dungeon=dungeon)


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
	policy_update_interval: int,
	theta_low: float,
	theta_high: float,
	out_root: Path,
) -> W3PolicyCellConfig:
	default_map = {config.condition: config for config in _default_policy_configs()}
	base = default_map[condition]
	return W3PolicyCellConfig(
		condition=base.condition,
		baseline_commitment=base.baseline_commitment,
		active_commitment=base.active_commitment,
		players=int(players),
		rounds=int(rounds),
		events_json=events_json,
		selection_strength=float(selection_strength),
		init_bias=float(init_bias),
		memory_kernel=int(memory_kernel),
		burn_in=int(burn_in),
		tail=int(tail),
		policy_update_interval=int(policy_update_interval),
		theta_low=float(theta_low),
		theta_high=float(theta_high),
		out_dir=out_root / condition,
	)


def run_w3_policy_cell(config: W3PolicyCellConfig, seed: int) -> dict[str, Any]:
	config.out_dir.mkdir(parents=True, exist_ok=True)
	adapter = W3PolicyAdapter(config=config, seed=int(seed))
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
	policy_steps_path = config.policy_steps_path(int(seed))
	_write_tsv(policy_steps_path, fieldnames=POLICY_STEP_FIELDNAMES, rows=adapter.policy_rows)
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
		"active_commitment": {
			"leader_action": config.active_commitment.leader_action,
			"a": float(config.active_commitment.a),
			"b": float(config.active_commitment.b),
			"matrix_cross_coupling": float(config.active_commitment.matrix_cross_coupling),
		},
		"policy_update_interval": int(config.policy_update_interval),
		"theta_low": float(config.theta_low),
		"theta_high": float(config.theta_high),
		"event_provenance": summarize_event_provenance(cfg.out_csv, events_json=config.events_json),
	}
	provenance_path.write_text(json.dumps(provenance_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
	cycle_level = int(seed_metric["cycle_level"])
	has_level3_seed = cycle_level >= 3
	policy_activation_share = _safe_mean([1.0 if row["regime_active"] == "yes" else 0.0 for row in adapter.policy_rows])
	n_policy_switches = sum(1 for row in adapter.policy_rows if row["regime_switched"] == "yes")
	mean_dominance_gap = _safe_mean([float(row["dominance_gap"]) for row in adapter.policy_rows])
	return {
		"protocol": PROTOCOL,
		"condition": config.condition,
		"leader_action": config.active_commitment.leader_action if not config.is_control else baseline.leader_action,
		"a": _format_float(float(config.active_commitment.a if not config.is_control else baseline.a)),
		"b": _format_float(float(config.active_commitment.b if not config.is_control else baseline.b)),
		"matrix_cross_coupling": _format_float(
			float(config.active_commitment.matrix_cross_coupling if not config.is_control else baseline.matrix_cross_coupling)
		),
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
		"policy_activation_share": _format_float(policy_activation_share),
		"n_policy_switches": int(n_policy_switches),
		"mean_dominance_gap": _format_float(mean_dominance_gap),
		"policy_steps_tsv": str(policy_steps_path),
		"out_csv": str(cfg.out_csv),
		"provenance_json": str(provenance_path),
	}


def _cell_summary(rows: list[dict[str, Any]], *, control_row: Mapping[str, Any] | None, config: W3PolicyCellConfig) -> dict[str, Any]:
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
	policy_activation_rate = _safe_mean([float(row["policy_activation_share"]) for row in rows])
	mean_policy_switches = _safe_mean([float(row["n_policy_switches"]) for row in rows])
	mean_dominance_gap = _safe_mean([float(row["mean_dominance_gap"]) for row in rows])
	promotion_gate_pass = (
		not config.is_control
		and level3_seed_count >= 1
		and mean_env_gamma >= 0.0
		and mean_stage3_score >= control_mean_stage3_score
	)
	if config.is_control:
		verdict = "control"
	elif promotion_gate_pass:
		verdict = "pass"
	elif leader_prefers_over_control and policy_activation_rate > 0.0:
		verdict = "weak_positive"
	else:
		verdict = "fail"
	return {
		"protocol": PROTOCOL,
		"condition": config.condition,
		"is_control": _yes_no(config.is_control),
		"leader_action": config.active_commitment.leader_action if not config.is_control else config.baseline_commitment.leader_action,
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
		"policy_activation_rate": _format_float(policy_activation_rate),
		"mean_policy_switches": _format_float(mean_policy_switches),
		"mean_dominance_gap": _format_float(mean_dominance_gap),
		"is_best_policy": "no",
		"promotion_gate_pass": _yes_no(promotion_gate_pass),
		"verdict": verdict,
		"events_json": str(config.events_json),
		"selection_strength": _format_float(float(config.selection_strength)),
		"memory_kernel": int(config.memory_kernel),
		"policy_update_interval": int(config.policy_update_interval),
		"theta_low": _format_float(float(config.theta_low)),
		"theta_high": _format_float(float(config.theta_high)),
		"out_dir": str(config.out_dir),
	}


def _write_decision(path: Path, *, decision: str, control_row: Mapping[str, Any], combined_rows: list[dict[str, Any]], best_condition: str | None) -> None:
	best_row = next((row for row in combined_rows if row["condition"] == best_condition), None)
	lines = [
		"# W3.2 Leader Policy Decision",
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
		"| Cell | leader_action | Level 3 seeds | mean_stage3_score | mean_env_gamma | policy_activation_rate | leader_prefers_over_control | verdict |",
		"|---|---|---:|---:|---:|---:|---|---|",
	]
	for row in combined_rows:
		lines.append(
			f"| {row['condition']} | {row['leader_action']} | {row['level3_seed_count']} | {row['mean_stage3_score']} | {row['mean_env_gamma']} | {row['policy_activation_rate']} | {row['leader_prefers_over_control']} | {row['verdict']} |"
		)
	lines.extend(
		[
			"",
			"## Gate",
			"",
			"- pass: 至少 1 個 non-control policy cell 同時滿足 level3_seed_count >= 1、mean_env_gamma >= 0、且 mean_stage3_score 不低於 control。",
			"- close_w3_2: 若所有 non-control policy cells 都沒有任何 Level 3 seed，則 W3.2 直接結案，不再做 theta、interval、或 regime 組合微調。",
			"",
			"## Interpretation",
			"",
			"- W3.2 的 value 在於檢查低頻 state-feedback 是否能打破固定 commitment 的 plateau，而不是找一個新的靜態 cell。",
			"- policy_activation_rate 若接近 0，表示該 policy 幾乎沒有真正進入 active regime；此時即使 aggregate 好看，也不應視為強證據。",
		]
	)
	path.parent.mkdir(parents=True, exist_ok=True)
	path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _decide_overall(combined_rows: list[dict[str, Any]]) -> str:
	candidate_rows = [row for row in combined_rows if row["is_control"] == "no"]
	if any(row["promotion_gate_pass"] == "yes" for row in candidate_rows):
		return "pass"
	if all(int(row["level3_seed_count"]) == 0 for row in candidate_rows):
		return "close_w3_2"
	if any(row["verdict"] == "weak_positive" for row in candidate_rows):
		return "weak_positive"
	return "fail"


def run_w3_policy_scout(
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
	policy_update_interval: int = DEFAULT_POLICY_UPDATE_INTERVAL,
	theta_low: float = DEFAULT_THETA_LOW,
	theta_high: float = DEFAULT_THETA_HIGH,
	cell_runner: CellRunner | None = None,
) -> dict[str, Any]:
	runner = cell_runner or run_w3_policy_cell
	selected_conditions = set(str(value) for value in conditions) if conditions is not None else None
	out_root.mkdir(parents=True, exist_ok=True)
	condition_order = [config.condition for config in _default_policy_configs()]
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
			policy_update_interval=policy_update_interval,
			theta_low=theta_low,
			theta_high=theta_high,
			out_root=out_root,
		)
		for condition in condition_order
	]
	if selected_conditions is not None:
		cell_configs = [config for config in cell_configs if config.condition in selected_conditions]
		if not cell_configs:
			raise ValueError("no W3 policy cells selected")
	summary_rows: list[dict[str, Any]] = []
	by_condition: dict[str, list[dict[str, Any]]] = {config.condition: [] for config in cell_configs}
	for config in cell_configs:
		for seed in seeds:
			row = dict(runner(config, int(seed)))
			summary_rows.append(row)
			by_condition[config.condition].append(row)
	_write_tsv(summary_tsv, fieldnames=SUMMARY_FIELDNAMES, rows=summary_rows)
	control_config = next((config for config in cell_configs if config.is_control), None)
	control_rows = by_condition.get("control_policy", [])
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
	parser = argparse.ArgumentParser(description="W3.2 state-feedback leader policy scout")
	parser.add_argument("--conditions", type=str, default="control_policy,w3_policy_crossguard,w3_policy_edgetilt,w3_policy_commitpush")
	parser.add_argument("--seeds", type=str, default="45,47,49")
	parser.add_argument("--players", type=int, default=300)
	parser.add_argument("--rounds", type=int, default=3000)
	parser.add_argument("--selection-strength", type=float, default=0.06)
	parser.add_argument("--init-bias", type=float, default=0.12)
	parser.add_argument("--memory-kernel", type=int, default=3)
	parser.add_argument("--burn-in", type=int, default=1000)
	parser.add_argument("--tail", type=int, default=1000)
	parser.add_argument("--policy-update-interval", type=int, default=DEFAULT_POLICY_UPDATE_INTERVAL)
	parser.add_argument("--theta-low", type=float, default=DEFAULT_THETA_LOW)
	parser.add_argument("--theta-high", type=float, default=DEFAULT_THETA_HIGH)
	parser.add_argument("--events-json", type=Path, default=DEFAULT_FULL_EVENTS_JSON)
	parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
	parser.add_argument("--summary-tsv", type=Path, default=DEFAULT_SUMMARY_TSV)
	parser.add_argument("--combined-tsv", type=Path, default=DEFAULT_COMBINED_TSV)
	parser.add_argument("--decision-md", type=Path, default=DEFAULT_DECISION_MD)
	args = parser.parse_args()
	run_w3_policy_scout(
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
		policy_update_interval=int(args.policy_update_interval),
		theta_low=float(args.theta_low),
		theta_high=float(args.theta_high),
		cell_runner=cell_runner,
	)
	return 0


if __name__ == "__main__":
	raise SystemExit(main())