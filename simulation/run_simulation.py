from core.game_engine import GameEngine
from dungeon.dungeon_ai import DungeonAI
from dungeon.event_loader import EventLoader
from analysis.metrics import average_reward, average_utility, strategy_distribution
from evolution.replicator_dynamics import replicator_step
from players.base_player import BasePlayer

import argparse
import csv
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from math import exp


@dataclass(frozen=True)
class SimConfig:
	n_players: int
	n_rounds: int
	seed: int | None
	payoff_mode: str
	popularity_mode: str
	gamma: float
	epsilon: float
	a: float
	b: float
	matrix_cross_coupling: float
	init_bias: float
	evolution_mode: str
	payoff_lag: int
	selection_strength: float
	enable_events: bool
	events_json: Path | None
	out_csv: Path
	event_failure_threshold: float | None = None
	event_health_penalty: float | None = None
	event_stress_risk_coefficient: float | None = None
	risk_ma_alpha: float | None = None
	risk_ma_multiplier: float | None = None
	stress_decay_c: float | None = None
	stress_decay_beta: float | None = None
	adaptive_ft_strength: float | None = None
	ft_update_interval: int | None = None
	adaptive_payoff_strength: float = 0.0
	payoff_update_interval: int = 500
	adaptive_payoff_target: float = 0.30


DEFAULT_EVENTS_JSON = Path(__file__).resolve().parents[1] / "docs" / "personality_dungeon_v1" / "02_event_templates_v1.json"


def _initial_weights(*, strategy_space: list[str], init_bias: float) -> dict[str, float]:
	if not strategy_space:
		return {}
	b = float(init_bias)
	if b == 0.0:
		return {s: 1.0 for s in strategy_space}
	if abs(b) >= 1.0:
		raise ValueError("init_bias must satisfy |init_bias| < 1.0 (keeps weights positive)")
	# Deterministic symmetry breaking: bias aggressive up and defensive down.
	# This is intentionally minimal but sufficient to escape the uniform fixed point.
	w = {s: 1.0 for s in strategy_space}
	if len(strategy_space) >= 1:
		w[strategy_space[0]] = 1.0 + b
	if len(strategy_space) >= 2:
		w[strategy_space[1]] = 1.0 - b
	return w


def _parse_args() -> SimConfig:
	p = argparse.ArgumentParser(description="personality-dungeon MVP simulation")
	p.add_argument("--players", type=int, default=50)
	p.add_argument("--rounds", type=int, default=20)
	p.add_argument(
		"--seed",
		type=int,
		default=None,
		help="Random seed for reproducible simulation (controls BasePlayer sampling)",
	)
	p.add_argument(
		"--payoff-mode",
		type=str,
		default="count_cycle",
		choices=["count_cycle", "matrix_ab"],
		help="Dungeon payoff model: count-based cyclic penalty/bonus, or matrix U=Ax using (a,b)",
	)
	p.add_argument(
		"--popularity-mode",
		type=str,
		default="sampled",
		choices=["sampled", "expected"],
		help="How dungeon popularity is updated each round: sampled choices (default) or expected distribution from weights (denoised)",
	)
	p.add_argument(
		"--evolution-mode",
		type=str,
		default="sampled",
		choices=["sampled", "mean_field"],
		help="Weight update rule. 'sampled' uses per-strategy averages from sampled players (default). 'mean_field' uses deterministic expected rewards u=Ax and replicator mapping (matrix_ab only).",
	)
	p.add_argument(
		"--payoff-lag",
		type=int,
		default=1,
		choices=[0, 1],
		help="Only used in evolution_mode=mean_field. 0: use current x_t for payoff u(x_t). 1: use lagged x_{t-1} for payoff u(x_{t-1}).",
	)
	p.add_argument("--gamma", type=float, default=0.1)
	p.add_argument("--epsilon", type=float, default=0.0)
	p.add_argument("--a", type=float, default=0.0, help="matrix_ab only: payoff matrix parameter a")
	p.add_argument("--b", type=float, default=0.0, help="matrix_ab only: payoff matrix parameter b")
	p.add_argument(
		"--matrix-cross-coupling",
		type=float,
		default=0.0,
		help="matrix_ab only: aggressive-defensive coexistence penalty strength c_AD (default: 0.0)",
	)
	p.add_argument(
		"--init-bias",
		type=float,
		default=0.0,
		help="Initial symmetry-breaking bias applied to weights: w0=(1+bias,1-bias,1) for (aggressive,defensive,balanced). Use with popularity_mode=expected to get deterministic phase rotation; requires |bias|<1.",
	)
	p.add_argument("--selection-strength", type=float, default=0.05)
	p.add_argument(
		"--enable-events",
		action="store_true",
		help="Enable Personality Dungeon event processing using the event schema JSON.",
	)
	p.add_argument(
		"--events-json",
		type=Path,
		default=None,
		help="Path to the Personality Dungeon event schema JSON. Defaults to docs/personality_dungeon_v1/02_event_templates_v1.json when --enable-events is set.",
	)
	p.add_argument(
		"--out",
		type=Path,
		default=Path("outputs") / "timeseries.csv",
		help="CSV output path (will create parent dirs)",
	)
	p.add_argument(
		"--event-failure-threshold",
		type=float,
		default=None,
		help="Override all per-action failure_threshold values with this value (grid search use). Range [0,1].",
	)
	p.add_argument(
		"--event-health-penalty",
		type=float,
		default=None,
		help="Override health penalty coefficient in final_risk (default: 0.10).",
	)
	p.add_argument(
		"--event-stress-risk-coefficient",
		type=float,
		default=None,
		help="Override stress→risk coupling coefficient in final_risk (default: 0.10).",
	)
	p.add_argument(
		"--risk-ma-alpha",
		type=float,
		default=None,
		help="EMA smoothing factor \u03b1 for risk memory (0=disabled, 0.9=slow decay). Default: 0.0.",
	)
	p.add_argument(
		"--risk-ma-multiplier",
		type=float,
		default=None,
		help="Weight of EMA risk_ma contribution to final_risk. Default: 0.0.",
	)
	p.add_argument(
		"--stress-decay-c",
		type=float,
		default=None,
		help="Nonlinear stress-decay coefficient c (0=disabled). effective_rate=1-(1-base)/(1+c*stress^beta). Default: 0.0.",
	)
	p.add_argument(
		"--stress-decay-beta",
		type=float,
		default=None,
		help="Exponent beta for stress-dependent decay asymmetry (default: 2.0).",
	)
	p.add_argument(
		"--adaptive-ft-strength",
		type=float,
		default=None,
		help="Adaptive failure-threshold coupling strength s (0=disabled). ft_new=ft_base*(1+s*(p_agg-1/3)). Default: 0.0.",
	)
	p.add_argument(
		"--ft-update-interval",
		type=int,
		default=None,
		help="Rounds between adaptive ft updates (default: 500).",
	)
	p.add_argument(
		"--adaptive-payoff-strength",
		type=float,
		default=0.0,
		help="Adaptive strength for a/b payoff adjustment based on p_agg (0=disabled).",
	)
	p.add_argument(
		"--payoff-update-interval",
		type=int,
		default=500,
		help="Rounds between adaptive payoff matrix updates (default: 500).",
	)
	p.add_argument(
		"--adaptive-payoff-target",
		type=float,
		default=0.30,
		help="Target p_agg level for adaptive payoff feedback (default: 0.30).",
	)
	a = p.parse_args()
	return SimConfig(
		n_players=a.players,
		n_rounds=a.rounds,
		seed=a.seed,
		payoff_mode=a.payoff_mode,
		popularity_mode=a.popularity_mode,
		gamma=a.gamma,
		epsilon=a.epsilon,
		a=a.a,
		b=a.b,
		matrix_cross_coupling=a.matrix_cross_coupling,
		init_bias=a.init_bias,
		evolution_mode=a.evolution_mode,
		payoff_lag=a.payoff_lag,
		selection_strength=a.selection_strength,
		enable_events=bool(a.enable_events),
		events_json=a.events_json,
		out_csv=a.out,
		event_failure_threshold=a.event_failure_threshold,
		event_health_penalty=a.event_health_penalty,
		event_stress_risk_coefficient=a.event_stress_risk_coefficient,
		risk_ma_alpha=a.risk_ma_alpha,
		risk_ma_multiplier=a.risk_ma_multiplier,
		stress_decay_c=a.stress_decay_c,
		stress_decay_beta=a.stress_decay_beta,
		adaptive_ft_strength=a.adaptive_ft_strength,
		ft_update_interval=a.ft_update_interval,
		adaptive_payoff_strength=a.adaptive_payoff_strength,
		payoff_update_interval=a.payoff_update_interval,
		adaptive_payoff_target=a.adaptive_payoff_target,
	)


def _normalize_simplex(weights: dict[str, float]) -> dict[str, float]:
	total = sum(float(v) for v in weights.values())
	if total <= 0:
		n = len(weights)
		return {k: (1.0 / float(n)) for k in weights} if n else {}
	return {k: (float(v) / float(total)) for k, v in weights.items()}


def _matrix_ab_payoff_vec(
	*,
	strategy_space: list[str],
	a: float,
	b: float,
	matrix_cross_coupling: float,
	x: dict[str, float],
) -> dict[str, float]:
	"""Compute u = A x for the matrix_ab cyclic payoff.

	A = [[0, a, -b],
	     [-b, 0, a],
	     [a, -b, 0]]

	Optional cross term c_AD adds (-c*x_D, -c*x_A, c*(x_A+x_D)).
	"""
	if len(strategy_space) != 3:
		raise ValueError("matrix_ab requires exactly 3 strategies")
	x0 = float(x.get(strategy_space[0], 0.0))
	x1 = float(x.get(strategy_space[1], 0.0))
	x2 = float(x.get(strategy_space[2], 0.0))
	A = (
		(0.0, float(a), -float(b)),
		(-float(b), 0.0, float(a)),
		(float(a), -float(b), 0.0),
	)
	u0 = A[0][0] * x0 + A[0][1] * x1 + A[0][2] * x2
	u1 = A[1][0] * x0 + A[1][1] * x1 + A[1][2] * x2
	u2 = A[2][0] * x0 + A[2][1] * x1 + A[2][2] * x2
	c = float(matrix_cross_coupling)
	if c != 0.0:
		u0 += -c * x1
		u1 += -c * x0
		u2 += c * (x0 + x1)
	return {
		strategy_space[0]: float(u0),
		strategy_space[1]: float(u1),
		strategy_space[2]: float(u2),
	}


def _write_timeseries_csv(
	out_csv: Path,
	*,
	strategy_space: list[str],
	rows: list[dict],
) -> None:
	out_csv.parent.mkdir(parents=True, exist_ok=True)
	fieldnames = (
		["round", "avg_reward", "avg_utility"]
		+ [f"p_{s}" for s in strategy_space]
		+ [f"w_{s}" for s in strategy_space]
		+ [
			"event_count",
			"success_count",
			"event_types_json",
			"event_ids_json",
			"action_names_json",
			"result_kinds_json",
			"successes_json",
			"final_risks_json",
			"success_probs_json",
			"trait_deltas_json",
			"trait_deltas_per_event_json",
			"popularity_shift_json",
			"state_effects_json",
		]
	)
	with out_csv.open("w", newline="") as f:
		writer = csv.DictWriter(f, fieldnames=fieldnames)
		writer.writeheader()
		writer.writerows(rows)


def _aggregate_event_records(step_records: list[dict]) -> dict[str, object]:
	event_records = [record.get("event_result") for record in step_records if record.get("event_result")]
	if not event_records:
		return {
			"event_count": 0,
			"success_count": 0,
			"event_types_json": "[]",
			"event_ids_json": "[]",
			"action_names_json": "[]",
			"result_kinds_json": "[]",
			"successes_json": "[]",
			"final_risks_json": "[]",
			"success_probs_json": "[]",
			"trait_deltas_json": "{}",
			"trait_deltas_per_event_json": "{}",
			"popularity_shift_json": "{}",
			"state_effects_json": "{}",
		}

	def _sum_dicts(items: list[dict[str, float]]) -> dict[str, float]:
		merged: dict[str, float] = {}
		for item in items:
			for key, value in item.items():
				merged[key] = float(merged.get(key, 0.0)) + float(value)
		return merged

	trait_deltas = _sum_dicts([dict(record.get("trait_deltas", {})) for record in event_records])
	popularity_shift = _sum_dicts([dict(record.get("popularity_shift", {})) for record in event_records])
	state_effects = _sum_dicts([dict(record.get("state_effects", {})) for record in event_records])
	trait_deltas_per_event: dict[str, dict[str, float]] = {}
	for record in event_records:
		event_id = str(record.get("event_id"))
		bucket = trait_deltas_per_event.setdefault(event_id, {})
		for key, value in dict(record.get("trait_deltas", {})).items():
			bucket[str(key)] = float(bucket.get(str(key), 0.0)) + float(value)

	return {
		"event_count": len(event_records),
		"success_count": sum(1 for record in event_records if bool(record.get("success"))),
		"event_types_json": json.dumps([record.get("event_type") for record in event_records], sort_keys=True),
		"event_ids_json": json.dumps([record.get("event_id") for record in event_records], sort_keys=True),
		"action_names_json": json.dumps([record.get("action_name") for record in event_records], sort_keys=True),
		"result_kinds_json": json.dumps([record.get("result_kind") for record in event_records], sort_keys=True),
		"successes_json": json.dumps([bool(record.get("success")) for record in event_records]),
		"final_risks_json": json.dumps([float(record.get("final_risk", 0.0)) for record in event_records]),
		"success_probs_json": json.dumps([float(record.get("success_prob", 0.0)) for record in event_records]),
		"trait_deltas_json": json.dumps(trait_deltas, sort_keys=True),
		"trait_deltas_per_event_json": json.dumps(trait_deltas_per_event, sort_keys=True),
		"popularity_shift_json": json.dumps(popularity_shift, sort_keys=True),
		"state_effects_json": json.dumps(state_effects, sort_keys=True),
	}


def simulate(cfg: SimConfig) -> tuple[list[str], list[dict]]:
	"""Run a simulation and return (strategy_space, rows) without any I/O.

	This is the preferred entry point for programmatic experiments (e.g. multi-seed stability)
	while keeping the CLI `main()` as a thin wrapper.
	"""

	strategy_space = ["aggressive", "defensive", "balanced"]
	if str(cfg.evolution_mode) == "mean_field":
		if str(cfg.payoff_mode) != "matrix_ab":
			raise ValueError("evolution_mode='mean_field' currently supports payoff_mode='matrix_ab' only")
		if int(cfg.payoff_lag) not in (0, 1):
			raise ValueError("payoff_lag must be 0 or 1")
		k = float(cfg.selection_strength)
		w_cur = _initial_weights(strategy_space=strategy_space, init_bias=float(cfg.init_bias))
		# Normalize weights to mean=1 (same convention as replicator_step output).
		m0 = (sum(w_cur.values()) / float(len(w_cur))) if w_cur else 1.0
		w_cur = {s: (float(v) / float(m0)) for s, v in w_cur.items()}
		x_cur = _normalize_simplex(w_cur)
		x_prev = dict(x_cur)
		rows: list[dict] = []
		for t in range(int(cfg.n_rounds)):
			x_pay = x_prev if int(cfg.payoff_lag) == 1 else x_cur
			u = _matrix_ab_payoff_vec(
				strategy_space=strategy_space,
				a=float(cfg.a),
				b=float(cfg.b),
				matrix_cross_coupling=float(cfg.matrix_cross_coupling),
				x=x_pay,
			)
			u_bar = sum(float(x_cur[s]) * float(u[s]) for s in strategy_space)
			w_next_raw = {s: (float(w_cur[s]) * exp(k * (float(u[s]) - float(u_bar)))) for s in strategy_space}
			m = sum(w_next_raw.values()) / float(len(w_next_raw))
			w_cur = {s: (float(v) / float(m)) for s, v in w_next_raw.items()} if m > 0 else {s: 1.0 for s in strategy_space}
			x_next = _normalize_simplex(w_cur)
			row = {"round": t, "avg_reward": "", "avg_utility": 0.0}
			for s in strategy_space:
				row[f"p_{s}"] = float(x_next[s])
			for s in strategy_space:
				row[f"w_{s}"] = float(w_cur[s])
			rows.append(row)
			x_prev = x_cur
			x_cur = x_next
		return strategy_space, rows

	# Deterministic sampling: give each player an independent RNG stream.
	# This avoids accidental coupling via a single shared RNG and makes runs reproducible.
	if cfg.seed is None:
		players = [BasePlayer(strategy_space) for _ in range(cfg.n_players)]
	else:
		players = [
			BasePlayer(strategy_space, rng=random.Random(int(cfg.seed) + i))
			for i in range(cfg.n_players)
		]

	event_loader = None
	if bool(cfg.enable_events):
		events_json = cfg.events_json if cfg.events_json is not None else DEFAULT_EVENTS_JSON
		event_loader = EventLoader(
			events_json,
			failure_threshold_override=cfg.event_failure_threshold,
			health_penalty_coefficient=(
				float(cfg.event_health_penalty)
				if cfg.event_health_penalty is not None
				else 0.10
			),
			stress_risk_coefficient=(
				float(cfg.event_stress_risk_coefficient)
				if cfg.event_stress_risk_coefficient is not None
				else 0.10
			),
			risk_ma_alpha=(float(cfg.risk_ma_alpha) if cfg.risk_ma_alpha is not None else 0.0),
			risk_ma_multiplier=(float(cfg.risk_ma_multiplier) if cfg.risk_ma_multiplier is not None else 0.0),
			stress_decay_c=(float(cfg.stress_decay_c) if cfg.stress_decay_c is not None else 0.0),
			stress_decay_beta=(float(cfg.stress_decay_beta) if cfg.stress_decay_beta is not None else 2.0),
		)

	# Symmetry breaking: start away from the uniform equilibrium.
	init_w = _initial_weights(strategy_space=strategy_space, init_bias=float(cfg.init_bias))
	if init_w:
		for pl in players:
			pl.update_weights(init_w)

	dungeon = DungeonAI(
		payoff_mode=cfg.payoff_mode,
		gamma=cfg.gamma,
		epsilon=cfg.epsilon,
		a=cfg.a,
		b=cfg.b,
		matrix_cross_coupling=cfg.matrix_cross_coupling,
		strategy_cycle=strategy_space,
		event_loader=event_loader,
		event_rng=random.Random(cfg.seed) if cfg.seed is not None else random.Random(),
	)
	engine = GameEngine(players, dungeon, popularity_mode=str(cfg.popularity_mode))
	if str(cfg.popularity_mode) == "expected":
		# Without this, round-0 popularity is empty -> identical rewards -> no movement.
		dungeon.popularity = dict(init_w)

	rows: list[dict] = []
	new_weights = {s: 1.0 for s in strategy_space}

	for t in range(cfg.n_rounds):
		step_records = engine.step()

		dist = strategy_distribution(players, strategy_space)
		avg_u = average_utility(players)
		avg_r = average_reward(players)

		# 演化更新：根據每一輪的 last_reward 來調整下一輪抽樣權重
		new_weights = replicator_step(
			players,
			strategy_space,
			selection_strength=cfg.selection_strength,
		)
		for pl in players:
			pl.update_weights(new_weights)

		row = {
			"round": t,
			"avg_reward": "" if avg_r is None else float(avg_r),
			"avg_utility": float(avg_u),
		}
		for s in strategy_space:
			row[f"p_{s}"] = float(dist[s])
		for s in strategy_space:
			row[f"w_{s}"] = float(new_weights[s])
		row.update(_aggregate_event_records(step_records))
		rows.append(row)

		# Adaptive rule-mutation: periodically adjust failure threshold based on p_aggressive.
		if event_loader is not None and (cfg.adaptive_ft_strength or 0.0) > 0.0:
			interval = int(cfg.ft_update_interval) if cfg.ft_update_interval is not None else 500
			if (t + 1) % interval == 0:
				ft_base = float(cfg.event_failure_threshold) if cfg.event_failure_threshold is not None else 0.72
				p_agg = float(dist.get("aggressive", 1.0 / 3.0))
				ft_new = ft_base * (1.0 + float(cfg.adaptive_ft_strength) * (p_agg - 1.0 / 3.0))
				event_loader.set_failure_threshold_override(ft_new)

		# Adaptive payoff feedback v2: additive formula + clipping.
		# delta = strength * (p_agg - target); a_new = a_base + delta (punish dominance);
		# b_new = b_base - delta (compensate); both clipped to [0.5, 1.2].
		if float(cfg.adaptive_payoff_strength) > 0.0:
			p_interval = int(cfg.payoff_update_interval)
			if (t + 1) % p_interval == 0:
				p_agg = float(dist.get("aggressive", 1.0 / 3.0))
				a_base = float(cfg.a) if float(cfg.a) != 0.0 else 0.8
				b_base = float(cfg.b) if float(cfg.b) != 0.0 else 0.9
				target = float(cfg.adaptive_payoff_target)
				delta = float(cfg.adaptive_payoff_strength) * (p_agg - target)
				dungeon.a = max(0.5, min(1.2, a_base + delta))
				dungeon.b = max(0.5, min(1.2, b_base - delta))

	return strategy_space, rows


def simulate_series_window(
	cfg: SimConfig,
	*,
	series: Literal["w", "p"],
	burn_in: int = 0,
	tail: int | None = None,
	strategy_space: list[str] | None = None,
) -> dict[str, list[float]]:
	"""Run a simulation and return only the requested time-series window.

	This is a memory-efficient alternative to `simulate()` for large `n_rounds`.
	It collects only the values needed by downstream analysis (cycle metrics),
	without creating per-round row dictionaries.

	Window semantics
	- Matches `analysis.cycle_metrics._slice_series()` behavior.
	- The collected index range is [begin, end) where:
	  - end = n_rounds
	  - begin = max(burn_in, end - tail) if tail is not None else burn_in
	
	Returned mapping keys are strategy names (e.g. "aggressive").
	"""
	if burn_in < 0:
		raise ValueError("burn_in must be >= 0")
	if tail is not None and tail < 0:
		raise ValueError("tail must be >= 0")
	if series not in ("w", "p"):
		raise ValueError("series must be 'w' or 'p'")

	if strategy_space is None:
		strategy_space = ["aggressive", "defensive", "balanced"]

	# Determine the window [begin, end) within 0..n_rounds.
	n = int(cfg.n_rounds)
	start = min(max(int(burn_in), 0), n)
	if tail is None:
		begin = start
	else:
		begin = max(start, n - int(tail))
	begin = min(max(begin, 0), n)

	if str(cfg.evolution_mode) == "mean_field":
		if str(cfg.payoff_mode) != "matrix_ab":
			raise ValueError("evolution_mode='mean_field' currently supports payoff_mode='matrix_ab' only")
		k = float(cfg.selection_strength)
		w_cur = _initial_weights(strategy_space=strategy_space, init_bias=float(cfg.init_bias))
		m0 = (sum(w_cur.values()) / float(len(w_cur))) if w_cur else 1.0
		w_cur = {s: (float(v) / float(m0)) for s, v in w_cur.items()}
		x_cur = _normalize_simplex(w_cur)
		x_prev = dict(x_cur)

		out: dict[str, list[float]] = {s: [] for s in strategy_space}
		for t in range(n):
			x_pay = x_prev if int(cfg.payoff_lag) == 1 else x_cur
			u = _matrix_ab_payoff_vec(strategy_space=strategy_space, a=float(cfg.a), b=float(cfg.b), x=x_pay)
			u_bar = sum(float(x_cur[s]) * float(u[s]) for s in strategy_space)
			w_next_raw = {s: (float(w_cur[s]) * exp(k * (float(u[s]) - float(u_bar)))) for s in strategy_space}
			m = sum(w_next_raw.values()) / float(len(w_next_raw))
			w_cur = {s: (float(v) / float(m)) for s, v in w_next_raw.items()} if m > 0 else {s: 1.0 for s in strategy_space}
			x_next = _normalize_simplex(w_cur)

			if t >= begin:
				if series == "p":
					for s in strategy_space:
						out[s].append(float(x_next[s]))
				else:
					for s in strategy_space:
						out[s].append(float(w_cur[s]))

			x_prev = x_cur
			x_cur = x_next

		return out

	# Deterministic sampling: give each player an independent RNG stream.
	if cfg.seed is None:
		players = [BasePlayer(strategy_space) for _ in range(cfg.n_players)]
	else:
		players = [
			BasePlayer(strategy_space, rng=random.Random(int(cfg.seed) + i))
			for i in range(cfg.n_players)
		]

	init_w = _initial_weights(strategy_space=strategy_space, init_bias=float(cfg.init_bias))
	if init_w:
		for pl in players:
			pl.update_weights(init_w)

	dungeon = DungeonAI(
		payoff_mode=cfg.payoff_mode,
		gamma=cfg.gamma,
		epsilon=cfg.epsilon,
		a=cfg.a,
		b=cfg.b,
		strategy_cycle=strategy_space,
	)
	engine = GameEngine(players, dungeon, popularity_mode=str(cfg.popularity_mode))
	if str(cfg.popularity_mode) == "expected":
		dungeon.popularity = dict(init_w)

	out: dict[str, list[float]] = {s: [] for s in strategy_space}
	new_weights = {s: 1.0 for s in strategy_space}

	for t in range(n):
		engine.step()
		dist = strategy_distribution(players, strategy_space)

		# Evolution update (weights used for the *next* round, same as CSV rows).
		new_weights = replicator_step(
			players,
			strategy_space,
			selection_strength=cfg.selection_strength,
		)
		for pl in players:
			pl.update_weights(new_weights)

		if t < begin:
			continue
		if series == "p":
			for s in strategy_space:
				out[s].append(float(dist[s]))
		else:  # series == "w"
			for s in strategy_space:
				out[s].append(float(new_weights[s]))

	return out


def main() -> None:
	cfg = _parse_args()
	strategy_space, rows = simulate(cfg)
	# For CLI summary only.
	new_weights = {f.split("w_", 1)[1]: float(v) for f, v in rows[-1].items() if f.startswith("w_")} if rows else {s: 1.0 for s in strategy_space}

	_write_timeseries_csv(cfg.out_csv, strategy_space=strategy_space, rows=rows)

	# 簡短摘要（stdout）
	print(f"Wrote CSV: {cfg.out_csv}")
	print(
		f"players={cfg.n_players} rounds={cfg.n_rounds} "
		f"seed={cfg.seed} payoff_mode={cfg.payoff_mode} gamma={cfg.gamma} epsilon={cfg.epsilon} a={cfg.a} b={cfg.b} init_bias={cfg.init_bias} evolution_mode={cfg.evolution_mode} payoff_lag={cfg.payoff_lag} "
		f"selection_strength={cfg.selection_strength} enable_events={cfg.enable_events} events_json={cfg.events_json if cfg.events_json is not None else DEFAULT_EVENTS_JSON if cfg.enable_events else None}"
	)
	print("Final weights:", new_weights)


if __name__ == "__main__":
	main()

