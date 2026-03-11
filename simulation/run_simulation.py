from core.game_engine import GameEngine
from dungeon.dungeon_ai import DungeonAI
from analysis.metrics import average_reward, average_utility, strategy_distribution
from evolution.replicator_dynamics import replicator_step
from players.base_player import BasePlayer

import argparse
import csv
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
	init_bias: float
	evolution_mode: str
	payoff_lag: int
	selection_strength: float
	out_csv: Path


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
		"--init-bias",
		type=float,
		default=0.0,
		help="Initial symmetry-breaking bias applied to weights: w0=(1+bias,1-bias,1) for (aggressive,defensive,balanced). Use with popularity_mode=expected to get deterministic phase rotation; requires |bias|<1.",
	)
	p.add_argument("--selection-strength", type=float, default=0.05)
	p.add_argument(
		"--out",
		type=Path,
		default=Path("outputs") / "timeseries.csv",
		help="CSV output path (will create parent dirs)",
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
		init_bias=a.init_bias,
		evolution_mode=a.evolution_mode,
		payoff_lag=a.payoff_lag,
		selection_strength=a.selection_strength,
		out_csv=a.out,
	)


def _normalize_simplex(weights: dict[str, float]) -> dict[str, float]:
	total = sum(float(v) for v in weights.values())
	if total <= 0:
		n = len(weights)
		return {k: (1.0 / float(n)) for k in weights} if n else {}
	return {k: (float(v) / float(total)) for k, v in weights.items()}


def _matrix_ab_payoff_vec(*, strategy_space: list[str], a: float, b: float, x: dict[str, float]) -> dict[str, float]:
	"""Compute u = A x for the matrix_ab cyclic payoff.

	A = [[0, a, -b],
	     [-b, 0, a],
	     [a, -b, 0]]
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
	)
	with out_csv.open("w", newline="") as f:
		writer = csv.DictWriter(f, fieldnames=fieldnames)
		writer.writeheader()
		writer.writerows(rows)


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
			u = _matrix_ab_payoff_vec(strategy_space=strategy_space, a=float(cfg.a), b=float(cfg.b), x=x_pay)
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
		strategy_cycle=strategy_space,
	)
	engine = GameEngine(players, dungeon, popularity_mode=str(cfg.popularity_mode))
	if str(cfg.popularity_mode) == "expected":
		# Without this, round-0 popularity is empty -> identical rewards -> no movement.
		dungeon.popularity = dict(init_w)

	rows: list[dict] = []
	new_weights = {s: 1.0 for s in strategy_space}

	for t in range(cfg.n_rounds):
		engine.step()

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
		rows.append(row)

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
		f"selection_strength={cfg.selection_strength}"
	)
	print("Final weights:", new_weights)


if __name__ == "__main__":
	main()

