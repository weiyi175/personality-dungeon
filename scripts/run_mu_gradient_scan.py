#!/usr/bin/env python3
"""H7 μ-Gradient Fine Scan

Scans mu_base ∈ {0.20, 0.25, 0.30, 0.35, 0.40} to identify the "golden point":
a mu_base value that achieves ≥50% L3 success rate while avoiding stagnation.

────────────────────────────────────────────────────────────────────────────────
STRICT ACCEPTANCE CONDITIONS (嚴格驗收條件)
────────────────────────────────────────────────────────────────────────────────
  Gate 1 (PRIMARY)    L3_control ≥ 0.50
                        At least 6/12 seeds reach L3 under control_none.
  Gate 2 (WALL)       L3_control ≥ L3_random + 0.10
                        Random wall must be present: control outperforms random
                        by ≥10 pp. If random ≥ control, personality is still
                        "acting as noise" rather than "regulating inertia".
  Gate 3 (VELOCITY)   mean_phase_velocity_control ∈ [0.001, 0.030]
                        Lower bound: system is not too slow (approaching L0 stasis).
                        Upper bound: system is not chaotic (over-shooting).
  Gate 4 (ANTI-STAG)  mean_max_dominance_control ≤ 0.75
                        mean_max_dominance = average over tail window of the
                        proportion held by the dominant strategy each round.
                        Healthy cycling: ≈0.45–0.65 (each strategy takes turns).
                        Stagnant (high-μ lock-in): ≈0.80–1.0 (one strategy wins).
────────────────────────────────────────────────────────────────────────────────
EXPERIMENT BOUNDARIES (固定邊界)
────────────────────────────────────────────────────────────────────────────────
  Payoff     : matrix_ab, a=1.0, b=0.9, cross_coupling=0.20
  Selection  : SS=0.15, k_clamp=[0.05, 0.25], lambda_k=0.20
  Personality: lambda_mu=0.05, mu_clamp=[0.0, 0.60]
  Simulation : players=300, seeds={12 standard}, burn_in=2000, tail=2000
  Variable   : mu_base ∈ {0.20, 0.25, 0.30, 0.35, 0.40}
────────────────────────────────────────────────────────────────────────────────
PARADIGM NOTE
────────────────────────────────────────────────────────────────────────────────
  Old view: personality as "noise" → perturbation of growth vector
  New view: personality as "inertia regulator" → modulation of velocity memory
  μ_base is the fundamental phase-accumulation capacity;
  personality ± adjusts individual players around this baseline.
────────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis.cycle_metrics import classify_cycle_level, phase_rotation_r2
from players.base_player import DEFAULT_PERSONALITY_KEYS
from simulation.run_simulation import SimConfig, _write_timeseries_csv, simulate

# ──────────────────────────────────────────────────────────────────────────────
# Fixed experiment boundaries
# ──────────────────────────────────────────────────────────────────────────────
FULL_MU_BASES: list[float] = [0.20, 0.25, 0.30, 0.35, 0.40]
QUICK_MU_BASES: list[float] = [0.20, 0.30, 0.40]

FULL_SEEDS = [45, 47, 49, 51, 53, 55, 91, 93, 95, 97, 99, 123]
QUICK_SEEDS = [45, 47, 49]

CONDITIONS = ["control_none", "random_9persona"]
SELECTION_STRENGTH = 0.15
K_LOWER = 0.05
K_UPPER = 0.25
LAMBDA_MU = 0.05
LAMBDA_K = 0.20
FULL_ROUNDS = 6000
QUICK_ROUNDS = 3000
BURN_IN = 2000
TAIL = 2000

# ──────────────────────────────────────────────────────────────────────────────
# Strict acceptance gate thresholds (嚴格驗收條件)
# ──────────────────────────────────────────────────────────────────────────────
GATE_L3_RATE_MIN: float = 0.50          # Gate 1 PRIMARY  — ≥50% seeds at L3
GATE_WALL_MARGIN_MIN: float = 0.10      # Gate 2 WALL     — ctrl ≥ random + 10pp
GATE_VELOCITY_MIN: float = 0.001        # Gate 3 VELOCITY — not too slow
GATE_VELOCITY_MAX: float = 0.030        # Gate 3 VELOCITY — not too fast / chaotic
GATE_MAX_DOMINANCE: float = 0.75        # Gate 4 ANTI-STAG — no single-strategy lock-in


# ──────────────────────────────────────────────────────────────────────────────
# Data types
# ──────────────────────────────────────────────────────────────────────────────
@dataclass
class RunMetric:
	mu_base: float
	condition: str
	seed: int
	csv_path: str
	cycle_level: int
	phase_velocity: float
	mean_max_dominance: float  # anti-stagnation metric (higher → more stagnant)
	l3_pass: bool


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────
def _random_9persona_setup(seed: int):
	def cb(players: list[object], _strategy_space: list[str], _cfg: SimConfig) -> None:
		for idx, player in enumerate(players):
			rng = random.Random(int(seed) * 10000 + idx)
			for key in DEFAULT_PERSONALITY_KEYS:
				player.personality[key] = rng.uniform(-0.4, 0.4)

	return cb


def _extract_series(rows: list[dict[str, Any]]) -> dict[str, list[float]]:
	return {
		"aggressive": [float(row.get("p_aggressive", 0.0)) for row in rows],
		"defensive": [float(row.get("p_defensive", 0.0)) for row in rows],
		"balanced": [float(row.get("p_balanced", 0.0)) for row in rows],
	}


def _mean_max_dominance(rows: list[dict[str, Any]], burn_in: int, tail: int) -> float:
	"""Mean of max-strategy proportion in the tail window.

	High (> 0.75) → one strategy dominates most rounds → stagnation.
	Healthy L3 cycling → ≈ 0.45–0.65 (each strategy takes turns).
	"""
	total = len(rows)
	start = max(burn_in, total - tail)
	tail_rows = rows[start:]
	if not tail_rows:
		return 0.0
	vals = [
		max(
			float(r.get("p_aggressive", 0.333)),
			float(r.get("p_defensive", 0.333)),
			float(r.get("p_balanced", 0.333)),
		)
		for r in tail_rows
	]
	return sum(vals) / len(vals)


# ──────────────────────────────────────────────────────────────────────────────
# Core run
# ──────────────────────────────────────────────────────────────────────────────
def run_one(
	*,
	mu_base: float,
	condition: str,
	seed: int,
	out_dir: Path,
	rounds: int,
) -> RunMetric:
	tag_dir = out_dir / f"mu{mu_base:.2f}" / condition
	tag_dir.mkdir(parents=True, exist_ok=True)
	out_csv = tag_dir / f"seed{seed}.csv"

	cfg = SimConfig(
		n_players=300,
		n_rounds=int(rounds),
		seed=int(seed),
		payoff_mode="matrix_ab",
		popularity_mode="sampled",
		gamma=0.16,
		epsilon=0.0,
		a=1.0,
		b=0.9,
		matrix_cross_coupling=0.20,
		init_bias=0.5,
		evolution_mode="personality_coupled",
		payoff_lag=1,
		selection_strength=float(SELECTION_STRENGTH),
		enable_events=False,
		events_json=None,
		out_csv=out_csv,
		memory_kernel=1,
		synergy_type="nonlinear",
		synergy_gamma=0.16,
		synergy_nonlinear_type="power",
		synergy_nonlinear_power=3.2,
		personality_coupling_mu_base=float(mu_base),
		personality_coupling_lambda_mu=float(LAMBDA_MU),
		personality_coupling_lambda_k=float(LAMBDA_K),
		personality_coupling_mu_lower=0.0,
		personality_coupling_mu_upper=0.60,
		personality_coupling_k_lower=float(K_LOWER),
		personality_coupling_k_upper=float(K_UPPER),
	)

	player_setup = _random_9persona_setup(seed) if condition == "random_9persona" else None
	strategy_space, rows = simulate(cfg, player_setup_callback=player_setup)
	_write_timeseries_csv(out_csv, strategy_space=strategy_space, rows=rows)

	series = _extract_series(rows)
	cyc = classify_cycle_level(
		series, burn_in=BURN_IN, tail=TAIL, amplitude_threshold=0.02, eta=0.6, min_turn_strength=0.0
	)
	rot = phase_rotation_r2(series, burn_in=BURN_IN, tail=TAIL)
	wlen = rot.window_length if rot.window_length > 0 else 1
	phase_velocity = float(rot.cumulative_rotation) / float(wlen)
	mean_max_dom = _mean_max_dominance(rows, burn_in=BURN_IN, tail=TAIL)

	return RunMetric(
		mu_base=float(mu_base),
		condition=condition,
		seed=int(seed),
		csv_path=str(out_csv.relative_to(ROOT)),
		cycle_level=int(cyc.level),
		phase_velocity=float(phase_velocity),
		mean_max_dominance=float(mean_max_dom),
		l3_pass=int(cyc.level) >= 3,
	)


# ──────────────────────────────────────────────────────────────────────────────
# Aggregation & acceptance gate evaluation
# ──────────────────────────────────────────────────────────────────────────────
def _agg(metrics: list[RunMetric]) -> dict[str, Any]:
	n = len(metrics)
	if n == 0:
		return {"n": 0, "l3_rate": 0.0, "mean_phase_velocity": 0.0, "mean_max_dominance": 0.0}
	l3_count = sum(1 for m in metrics if m.l3_pass)
	l3_rate = l3_count / n
	mean_v = sum(m.phase_velocity for m in metrics) / n
	mean_dom = sum(m.mean_max_dominance for m in metrics) / n
	return {
		"n": n,
		"l3_pass_count": l3_count,
		"l3_rate": l3_rate,
		"mean_phase_velocity": mean_v,
		"mean_max_dominance": mean_dom,
		"gate_l3_ok": l3_rate >= GATE_L3_RATE_MIN,
		"gate_velocity_ok": GATE_VELOCITY_MIN <= mean_v <= GATE_VELOCITY_MAX,
		"gate_stagnation_ok": mean_dom <= GATE_MAX_DOMINANCE,
	}


def _evaluate_mu(
	mu_base: float,
	all_results: list[RunMetric],
) -> dict[str, Any]:
	ctrl_metrics = [r for r in all_results if r.mu_base == mu_base and r.condition == "control_none"]
	rand_metrics = [r for r in all_results if r.mu_base == mu_base and r.condition == "random_9persona"]
	ctrl = _agg(ctrl_metrics)
	rand = _agg(rand_metrics)
	wall_ok = ctrl["l3_rate"] >= rand["l3_rate"] + GATE_WALL_MARGIN_MIN
	all_gates = (
		ctrl.get("gate_l3_ok", False)
		and wall_ok
		and ctrl.get("gate_velocity_ok", False)
		and ctrl.get("gate_stagnation_ok", False)
	)
	return {
		"mu_base": mu_base,
		"control": ctrl,
		"random": rand,
		"gate_wall_ok": wall_ok,
		"all_gates_pass": all_gates,
	}


# ──────────────────────────────────────────────────────────────────────────────
# Output
# ──────────────────────────────────────────────────────────────────────────────
def _print_acceptance_table(all_results: list[RunMetric]) -> None:
	mu_values = sorted({r.mu_base for r in all_results})
	mu_evals = {mu: _evaluate_mu(mu, all_results) for mu in mu_values}

	print()
	print("=" * 105)
	print("μ-Gradient Acceptance Table (嚴格驗收)")
	print("=" * 105)
	header = (
		f"{'μ_base':>7}  {'Condition':<18}  {'L3%':>6}  {'G1':>3}  "
		f"{'G2(wall)':>8}  {'v_mean':>10}  {'G3':>3}  {'MaxDom':>8}  {'G4':>3}  {'PASS':>6}"
	)
	print(header)
	print("-" * 105)
	for mu in mu_values:
		ev = mu_evals[mu]
		for cond in CONDITIONS:
			agg_data = ev["control"] if cond == "control_none" else ev["random"]
			g1 = "✓" if agg_data.get("gate_l3_ok", False) else "✗"
			g2 = ("✓" if ev["gate_wall_ok"] else "✗") if cond == "control_none" else "  –"
			g3 = "✓" if agg_data.get("gate_velocity_ok", False) else "✗"
			g4 = "✓" if agg_data.get("gate_stagnation_ok", False) else "✗"
			if cond == "control_none":
				pass_str = "★ YES" if ev["all_gates_pass"] else "   no"
			else:
				pass_str = ""
			print(
				f"{mu:>7.2f}  {cond:<18}  {agg_data['l3_rate']:>6.1%}  {g1:>3}  "
				f"{g2:>8}  {agg_data['mean_phase_velocity']:>10.6f}  {g3:>3}  "
				f"{agg_data['mean_max_dominance']:>8.4f}  {g4:>3}  {pass_str:>6}"
			)
		print()

	print("Gates: G1=L3≥50%  G2=Ctrl−Rand≥10pp  G3=v∈[0.001,0.030]  G4=MaxDom≤0.75")
	print("★ = passes all 4 gates simultaneously → Golden Point candidate")
	print("=" * 105)


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────
def main() -> int:
	parser = argparse.ArgumentParser(description="H7 μ-gradient fine scan")
	parser.add_argument("--quick", action="store_true", help="Quick mode: 3 seeds × 3 μ values × 3000 rounds")
	parser.add_argument("--rounds", type=int, default=FULL_ROUNDS, help="Rounds per run (full mode)")
	parser.add_argument(
		"--mu-bases",
		type=str,
		default=",".join(f"{v:.2f}" for v in FULL_MU_BASES),
		help="Comma-separated mu_base values to scan",
	)
	parser.add_argument(
		"--seeds",
		type=str,
		default=",".join(str(s) for s in FULL_SEEDS),
		help="Comma-separated seed list",
	)
	args = parser.parse_args()

	if args.quick:
		mu_bases = QUICK_MU_BASES
		seeds = QUICK_SEEDS
		rounds = QUICK_ROUNDS
		out_dir = ROOT / "outputs" / "mu_gradient_quick"
	else:
		mu_bases = [float(v) for v in args.mu_bases.split(",") if v.strip()]
		seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
		rounds = args.rounds
		out_dir = ROOT / "outputs" / "mu_gradient_scan"

	out_dir.mkdir(parents=True, exist_ok=True)
	total = len(mu_bases) * len(CONDITIONS) * len(seeds)

	print("H7 μ-Gradient Fine Scan")
	print(f"  μ_base values  : {mu_bases}")
	print(f"  seeds          : {len(seeds)}")
	print(f"  conditions     : {CONDITIONS}")
	print(f"  rounds         : {rounds} (burn_in={BURN_IN}, tail={TAIL})")
	print(f"  total runs     : {total}")
	print(f"  output dir     : {out_dir}")
	print()
	print("Acceptance gates:")
	print(f"  G1 PRIMARY     L3_control ≥ {GATE_L3_RATE_MIN:.0%}")
	print(f"  G2 WALL        L3_control ≥ L3_random + {GATE_WALL_MARGIN_MIN:.0%}")
	print(f"  G3 VELOCITY    v ∈ [{GATE_VELOCITY_MIN}, {GATE_VELOCITY_MAX}]")
	print(f"  G4 ANTI-STAG   mean_max_dominance ≤ {GATE_MAX_DOMINANCE}")
	print()

	all_results: list[RunMetric] = []
	index = 0
	for mu in mu_bases:
		for cond in CONDITIONS:
			for seed in seeds:
				index += 1
				print(
					f"[{index:3d}/{total}] μ={mu:.2f} {cond:<18} seed={seed:3d}",
					end="  ",
					flush=True,
				)
				m = run_one(mu_base=mu, condition=cond, seed=seed, out_dir=out_dir, rounds=rounds)
				all_results.append(m)
				stag_flag = "!" if m.mean_max_dominance > GATE_MAX_DOMINANCE else " "
				print(
					f"L{m.cycle_level}  v={m.phase_velocity:.5f}  dom={m.mean_max_dominance:.3f}{stag_flag}  "
					f"{'L3' if m.l3_pass else '  '}"
				)

	_print_acceptance_table(all_results)

	# Identify golden points
	mu_evals = {mu: _evaluate_mu(mu, all_results) for mu in mu_bases}
	golden = sorted(mu for mu, ev in mu_evals.items() if ev["all_gates_pass"])
	print()
	if golden:
		print(f"★ Golden Point(s): μ_base ∈ {golden}")
		best = min(golden)  # prefer smallest passing value → lowest stagnation risk
		print(f"  Recommendation: μ_base = {best:.2f}  (smallest that passes all gates)")
	else:
		closest = max(mu_bases, key=lambda mu: (
			mu_evals[mu]["control"].get("l3_rate", 0.0),
			-mu_evals[mu]["control"].get("mean_max_dominance", 1.0),
		))
		print(f"✗ No μ_base passed all 4 gates.")
		print(f"  Best candidate: μ_base = {closest:.2f} (highest L3 rate with lowest stagnation)")

	# Write summary JSON
	summary = {
		"experiment": "H7 μ-Gradient Fine Scan",
		"config": {
			"mu_bases": mu_bases,
			"seeds": seeds,
			"conditions": CONDITIONS,
			"rounds": rounds,
			"burn_in": BURN_IN,
			"tail": TAIL,
			"selection_strength": SELECTION_STRENGTH,
			"k_clamp": [K_LOWER, K_UPPER],
			"lambda_mu": LAMBDA_MU,
			"lambda_k": LAMBDA_K,
		},
		"acceptance_gates": {
			"gate1_l3_rate_min": GATE_L3_RATE_MIN,
			"gate2_wall_margin_min": GATE_WALL_MARGIN_MIN,
			"gate3_velocity_range": [GATE_VELOCITY_MIN, GATE_VELOCITY_MAX],
			"gate4_max_dominance_max": GATE_MAX_DOMINANCE,
		},
		"per_mu_evaluation": {f"{mu:.2f}": mu_evals[mu] for mu in mu_bases},
		"golden_points": golden,
		"results": [asdict(r) for r in all_results],
	}
	summary_path = out_dir / "summary.json"
	summary_path.write_text(json.dumps(summary, indent=2))
	print(f"\nSummary written to: {summary_path}")
	return 0


if __name__ == "__main__":
	sys.exit(main())
