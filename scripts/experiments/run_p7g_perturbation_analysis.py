"""P7-G 吸引子穩定性與微擾分析（Perturbation Analysis）

Protocol lock:
  - α = 0.2 (S2 穩定域中心)
  - n_players = 4, n_rounds = 200, burn_in = 50
  - personality_mode = "static", personality_update_enabled = True
  - perturbation_axes = 8 (垂直于 P7-F 主軸)
  - perturbation_scales = 3 (ε ∈ {0.005, 0.010, 0.020})
  - perturbation_directions = 2 (±ε)
  - base_seeds = {42, 43, 44}
  - total: 3 base + 144 perturbed = 147 runs (full) or 3 + 48 = 51 runs (medium)

Gate checks:
  G7G-01: Execution completeness (Phase 1+2 at minimum)
  G7G-02: Base run consistency with P7-F attractor at α=0.2
  G7G-03: Perturbation decay (recovery time < 150 rounds)
  G7G-04: Lyapunov stability (λ_max < 0)
  G7G-05: Axis consistency (cross-seed recovery time similarity)

Usage:
    ./venv/bin/python scripts/experiments/run_p7g_perturbation_analysis.py \\
        --alpha 0.2 \\
        --seeds 42 43 44 \\
        --perturbation-scales 0.005 0.010 0.020 \\
        --phase 2 \\
        --out reports/experiments/p7g_perturbation_analysis
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np  # type: ignore
from scipy import linalg, stats  # type: ignore

from api.personality_sbert_inference import infer_personality_vector_sbert
from simulation.rl_session_engine import RLSessionConfig, RLSessionEngine

# Protocol constants
N_PLAYERS = 4
N_ROUNDS = 200
BURN_IN = 50
TAIL_START = BURN_IN
ALPHA_P7G = 0.2

LAMBDA_ALPHA = 0.15
LAMBDA_BETA = 0.10
LAMBDA_R = 0.20
LAMBDA_RISK = 0.20
LAMBDA_BETA_COMP = 0.0

PERSONALITY_TRAITS = [
    "impulsiveness", "assertiveness", "optimism",
    "risk_aversion", "suspicion", "endurance",
    "randomness", "stability_seeking", "curiosity",
]


@dataclass
class PerturbationMetrics:
    """Perturbation analysis metrics for one run."""
    recovery_time: int              # rounds until returning to attractor
    recovery_success: bool
    lyapunov_exponent: float        # λ_max
    decay_rate: float               # exponential decay rate
    half_life: int                  # rounds for amplitude to halve
    final_perturbation_magnitude: float  # tail mean ||Δx||
    final_distance_to_baseline: float    # ||x_final - x_baseline||


@dataclass
class PerturbationRunResult:
    seed: int
    alpha: float                    # = 0.2
    group: str
    is_perturbed: bool
    perturbation_axis: int | None   # 0-7 or None
    epsilon: float                  # 0.0 if base
    perturbation_direction: str     # "positive", "negative", or "none"
    
    initial_personality: dict[str, float]
    baseline_attractor: dict[str, float]
    
    metrics: PerturbationMetrics
    step_log: list[dict[str, Any]]


def infer_group_personality(text: str) -> dict[str, float]:
    """Infer personality from text."""
    vector, _meta = infer_personality_vector_sbert(text)
    return {k: float(v) for k, v in vector.items()}


def load_p7f_attractor(alpha: float) -> dict[str, list[dict[str, float]]]:
    """
    Load P7-F attractor coordinates for given alpha.
    Returns: {seed: attractor_dict for each seed}
    """
    p7f_coords_file = ROOT / "reports/experiments/p7f_attractor_mapping/p7f_attractor_coordinates.json"
    
    if not p7f_coords_file.exists():
        print(f"[P7-G] Warning: P7-F coordinates not found at {p7f_coords_file}")
        return {}
    
    with p7f_coords_file.open() as f:
        all_coords = json.load(f)
    
    alpha_str = str(round(alpha, 2))
    if alpha_str not in all_coords:
        print(f"[P7-G] Warning: α={alpha} not in P7-F results")
        return {}
    
    return {seed_idx: att for seed_idx, att in enumerate(all_coords[alpha_str])}


def compute_perturbation_axes(
    all_attractors: list[dict[str, float]],
    n_axes: int = 8
) -> np.ndarray:
    """
    Compute orthonormal perturbation axes using Gram-Schmidt.
    
    P7-F: SVD main direction (σ₁) captures 99.93% variance
    P7-G: Generate n_axes orthonormal vectors perpendicular to σ₁
    
    Returns: (n_axes, 9) matrix of orthonormal axes
    """
    # SVD on attractors
    attractors_matrix = np.array([
        [att.get(t, 0.0) for t in PERSONALITY_TRAITS]
        for att in all_attractors
    ])
    
    U, S, Vt = np.linalg.svd(attractors_matrix, full_matrices=False)
    
    # Main axis (first principal component)
    v1 = Vt[0, :]  # shape (9,)
    
    # Generate orthonormal basis perpendicular to v1
    # Use Gram-Schmidt
    basis = []
    standard_basis = np.eye(9)
    
    for e_i in standard_basis:
        # Project out v1 component
        v = e_i - np.dot(e_i, v1) * v1
        v = v / (np.linalg.norm(v) + 1e-10)
        
        # Gram-Schmidt against existing basis vectors
        for b in basis:
            v = v - np.dot(v, b) * b
        
        v = v / (np.linalg.norm(v) + 1e-10)
        
        if np.linalg.norm(v) > 1e-6:
            basis.append(v)
            if len(basis) == n_axes:
                break
    
    # Ensure we have n_axes
    if len(basis) < n_axes:
        print(f"[P7-G] Warning: Only {len(basis)} orthonormal axes generated (requested {n_axes})")
    
    return np.array(basis[:n_axes])


def compute_recovery_time(
    step_log: list[dict[str, Any]],
    baseline_attractor: dict[str, float],
    threshold: float = 0.01
) -> tuple[int, bool]:
    """
    Measure rounds until trajectory returns to within threshold of baseline attractor.
    """
    for i, step in enumerate(step_log):
        p_vec = step.get("personality_vector", {})
        distance = np.linalg.norm(np.array([
            p_vec.get(t, 0.0) - baseline_attractor.get(t, 0.0)
            for t in PERSONALITY_TRAITS
        ]))
        
        if distance < threshold:
            return i, True
    
    # Never returned
    return len(step_log), False


def compute_lyapunov_exponent(
    step_log_perturbed: list[dict[str, Any]],
    step_log_base: list[dict[str, Any]],
    start_idx: int = 0
) -> float:
    """
    Compute local maximum Lyapunov exponent.
    
    λ_max ≈ (1/T) Σ_t log(||Δx(t)|| / ||Δx(0)||)
    
    where Δx(t) = x_perturbed(t) - x_base(t)
    """
    if len(step_log_perturbed) < start_idx + 10:
        return 0.0
    
    deltas = []
    for i in range(start_idx, len(step_log_perturbed)):
        p_pert = step_log_perturbed[i].get("personality_vector", {})
        p_base = step_log_base[i].get("personality_vector", {}) if i < len(step_log_base) else {}
        
        delta = np.linalg.norm(np.array([
            p_pert.get(t, 0.0) - p_base.get(t, 0.0)
            for t in PERSONALITY_TRAITS
        ]))
        deltas.append(delta)
    
    deltas = np.array(deltas)
    
    # Compute log ratios
    delta_0 = deltas[0] + 1e-10
    log_ratios = np.log((deltas + 1e-10) / delta_0)
    
    # Lyapunov exponent per round
    lyapunov = np.mean(log_ratios) if len(log_ratios) > 0 else 0.0
    
    return float(lyapunov)


def compute_decay_metrics(
    perturbation_magnitudes: list[float],
    burn_in: int = BURN_IN
) -> tuple[float, int]:
    """
    Compute exponential decay rate and half-life.
    
    perturbation_magnitudes: list of ||Δx(t)|| over time
    Returns: (decay_rate, half_life)
    """
    if len(perturbation_magnitudes) < burn_in + 10:
        return 0.0, 0
    
    tail = perturbation_magnitudes[burn_in:]
    
    # Fit exponential: ||Δx(t)|| = A exp(-λ t)
    times = np.arange(len(tail))
    
    # Log-linear fit
    tail_arr = np.array(tail) + 1e-10
    log_tail = np.log(tail_arr)
    
    # Linear regression: log(tail) = log(A) - λ*t
    coeffs = np.polyfit(times, log_tail, 1)
    decay_rate = -coeffs[0]  # λ (positive value)
    
    # Half-life: log(0.5) = -λ * t_half → t_half = log(2) / λ
    if decay_rate > 1e-6:
        half_life = int(np.log(2.0) / decay_rate)
    else:
        half_life = len(tail)
    
    return float(decay_rate), int(half_life)


def run_base(seed: int, personality: dict[str, float]) -> PerturbationRunResult:
    """Run base case (no perturbation)."""
    config = RLSessionConfig(
        n_players=N_PLAYERS,
        n_rounds=N_ROUNDS,
        burn_in=BURN_IN,
        tail=N_ROUNDS - BURN_IN,
        seed=seed,
        personality_mode="static",
        fixed_personality_vector=personality,
        lambda_alpha=LAMBDA_ALPHA,
        lambda_beta=LAMBDA_BETA,
        lambda_r=LAMBDA_R,
        lambda_risk=LAMBDA_RISK,
        lambda_beta_comp=LAMBDA_BETA_COMP,
        personality_update_enabled=True,
        personality_feedback_strength=ALPHA_P7G,
        personality_learning_rate=0.05,
    )
    engine = RLSessionEngine(config, session_id=f"p7g_base_{seed}")
    engine.reset()

    initial_personality = dict(engine.players[0].personality if engine.players else {})
    step_log: list[dict[str, Any]] = []

    for _r in range(N_ROUNDS):
        snap = engine.step()
        p0 = engine.players[0].personality if engine.players else {}
        
        record = {
            "round": snap.round,
            "p_aggressive": snap.p_aggressive,
            "p_defensive": snap.p_defensive,
            "p_balanced": snap.p_balanced,
            "avg_reward": snap.avg_reward,
            "personality_vector": dict(p0),
            "phase": snap.phase,
        }
        step_log.append(record)

    # Compute attractor (tail mean)
    attractor = {}
    tail_trajectories = [step_log[i]["personality_vector"] for i in range(TAIL_START, N_ROUNDS)]
    for trait in PERSONALITY_TRAITS:
        attractor[trait] = float(np.mean([t.get(trait, 0.0) for t in tail_trajectories]))

    # Dummy metrics (no perturbation)
    metrics = PerturbationMetrics(
        recovery_time=0,
        recovery_success=True,
        lyapunov_exponent=0.0,
        decay_rate=0.0,
        half_life=0,
        final_perturbation_magnitude=0.0,
        final_distance_to_baseline=0.0,
    )

    return PerturbationRunResult(
        seed=seed,
        alpha=ALPHA_P7G,
        group="G-AGG",
        is_perturbed=False,
        perturbation_axis=None,
        epsilon=0.0,
        perturbation_direction="none",
        initial_personality=initial_personality,
        baseline_attractor=attractor,
        metrics=metrics,
        step_log=step_log,
    )


def run_perturbed(
    seed: int,
    personality: dict[str, float],
    axis_idx: int,
    perturbation_axis_vector: np.ndarray,
    epsilon: float,
    direction: str,  # "positive" or "negative"
    base_step_log: list[dict[str, Any]]
) -> PerturbationRunResult:
    """Run with perturbation."""
    # Apply perturbation to initial personality
    perturbed_personality = dict(personality)
    perturbation_vec = epsilon * perturbation_axis_vector if direction == "positive" else -epsilon * perturbation_axis_vector
    
    for i, trait in enumerate(PERSONALITY_TRAITS):
        perturbed_personality[trait] = personality[trait] + perturbation_vec[i]

    config = RLSessionConfig(
        n_players=N_PLAYERS,
        n_rounds=N_ROUNDS,
        burn_in=BURN_IN,
        tail=N_ROUNDS - BURN_IN,
        seed=seed,
        personality_mode="static",
        fixed_personality_vector=perturbed_personality,
        lambda_alpha=LAMBDA_ALPHA,
        lambda_beta=LAMBDA_BETA,
        lambda_r=LAMBDA_R,
        lambda_risk=LAMBDA_RISK,
        lambda_beta_comp=LAMBDA_BETA_COMP,
        personality_update_enabled=True,
        personality_feedback_strength=ALPHA_P7G,
        personality_learning_rate=0.05,
    )
    engine = RLSessionEngine(config, session_id=f"p7g_a{ALPHA_P7G}_s{seed}_ax{axis_idx}_eps{epsilon}_{direction}")
    engine.reset()

    step_log: list[dict[str, Any]] = []
    perturbation_magnitudes: list[float] = []

    for _r in range(N_ROUNDS):
        snap = engine.step()
        p0 = engine.players[0].personality if engine.players else {}
        
        # Compute perturbation magnitude vs base
        p_base = base_step_log[_r].get("personality_vector", {}) if _r < len(base_step_log) else {}
        delta_magnitude = np.linalg.norm(np.array([
            p0.get(t, 0.0) - p_base.get(t, 0.0)
            for t in PERSONALITY_TRAITS
        ]))
        
        record = {
            "round": snap.round,
            "personality_vector": dict(p0),
            "perturbation_magnitude": delta_magnitude,
            "phase": snap.phase,
        }
        step_log.append(record)
        perturbation_magnitudes.append(delta_magnitude)

    # Baseline attractor (from base run)
    baseline_attractor = {}
    tail_trajectories_base = [base_step_log[i]["personality_vector"] for i in range(TAIL_START, min(N_ROUNDS, len(base_step_log)))]
    for trait in PERSONALITY_TRAITS:
        baseline_attractor[trait] = float(np.mean([t.get(trait, 0.0) for t in tail_trajectories_base]))

    # Compute metrics
    recovery_time, recovery_success = compute_recovery_time(step_log, baseline_attractor, threshold=0.01)
    lyapunov_exp = compute_lyapunov_exponent(step_log, base_step_log, start_idx=5)
    decay_rate, half_life = compute_decay_metrics(perturbation_magnitudes)
    
    final_perturbation_mag = float(np.mean(perturbation_magnitudes[TAIL_START:]))
    
    final_p_vec = step_log[-1].get("personality_vector", {})
    final_distance = np.linalg.norm(np.array([
        final_p_vec.get(t, 0.0) - baseline_attractor.get(t, 0.0)
        for t in PERSONALITY_TRAITS
    ]))

    metrics = PerturbationMetrics(
        recovery_time=recovery_time,
        recovery_success=recovery_success,
        lyapunov_exponent=lyapunov_exp,
        decay_rate=decay_rate,
        half_life=half_life,
        final_perturbation_magnitude=final_perturbation_mag,
        final_distance_to_baseline=final_distance,
    )

    return PerturbationRunResult(
        seed=seed,
        alpha=ALPHA_P7G,
        group="G-AGG",
        is_perturbed=True,
        perturbation_axis=axis_idx,
        epsilon=epsilon,
        perturbation_direction=direction,
        initial_personality=dict(personality),
        baseline_attractor=baseline_attractor,
        metrics=metrics,
        step_log=step_log,
    )


def check_gates(results: list[PerturbationRunResult]) -> dict[str, bool | str]:
    """Evaluate G7G-01 ~ G7G-05."""
    gates: dict[str, bool | str] = {}

    # G7G-01: Execution completeness
    base_runs = [r for r in results if not r.is_perturbed]
    pert_runs = [r for r in results if r.is_perturbed]
    
    gates["G7G-01"] = len(base_runs) >= 3 and len(pert_runs) >= 24  # Phase 1+2 minimum

    # G7G-02: Base run consistency (load P7-F α=0.2 attractor)
    p7f_attractors = load_p7f_attractor(0.2)
    if p7f_attractors:
        cosine_sims = []
        for r in base_runs:
            for p7f_att in p7f_attractors.values():
                att_vec = np.array([r.baseline_attractor.get(t, 0.0) for t in PERSONALITY_TRAITS])
                p7f_vec = np.array([p7f_att.get(t, 0.0) for t in PERSONALITY_TRAITS])
                sim = np.dot(att_vec, p7f_vec) / (np.linalg.norm(att_vec) * np.linalg.norm(p7f_vec) + 1e-10)
                cosine_sims.append(sim)
        gates["G7G-02"] = bool(np.mean(cosine_sims) > 0.99) if cosine_sims else False
        gates["G7G-02_detail"] = f"mean_cosine_sim={np.mean(cosine_sims):.4f}"
    else:
        gates["G7G-02"] = False
        gates["G7G-02_detail"] = "P7-F data not available"

    # G7G-03: Perturbation decay
    recovery_times = [r.metrics.recovery_time for r in pert_runs]
    if recovery_times:
        mean_recovery = np.mean(recovery_times)
        all_recovered = all(r.metrics.recovery_success for r in pert_runs)
        gates["G7G-03"] = bool(mean_recovery < 150 and all_recovered)
        gates["G7G-03_detail"] = f"mean_recovery={mean_recovery:.1f}, success_rate={sum(1 for r in pert_runs if r.metrics.recovery_success)}/{len(pert_runs)}"
    else:
        gates["G7G-03"] = False
        gates["G7G-03_detail"] = "No perturbation runs"

    # G7G-04: Lyapunov stability
    lyap_exps = [r.metrics.lyapunov_exponent for r in pert_runs]
    if lyap_exps:
        mean_lyap = np.mean(lyap_exps)
        quantile_75 = np.quantile(lyap_exps, 0.75)
        gates["G7G-04"] = bool(mean_lyap < 0 and quantile_75 < 0)
        gates["G7G-04_detail"] = f"mean_λ={mean_lyap:.6f}, q75_λ={quantile_75:.6f}"
    else:
        gates["G7G-04"] = False
        gates["G7G-04_detail"] = "No Lyapunov data"

    # G7G-05: Axis consistency
    axis_consistency = True
    for axis_idx in range(8):
        for epsilon in [0.005, 0.010, 0.020]:
            times_for_axis_eps = [
                r.metrics.recovery_time for r in pert_runs
                if r.perturbation_axis == axis_idx and r.epsilon == epsilon
            ]
            if len(times_for_axis_eps) >= 2:
                cv = np.std(times_for_axis_eps) / (np.mean(times_for_axis_eps) + 1e-10)
                if cv > 0.5:
                    axis_consistency = False
                    break
    
    gates["G7G-05"] = bool(axis_consistency)
    gates["G7G-05_detail"] = "Cross-seed consistency check"

    return gates


def main() -> None:
    parser = argparse.ArgumentParser(description="P7-G Perturbation Analysis")
    parser.add_argument("--alpha", type=float, default=0.2)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--perturbation-scales", type=float, nargs="+", default=[0.005, 0.010, 0.020])
    parser.add_argument("--phase", type=int, default=2, choices=[1, 2, 3])
    parser.add_argument("--out", type=str, default="reports/experiments/p7g_perturbation_analysis")
    args = parser.parse_args()

    out_dir = Path(ROOT / args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    alpha = args.alpha
    seeds = args.seeds
    perturbation_scales = args.perturbation_scales
    phase = args.phase

    print(f"[P7-G] Alpha={alpha}, Seeds={seeds}, Phase={phase}, Out={out_dir}")

    # Infer personality
    text = "我喜歡冒險挑戰"
    personality = infer_group_personality(text)
    print(f"[P7-G] Inferred G-AGG ({text!r})")

    # Phase 1: Base runs
    print(f"\n[P7-G] Phase 1: Running {len(seeds)} base runs (no perturbation) ...")
    base_results = []
    base_step_logs = {}
    
    for seed in seeds:
        print(f"  Base seed={seed} ...", end="", flush=True)
        try:
            result = run_base(seed, personality)
            base_results.append(result)
            base_step_logs[seed] = result.step_log
            print(f" done")
        except Exception as e:
            print(f" FAILED: {e}")

    # Phase 2: Perturbation runs (subset: 2 axes × 3 scales × 2 directions × 3 seeds = 36 runs)
    print(f"\n[P7-G] Phase 2: Computing perturbation axes ...")
    
    # Compute axes
    p7f_attractors_list = []
    for seed_idx in range(len(base_results)):
        p7f_attractors_list.append(base_results[seed_idx].baseline_attractor)
    
    perturbation_axes = compute_perturbation_axes(p7f_attractors_list, n_axes=8)
    print(f"[P7-G] Generated {perturbation_axes.shape[0]} orthonormal perturbation axes (9D)")

    print(f"\n[P7-G] Phase {phase}: Running perturbation runs ...")
    pert_results = []
    
    if phase >= 2:
        # Run subset (2 main axes)
        axes_to_run = [0, 1]  # 2 main axes
        n_total = len(seeds) * len(axes_to_run) * len(perturbation_scales) * 2
        run_idx = 0
        
        for seed_idx, seed in enumerate(seeds):
            base_log = base_step_logs.get(seed, [])
            
            for axis_idx in axes_to_run:
                axis_vec = perturbation_axes[axis_idx]
                
                for epsilon in perturbation_scales:
                    for direction in ["positive", "negative"]:
                        run_idx += 1
                        print(f"  [{run_idx}/{n_total}] seed={seed} axis={axis_idx} ε={epsilon} {direction} ...", end="", flush=True)
                        
                        try:
                            result = run_perturbed(
                                seed=seed,
                                personality=personality,
                                axis_idx=axis_idx,
                                perturbation_axis_vector=axis_vec,
                                epsilon=epsilon,
                                direction=direction,
                                base_step_log=base_log,
                            )
                            pert_results.append(result)
                            print(f" done | recovery_t={result.metrics.recovery_time}, λ={result.metrics.lyapunov_exponent:.6f}")
                        except Exception as e:
                            print(f" FAILED: {e}")

    if phase >= 3:
        # Run full set (8 axes)
        print(f"\n[P7-G] Phase 3: Running full perturbation matrix (8 axes) ...")
        axes_to_run = list(range(8))
        n_total = len(seeds) * len(axes_to_run) * len(perturbation_scales) * 2
        run_idx = 0
        
        for seed_idx, seed in enumerate(seeds):
            base_log = base_step_logs.get(seed, [])
            
            for axis_idx in axes_to_run:
                if axis_idx < 2:
                    continue  # Already done in Phase 2
                
                axis_vec = perturbation_axes[axis_idx]
                
                for epsilon in perturbation_scales:
                    for direction in ["positive", "negative"]:
                        run_idx += 1
                        
                        try:
                            result = run_perturbed(
                                seed=seed,
                                personality=personality,
                                axis_idx=axis_idx,
                                perturbation_axis_vector=axis_vec,
                                epsilon=epsilon,
                                direction=direction,
                                base_step_log=base_log,
                            )
                            pert_results.append(result)
                        except Exception as e:
                            pass

    # Summary CSV
    all_results = base_results + pert_results
    summary_path = out_dir / "p7g_perturbation_summary.csv"
    
    fieldnames = [
        "seed", "is_perturbed", "perturbation_axis", "epsilon", "direction",
        "recovery_time", "recovery_success", "lyapunov_exponent", "decay_rate",
        "half_life", "final_perturbation_magnitude", "final_distance_to_baseline"
    ]
    
    with summary_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in all_results:
            row = {
                "seed": r.seed,
                "is_perturbed": r.is_perturbed,
                "perturbation_axis": r.perturbation_axis if r.is_perturbed else "N/A",
                "epsilon": round(r.epsilon, 5),
                "direction": r.perturbation_direction,
                "recovery_time": r.metrics.recovery_time,
                "recovery_success": r.metrics.recovery_success,
                "lyapunov_exponent": round(r.metrics.lyapunov_exponent, 9),
                "decay_rate": round(r.metrics.decay_rate, 6),
                "half_life": r.metrics.half_life,
                "final_perturbation_magnitude": round(r.metrics.final_perturbation_magnitude, 9),
                "final_distance_to_baseline": round(r.metrics.final_distance_to_baseline, 6),
            }
            writer.writerow(row)

    print(f"\n[P7-G] Summary CSV → {summary_path}")

    # Perturbation axes JSON
    axes_path = out_dir / "p7g_perturbation_axes.json"
    axes_data = {
        f"axis_{i}": [float(v) for v in perturbation_axes[i]]
        for i in range(perturbation_axes.shape[0])
    }
    axes_path.write_text(json.dumps(axes_data, indent=2, ensure_ascii=False))

    # Lyapunov analysis
    lyapunov_by_axis = {}
    for axis_idx in range(perturbation_axes.shape[0]):
        axis_results = [r for r in pert_results if r.perturbation_axis == axis_idx]
        lyaps = [r.metrics.lyapunov_exponent for r in axis_results]
        if lyaps:
            lyapunov_by_axis[axis_idx] = {
                "mean": float(np.mean(lyaps)),
                "std": float(np.std(lyaps)),
                "min": float(np.min(lyaps)),
                "max": float(np.max(lyaps)),
                "values": [round(v, 9) for v in lyaps],
            }

    lyapunov_path = out_dir / "p7g_lyapunov_analysis.json"
    lyapunov_path.write_text(json.dumps(lyapunov_by_axis, indent=2, ensure_ascii=False))

    # Gate checks
    gates = check_gates(all_results)
    print("\n" + "=" * 60)
    print("Gate Check Results")
    print("=" * 60)
    for gate_id in ["G7G-01", "G7G-02", "G7G-03", "G7G-04", "G7G-05"]:
        passed = gates[gate_id]
        symbol = "✓ PASS" if passed else "✗ FAIL"
        detail = gates.get(f"{gate_id}_detail", "")
        print(f"  {gate_id}: {symbol}  {detail}")

    gates_path = out_dir / "p7g_gates.json"
    gates_serializable = {k: bool(v) if isinstance(v, bool) else str(v) for k, v in gates.items()}
    gates_path.write_text(json.dumps(gates_serializable, indent=2, ensure_ascii=False))

    all_passed = all(bool(gates[g]) for g in ["G7G-01", "G7G-02", "G7G-03", "G7G-04", "G7G-05"])
    print("=" * 60)
    if all_passed:
        print("[P7-G] All gates PASSED → Attractor stability confirmed.")
        print(f"       Mean Lyapunov exponent: {np.mean([r.metrics.lyapunov_exponent for r in pert_results]):.6f}")
        print(f"       Mean recovery time: {np.mean([r.metrics.recovery_time for r in pert_results]):.1f} rounds")
    else:
        print("[P7-G] Some gates FAILED → Review results.")
    print(f"       Artifacts → {out_dir}/")

    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
