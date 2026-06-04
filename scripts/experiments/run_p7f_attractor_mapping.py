"""P7-F 吸引子人格空間映射（Attractor Personality Space Mapping）

Protocol lock:
  - n_players=4, n_rounds=200, burn_in=50
  - personality_mode="static", personality_update_enabled=True
  - α ∈ {0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4} (7 values)
  - seeds={42, 43, 44} (3 seeds)
  - Total: 7 alpha × 3 seeds = 21 runs
  - personality_learning_rate η = 0.05 (fixed)

Gate checks:
  G7F-01: 21/21 runs complete, no exception
  G7F-02: All runs have finite attractor coordinates
  G7F-03: Same α across 3 seeds: mean pairwise distance < 0.05
  G7F-04: ||attractor|| vs α linear fit R² > 0.95
  G7F-05: Invariant subspace dimension ≤ 3

Usage:
    ./venv/bin/python scripts/experiments/run_p7f_attractor_mapping.py \\
        --alphas 0.1 0.15 0.2 0.25 0.3 0.35 0.4 \\
        --seeds 42 43 44 \\
        --group G-AGG \\
        --learning-rate 0.05 \\
        --out reports/experiments/p7f_attractor_mapping
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
from scipy import linalg  # type: ignore

from api.personality_sbert_inference import infer_personality_vector_sbert
from simulation.rl_session_engine import RLSessionConfig, RLSessionEngine

# Protocol constants
N_PLAYERS = 4
N_ROUNDS = 200
BURN_IN = 50
TAIL_START = BURN_IN

# Personality coupling lambdas (B1 proven config)
LAMBDA_ALPHA = 0.15
LAMBDA_BETA = 0.10
LAMBDA_R = 0.20
LAMBDA_RISK = 0.20
LAMBDA_BETA_COMP = 0.0

GROUP_TEXTS: dict[str, str] = {
    "G-AGG": "我喜歡冒險挑戰",
    "G-DEF": "我謹慎保守行事",
    "G-BAL": "我靈活應對局面",
}

PERSONALITY_TRAITS = [
    "impulsiveness", "assertiveness", "optimism",
    "risk_aversion", "suspicion", "endurance",
    "randomness", "stability_seeking", "curiosity",
]


@dataclass
class AttractorMetrics:
    """Attractor characteristics for one run."""
    attractor_coordinates: dict[str, float]  # 9D attractor position (tail mean)
    attractor_distance: float                 # ||attractor - initial||
    attractor_norm: float                     # ||attractor|| / sqrt(9)
    attractor_variance: float                 # variance of tail trajectory


@dataclass
class RunResult:
    seed: int
    alpha: float
    group: str
    initial_personality: dict[str, float]
    metrics: AttractorMetrics
    step_log: list[dict[str, Any]]


def infer_group_personality(text: str) -> dict[str, float]:
    """Infer 9D personality vector from text."""
    vector, _meta = infer_personality_vector_sbert(text)
    return {k: float(v) for k, v in vector.items()}


def compute_attractor_coordinates(
    step_log: list[dict[str, Any]],
    tail_start: int = TAIL_START
) -> dict[str, float]:
    """
    Compute attractor position as tail mean of personality vector.
    """
    if len(step_log) < tail_start:
        return {t: 0.0 for t in PERSONALITY_TRAITS}
    
    tail_trajectories = [step_log[i]["personality_vector"] for i in range(tail_start, len(step_log))]
    
    attractor = {}
    for trait in PERSONALITY_TRAITS:
        values = [t.get(trait, 0.0) for t in tail_trajectories]
        attractor[trait] = float(np.mean(values)) if values else 0.0
    
    return attractor


def compute_attractor_distance(
    initial_personality: dict[str, float],
    attractor: dict[str, float]
) -> tuple[float, float]:
    """
    Compute Euclidean distance from initial state to attractor.
    Returns: (distance, normalized_norm)
    """
    delta = np.array([
        attractor.get(t, 0.0) - initial_personality.get(t, 0.0)
        for t in PERSONALITY_TRAITS
    ])
    distance = float(np.linalg.norm(delta))
    norm = float(np.linalg.norm([attractor.get(t, 0.0) for t in PERSONALITY_TRAITS]) / np.sqrt(len(PERSONALITY_TRAITS)))
    
    return distance, norm


def compute_attractor_variance(
    step_log: list[dict[str, Any]],
    tail_start: int = TAIL_START
) -> float:
    """
    Compute variance of tail trajectory (tighter attractor = lower variance).
    """
    if len(step_log) < tail_start:
        return 0.0
    
    tail_norms = [step_log[i]["personality_norm"] for i in range(tail_start, len(step_log))]
    return float(np.var(tail_norms)) if tail_norms else 0.0


def compute_pairwise_distances(
    attractors: list[dict[str, float]]
) -> np.ndarray:
    """
    Compute pairwise Euclidean distances between attractors.
    Returns: n×n symmetric matrix (n = len(attractors))
    """
    n = len(attractors)
    distances = np.zeros((n, n))
    
    for i in range(n):
        for j in range(i+1, n):
            delta = np.array([
                attractors[i].get(t, 0.0) - attractors[j].get(t, 0.0)
                for t in PERSONALITY_TRAITS
            ])
            d = float(np.linalg.norm(delta))
            distances[i, j] = d
            distances[j, i] = d
    
    return distances


def estimate_invariant_subspace_dim(
    attractors_matrix: np.ndarray,
    threshold: float = 0.95
) -> tuple[int, list[float]]:
    """
    Estimate attractor subspace dimensionality via SVD.
    
    attractors_matrix: (n_runs, 9) array of attractor vectors
    threshold: cumulative variance threshold (default 0.95 → 95%)
    
    Returns: (dimension, first_5_singular_values)
    """
    if attractors_matrix.shape[0] < 2:
        return 9, []
    
    # SVD
    U, S, Vt = np.linalg.svd(attractors_matrix, full_matrices=False)
    
    # Cumulative explained variance
    cumsum = np.cumsum(S**2) / np.sum(S**2)
    
    # Find dimension
    dim = np.argmax(cumsum >= threshold) + 1
    dim = min(dim, len(S))
    
    singular_values = [float(s) for s in S[:min(5, len(S))]]
    
    return int(dim), singular_values


def detect_bifurcation(
    all_attractors_by_alpha: dict[float, list[dict[str, float]]]
) -> dict[str, Any]:
    """
    Detect bifurcation points by analyzing rate of change in attractors.
    """
    alphas = sorted(all_attractors_by_alpha.keys())
    bifurcation_results = {
        "bifurcation_points": [],
        "max_rate_of_change": 0.0,
        "smooth": True
    }
    
    if len(alphas) < 2:
        return bifurcation_results
    
    for i in range(len(alphas) - 1):
        alpha1, alpha2 = alphas[i], alphas[i+1]
        delta_alpha = alpha2 - alpha1
        
        attractors1 = all_attractors_by_alpha[alpha1]
        attractors2 = all_attractors_by_alpha[alpha2]
        
        for seed_idx in range(min(len(attractors1), len(attractors2))):
            att1 = attractors1[seed_idx]
            att2 = attractors2[seed_idx]
            
            delta_att = np.linalg.norm(np.array([
                att2.get(t, 0.0) - att1.get(t, 0.0) for t in PERSONALITY_TRAITS
            ]))
            
            rate = delta_att / delta_alpha if delta_alpha > 0 else 0.0
            bifurcation_results["max_rate_of_change"] = max(
                bifurcation_results["max_rate_of_change"],
                rate
            )
            
            # Simple bifurcation threshold: rate > 0.5
            if rate > 0.5:
                bifurcation_results["bifurcation_points"].append({
                    "alpha_range": f"[{alpha1:.2f}, {alpha2:.2f}]",
                    "seed": seed_idx,
                    "rate": round(rate, 4)
                })
                bifurcation_results["smooth"] = False
    
    return bifurcation_results


def run_single(
    seed: int,
    alpha: float,
    personality_vector: dict[str, float]
) -> RunResult:
    """Run one P7-F session."""
    config = RLSessionConfig(
        n_players=N_PLAYERS,
        n_rounds=N_ROUNDS,
        burn_in=BURN_IN,
        tail=N_ROUNDS - BURN_IN,
        check_interval=50,
        seed=seed,
        personality_mode="static",
        fixed_personality_vector=personality_vector,
        lambda_alpha=LAMBDA_ALPHA,
        lambda_beta=LAMBDA_BETA,
        lambda_r=LAMBDA_R,
        lambda_risk=LAMBDA_RISK,
        lambda_beta_comp=LAMBDA_BETA_COMP,
        personality_update_enabled=True,
        personality_feedback_strength=alpha,
        personality_learning_rate=0.05,
    )
    engine = RLSessionEngine(config, session_id=f"p7f_a{alpha:.2f}_{seed}")
    engine.reset()

    initial_personality = dict(engine.players[0].personality) if engine.players else {}

    step_log: list[dict[str, Any]] = []

    for _r in range(N_ROUNDS):
        snap = engine.step()
        
        p0 = engine.players[0].personality if engine.players else {}
        p_norm = np.sqrt(sum((v - initial_personality.get(k, 0.0)) ** 2 for k, v in p0.items()) / 9.0)

        eps = 1e-10
        x_list = [snap.p_aggressive, snap.p_defensive, snap.p_balanced]
        h = -sum(x * np.log(x + eps) for x in x_list)

        record = {
            "round": snap.round,
            "p_aggressive": snap.p_aggressive,
            "p_defensive": snap.p_defensive,
            "p_balanced": snap.p_balanced,
            "avg_reward": snap.avg_reward,
            "strategy_entropy": h,
            "personality_norm": p_norm,
            "personality_vector": dict(p0),
            "phase": snap.phase,
        }
        step_log.append(record)

    # Compute attractor metrics
    attractor_coords = compute_attractor_coordinates(step_log, TAIL_START)
    attractor_dist, attractor_norm = compute_attractor_distance(initial_personality, attractor_coords)
    attractor_var = compute_attractor_variance(step_log, TAIL_START)

    metrics = AttractorMetrics(
        attractor_coordinates=attractor_coords,
        attractor_distance=attractor_dist,
        attractor_norm=attractor_norm,
        attractor_variance=attractor_var,
    )

    return RunResult(
        seed=seed,
        alpha=alpha,
        group="",
        initial_personality=initial_personality,
        metrics=metrics,
        step_log=step_log,
    )


def check_gates(results: list[RunResult]) -> dict[str, bool | str]:
    """Evaluate G7F-01 ~ G7F-05."""
    gates: dict[str, bool | str] = {}

    # G7F-01: all 21 runs complete
    gates["G7F-01"] = len(results) == 21

    # G7F-02: no NaN/inf in attractor coordinates
    all_finite = all(
        all(np.isfinite(v) for v in r.metrics.attractor_coordinates.values())
        for r in results
    )
    gates["G7F-02"] = bool(all_finite)

    # G7F-03: same α convergence (pairwise distance < 0.05)
    alphas = sorted(set(r.alpha for r in results))
    g7f03_pass = True
    convergence_details = {}
    
    for alpha in alphas:
        attractors = [r.metrics.attractor_coordinates for r in results if r.alpha == alpha]
        if len(attractors) >= 2:
            dists = compute_pairwise_distances(attractors)
            upper_indices = np.triu_indices_from(dists, k=1)
            mean_dist = float(np.mean(dists[upper_indices])) if len(upper_indices[0]) > 0 else 0.0
            convergence_details[alpha] = mean_dist
            if mean_dist > 0.05:
                g7f03_pass = False
    
    gates["G7F-03"] = bool(g7f03_pass)
    gates["G7F-03_detail"] = json.dumps({round(k, 2): round(v, 5) for k, v in convergence_details.items()})

    # G7F-04: linear fit of ||attractor|| vs α (R² > 0.95)
    norms_by_alpha = {}
    for alpha in alphas:
        norms = [r.metrics.attractor_distance for r in results if r.alpha == alpha]
        norms_by_alpha[alpha] = float(np.mean(norms)) if norms else 0.0
    
    if len(norms_by_alpha) >= 3:
        alpha_arr = np.array(list(norms_by_alpha.keys()))
        norm_arr = np.array(list(norms_by_alpha.values()))
        
        fit = np.polyfit(alpha_arr, norm_arr, 1)
        y_pred = np.polyval(fit, alpha_arr)
        ss_res = np.sum((norm_arr - y_pred)**2)
        ss_tot = np.sum((norm_arr - np.mean(norm_arr))**2)
        r_squared = 1.0 - (ss_res / (ss_tot + 1e-10))
        
        gates["G7F-04"] = bool(r_squared > 0.95)
        gates["G7F-04_detail"] = f"R²={round(r_squared, 4)}, slope={round(fit[0], 4)}"
    else:
        gates["G7F-04"] = False
        gates["G7F-04_detail"] = "Insufficient α values"

    # G7F-05: attractor subspace dimension ≤ 3
    all_attractors = np.array([
        [r.metrics.attractor_coordinates.get(t, 0.0) for t in PERSONALITY_TRAITS]
        for r in results
    ])
    
    if all_attractors.shape[0] >= 2:
        dim, singular_vals = estimate_invariant_subspace_dim(all_attractors, threshold=0.95)
        gates["G7F-05"] = (dim <= 3)
        gates["G7F-05_detail"] = f"dim={dim}, singular_values={[round(v, 4) for v in singular_vals[:3]]}"
    else:
        gates["G7F-05"] = False
        gates["G7F-05_detail"] = "Insufficient runs"

    return gates


def main() -> None:
    parser = argparse.ArgumentParser(description="P7-F Attractor Personality Space Mapping")
    parser.add_argument("--alphas", type=float, nargs="+", default=[0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4])
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--group", type=str, default="G-AGG")
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--out", type=str, default="reports/experiments/p7f_attractor_mapping")
    args = parser.parse_args()

    out_dir = Path(ROOT / args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    alphas: list[float] = sorted(args.alphas)
    seeds: list[int] = args.seeds
    group: str = args.group

    print(f"[P7-F] Alphas: {alphas}  Seeds: {seeds}  Group: {group}  Out: {out_dir}")

    # Infer personality
    text = GROUP_TEXTS[group]
    personality = infer_group_personality(text)
    print(f"[P7-F] Inferred {group} ({text!r})")

    # Run all (alpha, seed) combinations
    all_results: list[RunResult] = []
    n_total = len(alphas) * len(seeds)
    print(f"\n[P7-F] Running {n_total} sessions (n_rounds={N_ROUNDS}) ...")

    for i, alpha in enumerate(alphas):
        for j, seed in enumerate(seeds):
            run_idx = i * len(seeds) + j + 1
            print(f"  [{run_idx}/{n_total}] α={alpha:.2f} seed={seed} ...", end="", flush=True)
            try:
                result = run_single(seed, alpha, personality)
                result.group = group
                all_results.append(result)
                print(
                    f" done | dist={result.metrics.attractor_distance:.4f}  "
                    f"var={result.metrics.attractor_variance:.6f}"
                )
                
                # Save per-run JSON
                run_file = out_dir / f"run_a{alpha:.2f}_{seed}.json"
                run_data = {
                    "seed": seed,
                    "alpha": alpha,
                    "group": group,
                    "initial_personality": result.initial_personality,
                    "metrics": {
                        "attractor_coordinates": result.metrics.attractor_coordinates,
                        "attractor_distance": result.metrics.attractor_distance,
                        "attractor_norm": result.metrics.attractor_norm,
                        "attractor_variance": result.metrics.attractor_variance,
                    },
                }
                run_file.write_text(json.dumps(run_data, indent=2, ensure_ascii=False))
                
                # Save full trajectory CSV
                traj_file = out_dir / f"run_a{alpha:.2f}_{seed}_trajectory.csv"
                if len(result.step_log) > 0:
                    fieldnames = list(result.step_log[0].keys())
                    # Remove nested dict from fieldnames
                    fieldnames = [f for f in fieldnames if f != "personality_vector"]
                    with traj_file.open("w", newline="", encoding="utf-8") as f:
                        writer = csv.DictWriter(f, fieldnames=fieldnames)
                        writer.writeheader()
                        for row in result.step_log:
                            row_filtered = {k: v for k, v in row.items() if k != "personality_vector"}
                            writer.writerow(row_filtered)
                
            except Exception as exc:
                print(f" FAILED: {exc}")
                import traceback
                traceback.print_exc()

    # Summary CSV
    summary_path = out_dir / "p7f_attractor_mapping_summary.csv"
    fieldnames = [
        "seed", "alpha", "group",
        "attractor_distance", "attractor_norm", "attractor_variance"
    ] + PERSONALITY_TRAITS

    with summary_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in all_results:
            row = {
                "seed": r.seed,
                "alpha": round(r.alpha, 2),
                "group": r.group,
                "attractor_distance": round(r.metrics.attractor_distance, 6),
                "attractor_norm": round(r.metrics.attractor_norm, 6),
                "attractor_variance": round(r.metrics.attractor_variance, 9),
            }
            for trait in PERSONALITY_TRAITS:
                row[trait] = round(r.metrics.attractor_coordinates.get(trait, 0.0), 6)
            writer.writerow(row)

    print(f"\n[P7-F] Summary CSV → {summary_path}")

    # Attractor coordinates JSON
    attractors_by_alpha = {}
    for alpha in alphas:
        attractors_by_alpha[alpha] = [
            r.metrics.attractor_coordinates for r in all_results if r.alpha == alpha
        ]
    
    attractors_path = out_dir / "p7f_attractor_coordinates.json"
    attractors_json = {
        str(k): v for k, v in attractors_by_alpha.items()
    }
    attractors_path.write_text(json.dumps(attractors_json, indent=2, ensure_ascii=False))

    # Subspace analysis
    all_attractors_arr = np.array([
        [r.metrics.attractor_coordinates.get(t, 0.0) for t in PERSONALITY_TRAITS]
        for r in all_results
    ])
    subspace_dim, singular_vals = estimate_invariant_subspace_dim(all_attractors_arr, threshold=0.95)
    
    subspace_path = out_dir / "p7f_subspace_analysis.json"
    subspace_data = {
        "subspace_dimension": int(subspace_dim),
        "singular_values": [float(v) for v in singular_vals],
        "explained_variance_ratio": [float(v**2) / np.sum(np.array(singular_vals)**2) for v in singular_vals],
    }
    subspace_path.write_text(json.dumps(subspace_data, indent=2, ensure_ascii=False))

    # Bifurcation analysis
    bifurcation_results = detect_bifurcation(attractors_by_alpha)
    bifurcation_path = out_dir / "p7f_bifurcation_analysis.json"
    bifurcation_path.write_text(json.dumps(bifurcation_results, indent=2, ensure_ascii=False))

    # Gate checks
    gates = check_gates(all_results)
    print("\n" + "=" * 60)
    print("Gate Check Results")
    print("=" * 60)
    for gate_id in ["G7F-01", "G7F-02", "G7F-03", "G7F-04", "G7F-05"]:
        passed = gates[gate_id]
        symbol = "✓ PASS" if passed else "✗ FAIL"
        detail = gates.get(f"{gate_id}_detail", "")
        print(f"  {gate_id}: {symbol}  {detail}")

    gates_path = out_dir / "p7f_gates.json"
    gates_serializable = {k: bool(v) if isinstance(v, bool) else str(v) for k, v in gates.items()}
    gates_path.write_text(json.dumps(gates_serializable, indent=2, ensure_ascii=False))

    all_passed = all(
        bool(gates[g]) for g in ["G7F-01", "G7F-02", "G7F-03", "G7F-04", "G7F-05"]
    )
    print("=" * 60)
    if all_passed:
        print("[P7-F] All gates PASSED → Attractor space fully characterized.")
        print(f"       Attractor subspace dimension: {subspace_dim}")
        print(f"       Bifurcation analysis: {'smooth' if bifurcation_results['smooth'] else 'bifurcations detected'}")
        print("       Ready for P7-G (Perturbation Analysis).")
    else:
        print("[P7-F] Some gates FAILED → Review results.")
    print(f"       Artifacts → {out_dir}/")

    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
