"""P7-E 長期人格軌跡特徵化（Long-term Trajectory Characterization）

Protocol lock:
  - n_players=4, n_rounds=1000, burn_in=100
  - personality_mode="static", personality_update_enabled=True
  - Scan α ∈ {0.2, 0.3}
  - seeds={42, 43, 44, 101, 102} (5 seeds)
  - Total: 2 alpha × 5 seeds = 10 runs
  - personality_learning_rate η = 0.05 (fixed)

Gate checks:
  G7E-01: 10/10 runs complete, no exception
  G7E-02: All runs have finite reward/personality (no NaN/inf)
  G7E-03: At least one run exhibits VDI < -0.01 (convergence)
  G7E-04: Same α across 5 seeds: std(plasticity_window) < 150 rounds
  G7E-05: Same α across 5 seeds: cosine_similarity(final_personality) > 0.8

Usage:
    ./venv/bin/python scripts/experiments/run_p7e_long_trajectory.py \\
        --alphas 0.2 0.3 \\
        --seeds 42 43 44 101 102 \\
        --group G-AGG \\
        --learning-rate 0.05 \\
        --out reports/experiments/p7e_long_trajectory
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
from scipy import stats  # type: ignore
from scipy.spatial.distance import cosine  # type: ignore

from api.personality_sbert_inference import infer_personality_vector_sbert
from simulation.rl_session_engine import RLSessionConfig, RLSessionEngine

# Protocol constants
N_PLAYERS = 4
N_ROUNDS = 1000
BURN_IN = 100
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
    """Attractor and long-term stability indicators."""
    vdi: float                         # Variance Decay Index
    act: int                           # Autocorrelation Timescale (rounds)
    plasticity_window: int | None      # When personality locks (rounds)
    coupling_strength: dict[str, float] # {"mean": ..., "std": ..., "trend": ...}
    final_personality_vector: dict[str, float]
    personality_saturation: float      # Fraction of traits at |P_i| > 0.95


@dataclass
class RunResult:
    seed: int
    alpha: float
    group: str
    initial_personality: dict[str, float]
    final_personality: dict[str, float]
    metrics: AttractorMetrics
    step_log: list[dict[str, Any]]


def infer_group_personality(text: str) -> dict[str, float]:
    """Infer 9D personality vector from text."""
    vector, _meta = infer_personality_vector_sbert(text)
    return {k: float(v) for k, v in vector.items()}


def compute_acf(signal: list[float], max_lag: int = 200) -> list[float]:
    """Compute autocorrelation function (ACF) via numpy."""
    if len(signal) < max_lag:
        return [1.0] * min(len(signal), max_lag)
    
    signal_arr = np.array(signal)
    mean = np.mean(signal_arr)
    c0 = np.sum((signal_arr - mean) ** 2) / len(signal_arr)
    
    acf = [1.0]
    for lag in range(1, max_lag):
        c_lag = np.sum((signal_arr[:-lag] - mean) * (signal_arr[lag:] - mean)) / len(signal_arr)
        acf.append(c_lag / c0 if c0 > 0 else 0.0)
    
    return acf


def compute_vdi(trajectory: list[float], window_size: int = 100) -> tuple[float, list[float]]:
    """
    Compute Variance Decay Index (VDI).
    Divides trajectory into windows and checks if variance monotonically decreases.
    
    VDI < -0.01: strong convergence to fixed point
    -0.01 ≤ VDI ≤ 0.01: steady state (no drift)
    VDI > 0.01: continued drift or divergence
    """
    if len(trajectory) < window_size * 2:
        return 0.0, [np.var(trajectory)]
    
    n_windows = len(trajectory) // window_size
    windows = [trajectory[i*window_size:(i+1)*window_size] for i in range(n_windows)]
    variances = [np.var(w) if len(w) > 0 else 0.0 for w in windows]
    
    if len(variances) < 2:
        return 0.0, variances
    
    # VDI = mean of variance differences
    slopes = [variances[i+1] - variances[i] for i in range(len(variances)-1)]
    vdi = float(np.mean(slopes))
    
    return vdi, variances


def compute_acf_timescale(trajectory: list[float], max_lag: int = 200) -> int:
    """
    Compute autocorrelation timescale (ACT).
    Returns the lag at which ACF first drops below e^(-1) ≈ 0.37.
    
    ACT < 50: highly random (short memory)
    50 ≤ ACT < 200: moderate correlation (dynamic evolution)
    ACT ≥ 200: strong correlation (long memory, possible large-scale loop)
    """
    if len(trajectory) < max_lag:
        return len(trajectory)
    
    acf = compute_acf(trajectory, max_lag)
    threshold = np.exp(-1)  # ≈ 0.37
    
    for lag, val in enumerate(acf):
        if val < threshold:
            return lag
    
    return max_lag


def find_plasticity_window(
    step_log: list[dict[str, Any]],
    threshold: float = 0.01,
    window_size: int = 50
) -> int | None:
    """
    Find the round when personality stops changing significantly.
    
    Plasticity window = first t where |dP/dt| < threshold for window_size consecutive rounds
    """
    if len(step_log) < window_size:
        return None
    
    velocities = [
        abs(step_log[i+1]["personality_norm"] - step_log[i]["personality_norm"])
        for i in range(len(step_log)-1)
    ]
    
    for i in range(len(velocities) - window_size):
        if all(v < threshold for v in velocities[i:i+window_size]):
            return i
    
    return None


def compute_coupling_strength(
    step_log: list[dict[str, Any]],
    burn_in: int = BURN_IN,
    window_size: int = 100
) -> dict[str, float]:
    """
    Compute strategy-personality coupling strength.
    Measures correlation between personality velocity and reward across windows.
    """
    if len(step_log) < burn_in + window_size:
        return {"mean_correlation": 0.0, "std_correlation": 0.0, "trend": "flat"}
    
    personality_velocities = [
        abs(step_log[i+1]["personality_norm"] - step_log[i]["personality_norm"])
        for i in range(len(step_log)-1)
    ]
    rewards = [step_log[i]["avg_reward"] for i in range(len(step_log))]
    
    # Extract windows from tail
    windows = []
    for i in range(burn_in, len(step_log) - window_size, window_size):
        pv_window = personality_velocities[i:i+window_size]
        r_window = rewards[i:i+window_size]
        if len(pv_window) > 1 and np.std(pv_window) > 0 and np.std(r_window) > 0:
            try:
                corr = float(np.corrcoef(pv_window, r_window)[0, 1])
                if not np.isnan(corr):
                    windows.append(corr)
            except:
                pass
    
    if len(windows) < 2:
        return {"mean_correlation": 0.0, "std_correlation": 0.0, "trend": "flat"}
    
    mean_corr = float(np.mean(windows))
    std_corr = float(np.std(windows))
    
    # Trend: strengthening or weakening?
    slope = np.polyfit(range(len(windows)), windows, 1)[0]
    trend = "strengthening" if slope > 0 else "weakening"
    
    return {
        "mean_correlation": mean_corr,
        "std_correlation": std_corr,
        "trend": trend,
        "n_windows": len(windows)
    }


def run_single(
    seed: int,
    alpha: float,
    personality_vector: dict[str, float]
) -> RunResult:
    """Run one P7-E session with long-term stability monitoring."""
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
    engine = RLSessionEngine(config, session_id=f"p7e_a{alpha:.2f}_{seed}")
    engine.reset()

    initial_personality = dict(engine.players[0].personality) if engine.players else {}

    step_log: list[dict[str, Any]] = []
    personality_trajectory: list[float] = []

    for _r in range(N_ROUNDS):
        snap = engine.step()
        
        # Compute personality norm (L2 from initial state)
        p0 = engine.players[0].personality if engine.players else {}
        p_norm = np.sqrt(sum((v - initial_personality.get(k, 0.0)) ** 2 for k, v in p0.items()) / 9.0)
        personality_trajectory.append(p_norm)

        # Strategy entropy H(x) = -sum(x_s * ln(x_s))
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
            "personality_velocity": abs(personality_trajectory[_r] - (personality_trajectory[_r-1] if _r > 0 else 0)),
            "phase": snap.phase,
        }
        step_log.append(record)

    final_personality = dict(engine.players[0].personality) if engine.players else {}

    # Compute attractor metrics
    vdi, _ = compute_vdi(personality_trajectory[BURN_IN:], window_size=100)
    act = compute_acf_timescale(personality_trajectory[BURN_IN:], max_lag=200)
    plasticity_window = find_plasticity_window(step_log, threshold=0.01, window_size=50)
    coupling = compute_coupling_strength(step_log, burn_in=BURN_IN, window_size=100)

    # Personality saturation
    saturation_count = 0
    for trait in PERSONALITY_TRAITS:
        if abs(final_personality.get(trait, 0.0)) > 0.95:
            saturation_count += 1
    saturation_ratio = saturation_count / 9.0

    metrics = AttractorMetrics(
        vdi=vdi,
        act=act,
        plasticity_window=plasticity_window,
        coupling_strength=coupling,
        final_personality_vector=final_personality,
        personality_saturation=saturation_ratio,
    )

    return RunResult(
        seed=seed,
        alpha=alpha,
        group="",
        initial_personality=initial_personality,
        final_personality=final_personality,
        metrics=metrics,
        step_log=step_log,
    )


def check_gates(results: list[RunResult]) -> dict[str, bool | str]:
    """Evaluate G7E-01 ~ G7E-05."""
    gates: dict[str, bool | str] = {}

    # G7E-01: all 10 runs complete
    gates["G7E-01"] = len(results) == 10

    # G7E-02: no NaN/inf
    all_finite = all(
        np.isfinite(r.metrics.coupling_strength.get("mean_correlation", 0.0)) and 
        np.isfinite(r.metrics.vdi) and
        all(np.isfinite(v) for v in r.final_personality.values())
        for r in results
    )
    gates["G7E-02"] = bool(all_finite)

    # G7E-03: at least one run with VDI < -0.01
    has_convergence = any(r.metrics.vdi < -0.01 for r in results)
    gates["G7E-03"] = bool(has_convergence)
    vdi_values = [r.metrics.vdi for r in results]
    gates["G7E-03_detail"] = f"min VDI: {min(vdi_values):.4f}, max VDI: {max(vdi_values):.4f}"

    # G7E-04: reproducibility of plasticity window
    pw_by_alpha = {}
    for alpha in [0.2, 0.3]:
        pw_list = [r.metrics.plasticity_window for r in results 
                   if r.alpha == alpha and r.metrics.plasticity_window is not None]
        if len(pw_list) >= 2:
            pw_std = float(np.std(pw_list))
            pw_by_alpha[alpha] = pw_std
    
    g7e04_pass = all(std < 150 for std in pw_by_alpha.values()) if pw_by_alpha else False
    gates["G7E-04"] = bool(g7e04_pass)
    gates["G7E-04_detail"] = f"std(PW) per α: {json.dumps({k: round(v, 1) for k, v in pw_by_alpha.items()})}"

    # G7E-05: personality vector consistency across seeds (cosine similarity > 0.8)
    g7e05_pass = True
    similarity_by_alpha = {}
    for alpha in [0.2, 0.3]:
        final_vecs = [r.final_personality for r in results if r.alpha == alpha]
        if len(final_vecs) >= 2:
            # Normalize vectors
            vecs_norm = []
            for v in final_vecs:
                vec = np.array([v.get(t, 0.0) for t in PERSONALITY_TRAITS])
                vec_norm = vec / (np.linalg.norm(vec) + 1e-10)
                vecs_norm.append(vec_norm)
            
            # Pairwise similarity
            sims = []
            for i in range(len(vecs_norm)):
                for j in range(i+1, len(vecs_norm)):
                    sim = 1.0 - cosine(vecs_norm[i], vecs_norm[j])
                    sims.append(sim)
            
            if sims:
                mean_sim = float(np.mean(sims))
                similarity_by_alpha[alpha] = mean_sim
                if mean_sim < 0.8:
                    g7e05_pass = False
    
    gates["G7E-05"] = bool(g7e05_pass)
    gates["G7E-05_detail"] = f"cosine similarity per α: {json.dumps({k: round(v, 3) for k, v in similarity_by_alpha.items()})}"

    return gates


def main() -> None:
    parser = argparse.ArgumentParser(description="P7-E Long-term Trajectory Characterization")
    parser.add_argument("--alphas", type=float, nargs="+", default=[0.2, 0.3])
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 101, 102])
    parser.add_argument("--group", type=str, default="G-AGG")
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--out", type=str, default="reports/experiments/p7e_long_trajectory")
    args = parser.parse_args()

    out_dir = Path(ROOT / args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    alphas: list[float] = sorted(args.alphas)
    seeds: list[int] = args.seeds
    group: str = args.group

    print(f"[P7-E] Alphas: {alphas}  Seeds: {seeds}  Group: {group}  Out: {out_dir}")

    # Infer personality
    text = GROUP_TEXTS[group]
    personality = infer_group_personality(text)
    print(f"[P7-E] Inferred {group} ({text!r})")

    # Run all (alpha, seed) combinations
    all_results: list[RunResult] = []
    n_total = len(alphas) * len(seeds)
    print(f"\n[P7-E] Running {n_total} sessions (n_rounds={N_ROUNDS}) ...")

    for i, alpha in enumerate(alphas):
        for j, seed in enumerate(seeds):
            run_idx = i * len(seeds) + j + 1
            print(f"  [{run_idx}/{n_total}] α={alpha:.2f} seed={seed} ...", end="", flush=True)
            try:
                result = run_single(seed, alpha, personality)
                result.group = group
                all_results.append(result)
                print(
                    f" done | VDI={result.metrics.vdi:.4f}  "
                    f"ACT={result.metrics.act}  "
                    f"PW={result.metrics.plasticity_window if result.metrics.plasticity_window else 'inf'}"
                )
                
                # Save per-run JSON (minimal: attractor metrics only, not full step_log for size)
                run_file = out_dir / f"run_a{alpha:.2f}_{seed}.json"
                run_data = {
                    "seed": seed,
                    "alpha": alpha,
                    "group": group,
                    "initial_personality": result.initial_personality,
                    "final_personality": result.final_personality,
                    "metrics": {
                        "vdi": result.metrics.vdi,
                        "act": result.metrics.act,
                        "plasticity_window": result.metrics.plasticity_window,
                        "coupling_strength": result.metrics.coupling_strength,
                        "personality_saturation": result.metrics.personality_saturation,
                    },
                    "step_log_size": len(result.step_log),  # Metadata only
                }
                run_file.write_text(json.dumps(run_data, indent=2, ensure_ascii=False))
                
                # Save full trajectory CSV (for detailed analysis)
                traj_file = out_dir / f"run_a{alpha:.2f}_{seed}_trajectory.csv"
                if len(result.step_log) > 0:
                    fieldnames = list(result.step_log[0].keys())
                    with traj_file.open("w", newline="", encoding="utf-8") as f:
                        writer = csv.DictWriter(f, fieldnames=fieldnames)
                        writer.writeheader()
                        writer.writerows(result.step_log)
                
            except Exception as exc:
                print(f" FAILED: {exc}")
                import traceback
                traceback.print_exc()

    # Summary CSV
    summary_path = out_dir / "p7e_long_trajectory_summary.csv"
    fieldnames = [
        "seed", "alpha", "group",
        "vdi", "act", "plasticity_window", "coupling_mean", "coupling_std", "coupling_trend",
        "personality_saturation"
    ]

    with summary_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in all_results:
            writer.writerow({
                "seed": r.seed,
                "alpha": round(r.alpha, 2),
                "group": r.group,
                "vdi": round(r.metrics.vdi, 6),
                "act": r.metrics.act,
                "plasticity_window": r.metrics.plasticity_window if r.metrics.plasticity_window is not None else "inf",
                "coupling_mean": round(r.metrics.coupling_strength.get("mean_correlation", 0.0), 6),
                "coupling_std": round(r.metrics.coupling_strength.get("std_correlation", 0.0), 6),
                "coupling_trend": r.metrics.coupling_strength.get("trend", "N/A"),
                "personality_saturation": round(r.metrics.personality_saturation, 4),
            })

    print(f"\n[P7-E] Summary CSV → {summary_path}")

    # Gate checks
    gates = check_gates(all_results)
    print("\n" + "=" * 60)
    print("Gate Check Results")
    print("=" * 60)
    for gate_id in ["G7E-01", "G7E-02", "G7E-03", "G7E-04", "G7E-05"]:
        passed = gates[gate_id]
        symbol = "✓ PASS" if passed else "✗ FAIL"
        detail = gates.get(f"{gate_id}_detail", "")
        print(f"  {gate_id}: {symbol}  {detail}")

    gates_path = out_dir / "p7e_gates.json"
    gates_serializable = {k: bool(v) if isinstance(v, bool) else str(v) for k, v in gates.items()}
    gates_path.write_text(json.dumps(gates_serializable, indent=2, ensure_ascii=False))

    all_passed = all(
        bool(gates[g]) for g in ["G7E-01", "G7E-02", "G7E-03", "G7E-04", "G7E-05"]
    )
    print("=" * 60)
    if all_passed:
        print("[P7-E] All gates PASSED → Long-term trajectory characterized.")
        print("       Attractor structure identified → Ready for P7-F (attractor analysis).")
    else:
        print("[P7-E] Some gates FAILED → Review results.")
    print(f"       Artifacts → {out_dir}/")

    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
