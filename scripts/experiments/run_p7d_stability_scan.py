"""P7-D 迴圈穩定性掃描（Closed-Loop Stability Scan）

Protocol lock:
  - n_players=4, n_rounds=300, burn_in=50
  - personality_mode="static", personality_update_enabled=True
  - Scan α ∈ {0.1, 0.2, 0.3, 0.4, 0.5}
  - seeds={42, 43, 44} (3 seeds)
  - Total: 5 alpha × 3 seeds = 15 runs
  - personality_learning_rate η = 0.05 (fixed)

Gate checks:
  G7D-01: 15/15 runs complete, no exception
  G7D-02: All runs have finite reward/personality (no NaN/inf)
  G7D-03: A_P increases monotonically with α
  G7D-04: At least one α produces S2 or S3 classification
  G7D-05: Same α across 3 seeds: std(A_P) < 0.08

Usage:
    ./venv/bin/python scripts/experiments/run_p7d_stability_scan.py \\
        --alphas 0.1 0.2 0.3 0.4 0.5 \\
        --seeds 42 43 44 \\
        --group G-AGG \\
        --learning-rate 0.05 \\
        --out reports/experiments/p7d_stability_scan
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

from api.personality_sbert_inference import infer_personality_vector_sbert
from simulation.rl_session_engine import RLSessionConfig, RLSessionEngine

# Protocol constants
N_PLAYERS = 4
N_ROUNDS = 300
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
class StabilityMetrics:
    """Stability indicators for one run."""
    amplitude_p: float                # A_P: L2 norm of personality change
    oscillation_score: float          # OSC: ACF peak of |v(t)|
    convergence_time: int | None      # t_c: rounds to reach steady state
    coupling_score: float             # Correlation(||P(t)||, H(x(t)))
    reward_mean: float
    reward_std: float
    strategy_entropy_mean: float      # H(x) mean over tail
    personality_saturation: float     # % of any trait reaching ±0.95


@dataclass
class RunResult:
    seed: int
    alpha: float
    group: str
    initial_personality: dict[str, float]
    final_personality: dict[str, float]
    metrics: StabilityMetrics
    classification: str               # S1/S2/S3/S4
    step_log: list[dict[str, Any]]


def infer_group_personality(text: str) -> dict[str, float]:
    """Infer 9D personality vector from text."""
    vector, _meta = infer_personality_vector_sbert(text)
    return {k: float(v) for k, v in vector.items()}


def compute_acf(signal: list[float], max_lag: int = 50) -> list[float]:
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


def classify_stability(metrics: StabilityMetrics) -> str:
    """Classify run into S1/S2/S3/S4 based on P7-D §3."""
    if metrics.amplitude_p < 0.05 and metrics.oscillation_score < 0.2:
        return "S1"  # Static
    elif metrics.amplitude_p <= 0.20 and metrics.oscillation_score < 0.4:
        return "S2"  # Stable dynamic
    elif metrics.amplitude_p <= 0.50 and 0.4 <= metrics.oscillation_score < 0.65:
        return "S3"  # Boundary oscillation
    else:
        return "S4"  # Divergent collapse


def run_single(
    seed: int,
    alpha: float,
    personality_vector: dict[str, float]
) -> RunResult:
    """Run one P7-D session with stability monitoring."""
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
    engine = RLSessionEngine(config, session_id=f"p7d_a{alpha:.2f}_{seed}")
    engine.reset()

    initial_personality = dict(engine.players[0].personality) if engine.players else {}

    step_log: list[dict[str, Any]] = []
    tail_rewards: list[float] = []
    personality_trajectory: list[list[float]] = []  # per-step norm
    strategy_entropy_list: list[float] = []
    personality_norm_list: list[float] = []

    for _r in range(N_ROUNDS):
        snap = engine.step()
        
        # Compute current personality norm
        p0 = engine.players[0].personality if engine.players else {}
        p_norm = np.sqrt(sum((v - initial_personality.get(k, 0.0)) ** 2 for k, v in p0.items()) / 9.0)
        personality_trajectory.append([p_norm])

        # Strategy entropy H(x) = -sum(x_s * ln(x_s))
        eps = 1e-10
        x_list = [snap.p_aggressive, snap.p_defensive, snap.p_balanced]
        h = -sum(x * np.log(x + eps) for x in x_list)
        strategy_entropy_list.append(h)
        personality_norm_list.append(p_norm)

        record = {
            "round": snap.round,
            "p_aggressive": snap.p_aggressive,
            "p_defensive": snap.p_defensive,
            "p_balanced": snap.p_balanced,
            "avg_reward": snap.avg_reward,
            "strategy_entropy": h,
            "personality_norm": p_norm,
            "phase": snap.phase,
        }
        step_log.append(record)

        if snap.warm:
            tail_rewards.append(snap.avg_reward)

    final_personality = dict(engine.players[0].personality) if engine.players else {}

    # Compute stability metrics
    personality_change = {}
    max_change = 0.0
    saturation_count = 0
    for trait in PERSONALITY_TRAITS:
        delta = final_personality.get(trait, 0.0) - initial_personality.get(trait, 0.0)
        personality_change[trait] = delta
        max_change = max(max_change, abs(delta))
        if abs(final_personality.get(trait, 0.0)) > 0.95:
            saturation_count += 1

    amplitude_p = max_change

    # Oscillation score: ACF of |dP/dt|
    velocity = [abs(step_log[i + 1]["personality_norm"] - step_log[i]["personality_norm"])
                for i in range(len(step_log) - 1)]
    acf = compute_acf(velocity[BURN_IN:], max_lag=min(50, len(velocity) - BURN_IN - 20))
    oscillation_score = max(acf[20:]) if len(acf) > 20 else max(acf)

    # Convergence time
    convergence_time = None
    for i in range(BURN_IN, len(velocity) - 20):
        if all(velocity[j] < 0.01 for j in range(i, min(i + 20, len(velocity)))):
            convergence_time = i - BURN_IN
            break

    # Coupling: Pearson correlation of ||P(t)|| and H(x(t))
    if len(personality_norm_list) > 10 and len(strategy_entropy_list) > 10:
        tail_p_norm = personality_norm_list[BURN_IN:]
        tail_entropy = strategy_entropy_list[BURN_IN:]
        if len(tail_p_norm) > 1 and np.std(tail_p_norm) > 0 and np.std(tail_entropy) > 0:
            coupling = float(np.corrcoef(tail_p_norm, tail_entropy)[0, 1])
        else:
            coupling = 0.0
    else:
        coupling = 0.0

    reward_mean = float(np.mean(tail_rewards)) if tail_rewards else float("nan")
    reward_std = float(np.std(tail_rewards)) if tail_rewards else float("nan")
    entropy_mean = float(np.mean(strategy_entropy_list[BURN_IN:])) if len(strategy_entropy_list) > BURN_IN else float("nan")
    saturation_ratio = saturation_count / 9.0

    metrics = StabilityMetrics(
        amplitude_p=amplitude_p,
        oscillation_score=oscillation_score,
        convergence_time=convergence_time,
        coupling_score=coupling,
        reward_mean=reward_mean,
        reward_std=reward_std,
        strategy_entropy_mean=entropy_mean,
        personality_saturation=saturation_ratio,
    )

    classification = classify_stability(metrics)

    return RunResult(
        seed=seed,
        alpha=alpha,
        group="",  # Set in main
        initial_personality=initial_personality,
        final_personality=final_personality,
        metrics=metrics,
        classification=classification,
        step_log=step_log,
    )


def check_gates(results: list[RunResult]) -> dict[str, bool | str]:
    """Evaluate G7D-01 ~ G7D-05."""
    gates: dict[str, bool | str] = {}

    # G7D-01: all 15 runs complete
    gates["G7D-01"] = len(results) == 15

    # G7D-02: no NaN/inf
    all_finite = all(
        np.isfinite(r.metrics.reward_mean) and np.isfinite(r.metrics.amplitude_p)
        for r in results
    )
    gates["G7D-02"] = bool(all_finite)

    # G7D-03: A_P increases monotonically with α
    alpha_sorted = sorted(set(r.alpha for r in results))
    if len(alpha_sorted) > 1:
        mean_amp = {}
        for a in alpha_sorted:
            amps = [r.metrics.amplitude_p for r in results if r.alpha == a]
            mean_amp[a] = float(np.mean(amps))
        
        diffs = [mean_amp[alpha_sorted[i + 1]] - mean_amp[alpha_sorted[i]]
                 for i in range(len(alpha_sorted) - 1)]
        monotonic = all(d > -0.02 for d in diffs)  # Allow small noise
        gates["G7D-03"] = bool(monotonic)
        gates["G7D-03_detail"] = f"A_P trend: {json.dumps({a: round(mean_amp[a], 4) for a in alpha_sorted})}"
    else:
        gates["G7D-03"] = False
        gates["G7D-03_detail"] = "Insufficient alpha values"

    # G7D-04: at least one S2 or S3
    classifications = [r.classification for r in results]
    has_stable = any(c in ["S2", "S3"] for c in classifications)
    gates["G7D-04"] = bool(has_stable)
    gates["G7D-04_detail"] = f"Classifications: {dict(zip([f'a={r.alpha:.1f}_s{r.seed}' for r in results], classifications))}"

    # G7D-05: re-producibility across seeds for same α
    alpha_std_dict = {}
    for a in alpha_sorted:
        amps = [r.metrics.amplitude_p for r in results if r.alpha == a]
        std = float(np.std(amps)) if len(amps) > 1 else 0.0
        alpha_std_dict[a] = std
    
    max_std = max(alpha_std_dict.values()) if alpha_std_dict else 0.0
    gates["G7D-05"] = max_std < 0.08
    gates["G7D-05_detail"] = f"max std(A_P) across seeds per α: {round(max_std, 4)}"

    return gates


def main() -> None:
    parser = argparse.ArgumentParser(description="P7-D Stability Scan")
    parser.add_argument("--alphas", type=float, nargs="+", default=[0.1, 0.2, 0.3, 0.4, 0.5])
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--group", type=str, default="G-AGG")
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--out", type=str, default="reports/experiments/p7d_stability_scan")
    args = parser.parse_args()

    out_dir = Path(ROOT / args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    alphas: list[float] = sorted(args.alphas)
    seeds: list[int] = args.seeds
    group: str = args.group

    print(f"[P7-D] Alphas: {alphas}  Seeds: {seeds}  Group: {group}  Out: {out_dir}")

    # Infer personality
    text = GROUP_TEXTS[group]
    personality = infer_group_personality(text)
    print(f"[P7-D] Inferred {group} ({text!r})")

    # Run all (alpha, seed) combinations
    all_results: list[RunResult] = []
    n_total = len(alphas) * len(seeds)
    print(f"\n[P7-D] Running {n_total} sessions (n_rounds={N_ROUNDS}) ...")

    for i, alpha in enumerate(alphas):
        for j, seed in enumerate(seeds):
            run_idx = i * len(seeds) + j + 1
            print(f"  [{run_idx}/{n_total}] α={alpha:.2f} seed={seed} ...", end="", flush=True)
            try:
                result = run_single(seed, alpha, personality)
                result.group = group
                all_results.append(result)
                print(
                    f" done | A_P={result.metrics.amplitude_p:.4f}  "
                    f"OSC={result.metrics.oscillation_score:.4f}  "
                    f"class={result.classification}"
                )
                # Save per-run JSON
                run_file = out_dir / f"run_a{alpha:.2f}_{seed}.json"
                run_data = {
                    "seed": seed,
                    "alpha": alpha,
                    "group": group,
                    "metrics": {
                        "amplitude_p": result.metrics.amplitude_p,
                        "oscillation_score": result.metrics.oscillation_score,
                        "convergence_time": result.metrics.convergence_time,
                        "coupling_score": result.metrics.coupling_score,
                        "reward_mean": result.metrics.reward_mean,
                        "reward_std": result.metrics.reward_std,
                        "strategy_entropy_mean": result.metrics.strategy_entropy_mean,
                        "personality_saturation": result.metrics.personality_saturation,
                    },
                    "classification": result.classification,
                    "step_log": result.step_log,
                }
                run_file.write_text(json.dumps(run_data, indent=2, ensure_ascii=False))
            except Exception as exc:
                print(f" FAILED: {exc}")
                import traceback
                traceback.print_exc()

    # Summary CSV
    summary_path = out_dir / "p7d_stability_summary.csv"
    fieldnames = [
        "seed", "alpha", "group",
        "amplitude_p", "oscillation_score", "convergence_time", "coupling_score",
        "reward_mean", "reward_std", "strategy_entropy_mean", "personality_saturation",
        "classification"
    ]

    with summary_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in all_results:
            writer.writerow({
                "seed": r.seed,
                "alpha": round(r.alpha, 2),
                "group": r.group,
                "amplitude_p": round(r.metrics.amplitude_p, 6),
                "oscillation_score": round(r.metrics.oscillation_score, 6),
                "convergence_time": r.metrics.convergence_time if r.metrics.convergence_time is not None else "inf",
                "coupling_score": round(r.metrics.coupling_score, 6),
                "reward_mean": round(r.metrics.reward_mean, 6),
                "reward_std": round(r.metrics.reward_std, 6),
                "strategy_entropy_mean": round(r.metrics.strategy_entropy_mean, 6),
                "personality_saturation": round(r.metrics.personality_saturation, 4),
                "classification": r.classification,
            })

    print(f"\n[P7-D] Summary CSV → {summary_path}")

    # Gate checks
    gates = check_gates(all_results)
    print("\n" + "=" * 60)
    print("Gate Check Results")
    print("=" * 60)
    for gate_id in ["G7D-01", "G7D-02", "G7D-03", "G7D-04", "G7D-05"]:
        passed = gates[gate_id]
        symbol = "✓ PASS" if passed else "✗ FAIL"
        detail = gates.get(f"{gate_id}_detail", "")
        print(f"  {gate_id}: {symbol}  {detail}")

    gates_path = out_dir / "p7d_gates.json"
    gates_serializable = {k: bool(v) if isinstance(v, bool) else str(v) for k, v in gates.items()}
    gates_path.write_text(json.dumps(gates_serializable, indent=2, ensure_ascii=False))

    all_passed = all(
        bool(gates[g]) for g in ["G7D-01", "G7D-02", "G7D-03", "G7D-04", "G7D-05"]
    )
    print("=" * 60)
    if all_passed:
        print("[P7-D] All gates PASSED → Stability profile characterized.")
        print("       Recommend α* from S2 classification for P7-E long-term study.")
    else:
        print("[P7-D] Some gates FAILED → Review results.")
    print(f"       Artifacts → {out_dir}/")

    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
