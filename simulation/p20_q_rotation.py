"""P20 Q-Rotation Bias — tangential drift in Q-space.

Phase 20 experiment: On the BL1 baseline (well_mixed ε=0.02, wide α, β=3),
test whether injecting a small antisymmetric rotation bias into Q-values
after each Q-update can improve rotation consistency (turn_strength / stab).

Sweep parameter: δ_rot ∈ {0.0 (control), 0.001, 0.005, 0.01, 0.02}

Architecture
------------
- Thin wrapper around E1's _run_e1_simulation (same pattern as P18).
- Uses the new q_rotation_delta param (Phase 20).
- CC1 fallback + revised Pass Gate (same as BL1).

Usage
-----
    ./venv/bin/python -m simulation.p20_q_rotation \\
        --delta-rots 0.001,0.005,0.01,0.02 \\
        --seeds 45,47,49
"""

from __future__ import annotations

import argparse
import csv
import json
from math import atan2, sqrt
from pathlib import Path
from typing import Any

from analysis.cycle_metrics import classify_cycle_level
from analysis.decay_rate import estimate_decay_gamma
from evolution.independent_rl import (
    STRATEGY_SPACE,
    boltzmann_weights,
    weight_entropy,
)
from simulation.e1_heterogeneous_rl import (
    _run_e1_simulation,
    _write_csv,
    _write_simplex_plot,
    _write_phase_amplitude_plot,
    _make_graph_spec,
    _fmt,
    _yn,
    _mean,
    _std,
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

G2_GATE = "g2_p20_q_rotation"

# BL1 locked parameters
ALPHA_LO = 0.005
ALPHA_HI = 0.40
BETA_CEILING = 3.0
TOPOLOGY = "well_mixed"
PAYOFF_EPSILON = 0.02  # BL1 anchor ε

# CC1 fallback thresholds
CC1_R2_THR = 0.85
CC1_MIN_ROT = 20.0

# Revised Pass Gate thresholds
GATE_ENTROPY = 1.06
GATE_Q_STD = 0.04

DEFAULT_OUT_ROOT = Path("outputs/p20_q_rotation")
DEFAULT_SUMMARY_TSV = DEFAULT_OUT_ROOT / "p20_summary.tsv"
DEFAULT_COMBINED_TSV = DEFAULT_OUT_ROOT / "p20_combined.tsv"
DEFAULT_DECISION_MD = DEFAULT_OUT_ROOT / "p20_decision.md"

SUMMARY_FIELDNAMES = [
    "gate", "condition", "topology", "beta_ceiling",
    "payoff_epsilon", "q_rotation_delta",
    "alpha_lo", "alpha_hi", "seed",
    "realized_alpha_std",
    "cycle_level", "stage3_score", "turn_strength",
    "env_gamma", "env_gamma_r2", "env_gamma_n_peaks",
    "mean_player_weight_entropy", "weight_heterogeneity_std",
    "mean_q_value_std", "mean_neighbor_weight_cosine",
    "strategy_cycle_stability",
    "spatial_strategy_clustering", "mean_edge_strategy_distance",
    "has_level3_seed", "out_csv", "provenance_json",
]

COMBINED_FIELDNAMES = [
    "gate", "condition", "topology", "beta_ceiling",
    "payoff_epsilon", "q_rotation_delta",
    "alpha_lo", "alpha_hi",
    "is_control", "n_seeds",
    "mean_cycle_level", "mean_stage3_score", "mean_turn_strength", "mean_env_gamma",
    "level_counts_json", "p_level_3", "level3_seed_count",
    "realized_alpha_std",
    "mean_player_weight_entropy", "weight_heterogeneity_std",
    "mean_q_value_std", "mean_neighbor_weight_cosine",
    "strategy_cycle_stability",
    "spatial_strategy_clustering", "mean_edge_strategy_distance",
    "short_scout_pass", "hard_stop_fail", "longer_confirm_candidate", "verdict",
    "representative_seed",
    "representative_simplex_png", "representative_phase_amplitude_png",
    "players", "rounds", "out_dir",
]

_NSTRATS = len(STRATEGY_SPACE)


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _parse_seeds(s: str) -> list[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip()]

def _parse_floats(s: str) -> list[float]:
    return [float(x.strip()) for x in s.split(",") if x.strip()]

def _write_tsv(path: Path, *, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t", extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def _cond_name(*, delta_rot: float) -> str:
    def _tok(v: float) -> str:
        return f"{v:.4f}".rstrip("0").rstrip(".").replace(".", "p")
    return f"g2_p20_{TOPOLOGY}_dr{_tok(delta_rot)}"


# ---------------------------------------------------------------------------
# Strategy cycle stability
# ---------------------------------------------------------------------------

def _phase_angle(pa: float, pd: float, pb: float) -> float:
    return float(atan2(sqrt(3.0) * (pd - pb), 2.0 * pa - pd - pb))


def _strategy_cycle_stability(rows: list[dict[str, Any]], *,
                               burn_in: int, tail: int) -> float:
    n_rows = len(rows)
    begin = max(int(burn_in), n_rows - int(tail))
    window = rows[begin:]
    if len(window) < 10:
        return 0.0
    try:
        import numpy as np
    except ImportError:
        return 0.0
    phases_raw = [_phase_angle(float(r["p_aggressive"]), float(r["p_defensive"]),
                                float(r["p_balanced"])) for r in window]
    phases = np.unwrap(phases_raw)
    x = np.arange(len(phases), dtype=float)
    x_mean = x.mean()
    p_mean = phases.mean()
    ss_xx = ((x - x_mean) ** 2).sum()
    ss_xy = ((x - x_mean) * (phases - p_mean)).sum()
    if ss_xx < 1e-30:
        return 0.0
    slope = ss_xy / ss_xx
    intercept = p_mean - slope * x_mean
    predicted = slope * x + intercept
    ss_res = ((phases - predicted) ** 2).sum()
    ss_tot = ((phases - p_mean) ** 2).sum()
    if ss_tot < 1e-30:
        return 0.0
    r2 = float(1.0 - ss_res / ss_tot)
    return max(0.0, r2)


# ---------------------------------------------------------------------------
# Seed metrics (with CC1 fallback)
# ---------------------------------------------------------------------------

def _seed_metrics(
    rows: list[dict[str, Any]], *, burn_in: int, tail: int,
    eta: float, corr_threshold: float,
) -> dict[str, float | int]:
    sm = {
        "aggressive": [float(r["p_aggressive"]) for r in rows],
        "defensive":  [float(r["p_defensive"])  for r in rows],
        "balanced":   [float(r["p_balanced"])   for r in rows],
    }
    cyc = classify_cycle_level(
        sm, burn_in=int(burn_in), tail=int(tail),
        amplitude_threshold=0.02, corr_threshold=float(corr_threshold),
        eta=float(eta), stage3_method="turning", phase_smoothing=1,
        stage2_fallback_r2_threshold=CC1_R2_THR,
        stage2_fallback_min_rotation=CC1_MIN_ROT,
    )
    fit = estimate_decay_gamma(sm, series_kind="p")
    return {
        "cycle_level":       int(cyc.level),
        "stage3_score":      float(cyc.stage3.score) if cyc.stage3 else 0.0,
        "turn_strength":     float(cyc.stage3.turn_strength) if cyc.stage3 else 0.0,
        "env_gamma":         float(fit.gamma) if fit else 0.0,
        "env_gamma_r2":      float(fit.r2) if fit else 0.0,
        "env_gamma_n_peaks": int(fit.n_peaks) if fit else 0,
    }


def _tail_begin(n_rows: int, *, burn_in: int, tail: int) -> int:
    if n_rows <= 0:
        return 0
    return max(int(burn_in), n_rows - int(tail))


def _tail_diagnostics(
    round_diag: list[dict[str, Any]],
    final_q: list[list[float]],
    adj: list[list[int]] | None,
    player_betas: list[float],
    *, n_rows: int, burn_in: int, tail: int,
) -> dict[str, float]:
    begin = _tail_begin(n_rows, burn_in=int(burn_in), tail=int(tail))
    window = round_diag[begin:]
    if not window:
        return {
            "mean_player_weight_entropy": 0.0,
            "weight_heterogeneity_std": 0.0,
            "mean_q_value_std": 0.0,
            "mean_neighbor_weight_cosine": 0.0,
            "spatial_strategy_clustering": 0.0,
            "mean_edge_strategy_distance": 0.0,
        }

    mh  = _mean([float(d["mean_player_weight_entropy"]) for d in window])
    ws  = _mean([float(d["weight_heterogeneity_std"]) for d in window])
    qs  = _mean([float(d["mean_q_value_std"]) for d in window])
    nwc = _mean([float(d["mean_neighbor_weight_cosine"]) for d in window])

    ssc = 0.0
    mesd = 0.0
    if adj is not None:
        from evolution.local_graph import edge_disagreement_rate, edge_strategy_distance
        final_w = [boltzmann_weights(q, beta=player_betas[i])
                    for i, q in enumerate(final_q)]
        dom = [STRATEGY_SPACE[w.index(max(w))] for w in final_w]
        ssc = 1.0 - edge_disagreement_rate(adj, dom)
        mesd = edge_strategy_distance(adj, final_w)

    return {
        "mean_player_weight_entropy": float(mh),
        "weight_heterogeneity_std": float(ws),
        "mean_q_value_std": float(qs),
        "mean_neighbor_weight_cosine": float(nwc),
        "spatial_strategy_clustering": float(ssc),
        "mean_edge_strategy_distance": float(mesd),
    }


# ---------------------------------------------------------------------------
# Summary builder
# ---------------------------------------------------------------------------

def _build_summary(
    *, condition: str, delta_rot: float, is_control: bool,
    metric_rows: list[dict[str, Any]],
    diag_rows: list[dict[str, float]], alpha_stds: list[float],
    cycle_stabs: list[float],
    out_dir: Path, players: int, rounds: int,
) -> dict[str, Any]:
    levels = [int(r["cycle_level"]) for r in metric_rows]
    lc = {lv: levels.count(lv) for lv in range(4)}
    den = float(len(metric_rows)) or 1.0
    return {
        "gate": G2_GATE, "condition": condition, "topology": TOPOLOGY,
        "beta_ceiling": _fmt(BETA_CEILING),
        "payoff_epsilon": _fmt(PAYOFF_EPSILON),
        "q_rotation_delta": _fmt(delta_rot),
        "alpha_lo": _fmt(ALPHA_LO), "alpha_hi": _fmt(ALPHA_HI),
        "is_control": _yn(is_control), "n_seeds": len(metric_rows),
        "mean_cycle_level":  _fmt(_mean([float(l) for l in levels])),
        "mean_stage3_score": _fmt(_mean([float(r["stage3_score"]) for r in metric_rows])),
        "mean_turn_strength": _fmt(_mean([float(r["turn_strength"]) for r in metric_rows])),
        "mean_env_gamma":    _fmt(_mean([float(r["env_gamma"]) for r in metric_rows])),
        "level_counts_json": json.dumps(lc, sort_keys=True),
        "p_level_3": _fmt(sum(1 for l in levels if l >= 3) / den),
        "level3_seed_count": sum(1 for l in levels if l >= 3),
        "realized_alpha_std": _fmt(_mean(alpha_stds)),
        "mean_player_weight_entropy": _fmt(_mean([float(d["mean_player_weight_entropy"]) for d in diag_rows])),
        "weight_heterogeneity_std":   _fmt(_mean([float(d["weight_heterogeneity_std"]) for d in diag_rows])),
        "mean_q_value_std":           _fmt(_mean([float(d["mean_q_value_std"]) for d in diag_rows])),
        "mean_neighbor_weight_cosine": _fmt(_mean([float(d["mean_neighbor_weight_cosine"]) for d in diag_rows])),
        "strategy_cycle_stability":    _fmt(_mean(cycle_stabs)),
        "spatial_strategy_clustering": _fmt(_mean([float(d["spatial_strategy_clustering"]) for d in diag_rows])),
        "mean_edge_strategy_distance": _fmt(_mean([float(d["mean_edge_strategy_distance"]) for d in diag_rows])),
        "short_scout_pass": "", "hard_stop_fail": "",
        "longer_confirm_candidate": "no",
        "verdict": "control" if is_control else "pending",
        "representative_seed": "",
        "representative_simplex_png": "",
        "representative_phase_amplitude_png": "",
        "players": int(players), "rounds": int(rounds), "out_dir": str(out_dir),
    }


def _rep_seed(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return max(rows, key=lambda r: (int(r["cycle_level"]), float(r["stage3_score"])))


# ---------------------------------------------------------------------------
# Decision writer
# ---------------------------------------------------------------------------

def _write_decision(
    path: Path, *, combined: list[dict[str, Any]],
    candidates: list[dict[str, Any]], all_fail: bool,
) -> None:
    lines = ["# P20 Q-Rotation Bias Decision", "",
             "## G2 Short Scout", ""]
    for r in combined:
        lines.append(
            f"- {r['condition']}: δ_rot={r['q_rotation_delta']}"
            f" level3={r['level3_seed_count']}"
            f" entropy={r['mean_player_weight_entropy']}"
            f" q_std={r['mean_q_value_std']}"
            f" stab={r['strategy_cycle_stability']}"
            f" s3={r['mean_stage3_score']}"
            f" turn={r['mean_turn_strength']}"
            f" pass={r['short_scout_pass']} verdict={r['verdict']}"
        )
    lines += ["", "## Recommendation", ""]
    if not candidates:
        lines.append("- longer_confirm_candidate: none")
    else:
        for r in candidates:
            lines.append(
                f"- longer_confirm_candidate: {r['condition']}"
                f" s3={r['mean_stage3_score']}"
                f" turn={r['mean_turn_strength']}"
                f" q_std={r['mean_q_value_std']}"
            )
    lines += ["", "## Pass Gate (revised, 4-way AND)", ""]
    lines += [
        f"1. level3_seed_count >= 2",
        f"2. mean_env_gamma >= 0",
        f"3. mean_player_weight_entropy >= {GATE_ENTROPY}",
        f"4. mean_q_value_std > {GATE_Q_STD}",
    ]
    lines += ["", "## BL1 Comparison Target", ""]
    lines += [
        "- BL1 anchor: L3=4/6, s3=0.863, turn=0.032, q_std=0.046, stab=0.840",
        "- Any improvement in s3 or turn_strength above BL1 is a positive signal.",
    ]
    lines += ["", "## Invariant Check", ""]
    lines += [
        "- δ_rot=0.0 control MUST match BL1 numerically (same seed → same output).",
        "- Q-rotation bias does NOT change payoff matrix or learning rates.",
    ]
    lines += ["", "## Stop Rule", ""]
    lines.append("- If ALL active conditions have L3=0, P20 closes.")
    lines.append(f"- overall_verdict: {'close_p20' if all_fail else 'keep_p20_open'}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# Build conditions
# ---------------------------------------------------------------------------

def _build_conditions(*, delta_rots: list[float]) -> list[dict[str, Any]]:
    conds: list[dict[str, Any]] = []
    for dr in delta_rots:
        if dr == 0.0:
            continue  # control is added automatically
        conds.append({
            "delta_rot": float(dr),
            "is_control": False,
        })
    return conds


# ---------------------------------------------------------------------------
# Main harness
# ---------------------------------------------------------------------------

def run_p20_scout(
    *,
    seeds: list[int],
    conditions: list[dict[str, Any]],
    out_root: Path,
    summary_tsv: Path,
    combined_tsv: Path,
    decision_md: Path,
    players: int = 300,
    rounds: int = 6000,
    burn_in: int = 2000,
    tail: int = 2000,
    a: float = 1.0,
    b: float = 0.9,
    cross: float = 0.20,
    init_q: float = 0.0,
    eta: float = 0.55,
    corr_threshold: float = 0.09,
) -> dict[str, Any]:
    out_root.mkdir(parents=True, exist_ok=True)
    gs = _make_graph_spec(TOPOLOGY)

    # Add BL1 control (δ_rot=0.0)
    all_conditions = [
        {"delta_rot": 0.0, "is_control": True},
    ] + list(conditions)

    all_summary: list[dict[str, Any]] = []
    all_combined: list[dict[str, Any]] = []
    cond_seed_rows: dict[str, list[dict[str, Any]]] = {}
    cond_ts: dict[str, dict[int, list[dict[str, Any]]]] = {}

    for cond in all_conditions:
        dr = float(cond["delta_rot"])
        is_ctrl = bool(cond["is_control"])

        cn = _cond_name(delta_rot=dr)
        od = out_root / cn
        od.mkdir(parents=True, exist_ok=True)

        per_seed: list[dict[str, Any]] = []
        metrics: list[dict[str, Any]] = []
        diags: list[dict[str, float]] = []
        alpha_stds: list[float] = []
        cycle_stabs: list[float] = []
        seed_ts: dict[int, list[dict[str, Any]]] = {}

        for seed in seeds:
            print(f"  P20 {cn} seed={seed} ...")
            rows, rdiag, fq, adj_out, p_alphas, p_betas = _run_e1_simulation(
                n_players=int(players), n_rounds=int(rounds), seed=int(seed),
                graph_spec=gs,
                alpha_lo=ALPHA_LO, alpha_hi=ALPHA_HI,
                beta_lo=BETA_CEILING, beta_hi=BETA_CEILING,
                a=float(a), b=float(b), cross=float(cross),
                init_q=float(init_q),
                entropy_lambda=0.0,
                payoff_epsilon=PAYOFF_EPSILON,
                q_rotation_delta=dr,
            )
            seed_ts[int(seed)] = rows

            csv_path = od / f"seed{seed}.csv"
            _write_csv(csv_path, rows=rows)

            sm = _seed_metrics(rows, burn_in=burn_in, tail=tail,
                               eta=eta, corr_threshold=corr_threshold)
            td = _tail_diagnostics(
                rdiag, fq, adj_out, p_betas,
                n_rows=len(rows), burn_in=burn_in, tail=tail,
            )

            scs = _strategy_cycle_stability(rows, burn_in=burn_in, tail=tail)
            a_std = _std(p_alphas)

            prov = {
                "condition": cn, "topology": TOPOLOGY,
                "beta_ceiling": BETA_CEILING,
                "payoff_epsilon": PAYOFF_EPSILON,
                "q_rotation_delta": dr,
                "alpha_lo": ALPHA_LO, "alpha_hi": ALPHA_HI,
                "seed": int(seed),
                "config": {
                    "players": int(players), "rounds": int(rounds),
                    "burn_in": int(burn_in), "tail": int(tail),
                    "a": float(a), "b": float(b), "cross": float(cross),
                    "init_q": float(init_q),
                },
                "realized_alpha_std": float(a_std),
                "cycle_metrics": dict(sm),
                "e1_diagnostics": {**dict(td), "strategy_cycle_stability": scs},
            }
            pj = od / f"seed{seed}_provenance.json"
            pj.write_text(json.dumps(prov, ensure_ascii=False, indent=2) + "\n",
                          encoding="utf-8")

            seed_row = {
                "gate": G2_GATE, "condition": cn, "topology": TOPOLOGY,
                "beta_ceiling": _fmt(BETA_CEILING),
                "payoff_epsilon": _fmt(PAYOFF_EPSILON),
                "q_rotation_delta": _fmt(dr),
                "alpha_lo": _fmt(ALPHA_LO), "alpha_hi": _fmt(ALPHA_HI),
                "seed": int(seed),
                "realized_alpha_std": _fmt(a_std),
                "cycle_level": int(sm["cycle_level"]),
                "stage3_score": _fmt(sm["stage3_score"]),
                "turn_strength": _fmt(sm["turn_strength"]),
                "env_gamma": _fmt(sm["env_gamma"]),
                "env_gamma_r2": _fmt(sm["env_gamma_r2"]),
                "env_gamma_n_peaks": int(sm["env_gamma_n_peaks"]),
                "mean_player_weight_entropy": _fmt(td["mean_player_weight_entropy"]),
                "weight_heterogeneity_std": _fmt(td["weight_heterogeneity_std"]),
                "mean_q_value_std": _fmt(td["mean_q_value_std"]),
                "mean_neighbor_weight_cosine": _fmt(td["mean_neighbor_weight_cosine"]),
                "strategy_cycle_stability": _fmt(scs),
                "spatial_strategy_clustering": _fmt(td["spatial_strategy_clustering"]),
                "mean_edge_strategy_distance": _fmt(td["mean_edge_strategy_distance"]),
                "has_level3_seed": _yn(int(sm["cycle_level"]) >= 3),
                "out_csv": str(csv_path), "provenance_json": str(pj),
            }
            per_seed.append(seed_row)
            metrics.append(sm)
            diags.append(td)
            alpha_stds.append(a_std)
            cycle_stabs.append(scs)

        cond_seed_rows[cn] = per_seed
        cond_ts[cn] = seed_ts
        all_combined.append(_build_summary(
            condition=cn, delta_rot=dr, is_control=is_ctrl,
            metric_rows=metrics, diag_rows=diags,
            alpha_stds=alpha_stds, cycle_stabs=cycle_stabs,
            out_dir=od, players=players, rounds=rounds,
        ))

    # --- Verdicts (revised gate) ---
    for cr in all_combined:
        if str(cr["is_control"]) == "yes":
            continue
        l3  = int(cr["level3_seed_count"])
        eg  = float(cr["mean_env_gamma"])
        ent = float(cr["mean_player_weight_entropy"])
        qs  = float(cr["mean_q_value_std"])

        sp = l3 >= 2 and eg >= 0.0 and ent >= GATE_ENTROPY and qs > GATE_Q_STD
        hs = l3 == 0
        if sp:
            verdict = "pass"
        elif hs:
            verdict = "fail"
        else:
            verdict = "weak_positive"
        cr["short_scout_pass"] = _yn(sp)
        cr["hard_stop_fail"] = _yn(hs)
        cr["verdict"] = verdict
        cr["longer_confirm_candidate"] = _yn(sp)

    # --- Representative seeds + plots ---
    for cr in all_combined:
        cn = str(cr["condition"])
        rep = _rep_seed(cond_seed_rows[cn])
        rs = int(rep["seed"])
        od = Path(cr["out_dir"])
        sp_png = od / f"seed{rs}_simplex.png"
        pp_png = od / f"seed{rs}_phase_amplitude.png"
        _write_simplex_plot(
            cond_ts[cn][rs], out_png=sp_png,
            title=f"{cn} s={rs}", burn_in=burn_in, tail=tail)
        _write_phase_amplitude_plot(
            cond_ts[cn][rs], out_png=pp_png,
            title=f"{cn} s={rs}", burn_in=burn_in, tail=tail)
        cr["representative_seed"] = rs
        cr["representative_simplex_png"] = str(sp_png)
        cr["representative_phase_amplitude_png"] = str(pp_png)

    # --- Flatten per-seed for summary ---
    for _cn, ps in cond_seed_rows.items():
        all_summary.extend(ps)

    # --- Stop rule ---
    active = [r for r in all_combined if str(r["is_control"]) == "no"]
    all_fail = bool(active) and all(int(r["level3_seed_count"]) == 0 for r in active)

    candidates = [r for r in all_combined if str(r["verdict"]) == "pass"]
    candidates.sort(
        key=lambda r: (int(r["level3_seed_count"]),
                       float(r["mean_stage3_score"]),
                       float(r["mean_turn_strength"])),
        reverse=True)

    _write_tsv(summary_tsv, fieldnames=SUMMARY_FIELDNAMES, rows=all_summary)
    _write_tsv(combined_tsv, fieldnames=COMBINED_FIELDNAMES, rows=all_combined)
    _write_decision(decision_md, combined=all_combined,
                    candidates=candidates, all_fail=all_fail)

    print(f"\nsummary_tsv={summary_tsv}")
    print(f"combined_tsv={combined_tsv}")
    print(f"decision_md={decision_md}")
    print(f"all_fail={all_fail}")

    return {
        "summary_tsv": str(summary_tsv),
        "combined_tsv": str(combined_tsv),
        "decision_md": str(decision_md),
        "all_fail": bool(all_fail),
        "candidates": [dict(r) for r in candidates],
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(description="P20 Q-Rotation Bias Scout")
    p.add_argument("--delta-rots", type=str, default="0.001,0.005,0.01,0.02",
                   help="Comma-separated δ_rot values (control δ=0 added automatically)")
    p.add_argument("--seeds", type=str, default="45,47,49")
    p.add_argument("--players", type=int, default=300)
    p.add_argument("--rounds", type=int, default=6000)
    p.add_argument("--burn-in", type=int, default=2000)
    p.add_argument("--tail", type=int, default=2000)
    # Game params (BL1 locked)
    p.add_argument("--a", type=float, default=1.0)
    p.add_argument("--b", type=float, default=0.9)
    p.add_argument("--cross", type=float, default=0.20)
    p.add_argument("--init-q", type=float, default=0.0)
    p.add_argument("--eta", type=float, default=0.55)
    p.add_argument("--corr-threshold", type=float, default=0.09)
    # Output
    p.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    p.add_argument("--summary-tsv", type=Path, default=DEFAULT_SUMMARY_TSV)
    p.add_argument("--combined-tsv", type=Path, default=DEFAULT_COMBINED_TSV)
    p.add_argument("--decision-md", type=Path, default=DEFAULT_DECISION_MD)
    args = p.parse_args()

    conditions = _build_conditions(
        delta_rots=_parse_floats(args.delta_rots),
    )

    if not conditions:
        print("No conditions generated. Check --delta-rots.")
        return

    seeds = _parse_seeds(args.seeds)
    print(f"P20 Q-Rotation Bias: {len(conditions)} conditions × {len(seeds)} seeds + control")
    print(f"  δ_rot values: {[c['delta_rot'] for c in conditions]}")

    run_p20_scout(
        seeds=seeds,
        conditions=conditions,
        out_root=args.out_root, summary_tsv=args.summary_tsv,
        combined_tsv=args.combined_tsv, decision_md=args.decision_md,
        players=args.players, rounds=args.rounds,
        burn_in=args.burn_in, tail=args.tail,
        a=args.a, b=args.b, cross=args.cross, init_q=args.init_q,
        eta=args.eta, corr_threshold=args.corr_threshold,
    )


if __name__ == "__main__":
    main()
