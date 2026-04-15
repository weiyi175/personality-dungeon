"""F1v2 Payoff Perturbation — E2-best base + mild per-player payoff bias.

Phase 16 experiment: on the E2 β=3 + wide α base that already passes the
revised Pass Gate, apply mild per-player payoff perturbation to test whether
breaking payoff symmetry can further push q_std.

Key difference vs F1 (Phase 12, failed):
    F1:   α_lo=0.02, α_hi=0.08, β=5.0   → L3=0 (narrow α killed cycling)
    F1v2: α_lo=0.005, α_hi=0.40, β=3.0  → preserves E2-best foundation

Sweep: payoff_epsilon ∈ {0.00, 0.02, 0.04, 0.06, 0.08}
       × {lattice4, well_mixed} × seeds

Architecture
------------
- Thin wrapper around E1's core simulation (_run_e1_simulation).
- Uses CC1 fallback in cycle level computation.
- Revised Pass Gate: ent≥1.06, q_std>0.04, L3≥2, γ≥0.

Usage
-----
    ./venv/bin/python -m simulation.f1v2_payoff_perturbation \\
        --payoff-epsilons 0.00,0.02,0.04,0.06,0.08 \\
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

G2_GATE = "g2_f1v2_payoff_perturbation"

ALPHA_LO = 0.005
ALPHA_HI = 0.40
BETA_CEILING = 3.0

# CC1 fallback thresholds
CC1_R2_THR = 0.85
CC1_MIN_ROT = 20.0

# Revised Pass Gate thresholds (Phase 15 approved)
GATE_ENTROPY = 1.06
GATE_Q_STD = 0.04

DEFAULT_OUT_ROOT = Path("outputs/f1v2_payoff_perturbation")
DEFAULT_SUMMARY_TSV = DEFAULT_OUT_ROOT / "f1v2_summary.tsv"
DEFAULT_COMBINED_TSV = DEFAULT_OUT_ROOT / "f1v2_combined.tsv"
DEFAULT_DECISION_MD = DEFAULT_OUT_ROOT / "f1v2_decision.md"

SUMMARY_FIELDNAMES = [
    "gate", "condition", "topology", "beta_ceiling", "payoff_epsilon",
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
    "gate", "condition", "topology", "beta_ceiling", "payoff_epsilon",
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

def _parse_strs(s: str) -> list[str]:
    return [x.strip() for x in s.split(",") if x.strip()]

def _parse_floats(s: str) -> list[float]:
    return [float(x.strip()) for x in s.split(",") if x.strip()]

def _write_tsv(path: Path, *, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t", extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def _cond_name(*, topology: str, payoff_epsilon: float) -> str:
    def _tok(v: float) -> str:
        return f"{v:.2f}".rstrip("0").rstrip(".").replace(".", "p")
    return f"g2_f1v2_{topology}_eps{_tok(payoff_epsilon)}"


# ---------------------------------------------------------------------------
# Strategy cycle stability (R² of phase angle linear trend)
# ---------------------------------------------------------------------------

def _phase_angle(pa: float, pd: float, pb: float) -> float:
    return float(atan2(sqrt(3.0) * (pd - pb), 2.0 * pa - pd - pb))


def _strategy_cycle_stability(rows: list[dict[str, Any]], *, burn_in: int, tail: int) -> float:
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
        "cycle_level":     int(cyc.level),
        "stage3_score":    float(cyc.stage3.score) if cyc.stage3 else 0.0,
        "turn_strength":   float(cyc.stage3.turn_strength) if cyc.stage3 else 0.0,
        "env_gamma":       float(fit.gamma) if fit else 0.0,
        "env_gamma_r2":    float(fit.r2) if fit else 0.0,
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
        from evolution.independent_rl import cosine_similarity as _cos
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
    *, condition: str, topology: str,
    payoff_epsilon: float, is_control: bool,
    metric_rows: list[dict[str, Any]],
    diag_rows: list[dict[str, float]], alpha_stds: list[float],
    cycle_stabs: list[float],
    out_dir: Path, players: int, rounds: int,
) -> dict[str, Any]:
    levels = [int(r["cycle_level"]) for r in metric_rows]
    lc = {lv: levels.count(lv) for lv in range(4)}
    den = float(len(metric_rows)) or 1.0
    return {
        "gate": G2_GATE, "condition": condition, "topology": topology,
        "beta_ceiling": _fmt(BETA_CEILING),
        "payoff_epsilon": _fmt(payoff_epsilon),
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
    candidates: list[dict[str, Any]], close_f1v2: bool,
) -> None:
    lines = ["# F1v2 Payoff Perturbation Decision", "", "## G2 Short Scout", ""]
    for r in combined:
        lines.append(
            f"- {r['condition']}: ε={r['payoff_epsilon']}"
            f" level3={r['level3_seed_count']}"
            f" entropy={r['mean_player_weight_entropy']}"
            f" q_std={r['mean_q_value_std']}"
            f" w_std={r['weight_heterogeneity_std']}"
            f" α_std={r['realized_alpha_std']}"
            f" cycle_stab={r['strategy_cycle_stability']}"
            f" pass={r['short_scout_pass']} verdict={r['verdict']}"
        )
    lines += ["", "## Recommendation", ""]
    if not candidates:
        lines.append("- longer_confirm_candidate: none")
    else:
        for r in candidates:
            lines.append(
                f"- longer_confirm_candidate: {r['condition']}"
                f" entropy={r['mean_player_weight_entropy']}"
                f" q_std={r['mean_q_value_std']}"
            )
    lines += ["", "## Pass Gate (revised, 4-way AND)", ""]
    lines += [
        f"1. level3_seed_count >= 2",
        f"2. mean_env_gamma >= 0",
        f"3. mean_player_weight_entropy >= {GATE_ENTROPY}",
        f"4. mean_q_value_std > {GATE_Q_STD}",
    ]
    lines += ["", "## Stop Rule", ""]
    lines.append("- If ALL active conditions (ε>0) have L3=0, F1v2 closes.")
    lines.append(f"- overall_verdict: {'close_f1v2' if close_f1v2 else 'keep_f1v2_open'}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# Main harness
# ---------------------------------------------------------------------------

def run_f1v2_scout(
    *,
    seeds: list[int],
    topologies: list[str],
    payoff_epsilons: list[float],
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

    # --- Build conditions ---
    conditions: list[dict[str, Any]] = []
    for topo in topologies:
        gs = _make_graph_spec(topo)
        for eps in payoff_epsilons:
            conditions.append({
                "condition": _cond_name(topology=topo, payoff_epsilon=eps),
                "topology": topo, "graph_spec": gs,
                "payoff_epsilon": float(eps),
                "is_control": float(eps) == 0.0,
            })

    all_summary: list[dict[str, Any]] = []
    all_combined: list[dict[str, Any]] = []
    cond_seed_rows: dict[str, list[dict[str, Any]]] = {}
    cond_ts: dict[str, dict[int, list[dict[str, Any]]]] = {}

    for cond in conditions:
        cn = str(cond["condition"])
        gs = cond["graph_spec"]
        eps = float(cond["payoff_epsilon"])
        topo = str(cond["topology"])
        is_ctrl = bool(cond["is_control"])
        od = out_root / cn
        od.mkdir(parents=True, exist_ok=True)

        per_seed: list[dict[str, Any]] = []
        metrics: list[dict[str, Any]] = []
        diags: list[dict[str, float]] = []
        alpha_stds: list[float] = []
        cycle_stabs: list[float] = []
        seed_ts: dict[int, list[dict[str, Any]]] = {}

        for seed in seeds:
            print(f"  F1v2 {cn} seed={seed} ...")
            rows, rdiag, fq, adj_out, p_alphas, p_betas = _run_e1_simulation(
                n_players=int(players), n_rounds=int(rounds), seed=int(seed),
                graph_spec=gs,
                alpha_lo=ALPHA_LO, alpha_hi=ALPHA_HI,
                beta_lo=BETA_CEILING, beta_hi=BETA_CEILING,  # homo β=3
                a=float(a), b=float(b), cross=float(cross),
                init_q=float(init_q),
                entropy_lambda=0.0,       # no G1 Q-centering
                payoff_epsilon=eps,
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
                "condition": cn, "topology": topo,
                "beta_ceiling": BETA_CEILING,
                "payoff_epsilon": eps,
                "alpha_lo": ALPHA_LO, "alpha_hi": ALPHA_HI,
                "seed": int(seed),
                "config": {
                    "players": int(players), "rounds": int(rounds),
                    "burn_in": int(burn_in), "tail": int(tail),
                    "a": float(a), "b": float(b), "cross": float(cross),
                    "init_q": float(init_q),
                    "payoff_epsilon": eps,
                },
                "realized_alpha_std": float(a_std),
                "cycle_metrics": dict(sm),
                "e1_diagnostics": {**dict(td), "strategy_cycle_stability": scs},
            }
            pj = od / f"seed{seed}_provenance.json"
            pj.write_text(json.dumps(prov, ensure_ascii=False, indent=2) + "\n",
                          encoding="utf-8")

            seed_row = {
                "gate": G2_GATE, "condition": cn, "topology": topo,
                "beta_ceiling": _fmt(BETA_CEILING),
                "payoff_epsilon": _fmt(eps),
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
            condition=cn, topology=topo,
            payoff_epsilon=eps, is_control=is_ctrl,
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
    close_f1v2 = bool(active) and all(int(r["level3_seed_count"]) == 0 for r in active)

    candidates = [r for r in all_combined if str(r["verdict"]) == "pass"]
    candidates.sort(
        key=lambda r: (int(r["level3_seed_count"]), float(r["mean_q_value_std"])),
        reverse=True)

    _write_tsv(summary_tsv, fieldnames=SUMMARY_FIELDNAMES, rows=all_summary)
    _write_tsv(combined_tsv, fieldnames=COMBINED_FIELDNAMES, rows=all_combined)
    _write_decision(decision_md, combined=all_combined,
                    candidates=candidates, close_f1v2=close_f1v2)

    print(f"summary_tsv={summary_tsv}")
    print(f"combined_tsv={combined_tsv}")
    print(f"decision_md={decision_md}")
    print(f"close_f1v2={close_f1v2}")

    return {
        "summary_tsv": str(summary_tsv),
        "combined_tsv": str(combined_tsv),
        "decision_md": str(decision_md),
        "close_f1v2": bool(close_f1v2),
        "candidates": [dict(r) for r in candidates],
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(description="F1v2 Payoff Perturbation G2 Scout")
    p.add_argument("--seeds", type=str, default="45,47,49")
    p.add_argument("--graph-topologies", type=str, default="lattice4,well_mixed")
    p.add_argument("--payoff-epsilons", type=str, default="0.00,0.02,0.04,0.06,0.08")
    p.add_argument("--players", type=int, default=300)
    p.add_argument("--rounds", type=int, default=6000)
    p.add_argument("--burn-in", type=int, default=2000)
    p.add_argument("--tail", type=int, default=2000)
    p.add_argument("--a", type=float, default=1.0)
    p.add_argument("--b", type=float, default=0.9)
    p.add_argument("--cross", type=float, default=0.20)
    p.add_argument("--init-q", type=float, default=0.0)
    p.add_argument("--eta", type=float, default=0.55)
    p.add_argument("--corr-threshold", type=float, default=0.09)
    p.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    p.add_argument("--summary-tsv", type=Path, default=DEFAULT_SUMMARY_TSV)
    p.add_argument("--combined-tsv", type=Path, default=DEFAULT_COMBINED_TSV)
    p.add_argument("--decision-md", type=Path, default=DEFAULT_DECISION_MD)
    args = p.parse_args()

    run_f1v2_scout(
        seeds=_parse_seeds(args.seeds),
        topologies=_parse_strs(args.graph_topologies),
        payoff_epsilons=_parse_floats(args.payoff_epsilons),
        out_root=args.out_root, summary_tsv=args.summary_tsv,
        combined_tsv=args.combined_tsv, decision_md=args.decision_md,
        players=args.players, rounds=args.rounds,
        burn_in=args.burn_in, tail=args.tail,
        a=args.a, b=args.b, cross=args.cross, init_q=args.init_q,
        eta=args.eta, corr_threshold=args.corr_threshold,
    )


if __name__ == "__main__":
    main()
