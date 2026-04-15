"""P18×P21 combo scout — dynamic epsilon plus asymmetric alpha.

Combine the strongest Phase 18a champion and Phase 21 champion on the
same E1 pipeline to test whether payoff-side annealing and strategy-side
learning asymmetry stack constructively.

Conditions:
    control      : BL1 static epsilon, symmetric alpha
    p18_only     : epsilon_start=0.04 -> epsilon_end=0.01, R=1000
    p21_only     : mild_cw multipliers [1.2, 1.0, 0.8]
    combo        : p18_only + p21_only

Usage
-----
    ./venv/bin/python -m simulation.p18_p21_combo \
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
from evolution.independent_rl import STRATEGY_SPACE, boltzmann_weights
from simulation.e1_heterogeneous_rl import (
    _fmt,
    _make_graph_spec,
    _mean,
    _run_e1_simulation,
    _std,
    _write_csv,
    _write_phase_amplitude_plot,
    _write_simplex_plot,
    _yn,
)


G2_GATE = "g2_p18_p21_combo"

ALPHA_LO = 0.005
ALPHA_HI = 0.40
BETA_CEILING = 3.0
TOPOLOGY = "well_mixed"
PAYOFF_EPSILON = 0.02

P18_EPSILON_START = 0.04
P18_EPSILON_END = 0.01
P18_RESAMPLE_INTERVAL = 1000
P21_MILD_CW = [1.2, 1.0, 0.8]

CC1_R2_THR = 0.85
CC1_MIN_ROT = 20.0

GATE_ENTROPY = 1.06
GATE_Q_STD = 0.04

DEFAULT_OUT_ROOT = Path("outputs/p18_p21_combo")
DEFAULT_SUMMARY_TSV = DEFAULT_OUT_ROOT / "combo_summary.tsv"
DEFAULT_COMBINED_TSV = DEFAULT_OUT_ROOT / "combo_combined.tsv"
DEFAULT_DECISION_MD = DEFAULT_OUT_ROOT / "combo_decision.md"

SUMMARY_FIELDNAMES = [
    "gate", "condition", "topology", "beta_ceiling",
    "payoff_epsilon", "epsilon_end", "resample_interval",
    "alpha_mult_config", "r_A", "r_D", "r_B",
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
    "payoff_epsilon", "epsilon_end", "resample_interval",
    "alpha_mult_config", "r_A", "r_D", "r_B",
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


def _parse_seeds(s: str) -> list[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def _write_tsv(path: Path, *, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _cond_name(*, config_name: str) -> str:
    return f"g2_combo_{TOPOLOGY}_{config_name}"


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
    phases_raw = [
        _phase_angle(float(r["p_aggressive"]), float(r["p_defensive"]), float(r["p_balanced"]))
        for r in window
    ]
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
    return max(0.0, float(1.0 - ss_res / ss_tot))


def _seed_metrics(
    rows: list[dict[str, Any]], *, burn_in: int, tail: int,
    eta: float, corr_threshold: float,
) -> dict[str, float | int]:
    sm = {
        "aggressive": [float(r["p_aggressive"]) for r in rows],
        "defensive": [float(r["p_defensive"]) for r in rows],
        "balanced": [float(r["p_balanced"]) for r in rows],
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
        "cycle_level": int(cyc.level),
        "stage3_score": float(cyc.stage3.score) if cyc.stage3 else 0.0,
        "turn_strength": float(cyc.stage3.turn_strength) if cyc.stage3 else 0.0,
        "env_gamma": float(fit.gamma) if fit else 0.0,
        "env_gamma_r2": float(fit.r2) if fit else 0.0,
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

    mh = _mean([float(d["mean_player_weight_entropy"]) for d in window])
    ws = _mean([float(d["weight_heterogeneity_std"]) for d in window])
    qs = _mean([float(d["mean_q_value_std"]) for d in window])
    nwc = _mean([float(d["mean_neighbor_weight_cosine"]) for d in window])

    ssc = 0.0
    mesd = 0.0
    if adj is not None:
        from evolution.local_graph import edge_disagreement_rate, edge_strategy_distance

        final_w = [boltzmann_weights(q, beta=player_betas[i]) for i, q in enumerate(final_q)]
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


def _build_summary(
    *, condition: str, payoff_epsilon: float, epsilon_end: float | None,
    resample_interval: int, alpha_mult_config: str, multipliers: list[float],
    is_control: bool, metric_rows: list[dict[str, Any]],
    diag_rows: list[dict[str, float]], alpha_stds: list[float],
    cycle_stabs: list[float], out_dir: Path, players: int, rounds: int,
) -> dict[str, Any]:
    levels = [int(r["cycle_level"]) for r in metric_rows]
    lc = {lv: levels.count(lv) for lv in range(4)}
    den = float(len(metric_rows)) or 1.0
    return {
        "gate": G2_GATE,
        "condition": condition,
        "topology": TOPOLOGY,
        "beta_ceiling": _fmt(BETA_CEILING),
        "payoff_epsilon": _fmt(payoff_epsilon),
        "epsilon_end": _fmt(epsilon_end if epsilon_end is not None else payoff_epsilon),
        "resample_interval": int(resample_interval),
        "alpha_mult_config": alpha_mult_config,
        "r_A": _fmt(multipliers[0]),
        "r_D": _fmt(multipliers[1]),
        "r_B": _fmt(multipliers[2]),
        "alpha_lo": _fmt(ALPHA_LO),
        "alpha_hi": _fmt(ALPHA_HI),
        "is_control": _yn(is_control),
        "n_seeds": len(metric_rows),
        "mean_cycle_level": _fmt(_mean([float(l) for l in levels])),
        "mean_stage3_score": _fmt(_mean([float(r["stage3_score"]) for r in metric_rows])),
        "mean_turn_strength": _fmt(_mean([float(r["turn_strength"]) for r in metric_rows])),
        "mean_env_gamma": _fmt(_mean([float(r["env_gamma"]) for r in metric_rows])),
        "level_counts_json": json.dumps(lc, sort_keys=True),
        "p_level_3": _fmt(sum(1 for l in levels if l >= 3) / den),
        "level3_seed_count": sum(1 for l in levels if l >= 3),
        "realized_alpha_std": _fmt(_mean(alpha_stds)),
        "mean_player_weight_entropy": _fmt(_mean([float(d["mean_player_weight_entropy"]) for d in diag_rows])),
        "weight_heterogeneity_std": _fmt(_mean([float(d["weight_heterogeneity_std"]) for d in diag_rows])),
        "mean_q_value_std": _fmt(_mean([float(d["mean_q_value_std"]) for d in diag_rows])),
        "mean_neighbor_weight_cosine": _fmt(_mean([float(d["mean_neighbor_weight_cosine"]) for d in diag_rows])),
        "strategy_cycle_stability": _fmt(_mean(cycle_stabs)),
        "spatial_strategy_clustering": _fmt(_mean([float(d["spatial_strategy_clustering"]) for d in diag_rows])),
        "mean_edge_strategy_distance": _fmt(_mean([float(d["mean_edge_strategy_distance"]) for d in diag_rows])),
        "short_scout_pass": "",
        "hard_stop_fail": "",
        "longer_confirm_candidate": "no",
        "verdict": "control" if is_control else "pending",
        "representative_seed": "",
        "representative_simplex_png": "",
        "representative_phase_amplitude_png": "",
        "players": int(players),
        "rounds": int(rounds),
        "out_dir": str(out_dir),
    }


def _rep_seed(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return max(rows, key=lambda r: (int(r["cycle_level"]), float(r["stage3_score"])))


def _write_decision(
    path: Path, *, combined: list[dict[str, Any]],
    candidates: list[dict[str, Any]], all_fail: bool,
) -> None:
    lines = ["# P18×P21 Combo Decision", "", "## G2 Short Scout", ""]
    for row in combined:
        lines.append(
            f"- {row['condition']}: ε_start={row['payoff_epsilon']}"
            f" ε_end={row['epsilon_end']} R={row['resample_interval']}"
            f" α=[{row['r_A']},{row['r_D']},{row['r_B']}]"
            f" level3={row['level3_seed_count']}"
            f" entropy={row['mean_player_weight_entropy']}"
            f" q_std={row['mean_q_value_std']}"
            f" stab={row['strategy_cycle_stability']}"
            f" s3={row['mean_stage3_score']}"
            f" turn={row['mean_turn_strength']}"
            f" pass={row['short_scout_pass']} verdict={row['verdict']}"
        )
    lines += ["", "## Recommendation", ""]
    if not candidates:
        lines.append("- longer_confirm_candidate: none")
    else:
        for row in candidates:
            lines.append(
                f"- longer_confirm_candidate: {row['condition']}"
                f" s3={row['mean_stage3_score']}"
                f" turn={row['mean_turn_strength']}"
                f" q_std={row['mean_q_value_std']}"
            )
    lines += ["", "## Pass Gate (revised, 4-way AND)", ""]
    lines += [
        "1. level3_seed_count >= 2",
        "2. mean_env_gamma >= 0",
        f"3. mean_player_weight_entropy >= {GATE_ENTROPY}",
        f"4. mean_q_value_std > {GATE_Q_STD}",
    ]
    lines += ["", "## Comparison Target", ""]
    lines += [
        "- BL1 anchor: L3=4/6, s3=0.863, turn=0.032, q_std=0.046, stab=0.840",
        "- P18a extended: L3=5/6, s3=0.734, turn=0.026, stab=0.926",
        "- P21a extended: L3=6/6, s3=0.940, turn=0.054, stab=0.940",
    ]
    lines += ["", "## Stop Rule", ""]
    lines.append("- If combo does not beat both single-mechanism conditions, do not extend it.")
    lines.append(f"- overall_verdict: {'close_combo' if all_fail else 'keep_combo_open'}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _default_conditions() -> list[dict[str, Any]]:
    return [
        {
            "config_name": "control",
            "payoff_epsilon": PAYOFF_EPSILON,
            "epsilon_end": None,
            "resample_interval": 0,
            "multipliers": [1.0, 1.0, 1.0],
            "alpha_mult_config": "symmetric",
            "is_control": True,
        },
        {
            "config_name": "p18_only",
            "payoff_epsilon": P18_EPSILON_START,
            "epsilon_end": P18_EPSILON_END,
            "resample_interval": P18_RESAMPLE_INTERVAL,
            "multipliers": [1.0, 1.0, 1.0],
            "alpha_mult_config": "symmetric",
            "is_control": False,
        },
        {
            "config_name": "p21_only",
            "payoff_epsilon": PAYOFF_EPSILON,
            "epsilon_end": None,
            "resample_interval": 0,
            "multipliers": list(P21_MILD_CW),
            "alpha_mult_config": "mild_cw",
            "is_control": False,
        },
        {
            "config_name": "combo",
            "payoff_epsilon": P18_EPSILON_START,
            "epsilon_end": P18_EPSILON_END,
            "resample_interval": P18_RESAMPLE_INTERVAL,
            "multipliers": list(P21_MILD_CW),
            "alpha_mult_config": "mild_cw",
            "is_control": False,
        },
    ]


def run_combo_scout(
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
    graph_spec = _make_graph_spec(TOPOLOGY)

    all_summary: list[dict[str, Any]] = []
    all_combined: list[dict[str, Any]] = []
    cond_seed_rows: dict[str, list[dict[str, Any]]] = {}
    cond_ts: dict[str, dict[int, list[dict[str, Any]]]] = {}

    for cond in conditions:
        cfg_name = str(cond["config_name"])
        payoff_epsilon = float(cond["payoff_epsilon"])
        epsilon_end = cond["epsilon_end"]
        resample_interval = int(cond["resample_interval"])
        multipliers = [float(m) for m in cond["multipliers"]]
        alpha_mult_config = str(cond["alpha_mult_config"])
        is_control = bool(cond["is_control"])

        condition_name = _cond_name(config_name=cfg_name)
        out_dir = out_root / condition_name
        out_dir.mkdir(parents=True, exist_ok=True)

        per_seed: list[dict[str, Any]] = []
        metrics: list[dict[str, Any]] = []
        diags: list[dict[str, float]] = []
        alpha_stds: list[float] = []
        cycle_stabs: list[float] = []
        seed_ts: dict[int, list[dict[str, Any]]] = {}

        for seed in seeds:
            print(f"  COMBO {condition_name} seed={seed} ...")
            rows, rdiag, fq, adj_out, p_alphas, p_betas = _run_e1_simulation(
                n_players=int(players),
                n_rounds=int(rounds),
                seed=int(seed),
                graph_spec=graph_spec,
                alpha_lo=ALPHA_LO,
                alpha_hi=ALPHA_HI,
                beta_lo=BETA_CEILING,
                beta_hi=BETA_CEILING,
                a=float(a),
                b=float(b),
                cross=float(cross),
                init_q=float(init_q),
                entropy_lambda=0.0,
                payoff_epsilon=payoff_epsilon,
                epsilon_resample_interval=resample_interval,
                epsilon_end=float(epsilon_end) if epsilon_end is not None else None,
                strategy_alpha_multipliers=None if alpha_mult_config == "symmetric" else multipliers,
            )
            seed_ts[int(seed)] = rows

            csv_path = out_dir / f"seed{seed}.csv"
            _write_csv(csv_path, rows=rows)

            seed_metrics = _seed_metrics(
                rows, burn_in=burn_in, tail=tail,
                eta=eta, corr_threshold=corr_threshold,
            )
            tail_diag = _tail_diagnostics(
                rdiag, fq, adj_out, p_betas,
                n_rows=len(rows), burn_in=burn_in, tail=tail,
            )
            cycle_stability = _strategy_cycle_stability(rows, burn_in=burn_in, tail=tail)
            alpha_std = _std(p_alphas)

            provenance = {
                "condition": condition_name,
                "topology": TOPOLOGY,
                "beta_ceiling": BETA_CEILING,
                "payoff_epsilon": payoff_epsilon,
                "epsilon_end": epsilon_end,
                "resample_interval": resample_interval,
                "alpha_mult_config": alpha_mult_config,
                "multipliers": multipliers,
                "alpha_lo": ALPHA_LO,
                "alpha_hi": ALPHA_HI,
                "seed": int(seed),
                "config": {
                    "players": int(players),
                    "rounds": int(rounds),
                    "burn_in": int(burn_in),
                    "tail": int(tail),
                    "a": float(a),
                    "b": float(b),
                    "cross": float(cross),
                    "init_q": float(init_q),
                },
                "realized_alpha_std": float(alpha_std),
                "cycle_metrics": dict(seed_metrics),
                "e1_diagnostics": {**dict(tail_diag), "strategy_cycle_stability": cycle_stability},
            }
            provenance_path = out_dir / f"seed{seed}_provenance.json"
            provenance_path.write_text(
                json.dumps(provenance, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )

            seed_row = {
                "gate": G2_GATE,
                "condition": condition_name,
                "topology": TOPOLOGY,
                "beta_ceiling": _fmt(BETA_CEILING),
                "payoff_epsilon": _fmt(payoff_epsilon),
                "epsilon_end": _fmt(epsilon_end if epsilon_end is not None else payoff_epsilon),
                "resample_interval": int(resample_interval),
                "alpha_mult_config": alpha_mult_config,
                "r_A": _fmt(multipliers[0]),
                "r_D": _fmt(multipliers[1]),
                "r_B": _fmt(multipliers[2]),
                "alpha_lo": _fmt(ALPHA_LO),
                "alpha_hi": _fmt(ALPHA_HI),
                "seed": int(seed),
                "realized_alpha_std": _fmt(alpha_std),
                "cycle_level": int(seed_metrics["cycle_level"]),
                "stage3_score": _fmt(seed_metrics["stage3_score"]),
                "turn_strength": _fmt(seed_metrics["turn_strength"]),
                "env_gamma": _fmt(seed_metrics["env_gamma"]),
                "env_gamma_r2": _fmt(seed_metrics["env_gamma_r2"]),
                "env_gamma_n_peaks": int(seed_metrics["env_gamma_n_peaks"]),
                "mean_player_weight_entropy": _fmt(tail_diag["mean_player_weight_entropy"]),
                "weight_heterogeneity_std": _fmt(tail_diag["weight_heterogeneity_std"]),
                "mean_q_value_std": _fmt(tail_diag["mean_q_value_std"]),
                "mean_neighbor_weight_cosine": _fmt(tail_diag["mean_neighbor_weight_cosine"]),
                "strategy_cycle_stability": _fmt(cycle_stability),
                "spatial_strategy_clustering": _fmt(tail_diag["spatial_strategy_clustering"]),
                "mean_edge_strategy_distance": _fmt(tail_diag["mean_edge_strategy_distance"]),
                "has_level3_seed": _yn(int(seed_metrics["cycle_level"]) >= 3),
                "out_csv": str(csv_path),
                "provenance_json": str(provenance_path),
            }
            per_seed.append(seed_row)
            metrics.append(seed_metrics)
            diags.append(tail_diag)
            alpha_stds.append(alpha_std)
            cycle_stabs.append(cycle_stability)

        cond_seed_rows[condition_name] = per_seed
        cond_ts[condition_name] = seed_ts
        all_combined.append(
            _build_summary(
                condition=condition_name,
                payoff_epsilon=payoff_epsilon,
                epsilon_end=float(epsilon_end) if epsilon_end is not None else None,
                resample_interval=resample_interval,
                alpha_mult_config=alpha_mult_config,
                multipliers=multipliers,
                is_control=is_control,
                metric_rows=metrics,
                diag_rows=diags,
                alpha_stds=alpha_stds,
                cycle_stabs=cycle_stabs,
                out_dir=out_dir,
                players=players,
                rounds=rounds,
            )
        )

    for combined_row in all_combined:
        if str(combined_row["is_control"]) == "yes":
            continue
        level3_count = int(combined_row["level3_seed_count"])
        env_gamma = float(combined_row["mean_env_gamma"])
        entropy = float(combined_row["mean_player_weight_entropy"])
        q_std = float(combined_row["mean_q_value_std"])
        short_pass = level3_count >= 2 and env_gamma >= 0.0 and entropy >= GATE_ENTROPY and q_std > GATE_Q_STD
        hard_stop = level3_count == 0
        if short_pass:
            verdict = "pass"
        elif hard_stop:
            verdict = "fail"
        else:
            verdict = "weak_positive"
        combined_row["short_scout_pass"] = _yn(short_pass)
        combined_row["hard_stop_fail"] = _yn(hard_stop)
        combined_row["verdict"] = verdict
        combined_row["longer_confirm_candidate"] = _yn(short_pass)

    for combined_row in all_combined:
        condition_name = str(combined_row["condition"])
        rep = _rep_seed(cond_seed_rows[condition_name])
        rep_seed = int(rep["seed"])
        out_dir = Path(combined_row["out_dir"])
        simplex_png = out_dir / f"seed{rep_seed}_simplex.png"
        phase_png = out_dir / f"seed{rep_seed}_phase_amplitude.png"
        _write_simplex_plot(
            cond_ts[condition_name][rep_seed],
            out_png=simplex_png,
            title=f"{condition_name} s={rep_seed}",
            burn_in=burn_in,
            tail=tail,
        )
        _write_phase_amplitude_plot(
            cond_ts[condition_name][rep_seed],
            out_png=phase_png,
            title=f"{condition_name} s={rep_seed}",
            burn_in=burn_in,
            tail=tail,
        )
        combined_row["representative_seed"] = rep_seed
        combined_row["representative_simplex_png"] = str(simplex_png)
        combined_row["representative_phase_amplitude_png"] = str(phase_png)

    for per_seed_rows in cond_seed_rows.values():
        all_summary.extend(per_seed_rows)

    active = [row for row in all_combined if str(row["is_control"]) == "no"]
    all_fail = bool(active) and all(int(row["level3_seed_count"]) == 0 for row in active)
    candidates = [row for row in all_combined if str(row["verdict"]) == "pass"]
    candidates.sort(
        key=lambda row: (
            int(row["level3_seed_count"]),
            float(row["mean_stage3_score"]),
            float(row["mean_turn_strength"]),
        ),
        reverse=True,
    )

    _write_tsv(summary_tsv, fieldnames=SUMMARY_FIELDNAMES, rows=all_summary)
    _write_tsv(combined_tsv, fieldnames=COMBINED_FIELDNAMES, rows=all_combined)
    _write_decision(decision_md, combined=all_combined, candidates=candidates, all_fail=all_fail)

    print(f"\nsummary_tsv={summary_tsv}")
    print(f"combined_tsv={combined_tsv}")
    print(f"decision_md={decision_md}")
    print(f"all_fail={all_fail}")

    return {
        "summary_tsv": str(summary_tsv),
        "combined_tsv": str(combined_tsv),
        "decision_md": str(decision_md),
        "all_fail": bool(all_fail),
        "candidates": [dict(row) for row in candidates],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="P18×P21 Combo Scout")
    parser.add_argument("--seeds", type=str, default="45,47,49")
    parser.add_argument("--players", type=int, default=300)
    parser.add_argument("--rounds", type=int, default=6000)
    parser.add_argument("--burn-in", type=int, default=2000)
    parser.add_argument("--tail", type=int, default=2000)
    parser.add_argument("--a", type=float, default=1.0)
    parser.add_argument("--b", type=float, default=0.9)
    parser.add_argument("--cross", type=float, default=0.20)
    parser.add_argument("--init-q", type=float, default=0.0)
    parser.add_argument("--eta", type=float, default=0.55)
    parser.add_argument("--corr-threshold", type=float, default=0.09)
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    parser.add_argument("--summary-tsv", type=Path, default=DEFAULT_SUMMARY_TSV)
    parser.add_argument("--combined-tsv", type=Path, default=DEFAULT_COMBINED_TSV)
    parser.add_argument("--decision-md", type=Path, default=DEFAULT_DECISION_MD)
    args = parser.parse_args()

    conditions = _default_conditions()
    seeds = _parse_seeds(args.seeds)
    print(f"P18×P21 Combo: {len(conditions)} conditions × {len(seeds)} seeds")
    for cond in conditions:
        print(
            f"  {cond['config_name']}: ε_start={cond['payoff_epsilon']}"
            f" ε_end={cond['epsilon_end'] if cond['epsilon_end'] is not None else cond['payoff_epsilon']}"
            f" R={cond['resample_interval']} α={cond['multipliers']}"
        )

    run_combo_scout(
        seeds=seeds,
        conditions=conditions,
        out_root=args.out_root,
        summary_tsv=args.summary_tsv,
        combined_tsv=args.combined_tsv,
        decision_md=args.decision_md,
        players=args.players,
        rounds=args.rounds,
        burn_in=args.burn_in,
        tail=args.tail,
        a=args.a,
        b=args.b,
        cross=args.cross,
        init_q=args.init_q,
        eta=args.eta,
        corr_threshold=args.corr_threshold,
    )


if __name__ == "__main__":
    main()