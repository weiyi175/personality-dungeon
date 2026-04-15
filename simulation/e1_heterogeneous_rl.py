"""E1 Heterogeneous RL – per-player α_i / β_i G2 Short Scout Harness.

Phase 11 experiment: each player has independently sampled learning rate
(α_i) and selection sharpness (β_i), drawn once from Uniform(lo, hi)
and held fixed for the entire simulation.

This breaks D1's Q-convergence bottleneck where symmetric game + identical
α/β caused all players to learn similar Q-profiles (q_std ≤ 0.03).

Architecture
------------
- Standalone simulation loop (does NOT use simulate()).
- Reuses evolution/independent_rl.py (no modification needed).
- Uses evolution/local_graph.py for topology construction.
- analysis/ layer for cycle_metrics + decay_rate.

Usage
-----
    ./venv/bin/python -m simulation.e1_heterogeneous_rl \\
        --graph-topologies lattice4,well_mixed \\
        --alpha-ranges 0.10:0.10,0.02:0.20,0.005:0.40 \\
        --beta-ranges 10.0:10.0,3.0:20.0,1.0:40.0 \\
        --seeds 45,47,49
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from math import atan2, sqrt
from pathlib import Path
from typing import Any

from analysis.cycle_metrics import classify_cycle_level
from analysis.decay_rate import estimate_decay_gamma
from evolution.independent_rl import (
    STRATEGY_SPACE,
    _NSTRATS,
    apply_per_player_payoff_perturbation,
    apply_q_rotation_bias,
    boltzmann_select,
    boltzmann_weights,
    cosine_similarity,
    one_hot_local_payoff,
    q_value_mean,
    rl_q_update,
    sample_payoff_perturbation,
    strategy_payoff_matrix,
    weight_entropy,
)
from evolution.local_graph import (
    GraphSpec,
    build_graph,
    edge_disagreement_rate,
    edge_strategy_distance,
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

G2_GATE = "g2_e1_hetero_rl"

DEFAULT_OUT_ROOT = Path("outputs/e1_heterogeneous_rl")
DEFAULT_SUMMARY_TSV = DEFAULT_OUT_ROOT / "e1_summary.tsv"
DEFAULT_COMBINED_TSV = DEFAULT_OUT_ROOT / "e1_combined.tsv"
DEFAULT_DECISION_MD = DEFAULT_OUT_ROOT / "e1_decision.md"

SUMMARY_FIELDNAMES = [
    "gate", "condition", "topology",
    "alpha_lo", "alpha_hi", "beta_lo", "beta_hi", "seed",
    "realized_alpha_std", "realized_beta_std",
    "cycle_level", "stage3_score", "turn_strength",
    "env_gamma", "env_gamma_r2", "env_gamma_n_peaks",
    "mean_player_weight_entropy", "weight_heterogeneity_std",
    "mean_q_value_std", "mean_neighbor_weight_cosine",
    "spatial_strategy_clustering", "mean_edge_strategy_distance",
    "has_level3_seed", "out_csv", "provenance_json",
]

COMBINED_FIELDNAMES = [
    "gate", "condition", "topology",
    "alpha_lo", "alpha_hi", "beta_lo", "beta_hi",
    "is_control", "n_seeds",
    "mean_cycle_level", "mean_stage3_score", "mean_turn_strength", "mean_env_gamma",
    "level_counts_json", "p_level_3", "level3_seed_count",
    "realized_alpha_std", "realized_beta_std",
    "mean_player_weight_entropy", "weight_heterogeneity_std",
    "mean_q_value_std", "mean_neighbor_weight_cosine",
    "spatial_strategy_clustering", "mean_edge_strategy_distance",
    "short_scout_pass", "hard_stop_fail", "longer_confirm_candidate", "verdict",
    "representative_seed",
    "representative_simplex_png", "representative_phase_amplitude_png",
    "players", "rounds", "out_dir",
]


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _parse_seeds(s: str) -> list[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip()]

def _parse_strs(s: str) -> list[str]:
    return [x.strip() for x in s.split(",") if x.strip()]

def _parse_ranges(s: str) -> list[tuple[float, float]]:
    """Parse 'lo:hi,lo:hi,...' into list of (lo, hi) tuples."""
    out: list[tuple[float, float]] = []
    for tok in s.split(","):
        tok = tok.strip()
        if not tok:
            continue
        parts = tok.split(":")
        if len(parts) != 2:
            raise ValueError(f"Range must be lo:hi, got {tok!r}")
        out.append((float(parts[0]), float(parts[1])))
    return out

def _fmt(v: float) -> str:
    return f"{float(v):.6f}"

def _yn(v: bool) -> str:
    return "yes" if v else "no"

def _mean(vals: list[float]) -> float:
    return float(sum(vals) / len(vals)) if vals else 0.0

def _std(vals: list[float]) -> float:
    if len(vals) < 2:
        return 0.0
    mu = _mean(vals)
    return float((sum((v - mu) ** 2 for v in vals) / len(vals)) ** 0.5)

def _write_tsv(path: Path, *, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t", extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def _cond_name(*, topology: str, alpha_lo: float, alpha_hi: float,
               beta_lo: float, beta_hi: float) -> str:
    def _tok(v: float) -> str:
        return f"{v:.3f}".rstrip("0").rstrip(".").replace(".", "p")
    return f"g2_e1_{topology}_a{_tok(alpha_lo)}_{_tok(alpha_hi)}_b{_tok(beta_lo)}_{_tok(beta_hi)}"

def _ctrl_name(topology: str) -> str:
    return f"g2_e1_frozen_{topology}"


# ---------------------------------------------------------------------------
# Phase / simplex helpers
# ---------------------------------------------------------------------------

def _phase_angle(pa: float, pd: float, pb: float) -> float:
    return float(atan2(sqrt(3.0) * (pd - pb), 2.0 * pa - pd - pb))

def _amplitude(pa: float, pd: float, pb: float) -> float:
    return float(((pa - 1/3)**2 + (pd - 1/3)**2 + (pb - 1/3)**2) ** 0.5)

def _simplex_xy(pa: float, pd: float, pb: float) -> tuple[float, float]:
    return (pd - pb, pa - 0.5 * (pd + pb))


# ---------------------------------------------------------------------------
# Graph helpers
# ---------------------------------------------------------------------------

def _make_graph_spec(topology: str) -> GraphSpec | None:
    if topology == "well_mixed":
        return None
    if topology == "lattice4":
        return GraphSpec(topology="lattice4", degree=4, lattice_rows=15, lattice_cols=20)
    raise ValueError(f"Unknown topology: {topology!r}")


# ---------------------------------------------------------------------------
# Per-player parameter sampling
# ---------------------------------------------------------------------------

def _sample_player_params(
    n: int, *, alpha_lo: float, alpha_hi: float,
    beta_lo: float, beta_hi: float, rng: random.Random,
) -> tuple[list[float], list[float]]:
    """Draw per-player α_i ~ U(alpha_lo, alpha_hi), β_i ~ U(beta_lo, beta_hi).

    Clips to safe ranges: α ∈ [0, 1], β ∈ [0.1, 100].
    """
    alphas: list[float] = []
    betas: list[float] = []
    for _ in range(n):
        ai = rng.uniform(float(alpha_lo), float(alpha_hi))
        ai = max(0.0, min(1.0, ai))
        bi = rng.uniform(float(beta_lo), float(beta_hi))
        bi = max(0.1, min(100.0, bi))
        alphas.append(ai)
        betas.append(bi)
    return alphas, betas


# ---------------------------------------------------------------------------
# Core simulation loop
# ---------------------------------------------------------------------------

def _run_e1_simulation(
    *,
    n_players: int,
    n_rounds: int,
    seed: int,
    graph_spec: GraphSpec | None,
    alpha_lo: float,
    alpha_hi: float,
    beta_lo: float,
    beta_hi: float,
    a: float,
    b: float,
    cross: float,
    init_q: float = 0.0,
    control_degree: int = 4,
    entropy_lambda: float = 0.0,
    payoff_epsilon: float = 0.0,
    # Phase 18: dynamic payoff perturbation
    epsilon_resample_interval: int = 0,
    epsilon_end: float | None = None,
    # Phase 20: Q-rotation bias
    q_rotation_delta: float = 0.0,
    # Phase 21: per-strategy α multipliers
    strategy_alpha_multipliers: list[float] | None = None,
) -> tuple[
    list[dict[str, Any]],    # global_rows
    list[dict[str, Any]],    # round_diagnostics
    list[list[float]],       # final_q_values
    list[list[int]] | None,  # adj
    list[float],             # per-player alphas
    list[float],             # per-player betas
]:
    rng = random.Random(int(seed))
    n = int(n_players)
    k = int(control_degree)

    # --- Per-player heterogeneous parameters ---
    player_alphas, player_betas = _sample_player_params(
        n, alpha_lo=float(alpha_lo), alpha_hi=float(alpha_hi),
        beta_lo=float(beta_lo), beta_hi=float(beta_hi), rng=rng,
    )

    # Build static graph
    adj: list[list[int]] | None = None
    if graph_spec is not None:
        adj = build_graph(n, graph_spec, seed=int(seed))

    # Payoff matrix
    pmat = strategy_payoff_matrix(a=float(a), b=float(b), cross=float(cross))

    # Per-player payoff perturbation (F1v2)
    payoff_deltas = sample_payoff_perturbation(
        n, epsilon=float(payoff_epsilon), rng=rng,
    )

    # Phase 18: resolve annealing endpoint
    _eps_end = float(epsilon_end) if epsilon_end is not None else float(payoff_epsilon)
    _eps_resample = int(epsilon_resample_interval)

    # Phase 21: per-strategy α multipliers (clip to [0,1] after multiply)
    _strat_mult: list[float] | None = None
    if strategy_alpha_multipliers is not None:
        _strat_mult = [float(m) for m in strategy_alpha_multipliers]
        assert len(_strat_mult) == _NSTRATS

    # Initialise per-player Q-tables
    q0 = float(init_q)
    q_tables: list[list[float]] = [[q0] * _NSTRATS for _ in range(n)]

    global_rows: list[dict[str, Any]] = []
    round_diag: list[dict[str, Any]] = []

    for t in range(int(n_rounds)):
        # --- effective adjacency ---
        if adj is not None:
            eff = adj
        else:
            eff = []
            for i in range(n):
                nbrs: set[int] = set()
                while len(nbrs) < k:
                    j = rng.randrange(n)
                    if j != i:
                        nbrs.add(j)
                eff.append(sorted(nbrs))

        # --- 1. Boltzmann strategy selection (per-player β_i) ---
        chosen: list[int] = []
        for i in range(n):
            chosen.append(boltzmann_select(q_tables[i], beta=player_betas[i], rng=rng))

        # --- 2. Local one-hot payoff ---
        rewards: list[float] = []
        for i in range(n):
            ns = [chosen[j] for j in eff[i]]
            rewards.append(one_hot_local_payoff(chosen[i], ns, pmat))

        # --- 2b. Per-player payoff perturbation (F1v2 / P18 dynamic) ---
        if _eps_resample > 0 and t > 0 and t % _eps_resample == 0:
            frac = t / int(n_rounds)
            cur_eps = float(payoff_epsilon) * (1.0 - frac) + _eps_end * frac
            payoff_deltas = sample_payoff_perturbation(
                n, epsilon=cur_eps, rng=rng,
            )
        if payoff_epsilon > 0.0 or _eps_resample > 0:
            rewards = apply_per_player_payoff_perturbation(
                rewards, chosen, payoff_deltas,
            )

        # --- 3. Q-update (per-player α_i, with optional per-strategy multiplier) ---
        for i in range(n):
            if _strat_mult is not None:
                eff_alpha = min(1.0, player_alphas[i] * _strat_mult[chosen[i]])
            else:
                eff_alpha = player_alphas[i]
            q_tables[i] = rl_q_update(q_tables[i], chosen[i], rewards[i],
                                       alpha=eff_alpha)

        # --- 3b. Q-rotation bias (Phase 20) ---
        if q_rotation_delta != 0.0:
            for i in range(n):
                q_tables[i] = apply_q_rotation_bias(
                    q_tables[i], delta_rot=q_rotation_delta,
                )

        # --- 3c. Q-value centering (G1 entropy regularization) ---
        if entropy_lambda > 0.0:
            for i in range(n):
                q = q_tables[i]
                q_mean = sum(q) / len(q)
                shrink = entropy_lambda * player_alphas[i]
                q_tables[i] = [v - shrink * (v - q_mean) for v in q]

        # --- 4. Global strategy distribution ---
        counts = [0] * _NSTRATS
        for c in chosen:
            counts[c] += 1
        row: dict[str, Any] = {"round": t}
        for si in range(_NSTRATS):
            row[f"p_{STRATEGY_SPACE[si]}"] = counts[si] / n
        global_rows.append(row)

        # --- 5. Diagnostics ---
        # Per-player weights from softmax(q) with per-player β
        all_w: list[list[float]] = [
            boltzmann_weights(q_tables[i], beta=player_betas[i]) for i in range(n)
        ]

        # mean_player_weight_entropy
        total_h = sum(weight_entropy(w) for w in all_w)
        mean_h = total_h / n

        # weight_heterogeneity_std
        mean_w = [0.0] * _NSTRATS
        for w in all_w:
            for si in range(_NSTRATS):
                mean_w[si] += w[si]
        mean_w = [v / n for v in mean_w]
        var_s = 0.0
        for w in all_w:
            for si in range(_NSTRATS):
                var_s += (w[si] - mean_w[si]) ** 2
        w_std = (var_s / (n * _NSTRATS)) ** 0.5

        # mean_q_value_std: std of per-player mean-Q across population
        q_means = [q_value_mean(q_tables[i]) for i in range(n)]
        q_mu = _mean(q_means)
        q_var = sum((v - q_mu) ** 2 for v in q_means) / n if n > 0 else 0.0
        q_std = q_var ** 0.5

        # mean_neighbor_weight_cosine
        nwc_sum = 0.0
        nwc_cnt = 0
        if adj is not None:
            for i, nbrs in enumerate(adj):
                for j in nbrs:
                    nwc_sum += cosine_similarity(all_w[i], all_w[j])
                    nwc_cnt += 1
        nwc = (nwc_sum / nwc_cnt) if nwc_cnt > 0 else 0.0

        round_diag.append({
            "mean_player_weight_entropy": float(mean_h),
            "weight_heterogeneity_std": float(w_std),
            "mean_q_value_std": float(q_std),
            "mean_neighbor_weight_cosine": float(nwc),
        })

    return global_rows, round_diag, q_tables, adj, player_alphas, player_betas


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def _tail_begin(n_rows: int, *, burn_in: int, tail: int) -> int:
    if n_rows <= 0:
        return 0
    return max(int(burn_in), n_rows - int(tail))


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


def _tail_e1_diagnostics(
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

    # Spatial diagnostics from final Q-tables + graph
    ssc = 0.0
    mesd = 0.0
    if adj is not None:
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
# CSV + Plots
# ---------------------------------------------------------------------------

def _write_csv(path: Path, *, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fns = ["round"] + [f"p_{s}" for s in STRATEGY_SPACE]
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fns, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def _write_simplex_plot(
    rows: list[dict[str, Any]], *, out_png: Path, title: str,
    burn_in: int, tail: int,
) -> None:
    try:
        import matplotlib.pyplot as plt
        from matplotlib.collections import LineCollection
        import numpy as np
    except ImportError:
        return
    begin = _tail_begin(len(rows), burn_in=int(burn_in), tail=int(tail))
    tail_rows = rows[begin:]
    if len(tail_rows) < 2:
        return
    pts = [_simplex_xy(float(r["p_aggressive"]), float(r["p_defensive"]),
                        float(r["p_balanced"])) for r in tail_rows]
    xs = np.array([p[0] for p in pts], dtype=float)
    ys = np.array([p[1] for p in pts], dtype=float)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5, 5), constrained_layout=True)
    segs = np.stack([np.column_stack([xs[:-1], ys[:-1]]),
                     np.column_stack([xs[1:], ys[1:]])], axis=1)
    lc = LineCollection(segs, cmap="viridis", linewidths=1.5)
    lc.set_array(np.linspace(0, 1, len(segs)))
    ax.add_collection(lc)
    ax.scatter(xs[0], ys[0], color="#1f77b4", s=30, marker="o")
    ax.scatter(xs[-1], ys[-1], color="#d62728", s=40, marker="X")
    ax.axhline(0, color="#ccc", lw=0.5)
    ax.axvline(0, color="#ccc", lw=0.5)
    ax.set_title(title, fontsize=9)
    ax.set_aspect("equal", adjustable="box")
    fig.savefig(out_png, dpi=160)
    plt.close(fig)


def _write_phase_amplitude_plot(
    rows: list[dict[str, Any]], *, out_png: Path, title: str,
    burn_in: int, tail: int,
) -> None:
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        return
    begin = _tail_begin(len(rows), burn_in=int(burn_in), tail=int(tail))
    tail_rows = rows[begin:]
    if not tail_rows:
        return
    rds = np.array([int(r["round"]) for r in tail_rows], dtype=float)
    phases = np.unwrap(np.array([
        _phase_angle(float(r["p_aggressive"]), float(r["p_defensive"]),
                     float(r["p_balanced"])) for r in tail_rows]))
    amps = np.array([
        _amplitude(float(r["p_aggressive"]), float(r["p_defensive"]),
                   float(r["p_balanced"])) for r in tail_rows])
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 1, figsize=(8, 5.5), sharex=True,
                             constrained_layout=True)
    axes[0].plot(rds, phases, color="#1f77b4", lw=1.5)
    axes[0].set_ylabel("phase angle")
    axes[0].set_title(title)
    axes[0].grid(alpha=0.25)
    axes[1].plot(rds, amps, color="#d62728", lw=1.5)
    axes[1].set_ylabel("amplitude")
    axes[1].set_xlabel("round")
    axes[1].grid(alpha=0.25)
    fig.savefig(out_png, dpi=160)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Summary builder
# ---------------------------------------------------------------------------

def _build_summary(
    *, condition: str, topology: str,
    alpha_lo: float, alpha_hi: float, beta_lo: float, beta_hi: float,
    is_control: bool, metric_rows: list[dict[str, Any]],
    diag_rows: list[dict[str, float]], param_stats: list[dict[str, float]],
    out_dir: Path, players: int, rounds: int,
) -> dict[str, Any]:
    levels = [int(r["cycle_level"]) for r in metric_rows]
    lc = {lv: levels.count(lv) for lv in range(4)}
    den = float(len(metric_rows)) or 1.0
    return {
        "gate": G2_GATE, "condition": condition, "topology": topology,
        "alpha_lo": _fmt(alpha_lo), "alpha_hi": _fmt(alpha_hi),
        "beta_lo": _fmt(beta_lo), "beta_hi": _fmt(beta_hi),
        "is_control": _yn(is_control), "n_seeds": len(metric_rows),
        "mean_cycle_level":  _fmt(_mean([float(l) for l in levels])),
        "mean_stage3_score": _fmt(_mean([float(r["stage3_score"]) for r in metric_rows])),
        "mean_turn_strength": _fmt(_mean([float(r["turn_strength"]) for r in metric_rows])),
        "mean_env_gamma":    _fmt(_mean([float(r["env_gamma"]) for r in metric_rows])),
        "level_counts_json": json.dumps(lc, sort_keys=True),
        "p_level_3": _fmt(sum(1 for l in levels if l >= 3) / den),
        "level3_seed_count": sum(1 for l in levels if l >= 3),
        "realized_alpha_std": _fmt(_mean([float(p["alpha_std"]) for p in param_stats])),
        "realized_beta_std":  _fmt(_mean([float(p["beta_std"]) for p in param_stats])),
        "mean_player_weight_entropy": _fmt(_mean([float(d["mean_player_weight_entropy"]) for d in diag_rows])),
        "weight_heterogeneity_std":   _fmt(_mean([float(d["weight_heterogeneity_std"]) for d in diag_rows])),
        "mean_q_value_std":           _fmt(_mean([float(d["mean_q_value_std"]) for d in diag_rows])),
        "mean_neighbor_weight_cosine": _fmt(_mean([float(d["mean_neighbor_weight_cosine"]) for d in diag_rows])),
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
    candidates: list[dict[str, Any]], close_e1: bool,
) -> None:
    lines = ["# E1 Heterogeneous RL Decision", "", "## G2 Short Scout", ""]
    for r in combined:
        lines.append(
            f"- {r['condition']}: topo={r['topology']}"
            f" α=[{r['alpha_lo']},{r['alpha_hi']}]"
            f" β=[{r['beta_lo']},{r['beta_hi']}]"
            f" level3={r['level3_seed_count']}"
            f" entropy={r['mean_player_weight_entropy']}"
            f" q_std={r['mean_q_value_std']}"
            f" w_std={r['weight_heterogeneity_std']}"
            f" α_std={r['realized_alpha_std']}"
            f" β_std={r['realized_beta_std']}"
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
    lines += ["", "## Pass Gate (4-way AND)", ""]
    lines += [
        "1. level3_seed_count ≥ 2",
        "2. mean_env_gamma ≥ 0",
        "3. mean_player_weight_entropy ≥ 1.18",
        "4. mean_q_value_std > 0.08",
    ]
    lines += ["", "## Stop Rule", ""]
    lines.append("- If ALL active conditions hard-stop fail, E1 closes as negative result.")
    lines.append("- Interpretation: heterogeneous α/β is insufficient to open Level 3 basin.")
    lines.append(f"- overall_verdict: {'close_e1' if close_e1 else 'keep_e1_open'}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# Main harness
# ---------------------------------------------------------------------------

def run_e1_scout(
    *,
    seeds: list[int],
    topologies: list[str],
    alpha_ranges: list[tuple[float, float]],
    beta_ranges: list[tuple[float, float]],
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

    # Frozen controls (alpha=0) per topology
    for topo in topologies:
        gs = _make_graph_spec(topo)
        conditions.append({
            "condition": _ctrl_name(topo),
            "topology": topo, "graph_spec": gs,
            "alpha_lo": 0.0, "alpha_hi": 0.0,
            "beta_lo": 5.0, "beta_hi": 5.0,
            "is_control": True,
        })

    # Active conditions: all combos of alpha_range × beta_range × topology
    for topo in topologies:
        gs = _make_graph_spec(topo)
        for alo, ahi in alpha_ranges:
            for blo, bhi in beta_ranges:
                conditions.append({
                    "condition": _cond_name(
                        topology=topo, alpha_lo=alo, alpha_hi=ahi,
                        beta_lo=blo, beta_hi=bhi),
                    "topology": topo, "graph_spec": gs,
                    "alpha_lo": float(alo), "alpha_hi": float(ahi),
                    "beta_lo": float(blo), "beta_hi": float(bhi),
                    "is_control": False,
                })

    all_summary: list[dict[str, Any]] = []
    all_combined: list[dict[str, Any]] = []
    cond_seed_rows: dict[str, list[dict[str, Any]]] = {}
    cond_ts: dict[str, dict[int, list[dict[str, Any]]]] = {}

    for cond in conditions:
        cn = str(cond["condition"])
        gs = cond["graph_spec"]
        alo = float(cond["alpha_lo"])
        ahi = float(cond["alpha_hi"])
        blo = float(cond["beta_lo"])
        bhi = float(cond["beta_hi"])
        topo = str(cond["topology"])
        is_ctrl = bool(cond["is_control"])
        od = out_root / cn
        od.mkdir(parents=True, exist_ok=True)

        per_seed: list[dict[str, Any]] = []
        metrics: list[dict[str, Any]] = []
        diags: list[dict[str, float]] = []
        param_stats: list[dict[str, float]] = []
        seed_ts: dict[int, list[dict[str, Any]]] = {}

        for seed in seeds:
            print(f"  E1 {cn} seed={seed} ...")
            rows, rdiag, fq, adj_out, p_alphas, p_betas = _run_e1_simulation(
                n_players=int(players), n_rounds=int(rounds), seed=int(seed),
                graph_spec=gs,
                alpha_lo=alo, alpha_hi=ahi,
                beta_lo=blo, beta_hi=bhi,
                a=float(a), b=float(b), cross=float(cross),
                init_q=float(init_q),
            )
            seed_ts[int(seed)] = rows

            csv_path = od / f"seed{seed}.csv"
            _write_csv(csv_path, rows=rows)

            sm = _seed_metrics(rows, burn_in=burn_in, tail=tail,
                               eta=eta, corr_threshold=corr_threshold)
            td = _tail_e1_diagnostics(
                rdiag, fq, adj_out, p_betas,
                n_rows=len(rows), burn_in=burn_in, tail=tail,
            )

            a_std = _std(p_alphas)
            b_std = _std(p_betas)
            ps = {"alpha_std": a_std, "beta_std": b_std}

            prov = {
                "condition": cn, "topology": topo,
                "alpha_lo": alo, "alpha_hi": ahi,
                "beta_lo": blo, "beta_hi": bhi,
                "seed": int(seed),
                "config": {
                    "players": int(players), "rounds": int(rounds),
                    "burn_in": int(burn_in), "tail": int(tail),
                    "a": float(a), "b": float(b), "cross": float(cross),
                    "init_q": float(init_q),
                },
                "realized_alpha_std": float(a_std),
                "realized_beta_std": float(b_std),
                "cycle_metrics": dict(sm),
                "e1_diagnostics": dict(td),
            }
            pj = od / f"seed{seed}_provenance.json"
            pj.write_text(json.dumps(prov, ensure_ascii=False, indent=2) + "\n",
                          encoding="utf-8")

            seed_row = {
                "gate": G2_GATE, "condition": cn, "topology": topo,
                "alpha_lo": _fmt(alo), "alpha_hi": _fmt(ahi),
                "beta_lo": _fmt(blo), "beta_hi": _fmt(bhi),
                "seed": int(seed),
                "realized_alpha_std": _fmt(a_std),
                "realized_beta_std": _fmt(b_std),
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
                "spatial_strategy_clustering": _fmt(td["spatial_strategy_clustering"]),
                "mean_edge_strategy_distance": _fmt(td["mean_edge_strategy_distance"]),
                "has_level3_seed": _yn(int(sm["cycle_level"]) >= 3),
                "out_csv": str(csv_path), "provenance_json": str(pj),
            }
            per_seed.append(seed_row)
            metrics.append(sm)
            diags.append(td)
            param_stats.append(ps)

        cond_seed_rows[cn] = per_seed
        cond_ts[cn] = seed_ts
        all_combined.append(_build_summary(
            condition=cn, topology=topo,
            alpha_lo=alo, alpha_hi=ahi, beta_lo=blo, beta_hi=bhi,
            is_control=is_ctrl, metric_rows=metrics, diag_rows=diags,
            param_stats=param_stats,
            out_dir=od, players=players, rounds=rounds,
        ))

    # --- Verdicts ---
    for cr in all_combined:
        if str(cr["is_control"]) == "yes":
            continue
        l3  = int(cr["level3_seed_count"])
        eg  = float(cr["mean_env_gamma"])
        ent = float(cr["mean_player_weight_entropy"])
        qs  = float(cr["mean_q_value_std"])

        sp = l3 >= 2 and eg >= 0.0 and ent >= 1.18 and qs > 0.08
        hs = l3 == 0 and ent < 1.10 and qs < 0.04
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
    close_e1 = bool(active) and all(
        str(r["hard_stop_fail"]) == "yes" for r in active)

    candidates = [r for r in all_combined if str(r["verdict"]) == "pass"]
    candidates.sort(
        key=lambda r: (int(r["level3_seed_count"]), float(r["mean_q_value_std"])),
        reverse=True)

    _write_tsv(summary_tsv, fieldnames=SUMMARY_FIELDNAMES, rows=all_summary)
    _write_tsv(combined_tsv, fieldnames=COMBINED_FIELDNAMES, rows=all_combined)
    _write_decision(decision_md, combined=all_combined,
                    candidates=candidates, close_e1=close_e1)

    print(f"summary_tsv={summary_tsv}")
    print(f"combined_tsv={combined_tsv}")
    print(f"decision_md={decision_md}")
    print(f"close_e1={close_e1}")

    return {
        "summary_tsv": str(summary_tsv),
        "combined_tsv": str(combined_tsv),
        "decision_md": str(decision_md),
        "close_e1": bool(close_e1),
        "candidates": [dict(r) for r in candidates],
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(description="E1 Heterogeneous RL G2 Scout")
    p.add_argument("--seeds", type=str, default="45,47,49")
    p.add_argument("--graph-topologies", type=str, default="lattice4,well_mixed")
    p.add_argument("--alpha-ranges", type=str,
                   default="0.10:0.10,0.02:0.20,0.005:0.40")
    p.add_argument("--beta-ranges", type=str,
                   default="10.0:10.0,3.0:20.0,1.0:40.0")
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

    run_e1_scout(
        seeds=_parse_seeds(args.seeds),
        topologies=_parse_strs(args.graph_topologies),
        alpha_ranges=_parse_ranges(args.alpha_ranges),
        beta_ranges=_parse_ranges(args.beta_ranges),
        out_root=args.out_root, summary_tsv=args.summary_tsv,
        combined_tsv=args.combined_tsv, decision_md=args.decision_md,
        players=args.players, rounds=args.rounds,
        burn_in=args.burn_in, tail=args.tail,
        a=args.a, b=args.b, cross=args.cross, init_q=args.init_q,
        eta=args.eta, corr_threshold=args.corr_threshold,
    )


if __name__ == "__main__":
    main()
