"""C1 Local Pairwise Imitation harness.

Implements the C-family first experiment: local pairwise imitation on
ring-4 and lattice-4 topologies, replacing the global shared growth vector
with per-player local imitation updates.

Architecture invariants (blueprint §3)
---------------------------------------
- Does NOT use simulation.run_simulation.simulate() inner loop.
- Does NOT import analysis/ directly in the inner loop.
- Does NOT use a global shared growth vector.
- Update is synchronous: all new weights w(t+1) computed from w(t).
- Neighbour selection is uniform random (per player RNG).
- Per-round CSV row = population-average of player weights/strategies.

Usage (blueprint §14.1)
-----------------------
./venv/bin/python -m simulation.c1_local_pairwise \\
  --seeds 45,47,49 \\
  --topologies ring4,lattice4 \\
  --pairwise-imitation-strengths 0.10,0.20,0.35 \\
  --pairwise-beta 8.0 \\
  --players 300 --rounds 3000 \\
  --burn-in 1000 --tail 1000 \\
  --a 1.0 --b 0.9 --matrix-cross-coupling 0.20 \\
  --out-root outputs/c1_local_pairwise_short_scout \\
  --summary-tsv outputs/c1_local_pairwise_short_scout_summary.tsv \\
  --combined-tsv outputs/c1_local_pairwise_short_scout_combined.tsv \\
  --decision-md outputs/c1_local_pairwise_short_scout_decision.md
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from dataclasses import dataclass
from math import atan2, isfinite, pi, sqrt
from pathlib import Path
from typing import Any

from analysis.cycle_metrics import classify_cycle_level
from analysis.decay_rate import estimate_decay_gamma
from evolution.local_graph import (
    GraphSpec,
    build_graph,
    edge_disagreement_rate,
    edge_strategy_distance,
    sample_neighbor,
)
from evolution.local_pairwise import (
    local_payoff,
    list_to_weights,
    pairwise_adoption_probability,
    pairwise_imitation_update,
    weights_to_simplex,
)
from players.base_player import BasePlayer
from simulation.run_simulation import (
    _initial_weights,
    _matrix_ab_payoff_vec,
)


REPO_ROOT = Path(__file__).resolve().parents[1]

STRATEGY_SPACE = ["aggressive", "defensive", "balanced"]

GATE_NAME = "g2_sampled"
FAMILY = "c1_local_pairwise"

SUMMARY_FIELDNAMES = [
    "gate",
    "family",
    "condition",
    "graph_topology",
    "graph_degree",
    "pairwise_imitation_strength",
    "pairwise_beta",
    "local_update_mode",
    "seed",
    "cycle_level",
    "stage3_score",
    "turn_strength",
    "env_gamma",
    "env_gamma_r2",
    "env_gamma_n_peaks",
    "control_env_gamma",
    "control_stage3_score",
    "stage3_uplift_vs_control_seed",
    "has_level3_seed",
    "mean_pairwise_adoption_prob",
    "mean_imitation_step_norm",
    "mean_edge_strategy_distance",
    "mean_neighbor_payoff_gap",
    "mean_local_phase_spread",
    "edge_disagreement_rate",
    "phase_amplitude_stability",
    "out_csv",
    "provenance_json",
]

COMBINED_FIELDNAMES = [
    "gate",
    "family",
    "condition",
    "graph_topology",
    "graph_degree",
    "pairwise_imitation_strength",
    "pairwise_beta",
    "local_update_mode",
    "is_control",
    "n_seeds",
    "mean_cycle_level",
    "mean_stage3_score",
    "mean_turn_strength",
    "mean_env_gamma",
    "level_counts_json",
    "p_level_3",
    "level3_seed_count",
    "control_mean_env_gamma",
    "control_mean_stage3_score",
    "stage3_uplift_vs_control",
    "mean_pairwise_adoption_prob",
    "mean_imitation_step_norm",
    "mean_edge_strategy_distance",
    "mean_neighbor_payoff_gap",
    "mean_local_phase_spread",
    "edge_disagreement_rate",
    "phase_amplitude_stability",
    "short_scout_pass",
    "hard_stop_fail",
    "verdict",
    "representative_seed",
    "representative_simplex_png",
    "representative_phase_amplitude_png",
    "players",
    "rounds",
    "out_dir",
]


# ---------------------------------------------------------------------------
# Config dataclass
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class C1CellConfig:
    condition: str
    graph_spec: GraphSpec
    pairwise_imitation_strength: float
    pairwise_beta: float
    players: int
    rounds: int
    seed: int
    a: float
    b: float
    cross: float
    burn_in: int
    tail: int
    init_bias: float
    memory_kernel: int
    out_dir: Path


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _parse_seeds(spec: str) -> list[int]:
    return [int(x.strip()) for x in str(spec).split(",") if x.strip()]


def _parse_float_list(spec: str) -> list[float]:
    return [float(x.strip()) for x in str(spec).split(",") if x.strip()]


def _parse_str_list(spec: str) -> list[str]:
    return [x.strip() for x in str(spec).split(",") if x.strip()]


def _fmt(v: float) -> str:
    return f"{float(v):.6f}"


def _yn(v: bool) -> str:
    return "yes" if bool(v) else "no"


def _safe_mean(xs: list[float]) -> float:
    if not xs:
        return 0.0
    return float(sum(xs)) / float(len(xs))


def _write_tsv(path: Path, *, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def _condition_name(*, topology: str, mu: float) -> str:
    mu_tok = f"{float(mu):.2f}".replace(".", "p")
    return f"c1_{topology}_mu{mu_tok}"


def _control_condition_name() -> str:
    return "control_wellmixed_sampled"


def _graph_spec_for(topology: str) -> GraphSpec:
    if topology == "ring4":
        return GraphSpec(topology="ring4", degree=4)
    if topology == "lattice4":
        return GraphSpec(topology="lattice4", degree=4, lattice_rows=15, lattice_cols=20)
    raise ValueError(f"Unsupported topology: {topology!r}")


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def _simplex_xy(p_a: float, p_d: float, p_b: float) -> tuple[float, float]:
    return (p_d - p_b, p_a - 0.5 * (p_d + p_b))


def _phase_angle(p_a: float, p_d: float, p_b: float) -> float:
    return float(atan2(sqrt(3.0) * (p_d - p_b), 2.0 * p_a - p_d - p_b))


def _amplitude(p_a: float, p_d: float, p_b: float) -> float:
    return float(((p_a - 1.0 / 3.0) ** 2 + (p_d - 1.0 / 3.0) ** 2 + (p_b - 1.0 / 3.0) ** 2) ** 0.5)


def _tail_begin(n_rows: int, *, burn_in: int, tail: int) -> int:
    return max(int(burn_in), n_rows - int(tail))


def _local_phase_spread(
    adj: list[list[int]],
    all_weights_simplex: list[list[float]],
) -> float:
    """Mean pairwise phase-angle difference over all edges (i→j)."""
    total = 0.0
    count = 0
    for i, nbrs in enumerate(adj):
        wi = all_weights_simplex[i]
        phi_i = _phase_angle(wi[0], wi[1], wi[2])
        for j in nbrs:
            wj = all_weights_simplex[j]
            phi_j = _phase_angle(wj[0], wj[1], wj[2])
            diff = abs(phi_i - phi_j)
            if diff > pi:
                diff = 2.0 * pi - diff
            total += diff
            count += 1
    return float(total / count) if count > 0 else 0.0


# ---------------------------------------------------------------------------
# Core simulation loop
# ---------------------------------------------------------------------------

def run_c1_cell(config: C1CellConfig) -> dict[str, Any]:
    """Run one C1 simulation cell and return rows + diagnostics.

    Returns
    -------
    dict with keys:
        'global_rows'  : list[dict]  – one row per round (timeseries CSV)
        'round_diag'   : list[dict]  – per-round local diagnostics
    """
    rng_global = random.Random(int(config.seed))
    # Per-player RNGs (deterministic, player-index offset)
    player_rngs = [
        random.Random(int(config.seed) + i)
        for i in range(int(config.players))
    ]

    # Initialise players
    players = [
        BasePlayer(STRATEGY_SPACE, rng=player_rngs[i])
        for i in range(int(config.players))
    ]
    init_w_dict = _initial_weights(
        strategy_space=STRATEGY_SPACE,
        init_bias=float(config.init_bias),
    )
    for pl in players:
        pl.update_weights(init_w_dict)

    # Build graph
    adj = build_graph(int(config.players), config.graph_spec, int(config.seed))

    global_rows: list[dict[str, Any]] = []
    round_diag: list[dict[str, Any]] = []

    for t in range(int(config.rounds)):
        n = int(config.players)

        # --- Snapshot current weights as simplex (synchronous: lock w(t)) ---
        simplex_t = [
            weights_to_simplex(pl.strategy_weights)
            for pl in players
        ]

        # --- Each player samples a strategy ---
        for pl in players:
            pl.choose_strategy()

        # Per-player sampled strategies
        strategies_t = [str(pl.last_strategy) for pl in players]

        # --- Compute local payoff for each player (using w(t) simplex) ---
        payoffs_t = []
        for i, pl in enumerate(players):
            nbr_strats = [strategies_t[j] for j in adj[i]]
            pi_i = local_payoff(
                simplex_t[i],
                nbr_strats,
                a=float(config.a),
                b=float(config.b),
                cross=float(config.cross),
            )
            payoffs_t.append(pi_i)
            pl.update_utility(pi_i)
            pl.last_reward = pi_i

        # --- Synchronous pairwise imitation update ---
        new_simplex: list[list[float]] = []
        adoption_probs: list[float] = []
        step_norms: list[float] = []
        nbr_payoff_gaps: list[float] = []

        for i, pl in enumerate(players):
            j = sample_neighbor(player_rngs[i], adj, i)
            q = pairwise_adoption_probability(
                payoffs_t[i],
                payoffs_t[j],
                beta_pair=float(config.pairwise_beta),
            )
            adoption_probs.append(q)
            nbr_payoff_gaps.append(abs(payoffs_t[j] - payoffs_t[i]))

            w_i_new = pairwise_imitation_update(
                simplex_t[i],
                simplex_t[j],
                mu=float(config.pairwise_imitation_strength),
                q=q,
            )
            step_norm = sum(abs(new - old) for new, old in zip(w_i_new, simplex_t[i]))
            step_norms.append(step_norm)
            new_simplex.append(w_i_new)

        # --- Broadcast new weights back to players ---
        for i, pl in enumerate(players):
            pl.update_weights(list_to_weights(new_simplex[i]))

        # --- Per-round global observation (population average) ---
        # p_*(t): proportion from sampled strategies this round
        strat_counts = {s: 0 for s in STRATEGY_SPACE}
        for s in strategies_t:
            strat_counts[s] += 1
        p_row = {s: float(strat_counts[s]) / float(n) for s in STRATEGY_SPACE}

        # w_*(t): mean weight after update
        w_mean = {s: _safe_mean([new_simplex[i][k] for i in range(n)])
                  for k, s in enumerate(STRATEGY_SPACE)}

        row: dict[str, Any] = {"round": t}
        for s in STRATEGY_SPACE:
            row[f"p_{s}"] = float(p_row[s])
        for s in STRATEGY_SPACE:
            row[f"w_{s}"] = float(w_mean[s])
        global_rows.append(row)

        # --- Per-round diagnostics ---
        mean_adopt = _safe_mean(adoption_probs)
        mean_step = _safe_mean(step_norms)
        mean_gap = _safe_mean(nbr_payoff_gaps)
        e_dist = edge_strategy_distance(adj, new_simplex)
        e_disagree = edge_disagreement_rate(adj, strategies_t)
        phase_spread = _local_phase_spread(adj, new_simplex)

        round_diag.append({
            "mean_pairwise_adoption_prob": mean_adopt,
            "mean_imitation_step_norm": mean_step,
            "mean_edge_strategy_distance": e_dist,
            "mean_neighbor_payoff_gap": mean_gap,
            "mean_local_phase_spread": phase_spread,
            "edge_disagreement_rate": e_disagree,
        })

    return {"global_rows": global_rows, "round_diag": round_diag}


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def _seed_metrics(
    rows: list[dict[str, Any]],
    *,
    burn_in: int,
    tail: int,
    eta: float = 0.55,
    corr_threshold: float = 0.09,
) -> dict[str, Any]:
    series_map = {s: [float(row[f"p_{s}"]) for row in rows] for s in STRATEGY_SPACE}
    cycle = classify_cycle_level(
        series_map,
        burn_in=int(burn_in),
        tail=int(tail),
        amplitude_threshold=0.02,
        corr_threshold=float(corr_threshold),
        eta=float(eta),
        stage3_method="turning",
        phase_smoothing=1,
    )
    fit = estimate_decay_gamma(series_map, series_kind="p")
    return {
        "cycle_level": int(cycle.level),
        "stage3_score": float(cycle.stage3.score) if cycle.stage3 is not None else 0.0,
        "turn_strength": float(cycle.stage3.turn_strength) if cycle.stage3 is not None else 0.0,
        "env_gamma": float(fit.gamma) if fit is not None else 0.0,
        "env_gamma_r2": float(fit.r2) if fit is not None else 0.0,
        "env_gamma_n_peaks": int(fit.n_peaks) if fit is not None else 0,
    }


def _phase_amplitude_stability(rows: list[dict[str, Any]], *, burn_in: int, tail: int) -> float:
    begin = _tail_begin(len(rows), burn_in=burn_in, tail=tail)
    window = rows[begin:]
    if not window:
        return 0.0
    amps = [_amplitude(float(r["p_aggressive"]), float(r["p_defensive"]), float(r["p_balanced"]))
            for r in window]
    mean_amp = _safe_mean(amps)
    if mean_amp <= 1e-12:
        return 0.0
    var = _safe_mean([(a - mean_amp) ** 2 for a in amps])
    return float(max(0.0, 1.0 - min(1.0, (var ** 0.5) / mean_amp)))


def _tail_local_diagnostics(
    round_diag: list[dict[str, Any]],
    *,
    n_rows: int,
    burn_in: int,
    tail: int,
) -> dict[str, float]:
    begin = _tail_begin(n_rows, burn_in=burn_in, tail=tail)
    window = round_diag[begin:]
    if not window:
        return {k: 0.0 for k in [
            "mean_pairwise_adoption_prob", "mean_imitation_step_norm",
            "mean_edge_strategy_distance", "mean_neighbor_payoff_gap",
            "mean_local_phase_spread", "edge_disagreement_rate",
        ]}
    return {
        key: _safe_mean([float(d[key]) for d in window])
        for key in [
            "mean_pairwise_adoption_prob", "mean_imitation_step_norm",
            "mean_edge_strategy_distance", "mean_neighbor_payoff_gap",
            "mean_local_phase_spread", "edge_disagreement_rate",
        ]
    }


# ---------------------------------------------------------------------------
# Verdict helpers (blueprint §7.2 – §7.4)
# ---------------------------------------------------------------------------

def _compute_verdict(
    *,
    is_control: bool,
    level3_seed_count: int,
    stage3_uplift: float | str,
    mean_env_gamma: float,
    phase_amplitude_stability: float,
    mean_edge_strategy_distance: float,
    mean_local_phase_spread: float,
) -> str:
    if is_control:
        return "control"
    uplift = float(stage3_uplift) if isinstance(stage3_uplift, (int, float)) and isfinite(float(stage3_uplift)) else 0.0

    # Pass: level3 + gamma ok + stability ok
    if (
        level3_seed_count >= 1
        and abs(mean_env_gamma) <= 5e-4
        and phase_amplitude_stability >= 0.3
    ):
        return "pass"

    # Weak positive (§7.4)
    if level3_seed_count >= 1 and abs(mean_env_gamma) > 5e-4:
        return "weak_positive"
    if uplift >= 0.02:
        return "weak_positive"
    if mean_edge_strategy_distance >= 0.08 and mean_local_phase_spread >= 0.15:
        return "weak_positive"

    return "fail"


# ---------------------------------------------------------------------------
# Control run (well-mixed sampled baseline)
# ---------------------------------------------------------------------------

def _run_control(
    *,
    seed: int,
    players: int,
    rounds: int,
    burn_in: int,
    tail: int,
    a: float,
    b: float,
    cross: float,
    init_bias: float,
    memory_kernel: int,
    selection_strength: float,
    out_dir: Path,
) -> dict[str, Any]:
    """Run well-mixed sampled baseline using simulate().

    Uses the existing simulation.run_simulation.simulate() path to ensure
    control results are identical to the established sampled baseline.
    """
    from simulation.run_simulation import SimConfig, simulate

    cfg = SimConfig(
        n_players=int(players),
        n_rounds=int(rounds),
        seed=int(seed),
        payoff_mode="matrix_ab",
        popularity_mode="sampled",
        gamma=0.1,
        epsilon=0.0,
        a=float(a),
        b=float(b),
        matrix_cross_coupling=float(cross),
        init_bias=float(init_bias),
        evolution_mode="sampled",
        payoff_lag=1,
        selection_strength=float(selection_strength),
        memory_kernel=int(memory_kernel),
    )
    _strategy_space, rows = simulate(cfg)
    csv_path = out_dir / f"seed{seed}.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        fieldnames = list(rows[0].keys()) if rows else []
        writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    metrics = _seed_metrics(rows, burn_in=burn_in, tail=tail)
    pa = _phase_amplitude_stability(rows, burn_in=burn_in, tail=tail)
    return {
        "rows": rows,
        "csv_path": csv_path,
        "metrics": metrics,
        "phase_amplitude_stability": pa,
    }


# ---------------------------------------------------------------------------
# Timeseries CSV writer
# ---------------------------------------------------------------------------

def _write_timeseries_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["round"] + [f"p_{s}" for s in STRATEGY_SPACE] + [f"w_{s}" for s in STRATEGY_SPACE]
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _write_simplex_plot(
    rows: list[dict[str, Any]],
    *,
    out_png: Path,
    title: str,
    burn_in: int,
    tail: int,
) -> None:
    try:
        import matplotlib.pyplot as plt
        from matplotlib.collections import LineCollection
        import numpy as np
    except ImportError:
        return

    begin = _tail_begin(len(rows), burn_in=burn_in, tail=tail)
    window = rows[begin:]
    if len(window) < 2:
        return
    pts = [_simplex_xy(float(r["p_aggressive"]), float(r["p_defensive"]), float(r["p_balanced"]))
           for r in window]
    xs = np.array([p[0] for p in pts], dtype=float)
    ys = np.array([p[1] for p in pts], dtype=float)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5, 5), constrained_layout=True)
    segs = np.stack([np.column_stack([xs[:-1], ys[:-1]]), np.column_stack([xs[1:], ys[1:]])], axis=1)
    lc = LineCollection(segs, cmap="viridis", linewidths=1.5)
    lc.set_array(np.linspace(0.0, 1.0, len(segs)))
    ax.add_collection(lc)
    ax.scatter(xs[0], ys[0], color="#1f77b4", s=30, marker="o")
    ax.scatter(xs[-1], ys[-1], color="#d62728", s=40, marker="X")
    ax.axhline(0.0, color="#ccc", linewidth=0.5)
    ax.axvline(0.0, color="#ccc", linewidth=0.5)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(title, fontsize=9)
    fig.savefig(out_png, dpi=160)
    plt.close(fig)


def _write_phase_amplitude_plot(
    rows: list[dict[str, Any]],
    *,
    out_png: Path,
    title: str,
    burn_in: int,
    tail: int,
) -> None:
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        return

    begin = _tail_begin(len(rows), burn_in=burn_in, tail=tail)
    window = rows[begin:]
    if not window:
        return
    rds = np.array([int(r["round"]) for r in window], dtype=float)
    phases = np.unwrap(np.array([_phase_angle(float(r["p_aggressive"]), float(r["p_defensive"]), float(r["p_balanced"])) for r in window], dtype=float))
    amps = np.array([_amplitude(float(r["p_aggressive"]), float(r["p_defensive"]), float(r["p_balanced"])) for r in window], dtype=float)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 1, figsize=(8, 5.5), sharex=True, constrained_layout=True)
    axes[0].plot(rds, phases, color="#1f77b4", linewidth=1.5)
    axes[0].set_ylabel("phase angle")
    axes[0].set_title(title)
    axes[0].grid(alpha=0.25)
    axes[1].plot(rds, amps, color="#d62728", linewidth=1.5)
    axes[1].set_ylabel("amplitude")
    axes[1].set_xlabel("round")
    axes[1].grid(alpha=0.25)
    fig.savefig(out_png, dpi=160)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Decision markdown writer
# ---------------------------------------------------------------------------

def _write_decision(
    path: Path,
    *,
    combined_rows: list[dict[str, Any]],
) -> None:
    lines = [
        "# C1 Local Pairwise Imitation — Short Scout Decision",
        "",
        "## G2 Short Scout",
        "",
    ]

    # Per-condition lines (blueprint §10.2 line format)
    control_rows = [r for r in combined_rows if str(r.get("is_control", "")) == "yes"]
    active_rows = [r for r in combined_rows if str(r.get("is_control", "")) != "yes"]

    for row in control_rows + active_rows:
        is_ctrl = str(row.get("is_control", "")) == "yes"
        uplift_str = "" if is_ctrl else f"  stage3_uplift={row.get('stage3_uplift_vs_control', '')}"
        lines.append(
            f"condition: {row['condition']}"
            + (f"  graph_topology={row['graph_topology']}"
               f"  pairwise_imitation_strength={row['pairwise_imitation_strength']}"
               if not is_ctrl else "")
            + f"  seed_count={row['n_seeds']}"
            f"  level3_seed_count={row['level3_seed_count']}"
            + uplift_str
            + f"  mean_env_gamma={row['mean_env_gamma']}"
            f"  phase_amplitude_stability={row['phase_amplitude_stability']}"
            f"  mean_edge_strategy_distance={row.get('mean_edge_strategy_distance', '')}"
            f"  mean_local_phase_spread={row.get('mean_local_phase_spread', '')}"
            f"  short_scout_pass={row.get('short_scout_pass', '')}"
            f"  hard_stop_fail={row.get('hard_stop_fail', '')}"
            f"  verdict={row['verdict']}"
        )

    # Overall verdict
    any_pass = any(str(r.get("verdict", "")) == "pass" for r in active_rows)
    any_weak = any(str(r.get("verdict", "")) == "weak_positive" for r in active_rows)
    all_fail = all(str(r.get("verdict", "")) in ("fail", "pending") for r in active_rows)

    hard_stop = (
        all(int(r.get("level3_seed_count", 0)) == 0 for r in active_rows)
        and all(
            (
                lambda up: (
                    float(up) < 0.02
                    if isinstance(up, str) and up.lstrip("+-").replace(".", "").isdigit()
                    else float(up) < 0.02 if isinstance(up, (int, float)) else True
                )
            )(r.get("stage3_uplift_vs_control", "0"))
            for r in active_rows
        )
        and all(
            float(r.get("mean_edge_strategy_distance", 0.0)) < 0.08
            for r in active_rows
        )
    )

    lines.extend(["", "## Mechanism Signal", ""])
    # Aggregate diagnostics across active cells
    all_esd = [float(r.get("mean_edge_strategy_distance", 0)) for r in active_rows]
    all_lps = [float(r.get("mean_local_phase_spread", 0)) for r in active_rows]
    max_esd = max(all_esd) if all_esd else 0.0
    max_lps = max(all_lps) if all_lps else 0.0
    if max_esd >= 0.08 or max_lps >= 0.15:
        lines.append(f"- graph-local signal **已形成** (max_edge_strategy_distance={max_esd:.4f}, max_local_phase_spread={max_lps:.4f} rad)")
        if all_fail:
            lines.append("- 但 Level 3 仍未打開 → signal 已形成但不足以打開 Level 3")
    else:
        lines.append(f"- graph-local signal **未形成** (max_edge_strategy_distance={max_esd:.4f}, max_local_phase_spread={max_lps:.4f} rad)")
        if all_fail:
            lines.append("- Level 3 亦未打開 → signal 未形成，local imitation 沒有建立差異化")

    lines.extend(["", "## Overall Verdict", ""])
    if any_pass:
        lines.append("- overall_verdict: pass_c1")
        lines.append("- 建議開 longer_confirm（seeds=0:29）")
    elif any_weak:
        lines.append("- overall_verdict: weak_positive_c1")
        lines.append("- 建議開 targeted follow-up（最多 9 runs）")
    elif hard_stop:
        lines.append("- overall_verdict: close_c1")
        lines.append("- hard_stop: 所有 active cells 0/3 Level 3，uplift < 0.02，no local signal → C1 直接 closure")
    else:
        lines.append("- overall_verdict: close_c1")
        lines.append("- 所有 active cells 未達標，進行 closure")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# Main scout entry point
# ---------------------------------------------------------------------------

def run_c1_scout(
    *,
    seeds: list[int],
    topologies: list[str],
    pairwise_imitation_strengths: list[float],
    pairwise_beta: float,
    players: int,
    rounds: int,
    burn_in: int,
    tail: int,
    a: float,
    b: float,
    cross: float,
    init_bias: float,
    memory_kernel: int,
    selection_strength: float,
    out_root: Path,
    summary_tsv: Path,
    combined_tsv: Path,
    decision_md: Path,
    eta: float = 0.55,
    corr_threshold: float = 0.09,
) -> dict[str, Any]:
    out_root.mkdir(parents=True, exist_ok=True)

    all_summary_rows: list[dict[str, Any]] = []

    # --- Control condition ---
    control_cond = _control_condition_name()
    ctrl_out_dir = out_root / control_cond
    ctrl_out_dir.mkdir(parents=True, exist_ok=True)

    control_per_seed: dict[int, dict[str, Any]] = {}
    for seed in seeds:
        ctrl_result = _run_control(
            seed=seed,
            players=players,
            rounds=rounds,
            burn_in=burn_in,
            tail=tail,
            a=a,
            b=b,
            cross=cross,
            init_bias=init_bias,
            memory_kernel=memory_kernel,
            selection_strength=selection_strength,
            out_dir=ctrl_out_dir,
        )
        control_per_seed[seed] = ctrl_result

        prov = {
            "family": "control_wellmixed_sampled",
            "condition": control_cond,
            "seed": int(seed),
            "graph_topology": "well_mixed",
            "graph_degree": 0,
            "pairwise_imitation_strength": 0.0,
            "pairwise_beta": 0.0,
            "out_csv": str(ctrl_result["csv_path"]),
            "cycle_level": int(ctrl_result["metrics"]["cycle_level"]),
            "stage3_score": float(ctrl_result["metrics"]["stage3_score"]),
            "turn_strength": float(ctrl_result["metrics"]["turn_strength"]),
            "env_gamma": float(ctrl_result["metrics"]["env_gamma"]),
        }
        prov_path = ctrl_out_dir / f"seed{seed}_provenance.json"
        prov_path.write_text(json.dumps(prov, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

        all_summary_rows.append({
            "gate": GATE_NAME,
            "family": FAMILY,
            "condition": control_cond,
            "graph_topology": "well_mixed",
            "graph_degree": 0,
            "pairwise_imitation_strength": "",
            "pairwise_beta": "",
            "local_update_mode": "well_mixed_sampled",
            "seed": int(seed),
            "cycle_level": int(ctrl_result["metrics"]["cycle_level"]),
            "stage3_score": _fmt(ctrl_result["metrics"]["stage3_score"]),
            "turn_strength": _fmt(ctrl_result["metrics"]["turn_strength"]),
            "env_gamma": _fmt(ctrl_result["metrics"]["env_gamma"]),
            "env_gamma_r2": _fmt(ctrl_result["metrics"]["env_gamma_r2"]),
            "env_gamma_n_peaks": int(ctrl_result["metrics"]["env_gamma_n_peaks"]),
            "control_env_gamma": "",
            "control_stage3_score": "",
            "stage3_uplift_vs_control_seed": "",
            "has_level3_seed": _yn(int(ctrl_result["metrics"]["cycle_level"]) >= 3),
            "mean_pairwise_adoption_prob": "",
            "mean_imitation_step_norm": "",
            "mean_edge_strategy_distance": "",
            "mean_neighbor_payoff_gap": "",
            "mean_local_phase_spread": "",
            "edge_disagreement_rate": "",
            "phase_amplitude_stability": _fmt(ctrl_result["phase_amplitude_stability"]),
            "out_csv": str(ctrl_result["csv_path"]),
            "provenance_json": str(prov_path),
        })

    # Build condition list: cartesian product topology × mu
    active_conditions: list[dict[str, Any]] = []
    for topo in topologies:
        for mu in pairwise_imitation_strengths:
            active_conditions.append({
                "condition": _condition_name(topology=topo, mu=mu),
                "topology": topo,
                "mu": float(mu),
                "graph_spec": _graph_spec_for(topo),
            })

    # --- Active cells ---
    cond_summary_map: dict[str, dict[str, Any]] = {}

    for cond_info in active_conditions:
        condition = str(cond_info["condition"])
        topo = str(cond_info["topology"])
        mu = float(cond_info["mu"])
        graph_spec = cond_info["graph_spec"]
        cond_out_dir = out_root / condition
        cond_out_dir.mkdir(parents=True, exist_ok=True)

        per_seed_metrics: list[dict[str, Any]] = []
        per_seed_diag: list[dict[str, float]] = []

        for seed in seeds:
            config = C1CellConfig(
                condition=condition,
                graph_spec=graph_spec,
                pairwise_imitation_strength=mu,
                pairwise_beta=float(pairwise_beta),
                players=int(players),
                rounds=int(rounds),
                seed=int(seed),
                a=float(a),
                b=float(b),
                cross=float(cross),
                burn_in=int(burn_in),
                tail=int(tail),
                init_bias=float(init_bias),
                memory_kernel=int(memory_kernel),
                out_dir=cond_out_dir,
            )
            result = run_c1_cell(config)
            global_rows = result["global_rows"]
            round_diag = result["round_diag"]

            # Write timeseries CSV
            csv_path = cond_out_dir / f"seed{seed}.csv"
            _write_timeseries_csv(csv_path, global_rows)

            # Compute metrics
            sm = _seed_metrics(global_rows, burn_in=burn_in, tail=tail, eta=eta, corr_threshold=corr_threshold)
            diag = _tail_local_diagnostics(round_diag, n_rows=len(global_rows), burn_in=burn_in, tail=tail)
            pa = _phase_amplitude_stability(global_rows, burn_in=burn_in, tail=tail)

            # Control paired values
            ctrl_m = control_per_seed[seed]["metrics"]
            uplift = float(sm["stage3_score"]) - float(ctrl_m["stage3_score"])

            prov = {
                "family": FAMILY,
                "condition": condition,
                "seed": int(seed),
                "graph_topology": topo,
                "graph_degree": graph_spec.degree,
                "pairwise_imitation_strength": float(mu),
                "pairwise_beta": float(pairwise_beta),
                "out_csv": str(csv_path),
                "cycle_level": int(sm["cycle_level"]),
                "stage3_score": float(sm["stage3_score"]),
                "turn_strength": float(sm["turn_strength"]),
                "env_gamma": float(sm["env_gamma"]),
                "tail_diagnostics": diag,
            }
            prov_path = cond_out_dir / f"seed{seed}_provenance.json"
            prov_path.write_text(json.dumps(prov, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

            per_seed_metrics.append(sm)
            per_seed_diag.append({**diag, "phase_amplitude_stability": pa})

            all_summary_rows.append({
                "gate": GATE_NAME,
                "family": FAMILY,
                "condition": condition,
                "graph_topology": topo,
                "graph_degree": graph_spec.degree,
                "pairwise_imitation_strength": _fmt(mu),
                "pairwise_beta": _fmt(pairwise_beta),
                "local_update_mode": "pairwise_imitation",
                "seed": int(seed),
                "cycle_level": int(sm["cycle_level"]),
                "stage3_score": _fmt(sm["stage3_score"]),
                "turn_strength": _fmt(sm["turn_strength"]),
                "env_gamma": _fmt(sm["env_gamma"]),
                "env_gamma_r2": _fmt(sm["env_gamma_r2"]),
                "env_gamma_n_peaks": int(sm["env_gamma_n_peaks"]),
                "control_env_gamma": _fmt(ctrl_m["env_gamma"]),
                "control_stage3_score": _fmt(ctrl_m["stage3_score"]),
                "stage3_uplift_vs_control_seed": _fmt(uplift),
                "has_level3_seed": _yn(int(sm["cycle_level"]) >= 3),
                "mean_pairwise_adoption_prob": _fmt(diag["mean_pairwise_adoption_prob"]),
                "mean_imitation_step_norm": _fmt(diag["mean_imitation_step_norm"]),
                "mean_edge_strategy_distance": _fmt(diag["mean_edge_strategy_distance"]),
                "mean_neighbor_payoff_gap": _fmt(diag["mean_neighbor_payoff_gap"]),
                "mean_local_phase_spread": _fmt(diag["mean_local_phase_spread"]),
                "edge_disagreement_rate": _fmt(diag["edge_disagreement_rate"]),
                "phase_amplitude_stability": _fmt(pa),
                "out_csv": str(csv_path),
                "provenance_json": str(prov_path),
            })

        # Aggregate per condition
        levels = [int(m["cycle_level"]) for m in per_seed_metrics]
        level_counts = {lv: levels.count(lv) for lv in range(4)}
        n = len(per_seed_metrics)
        mean_s3 = _safe_mean([float(m["stage3_score"]) for m in per_seed_metrics])
        mean_ctrl_s3 = _safe_mean([float(control_per_seed[s]["metrics"]["stage3_score"]) for s in seeds])
        uplift_agg = mean_s3 - mean_ctrl_s3
        mean_gamma = _safe_mean([float(m["env_gamma"]) for m in per_seed_metrics])
        mean_ctrl_gamma = _safe_mean([float(control_per_seed[s]["metrics"]["env_gamma"]) for s in seeds])
        l3_count = sum(1 for l in levels if l >= 3)
        pa_agg = _safe_mean([float(d["phase_amplitude_stability"]) for d in per_seed_diag])
        mean_esd = _safe_mean([float(d["mean_edge_strategy_distance"]) for d in per_seed_diag])
        mean_lps = _safe_mean([float(d["mean_local_phase_spread"]) for d in per_seed_diag])
        mean_adopt = _safe_mean([float(d["mean_pairwise_adoption_prob"]) for d in per_seed_diag])
        mean_step = _safe_mean([float(d["mean_imitation_step_norm"]) for d in per_seed_diag])
        mean_payoff_gap = _safe_mean([float(d["mean_neighbor_payoff_gap"]) for d in per_seed_diag])
        mean_disagree = _safe_mean([float(d["edge_disagreement_rate"]) for d in per_seed_diag])

        verdict = _compute_verdict(
            is_control=False,
            level3_seed_count=l3_count,
            stage3_uplift=uplift_agg,
            mean_env_gamma=mean_gamma,
            phase_amplitude_stability=pa_agg,
            mean_edge_strategy_distance=mean_esd,
            mean_local_phase_spread=mean_lps,
        )
        pass_flag = verdict == "pass"
        hard_stop_flag = (l3_count == 0 and uplift_agg < 0.02 and mean_esd < 0.08)

        cond_summary_map[condition] = {
            "gate": GATE_NAME,
            "family": FAMILY,
            "condition": condition,
            "graph_topology": topo,
            "graph_degree": graph_spec.degree,
            "pairwise_imitation_strength": _fmt(mu),
            "pairwise_beta": _fmt(pairwise_beta),
            "local_update_mode": "pairwise_imitation",
            "is_control": _yn(False),
            "n_seeds": n,
            "mean_cycle_level": _fmt(_safe_mean([float(m["cycle_level"]) for m in per_seed_metrics])),
            "mean_stage3_score": _fmt(mean_s3),
            "mean_turn_strength": _fmt(_safe_mean([float(m["turn_strength"]) for m in per_seed_metrics])),
            "mean_env_gamma": _fmt(mean_gamma),
            "level_counts_json": json.dumps(level_counts, sort_keys=True),
            "p_level_3": _fmt(float(l3_count) / float(n) if n > 0 else 0.0),
            "level3_seed_count": l3_count,
            "control_mean_env_gamma": _fmt(mean_ctrl_gamma),
            "control_mean_stage3_score": _fmt(mean_ctrl_s3),
            "stage3_uplift_vs_control": _fmt(uplift_agg),
            "mean_pairwise_adoption_prob": _fmt(mean_adopt),
            "mean_imitation_step_norm": _fmt(mean_step),
            "mean_edge_strategy_distance": _fmt(mean_esd),
            "mean_neighbor_payoff_gap": _fmt(mean_payoff_gap),
            "mean_local_phase_spread": _fmt(mean_lps),
            "edge_disagreement_rate": _fmt(mean_disagree),
            "phase_amplitude_stability": _fmt(pa_agg),
            "short_scout_pass": _yn(pass_flag),
            "hard_stop_fail": _yn(hard_stop_flag),
            "verdict": verdict,
            "representative_seed": "",
            "representative_simplex_png": "",
            "representative_phase_amplitude_png": "",
            "players": int(players),
            "rounds": int(rounds),
            "out_dir": str(cond_out_dir),
        }

        # Representative seed: best by level then stage3_score
        rep_seed = max(
            seeds,
            key=lambda s: (
                int(control_per_seed[s]["metrics"]["cycle_level"]) if condition == control_cond
                else int(per_seed_metrics[seeds.index(s)]["cycle_level"]),
                float(per_seed_metrics[seeds.index(s)]["stage3_score"]),
            ),
        )
        cond_summary_map[condition]["representative_seed"] = int(rep_seed)

        # Plots for representative seed
        rep_rows_csv = out_root / condition / f"seed{rep_seed}.csv"
        rep_rows_data: list[dict[str, Any]] = []
        with rep_rows_csv.open(encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                rep_rows_data.append(dict(row))

        simplex_png = cond_out_dir / f"simplex_seed{rep_seed}.png"
        phase_png = cond_out_dir / f"phase_amplitude_seed{rep_seed}.png"
        _write_simplex_plot(rep_rows_data, out_png=simplex_png, title=f"{condition} seed={rep_seed}", burn_in=burn_in, tail=tail)
        _write_phase_amplitude_plot(rep_rows_data, out_png=phase_png, title=f"{condition} seed={rep_seed}", burn_in=burn_in, tail=tail)
        cond_summary_map[condition]["representative_simplex_png"] = str(simplex_png)
        cond_summary_map[condition]["representative_phase_amplitude_png"] = str(phase_png)

    # --- Control combined row ---
    ctrl_levels = [int(control_per_seed[s]["metrics"]["cycle_level"]) for s in seeds]
    ctrl_level_counts = {lv: ctrl_levels.count(lv) for lv in range(4)}
    n_ctrl = len(seeds)
    ctrl_combined = {
        "gate": GATE_NAME,
        "family": FAMILY,
        "condition": control_cond,
        "graph_topology": "well_mixed",
        "graph_degree": 0,
        "pairwise_imitation_strength": "",
        "pairwise_beta": "",
        "local_update_mode": "well_mixed_sampled",
        "is_control": _yn(True),
        "n_seeds": n_ctrl,
        "mean_cycle_level": _fmt(_safe_mean([float(control_per_seed[s]["metrics"]["cycle_level"]) for s in seeds])),
        "mean_stage3_score": _fmt(_safe_mean([float(control_per_seed[s]["metrics"]["stage3_score"]) for s in seeds])),
        "mean_turn_strength": _fmt(_safe_mean([float(control_per_seed[s]["metrics"]["turn_strength"]) for s in seeds])),
        "mean_env_gamma": _fmt(_safe_mean([float(control_per_seed[s]["metrics"]["env_gamma"]) for s in seeds])),
        "level_counts_json": json.dumps(ctrl_level_counts, sort_keys=True),
        "p_level_3": _fmt(sum(1 for l in ctrl_levels if l >= 3) / float(n_ctrl) if n_ctrl > 0 else 0.0),
        "level3_seed_count": sum(1 for l in ctrl_levels if l >= 3),
        "control_mean_env_gamma": "",
        "control_mean_stage3_score": "",
        "stage3_uplift_vs_control": "",
        "mean_pairwise_adoption_prob": "",
        "mean_imitation_step_norm": "",
        "mean_edge_strategy_distance": "",
        "mean_neighbor_payoff_gap": "",
        "mean_local_phase_spread": "",
        "edge_disagreement_rate": "",
        "phase_amplitude_stability": _fmt(_safe_mean([float(control_per_seed[s]["phase_amplitude_stability"]) for s in seeds])),
        "short_scout_pass": "",
        "hard_stop_fail": "",
        "verdict": "control",
        "representative_seed": seeds[0],
        "representative_simplex_png": "",
        "representative_phase_amplitude_png": "",
        "players": int(players),
        "rounds": int(rounds),
        "out_dir": str(ctrl_out_dir),
    }
    all_combined_rows = [ctrl_combined] + list(cond_summary_map.values())

    # Write outputs
    _write_tsv(summary_tsv, fieldnames=SUMMARY_FIELDNAMES, rows=all_summary_rows)
    _write_tsv(combined_tsv, fieldnames=COMBINED_FIELDNAMES, rows=all_combined_rows)
    _write_decision(decision_md, combined_rows=all_combined_rows)

    return {
        "summary_tsv": str(summary_tsv),
        "combined_tsv": str(combined_tsv),
        "decision_md": str(decision_md),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="C1 Local Pairwise Imitation short scout")
    parser.add_argument("--seeds", type=str, default="45,47,49")
    parser.add_argument("--topologies", type=str, default="ring4,lattice4")
    parser.add_argument("--pairwise-imitation-strengths", type=str, default="0.10,0.20,0.35")
    parser.add_argument("--pairwise-beta", type=float, default=8.0)
    parser.add_argument("--players", type=int, default=300)
    parser.add_argument("--rounds", type=int, default=3000)
    parser.add_argument("--burn-in", type=int, default=1000)
    parser.add_argument("--tail", type=int, default=1000)
    parser.add_argument("--a", type=float, default=1.0)
    parser.add_argument("--b", type=float, default=0.9)
    parser.add_argument("--matrix-cross-coupling", type=float, default=0.20)
    parser.add_argument("--init-bias", type=float, default=0.12)
    parser.add_argument("--memory-kernel", type=int, default=3)
    parser.add_argument("--selection-strength", type=float, default=0.06,
                        help="Only used for the well-mixed control run")
    parser.add_argument("--out-root", type=Path,
                        default=REPO_ROOT / "outputs" / "c1_local_pairwise_short_scout")
    parser.add_argument("--summary-tsv", type=Path,
                        default=REPO_ROOT / "outputs" / "c1_local_pairwise_short_scout_summary.tsv")
    parser.add_argument("--combined-tsv", type=Path,
                        default=REPO_ROOT / "outputs" / "c1_local_pairwise_short_scout_combined.tsv")
    parser.add_argument("--decision-md", type=Path,
                        default=REPO_ROOT / "outputs" / "c1_local_pairwise_short_scout_decision.md")

    args = parser.parse_args()

    result = run_c1_scout(
        seeds=_parse_seeds(args.seeds),
        topologies=_parse_str_list(args.topologies),
        pairwise_imitation_strengths=_parse_float_list(args.pairwise_imitation_strengths),
        pairwise_beta=float(args.pairwise_beta),
        players=int(args.players),
        rounds=int(args.rounds),
        burn_in=int(args.burn_in),
        tail=int(args.tail),
        a=float(args.a),
        b=float(args.b),
        cross=float(args.matrix_cross_coupling),
        init_bias=float(args.init_bias),
        memory_kernel=int(args.memory_kernel),
        selection_strength=float(args.selection_strength),
        out_root=args.out_root,
        summary_tsv=args.summary_tsv,
        combined_tsv=args.combined_tsv,
        decision_md=args.decision_md,
    )
    print(f"summary_tsv={result['summary_tsv']}")
    print(f"combined_tsv={result['combined_tsv']}")
    print(f"decision_md={result['decision_md']}")


if __name__ == "__main__":
    main()
