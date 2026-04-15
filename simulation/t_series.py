"""T-series: Topology × Update 2×2 harness with random Dirichlet init.

Implements a 2×2 factorial experiment:
  - Topology:  lattice4 (torus 15×20, degree=4)  vs  small_world (WS k=4, p=0.10)
  - Update:    local pairwise imitation (C1 style) vs  local mini-batch replicator (C2 style)

All active cells use random Dirichlet(1,1,1) per-player initialization.
Two controls are run:
  - control_random_init:  well-mixed sampled + random init  (isolates init effect)
  - control_uniform_init: well-mixed sampled + uniform init (original baseline)

Architecture invariants
-----------------------
- Does NOT use global shared growth vector in active cells.
- Does NOT use simulation.run_simulation.simulate() in active cells.
- Update is synchronous: all new weights w(t+1) computed from w(t).
- Per-round CSV row = population-average of player weights/strategies.
- Dispatches to C1/C2 helper functions for the inner update step.

Usage
-----
./venv/bin/python -m simulation.t_series \\
  --seeds 45,47,49 \\
  --topologies lattice4,small_world \\
  --update-modes pairwise,minibatch \\
  --p-rewire 0.10 \\
  --pairwise-imitation-strength 0.35 \\
  --pairwise-beta 8.0 \\
  --local-selection-strength 0.08 \\
  --players 300 --rounds 3000 \\
  --burn-in 1000 --tail 1000 \\
  --a 1.0 --b 0.9 --matrix-cross-coupling 0.20 \\
  --random-init \\
  --out-root outputs/t_series_short_scout \\
  --summary-tsv outputs/t_series_short_scout_summary.tsv \\
  --combined-tsv outputs/t_series_short_scout_combined.tsv \\
  --decision-md outputs/t_series_short_scout_decision.md
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from dataclasses import dataclass
from math import atan2, isfinite, log, pi, sqrt
from pathlib import Path
from typing import Any

from analysis.cycle_metrics import classify_cycle_level
from analysis.decay_rate import estimate_decay_gamma
from evolution.init_weights import (
    init_weight_dispersion,
    random_simplex_weights,
)
from evolution.local_graph import (
    GraphSpec,
    build_graph,
    edge_disagreement_rate,
    edge_strategy_distance,
    graph_clustering_coefficient,
    graph_mean_shortest_path_approx,
    sample_neighbor,
    spatial_autocorrelation,
)
from evolution.local_pairwise import (
    local_payoff,
    pairwise_adoption_probability,
    pairwise_imitation_update,
)
from evolution.local_replicator import (
    STRATEGY_SPACE,
    batch_phase_spread,
    list_to_weights,
    local_growth_cosine_vs_global,
    local_growth_norm,
    local_popularity,
    local_replicator_advantage,
    local_replicator_update,
    neighbor_popularity_dispersion,
    player_growth_dispersion,
    weights_to_simplex,
)
from players.base_player import BasePlayer
from simulation.run_simulation import _initial_weights

REPO_ROOT = Path(__file__).resolve().parents[1]

GATE_NAME = "g2_sampled"
FAMILY = "t_series"

SUMMARY_FIELDNAMES = [
    "gate", "family", "condition",
    "graph_topology", "graph_degree", "p_rewire",
    "update_mode",
    "pairwise_imitation_strength", "pairwise_beta",
    "local_selection_strength",
    "init_mode",
    "seed",
    "cycle_level", "stage3_score", "turn_strength",
    "env_gamma", "env_gamma_r2", "env_gamma_n_peaks",
    "control_env_gamma", "control_stage3_score",
    "stage3_uplift_vs_control_seed", "has_level3_seed",
    "init_weight_dispersion",
    "graph_clustering_coefficient", "graph_mean_path_length",
    "spatial_autocorrelation_d1", "spatial_autocorrelation_d2",
    "mean_batch_phase_spread",
    "mean_edge_strategy_distance",
    "edge_disagreement_rate",
    "mean_local_growth_cosine_vs_global",
    "mean_player_growth_dispersion",
    "phase_amplitude_stability",
    "out_csv", "provenance_json",
]

COMBINED_FIELDNAMES = [
    "gate", "family", "condition",
    "graph_topology", "graph_degree", "p_rewire",
    "update_mode",
    "pairwise_imitation_strength", "pairwise_beta",
    "local_selection_strength",
    "init_mode",
    "is_control", "n_seeds",
    "mean_cycle_level", "mean_stage3_score", "mean_turn_strength",
    "mean_env_gamma",
    "level_counts_json", "p_level_3", "level3_seed_count",
    "control_mean_env_gamma", "control_mean_stage3_score",
    "stage3_uplift_vs_control",
    "init_weight_dispersion",
    "graph_clustering_coefficient", "graph_mean_path_length",
    "spatial_autocorrelation_d1", "spatial_autocorrelation_d2",
    "spatial_autocorrelation_decay",
    "mean_batch_phase_spread",
    "mean_edge_strategy_distance",
    "edge_disagreement_rate",
    "mean_local_growth_cosine_vs_global",
    "mean_player_growth_dispersion",
    "phase_amplitude_stability",
    "short_scout_pass", "hard_stop_fail", "verdict",
    "representative_seed",
    "representative_simplex_png", "representative_phase_amplitude_png",
    "players", "rounds", "out_dir",
]


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TCellConfig:
    condition: str
    graph_spec: GraphSpec
    update_mode: str            # "pairwise" | "minibatch"
    # pairwise params
    pairwise_imitation_strength: float
    pairwise_beta: float
    # minibatch params
    local_selection_strength: float
    # common
    players: int
    rounds: int
    seed: int
    a: float
    b: float
    cross: float
    burn_in: int
    tail: int
    random_init: bool
    init_bias: float
    memory_kernel: int
    mutation_rate: float
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

def _condition_name(*, topology: str, update_mode: str) -> str:
    return f"t_{topology}_{update_mode}"

def _graph_spec_for(topology: str, *, p_rewire: float = 0.0) -> GraphSpec:
    if topology == "lattice4":
        return GraphSpec(topology="lattice4", degree=4, lattice_rows=15, lattice_cols=20)
    if topology == "small_world":
        return GraphSpec(topology="small_world", degree=4, p_rewire=p_rewire)
    if topology == "ring4":
        return GraphSpec(topology="ring4", degree=4)
    raise ValueError(f"Unsupported topology: {topology!r}")

def _tail_begin(n_rows: int, *, burn_in: int, tail: int) -> int:
    return max(int(burn_in), n_rows - int(tail))

# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def _simplex_xy(p_a: float, p_d: float, p_b: float) -> tuple[float, float]:
    return (p_d - p_b, p_a - 0.5 * (p_d + p_b))

def _phase_angle(p_a: float, p_d: float, p_b: float) -> float:
    return float(atan2(sqrt(3.0) * (p_d - p_b), 2.0 * p_a - p_d - p_b))

def _amplitude(p_a: float, p_d: float, p_b: float) -> float:
    return float(((p_a - 1.0/3.0)**2 + (p_d - 1.0/3.0)**2 + (p_b - 1.0/3.0)**2)**0.5)


# ---------------------------------------------------------------------------
# Core simulation loop (dispatches to C1 or C2 inner update)
# ---------------------------------------------------------------------------

def run_t_cell(config: TCellConfig) -> dict[str, Any]:
    """Run one T-series cell and return rows + diagnostics."""
    n = int(config.players)
    mode = str(config.update_mode)

    # Per-player RNGs
    player_rngs = [random.Random(int(config.seed) + i) for i in range(n)]

    # Initialise players
    players = [BasePlayer(STRATEGY_SPACE, rng=player_rngs[i]) for i in range(n)]

    if config.random_init:
        for i, pl in enumerate(players):
            w_dict = random_simplex_weights(
                STRATEGY_SPACE, seed=int(config.seed), player_index=i,
            )
            pl.update_weights(w_dict)
    else:
        init_w_dict = _initial_weights(
            strategy_space=STRATEGY_SPACE,
            init_bias=float(config.init_bias),
        )
        for pl in players:
            pl.update_weights(init_w_dict)

    # Build graph
    adj = build_graph(n, config.graph_spec, int(config.seed))

    # Graph-level diagnostics (computed once)
    g_cc = graph_clustering_coefficient(adj)
    g_path = graph_mean_shortest_path_approx(adj, sample_size=50, seed=int(config.seed))

    # Pre-compute batch indices for minibatch mode
    batch_indices: list[list[int]] = [[i] + list(adj[i]) for i in range(n)]

    # Initial weight dispersion
    init_simplex = [weights_to_simplex(pl.strategy_weights) for pl in players]
    iwd = init_weight_dispersion(init_simplex)

    global_rows: list[dict[str, Any]] = []
    round_diag: list[dict[str, Any]] = []

    for t in range(int(config.rounds)):
        # --- Lock w(t) as simplex (synchronous) ---
        simplex_t = [weights_to_simplex(pl.strategy_weights) for pl in players]

        # --- Each player samples a strategy ---
        for pl in players:
            pl.choose_strategy()
        strategies_t = [str(pl.last_strategy) for pl in players]

        # --- Dispatch to update mode ---
        if mode == "pairwise":
            new_simplex, diag_t = _pairwise_update_step(
                config, players, player_rngs, adj, simplex_t, strategies_t, n,
            )
        elif mode == "minibatch":
            new_simplex, diag_t = _minibatch_update_step(
                config, adj, batch_indices, simplex_t, strategies_t, n,
            )
        else:
            raise ValueError(f"Unknown update_mode: {mode!r}")

        # --- Per-player mutation (Dirichlet mixing, if enabled) ---
        if config.mutation_rate > 0.0:
            new_simplex = [
                _apply_mutation(new_simplex[i], player_rngs[i], float(config.mutation_rate))
                for i in range(n)
            ]

        # --- Broadcast new weights ---
        for i, pl in enumerate(players):
            pl.update_weights(list_to_weights(new_simplex[i]))

        # --- Per-round global observation ---
        strat_counts = {s: 0 for s in STRATEGY_SPACE}
        for s in strategies_t:
            strat_counts[s] += 1
        p_row = {s: float(strat_counts[s]) / float(n) for s in STRATEGY_SPACE}
        w_mean = {s: _safe_mean([new_simplex[i][k] for i in range(n)])
                  for k, s in enumerate(STRATEGY_SPACE)}

        row: dict[str, Any] = {"round": t}
        for s in STRATEGY_SPACE:
            row[f"p_{s}"] = float(p_row[s])
        for s in STRATEGY_SPACE:
            row[f"w_{s}"] = float(w_mean[s])
        global_rows.append(row)

        # --- Common per-round diagnostics ---
        e_disagree = edge_disagreement_rate(adj, strategies_t)
        e_dist = edge_strategy_distance(adj, new_simplex)
        bps = batch_phase_spread(adj, new_simplex)

        diag_t["edge_disagreement_rate"] = e_disagree
        diag_t["mean_edge_strategy_distance"] = e_dist
        diag_t["mean_batch_phase_spread"] = bps
        round_diag.append(diag_t)

    # Tail-window spatial autocorrelation
    begin = _tail_begin(len(global_rows), burn_in=int(config.burn_in), tail=int(config.tail))
    final_simplex = [weights_to_simplex(pl.strategy_weights) for pl in players]
    sac = spatial_autocorrelation(adj, final_simplex, max_hop=2)

    return {
        "global_rows": global_rows,
        "round_diag": round_diag,
        "graph_clustering_coefficient": g_cc,
        "graph_mean_path_length": g_path,
        "init_weight_dispersion": iwd,
        "spatial_autocorrelation_d1": sac.get(1, 0.0),
        "spatial_autocorrelation_d2": sac.get(2, 0.0),
    }


def _apply_mutation(simplex: list[float], rng: random.Random, eta: float) -> list[float]:
    """Mix current weights with Dirichlet(1,1,1) sample: w' = (1-eta)*w + eta*u.

    This implements continuous mutation in strategy space, maintaining the
    simplex constraint while injecting per-player heterogeneity each round.
    """
    k = len(simplex)
    # Dirichlet(1,1,1) via Gamma(1,1) = Exponential(1) = -log(U)
    u = [-log(max(1e-300, rng.random())) for _ in range(k)]
    total = sum(u)
    u_norm = [v / total for v in u]
    return [(1.0 - eta) * simplex[j] + eta * u_norm[j] for j in range(k)]


# ---------------------------------------------------------------------------
# C1-style pairwise inner update
# ---------------------------------------------------------------------------

def _pairwise_update_step(
    config: TCellConfig,
    players: list[BasePlayer],
    player_rngs: list[random.Random],
    adj: list[list[int]],
    simplex_t: list[list[float]],
    strategies_t: list[str],
    n: int,
) -> tuple[list[list[float]], dict[str, Any]]:
    """One round of C1-style pairwise imitation update. Returns (new_simplex, diag)."""
    # Compute local payoffs
    payoffs_t: list[float] = []
    for i in range(n):
        nbr_strats = [strategies_t[j] for j in adj[i]]
        pi_i = local_payoff(
            simplex_t[i], nbr_strats,
            a=float(config.a), b=float(config.b), cross=float(config.cross),
        )
        payoffs_t.append(pi_i)
        players[i].update_utility(pi_i)
        players[i].last_reward = pi_i

    new_simplex: list[list[float]] = []
    adoption_probs: list[float] = []
    step_norms: list[float] = []

    for i in range(n):
        j = sample_neighbor(player_rngs[i], adj, i)
        q = pairwise_adoption_probability(
            payoffs_t[i], payoffs_t[j],
            beta_pair=float(config.pairwise_beta),
        )
        adoption_probs.append(q)
        w_i_new = pairwise_imitation_update(
            simplex_t[i], simplex_t[j],
            mu=float(config.pairwise_imitation_strength), q=q,
        )
        step_norm = sum(abs(nw - ow) for nw, ow in zip(w_i_new, simplex_t[i]))
        step_norms.append(step_norm)
        new_simplex.append(w_i_new)

    diag: dict[str, Any] = {
        "mean_pairwise_adoption_prob": _safe_mean(adoption_probs),
        "mean_imitation_step_norm": _safe_mean(step_norms),
        "mean_local_growth_cosine_vs_global": "",  # N/A for pairwise
        "mean_player_growth_dispersion": "",
    }
    return new_simplex, diag


# ---------------------------------------------------------------------------
# C2-style minibatch inner update
# ---------------------------------------------------------------------------

def _minibatch_update_step(
    config: TCellConfig,
    adj: list[list[int]],
    batch_indices: list[list[int]],
    simplex_t: list[list[float]],
    strategies_t: list[str],
    n: int,
) -> tuple[list[list[float]], dict[str, Any]]:
    """One round of C2-style mini-batch replicator update. Returns (new_simplex, diag)."""
    # Global popularity (for cosine diagnostic)
    global_pop = [_safe_mean([simplex_t[i][k] for i in range(n)]) for k in range(3)]
    global_adv = local_replicator_advantage(
        global_pop, global_pop,
        a=float(config.a), b=float(config.b), cross=float(config.cross),
    )

    new_simplex: list[list[float]] = []
    adv_list: list[list[float]] = []
    step_norms: list[float] = []
    cosines: list[float] = []

    for i in range(n):
        loc_pop = local_popularity(simplex_t, batch_indices[i])
        adv_i = local_replicator_advantage(
            simplex_t[i], loc_pop,
            a=float(config.a), b=float(config.b), cross=float(config.cross),
        )
        adv_list.append(adv_i)
        cos = local_growth_cosine_vs_global(adv_i, global_adv)
        cosines.append(cos)
        w_i_new = local_replicator_update(
            simplex_t[i], adv_i,
            k_local=float(config.local_selection_strength),
        )
        step_norm = sum(abs(nw - ow) for nw, ow in zip(w_i_new, simplex_t[i]))
        step_norms.append(step_norm)
        new_simplex.append(w_i_new)

    diag: dict[str, Any] = {
        "mean_local_growth_cosine_vs_global": _safe_mean(cosines),
        "mean_player_growth_dispersion": player_growth_dispersion(adv_list),
        "mean_pairwise_adoption_prob": "",  # N/A for minibatch
        "mean_imitation_step_norm": _safe_mean(step_norms),
    }
    return new_simplex, diag


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def _seed_metrics(
    rows: list[dict[str, Any]],
    *, burn_in: int, tail: int,
    eta: float = 0.55, corr_threshold: float = 0.09,
) -> dict[str, Any]:
    series_map = {s: [float(row[f"p_{s}"]) for row in rows] for s in STRATEGY_SPACE}
    cycle = classify_cycle_level(
        series_map, burn_in=int(burn_in), tail=int(tail),
        amplitude_threshold=0.02, corr_threshold=float(corr_threshold),
        eta=float(eta), stage3_method="turning", phase_smoothing=1,
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


def _tail_diagnostics(
    round_diag: list[dict[str, Any]],
    *, n_rows: int, burn_in: int, tail: int,
) -> dict[str, float]:
    begin = _tail_begin(n_rows, burn_in=burn_in, tail=tail)
    window = round_diag[begin:]
    keys = [
        "mean_batch_phase_spread",
        "mean_edge_strategy_distance",
        "edge_disagreement_rate",
        "mean_local_growth_cosine_vs_global",
        "mean_player_growth_dispersion",
    ]
    if not window:
        return {k: 0.0 for k in keys}
    result: dict[str, float] = {}
    for key in keys:
        vals = [float(d[key]) for d in window if d.get(key, "") != ""]
        result[key] = _safe_mean(vals) if vals else 0.0
    return result


# ---------------------------------------------------------------------------
# Verdict
# ---------------------------------------------------------------------------

def _compute_verdict(
    *,
    is_control: bool,
    level3_seed_count: int,
    stage3_uplift: float,
    mean_env_gamma: float,
    phase_amplitude_stability: float,
    mean_local_growth_cosine_vs_global: float,
    mean_player_growth_dispersion: float,
    mean_batch_phase_spread: float,
) -> str:
    if is_control:
        return "control"
    # Pass
    if (level3_seed_count >= 1
            and abs(mean_env_gamma) <= 5e-4
            and phase_amplitude_stability >= 0.3):
        return "pass"
    # Weak positive
    if level3_seed_count >= 1 and abs(mean_env_gamma) > 5e-4:
        return "weak_positive"
    if stage3_uplift >= 0.02:
        return "weak_positive"
    if mean_local_growth_cosine_vs_global <= 0.85 and mean_player_growth_dispersion >= 0.03:
        return "weak_positive"
    if mean_batch_phase_spread >= 0.15:
        return "weak_positive"
    return "fail"


# ---------------------------------------------------------------------------
# Control runs
# ---------------------------------------------------------------------------

def _run_control_wellmixed(
    *, seed: int, players: int, rounds: int, burn_in: int, tail: int,
    a: float, b: float, cross: float,
    init_bias: float, memory_kernel: int, selection_strength: float,
    random_init: bool, out_dir: Path,
) -> dict[str, Any]:
    """Run well-mixed sampled baseline.

    If random_init=True, uses random Dirichlet init via the standard simulate()
    path but with a custom init bias of 0.0 (uniform from _initial_weights)
    and then manually overrides weights — but actually, simulate() doesn't
    support per-player random init. So for random_init control, we build
    a simple inner loop right here (same global update semantics).
    """
    from simulation.run_simulation import SimConfig, simulate

    if not random_init:
        # Standard well-mixed sampled
        cfg = SimConfig(
            n_players=int(players), n_rounds=int(rounds), seed=int(seed),
            payoff_mode="matrix_ab", popularity_mode="sampled",
            gamma=0.1, epsilon=0.0,
            a=float(a), b=float(b), matrix_cross_coupling=float(cross),
            init_bias=float(init_bias),
            evolution_mode="sampled", payoff_lag=1,
            selection_strength=float(selection_strength),
            memory_kernel=int(memory_kernel),
        )
        _ss, rows = simulate(cfg)
        csv_path = out_dir / f"seed{seed}.csv"
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        with csv_path.open("w", newline="", encoding="utf-8") as fh:
            fieldnames = list(rows[0].keys()) if rows else []
            writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
        metrics = _seed_metrics(rows, burn_in=burn_in, tail=tail)
        pa = _phase_amplitude_stability(rows, burn_in=burn_in, tail=tail)
        return {"rows": rows, "csv_path": csv_path, "metrics": metrics,
                "phase_amplitude_stability": pa, "init_weight_dispersion": 0.0}

    # --- Random init control: well-mixed + global sampled update ---
    # Replicate simulate() logic but with per-player Dirichlet init
    from simulation.run_simulation import _matrix_ab_payoff_vec

    n = int(players)
    player_rngs = [random.Random(int(seed) + i) for i in range(n)]
    pls = [BasePlayer(STRATEGY_SPACE, rng=player_rngs[i]) for i in range(n)]

    for i, pl in enumerate(pls):
        w_dict = random_simplex_weights(STRATEGY_SPACE, seed=int(seed), player_index=i)
        pl.update_weights(w_dict)

    init_simplex = [weights_to_simplex(pl.strategy_weights) for pl in pls]
    iwd = init_weight_dispersion(init_simplex)

    rows: list[dict[str, Any]] = []
    for t in range(int(rounds)):
        # Choose strategies
        for pl in pls:
            pl.choose_strategy()
        strategies_t = [str(pl.last_strategy) for pl in pls]

        # Global popularity from current weights
        simplex_t = [weights_to_simplex(pl.strategy_weights) for pl in pls]
        global_x = [_safe_mean([simplex_t[i][k] for i in range(n)]) for k in range(3)]

        # Global payoff vector (dict[str, float])
        x_dict = {s: global_x[k] for k, s in enumerate(STRATEGY_SPACE)}
        u_dict = _matrix_ab_payoff_vec(
            strategy_space=STRATEGY_SPACE, a=float(a), b=float(b),
            matrix_cross_coupling=float(cross), x=x_dict,
        )
        u_vals = [float(u_dict[s]) for s in STRATEGY_SPACE]

        # Update each player's weights using global sampled replicator
        for i, pl in enumerate(pls):
            wi = simplex_t[i]
            # replicator: w_k' ∝ w_k * exp(selection_strength * u_k)
            from math import exp
            raw = [wi[k] * exp(max(-500, min(500, float(selection_strength) * u_vals[k]))) for k in range(3)]
            total = sum(raw)
            new_w = [v / total for v in raw]
            pl.update_weights(list_to_weights(new_w))

        # Record
        strat_counts = {s: 0 for s in STRATEGY_SPACE}
        for s in strategies_t:
            strat_counts[s] += 1
        new_simplex = [weights_to_simplex(pl.strategy_weights) for pl in pls]
        w_mean = {s: _safe_mean([new_simplex[i][k] for i in range(n)])
                  for k, s in enumerate(STRATEGY_SPACE)}
        row: dict[str, Any] = {"round": t}
        for s in STRATEGY_SPACE:
            row[f"p_{s}"] = float(strat_counts[s]) / float(n)
        for s in STRATEGY_SPACE:
            row[f"w_{s}"] = float(w_mean[s])
        rows.append(row)

    csv_path = out_dir / f"seed{seed}.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    _write_timeseries_csv(csv_path, rows)
    metrics = _seed_metrics(rows, burn_in=burn_in, tail=tail)
    pa = _phase_amplitude_stability(rows, burn_in=burn_in, tail=tail)
    return {"rows": rows, "csv_path": csv_path, "metrics": metrics,
            "phase_amplitude_stability": pa, "init_weight_dispersion": iwd}


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

def _write_simplex_plot(rows, *, out_png, title, burn_in, tail):
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

def _write_phase_amplitude_plot(rows, *, out_png, title, burn_in, tail):
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
    phases = np.unwrap(np.array([_phase_angle(float(r["p_aggressive"]), float(r["p_defensive"]), float(r["p_balanced"])) for r in window]))
    amps = np.array([_amplitude(float(r["p_aggressive"]), float(r["p_defensive"]), float(r["p_balanced"])) for r in window])
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
# Decision markdown
# ---------------------------------------------------------------------------

def _write_decision(path: Path, *, combined_rows: list[dict[str, Any]]) -> None:
    lines = [
        "# T-series: Topology × Update 2×2 — Short Scout Decision",
        "",
        "## G2 Short Scout",
        "",
    ]
    control_rows = [r for r in combined_rows if str(r.get("is_control", "")) == "yes"]
    active_rows = [r for r in combined_rows if str(r.get("is_control", "")) != "yes"]

    for row in control_rows + active_rows:
        is_ctrl = str(row.get("is_control", "")) == "yes"
        uplift_str = "" if is_ctrl else f"  stage3_uplift={row.get('stage3_uplift_vs_control', '')}"
        lines.append(
            f"condition: {row['condition']}"
            + (f"  graph={row['graph_topology']}"
               f"  update={row['update_mode']}"
               f"  init={row['init_mode']}"
               if not is_ctrl else f"  init={row['init_mode']}")
            + f"  seeds={row['n_seeds']}"
            f"  lv3={row['level3_seed_count']}"
            + uplift_str
            + f"  gamma={row['mean_env_gamma']}"
            f"  pa_stab={row['phase_amplitude_stability']}"
            f"  iwd={row.get('init_weight_dispersion', '')}"
            f"  cosine={row.get('mean_local_growth_cosine_vs_global', '')}"
            f"  pgd={row.get('mean_player_growth_dispersion', '')}"
            f"  bps={row.get('mean_batch_phase_spread', '')}"
            f"  sac_d1={row.get('spatial_autocorrelation_d1', '')}"
            f"  sac_d2={row.get('spatial_autocorrelation_d2', '')}"
            f"  verdict={row['verdict']}"
        )

    # Mechanism signal
    lines.extend(["", "## Mechanism Signal", ""])

    for row in active_rows:
        topo = str(row['graph_topology'])
        umode = str(row['update_mode'])
        iwd = float(row.get('init_weight_dispersion', 0.0))
        bps = float(row.get('mean_batch_phase_spread', 0.0))
        sac1 = float(row.get('spatial_autocorrelation_d1', 0.0))
        sac2 = float(row.get('spatial_autocorrelation_d2', 0.0))
        sac_decay = f"{sac1/sac2:.2f}" if sac2 > 1e-9 else "inf"
        lines.append(
            f"- {row['condition']}: iwd={iwd:.4f}  bps={bps:.4f} rad"
            f"  sac_d1={sac1:.6f}  sac_d2={sac2:.6f}  decay_ratio={sac_decay}"
        )

    # Overall verdict
    lines.extend(["", "## Overall Verdict", ""])

    any_pass = any(str(r.get("verdict", "")) == "pass" for r in active_rows)
    any_weak = any(str(r.get("verdict", "")) == "weak_positive" for r in active_rows)
    all_fail = all(str(r.get("verdict", "")) in ("fail",) for r in active_rows)

    hard_stop = (
        all(int(r.get("level3_seed_count", 0)) == 0 for r in active_rows)
        and all(
            float(r.get("stage3_uplift_vs_control", 0)) < 0.02
            for r in active_rows
        )
    )

    if any_pass:
        lines.append("- overall_verdict: pass_t")
        lines.append("- 建議開 longer_confirm（seeds=0:29）")
    elif any_weak:
        lines.append("- overall_verdict: weak_positive_t")
        lines.append("- 建議 targeted follow-up（最多 9 runs）")
    elif hard_stop:
        lines.append("- overall_verdict: close_t")
        lines.append("- hard_stop: 所有 4 active cells（2 topologies × 2 update modes）0/3 Level 3 + uplift < 0.02")
        lines.append("- 結論：即使初始空間異質性（random Dirichlet init）+ local clustering + local update，")
        lines.append("  仍無法讓 phase domain 存活至全局可見的 Level 3 basin")
        lines.append("- 與 B/C-series 負結果共同指向：sampled discrete update 本身是 Level 2 plateau 的結構性瓶頸")
    else:
        lines.append("- overall_verdict: close_t")
        lines.append("- 所有 active cells 未達標，T-series closure")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# Main scout
# ---------------------------------------------------------------------------

def run_t_scout(
    *,
    seeds: list[int],
    topologies: list[str],
    update_modes: list[str],
    p_rewire: float,
    pairwise_imitation_strength: float,
    pairwise_beta: float,
    local_selection_strength: float,
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
    random_init: bool,
    out_root: Path,
    summary_tsv: Path,
    combined_tsv: Path,
    decision_md: Path,
) -> dict[str, Any]:
    out_root.mkdir(parents=True, exist_ok=True)
    all_summary_rows: list[dict[str, Any]] = []

    # --- Control 1: well-mixed + random init ---
    ctrl_ri_cond = "control_random_init"
    ctrl_ri_dir = out_root / ctrl_ri_cond
    ctrl_ri_dir.mkdir(parents=True, exist_ok=True)
    ctrl_ri_per_seed: dict[int, dict[str, Any]] = {}
    for seed in seeds:
        r = _run_control_wellmixed(
            seed=seed, players=players, rounds=rounds, burn_in=burn_in, tail=tail,
            a=a, b=b, cross=cross, init_bias=init_bias, memory_kernel=memory_kernel,
            selection_strength=selection_strength, random_init=True, out_dir=ctrl_ri_dir,
        )
        ctrl_ri_per_seed[seed] = r
        all_summary_rows.append(_ctrl_summary_row(
            ctrl_ri_cond, "random_init", seed, r,
        ))

    # --- Control 2: well-mixed + uniform init ---
    ctrl_ui_cond = "control_uniform_init"
    ctrl_ui_dir = out_root / ctrl_ui_cond
    ctrl_ui_dir.mkdir(parents=True, exist_ok=True)
    ctrl_ui_per_seed: dict[int, dict[str, Any]] = {}
    for seed in seeds:
        r = _run_control_wellmixed(
            seed=seed, players=players, rounds=rounds, burn_in=burn_in, tail=tail,
            a=a, b=b, cross=cross, init_bias=init_bias, memory_kernel=memory_kernel,
            selection_strength=selection_strength, random_init=False, out_dir=ctrl_ui_dir,
        )
        ctrl_ui_per_seed[seed] = r
        all_summary_rows.append(_ctrl_summary_row(
            ctrl_ui_cond, "uniform_init", seed, r,
        ))

    # Use random-init control as the primary comparison baseline
    control_per_seed = ctrl_ri_per_seed

    # --- Active cells: topology × update_mode ---
    active_conditions: list[dict[str, Any]] = []
    for topo in topologies:
        for umode in update_modes:
            active_conditions.append({
                "condition": _condition_name(topology=topo, update_mode=umode),
                "topology": topo,
                "update_mode": umode,
                "graph_spec": _graph_spec_for(topo, p_rewire=p_rewire),
            })

    cond_summary_map: dict[str, dict[str, Any]] = {}

    for cond_info in active_conditions:
        condition = str(cond_info["condition"])
        topo = str(cond_info["topology"])
        umode = str(cond_info["update_mode"])
        graph_spec = cond_info["graph_spec"]
        cond_out_dir = out_root / condition
        cond_out_dir.mkdir(parents=True, exist_ok=True)

        per_seed_metrics: list[dict[str, Any]] = []
        per_seed_extra: list[dict[str, float]] = []

        for seed in seeds:
            config = TCellConfig(
                condition=condition, graph_spec=graph_spec,
                update_mode=umode,
                pairwise_imitation_strength=pairwise_imitation_strength,
                pairwise_beta=pairwise_beta,
                local_selection_strength=local_selection_strength,
                players=players, rounds=rounds, seed=seed,
                a=a, b=b, cross=cross,
                burn_in=burn_in, tail=tail,
                random_init=random_init, init_bias=init_bias,
                memory_kernel=memory_kernel, mutation_rate=0.0, out_dir=cond_out_dir,
            )
            result = run_t_cell(config)
            global_rows = result["global_rows"]
            round_diag = result["round_diag"]

            csv_path = cond_out_dir / f"seed{seed}.csv"
            _write_timeseries_csv(csv_path, global_rows)

            sm = _seed_metrics(global_rows, burn_in=burn_in, tail=tail)
            diag = _tail_diagnostics(round_diag, n_rows=len(global_rows), burn_in=burn_in, tail=tail)
            pa = _phase_amplitude_stability(global_rows, burn_in=burn_in, tail=tail)

            ctrl_m = control_per_seed[seed]["metrics"]
            uplift = float(sm["stage3_score"]) - float(ctrl_m["stage3_score"])

            prov = {
                "family": FAMILY, "condition": condition, "seed": int(seed),
                "graph_topology": topo, "graph_degree": graph_spec.degree,
                "p_rewire": graph_spec.p_rewire,
                "update_mode": umode,
                "init_mode": "random_init" if random_init else "uniform_init",
                "out_csv": str(csv_path),
                "cycle_level": int(sm["cycle_level"]),
                "stage3_score": float(sm["stage3_score"]),
            }
            prov_path = cond_out_dir / f"seed{seed}_provenance.json"
            prov_path.write_text(json.dumps(prov, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

            per_seed_metrics.append(sm)
            extra = {
                **diag,
                "phase_amplitude_stability": pa,
                "init_weight_dispersion": result["init_weight_dispersion"],
                "graph_clustering_coefficient": result["graph_clustering_coefficient"],
                "graph_mean_path_length": result["graph_mean_path_length"],
                "spatial_autocorrelation_d1": result["spatial_autocorrelation_d1"],
                "spatial_autocorrelation_d2": result["spatial_autocorrelation_d2"],
                "uplift": uplift,
            }
            per_seed_extra.append(extra)

            all_summary_rows.append({
                "gate": GATE_NAME, "family": FAMILY, "condition": condition,
                "graph_topology": topo, "graph_degree": graph_spec.degree,
                "p_rewire": _fmt(graph_spec.p_rewire),
                "update_mode": umode,
                "pairwise_imitation_strength": _fmt(pairwise_imitation_strength) if umode == "pairwise" else "",
                "pairwise_beta": _fmt(pairwise_beta) if umode == "pairwise" else "",
                "local_selection_strength": _fmt(local_selection_strength) if umode == "minibatch" else "",
                "init_mode": "random_init" if random_init else "uniform_init",
                "seed": seed,
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
                "init_weight_dispersion": _fmt(result["init_weight_dispersion"]),
                "graph_clustering_coefficient": _fmt(result["graph_clustering_coefficient"]),
                "graph_mean_path_length": _fmt(result["graph_mean_path_length"]),
                "spatial_autocorrelation_d1": _fmt(result["spatial_autocorrelation_d1"]),
                "spatial_autocorrelation_d2": _fmt(result["spatial_autocorrelation_d2"]),
                "mean_batch_phase_spread": _fmt(diag["mean_batch_phase_spread"]),
                "mean_edge_strategy_distance": _fmt(diag["mean_edge_strategy_distance"]),
                "edge_disagreement_rate": _fmt(diag["edge_disagreement_rate"]),
                "mean_local_growth_cosine_vs_global": _fmt(diag["mean_local_growth_cosine_vs_global"]) if diag["mean_local_growth_cosine_vs_global"] else "",
                "mean_player_growth_dispersion": _fmt(diag["mean_player_growth_dispersion"]) if diag["mean_player_growth_dispersion"] else "",
                "phase_amplitude_stability": _fmt(pa),
                "out_csv": str(csv_path),
                "provenance_json": str(prov_path),
            })

        # --- Aggregate ---
        levels = [int(m["cycle_level"]) for m in per_seed_metrics]
        level_counts = {lv: levels.count(lv) for lv in range(4)}
        n_cond = len(per_seed_metrics)
        mean_s3 = _safe_mean([float(m["stage3_score"]) for m in per_seed_metrics])
        mean_ctrl_s3 = _safe_mean([float(control_per_seed[s]["metrics"]["stage3_score"]) for s in seeds])
        uplift_agg = mean_s3 - mean_ctrl_s3
        mean_gamma = _safe_mean([float(m["env_gamma"]) for m in per_seed_metrics])
        mean_ctrl_gamma = _safe_mean([float(control_per_seed[s]["metrics"]["env_gamma"]) for s in seeds])
        l3_count = sum(1 for l in levels if l >= 3)

        def _agg(key: str) -> float:
            vals = [float(e[key]) for e in per_seed_extra if e.get(key, "") != ""]
            return _safe_mean(vals) if vals else 0.0

        pa_agg = _agg("phase_amplitude_stability")
        cos_agg = _agg("mean_local_growth_cosine_vs_global")
        pgd_agg = _agg("mean_player_growth_dispersion")
        bps_agg = _agg("mean_batch_phase_spread")
        sac1_agg = _agg("spatial_autocorrelation_d1")
        sac2_agg = _agg("spatial_autocorrelation_d2")
        sac_decay = sac1_agg / sac2_agg if abs(sac2_agg) > 1e-12 else 0.0

        verdict = _compute_verdict(
            is_control=False, level3_seed_count=l3_count,
            stage3_uplift=uplift_agg, mean_env_gamma=mean_gamma,
            phase_amplitude_stability=pa_agg,
            mean_local_growth_cosine_vs_global=cos_agg,
            mean_player_growth_dispersion=pgd_agg,
            mean_batch_phase_spread=bps_agg,
        )
        pass_flag = verdict == "pass"
        hard_stop_flag = l3_count == 0 and uplift_agg < 0.02

        rep_seed = max(seeds, key=lambda s: (
            int(per_seed_metrics[seeds.index(s)]["cycle_level"]),
            float(per_seed_metrics[seeds.index(s)]["stage3_score"]),
        ))

        cond_summary_map[condition] = {
            "gate": GATE_NAME, "family": FAMILY, "condition": condition,
            "graph_topology": topo, "graph_degree": graph_spec.degree,
            "p_rewire": _fmt(graph_spec.p_rewire),
            "update_mode": umode,
            "pairwise_imitation_strength": _fmt(pairwise_imitation_strength) if umode == "pairwise" else "",
            "pairwise_beta": _fmt(pairwise_beta) if umode == "pairwise" else "",
            "local_selection_strength": _fmt(local_selection_strength) if umode == "minibatch" else "",
            "init_mode": "random_init" if random_init else "uniform_init",
            "is_control": _yn(False),
            "n_seeds": n_cond,
            "mean_cycle_level": _fmt(_safe_mean([float(m["cycle_level"]) for m in per_seed_metrics])),
            "mean_stage3_score": _fmt(mean_s3),
            "mean_turn_strength": _fmt(_safe_mean([float(m["turn_strength"]) for m in per_seed_metrics])),
            "mean_env_gamma": _fmt(mean_gamma),
            "level_counts_json": json.dumps(level_counts, sort_keys=True),
            "p_level_3": _fmt(float(l3_count) / float(n_cond) if n_cond > 0 else 0.0),
            "level3_seed_count": l3_count,
            "control_mean_env_gamma": _fmt(mean_ctrl_gamma),
            "control_mean_stage3_score": _fmt(mean_ctrl_s3),
            "stage3_uplift_vs_control": _fmt(uplift_agg),
            "init_weight_dispersion": _fmt(_agg("init_weight_dispersion")),
            "graph_clustering_coefficient": _fmt(_agg("graph_clustering_coefficient")),
            "graph_mean_path_length": _fmt(_agg("graph_mean_path_length")),
            "spatial_autocorrelation_d1": _fmt(sac1_agg),
            "spatial_autocorrelation_d2": _fmt(sac2_agg),
            "spatial_autocorrelation_decay": _fmt(sac_decay),
            "mean_batch_phase_spread": _fmt(bps_agg),
            "mean_edge_strategy_distance": _fmt(_agg("mean_edge_strategy_distance")),
            "edge_disagreement_rate": _fmt(_agg("edge_disagreement_rate")),
            "mean_local_growth_cosine_vs_global": _fmt(cos_agg) if cos_agg else "",
            "mean_player_growth_dispersion": _fmt(pgd_agg) if pgd_agg else "",
            "phase_amplitude_stability": _fmt(pa_agg),
            "short_scout_pass": _yn(pass_flag),
            "hard_stop_fail": _yn(hard_stop_flag),
            "verdict": verdict,
            "representative_seed": int(rep_seed),
            "representative_simplex_png": "",
            "representative_phase_amplitude_png": "",
            "players": players, "rounds": rounds,
            "out_dir": str(cond_out_dir),
        }

        # Plots
        rep_rows: list[dict[str, Any]] = []
        with (cond_out_dir / f"seed{rep_seed}.csv").open(encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                rep_rows.append(dict(row))
        s_png = cond_out_dir / f"simplex_seed{rep_seed}.png"
        p_png = cond_out_dir / f"phase_amplitude_seed{rep_seed}.png"
        _write_simplex_plot(rep_rows, out_png=s_png, title=f"{condition} seed={rep_seed}", burn_in=burn_in, tail=tail)
        _write_phase_amplitude_plot(rep_rows, out_png=p_png, title=f"{condition} seed={rep_seed}", burn_in=burn_in, tail=tail)
        cond_summary_map[condition]["representative_simplex_png"] = str(s_png)
        cond_summary_map[condition]["representative_phase_amplitude_png"] = str(p_png)

    # --- Combined rows for controls ---
    def _ctrl_combined(cond_name: str, init_mode: str, per_seed: dict[int, dict]) -> dict[str, Any]:
        lvls = [int(per_seed[s]["metrics"]["cycle_level"]) for s in seeds]
        lc = {lv: lvls.count(lv) for lv in range(4)}
        return {
            "gate": GATE_NAME, "family": FAMILY, "condition": cond_name,
            "graph_topology": "well_mixed", "graph_degree": 0,
            "p_rewire": "", "update_mode": "well_mixed_sampled",
            "pairwise_imitation_strength": "", "pairwise_beta": "",
            "local_selection_strength": "",
            "init_mode": init_mode,
            "is_control": _yn(True), "n_seeds": len(seeds),
            "mean_cycle_level": _fmt(_safe_mean([float(per_seed[s]["metrics"]["cycle_level"]) for s in seeds])),
            "mean_stage3_score": _fmt(_safe_mean([float(per_seed[s]["metrics"]["stage3_score"]) for s in seeds])),
            "mean_turn_strength": _fmt(_safe_mean([float(per_seed[s]["metrics"]["turn_strength"]) for s in seeds])),
            "mean_env_gamma": _fmt(_safe_mean([float(per_seed[s]["metrics"]["env_gamma"]) for s in seeds])),
            "level_counts_json": json.dumps(lc, sort_keys=True),
            "p_level_3": _fmt(sum(1 for l in lvls if l >= 3) / len(seeds)),
            "level3_seed_count": sum(1 for l in lvls if l >= 3),
            "control_mean_env_gamma": "", "control_mean_stage3_score": "",
            "stage3_uplift_vs_control": "",
            "init_weight_dispersion": _fmt(_safe_mean([float(per_seed[s].get("init_weight_dispersion", 0)) for s in seeds])),
            "graph_clustering_coefficient": "", "graph_mean_path_length": "",
            "spatial_autocorrelation_d1": "", "spatial_autocorrelation_d2": "",
            "spatial_autocorrelation_decay": "",
            "mean_batch_phase_spread": "", "mean_edge_strategy_distance": "",
            "edge_disagreement_rate": "",
            "mean_local_growth_cosine_vs_global": "", "mean_player_growth_dispersion": "",
            "phase_amplitude_stability": _fmt(_safe_mean([float(per_seed[s]["phase_amplitude_stability"]) for s in seeds])),
            "short_scout_pass": "", "hard_stop_fail": "",
            "verdict": "control",
            "representative_seed": seeds[0],
            "representative_simplex_png": "", "representative_phase_amplitude_png": "",
            "players": players, "rounds": rounds,
            "out_dir": str(out_root / cond_name),
        }

    all_combined = [
        _ctrl_combined(ctrl_ri_cond, "random_init", ctrl_ri_per_seed),
        _ctrl_combined(ctrl_ui_cond, "uniform_init", ctrl_ui_per_seed),
    ] + list(cond_summary_map.values())

    _write_tsv(summary_tsv, fieldnames=SUMMARY_FIELDNAMES, rows=all_summary_rows)
    _write_tsv(combined_tsv, fieldnames=COMBINED_FIELDNAMES, rows=all_combined)
    _write_decision(decision_md, combined_rows=all_combined)

    return {
        "summary_tsv": str(summary_tsv),
        "combined_tsv": str(combined_tsv),
        "decision_md": str(decision_md),
    }


def _ctrl_summary_row(cond: str, init_mode: str, seed: int, r: dict) -> dict[str, Any]:
    return {
        "gate": GATE_NAME, "family": FAMILY, "condition": cond,
        "graph_topology": "well_mixed", "graph_degree": 0,
        "p_rewire": "",
        "update_mode": "well_mixed_sampled",
        "pairwise_imitation_strength": "", "pairwise_beta": "",
        "local_selection_strength": "",
        "init_mode": init_mode,
        "seed": seed,
        "cycle_level": int(r["metrics"]["cycle_level"]),
        "stage3_score": _fmt(r["metrics"]["stage3_score"]),
        "turn_strength": _fmt(r["metrics"]["turn_strength"]),
        "env_gamma": _fmt(r["metrics"]["env_gamma"]),
        "env_gamma_r2": _fmt(r["metrics"]["env_gamma_r2"]),
        "env_gamma_n_peaks": int(r["metrics"]["env_gamma_n_peaks"]),
        "control_env_gamma": "", "control_stage3_score": "",
        "stage3_uplift_vs_control_seed": "",
        "has_level3_seed": _yn(int(r["metrics"]["cycle_level"]) >= 3),
        "init_weight_dispersion": _fmt(r.get("init_weight_dispersion", 0.0)),
        "graph_clustering_coefficient": "", "graph_mean_path_length": "",
        "spatial_autocorrelation_d1": "", "spatial_autocorrelation_d2": "",
        "mean_batch_phase_spread": "", "mean_edge_strategy_distance": "",
        "edge_disagreement_rate": "",
        "mean_local_growth_cosine_vs_global": "", "mean_player_growth_dispersion": "",
        "phase_amplitude_stability": _fmt(r["phase_amplitude_stability"]),
        "out_csv": str(r["csv_path"]),
        "provenance_json": "",
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="T-series: Topology × Update 2×2 short scout")
    parser.add_argument("--seeds", type=str, default="45,47,49")
    parser.add_argument("--topologies", type=str, default="lattice4,small_world")
    parser.add_argument("--update-modes", type=str, default="pairwise,minibatch")
    parser.add_argument("--p-rewire", type=float, default=0.10)
    parser.add_argument("--pairwise-imitation-strength", type=float, default=0.35)
    parser.add_argument("--pairwise-beta", type=float, default=8.0)
    parser.add_argument("--local-selection-strength", type=float, default=0.08)
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
                        help="Only used for well-mixed control runs")
    parser.add_argument("--random-init", action="store_true", default=True)
    parser.add_argument("--no-random-init", dest="random_init", action="store_false")
    parser.add_argument("--out-root", type=Path,
                        default=REPO_ROOT / "outputs" / "t_series_short_scout")
    parser.add_argument("--summary-tsv", type=Path,
                        default=REPO_ROOT / "outputs" / "t_series_short_scout_summary.tsv")
    parser.add_argument("--combined-tsv", type=Path,
                        default=REPO_ROOT / "outputs" / "t_series_short_scout_combined.tsv")
    parser.add_argument("--decision-md", type=Path,
                        default=REPO_ROOT / "outputs" / "t_series_short_scout_decision.md")

    args = parser.parse_args()

    result = run_t_scout(
        seeds=_parse_seeds(args.seeds),
        topologies=_parse_str_list(args.topologies),
        update_modes=_parse_str_list(args.update_modes),
        p_rewire=float(args.p_rewire),
        pairwise_imitation_strength=float(args.pairwise_imitation_strength),
        pairwise_beta=float(args.pairwise_beta),
        local_selection_strength=float(args.local_selection_strength),
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
        random_init=bool(args.random_init),
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
