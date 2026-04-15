"""C1 Local Pairwise Imitation – G2 Short Scout Harness.

Phase 9 experiment: replaces global replicator broadcast with per-player
Fermi pairwise imitation on a structured interaction graph.

Architecture
------------
- Standalone simulation loop (does NOT use simulate()).
- Uses evolution/local_pairwise.py for update mechanics.
- Uses evolution/local_graph.py for topology construction.
- analysis/ layer for cycle_metrics + decay_rate.

Usage
-----
    ./venv/bin/python -m simulation.c1_pairwise_scout \\
        --graph-topologies lattice4,small_world,ring4 \\
        --beta-pairs 5.0,10.0 \\
        --mu-pairs 0.8,0.5 \\
        --seeds 45,47,49
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from math import atan2, log, pi, sqrt
from pathlib import Path
from typing import Any

from analysis.cycle_metrics import classify_cycle_level
from analysis.decay_rate import estimate_decay_gamma
from evolution.local_graph import (
    GraphSpec,
    build_graph,
    edge_disagreement_rate,
    edge_strategy_distance,
    graph_clustering_coefficient,
    sample_neighbor,
)
from evolution.local_pairwise import (
    STRATEGY_SPACE,
    local_payoff,
    pairwise_adoption_probability,
    pairwise_imitation_update,
)
from simulation.run_simulation import _initial_weights


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

G2_GATE = "g2_c1_pairwise"

DEFAULT_OUT_ROOT = Path("outputs/c1_pairwise_scout")
DEFAULT_SUMMARY_TSV = DEFAULT_OUT_ROOT / "c1_summary.tsv"
DEFAULT_COMBINED_TSV = DEFAULT_OUT_ROOT / "c1_combined.tsv"
DEFAULT_DECISION_MD = DEFAULT_OUT_ROOT / "c1_decision.md"

_N_STRATS = len(STRATEGY_SPACE)  # 3

SUMMARY_FIELDNAMES = [
    "gate",
    "condition",
    "topology",
    "beta_pair",
    "mu_pair",
    "seed",
    "cycle_level",
    "stage3_score",
    "turn_strength",
    "env_gamma",
    "env_gamma_r2",
    "env_gamma_n_peaks",
    "mean_player_weight_entropy",
    "weight_heterogeneity_std",
    "spatial_strategy_clustering",
    "mean_edge_strategy_distance",
    "graph_clustering_coeff",
    "has_level3_seed",
    "out_csv",
    "provenance_json",
]

COMBINED_FIELDNAMES = [
    "gate",
    "condition",
    "topology",
    "beta_pair",
    "mu_pair",
    "is_control",
    "n_seeds",
    "mean_cycle_level",
    "mean_stage3_score",
    "mean_turn_strength",
    "mean_env_gamma",
    "level_counts_json",
    "p_level_3",
    "level3_seed_count",
    "mean_player_weight_entropy",
    "weight_heterogeneity_std",
    "spatial_strategy_clustering",
    "mean_edge_strategy_distance",
    "graph_clustering_coeff",
    "short_scout_pass",
    "hard_stop_fail",
    "longer_confirm_candidate",
    "verdict",
    "representative_seed",
    "representative_simplex_png",
    "representative_phase_amplitude_png",
    "players",
    "rounds",
    "out_dir",
]


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _parse_seeds(s: str) -> list[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def _parse_floats(s: str) -> list[float]:
    return [float(x.strip()) for x in s.split(",") if x.strip()]


def _parse_strs(s: str) -> list[str]:
    return [x.strip() for x in s.split(",") if x.strip()]


def _fmt(v: float) -> str:
    return f"{float(v):.6f}"


def _yn(v: bool) -> str:
    return "yes" if v else "no"


def _mean(vals: list[float]) -> float:
    return float(sum(vals) / len(vals)) if vals else 0.0


def _write_tsv(
    path: Path, *, fieldnames: list[str], rows: list[dict[str, Any]]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f, fieldnames=fieldnames, delimiter="\t", extrasaction="ignore"
        )
        w.writeheader()
        w.writerows(rows)


def _cond_name(*, topology: str, beta: float, mu: float) -> str:
    b = f"{beta:.1f}".replace(".", "p")
    m = f"{mu:.1f}".replace(".", "p")
    return f"g2_c1_{topology}_b{b}_m{m}"


def _ctrl_name() -> str:
    return "g2_c1_control"


# ---------------------------------------------------------------------------
# Simplex / phase helpers
# ---------------------------------------------------------------------------

def _phase_angle(pa: float, pd: float, pb: float) -> float:
    return float(atan2(sqrt(3.0) * (pd - pb), 2.0 * pa - pd - pb))


def _amplitude(pa: float, pd: float, pb: float) -> float:
    return float(
        ((pa - 1 / 3) ** 2 + (pd - 1 / 3) ** 2 + (pb - 1 / 3) ** 2) ** 0.5
    )


def _simplex_xy(pa: float, pd: float, pb: float) -> tuple[float, float]:
    return (pd - pb, pa - 0.5 * (pd + pb))


# ---------------------------------------------------------------------------
# Graph spec builder
# ---------------------------------------------------------------------------

def _make_graph_spec(
    topology: str, *, small_world_p: float = 0.10
) -> GraphSpec:
    if topology == "lattice4":
        return GraphSpec(
            topology="lattice4", degree=4, lattice_rows=15, lattice_cols=20
        )
    if topology == "small_world":
        return GraphSpec(
            topology="small_world", degree=4, p_rewire=float(small_world_p)
        )
    if topology == "ring4":
        return GraphSpec(topology="ring4", degree=4)
    raise ValueError(f"Unknown topology: {topology!r}")


# ---------------------------------------------------------------------------
# Core simulation loop
# ---------------------------------------------------------------------------

def _run_c1_simulation(
    *,
    n_players: int,
    n_rounds: int,
    seed: int,
    graph_spec: GraphSpec | None,
    beta_pair: float,
    mu_pair: float,
    a: float,
    b: float,
    cross: float,
    init_bias: float,
    control_degree: int = 4,
    init_weight_jitter: float = 0.02,
) -> tuple[
    list[dict[str, Any]],    # global_rows
    list[dict[str, Any]],    # round_diagnostics
    list[list[float]],       # final_weights
    list[list[int]] | None,  # adj (static graph, None for control)
]:
    """Run one C1 pairwise imitation simulation.

    Returns (global_rows, round_diagnostics, final_weights, adj).
    """
    rng = random.Random(int(seed))
    n = int(n_players)
    beta = float(beta_pair)
    mu = float(mu_pair)
    a_v, b_v, c_v = float(a), float(b), float(cross)
    k = int(control_degree)

    # Build static graph (or None for well-mixed control)
    adj: list[list[int]] | None = None
    if graph_spec is not None:
        adj = build_graph(n, graph_spec, seed=int(seed))

    # Initialise per-player weights (with jitter for symmetry-breaking)
    iw = _initial_weights(strategy_space=STRATEGY_SPACE, init_bias=float(init_bias))
    init_w = [float(iw[s]) for s in STRATEGY_SPACE]
    ws = sum(init_w)
    init_w = [v / ws for v in init_w]

    jitter = float(init_weight_jitter)
    weights: list[list[float]] = []
    for _i in range(n):
        w_i = list(init_w)
        if jitter > 0:
            for k in range(len(w_i)):
                w_i[k] += rng.uniform(-jitter, jitter)
            w_i = [max(v, 1e-8) for v in w_i]
            s = sum(w_i)
            w_i = [v / s for v in w_i]
        weights.append(w_i)

    global_rows: list[dict[str, Any]] = []
    round_diag: list[dict[str, Any]] = []

    for t in range(int(n_rounds)):
        # --- effective adjacency (static graph or random-per-round) ---
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

        # --- 1. strategy sampling from individual weights ---
        strategies: list[str] = []
        for i in range(n):
            w = weights[i]
            r = rng.random()
            cum = 0.0
            chosen = STRATEGY_SPACE[-1]
            for si in range(_N_STRATS):
                cum += w[si]
                if r <= cum:
                    chosen = STRATEGY_SPACE[si]
                    break
            strategies.append(chosen)

        # --- 2. local payoff ---
        payoffs: list[float] = []
        for i in range(n):
            ns = [strategies[j] for j in eff[i]]
            payoffs.append(
                local_payoff(weights[i], ns, a=a_v, b=b_v, cross=c_v)
            )

        # --- 3. pairwise imitation (synchronous) ---
        new_w: list[list[float]] = []
        for i in range(n):
            if adj is not None:
                j = sample_neighbor(rng, adj, i)
            else:
                j = rng.choice(eff[i])
            q = pairwise_adoption_probability(
                payoffs[i], payoffs[j], beta_pair=beta
            )
            new_w.append(
                pairwise_imitation_update(weights[i], weights[j], mu=mu, q=q)
            )

        weights = new_w

        # --- 4. global strategy distribution ---
        counts = [0] * _N_STRATS
        for s in strategies:
            counts[STRATEGY_SPACE.index(s)] += 1
        row: dict[str, Any] = {"round": t}
        for si in range(_N_STRATS):
            row[f"p_{STRATEGY_SPACE[si]}"] = counts[si] / n
        global_rows.append(row)

        # --- 5. lightweight per-round diagnostics ---
        total_h = 0.0
        for i in range(n):
            for v in weights[i]:
                if v > 1e-15:
                    total_h -= v * log(v)
        mean_h = total_h / n

        mean_w = [0.0] * _N_STRATS
        for i in range(n):
            for si in range(_N_STRATS):
                mean_w[si] += weights[i][si]
        mean_w = [v / n for v in mean_w]
        var_s = 0.0
        for i in range(n):
            for si in range(_N_STRATS):
                var_s += (weights[i][si] - mean_w[si]) ** 2
        w_std = (var_s / (n * _N_STRATS)) ** 0.5

        round_diag.append(
            {
                "mean_player_weight_entropy": float(mean_h),
                "weight_heterogeneity_std": float(w_std),
            }
        )

    return global_rows, round_diag, weights, adj


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def _tail_begin(n_rows: int, *, burn_in: int, tail: int) -> int:
    if n_rows <= 0:
        return 0
    return max(int(burn_in), n_rows - int(tail))


def _seed_metrics(
    rows: list[dict[str, Any]],
    *,
    burn_in: int,
    tail: int,
    eta: float,
    corr_threshold: float,
) -> dict[str, float | int]:
    sm = {
        "aggressive": [float(r["p_aggressive"]) for r in rows],
        "defensive": [float(r["p_defensive"]) for r in rows],
        "balanced": [float(r["p_balanced"]) for r in rows],
    }
    cyc = classify_cycle_level(
        sm,
        burn_in=int(burn_in),
        tail=int(tail),
        amplitude_threshold=0.02,
        corr_threshold=float(corr_threshold),
        eta=float(eta),
        stage3_method="turning",
        phase_smoothing=1,
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


def _tail_c1_diagnostics(
    round_diag: list[dict[str, Any]],
    final_weights: list[list[float]],
    adj: list[list[int]] | None,
    *,
    n_rows: int,
    burn_in: int,
    tail: int,
) -> dict[str, float]:
    begin = _tail_begin(n_rows, burn_in=int(burn_in), tail=int(tail))
    window = round_diag[begin:]
    if not window:
        return {
            "mean_player_weight_entropy": 0.0,
            "weight_heterogeneity_std": 0.0,
            "spatial_strategy_clustering": 0.0,
            "mean_edge_strategy_distance": 0.0,
            "graph_clustering_coeff": 0.0,
        }

    mh = _mean([float(d["mean_player_weight_entropy"]) for d in window])
    ws = _mean([float(d["weight_heterogeneity_std"]) for d in window])

    # Spatial diagnostics from final weights + graph
    ssc = 0.0
    mesd = 0.0
    gcc = 0.0
    if adj is not None:
        dom = [
            STRATEGY_SPACE[w.index(max(w))] for w in final_weights
        ]
        ssc = 1.0 - edge_disagreement_rate(adj, dom)
        mesd = edge_strategy_distance(adj, final_weights)
        gcc = graph_clustering_coefficient(adj)

    return {
        "mean_player_weight_entropy": float(mh),
        "weight_heterogeneity_std": float(ws),
        "spatial_strategy_clustering": float(ssc),
        "mean_edge_strategy_distance": float(mesd),
        "graph_clustering_coeff": float(gcc),
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

    begin = _tail_begin(len(rows), burn_in=int(burn_in), tail=int(tail))
    tail_rows = rows[begin:]
    if len(tail_rows) < 2:
        return

    pts = [
        _simplex_xy(
            float(r["p_aggressive"]),
            float(r["p_defensive"]),
            float(r["p_balanced"]),
        )
        for r in tail_rows
    ]
    xs = np.array([p[0] for p in pts], dtype=float)
    ys = np.array([p[1] for p in pts], dtype=float)

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5, 5), constrained_layout=True)
    segs = np.stack(
        [
            np.column_stack([xs[:-1], ys[:-1]]),
            np.column_stack([xs[1:], ys[1:]]),
        ],
        axis=1,
    )
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

    begin = _tail_begin(len(rows), burn_in=int(burn_in), tail=int(tail))
    tail_rows = rows[begin:]
    if not tail_rows:
        return

    rounds_arr = np.array([int(r["round"]) for r in tail_rows], dtype=float)
    phases = np.unwrap(
        np.array(
            [
                _phase_angle(
                    float(r["p_aggressive"]),
                    float(r["p_defensive"]),
                    float(r["p_balanced"]),
                )
                for r in tail_rows
            ]
        )
    )
    amps = np.array(
        [
            _amplitude(
                float(r["p_aggressive"]),
                float(r["p_defensive"]),
                float(r["p_balanced"]),
            )
            for r in tail_rows
        ]
    )

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(
        2, 1, figsize=(8, 5.5), sharex=True, constrained_layout=True
    )
    axes[0].plot(rounds_arr, phases, color="#1f77b4", lw=1.5)
    axes[0].set_ylabel("phase angle")
    axes[0].set_title(title)
    axes[0].grid(alpha=0.25)
    axes[1].plot(rounds_arr, amps, color="#d62728", lw=1.5)
    axes[1].set_ylabel("amplitude")
    axes[1].set_xlabel("round")
    axes[1].grid(alpha=0.25)
    fig.savefig(out_png, dpi=160)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Summary builder
# ---------------------------------------------------------------------------

def _build_summary(
    *,
    condition: str,
    topology: str,
    beta: float,
    mu: float,
    is_control: bool,
    metric_rows: list[dict[str, Any]],
    diag_rows: list[dict[str, float]],
    out_dir: Path,
    players: int,
    rounds: int,
) -> dict[str, Any]:
    levels = [int(r["cycle_level"]) for r in metric_rows]
    lc = {lv: levels.count(lv) for lv in range(4)}
    den = float(len(metric_rows)) or 1.0
    return {
        "gate": G2_GATE,
        "condition": condition,
        "topology": topology,
        "beta_pair": _fmt(beta),
        "mu_pair": _fmt(mu),
        "is_control": _yn(is_control),
        "n_seeds": len(metric_rows),
        "mean_cycle_level": _fmt(_mean([float(l) for l in levels])),
        "mean_stage3_score": _fmt(
            _mean([float(r["stage3_score"]) for r in metric_rows])
        ),
        "mean_turn_strength": _fmt(
            _mean([float(r["turn_strength"]) for r in metric_rows])
        ),
        "mean_env_gamma": _fmt(
            _mean([float(r["env_gamma"]) for r in metric_rows])
        ),
        "level_counts_json": json.dumps(lc, sort_keys=True),
        "p_level_3": _fmt(sum(1 for l in levels if l >= 3) / den),
        "level3_seed_count": sum(1 for l in levels if l >= 3),
        "mean_player_weight_entropy": _fmt(
            _mean([float(r["mean_player_weight_entropy"]) for r in diag_rows])
        ),
        "weight_heterogeneity_std": _fmt(
            _mean([float(r["weight_heterogeneity_std"]) for r in diag_rows])
        ),
        "spatial_strategy_clustering": _fmt(
            _mean(
                [float(r["spatial_strategy_clustering"]) for r in diag_rows]
            )
        ),
        "mean_edge_strategy_distance": _fmt(
            _mean(
                [float(r["mean_edge_strategy_distance"]) for r in diag_rows]
            )
        ),
        "graph_clustering_coeff": _fmt(
            _mean([float(r["graph_clustering_coeff"]) for r in diag_rows])
        ),
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


def _rep_seed_row(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return max(
        rows,
        key=lambda r: (int(r["cycle_level"]), float(r["stage3_score"])),
    )


# ---------------------------------------------------------------------------
# Decision writer
# ---------------------------------------------------------------------------

def _write_decision(
    path: Path,
    *,
    combined: list[dict[str, Any]],
    candidates: list[dict[str, Any]],
    close_c1: bool,
) -> None:
    lines = [
        "# C1 Local Pairwise Imitation Decision",
        "",
        "## G2 Short Scout",
        "",
    ]
    for r in combined:
        lines.append(
            f"- {r['condition']}: topo={r['topology']}"
            f" β={r['beta_pair']} μ={r['mu_pair']}"
            f" level3={r['level3_seed_count']}"
            f" mean_entropy={r['mean_player_weight_entropy']}"
            f" spatial_clust={r['spatial_strategy_clustering']}"
            f" weight_std={r['weight_heterogeneity_std']}"
            f" pass={r['short_scout_pass']}"
            f" verdict={r['verdict']}"
        )
    lines += ["", "## Recommendation", ""]
    if not candidates:
        lines.append("- longer_confirm_candidate: none")
    else:
        for r in candidates:
            lines.append(
                f"- longer_confirm_candidate: {r['condition']}"
                f" entropy={r['mean_player_weight_entropy']}"
            )
    lines += ["", "## Pass Gate (4-way AND)", ""]
    lines.append("1. level3_seed_count ≥ 2")
    lines.append("2. mean_env_gamma ≥ 0")
    lines.append("3. mean_player_weight_entropy ≥ 1.18")
    lines.append("4. spatial_strategy_clustering > 0.25")
    lines += ["", "## Stop Rule", ""]
    lines.append(
        "- If ALL active conditions hard-stop fail, C1 closes as negative result."
    )
    lines.append(
        "- Interpretation: per-player pairwise Fermi imitation on structured"
        " graph is insufficient to break entropy lock."
    )
    lines.append(
        f"- overall_verdict: {'close_c1' if close_c1 else 'keep_c1_open'}"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# Main harness
# ---------------------------------------------------------------------------

def run_c1_scout(
    *,
    seeds: list[int],
    topologies: list[str],
    beta_pairs: list[float],
    mu_pairs: list[float],
    small_world_p: float = 0.10,
    out_root: Path,
    summary_tsv: Path,
    combined_tsv: Path,
    decision_md: Path,
    players: int = 300,
    rounds: int = 5000,
    burn_in: int = 1500,
    tail: int = 1500,
    init_bias: float = 0.12,
    a: float = 1.0,
    b: float = 0.9,
    cross: float = 0.20,
    eta: float = 0.55,
    corr_threshold: float = 0.09,
    control_beta: float = 5.0,
    control_mu: float = 0.5,
    init_weight_jitter: float = 0.02,
) -> dict[str, Any]:
    out_root.mkdir(parents=True, exist_ok=True)

    # Build conditions: well-mixed control + active graph conditions
    conditions: list[dict[str, Any]] = []
    conditions.append(
        {
            "condition": _ctrl_name(),
            "topology": "none",
            "graph_spec": None,
            "beta_pair": float(control_beta),
            "mu_pair": float(control_mu),
            "is_control": True,
        }
    )

    # Paired combos: zip beta_pairs with mu_pairs
    combos = list(zip(beta_pairs, mu_pairs))
    for topo in topologies:
        gs = _make_graph_spec(topo, small_world_p=small_world_p)
        for beta, mu in combos:
            conditions.append(
                {
                    "condition": _cond_name(
                        topology=topo, beta=beta, mu=mu
                    ),
                    "topology": topo,
                    "graph_spec": gs,
                    "beta_pair": float(beta),
                    "mu_pair": float(mu),
                    "is_control": False,
                }
            )

    all_summary: list[dict[str, Any]] = []
    all_combined: list[dict[str, Any]] = []
    cond_seed_rows: dict[str, list[dict[str, Any]]] = {}
    cond_timeseries: dict[str, dict[int, list[dict[str, Any]]]] = {}

    for cond in conditions:
        cn = str(cond["condition"])
        gs = cond["graph_spec"]
        beta = float(cond["beta_pair"])
        mu = float(cond["mu_pair"])
        topo = str(cond["topology"])
        is_ctrl = bool(cond["is_control"])
        od = out_root / cn
        od.mkdir(parents=True, exist_ok=True)

        per_seed: list[dict[str, Any]] = []
        metrics: list[dict[str, Any]] = []
        diags: list[dict[str, float]] = []
        seed_ts: dict[int, list[dict[str, Any]]] = {}

        for seed in seeds:
            print(f"  C1 {cn} seed={seed} ...")
            rows, rdiag, fw, adj_out = _run_c1_simulation(
                n_players=int(players),
                n_rounds=int(rounds),
                seed=int(seed),
                graph_spec=gs,
                beta_pair=beta,
                mu_pair=mu,
                a=float(a),
                b=float(b),
                cross=float(cross),
                init_bias=float(init_bias),
                init_weight_jitter=float(init_weight_jitter),
            )
            seed_ts[int(seed)] = rows

            csv_path = od / f"seed{seed}.csv"
            _write_csv(csv_path, rows=rows)

            sm = _seed_metrics(
                rows,
                burn_in=burn_in,
                tail=tail,
                eta=eta,
                corr_threshold=corr_threshold,
            )
            td = _tail_c1_diagnostics(
                rdiag,
                fw,
                adj_out,
                n_rows=len(rows),
                burn_in=burn_in,
                tail=tail,
            )

            prov = {
                "condition": cn,
                "topology": topo,
                "beta_pair": beta,
                "mu_pair": mu,
                "seed": int(seed),
                "config": {
                    "players": int(players),
                    "rounds": int(rounds),
                    "burn_in": int(burn_in),
                    "tail": int(tail),
                    "init_bias": float(init_bias),
                    "a": float(a),
                    "b": float(b),
                    "cross": float(cross),
                },
                "cycle_metrics": dict(sm),
                "c1_diagnostics": dict(td),
            }
            pj = od / f"seed{seed}_provenance.json"
            pj.write_text(
                json.dumps(prov, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )

            seed_row = {
                "gate": G2_GATE,
                "condition": cn,
                "topology": topo,
                "beta_pair": _fmt(beta),
                "mu_pair": _fmt(mu),
                "seed": int(seed),
                "cycle_level": int(sm["cycle_level"]),
                "stage3_score": _fmt(sm["stage3_score"]),
                "turn_strength": _fmt(sm["turn_strength"]),
                "env_gamma": _fmt(sm["env_gamma"]),
                "env_gamma_r2": _fmt(sm["env_gamma_r2"]),
                "env_gamma_n_peaks": int(sm["env_gamma_n_peaks"]),
                "mean_player_weight_entropy": _fmt(
                    td["mean_player_weight_entropy"]
                ),
                "weight_heterogeneity_std": _fmt(
                    td["weight_heterogeneity_std"]
                ),
                "spatial_strategy_clustering": _fmt(
                    td["spatial_strategy_clustering"]
                ),
                "mean_edge_strategy_distance": _fmt(
                    td["mean_edge_strategy_distance"]
                ),
                "graph_clustering_coeff": _fmt(td["graph_clustering_coeff"]),
                "has_level3_seed": _yn(int(sm["cycle_level"]) >= 3),
                "out_csv": str(csv_path),
                "provenance_json": str(pj),
            }
            per_seed.append(seed_row)
            metrics.append(sm)
            diags.append(td)

        cond_seed_rows[cn] = per_seed
        cond_timeseries[cn] = seed_ts
        all_combined.append(
            _build_summary(
                condition=cn,
                topology=topo,
                beta=beta,
                mu=mu,
                is_control=is_ctrl,
                metric_rows=metrics,
                diag_rows=diags,
                out_dir=od,
                players=players,
                rounds=rounds,
            )
        )

    # --- Verdicts ---
    for cr in all_combined:
        if str(cr["is_control"]) == "yes":
            continue
        l3 = int(cr["level3_seed_count"])
        eg = float(cr["mean_env_gamma"])
        ent = float(cr["mean_player_weight_entropy"])
        ssc = float(cr["spatial_strategy_clustering"])

        sp = l3 >= 2 and eg >= 0.0 and ent >= 1.18 and ssc > 0.25
        hs = l3 == 0 and ent < 1.10

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
        rep = _rep_seed_row(cond_seed_rows[cn])
        rs = int(rep["seed"])
        od = Path(cr["out_dir"])

        sp_png = od / f"seed{rs}_simplex.png"
        pp_png = od / f"seed{rs}_phase_amplitude.png"
        _write_simplex_plot(
            cond_timeseries[cn][rs],
            out_png=sp_png,
            title=f"{cn} s={rs}",
            burn_in=burn_in,
            tail=tail,
        )
        _write_phase_amplitude_plot(
            cond_timeseries[cn][rs],
            out_png=pp_png,
            title=f"{cn} s={rs}",
            burn_in=burn_in,
            tail=tail,
        )
        cr["representative_seed"] = rs
        cr["representative_simplex_png"] = str(sp_png)
        cr["representative_phase_amplitude_png"] = str(pp_png)

    # --- Flatten per-seed for summary ---
    for _cn, ps in cond_seed_rows.items():
        all_summary.extend(ps)

    # --- Stop rule ---
    active = [r for r in all_combined if str(r["is_control"]) == "no"]
    close_c1 = bool(active) and all(
        str(r["hard_stop_fail"]) == "yes" for r in active
    )

    candidates = [r for r in all_combined if str(r["verdict"]) == "pass"]
    candidates.sort(
        key=lambda r: (
            int(r["level3_seed_count"]),
            float(r["mean_player_weight_entropy"]),
        ),
        reverse=True,
    )

    _write_tsv(summary_tsv, fieldnames=SUMMARY_FIELDNAMES, rows=all_summary)
    _write_tsv(
        combined_tsv, fieldnames=COMBINED_FIELDNAMES, rows=all_combined
    )
    _write_decision(
        decision_md,
        combined=all_combined,
        candidates=candidates,
        close_c1=close_c1,
    )

    print(f"summary_tsv={summary_tsv}")
    print(f"combined_tsv={combined_tsv}")
    print(f"decision_md={decision_md}")
    print(f"close_c1={close_c1}")

    return {
        "summary_tsv": str(summary_tsv),
        "combined_tsv": str(combined_tsv),
        "decision_md": str(decision_md),
        "close_c1": bool(close_c1),
        "candidates": [dict(r) for r in candidates],
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(
        description="C1 Local Pairwise Imitation G2 Scout"
    )
    p.add_argument("--seeds", type=str, default="45,47,49")
    p.add_argument(
        "--graph-topologies",
        type=str,
        default="lattice4,small_world,ring4",
    )
    p.add_argument("--beta-pairs", type=str, default="5.0,10.0")
    p.add_argument("--mu-pairs", type=str, default="0.8,0.5")
    p.add_argument("--small-world-p", type=float, default=0.10)
    p.add_argument("--players", type=int, default=300)
    p.add_argument("--rounds", type=int, default=5000)
    p.add_argument("--burn-in", type=int, default=1500)
    p.add_argument("--tail", type=int, default=1500)
    p.add_argument("--init-bias", type=float, default=0.12)
    p.add_argument("--a", type=float, default=1.0)
    p.add_argument("--b", type=float, default=0.9)
    p.add_argument("--cross", type=float, default=0.20)
    p.add_argument("--eta", type=float, default=0.55)
    p.add_argument("--corr-threshold", type=float, default=0.09)
    p.add_argument("--control-beta", type=float, default=5.0)
    p.add_argument("--control-mu", type=float, default=0.5)
    p.add_argument("--init-weight-jitter", type=float, default=0.02)
    p.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    p.add_argument("--summary-tsv", type=Path, default=DEFAULT_SUMMARY_TSV)
    p.add_argument("--combined-tsv", type=Path, default=DEFAULT_COMBINED_TSV)
    p.add_argument("--decision-md", type=Path, default=DEFAULT_DECISION_MD)
    args = p.parse_args()

    run_c1_scout(
        seeds=_parse_seeds(args.seeds),
        topologies=_parse_strs(args.graph_topologies),
        beta_pairs=_parse_floats(args.beta_pairs),
        mu_pairs=_parse_floats(args.mu_pairs),
        small_world_p=args.small_world_p,
        out_root=args.out_root,
        summary_tsv=args.summary_tsv,
        combined_tsv=args.combined_tsv,
        decision_md=args.decision_md,
        players=args.players,
        rounds=args.rounds,
        burn_in=args.burn_in,
        tail=args.tail,
        init_bias=args.init_bias,
        a=args.a,
        b=args.b,
        cross=args.cross,
        eta=args.eta,
        corr_threshold=args.corr_threshold,
        control_beta=args.control_beta,
        control_mu=args.control_mu,
        init_weight_jitter=args.init_weight_jitter,
    )


if __name__ == "__main__":
    main()
