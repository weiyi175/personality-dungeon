"""Tests for T-series: Topology × Update 2×2 harness.

G0 gate tests
-------------
G0-1  p_rewire=0 → WS ≡ ring-4 (graph equivalence)
G0-2  random init → different player weights (max pairwise distance > 0)
G0-3  fixed seed → fully reproducible
G0-4  mini-batch k_local=0 → exact no-op
G0-5  mini-batch + uniform init → cosine ≈ 1.0 (C2 degeneracy confirmation)
G0-6  random init vs uniform init → init_weight_dispersion diff > 0.05

Integration smoke
-----------------
IS-1  row count = rounds
IS-2  p/w validity (0 ≤ v ≤ 1, sum ≈ 1)
IS-3  fixed seed → fully reproducible (cell-level)
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from evolution.init_weights import init_weight_dispersion, random_simplex_weights
from evolution.local_graph import (
    GraphSpec,
    build_graph,
    graph_clustering_coefficient,
    graph_mean_shortest_path_approx,
    spatial_autocorrelation,
)
from evolution.local_replicator import STRATEGY_SPACE, weights_to_simplex
from simulation.t_series import (
    TCellConfig,
    _graph_spec_for,
    _safe_mean,
    run_t_cell,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_config(
    *,
    condition: str = "test",
    topology: str = "lattice4",
    update_mode: str = "pairwise",
    pairwise_imitation_strength: float = 0.35,
    pairwise_beta: float = 8.0,
    local_selection_strength: float = 0.08,
    players: int = 30,
    rounds: int = 20,
    seed: int = 45,
    a: float = 1.0,
    b: float = 0.9,
    cross: float = 0.20,
    burn_in: int = 5,
    tail: int = 10,
    random_init: bool = True,
    init_bias: float = 0.12,
    memory_kernel: int = 3,
    mutation_rate: float = 0.0,
    p_rewire: float = 0.0,
    lattice_rows: int | None = None,
    lattice_cols: int | None = None,
) -> TCellConfig:
    if topology == "lattice4":
        lattice_rows = lattice_rows or 5
        lattice_cols = lattice_cols or 6
        gs = GraphSpec(topology="lattice4", degree=4,
                       lattice_rows=lattice_rows, lattice_cols=lattice_cols)
        players = lattice_rows * lattice_cols
    elif topology == "small_world":
        gs = GraphSpec(topology="small_world", degree=4, p_rewire=p_rewire)
    elif topology == "ring4":
        gs = GraphSpec(topology="ring4", degree=4)
    else:
        raise ValueError(topology)

    return TCellConfig(
        condition=condition,
        graph_spec=gs,
        update_mode=update_mode,
        pairwise_imitation_strength=pairwise_imitation_strength,
        pairwise_beta=pairwise_beta,
        local_selection_strength=local_selection_strength,
        players=players, rounds=rounds, seed=seed,
        a=a, b=b, cross=cross,
        burn_in=burn_in, tail=tail,
        random_init=random_init, init_bias=init_bias,
        memory_kernel=memory_kernel, mutation_rate=mutation_rate,
        out_dir=Path(tempfile.mkdtemp()),
    )


# ===================================================================
# G0-1: p_rewire=0 → WS ≡ ring-4
# ===================================================================

class TestG01WattsStrogatzRing:
    """With p_rewire=0, Watts–Strogatz(k=4) must produce the same graph as ring4."""

    def test_ws_p0_equals_ring4(self):
        n = 30
        ring_adj = build_graph(n, GraphSpec(topology="ring4", degree=4), seed=0)
        ws_adj = build_graph(n, GraphSpec(topology="small_world", degree=4, p_rewire=0.0), seed=0)
        # Same adjacency
        for i in range(n):
            assert sorted(ring_adj[i]) == sorted(ws_adj[i]), f"mismatch at node {i}"


# ===================================================================
# G0-2: random init → different player weights
# ===================================================================

class TestG02RandomInitHeterogeneity:
    """Random Dirichlet init must produce heterogeneous player weights."""

    def test_different_player_weights(self):
        n = 30
        seed = 45
        weights = [random_simplex_weights(STRATEGY_SPACE, seed=seed, player_index=i)
                    for i in range(n)]
        # At least two players should differ
        all_same = True
        for i in range(1, n):
            for s in STRATEGY_SPACE:
                if abs(weights[i][s] - weights[0][s]) > 1e-8:
                    all_same = False
                    break
        assert not all_same, "All players have identical weights"

    def test_max_pairwise_distance_positive(self):
        n = 30
        seed = 47
        simplices = [list(random_simplex_weights(STRATEGY_SPACE, seed=seed, player_index=i).values())
                     for i in range(n)]
        max_dist = 0.0
        for i in range(n):
            for j in range(i + 1, n):
                d = sum((simplices[i][k] - simplices[j][k]) ** 2 for k in range(3)) ** 0.5
                max_dist = max(max_dist, d)
        assert max_dist > 0.01, f"max_pairwise_distance={max_dist} too small"


# ===================================================================
# G0-3: fixed seed → reproducible
# ===================================================================

class TestG03Reproducibility:
    """Same seed, same config → identical output rows."""

    def test_pairwise_reproducible(self):
        cfg = _make_config(topology="lattice4", update_mode="pairwise",
                           seed=45, rounds=15, random_init=True,
                           lattice_rows=5, lattice_cols=6)
        r1 = run_t_cell(cfg)
        r2 = run_t_cell(cfg)
        for i, (a, b) in enumerate(zip(r1["global_rows"], r2["global_rows"])):
            for s in STRATEGY_SPACE:
                assert abs(float(a[f"p_{s}"]) - float(b[f"p_{s}"])) < 1e-12, \
                    f"p_{s} mismatch at round {i}"

    def test_minibatch_reproducible(self):
        cfg = _make_config(topology="small_world", update_mode="minibatch",
                           seed=49, rounds=15, random_init=True,
                           players=30, p_rewire=0.10)
        r1 = run_t_cell(cfg)
        r2 = run_t_cell(cfg)
        for i, (a, b) in enumerate(zip(r1["global_rows"], r2["global_rows"])):
            for s in STRATEGY_SPACE:
                assert abs(float(a[f"w_{s}"]) - float(b[f"w_{s}"])) < 1e-12, \
                    f"w_{s} mismatch at round {i}"


# ===================================================================
# G0-4: mini-batch k_local=0 → exact no-op
# ===================================================================

class TestG04MinibatchNoOp:
    """With k_local=0, mini-batch update should be a no-op on weights."""

    def test_no_weight_change(self):
        cfg = _make_config(
            topology="lattice4", update_mode="minibatch",
            local_selection_strength=0.0,
            seed=45, rounds=10, random_init=True,
            lattice_rows=5, lattice_cols=6,
        )
        result = run_t_cell(cfg)
        rows = result["global_rows"]
        # With k_local=0 the weights should not change from init
        first_w = {s: float(rows[0][f"w_{s}"]) for s in STRATEGY_SPACE}
        for t, row in enumerate(rows):
            for s in STRATEGY_SPACE:
                assert abs(float(row[f"w_{s}"]) - first_w[s]) < 1e-10, \
                    f"w_{s} changed at round {t} with k_local=0"


# ===================================================================
# G0-5: mini-batch + uniform init → cosine ≈ 1.0 (C2 degeneracy)
# ===================================================================

class TestG05MinibatchUniformDegeneracy:
    """Uniform init + minibatch → local growth ≡ global growth (cosine = 1.0)."""

    def test_cosine_equals_one(self):
        cfg = _make_config(
            topology="lattice4", update_mode="minibatch",
            local_selection_strength=0.08,
            seed=45, rounds=10, random_init=False,
            lattice_rows=5, lattice_cols=6,
        )
        result = run_t_cell(cfg)
        diag = result["round_diag"]
        for d in diag:
            cos = d.get("mean_local_growth_cosine_vs_global", "")
            if cos != "":
                assert abs(float(cos) - 1.0) < 1e-8, \
                    f"cosine_vs_global={cos} != 1.0 with uniform init"


# ===================================================================
# G0-6: random init vs uniform init → dispersion difference > 0.05
# ===================================================================

class TestG06InitDispersionDifference:
    """Random Dirichlet init should have higher init_weight_dispersion than uniform."""

    def test_dispersion_gap(self):
        n = 30
        seed = 45
        random_simplices = [
            list(random_simplex_weights(STRATEGY_SPACE, seed=seed, player_index=i).values())
            for i in range(n)
        ]
        uniform_val = [1.0 / 3.0] * 3
        uniform_simplices = [uniform_val[:] for _ in range(n)]

        iwd_random = init_weight_dispersion(random_simplices)
        iwd_uniform = init_weight_dispersion(uniform_simplices)
        gap = iwd_random - iwd_uniform
        assert gap > 0.05, f"dispersion gap={gap:.4f} too small"


# ===================================================================
# IS-1: Row count = rounds
# ===================================================================

class TestIS1RowCount:
    def test_pairwise_row_count(self):
        rounds = 15
        cfg = _make_config(topology="lattice4", update_mode="pairwise",
                           seed=45, rounds=rounds, lattice_rows=5, lattice_cols=6)
        result = run_t_cell(cfg)
        assert len(result["global_rows"]) == rounds

    def test_minibatch_row_count(self):
        rounds = 12
        cfg = _make_config(topology="small_world", update_mode="minibatch",
                           seed=47, rounds=rounds, players=30, p_rewire=0.10)
        result = run_t_cell(cfg)
        assert len(result["global_rows"]) == rounds


# ===================================================================
# IS-2: p/w validity (0 ≤ v ≤ 1, sum ≈ 1)
# ===================================================================

class TestIS2Validity:
    def test_proportions_valid(self):
        cfg = _make_config(topology="lattice4", update_mode="pairwise",
                           seed=45, rounds=20, lattice_rows=5, lattice_cols=6)
        result = run_t_cell(cfg)
        for row in result["global_rows"]:
            p_sum = sum(float(row[f"p_{s}"]) for s in STRATEGY_SPACE)
            assert abs(p_sum - 1.0) < 1e-8, f"p sum={p_sum}"
            w_sum = sum(float(row[f"w_{s}"]) for s in STRATEGY_SPACE)
            assert abs(w_sum - 1.0) < 1e-6, f"w sum={w_sum}"
            for s in STRATEGY_SPACE:
                assert 0.0 <= float(row[f"p_{s}"]) <= 1.0
                assert -1e-8 <= float(row[f"w_{s}"]) <= 1.0 + 1e-8


# ===================================================================
# IS-3: Graph diagnostics populated
# ===================================================================

class TestIS3GraphDiagnostics:
    def test_has_graph_metrics(self):
        cfg = _make_config(topology="small_world", update_mode="pairwise",
                           seed=45, rounds=10, players=30, p_rewire=0.10)
        result = run_t_cell(cfg)
        assert "graph_clustering_coefficient" in result
        assert "graph_mean_path_length" in result
        assert "spatial_autocorrelation_d1" in result
        assert "spatial_autocorrelation_d2" in result
        assert result["graph_clustering_coefficient"] > 0.0
        assert result["graph_mean_path_length"] > 0.0

    def test_init_weight_dispersion_positive_for_random(self):
        cfg = _make_config(topology="lattice4", update_mode="minibatch",
                           seed=45, rounds=10, random_init=True,
                           lattice_rows=5, lattice_cols=6)
        result = run_t_cell(cfg)
        assert result["init_weight_dispersion"] > 0.05


# ===================================================================
# G0-X: mutation_rate
# ===================================================================

class TestG0XMutation:
    """Per-round mutation (Dirichlet mixing) tests."""

    def test_mutation_breaks_uniform_degeneracy(self):
        """With mutation_rate>0 + uniform init + minibatch, cosine should drop below 1.0."""
        cfg = _make_config(
            topology="lattice4", update_mode="minibatch",
            local_selection_strength=0.08, mutation_rate=0.05,
            seed=45, rounds=15, random_init=False,
            lattice_rows=5, lattice_cols=6,
        )
        result = run_t_cell(cfg)
        # After first mutation round, cosines should be < 1.0
        cosines = [
            float(d["mean_local_growth_cosine_vs_global"])
            for d in result["round_diag"][1:]
            if d.get("mean_local_growth_cosine_vs_global", "") != ""
        ]
        assert any(c < 0.999 for c in cosines), \
            "mutation should break cosine=1.0 degeneracy"

    def test_mutation_reproducible(self):
        """Same seed + mutation_rate → identical results."""
        cfg = _make_config(
            topology="small_world", update_mode="minibatch",
            mutation_rate=0.05, seed=47, rounds=15,
            players=30, p_rewire=0.10,
        )
        r1 = run_t_cell(cfg)
        r2 = run_t_cell(cfg)
        for i, (a, b) in enumerate(zip(r1["global_rows"], r2["global_rows"])):
            for s in STRATEGY_SPACE:
                assert abs(float(a[f"w_{s}"]) - float(b[f"w_{s}"])) < 1e-12, \
                    f"w_{s} mismatch at round {i} with mutation"

