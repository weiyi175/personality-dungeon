"""G0 + IR tests for evolution/local_graph.py.

Blueprint §11.1 (5 items):
  G0-1  ring4 degree check
  G0-2  ring4 no self-loops
  G0-3  lattice4 degree check + torus wrap
  G0-4  lattice4 no self-loops
  G0-5  edge_strategy_distance / edge_disagreement_rate shape correctness
  IR-1  sample_neighbor always returns a valid neighbor index
  IR-2  build_graph raises on unsupported topology
  IR-3  ring4 raises on <5 players
  IR-4  lattice4 raises on mismatched rows*cols
"""

from __future__ import annotations

import random

import pytest

from evolution.local_graph import (
    GraphSpec,
    build_graph,
    edge_disagreement_rate,
    edge_strategy_distance,
    sample_neighbor,
)

STRAT_SPACE = ["aggressive", "defensive", "balanced"]


# ---------------------------------------------------------------------------
# G0-1  ring4 degree = 4 for all players
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("n", [5, 10, 100, 300])
def test_ring4_degree(n: int) -> None:
    spec = GraphSpec(topology="ring4", degree=4)
    adj = build_graph(n, spec, seed=0)
    assert len(adj) == n
    for i, nbrs in enumerate(adj):
        assert len(nbrs) == 4, f"player {i} has {len(nbrs)} neighbours, expected 4"


# ---------------------------------------------------------------------------
# G0-2  ring4: no self-loops, sorted, unique
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("n", [5, 20, 300])
def test_ring4_no_self_loop(n: int) -> None:
    spec = GraphSpec(topology="ring4", degree=4)
    adj = build_graph(n, spec, seed=0)
    for i, nbrs in enumerate(adj):
        assert i not in nbrs, f"player {i} appears in its own neighbour list"
        assert sorted(nbrs) == nbrs, "neighbour list not sorted"
        assert len(set(nbrs)) == len(nbrs), "duplicate neighbours"


# ---------------------------------------------------------------------------
# G0-3  lattice4 degree = 4 + torus wrap (corners)
# ---------------------------------------------------------------------------

def test_lattice4_degree() -> None:
    rows, cols = 15, 20
    n = rows * cols
    spec = GraphSpec(topology="lattice4", degree=4, lattice_rows=rows, lattice_cols=cols)
    adj = build_graph(n, spec, seed=0)
    assert len(adj) == n
    for i, nbrs in enumerate(adj):
        assert len(nbrs) == 4, f"player {i} has {len(nbrs)} neighbours, expected 4"


def test_lattice4_torus_wrap() -> None:
    """Corner player (0,0) must connect to (rows-1, 0), (0, cols-1), etc."""
    rows, cols = 4, 5
    n = rows * cols
    spec = GraphSpec(topology="lattice4", degree=4, lattice_rows=rows, lattice_cols=cols)
    adj = build_graph(n, spec, seed=0)
    # Player (0,0) = index 0
    nbrs_0 = adj[0]
    # Expected: up=(rows-1)*cols+0, down=1*cols+0, left=0*cols+(cols-1), right=0*cols+1
    expected = sorted({
        (rows - 1) * cols + 0,  # up (torus wrap)
        1 * cols + 0,           # down
        0 * cols + (cols - 1),  # left (torus wrap)
        0 * cols + 1,           # right
    })
    assert nbrs_0 == expected


# ---------------------------------------------------------------------------
# G0-4  lattice4: no self-loops, sorted, unique
# ---------------------------------------------------------------------------

def test_lattice4_no_self_loop() -> None:
    rows, cols = 6, 7
    n = rows * cols
    spec = GraphSpec(topology="lattice4", degree=4, lattice_rows=rows, lattice_cols=cols)
    adj = build_graph(n, spec, seed=0)
    for i, nbrs in enumerate(adj):
        assert i not in nbrs, f"player {i} appears in its own neighbour list"
        assert sorted(nbrs) == nbrs
        assert len(set(nbrs)) == len(nbrs)


# ---------------------------------------------------------------------------
# G0-5  edge_strategy_distance / edge_disagreement_rate
# ---------------------------------------------------------------------------

def test_edge_strategy_distance_uniform() -> None:
    """When all players have identical weights, distance = 0."""
    spec = GraphSpec(topology="ring4", degree=4)
    adj = build_graph(10, spec, seed=0)
    # All uniform [1/3, 1/3, 1/3]
    weights = [[1/3, 1/3, 1/3]] * 10
    dist = edge_strategy_distance(adj, weights)
    assert abs(dist) < 1e-10


def test_edge_strategy_distance_polarised() -> None:
    """Alternating opposite corner strategies → distance > 0."""
    spec = GraphSpec(topology="ring4", degree=4)
    # n=6 so players 0,2,4 → [1,0,0] and 1,3,5 → [0,0,1]
    adj = build_graph(6, spec, seed=0)
    weights = [[1.0, 0.0, 0.0] if i % 2 == 0 else [0.0, 0.0, 1.0] for i in range(6)]
    dist = edge_strategy_distance(adj, weights)
    assert dist > 0.0


def test_edge_strategy_distance_range() -> None:
    """Distance must be in [0, 2] (L1 of simplex vectors)."""
    spec = GraphSpec(topology="ring4", degree=4)
    adj = build_graph(20, spec, seed=0)
    rng = random.Random(42)
    for _ in range(10):
        ws = []
        for _ in range(20):
            r = [rng.random() for _ in range(3)]
            s = sum(r)
            ws.append([v / s for v in r])
        dist = edge_strategy_distance(adj, ws)
        assert 0.0 <= dist <= 2.0 + 1e-12


def test_edge_disagreement_rate_uniform() -> None:
    """Same strategy for all → disagreement rate = 0."""
    spec = GraphSpec(topology="ring4", degree=4)
    adj = build_graph(8, spec, seed=0)
    strategies = ["aggressive"] * 8
    rate = edge_disagreement_rate(adj, strategies)
    assert abs(rate) < 1e-10


def test_edge_disagreement_rate_alternating() -> None:
    """Alternating strategies → disagreement rate in (0, 1]."""
    spec = GraphSpec(topology="ring4", degree=4)
    adj = build_graph(6, spec, seed=0)
    strategies = [STRAT_SPACE[i % 3] for i in range(6)]
    rate = edge_disagreement_rate(adj, strategies)
    assert 0.0 < rate <= 1.0


def test_edge_disagreement_rate_range() -> None:
    spec = GraphSpec(topology="ring4", degree=4)
    adj = build_graph(12, spec, seed=0)
    rng = random.Random(7)
    strategies = [rng.choice(STRAT_SPACE) for _ in range(12)]
    rate = edge_disagreement_rate(adj, strategies)
    assert 0.0 <= rate <= 1.0


# ---------------------------------------------------------------------------
# IR-1  sample_neighbor always returns a valid neighbor
# ---------------------------------------------------------------------------

def test_sample_neighbor_valid() -> None:
    spec = GraphSpec(topology="ring4", degree=4)
    adj = build_graph(10, spec, seed=0)
    rng = random.Random(99)
    for _ in range(200):
        i = rng.randrange(10)
        j = sample_neighbor(rng, adj, i)
        assert j in adj[i], f"sampled {j} is not in adj[{i}]={adj[i]}"
        assert j != i, "self-loop returned from sample_neighbor"


# ---------------------------------------------------------------------------
# IR-2  Unsupported topology raises ValueError
# ---------------------------------------------------------------------------

def test_build_graph_unsupported_topology() -> None:
    spec = GraphSpec(topology="hypercube", degree=4)
    with pytest.raises(ValueError, match="[Uu]nsupported"):
        build_graph(8, spec, seed=0)


# ---------------------------------------------------------------------------
# IR-3  ring4 raises on < 5 players
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("n", [1, 2, 3, 4])
def test_ring4_too_few_players(n: int) -> None:
    spec = GraphSpec(topology="ring4", degree=4)
    with pytest.raises(ValueError):
        build_graph(n, spec, seed=0)


# ---------------------------------------------------------------------------
# IR-4  lattice4 raises when rows*cols != n_players
# ---------------------------------------------------------------------------

def test_lattice4_mismatched_size() -> None:
    spec = GraphSpec(topology="lattice4", degree=4, lattice_rows=5, lattice_cols=6)
    with pytest.raises(ValueError):
        build_graph(999, spec, seed=0)


def test_lattice4_missing_rows_cols() -> None:
    spec = GraphSpec(topology="lattice4", degree=4)  # no lattice_rows/cols
    with pytest.raises((ValueError, TypeError)):
        build_graph(300, spec, seed=0)
