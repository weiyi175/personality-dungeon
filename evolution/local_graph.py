"""Local graph helper for C-family and T-series experiments.

Builds and manages static, undirected, unweighted neighbour lists for the
local-interaction experiments.

Supported topologies:
    ring4    – every player i is connected to i±1 and i±2 (mod N), degree = 4
    lattice4 – 15×20 torus, every cell connected to its 4 Von-Neumann neighbours
    small_world – Watts–Strogatz model: start from ring-k, rewire with prob p

Architecture invariants
-----------------------
- No I/O (no file reading/writing).
- No dependency on plotting or simulation layers.
- All functions are pure (same inputs → same outputs).
"""

from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass
from math import isfinite


# ---------------------------------------------------------------------------
# GraphSpec
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class GraphSpec:
    """Specification for a static interaction graph.

    topology: 'ring4' | 'lattice4' | 'small_world'
    degree:   4 (first-round only)
    lattice_rows / lattice_cols: required for lattice4, None for others
    p_rewire: rewiring probability for small_world (default 0.0)
    """
    topology: str
    degree: int
    lattice_rows: int | None = None
    lattice_cols: int | None = None
    p_rewire: float = 0.0


# ---------------------------------------------------------------------------
# Graph builder
# ---------------------------------------------------------------------------

def build_graph(n_players: int, graph_spec: GraphSpec, seed: int) -> list[list[int]]:
    """Return an adjacency list (list of neighbour lists).

    Parameters
    ----------
    n_players:
        Total number of players.
    graph_spec:
        Topology specification.
    seed:
        Random seed (currently unused; kept for API consistency and future
        random-graph extension).  Deterministic topologies (ring4, lattice4)
        are constructed deterministically regardless of seed.

    Returns
    -------
    adj : list[list[int]]
        adj[i] is a sorted list of neighbours of player i.
        No self-loops.  adj[i] has exactly graph_spec.degree entries.

    Raises
    ------
    ValueError
        If the topology is unsupported or the player count is incompatible with
        the requested topology.
    """
    topo = str(graph_spec.topology)
    if topo == "ring4":
        return _build_ring4(n_players)
    if topo == "lattice4":
        rows = graph_spec.lattice_rows
        cols = graph_spec.lattice_cols
        if rows is None or cols is None:
            raise ValueError(
                "lattice4 requires lattice_rows and lattice_cols in GraphSpec"
            )
        if int(rows) * int(cols) != int(n_players):
            raise ValueError(
                f"lattice4 requires rows*cols == n_players, "
                f"got {rows}*{cols}={rows*cols} != {n_players}"
            )
        return _build_lattice4(int(rows), int(cols))
    if topo == "small_world":
        return _build_watts_strogatz(
            n_players,
            k=int(graph_spec.degree),
            p_rewire=float(graph_spec.p_rewire),
            seed=int(seed),
        )
    raise ValueError(
        f"Unsupported graph topology: {topo!r}. "
        "Supported topologies: 'ring4', 'lattice4', 'small_world'."
    )


def _build_ring4(n: int) -> list[list[int]]:
    """Build ring-4: player i connects to i-2, i-1, i+1, i+2 (mod n)."""
    if int(n) < 5:
        raise ValueError(f"ring4 requires at least 5 players, got {n}")
    n = int(n)
    adj: list[list[int]] = []
    for i in range(n):
        nbrs = sorted({
            (i - 2) % n,
            (i - 1) % n,
            (i + 1) % n,
            (i + 2) % n,
        })
        adj.append(nbrs)
    return adj


def _build_lattice4(rows: int, cols: int) -> list[list[int]]:
    """Build 15×20 (or any r×c) torus with Von-Neumann neighbourhood.

    Player index: i = r*cols + c.
    """
    rows, cols = int(rows), int(cols)
    n = rows * cols
    adj: list[list[int]] = []
    for r in range(rows):
        for c in range(cols):
            i = r * cols + c
            nbrs = sorted({
                ((r - 1) % rows) * cols + c,   # up
                ((r + 1) % rows) * cols + c,   # down
                r * cols + (c - 1) % cols,     # left
                r * cols + (c + 1) % cols,     # right
            })
            assert len(nbrs) == 4, f"Expected 4 distinct neighbours for ({r},{c})"
            adj.append(nbrs)
    assert len(adj) == n
    return adj


# ---------------------------------------------------------------------------
# Graph queries
# ---------------------------------------------------------------------------

def sample_neighbor(
    rng: random.Random,
    adj: list[list[int]],
    player_index: int,
) -> int:
    """Uniformly sample one neighbour of *player_index* from *adj*.

    Parameters
    ----------
    rng:
        Player-local RNG (deterministic, per blueprint §3 #11).
    adj:
        Adjacency list returned by build_graph().
    player_index:
        Index of the focal player.
    """
    nbrs = adj[int(player_index)]
    if not nbrs:
        raise ValueError(f"Player {player_index} has no neighbours")
    return rng.choice(nbrs)


# ---------------------------------------------------------------------------
# Graph-level statistics (used for diagnostics)
# ---------------------------------------------------------------------------

def edge_strategy_distance(
    adj: list[list[int]],
    weights: list[list[float]],
) -> float:
    """Mean L1 strategy distance over all directed edges (i → j for j in N(i)).

    Parameters
    ----------
    adj:
        Adjacency list.
    weights:
        weights[i] = [w_aggressive, w_defensive, w_balanced] for player i
        (simplex normalised, i.e. sum=1).
    """
    total = 0.0
    count = 0
    for i, nbrs in enumerate(adj):
        wi = weights[i]
        for j in nbrs:
            wj = weights[j]
            total += sum(abs(float(a) - float(b)) for a, b in zip(wi, wj))
            count += 1
    if count == 0:
        return 0.0
    return float(total / count)


def edge_disagreement_rate(
    adj: list[list[int]],
    strategies: list[str],
) -> float:
    """Fraction of directed edges (i, j) where strategies[i] != strategies[j]."""
    total = 0
    disagreements = 0
    for i, nbrs in enumerate(adj):
        si = strategies[i]
        for j in nbrs:
            if strategies[j] != si:
                disagreements += 1
            total += 1
    if total == 0:
        return 0.0
    return float(disagreements) / float(total)


# ---------------------------------------------------------------------------
# Watts–Strogatz small-world builder
# ---------------------------------------------------------------------------

def _build_watts_strogatz(
    n: int,
    *,
    k: int,
    p_rewire: float,
    seed: int,
) -> list[list[int]]:
    """Build a Watts–Strogatz small-world graph.

    1. Start from a ring where each node connects to k/2 neighbours on each side.
    2. For each edge (i, j) with j > i in the clockwise direction, rewire j
       to a uniformly random node with probability p_rewire.

    Parameters
    ----------
    n : int
        Number of nodes.  Must be >= k+1.
    k : int
        Even degree of the initial ring (each node has k neighbours).
    p_rewire : float
        Rewiring probability in [0, 1].
    seed : int
        Random seed for reproducible rewiring.

    Returns
    -------
    list[list[int]]
        Adjacency list (sorted, undirected, no self-loops).
    """
    if k % 2 != 0:
        raise ValueError(f"k must be even, got {k}")
    if n < k + 1:
        raise ValueError(f"Need n >= k+1, got n={n}, k={k}")
    if not (0.0 <= p_rewire <= 1.0):
        raise ValueError(f"p_rewire must be in [0,1], got {p_rewire}")

    half_k = k // 2
    # Build initial ring adjacency as sets
    adj_sets: list[set[int]] = [set() for _ in range(n)]
    for i in range(n):
        for offset in range(1, half_k + 1):
            j = (i + offset) % n
            adj_sets[i].add(j)
            adj_sets[j].add(i)

    # Rewire
    rng = random.Random(seed)
    for i in range(n):
        for offset in range(1, half_k + 1):
            j = (i + offset) % n
            if j not in adj_sets[i]:
                continue  # already rewired away
            if rng.random() < p_rewire:
                # Pick a new target (not self, not already neighbour)
                candidates = [
                    v for v in range(n) if v != i and v not in adj_sets[i]
                ]
                if not candidates:
                    continue
                new_j = rng.choice(candidates)
                adj_sets[i].discard(j)
                adj_sets[j].discard(i)
                adj_sets[i].add(new_j)
                adj_sets[new_j].add(i)

    return [sorted(s) for s in adj_sets]


# ---------------------------------------------------------------------------
# Graph-level diagnostics (T-series)
# ---------------------------------------------------------------------------

def graph_clustering_coefficient(adj: list[list[int]]) -> float:
    """Compute the average local clustering coefficient.

    For each node i, the local clustering coefficient is the fraction of
    pairs of neighbours of i that are themselves connected.
    """
    n = len(adj)
    if n == 0:
        return 0.0
    total_cc = 0.0
    for i in range(n):
        nbrs = adj[i]
        ki = len(nbrs)
        if ki < 2:
            continue
        nbr_set = set(nbrs)
        triangles = 0
        for idx_a in range(ki):
            for idx_b in range(idx_a + 1, ki):
                if nbrs[idx_b] in set(adj[nbrs[idx_a]]):
                    triangles += 1
        total_cc += 2.0 * triangles / (ki * (ki - 1))
    return float(total_cc / n)


def graph_mean_shortest_path_approx(
    adj: list[list[int]],
    sample_size: int = 50,
    seed: int = 0,
) -> float:
    """Estimate mean shortest path length via BFS from a random sample of nodes.

    Parameters
    ----------
    adj : list[list[int]]
        Adjacency list.
    sample_size : int
        Number of source nodes to BFS from.
    seed : int
        Random seed for node sampling.
    """
    n = len(adj)
    if n < 2:
        return 0.0
    rng = random.Random(seed)
    sources = rng.sample(range(n), min(sample_size, n))
    total_dist = 0.0
    total_pairs = 0
    for src in sources:
        dist = [-1] * n
        dist[src] = 0
        queue: deque[int] = deque([src])
        while queue:
            u = queue.popleft()
            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    queue.append(v)
        for d in dist:
            if d > 0:
                total_dist += d
                total_pairs += 1
    return float(total_dist / total_pairs) if total_pairs > 0 else 0.0


def spatial_autocorrelation(
    adj: list[list[int]],
    all_weights_simplex: list[list[float]],
    *,
    max_hop: int = 2,
) -> dict[int, float]:
    """Compute mean dot-product similarity at distance 1..max_hop.

    Returns {hop: mean_dot_product}.  Measures how correlated w_i and w_j
    are at various graph distances.
    """
    n = len(adj)
    if n == 0:
        return {h: 0.0 for h in range(1, max_hop + 1)}

    # global mean for centring
    nstrats = len(all_weights_simplex[0]) if all_weights_simplex else 3
    gmean = [0.0] * nstrats
    for wi in all_weights_simplex:
        for k in range(nstrats):
            gmean[k] += wi[k]
    gmean = [v / n for v in gmean]

    # For each hop, accumulate
    hop_total: dict[int, float] = {h: 0.0 for h in range(1, max_hop + 1)}
    hop_count: dict[int, int] = {h: 0 for h in range(1, max_hop + 1)}

    # BFS from every node (small N=300 is fine)
    for i in range(n):
        dist = [-1] * n
        dist[i] = 0
        queue: deque[int] = deque([i])
        while queue:
            u = queue.popleft()
            if dist[u] >= max_hop:
                continue
            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    queue.append(v)
        wi_c = [all_weights_simplex[i][k] - gmean[k] for k in range(nstrats)]
        for j in range(n):
            d = dist[j]
            if 1 <= d <= max_hop:
                wj_c = [all_weights_simplex[j][k] - gmean[k] for k in range(nstrats)]
                dot = sum(wi_c[k] * wj_c[k] for k in range(nstrats))
                hop_total[d] += dot
                hop_count[d] += 1

    return {
        h: float(hop_total[h] / hop_count[h]) if hop_count[h] > 0 else 0.0
        for h in range(1, max_hop + 1)
    }
