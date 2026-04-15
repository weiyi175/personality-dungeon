"""C2 local mini-batch replicator helper.

Implements the per-player update semantics defined in blueprint §6.2.

Update rule (synchronous, per player i each round):
    1.  Collect local batch B_i = {i} ∪ N(i)  (ego + one-hop neighbours)
    2.  Estimate local popularity:
            x_i = (1/|B_i|) Σ_{k ∈ B_i} w_k   (equal-weight mean of simplex weights)
    3.  Compute local payoff vector (A applied to x_i):
            u_i = A x_i
    4.  Compute local advantage (centred replicator):
            g_i = u_i - <w_i, u_i> · 1
        Property: <w_i, g_i> = 0  (orthogonal to current strategy, always)
    5.  Replicator-like exponential update:
            w_i' = normalise(w_i ⊙ exp(k_local · g_i))

Architecture invariants
-----------------------
- No I/O (no file reading/writing).
- No dependency on plotting, simulation, or analysis layers.
- All public functions are pure.
"""

from __future__ import annotations

from math import exp, isfinite, sqrt


STRATEGY_SPACE = ["aggressive", "defensive", "balanced"]
_NSTRATS = 3


# ---------------------------------------------------------------------------
# Payoff matrix (inline, same as _matrix_ab_payoff_vec in run_simulation.py)
# ---------------------------------------------------------------------------

def _payoff_vec(
    x: list[float],
    *,
    a: float,
    b: float,
    cross: float,
) -> list[float]:
    """Compute u = A·x for the cyclic 3-strategy matrix.

    A = [[0, a, -b], [-b, 0, a], [a, -b, 0]]
    plus cross-coupling: u[0] -= cross*x[1], u[1] -= cross*x[0], u[2] += cross*(x[0]+x[1])

    Parameters
    ----------
    x:
        [x_agg, x_def, x_bal] – simplex population frequency vector.
    a, b, cross:
        Payoff matrix parameters.

    Returns
    -------
    list[float]
        [u_agg, u_def, u_bal]
    """
    x0, x1, x2 = float(x[0]), float(x[1]), float(x[2])
    a, b, cross = float(a), float(b), float(cross)
    u0 = a * x1 - b * x2
    u1 = -b * x0 + a * x2
    u2 = a * x0 - b * x1
    if cross != 0.0:
        u0 -= cross * x1
        u1 -= cross * x0
        u2 += cross * (x0 + x1)
    return [u0, u1, u2]


# ---------------------------------------------------------------------------
# Local popularity estimation
# ---------------------------------------------------------------------------

def local_popularity(
    player_weights: list[list[float]],
    batch_indices: list[int],
) -> list[float]:
    """Estimate local popularity as equal-weight mean over the batch.

    Parameters
    ----------
    player_weights:
        player_weights[i] = [w_agg, w_def, w_bal] for all players.
        Must be simplex-normalised (sum ≈ 1).
    batch_indices:
        Indices of players in the local batch B_i = {i} ∪ N(i).

    Returns
    -------
    list[float]
        [x_agg, x_def, x_bal] – equal-weight mean over the batch.
    """
    n = len(batch_indices)
    if n == 0:
        return [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0]
    totals = [0.0, 0.0, 0.0]
    for idx in batch_indices:
        wi = player_weights[idx]
        for k in range(_NSTRATS):
            totals[k] += float(wi[k])
    return [t / n for t in totals]


# ---------------------------------------------------------------------------
# Local advantage (replicator centring)
# ---------------------------------------------------------------------------

def local_replicator_advantage(
    weights_i: list[float],
    local_pop: list[float],
    *,
    a: float,
    b: float,
    cross: float,
) -> list[float]:
    """Compute the local advantage vector for player i.

        g_i = u_i - <w_i, u_i> · 1

    where u_i = A · x_i  (local payoff vector).

    The inner product <w_i, u_i> is the expected payoff of player i
    under its current mixed strategy against the local population x_i.

    Property: <w_i, g_i> = 0  (advantage is centred, always).

    Parameters
    ----------
    weights_i:
        [w_agg, w_def, w_bal] – player i's current simplex weights.
    local_pop:
        [x_agg, x_def, x_bal] – local popularity estimate from local_popularity().
    a, b, cross:
        Payoff matrix parameters.

    Returns
    -------
    list[float]
        [g_agg, g_def, g_bal]
    """
    u = _payoff_vec(local_pop, a=a, b=b, cross=cross)
    # expected payoff = w_i · u
    expected = sum(float(weights_i[k]) * u[k] for k in range(_NSTRATS))
    return [u[k] - expected for k in range(_NSTRATS)]


# ---------------------------------------------------------------------------
# Replicator-like exponential update
# ---------------------------------------------------------------------------

def local_replicator_update(
    weights_i: list[float],
    advantage_i: list[float],
    *,
    k_local: float,
) -> list[float]:
    """Apply the replicator-like exponential update.

        w_i' = normalise(w_i ⊙ exp(k_local · g_i))

    Parameters
    ----------
    weights_i:
        [w_agg, w_def, w_bal] – current simplex weights.
    advantage_i:
        [g_agg, g_def, g_bal] – from local_replicator_advantage().
    k_local:
        Local selection strength (≥ 0).  k_local = 0 → exact no-op.

    Returns
    -------
    list[float]
        Normalised weight vector (sum = 1, all entries > 0).

    Raises
    ------
    ValueError
        If k_local < 0 or any resulting raw weight is ≤ 0 before normalisation.
    """
    k = float(k_local)
    if k < 0.0:
        raise ValueError(f"k_local must be >= 0, got {k}")
    if k == 0.0:
        # Exact no-op: normalise only (preserves simplex without touching values)
        return _normalise(list(weights_i))
    raw = [
        float(weights_i[j]) * exp(max(-500.0, min(500.0, k * float(advantage_i[j]))))
        for j in range(_NSTRATS)
    ]
    return _normalise(raw)


# ---------------------------------------------------------------------------
# Normalisation
# ---------------------------------------------------------------------------

def _normalise(w: list[float]) -> list[float]:
    """Simplex normalisation: w_k / sum(w).

    Raises ValueError if sum ≤ 0 or any resulting entry is not strictly positive.
    """
    total = sum(float(v) for v in w)
    if not isfinite(total) or total <= 0.0:
        raise ValueError(
            f"Cannot normalise weight vector with non-positive sum={total}: {w}"
        )
    result = [float(v) / total for v in w]
    for k, v in enumerate(result):
        if not isfinite(v) or v <= 0.0:
            raise ValueError(
                f"Normalised weight[{k}]={v} is not positive. Input: {w}"
            )
    return result


# ---------------------------------------------------------------------------
# Diagnostic helpers
# ---------------------------------------------------------------------------

def local_growth_norm(advantage: list[float]) -> float:
    """L2 norm of the advantage vector."""
    return float(sqrt(sum(float(v) ** 2 for v in advantage)))


def local_growth_cosine_vs_global(
    local_adv: list[float],
    global_adv: list[float],
) -> float:
    """Cosine similarity between local advantage and global advantage.

    Returns 1.0 if vectors are identical, <1.0 if they diverge.
    Returns 1.0 if either vector is (~)zero to avoid division by zero.
    """
    dot = sum(float(la) * float(ga) for la, ga in zip(local_adv, global_adv))
    n_l = sqrt(sum(float(v) ** 2 for v in local_adv))
    n_g = sqrt(sum(float(v) ** 2 for v in global_adv))
    if n_l < 1e-15 or n_g < 1e-15:
        return 1.0
    return float(max(-1.0, min(1.0, dot / (n_l * n_g))))


def player_growth_dispersion(advantages: list[list[float]]) -> float:
    """Mean L2 norm of per-player advantage vectors.

    High dispersion = players are being pushed in different directions
    = local diversity is forming.  Near-zero = all players face the same
    local growth (well-mixed-like).
    """
    if not advantages:
        return 0.0
    return float(sum(local_growth_norm(adv) for adv in advantages) / len(advantages))


def batch_phase_spread(
    adj: list[list[int]],
    all_weights_simplex: list[list[float]],
) -> float:
    """Mean pairwise phase-angle difference over all ego-to-batch-member edges.

    Phase angle: atan2(sqrt(3)*(w_def - w_bal), 2*w_agg - w_def - w_bal).
    """
    from math import atan2, pi as _PI

    total = 0.0
    count = 0
    for i, nbrs in enumerate(adj):
        wi = all_weights_simplex[i]
        phi_i = atan2(
            (3.0 ** 0.5) * (wi[1] - wi[2]),
            2.0 * wi[0] - wi[1] - wi[2],
        )
        # ego vs each batch member (excluding ego itself)
        for j in nbrs:
            wj = all_weights_simplex[j]
            phi_j = atan2(
                (3.0 ** 0.5) * (wj[1] - wj[2]),
                2.0 * wj[0] - wj[1] - wj[2],
            )
            diff = abs(phi_i - phi_j)
            if diff > _PI:
                diff = 2.0 * _PI - diff
            total += diff
            count += 1
    return float(total / count) if count > 0 else 0.0


def neighbor_popularity_dispersion(
    adj: list[list[int]],
    all_weights_simplex: list[list[float]],
) -> float:
    """Mean L1 norm of (local_popularity_i - global_mean) over all players.

    Measures how much each player's local neighbourhood deviates from
    the global mean frequency.  High = spatial heterogeneity in popularity.
    """
    n = len(adj)
    if n == 0:
        return 0.0
    # Global mean
    global_mean = [0.0] * _NSTRATS
    for wi in all_weights_simplex:
        for k in range(_NSTRATS):
            global_mean[k] += wi[k]
    global_mean = [v / n for v in global_mean]

    total = 0.0
    for i, nbrs in enumerate(adj):
        batch = [i] + list(nbrs)
        loc_pop = local_popularity(all_weights_simplex, batch)
        total += sum(abs(loc_pop[k] - global_mean[k]) for k in range(_NSTRATS))
    return float(total / n)


# ---------------------------------------------------------------------------
# Weights ↔ list conversion helpers
# ---------------------------------------------------------------------------

def weights_to_simplex(w: dict[str, float]) -> list[float]:
    """Convert {strategy: weight} dict to normalised simplex list [w_agg, w_def, w_bal]."""
    raw = [float(w.get(s, 1.0)) for s in STRATEGY_SPACE]
    total = sum(raw)
    if total <= 0.0:
        return [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0]
    return [v / total for v in raw]


def list_to_weights(lst: list[float]) -> dict[str, float]:
    """Convert ordered [w_agg, w_def, w_bal] back to {strategy: weight} dict."""
    return {s: float(lst[k]) for k, s in enumerate(STRATEGY_SPACE)}
