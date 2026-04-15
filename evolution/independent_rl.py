"""D1 Independent Reinforcement Learning helpers.

Per-player Q-table update and Boltzmann softmax selection.
No imitation, no broadcast — each player learns independently
from its own reward stream.

Update rule (per player i each round):
    1.  Select strategy via Boltzmann softmax:
            Pr(s) = exp(β·Q_i(s)) / Σ exp(β·Q_i(s'))
    2.  Observe one-hot payoff against neighbors:
            r_i = (1/|N(i)|) Σ_{j∈N(i)} e_{s_i}^T · A · e_{s_j}
    3.  Update Q-table (exponential recency):
            Q_i(chosen)  ← (1-α)·Q_i(chosen) + α·r_i
            Q_i(other)   ← (1-α)·Q_i(other)             # forgetting

Architecture invariants
-----------------------
- No I/O (no file reading/writing).
- No dependency on plotting, simulation, or analysis layers.
- All public functions are pure.
"""

from __future__ import annotations

import random
from math import exp, isfinite, log


STRATEGY_SPACE = ["aggressive", "defensive", "balanced"]
_NSTRATS = len(STRATEGY_SPACE)

# Strategy index map
_SIDX = {s: i for i, s in enumerate(STRATEGY_SPACE)}


# ---------------------------------------------------------------------------
# Payoff (one-hot strategy vs one-hot strategy)
# ---------------------------------------------------------------------------

def strategy_payoff_matrix(*, a: float, b: float, cross: float) -> list[list[float]]:
    """Return the 3×3 payoff matrix A[i][j] = payoff of strategy i vs strategy j.

    A = [[0, a, -b],
         [-b, 0, a],
         [a, -b, 0]]
    plus cross-coupling terms.
    """
    a, b, cr = float(a), float(b), float(cross)
    return [
        [0.0,             a - cr,          -b],
        [-b - cr,         0.0,             a],
        [a + cr,          -b + cr,         0.0],
    ]


def one_hot_local_payoff(
    strategy_i: int,
    neighbor_strategies: list[int],
    payoff_mat: list[list[float]],
) -> float:
    """Compute player i's payoff from one-hot strategy interaction.

    r_i = (1/|N(i)|) Σ_{j∈N(i)} A[s_i][s_j]
    """
    if not neighbor_strategies:
        return 0.0
    row = payoff_mat[strategy_i]
    total = sum(row[sj] for sj in neighbor_strategies)
    return float(total / len(neighbor_strategies))


# ---------------------------------------------------------------------------
# Q-table update
# ---------------------------------------------------------------------------

def rl_q_update(
    q_values: list[float],
    chosen_idx: int,
    reward: float,
    *,
    alpha: float,
) -> list[float]:
    """Update Q-table with exponential recency.

    Q(chosen)  ← (1-α)·Q(chosen) + α·reward
    Q(other)   ← (1-α)·Q(other)                # forgetting/decay

    Parameters
    ----------
    q_values : list[float]
        Current Q-values [Q_agg, Q_def, Q_bal].
    chosen_idx : int
        Index of the strategy played this round.
    reward : float
        Realized payoff this round.
    alpha : float
        Learning rate ∈ [0, 1].

    Returns
    -------
    list[float]
        Updated Q-values.
    """
    a = float(alpha)
    new_q = []
    for i, q in enumerate(q_values):
        if i == chosen_idx:
            new_q.append((1.0 - a) * float(q) + a * float(reward))
        else:
            new_q.append((1.0 - a) * float(q))
    return new_q


# ---------------------------------------------------------------------------
# Boltzmann softmax selection
# ---------------------------------------------------------------------------

def boltzmann_weights(q_values: list[float], *, beta: float) -> list[float]:
    """Convert Q-values to probability simplex via Boltzmann softmax.

    w(s) = exp(β·Q(s)) / Σ exp(β·Q(s'))

    Numerically stable via max subtraction.
    """
    b = float(beta)
    scaled = [b * float(q) for q in q_values]
    max_s = max(scaled)
    exps = [exp(s - max_s) for s in scaled]
    total = sum(exps)
    if total <= 0.0 or not isfinite(total):
        n = len(q_values)
        return [1.0 / n] * n
    return [e / total for e in exps]


def boltzmann_select(
    q_values: list[float],
    *,
    beta: float,
    rng: random.Random,
) -> int:
    """Select strategy index via Boltzmann softmax sampling.

    Returns the index of the chosen strategy.
    """
    ws = boltzmann_weights(q_values, beta=beta)
    r = rng.random()
    cum = 0.0
    for i, w in enumerate(ws):
        cum += w
        if r <= cum:
            return i
    return len(ws) - 1


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------

def weight_entropy(w: list[float]) -> float:
    """Shannon entropy of a weight/probability vector."""
    h = 0.0
    for v in w:
        if v > 1e-15:
            h -= v * log(v)
    return float(h)


def q_value_mean(q_values: list[float]) -> float:
    """Mean of Q-values (scalar summary per player)."""
    return float(sum(q_values) / len(q_values)) if q_values else 0.0


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Cosine similarity between two vectors."""
    dot = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(x * x for x in b) ** 0.5
    if na < 1e-15 or nb < 1e-15:
        return 0.0
    return float(dot / (na * nb))


# ---------------------------------------------------------------------------
# F1: Per-player payoff perturbation
# ---------------------------------------------------------------------------

def sample_payoff_perturbation(
    n_players: int, *, epsilon: float, rng: random.Random,
) -> list[list[float]]:
    """Sample per-player payoff bias δ_i[s] ~ Uniform(-ε, +ε).

    Each player gets a 3-element perturbation vector drawn independently.
    If epsilon <= 0, returns all-zero vectors (no perturbation).

    Parameters
    ----------
    n_players : int
        Number of players.
    epsilon : float
        Half-width of perturbation range.
    rng : random.Random
        Seeded RNG for reproducibility.

    Returns
    -------
    list[list[float]]
        Per-player perturbation vectors, shape (n_players, 3).
    """
    eps = float(epsilon)
    if eps <= 0.0:
        return [[0.0] * _NSTRATS for _ in range(n_players)]
    return [
        [rng.uniform(-eps, eps) for _ in range(_NSTRATS)]
        for _ in range(n_players)
    ]


def apply_per_player_payoff_perturbation(
    rewards: list[float],
    chosen: list[int],
    perturbations: list[list[float]],
) -> list[float]:
    """Add per-player payoff bias to rewards.

    r_i' = r_i + δ_i[chosen_i]

    Parameters
    ----------
    rewards : list[float]
        Base rewards from ``one_hot_local_payoff``.
    chosen : list[int]
        Strategy index chosen by each player this round.
    perturbations : list[list[float]]
        Per-player perturbation vectors from ``sample_payoff_perturbation``.

    Returns
    -------
    list[float]
        Perturbed rewards.
    """
    return [
        float(rewards[i]) + float(perturbations[i][chosen[i]])
        for i in range(len(rewards))
    ]


# ---------------------------------------------------------------------------
# Q-Rotation Bias (Phase 20)
# ---------------------------------------------------------------------------

def apply_q_rotation_bias(
    q_values: list[float],
    *,
    delta_rot: float,
) -> list[float]:
    """Apply antisymmetric tangential drift to Q-values.

    Rotation direction: A→D→B→A (indices 0→1→2→0).

    For each Q-vector, compute centered values then add:
        drift[A] = δ_rot × (centered[D] - centered[B])
        drift[D] = δ_rot × (centered[B] - centered[A])
        drift[B] = δ_rot × (centered[A] - centered[D])

    This is a zero-sum perturbation on the simplex tangent plane.
    When δ_rot=0.0, returns q_values unchanged.

    Parameters
    ----------
    q_values : list[float]
        Current Q-values [Q_agg, Q_def, Q_bal].
    delta_rot : float
        Rotation bias strength. 0.0 = no bias.

    Returns
    -------
    list[float]
        Q-values with rotation bias applied.
    """
    d = float(delta_rot)
    if d == 0.0:
        return q_values
    q_mean = sum(q_values) / len(q_values)
    c = [v - q_mean for v in q_values]
    # A(0)→D(1)→B(2)→A(0)
    return [
        q_values[0] + d * (c[1] - c[2]),  # A gets (D - B)
        q_values[1] + d * (c[2] - c[0]),  # D gets (B - A)
        q_values[2] + d * (c[0] - c[1]),  # B gets (A - D)
    ]
