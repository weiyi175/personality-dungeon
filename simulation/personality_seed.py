"""Sub-critical personality seeding for P7-H.

Places an RL population *near the baseline attractor with proximity headroom*, so
the bifurcation DV is not saturated from round 1. This is the initial-condition
regime under which the validated H1 effect (d≈4.4) was measured: the engine-sim
and player-test drivers seed a single P₀ via ``sample_initial(headroom)`` and run
``personality_mode="static"``. The live ``/rl_sessions`` path could not reproduce
it because the request schema exposed no way to place the population sub-critical
(none/static/random_9persona all start far from baseline). This module is the
importable home for that seeding so the API can offer it.

Definition (identical to run_p7h_player_test.sample_initial):

    P₀ = BASELINE_ATTRACTOR + headroom · (jittered_real_attractor − BASELINE_ATTRACTOR)

``headroom=1.0`` reaches the full P7-F attractor spread; ``0.5`` sits halfway to
baseline (leaving proximity headroom for events to drive measurable displacement).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from simulation.bifurcation_detector import BASELINE_ATTRACTOR, FEATURE_NAMES

_ROOT = Path(__file__).resolve().parents[1]
_ATTRACTOR_JSON = (
    _ROOT / "reports" / "experiments" / "p7f_attractor_mapping"
    / "p7f_attractor_coordinates.json"
)

_pool_cache: np.ndarray | None = None


def load_attractor_pool() -> np.ndarray:
    """Return the (N, 9) pool of real P7-F attractor coordinates (FEATURE_NAMES order).

    Spans the real range of personality states the system produces across
    α=0.1–0.4. Cached after first load (the file is read-only).
    """
    global _pool_cache
    if _pool_cache is None:
        coords = json.loads(_ATTRACTOR_JSON.read_text())
        pts = [
            [a[f] for f in FEATURE_NAMES]
            for _alpha, attractors in coords.items()
            for a in attractors
        ]
        _pool_cache = np.asarray(pts, dtype=float)
    return _pool_cache


def sample_sub_critical(
    rng: np.random.RandomState,
    headroom: float = 0.5,
    pool: np.ndarray | None = None,
) -> np.ndarray:
    """Sample one sub-critical Space-A personality near the baseline attractor.

    Picks a random real attractor, adds small jitter, then scales the offset from
    baseline by ``headroom`` so the point starts sub-critical. Returns a 9D vector
    in FEATURE_NAMES order.
    """
    if not 0.0 < headroom <= 1.0:
        raise ValueError(f"headroom must be in (0, 1], got {headroom}")
    if pool is None:
        pool = load_attractor_pool()
    base = pool[rng.randint(len(pool))]
    jittered = base + rng.randn(9) * 0.01
    return BASELINE_ATTRACTOR + headroom * (jittered - BASELINE_ATTRACTOR)
