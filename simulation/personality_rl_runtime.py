"""Personality RL Runtime – bridge between BL2 RL dynamics and Personality Dungeon.

Bridge spec §3.3.  Orchestrates:
  1. Initialise RLPlayer population with personality → RL mapping
  2. Run BL2-compatible RL loop  (via evolution/independent_rl.py)
  3. Optional event overlay       (smoke subset from event templates)
  4. Output round-level CSV, seed-level provenance JSON, per-player snapshot TSV

Usage
-----
    ./venv/bin/python -m simulation.personality_rl_runtime \\
        --seeds 42,43,44 \\
        --personality-mode none \\
        --out-dir outputs/personality_rl

Architecture: simulation/ layer.  Handles I/O and orchestration only.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
from dataclasses import dataclass, field
from math import sqrt
from pathlib import Path
from typing import Any

from evolution.independent_rl import (
    STRATEGY_SPACE,
    _NSTRATS,
    apply_per_player_payoff_perturbation,
    boltzmann_select,
    boltzmann_weights,
    one_hot_local_payoff,
    rl_q_update,
    sample_payoff_perturbation,
    strategy_payoff_matrix,
)
from evolution.local_graph import GraphSpec, build_graph
from players.rl_player import (
    RLPlayer,
    init_rl_player,
    sample_personality,
)


# ===================================================================
# Config
# ===================================================================

@dataclass
class PersonalityRLConfig:
    """Configuration for a single personality-RL run."""

    n_players: int = 300
    n_rounds: int = 12000

    # ---- BL2 locked parameters ----
    alpha_lo: float = 0.005
    alpha_hi: float = 0.40
    beta: float = 3.0
    payoff_epsilon: float = 0.02
    strategy_alpha_multipliers: list[float] = field(
        default_factory=lambda: [1.2, 1.0, 0.8],
    )
    a: float = 1.0
    b: float = 0.9
    cross: float = 0.20
    init_q: float = 0.0
    topology: str = "well_mixed"
    control_degree: int = 4
    burn_in: int = 4000
    tail: int = 4000

    # ---- Personality modulation ----
    personality_mode: str = "none"      # "none" | "random"
    lambda_alpha: float = 0.0
    lambda_beta: float = 0.0
    lambda_r: float = 0.0               # strategy multiplier offset strength
    lambda_risk: float = 0.0            # risk sensitivity strength
    lambda_beta_comp: float = 0.0       # adaptive β compensation for low-z_β players

    # ---- Event integration (smoke subset) ----
    events_json: str = ""               # path to event templates JSON; "" = off
    event_rate: float = 0.0             # probability of event per round
    event_reward_scale: float = 0.01    # scale factor for event reward modifiers
    event_reward_mode: str = "additive"  # additive | multiplicative
    event_reward_multiplier_cap: float = 0.25  # used in multiplicative mode
    event_impact_mode: str = "instant"  # instant | spread
    event_impact_horizon: int = 1        # spread horizon in rounds
    event_impact_decay: float = 0.70     # geometric decay kernel base
    event_per_player: bool = False       # K4: per-player independent firing (breaks correlated noise)
    event_risk_enabled: bool = True      # K3: False = zero out risk/stress from events
    event_warmup_rounds: int = 0         # K1: skip events before this round (0 = immediate)
    event_reward_clamp: float = 0.0      # K2: max |reward_mod| per event (0 = no clamp)
    # ---- B1 async-dispatch contract ----
    event_dispatch_mode: str = "sync"   # sync | async_round_robin | async_poisson
    event_dispatch_target_rate: float = 0.0
    event_dispatch_batch_size: int = 0
    event_dispatch_seed_offset: int = 0
    event_dispatch_fairness_window: int = 200
    event_dispatch_fairness_tolerance: float = 0.15

    # ---- World feedback (Little Dragon) ----
    world_feedback: bool = False        # enable adaptive world state
    lambda_world: float = 0.08          # world update gain
    world_update_interval: int = 200    # rounds between world updates

    # ---- Output ----
    out_dir: str = "outputs/personality_rl"


# ===================================================================
# Event Bridge – smoke subset adapter (bridge spec §5.3)
# ===================================================================

_PERSONALITY_KEYS_ORDERED = [
    "impulsiveness", "caution", "greed", "optimism", "suspicion",
    "persistence", "randomness", "stability_seeking", "ambition",
    "patience", "curiosity", "fearfulness",
]


class EventBridge:
    """Minimal event adapter for smoke testing.

    Loads event templates JSON and computes personality-aligned
    reward / risk modifiers without depending on the full EventLoader.
    """

    def __init__(self, json_path: str | Path) -> None:
        with open(json_path, encoding="utf-8") as f:
            data = json.load(f)
        self.events: list[dict[str, Any]] = data.get("templates", [])
        if not self.events:
            raise ValueError(f"No event templates found in {json_path}")

    def sample_event(self, *, rng: random.Random) -> dict[str, Any]:
        return rng.choice(self.events)

    def compute_reward_risk(
        self,
        event: dict[str, Any],
        personality: dict[str, float],
        *,
        scale: float,
        rng: random.Random,
    ) -> tuple[float, float]:
        """Return ``(reward_mod, risk_mod)`` for a player facing *event*.

        Selects the best personality-aligned action, then scales its
        utility and base_risk.
        """
        actions = event.get("actions", [])
        if not actions:
            return 0.0, 0.0

        best_u = -1e9
        best_risk = 0.0
        for act in actions:
            ws = act.get("weights", [0.0] * 12)
            u = sum(
                ws[i] * personality.get(_PERSONALITY_KEYS_ORDERED[i], 0.0)
                for i in range(min(len(ws), 12))
            )
            noise_amp = abs(personality.get("randomness", 0.0))
            if noise_amp > 0:
                u += rng.gauss(0, 0.01 * noise_amp)
            if u > best_u:
                best_u = u
                best_risk = float(act.get("base_risk", 0.0))

        return best_u * scale, best_risk * scale


# ===================================================================
# World Feedback – Little Dragon adapter (blueprint §5.4)
# ===================================================================

_WORLD_DIMS = ("scarcity", "threat", "noise", "intel")
_EVENT_FAMILIES = ("Threat", "Resource", "Uncertainty", "Navigation", "Internal")

# B_P: how strategy proportion deviations drive world state
# Keys: world dim → (agg_coeff, def_coeff, bal_coeff)
_B_P: dict[str, tuple[float, float, float]] = {
    "scarcity": (0.60, 0.20, -0.40),
    "threat":   (0.60, -0.10, -0.30),
    "noise":    (-0.25, -0.25, 0.40),
    "intel":    (-0.30, -0.10, 0.45),
}

_REWARD_TARGET = 0.27

# B_E: how event shares × reward gap drive world state
# Keys: world dim → (Threat, Resource, Uncertainty, Navigation, Internal)
_B_E: dict[str, tuple[float, float, float, float, float]] = {
    "scarcity": (-0.10, 0.40, 0.10, -0.10, 0.10),
    "threat":   (0.45, -0.15, 0.10, -0.05, 0.05),
    "noise":    (0.05, -0.10, 0.35, -0.05, 0.30),
    "intel":    (-0.20, 0.15, -0.20, 0.40, -0.05),
}


def _clamp_world(v: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, v))


def _world_to_event_weights(st: dict[str, float]) -> dict[str, float]:
    """Map world state → event family sampling weights."""
    sc = st["scarcity"]; th = st["threat"]; no = st["noise"]; it = st["intel"]
    return {
        "Threat":      _clamp_world(1.0 + 0.80*(th-0.5) + 0.30*(no-0.5) - 0.20*(it-0.5), 0.20, 3.0),
        "Resource":    _clamp_world(1.0 - 0.80*(sc-0.5) + 0.35*(it-0.5), 0.20, 3.0),
        "Uncertainty": _clamp_world(1.0 + 0.90*(no-0.5) + 0.20*(th-0.5), 0.20, 3.0),
        "Navigation":  _clamp_world(1.0 + 0.75*(it-0.5) - 0.30*(no-0.5), 0.20, 3.0),
        "Internal":    _clamp_world(1.0 + 0.55*(no-0.5) + 0.25*(sc-0.5) - 0.20*(it-0.5), 0.20, 3.0),
    }


def _world_to_reward_mult(st: dict[str, float]) -> dict[str, float]:
    sc = st["scarcity"]; th = st["threat"]; no = st["noise"]; it = st["intel"]
    return {
        "Threat":      _clamp_world(1.0 + 0.20*(th-0.5), 0.85, 1.15),
        "Resource":    _clamp_world(1.0 - 0.30*(sc-0.5), 0.85, 1.15),
        "Uncertainty": _clamp_world(1.0 + 0.40*(no-0.5), 0.80, 1.20),
        "Navigation":  _clamp_world(1.0 + 0.30*(it-0.5), 0.85, 1.15),
        "Internal":    _clamp_world(1.0 + 0.40*(no-0.5), 0.80, 1.20),
    }


def _world_to_risk_mult(st: dict[str, float]) -> dict[str, float]:
    sc = st["scarcity"]; th = st["threat"]; no = st["noise"]; it = st["intel"]
    return {
        "Threat":      _clamp_world(1.0 + 0.30*(th-0.5), 0.85, 1.15),
        "Resource":    _clamp_world(1.0 + 0.15*(sc-0.5) - 0.10*(it-0.5), 0.90, 1.10),
        "Uncertainty": _clamp_world(1.0 + 0.35*(no-0.5), 0.85, 1.15),
        "Navigation":  _clamp_world(1.0 - 0.20*(it-0.5) + 0.10*(no-0.5), 0.90, 1.10),
        "Internal":    _clamp_world(1.0 + 0.20*(no-0.5) + 0.10*(sc-0.5), 0.90, 1.10),
    }


class WorldFeedback:
    """Little Dragon world feedback for the RL runtime.

    Reads population p_* and event shares each window, updates 4D world state,
    then modulates event sampling weights and reward/risk multipliers.

    When ``lambda_world=0`` the world stays at (0.5, 0.5, 0.5, 0.5) — pure
    control / BL2 degeneration.
    """

    def __init__(
        self, *,
        lambda_world: float = 0.08,
        update_interval: int = 200,
    ) -> None:
        self.lambda_world = float(lambda_world)
        self.update_interval = int(update_interval)
        self.state: dict[str, float] = {d: 0.5 for d in _WORLD_DIMS}
        self._window_p: list[tuple[float, float, float]] = []
        self._window_rewards: list[float] = []
        self._window_event_counts: dict[str, int] = {
            f: 0 for f in _EVENT_FAMILIES
        }
        self.event_weights: dict[str, float] = {
            f: 1.0 for f in _EVENT_FAMILIES
        }
        self.reward_mult: dict[str, float] = {
            f: 1.0 for f in _EVENT_FAMILIES
        }
        self.risk_mult: dict[str, float] = {
            f: 1.0 for f in _EVENT_FAMILIES
        }

    def record_round(
        self, *,
        p_agg: float, p_def: float, p_bal: float,
        avg_reward: float,
        event_type: str,
    ) -> None:
        self._window_p.append((p_agg, p_def, p_bal))
        self._window_rewards.append(avg_reward)
        if event_type in self._window_event_counts:
            self._window_event_counts[event_type] += 1

    def maybe_update(self, round_idx: int) -> bool:
        """Flush window and update world state if interval reached.

        Returns True if an update was performed.
        """
        if (round_idx + 1) % self.update_interval != 0:
            return False
        if not self._window_p:
            return False
        self._update_state()
        self.event_weights = _world_to_event_weights(self.state)
        self.reward_mult = _world_to_reward_mult(self.state)
        self.risk_mult = _world_to_risk_mult(self.state)
        self._window_p = []
        self._window_rewards = []
        self._window_event_counts = {f: 0 for f in _EVENT_FAMILIES}
        return True

    def _update_state(self) -> None:
        if self.lambda_world == 0.0:
            return
        n = len(self._window_p)
        if n == 0:
            return
        mean_a = sum(t[0] for t in self._window_p) / n
        mean_d = sum(t[1] for t in self._window_p) / n
        mean_b = sum(t[2] for t in self._window_p) / n
        p_delta = (mean_a - 1/3, mean_d - 1/3, mean_b - 1/3)

        total_events = sum(self._window_event_counts.values())
        if total_events > 0:
            shares = [self._window_event_counts[f] / total_events for f in _EVENT_FAMILIES]
        else:
            shares = [0.0] * len(_EVENT_FAMILIES)

        reward_gap = (
            sum(self._window_rewards) / len(self._window_rewards)
            - _REWARD_TARGET
        ) if self._window_rewards else 0.0

        for dim in _WORLD_DIMS:
            bp = _B_P[dim]
            delta_p = bp[0]*p_delta[0] + bp[1]*p_delta[1] + bp[2]*p_delta[2]
            be = _B_E[dim]
            delta_e = reward_gap * sum(be[j]*shares[j] for j in range(len(shares)))
            self.state[dim] = _clamp_world(
                self.state[dim] + self.lambda_world * (delta_p + delta_e),
            )

    def sample_event_weighted(
        self,
        events: list[dict[str, Any]],
        *,
        rng: random.Random,
    ) -> dict[str, Any]:
        """Sample an event template using world-modulated type weights."""
        if not events:
            return {}
        weights: list[float] = []
        for ev in events:
            etype = ev.get("type", ev.get("event_type", ""))
            w = self.event_weights.get(etype, 1.0)
            weights.append(max(w, 0.01))
        total = sum(weights)
        r = rng.random() * total
        cum = 0.0
        for i, w in enumerate(weights):
            cum += w
            if r <= cum:
                return events[i]
        return events[-1]

    def apply_multipliers(
        self,
        event_type: str,
        reward_mod: float,
        risk_mod: float,
    ) -> tuple[float, float]:
        """Apply world-derived reward/risk multipliers for an event type."""
        rm = self.reward_mult.get(event_type, 1.0)
        rk = self.risk_mult.get(event_type, 1.0)
        return reward_mod * rm, risk_mod * rk


# ===================================================================
# Topology helper
# ===================================================================

def _make_graph_spec(topology: str) -> GraphSpec | None:
    if topology == "well_mixed":
        return None
    if topology == "lattice4":
        return GraphSpec(
            topology="lattice4", degree=4,
            lattice_rows=15, lattice_cols=20,
        )
    raise ValueError(f"Unknown topology: {topology!r}")


# ===================================================================
# Output contracts (bridge spec §6)
# ===================================================================

ROUND_CSV_FIELDS: list[str] = [
    "round",
    "avg_reward", "avg_utility", "success_rate",
    "risk_mean", "stress_mean",
    "event_affected_count", "event_affected_ratio",
    "event_sync_index",
    "p_aggressive", "p_defensive", "p_balanced",
    "pi_aggressive", "pi_defensive", "pi_balanced",
    "q_mean_aggressive", "q_mean_defensive", "q_mean_balanced",
    "world_scarcity", "world_threat", "world_noise", "world_intel",
    "dominant_event_type",
]


def _dispatch_seed_stream(seed: int, offset: int) -> int:
    raw = f"{int(seed)}:{int(offset)}:dispatch".encode("utf-8")
    # Use fixed hash function for stable cross-process reproducibility.
    return int(hashlib.sha256(raw).hexdigest()[:16], 16)


def _select_async_round_robin(
    *,
    n_players: int,
    rr_order: list[int],
    rr_cursor: int,
    target_rate: float,
    batch_size_cfg: int,
) -> tuple[list[int], int]:
    if n_players <= 0:
        return [], rr_cursor
    if batch_size_cfg > 0:
        batch = min(int(batch_size_cfg), n_players)
    else:
        batch = int(round(target_rate * n_players))
        if target_rate > 0.0:
            batch = max(1, batch)
        batch = min(batch, n_players)
    if batch <= 0:
        return [], rr_cursor

    affected: list[int] = []
    c = rr_cursor
    for _ in range(batch):
        affected.append(rr_order[c])
        c = (c + 1) % n_players
    return affected, c


def _select_async_poisson(
    *,
    n_players: int,
    target_rate: float,
    rng_dispatch: random.Random,
) -> list[int]:
    if target_rate <= 0.0:
        return []
    p = max(0.0, min(1.0, float(target_rate)))
    return [i for i in range(n_players) if rng_dispatch.random() < p]


def _evaluate_fairness_window(
    *,
    counts: list[int],
    tolerance: float,
) -> tuple[bool, int, float]:
    n = len(counts)
    total = sum(counts)
    if n <= 0 or total <= 0:
        return True, total, 0.0
    expected = total / n
    tol = max(0.0, float(tolerance))
    lo = expected * (1.0 - tol)
    hi = expected * (1.0 + tol)
    ok = all(lo <= c <= hi for c in counts)
    return ok, total, expected


def _impact_kernel_weights(*, horizon: int, decay: float) -> list[float]:
    h = max(1, int(horizon))
    d = max(0.0, float(decay))
    if h == 1:
        return [1.0]
    if abs(d - 1.0) <= 1e-12:
        v = 1.0 / h
        return [v] * h
    raw = [d ** k for k in range(h)]
    total = sum(raw)
    if total <= 0.0:
        return [1.0] + [0.0] * (h - 1)
    return [v / total for v in raw]


def _compute_reward_perturb_corr(
    *,
    sum_x: list[float],
    sum_x2: list[float],
    sum_xy: list[float],
    sum_y: float,
    sum_y2: float,
    n_rounds: int,
) -> float:
    """Compute mean player-vs-round-mean perturbation correlation.

    This is a low-memory proxy for cross-player perturbation synchrony.
    1.0 means players track a shared perturbation almost perfectly;
    values near 0 indicate weak common-mode perturbation.
    """
    if n_rounds < 2:
        return 0.0
    den_y = n_rounds * sum_y2 - sum_y * sum_y
    if den_y <= 0.0:
        return 0.0
    corrs: list[float] = []
    for i in range(len(sum_x)):
        den_x = n_rounds * sum_x2[i] - sum_x[i] * sum_x[i]
        if den_x <= 0.0:
            continue
        num = n_rounds * sum_xy[i] - sum_x[i] * sum_y
        c = num / sqrt(den_x * den_y)
        corrs.append(max(-1.0, min(1.0, c)))
    if not corrs:
        return 0.0
    return float(sum(corrs) / len(corrs))


def _estimate_trap_entry_round(
    rows: list[dict[str, Any]],
    *,
    scan_window: int = 1200,
    scan_step: int = 200,
) -> int | None:
    """Estimate first entry into the known s3≈0.555 flat-equilibrium trap.

    Diagnostic rule:
    - compute cycle metrics on rolling windows,
    - trap if stage3 score in [0.54, 0.57] and turn strength <= 0.01.
    """
    n = len(rows)
    if n < max(600, scan_window // 2):
        return None

    # Local import avoids coupling runtime startup to analysis layer.
    from analysis.cycle_metrics import classify_cycle_level

    series_all = {
        "aggressive": [float(r["p_aggressive"]) for r in rows],
        "defensive": [float(r["p_defensive"]) for r in rows],
        "balanced": [float(r["p_balanced"]) for r in rows],
    }

    for end in range(scan_window, n + 1, scan_step):
        start = max(0, end - scan_window)
        local = {
            k: v[start:end]
            for k, v in series_all.items()
        }
        m = len(local["aggressive"])
        if m < 400:
            continue
        local_burn = max(100, int(0.25 * m))
        local_tail = max(200, m - local_burn)
        cyc = classify_cycle_level(
            local,
            burn_in=local_burn,
            tail=local_tail,
            amplitude_threshold=0.02,
            corr_threshold=0.09,
            eta=0.55,
            stage3_method="turning",
            phase_smoothing=1,
            stage2_fallback_r2_threshold=0.85,
            stage2_fallback_min_rotation=20.0,
        )
        if cyc.stage3 is None:
            continue
        s3 = float(cyc.stage3.score)
        turn = abs(float(cyc.stage3.turn_strength))
        if 0.54 <= s3 <= 0.57 and turn <= 0.01:
            return int(end - 1)
    return None


def _write_round_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f, fieldnames=ROUND_CSV_FIELDS, extrasaction="ignore",
        )
        w.writeheader()
        w.writerows(rows)


def _write_provenance(path: Path, prov: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(prov, f, indent=2, ensure_ascii=False)


def _write_player_snapshot(path: Path, players: list[RLPlayer]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    snap_fields = [
        "player_id", "personality_json",
        "alpha", "beta",
        "r_a", "r_d", "r_b",
        "q_a", "q_d", "q_b",
        "utility_final", "risk_final", "dominant_strategy",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f, fieldnames=snap_fields, delimiter="\t", extrasaction="ignore",
        )
        w.writeheader()
        for p in players:
            dom_idx = p.q_values.index(max(p.q_values))
            w.writerow({
                "player_id": p.player_id,
                "personality_json": json.dumps(
                    p.personality, ensure_ascii=False,
                ),
                "alpha": f"{p.alpha:.6f}",
                "beta": f"{p.beta:.6f}",
                "r_a": f"{p.strategy_alpha_multipliers[0]:.4f}",
                "r_d": f"{p.strategy_alpha_multipliers[1]:.4f}",
                "r_b": f"{p.strategy_alpha_multipliers[2]:.4f}",
                "q_a": f"{p.q_values[0]:.6f}",
                "q_d": f"{p.q_values[1]:.6f}",
                "q_b": f"{p.q_values[2]:.6f}",
                "utility_final": f"{p.cumulative_utility:.6f}",
                "risk_final": f"{p.cumulative_risk:.6f}",
                "dominant_strategy": STRATEGY_SPACE[dom_idx],
            })


# ===================================================================
# Main simulation loop
# ===================================================================

@dataclass
class RunResult:
    rows: list[dict[str, Any]]
    players: list[RLPlayer]
    adj: list[list[int]] | None
    seed: int
    diagnostics: dict[str, Any] = field(default_factory=dict)


def run_personality_rl(cfg: PersonalityRLConfig, *, seed: int) -> RunResult:
    """Run a single personality-RL simulation seed.

    Core loop reuses ``evolution/independent_rl.py`` pure functions
    to stay BL2-compatible.
    """
    rng = random.Random(seed)
    n = cfg.n_players

    # 1. Payoff matrix (BL2 locked)
    pmat = strategy_payoff_matrix(a=cfg.a, b=cfg.b, cross=cfg.cross)

    # 2. Sample per-player alphas (and dummy betas) — RNG order must
    #    match E1's _sample_player_params to preserve BL2 seed compatibility.
    player_alphas: list[float] = []
    for _ in range(n):
        ai = rng.uniform(cfg.alpha_lo, cfg.alpha_hi)
        ai = max(0.0, min(1.0, ai))
        player_alphas.append(ai)
        # E1 also draws beta_i here even when beta_lo==beta_hi;
        # consume the RNG to keep streams aligned.
        _bi = rng.uniform(cfg.beta, cfg.beta)  # noqa: F841

    # 3. Build graph topology
    graph_spec = _make_graph_spec(cfg.topology)
    adj: list[list[int]] | None = None
    if graph_spec is not None:
        adj = build_graph(n, graph_spec, seed=seed)

    # 4. Per-player payoff perturbation — static ε (BL2: 0.02)
    payoff_deltas = sample_payoff_perturbation(
        n, epsilon=cfg.payoff_epsilon, rng=rng,
    )

    # 5. Initialise players
    players: list[RLPlayer] = []
    for i in range(n):
        if cfg.personality_mode == "random":
            personality = sample_personality(rng=rng)
        else:
            personality = {}
        p = init_rl_player(
            i, personality,
            alpha_base=player_alphas[i],
            beta_base=cfg.beta,
            strategy_alpha_multipliers=list(cfg.strategy_alpha_multipliers),
            payoff_bias=list(payoff_deltas[i]),
            lambda_alpha=cfg.lambda_alpha,
            lambda_beta=cfg.lambda_beta,
            lambda_r=cfg.lambda_r,
            lambda_risk=cfg.lambda_risk,
            lambda_beta_comp=cfg.lambda_beta_comp,
            init_q=cfg.init_q,
        )
        players.append(p)

    # 6. Event bridge (optional)
    event_bridge: EventBridge | None = None
    if cfg.events_json:
        event_bridge = EventBridge(cfg.events_json)

    # 6b. World feedback (Little Dragon)
    world: WorldFeedback | None = None
    if cfg.world_feedback:
        world = WorldFeedback(
            lambda_world=cfg.lambda_world,
            update_interval=cfg.world_update_interval,
        )

    dispatch_mode = str(cfg.event_dispatch_mode).strip().lower() or "sync"
    if dispatch_mode not in {"sync", "async_round_robin", "async_poisson"}:
        raise ValueError(f"Unsupported event_dispatch_mode: {cfg.event_dispatch_mode!r}")
    # Backward compatibility with the previous K4 flag.
    if cfg.event_per_player and dispatch_mode == "sync":
        dispatch_mode = "async_poisson"
    dispatch_target_rate = float(cfg.event_dispatch_target_rate)
    if dispatch_target_rate <= 0.0:
        dispatch_target_rate = float(cfg.event_rate)
    dispatch_target_rate = max(0.0, min(1.0, dispatch_target_rate))

    reward_mode = str(cfg.event_reward_mode).strip().lower() or "additive"
    if reward_mode not in {"additive", "multiplicative"}:
        raise ValueError(f"Unsupported event_reward_mode: {cfg.event_reward_mode!r}")
    # Keep multiplicative reward bounded and non-negative.
    multiplier_cap = float(cfg.event_reward_multiplier_cap)
    multiplier_cap = max(1e-6, min(0.999, multiplier_cap))

    impact_mode = str(cfg.event_impact_mode).strip().lower() or "instant"
    if impact_mode not in {"instant", "spread"}:
        raise ValueError(f"Unsupported event_impact_mode: {cfg.event_impact_mode!r}")
    impact_horizon = max(1, int(cfg.event_impact_horizon))
    impact_decay = max(0.0, float(cfg.event_impact_decay))
    impact_kernel = _impact_kernel_weights(
        horizon=impact_horizon,
        decay=impact_decay,
    )

    impact_reward_queue: list[list[float]] = [
        [0.0] * n for _ in range(impact_horizon)
    ]
    impact_risk_queue: list[list[float]] = [
        [0.0] * n for _ in range(impact_horizon)
    ]

    dispatch_seed_stream = _dispatch_seed_stream(seed, cfg.event_dispatch_seed_offset)
    rng_dispatch = random.Random(dispatch_seed_stream)
    rr_order = list(range(n))
    rng_dispatch.shuffle(rr_order)
    rr_cursor = 0

    # 7. Main loop
    rows: list[dict[str, Any]] = []
    k = cfg.control_degree

    # Low-memory accumulators for diagnostics.
    # reward_perturb_corr uses per-player perturbation vs round-mean perturbation.
    sum_x = [0.0] * n
    sum_x2 = [0.0] * n
    sum_xy = [0.0] * n
    sum_y = 0.0
    sum_y2 = 0.0
    corr_rounds = 0
    event_sync_sum = 0.0

    # B1 dispatch fairness diagnostics
    activation_counts = [0] * n
    fairness_window_counts = [0] * n
    fairness_window_rounds = 0
    fairness_window_checks = 0
    fairness_window_failures = 0
    dispatch_affected_ratio_sum = 0.0
    dispatch_rounds = 0

    for t in range(cfg.n_rounds):
        # 7a. Effective adjacency (well_mixed: resample each round)
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

        # 7b. Boltzmann selection (per-player β)
        chosen: list[int] = [
            boltzmann_select(players[i].q_values, beta=players[i].beta, rng=rng)
            for i in range(n)
        ]

        # 7c. Local one-hot payoff
        rewards: list[float] = [
            one_hot_local_payoff(
                chosen[i], [chosen[j] for j in eff[i]], pmat,
            )
            for i in range(n)
        ]

        # 7d. Per-player payoff perturbation (static)
        if cfg.payoff_epsilon > 0:
            rewards = apply_per_player_payoff_perturbation(
                rewards, chosen, payoff_deltas,
            )

        # 7e. Event overlay (smoke subset) + world feedback
        event_type = ""
        raw_event_reward_mods = [0.0] * n
        raw_event_risk_mods = [0.0] * n
        affected_count = 0
        affected_players: list[int] = []
        has_dispatch = (
            dispatch_target_rate > 0.0
            or cfg.event_dispatch_batch_size > 0
            or cfg.event_rate > 0.0
        )
        _ev_active = (
            event_bridge is not None
            and has_dispatch
            and t >= cfg.event_warmup_rounds
        )
        if _ev_active:
            if dispatch_mode == "sync":
                if rng.random() < cfg.event_rate:
                    affected_players = list(range(n))
            elif dispatch_mode == "async_round_robin":
                affected_players, rr_cursor = _select_async_round_robin(
                    n_players=n,
                    rr_order=rr_order,
                    rr_cursor=rr_cursor,
                    target_rate=dispatch_target_rate,
                    batch_size_cfg=cfg.event_dispatch_batch_size,
                )
            else:  # async_poisson
                affected_players = _select_async_poisson(
                    n_players=n,
                    target_rate=dispatch_target_rate,
                    rng_dispatch=rng_dispatch,
                )

            affected_count = len(affected_players)
            if affected_count > 0:
                # Sample one event for this round (shared event context).
                if world is not None:
                    event = world.sample_event_weighted(
                        event_bridge.events, rng=rng,
                    )
                else:
                    event = event_bridge.sample_event(rng=rng)
                event_type = event.get("type", event.get("event_type", ""))

                for i in affected_players:
                    rm, rk = event_bridge.compute_reward_risk(
                        event, players[i].personality,
                        scale=cfg.event_reward_scale, rng=rng,
                    )
                    if world is not None:
                        rm, rk = world.apply_multipliers(event_type, rm, rk)
                    if cfg.event_reward_clamp > 0:
                        rm = max(-cfg.event_reward_clamp, min(cfg.event_reward_clamp, rm))
                    if not cfg.event_risk_enabled:
                        rk = 0.0
                    raw_event_reward_mods[i] = rm
                    raw_event_risk_mods[i] = rk

        if impact_mode == "spread":
            for i in range(n):
                rm = raw_event_reward_mods[i]
                rk = raw_event_risk_mods[i]
                if rm == 0.0 and rk == 0.0:
                    continue
                for kidx, w in enumerate(impact_kernel):
                    if w == 0.0:
                        continue
                    impact_reward_queue[kidx][i] += rm * w
                    impact_risk_queue[kidx][i] += rk * w

            event_reward_mods = impact_reward_queue.pop(0)
            impact_reward_queue.append([0.0] * n)
            event_risk_mods = impact_risk_queue.pop(0)
            impact_risk_queue.append([0.0] * n)
        else:
            event_reward_mods = raw_event_reward_mods
            event_risk_mods = raw_event_risk_mods

        if reward_mode == "additive":
            rewards = [rewards[i] + event_reward_mods[i] for i in range(n)]
        else:
            rewards = [
                rewards[i] * max(1e-6, min(1.0 + multiplier_cap, max(1.0 - multiplier_cap, 1.0 + event_reward_mods[i])))
                for i in range(n)
            ]

        affected_ratio = (affected_count / n) if n > 0 else 0.0
        dispatch_affected_ratio_sum += affected_ratio
        dispatch_rounds += 1

        for i in affected_players:
            activation_counts[i] += 1
            fairness_window_counts[i] += 1
        fairness_window_rounds += 1
        if fairness_window_rounds >= max(1, int(cfg.event_dispatch_fairness_window)):
            ok, window_total, _expected = _evaluate_fairness_window(
                counts=fairness_window_counts,
                tolerance=cfg.event_dispatch_fairness_tolerance,
            )
            if window_total > 0:
                fairness_window_checks += 1
                if not ok:
                    fairness_window_failures += 1
            fairness_window_counts = [0] * n
            fairness_window_rounds = 0

        # Event synchronization index = probability two random players
        # are co-affected in this round.
        if n > 1:
            event_sync_index = (
                affected_count * (affected_count - 1)
            ) / (n * (n - 1))
        else:
            event_sync_index = 0.0
        event_sync_sum += event_sync_index

        # Update reward perturbation correlation accumulators.
        mean_perturb = sum(event_reward_mods) / n
        sum_y += mean_perturb
        sum_y2 += mean_perturb * mean_perturb
        for i, val in enumerate(event_reward_mods):
            x = float(val)
            sum_x[i] += x
            sum_x2[i] += x * x
            sum_xy[i] += x * mean_perturb
        corr_rounds += 1

        # 7f. Q-update (BL2-compatible, per-strategy α multipliers,
        #     risk-sensitive reward bias per §4.2 outlet 4)
        # risk_sign: aggressive=-1 (risky), defensive=+1 (safe), balanced=0
        _RISK_SIGN = (-1.0, 1.0, 0.0)
        for i in range(n):
            p = players[i]
            si = chosen[i]
            eff_alpha = min(1.0, p.alpha * p.strategy_alpha_multipliers[si])
            eff_reward = rewards[i] + p.risk_sensitivity * _RISK_SIGN[si]
            p.q_values = rl_q_update(
                p.q_values, si, eff_reward, alpha=eff_alpha,
            )
            p.cumulative_utility += rewards[i]
            risk_factor = 1.0 + p.risk_sensitivity
            p.cumulative_risk += event_risk_mods[i] * risk_factor
            p.stress = max(0.0, p.stress * 0.99 + event_risk_mods[i] * risk_factor)

        # 7g. Round-level statistics
        counts = [0] * _NSTRATS
        for c in chosen:
            counts[c] += 1

        pi_sum = [0.0] * _NSTRATS
        q_sum = [0.0] * _NSTRATS
        for i in range(n):
            ws = boltzmann_weights(players[i].q_values, beta=players[i].beta)
            for s in range(_NSTRATS):
                pi_sum[s] += ws[s]
                q_sum[s] += players[i].q_values[s]

        avg_rew = sum(rewards) / n
        avg_util = sum(p.cumulative_utility for p in players) / n
        risk_mean = sum(event_risk_mods) / n
        stress_mean = sum(p.stress for p in players) / n

        row: dict[str, Any] = {
            "round": t,
            "avg_reward": avg_rew,
            "avg_utility": avg_util,
            "success_rate": 1.0,        # placeholder until Step 5
            "risk_mean": risk_mean,
            "stress_mean": stress_mean,
            "event_affected_count": affected_count,
            "event_affected_ratio": affected_ratio,
            "event_sync_index": event_sync_index,
        }
        for si in range(_NSTRATS):
            s = STRATEGY_SPACE[si]
            row[f"p_{s}"] = counts[si] / n
            row[f"pi_{s}"] = pi_sum[si] / n
            row[f"q_mean_{s}"] = q_sum[si] / n

        # World state channels
        if world is not None:
            row["world_scarcity"] = world.state["scarcity"]
            row["world_threat"] = world.state["threat"]
            row["world_noise"] = world.state["noise"]
            row["world_intel"] = world.state["intel"]
            # Feed round data into world accumulator
            world.record_round(
                p_agg=counts[0] / n,
                p_def=counts[1] / n,
                p_bal=counts[2] / n,
                avg_reward=avg_rew,
                event_type=event_type,
            )
            world.maybe_update(t)
        else:
            row["world_scarcity"] = 0.0
            row["world_threat"] = 0.0
            row["world_noise"] = 0.0
            row["world_intel"] = 0.0
        row["dominant_event_type"] = event_type

        rows.append(row)

    if fairness_window_rounds > 0:
        ok, window_total, _expected = _evaluate_fairness_window(
            counts=fairness_window_counts,
            tolerance=cfg.event_dispatch_fairness_tolerance,
        )
        if window_total > 0:
            fairness_window_checks += 1
            if not ok:
                fairness_window_failures += 1

    mean_act = (sum(activation_counts) / n) if n > 0 else 0.0
    if mean_act > 0.0 and n > 0:
        var_act = sum((c - mean_act) ** 2 for c in activation_counts) / n
        cv_act = (var_act ** 0.5) / mean_act
    else:
        cv_act = 0.0

    diagnostics = {
        "event_sync_index_mean": (
            event_sync_sum / len(rows)
        ) if rows else 0.0,
        "reward_perturb_corr": _compute_reward_perturb_corr(
            sum_x=sum_x,
            sum_x2=sum_x2,
            sum_xy=sum_xy,
            sum_y=sum_y,
            sum_y2=sum_y2,
            n_rounds=corr_rounds,
        ),
        "trap_entry_round": _estimate_trap_entry_round(rows),
        "dispatch_seed_stream": dispatch_seed_stream,
        "dispatch_mode": dispatch_mode,
        "dispatch_target_rate": dispatch_target_rate,
        "event_reward_mode": reward_mode,
        "event_reward_multiplier_cap": multiplier_cap,
        "event_impact_mode": impact_mode,
        "event_impact_horizon": impact_horizon,
        "event_impact_decay": impact_decay,
        "dispatch_mean_affected_ratio": (
            dispatch_affected_ratio_sum / dispatch_rounds
        ) if dispatch_rounds else 0.0,
        "dispatch_player_activation_min": min(activation_counts) if activation_counts else 0,
        "dispatch_player_activation_max": max(activation_counts) if activation_counts else 0,
        "dispatch_player_activation_cv": cv_act,
        "dispatch_fairness_window": int(cfg.event_dispatch_fairness_window),
        "dispatch_fairness_tolerance": float(cfg.event_dispatch_fairness_tolerance),
        "dispatch_fairness_checks": fairness_window_checks,
        "dispatch_fairness_failures": fairness_window_failures,
        "dispatch_fairness_pass": fairness_window_failures == 0,
    }

    return RunResult(
        rows=rows,
        players=players,
        adj=adj,
        seed=seed,
        diagnostics=diagnostics,
    )


# ===================================================================
# Provenance (bridge spec §6.2)
# ===================================================================

def _build_provenance(
    cfg: PersonalityRLConfig,
    seed: int,
    players: list[RLPlayer],
    diagnostics: dict[str, Any] | None = None,
) -> dict[str, Any]:
    alphas = [p.alpha for p in players]
    betas = [p.beta for p in players]
    n = len(alphas)
    mean_a = sum(alphas) / n if n else 0.0
    mean_b = sum(betas) / n if n else 0.0
    std_a = (sum((a - mean_a) ** 2 for a in alphas) / n) ** 0.5 if n else 0.0
    std_b = (sum((b - mean_b) ** 2 for b in betas) / n) ** 0.5 if n else 0.0
    prov: dict[str, Any] = {
        "alpha_lo": cfg.alpha_lo,
        "alpha_hi": cfg.alpha_hi,
        "beta_ceiling": cfg.beta,
        "strategy_alpha_multipliers": cfg.strategy_alpha_multipliers,
        "payoff_epsilon": cfg.payoff_epsilon,
        "personality_mode": cfg.personality_mode,
        "lambda_alpha": cfg.lambda_alpha,
        "lambda_beta": cfg.lambda_beta,
        "lambda_r": cfg.lambda_r,
        "lambda_risk": cfg.lambda_risk,
        "events_json": cfg.events_json,
        "world_mode": "adaptive_world" if cfg.world_feedback else cfg.topology,
        "lambda_world": cfg.lambda_world if cfg.world_feedback else 0.0,
        "world_update_interval": cfg.world_update_interval if cfg.world_feedback else 0,
        "seed": seed,
        "mean_alpha": mean_a,
        "std_alpha": std_a,
        "mean_beta": mean_b,
        "std_beta": std_b,
        "risk_rule_version": "v1",
    }
    diag = diagnostics or {}
    prov["event_sync_index_mean"] = float(diag.get("event_sync_index_mean", 0.0))
    prov["reward_perturb_corr"] = float(diag.get("reward_perturb_corr", 0.0))
    prov["trap_entry_round"] = diag.get("trap_entry_round")
    prov["dispatch_seed_stream"] = int(diag.get("dispatch_seed_stream", 0))
    prov["dispatch_mode"] = str(diag.get("dispatch_mode", "sync"))
    prov["dispatch_target_rate"] = float(diag.get("dispatch_target_rate", 0.0))
    prov["event_reward_mode"] = str(diag.get("event_reward_mode", cfg.event_reward_mode))
    prov["event_reward_multiplier_cap"] = float(
        diag.get("event_reward_multiplier_cap", cfg.event_reward_multiplier_cap)
    )
    prov["event_impact_mode"] = str(diag.get("event_impact_mode", cfg.event_impact_mode))
    prov["event_impact_horizon"] = int(diag.get("event_impact_horizon", cfg.event_impact_horizon))
    prov["event_impact_decay"] = float(diag.get("event_impact_decay", cfg.event_impact_decay))
    prov["dispatch_mean_affected_ratio"] = float(diag.get("dispatch_mean_affected_ratio", 0.0))
    prov["dispatch_player_activation_min"] = int(diag.get("dispatch_player_activation_min", 0))
    prov["dispatch_player_activation_max"] = int(diag.get("dispatch_player_activation_max", 0))
    prov["dispatch_player_activation_cv"] = float(diag.get("dispatch_player_activation_cv", 0.0))
    prov["dispatch_fairness_window"] = int(diag.get("dispatch_fairness_window", 0))
    prov["dispatch_fairness_tolerance"] = float(diag.get("dispatch_fairness_tolerance", 0.0))
    prov["dispatch_fairness_checks"] = int(diag.get("dispatch_fairness_checks", 0))
    prov["dispatch_fairness_failures"] = int(diag.get("dispatch_fairness_failures", 0))
    prov["dispatch_fairness_pass"] = bool(diag.get("dispatch_fairness_pass", True))
    prov["diagnostic_rule_version"] = "diag_v1"
    return prov


# ===================================================================
# Single-seed runner + CLI
# ===================================================================

def _run_seed(cfg: PersonalityRLConfig, seed: int) -> None:
    out_dir = Path(cfg.out_dir) / f"seed_{seed}"
    result = run_personality_rl(cfg, seed=seed)
    _write_round_csv(out_dir / "timeseries.csv", result.rows)
    _write_provenance(
        out_dir / "provenance.json",
        _build_provenance(
            cfg,
            seed,
            result.players,
            diagnostics=result.diagnostics,
        ),
    )
    _write_player_snapshot(out_dir / "player_snapshot.tsv", result.players)
    print(f"seed={seed} done → {out_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Personality RL Runtime (bridge spec)",
    )
    parser.add_argument("--seeds", default="42,43,44")
    parser.add_argument("--n-players", type=int, default=300)
    parser.add_argument("--n-rounds", type=int, default=12000)
    parser.add_argument("--topology", default="well_mixed")
    parser.add_argument(
        "--personality-mode", default="none",
        choices=["none", "random"],
    )
    parser.add_argument("--lambda-alpha", type=float, default=0.0)
    parser.add_argument("--lambda-beta", type=float, default=0.0)
    parser.add_argument("--lambda-r", type=float, default=0.0)
    parser.add_argument("--lambda-risk", type=float, default=0.0)
    parser.add_argument("--events-json", default="")
    parser.add_argument("--event-rate", type=float, default=0.0)
    parser.add_argument("--event-reward-scale", type=float, default=0.01)
    parser.add_argument(
        "--event-reward-mode",
        default="additive",
        choices=["additive", "multiplicative"],
    )
    parser.add_argument("--event-reward-multiplier-cap", type=float, default=0.25)
    parser.add_argument(
        "--event-impact-mode",
        default="instant",
        choices=["instant", "spread"],
    )
    parser.add_argument("--event-impact-horizon", type=int, default=1)
    parser.add_argument("--event-impact-decay", type=float, default=0.70)
    parser.add_argument(
        "--event-dispatch-mode",
        default="sync",
        choices=["sync", "async_round_robin", "async_poisson"],
    )
    parser.add_argument("--event-dispatch-target-rate", type=float, default=0.0)
    parser.add_argument("--event-dispatch-batch-size", type=int, default=0)
    parser.add_argument("--event-dispatch-seed-offset", type=int, default=0)
    parser.add_argument("--event-dispatch-fairness-window", type=int, default=200)
    parser.add_argument("--event-dispatch-fairness-tolerance", type=float, default=0.15)
    parser.add_argument("--world-feedback", action="store_true", default=False)
    parser.add_argument("--lambda-world", type=float, default=0.08)
    parser.add_argument("--world-update-interval", type=int, default=200)
    parser.add_argument("--out-dir", default="outputs/personality_rl")
    args = parser.parse_args()

    cfg = PersonalityRLConfig(
        n_players=args.n_players,
        n_rounds=args.n_rounds,
        topology=args.topology,
        personality_mode=args.personality_mode,
        lambda_alpha=args.lambda_alpha,
        lambda_beta=args.lambda_beta,
        lambda_r=args.lambda_r,
        lambda_risk=args.lambda_risk,
        events_json=args.events_json,
        event_rate=args.event_rate,
        event_reward_scale=args.event_reward_scale,
        event_reward_mode=args.event_reward_mode,
        event_reward_multiplier_cap=args.event_reward_multiplier_cap,
        event_impact_mode=args.event_impact_mode,
        event_impact_horizon=args.event_impact_horizon,
        event_impact_decay=args.event_impact_decay,
        event_dispatch_mode=args.event_dispatch_mode,
        event_dispatch_target_rate=args.event_dispatch_target_rate,
        event_dispatch_batch_size=args.event_dispatch_batch_size,
        event_dispatch_seed_offset=args.event_dispatch_seed_offset,
        event_dispatch_fairness_window=args.event_dispatch_fairness_window,
        event_dispatch_fairness_tolerance=args.event_dispatch_fairness_tolerance,
        world_feedback=args.world_feedback,
        lambda_world=args.lambda_world,
        world_update_interval=args.world_update_interval,
        out_dir=args.out_dir,
    )

    for seed in (int(s) for s in args.seeds.split(",")):
        _run_seed(cfg, seed)


if __name__ == "__main__":
    main()
