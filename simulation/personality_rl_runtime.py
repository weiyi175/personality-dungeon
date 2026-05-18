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
from collections import deque
from dataclasses import dataclass, field
from math import exp, log, sqrt
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
    event_modulation_mode: str = "off"   # off | multiplicative_v2
    event_modulation_gain: float = 0.0
    event_modulation_log_center: float = 0.0
    event_modulation_zero_mean: bool = False
    event_modulation_floor: float = 1.0
    event_modulation_ceiling: float = 1.0
    event_neutralize_payoff: bool = False  # E6: zero-mean event payoff over affected players
    event_neutralize_eps: float = 1e-9     # E6 diagnostics tolerance
    event_impact_mode: str = "instant"  # instant | spread
    event_impact_horizon: int = 1        # spread horizon in rounds
    event_impact_decay: float = 0.70     # geometric decay kernel base
    impact_spread_kernel_id: str = "legacy_v1"  # legacy_v1 | hierarchical_v2
    impact_spread_local_mass: float = 1.0
    impact_spread_neighbor_mass: float = 0.0
    impact_spread_neighbor_hop: int = 0
    impact_spread_memory_kernel: int = 1
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
    event_trigger_mode: str = "always"  # always | entropy_guard
    event_trigger_entropy_threshold: float = 0.85  # normalized entropy in [0, 1]
    # ---- B1 v2 async operator / queue ----
    replicator_update_mode: str = "sync_global"  # sync_global | async_per_player
    replicator_async_minibatch: int = 0            # <=0 means auto
    replicator_async_jitter: float = 0.0           # [0, 1]
    event_queue_mode: str = "off"                # off | per_player
    event_queue_cap: int = 0                       # <=0 means unbounded
    event_queue_drain_rate: float = 1.0            # queue entries drained per round

    # ---- World feedback (Little Dragon) ----
    world_feedback: bool = False        # enable adaptive world state
    world_feedback_mode: str = "off"   # off | adaptive_world | read_only | difficulty_only
    lambda_world: float = 0.04          # world update gain (Hypothesis 1 recalibration: 50% down)
    world_update_interval: int = 200    # rounds between world updates
    world_feedback_delay_windows: int = 0  # H3: delayed world update signal in window units
    world_feedback_smooth_windows: int = 1  # H3: moving-average smoothing over window signals

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


def _world_difficulty_index(st: dict[str, float]) -> float:
    """Map world state to a bounded scalar difficulty index in [0, 1]."""
    raw = (
        0.30 * float(st["scarcity"])
        + 0.30 * float(st["threat"])
        + 0.25 * float(st["noise"])
        - 0.20 * float(st["intel"])
    )
    # raw range is approximately [-0.20, 0.85]; normalize around 0.5 at state=0.5.
    return _clamp_world((raw + 0.20) / 1.05)


def _event_difficulty_multiplier(
    event_weights: dict[str, float],
    event_type: str,
) -> float:
    if not event_type:
        return 1.0
    vals = [max(0.0, float(v)) for v in event_weights.values()]
    if not vals:
        return 1.0
    mean_w = sum(vals) / len(vals)
    if mean_w <= 0.0:
        return 1.0
    return max(0.0, float(event_weights.get(event_type, mean_w)) / mean_w)


class WorldFeedback:
    """Little Dragon world feedback for the RL runtime.

    Reads population p_* and event shares each window, updates 4D world state,
    then modulates event sampling weights and reward/risk multipliers.

    When ``lambda_world=0`` the world stays at (0.5, 0.5, 0.5, 0.5) — pure
    control / BL2 degeneration.
    """

    def __init__(
        self, *,
        lambda_world: float = 0.04,
        update_interval: int = 200,
        delay_windows: int = 0,
        smooth_windows: int = 1,
    ) -> None:
        self.lambda_world = float(lambda_world)
        self.update_interval = int(update_interval)
        self.delay_windows = max(0, int(delay_windows))
        self.smooth_windows = max(1, int(smooth_windows))
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
        # diagnostic/history for provenance: list of world-update windows
        self.world_update_rows: list[dict[str, Any]] = []
        self._window_index: int = 0
        self._update_signal_history: list[dict[str, float]] = []
        # monitoring thresholds (keep local to runtime)
        self.HARD_LOWER_BOUND = 0.0
        self.HARD_UPPER_BOUND = 1.0
        self.STABILITY_THRESHOLD = 0.20
        # attach world-update rows and aggregate boundary/instability stats
        self.world_update_rows: list[dict[str, Any]] = []
        self._window_index: int = 0
        self.HARD_LOWER_BOUND = 0.0
        self.HARD_UPPER_BOUND = 1.0
        self.STABILITY_THRESHOLD = 0.20
        
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
        # capture diagnostics about this window before and after updating
        n = len(self._window_p)
        start_round = (round_idx + 1) - n
        end_round = round_idx
        prev_state = None
        if self.world_update_rows:
            prev_state = {
                d: float(self.world_update_rows[-1].get(f"state_{d}", 0.5))
                for d in _WORLD_DIMS
            }
        # perform update
        self._update_state()
        # compute derived multipliers
        self.event_weights = _world_to_event_weights(self.state)
        self.reward_mult = _world_to_reward_mult(self.state)
        self.risk_mult = _world_to_risk_mult(self.state)
        # diagnostics: boundary hit checks and instability
        boundary_hit = False
        boundary_hit_vars: list[str] = []
        boundary_hit_type = ""
        for d in _WORLD_DIMS:
            v = float(self.state.get(d, 0.0))
            if v <= self.HARD_LOWER_BOUND:
                boundary_hit = True
                boundary_hit_vars.append(d)
                boundary_hit_type = "lower"
            if v >= self.HARD_UPPER_BOUND:
                boundary_hit = True
                boundary_hit_vars.append(d)
                # if mixed, mark as mixed
                if boundary_hit_type and boundary_hit_type != "upper":
                    boundary_hit_type = "mixed"
                else:
                    boundary_hit_type = "upper"
        
        max_state_delta = 0.0
        instability_warning = False
        if prev_state is not None:
            deltas = [abs(float(self.state.get(d, 0.0)) - float(prev_state.get(d, 0.0))) for d in _WORLD_DIMS]
            max_state_delta = max(deltas) if deltas else 0.0
            if max_state_delta > self.STABILITY_THRESHOLD:
                instability_warning = True
        
        row_entry: dict[str, Any] = {
            "window_index": int(self._window_index),
            "start_round": int(start_round),
            "end_round": int(end_round),
            "window_rows": int(n),
            "delay_windows": int(self.delay_windows),
            "smooth_windows": int(self.smooth_windows),
            "boundary_hit": bool(boundary_hit),
            "boundary_hit_vars": list(boundary_hit_vars),
            "boundary_hit_type": str(boundary_hit_type),
            "instability_warning": bool(instability_warning),
            "max_state_delta": float(max_state_delta),
        }
        for d in _WORLD_DIMS:
            row_entry[f"state_{d}"] = float(self.state.get(d, 0.0))
        self.world_update_rows.append(row_entry)
        self._window_index += 1
        
        # clear window accumulators
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

        raw_signal: dict[str, float] = {}
        for dim in _WORLD_DIMS:
            bp = _B_P[dim]
            delta_p = bp[0]*p_delta[0] + bp[1]*p_delta[1] + bp[2]*p_delta[2]
            be = _B_E[dim]
            delta_e = reward_gap * sum(be[j]*shares[j] for j in range(len(shares)))
            raw_signal[dim] = float(delta_p + delta_e)

        # H3 delayed feedback: apply historical window signal with optional smoothing.
        self._update_signal_history.append(raw_signal)
        apply_idx = max(0, len(self._update_signal_history) - 1 - self.delay_windows)
        smooth_start = max(0, apply_idx - self.smooth_windows + 1)
        smooth_slice = self._update_signal_history[smooth_start : apply_idx + 1]

        for dim in _WORLD_DIMS:
            avg_signal = (
                sum(sig.get(dim, 0.0) for sig in smooth_slice) / float(len(smooth_slice))
            ) if smooth_slice else 0.0
            self.state[dim] = _clamp_world(
                self.state[dim] + self.lambda_world * avg_signal,
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
    "world_state_scarcity", "world_state_threat", "world_state_noise", "world_state_intel",
    "world_readonly_applied", "readonly_leak_score",
    "difficulty_index", "event_difficulty_multiplier", "difficulty_modulation_applied", "payoff_static_pass",
    "reward_multiplier_raw", "reward_multiplier_clamped", "log_reward_multiplier",
    "modulation_gain_effective", "modulation_zero_mean_residual", "multiplicative_static_pass",
    "impact_kernel_id", "impact_kernel_mass_local", "impact_kernel_mass_neighbor",
    "impact_spread_radius", "impact_spread_delay_mean", "impact_spread_alignment",
    "impact_kernel_mass_error", "impact_spread_applied",
    "replicator_update_mode", "async_update_applied", "async_update_ratio",
    "player_event_queue_depth_mean", "player_event_queue_depth_p95", "queue_drop_count",
    "update_skew_index", "phase_lag_index",
    "dominant_event_type",
]


def _resolve_world_feedback_mode(cfg: PersonalityRLConfig) -> str:
    aliases = {
        "": "off",
        "none": "off",
        "false": "off",
        "adaptive": "adaptive_world",
        "on": "adaptive_world",
        "true": "adaptive_world",
        "read-only": "read_only",
        "readonly": "read_only",
        "difficulty": "difficulty_only",
        "difficulty-only": "difficulty_only",
    }
    raw = str(getattr(cfg, "world_feedback_mode", "") or "").strip().lower()
    mode = aliases.get(raw, raw)
    if mode in {"", "off"}:
        return "adaptive_world" if bool(cfg.world_feedback) else "off"
    if mode not in {"off", "adaptive_world", "read_only", "difficulty_only"}:
        raise ValueError(f"Unsupported world_feedback_mode: {cfg.world_feedback_mode!r}")
    return mode


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


def _select_async_update_indices(
    *,
    n_players: int,
    minibatch: int,
    jitter: float,
    rng_dispatch: random.Random,
) -> list[int]:
    if n_players <= 0:
        return []
    base = int(minibatch)
    if base <= 0:
        base = max(1, int(round(0.10 * n_players)))
    base = max(1, min(base, n_players))

    j = max(0.0, min(1.0, float(jitter)))
    if j > 0.0:
        scale = 1.0 + rng_dispatch.uniform(-j, j)
        eff = int(round(base * scale))
    else:
        eff = base
    eff = max(1, min(eff, n_players))
    return rng_dispatch.sample(list(range(n_players)), eff)


def _drain_player_event_queue(
    *,
    queue: deque[tuple[float, float]],
    drain_rate: float,
) -> tuple[float, float]:
    budget = max(0.0, float(drain_rate))
    reward_out = 0.0
    risk_out = 0.0
    if budget <= 0.0 or not queue:
        return reward_out, risk_out

    while budget > 1e-12 and queue:
        reward_head, risk_head = queue[0]
        if budget >= 1.0 - 1e-12:
            reward_out += float(reward_head)
            risk_out += float(risk_head)
            queue.popleft()
            budget -= 1.0
            continue

        frac = budget
        reward_out += float(reward_head) * frac
        risk_out += float(risk_head) * frac
        queue[0] = (
            float(reward_head) * (1.0 - frac),
            float(risk_head) * (1.0 - frac),
        )
        budget = 0.0

    return reward_out, risk_out


def _p95(values: list[int]) -> float:
    if not values:
        return 0.0
    arr = sorted(int(v) for v in values)
    idx = int(round((len(arr) - 1) * 0.95))
    return float(arr[idx])


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


def _resolve_impact_spread_kernel_id(cfg: PersonalityRLConfig) -> str:
    aliases = {
        "": "legacy_v1",
        "legacy": "legacy_v1",
        "legacy-v1": "legacy_v1",
        "hierarchical": "hierarchical_v2",
        "hierarchical-v2": "hierarchical_v2",
    }
    raw = str(getattr(cfg, "impact_spread_kernel_id", "") or "").strip().lower()
    kernel_id = aliases.get(raw, raw)
    if kernel_id not in {"legacy_v1", "hierarchical_v2"}:
        raise ValueError(
            f"Unsupported impact_spread_kernel_id: {cfg.impact_spread_kernel_id!r}"
        )
    return kernel_id


def _hop_neighbors(eff: list[list[int]], start: int, hop: int) -> list[int]:
    if hop <= 0:
        return []
    visited = {int(start)}
    frontier = {int(start)}
    for _ in range(int(hop)):
        nxt: set[int] = set()
        for node in frontier:
            for nb in eff[node]:
                j = int(nb)
                if j in visited:
                    continue
                visited.add(j)
                nxt.add(j)
        if not nxt:
            break
        frontier = nxt
    visited.discard(int(start))
    return sorted(visited)


def _vector_alignment(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = 0.0
    na = 0.0
    nb = 0.0
    for x, y in zip(a, b):
        fx = float(x)
        fy = float(y)
        dot += fx * fy
        na += fx * fx
        nb += fy * fy
    if na <= 0.0 and nb <= 0.0:
        return 1.0
    if na <= 0.0 or nb <= 0.0:
        return 0.0
    return max(-1.0, min(1.0, dot / sqrt(na * nb)))


def _resolve_event_modulation_mode(cfg: PersonalityRLConfig) -> str:
    aliases = {
        "": "off",
        "none": "off",
        "disabled": "off",
        "false": "off",
        "v2": "multiplicative_v2",
        "multiplicative-v2": "multiplicative_v2",
    }
    raw = str(getattr(cfg, "event_modulation_mode", "") or "").strip().lower()
    mode = aliases.get(raw, raw)
    if mode in {"", "off"}:
        return "off"
    if mode != "multiplicative_v2":
        raise ValueError(
            f"Unsupported event_modulation_mode: {cfg.event_modulation_mode!r}"
        )
    return mode


def _apply_log_multiplicative_modulation(
    *,
    event_reward_mods: list[float],
    gain: float,
    log_center: float,
    zero_mean: bool,
    floor: float,
    ceiling: float,
    eps: float = 1e-6,
) -> tuple[list[float], dict[str, float | bool]]:
    n = len(event_reward_mods)
    if n <= 0:
        return [], {
            "reward_multiplier_raw": 1.0,
            "reward_multiplier_clamped": 1.0,
            "log_reward_multiplier": 0.0,
            "modulation_gain_effective": 0.0,
            "modulation_zero_mean_residual": 0.0,
            "multiplicative_static_pass": True,
        }

    lo = min(float(floor), float(ceiling))
    hi = max(float(floor), float(ceiling))
    lo = max(eps, lo)
    hi = max(lo, hi)

    active_idx = [i for i, d in enumerate(event_reward_mods) if abs(float(d)) > 0.0]
    if not active_idx:
        active_idx = list(range(n))

    log_base = [0.0] * n
    gain_delta = [0.0] * n
    for i in range(n):
        m0 = max(eps, 1.0 + float(event_reward_mods[i]))
        log_base[i] = log(m0)
        gain_delta[i] = float(gain) * (log_base[i] - float(log_center))

    if zero_mean and active_idx:
        mean_gain = sum(gain_delta[i] for i in active_idx) / float(len(active_idx))
        for i in active_idx:
            gain_delta[i] -= mean_gain

    residual = 0.0
    if active_idx:
        residual = abs(sum(gain_delta[i] for i in active_idx) / float(len(active_idx)))

    multipliers_raw = [0.0] * n
    multipliers_clamped = [0.0] * n
    log_modulated = [0.0] * n
    for i in range(n):
        log_i = log_base[i] + gain_delta[i]
        log_modulated[i] = log_i
        m_raw = exp(log_i)
        multipliers_raw[i] = m_raw
        multipliers_clamped[i] = max(lo, min(hi, m_raw))

    obs = active_idx if active_idx else list(range(n))
    mean_raw = sum(multipliers_raw[i] for i in obs) / float(len(obs))
    mean_clamped = sum(multipliers_clamped[i] for i in obs) / float(len(obs))
    mean_log = sum(log_modulated[i] for i in obs) / float(len(obs))
    mean_gain_eff = sum(abs(gain_delta[i]) for i in obs) / float(len(obs))

    return multipliers_clamped, {
        "reward_multiplier_raw": float(mean_raw),
        "reward_multiplier_clamped": float(mean_clamped),
        "log_reward_multiplier": float(mean_log),
        "modulation_gain_effective": float(mean_gain_eff),
        "modulation_zero_mean_residual": float(residual),
        "multiplicative_static_pass": (float(residual) <= float(eps)) if zero_mean else True,
    }


def _normalized_strategy_entropy(chosen: list[int], n_strats: int) -> float:
    if n_strats <= 1 or not chosen:
        return 0.0
    counts = [0] * n_strats
    for s in chosen:
        if 0 <= s < n_strats:
            counts[s] += 1
    n = len(chosen)
    if n <= 0:
        return 0.0
    ent = 0.0
    for c in counts:
        if c <= 0:
            continue
        p = c / n
        ent -= p * log(p)
    max_ent = log(n_strats)
    if max_ent <= 0.0:
        return 0.0
    return max(0.0, min(1.0, ent / max_ent))


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
    world_mode = _resolve_world_feedback_mode(cfg)
    world_enabled = world_mode != "off"
    world_apply_event_weights = world_mode in {"adaptive_world", "difficulty_only"}
    world_apply_multipliers = world_mode == "adaptive_world"
    world_readonly_applied = world_mode == "read_only"
    world_difficulty_only_applied = world_mode == "difficulty_only"

    world: WorldFeedback | None = None
    if world_enabled:
        world = WorldFeedback(
            lambda_world=cfg.lambda_world,
            update_interval=cfg.world_update_interval,
            delay_windows=cfg.world_feedback_delay_windows,
            smooth_windows=cfg.world_feedback_smooth_windows,
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

    replicator_update_mode = str(cfg.replicator_update_mode).strip().lower() or "sync_global"
    if replicator_update_mode in {"sync", "sync_all"}:
        replicator_update_mode = "sync_global"
    if replicator_update_mode not in {"sync_global", "async_per_player"}:
        raise ValueError(
            f"Unsupported replicator_update_mode: {cfg.replicator_update_mode!r}"
        )
    replicator_async_minibatch = int(cfg.replicator_async_minibatch)
    replicator_async_jitter = max(0.0, min(1.0, float(cfg.replicator_async_jitter)))
    async_update_applied = replicator_update_mode == "async_per_player"

    queue_mode = str(cfg.event_queue_mode).strip().lower() or "off"
    if queue_mode in {"none", "disabled"}:
        queue_mode = "off"
    if queue_mode not in {"off", "per_player"}:
        raise ValueError(f"Unsupported event_queue_mode: {cfg.event_queue_mode!r}")
    queue_cap = int(cfg.event_queue_cap)
    queue_cap = max(0, queue_cap)
    queue_drain_rate = max(0.0, float(cfg.event_queue_drain_rate))
    queue_enabled = queue_mode == "per_player"
    player_event_queues: list[deque[tuple[float, float]]] = [
        deque() for _ in range(n)
    ]

    reward_mode = str(cfg.event_reward_mode).strip().lower() or "additive"
    if reward_mode not in {"additive", "multiplicative"}:
        raise ValueError(f"Unsupported event_reward_mode: {cfg.event_reward_mode!r}")
    event_modulation_mode = _resolve_event_modulation_mode(cfg)
    event_modulation_gain = float(cfg.event_modulation_gain)
    event_modulation_log_center = float(cfg.event_modulation_log_center)
    event_modulation_zero_mean = bool(cfg.event_modulation_zero_mean)
    event_modulation_floor = float(cfg.event_modulation_floor)
    event_modulation_ceiling = float(cfg.event_modulation_ceiling)
    modulation_zero_mean_eps = 1e-6
    # Keep multiplicative reward bounded and non-negative.
    multiplier_cap = float(cfg.event_reward_multiplier_cap)
    multiplier_cap = max(1e-6, min(0.999, multiplier_cap))
    neutralize_payoff = bool(cfg.event_neutralize_payoff)
    neutralize_eps = max(0.0, float(cfg.event_neutralize_eps))

    trigger_mode = str(cfg.event_trigger_mode).strip().lower() or "always"
    if trigger_mode not in {"always", "entropy_guard"}:
        raise ValueError(f"Unsupported event_trigger_mode: {cfg.event_trigger_mode!r}")
    trigger_entropy_threshold = max(0.0, min(1.0, float(cfg.event_trigger_entropy_threshold)))

    impact_mode = str(cfg.event_impact_mode).strip().lower() or "instant"
    if impact_mode not in {"instant", "spread"}:
        raise ValueError(f"Unsupported event_impact_mode: {cfg.event_impact_mode!r}")
    impact_horizon = max(1, int(cfg.event_impact_horizon))
    impact_decay = max(0.0, float(cfg.event_impact_decay))
    impact_spread_kernel_id = _resolve_impact_spread_kernel_id(cfg)
    impact_local_mass = max(0.0, float(cfg.impact_spread_local_mass))
    impact_neighbor_mass = max(0.0, float(cfg.impact_spread_neighbor_mass))
    impact_spread_neighbor_hop = max(0, int(cfg.impact_spread_neighbor_hop))
    impact_spread_memory_kernel = max(1, int(cfg.impact_spread_memory_kernel))
    if impact_spread_kernel_id == "legacy_v1":
        impact_local_mass = 1.0
        impact_neighbor_mass = 0.0
        impact_spread_neighbor_hop = 0
        impact_spread_memory_kernel = 1
    else:
        mass_total = impact_local_mass + impact_neighbor_mass
        if mass_total <= 0.0:
            impact_local_mass = 1.0
            impact_neighbor_mass = 0.0
        else:
            impact_local_mass /= mass_total
            impact_neighbor_mass /= mass_total
    impact_effective_horizon = (
        max(impact_horizon, impact_spread_memory_kernel)
        if impact_spread_kernel_id == "hierarchical_v2"
        else impact_horizon
    )
    impact_kernel = _impact_kernel_weights(
        horizon=impact_effective_horizon,
        decay=impact_decay,
    )
    impact_kernel_delay_mean = float(
        sum(float(kidx) * float(w) for kidx, w in enumerate(impact_kernel))
    )

    impact_reward_queue: list[list[float]] = [
        [0.0] * n for _ in range(len(impact_kernel))
    ]
    impact_risk_queue: list[list[float]] = [
        [0.0] * n for _ in range(len(impact_kernel))
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
    event_reward_mean_raw_sum = 0.0
    event_reward_mean_post_sum = 0.0
    event_reward_abs_mean_post_sum = 0.0
    event_neutrality_rounds = 0
    event_neutrality_max_abs_mean = 0.0
    event_trigger_guard_check_count = 0
    event_trigger_guard_block_count = 0
    readonly_leak_eps = 1e-6
    readonly_leak_max = 0.0
    readonly_leak_sum = 0.0
    readonly_leak_rounds = 0
    payoff_static_eps = 1e-9
    payoff_static_max = 0.0
    payoff_static_sum = 0.0
    payoff_static_rounds = 0
    difficulty_index_sum = 0.0
    event_difficulty_multiplier_sum = 0.0
    difficulty_obs_rounds = 0
    reward_multiplier_raw_sum = 0.0
    reward_multiplier_clamped_sum = 0.0
    log_reward_multiplier_sum = 0.0
    modulation_gain_effective_sum = 0.0
    modulation_zero_mean_residual_max = 0.0
    modulation_obs_rounds = 0
    multiplicative_static_failures = 0
    impact_spread_alignment_sum = 0.0
    impact_spread_rounds = 0
    impact_kernel_mass_error_max = 0.0

    # B1 dispatch fairness diagnostics
    activation_counts = [0] * n
    fairness_window_counts = [0] * n
    fairness_window_rounds = 0
    fairness_window_checks = 0
    fairness_window_failures = 0
    dispatch_affected_ratio_sum = 0.0
    dispatch_rounds = 0
    dispatch_configured = (
        dispatch_target_rate > 0.0
        or cfg.event_dispatch_batch_size > 0
        or cfg.event_rate > 0.0
    )

    # B1 v2 async-update + queue diagnostics
    async_update_count_total = 0
    async_update_ratio_sum = 0.0
    async_update_ratio_min = 1.0
    async_update_ratio_max = 0.0
    async_update_rounds = 0
    player_update_counts = [0] * n
    update_skew_index_round = 0.0
    phase_lag_index_sum = 0.0
    phase_lag_rounds = 0

    queue_drop_count_total = 0
    queue_depth_mean_sum = 0.0
    queue_depth_p95_sum = 0.0
    queue_depth_rounds = 0
    queue_depth_mean_round = 0.0
    queue_depth_p95_round = 0.0

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
        readonly_leak_score_round = 0.0
        payoff_static_score_round = 0.0
        event_difficulty_multiplier_round = 1.0
        has_dispatch = dispatch_configured
        _ev_active = (
            event_bridge is not None
            and has_dispatch
            and t >= cfg.event_warmup_rounds
        )
        if _ev_active and trigger_mode == "entropy_guard":
            event_trigger_guard_check_count += 1
            entropy_now = _normalized_strategy_entropy(chosen, _NSTRATS)
            if entropy_now <= trigger_entropy_threshold:
                event_trigger_guard_block_count += 1
                _ev_active = False

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
                if world is not None and world_apply_event_weights:
                    event = world.sample_event_weighted(
                        event_bridge.events, rng=rng,
                    )
                else:
                    event = event_bridge.sample_event(rng=rng)
                event_type = event.get("type", event.get("event_type", ""))
                if world is not None and world_apply_event_weights:
                    event_difficulty_multiplier_round = _event_difficulty_multiplier(
                        world.event_weights,
                        str(event_type),
                    )

                for i in affected_players:
                    rm_base, rk_base = event_bridge.compute_reward_risk(
                        event, players[i].personality,
                        scale=cfg.event_reward_scale, rng=rng,
                    )
                    rm = rm_base
                    rk = rk_base
                    if world is not None and world_apply_multipliers:
                        rm, rk = world.apply_multipliers(event_type, rm, rk)
                    readonly_leak_score_round = max(
                        readonly_leak_score_round,
                        abs(rm - rm_base),
                        abs(rk - rk_base),
                    )
                    payoff_static_score_round = max(
                        payoff_static_score_round,
                        abs(rm - rm_base),
                        abs(rk - rk_base),
                    )
                    if cfg.event_reward_clamp > 0:
                        rm = max(-cfg.event_reward_clamp, min(cfg.event_reward_clamp, rm))
                    if not cfg.event_risk_enabled:
                        rk = 0.0
                    raw_event_reward_mods[i] = rm
                    raw_event_risk_mods[i] = rk

                raw_mean = sum(raw_event_reward_mods[i] for i in affected_players) / affected_count
                if neutralize_payoff:
                    for i in affected_players:
                        raw_event_reward_mods[i] -= raw_mean
                post_mean = sum(raw_event_reward_mods[i] for i in affected_players) / affected_count
                event_reward_mean_raw_sum += raw_mean
                event_reward_mean_post_sum += post_mean
                event_reward_abs_mean_post_sum += abs(post_mean)
                event_neutrality_rounds += 1
                event_neutrality_max_abs_mean = max(event_neutrality_max_abs_mean, abs(post_mean))

        if impact_mode == "spread":
            injected_reward_total = 0.0
            expected_reward_total = 0.0
            for i in range(n):
                rm = raw_event_reward_mods[i]
                rk = raw_event_risk_mods[i]
                if rm == 0.0 and rk == 0.0:
                    continue
                expected_reward_total += float(rm)
                neighbors: list[int] = []
                if impact_neighbor_mass > 0.0 and impact_spread_neighbor_hop > 0:
                    neighbors = _hop_neighbors(eff, i, impact_spread_neighbor_hop)
                local_mass = impact_local_mass
                neighbor_share = 0.0
                if neighbors:
                    neighbor_share = impact_neighbor_mass / float(len(neighbors))
                else:
                    # Keep strict mass conservation even without reachable neighbors.
                    local_mass += impact_neighbor_mass
                for kidx, w in enumerate(impact_kernel):
                    if w == 0.0:
                        continue
                    local_rm = float(rm) * float(w) * float(local_mass)
                    local_rk = float(rk) * float(w) * float(local_mass)
                    impact_reward_queue[kidx][i] += local_rm
                    impact_risk_queue[kidx][i] += local_rk
                    injected_reward_total += local_rm
                    if neighbor_share > 0.0:
                        nb_rm = float(rm) * float(w) * float(neighbor_share)
                        nb_rk = float(rk) * float(w) * float(neighbor_share)
                        for j in neighbors:
                            impact_reward_queue[kidx][j] += nb_rm
                            impact_risk_queue[kidx][j] += nb_rk
                            injected_reward_total += nb_rm

            event_reward_mods = impact_reward_queue.pop(0)
            impact_reward_queue.append([0.0] * n)
            event_risk_mods = impact_risk_queue.pop(0)
            impact_risk_queue.append([0.0] * n)
            impact_kernel_mass_error_round = abs(
                float(injected_reward_total) - float(expected_reward_total)
            )
        else:
            event_reward_mods = raw_event_reward_mods
            event_risk_mods = raw_event_risk_mods
            impact_kernel_mass_error_round = 0.0

        impact_spread_alignment_round = _vector_alignment(
            raw_event_reward_mods,
            event_reward_mods,
        ) if impact_mode == "spread" else 0.0
        impact_spread_alignment_sum += impact_spread_alignment_round
        impact_spread_rounds += 1
        impact_kernel_mass_error_max = max(
            impact_kernel_mass_error_max,
            float(impact_kernel_mass_error_round),
        )

        if queue_enabled:
            for i in range(n):
                rm = float(event_reward_mods[i])
                rk = float(event_risk_mods[i])
                if rm == 0.0 and rk == 0.0:
                    continue
                q = player_event_queues[i]
                q.append((rm, rk))
                if queue_cap > 0:
                    while len(q) > queue_cap:
                        q.popleft()
                        queue_drop_count_total += 1

            queued_reward_mods = [0.0] * n
            queued_risk_mods = [0.0] * n
            for i in range(n):
                rm, rk = _drain_player_event_queue(
                    queue=player_event_queues[i],
                    drain_rate=queue_drain_rate,
                )
                queued_reward_mods[i] = rm
                queued_risk_mods[i] = rk

            event_reward_mods = queued_reward_mods
            event_risk_mods = queued_risk_mods

            queue_depths = [len(q) for q in player_event_queues]
            queue_depth_mean_round = (
                sum(queue_depths) / float(n)
            ) if n > 0 else 0.0
            queue_depth_p95_round = _p95(queue_depths)
        else:
            queue_depth_mean_round = 0.0
            queue_depth_p95_round = 0.0

        queue_depth_mean_sum += queue_depth_mean_round
        queue_depth_p95_sum += queue_depth_p95_round
        queue_depth_rounds += 1

        readonly_leak_score_round = float(readonly_leak_score_round if world_readonly_applied else 0.0)
        payoff_static_score_round = float(
            payoff_static_score_round if world_difficulty_only_applied else 0.0
        )
        readonly_leak_max = max(readonly_leak_max, readonly_leak_score_round)
        readonly_leak_sum += readonly_leak_score_round
        readonly_leak_rounds += 1
        payoff_static_max = max(payoff_static_max, payoff_static_score_round)
        payoff_static_sum += payoff_static_score_round
        payoff_static_rounds += 1

        reward_multiplier_raw_round = 1.0
        reward_multiplier_clamped_round = 1.0
        log_reward_multiplier_round = 0.0
        modulation_gain_effective_round = 0.0
        modulation_zero_mean_residual_round = 0.0
        multiplicative_static_pass_round = True

        if reward_mode == "additive":
            rewards = [rewards[i] + event_reward_mods[i] for i in range(n)]
        else:
            if event_modulation_mode == "multiplicative_v2":
                mod_multipliers, mod_diag = _apply_log_multiplicative_modulation(
                    event_reward_mods=event_reward_mods,
                    gain=event_modulation_gain,
                    log_center=event_modulation_log_center,
                    zero_mean=event_modulation_zero_mean,
                    floor=event_modulation_floor,
                    ceiling=event_modulation_ceiling,
                    eps=modulation_zero_mean_eps,
                )
                raw_multipliers = [max(1e-6, 1.0 + float(event_reward_mods[i])) for i in range(n)]
                multipliers = [
                    max(
                        1e-6,
                        min(
                            1.0 + multiplier_cap,
                            max(1.0 - multiplier_cap, float(mod_multipliers[i])),
                        ),
                    )
                    for i in range(n)
                ]
                rewards = [rewards[i] * multipliers[i] for i in range(n)]

                reward_multiplier_raw_round = float(mod_diag["reward_multiplier_raw"])
                reward_multiplier_clamped_round = sum(multipliers) / float(n) if n > 0 else 1.0
                log_reward_multiplier_round = float(mod_diag["log_reward_multiplier"])
                modulation_gain_effective_round = float(mod_diag["modulation_gain_effective"])
                modulation_zero_mean_residual_round = float(mod_diag["modulation_zero_mean_residual"])
                multiplicative_static_pass_round = bool(mod_diag["multiplicative_static_pass"])

                if n > 0:
                    reward_multiplier_raw_round = sum(raw_multipliers) / float(n)
            else:
                multipliers = [
                    max(
                        1e-6,
                        min(
                            1.0 + multiplier_cap,
                            max(1.0 - multiplier_cap, 1.0 + event_reward_mods[i]),
                        ),
                    )
                    for i in range(n)
                ]
                rewards = [rewards[i] * multipliers[i] for i in range(n)]
                if n > 0:
                    reward_multiplier_raw_round = sum(
                        max(1e-6, 1.0 + float(event_reward_mods[i]))
                        for i in range(n)
                    ) / float(n)
                    reward_multiplier_clamped_round = sum(multipliers) / float(n)
                    log_reward_multiplier_round = sum(log(max(1e-6, m)) for m in multipliers) / float(n)

        reward_multiplier_raw_sum += reward_multiplier_raw_round
        reward_multiplier_clamped_sum += reward_multiplier_clamped_round
        log_reward_multiplier_sum += log_reward_multiplier_round
        modulation_gain_effective_sum += modulation_gain_effective_round
        modulation_zero_mean_residual_max = max(
            modulation_zero_mean_residual_max,
            modulation_zero_mean_residual_round,
        )
        if not multiplicative_static_pass_round:
            multiplicative_static_failures += 1
        modulation_obs_rounds += 1

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

        if async_update_applied:
            update_indices = _select_async_update_indices(
                n_players=n,
                minibatch=replicator_async_minibatch,
                jitter=replicator_async_jitter,
                rng_dispatch=rng_dispatch,
            )
        else:
            update_indices = list(range(n))
        update_set = set(update_indices)

        async_update_ratio_round = (len(update_indices) / float(n)) if n > 0 else 0.0
        async_update_count_total += len(update_indices)
        async_update_ratio_sum += async_update_ratio_round
        async_update_ratio_min = min(async_update_ratio_min, async_update_ratio_round)
        async_update_ratio_max = max(async_update_ratio_max, async_update_ratio_round)
        async_update_rounds += 1

        for idx in update_indices:
            player_update_counts[idx] += 1

        if async_update_applied:
            mean_updates_now = sum(player_update_counts) / float(n) if n > 0 else 0.0
            if mean_updates_now > 0.0 and n > 0:
                var_updates_now = (
                    sum((c - mean_updates_now) ** 2 for c in player_update_counts) / float(n)
                )
                update_skew_index_round = (var_updates_now ** 0.5) / mean_updates_now
            else:
                update_skew_index_round = 0.0
        else:
            update_skew_index_round = 0.0

        phase_lag_index_round = (
            abs(affected_ratio - async_update_ratio_round)
            if async_update_applied else 0.0
        )
        phase_lag_index_sum += phase_lag_index_round
        phase_lag_rounds += 1

        # 7f. Q-update (BL2-compatible, per-strategy α multipliers,
        #     risk-sensitive reward bias per §4.2 outlet 4)
        # risk_sign: aggressive=-1 (risky), defensive=+1 (safe), balanced=0
        _RISK_SIGN = (-1.0, 1.0, 0.0)
        for i in range(n):
            p = players[i]
            if i in update_set:
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
            row["world_state_scarcity"] = world.state["scarcity"]
            row["world_state_threat"] = world.state["threat"]
            row["world_state_noise"] = world.state["noise"]
            row["world_state_intel"] = world.state["intel"]
            difficulty_index_round = _world_difficulty_index(world.state)
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
            row["world_state_scarcity"] = 0.0
            row["world_state_threat"] = 0.0
            row["world_state_noise"] = 0.0
            row["world_state_intel"] = 0.0
            difficulty_index_round = 0.0
        row["world_readonly_applied"] = 1 if world_readonly_applied else 0
        row["readonly_leak_score"] = readonly_leak_score_round
        row["difficulty_index"] = difficulty_index_round
        row["event_difficulty_multiplier"] = event_difficulty_multiplier_round
        row["difficulty_modulation_applied"] = 1 if world_difficulty_only_applied else 0
        row["payoff_static_pass"] = (
            payoff_static_score_round <= payoff_static_eps
        ) if world_difficulty_only_applied else True
        row["reward_multiplier_raw"] = reward_multiplier_raw_round
        row["reward_multiplier_clamped"] = reward_multiplier_clamped_round
        row["log_reward_multiplier"] = log_reward_multiplier_round
        row["modulation_gain_effective"] = modulation_gain_effective_round
        row["modulation_zero_mean_residual"] = modulation_zero_mean_residual_round
        row["multiplicative_static_pass"] = multiplicative_static_pass_round
        row["impact_kernel_id"] = impact_spread_kernel_id
        row["impact_kernel_mass_local"] = impact_local_mass
        row["impact_kernel_mass_neighbor"] = impact_neighbor_mass
        row["impact_spread_radius"] = float(impact_spread_neighbor_hop)
        row["impact_spread_delay_mean"] = impact_kernel_delay_mean
        row["impact_spread_alignment"] = impact_spread_alignment_round
        row["impact_kernel_mass_error"] = impact_kernel_mass_error_round
        row["impact_spread_applied"] = 1 if impact_mode == "spread" else 0
        row["replicator_update_mode"] = replicator_update_mode
        row["async_update_applied"] = 1 if async_update_applied else 0
        row["async_update_ratio"] = async_update_ratio_round
        row["player_event_queue_depth_mean"] = queue_depth_mean_round
        row["player_event_queue_depth_p95"] = queue_depth_p95_round
        row["queue_drop_count"] = int(queue_drop_count_total)
        row["update_skew_index"] = update_skew_index_round
        row["phase_lag_index"] = phase_lag_index_round
        row["dominant_event_type"] = event_type

        difficulty_index_sum += difficulty_index_round
        event_difficulty_multiplier_sum += event_difficulty_multiplier_round
        difficulty_obs_rounds += 1

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

    if n > 0:
        mean_updates = sum(player_update_counts) / float(n)
        if mean_updates > 0.0:
            var_updates = sum((c - mean_updates) ** 2 for c in player_update_counts) / float(n)
            update_skew_index = (var_updates ** 0.5) / mean_updates
        else:
            update_skew_index = 0.0
    else:
        update_skew_index = 0.0

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
        "world_feedback_mode": world_mode,
        "world_feedback_delay_windows": int(cfg.world_feedback_delay_windows),
        "world_feedback_smooth_windows": int(cfg.world_feedback_smooth_windows),
        "world_feedback_effective_mode": (
            "adaptive_world_delayed"
            if (
                world_mode == "adaptive_world"
                and (
                    int(cfg.world_feedback_delay_windows) > 0
                    or int(cfg.world_feedback_smooth_windows) > 1
                )
            )
            else world_mode
        ),
        "world_readonly_applied": world_readonly_applied,
        "readonly_leak_eps": readonly_leak_eps,
        "readonly_leak_score": readonly_leak_max,
        "readonly_leak_score_mean": (
            readonly_leak_sum / readonly_leak_rounds
        ) if readonly_leak_rounds else 0.0,
        "readonly_leak_pass": (
            readonly_leak_max <= readonly_leak_eps
        ) if world_readonly_applied else True,
        "difficulty_modulation_applied": world_difficulty_only_applied,
        "difficulty_index_mean": (
            difficulty_index_sum / difficulty_obs_rounds
        ) if difficulty_obs_rounds else 0.0,
        "event_difficulty_multiplier_mean": (
            event_difficulty_multiplier_sum / difficulty_obs_rounds
        ) if difficulty_obs_rounds else 1.0,
        "payoff_static_eps": payoff_static_eps,
        "payoff_static_score": payoff_static_max,
        "payoff_static_score_mean": (
            payoff_static_sum / payoff_static_rounds
        ) if payoff_static_rounds else 0.0,
        "payoff_static_pass": (
            payoff_static_max <= payoff_static_eps
        ) if world_difficulty_only_applied else True,
        "event_reward_mode": reward_mode,
        "event_reward_multiplier_cap": multiplier_cap,
        "event_modulation_mode": event_modulation_mode,
        "event_modulation_gain": event_modulation_gain,
        "event_modulation_log_center": event_modulation_log_center,
        "event_modulation_zero_mean": event_modulation_zero_mean,
        "event_modulation_floor": min(event_modulation_floor, event_modulation_ceiling),
        "event_modulation_ceiling": max(event_modulation_floor, event_modulation_ceiling),
        "modulation_zero_mean_eps": modulation_zero_mean_eps,
        "reward_multiplier_raw_mean": (
            reward_multiplier_raw_sum / modulation_obs_rounds
        ) if modulation_obs_rounds else 1.0,
        "reward_multiplier_clamped_mean": (
            reward_multiplier_clamped_sum / modulation_obs_rounds
        ) if modulation_obs_rounds else 1.0,
        "log_reward_multiplier_mean": (
            log_reward_multiplier_sum / modulation_obs_rounds
        ) if modulation_obs_rounds else 0.0,
        "modulation_gain_effective_mean": (
            modulation_gain_effective_sum / modulation_obs_rounds
        ) if modulation_obs_rounds else 0.0,
        "modulation_zero_mean_residual_max": modulation_zero_mean_residual_max,
        "multiplicative_static_failures": int(multiplicative_static_failures),
        "multiplicative_static_pass": multiplicative_static_failures == 0,
        "event_neutralize_payoff": neutralize_payoff,
        "event_neutralize_eps": neutralize_eps,
        "event_reward_mean_raw_over_affected": (
            event_reward_mean_raw_sum / event_neutrality_rounds
        ) if event_neutrality_rounds else 0.0,
        "event_reward_mean_after_neutralize": (
            event_reward_mean_post_sum / event_neutrality_rounds
        ) if event_neutrality_rounds else 0.0,
        "event_reward_abs_mean_after_neutralize": (
            event_reward_abs_mean_post_sum / event_neutrality_rounds
        ) if event_neutrality_rounds else 0.0,
        "event_neutrality_rounds": event_neutrality_rounds,
        "event_neutrality_max_abs_mean": event_neutrality_max_abs_mean,
        "event_neutrality_pass": (
            (event_neutrality_max_abs_mean <= neutralize_eps) if neutralize_payoff else True
        ),
        "event_impact_mode": impact_mode,
        "event_impact_horizon": impact_horizon,
        "event_impact_decay": impact_decay,
        "impact_spread_kernel_id": impact_spread_kernel_id,
        "impact_kernel_mass_local": float(impact_local_mass),
        "impact_kernel_mass_neighbor": float(impact_neighbor_mass),
        "impact_spread_radius": float(impact_spread_neighbor_hop),
        "impact_spread_memory_kernel": int(impact_spread_memory_kernel),
        "impact_spread_delay_mean": impact_kernel_delay_mean,
        "impact_spread_alignment_mean": (
            impact_spread_alignment_sum / impact_spread_rounds
        ) if impact_spread_rounds else 0.0,
        "impact_kernel_mass_error": float(impact_kernel_mass_error_max),
        "impact_spread_applied": (impact_mode == "spread"),
        "event_trigger_mode": trigger_mode,
        "event_trigger_entropy_threshold": trigger_entropy_threshold,
        "event_trigger_guard_check_count": event_trigger_guard_check_count,
        "event_trigger_guard_block_count": event_trigger_guard_block_count,
        "event_trigger_guard_block_rate": (
            event_trigger_guard_block_count / event_trigger_guard_check_count
        ) if event_trigger_guard_check_count else 0.0,
        "event_trigger_guard_pass": (
            True
            if (trigger_mode != "entropy_guard" or not dispatch_configured)
            else (
                event_trigger_guard_check_count > 0
                and event_trigger_guard_block_count < event_trigger_guard_check_count
            )
        ),
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
        "replicator_update_mode": replicator_update_mode,
        "async_update_applied": async_update_applied,
        "async_update_ratio_mean": (
            async_update_ratio_sum / async_update_rounds
        ) if async_update_rounds else 0.0,
        "async_update_ratio_min": async_update_ratio_min if async_update_rounds else 0.0,
        "async_update_ratio_max": async_update_ratio_max if async_update_rounds else 0.0,
        "async_update_count_total": int(async_update_count_total),
        "update_skew_index": float(update_skew_index),
        "event_queue_mode": queue_mode,
        "event_queue_cap": int(queue_cap),
        "event_queue_drain_rate": float(queue_drain_rate),
        "queue_overflow_count": int(queue_drop_count_total),
        "player_event_queue_depth_mean": (
            queue_depth_mean_sum / queue_depth_rounds
        ) if queue_depth_rounds else 0.0,
        "player_event_queue_depth_p95": (
            queue_depth_p95_sum / queue_depth_rounds
        ) if queue_depth_rounds else 0.0,
        "phase_lag_index_mean": (
            phase_lag_index_sum / phase_lag_rounds
        ) if phase_lag_rounds else 0.0,
    }

    # attach world-update rows and aggregate boundary/instability stats
    try:
        if world is not None and hasattr(world, "world_update_rows"):
            wrows = list(world.world_update_rows)
            diagnostics["world_update_rows"] = wrows
            windows = len(wrows)
            boundary_hit_count = sum(1 for r in wrows if bool(r.get("boundary_hit")))
            instability_warning_count = sum(1 for r in wrows if bool(r.get("instability_warning")))
            diagnostics["boundary_hit_count"] = int(boundary_hit_count)
            diagnostics["instability_warning_count"] = int(instability_warning_count)
            diagnostics["world_update_windows"] = int(windows)
            diagnostics["boundary_hit_rate"] = float(boundary_hit_count / windows) if windows else 0.0
        else:
            diagnostics["world_update_rows"] = []
            diagnostics["boundary_hit_count"] = 0
            diagnostics["instability_warning_count"] = 0
            diagnostics["world_update_windows"] = 0
            diagnostics["boundary_hit_rate"] = 0.0
    except Exception:
        diagnostics.setdefault("world_update_rows", [])
        diagnostics.setdefault("boundary_hit_count", 0)
        diagnostics.setdefault("instability_warning_count", 0)
        diagnostics.setdefault("world_update_windows", 0)
        diagnostics.setdefault("boundary_hit_rate", 0.0)

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
    diag = diagnostics or {}
    diag_world_mode = str(
        diag.get(
            "world_feedback_mode",
            ("adaptive_world" if cfg.world_feedback else "off"),
        )
    )
    if diag_world_mode in {"", "off"}:
        world_mode_value = cfg.topology
        lambda_world_value = 0.0
        world_update_interval_value = 0
    else:
        world_mode_value = diag_world_mode
        lambda_world_value = cfg.lambda_world
        world_update_interval_value = cfg.world_update_interval

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
        "world_mode": world_mode_value,
        "lambda_world": lambda_world_value,
        "world_update_interval": world_update_interval_value,
        "seed": seed,
        "mean_alpha": mean_a,
        "std_alpha": std_a,
        "mean_beta": mean_b,
        "std_beta": std_b,
        "risk_rule_version": "v1",
    }
    prov["event_sync_index_mean"] = float(diag.get("event_sync_index_mean", 0.0))
    prov["reward_perturb_corr"] = float(diag.get("reward_perturb_corr", 0.0))
    prov["trap_entry_round"] = diag.get("trap_entry_round")
    prov["dispatch_seed_stream"] = int(diag.get("dispatch_seed_stream", 0))
    prov["dispatch_mode"] = str(diag.get("dispatch_mode", "sync"))
    prov["dispatch_target_rate"] = float(diag.get("dispatch_target_rate", 0.0))
    prov["world_readonly_applied"] = bool(diag.get("world_readonly_applied", False))
    prov["world_feedback_delay_windows"] = int(
        diag.get("world_feedback_delay_windows", cfg.world_feedback_delay_windows)
    )
    prov["world_feedback_smooth_windows"] = int(
        diag.get("world_feedback_smooth_windows", cfg.world_feedback_smooth_windows)
    )
    prov["world_feedback_effective_mode"] = str(
        diag.get(
            "world_feedback_effective_mode",
            "adaptive_world_delayed"
            if (
                diag_world_mode == "adaptive_world"
                and (
                    int(cfg.world_feedback_delay_windows) > 0
                    or int(cfg.world_feedback_smooth_windows) > 1
                )
            )
            else diag_world_mode,
        )
    )
    prov["readonly_leak_score"] = float(diag.get("readonly_leak_score", 0.0))
    prov["readonly_leak_pass"] = bool(diag.get("readonly_leak_pass", True))
    prov["difficulty_modulation_applied"] = bool(diag.get("difficulty_modulation_applied", False))
    prov["difficulty_index_mean"] = float(diag.get("difficulty_index_mean", 0.0))
    prov["event_difficulty_multiplier_mean"] = float(diag.get("event_difficulty_multiplier_mean", 1.0))
    prov["payoff_static_score"] = float(diag.get("payoff_static_score", 0.0))
    prov["payoff_static_pass"] = bool(diag.get("payoff_static_pass", True))
    prov["event_reward_mode"] = str(diag.get("event_reward_mode", cfg.event_reward_mode))
    prov["event_reward_multiplier_cap"] = float(
        diag.get("event_reward_multiplier_cap", cfg.event_reward_multiplier_cap)
    )
    prov["event_modulation_mode"] = str(
        diag.get("event_modulation_mode", cfg.event_modulation_mode)
    )
    prov["event_modulation_gain"] = float(
        diag.get("event_modulation_gain", cfg.event_modulation_gain)
    )
    prov["event_modulation_log_center"] = float(
        diag.get("event_modulation_log_center", cfg.event_modulation_log_center)
    )
    prov["event_modulation_zero_mean"] = bool(
        diag.get("event_modulation_zero_mean", cfg.event_modulation_zero_mean)
    )
    prov["event_modulation_floor"] = float(
        diag.get("event_modulation_floor", cfg.event_modulation_floor)
    )
    prov["event_modulation_ceiling"] = float(
        diag.get("event_modulation_ceiling", cfg.event_modulation_ceiling)
    )
    prov["reward_multiplier_raw_mean"] = float(
        diag.get("reward_multiplier_raw_mean", 1.0)
    )
    prov["reward_multiplier_clamped_mean"] = float(
        diag.get("reward_multiplier_clamped_mean", 1.0)
    )
    prov["log_reward_multiplier_mean"] = float(
        diag.get("log_reward_multiplier_mean", 0.0)
    )
    prov["modulation_gain_effective_mean"] = float(
        diag.get("modulation_gain_effective_mean", 0.0)
    )
    prov["modulation_zero_mean_residual_max"] = float(
        diag.get("modulation_zero_mean_residual_max", 0.0)
    )
    prov["multiplicative_static_pass"] = bool(
        diag.get("multiplicative_static_pass", True)
    )
    prov["event_neutralize_payoff"] = bool(
        diag.get("event_neutralize_payoff", cfg.event_neutralize_payoff)
    )
    prov["event_neutralize_eps"] = float(
        diag.get("event_neutralize_eps", cfg.event_neutralize_eps)
    )
    prov["event_neutrality_max_abs_mean"] = float(
        diag.get("event_neutrality_max_abs_mean", 0.0)
    )
    prov["event_neutrality_pass"] = bool(diag.get("event_neutrality_pass", True))
    prov["event_impact_mode"] = str(diag.get("event_impact_mode", cfg.event_impact_mode))
    prov["event_impact_horizon"] = int(diag.get("event_impact_horizon", cfg.event_impact_horizon))
    prov["event_impact_decay"] = float(diag.get("event_impact_decay", cfg.event_impact_decay))
    prov["impact_spread_kernel_id"] = str(diag.get("impact_spread_kernel_id", cfg.impact_spread_kernel_id))
    prov["impact_kernel_mass_local"] = float(diag.get("impact_kernel_mass_local", cfg.impact_spread_local_mass))
    prov["impact_kernel_mass_neighbor"] = float(diag.get("impact_kernel_mass_neighbor", cfg.impact_spread_neighbor_mass))
    prov["impact_spread_radius"] = float(diag.get("impact_spread_radius", cfg.impact_spread_neighbor_hop))
    prov["impact_spread_memory_kernel"] = int(diag.get("impact_spread_memory_kernel", cfg.impact_spread_memory_kernel))
    prov["impact_spread_delay_mean"] = float(diag.get("impact_spread_delay_mean", 0.0))
    prov["impact_spread_alignment"] = float(diag.get("impact_spread_alignment_mean", 0.0))
    prov["impact_kernel_mass_error"] = float(diag.get("impact_kernel_mass_error", 0.0))
    prov["impact_spread_applied"] = bool(diag.get("impact_spread_applied", False))
    prov["event_trigger_mode"] = str(diag.get("event_trigger_mode", cfg.event_trigger_mode))
    prov["event_trigger_entropy_threshold"] = float(
        diag.get("event_trigger_entropy_threshold", cfg.event_trigger_entropy_threshold)
    )
    prov["event_trigger_guard_check_count"] = int(diag.get("event_trigger_guard_check_count", 0))
    prov["event_trigger_guard_block_count"] = int(diag.get("event_trigger_guard_block_count", 0))
    prov["event_trigger_guard_block_rate"] = float(diag.get("event_trigger_guard_block_rate", 0.0))
    prov["event_trigger_guard_pass"] = bool(diag.get("event_trigger_guard_pass", True))
    prov["dispatch_mean_affected_ratio"] = float(diag.get("dispatch_mean_affected_ratio", 0.0))
    prov["dispatch_player_activation_min"] = int(diag.get("dispatch_player_activation_min", 0))
    prov["dispatch_player_activation_max"] = int(diag.get("dispatch_player_activation_max", 0))
    prov["dispatch_player_activation_cv"] = float(diag.get("dispatch_player_activation_cv", 0.0))
    prov["dispatch_fairness_window"] = int(diag.get("dispatch_fairness_window", 0))
    prov["dispatch_fairness_tolerance"] = float(diag.get("dispatch_fairness_tolerance", 0.0))
    prov["dispatch_fairness_checks"] = int(diag.get("dispatch_fairness_checks", 0))
    prov["dispatch_fairness_failures"] = int(diag.get("dispatch_fairness_failures", 0))
    prov["dispatch_fairness_pass"] = bool(diag.get("dispatch_fairness_pass", True))
    prov["replicator_update_mode"] = str(diag.get("replicator_update_mode", cfg.replicator_update_mode))
    prov["async_update_applied"] = bool(diag.get("async_update_applied", False))
    prov["async_update_ratio_mean"] = float(diag.get("async_update_ratio_mean", 0.0))
    prov["async_update_ratio_min"] = float(diag.get("async_update_ratio_min", 0.0))
    prov["async_update_ratio_max"] = float(diag.get("async_update_ratio_max", 0.0))
    prov["async_update_count_total"] = int(diag.get("async_update_count_total", 0))
    prov["update_skew_index"] = float(diag.get("update_skew_index", 0.0))
    prov["event_queue_mode"] = str(diag.get("event_queue_mode", cfg.event_queue_mode))
    prov["event_queue_cap"] = int(diag.get("event_queue_cap", cfg.event_queue_cap))
    prov["event_queue_drain_rate"] = float(diag.get("event_queue_drain_rate", cfg.event_queue_drain_rate))
    prov["queue_overflow_count"] = int(diag.get("queue_overflow_count", 0))
    prov["player_event_queue_depth_mean"] = float(diag.get("player_event_queue_depth_mean", 0.0))
    prov["player_event_queue_depth_p95"] = float(diag.get("player_event_queue_depth_p95", 0.0))
    prov["phase_lag_index_mean"] = float(diag.get("phase_lag_index_mean", 0.0))
    prov["diagnostic_rule_version"] = "diag_v1"
    prov["world_update_windows"] = int(diag.get("world_update_windows", 0))
    prov["boundary_hit_count"] = int(diag.get("boundary_hit_count", 0))
    prov["instability_warning_count"] = int(diag.get("instability_warning_count", 0))
    prov["boundary_hit_rate"] = float(diag.get("boundary_hit_rate", 0.0))
    prov["world_update_rows"] = diag.get("world_update_rows", [])
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
        "--event-modulation-mode",
        default="off",
        choices=["off", "multiplicative_v2"],
    )
    parser.add_argument("--event-modulation-gain", type=float, default=0.0)
    parser.add_argument("--event-modulation-log-center", type=float, default=0.0)
    parser.add_argument("--event-modulation-zero-mean", action="store_true", default=False)
    parser.add_argument("--event-modulation-floor", type=float, default=1.0)
    parser.add_argument("--event-modulation-ceiling", type=float, default=1.0)
    parser.add_argument("--event-neutralize-payoff", action="store_true", default=False)
    parser.add_argument("--event-neutralize-eps", type=float, default=1e-9)
    parser.add_argument(
        "--event-impact-mode",
        default="instant",
        choices=["instant", "spread"],
    )
    parser.add_argument("--event-impact-horizon", type=int, default=1)
    parser.add_argument("--event-impact-decay", type=float, default=0.70)
    parser.add_argument("--impact-spread-kernel-id", default="legacy_v1")
    parser.add_argument("--impact-spread-local-mass", type=float, default=1.0)
    parser.add_argument("--impact-spread-neighbor-mass", type=float, default=0.0)
    parser.add_argument("--impact-spread-neighbor-hop", type=int, default=0)
    parser.add_argument("--impact-spread-memory-kernel", type=int, default=1)
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
    parser.add_argument(
        "--event-trigger-mode",
        default="always",
        choices=["always", "entropy_guard"],
    )
    parser.add_argument("--event-trigger-entropy-threshold", type=float, default=0.85)
    parser.add_argument(
        "--replicator-update-mode",
        default="sync_global",
        choices=["sync_global", "async_per_player"],
    )
    parser.add_argument("--replicator-async-minibatch", type=int, default=0)
    parser.add_argument("--replicator-async-jitter", type=float, default=0.0)
    parser.add_argument(
        "--event-queue-mode",
        default="off",
        choices=["off", "per_player"],
    )
    parser.add_argument("--event-queue-cap", type=int, default=0)
    parser.add_argument("--event-queue-drain-rate", type=float, default=1.0)
    parser.add_argument("--world-feedback", action="store_true", default=False)
    parser.add_argument(
        "--world-feedback-mode",
        default="off",
        choices=["off", "adaptive_world", "read_only", "difficulty_only"],
    )
    parser.add_argument("--lambda-world", type=float, default=0.04)
    parser.add_argument("--world-update-interval", type=int, default=200)
    parser.add_argument("--world-feedback-delay-windows", type=int, default=0)
    parser.add_argument("--world-feedback-smooth-windows", type=int, default=1)
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
        event_modulation_mode=args.event_modulation_mode,
        event_modulation_gain=args.event_modulation_gain,
        event_modulation_log_center=args.event_modulation_log_center,
        event_modulation_zero_mean=args.event_modulation_zero_mean,
        event_modulation_floor=args.event_modulation_floor,
        event_modulation_ceiling=args.event_modulation_ceiling,
        event_neutralize_payoff=args.event_neutralize_payoff,
        event_neutralize_eps=args.event_neutralize_eps,
        event_impact_mode=args.event_impact_mode,
        event_impact_horizon=args.event_impact_horizon,
        event_impact_decay=args.event_impact_decay,
        impact_spread_kernel_id=args.impact_spread_kernel_id,
        impact_spread_local_mass=args.impact_spread_local_mass,
        impact_spread_neighbor_mass=args.impact_spread_neighbor_mass,
        impact_spread_neighbor_hop=args.impact_spread_neighbor_hop,
        impact_spread_memory_kernel=args.impact_spread_memory_kernel,
        event_dispatch_mode=args.event_dispatch_mode,
        event_dispatch_target_rate=args.event_dispatch_target_rate,
        event_dispatch_batch_size=args.event_dispatch_batch_size,
        event_dispatch_seed_offset=args.event_dispatch_seed_offset,
        event_dispatch_fairness_window=args.event_dispatch_fairness_window,
        event_dispatch_fairness_tolerance=args.event_dispatch_fairness_tolerance,
        event_trigger_mode=args.event_trigger_mode,
        event_trigger_entropy_threshold=args.event_trigger_entropy_threshold,
        replicator_update_mode=args.replicator_update_mode,
        replicator_async_minibatch=args.replicator_async_minibatch,
        replicator_async_jitter=args.replicator_async_jitter,
        event_queue_mode=args.event_queue_mode,
        event_queue_cap=args.event_queue_cap,
        event_queue_drain_rate=args.event_queue_drain_rate,
        world_feedback=args.world_feedback,
        world_feedback_mode=args.world_feedback_mode,
        lambda_world=args.lambda_world,
        world_update_interval=args.world_update_interval,
        world_feedback_delay_windows=max(0, int(args.world_feedback_delay_windows)),
        world_feedback_smooth_windows=max(1, int(args.world_feedback_smooth_windows)),
        out_dir=args.out_dir,
    )

    for seed in (int(s) for s in args.seeds.split(",")):
        _run_seed(cfg, seed)


if __name__ == "__main__":
    main()
