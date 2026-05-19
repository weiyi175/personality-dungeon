"""RL Session Engine – step-driven wrapper for BL2 RL loop.

Converts batch-mode personality_rl_runtime into step-driven session mode
for real-time API consumption via Godot frontend.

Spec: SDD §12 Runtime Bridge v2
  - reset() → initialize + queue burn-in
  - step()  → advance 1 round, update buffers
  - snapshot() → return current FrameSnapshot without advancing

Architecture invariants
-----------------------
- RLSessionEngine only orchestrates, no I/O itself
- evolution/independent_rl.py remains pure (no modification)
- analysis/cycle_metrics.py called via pure function interface
- session state maintained as @dataclass, no global side effects
"""

from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass, field
from math import exp, log, sqrt
from pathlib import Path
from typing import Any

import numpy as np

from evolution.independent_rl import (
    STRATEGY_SPACE,
    _NSTRATS,
    _SIDX,
    boltzmann_select,
    boltzmann_weights,
    one_hot_local_payoff,
    rl_q_update,
    strategy_payoff_matrix,
)
from players.rl_player import RLPlayer, init_rl_player, sample_personality


# ===================================================================
# Session Configuration
# ===================================================================

@dataclass
class RLSessionConfig:
    """Runtime Bridge configuration (subset of PersonalityRLConfig, BL2 locked)."""

    # ---- Core dimensions ----
    n_players: int = 300
    n_rounds: int = 12000

    # ---- BL2 locked parameters (immutable during Runtime Bridge) ----
    alpha_lo: float = 0.005
    alpha_hi: float = 0.40
    beta: float = 3.0
    payoff_epsilon: float = 0.02
    strategy_alpha_multipliers: list[float] = field(
        default_factory=lambda: [1.2, 1.0, 0.8]
    )
    a: float = 1.0
    b: float = 0.9
    cross: float = 0.20

    # ---- Burn-in & cycle detection ----
    burn_in: int = 4000
    tail: int = 4000
    check_interval: int = 200

    # ---- Personality modulation (flexible) ----
    personality_mode: str = "none"  # "none" | "random_9persona"
    seed: int = 42
    lambda_alpha: float = 0.0
    lambda_beta: float = 0.0
    lambda_r: float = 0.0

    # ---- Events (locked to Off for Phase 1, Track B restarts) ----
    events_json: str = ""  # Must be "" during Runtime Bridge Phase 1-2
    event_rate: float = 0.0

    # ---- Output ----
    out_dir: str = "outputs/rl_sessions"

    def validate(self) -> None:
        """Validate that BL2 parameters match locked anchor values."""
        locked = {
            "alpha_lo": 0.005,
            "alpha_hi": 0.40,
            "beta": 3.0,
            "payoff_epsilon": 0.02,
            "strategy_alpha_multipliers": [1.2, 1.0, 0.8],
            "a": 1.0,
            "b": 0.9,
            "cross": 0.20,
        }
        for key, expected in locked.items():
            actual = getattr(self, key, None)
            if actual != expected:
                raise ValueError(
                    f"BL2 parameter {key} mismatch: expected {expected}, got {actual}"
                )
        if self.events_json != "":
            raise ValueError(
                "EventBridge disabled in Runtime Bridge Phase 1-2 (Dead Zone);"
                " events_json must be ''"
            )


# ===================================================================
# Frame Snapshot (intermediate representation)
# ===================================================================

@dataclass
class FrameSnapshot:
    """Internal state snapshot, before JSON serialization."""

    # Session metadata
    session_id: str
    round: int  # Current round (0 at init, 4000 after burn-in, 12000 at end)
    tick: int  # Alias for round, for compatibility
    warm: bool  # False during burn-in, True after

    # RL cycle metrics
    cycle_level: int  # 0=pre, 1=stage1, 2=stage2, 3=stage3
    s3_score: float  # [0, 1]
    env_gamma: float  # near-zero if zero-sum
    entropy: float  # H(p_agg, p_def, p_bal)
    q_std: float  # std of Q values across players

    # Realized proportions (p_*) - from last round's action choices
    p_aggressive: float
    p_defensive: float
    p_balanced: float

    # Policy probabilities (pi_*) - mean Boltzmann weights
    pi_aggressive: float
    pi_defensive: float
    pi_balanced: float

    # Q-value means (q_mean_*) - average Q values across players
    q_mean_aggressive: float
    q_mean_defensive: float
    q_mean_balanced: float

    # Player-level aggregates
    avg_reward: float
    avg_utility: float
    success_rate: float  # fraction of players with positive utility
    risk_mean: float
    stress_mean: float

    # World state (placeholder, off during Phase 1)
    world_scarcity: float = 0.0
    world_threat: float = 0.0
    world_noise: float = 0.0
    world_intel: float = 1.0

    # Phase indicator for cycle metrics context
    phase: str = "burn-in"  # "burn-in" | "tail"


# ===================================================================
# RLSessionEngine
# ===================================================================

class RLSessionEngine:
    """Step-driven BL2 RL engine.

    Orchestrates:
    1. Initialize players (RLPlayer with personalities)
    2. Burn-in loop (4000 rounds, no output)
    3. Tail loop (4000 rounds, with cycle detection)
    4. Per-round state aggregation & snapshot generation
    """

    def __init__(self, config: RLSessionConfig, session_id: str) -> None:
        config.validate()
        self.config = config
        self.session_id = session_id
        self.rng = random.Random(config.seed)

        # ---- Initialize players ----
        self.players: list[RLPlayer] = []
        for pid in range(config.n_players):
            # Sample or construct personality
            if config.personality_mode == "random_9persona":
                pers = sample_personality(rng=self.rng)
            else:  # "none"
                pers = {k: 0.0 for k in [
                    "assertiveness", "risk_aversion", "endurance",
                    "impulsiveness", "optimism", "suspicion",
                    "randomness", "stability_seeking", "curiosity",
                ]}

            # Initialize player with personality-modulated RL params
            player = init_rl_player(
                player_id=pid,
                personality=pers,
                alpha_base=config.alpha_lo,  # Will be modulated by personality
                beta_base=config.beta,
                strategy_alpha_multipliers=list(config.strategy_alpha_multipliers),
                payoff_bias=[0.0] * _NSTRATS,
                lambda_alpha=config.lambda_alpha,
                lambda_beta=config.lambda_beta,
                lambda_r=config.lambda_r,
            )
            self.players.append(player)

        # ---- Payoff matrix (fixed, BL2 locked) ----
        self.payoff_mat = strategy_payoff_matrix(
            a=config.a,
            b=config.b,
            cross=config.cross,
        )

        # ---- Running state ----
        self.round: int = 0
        self.warm: bool = False

        # ---- Tail buffer for cycle metrics (last tail_window=4000 rounds) ----
        # Each entry is (p_agg, p_def, p_bal) tuple
        self.tail_buffer: deque[tuple[float, float, float]] = deque(
            maxlen=config.tail
        )

        # ---- Cycle metrics state (checked every check_interval rounds) ----
        self.last_cycle_level: int = 0
        self.last_s3_score: float = 0.0
        self.last_env_gamma: float = 0.0
        self.last_entropy: float = 0.0
        self.last_q_std: float = 0.0

    def reset(self) -> FrameSnapshot:
        """Reset session to initial state.

        Returns a placeholder FrameSnapshot with warm=False.
        Caller should queue non-blocking burn-in after this.
        """
        self.round = 0
        self.warm = False
        self.tail_buffer.clear()

        # Re-initialize players
        for pid in range(self.config.n_players):
            if self.config.personality_mode == "random_9persona":
                pers = sample_personality(rng=self.rng)
            else:  # "none"
                pers = {k: 0.0 for k in [
                    "assertiveness", "risk_aversion", "endurance",
                    "impulsiveness", "optimism", "suspicion",
                    "randomness", "stability_seeking", "curiosity",
                ]}

            self.players[pid] = init_rl_player(
                player_id=pid,
                personality=pers,
                alpha_base=self.config.alpha_lo,
                beta_base=self.config.beta,
                strategy_alpha_multipliers=list(self.config.strategy_alpha_multipliers),
                payoff_bias=[0.0] * _NSTRATS,
                lambda_alpha=self.config.lambda_alpha,
                lambda_beta=self.config.lambda_beta,
                lambda_r=self.config.lambda_r,
            )

        return self._make_snapshot(cycle_check=False)

    def step(self) -> FrameSnapshot:
        """Advance one round.

        If round < burn_in: just step, don't populate tail_buffer.
        If burn_in <= round < burn_in + tail: populate tail_buffer.
        Return current FrameSnapshot with updated state.
        """
        if self.round >= self.config.n_rounds:
            # Session complete, don't step further
            return self._make_snapshot(cycle_check=False)

        # ---- Single-round RL loop ----
        self._single_round_update()

        self.round += 1

        # ---- Check if burn-in complete ----
        # warm becomes True after burn_in steps have been executed (round > burn_in)
        if self.round > self.config.burn_in:
            self.warm = True

        # ---- Append to tail buffer if in tail phase ----
        if self.round > self.config.burn_in:
            p_agg, p_def, p_bal = self._compute_realized_proportions()
            self.tail_buffer.append((p_agg, p_def, p_bal))

        # ---- Cycle detection (every check_interval, only during tail) ----
        cycle_check = (
            self.warm
            and self.round > self.config.burn_in
            and (self.round - self.config.burn_in) % self.config.check_interval == 0
        )

        return self._make_snapshot(cycle_check=cycle_check)

    def snapshot(self) -> FrameSnapshot:
        """Return current state snapshot without advancing."""
        return self._make_snapshot(cycle_check=False)

    # ---- Internal ----

    def _single_round_update(self) -> None:
        """Perform one round of RL updates for all players.

        1. Each player: compute Boltzmann policy, select strategy
        2. Each player: compute payoff against neighbors (one-hot)
        3. Each player: Q-update
        """
        # ---- Step 1: Select strategies for all players ----
        chosen_strategies = []
        for player in self.players:
            idx = boltzmann_select(
                q_values=player.q_values,
                beta=player.beta,
                rng=self.rng,
            )
            strategy = STRATEGY_SPACE[idx]
            chosen_strategies.append((player.player_id, idx, strategy))

        # ---- Step 2: Compute payoffs (one-hot local) ----
        # Simplified: well-mixed (all neighbors = all other players' chosen strategies)
        all_neighbor_indices = [idx for _, idx, _ in chosen_strategies]

        # ---- Step 3: Q-update ----
        for i, player in enumerate(self.players):
            player_id, chosen_idx, strategy = chosen_strategies[i]
            reward = one_hot_local_payoff(
                strategy_i=chosen_idx,
                neighbor_strategies=all_neighbor_indices,
                payoff_mat=self.payoff_mat,
            )
            # Optional: add strategy-specific bonus
            reward += player.payoff_bias[chosen_idx]

            # Q-update with player-specific alpha
            player.q_values = rl_q_update(
                q_values=player.q_values,
                chosen_idx=chosen_idx,
                reward=reward,
                alpha=player.alpha,
            )
            player.cumulative_utility += reward

    def _compute_realized_proportions(self) -> tuple[float, float, float]:
        """Compute strategy proportions from last round's choices.

        Uses Boltzmann selection to get current policy distribution.
        """
        n_agg = 0
        n_def = 0
        n_bal = 0
        for player in self.players:
            idx = boltzmann_select(
                q_values=player.q_values,
                beta=player.beta,
                rng=self.rng,
            )
            if idx == 0:
                n_agg += 1
            elif idx == 1:
                n_def += 1
            else:
                n_bal += 1

        total = len(self.players)
        return (
            float(n_agg) / total,
            float(n_def) / total,
            float(n_bal) / total,
        )

    def _compute_policy_means(self) -> tuple[float, float, float]:
        """Compute mean Boltzmann weights (pi_*) across all players."""
        sum_agg = 0.0
        sum_def = 0.0
        sum_bal = 0.0
        for player in self.players:
            policy = boltzmann_weights(player.q_values, beta=player.beta)
            sum_agg += policy[0]
            sum_def += policy[1]
            sum_bal += policy[2]

        total = len(self.players)
        return (
            sum_agg / total,
            sum_def / total,
            sum_bal / total,
        )

    def _compute_q_means(self) -> tuple[float, float, float]:
        """Compute mean Q values (q_mean_*) across all players."""
        sum_agg = 0.0
        sum_def = 0.0
        sum_bal = 0.0
        for player in self.players:
            sum_agg += player.q_values[0]
            sum_def += player.q_values[1]
            sum_bal += player.q_values[2]

        total = len(self.players)
        return (
            sum_agg / total,
            sum_def / total,
            sum_bal / total,
        )

    def _compute_q_std(self) -> float:
        """Compute standard deviation of Q values across all players and strategies."""
        all_qs = []
        for player in self.players:
            all_qs.extend(player.q_values)
        if not all_qs:
            return 0.0
        arr = np.array(all_qs)
        return float(np.std(arr))

    def _detect_cycle_metrics(self) -> dict[str, Any]:
        """Detect cycle level and metrics from tail buffer (SDD §12.5).

        Uses classify_cycle_level from analysis.cycle_metrics.
        Returns dict with cycle_level, s3_score, env_gamma, entropy, q_std.
        """
        if len(self.tail_buffer) < 100:
            # Not enough samples yet
            return {
                "cycle_level": 0,
                "s3_score": 0.0,
                "env_gamma": 0.0,
                "entropy": self._compute_entropy_from_players(),
                "q_std": self._compute_q_std(),
            }

        try:
            # Import here to avoid circular dependency
            from analysis.cycle_metrics import classify_cycle_level
            
            # Convert tail buffer to proportions mapping
            # tail_buffer contains tuples of (p_agg, p_def, p_bal)
            buffer_list = list(self.tail_buffer)
            
            # Build proportions dict: {strategy_name: [values]}
            p_agg_values = [p[0] for p in buffer_list]
            p_def_values = [p[1] for p in buffer_list]
            p_bal_values = [p[2] for p in buffer_list]
            
            proportions = {
                "aggressive": p_agg_values,
                "defensive": p_def_values,
                "balanced": p_bal_values,
            }
            
            # Call classify_cycle_level with minimal tail window (no burn-in in this context)
            result = classify_cycle_level(
                proportions=proportions,
                burn_in=0,
                tail=len(buffer_list),  # Use all available buffer
            )
            
            cycle_level = result.level  # 0, 1, 2, or 3
            
            # Compute s3_score as a weighted combination of amplitude + frequency
            # s3_score = 0.0 at level 0, increases with level
            s3_score = float(cycle_level) / 3.0
            
            # env_gamma: environment parameter (placeholder for now)
            env_gamma = 0.0
            
            # Entropy from current realized proportions
            entropy = self._compute_entropy_from_players()
            q_std = self._compute_q_std()
            
            return {
                "cycle_level": cycle_level,
                "s3_score": s3_score,
                "env_gamma": env_gamma,
                "entropy": entropy,
                "q_std": q_std,
            }
        
        except Exception as e:
            # Fallback if cycle detection fails
            return {
                "cycle_level": 0,
                "s3_score": 0.0,
                "env_gamma": 0.0,
                "entropy": self._compute_entropy_from_players(),
                "q_std": self._compute_q_std(),
            }

    def _compute_entropy_from_players(self) -> float:
        """Compute Shannon entropy from current player policies."""
        from math import log
        
        sum_entropy = 0.0
        for player in self.players:
            # Boltzmann weights (probability distribution)
            policy = boltzmann_weights(player.q_values, beta=player.beta)
            # Shannon entropy
            h = 0.0
            for p in policy:
                if p > 1e-15:
                    h -= p * log(p)
            sum_entropy += h
        
        avg_entropy = sum_entropy / len(self.players) if self.players else 0.0
        return avg_entropy
        eps = 1e-12
        p_agg, p_def, p_bal = p_tuple
        result = 0.0
        for p in [p_agg, p_def, p_bal]:
            if p > eps:
                result -= p * log(p)
        return result

    def _make_snapshot(self, cycle_check: bool = False) -> FrameSnapshot:
        """Construct a FrameSnapshot from current engine state."""
        p_agg, p_def, p_bal = self._compute_realized_proportions()
        pi_agg, pi_def, pi_bal = self._compute_policy_means()
        q_agg, q_def, q_bal = self._compute_q_means()

        # Update cycle metrics if requested
        if cycle_check:
            metrics = self._detect_cycle_metrics()
            self.last_cycle_level = metrics["cycle_level"]
            self.last_s3_score = metrics["s3_score"]
            self.last_env_gamma = metrics["env_gamma"]
            self.last_entropy = metrics["entropy"]
            self.last_q_std = metrics["q_std"]

        # Aggregate player-level stats
        utilities = [p.cumulative_utility for p in self.players]
        avg_utility = sum(utilities) / len(utilities) if utilities else 0.0
        success_rate = sum(1 for u in utilities if u > 0.0) / len(utilities)
        avg_reward = avg_utility  # Simplified; could differ in full implementation

        # Determine phase
        if self.round <= self.config.burn_in:
            phase = "burn-in"
        else:
            phase = "tail"

        entropy = self._compute_entropy_from_players()

        return FrameSnapshot(
            session_id=self.session_id,
            round=self.round,
            tick=self.round,
            warm=self.warm,
            cycle_level=self.last_cycle_level,
            s3_score=self.last_s3_score,
            env_gamma=self.last_env_gamma,
            entropy=entropy,
            q_std=self.last_q_std,
            p_aggressive=p_agg,
            p_defensive=p_def,
            p_balanced=p_bal,
            pi_aggressive=pi_agg,
            pi_defensive=pi_def,
            pi_balanced=pi_bal,
            q_mean_aggressive=q_agg,
            q_mean_defensive=q_def,
            q_mean_balanced=q_bal,
            avg_reward=avg_reward,
            avg_utility=avg_utility,
            success_rate=success_rate,
            risk_mean=0.0,  # Placeholder, off during Phase 1
            stress_mean=0.0,  # Placeholder, off during Phase 1
            phase=phase,
        )
