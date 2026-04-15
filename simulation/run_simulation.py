from core.game_engine import GameEngine
from dungeon.dungeon_ai import DungeonAI
from dungeon.event_loader import EventLoader
from analysis.metrics import average_reward, average_utility, strategy_distribution
from evolution.replicator_dynamics import anchored_subgroup_payoff_shift, anchored_subgroup_weight_pull, async_replicator_step, bidirectional_subgroup_payoff_shifts, deterministic_replicator_step, inertial_deterministic_replicator_step, inertial_growth_step, inertial_sampled_replicator_step, replicator_step, sampled_growth_vector, state_dependent_anchored_subgroup_payoff_shift, tangential_projection_replicator_step
from players.base_player import BasePlayer
from simulation.personality_coupling import personality_state_k_factor, resolve_personality_coupling

import argparse
import csv
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Literal

from math import exp
from math import isfinite


@dataclass(frozen=True)
class SimConfig:
	n_players: int
	n_rounds: int
	seed: int | None
	payoff_mode: str
	popularity_mode: str
	gamma: float
	epsilon: float
	a: float
	b: float
	matrix_cross_coupling: float
	init_bias: float
	evolution_mode: str
	payoff_lag: int
	selection_strength: float
	strategy_selection_strengths: tuple[float, float, float] | None = None
	hybrid_update_share: float = 0.0
	hybrid_inertia: float = 0.0
	sampled_inertia: float = 0.0
	fixed_subgroup_share: float = 0.0
	fixed_subgroup_weights: tuple[float, float, float] | None = None
	fixed_subgroup_anchor_pull_strength: float = 1.0
	fixed_subgroup_coupling_strength: float = 0.0
	fixed_subgroup_bidirectional_coupling_strength: float = 0.0
	fixed_subgroup_state_coupling_strength: float = 0.0
	fixed_subgroup_state_coupling_beta: float = 8.0
	fixed_subgroup_state_coupling_theta: float = 0.0
	fixed_subgroup_state_signal: str = "gap_norm"
	enable_events: bool = False
	events_json: Path | None = None
	out_csv: Path = Path("outputs") / "timeseries.csv"
	event_failure_threshold: float | None = None
	event_health_penalty: float | None = None
	event_stress_risk_coefficient: float | None = None
	risk_ma_alpha: float | None = None
	risk_ma_multiplier: float | None = None
	stress_decay_c: float | None = None
	stress_decay_beta: float | None = None
	adaptive_ft_strength: float | None = None
	ft_update_interval: int | None = None
	adaptive_payoff_strength: float = 0.0
	payoff_update_interval: int = 500
	adaptive_payoff_target: float = 0.30
	apply_event_trait_deltas: bool = True
	memory_kernel: int = 1
	threshold_theta: float = 0.40
	threshold_theta_low: float | None = None
	threshold_theta_high: float | None = None
	threshold_trigger: str = "ad_share"
	threshold_state_alpha: float = 1.0
	threshold_a_hi: float | None = None
	threshold_b_hi: float | None = None
	personality_coupling_mu_base: float = 0.0
	personality_coupling_lambda_mu: float = 0.0
	personality_coupling_lambda_k: float = 0.0
	personality_coupling_beta_state_k: float = 0.0
	sampled_growth_n_strata: int = 1
	tangential_drift_delta: float = 0.0
	tangential_alpha: float = 0.0
	async_update_fraction: float = 1.0
	sampling_beta: float = 1.0
	mutation_rate: float = 0.0
	local_group_size: int = 0
	payoff_niche_epsilon: float = 0.0
	niche_group_size: int = 0


DEFAULT_EVENTS_JSON = Path(__file__).resolve().parents[1] / "docs" / "personality_dungeon_v1" / "02_event_templates_v1.json"
PlayerSetupCallback = Callable[[list[object], list[str], SimConfig], None]
RoundCallback = Callable[[int, SimConfig, list[object], DungeonAI, list[dict[str, object]], dict[str, Any]], None]


def _initial_weights(*, strategy_space: list[str], init_bias: float) -> dict[str, float]:
	if not strategy_space:
		return {}
	b = float(init_bias)
	if b == 0.0:
		return {s: 1.0 for s in strategy_space}
	if abs(b) >= 1.0:
		raise ValueError("init_bias must satisfy |init_bias| < 1.0 (keeps weights positive)")
	# Deterministic symmetry breaking: bias aggressive up and defensive down.
	# This is intentionally minimal but sufficient to escape the uniform fixed point.
	w = {s: 1.0 for s in strategy_space}
	if len(strategy_space) >= 1:
		w[strategy_space[0]] = 1.0 + b
	if len(strategy_space) >= 2:
		w[strategy_space[1]] = 1.0 - b
	return w


def _parse_args() -> SimConfig:
	p = argparse.ArgumentParser(description="personality-dungeon MVP simulation")
	strategy_space = ["aggressive", "defensive", "balanced"]
	p.add_argument("--players", type=int, default=50)
	p.add_argument("--rounds", type=int, default=20)
	p.add_argument(
		"--seed",
		type=int,
		default=None,
		help="Random seed for reproducible simulation (controls BasePlayer sampling)",
	)
	p.add_argument(
		"--payoff-mode",
		type=str,
		default="count_cycle",
		choices=["count_cycle", "matrix_ab", "threshold_ab"],
		help="Dungeon payoff model: count-based cyclic penalty/bonus, or matrix U=Ax using (a,b)",
	)
	p.add_argument(
		"--popularity-mode",
		type=str,
		default="sampled",
		choices=["sampled", "expected"],
		help="How dungeon popularity is updated each round: sampled choices (default) or expected distribution from weights (denoised)",
	)
	p.add_argument(
		"--evolution-mode",
		type=str,
		default="sampled",
		choices=["sampled", "mean_field", "hetero", "hybrid", "sampled_inertial", "personality_coupled"],
		help="Weight update rule. 'sampled' uses per-strategy averages from sampled players (default). 'mean_field' uses deterministic expected rewards u=Ax and replicator mapping. 'hetero' keeps sampled updates but uses per-strategy selection strengths for H3. 'sampled_inertial' adds one-step inertia directly on the sampled update operator. 'personality_coupled' applies per-player inertia and selection strength from personality signals.",
	)
	p.add_argument(
		"--strategy-selection-strengths",
		type=str,
		default=None,
		help="H3 only: comma-separated per-strategy selection strengths kA,kD,kB. When omitted, hetero defaults to repeating --selection-strength.",
	)
	p.add_argument(
		"--hybrid-update-share",
		type=float,
		default=0.0,
		help="H4 only: share of sampled players that use deterministic expected-payoff updates while actions/popularity remain sampled.",
	)
	p.add_argument(
		"--hybrid-inertia",
		type=float,
		default=0.0,
		help="H4.1 only: one-step inertia coefficient for hybrid deterministic updates. 0 recovers the original H4 rule.",
	)
	p.add_argument(
		"--sampled-inertia",
		type=float,
		default=0.0,
		help="H5.1: one-step inertia coefficient for the sampled update operator. In mean_field it acts only as the deterministic gate control knob.",
	)
	p.add_argument(
		"--personality-coupling-mu-base",
		type=float,
		default=0.0,
		help="H7.1: base per-player inertia before applying the personality signal.",
	)
	p.add_argument(
		"--personality-coupling-lambda-mu",
		type=float,
		default=0.0,
		help="H7.1: coupling strength from personality signal z_mu into per-player inertia mu_p.",
	)
	p.add_argument(
		"--personality-coupling-lambda-k",
		type=float,
		default=0.0,
		help="H7.1: coupling strength from personality signal z_k into per-player selection strength k_p.",
	)
	p.add_argument(
		"--beta-state-k",
		type=float,
		default=0.0,
		help="B4: linear state-dependent modulation strength for per-player selection strength using current dominance max(p_s).",
	)
	p.add_argument(
		"--sampled-growth-n-strata",
		type=int,
		default=1,
		choices=[1, 3, 5, 10],
		help="B3.1: number of fixed index-based strata used before combining sampled growth. 1 exactly recovers the baseline sampled path.",
	)
	p.add_argument(
		"--tangential-drift-delta",
		type=float,
		default=0.0,
		help="B5: tangential drift magnitude injected before exp(k*g). 0 exactly recovers the baseline operator.",
	)
	p.add_argument(
		"--fixed-subgroup-share",
		type=float,
		default=0.0,
		help="H3.1 only: share of players whose strategy weights remain fixed across rounds.",
	)
	p.add_argument(
		"--fixed-subgroup-weights",
		type=str,
		default=None,
		help="H3.1 only: comma-separated fixed subgroup weights wA,wD,wB. Positive values are normalized to mean=1.",
	)
	p.add_argument(
		"--fixed-subgroup-anchor-pull-strength",
		type=float,
		default=1.0,
		help="H3.4 only: anchor pull strength rho in [0,1]. 1 keeps the subgroup fully frozen (H3.1); 0 removes anchor pullback.",
	)
	p.add_argument(
		"--fixed-subgroup-coupling-strength",
		type=float,
		default=0.0,
		help="H3.2 only: payoff coupling strength from the frozen subgroup anchor into the adaptive subgroup (>=0).",
	)
	p.add_argument(
		"--fixed-subgroup-bidirectional-coupling-strength",
		type=float,
		default=0.0,
		help="H3.5 only: equal-and-opposite payoff coupling strength between the semi-frozen subgroup and the adaptive subgroup (>=0).",
	)
	p.add_argument(
		"--fixed-subgroup-state-coupling-strength",
		type=float,
		default=0.0,
		help="H3.3B only: base state-dependent coupling strength from frozen subgroup into adaptive subgroup (>=0).",
	)
	p.add_argument(
		"--fixed-subgroup-state-coupling-beta",
		type=float,
		default=8.0,
		help="H3.3B only: sigmoid slope beta for the subgroup state gate (>0).",
	)
	p.add_argument(
		"--fixed-subgroup-state-coupling-theta",
		type=float,
		default=0.0,
		help="H3.3B only: subgroup state threshold theta_z (>=0).",
	)
	p.add_argument(
		"--fixed-subgroup-state-signal",
		type=str,
		default="gap_norm",
		choices=["gap_norm"],
		help="H3.3B only: state signal used to gate subgroup coupling. First version only supports 'gap_norm'.",
	)
	p.add_argument(
		"--payoff-lag",
		type=int,
		default=1,
		choices=[0, 1],
		help="Only used in evolution_mode=mean_field. 0: use current x_t for payoff u(x_t). 1: use lagged x_{t-1} for payoff u(x_{t-1}).",
	)
	p.add_argument(
		"--memory-kernel",
		type=int,
		default=1,
		choices=[1, 3, 5],
		help="Payoff input memory length for H1. Uses prefix average of the last m simplex states (allowed: 1,3,5).",
	)
	p.add_argument(
		"--threshold-theta",
		type=float,
		default=0.40,
		help="threshold_ab only: regime switch threshold on q_AD=x_A+x_D computed from payoff input state.",
	)
	p.add_argument(
		"--threshold-theta-low",
		type=float,
		default=None,
		help="threshold_ab only: optional hysteresis low threshold. Defaults to --threshold-theta.",
	)
	p.add_argument(
		"--threshold-theta-high",
		type=float,
		default=None,
		help="threshold_ab only: optional hysteresis high threshold. Defaults to --threshold-theta.",
	)
	p.add_argument(
		"--threshold-trigger",
		type=str,
		default="ad_share",
		choices=["ad_share", "ad_product"],
		help="threshold_ab only: trigger function used by H2.2 regime switching.",
	)
	p.add_argument(
		"--threshold-state-alpha",
		type=float,
		default=1.0,
		help="threshold_ab only: H2.2 state smoothing alpha in (0,1]. 1.0 means no extra state memory.",
	)
	p.add_argument(
		"--threshold-a-hi",
		type=float,
		default=None,
		help="threshold_ab only: high-regime value for a. Defaults to base --a when omitted.",
	)
	p.add_argument(
		"--threshold-b-hi",
		type=float,
		default=None,
		help="threshold_ab only: high-regime value for b. Defaults to base --b when omitted.",
	)
	p.add_argument("--gamma", type=float, default=0.1)
	p.add_argument("--epsilon", type=float, default=0.0)
	p.add_argument("--a", type=float, default=0.0, help="matrix_ab only: payoff matrix parameter a")
	p.add_argument("--b", type=float, default=0.0, help="matrix_ab only: payoff matrix parameter b")
	p.add_argument(
		"--matrix-cross-coupling",
		type=float,
		default=0.0,
		help="matrix_ab only: aggressive-defensive coexistence penalty strength c_AD (default: 0.0)",
	)
	p.add_argument(
		"--init-bias",
		type=float,
		default=0.0,
		help="Initial symmetry-breaking bias applied to weights: w0=(1+bias,1-bias,1) for (aggressive,defensive,balanced). Use with popularity_mode=expected to get deterministic phase rotation; requires |bias|<1.",
	)
	p.add_argument("--selection-strength", type=float, default=0.05)
	p.add_argument(
		"--enable-events",
		action="store_true",
		help="Enable Personality Dungeon event processing using the event schema JSON.",
	)
	p.add_argument(
		"--events-json",
		type=Path,
		default=None,
		help="Path to the Personality Dungeon event schema JSON. Defaults to docs/personality_dungeon_v1/02_event_templates_v1.json when --enable-events is set.",
	)
	p.add_argument(
		"--disable-event-trait-deltas",
		action="store_true",
		help="Disable applying event trait_deltas to player personalities while keeping the event layer enabled.",
	)
	p.add_argument(
		"--out",
		type=Path,
		default=Path("outputs") / "timeseries.csv",
		help="CSV output path (will create parent dirs)",
	)
	p.add_argument(
		"--event-failure-threshold",
		type=float,
		default=None,
		help="Override all per-action failure_threshold values with this value (grid search use). Range [0,1].",
	)
	p.add_argument(
		"--event-health-penalty",
		type=float,
		default=None,
		help="Override health penalty coefficient in final_risk (default: 0.10).",
	)
	p.add_argument(
		"--event-stress-risk-coefficient",
		type=float,
		default=None,
		help="Override stress→risk coupling coefficient in final_risk (default: 0.10).",
	)
	p.add_argument(
		"--risk-ma-alpha",
		type=float,
		default=None,
		help="EMA smoothing factor \u03b1 for risk memory (0=disabled, 0.9=slow decay). Default: 0.0.",
	)
	p.add_argument(
		"--risk-ma-multiplier",
		type=float,
		default=None,
		help="Weight of EMA risk_ma contribution to final_risk. Default: 0.0.",
	)
	p.add_argument(
		"--stress-decay-c",
		type=float,
		default=None,
		help="Nonlinear stress-decay coefficient c (0=disabled). effective_rate=1-(1-base)/(1+c*stress^beta). Default: 0.0.",
	)
	p.add_argument(
		"--stress-decay-beta",
		type=float,
		default=None,
		help="Exponent beta for stress-dependent decay asymmetry (default: 2.0).",
	)
	p.add_argument(
		"--adaptive-ft-strength",
		type=float,
		default=None,
		help="Adaptive failure-threshold coupling strength s (0=disabled). ft_new=ft_base*(1+s*(p_agg-1/3)). Default: 0.0.",
	)
	p.add_argument(
		"--ft-update-interval",
		type=int,
		default=None,
		help="Rounds between adaptive ft updates (default: 500).",
	)
	p.add_argument(
		"--adaptive-payoff-strength",
		type=float,
		default=0.0,
		help="Adaptive strength for a/b payoff adjustment based on p_agg (0=disabled).",
	)
	p.add_argument(
		"--payoff-update-interval",
		type=int,
		default=500,
		help="Rounds between adaptive payoff matrix updates (default: 500).",
	)
	p.add_argument(
		"--adaptive-payoff-target",
		type=float,
		default=0.30,
		help="Target p_agg level for adaptive payoff feedback (default: 0.30).",
	)
	a = p.parse_args()
	return SimConfig(
		n_players=a.players,
		n_rounds=a.rounds,
		seed=a.seed,
		payoff_mode=a.payoff_mode,
		popularity_mode=a.popularity_mode,
		gamma=a.gamma,
		epsilon=a.epsilon,
		a=a.a,
		b=a.b,
		matrix_cross_coupling=a.matrix_cross_coupling,
		init_bias=a.init_bias,
		evolution_mode=a.evolution_mode,
		payoff_lag=a.payoff_lag,
		selection_strength=a.selection_strength,
		strategy_selection_strengths=_coerce_strategy_selection_strengths(
			a.strategy_selection_strengths,
			strategy_space=strategy_space,
		),
		hybrid_update_share=float(a.hybrid_update_share),
		hybrid_inertia=float(a.hybrid_inertia),
		sampled_inertia=float(a.sampled_inertia),
		fixed_subgroup_share=float(a.fixed_subgroup_share),
		fixed_subgroup_weights=_coerce_fixed_subgroup_weights(
			a.fixed_subgroup_weights,
			strategy_space=strategy_space,
		),
		fixed_subgroup_anchor_pull_strength=float(a.fixed_subgroup_anchor_pull_strength),
		fixed_subgroup_coupling_strength=float(a.fixed_subgroup_coupling_strength),
		fixed_subgroup_bidirectional_coupling_strength=float(a.fixed_subgroup_bidirectional_coupling_strength),
		fixed_subgroup_state_coupling_strength=float(a.fixed_subgroup_state_coupling_strength),
		fixed_subgroup_state_coupling_beta=float(a.fixed_subgroup_state_coupling_beta),
		fixed_subgroup_state_coupling_theta=float(a.fixed_subgroup_state_coupling_theta),
		fixed_subgroup_state_signal=str(a.fixed_subgroup_state_signal),
		enable_events=bool(a.enable_events),
		events_json=a.events_json,
		out_csv=a.out,
		event_failure_threshold=a.event_failure_threshold,
		event_health_penalty=a.event_health_penalty,
		event_stress_risk_coefficient=a.event_stress_risk_coefficient,
		risk_ma_alpha=a.risk_ma_alpha,
		risk_ma_multiplier=a.risk_ma_multiplier,
		stress_decay_c=a.stress_decay_c,
		stress_decay_beta=a.stress_decay_beta,
		adaptive_ft_strength=a.adaptive_ft_strength,
		ft_update_interval=a.ft_update_interval,
		adaptive_payoff_strength=a.adaptive_payoff_strength,
		payoff_update_interval=a.payoff_update_interval,
		adaptive_payoff_target=a.adaptive_payoff_target,
		apply_event_trait_deltas=not bool(a.disable_event_trait_deltas),
		memory_kernel=a.memory_kernel,
		threshold_theta=a.threshold_theta,
		threshold_theta_low=a.threshold_theta_low,
		threshold_theta_high=a.threshold_theta_high,
		threshold_trigger=a.threshold_trigger,
		threshold_state_alpha=a.threshold_state_alpha,
		threshold_a_hi=a.threshold_a_hi,
		threshold_b_hi=a.threshold_b_hi,
		personality_coupling_mu_base=float(a.personality_coupling_mu_base),
		personality_coupling_lambda_mu=float(a.personality_coupling_lambda_mu),
		personality_coupling_lambda_k=float(a.personality_coupling_lambda_k),
		personality_coupling_beta_state_k=float(a.beta_state_k),
		sampled_growth_n_strata=int(a.sampled_growth_n_strata),
		tangential_drift_delta=float(a.tangential_drift_delta),
	)


def _coerce_strategy_selection_strengths(
	value: str | tuple[float, ...] | list[float] | None,
	*,
	strategy_space: list[str],
) -> tuple[float, ...] | None:
	if value is None:
		return None
	if isinstance(value, str):
		parts = [part.strip() for part in value.split(",") if part.strip()]
		vals = tuple(float(part) for part in parts)
	else:
		vals = tuple(float(part) for part in value)
	if len(vals) != len(strategy_space):
		raise ValueError("strategy_selection_strengths must match strategy_space length")
	for k in vals:
		if not isfinite(float(k)) or float(k) < 0.0:
			raise ValueError("strategy_selection_strengths must contain finite values >= 0")
	return vals


def _coerce_fixed_subgroup_weights(
	value: str | tuple[float, ...] | list[float] | None,
	*,
	strategy_space: list[str],
) -> tuple[float, ...] | None:
	if value is None:
		return None
	if isinstance(value, str):
		parts = [part.strip() for part in value.split(",") if part.strip()]
		vals = tuple(float(part) for part in parts)
	else:
		vals = tuple(float(part) for part in value)
	if len(vals) != len(strategy_space):
		raise ValueError("fixed_subgroup_weights must match strategy_space length")
	for w in vals:
		if not isfinite(float(w)) or float(w) <= 0.0:
			raise ValueError("fixed_subgroup_weights must contain finite values > 0")
	mean_w = sum(float(w) for w in vals) / float(len(vals))
	if mean_w <= 0.0:
		raise ValueError("fixed_subgroup_weights mean must be > 0")
	return tuple(float(w) / float(mean_w) for w in vals)


def _resolve_fixed_subgroup_count(*, n_players: int, fixed_subgroup_share: float) -> int:
	share = float(fixed_subgroup_share)
	if not isfinite(share) or not (0.0 <= share <= 1.0):
		raise ValueError("fixed_subgroup_share must be finite and lie in [0,1]")
	return int(round(float(n_players) * share))


def _resolve_hybrid_count(*, n_players: int, hybrid_update_share: float) -> int:
	share = float(hybrid_update_share)
	if not isfinite(share) or not (0.0 <= share <= 1.0):
		raise ValueError("hybrid_update_share must be finite and lie in [0,1]")
	return int(round(float(n_players) * share))


def _fixed_weights_dict(*, strategy_space: list[str], fixed_subgroup_weights: tuple[float, ...] | None) -> dict[str, float] | None:
	if fixed_subgroup_weights is None:
		return None
	return {
		s: float(w)
		for s, w in zip(strategy_space, fixed_subgroup_weights, strict=True)
	}


def _validate_fixed_subgroup_params(
	*,
	fixed_subgroup_share: float,
	fixed_subgroup_weights: tuple[float, ...] | None,
	fixed_subgroup_anchor_pull_strength: float = 1.0,
	fixed_subgroup_coupling_strength: float = 0.0,
	fixed_subgroup_bidirectional_coupling_strength: float = 0.0,
	fixed_subgroup_state_coupling_strength: float = 0.0,
	fixed_subgroup_state_coupling_beta: float = 8.0,
	fixed_subgroup_state_coupling_theta: float = 0.0,
	fixed_subgroup_state_signal: str = "gap_norm",
	evolution_mode: str = "sampled",
) -> None:
	share = float(fixed_subgroup_share)
	if share > 0.0 and fixed_subgroup_weights is None:
		raise ValueError("fixed_subgroup_weights are required when fixed_subgroup_share > 0")
	anchor_pull = float(fixed_subgroup_anchor_pull_strength)
	if not isfinite(anchor_pull) or not (0.0 <= anchor_pull <= 1.0):
		raise ValueError("fixed_subgroup_anchor_pull_strength must be finite and lie in [0,1]")
	coupling = float(fixed_subgroup_coupling_strength)
	if not isfinite(coupling) or coupling < 0.0:
		raise ValueError("fixed_subgroup_coupling_strength must be finite and >= 0")
	bidirectional_coupling = float(fixed_subgroup_bidirectional_coupling_strength)
	if not isfinite(bidirectional_coupling) or bidirectional_coupling < 0.0:
		raise ValueError("fixed_subgroup_bidirectional_coupling_strength must be finite and >= 0")
	state_coupling = float(fixed_subgroup_state_coupling_strength)
	if not isfinite(state_coupling) or state_coupling < 0.0:
		raise ValueError("fixed_subgroup_state_coupling_strength must be finite and >= 0")
	beta = float(fixed_subgroup_state_coupling_beta)
	if not isfinite(beta) or beta <= 0.0:
		raise ValueError("fixed_subgroup_state_coupling_beta must be finite and > 0")
	theta = float(fixed_subgroup_state_coupling_theta)
	if not isfinite(theta) or theta < 0.0:
		raise ValueError("fixed_subgroup_state_coupling_theta must be finite and >= 0")
	if str(fixed_subgroup_state_signal) != "gap_norm":
		raise ValueError("fixed_subgroup_state_signal must be 'gap_norm'")
	if anchor_pull < 1.0:
		if share <= 0.0 or fixed_subgroup_weights is None:
			raise ValueError("fixed_subgroup_share > 0 and fixed_subgroup_weights are required when fixed_subgroup_anchor_pull_strength < 1")
		if str(evolution_mode) == "mean_field":
			raise ValueError("fixed_subgroup_anchor_pull_strength < 1 is only supported for sampled player paths")
		if coupling > 0.0 or state_coupling > 0.0:
			raise ValueError("semi-frozen subgroup topology cannot be combined with H3.2 or H3.3B in the first version")
	if bidirectional_coupling > 0.0:
		if share <= 0.0 or fixed_subgroup_weights is None:
			raise ValueError("fixed_subgroup_share > 0 and fixed_subgroup_weights are required when fixed_subgroup_bidirectional_coupling_strength > 0")
		if str(evolution_mode) == "mean_field":
			raise ValueError("fixed_subgroup_bidirectional_coupling_strength is only supported for sampled player paths")
		if anchor_pull >= 1.0:
			raise ValueError("fixed_subgroup_bidirectional_coupling_strength requires fixed_subgroup_anchor_pull_strength < 1 for true bidirectional dynamics")
		if coupling > 0.0 or state_coupling > 0.0:
			raise ValueError("fixed_subgroup_bidirectional_coupling_strength cannot be enabled together with H3.2 or H3.3B")
	if coupling > 0.0:
		if share <= 0.0 or fixed_subgroup_weights is None:
			raise ValueError("fixed_subgroup_share > 0 and fixed_subgroup_weights are required when fixed_subgroup_coupling_strength > 0")
		if str(evolution_mode) == "mean_field":
			raise ValueError("fixed_subgroup_coupling_strength is only supported for sampled player paths")
	if state_coupling > 0.0:
		if share <= 0.0 or fixed_subgroup_weights is None:
			raise ValueError("fixed_subgroup_share > 0 and fixed_subgroup_weights are required when fixed_subgroup_state_coupling_strength > 0")
		if str(evolution_mode) == "mean_field":
			raise ValueError("fixed_subgroup_state_coupling_strength is only supported for sampled player paths")
		if coupling > 0.0 or bidirectional_coupling > 0.0:
			raise ValueError("fixed_subgroup_coupling_strength and fixed_subgroup_state_coupling_strength cannot be enabled together")


def _validate_hybrid_params(
	*,
	evolution_mode: str,
	popularity_mode: str,
	payoff_mode: str,
	hybrid_update_share: float,
	hybrid_inertia: float,
	fixed_subgroup_share: float,
	fixed_subgroup_coupling_strength: float,
	fixed_subgroup_bidirectional_coupling_strength: float,
	fixed_subgroup_state_coupling_strength: float,
) -> None:
	share = float(hybrid_update_share)
	if not isfinite(share) or not (0.0 <= share <= 1.0):
		raise ValueError("hybrid_update_share must be finite and lie in [0,1]")
	inertia = float(hybrid_inertia)
	if not isfinite(inertia) or not (0.0 <= inertia < 1.0):
		raise ValueError("hybrid_inertia must be finite and lie in [0,1)")
	if str(evolution_mode) != "hybrid":
		if inertia > 0.0:
			raise ValueError("hybrid_inertia > 0 requires evolution_mode='hybrid'")
		return
	if str(popularity_mode) != "sampled":
		raise ValueError("evolution_mode='hybrid' requires popularity_mode='sampled'")
	if str(payoff_mode) != "matrix_ab":
		raise ValueError("evolution_mode='hybrid' currently supports payoff_mode='matrix_ab' only")
	if float(fixed_subgroup_share) > 0.0:
		raise ValueError("evolution_mode='hybrid' cannot be combined with fixed subgroup mechanisms in the first version")
	if float(fixed_subgroup_coupling_strength) > 0.0:
		raise ValueError("evolution_mode='hybrid' cannot be combined with H3.2 in the first version")
	if float(fixed_subgroup_bidirectional_coupling_strength) > 0.0:
		raise ValueError("evolution_mode='hybrid' cannot be combined with H3.5 in the first version")
	if float(fixed_subgroup_state_coupling_strength) > 0.0:
		raise ValueError("evolution_mode='hybrid' cannot be combined with H3.3B in the first version")


def _validate_sampled_inertial_params(
	*,
	evolution_mode: str,
	popularity_mode: str,
	payoff_mode: str,
	sampled_inertia: float,
	hybrid_update_share: float,
	hybrid_inertia: float,
	fixed_subgroup_share: float,
	fixed_subgroup_weights: tuple[float, ...] | None,
	fixed_subgroup_anchor_pull_strength: float,
	fixed_subgroup_coupling_strength: float,
	fixed_subgroup_bidirectional_coupling_strength: float,
	fixed_subgroup_state_coupling_strength: float,
) -> None:
	inertia = float(sampled_inertia)
	if not isfinite(inertia) or not (0.0 <= inertia < 1.0):
		raise ValueError("sampled_inertia must be finite and lie in [0,1)")
	mode = str(evolution_mode)
	if mode == "sampled_inertial":
		if str(popularity_mode) != "sampled":
			raise ValueError("evolution_mode='sampled_inertial' requires popularity_mode='sampled'")
		if str(payoff_mode) != "matrix_ab":
			raise ValueError("evolution_mode='sampled_inertial' currently supports payoff_mode='matrix_ab' only")
		if float(hybrid_update_share) > 0.0 or float(hybrid_inertia) > 0.0:
			raise ValueError("evolution_mode='sampled_inertial' cannot be combined with H4 parameters")
		if float(fixed_subgroup_share) > 0.0 or fixed_subgroup_weights is not None:
			raise ValueError("evolution_mode='sampled_inertial' cannot be combined with H3 subgroup parameters")
		if float(fixed_subgroup_anchor_pull_strength) < 1.0:
			raise ValueError("evolution_mode='sampled_inertial' cannot be combined with H3 subgroup parameters")
		if float(fixed_subgroup_coupling_strength) > 0.0:
			raise ValueError("evolution_mode='sampled_inertial' cannot be combined with H3 subgroup parameters")
		if float(fixed_subgroup_bidirectional_coupling_strength) > 0.0:
			raise ValueError("evolution_mode='sampled_inertial' cannot be combined with H3 subgroup parameters")
		if float(fixed_subgroup_state_coupling_strength) > 0.0:
			raise ValueError("evolution_mode='sampled_inertial' cannot be combined with H3 subgroup parameters")
		return
	if mode == "mean_field":
		if inertia > 0.0 and str(payoff_mode) != "matrix_ab":
			raise ValueError("sampled_inertia > 0 in evolution_mode='mean_field' currently supports payoff_mode='matrix_ab' only")
		return
	if inertia > 0.0:
		raise ValueError("sampled_inertia > 0 requires evolution_mode in {'sampled_inertial','mean_field'}")


def _validate_personality_coupling_params(
	*,
	evolution_mode: str,
	popularity_mode: str,
	payoff_mode: str,
	personality_coupling_mu_base: float,
	personality_coupling_lambda_mu: float,
	personality_coupling_lambda_k: float,
	personality_coupling_beta_state_k: float,
	sampled_inertia: float,
	hybrid_update_share: float,
	hybrid_inertia: float,
	fixed_subgroup_share: float,
	fixed_subgroup_weights: tuple[float, ...] | None,
	fixed_subgroup_anchor_pull_strength: float,
	fixed_subgroup_coupling_strength: float,
	fixed_subgroup_bidirectional_coupling_strength: float,
	fixed_subgroup_state_coupling_strength: float,
) -> None:
	mu_base = float(personality_coupling_mu_base)
	lambda_mu = float(personality_coupling_lambda_mu)
	lambda_k = float(personality_coupling_lambda_k)
	beta_state_k = float(personality_coupling_beta_state_k)
	if not isfinite(mu_base) or not (0.0 <= mu_base < 1.0):
		raise ValueError("personality_coupling_mu_base must be finite and lie in [0,1)")
	if not isfinite(lambda_mu) or lambda_mu < 0.0:
		raise ValueError("personality_coupling_lambda_mu must be finite and >= 0")
	if not isfinite(lambda_k) or lambda_k < 0.0:
		raise ValueError("personality_coupling_lambda_k must be finite and >= 0")
	if not isfinite(beta_state_k) or not (0.0 <= beta_state_k <= 1.5):
		raise ValueError("personality_coupling_beta_state_k must be finite and lie in [0,1.5]")
	mode = str(evolution_mode)
	if mode != "personality_coupled":
		if lambda_mu > 0.0 or lambda_k > 0.0 or beta_state_k > 0.0:
			raise ValueError("personality coupling lambdas and beta_state_k require evolution_mode='personality_coupled'")
		return
	if str(popularity_mode) != "sampled":
		raise ValueError("evolution_mode='personality_coupled' requires popularity_mode='sampled'")
	if str(payoff_mode) != "matrix_ab":
		raise ValueError("evolution_mode='personality_coupled' currently supports payoff_mode='matrix_ab' only")
	if float(sampled_inertia) > 0.0:
		raise ValueError("evolution_mode='personality_coupled' cannot be combined with sampled_inertia")
	if float(hybrid_update_share) > 0.0 or float(hybrid_inertia) > 0.0:
		raise ValueError("evolution_mode='personality_coupled' cannot be combined with H4 parameters")
	if float(fixed_subgroup_share) > 0.0 or fixed_subgroup_weights is not None:
		raise ValueError("evolution_mode='personality_coupled' cannot be combined with H3 subgroup parameters")
	if float(fixed_subgroup_anchor_pull_strength) < 1.0:
		raise ValueError("evolution_mode='personality_coupled' cannot be combined with H3 subgroup parameters")
	if float(fixed_subgroup_coupling_strength) > 0.0:
		raise ValueError("evolution_mode='personality_coupled' cannot be combined with H3 subgroup parameters")
	if float(fixed_subgroup_bidirectional_coupling_strength) > 0.0:
		raise ValueError("evolution_mode='personality_coupled' cannot be combined with H3 subgroup parameters")
	if float(fixed_subgroup_state_coupling_strength) > 0.0:
		raise ValueError("evolution_mode='personality_coupled' cannot be combined with H3 subgroup parameters")


def _validate_stratified_growth_params(
	*,
	evolution_mode: str,
	popularity_mode: str,
	payoff_mode: str,
	sampled_growth_n_strata: int,
) -> None:
	strata = int(sampled_growth_n_strata)
	if strata not in {1, 3, 5, 10}:
		raise ValueError("sampled_growth_n_strata must be one of {1,3,5,10}")
	if strata == 1:
		return
	if str(evolution_mode) != "sampled":
		raise ValueError("sampled_growth_n_strata > 1 requires evolution_mode='sampled'")
	if str(popularity_mode) != "sampled":
		raise ValueError("sampled_growth_n_strata > 1 requires popularity_mode='sampled'")
	if str(payoff_mode) != "matrix_ab":
		raise ValueError("sampled_growth_n_strata > 1 currently supports payoff_mode='matrix_ab' only")


def _validate_tangential_drift_params(
	*,
	evolution_mode: str,
	popularity_mode: str,
	payoff_mode: str,
	sampled_growth_n_strata: int,
	sampled_inertia: float,
	tangential_drift_delta: float,
) -> None:
	delta = float(tangential_drift_delta)
	if not isfinite(delta) or delta < 0.0:
		raise ValueError("tangential_drift_delta must be finite and >= 0")
	if delta == 0.0:
		return
	mode = str(evolution_mode)
	if mode not in {"sampled", "mean_field"}:
		raise ValueError("tangential_drift_delta currently requires evolution_mode in {'sampled','mean_field'}")
	if str(payoff_mode) != "matrix_ab":
		raise ValueError("tangential_drift_delta currently supports payoff_mode='matrix_ab' only")
	if float(sampled_inertia) > 0.0:
		raise ValueError("tangential_drift_delta cannot be combined with sampled_inertia in the first-round runtime")
	if mode == "sampled":
		if str(popularity_mode) != "sampled":
			raise ValueError("tangential_drift_delta with evolution_mode='sampled' requires popularity_mode='sampled'")
		if int(sampled_growth_n_strata) != 1:
			raise ValueError("tangential_drift_delta cannot be combined with sampled_growth_n_strata > 1 in the first-round runtime")


def _validate_tangential_alpha_params(
	*,
	evolution_mode: str,
	popularity_mode: str,
	payoff_mode: str,
	tangential_alpha: float,
	tangential_drift_delta: float,
) -> None:
	alpha = float(tangential_alpha)
	if not isfinite(alpha) or alpha < 0.0:
		raise ValueError("tangential_alpha must be finite and >= 0")
	if alpha == 0.0:
		return
	if str(evolution_mode) != "sampled":
		raise ValueError("tangential_alpha currently requires evolution_mode='sampled'")
	if str(payoff_mode) != "matrix_ab":
		raise ValueError("tangential_alpha currently supports payoff_mode='matrix_ab' only")
	if str(popularity_mode) != "sampled":
		raise ValueError("tangential_alpha with evolution_mode='sampled' requires popularity_mode='sampled'")
	if float(tangential_drift_delta) > 0.0:
		raise ValueError("tangential_alpha cannot be combined with tangential_drift_delta > 0 (B1 and B5 are mutually exclusive)")


def _validate_async_update_params(
	*,
	evolution_mode: str,
	async_update_fraction: float,
	tangential_alpha: float,
	tangential_drift_delta: float,
) -> None:
	frac = float(async_update_fraction)
	if not isfinite(frac) or frac < 0.0 or frac > 1.0:
		raise ValueError("async_update_fraction must be finite and in [0, 1]")
	if frac >= 1.0:
		return
	if str(evolution_mode) != "sampled":
		raise ValueError("async_update_fraction < 1.0 requires evolution_mode='sampled'")
	if float(tangential_alpha) > 0.0:
		raise ValueError("async_update_fraction < 1.0 cannot be combined with tangential_alpha > 0 (A1 and B1 are mutually exclusive)")
	if float(tangential_drift_delta) > 0.0:
		raise ValueError("async_update_fraction < 1.0 cannot be combined with tangential_drift_delta > 0 (A1 and B5 are mutually exclusive)")


def _validate_sampling_beta_params(
	*,
	sampling_beta: float,
	async_update_fraction: float,
) -> None:
	beta = float(sampling_beta)
	if not isfinite(beta) or beta < 0.0:
		raise ValueError("sampling_beta must be finite and >= 0")
	if beta == 1.0:
		return
	if float(async_update_fraction) < 1.0:
		raise ValueError("sampling_beta != 1.0 cannot be combined with async_update_fraction < 1.0 (S1 and A1 are mutually exclusive)")


def _validate_mutation_params(
	*,
	mutation_rate: float,
	async_update_fraction: float,
) -> None:
	eta = float(mutation_rate)
	if not isfinite(eta) or eta < 0.0 or eta > 1.0:
		raise ValueError("mutation_rate must be finite and in [0, 1]")
	if eta == 0.0:
		return
	if float(async_update_fraction) < 1.0:
		raise ValueError("mutation_rate > 0 cannot be combined with async_update_fraction < 1.0 (M1 and A1 are mutually exclusive)")


def _validate_local_group_params(
	*,
	local_group_size: int,
	async_update_fraction: float,
	sampling_beta: float,
	mutation_rate: float,
	tangential_alpha: float,
) -> None:
	gs = int(local_group_size)
	if gs < 0:
		raise ValueError("local_group_size must be >= 0")
	if gs == 0:
		return
	if float(async_update_fraction) < 1.0:
		raise ValueError("local_group_size > 0 cannot be combined with async_update_fraction < 1.0 (L2 and A1 are mutually exclusive)")
	if float(sampling_beta) != 1.0:
		raise ValueError("local_group_size > 0 cannot be combined with sampling_beta != 1.0 (L2 and S1 are mutually exclusive)")
	if float(mutation_rate) > 0.0:
		raise ValueError("local_group_size > 0 cannot be combined with mutation_rate > 0 (L2 and M1 are mutually exclusive)")
	if float(tangential_alpha) > 0.0:
		raise ValueError("local_group_size > 0 cannot be combined with tangential_alpha > 0 (L2 and B1 are mutually exclusive)")


def _validate_h1_niche_params(
	*,
	payoff_niche_epsilon: float,
	niche_group_size: int,
	async_update_fraction: float,
	sampling_beta: float,
	mutation_rate: float,
	tangential_alpha: float,
	local_group_size: int,
) -> None:
	eps = float(payoff_niche_epsilon)
	ngs = int(niche_group_size)
	if eps < 0:
		raise ValueError("payoff_niche_epsilon must be >= 0")
	if ngs < 0:
		raise ValueError("niche_group_size must be >= 0")
	if eps > 0 and ngs <= 0:
		raise ValueError("payoff_niche_epsilon > 0 requires niche_group_size > 0 (H1 must be combined with per-group evolution)")
	if ngs > 0 and eps <= 0:
		raise ValueError("niche_group_size > 0 requires payoff_niche_epsilon > 0")
	if eps > 0 or ngs > 0:
		if float(async_update_fraction) < 1.0:
			raise ValueError("H1 (payoff_niche_epsilon > 0) cannot be combined with async_update_fraction < 1.0 (H1 and A1 are mutually exclusive)")
		if float(sampling_beta) != 1.0:
			raise ValueError("H1 (payoff_niche_epsilon > 0) cannot be combined with sampling_beta != 1.0 (H1 and S1 are mutually exclusive)")
		if float(mutation_rate) > 0.0:
			raise ValueError("H1 (payoff_niche_epsilon > 0) cannot be combined with mutation_rate > 0 (H1 and M1 are mutually exclusive)")
		if float(tangential_alpha) > 0.0:
			raise ValueError("H1 (payoff_niche_epsilon > 0) cannot be combined with tangential_alpha > 0 (H1 and B1 are mutually exclusive)")
		if int(local_group_size) > 0:
			raise ValueError("H1 (niche_group_size > 0) cannot be combined with local_group_size > 0 (H1 and L2 use separate group partitions)")


def _partition_players_into_groups(
	players: list[object],
	group_size: int,
) -> list[list[object]]:
	"""Partition players into fixed sub-groups of approximately *group_size*."""
	n = len(players)
	if group_size <= 0 or group_size >= n:
		return [list(players)]
	n_groups = max(1, n // group_size)
	base = n // n_groups
	remainder = n % n_groups
	groups: list[list[object]] = []
	idx = 0
	for g in range(n_groups):
		sz = base + (1 if g < remainder else 0)
		groups.append(players[idx:idx + sz])
		idx += sz
	return groups


def _apply_mutation_weights(
	weights: dict[str, float],
	strategy_space: list[str],
	rng: random.Random,
	eta: float,
) -> dict[str, float]:
	"""Mix current weights with Dirichlet(1,1,1) sample: w' = (1-eta)*w + eta*u."""
	from math import log as _log
	u = [-_log(max(1e-300, rng.random())) for _ in strategy_space]
	total = sum(u)
	u_norm = [v / total for v in u]
	return {
		s: (1.0 - eta) * float(weights.get(s, 1.0)) + eta * u_norm[j]
		for j, s in enumerate(strategy_space)
	}


def _resolved_fixed_player_weights(
	*,
	strategy_space: list[str],
	adaptive_weights: dict[str, float],
	fixed_weights: dict[str, float] | None,
	fixed_subgroup_anchor_pull_strength: float,
) -> dict[str, float]:
	if fixed_weights is None:
		return dict(adaptive_weights)
	return anchored_subgroup_weight_pull(
		strategy_space,
		adaptive_weights=adaptive_weights,
		anchor_weights=fixed_weights,
		anchor_pull_strength=float(fixed_subgroup_anchor_pull_strength),
	)


def _normalize_simplex(weights: dict[str, float]) -> dict[str, float]:
	total = sum(float(v) for v in weights.values())
	if total <= 0:
		n = len(weights)
		return {k: (1.0 / float(n)) for k in weights} if n else {}
	return {k: (float(v) / float(total)) for k, v in weights.items()}


def _mean_player_weights(players: list[object], *, strategy_space: list[str]) -> dict[str, float]:
	if not players:
		return {s: 1.0 for s in strategy_space}
	merged = {s: 0.0 for s in strategy_space}
	for pl in players:
		weights = getattr(pl, "strategy_weights", {})
		for s in strategy_space:
			merged[s] += float(weights.get(s, 1.0))
	den = float(len(players))
	return {s: (float(merged[s]) / den) for s in strategy_space}


def _personality_coupling_is_noop(cfg: SimConfig) -> bool:
	return (
		str(cfg.evolution_mode) == "personality_coupled"
		and float(cfg.personality_coupling_lambda_mu) == 0.0
		and float(cfg.personality_coupling_lambda_k) == 0.0
		and float(cfg.personality_coupling_mu_base) == 0.0
		and float(cfg.personality_coupling_beta_state_k) == 0.0
	)


def _current_state_dominance(distribution: dict[str, float]) -> float:
	if not distribution:
		return 1.0 / 3.0
	return max(1.0 / 3.0, min(1.0, max(float(value) for value in distribution.values())))


def _subgroup_strategy_distribution(
	players: list[object],
	*,
	strategy_space: list[str],
	fixed_subgroup: bool,
) -> dict[str, float]:
	counts = {s: 0.0 for s in strategy_space}
	total = 0.0
	for pl in players:
		if bool(getattr(pl, "fixed_subgroup", False)) != bool(fixed_subgroup):
			continue
		strategy = getattr(pl, "last_strategy", None)
		if strategy not in counts:
			continue
		counts[strategy] += 1.0
		total += 1.0
	if total <= 0.0:
		return {s: 0.0 for s in strategy_space}
	return {s: (float(counts[s]) / total) for s in strategy_space}


def _apply_fixed_subgroup_coupling(
	players: list[object],
	*,
	strategy_space: list[str],
	fixed_weights: dict[str, float] | None,
	fixed_subgroup_coupling_strength: float,
	a: float,
	b: float,
	step_records: list[dict[str, object]] | None = None,
) -> None:
	coupling = float(fixed_subgroup_coupling_strength)
	if coupling <= 0.0 or fixed_weights is None:
		return
	anchor_simplex = _normalize_simplex(fixed_weights)
	adaptive_simplex = _subgroup_strategy_distribution(
		players,
		strategy_space=strategy_space,
		fixed_subgroup=False,
	)
	if sum(float(v) for v in adaptive_simplex.values()) <= 0.0:
		return
	shift = anchored_subgroup_payoff_shift(
		strategy_space,
		a=float(a),
		b=float(b),
		anchor_simplex=anchor_simplex,
		adaptive_simplex=adaptive_simplex,
	)
	for idx, pl in enumerate(players):
		if bool(getattr(pl, "fixed_subgroup", False)):
			continue
		strategy = getattr(pl, "last_strategy", None)
		reward = getattr(pl, "last_reward", None)
		if strategy not in shift or reward is None:
			continue
		delta = coupling * float(shift[strategy])
		new_reward = float(reward) + float(delta)
		pl.last_reward = float(new_reward)
		pl.utility = float(getattr(pl, "utility", 0.0)) + float(delta)
		if step_records is not None and idx < len(step_records):
			step_records[idx]["reward"] = float(new_reward)


def _apply_fixed_subgroup_state_coupling(
	players: list[object],
	*,
	strategy_space: list[str],
	fixed_weights: dict[str, float] | None,
	fixed_subgroup_state_coupling_strength: float,
	fixed_subgroup_state_coupling_beta: float,
	fixed_subgroup_state_coupling_theta: float,
	fixed_subgroup_state_signal: str,
	a: float,
	b: float,
	step_records: list[dict[str, object]] | None = None,
) -> None:
	state_coupling = float(fixed_subgroup_state_coupling_strength)
	if state_coupling <= 0.0 or fixed_weights is None:
		return
	anchor_simplex = _normalize_simplex(fixed_weights)
	adaptive_simplex = _subgroup_strategy_distribution(
		players,
		strategy_space=strategy_space,
		fixed_subgroup=False,
	)
	if sum(float(v) for v in adaptive_simplex.values()) <= 0.0:
		return
	shift, _gap_norm, _gate = state_dependent_anchored_subgroup_payoff_shift(
		strategy_space,
		a=float(a),
		b=float(b),
		anchor_simplex=anchor_simplex,
		adaptive_simplex=adaptive_simplex,
		base_coupling_strength=state_coupling,
		beta=float(fixed_subgroup_state_coupling_beta),
		theta=float(fixed_subgroup_state_coupling_theta),
		signal=str(fixed_subgroup_state_signal),
	)
	for idx, pl in enumerate(players):
		if bool(getattr(pl, "fixed_subgroup", False)):
			continue
		strategy = getattr(pl, "last_strategy", None)
		reward = getattr(pl, "last_reward", None)
		if strategy not in shift or reward is None:
			continue
		delta = float(shift[strategy])
		new_reward = float(reward) + float(delta)
		pl.last_reward = float(new_reward)
		pl.utility = float(getattr(pl, "utility", 0.0)) + float(delta)
		if step_records is not None and idx < len(step_records):
			step_records[idx]["reward"] = float(new_reward)


def _apply_fixed_subgroup_bidirectional_coupling(
	players: list[object],
	*,
	strategy_space: list[str],
	fixed_subgroup_bidirectional_coupling_strength: float,
	a: float,
	b: float,
	step_records: list[dict[str, object]] | None = None,
) -> None:
	bidirectional_coupling = float(fixed_subgroup_bidirectional_coupling_strength)
	if bidirectional_coupling <= 0.0:
		return
	fixed_simplex = _subgroup_strategy_distribution(
		players,
		strategy_space=strategy_space,
		fixed_subgroup=True,
	)
	adaptive_simplex = _subgroup_strategy_distribution(
		players,
		strategy_space=strategy_space,
		fixed_subgroup=False,
	)
	if sum(float(v) for v in fixed_simplex.values()) <= 0.0:
		return
	if sum(float(v) for v in adaptive_simplex.values()) <= 0.0:
		return
	fixed_shift, adaptive_shift = bidirectional_subgroup_payoff_shifts(
		strategy_space,
		a=float(a),
		b=float(b),
		fixed_simplex=fixed_simplex,
		adaptive_simplex=adaptive_simplex,
		coupling_strength=bidirectional_coupling,
	)
	for idx, pl in enumerate(players):
		strategy = getattr(pl, "last_strategy", None)
		reward = getattr(pl, "last_reward", None)
		if reward is None:
			continue
		shift = fixed_shift if bool(getattr(pl, "fixed_subgroup", False)) else adaptive_shift
		if strategy not in shift:
			continue
		delta = float(shift[strategy])
		new_reward = float(reward) + float(delta)
		pl.last_reward = float(new_reward)
		pl.utility = float(getattr(pl, "utility", 0.0)) + float(delta)
		if step_records is not None and idx < len(step_records):
			step_records[idx]["reward"] = float(new_reward)


def _average_simplex_history(
	history: list[dict[str, float]],
	*,
	strategy_space: list[str],
	kernel: int,
	lag: int,
) -> dict[str, float]:
	"""Average a lagged simplex history using prefix semantics.

	If there is not enough history, average over the available prefix only.
	This matches the H1 spec: do not extrapolate negative-time states.
	"""
	if kernel <= 0 or (kernel % 2) == 0:
		raise ValueError("memory_kernel must be a positive odd integer")
	if lag < 0:
		raise ValueError("lag must be >= 0")
	if not history:
		return {s: 0.0 for s in strategy_space}
	end = len(history) - int(lag)
	if end <= 0:
		end = 1
	start = max(0, end - int(kernel))
	window = history[start:end]
	if not window:
		window = history[:1]
	denom = float(len(window))
	return {
		s: sum(float(state.get(s, 0.0)) for state in window) / denom
		for s in strategy_space
	}


def _matrix_ab_payoff_vec(
	*,
	strategy_space: list[str],
	a: float,
	b: float,
	matrix_cross_coupling: float,
	x: dict[str, float],
) -> dict[str, float]:
	"""Compute u = A x for the matrix_ab cyclic payoff.

	A = [[0, a, -b],
	     [-b, 0, a],
	     [a, -b, 0]]

	Optional cross term c_AD adds (-c*x_D, -c*x_A, c*(x_A+x_D)).
	"""
	if len(strategy_space) != 3:
		raise ValueError("matrix_ab requires exactly 3 strategies")
	x0 = float(x.get(strategy_space[0], 0.0))
	x1 = float(x.get(strategy_space[1], 0.0))
	x2 = float(x.get(strategy_space[2], 0.0))
	A = (
		(0.0, float(a), -float(b)),
		(-float(b), 0.0, float(a)),
		(float(a), -float(b), 0.0),
	)
	u0 = A[0][0] * x0 + A[0][1] * x1 + A[0][2] * x2
	u1 = A[1][0] * x0 + A[1][1] * x1 + A[1][2] * x2
	u2 = A[2][0] * x0 + A[2][1] * x1 + A[2][2] * x2
	c = float(matrix_cross_coupling)
	if c != 0.0:
		u0 += -c * x1
		u1 += -c * x0
		u2 += c * (x0 + x1)
	return {
		strategy_space[0]: float(u0),
		strategy_space[1]: float(u1),
		strategy_space[2]: float(u2),
	}


def _resolve_threshold_band(*, theta: float, theta_low: float | None, theta_high: float | None) -> tuple[float, float]:
	lo = float(theta) if theta_low is None else float(theta_low)
	hi = float(theta) if theta_high is None else float(theta_high)
	return lo, hi


def _validate_threshold_params(
	*,
	theta: float,
	theta_low: float | None,
	theta_high: float | None,
	trigger: str,
	state_alpha: float,
	a_hi: float | None,
	b_hi: float | None,
) -> None:
	if not isfinite(float(theta)) or not (0.0 <= float(theta) <= 1.0):
		raise ValueError("threshold_theta must be finite and lie in [0,1]")
	lo, hi = _resolve_threshold_band(theta=float(theta), theta_low=theta_low, theta_high=theta_high)
	if not isfinite(lo) or not (0.0 <= lo <= 1.0):
		raise ValueError("threshold_theta_low must be finite and lie in [0,1]")
	if not isfinite(hi) or not (0.0 <= hi <= 1.0):
		raise ValueError("threshold_theta_high must be finite and lie in [0,1]")
	if lo > hi:
		raise ValueError("threshold_theta_low must be <= threshold_theta_high")
	if str(trigger) not in {"ad_share", "ad_product"}:
		raise ValueError("threshold_trigger must be one of {'ad_share', 'ad_product'}")
	if not isfinite(float(state_alpha)) or not (0.0 < float(state_alpha) <= 1.0):
		raise ValueError("threshold_state_alpha must be finite and lie in (0,1]")
	if a_hi is not None and not isfinite(float(a_hi)):
		raise ValueError("threshold_a_hi must be finite")
	if b_hi is not None and not isfinite(float(b_hi)):
		raise ValueError("threshold_b_hi must be finite")


def _threshold_trigger_value(*, trigger: str, strategy_space: list[str], x: dict[str, float]) -> float:
	x_a = float(x.get(strategy_space[0], 0.0))
	x_d = float(x.get(strategy_space[1], 0.0))
	if str(trigger) == "ad_product":
		return float(4.0 * x_a * x_d)
	return float(x_a + x_d)


def _next_threshold_state_value(*, trigger_value: float, state_alpha: float, current_state_value: float | None) -> float:
	alpha = float(state_alpha)
	if current_state_value is None or alpha >= 1.0:
		return float(trigger_value)
	return alpha * float(trigger_value) + (1.0 - alpha) * float(current_state_value)


def _next_threshold_regime(
	*,
	q_ad: float,
	theta: float,
	theta_low: float | None,
	theta_high: float | None,
	current_regime_hi: bool | None,
) -> bool:
	lo, hi = _resolve_threshold_band(theta=float(theta), theta_low=theta_low, theta_high=theta_high)
	if float(q_ad) >= hi:
		return True
	if float(q_ad) <= lo:
		return False
	if current_regime_hi is None:
		return False
	return bool(current_regime_hi)


def _threshold_ab_payoff_vec(
	*,
	strategy_space: list[str],
	a: float,
	b: float,
	threshold_theta: float,
	threshold_theta_low: float | None,
	threshold_theta_high: float | None,
	threshold_trigger: str,
	threshold_state_alpha: float,
	threshold_a_hi: float | None,
	threshold_b_hi: float | None,
	matrix_cross_coupling: float,
	x: dict[str, float],
	current_regime_hi: bool | None = None,
	current_state_value: float | None = None,
) -> tuple[dict[str, float], bool, float]:
	if len(strategy_space) != 3:
		raise ValueError("threshold_ab requires exactly 3 strategies")
	_validate_threshold_params(
		theta=float(threshold_theta),
		theta_low=threshold_theta_low,
		theta_high=threshold_theta_high,
		trigger=threshold_trigger,
		state_alpha=threshold_state_alpha,
		a_hi=threshold_a_hi,
		b_hi=threshold_b_hi,
	)
	trigger_value = _threshold_trigger_value(trigger=threshold_trigger, strategy_space=strategy_space, x=x)
	state_value = _next_threshold_state_value(
		trigger_value=trigger_value,
		state_alpha=threshold_state_alpha,
		current_state_value=current_state_value,
	)
	regime_hi = _next_threshold_regime(
		q_ad=state_value,
		theta=float(threshold_theta),
		theta_low=threshold_theta_low,
		theta_high=threshold_theta_high,
		current_regime_hi=current_regime_hi,
	)
	a_eff = float(threshold_a_hi) if (regime_hi and threshold_a_hi is not None) else float(a)
	b_eff = float(threshold_b_hi) if (regime_hi and threshold_b_hi is not None) else float(b)
	return (
		_matrix_ab_payoff_vec(
		strategy_space=strategy_space,
		a=a_eff,
		b=b_eff,
		matrix_cross_coupling=matrix_cross_coupling,
		x=x,
		),
		bool(regime_hi),
		float(state_value),
	)


def _payoff_vec_from_state(
	*,
	payoff_mode: str,
	strategy_space: list[str],
	a: float,
	b: float,
	threshold_theta: float,
	threshold_theta_low: float | None,
	threshold_theta_high: float | None,
	threshold_trigger: str,
	threshold_state_alpha: float,
	threshold_a_hi: float | None,
	threshold_b_hi: float | None,
	matrix_cross_coupling: float,
	x: dict[str, float],
	current_regime_hi: bool | None = None,
	current_state_value: float | None = None,
) -> tuple[dict[str, float], bool | None, float | None]:
	if str(payoff_mode) == "matrix_ab":
		return (
			_matrix_ab_payoff_vec(
			strategy_space=strategy_space,
			a=a,
			b=b,
			matrix_cross_coupling=matrix_cross_coupling,
			x=x,
			),
			current_regime_hi,
			current_state_value,
		)
	if str(payoff_mode) == "threshold_ab":
		return _threshold_ab_payoff_vec(
			strategy_space=strategy_space,
			a=a,
			b=b,
			threshold_theta=threshold_theta,
			threshold_theta_low=threshold_theta_low,
			threshold_theta_high=threshold_theta_high,
			threshold_trigger=threshold_trigger,
			threshold_state_alpha=threshold_state_alpha,
			threshold_a_hi=threshold_a_hi,
			threshold_b_hi=threshold_b_hi,
			matrix_cross_coupling=matrix_cross_coupling,
			x=x,
			current_regime_hi=current_regime_hi,
			current_state_value=current_state_value,
		)
	raise ValueError(f"Unsupported payoff_mode for matrix-like evolution: {payoff_mode!r}")


def _write_timeseries_csv(
	out_csv: Path,
	*,
	strategy_space: list[str],
	rows: list[dict],
) -> None:
	out_csv.parent.mkdir(parents=True, exist_ok=True)
	fieldnames = (
		["round", "avg_reward", "avg_utility"]
		+ [f"p_{s}" for s in strategy_space]
		+ [f"w_{s}" for s in strategy_space]
		+ [
			"threshold_regime_hi",
			"threshold_state_value",
			"event_count",
			"success_count",
			"event_types_json",
			"event_ids_json",
			"action_names_json",
			"result_kinds_json",
			"successes_json",
			"final_risks_json",
			"success_probs_json",
			"trait_deltas_json",
			"trait_deltas_per_event_json",
			"popularity_shift_json",
			"state_effects_json",
		]
	)
	with out_csv.open("w", newline="") as f:
		writer = csv.DictWriter(f, fieldnames=fieldnames)
		writer.writeheader()
		writer.writerows(rows)


def _aggregate_event_records(step_records: list[dict]) -> dict[str, object]:
	event_records = [record.get("event_result") for record in step_records if record.get("event_result")]
	if not event_records:
		return {
			"event_count": 0,
			"success_count": 0,
			"event_types_json": "[]",
			"event_ids_json": "[]",
			"action_names_json": "[]",
			"result_kinds_json": "[]",
			"successes_json": "[]",
			"final_risks_json": "[]",
			"success_probs_json": "[]",
			"trait_deltas_json": "{}",
			"trait_deltas_per_event_json": "{}",
			"popularity_shift_json": "{}",
			"state_effects_json": "{}",
		}

	def _sum_dicts(items: list[dict[str, float]]) -> dict[str, float]:
		merged: dict[str, float] = {}
		for item in items:
			for key, value in item.items():
				merged[key] = float(merged.get(key, 0.0)) + float(value)
		return merged

	trait_deltas = _sum_dicts([dict(record.get("trait_deltas", {})) for record in event_records])
	popularity_shift = _sum_dicts([dict(record.get("popularity_shift", {})) for record in event_records])
	state_effects = _sum_dicts([dict(record.get("state_effects", {})) for record in event_records])
	trait_deltas_per_event: dict[str, dict[str, float]] = {}
	for record in event_records:
		event_id = str(record.get("event_id"))
		bucket = trait_deltas_per_event.setdefault(event_id, {})
		for key, value in dict(record.get("trait_deltas", {})).items():
			bucket[str(key)] = float(bucket.get(str(key), 0.0)) + float(value)

	return {
		"event_count": len(event_records),
		"success_count": sum(1 for record in event_records if bool(record.get("success"))),
		"event_types_json": json.dumps([record.get("event_type") for record in event_records], sort_keys=True),
		"event_ids_json": json.dumps([record.get("event_id") for record in event_records], sort_keys=True),
		"action_names_json": json.dumps([record.get("action_name") for record in event_records], sort_keys=True),
		"result_kinds_json": json.dumps([record.get("result_kind") for record in event_records], sort_keys=True),
		"successes_json": json.dumps([bool(record.get("success")) for record in event_records]),
		"final_risks_json": json.dumps([float(record.get("final_risk", 0.0)) for record in event_records]),
		"success_probs_json": json.dumps([float(record.get("success_prob", 0.0)) for record in event_records]),
		"trait_deltas_json": json.dumps(trait_deltas, sort_keys=True),
		"trait_deltas_per_event_json": json.dumps(trait_deltas_per_event, sort_keys=True),
		"popularity_shift_json": json.dumps(popularity_shift, sort_keys=True),
		"state_effects_json": json.dumps(state_effects, sort_keys=True),
	}


def simulate(
	cfg: SimConfig,
	*,
	player_setup_callback: PlayerSetupCallback | None = None,
	round_callback: RoundCallback | None = None,
) -> tuple[list[str], list[dict]]:
	"""Run a simulation and return (strategy_space, rows) without any I/O.

	This is the preferred entry point for programmatic experiments (e.g. multi-seed stability)
	while keeping the CLI `main()` as a thin wrapper.
	"""

	strategy_space = ["aggressive", "defensive", "balanced"]
	strategy_selection_strengths = _coerce_strategy_selection_strengths(
		cfg.strategy_selection_strengths,
		strategy_space=strategy_space,
	)
	fixed_subgroup_weights = _coerce_fixed_subgroup_weights(
		cfg.fixed_subgroup_weights,
		strategy_space=strategy_space,
	)
	_validate_fixed_subgroup_params(
		fixed_subgroup_share=float(cfg.fixed_subgroup_share),
		fixed_subgroup_weights=fixed_subgroup_weights,
		fixed_subgroup_anchor_pull_strength=float(cfg.fixed_subgroup_anchor_pull_strength),
		fixed_subgroup_coupling_strength=float(cfg.fixed_subgroup_coupling_strength),
		fixed_subgroup_bidirectional_coupling_strength=float(cfg.fixed_subgroup_bidirectional_coupling_strength),
		fixed_subgroup_state_coupling_strength=float(cfg.fixed_subgroup_state_coupling_strength),
		fixed_subgroup_state_coupling_beta=float(cfg.fixed_subgroup_state_coupling_beta),
		fixed_subgroup_state_coupling_theta=float(cfg.fixed_subgroup_state_coupling_theta),
		fixed_subgroup_state_signal=str(cfg.fixed_subgroup_state_signal),
		evolution_mode=str(cfg.evolution_mode),
	)
	_validate_hybrid_params(
		evolution_mode=str(cfg.evolution_mode),
		popularity_mode=str(cfg.popularity_mode),
		payoff_mode=str(cfg.payoff_mode),
		hybrid_update_share=float(cfg.hybrid_update_share),
		hybrid_inertia=float(cfg.hybrid_inertia),
		fixed_subgroup_share=float(cfg.fixed_subgroup_share),
		fixed_subgroup_coupling_strength=float(cfg.fixed_subgroup_coupling_strength),
		fixed_subgroup_bidirectional_coupling_strength=float(cfg.fixed_subgroup_bidirectional_coupling_strength),
		fixed_subgroup_state_coupling_strength=float(cfg.fixed_subgroup_state_coupling_strength),
	)
	_validate_sampled_inertial_params(
		evolution_mode=str(cfg.evolution_mode),
		popularity_mode=str(cfg.popularity_mode),
		payoff_mode=str(cfg.payoff_mode),
		sampled_inertia=float(cfg.sampled_inertia),
		hybrid_update_share=float(cfg.hybrid_update_share),
		hybrid_inertia=float(cfg.hybrid_inertia),
		fixed_subgroup_share=float(cfg.fixed_subgroup_share),
		fixed_subgroup_weights=fixed_subgroup_weights,
		fixed_subgroup_anchor_pull_strength=float(cfg.fixed_subgroup_anchor_pull_strength),
		fixed_subgroup_coupling_strength=float(cfg.fixed_subgroup_coupling_strength),
		fixed_subgroup_bidirectional_coupling_strength=float(cfg.fixed_subgroup_bidirectional_coupling_strength),
		fixed_subgroup_state_coupling_strength=float(cfg.fixed_subgroup_state_coupling_strength),
	)
	_validate_personality_coupling_params(
		evolution_mode=str(cfg.evolution_mode),
		popularity_mode=str(cfg.popularity_mode),
		payoff_mode=str(cfg.payoff_mode),
		personality_coupling_mu_base=float(cfg.personality_coupling_mu_base),
		personality_coupling_lambda_mu=float(cfg.personality_coupling_lambda_mu),
		personality_coupling_lambda_k=float(cfg.personality_coupling_lambda_k),
		personality_coupling_beta_state_k=float(cfg.personality_coupling_beta_state_k),
		sampled_inertia=float(cfg.sampled_inertia),
		hybrid_update_share=float(cfg.hybrid_update_share),
		hybrid_inertia=float(cfg.hybrid_inertia),
		fixed_subgroup_share=float(cfg.fixed_subgroup_share),
		fixed_subgroup_weights=fixed_subgroup_weights,
		fixed_subgroup_anchor_pull_strength=float(cfg.fixed_subgroup_anchor_pull_strength),
		fixed_subgroup_coupling_strength=float(cfg.fixed_subgroup_coupling_strength),
		fixed_subgroup_bidirectional_coupling_strength=float(cfg.fixed_subgroup_bidirectional_coupling_strength),
		fixed_subgroup_state_coupling_strength=float(cfg.fixed_subgroup_state_coupling_strength),
	)
	_validate_stratified_growth_params(
		evolution_mode=str(cfg.evolution_mode),
		popularity_mode=str(cfg.popularity_mode),
		payoff_mode=str(cfg.payoff_mode),
		sampled_growth_n_strata=int(cfg.sampled_growth_n_strata),
	)
	_validate_tangential_drift_params(
		evolution_mode=str(cfg.evolution_mode),
		popularity_mode=str(cfg.popularity_mode),
		payoff_mode=str(cfg.payoff_mode),
		sampled_growth_n_strata=int(cfg.sampled_growth_n_strata),
		sampled_inertia=float(cfg.sampled_inertia),
		tangential_drift_delta=float(cfg.tangential_drift_delta),
	)
	_validate_tangential_alpha_params(
		evolution_mode=str(cfg.evolution_mode),
		popularity_mode=str(cfg.popularity_mode),
		payoff_mode=str(cfg.payoff_mode),
		tangential_alpha=float(cfg.tangential_alpha),
		tangential_drift_delta=float(cfg.tangential_drift_delta),
	)
	_validate_async_update_params(
		evolution_mode=str(cfg.evolution_mode),
		async_update_fraction=float(cfg.async_update_fraction),
		tangential_alpha=float(cfg.tangential_alpha),
		tangential_drift_delta=float(cfg.tangential_drift_delta),
	)
	_validate_sampling_beta_params(
		sampling_beta=float(cfg.sampling_beta),
		async_update_fraction=float(cfg.async_update_fraction),
	)
	_validate_mutation_params(
		mutation_rate=float(cfg.mutation_rate),
		async_update_fraction=float(cfg.async_update_fraction),
	)
	_validate_local_group_params(
		local_group_size=int(cfg.local_group_size),
		async_update_fraction=float(cfg.async_update_fraction),
		sampling_beta=float(cfg.sampling_beta),
		mutation_rate=float(cfg.mutation_rate),
		tangential_alpha=float(cfg.tangential_alpha),
	)
	_validate_h1_niche_params(
		payoff_niche_epsilon=float(cfg.payoff_niche_epsilon),
		niche_group_size=int(cfg.niche_group_size),
		async_update_fraction=float(cfg.async_update_fraction),
		sampling_beta=float(cfg.sampling_beta),
		mutation_rate=float(cfg.mutation_rate),
		tangential_alpha=float(cfg.tangential_alpha),
		local_group_size=int(cfg.local_group_size),
	)
	_validate_personality_coupling_params(
		evolution_mode=str(cfg.evolution_mode),
		popularity_mode=str(cfg.popularity_mode),
		payoff_mode=str(cfg.payoff_mode),
		personality_coupling_mu_base=float(cfg.personality_coupling_mu_base),
		personality_coupling_lambda_mu=float(cfg.personality_coupling_lambda_mu),
		personality_coupling_lambda_k=float(cfg.personality_coupling_lambda_k),
		personality_coupling_beta_state_k=float(cfg.personality_coupling_beta_state_k),
		sampled_inertia=float(cfg.sampled_inertia),
		hybrid_update_share=float(cfg.hybrid_update_share),
		hybrid_inertia=float(cfg.hybrid_inertia),
		fixed_subgroup_share=float(cfg.fixed_subgroup_share),
		fixed_subgroup_weights=fixed_subgroup_weights,
		fixed_subgroup_anchor_pull_strength=float(cfg.fixed_subgroup_anchor_pull_strength),
		fixed_subgroup_coupling_strength=float(cfg.fixed_subgroup_coupling_strength),
		fixed_subgroup_bidirectional_coupling_strength=float(cfg.fixed_subgroup_bidirectional_coupling_strength),
		fixed_subgroup_state_coupling_strength=float(cfg.fixed_subgroup_state_coupling_strength),
	)
	if str(cfg.evolution_mode) == "mean_field":
		if str(cfg.payoff_mode) not in ("matrix_ab", "threshold_ab"):
			raise ValueError("evolution_mode='mean_field' currently supports payoff_mode in {'matrix_ab','threshold_ab'} only")
		if int(cfg.payoff_lag) not in (0, 1):
			raise ValueError("payoff_lag must be 0 or 1")
		if int(cfg.memory_kernel) not in (1, 3, 5):
			raise ValueError("memory_kernel must be one of {1,3,5}")
		if str(cfg.payoff_mode) == "threshold_ab":
			_validate_threshold_params(
				theta=float(cfg.threshold_theta),
				theta_low=cfg.threshold_theta_low,
				theta_high=cfg.threshold_theta_high,
				trigger=str(cfg.threshold_trigger),
				state_alpha=float(cfg.threshold_state_alpha),
				a_hi=cfg.threshold_a_hi,
				b_hi=cfg.threshold_b_hi,
			)
		k = float(cfg.selection_strength)
		w_cur = _initial_weights(strategy_space=strategy_space, init_bias=float(cfg.init_bias))
		# Normalize weights to mean=1 (same convention as replicator_step output).
		m0 = (sum(w_cur.values()) / float(len(w_cur))) if w_cur else 1.0
		w_cur = {s: (float(v) / float(m0)) for s, v in w_cur.items()}
		x_cur = _normalize_simplex(w_cur)
		x_history: list[dict[str, float]] = [dict(x_cur)]
		threshold_regime_hi: bool | None = None
		threshold_state_value: float | None = None
		rows: list[dict] = []
		for t in range(int(cfg.n_rounds)):
			x_pay = _average_simplex_history(
				x_history,
				strategy_space=strategy_space,
				kernel=int(cfg.memory_kernel),
				lag=int(cfg.payoff_lag),
			)
			u, threshold_regime_hi, threshold_state_value = _payoff_vec_from_state(
				payoff_mode=str(cfg.payoff_mode),
				strategy_space=strategy_space,
				a=float(cfg.a),
				b=float(cfg.b),
				threshold_theta=float(cfg.threshold_theta),
				threshold_theta_low=cfg.threshold_theta_low,
				threshold_theta_high=cfg.threshold_theta_high,
				threshold_trigger=str(cfg.threshold_trigger),
				threshold_state_alpha=float(cfg.threshold_state_alpha),
				threshold_a_hi=cfg.threshold_a_hi,
				threshold_b_hi=cfg.threshold_b_hi,
				matrix_cross_coupling=float(cfg.matrix_cross_coupling),
				x=x_pay,
				current_regime_hi=threshold_regime_hi,
				current_state_value=threshold_state_value,
			)
			if float(cfg.sampled_inertia) > 0.0:
				w_cur, mean_field_velocity = inertial_deterministic_replicator_step(
					w_cur,
					strategy_space,
					payoff_vector=u,
					previous_velocity=mean_field_velocity,
					inertia=float(cfg.sampled_inertia),
					selection_strength=k,
				)
			else:
				w_cur = deterministic_replicator_step(
					w_cur,
					strategy_space,
					payoff_vector=u,
					selection_strength=k,
					tangential_drift_simplex=x_cur,
					tangential_drift_delta=float(cfg.tangential_drift_delta),
				)
			x_next = _normalize_simplex(w_cur)
			row = {
				"round": t,
				"avg_reward": "",
				"avg_utility": 0.0,
				"threshold_regime_hi": "" if threshold_regime_hi is None else int(bool(threshold_regime_hi)),
				"threshold_state_value": "" if threshold_state_value is None else float(threshold_state_value),
			}
			for s in strategy_space:
				row[f"p_{s}"] = float(x_next[s])
			for s in strategy_space:
				row[f"w_{s}"] = float(w_cur[s])
			rows.append(row)
			x_cur = x_next
			x_history.append(dict(x_cur))
		return strategy_space, rows

	# Deterministic sampling: give each player an independent RNG stream.
	# This avoids accidental coupling via a single shared RNG and makes runs reproducible.
	if cfg.seed is None:
		players = [BasePlayer(strategy_space) for _ in range(cfg.n_players)]
	else:
		players = [
			BasePlayer(strategy_space, rng=random.Random(int(cfg.seed) + i))
			for i in range(cfg.n_players)
		]
	fixed_n = _resolve_fixed_subgroup_count(n_players=int(cfg.n_players), fixed_subgroup_share=float(cfg.fixed_subgroup_share))
	hybrid_n = _resolve_hybrid_count(n_players=int(cfg.n_players), hybrid_update_share=float(cfg.hybrid_update_share))
	fixed_weights = _fixed_weights_dict(strategy_space=strategy_space, fixed_subgroup_weights=fixed_subgroup_weights)
	for idx, pl in enumerate(players):
		is_fixed = bool(idx < fixed_n and fixed_weights is not None)
		setattr(pl, "stratum", int(idx % max(1, int(cfg.sampled_growth_n_strata))))
		setattr(pl, "fixed_subgroup", is_fixed)
		is_hybrid = bool(str(cfg.evolution_mode) == "hybrid" and idx < hybrid_n)
		setattr(pl, "hybrid_update", is_hybrid)
		if is_hybrid:
			setattr(pl, "hybrid_velocity", {s: 0.0 for s in strategy_space})
		if str(cfg.evolution_mode) == "sampled_inertial" and not is_fixed:
			setattr(pl, "sampled_velocity", {s: 0.0 for s in strategy_space})
		if str(cfg.evolution_mode) == "personality_coupled" and not is_fixed:
			setattr(pl, "personality_coupled_velocity", {s: 0.0 for s in strategy_space})
		if str(cfg.evolution_mode) == "personality_coupled" and not is_fixed:
			setattr(pl, "personality_coupled_velocity", {s: 0.0 for s in strategy_space})
		if float(cfg.sampling_beta) != 1.0:
			setattr(pl, "sampling_beta", float(cfg.sampling_beta))

	# M1: per-player mutation RNGs (seeded deterministically from cfg.seed)
	mutation_rngs: list[random.Random] | None = None
	if float(cfg.mutation_rate) > 0.0:
		base_seed = int(cfg.seed or 0)
		mutation_rngs = [random.Random(base_seed + 900000 + i) for i in range(int(cfg.n_players))]

	# L2: partition players into fixed local sub-groups
	local_groups: list[list[object]] | None = None
	l2_gs = int(cfg.local_group_size)
	if l2_gs > 0 and l2_gs < int(cfg.n_players):
		local_groups = _partition_players_into_groups(players, l2_gs)

	# H1: partition players into niche sub-groups (separate from L2)
	h1_groups: list[list[object]] | None = None
	h1_epsilon = float(cfg.payoff_niche_epsilon)
	h1_ngs = int(cfg.niche_group_size)
	if h1_epsilon > 0 and h1_ngs > 0 and h1_ngs < int(cfg.n_players):
		h1_groups = _partition_players_into_groups(players, h1_ngs)

	event_loader = None
	if bool(cfg.enable_events):
		events_json = cfg.events_json if cfg.events_json is not None else DEFAULT_EVENTS_JSON
		event_loader = EventLoader(
			events_json,
			apply_trait_deltas=bool(cfg.apply_event_trait_deltas),
			failure_threshold_override=cfg.event_failure_threshold,
			health_penalty_coefficient=(
				float(cfg.event_health_penalty)
				if cfg.event_health_penalty is not None
				else 0.10
			),
			stress_risk_coefficient=(
				float(cfg.event_stress_risk_coefficient)
				if cfg.event_stress_risk_coefficient is not None
				else 0.10
			),
			risk_ma_alpha=(float(cfg.risk_ma_alpha) if cfg.risk_ma_alpha is not None else 0.0),
			risk_ma_multiplier=(float(cfg.risk_ma_multiplier) if cfg.risk_ma_multiplier is not None else 0.0),
			stress_decay_c=(float(cfg.stress_decay_c) if cfg.stress_decay_c is not None else 0.0),
			stress_decay_beta=(float(cfg.stress_decay_beta) if cfg.stress_decay_beta is not None else 2.0),
		)

	# Symmetry breaking: start away from the uniform equilibrium.
	init_w = _initial_weights(strategy_space=strategy_space, init_bias=float(cfg.init_bias))
	if init_w:
		for pl in players:
			if bool(getattr(pl, "fixed_subgroup", False)) and fixed_weights is not None:
				pl.update_weights(
					_resolved_fixed_player_weights(
						strategy_space=strategy_space,
						adaptive_weights=init_w,
						fixed_weights=fixed_weights,
						fixed_subgroup_anchor_pull_strength=float(cfg.fixed_subgroup_anchor_pull_strength),
					)
				)
			else:
				pl.update_weights(init_w)
	if player_setup_callback is not None:
		player_setup_callback(players, strategy_space, cfg)

	dungeon = DungeonAI(
		payoff_mode=cfg.payoff_mode,
		gamma=cfg.gamma,
		epsilon=cfg.epsilon,
		a=cfg.a,
		b=cfg.b,
		matrix_cross_coupling=cfg.matrix_cross_coupling,
		memory_kernel=cfg.memory_kernel,
		threshold_theta=cfg.threshold_theta,
		threshold_theta_low=cfg.threshold_theta_low,
		threshold_theta_high=cfg.threshold_theta_high,
		threshold_trigger=cfg.threshold_trigger,
		threshold_state_alpha=cfg.threshold_state_alpha,
		threshold_a_hi=cfg.threshold_a_hi,
		threshold_b_hi=cfg.threshold_b_hi,
		strategy_cycle=strategy_space,
		event_loader=event_loader,
		event_rng=random.Random(cfg.seed) if cfg.seed is not None else random.Random(),
	)
	engine = GameEngine(players, dungeon, popularity_mode=str(cfg.popularity_mode))
	if str(cfg.popularity_mode) == "expected":
		# Without this, round-0 popularity is empty -> identical rewards -> no movement.
		dungeon.set_popularity(dict(init_w))

	rows: list[dict] = []
	new_weights = {s: 1.0 for s in strategy_space}
	x_history: list[dict[str, float]] = [
		_normalize_simplex(_mean_player_weights(players, strategy_space=strategy_space))
	]

	for t in range(cfg.n_rounds):
		step_records = engine.step()
		_apply_fixed_subgroup_coupling(
			players,
			strategy_space=strategy_space,
			fixed_weights=fixed_weights,
			fixed_subgroup_coupling_strength=float(cfg.fixed_subgroup_coupling_strength),
			a=float(dungeon.a),
			b=float(dungeon.b),
			step_records=step_records,
		)
		_apply_fixed_subgroup_bidirectional_coupling(
			players,
			strategy_space=strategy_space,
			fixed_subgroup_bidirectional_coupling_strength=float(cfg.fixed_subgroup_bidirectional_coupling_strength),
			a=float(dungeon.a),
			b=float(dungeon.b),
			step_records=step_records,
		)
		_apply_fixed_subgroup_state_coupling(
			players,
			strategy_space=strategy_space,
			fixed_weights=fixed_weights,
			fixed_subgroup_state_coupling_strength=float(cfg.fixed_subgroup_state_coupling_strength),
			fixed_subgroup_state_coupling_beta=float(cfg.fixed_subgroup_state_coupling_beta),
			fixed_subgroup_state_coupling_theta=float(cfg.fixed_subgroup_state_coupling_theta),
			fixed_subgroup_state_signal=str(cfg.fixed_subgroup_state_signal),
			a=float(dungeon.a),
			b=float(dungeon.b),
			step_records=step_records,
		)

		# H1: per-group payoff niche bonus
		if h1_groups is not None and h1_epsilon > 0:
			for gk, group in enumerate(h1_groups):
				niche_strategy = strategy_space[gk % len(strategy_space)]
				for pl in group:
					if getattr(pl, "last_strategy", None) == niche_strategy:
						pl.last_reward = float(pl.last_reward) + h1_epsilon

		dist = strategy_distribution(players, strategy_space)
		avg_u = average_utility(players)
		avg_r = average_reward(players)
		x_pay = _average_simplex_history(
			x_history,
			strategy_space=strategy_space,
			kernel=int(cfg.memory_kernel),
			lag=int(cfg.payoff_lag),
		)
		hybrid_payoff = _matrix_ab_payoff_vec(
			strategy_space=strategy_space,
			a=float(cfg.a),
			b=float(cfg.b),
			matrix_cross_coupling=float(cfg.matrix_cross_coupling),
			x=x_pay,
		)

		# 演化更新：根據每一輪的 last_reward 來調整下一輪抽樣權重
		if float(cfg.async_update_fraction) < 1.0:
			new_weights, _a1_diag = async_replicator_step(
				players,
				strategy_space,
				selection_strength=cfg.selection_strength,
				async_update_fraction=float(cfg.async_update_fraction),
				seed=int(cfg.seed or 0),
				round_index=t,
			)
		elif str(cfg.evolution_mode) == "sampled_inertial":
			new_weights, sampled_velocity = inertial_sampled_replicator_step(
				players,
				strategy_space,
				previous_velocity=getattr(players[0], "sampled_velocity", {}) if players else None,
				inertia=float(cfg.sampled_inertia),
				selection_strength=cfg.selection_strength,
			)
		elif str(cfg.evolution_mode) == "personality_coupled":
			if _personality_coupling_is_noop(cfg):
				new_weights = replicator_step(
					players,
					strategy_space,
					selection_strength=cfg.selection_strength,
					tangential_drift_simplex=dist,
					tangential_drift_delta=float(cfg.tangential_drift_delta),
				)
			else:
				growth_vector = sampled_growth_vector(players, strategy_space)
				state_dominance = _current_state_dominance(dist)
				state_factor = personality_state_k_factor(
					state_dominance=state_dominance,
					beta_state_k=float(cfg.personality_coupling_beta_state_k),
				)
				for pl in players:
					params = resolve_personality_coupling(
						getattr(pl, "personality", {}),
						mu_base=float(cfg.personality_coupling_mu_base),
						lambda_mu=float(cfg.personality_coupling_lambda_mu),
						k_base=float(cfg.selection_strength),
						lambda_k=float(cfg.personality_coupling_lambda_k),
						state_dominance=state_dominance,
						beta_state_k=float(cfg.personality_coupling_beta_state_k),
					)
					next_weights, next_velocity = inertial_growth_step(
						getattr(pl, "strategy_weights", {}),
						strategy_space,
						growth_vector=growth_vector,
						previous_velocity=getattr(pl, "personality_coupled_velocity", {}),
						inertia=float(params["mu"]),
						selection_strength=float(params["k"]),
					)
					pl.update_weights(next_weights)
					setattr(pl, "personality_coupled_velocity", next_velocity)
					setattr(pl, "personality_signal_mu", float(params["signal_mu"]))
					setattr(pl, "personality_signal_k", float(params["signal_k"]))
					setattr(pl, "personality_inertia", float(params["mu"]))
					setattr(pl, "personality_selection_strength", float(params["k"]))
					setattr(pl, "personality_state_dominance", float(state_dominance))
					setattr(pl, "personality_state_k_factor", float(state_factor))
				new_weights = _mean_player_weights(players, strategy_space=strategy_space)
		else:
			_evo_groups = local_groups if local_groups is not None else (h1_groups if h1_groups is not None else None)
			if _evo_groups is not None:
				# L2/H1: per-group replicator_step — each group computes its own growth vector
				for group_players in _evo_groups:
					group_new_weights = replicator_step(
						group_players,
						strategy_space,
						selection_strength=cfg.selection_strength,
						sampled_growth_n_strata=1,
					)
					for gpl in group_players:
						gpl.update_weights(group_new_weights)
				new_weights = _mean_player_weights(players, strategy_space=strategy_space)
			elif float(cfg.tangential_alpha) > 0.0:
				new_weights, _b1_diag = tangential_projection_replicator_step(
					players,
					strategy_space,
					selection_strength=cfg.selection_strength,
					tangential_alpha=float(cfg.tangential_alpha),
					sampled_growth_n_strata=int(cfg.sampled_growth_n_strata),
				)
			else:
				new_weights = replicator_step(
					players,
					strategy_space,
					selection_strength=cfg.selection_strength,
					strategy_selection_strengths=(strategy_selection_strengths if str(cfg.evolution_mode) == "hetero" else None),
					sampled_growth_n_strata=int(cfg.sampled_growth_n_strata),
					tangential_drift_simplex=dist,
					tangential_drift_delta=float(cfg.tangential_drift_delta),
				)
		for pl in players:
			if bool(getattr(pl, "fixed_subgroup", False)) and fixed_weights is not None:
				pl.update_weights(
					_resolved_fixed_player_weights(
						strategy_space=strategy_space,
						adaptive_weights=new_weights,
						fixed_weights=fixed_weights,
						fixed_subgroup_anchor_pull_strength=float(cfg.fixed_subgroup_anchor_pull_strength),
					)
				)
			elif bool(getattr(pl, "hybrid_update", False)):
				next_weights, next_velocity = inertial_deterministic_replicator_step(
					getattr(pl, "strategy_weights", {}),
					strategy_space,
					payoff_vector=hybrid_payoff,
					previous_velocity=getattr(pl, "hybrid_velocity", {}),
					inertia=float(cfg.hybrid_inertia),
					selection_strength=float(cfg.selection_strength),
				)
				pl.update_weights(next_weights)
				setattr(pl, "hybrid_velocity", next_velocity)
			elif str(cfg.evolution_mode) == "personality_coupled" and not _personality_coupling_is_noop(cfg):
				continue
			elif float(cfg.async_update_fraction) < 1.0:
				continue  # A1: already updated per-player in async dispatch
			elif local_groups is not None or h1_groups is not None:
				continue  # L2/H1: already updated per-group in evolution step
			else:
				pl.update_weights(new_weights)
				if str(cfg.evolution_mode) == "sampled_inertial":
					setattr(pl, "sampled_velocity", sampled_velocity)
		# M1: per-player Dirichlet mutation
		if mutation_rngs is not None and float(cfg.mutation_rate) > 0.0:
			m_eta = float(cfg.mutation_rate)
			for i, pl in enumerate(players):
				if bool(getattr(pl, "fixed_subgroup", False)):
					continue
				pl.update_weights(
					_apply_mutation_weights(
						getattr(pl, "strategy_weights", {}),
						strategy_space,
						mutation_rngs[i],
						m_eta,
					)
				)
		mean_weights = _mean_player_weights(players, strategy_space=strategy_space)
		x_history.append(dict(dist))

		threshold_regime_hi = getattr(dungeon, "_threshold_regime_hi", None)
		threshold_state_value = getattr(dungeon, "_threshold_state_value", None)
		row = {
			"round": t,
			"avg_reward": "" if avg_r is None else float(avg_r),
			"avg_utility": float(avg_u),
			"threshold_regime_hi": "" if threshold_regime_hi is None else int(bool(threshold_regime_hi)),
			"threshold_state_value": "" if threshold_state_value is None else float(threshold_state_value),
		}
		for s in strategy_space:
			row[f"p_{s}"] = float(dist[s])
		for s in strategy_space:
			row[f"w_{s}"] = float(mean_weights[s])
		row.update(_aggregate_event_records(step_records))
		rows.append(row)

		# Adaptive rule-mutation: periodically adjust failure threshold based on p_aggressive.
		if event_loader is not None and (cfg.adaptive_ft_strength or 0.0) > 0.0:
			interval = int(cfg.ft_update_interval) if cfg.ft_update_interval is not None else 500
			if (t + 1) % interval == 0:
				ft_base = float(cfg.event_failure_threshold) if cfg.event_failure_threshold is not None else 0.72
				p_agg = float(dist.get("aggressive", 1.0 / 3.0))
				ft_new = ft_base * (1.0 + float(cfg.adaptive_ft_strength) * (p_agg - 1.0 / 3.0))
				event_loader.set_failure_threshold_override(ft_new)

		# Adaptive payoff feedback v2: additive formula + clipping.
		# delta = strength * (p_agg - target); a_new = a_base + delta (punish dominance);
		# b_new = b_base - delta (compensate); both clipped to [0.5, 1.2].
		if float(cfg.adaptive_payoff_strength) > 0.0:
			p_interval = int(cfg.payoff_update_interval)
			if (t + 1) % p_interval == 0:
				p_agg = float(dist.get("aggressive", 1.0 / 3.0))
				a_base = float(cfg.a) if float(cfg.a) != 0.0 else 0.8
				b_base = float(cfg.b) if float(cfg.b) != 0.0 else 0.9
				target = float(cfg.adaptive_payoff_target)
				delta = float(cfg.adaptive_payoff_strength) * (p_agg - target)
				dungeon.a = max(0.5, min(1.2, a_base + delta))
				dungeon.b = max(0.5, min(1.2, b_base - delta))

		if round_callback is not None:
			round_callback(t, cfg, players, dungeon, step_records, row)

	return strategy_space, rows


def simulate_series_window(
	cfg: SimConfig,
	*,
	series: Literal["w", "p"],
	burn_in: int = 0,
	tail: int | None = None,
	strategy_space: list[str] | None = None,
	player_setup_callback: PlayerSetupCallback | None = None,
) -> dict[str, list[float]]:
	"""Run a simulation and return only the requested time-series window.

	This is a memory-efficient alternative to `simulate()` for large `n_rounds`.
	It collects only the values needed by downstream analysis (cycle metrics),
	without creating per-round row dictionaries.

	Window semantics
	- Matches `analysis.cycle_metrics._slice_series()` behavior.
	- The collected index range is [begin, end) where:
	  - end = n_rounds
	  - begin = max(burn_in, end - tail) if tail is not None else burn_in
	
	Returned mapping keys are strategy names (e.g. "aggressive").
	"""
	if burn_in < 0:
		raise ValueError("burn_in must be >= 0")
	if tail is not None and tail < 0:
		raise ValueError("tail must be >= 0")
	if series not in ("w", "p"):
		raise ValueError("series must be 'w' or 'p'")

	if strategy_space is None:
		strategy_space = ["aggressive", "defensive", "balanced"]
	strategy_selection_strengths = _coerce_strategy_selection_strengths(
		cfg.strategy_selection_strengths,
		strategy_space=strategy_space,
	)
	fixed_subgroup_weights = _coerce_fixed_subgroup_weights(
		cfg.fixed_subgroup_weights,
		strategy_space=strategy_space,
	)
	_validate_fixed_subgroup_params(
		fixed_subgroup_share=float(cfg.fixed_subgroup_share),
		fixed_subgroup_weights=fixed_subgroup_weights,
		fixed_subgroup_anchor_pull_strength=float(cfg.fixed_subgroup_anchor_pull_strength),
		fixed_subgroup_coupling_strength=float(cfg.fixed_subgroup_coupling_strength),
		fixed_subgroup_bidirectional_coupling_strength=float(cfg.fixed_subgroup_bidirectional_coupling_strength),
		fixed_subgroup_state_coupling_strength=float(cfg.fixed_subgroup_state_coupling_strength),
		fixed_subgroup_state_coupling_beta=float(cfg.fixed_subgroup_state_coupling_beta),
		fixed_subgroup_state_coupling_theta=float(cfg.fixed_subgroup_state_coupling_theta),
		fixed_subgroup_state_signal=str(cfg.fixed_subgroup_state_signal),
		evolution_mode=str(cfg.evolution_mode),
	)
	_validate_hybrid_params(
		evolution_mode=str(cfg.evolution_mode),
		popularity_mode=str(cfg.popularity_mode),
		payoff_mode=str(cfg.payoff_mode),
		hybrid_update_share=float(cfg.hybrid_update_share),
		hybrid_inertia=float(cfg.hybrid_inertia),
		fixed_subgroup_share=float(cfg.fixed_subgroup_share),
		fixed_subgroup_coupling_strength=float(cfg.fixed_subgroup_coupling_strength),
		fixed_subgroup_bidirectional_coupling_strength=float(cfg.fixed_subgroup_bidirectional_coupling_strength),
		fixed_subgroup_state_coupling_strength=float(cfg.fixed_subgroup_state_coupling_strength),
	)
	_validate_sampled_inertial_params(
		evolution_mode=str(cfg.evolution_mode),
		popularity_mode=str(cfg.popularity_mode),
		payoff_mode=str(cfg.payoff_mode),
		sampled_inertia=float(cfg.sampled_inertia),
		hybrid_update_share=float(cfg.hybrid_update_share),
		hybrid_inertia=float(cfg.hybrid_inertia),
		fixed_subgroup_share=float(cfg.fixed_subgroup_share),
		fixed_subgroup_weights=fixed_subgroup_weights,
		fixed_subgroup_anchor_pull_strength=float(cfg.fixed_subgroup_anchor_pull_strength),
		fixed_subgroup_coupling_strength=float(cfg.fixed_subgroup_coupling_strength),
		fixed_subgroup_bidirectional_coupling_strength=float(cfg.fixed_subgroup_bidirectional_coupling_strength),
		fixed_subgroup_state_coupling_strength=float(cfg.fixed_subgroup_state_coupling_strength),
	)
	_validate_stratified_growth_params(
		evolution_mode=str(cfg.evolution_mode),
		popularity_mode=str(cfg.popularity_mode),
		payoff_mode=str(cfg.payoff_mode),
		sampled_growth_n_strata=int(cfg.sampled_growth_n_strata),
	)
	_validate_tangential_drift_params(
		evolution_mode=str(cfg.evolution_mode),
		popularity_mode=str(cfg.popularity_mode),
		payoff_mode=str(cfg.payoff_mode),
		sampled_growth_n_strata=int(cfg.sampled_growth_n_strata),
		sampled_inertia=float(cfg.sampled_inertia),
		tangential_drift_delta=float(cfg.tangential_drift_delta),
	)

	# Determine the window [begin, end) within 0..n_rounds.
	n = int(cfg.n_rounds)
	start = min(max(int(burn_in), 0), n)
	if tail is None:
		begin = start
	else:
		begin = max(start, n - int(tail))
	begin = min(max(begin, 0), n)

	if str(cfg.evolution_mode) == "mean_field":
		if str(cfg.payoff_mode) not in ("matrix_ab", "threshold_ab"):
			raise ValueError("evolution_mode='mean_field' currently supports payoff_mode in {'matrix_ab','threshold_ab'} only")
		if int(cfg.memory_kernel) not in (1, 3, 5):
			raise ValueError("memory_kernel must be one of {1,3,5}")
		if str(cfg.payoff_mode) == "threshold_ab":
			_validate_threshold_params(
				theta=float(cfg.threshold_theta),
				theta_low=cfg.threshold_theta_low,
				theta_high=cfg.threshold_theta_high,
				trigger=str(cfg.threshold_trigger),
				state_alpha=float(cfg.threshold_state_alpha),
				a_hi=cfg.threshold_a_hi,
				b_hi=cfg.threshold_b_hi,
			)
		k = float(cfg.selection_strength)
		w_cur = _initial_weights(strategy_space=strategy_space, init_bias=float(cfg.init_bias))
		m0 = (sum(w_cur.values()) / float(len(w_cur))) if w_cur else 1.0
		w_cur = {s: (float(v) / float(m0)) for s, v in w_cur.items()}
		x_cur = _normalize_simplex(w_cur)
		x_history: list[dict[str, float]] = [dict(x_cur)]
		threshold_regime_hi: bool | None = None
		threshold_state_value: float | None = None
		mean_field_velocity = {s: 0.0 for s in strategy_space}

		out: dict[str, list[float]] = {s: [] for s in strategy_space}
		for t in range(n):
			x_pay = _average_simplex_history(
				x_history,
				strategy_space=strategy_space,
				kernel=int(cfg.memory_kernel),
				lag=int(cfg.payoff_lag),
			)
			u, threshold_regime_hi, threshold_state_value = _payoff_vec_from_state(
				payoff_mode=str(cfg.payoff_mode),
				strategy_space=strategy_space,
				a=float(cfg.a),
				b=float(cfg.b),
				threshold_theta=float(cfg.threshold_theta),
				threshold_theta_low=cfg.threshold_theta_low,
				threshold_theta_high=cfg.threshold_theta_high,
				threshold_trigger=str(cfg.threshold_trigger),
				threshold_state_alpha=float(cfg.threshold_state_alpha),
				threshold_a_hi=cfg.threshold_a_hi,
				threshold_b_hi=cfg.threshold_b_hi,
				matrix_cross_coupling=float(cfg.matrix_cross_coupling),
				x=x_pay,
				current_regime_hi=threshold_regime_hi,
				current_state_value=threshold_state_value,
			)
			if float(cfg.sampled_inertia) > 0.0:
				w_cur, mean_field_velocity = inertial_deterministic_replicator_step(
					w_cur,
					strategy_space,
					payoff_vector=u,
					previous_velocity=mean_field_velocity,
					inertia=float(cfg.sampled_inertia),
					selection_strength=k,
				)
			else:
				w_cur = deterministic_replicator_step(
					w_cur,
					strategy_space,
					payoff_vector=u,
					selection_strength=k,
					tangential_drift_simplex=x_cur,
					tangential_drift_delta=float(cfg.tangential_drift_delta),
				)
			x_next = _normalize_simplex(w_cur)

			if t >= begin:
				if series == "p":
					for s in strategy_space:
						out[s].append(float(x_next[s]))
				else:
					for s in strategy_space:
						out[s].append(float(w_cur[s]))

			x_cur = x_next
			x_history.append(dict(x_cur))

		return out

	# Deterministic sampling: give each player an independent RNG stream.
	if cfg.seed is None:
		players = [BasePlayer(strategy_space) for _ in range(cfg.n_players)]
	else:
		players = [
			BasePlayer(strategy_space, rng=random.Random(int(cfg.seed) + i))
			for i in range(cfg.n_players)
		]
	fixed_n = _resolve_fixed_subgroup_count(n_players=int(cfg.n_players), fixed_subgroup_share=float(cfg.fixed_subgroup_share))
	hybrid_n = _resolve_hybrid_count(n_players=int(cfg.n_players), hybrid_update_share=float(cfg.hybrid_update_share))
	fixed_weights = _fixed_weights_dict(strategy_space=strategy_space, fixed_subgroup_weights=fixed_subgroup_weights)
	for idx, pl in enumerate(players):
		is_fixed = bool(idx < fixed_n and fixed_weights is not None)
		setattr(pl, "stratum", int(idx % max(1, int(cfg.sampled_growth_n_strata))))
		setattr(pl, "fixed_subgroup", is_fixed)
		is_hybrid = bool(str(cfg.evolution_mode) == "hybrid" and idx < hybrid_n)
		setattr(pl, "hybrid_update", is_hybrid)
		if is_hybrid:
			setattr(pl, "hybrid_velocity", {s: 0.0 for s in strategy_space})
		if str(cfg.evolution_mode) == "sampled_inertial" and not is_fixed:
			setattr(pl, "sampled_velocity", {s: 0.0 for s in strategy_space})
		if float(cfg.sampling_beta) != 1.0:
			setattr(pl, "sampling_beta", float(cfg.sampling_beta))

	# M1: per-player mutation RNGs (seeded deterministically from cfg.seed)
	mutation_rngs_sw: list[random.Random] | None = None
	if float(cfg.mutation_rate) > 0.0:
		base_seed = int(cfg.seed or 0)
		mutation_rngs_sw = [random.Random(base_seed + 900000 + i) for i in range(int(cfg.n_players))]

	# L2: partition players into fixed local sub-groups
	local_groups_sw: list[list[object]] | None = None
	l2_gs_sw = int(cfg.local_group_size)
	if l2_gs_sw > 0 and l2_gs_sw < int(cfg.n_players):
		local_groups_sw = _partition_players_into_groups(players, l2_gs_sw)

	# H1: partition players into niche sub-groups (separate from L2)
	h1_groups_sw: list[list[object]] | None = None
	h1_epsilon_sw = float(cfg.payoff_niche_epsilon)
	h1_ngs_sw = int(cfg.niche_group_size)
	if h1_epsilon_sw > 0 and h1_ngs_sw > 0 and h1_ngs_sw < int(cfg.n_players):
		h1_groups_sw = _partition_players_into_groups(players, h1_ngs_sw)

	init_w = _initial_weights(strategy_space=strategy_space, init_bias=float(cfg.init_bias))
	if init_w:
		for pl in players:
			if bool(getattr(pl, "fixed_subgroup", False)) and fixed_weights is not None:
				pl.update_weights(
					_resolved_fixed_player_weights(
						strategy_space=strategy_space,
						adaptive_weights=init_w,
						fixed_weights=fixed_weights,
						fixed_subgroup_anchor_pull_strength=float(cfg.fixed_subgroup_anchor_pull_strength),
					)
				)
			else:
				pl.update_weights(init_w)
	if player_setup_callback is not None:
		player_setup_callback(players, strategy_space, cfg)

	dungeon = DungeonAI(
		payoff_mode=cfg.payoff_mode,
		gamma=cfg.gamma,
		epsilon=cfg.epsilon,
		a=cfg.a,
		b=cfg.b,
		matrix_cross_coupling=cfg.matrix_cross_coupling,
		memory_kernel=cfg.memory_kernel,
		threshold_theta=cfg.threshold_theta,
		threshold_theta_low=cfg.threshold_theta_low,
		threshold_theta_high=cfg.threshold_theta_high,
		threshold_trigger=cfg.threshold_trigger,
		threshold_state_alpha=cfg.threshold_state_alpha,
		threshold_a_hi=cfg.threshold_a_hi,
		threshold_b_hi=cfg.threshold_b_hi,
		strategy_cycle=strategy_space,
	)
	engine = GameEngine(players, dungeon, popularity_mode=str(cfg.popularity_mode))
	if str(cfg.popularity_mode) == "expected":
		dungeon.set_popularity(dict(init_w))

	out: dict[str, list[float]] = {s: [] for s in strategy_space}
	new_weights = {s: 1.0 for s in strategy_space}
	x_history: list[dict[str, float]] = [
		_normalize_simplex(_mean_player_weights(players, strategy_space=strategy_space))
	]

	for t in range(n):
		step_records = engine.step()
		_apply_fixed_subgroup_coupling(
			players,
			strategy_space=strategy_space,
			fixed_weights=fixed_weights,
			fixed_subgroup_coupling_strength=float(cfg.fixed_subgroup_coupling_strength),
			a=float(dungeon.a),
			b=float(dungeon.b),
			step_records=step_records,
		)
		_apply_fixed_subgroup_bidirectional_coupling(
			players,
			strategy_space=strategy_space,
			fixed_subgroup_bidirectional_coupling_strength=float(cfg.fixed_subgroup_bidirectional_coupling_strength),
			a=float(dungeon.a),
			b=float(dungeon.b),
			step_records=step_records,
		)
		_apply_fixed_subgroup_state_coupling(
			players,
			strategy_space=strategy_space,
			fixed_weights=fixed_weights,
			fixed_subgroup_state_coupling_strength=float(cfg.fixed_subgroup_state_coupling_strength),
			fixed_subgroup_state_coupling_beta=float(cfg.fixed_subgroup_state_coupling_beta),
			fixed_subgroup_state_coupling_theta=float(cfg.fixed_subgroup_state_coupling_theta),
			fixed_subgroup_state_signal=str(cfg.fixed_subgroup_state_signal),
			a=float(dungeon.a),
			b=float(dungeon.b),
			step_records=step_records,
		)

		# H1: per-group payoff niche bonus
		if h1_groups_sw is not None and h1_epsilon_sw > 0:
			for gk, group in enumerate(h1_groups_sw):
				niche_strategy = strategy_space[gk % len(strategy_space)]
				for pl in group:
					if getattr(pl, "last_strategy", None) == niche_strategy:
						pl.last_reward = float(pl.last_reward) + h1_epsilon_sw

		dist = strategy_distribution(players, strategy_space)
		x_pay = _average_simplex_history(
			x_history,
			strategy_space=strategy_space,
			kernel=int(cfg.memory_kernel),
			lag=int(cfg.payoff_lag),
		)
		hybrid_payoff = _matrix_ab_payoff_vec(
			strategy_space=strategy_space,
			a=float(cfg.a),
			b=float(cfg.b),
			matrix_cross_coupling=float(cfg.matrix_cross_coupling),
			x=x_pay,
		)

		# Evolution update (weights used for the *next* round, same as CSV rows).
		if float(cfg.async_update_fraction) < 1.0:
			new_weights, _a1_diag = async_replicator_step(
				players,
				strategy_space,
				selection_strength=cfg.selection_strength,
				async_update_fraction=float(cfg.async_update_fraction),
				seed=int(cfg.seed or 0),
				round_index=t,
			)
		elif str(cfg.evolution_mode) == "sampled_inertial":
			new_weights, sampled_velocity = inertial_sampled_replicator_step(
				players,
				strategy_space,
				previous_velocity=getattr(players[0], "sampled_velocity", {}) if players else None,
				inertia=float(cfg.sampled_inertia),
				selection_strength=cfg.selection_strength,
			)
		elif str(cfg.evolution_mode) == "personality_coupled":
			if _personality_coupling_is_noop(cfg):
				new_weights = replicator_step(
					players,
					strategy_space,
					selection_strength=cfg.selection_strength,
					tangential_drift_simplex=dist,
					tangential_drift_delta=float(cfg.tangential_drift_delta),
				)
			else:
				growth_vector = sampled_growth_vector(players, strategy_space)
				state_dominance = _current_state_dominance(dist)
				state_factor = personality_state_k_factor(
					state_dominance=state_dominance,
					beta_state_k=float(cfg.personality_coupling_beta_state_k),
				)
				for pl in players:
					params = resolve_personality_coupling(
						getattr(pl, "personality", {}),
						mu_base=float(cfg.personality_coupling_mu_base),
						lambda_mu=float(cfg.personality_coupling_lambda_mu),
						k_base=float(cfg.selection_strength),
						lambda_k=float(cfg.personality_coupling_lambda_k),
						state_dominance=state_dominance,
						beta_state_k=float(cfg.personality_coupling_beta_state_k),
					)
					next_weights, next_velocity = inertial_growth_step(
						getattr(pl, "strategy_weights", {}),
						strategy_space,
						growth_vector=growth_vector,
						previous_velocity=getattr(pl, "personality_coupled_velocity", {}),
						inertia=float(params["mu"]),
						selection_strength=float(params["k"]),
					)
					pl.update_weights(next_weights)
					setattr(pl, "personality_coupled_velocity", next_velocity)
					setattr(pl, "personality_signal_mu", float(params["signal_mu"]))
					setattr(pl, "personality_signal_k", float(params["signal_k"]))
					setattr(pl, "personality_inertia", float(params["mu"]))
					setattr(pl, "personality_selection_strength", float(params["k"]))
					setattr(pl, "personality_state_dominance", float(state_dominance))
					setattr(pl, "personality_state_k_factor", float(state_factor))
				new_weights = _mean_player_weights(players, strategy_space=strategy_space)
		else:
			_evo_groups_sw = local_groups_sw if local_groups_sw is not None else (h1_groups_sw if h1_groups_sw is not None else None)
			if _evo_groups_sw is not None:
				# L2/H1: per-group replicator_step — each group computes its own growth vector
				for group_players in _evo_groups_sw:
					group_new_weights = replicator_step(
						group_players,
						strategy_space,
						selection_strength=cfg.selection_strength,
						sampled_growth_n_strata=1,
					)
					for gpl in group_players:
						gpl.update_weights(group_new_weights)
				new_weights = _mean_player_weights(players, strategy_space=strategy_space)
			elif float(cfg.tangential_alpha) > 0.0:
				new_weights, _b1_diag = tangential_projection_replicator_step(
					players,
					strategy_space,
					selection_strength=cfg.selection_strength,
					tangential_alpha=float(cfg.tangential_alpha),
					sampled_growth_n_strata=int(cfg.sampled_growth_n_strata),
				)
			else:
				new_weights = replicator_step(
					players,
					strategy_space,
					selection_strength=cfg.selection_strength,
					strategy_selection_strengths=(strategy_selection_strengths if str(cfg.evolution_mode) == "hetero" else None),
					sampled_growth_n_strata=int(cfg.sampled_growth_n_strata),
					tangential_drift_simplex=dist,
					tangential_drift_delta=float(cfg.tangential_drift_delta),
				)
		for pl in players:
			if bool(getattr(pl, "fixed_subgroup", False)) and fixed_weights is not None:
				pl.update_weights(
					_resolved_fixed_player_weights(
						strategy_space=strategy_space,
						adaptive_weights=new_weights,
						fixed_weights=fixed_weights,
						fixed_subgroup_anchor_pull_strength=float(cfg.fixed_subgroup_anchor_pull_strength),
					)
				)
			elif bool(getattr(pl, "hybrid_update", False)):
				next_weights, next_velocity = inertial_deterministic_replicator_step(
					getattr(pl, "strategy_weights", {}),
					strategy_space,
					payoff_vector=hybrid_payoff,
					previous_velocity=getattr(pl, "hybrid_velocity", {}),
					inertia=float(cfg.hybrid_inertia),
					selection_strength=float(cfg.selection_strength),
				)
				pl.update_weights(next_weights)
				setattr(pl, "hybrid_velocity", next_velocity)
			elif str(cfg.evolution_mode) == "personality_coupled" and not _personality_coupling_is_noop(cfg):
				continue
			elif float(cfg.async_update_fraction) < 1.0:
				continue  # A1: already updated per-player in async dispatch
			elif local_groups_sw is not None or h1_groups_sw is not None:
				continue  # L2/H1: already updated per-group in evolution step
			else:
				pl.update_weights(new_weights)
				if str(cfg.evolution_mode) == "sampled_inertial":
					setattr(pl, "sampled_velocity", sampled_velocity)
		# M1: per-player Dirichlet mutation
		if mutation_rngs_sw is not None and float(cfg.mutation_rate) > 0.0:
			m_eta = float(cfg.mutation_rate)
			for i, pl in enumerate(players):
				if bool(getattr(pl, "fixed_subgroup", False)):
					continue
				pl.update_weights(
					_apply_mutation_weights(
						getattr(pl, "strategy_weights", {}),
						strategy_space,
						mutation_rngs_sw[i],
						m_eta,
					)
				)
		mean_weights = _mean_player_weights(players, strategy_space=strategy_space)
		x_history.append(dict(dist))

		if t < begin:
			continue
		if series == "p":
			for s in strategy_space:
				out[s].append(float(dist[s]))
		else:  # series == "w"
			for s in strategy_space:
				out[s].append(float(mean_weights[s]))

	return out


def main() -> None:
	cfg = _parse_args()
	strategy_space, rows = simulate(cfg)
	# For CLI summary only.
	new_weights = {f.split("w_", 1)[1]: float(v) for f, v in rows[-1].items() if f.startswith("w_")} if rows else {s: 1.0 for s in strategy_space}

	_write_timeseries_csv(cfg.out_csv, strategy_space=strategy_space, rows=rows)

	# 簡短摘要（stdout）
	print(f"Wrote CSV: {cfg.out_csv}")
	print(
		f"players={cfg.n_players} rounds={cfg.n_rounds} "
		f"seed={cfg.seed} payoff_mode={cfg.payoff_mode} gamma={cfg.gamma} epsilon={cfg.epsilon} a={cfg.a} b={cfg.b} threshold_theta={cfg.threshold_theta} threshold_theta_low={cfg.threshold_theta_low} threshold_theta_high={cfg.threshold_theta_high} threshold_trigger={cfg.threshold_trigger} threshold_state_alpha={cfg.threshold_state_alpha} threshold_a_hi={cfg.threshold_a_hi} threshold_b_hi={cfg.threshold_b_hi} init_bias={cfg.init_bias} evolution_mode={cfg.evolution_mode} strategy_selection_strengths={cfg.strategy_selection_strengths} hybrid_update_share={cfg.hybrid_update_share} hybrid_inertia={cfg.hybrid_inertia} sampled_inertia={cfg.sampled_inertia} personality_coupling_mu_base={cfg.personality_coupling_mu_base} personality_coupling_lambda_mu={cfg.personality_coupling_lambda_mu} personality_coupling_lambda_k={cfg.personality_coupling_lambda_k} personality_coupling_beta_state_k={cfg.personality_coupling_beta_state_k} apply_event_trait_deltas={cfg.apply_event_trait_deltas} fixed_subgroup_share={cfg.fixed_subgroup_share} fixed_subgroup_weights={cfg.fixed_subgroup_weights} fixed_subgroup_coupling_strength={cfg.fixed_subgroup_coupling_strength} fixed_subgroup_state_coupling_strength={cfg.fixed_subgroup_state_coupling_strength} fixed_subgroup_state_coupling_beta={cfg.fixed_subgroup_state_coupling_beta} fixed_subgroup_state_coupling_theta={cfg.fixed_subgroup_state_coupling_theta} fixed_subgroup_state_signal={cfg.fixed_subgroup_state_signal} payoff_lag={cfg.payoff_lag} memory_kernel={cfg.memory_kernel} "
		f"selection_strength={cfg.selection_strength} sampled_growth_n_strata={cfg.sampled_growth_n_strata} tangential_drift_delta={cfg.tangential_drift_delta} enable_events={cfg.enable_events} events_json={cfg.events_json if cfg.events_json is not None else DEFAULT_EVENTS_JSON if cfg.enable_events else None}"
	)
	print("Final weights:", new_weights)


if __name__ == "__main__":
	main()

