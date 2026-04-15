"""Regression tests for the personality RL runtime bridge.

Test groups (bridge spec §8):
  8.1  Degeneration tests
  8.2  Contract tests
  8.3  Personality mapping tests
  8.4  I/O tests
"""

from __future__ import annotations

import csv
import json
import tempfile
from pathlib import Path

import pytest

from analysis.decay_rate import _read_timeseries_csv
from analysis.visualization import load_timeseries_csv
from players.rl_player import (
    RLPlayer,
    init_rl_player,
    personality_latent_signals,
    sample_personality,
)
from simulation.personality_rl_runtime import (
    ROUND_CSV_FIELDS,
    EventBridge,
    PersonalityRLConfig,
    RunResult,
    WorldFeedback,
    _build_provenance,
    _write_player_snapshot,
    _write_provenance,
    _write_round_csv,
    run_personality_rl,
)


# -------------------------------------------------------------------
# Fixtures
# -------------------------------------------------------------------

EVENT_JSON = Path("docs/personality_dungeon_v1/02_event_templates_v1.json")

_SMALL_CFG = PersonalityRLConfig(
    n_players=30,
    n_rounds=100,
    personality_mode="none",
)


_TRACK_A2_PROVENANCE_MIN_KEYS = {
    "alpha_lo", "alpha_hi", "beta_ceiling", "strategy_alpha_multipliers", "payoff_epsilon",
    "personality_mode", "lambda_alpha", "lambda_beta", "lambda_r", "lambda_risk",
    "events_json", "world_mode", "lambda_world", "world_update_interval", "seed",
    "mean_alpha", "std_alpha", "mean_beta", "std_beta",
    "risk_rule_version", "diagnostic_rule_version",
    "event_sync_index_mean", "reward_perturb_corr", "trap_entry_round",
    "dispatch_seed_stream", "dispatch_mode", "dispatch_target_rate",
    "dispatch_mean_affected_ratio", "dispatch_player_activation_min",
    "dispatch_player_activation_max", "dispatch_player_activation_cv",
    "dispatch_fairness_window", "dispatch_fairness_tolerance",
    "dispatch_fairness_checks", "dispatch_fairness_failures", "dispatch_fairness_pass",
    "event_reward_mode", "event_reward_multiplier_cap",
    "event_impact_mode", "event_impact_horizon", "event_impact_decay",
}


# ===================================================================
# 8.1  Degeneration tests
# ===================================================================


class TestDegeneration:
    def test_no_events_no_personality_completes(self):
        """With events off and personality=none, should complete."""
        result = run_personality_rl(_SMALL_CFG, seed=42)
        assert len(result.rows) == 100

    def test_symmetric_multipliers_no_extreme_dominance(self):
        """[1,1,1] multipliers → no single strategy dominates > 90 %."""
        cfg = PersonalityRLConfig(
            n_players=30,
            n_rounds=500,
            strategy_alpha_multipliers=[1.0, 1.0, 1.0],
            personality_mode="none",
        )
        result = run_personality_rl(cfg, seed=42)
        tail = result.rows[-100:]
        for row in tail:
            for s in ("aggressive", "defensive", "balanced"):
                assert row[f"p_{s}"] <= 0.95, (
                    f"p_{s} = {row[f'p_{s}']:.3f} exceeds 0.95 in symmetric case"
                )

    def test_deterministic_with_same_seed(self):
        """Same config + seed → same trajectory."""
        r1 = run_personality_rl(_SMALL_CFG, seed=99)
        r2 = run_personality_rl(_SMALL_CFG, seed=99)
        for a, b in zip(r1.rows, r2.rows):
            assert a["p_aggressive"] == b["p_aggressive"]
            assert a["p_defensive"] == b["p_defensive"]


# ===================================================================
# 8.2  Contract tests
# ===================================================================


class TestContracts:
    def test_p_star_sums_to_one(self):
        result = run_personality_rl(_SMALL_CFG, seed=42)
        for row in result.rows:
            total = (
                row["p_aggressive"] + row["p_defensive"] + row["p_balanced"]
            )
            assert abs(total - 1.0) < 1e-9

    def test_pi_star_sums_to_one(self):
        result = run_personality_rl(_SMALL_CFG, seed=42)
        for row in result.rows:
            total = (
                row["pi_aggressive"] + row["pi_defensive"] + row["pi_balanced"]
            )
            assert abs(total - 1.0) < 1e-6

    def test_no_legacy_w_star_fields(self):
        """CSV must not contain w_* fields (bridge spec §6.1 note)."""
        result = run_personality_rl(_SMALL_CFG, seed=42)
        for row in result.rows:
            for key in row:
                assert not key.startswith("w_"), f"Legacy w_* field: {key}"

    def test_all_csv_fields_present(self):
        result = run_personality_rl(_SMALL_CFG, seed=42)
        for row in result.rows:
            for fld in ROUND_CSV_FIELDS:
                assert fld in row, f"Missing field: {fld}"

    def test_event_sync_index_bounded(self):
        result = run_personality_rl(_SMALL_CFG, seed=42)
        for row in result.rows:
            assert 0.0 <= float(row["event_sync_index"]) <= 1.0

    def test_event_affected_ratio_bounded(self):
        result = run_personality_rl(_SMALL_CFG, seed=42)
        for row in result.rows:
            assert 0.0 <= float(row["event_affected_ratio"]) <= 1.0
            assert int(row["event_affected_count"]) >= 0

    def test_event_channels_zero_when_events_disabled(self):
        """Track A lock: events-off runs still emit event fields as zeros."""
        result = run_personality_rl(_SMALL_CFG, seed=42)
        for row in result.rows:
            assert int(row["event_affected_count"]) == 0
            assert float(row["event_affected_ratio"]) == 0.0
            assert float(row["event_sync_index"]) == 0.0
            assert str(row["dominant_event_type"]) == ""

    def test_provenance_contains_track_a2_minimum_keys(self):
        result = run_personality_rl(_SMALL_CFG, seed=42)
        prov = _build_provenance(
            _SMALL_CFG,
            42,
            result.players,
            diagnostics=result.diagnostics,
        )
        missing = sorted(
            key for key in _TRACK_A2_PROVENANCE_MIN_KEYS
            if key not in prov
        )
        assert not missing, f"Missing Track A2 provenance keys: {missing}"


# ===================================================================
# 8.3  Personality mapping tests
# ===================================================================

_ZEROS = {
    "impulsiveness": 0.0, "caution": 0.0, "greed": 0.0,
    "optimism": 0.0, "suspicion": 0.0, "persistence": 0.0,
    "randomness": 0.0, "stability_seeking": 0.0, "ambition": 0.0,
    "patience": 0.0, "curiosity": 0.0, "fearfulness": 0.0,
}


class TestPersonalityMapping:
    def test_high_drive_higher_alpha(self):
        high = {**_ZEROS, "impulsiveness": 0.9, "greed": 0.9, "ambition": 0.9}
        low = {**_ZEROS, "impulsiveness": -0.9, "greed": -0.9, "ambition": -0.9}
        p_hi = init_rl_player(
            0, high, alpha_base=0.10, beta_base=3.0,
            strategy_alpha_multipliers=[1.2, 1.0, 0.8],
            payoff_bias=[0, 0, 0], lambda_alpha=0.3,
        )
        p_lo = init_rl_player(
            1, low, alpha_base=0.10, beta_base=3.0,
            strategy_alpha_multipliers=[1.2, 1.0, 0.8],
            payoff_bias=[0, 0, 0], lambda_alpha=0.3,
        )
        assert p_hi.alpha > p_lo.alpha

    def test_high_guard_higher_beta(self):
        high = {**_ZEROS, "caution": 0.9, "fearfulness": 0.9, "suspicion": 0.9}
        low = {**_ZEROS, "caution": -0.9, "fearfulness": -0.9, "suspicion": -0.9}
        p_hi = init_rl_player(
            0, high, alpha_base=0.10, beta_base=3.0,
            strategy_alpha_multipliers=[1.2, 1.0, 0.8],
            payoff_bias=[0, 0, 0], lambda_beta=0.3,
        )
        p_lo = init_rl_player(
            1, low, alpha_base=0.10, beta_base=3.0,
            strategy_alpha_multipliers=[1.2, 1.0, 0.8],
            payoff_bias=[0, 0, 0], lambda_beta=0.3,
        )
        assert p_hi.beta > p_lo.beta

    def test_zero_lambda_no_modulation(self):
        """lambda=0 → personality has no effect on alpha/beta."""
        p1 = init_rl_player(
            0, {**_ZEROS, "impulsiveness": 0.9}, alpha_base=0.10,
            beta_base=3.0, strategy_alpha_multipliers=[1, 1, 1],
            payoff_bias=[0, 0, 0], lambda_alpha=0.0, lambda_beta=0.0,
        )
        p2 = init_rl_player(
            1, {**_ZEROS, "impulsiveness": -0.9}, alpha_base=0.10,
            beta_base=3.0, strategy_alpha_multipliers=[1, 1, 1],
            payoff_bias=[0, 0, 0], lambda_alpha=0.0, lambda_beta=0.0,
        )
        assert p1.alpha == p2.alpha
        assert p1.beta == p2.beta

    def test_latent_signals_range(self):
        import random as _rnd
        rng = _rnd.Random(0)
        for _ in range(50):
            pers = sample_personality(rng=rng)
            sig = personality_latent_signals(pers)
            for v in sig.values():
                assert -1.5 <= v <= 1.5

    def test_beta_includes_z_temporal(self):
        """beta mapping should include z_temporal: high patience → higher beta."""
        high_temporal = {**_ZEROS, "patience": 0.9, "persistence": 0.9, "stability_seeking": 0.9}
        low_temporal = {**_ZEROS, "patience": -0.9, "persistence": -0.9, "stability_seeking": -0.9}
        p_hi = init_rl_player(
            0, high_temporal, alpha_base=0.10, beta_base=3.0,
            strategy_alpha_multipliers=[1.2, 1.0, 0.8],
            payoff_bias=[0, 0, 0], lambda_beta=0.3,
        )
        p_lo = init_rl_player(
            1, low_temporal, alpha_base=0.10, beta_base=3.0,
            strategy_alpha_multipliers=[1.2, 1.0, 0.8],
            payoff_bias=[0, 0, 0], lambda_beta=0.3,
        )
        assert p_hi.beta > p_lo.beta, "z_temporal should increase beta"

    def test_lambda_r_offsets_strategy_multipliers(self):
        """lambda_r > 0 → per-player strategy multiplier offsets."""
        high_drive = {**_ZEROS, "impulsiveness": 0.9, "greed": 0.9, "ambition": 0.9}
        p = init_rl_player(
            0, high_drive, alpha_base=0.10, beta_base=3.0,
            strategy_alpha_multipliers=[1.2, 1.0, 0.8],
            payoff_bias=[0, 0, 0], lambda_r=0.10,
        )
        # high z_drive → r_A should increase above baseline 1.2
        assert p.strategy_alpha_multipliers[0] > 1.2

    def test_lambda_r_zero_no_offset(self):
        """lambda_r=0 → multipliers stay at baseline even with personality."""
        pers = {**_ZEROS, "impulsiveness": 0.9, "greed": 0.9}
        p = init_rl_player(
            0, pers, alpha_base=0.10, beta_base=3.0,
            strategy_alpha_multipliers=[1.2, 1.0, 0.8],
            payoff_bias=[0, 0, 0], lambda_r=0.0,
        )
        assert p.strategy_alpha_multipliers == [1.2, 1.0, 0.8]

    def test_risk_sensitivity_high_guard(self):
        """High guard + fearfulness → positive risk_sensitivity."""
        high_guard = {**_ZEROS, "caution": 0.9, "fearfulness": 0.9, "suspicion": 0.9}
        p = init_rl_player(
            0, high_guard, alpha_base=0.10, beta_base=3.0,
            strategy_alpha_multipliers=[1.2, 1.0, 0.8],
            payoff_bias=[0, 0, 0], lambda_risk=0.5,
        )
        assert p.risk_sensitivity > 0

    def test_risk_sensitivity_zero_lambda(self):
        """lambda_risk=0 → risk_sensitivity is 0 regardless of personality."""
        pers = {**_ZEROS, "caution": 0.9, "fearfulness": 0.9}
        p = init_rl_player(
            0, pers, alpha_base=0.10, beta_base=3.0,
            strategy_alpha_multipliers=[1.2, 1.0, 0.8],
            payoff_bias=[0, 0, 0], lambda_risk=0.0,
        )
        assert p.risk_sensitivity == 0.0

    def test_all_lambdas_zero_equals_baseline(self):
        """All lambda=0 with any personality → same as empty personality."""
        pers = {**_ZEROS, "impulsiveness": 0.7, "caution": -0.3}
        p_pers = init_rl_player(
            0, pers, alpha_base=0.10, beta_base=3.0,
            strategy_alpha_multipliers=[1.2, 1.0, 0.8],
            payoff_bias=[0, 0, 0],
            lambda_alpha=0.0, lambda_beta=0.0, lambda_r=0.0, lambda_risk=0.0,
        )
        p_empty = init_rl_player(
            1, {}, alpha_base=0.10, beta_base=3.0,
            strategy_alpha_multipliers=[1.2, 1.0, 0.8],
            payoff_bias=[0, 0, 0],
        )
        assert p_pers.alpha == p_empty.alpha
        assert p_pers.beta == p_empty.beta
        assert p_pers.strategy_alpha_multipliers == p_empty.strategy_alpha_multipliers
        assert p_pers.risk_sensitivity == p_empty.risk_sensitivity


# ===================================================================
# 8.4  I/O tests
# ===================================================================


class TestIO:
    def test_writes_round_csv(self):
        result = run_personality_rl(_SMALL_CFG, seed=42)
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "ts.csv"
            _write_round_csv(csv_path, result.rows)
            assert csv_path.exists()
            with csv_path.open() as f:
                reader = csv.DictReader(f)
                rows = list(reader)
            assert len(rows) == 100
            assert set(ROUND_CSV_FIELDS).issubset(rows[0].keys())

    def test_writes_provenance_json(self):
        result = run_personality_rl(_SMALL_CFG, seed=42)
        prov = _build_provenance(_SMALL_CFG, 42, result.players)
        with tempfile.TemporaryDirectory() as tmpdir:
            prov_path = Path(tmpdir) / "prov.json"
            _write_provenance(prov_path, prov)
            assert prov_path.exists()
            data = json.loads(prov_path.read_text())
            assert "alpha_lo" in data
            assert data["seed"] == 42
            assert "event_sync_index_mean" in data
            assert "reward_perturb_corr" in data
            assert "trap_entry_round" in data
            assert "dispatch_seed_stream" in data
            assert "dispatch_mode" in data
            assert "dispatch_target_rate" in data
            assert "dispatch_mean_affected_ratio" in data
            assert "dispatch_player_activation_min" in data
            assert "dispatch_player_activation_max" in data
            assert "dispatch_player_activation_cv" in data
            assert "dispatch_fairness_window" in data
            assert "dispatch_fairness_tolerance" in data
            assert "dispatch_fairness_pass" in data

    def test_writes_player_snapshot_tsv(self):
        result = run_personality_rl(_SMALL_CFG, seed=42)
        with tempfile.TemporaryDirectory() as tmpdir:
            snap_path = Path(tmpdir) / "snap.tsv"
            _write_player_snapshot(snap_path, result.players)
            assert snap_path.exists()
            with snap_path.open() as f:
                lines = f.readlines()
            # header + 30 players
            assert len(lines) == 31

    def test_analysis_decay_rate_reads_runtime_csv(self):
        """Track A lock: analysis.decay_rate must read runtime CSV via p_* fields."""
        result = run_personality_rl(_SMALL_CFG, seed=42)
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "ts.csv"
            _write_round_csv(csv_path, result.rows)
            series_map = _read_timeseries_csv(csv_path, series="p")
        assert set(series_map.keys()) == {"aggressive", "defensive", "balanced"}
        assert len(series_map["aggressive"]) == 100

    def test_analysis_visualization_loads_runtime_csv_without_w_fields(self):
        """Track A lock: visualization loader must remain compatible without w_* columns."""
        result = run_personality_rl(_SMALL_CFG, seed=42)
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "ts.csv"
            _write_round_csv(csv_path, result.rows)
            ts = load_timeseries_csv(csv_path)
        assert len(ts.rounds) == 100
        assert set(ts.proportions.keys()) == {"aggressive", "defensive", "balanced"}
        assert set(ts.weights.keys()) == {"aggressive", "defensive", "balanced"}


# ===================================================================
# Event bridge smoke tests
# ===================================================================


class TestEventBridge:
    @pytest.mark.skipif(
        not EVENT_JSON.exists(),
        reason="event templates JSON not found",
    )
    def test_load_and_sample(self):
        import random as _rnd
        bridge = EventBridge(EVENT_JSON)
        assert len(bridge.events) >= 1
        rng = _rnd.Random(7)
        ev = bridge.sample_event(rng=rng)
        assert "type" in ev or "event_type" in ev

    @pytest.mark.skipif(
        not EVENT_JSON.exists(),
        reason="event templates JSON not found",
    )
    def test_reward_risk_not_nan(self):
        import random as _rnd
        bridge = EventBridge(EVENT_JSON)
        rng = _rnd.Random(7)
        ev = bridge.sample_event(rng=rng)
        pers = sample_personality(rng=rng)
        rm, rk = bridge.compute_reward_risk(
            ev, pers, scale=0.01, rng=rng,
        )
        assert rm == rm  # not NaN
        assert rk == rk

    @pytest.mark.skipif(
        not EVENT_JSON.exists(),
        reason="event templates JSON not found",
    )
    def test_runtime_with_events(self):
        cfg = PersonalityRLConfig(
            n_players=20,
            n_rounds=50,
            personality_mode="random",
            lambda_alpha=0.1,
            lambda_beta=0.1,
            events_json=str(EVENT_JSON),
            event_rate=0.0,
            event_reward_scale=0.01,
            event_dispatch_mode="async_poisson",
            event_dispatch_target_rate=0.3,
            event_dispatch_seed_offset=17,
        )
        result = run_personality_rl(cfg, seed=42)
        assert len(result.rows) == 50
        # At least some rounds should have an event
        event_rounds = [r for r in result.rows if r["dominant_event_type"]]
        assert len(event_rounds) >= 1
        assert "event_sync_index_mean" in result.diagnostics
        assert "reward_perturb_corr" in result.diagnostics
        assert "trap_entry_round" in result.diagnostics
        assert "dispatch_seed_stream" in result.diagnostics
        assert "dispatch_mode" in result.diagnostics
        assert "dispatch_target_rate" in result.diagnostics
        assert "dispatch_fairness_pass" in result.diagnostics
        assert -1.0 <= float(result.diagnostics["reward_perturb_corr"]) <= 1.0
        assert 0.0 <= float(result.diagnostics["dispatch_mean_affected_ratio"]) <= 1.0


# ===================================================================
# World Feedback tests
# ===================================================================


class TestWorldFeedback:
    def test_world_stays_at_half_when_lambda_zero(self):
        """lambda_world=0 → world state stays at (0.5, 0.5, 0.5, 0.5)."""
        wf = WorldFeedback(lambda_world=0.0, update_interval=10)
        for t in range(30):
            wf.record_round(
                p_agg=0.5, p_def=0.3, p_bal=0.2,
                avg_reward=0.3, event_type="Threat",
            )
            wf.maybe_update(t)
        for dim in ("scarcity", "threat", "noise", "intel"):
            assert wf.state[dim] == 0.5

    def test_world_deviates_with_nonzero_lambda(self):
        """lambda_world>0 + biased p_* → world state deviates."""
        wf = WorldFeedback(lambda_world=0.15, update_interval=10)
        for t in range(20):
            wf.record_round(
                p_agg=0.7, p_def=0.2, p_bal=0.1,
                avg_reward=0.1, event_type="Threat",
            )
            wf.maybe_update(t)
        # Aggressive dominance should push threat up and noise down
        assert wf.state["threat"] > 0.5
        assert wf.state["noise"] < 0.5

    def test_world_event_weights_change(self):
        """After world update, event weights should differ from uniform."""
        wf = WorldFeedback(lambda_world=0.15, update_interval=5)
        for t in range(10):
            wf.record_round(
                p_agg=0.6, p_def=0.2, p_bal=0.2,
                avg_reward=0.4, event_type="Resource",
            )
            wf.maybe_update(t)
        # Not all weights should be 1.0 anymore
        assert any(abs(w - 1.0) > 0.001 for w in wf.event_weights.values())

    @pytest.mark.skipif(
        not EVENT_JSON.exists(),
        reason="event templates JSON not found",
    )
    def test_runtime_with_world_feedback(self):
        """Full runtime with world feedback enabled should complete and
        produce non-zero world state values."""
        cfg = PersonalityRLConfig(
            n_players=20,
            n_rounds=500,
            personality_mode="random",
            lambda_alpha=0.1,
            lambda_beta=0.1,
            events_json=str(EVENT_JSON),
            event_rate=0.5,
            event_reward_scale=0.01,
            world_feedback=True,
            lambda_world=0.15,
            world_update_interval=50,
        )
        result = run_personality_rl(cfg, seed=42)
        assert len(result.rows) == 500
        # World state should have deviated in at least some later rounds
        tail = result.rows[-100:]
        has_deviation = any(
            abs(r["world_scarcity"] - 0.5) > 0.001
            or abs(r["world_threat"] - 0.5) > 0.001
            for r in tail
        )
        assert has_deviation, "World state did not deviate with lambda_world=0.15"

    def test_no_world_feedback_stays_zero(self):
        """world_feedback=False → world channels stay at 0.0."""
        cfg = PersonalityRLConfig(
            n_players=20,
            n_rounds=50,
            personality_mode="none",
            world_feedback=False,
        )
        result = run_personality_rl(cfg, seed=42)
        for row in result.rows:
            assert row["world_scarcity"] == 0.0
            assert row["world_threat"] == 0.0
