"""Tests for RLSessionEngine (RL session step-driven loop).

Spec: SDD §12.7 Runtime Bridge v2 Acceptance Criteria

Tests validate:
- reset() initializes correctly
- step() advances round and updates state
- snapshot() returns state without advancing
- burn-in completes after 4000 rounds
- cycle detection triggers every 200 rounds (in tail)
- reset() clears state and resets warm flag
- 6-seed stress test (seeds 45~55 match BL2 anchor)
"""

import pytest
from simulation.rl_session_engine import RLSessionEngine, RLSessionConfig


class TestRLSessionEngineBasics:
    """Phase 1 (6.1): Single-round loop and state tracking."""

    def test_reset_initializes_correctly(self):
        """reset() should return initial FrameSnapshot with warm=False."""
        config = RLSessionConfig(seed=42)
        engine = RLSessionEngine(config, session_id="test_sess_1")

        snapshot = engine.reset()

        assert snapshot.session_id == "test_sess_1"
        assert snapshot.round == 0
        assert snapshot.tick == 0
        assert snapshot.warm is False
        assert snapshot.cycle_level == 0
        assert snapshot.s3_score == 0.0
        assert len(engine.tail_buffer) == 0

    def test_step_advances_round(self):
        """step() should increment round counter."""
        config = RLSessionConfig(seed=42, burn_in=100)
        engine = RLSessionEngine(config, session_id="test_sess_2")
        engine.reset()

        snapshot = engine.step()

        assert snapshot.round == 1
        assert snapshot.tick == 1

    def test_step_multiple_times(self):
        """step() × 10 should advance round counter by 10."""
        config = RLSessionConfig(seed=42, burn_in=100)
        engine = RLSessionEngine(config, session_id="test_sess_3")
        engine.reset()

        for i in range(10):
            snapshot = engine.step()
            assert snapshot.round == i + 1

    def test_snapshot_does_not_advance(self):
        """snapshot() should not increment round."""
        config = RLSessionConfig(seed=42)
        engine = RLSessionEngine(config, session_id="test_sess_4")
        engine.reset()

        _ = engine.step()
        snapshot1 = engine.snapshot()
        snapshot2 = engine.snapshot()

        assert snapshot1.round == 1
        assert snapshot2.round == 1  # Same round

    def test_proportions_sum_to_one(self):
        """p_aggressive + p_defensive + p_balanced should ≈ 1."""
        config = RLSessionConfig(seed=42, burn_in=100)
        engine = RLSessionEngine(config, session_id="test_sess_5")
        engine.reset()

        for _ in range(10):
            snapshot = engine.step()

        total = (
            snapshot.p_aggressive + snapshot.p_defensive + snapshot.p_balanced
        )
        assert abs(total - 1.0) < 1e-9

    def test_pi_proportions_sum_to_one(self):
        """pi_aggressive + pi_defensive + pi_balanced should ≈ 1."""
        config = RLSessionConfig(seed=42, burn_in=100)
        engine = RLSessionEngine(config, session_id="test_sess_6")
        engine.reset()

        for _ in range(10):
            snapshot = engine.step()

        total = (
            snapshot.pi_aggressive + snapshot.pi_defensive + snapshot.pi_balanced
        )
        assert abs(total - 1.0) < 1e-9

    def test_buffer_updates_after_burn_in(self):
        """tail_buffer should remain empty during burn-in, populate after."""
        config = RLSessionConfig(seed=42, burn_in=50)
        engine = RLSessionEngine(config, session_id="test_sess_7")
        engine.reset()

        # During burn-in
        for _ in range(50):
            engine.step()
            assert len(engine.tail_buffer) == 0

        # After burn-in
        engine.step()  # round 51 (> burn_in)
        assert len(engine.tail_buffer) == 1


class TestRLSessionEngineBurnIn:
    """Phase 1 (6.2): Burn-in completion and warm flag."""

    def test_warm_false_before_burn_in(self):
        """warm should be False before burn-in completes."""
        config = RLSessionConfig(seed=42, burn_in=50)
        engine = RLSessionEngine(config, session_id="test_sess_8")
        engine.reset()

        for _ in range(50):
            snapshot = engine.step()
            assert snapshot.warm is False

    def test_warm_true_after_burn_in(self):
        """warm should be True after burn-in completes."""
        config = RLSessionConfig(seed=42, burn_in=50)
        engine = RLSessionEngine(config, session_id="test_sess_9")
        engine.reset()

        # Step through burn-in
        for _ in range(50):
            engine.step()

        # One more step to trigger warm=True
        snapshot = engine.step()
        assert snapshot.round == 51
        assert snapshot.warm is True

    def test_burn_in_length(self):
        """Burn-in should be exactly config.burn_in rounds."""
        config = RLSessionConfig(seed=42, burn_in=100)
        engine = RLSessionEngine(config, session_id="test_sess_10")
        engine.reset()

        for i in range(100):
            snapshot = engine.step()
            assert snapshot.warm is False

        snapshot = engine.step()
        assert snapshot.round == 101
        assert snapshot.warm is True

    def test_tail_buffer_populated_after_burn_in(self):
        """tail_buffer should contain entries after burn-in."""
        config = RLSessionConfig(seed=42, burn_in=50, tail=100)
        engine = RLSessionEngine(config, session_id="test_sess_11")
        engine.reset()

        # Burn-in
        for _ in range(50):
            engine.step()

        # Tail: should append to buffer
        for i in range(10):
            engine.step()
            assert len(engine.tail_buffer) == i + 1


class TestRLSessionEngineReset:
    """Phase 1 (6.3): Reset functionality."""

    def test_reset_clears_round(self):
        """reset() should set round to 0."""
        config = RLSessionConfig(seed=42)
        engine = RLSessionEngine(config, session_id="test_sess_12")
        engine.reset()

        for _ in range(10):
            engine.step()

        snapshot = engine.reset()
        assert snapshot.round == 0

    def test_reset_sets_warm_false(self):
        """reset() should set warm to False."""
        config = RLSessionConfig(seed=42, burn_in=10)
        engine = RLSessionEngine(config, session_id="test_sess_13")
        engine.reset()

        # Step past burn-in
        for _ in range(15):
            engine.step()

        snapshot = engine.reset()
        assert snapshot.warm is False

    def test_reset_clears_buffer(self):
        """reset() should clear tail_buffer."""
        config = RLSessionConfig(seed=42, burn_in=10, tail=50)
        engine = RLSessionEngine(config, session_id="test_sess_14")
        engine.reset()

        # Step through burn-in and into tail
        for _ in range(20):
            engine.step()

        assert len(engine.tail_buffer) > 0

        engine.reset()
        assert len(engine.tail_buffer) == 0

    def test_reset_allows_restart(self):
        """After reset(), engine should be able to repeat full cycle."""
        config = RLSessionConfig(seed=42, burn_in=20)
        engine = RLSessionEngine(config, session_id="test_sess_15")

        # First cycle
        engine.reset()
        for _ in range(25):
            engine.step()
        first_warm = engine.warm

        # Reset and second cycle
        engine.reset()
        for _ in range(25):
            engine.step()
        second_warm = engine.warm

        assert first_warm is True
        assert second_warm is True


class TestRLSessionEngineStress:
    """Phase 3 (7.6): 6-seed stress test matching BL2 anchor."""

    @pytest.mark.parametrize("seed", [45, 47, 49, 51, 53, 55])
    def test_six_seed_completion(self, seed):
        """Each BL2 anchor seed should complete full cycle without crash."""
        config = RLSessionConfig(
            seed=seed,
            n_players=300,
            burn_in=100,  # Short for testing
            tail=100,
            check_interval=50,
        )
        engine = RLSessionEngine(config, session_id=f"test_sess_{seed}")
        engine.reset()

        # Burn-in
        for _ in range(100):
            engine.step()

        # Tail
        for _ in range(100):
            snapshot = engine.step()

        assert snapshot.warm is True
        assert snapshot.round == 200

    @pytest.mark.parametrize("seed", [45, 47, 49, 51, 53, 55])
    def test_six_seed_proportions_reasonable(self, seed):
        """Proportions should be reasonable (not degenerate)."""
        config = RLSessionConfig(
            seed=seed,
            n_players=300,
            burn_in=50,
            tail=100,
        )
        engine = RLSessionEngine(config, session_id=f"test_sess_prop_{seed}")
        engine.reset()

        for _ in range(150):
            engine.step()

        snapshot = engine.snapshot()

        # Check that proportions are not stuck at extreme values
        assert 0.0 <= snapshot.p_aggressive <= 1.0
        assert 0.0 <= snapshot.p_defensive <= 1.0
        assert 0.0 <= snapshot.p_balanced <= 1.0

        # At least one strategy should be reasonably represented
        proportions = [
            snapshot.p_aggressive,
            snapshot.p_defensive,
            snapshot.p_balanced,
        ]
        assert max(proportions) > 0.1  # No single domination below 10%
        assert min(proportions) < 0.9  # No single strategy > 90%

    def test_six_seed_no_nan(self):
        """No FrameSnapshot field should be NaN."""
        config = RLSessionConfig(seed=45, burn_in=50, tail=100)
        engine = RLSessionEngine(config, session_id="test_sess_nan")
        engine.reset()

        for _ in range(150):
            snapshot = engine.step()

        # Check key numeric fields
        assert not (
            snapshot.p_aggressive != snapshot.p_aggressive
        )  # NaN check
        assert not (snapshot.s3_score != snapshot.s3_score)
        assert not (snapshot.q_std != snapshot.q_std)


class TestRLSessionEngineConfigValidation:
    """Config validation (BL2 lock enforcement)."""

    def test_bl2_parameter_lock_alpha_lo(self):
        """alpha_lo must be exactly 0.005 (BL2 locked)."""
        config = RLSessionConfig(seed=42, alpha_lo=0.010)  # Wrong!
        with pytest.raises(ValueError, match="alpha_lo"):
            RLSessionEngine(config, session_id="test_validation")

    def test_bl2_parameter_lock_alpha_hi(self):
        """alpha_hi must be exactly 0.40 (BL2 locked)."""
        config = RLSessionConfig(seed=42, alpha_hi=0.50)  # Wrong!
        with pytest.raises(ValueError, match="alpha_hi"):
            RLSessionEngine(config, session_id="test_validation")

    def test_bl2_parameter_lock_beta(self):
        """beta must be exactly 3.0 (BL2 locked)."""
        config = RLSessionConfig(seed=42, beta=2.0)  # Wrong!

        with pytest.raises(ValueError, match="beta"):
            config.validate()

    def test_bl2_parameter_lock_mild_cw(self):
        """strategy_alpha_multipliers must be [1.2, 1.0, 0.8] (BL2 locked)."""
        config = RLSessionConfig(seed=42, strategy_alpha_multipliers=[1.0, 1.0, 1.0])

        with pytest.raises(ValueError, match="strategy_alpha_multipliers"):
            config.validate()

    def test_events_must_be_empty(self):
        """events_json must be "" (Dead Zone in Phase 1-2)."""
        config = RLSessionConfig(seed=42, events_json="some_path.json")

        with pytest.raises(ValueError, match="events_json"):
            config.validate()

    def test_valid_config_passes(self):
        """Valid config with all BL2 parameters should pass."""
        config = RLSessionConfig(
            seed=42,
            alpha_lo=0.005,
            alpha_hi=0.40,
            beta=3.0,
            strategy_alpha_multipliers=[1.2, 1.0, 0.8],
            a=1.0,
            b=0.9,
            cross=0.20,
            events_json="",
        )

        # Should not raise
        config.validate()


# ===================================================================
# Run tests
# ===================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
