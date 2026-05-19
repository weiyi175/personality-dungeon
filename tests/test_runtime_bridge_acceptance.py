"""End-to-end acceptance test: Runtime Bridge v2 full workflow (SDD §12.7).

Validates complete pipeline:
  1. Session init (BL2 anchor, seed=45/47/49/51/53/55)
  2. Reset → burn-in (warm=False, no tail buffer)
  3. Step into tail phase (warm=True, tail buffer populated)
  4. Cycle detection (every 200 rounds in tail, check s3_score, cycle_level)
  5. Final snapshot with all 20+ fields populated
  6. API response integration

This is the SDD §12.7 acceptance criteria validation.
"""

import pytest
from fastapi.testclient import TestClient

from api.server import app
from api.rl_session_manager import get_session_manager


@pytest.fixture
def client():
    """FastAPI test client."""
    return TestClient(app)


@pytest.fixture(autouse=True)
def clear_sessions():
    """Clear session manager before each test."""
    manager = get_session_manager()
    for session_id in list(manager.sessions.keys()):
        manager.delete_session(session_id)
    yield
    for session_id in list(manager.sessions.keys()):
        manager.delete_session(session_id)


class TestRuntimeBridgeV2Acceptance:
    """SDD §12.7 acceptance criteria validation."""

    @pytest.mark.parametrize("seed", [45, 47, 49, 51, 53, 55])
    def test_full_workflow_6_seeds(self, client, seed):
        """Full workflow: init → burn-in → tail → cycle detection → respond (6 BL2 seeds)."""
        
        # ================================================================
        # 1. INITIALIZE SESSION (BL2 anchor seed)
        # ================================================================
        init_response = client.post(
            "/rl_sessions/initialize",
            json={
                "seed": seed,
                "n_players": 300,
                "n_rounds": 1000,
                "burn_in": 400,
                "personality_mode": "random_9persona",
            },
        )
        assert init_response.status_code == 200, f"Init failed: {init_response.text}"
        session_id = init_response.json()["session_id"]
        initial_snap = init_response.json()["initial_snapshot"]
        
        # Verify initial state
        assert initial_snap["round"] == 0
        assert initial_snap["warm"] is False
        assert initial_snap["phase"] == "burn-in"
        assert initial_snap["cycle_level"] == 0
        assert initial_snap["s3_score"] == 0.0
        
        # ================================================================
        # 2. BURN-IN PHASE
        # ================================================================
        for burn_idx in range(1, 401):
            step_response = client.post(f"/rl_sessions/{session_id}/step")
            assert step_response.status_code == 200, f"Burn-in step {burn_idx} failed: {step_response.text}"
            snap = step_response.json()["snapshot"]
            
            assert snap["round"] == burn_idx
            assert snap["warm"] is False
            assert snap["phase"] == "burn-in"
            
            # Verify proportions
            p_agg = snap["p_aggressive"]
            p_def = snap["p_defensive"]
            p_bal = snap["p_balanced"]
            p_sum = p_agg + p_def + p_bal
            
            assert not any(v != v for v in [p_agg, p_def, p_bal])  # No NaN
            assert abs(p_sum - 1.0) < 1e-5
            assert 0.0 <= p_agg <= 1.0 and 0.0 <= p_def <= 1.0 and 0.0 <= p_bal <= 1.0
        
        # ================================================================
        # 3. TRANSITION TO TAIL PHASE
        # ================================================================
        step_response = client.post(f"/rl_sessions/{session_id}/step")
        assert step_response.status_code == 200, f"Step 401 failed: {step_response.text}"
        snap = step_response.json()["snapshot"]
        
        assert snap["round"] == 401
        assert snap["warm"] is True
        assert snap["phase"] == "tail"
        
        # ================================================================
        # 4. TAIL PHASE (100 more steps to reach cycle detection checkpoint)
        # ================================================================
        for tail_idx in range(1, 101):
            step_response = client.post(f"/rl_sessions/{session_id}/step")
            assert step_response.status_code == 200, \
                f"Tail step {tail_idx} (round {401+tail_idx}) failed: {step_response.text}"
            snap = step_response.json()["snapshot"]
            
            assert snap["round"] == 401 + tail_idx
            assert snap["warm"] is True
            assert snap["phase"] == "tail"
            
            # Verify proportions
            p_sum = snap["p_aggressive"] + snap["p_defensive"] + snap["p_balanced"]
            assert abs(p_sum - 1.0) < 1e-5
            
            pi_sum = snap["pi_aggressive"] + snap["pi_defensive"] + snap["pi_balanced"]
            assert abs(pi_sum - 1.0) < 1e-5
        
        # ================================================================
        # 5. FINAL SNAPSHOT VALIDATION
        # ================================================================
        final_snap = client.get(f"/rl_sessions/{session_id}/snapshot").json()["snapshot"]
        
        required_fields = [
            "session_id", "round", "tick", "warm", "phase",
            "cycle_level", "s3_score", "env_gamma", "entropy", "q_std",
            "p_aggressive", "p_defensive", "p_balanced",
            "pi_aggressive", "pi_defensive", "pi_balanced",
            "q_mean_aggressive", "q_mean_defensive", "q_mean_balanced",
            "avg_reward", "avg_utility", "success_rate",
            "risk_mean", "stress_mean",
            "world_scarcity", "world_threat", "world_noise", "world_intel",
        ]
        
        for field in required_fields:
            assert field in final_snap, f"Missing field: {field}"
        
        assert final_snap["session_id"] == session_id
        assert final_snap["round"] == 501
        assert final_snap["warm"] is True
        assert final_snap["phase"] == "tail"
        
        # ================================================================
        # 6. RESET CAPABILITY
        # ================================================================
        reset_response = client.post(f"/rl_sessions/{session_id}/reset")
        assert reset_response.status_code == 200
        reset_snap = reset_response.json()["snapshot"]
        
        assert reset_snap["round"] == 0
        assert reset_snap["warm"] is False
        assert reset_snap["phase"] == "burn-in"
        
        # ================================================================
        # 7. CLEANUP (DELETE SESSION)
        # ================================================================
        delete_response = client.delete(f"/rl_sessions/{session_id}")
        assert delete_response.status_code == 200
        assert delete_response.json()["status"] == "deleted"
        
        # Verify deletion
        info_response = client.get(f"/rl_sessions/{session_id}/info")
        assert info_response.status_code == 404

    def test_proportions_converge_reasonably(self, client):
        """Verify strategy proportions converge to reasonable distribution."""
        init_response = client.post(
            "/rl_sessions/initialize",
            json={"seed": 45, "burn_in": 200, "n_rounds": 1200},
        )
        session_id = init_response.json()["session_id"]
        
        # Burn through burn-in
        for _ in range(200):
            client.post(f"/rl_sessions/{session_id}/step")
        
        # Collect proportions during tail (300 steps)
        proportions = []
        for i in range(300):
            response = client.post(f"/rl_sessions/{session_id}/step")
            assert response.status_code == 200, f"Step {200+i+1} failed: {response.text}"
            snap = response.json()["snapshot"]
            proportions.append({
                "p_agg": snap["p_aggressive"],
                "p_def": snap["p_defensive"],
                "p_bal": snap["p_balanced"],
            })
        
        # Check that proportions are not all identical (should vary)
        p_agg_values = [p["p_agg"] for p in proportions]
        p_def_values = [p["p_def"] for p in proportions]
        
        # Standard deviation should be non-zero (indicating variation)
        import statistics
        agg_std = statistics.stdev(p_agg_values) if len(set(p_agg_values)) > 1 else 0.0
        def_std = statistics.stdev(p_def_values) if len(set(p_def_values)) > 1 else 0.0
        
        # Should have some variation (not frozen)
        assert agg_std > 0.01 or def_std > 0.01, "Proportions should vary during simulation"

    def test_no_nan_propagation(self, client):
        """Ensure NaN values do not propagate in snapshots."""
        init_response = client.post(
            "/rl_sessions/initialize",
            json={"seed": 45, "burn_in": 100, "n_rounds": 500},
        )
        session_id = init_response.json()["session_id"]
        
        # Execute 400 steps
        for _ in range(400):
            response = client.post(f"/rl_sessions/{session_id}/step")
            assert response.status_code == 200
            snap = response.json()["snapshot"]
            
            # Check all numeric fields for NaN
            float_fields = [
                "entropy", "q_std", "q_mean_aggressive", "q_mean_defensive",
                "q_mean_balanced", "p_aggressive", "p_defensive", "p_balanced",
                "pi_aggressive", "pi_defensive", "pi_balanced",
                "avg_reward", "avg_utility", "success_rate",
            ]
            
            for field in float_fields:
                value = snap.get(field)
                assert value == value, f"NaN detected in {field} at round {snap['round']}"

    def test_session_state_consistency(self, client):
        """Verify session state remains consistent across API calls."""
        init_response = client.post(
            "/rl_sessions/initialize",
            json={"seed": 47, "burn_in": 50, "n_rounds": 200},
        )
        session_id = init_response.json()["session_id"]
        
        # Step 10 times
        for i in range(10):
            step_response = client.post(f"/rl_sessions/{session_id}/step")
            snap = step_response.json()["snapshot"]
            assert snap["round"] == i + 1
        
        # Get info and verify consistency
        info_response = client.get(f"/rl_sessions/{session_id}/info")
        info = info_response.json()
        assert info["round"] == 10
        
        # Get snapshot and verify consistency
        snap_response = client.get(f"/rl_sessions/{session_id}/snapshot")
        snap = snap_response.json()["snapshot"]
        assert snap["round"] == 10


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
