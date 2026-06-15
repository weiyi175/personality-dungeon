"""Unit tests for the P7-H real-study managers.

Covers ABTestManager, PlayerTestTracker, and SurveyManager — in particular the
save()/load() round-trips that back the server's restart-tolerant persistence
(see api/server.py _lifespan / _p7h_save). These managers are in-memory
singletons in production, so a restart mid-study relies on load() restoring the
exact prior state (count balance, trajectories, responses).
"""

from __future__ import annotations

import numpy as np
import pytest

from api.ab_test_manager import ABTestManager, SessionRecord
from api.player_test_tracker import PlayerTestTracker
from api.survey_manager import SurveyManager


# ===================================================================
# ABTestManager
# ===================================================================


def test_ab_assign_is_idempotent():
    mgr = ABTestManager()
    first = mgr.assign_session("s1")
    second = mgr.assign_session("s1")
    assert first["existing"] is False
    assert second["existing"] is True
    assert first["group"] == second["group"]


def test_ab_assign_count_balances_groups():
    """Each new session goes to the smaller arm → arms stay within 1 of each other."""
    mgr = ABTestManager()
    for i in range(20):
        mgr.assign_session(f"s{i}")
    counts = mgr._group_counts()
    assert counts["control"] + counts["experiment"] == 20
    assert abs(counts["control"] - counts["experiment"]) <= 1
    # 20 is even → exactly balanced
    assert counts["control"] == counts["experiment"] == 10


def test_ab_record_event_step_tracks_displacement():
    mgr = ABTestManager()
    mgr.assign_session("s1")
    p0 = [0.0] * 9
    p1 = [1.0] + [0.0] * 8
    mgr.record_event_step("s1", p0, p1, proximity=0.5, step_index=0)
    rec = mgr.get_session("s1")
    assert rec["event_count"] == 1
    assert rec["max_proximity"] == 0.5
    assert rec["total_displacement"] == pytest.approx(1.0)
    assert rec["initial_personality"] == p0
    assert rec["final_personality"] == p1


def test_ab_record_event_step_unknown_session_is_noop():
    mgr = ABTestManager()
    # Should not raise, just ignore
    mgr.record_event_step("ghost", [0.0] * 9, [0.0] * 9, proximity=0.1, step_index=0)
    assert mgr.get_session("ghost") is None


def test_ab_save_load_roundtrip(tmp_path):
    mgr = ABTestManager(seed=7)
    mgr.assign_session("s1")
    mgr.assign_session("s2")
    mgr.record_event_step("s1", [0.0] * 9, [1.0] + [0.0] * 8, proximity=0.9, step_index=0)
    mgr.save(tmp_path)

    restored = ABTestManager()
    assert restored.load(tmp_path) is True
    assert restored._group_counts() == mgr._group_counts()
    assert restored.get_session("s1") == mgr.get_session("s1")
    assert restored.get_session("s2") == mgr.get_session("s2")
    # Records rehydrate as dataclasses, not dicts
    assert isinstance(restored._sessions["s1"], SessionRecord)


def test_ab_load_missing_file_returns_false(tmp_path):
    assert ABTestManager().load(tmp_path) is False


def _pilot_tally(mgr):
    from api.ab_test_manager import _pid_is_pilot_eligible
    c = {"iterated": 0, "reset": 0}
    for rid, a in mgr._iteration_runs.items():
        if _pid_is_pilot_eligible(mgr._iteration_pid.get(rid, "dev")):
            c[a] += 1
    return c


def test_iteration_arm_balance_counts_only_pilot_eligible():
    """count-balance tally must count only naive (P\\d{2,}) runs, not dev/EXP_PREPILOT.

    Guards the contamination fix: interleaved experimenter (dev) playtests and the
    quarantined pre-pilot runs must NOT skew the naive cohort's arm assignment.
    """
    mgr = ABTestManager()
    # Simulate quarantined pre-pilot pool (skewed 2 iterated / 1 reset) with no pid
    # recorded — exactly how the live EXP_PREPILOT runs load (default "dev" → ineligible).
    mgr._iteration_runs = {"pp1": "iterated", "pp2": "iterated", "pp3": "reset"}
    assert _pilot_tally(mgr) == {"iterated": 0, "reset": 0}  # pre-pilot excluded

    # (a) a dev playtest receives an arm but never enters the pilot tally
    mgr.assign_session("s_dev", "run_dev", "dev")
    assert mgr._iteration_runs["run_dev"] in ("iterated", "reset")
    assert _pilot_tally(mgr) == {"iterated": 0, "reset": 0}

    # (b) P01 cold-start: empty pilot pool → arm assigned, sticky across its cycles
    arm = mgr.assign_session("s_p01c0", "run_P01", "P01")["iteration_arm"]
    assert arm in ("iterated", "reset")
    assert mgr.assign_session("s_p01c1", "run_P01", "P01")["iteration_arm"] == arm  # sticky

    # interleave more dev + naive; naive cohort self-balances regardless of dev noise
    for pid, rid in [("P02", "run_P02"), ("dev", "run_dev2"), ("P03", "run_P03"),
                     ("P04", "run_P04"), ("dev", "run_dev3"), ("P05", "run_P05"),
                     ("P06", "run_P06")]:
        mgr.assign_session("s_" + rid, rid, pid)
    t = _pilot_tally(mgr)
    assert t["iterated"] + t["reset"] == 6           # 6 naive runs counted, dev ignored
    assert abs(t["iterated"] - t["reset"]) <= 1      # balanced among naive only


def test_iteration_pid_save_load_roundtrip(tmp_path):
    mgr = ABTestManager()
    mgr.assign_session("s_p01", "run_P01", "P01")
    mgr.assign_session("s_dev", "run_dev", "dev")
    mgr.save(tmp_path)
    restored = ABTestManager()
    assert restored.load(tmp_path) is True
    assert restored._iteration_pid == mgr._iteration_pid
    assert restored._iteration_runs == mgr._iteration_runs


def test_ab_summary_reports_effect_size():
    mgr = ABTestManager()
    # Force one session into each group, give experiment a larger displacement.
    mgr._sessions["c1"] = SessionRecord(session_id="c1", group="control")
    mgr._sessions["e1"] = SessionRecord(session_id="e1", group="experiment")
    mgr.record_event_step("c1", [0.0] * 9, [0.1] + [0.0] * 8, proximity=0.2, step_index=0)
    mgr.record_event_step("e1", [0.0] * 9, [0.9] + [0.0] * 8, proximity=0.8, step_index=0)
    summary = mgr.summary()
    assert summary["total_sessions"] == 2
    assert summary["control"]["n"] == 1
    assert summary["experiment"]["n"] == 1
    assert "effect_size_cohens_d" in summary


# ===================================================================
# PlayerTestTracker
# ===================================================================


def test_player_start_rejects_duplicate():
    tracker = PlayerTestTracker()
    assert tracker.start_session("s1", "experiment")["ok"] is True
    dup = tracker.start_session("s1", "experiment")
    assert dup["ok"] is False


def test_player_record_step_requires_session():
    tracker = PlayerTestTracker()
    res = tracker.record_step("ghost", "a", [0.0] * 9, [0.0] * 9, 0.0, 0.0)
    assert res["ok"] is False


def test_player_end_orders_out_of_order_steps_by_timestamp():
    """The DV (total_displacement) must use timestamp order, not arrival order.

    Steps are posted fire-and-forget, so they can land out of order. end_session
    sorts by timestamp before computing endpoint displacement (P7-H DV bug fix).
    """
    tracker = PlayerTestTracker()
    tracker.start_session("s1", "experiment")
    sess = tracker._sessions["s1"]

    # Manually append three steps with shuffled timestamps but a clear true order.
    from api.player_test_tracker import TrajectoryStep

    steps = [
        TrajectoryStep(2, 200.0, "c", [0.5] * 9, [1.0] * 9, 0.5, 0.9, "shift", 100.0),
        TrajectoryStep(0, 100.0, "a", [0.0] * 9, [0.2] * 9, 0.0, 0.2, "shift", 100.0),
        TrajectoryStep(1, 150.0, "b", [0.2] * 9, [0.5] * 9, 0.2, 0.5, "shift", 100.0),
    ]
    sess.trajectory = steps  # deliberately out of order

    result = tracker.end_session("s1")
    # After ordering: p0 = first step's before ([0]*9), pf = last step's after ([1]*9)
    expected = float(np.linalg.norm(np.array([1.0] * 9) - np.array([0.0] * 9)))
    assert result["total_displacement"] == pytest.approx(expected)
    # step_index reassigned to timestamp order
    assert [s.step_index for s in tracker._sessions["s1"].trajectory] == [0, 1, 2]


def test_player_counts_critical_crossings_and_events():
    tracker = PlayerTestTracker()
    tracker.start_session("s1", "control")
    # crossing: proximity_before < 0.8 <= proximity_after
    tracker.record_step("s1", "x", [0.0] * 9, [0.1] * 9, 0.7, 0.85, event_type="shift")
    tracker.record_step("s1", "y", [0.1] * 9, [0.2] * 9, 0.85, 0.9, event_type="none")
    sess = tracker._sessions["s1"]
    assert sess.n_critical_crossings == 1
    assert sess.n_bifurcation_events == 1  # only the "shift" one counts


def test_player_save_load_roundtrip(tmp_path):
    tracker = PlayerTestTracker()
    tracker.start_session("s1", "experiment", player_alias="p7")
    tracker.record_step("s1", "act", [0.0] * 9, [0.3] * 9, 0.1, 0.4, event_type="shift")
    tracker.end_session("s1")
    tracker.save(tmp_path)

    restored = PlayerTestTracker()
    assert restored.load(tmp_path) is True
    assert restored.get_session("s1") == tracker.get_session("s1")


# ===================================================================
# SurveyManager
# ===================================================================


def test_survey_submit_validates_range():
    survey = SurveyManager()
    bad = survey.submit("s1", "control", q1=11, q2=5, q3=5)   # 11 > 10 still rejected
    assert bad["ok"] is False
    good = survey.submit("s1", "control", q1=8, q2=7, q3=9)
    assert good["ok"] is True


def test_survey_submit_accepts_zero_for_removed_q1_q2():
    """q1_naturalness/q2_fun removed from the UI (2026-06-13) → frontend sends 0.

    Regression: the trimmed survey sends q1=0/q2=0; submit() must accept it (q3 is
    the only mandatory scale question now). Previously 0 failed the 1–10 check → 422
    → surveys silently never persisted while the UI still printed "問卷完成".
    """
    survey = SurveyManager()
    r = survey.submit("s1", "experiment", q1=0, q2=0, q3=5, q4=6, manipulation_awareness="")
    assert r["ok"] is True
    resp = survey.get_response("s1")
    assert resp["q3_replay"] == 5 and resp["q4_continuity"] == 6
    # q3 still mandatory; 0 (unanswered) must be rejected
    assert survey.submit("s2", "experiment", q1=0, q2=0, q3=0)["ok"] is False


def test_survey_truncates_long_comments():
    survey = SurveyManager()
    survey.submit("s1", "control", 5, 5, 5, q1_comment="x" * 999)
    resp = survey.get_response("s1")
    assert len(resp["q1_comment"]) == 500


def test_survey_summary_composite_and_lift():
    survey = SurveyManager()
    survey.submit("c1", "control", 4, 4, 4)
    survey.submit("e1", "experiment", 8, 8, 8)
    summary = survey.summary()
    assert summary["total_responses"] == 2
    # composite_ux = mean over all individual item scores (not the sum)
    assert summary["control"]["composite_ux"] == pytest.approx(4.0)
    assert summary["experiment"]["composite_ux"] == pytest.approx(8.0)
    assert summary["ux_lift"] == pytest.approx(4.0)


def test_survey_save_load_roundtrip(tmp_path):
    survey = SurveyManager()
    survey.submit("s1", "experiment", 9, 8, 10, overall_comments="great")
    survey.save(tmp_path)

    restored = SurveyManager()
    assert restored.load(tmp_path) is True
    assert restored.get_response("s1") == survey.get_response("s1")
