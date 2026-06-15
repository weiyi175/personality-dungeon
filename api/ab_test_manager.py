"""
A/B test session manager for bifurcation event experiments.

Group assignment
----------------
  control    : Events chosen with a random direction (baseline)
  experiment : Events aligned with the sensitive direction v1 (bifurcation-optimised)

Each session_id is deterministically assigned based on a hash of the id so that
repeated lookups are stable and reproducible. The seed is configurable so the
experimenter can re-balance groups by changing it.

Tracked per session
-------------------
  - group assignment
  - personality snapshots (before/after each event step)
  - maximum proximity reached
  - total displacement (||P_final - P_initial||)
  - event count

The manager is intentionally in-process (no database). Restart-tolerant storage
is out of scope for Phase III; results are serialised to JSON on demand.
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Literal

import numpy as np

Group = Literal["control", "experiment"]

# naive 受試者代碼樣式（P01, P02, …）。實驗者試玩 "dev" / pre-pilot "EXP_PREPILOT" 都不符。
# 必須與 analyzer (analyze_iteration_study.py PILOT_PID_PATTERN) 同一套判別子：
# count-balance 配臂計數與分析納入用同源規則，naive cohort 才會在自己內部平衡，
# 且交錯的 dev 試玩不污染配臂計數（dev run 仍配到臂、但不計入平衡 tally）。
PILOT_PID_PATTERN = re.compile(r"^P\d{2,}$")


def _pid_is_pilot_eligible(participant_id: str) -> bool:
    return bool(PILOT_PID_PATTERN.match(participant_id or ""))


@dataclass
class SessionRecord:
    session_id: str
    group: Group
    created_at: float = field(default_factory=time.time)
    personality_snapshots: list[dict] = field(default_factory=list)
    max_proximity: float = 0.0
    initial_personality: list[float] | None = None
    final_personality: list[float] | None = None
    total_displacement: float = 0.0
    event_count: int = 0


class ABTestManager:
    """In-memory A/B test session manager."""

    def __init__(self, seed: int = 0, target_per_group: int = 106) -> None:
        self._seed = seed
        self._target_per_group = target_per_group
        self._sessions: dict[str, SessionRecord] = {}
        # 人格迭代研究：run_id → "iterated"|"reset"（sticky，跨同一受試者的 3 週期）。
        # 與上面的 control/experiment 事件方向分組正交、獨立計數平衡。
        self._iteration_runs: dict[str, str] = {}
        # run_id → participant_id（配臂時帶入）。count-balance 平衡 tally 只數 pilot-eligible
        # 的 run（participant_id 配 ^P\d{2,}$）；dev/EXP_PREPILOT 仍記錄、仍配到臂，但不計入
        # 平衡，故交錯 dev 試玩不污染 naive cohort 的配臂平衡。
        self._iteration_pid: dict[str, str] = {}

    # ── Session management ────────────────────────────────────────────────────

    def _group_counts(self) -> dict[str, int]:
        counts = {"control": 0, "experiment": 0}
        for rec in self._sessions.values():
            counts[rec.group] += 1
        return counts

    def assign_session(self, session_id: str, run_id: str = "",
                       participant_id: str = "dev") -> dict:
        """Assign or retrieve the A/B group for a session.

        Count-balanced: each new session goes to the currently smaller group so
        the two arms stay equal (guaranteeing 106/106 at N=212). Ties are broken
        with a deterministic sha256(session_id + seed) hash so the assignment is
        reproducible given the same arrival order. Re-looking up an existing
        session always returns its original group (idempotent).

        Personality-iteration study: when ``run_id`` is supplied the caller is in
        the iteration study. Event direction is fixed to "experiment" (decision B)
        and a sticky, count-balanced ``iteration_arm`` ("iterated"|"reset") is
        returned for that run_id (stable across the run's 3 cycles). Legacy
        callers (no run_id) are completely unaffected.
        """
        if run_id:
            existing = run_id in self._iteration_runs
            arm = self._assign_iteration_arm(run_id, participant_id)
            return {
                "session_id": session_id,
                "group": "experiment",
                "iteration_arm": arm,
                "run_id": run_id,
                "existing": existing,
            }

        if session_id in self._sessions:
            rec = self._sessions[session_id]
            return {"session_id": session_id, "group": rec.group, "existing": True}

        counts = self._group_counts()
        if counts["control"] < counts["experiment"]:
            group: Group = "control"
        elif counts["experiment"] < counts["control"]:
            group = "experiment"
        else:
            # Tie: deterministic hash so equal-arrival states are reproducible.
            digest = hashlib.sha256(f"{session_id}:{self._seed}".encode()).hexdigest()
            group = "experiment" if int(digest[0], 16) % 2 == 0 else "control"

        self._sessions[session_id] = SessionRecord(
            session_id=session_id, group=group
        )
        return {"session_id": session_id, "group": group, "existing": False}

    def _assign_iteration_arm(self, run_id: str, participant_id: str = "dev") -> str:
        """Sticky, count-balanced iteration arm for a participant run.

        Balanced across *pilot-eligible runs* (naive participants P\\d{2,}), not
        cycles and not experimenter/pre-pilot runs: the smaller arm gets the next
        new run; ties broken by deterministic hash. Re-looking up a known run_id
        returns its original arm (idempotent across the 3 cycles).

        The balance tally counts ONLY pilot-eligible runs (same discriminator as
        the analyzer). dev / EXP_PREPILOT runs still receive a (hash-broken) arm so
        playtests work, but they do NOT count toward the naive cohort's balance —
        so interleaved dev playtests never skew P-code assignment.
        """
        if run_id in self._iteration_runs:
            return self._iteration_runs[run_id]
        self._iteration_pid[run_id] = participant_id
        counts = {"iterated": 0, "reset": 0}
        for rid, a in self._iteration_runs.items():
            if _pid_is_pilot_eligible(self._iteration_pid.get(rid, "dev")):
                counts[a] = counts.get(a, 0) + 1
        if counts["iterated"] < counts["reset"]:
            arm = "iterated"
        elif counts["reset"] < counts["iterated"]:
            arm = "reset"
        else:
            digest = hashlib.sha256(f"{run_id}:{self._seed}:iter".encode()).hexdigest()
            arm = "iterated" if int(digest[0], 16) % 2 == 0 else "reset"
        self._iteration_runs[run_id] = arm
        return arm

    def get_session(self, session_id: str) -> dict | None:
        rec = self._sessions.get(session_id)
        return asdict(rec) if rec else None

    def record_event_step(
        self,
        session_id: str,
        personality_before: list[float],
        personality_after: list[float],
        proximity: float,
        step_index: int,
    ) -> None:
        rec = self._sessions.get(session_id)
        if rec is None:
            return
        if rec.initial_personality is None:
            rec.initial_personality = personality_before[:]

        rec.personality_snapshots.append({
            "step": step_index,
            "before": personality_before[:],
            "after": personality_after[:],
            "proximity": proximity,
        })
        rec.max_proximity = max(rec.max_proximity, proximity)
        rec.final_personality = personality_after[:]
        rec.event_count += 1

        # Update total displacement from initial
        if rec.initial_personality:
            p0 = np.array(rec.initial_personality)
            p1 = np.array(personality_after)
            rec.total_displacement = float(np.linalg.norm(p1 - p0))

    # ── Analytics ─────────────────────────────────────────────────────────────

    def summary(self) -> dict:
        """Compute group-level statistics."""
        groups: dict[str, list[SessionRecord]] = {"control": [], "experiment": []}
        for rec in self._sessions.values():
            groups[rec.group].append(rec)

        def _stats(records: list[SessionRecord]) -> dict:
            if not records:
                return {"n": 0}
            displacements = [r.total_displacement for r in records if r.event_count > 0]
            max_prox = [r.max_proximity for r in records if r.event_count > 0]
            return {
                "n": len(records),
                "n_with_events": len(displacements),
                "mean_displacement": float(np.mean(displacements)) if displacements else 0.0,
                "std_displacement": float(np.std(displacements)) if displacements else 0.0,
                "mean_max_proximity": float(np.mean(max_prox)) if max_prox else 0.0,
                "mean_event_count": float(np.mean([r.event_count for r in records])),
            }

        ctrl_stats = _stats(groups["control"])
        exp_stats = _stats(groups["experiment"])

        # Effect size (Cohen's d) on displacement
        cohen_d = 0.0
        if (ctrl_stats.get("n_with_events", 0) > 0
                and exp_stats.get("n_with_events", 0) > 0):
            pooled_std = (
                (ctrl_stats["std_displacement"] + exp_stats["std_displacement"]) / 2
            )
            if pooled_std > 0:
                cohen_d = (
                    (exp_stats["mean_displacement"] - ctrl_stats["mean_displacement"])
                    / pooled_std
                )

        return {
            "total_sessions": len(self._sessions),
            "control": ctrl_stats,
            "experiment": exp_stats,
            "effect_size_cohens_d": round(cohen_d, 4),
        }

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self, out_dir: str | Path) -> Path:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        path = out / "ab_test_sessions.json"
        payload = {
            "seed": self._seed,
            "sessions": {sid: asdict(rec) for sid, rec in self._sessions.items()},
            "iteration_runs": self._iteration_runs,
            "iteration_pid": self._iteration_pid,
            "summary": self.summary(),
        }
        with open(path, "w") as f:
            json.dump(payload, f, indent=2)
        return path

    def load(self, out_dir: str | Path) -> bool:
        """Restore sessions (and group counts) from a prior save().

        Returns True if a file was found and loaded. Used on server startup so a
        restart mid-study does not reset the count-balance or lose assignments.
        """
        path = Path(out_dir) / "ab_test_sessions.json"
        if not path.exists():
            return False
        with open(path) as f:
            payload = json.load(f)
        self._sessions = {
            sid: SessionRecord(**rec)
            for sid, rec in payload.get("sessions", {}).items()
        }
        self._iteration_runs = dict(payload.get("iteration_runs", {}))
        # 既有存檔無 iteration_pid（如目前 15 個 EXP_PREPILOT）→ get 預設 "dev" → 不計入平衡。
        self._iteration_pid = dict(payload.get("iteration_pid", {}))
        return True


# ── Module-level singleton (shared across API requests) ──────────────────────
_manager: ABTestManager | None = None


def get_manager(seed: int = 0) -> ABTestManager:
    global _manager
    if _manager is None:
        _manager = ABTestManager(seed=seed)
    return _manager
