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
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Literal

import numpy as np

Group = Literal["control", "experiment"]

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

    # ── Session management ────────────────────────────────────────────────────

    def _group_counts(self) -> dict[str, int]:
        counts = {"control": 0, "experiment": 0}
        for rec in self._sessions.values():
            counts[rec.group] += 1
        return counts

    def assign_session(self, session_id: str) -> dict:
        """Assign or retrieve the A/B group for a session.

        Count-balanced: each new session goes to the currently smaller group so
        the two arms stay equal (guaranteeing 106/106 at N=212). Ties are broken
        with a deterministic sha256(session_id + seed) hash so the assignment is
        reproducible given the same arrival order. Re-looking up an existing
        session always returns its original group (idempotent).

        Every session is assigned by session_id alone. The former run_id special
        case (force "experiment" + sticky iteration arm) was removed on 2026-06-18
        when the personality-iteration study was abandoned and lives decoupled —
        a multi-cycle playthrough's cycles are now balanced independently like any
        other session. run_id is purely a linking field recorded downstream.
        """
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
        # 既有存檔可能含 run_ids / iteration_runs（已棄用）→ 忽略，分組純看 sessions。
        return True


# ── Module-level singleton (shared across API requests) ──────────────────────
_manager: ABTestManager | None = None


def get_manager(seed: int = 0) -> ABTestManager:
    global _manager
    if _manager is None:
        _manager = ABTestManager(seed=seed)
    return _manager
