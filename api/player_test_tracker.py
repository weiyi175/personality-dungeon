"""
Player test tracker for P7-H Phase IV.

Tracks per-session trajectory data during an actual (or simulated) player test:
  - personality vector sequence (before / after each game action)
  - event response times
  - bifurcation success flags
  - session metadata (group, start/end time, total actions)

Complements ABTestManager (which tracks group assignment) by storing the
full time-series trajectory for post-hoc analysis.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Literal

Group = Literal["control", "experiment"]


@dataclass
class TrajectoryStep:
    step_index: int
    timestamp: float
    action_text: str           # player's in-game decision text
    personality_before: list[float]
    personality_after: list[float]
    proximity_before: float
    proximity_after: float
    event_type: str            # "personality_shift" | "explore_risk" | "none" | …
    response_time_ms: float    # milliseconds from action prompt to player choice


@dataclass
class PlayerTestSession:
    session_id: str
    group: Group
    player_alias: str          # anonymised identifier
    started_at: float = field(default_factory=time.time)
    ended_at: float | None = None
    trajectory: list[TrajectoryStep] = field(default_factory=list)
    # Derived metrics (computed on session_end)
    max_proximity: float = 0.0
    total_displacement: float = 0.0   # net ‖p_final − p_initial‖ (H1 primary DV)
    path_displacement: float = 0.0    # Σ‖p_k − p_{k-1}‖ over the ordered path
    n_bifurcation_events: int = 0
    n_critical_crossings: int = 0


class PlayerTestTracker:
    """In-memory player test session tracker."""

    def __init__(self) -> None:
        self._sessions: dict[str, PlayerTestSession] = {}

    # ── Session lifecycle ─────────────────────────────────────────────────────

    def start_session(
        self,
        session_id: str,
        group: Group,
        player_alias: str = "anon",
    ) -> dict:
        if session_id in self._sessions:
            return {"ok": False, "error": "session already exists"}
        self._sessions[session_id] = PlayerTestSession(
            session_id=session_id, group=group, player_alias=player_alias
        )
        return {"ok": True, "session_id": session_id, "group": group}

    def record_step(
        self,
        session_id: str,
        action_text: str,
        personality_before: list[float],
        personality_after: list[float],
        proximity_before: float,
        proximity_after: float,
        event_type: str = "none",
        response_time_ms: float = 0.0,
    ) -> dict:
        sess = self._sessions.get(session_id)
        if sess is None:
            return {"ok": False, "error": "session not found"}

        step = TrajectoryStep(
            step_index=len(sess.trajectory),
            timestamp=time.time(),
            action_text=action_text[:100],
            personality_before=personality_before[:],
            personality_after=personality_after[:],
            proximity_before=proximity_before,
            proximity_after=proximity_after,
            event_type=event_type,
            response_time_ms=response_time_ms,
        )
        sess.trajectory.append(step)

        # Update rolling max proximity
        sess.max_proximity = max(sess.max_proximity, proximity_after)

        # Count bifurcation events (non-trivial events)
        if event_type not in ("none", ""):
            sess.n_bifurcation_events += 1

        # Count critical crossings (0.8 threshold)
        if proximity_before < 0.8 <= proximity_after:
            sess.n_critical_crossings += 1

        return {"ok": True, "step_index": step.step_index}

    def end_session(self, session_id: str) -> dict:
        sess = self._sessions.get(session_id)
        if sess is None:
            return {"ok": False, "error": "session not found"}

        sess.ended_at = time.time()
        if sess.trajectory:
            import numpy as np
            # Steps may arrive out of order (the client posts them fire-and-forget,
            # so HTTP responses can interleave). Order by timestamp before deriving
            # any endpoint/path metric, otherwise total_displacement is computed
            # from whichever steps happened to land first/last — a non-deterministic
            # value that does not match the persisted trajectory. (P7-H DV bug fix.)
            ordered = sorted(sess.trajectory, key=lambda s: (s.timestamp, s.step_index))
            sess.trajectory = ordered
            for i, s in enumerate(ordered):
                s.step_index = i

            p0 = np.array(ordered[0].personality_before)
            pf = np.array(ordered[-1].personality_after)
            sess.total_displacement = float(np.linalg.norm(pf - p0))

            # Path length: cumulative step-to-step motion. Robust to the choice of
            # endpoints and reflects how much the personality actually travelled.
            path = 0.0
            prev = p0
            for s in ordered:
                cur = np.array(s.personality_after)
                path += float(np.linalg.norm(cur - prev))
                prev = cur
            sess.path_displacement = path

        duration_min = (
            (sess.ended_at - sess.started_at) / 60.0
            if sess.ended_at else None
        )
        return {
            "ok": True,
            "session_id": session_id,
            "n_steps": len(sess.trajectory),
            "max_proximity": sess.max_proximity,
            "total_displacement": sess.total_displacement,
            "path_displacement": sess.path_displacement,
            "n_bifurcation_events": sess.n_bifurcation_events,
            "n_critical_crossings": sess.n_critical_crossings,
            "duration_minutes": round(duration_min, 2) if duration_min else None,
        }

    def get_session(self, session_id: str) -> dict | None:
        sess = self._sessions.get(session_id)
        return asdict(sess) if sess else None

    # ── Analytics ─────────────────────────────────────────────────────────────

    def group_summary(self) -> dict:
        """Compute group-level trajectory statistics."""
        import numpy as np

        groups: dict[str, list[PlayerTestSession]] = {"control": [], "experiment": []}
        for sess in self._sessions.values():
            groups[sess.group].append(sess)

        def _stats(records: list[PlayerTestSession]) -> dict:
            if not records:
                return {"n": 0}
            completed = [r for r in records if r.ended_at is not None]
            displacements = [r.total_displacement for r in completed]
            path_disp = [r.path_displacement for r in completed]
            max_prox = [r.max_proximity for r in completed]
            crossings = [r.n_critical_crossings for r in completed]
            n_steps_list = [len(r.trajectory) for r in completed]
            rt_list = [
                s.response_time_ms
                for r in completed
                for s in r.trajectory
                if s.response_time_ms > 0
            ]
            return {
                "n_total": len(records),
                "n_completed": len(completed),
                "mean_displacement": float(np.mean(displacements)) if displacements else 0.0,
                "std_displacement": float(np.std(displacements)) if displacements else 0.0,
                "mean_path_displacement": float(np.mean(path_disp)) if path_disp else 0.0,
                "mean_max_proximity": float(np.mean(max_prox)) if max_prox else 0.0,
                "mean_critical_crossings": float(np.mean(crossings)) if crossings else 0.0,
                "mean_n_steps": float(np.mean(n_steps_list)) if n_steps_list else 0.0,
                "mean_response_time_ms": float(np.mean(rt_list)) if rt_list else 0.0,
            }

        ctrl = _stats(groups["control"])
        exp = _stats(groups["experiment"])

        cohen_d = 0.0
        if ctrl.get("n_completed", 0) > 0 and exp.get("n_completed", 0) > 0:
            pooled = (ctrl["std_displacement"] + exp["std_displacement"]) / 2
            if pooled > 0:
                cohen_d = (exp["mean_displacement"] - ctrl["mean_displacement"]) / pooled

        return {
            "total_sessions": len(self._sessions),
            "control": ctrl,
            "experiment": exp,
            "effect_size_cohens_d": round(cohen_d, 4),
        }

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self, out_dir: str | Path) -> Path:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        path = out / "p7h_player_test_sessions.json"
        with open(path, "w") as f:
            json.dump(
                {
                    "sessions": {
                        sid: asdict(sess) for sid, sess in self._sessions.items()
                    },
                    "group_summary": self.group_summary(),
                },
                f,
                indent=2,
            )
        return path

    def load(self, out_dir: str | Path) -> bool:
        """Restore sessions from a prior save(). Returns True if a file loaded.

        Used on server startup so a restart mid-study keeps already-collected
        trajectories instead of losing them with the in-memory singleton.
        """
        path = Path(out_dir) / "p7h_player_test_sessions.json"
        if not path.exists():
            return False
        with open(path) as f:
            data = json.load(f)
        restored: dict[str, PlayerTestSession] = {}
        for sid, s in data.get("sessions", {}).items():
            traj = [TrajectoryStep(**step) for step in s.pop("trajectory", [])]
            restored[sid] = PlayerTestSession(trajectory=traj, **s)
        self._sessions = restored
        return True


# ── Module-level singleton ────────────────────────────────────────────────────
_tracker: PlayerTestTracker | None = None


def get_tracker() -> PlayerTestTracker:
    global _tracker
    if _tracker is None:
        _tracker = PlayerTestTracker()
    return _tracker
