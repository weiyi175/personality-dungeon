"""Runtime Bridge v2: RL Session Manager with session pool & lifecycle (SDD §12.8).

This module implements the session pool management layer between the FastAPI server
and RLSessionEngine instances. Supports:
  - Session creation with RLSessionConfig
  - Per-session step-by-step advancement
  - Snapshot polling without advancement
  - Reset capability (restart from round=0, warm=False)
  - Session pooling with FIFO eviction (max_sessions=100)
  - Thread-safe operations (via session-level locks)

Protocol: Session lifecycle follows SDD §12.4 diagram:
  reset() → burn-in (async) → warm=False → step() → warm=True → snapshot() → respond

Integration point: api/server.py routes → rl_session_manager methods → RLSessionEngine
"""

from __future__ import annotations

import threading
import uuid
from collections import OrderedDict
from dataclasses import asdict
from typing import Any

from simulation.rl_session_engine import (
    RLSessionConfig,
    RLSessionEngine,
    FrameSnapshot,
)


class RLSessionManager:
    """Session pool manager with FIFO eviction policy (max 100 sessions).
    
    Thread-safe per-session. All public methods are safe for concurrent
    access from FastAPI handlers.
    """

    def __init__(self, max_sessions: int = 100):
        """Initialize session pool.
        
        Args:
            max_sessions: Maximum number of concurrent sessions (default 100).
                         When exceeded, oldest session is evicted (FIFO).
        """
        self.max_sessions = max_sessions
        self.sessions: OrderedDict[str, RLSessionEngine] = OrderedDict()
        self.session_configs: dict[str, RLSessionConfig] = {}
        self.session_locks: dict[str, threading.Lock] = {}
        self._manager_lock = threading.Lock()

    def initialize_session(
        self,
        config: RLSessionConfig | None = None,
        session_id: str | None = None,
    ) -> tuple[str, FrameSnapshot]:
        """Create and initialize new RL session.
        
        Args:
            config: RLSessionConfig instance. If None, uses defaults.
            session_id: Optional session ID. If None, generates UUID.
        
        Returns:
            (session_id, initial_snapshot)
        
        Raises:
            ValueError: If config validation fails (BL2 lock violation).
        """
        if config is None:
            config = RLSessionConfig()
        
        if session_id is None:
            session_id = str(uuid.uuid4())

        # Validate config (BL2 parameter lock enforcement)
        config.validate()

        # Check session pool capacity
        with self._manager_lock:
            if len(self.sessions) >= self.max_sessions:
                # FIFO eviction: remove oldest session
                oldest_id, _ = self.sessions.popitem(last=False)
                del self.session_configs[oldest_id]
                del self.session_locks[oldest_id]

            # Create new session and lock
            engine = RLSessionEngine(config, session_id=session_id)
            self.sessions[session_id] = engine
            self.session_configs[session_id] = config
            self.session_locks[session_id] = threading.Lock()

        # Initialize with reset()
        initial_snapshot = engine.reset()
        
        return session_id, initial_snapshot

    def step_session(self, session_id: str) -> FrameSnapshot:
        """Advance session by one step.
        
        Args:
            session_id: Session identifier
        
        Returns:
            FrameSnapshot after step
        
        Raises:
            KeyError: If session not found
            RuntimeError: If session already complete
        """
        engine = self._get_engine(session_id)
        lock = self.session_locks[session_id]

        with lock:
            snapshot = engine.step()
        
        return snapshot

    def snapshot_session(self, session_id: str) -> FrameSnapshot:
        """Retrieve current snapshot without advancing.
        
        Args:
            session_id: Session identifier
        
        Returns:
            FrameSnapshot (does not advance round counter)
        
        Raises:
            KeyError: If session not found
        """
        engine = self._get_engine(session_id)
        lock = self.session_locks[session_id]

        with lock:
            snapshot = engine.snapshot()
        
        return snapshot

    def reset_session(self, session_id: str) -> FrameSnapshot:
        """Reset session to initial state (round=0, warm=False).
        
        Args:
            session_id: Session identifier
        
        Returns:
            Initial FrameSnapshot after reset
        
        Raises:
            KeyError: If session not found
        """
        engine = self._get_engine(session_id)
        lock = self.session_locks[session_id]

        with lock:
            snapshot = engine.reset()
        
        return snapshot

    def get_session_info(self, session_id: str) -> dict[str, Any]:
        """Retrieve session metadata.
        
        Args:
            session_id: Session identifier
        
        Returns:
            Dictionary with session_id, config, round, warm, phase
        
        Raises:
            KeyError: If session not found
        """
        engine = self._get_engine(session_id)
        config = self.session_configs[session_id]

        return {
            "session_id": session_id,
            "config": {
                "n_players": config.n_players,
                "n_rounds": config.n_rounds,
                "burn_in": config.burn_in,
                "seed": config.seed,
                "personality_mode": config.personality_mode,
            },
            "round": engine.round,
            "warm": engine.warm,
            "phase": "burn-in" if not engine.warm else "tail",
        }

    def list_sessions(self) -> list[str]:
        """List all active session IDs.
        
        Returns:
            List of session identifiers
        """
        with self._manager_lock:
            return list(self.sessions.keys())

    def delete_session(self, session_id: str) -> None:
        """Manually delete session (outside FIFO eviction).
        
        Args:
            session_id: Session identifier
        
        Raises:
            KeyError: If session not found
        """
        self._get_engine(session_id)  # Verify exists
        
        with self._manager_lock:
            del self.sessions[session_id]
            del self.session_configs[session_id]
            del self.session_locks[session_id]

    def _get_engine(self, session_id: str) -> RLSessionEngine:
        """Retrieve engine by session ID (with existence check).
        
        Args:
            session_id: Session identifier
        
        Returns:
            RLSessionEngine instance
        
        Raises:
            KeyError: If session not found
        """
        if session_id not in self.sessions:
            raise KeyError(f"Session {session_id} not found")
        return self.sessions[session_id]


# Global session manager instance (singleton)
_global_manager: RLSessionManager | None = None


def get_session_manager(max_sessions: int = 100) -> RLSessionManager:
    """Get or create global session manager (lazy singleton pattern).
    
    Args:
        max_sessions: Maximum concurrent sessions (only used on first call)
    
    Returns:
        Global RLSessionManager instance
    """
    global _global_manager
    if _global_manager is None:
        _global_manager = RLSessionManager(max_sessions=max_sessions)
    return _global_manager
