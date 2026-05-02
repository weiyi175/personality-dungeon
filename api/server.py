"""Core↔Frontend API server.

Wraps core.GameEngine and serves ResponseEnvelope over HTTP.
Maintains session state, tick counter, and provides stable output contract.

Contract: All endpoints return JSON-serialized ResponseEnvelope (api/schemas.py).
API Version lock: "1.0.0" (immutable).

Usage:
    python -m api.server
    # Server runs on localhost:8000
    # POST /sessions/initialize → session_id
    # POST /sessions/{session_id}/step → action → ResponseEnvelope
    # GET /sessions/{session_id}/snapshot → ResponseEnvelope
    # POST /sessions/{session_id}/reset → ResponseEnvelope
"""

from __future__ import annotations

import hashlib
import uuid
from dataclasses import dataclass
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from api.schemas import (
    API_VERSION,
    ResponseEnvelope,
    WorldState,
    PlayerState,
    ResultState,
    normalize_response_envelope,
)
from core.game_engine import GameEngine


# ===================================================================
# Session Management
# ===================================================================


@dataclass
class MockDungeon:
    """Minimal mock dungeon for testing without full simulation."""

    def evaluate_player(self, player: object, strategy: str) -> float:
        """Return mock reward."""
        return 0.5  # Neutral reward

    def update_popularity(self, chosen_strategies: list[str]) -> None:
        """Track popularity (no-op for mock)."""
        pass

    def set_popularity(self, popularity: dict[str, float]) -> None:
        """Set expected popularity (no-op for mock)."""
        pass


@dataclass
class MockPlayer:
    """Minimal mock player for testing."""

    def choose_strategy(self) -> str:
        """Return a strategy."""
        return "balanced"

    def update_utility(self, reward: float) -> None:
        """Update internal utility (no-op for mock)."""
        pass


@dataclass
class GameSession:
    """In-memory session tracking tick counter and engine state."""

    session_id: str
    engine: GameEngine
    tick: int = 0
    world_state: dict[str, Any] | None = None

    def compute_state_hash(self) -> str:
        """Compute deterministic hash of current game state."""
        state_str = f"{self.session_id}:{self.tick}:{self.world_state}"
        return hashlib.sha256(state_str.encode()).hexdigest()[:16]


# ===================================================================
# FastAPI App
# ===================================================================

app = FastAPI(
    title="Personality Dungeon API",
    description="Contract-locked core↔frontend API",
    version=API_VERSION,
)

# In-memory session store
SESSIONS: dict[str, GameSession] = {}


# ===================================================================
# Pydantic Request/Response Models
# ===================================================================


class StepRequest(BaseModel):
    """POST /sessions/{session_id}/step payload."""

    action: str | None = None


class InitializeResponse(BaseModel):
    """Response to POST /sessions/initialize."""

    session_id: str


# ===================================================================
# Endpoint: Initialize Session
# ===================================================================


@app.post("/sessions/initialize", response_model=InitializeResponse)
async def initialize_session(
    n_players: int = 10,
    seed: int | None = None,
) -> InitializeResponse:
    """Create new game session.

    Args:
        n_players: Number of players to initialize.
        seed: Optional random seed for reproducibility.

    Returns:
        InitializeResponse with session_id.
    """
    session_id = str(uuid.uuid4())
    
    # Initialize mock players and dungeon
    players = [MockPlayer() for _ in range(n_players)]
    dungeon = MockDungeon()
    
    engine = GameEngine(players=players, dungeon=dungeon)
    
    session = GameSession(
        session_id=session_id,
        engine=engine,
        tick=0,
        world_state={"scarcity": 0.5, "threat": 0.3, "noise": 0.2, "intel": 0.4},
    )
    
    SESSIONS[session_id] = session
    return InitializeResponse(session_id=session_id)


# ===================================================================
# Endpoint: Step (perform one action)
# ===================================================================


@app.post("/sessions/{session_id}/step")
async def step(session_id: str, req: StepRequest) -> dict[str, Any]:
    """Execute one game step with optional player action.

    Args:
        session_id: Session identifier.
        req: StepRequest with optional action.

    Returns:
        JSON-serialized ResponseEnvelope with kind="step".
    """
    if session_id not in SESSIONS:
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found")
    
    session = SESSIONS[session_id]
    
    # Call engine.step() to get records from all players
    step_records = session.engine.step()
    
    # Build result from step_records
    # Each record: {"strategy": str, "reward": float, "base_reward": float, "event_result": ...}
    rewards = [rec.get("reward", 0.0) for rec in step_records]
    avg_reward = sum(rewards) / len(rewards) if rewards else 0.0
    
    result = ResultState(
        selected_choice_id=req.action or "default",
        reward=avg_reward,
        utility_delta=avg_reward,
        risk_delta=0.0,
        terminated=False,
        termination_reason=None,
    )
    
    # Wrap in ResponseEnvelope
    envelope_data = {
        "api_version": API_VERSION,
        "kind": "step",
        "ok": True,
        "session_id": session_id,
        "tick": session.tick,
        "state_hash": session.compute_state_hash(),
        "world_state": session.world_state,
        "result": result.to_dict(),
    }
    
    envelope = normalize_response_envelope(envelope_data)
    
    # Increment tick for next step
    session.tick += 1
    
    return envelope.to_dict()


# ===================================================================
# Endpoint: Snapshot (current state)
# ===================================================================


@app.get("/sessions/{session_id}/snapshot")
async def snapshot(session_id: str) -> dict[str, Any]:
    """Get current game state snapshot without executing a step.

    Args:
        session_id: Session identifier.

    Returns:
        JSON-serialized ResponseEnvelope with kind="snapshot".
    """
    if session_id not in SESSIONS:
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found")
    
    session = SESSIONS[session_id]
    
    # Build snapshot envelope
    envelope_data = {
        "api_version": API_VERSION,
        "kind": "snapshot",
        "ok": True,
        "session_id": session_id,
        "tick": session.tick,
        "state_hash": session.compute_state_hash(),
        "world_state": session.world_state,
    }
    
    envelope = normalize_response_envelope(envelope_data)
    return envelope.to_dict()


# ===================================================================
# Endpoint: Reset Session
# ===================================================================


@app.post("/sessions/{session_id}/reset")
async def reset_session(session_id: str) -> dict[str, Any]:
    """Reset session to initial state (tick=0).

    Args:
        session_id: Session identifier.

    Returns:
        JSON-serialized ResponseEnvelope with kind="init".
    """
    if session_id not in SESSIONS:
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found")
    
    session = SESSIONS[session_id]
    session.tick = 0
    session.world_state = {"scarcity": 0.5, "threat": 0.3, "noise": 0.2, "intel": 0.4}
    
    # Build reset envelope
    envelope_data = {
        "api_version": API_VERSION,
        "kind": "reset",
        "ok": True,
        "session_id": session_id,
        "tick": session.tick,
        "state_hash": session.compute_state_hash(),
        "world_state": session.world_state,
    }
    
    envelope = normalize_response_envelope(envelope_data)
    return envelope.to_dict()


# ===================================================================
# Endpoint: Get State Hash
# ===================================================================


@app.get("/sessions/{session_id}/state-hash")
async def get_state_hash(session_id: str) -> dict[str, str]:
    """Get state hash for debugging/provenance.

    Args:
        session_id: Session identifier.

    Returns:
        JSON dict with state_hash.
    """
    if session_id not in SESSIONS:
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found")
    
    session = SESSIONS[session_id]
    return {"state_hash": session.compute_state_hash()}


# ===================================================================
# Entry Point
# ===================================================================


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info",
    )

