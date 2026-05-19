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
from api.personality_text_inference import (
    infer_personality_vector,
    load_inference_config,
    log_personality_pair,
)
from core.game_engine import GameEngine
from api.rl_session_manager import get_session_manager
from simulation.rl_session_engine import RLSessionConfig


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
# RL Session Request/Response Models
# ===================================================================


class RLSessionInitRequest(BaseModel):
    """POST /rl_sessions/initialize payload."""

    n_players: int = 300
    n_rounds: int = 12000
    burn_in: int = 4000
    seed: int | None = None
    personality_mode: str = "random_9persona"


class RLSessionInitResponse(BaseModel):
    """Response to POST /rl_sessions/initialize."""

    session_id: str
    initial_snapshot: dict[str, Any]


class RLSessionStepResponse(BaseModel):
    """Response to POST /rl_sessions/{session_id}/step."""

    session_id: str
    snapshot: dict[str, Any]


class RLSessionInfoResponse(BaseModel):
    """Response to GET /rl_sessions/{session_id}/info."""

    session_id: str
    config: dict[str, Any]
    round: int
    warm: bool
    phase: str


# ===================================================================
# Personality Text Inference (LLM-backed)
# ===================================================================


class PersonalityInferRequest(BaseModel):
    """POST /personality/infer payload."""

    text: str
    source: str | None = None
    session_id: str | None = None
    user_id: str | None = None
    temperature: float | None = None


class PersonalityInferResponse(BaseModel):
    """Response to POST /personality/infer."""

    request_id: str
    text: str
    vector: dict[str, float]
    model: str
    temperature: float
    logged: bool
    log_error: str | None = None


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
# Runtime Bridge v2: RL Session Endpoints (SDD §12.8-12.9)
# ===================================================================


@app.post("/rl_sessions/initialize", response_model=RLSessionInitResponse)
async def rl_initialize_session(req: RLSessionInitRequest) -> RLSessionInitResponse:
    """Create and initialize new RL session (BL2 anchor locked).
    
    Validates RLSessionConfig against BL2 parameter lock before creation.
    If config passes, returns session_id + initial FrameSnapshot with warm=False.
    
    Args:
        req: RLSessionInitRequest with n_players, n_rounds, burn_in, seed,
             personality_mode (defaults to BL2-locked values).
    
    Returns:
        RLSessionInitResponse with session_id and initial_snapshot.
    
    Raises:
        HTTPException 400: If BL2 parameter validation fails
        HTTPException 500: If session creation fails
    """
    try:
        # Build RLSessionConfig with provided parameters
        config = RLSessionConfig(
            n_players=req.n_players,
            n_rounds=req.n_rounds,
            burn_in=req.burn_in,
            seed=req.seed if req.seed is not None else 42,
            personality_mode=req.personality_mode,
            # All other fields use defaults (BL2-locked: alpha_lo=0.005, alpha_hi=0.40, beta=3.0, etc.)
        )
        
        # Get global session manager
        manager = get_session_manager()
        
        # Initialize session (validates config, creates engine, returns initial snapshot)
        session_id, initial_snapshot = manager.initialize_session(config=config)
        
        # Convert FrameSnapshot to dict for JSON response
        snapshot_dict = {
            "session_id": initial_snapshot.session_id,
            "round": initial_snapshot.round,
            "tick": initial_snapshot.tick,
            "warm": initial_snapshot.warm,
            "cycle_level": initial_snapshot.cycle_level,
            "s3_score": initial_snapshot.s3_score,
            "env_gamma": initial_snapshot.env_gamma,
            "entropy": initial_snapshot.entropy,
            "q_std": initial_snapshot.q_std,
            "p_aggressive": initial_snapshot.p_aggressive,
            "p_defensive": initial_snapshot.p_defensive,
            "p_balanced": initial_snapshot.p_balanced,
            "pi_aggressive": initial_snapshot.pi_aggressive,
            "pi_defensive": initial_snapshot.pi_defensive,
            "pi_balanced": initial_snapshot.pi_balanced,
            "q_mean_aggressive": initial_snapshot.q_mean_aggressive,
            "q_mean_defensive": initial_snapshot.q_mean_defensive,
            "q_mean_balanced": initial_snapshot.q_mean_balanced,
            "avg_reward": initial_snapshot.avg_reward,
            "avg_utility": initial_snapshot.avg_utility,
            "success_rate": initial_snapshot.success_rate,
            "risk_mean": initial_snapshot.risk_mean,
            "stress_mean": initial_snapshot.stress_mean,
            "world_scarcity": initial_snapshot.world_scarcity,
            "world_threat": initial_snapshot.world_threat,
            "world_noise": initial_snapshot.world_noise,
            "world_intel": initial_snapshot.world_intel,
            "phase": initial_snapshot.phase,
        }
        
        return RLSessionInitResponse(
            session_id=session_id,
            initial_snapshot=snapshot_dict,
        )
    
    except ValueError as e:
        # BL2 parameter lock violation
        raise HTTPException(status_code=400, detail=f"BL2 parameter validation failed: {str(e)}")
    except Exception as e:
        # Unexpected error
        raise HTTPException(status_code=500, detail=f"Session creation failed: {str(e)}")


@app.post("/rl_sessions/{session_id}/step", response_model=RLSessionStepResponse)
async def rl_step_session(session_id: str) -> RLSessionStepResponse:
    """Advance RL session by one step.
    
    Executes single-round RL update (Boltzmann selection, payoff, Q-update).
    After step, appends to tail buffer if warm=True and beyond burn-in.
    Checks cycle metrics every 200 rounds during tail phase.
    
    Args:
        session_id: Session identifier.
    
    Returns:
        RLSessionStepResponse with session_id and current FrameSnapshot.
    
    Raises:
        HTTPException 404: If session not found
        HTTPException 500: If step execution fails
    """
    try:
        manager = get_session_manager()
        snapshot = manager.step_session(session_id)
        
        # Convert FrameSnapshot to dict
        snapshot_dict = {
            "session_id": snapshot.session_id,
            "round": snapshot.round,
            "tick": snapshot.tick,
            "warm": snapshot.warm,
            "cycle_level": snapshot.cycle_level,
            "s3_score": snapshot.s3_score,
            "env_gamma": snapshot.env_gamma,
            "entropy": snapshot.entropy,
            "q_std": snapshot.q_std,
            "p_aggressive": snapshot.p_aggressive,
            "p_defensive": snapshot.p_defensive,
            "p_balanced": snapshot.p_balanced,
            "pi_aggressive": snapshot.pi_aggressive,
            "pi_defensive": snapshot.pi_defensive,
            "pi_balanced": snapshot.pi_balanced,
            "q_mean_aggressive": snapshot.q_mean_aggressive,
            "q_mean_defensive": snapshot.q_mean_defensive,
            "q_mean_balanced": snapshot.q_mean_balanced,
            "avg_reward": snapshot.avg_reward,
            "avg_utility": snapshot.avg_utility,
            "success_rate": snapshot.success_rate,
            "risk_mean": snapshot.risk_mean,
            "stress_mean": snapshot.stress_mean,
            "world_scarcity": snapshot.world_scarcity,
            "world_threat": snapshot.world_threat,
            "world_noise": snapshot.world_noise,
            "world_intel": snapshot.world_intel,
            "phase": snapshot.phase,
        }
        
        return RLSessionStepResponse(
            session_id=session_id,
            snapshot=snapshot_dict,
        )
    
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found")
    except RuntimeError as e:
        # Session complete or other runtime error
        raise HTTPException(status_code=400, detail=f"Step failed: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")


@app.get("/rl_sessions/{session_id}/snapshot", response_model=RLSessionStepResponse)
async def rl_snapshot_session(session_id: str) -> RLSessionStepResponse:
    """Retrieve current RL session snapshot without advancing.
    
    Returns current FrameSnapshot without incrementing round counter.
    
    Args:
        session_id: Session identifier.
    
    Returns:
        RLSessionStepResponse with current FrameSnapshot.
    
    Raises:
        HTTPException 404: If session not found
    """
    try:
        manager = get_session_manager()
        snapshot = manager.snapshot_session(session_id)
        
        # Convert FrameSnapshot to dict
        snapshot_dict = {
            "session_id": snapshot.session_id,
            "round": snapshot.round,
            "tick": snapshot.tick,
            "warm": snapshot.warm,
            "cycle_level": snapshot.cycle_level,
            "s3_score": snapshot.s3_score,
            "env_gamma": snapshot.env_gamma,
            "entropy": snapshot.entropy,
            "q_std": snapshot.q_std,
            "p_aggressive": snapshot.p_aggressive,
            "p_defensive": snapshot.p_defensive,
            "p_balanced": snapshot.p_balanced,
            "pi_aggressive": snapshot.pi_aggressive,
            "pi_defensive": snapshot.pi_defensive,
            "pi_balanced": snapshot.pi_balanced,
            "q_mean_aggressive": snapshot.q_mean_aggressive,
            "q_mean_defensive": snapshot.q_mean_defensive,
            "q_mean_balanced": snapshot.q_mean_balanced,
            "avg_reward": snapshot.avg_reward,
            "avg_utility": snapshot.avg_utility,
            "success_rate": snapshot.success_rate,
            "risk_mean": snapshot.risk_mean,
            "stress_mean": snapshot.stress_mean,
            "world_scarcity": snapshot.world_scarcity,
            "world_threat": snapshot.world_threat,
            "world_noise": snapshot.world_noise,
            "world_intel": snapshot.world_intel,
            "phase": snapshot.phase,
        }
        
        return RLSessionStepResponse(
            session_id=session_id,
            snapshot=snapshot_dict,
        )
    
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found")


@app.post("/rl_sessions/{session_id}/reset", response_model=RLSessionStepResponse)
async def rl_reset_session(session_id: str) -> RLSessionStepResponse:
    """Reset RL session to initial state (round=0, warm=False).
    
    Clears players' Q-values, resets round counter, clears tail buffer.
    Returns FrameSnapshot identical to initial state.
    
    Args:
        session_id: Session identifier.
    
    Returns:
        RLSessionStepResponse with reset FrameSnapshot.
    
    Raises:
        HTTPException 404: If session not found
    """
    try:
        manager = get_session_manager()
        snapshot = manager.reset_session(session_id)
        
        # Convert FrameSnapshot to dict
        snapshot_dict = {
            "session_id": snapshot.session_id,
            "round": snapshot.round,
            "tick": snapshot.tick,
            "warm": snapshot.warm,
            "cycle_level": snapshot.cycle_level,
            "s3_score": snapshot.s3_score,
            "env_gamma": snapshot.env_gamma,
            "entropy": snapshot.entropy,
            "q_std": snapshot.q_std,
            "p_aggressive": snapshot.p_aggressive,
            "p_defensive": snapshot.p_defensive,
            "p_balanced": snapshot.p_balanced,
            "pi_aggressive": snapshot.pi_aggressive,
            "pi_defensive": snapshot.pi_defensive,
            "pi_balanced": snapshot.pi_balanced,
            "q_mean_aggressive": snapshot.q_mean_aggressive,
            "q_mean_defensive": snapshot.q_mean_defensive,
            "q_mean_balanced": snapshot.q_mean_balanced,
            "avg_reward": snapshot.avg_reward,
            "avg_utility": snapshot.avg_utility,
            "success_rate": snapshot.success_rate,
            "risk_mean": snapshot.risk_mean,
            "stress_mean": snapshot.stress_mean,
            "world_scarcity": snapshot.world_scarcity,
            "world_threat": snapshot.world_threat,
            "world_noise": snapshot.world_noise,
            "world_intel": snapshot.world_intel,
            "phase": snapshot.phase,
        }
        
        return RLSessionStepResponse(
            session_id=session_id,
            snapshot=snapshot_dict,
        )
    
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found")


@app.get("/rl_sessions/{session_id}/info", response_model=RLSessionInfoResponse)
async def rl_session_info(session_id: str) -> RLSessionInfoResponse:
    """Retrieve RL session metadata and configuration.
    
    Args:
        session_id: Session identifier.
    
    Returns:
        RLSessionInfoResponse with config, round, warm, phase.
    
    Raises:
        HTTPException 404: If session not found
    """
    try:
        manager = get_session_manager()
        info = manager.get_session_info(session_id)
        
        return RLSessionInfoResponse(
            session_id=info["session_id"],
            config=info["config"],
            round=info["round"],
            warm=info["warm"],
            phase=info["phase"],
        )
    
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found")


@app.get("/rl_sessions", response_model=dict[str, Any])
async def rl_list_sessions() -> dict[str, Any]:
    """List all active RL sessions.
    
    Returns:
        JSON dict with session_ids list.
    """
    manager = get_session_manager()
    session_ids = manager.list_sessions()
    
    return {
        "session_ids": session_ids,
        "count": len(session_ids),
    }


@app.delete("/rl_sessions/{session_id}")
async def rl_delete_session(session_id: str) -> dict[str, str]:
    """Manually delete RL session (outside FIFO eviction).
    
    Args:
        session_id: Session identifier.
    
    Returns:
        JSON dict with deletion confirmation.
    
    Raises:
        HTTPException 404: If session not found
    """
    try:
        manager = get_session_manager()
        manager.delete_session(session_id)
        
        return {"status": "deleted", "session_id": session_id}
    
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found")


# ===================================================================
# Endpoint: Personality Text Inference
# ===================================================================


@app.post("/personality/infer", response_model=PersonalityInferResponse)
async def personality_infer(req: PersonalityInferRequest) -> PersonalityInferResponse:
    """Infer 9-trait personality vector from a short text input."""
    try:
        config = load_inference_config()
    except ValueError as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    if req.temperature is not None and req.temperature < 0.0:
        raise HTTPException(status_code=400, detail="temperature must be >= 0")

    try:
        vector, meta = infer_personality_vector(
            req.text,
            config=config,
            temperature_override=req.temperature,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Inference failed: {exc}")

    try:
        logged, log_error = log_personality_pair(
            req.text.strip(),
            vector,
            meta,
            config=config,
            source=req.source,
            session_id=req.session_id,
            user_id=req.user_id,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Logging failed: {exc}")

    return PersonalityInferResponse(
        request_id=str(meta.get("request_id")),
        text=req.text.strip(),
        vector=vector,
        model=str(meta.get("model")),
        temperature=float(meta.get("temperature")),
        logged=logged,
        log_error=log_error,
    )


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

