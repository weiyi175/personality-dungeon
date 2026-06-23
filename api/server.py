"""Core↔Frontend API server.

Wraps core.GameEngine and serves ResponseEnvelope over HTTP.
Maintains session state, tick counter, and provides stable output contract.

Contract: All endpoints return JSON-serialized ResponseEnvelope (api/schemas.py).
API Version lock: "1.0.0" (immutable).

Usage:
    python -m api.server
    # Server runs on localhost:8001 (8000 is reserved for the Godot AI MCP server)
    # POST /sessions/initialize → session_id
    # POST /sessions/{session_id}/step → action → ResponseEnvelope
    # GET /sessions/{session_id}/snapshot → ResponseEnvelope
    # POST /sessions/{session_id}/reset → ResponseEnvelope
"""

from __future__ import annotations

import hashlib
import json
from contextlib import asynccontextmanager
from dataclasses import asdict
from typing import Any
import sys
from pathlib import Path

# Ensure project root is on sys.path so local `api` package resolves
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

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
from api.personality_sbert_inference import infer_personality_vector_sbert
from api.rl_session_manager import get_session_manager
from simulation.rl_session_engine import RLSessionConfig, VALID_PERSONALITY_MODES
from api.ab_test_manager import get_manager as _get_ab_manager
from api.player_test_tracker import get_tracker as _get_tracker
from api.survey_manager import get_survey as _get_survey
from api.ecology_tracker import get_tracker as _get_ecology
from api.pvp_manager import get_manager as _get_pvp
from api.wallet_manager import InsufficientFunds
from api.wallet_manager import get_manager as _get_wallet
from simulation.bifurcation_detector import (
    compute_bifurcation_distance,
    compute_sensitive_direction,
    get_jacobian,
    FEATURE_NAMES as _BF_FEATURE_NAMES,
)
from simulation.event_generator import (
    design_bifurcation_event,
    event_sequence_planner,
    intensity_modulation,
    is_in_effective_zone,
    rollback_distance,
)
from simulation.passive_choice import (
    resolve_passive_choice,
    vector_to_personality,
)
from simulation.personality_space import (
    a_to_b as space_a_to_b,
    b_to_a as space_b_to_a,
    displacement_a_to_b as space_disp_a_to_b,
)
from simulation.personality_seed import sample_sub_critical
from dungeon.event_loader import EventLoader
import asyncio
import os
import random
import time
import numpy as np
try:
    from api.instrumentation import log_metric, persist_metrics_batch
except Exception:
    # Fallback minimal implementations if instrumentation module isn't available
    from pathlib import Path
    import time, json

    LOG_DIR = ROOT / "logs"
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    def _default_log_path():
        name = f"metrics_{time.strftime('%Y-%m-%d')}.jsonl"
        return LOG_DIR / name

    def log_metric(event_type: str, **kwargs):
        ev = {"timestamp": int(time.time() * 1000), "event_type": event_type}
        ev.update(kwargs)
        p = _default_log_path()
        with p.open("a", encoding="utf-8") as f:
            f.write(json.dumps(ev, ensure_ascii=False) + "\n")
        return p

    def persist_metrics_batch(events, path=None):
        p = _default_log_path() if path is None else Path(path)
        with p.open("a", encoding="utf-8") as f:
            for ev in events:
                if "timestamp" not in ev:
                    ev["timestamp"] = int(time.time() * 1000)
                f.write(json.dumps(ev, ensure_ascii=False) + "\n")
        return p


# ===================================================================
# P7-H real-study persistence
# Real-human collection spans days and the server may restart, so the
# A/B / tracker / survey singletons (in-memory) are mirrored to disk.
# Output goes to P7H_OUT_DIR (default p7h_real_study) — kept separate
# from the simulated p7h_player_test/ validation data.
# ===================================================================

P7H_OUT_DIR = os.environ.get(
    "P7H_OUT_DIR", "reports/experiments/p7h_real_study"
)

# Personality-ecology meta-layer store — kept fully separate from P7-H study data
# (見 人格生態評分_規劃_v1.md §0/§4：不污染 P7-H).
ECOLOGY_OUT_DIR = os.environ.get("ECOLOGY_OUT_DIR", "reports/ecology")

# PVP / 地牢挑戰 store（最小 combat loop，獨立 store；不污染 P7-H / ecology）。
PVP_OUT_DIR = os.environ.get("PVP_OUT_DIR", "reports/pvp")

# 玩家錢包 store（Increment 2 經濟；獨立 store）。
WALLET_OUT_DIR = os.environ.get("WALLET_OUT_DIR", "reports/wallet")
# 前端可自行 credit 的 source 白名單（agnostic）：存活。生態 coins 為後端權威 credit。
_CLIENT_CREDIT_SOURCES = {"survival"}


async def _p7h_save(manager) -> None:
    """Persist a P7-H manager to disk off the event loop (best-effort)."""
    try:
        await asyncio.to_thread(manager.save, P7H_OUT_DIR)
    except Exception as exc:  # never fail a request because a save hiccupped
        log_metric("p7h_save_error", error=str(exc))


async def _ecology_save() -> None:
    """Persist the ecology tracker off the event loop (best-effort)."""
    try:
        await asyncio.to_thread(_get_ecology().save, ECOLOGY_OUT_DIR)
    except Exception as exc:
        log_metric("ecology_save_error", error=str(exc))


async def _pvp_save() -> None:
    """Persist the PVP manager off the event loop (best-effort)."""
    try:
        await asyncio.to_thread(_get_pvp().save, PVP_OUT_DIR)
    except Exception as exc:
        log_metric("pvp_save_error", error=str(exc))


async def _wallet_save() -> None:
    """Persist the wallet off the event loop (best-effort)."""
    try:
        await asyncio.to_thread(_get_wallet().save, WALLET_OUT_DIR)
    except Exception as exc:
        log_metric("wallet_save_error", error=str(exc))


@asynccontextmanager
async def _lifespan(app: FastAPI):
    """Backfill managers from disk so a restart mid-study keeps assignments,
    trajectories, and survey responses (and the 106/106 count balance).

    Uses the lifespan protocol (the @app.on_event("startup") hook is
    deprecated in current FastAPI/Starlette).
    """
    for mgr in (_get_ab_manager(), _get_tracker(), _get_survey()):
        try:
            mgr.load(P7H_OUT_DIR)
        except Exception as exc:
            log_metric("p7h_load_error", error=str(exc))
    try:
        _get_ecology().load(ECOLOGY_OUT_DIR)
    except Exception as exc:
        log_metric("ecology_load_error", error=str(exc))
    try:
        _get_pvp().load(PVP_OUT_DIR)
    except Exception as exc:
        log_metric("pvp_load_error", error=str(exc))
    try:
        _get_wallet().load(WALLET_OUT_DIR)
    except Exception as exc:
        log_metric("wallet_load_error", error=str(exc))
    yield


# ===================================================================
# FastAPI App
# ===================================================================

app = FastAPI(
    title="Personality Dungeon API",
    description="Contract-locked core↔frontend API",
    version=API_VERSION,
    lifespan=_lifespan,
)

# ===================================================================
# Pydantic Request/Response Models
# ===================================================================


class StepRequest(BaseModel):
    """POST /sessions/{session_id}/step payload."""

    action: str | None = None


class InitializeResponse(BaseModel):
    """Response to POST /sessions/initialize."""

    session_id: str
    status: str
    tick: int
    warm: bool


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
    # Enable Space-A bifurcation events for this session (P7-H). Off by default
    # so existing Runtime-Bridge sessions are unchanged.
    space_a_events_enabled: bool = False
    # ── Sub-critical seeding (P7-H, the validated H1 regime) ──────────────────
    # Place the whole population near the baseline attractor with proximity
    # headroom, so the bifurcation DV is NOT saturated from round 1. When set
    # (0, 1], the server samples one Space-A P₀ via sample_sub_critical() and runs
    # personality_mode="static" with it (overriding personality_mode). 0.5 = halfway
    # to baseline. This reproduces the engine-sim/player-test regime live.
    sub_critical_headroom: float | None = None
    # Explicit Space-A P₀ (9D, FEATURE_NAMES order) shared by all players
    # (personality_mode forced to "static"). Mutually exclusive with
    # sub_critical_headroom. For callers that compute their own seed.
    fixed_personality_vector: list[float] | None = None


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


class MetricsEventsResponse(BaseModel):
    """Response to POST /metrics/events."""

    ok: bool
    received: int


# ===================================================================
# Session Snapshot Helpers (/sessions)
# ===================================================================


def _snapshot_state_hash(snapshot: object) -> str:
    """Compute deterministic hash for the current session snapshot."""
    payload = asdict(snapshot)
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(encoded.encode()).hexdigest()[:16]


def _snapshot_world_state(snapshot: object) -> dict[str, float]:
    """Map RL snapshot world state to ResponseEnvelope world_state."""
    return {
        "scarcity": float(snapshot.world_scarcity),
        "threat": float(snapshot.world_threat),
        "noise": float(snapshot.world_noise),
        "intel": float(snapshot.world_intel),
    }


def _snapshot_extensions(snapshot: object) -> dict[str, Any]:
    """Expose RL snapshot metrics via ResponseEnvelope extensions."""
    return {
        "rl_cycle_level": int(snapshot.cycle_level),
        "rl_s3_score": float(snapshot.s3_score),
        "rl_env_gamma": float(snapshot.env_gamma),
        "rl_entropy": float(snapshot.entropy),
        "rl_q_std": float(snapshot.q_std),
        "rl_p_aggressive": float(snapshot.p_aggressive),
        "rl_p_defensive": float(snapshot.p_defensive),
        "rl_p_balanced": float(snapshot.p_balanced),
        "rl_pi_aggressive": float(snapshot.pi_aggressive),
        "rl_pi_defensive": float(snapshot.pi_defensive),
        "rl_pi_balanced": float(snapshot.pi_balanced),
        "rl_q_mean_aggressive": float(snapshot.q_mean_aggressive),
        "rl_q_mean_defensive": float(snapshot.q_mean_defensive),
        "rl_q_mean_balanced": float(snapshot.q_mean_balanced),
        "rl_avg_reward": float(snapshot.avg_reward),
        "rl_avg_utility": float(snapshot.avg_utility),
        "rl_success_rate": float(snapshot.success_rate),
        "rl_risk_mean": float(snapshot.risk_mean),
        "rl_stress_mean": float(snapshot.stress_mean),
        "rl_round": int(snapshot.round),
        "rl_warm": bool(snapshot.warm),
        "rl_phase": str(snapshot.phase),
    }


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
    """Create new session backed by RLSessionEngine.

    Args:
        n_players: Number of players to initialize.
        seed: Optional random seed for reproducibility.

    Returns:
        InitializeResponse with session_id, warm status, and tick.
    """
    try:
        config_kwargs: dict[str, Any] = {"n_players": n_players}
        if seed is not None:
            config_kwargs["seed"] = seed
        config = RLSessionConfig(**config_kwargs)

        manager = get_session_manager()
        start = time.time()
        session_id, initial_snapshot = manager.initialize_session(config=config)
        latency_ms = int((time.time() - start) * 1000)

        # Emit session start metric and API latency
        log_metric(
            "session_start",
            session_id=session_id,
            n_players=n_players,
            n_rounds=getattr(config, "n_rounds", None),
            burn_in=getattr(config, "burn_in", None),
            seed=getattr(config, "seed", None),
            personality_mode=getattr(config, "personality_mode", None),
        )
        log_metric("rl_init_latency", session_id=session_id, latency_ms=latency_ms, status=200)

        return InitializeResponse(
            session_id=session_id,
            status=initial_snapshot.phase,
            tick=initial_snapshot.tick,
            warm=initial_snapshot.warm,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=f"Session init failed: {exc}")
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Session init failed: {exc}")


# ===================================================================
# Endpoint: Step (perform one action)
# ===================================================================


@app.post("/sessions/{session_id}/step")
async def step(session_id: str, req: StepRequest) -> dict[str, Any]:
    """Execute one RL step with optional action hint.

    Args:
        session_id: Session identifier.
        req: StepRequest with optional action.

    Returns:
        JSON-serialized ResponseEnvelope with kind="step".
    """
    try:
        manager = get_session_manager()
        # Offload the synchronous RL step to a worker thread so the event loop
        # stays responsive to concurrent requests (metrics, snapshot polls).
        # A blocked loop refuses concurrent connections → RESULT_CANT_CONNECT.
        start = time.time()
        snapshot = await asyncio.to_thread(manager.step_session, session_id)
        latency_ms = int((time.time() - start) * 1000)

        result = ResultState(
            selected_choice_id=req.action,
            reward=snapshot.avg_reward,
            utility_delta=snapshot.avg_utility,
            risk_delta=snapshot.risk_mean,
            terminated=False,
            termination_reason=None,
        )

        tick = max(snapshot.round - 1, 0)
        envelope_data = {
            "api_version": API_VERSION,
            "kind": "step",
            "ok": True,
            "session_id": session_id,
            "tick": tick,
            "state_hash": _snapshot_state_hash(snapshot),
            "world_state": _snapshot_world_state(snapshot),
            "result": result.to_dict(),
            "extensions": _snapshot_extensions(snapshot),
        }

        envelope = normalize_response_envelope(envelope_data)

        # Emit step metric with latency and snapshot fields
        log_metric(
            "step_complete",
            session_id=session_id,
            round=snapshot.round,
            phase=str(snapshot.phase),
            latency_ms=latency_ms,
            http_status=200,
            avg_reward=float(getattr(snapshot, "avg_reward", None)),
            avg_utility=float(getattr(snapshot, "avg_utility", None)),
            success_rate=float(getattr(snapshot, "success_rate", None)),
        )

        return envelope.to_dict()
    except KeyError:
        log_metric("error", error_kind="step_not_found", session_id=session_id)
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found")
    except RuntimeError as exc:
        log_metric("error", error_kind="step_runtime_error", session_id=session_id, error=str(exc))
        raise HTTPException(status_code=400, detail=f"Step failed: {exc}")
    except Exception as exc:
        log_metric("error", error_kind="step_unexpected", session_id=session_id, error=str(exc))
        raise HTTPException(status_code=500, detail=f"Step failed: {exc}")


# ===================================================================
# Endpoint: Snapshot (current state)
# ===================================================================


@app.get("/sessions/{session_id}/snapshot")
async def snapshot(session_id: str) -> dict[str, Any]:
    """Get current RL state snapshot without executing a step.

    Args:
        session_id: Session identifier.

    Returns:
        JSON-serialized ResponseEnvelope with kind="snapshot".
    """
    try:
        manager = get_session_manager()
        snapshot = manager.snapshot_session(session_id)

        envelope_data = {
            "api_version": API_VERSION,
            "kind": "snapshot",
            "ok": True,
            "session_id": session_id,
            "tick": snapshot.round,
            "state_hash": _snapshot_state_hash(snapshot),
            "world_state": _snapshot_world_state(snapshot),
            "extensions": _snapshot_extensions(snapshot),
        }

        envelope = normalize_response_envelope(envelope_data)
        log_metric("snapshot", session_id=session_id, round=snapshot.round, phase=str(snapshot.phase))
        return envelope.to_dict()
    except KeyError:
        log_metric("error", error_kind="snapshot_not_found", session_id=session_id)
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found")


# ===================================================================
# Endpoint: Reset Session
# ===================================================================


@app.post("/sessions/{session_id}/reset")
async def reset_session(session_id: str) -> dict[str, Any]:
    """Reset session to initial state (tick=0).

    Args:
        session_id: Session identifier.

    Returns:
        JSON-serialized ResponseEnvelope with kind="reset".
    """
    try:
        manager = get_session_manager()
        snapshot = manager.reset_session(session_id)

        envelope_data = {
            "api_version": API_VERSION,
            "kind": "reset",
            "ok": True,
            "session_id": session_id,
            "tick": snapshot.round,
            "state_hash": _snapshot_state_hash(snapshot),
            "world_state": _snapshot_world_state(snapshot),
            "extensions": _snapshot_extensions(snapshot),
        }

        envelope = normalize_response_envelope(envelope_data)
        log_metric("reset", session_id=session_id, round=snapshot.round, phase=str(snapshot.phase))
        return envelope.to_dict()
    except KeyError:
        log_metric("error", error_kind="reset_not_found", session_id=session_id)
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found")


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
    try:
        manager = get_session_manager()
        snapshot = manager.snapshot_session(session_id)
        return {"state_hash": _snapshot_state_hash(snapshot)}
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found")


# ===================================================================
# Runtime Bridge v2: RL Session Endpoints (SDD §12.8-12.9)
# ===================================================================


def _rl_snapshot_to_dict(s: Any) -> dict[str, Any]:
    """Serialize a FrameSnapshot to the RL-session response dict.

    Single source of truth for the /rl_sessions/* snapshot shape (init, step,
    snapshot, reset, apply-event all use this).
    """
    return {
        "session_id": s.session_id,
        "round": s.round,
        "tick": s.tick,
        "warm": s.warm,
        "cycle_level": s.cycle_level,
        "s3_score": s.s3_score,
        "env_gamma": s.env_gamma,
        "entropy": s.entropy,
        "q_std": s.q_std,
        "p_aggressive": s.p_aggressive,
        "p_defensive": s.p_defensive,
        "p_balanced": s.p_balanced,
        "pi_aggressive": s.pi_aggressive,
        "pi_defensive": s.pi_defensive,
        "pi_balanced": s.pi_balanced,
        "q_mean_aggressive": s.q_mean_aggressive,
        "q_mean_defensive": s.q_mean_defensive,
        "q_mean_balanced": s.q_mean_balanced,
        "avg_reward": s.avg_reward,
        "avg_utility": s.avg_utility,
        "success_rate": s.success_rate,
        "risk_mean": s.risk_mean,
        "stress_mean": s.stress_mean,
        "world_scarcity": s.world_scarcity,
        "world_threat": s.world_threat,
        "world_noise": s.world_noise,
        "world_intel": s.world_intel,
        "phase": s.phase,
        # Space-A personality aggregate + displacement DV (P7-H)
        "mean_personality": s.mean_personality,
        "personality_displacement": s.personality_displacement,
        # Population-mean bifurcation proximity (Space-A truth, app-calibrated) —
        # the H1 primary DV. The frontend records THIS as the experiment proximity
        # (not the bounded display avatar's). None until the Space-A personality
        # aggregate is populated (e.g. pre-warm).
        "bifurcation_proximity": (
            compute_bifurcation_distance(
                np.asarray(s.mean_personality, dtype=float), mode="projection"
            )["bifurcation_proximity"]
            if len(s.mean_personality) == 9 else None
        ),
    }


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
    # Reject unknown personality_mode loudly (422) at the boundary. Otherwise the
    # engine dispatch silently falls back to an all-zero personality — far from
    # the baseline attractor — corrupting the session (e.g. Godot's 'balanced').
    if req.personality_mode not in VALID_PERSONALITY_MODES:
        raise HTTPException(
            status_code=422,
            detail=(
                f"personality_mode must be one of {sorted(VALID_PERSONALITY_MODES)}; "
                f"got {req.personality_mode!r}"
            ),
        )

    # ── Resolve sub-critical seeding (the validated H1 regime) ────────────────
    seed = req.seed if req.seed is not None else 42
    effective_mode = req.personality_mode
    fixed_vec: dict[str, float] | None = None
    if req.sub_critical_headroom is not None and req.fixed_personality_vector is not None:
        raise HTTPException(
            status_code=422,
            detail="sub_critical_headroom and fixed_personality_vector are mutually exclusive",
        )
    if req.sub_critical_headroom is not None:
        if not 0.0 < req.sub_critical_headroom <= 1.0:
            raise HTTPException(
                status_code=422,
                detail=f"sub_critical_headroom must be in (0, 1]; got {req.sub_critical_headroom}",
            )
        p0 = sample_sub_critical(np.random.RandomState(seed), headroom=req.sub_critical_headroom)
        fixed_vec = {k: float(p0[i]) for i, k in enumerate(_BF_FEATURE_NAMES)}
        effective_mode = "static"  # all players share the sub-critical P₀
    elif req.fixed_personality_vector is not None:
        if len(req.fixed_personality_vector) != 9:
            raise HTTPException(
                status_code=422,
                detail=f"fixed_personality_vector must have 9 elements, got {len(req.fixed_personality_vector)}",
            )
        fixed_vec = {k: float(v) for k, v in zip(_BF_FEATURE_NAMES, req.fixed_personality_vector)}
        effective_mode = "static"

    try:
        # Build RLSessionConfig with provided parameters
        config = RLSessionConfig(
            n_players=req.n_players,
            n_rounds=req.n_rounds,
            burn_in=req.burn_in,
            seed=seed,
            personality_mode=effective_mode,
            fixed_personality_vector=fixed_vec,
            space_a_events_enabled=req.space_a_events_enabled,
            # All other fields use defaults (BL2-locked: alpha_lo=0.005, alpha_hi=0.40, beta=3.0, etc.)
        )
        
        # Get global session manager
        manager = get_session_manager()
        
        # Initialize session (validates config, creates engine, returns initial snapshot).
        # Run the synchronous, CPU-bound init in a worker thread so the event loop
        # stays free to accept concurrent connections (e.g. /metrics/events from the
        # Godot client). Otherwise the burn-in loop blocks uvicorn and concurrent
        # requests get refused → RESULT_CANT_CONNECT on the client side.
        start = time.time()
        session_id, initial_snapshot = await asyncio.to_thread(
            manager.initialize_session, config=config
        )
        latency_ms = int((time.time() - start) * 1000)

        # Emit RL session init metric
        try:
            log_metric(
                "rl_session_start",
                session_id=session_id,
                n_players=getattr(config, "n_players", None),
                n_rounds=getattr(config, "n_rounds", None),
                burn_in=getattr(config, "burn_in", None),
                seed=getattr(config, "seed", None),
                personality_mode=getattr(config, "personality_mode", None),
                latency_ms=latency_ms,
                status=200,
            )
        except Exception:
            pass
        
        # Convert FrameSnapshot to dict for JSON response
        snapshot_dict = _rl_snapshot_to_dict(initial_snapshot)

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
        # Offload the synchronous RL step to a worker thread so the event loop
        # stays responsive to concurrent requests (metrics, snapshot polls).
        # A blocked loop refuses concurrent connections → RESULT_CANT_CONNECT.
        start = time.time()
        snapshot = await asyncio.to_thread(manager.step_session, session_id)
        latency_ms = int((time.time() - start) * 1000)
        
        # Convert FrameSnapshot to dict
        snapshot_dict = _rl_snapshot_to_dict(snapshot)
        
        # Emit RL step metric
        try:
            log_metric(
                "rl_step_complete",
                session_id=session_id,
                round=snapshot.round,
                phase=str(snapshot.phase),
                latency_ms=latency_ms,
                http_status=200,
                avg_reward=float(getattr(snapshot, "avg_reward", None)),
            )
        except Exception:
            pass

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
        snapshot_dict = _rl_snapshot_to_dict(snapshot)
        
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
        snapshot_dict = _rl_snapshot_to_dict(snapshot)
        
        return RLSessionStepResponse(
            session_id=session_id,
            snapshot=snapshot_dict,
        )
    
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found")


# ── Space-A personality events on an RL session (P7-H) ────────────────────────


class RLApplyEventRequest(BaseModel):
    """POST /rl_sessions/{session_id}/apply-event payload.

    Either pass an explicit ``displacement`` (a 9D Space-A delta), or omit it to
    have the server design one from the session's current population-mean
    personality. ``group`` selects the A/B arm: experiment → v1-aligned,
    control → random direction of equal magnitude.
    """

    displacement: list[float] | None = None
    group: str = "experiment"
    target: str = "personality_shift"
    intensity_scale: float = 1.0
    seed: int | None = None


@app.post("/rl_sessions/{session_id}/apply-event", response_model=RLSessionStepResponse)
async def rl_apply_event(session_id: str, req: RLApplyEventRequest) -> RLSessionStepResponse:
    """Apply a Space-A bifurcation event to an RL session's population.

    The event perturbs every player's personality in Space A and persists across
    later rounds; the resulting snapshot carries the updated mean_personality and
    the displacement DV. Requires the session to have been initialized with
    space_a_events_enabled=True.
    """
    manager = get_session_manager()

    # Resolve the displacement: explicit, or designed from the current mean.
    if req.displacement is not None:
        if len(req.displacement) != 9:
            raise HTTPException(
                status_code=422,
                detail=f"displacement must have 9 elements, got {len(req.displacement)}",
            )
        displacement = req.displacement
    else:
        try:
            current = manager.snapshot_session(session_id)
        except KeyError:
            raise HTTPException(status_code=404, detail=f"Session {session_id} not found")
        mean_a = current.mean_personality
        rng = np.random.RandomState(req.seed) if req.group == "control" else None
        event = design_bifurcation_event(
            mean_a,
            target=req.target,
            intensity_scale=req.intensity_scale,
            app_calibrated=True,
            direction_mode=_direction_mode_for_group(req.group),
            rng=rng,
        )
        displacement = event["displacement"]

    try:
        snapshot = await asyncio.to_thread(
            manager.apply_personality_event, session_id, displacement
        )
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found")
    except RuntimeError as exc:
        # space_a_events_enabled is False on this session.
        raise HTTPException(status_code=409, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))

    return RLSessionStepResponse(session_id=session_id, snapshot=_rl_snapshot_to_dict(snapshot))


@app.get("/rl_sessions/{session_id}/personality")
async def rl_session_personality(session_id: str) -> dict[str, Any]:
    """Report the session's Space-A population personality, the Space-B mapping,
    the displacement DV, and the current bifurcation proximity.
    """
    manager = get_session_manager()
    try:
        snapshot = manager.snapshot_session(session_id)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found")

    mean_a = np.asarray(snapshot.mean_personality, dtype=float)
    proximity = compute_bifurcation_distance(mean_a, mode="projection")
    return {
        "session_id": session_id,
        "round": snapshot.round,
        "feature_names": _BF_FEATURE_NAMES,
        "mean_personality_space_a": snapshot.mean_personality,
        "mean_personality_space_b": space_a_to_b(mean_a).tolist(),
        "personality_displacement": snapshot.personality_displacement,
        "bifurcation": proximity,
    }


@app.post("/metrics/events", response_model=MetricsEventsResponse)
async def post_metrics_events(events: list[dict[str, Any]]) -> MetricsEventsResponse:
    """Persist a batch of metric events to today's JSONL log."""
    try:
        persist_metrics_batch(events)
        return MetricsEventsResponse(ok=True, received=len(events))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Metrics write failed: {exc}")


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
# Endpoint: SBERT Personality Inference (offline, v7 MLP)
# ===================================================================


@app.post("/personality/infer_sbert", response_model=PersonalityInferResponse)
async def personality_infer_sbert(req: PersonalityInferRequest) -> PersonalityInferResponse:
    """Infer 9-trait personality vector using offline SBERT + v7 MLP.

    Does NOT require LLM environment variables.  Returns a deterministic
    (temperature=0) result from the locally trained model.
    """
    try:
        vector, meta = infer_personality_vector_sbert(req.text)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"SBERT inference failed: {exc}")

    return PersonalityInferResponse(
        request_id=str(meta.get("request_id")),
        text=req.text.strip(),
        vector=vector,
        model=str(meta.get("model")),
        temperature=float(meta.get("temperature")),
        logged=False,
        log_error=None,
    )


# NOTE: The __main__ / uvicorn entry point lives at the END of this file.
# It must be defined after ALL @app routes — otherwise `python -m api.server`
# calls uvicorn.run() (which blocks) before later routes register, silently
# dropping every endpoint defined below this point (bifurcation, player-test,
# survey). See git history / dev log "問題 3: server entrypoint ordering".


# ── Bifurcation API ──────────────────────────────────────────────────────────

# NOTE: app_calibrated defaults to True — the bifurcation API is the
# application layer, so proximity uses the v1-v2 projection + ε_c_app=0.11
# (avoids instant saturation; see bifurcation_detector.py B1 calibration).
class BifurcationDetectRequest(BaseModel):
    personality_vector: list[float]
    app_calibrated: bool = True

class BifurcationEventRequest(BaseModel):
    personality_vector: list[float]
    target: str = "personality_shift"
    intensity_scale: float = 1.0
    app_calibrated: bool = True
    # A/B arm: "experiment" → v1-aligned, "control" → random direction. The
    # control arm MUST be requested with group="control" so the manipulation is
    # real; otherwise both arms receive identical aligned events.
    group: str = "experiment"
    seed: int | None = None

class BifurcationSequenceRequest(BaseModel):
    personality_vector: list[float]
    n_steps: int = 5
    target: str = "personality_shift"
    app_calibrated: bool = True
    group: str = "experiment"
    seed: int | None = None


def _direction_mode_for_group(group: str) -> str:
    """Map an A/B group label to the event direction mode."""
    return "random" if group == "control" else "aligned"


@app.post("/bifurcation/detect")
async def bifurcation_detect(req: BifurcationDetectRequest) -> dict[str, Any]:
    """Compute bifurcation proximity and sensitive direction for a personality vector."""
    if len(req.personality_vector) != 9:
        raise HTTPException(
            status_code=422,
            detail=f"personality_vector must have 9 elements, got {len(req.personality_vector)}",
        )
    import numpy as np
    pv = np.array(req.personality_vector)
    mode = "projection" if req.app_calibrated else "euclidean"
    bifurc = compute_bifurcation_distance(pv, mode=mode)
    sens = compute_sensitive_direction(pv)
    return {
        "feature_names": _BF_FEATURE_NAMES,
        "bifurcation": bifurc,
        "sensitive_direction": sens,
    }


@app.post("/bifurcation/event")
async def bifurcation_event(req: BifurcationEventRequest) -> dict[str, Any]:
    """Design a single bifurcation-triggering game event."""
    if len(req.personality_vector) != 9:
        raise HTTPException(
            status_code=422,
            detail=f"personality_vector must have 9 elements, got {len(req.personality_vector)}",
        )
    import numpy as np
    pv = np.array(req.personality_vector)
    rng = np.random.RandomState(req.seed) if req.group == "control" else None
    event = design_bifurcation_event(
        pv, target=req.target, intensity_scale=req.intensity_scale,
        app_calibrated=req.app_calibrated,
        direction_mode=_direction_mode_for_group(req.group), rng=rng,
    )
    return event


@app.post("/bifurcation/sequence")
async def bifurcation_sequence(req: BifurcationSequenceRequest) -> dict[str, Any]:
    """Plan a multi-step event sequence to guide a trajectory toward bifurcation."""
    if len(req.personality_vector) != 9:
        raise HTTPException(
            status_code=422,
            detail=f"personality_vector must have 9 elements, got {len(req.personality_vector)}",
        )
    if not (1 <= req.n_steps <= 20):
        raise HTTPException(status_code=422, detail="n_steps must be between 1 and 20")
    import numpy as np
    pv = np.array(req.personality_vector)
    sequence = event_sequence_planner(
        pv, n_steps=req.n_steps, target=req.target,
        app_calibrated=req.app_calibrated,
        direction_mode=_direction_mode_for_group(req.group), seed=req.seed,
    )
    return sequence


@app.post("/bifurcation/zone-check")
async def bifurcation_zone_check(req: BifurcationDetectRequest) -> dict[str, Any]:
    """Check whether the personality is in the effective bifurcation event zone."""
    if len(req.personality_vector) != 9:
        raise HTTPException(
            status_code=422,
            detail=f"personality_vector must have 9 elements, got {len(req.personality_vector)}",
        )
    import numpy as np
    pv = np.array(req.personality_vector)
    return is_in_effective_zone(pv, app_calibrated=req.app_calibrated)


@app.post("/bifurcation/rollback")
async def bifurcation_rollback(req: BifurcationDetectRequest) -> dict[str, Any]:
    """Estimate free-dynamics rounds needed to recover from current bifurcation proximity."""
    if len(req.personality_vector) != 9:
        raise HTTPException(
            status_code=422,
            detail=f"personality_vector must have 9 elements, got {len(req.personality_vector)}",
        )
    import numpy as np
    pv = np.array(req.personality_vector)
    return rollback_distance(pv, app_calibrated=req.app_calibrated)


# ── Space-B-input bifurcation routes ──────────────────────────────────────────
# Godot holds personalities in Space B (display / operating coordinates). The
# live loop must NOT compute proximity or design events on the raw Space-B scale
# (it saturates proximity at 1.0 and applies events on the wrong scale). These
# parallel routes accept a Space-B personality_vector, map it to Space A via
# b_to_a() so the bifurcation geometry (ε_c_app, v1-v2 plane) is the validated
# apparatus, then map event DISPLACEMENTS back to Space B for the frontend to
# apply. proximity is the canonical Space-A DV and is returned unchanged ([0,1]).
# The A↔B transform is an isotropic affine, so unit DIRECTIONS are invariant.
# See simulation/personality_space.py.

def _event_to_space_b(event: dict[str, Any]) -> dict[str, Any]:
    """Re-express a Space-A event's displacement fields in Space B for the frontend.

    Only displacement-type fields rescale (displacement, feature_deltas,
    magnitude); the unit ``direction`` and the ``proximity`` DV are scale-free
    and kept as designed in Space A.
    """
    import numpy as np
    out = dict(event)
    disp_b = space_disp_a_to_b(np.asarray(event["displacement"], dtype=float))
    out["displacement"] = disp_b.tolist()
    out["feature_deltas"] = dict(zip(_BF_FEATURE_NAMES, disp_b.tolist()))
    out["magnitude"] = float(np.linalg.norm(disp_b))
    out["space"] = "B"  # displacement/feature_deltas are Space-B; proximity is Space-A
    return out


@app.post("/bifurcation/b/detect")
async def bifurcation_b_detect(req: BifurcationDetectRequest) -> dict[str, Any]:
    """Space-B input variant of /bifurcation/detect (maps B→A, measures DV in A)."""
    if len(req.personality_vector) != 9:
        raise HTTPException(
            status_code=422,
            detail=f"personality_vector must have 9 elements, got {len(req.personality_vector)}",
        )
    pv_a = space_b_to_a(req.personality_vector)
    mode = "projection" if req.app_calibrated else "euclidean"
    return {
        "feature_names": _BF_FEATURE_NAMES,
        "bifurcation": compute_bifurcation_distance(pv_a, mode=mode),
        "sensitive_direction": compute_sensitive_direction(pv_a),
        "input_space": "B",
    }


@app.post("/bifurcation/b/event")
async def bifurcation_b_event(req: BifurcationEventRequest) -> dict[str, Any]:
    """Space-B input variant of /bifurcation/event (event designed in A, displacement returned in B)."""
    if len(req.personality_vector) != 9:
        raise HTTPException(
            status_code=422,
            detail=f"personality_vector must have 9 elements, got {len(req.personality_vector)}",
        )
    import numpy as np
    pv_a = space_b_to_a(req.personality_vector)
    rng = np.random.RandomState(req.seed) if req.group == "control" else None
    event = design_bifurcation_event(
        pv_a, target=req.target, intensity_scale=req.intensity_scale,
        app_calibrated=req.app_calibrated,
        direction_mode=_direction_mode_for_group(req.group), rng=rng,
    )
    return {**_event_to_space_b(event), "input_space": "B"}


@app.post("/bifurcation/b/sequence")
async def bifurcation_b_sequence(req: BifurcationSequenceRequest) -> dict[str, Any]:
    """Space-B input variant of /bifurcation/sequence (each step's displacement returned in B)."""
    if len(req.personality_vector) != 9:
        raise HTTPException(
            status_code=422,
            detail=f"personality_vector must have 9 elements, got {len(req.personality_vector)}",
        )
    if not (1 <= req.n_steps <= 20):
        raise HTTPException(status_code=422, detail="n_steps must be between 1 and 20")
    import numpy as np
    pv_a = space_b_to_a(req.personality_vector)
    sequence = event_sequence_planner(
        pv_a, n_steps=req.n_steps, target=req.target,
        app_calibrated=req.app_calibrated,
        direction_mode=_direction_mode_for_group(req.group), seed=req.seed,
    )
    sequence["events"] = [_event_to_space_b(ev) for ev in sequence["events"]]
    sequence["total_displacement"] = space_disp_a_to_b(
        np.asarray(sequence["total_displacement"], dtype=float)
    ).tolist()
    sequence["input_space"] = "B"
    return sequence


@app.post("/bifurcation/b/zone-check")
async def bifurcation_b_zone_check(req: BifurcationDetectRequest) -> dict[str, Any]:
    """Space-B input variant of /bifurcation/zone-check (proximity DV measured in A)."""
    if len(req.personality_vector) != 9:
        raise HTTPException(
            status_code=422,
            detail=f"personality_vector must have 9 elements, got {len(req.personality_vector)}",
        )
    pv_a = space_b_to_a(req.personality_vector)
    return {**is_in_effective_zone(pv_a, app_calibrated=req.app_calibrated), "input_space": "B"}


@app.post("/bifurcation/b/rollback")
async def bifurcation_b_rollback(req: BifurcationDetectRequest) -> dict[str, Any]:
    """Space-B input variant of /bifurcation/rollback (recovery estimated in A)."""
    if len(req.personality_vector) != 9:
        raise HTTPException(
            status_code=422,
            detail=f"personality_vector must have 9 elements, got {len(req.personality_vector)}",
        )
    pv_a = space_b_to_a(req.personality_vector)
    return {**rollback_distance(pv_a, app_calibrated=req.app_calibrated), "input_space": "B"}


# ── Passive event choice API ──────────────────────────────────────────────────
# 「被動版」：給人格向量 → AI 分身依 weights·personality 自動選擇選項並結算，
# 供「你的分身做了選擇 X，結果…」展示，以及人格→選擇對應關係的驗證。

_EVENT_TEMPLATES_JSON = ROOT / "docs" / "personality_dungeon_v1" / "02_event_templates_v1.json"
_event_loader: EventLoader | None = None


def _get_event_loader() -> EventLoader:
    """Lazily construct and cache the EventLoader (templates are read-only)."""
    global _event_loader
    if _event_loader is None:
        _event_loader = EventLoader(_EVENT_TEMPLATES_JSON)
    return _event_loader


class EventChooseRequest(BaseModel):
    personality_vector: list[float]
    # 指定事件；None → 隨機抽一個
    event_id: str | None = None
    # 可選的玩家狀態（stress/noise/...），影響 risk；省略則全 0
    state: dict[str, float] | None = None
    # 隨機種子（控制隨機抽事件與成敗擲骰），供可重現
    seed: int | None = None
    # True → 不擲骰，success = success_prob >= 0.5
    deterministic_outcome: bool = False


@app.post("/event/choose")
async def event_choose(req: EventChooseRequest) -> dict[str, Any]:
    """Let the personality avatar passively face an event and resolve the outcome.

    Returns the chosen action plus per-option utilities / lean probabilities and
    the success/failure result. Read-only: applies no trait or state deltas.
    """
    loader = _get_event_loader()
    if len(req.personality_vector) != len(loader.dimensions_order):
        raise HTTPException(
            status_code=422,
            detail=(
                f"personality_vector must have {len(loader.dimensions_order)} "
                f"elements, got {len(req.personality_vector)}"
            ),
        )
    if req.event_id is not None and req.event_id not in loader.template_by_id:
        raise HTTPException(status_code=404, detail=f"unknown event_id: {req.event_id}")

    personality = vector_to_personality(loader, req.personality_vector)
    rng = random.Random(req.seed) if req.seed is not None else None
    return resolve_passive_choice(
        loader,
        personality,
        event_id=req.event_id,
        state=req.state,
        rng=rng,
        deterministic_outcome=req.deterministic_outcome,
    )


# ── A/B Test API ──────────────────────────────────────────────────────────────

class ABAssignRequest(BaseModel):
    session_id: str
    # run_id / participant_id 仍可由前端送來（向後相容），但 assign 不再依它們分組——
    # 人格迭代研究已於 2026-06-18 棄用、生命解耦，分組純走 session 的 count-balance。
    # 連結用的 run_id/participant_id 走 /player-test/start 記錄。
    run_id: str = ""
    participant_id: str = "dev"

class ABRecordStepRequest(BaseModel):
    session_id: str
    personality_before: list[float]
    personality_after: list[float]
    proximity: float
    step_index: int = 0


@app.post("/bifurcation/ab-test/assign")
async def ab_test_assign(req: ABAssignRequest) -> dict[str, Any]:
    """Assign or retrieve A/B test group for a session.

    Returns {"session_id", "group": "control"|"experiment", "existing": bool}.
    """
    mgr = _get_ab_manager()
    result = mgr.assign_session(req.session_id)
    await _p7h_save(mgr)  # persist count-balance across restarts
    return result


@app.post("/bifurcation/ab-test/record-step")
async def ab_test_record_step(req: ABRecordStepRequest) -> dict[str, Any]:
    """Record one event step's before/after personality snapshot."""
    if len(req.personality_before) != 9 or len(req.personality_after) != 9:
        raise HTTPException(status_code=422, detail="personality vectors must have 9 elements")
    _get_ab_manager().record_event_step(
        session_id=req.session_id,
        personality_before=req.personality_before,
        personality_after=req.personality_after,
        proximity=req.proximity,
        step_index=req.step_index,
    )
    return {"ok": True, "session_id": req.session_id}


# NOTE: static "/summary" MUST precede the "/{session_id}" catch-all (route shadowing).
@app.get("/bifurcation/ab-test/summary")
async def ab_test_summary() -> dict[str, Any]:
    """Return group-level statistics across all recorded sessions."""
    return _get_ab_manager().summary()


@app.get("/bifurcation/ab-test/{session_id}")
async def ab_test_get_session(session_id: str) -> dict[str, Any]:
    """Get the full record for a session."""
    record = _get_ab_manager().get_session(session_id)
    if record is None:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found")
    return record


# ── Player Test API ───────────────────────────────────────────────────────────

class PlayerTestStartRequest(BaseModel):
    session_id: str
    group: str
    player_alias: str = "anon"
    # 遺言診斷（選填；由 Godot 前端在 start 時一起送來）
    will_text: str = ""
    will_sbert_vector: list[float] = []
    will_personality_vector: list[float] = []
    will_recklessness: float = -1.0
    will_intensity: float = -1.0
    will_cadence: int = -1
    is_human: bool = False  # True=真人 Godot 前端；False=程式 API 呼叫（wsim 等）
    # 連結同一受試者的 3 週期（人格迭代 A/B 已於 2026-06-18 棄用——生命已解耦）
    run_id: str = ""
    cycle_index: int = -1
    # naive vs 實驗者權威判別子：naive 用分配代碼（P01…），實驗者試玩用 "dev"。預設 "dev"
    # 確保未帶此欄的舊客戶端/試玩不會被誤計為 naive pilot。
    participant_id: str = "dev"

class PlayerTestStepRequest(BaseModel):
    session_id: str
    action_text: str = ""
    personality_before: list[float]
    personality_after: list[float]
    proximity_before: float = 0.0
    proximity_after: float = 0.0
    event_type: str = "none"
    response_time_ms: float = 0.0

class PlayerTestEndRequest(BaseModel):
    session_id: str
    collapse_reason: str = ""  # "proximity+passive_failure"|"max_rounds"|"phase_ended"


@app.post("/player-test/start")
async def player_test_start(req: PlayerTestStartRequest) -> dict[str, Any]:
    """Start a new player test session."""
    return _get_tracker().start_session(
        session_id=req.session_id,
        group=req.group,  # type: ignore[arg-type]
        player_alias=req.player_alias,
        will_text=req.will_text,
        will_sbert_vector=req.will_sbert_vector,
        will_personality_vector=req.will_personality_vector,
        will_recklessness=req.will_recklessness,
        will_intensity=req.will_intensity,
        will_cadence=req.will_cadence,
        is_human=req.is_human,
        run_id=req.run_id,
        cycle_index=req.cycle_index,
        participant_id=req.participant_id,
    )


@app.post("/player-test/step")
async def player_test_step(req: PlayerTestStepRequest) -> dict[str, Any]:
    """Record one step of a player test session."""
    for label, v in [("before", req.personality_before), ("after", req.personality_after)]:
        if len(v) != 9:
            raise HTTPException(
                status_code=422,
                detail=f"personality_{label} must have 9 elements, got {len(v)}",
            )
    return _get_tracker().record_step(
        session_id=req.session_id,
        action_text=req.action_text,
        personality_before=req.personality_before,
        personality_after=req.personality_after,
        proximity_before=req.proximity_before,
        proximity_after=req.proximity_after,
        event_type=req.event_type,
        response_time_ms=req.response_time_ms,
    )


@app.post("/player-test/end")
async def player_test_end(req: PlayerTestEndRequest) -> dict[str, Any]:
    """End a player test session and compute derived metrics."""
    tracker = _get_tracker()
    result = tracker.end_session(req.session_id, collapse_reason=req.collapse_reason)
    if not result.get("ok"):
        raise HTTPException(status_code=404, detail=result.get("error", "unknown"))
    await _p7h_save(tracker)  # persist completed session
    return result


# NOTE: static "/summary" MUST be declared before the "/{session_id}" catch-all,
# otherwise FastAPI matches "summary" as a session_id (route shadowing).
@app.get("/player-test/summary")
async def player_test_group_summary() -> dict[str, Any]:
    """Group-level trajectory statistics across all completed player test sessions."""
    return _get_tracker().group_summary()


@app.get("/player-test/{session_id}")
async def player_test_get(session_id: str) -> dict[str, Any]:
    """Get full trajectory data for a player test session."""
    data = _get_tracker().get_session(session_id)
    if data is None:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found")
    return data


# ── Survey API ────────────────────────────────────────────────────────────────

class SurveySubmitRequest(BaseModel):
    session_id: str
    group: str
    q1_naturalness: int
    q2_fun: int
    q3_replay: int
    q4_continuity: int = 0  # 角色延續感（迭代研究 secondary DV）；0=未作答（舊客戶端相容）
    manipulation_awareness: str = ""  # 開放 debrief（demand-characteristics 稽核）
    q1_comment: str = ""
    q2_comment: str = ""
    q3_comment: str = ""
    overall_comments: str = ""


@app.get("/survey/questions")
async def survey_questions() -> dict[str, Any]:
    """Return the survey question list."""
    from api.survey_manager import QUESTIONS
    return {"questions": QUESTIONS}


@app.post("/survey/submit")
async def survey_submit(req: SurveySubmitRequest) -> dict[str, Any]:
    """Submit a survey response for a completed player test session."""
    survey = _get_survey()
    result = survey.submit(
        session_id=req.session_id,
        group=req.group,
        q1=req.q1_naturalness,
        q2=req.q2_fun,
        q3=req.q3_replay,
        q4=req.q4_continuity,
        manipulation_awareness=req.manipulation_awareness,
        q1_comment=req.q1_comment,
        q2_comment=req.q2_comment,
        q3_comment=req.q3_comment,
        overall_comments=req.overall_comments,
    )
    if not result.get("ok"):
        raise HTTPException(status_code=422, detail=result.get("error"))
    await _p7h_save(survey)  # persist survey response
    return result


@app.get("/survey/summary")
async def survey_summary() -> dict[str, Any]:
    """Return group-level survey statistics."""
    return _get_survey().summary()


# ── Personality Ecology API (meta-layer, Path B rotation) ─────────────────────
# 跨玩家人格生態：上傳冒險→投影原型→循環克制評分→更新生態→L0–L3 旋轉判定。
# 設計見 人格生態評分_規劃_v1.md。

class EcologySubmitRequest(BaseModel):
    personality_9d: list[float]
    run_id: str = ""
    session_id: str = ""
    outcome: dict[str, Any] = {}
    seen_scarcity: list[float] = []   # 乙：author 前前端顯示的稀缺佔比（純記錄，β-instrument 用）


@app.post("/ecology/submit")
async def ecology_submit(req: EcologySubmitRequest) -> dict[str, Any]:
    """Ingest one adventure: project archetype, score via cyclic dominance,
    update ecology, return the player's score + current ecology snapshot."""
    if len(req.personality_9d) != 9:
        raise HTTPException(status_code=422, detail="personality_9d must be length 9")
    if req.seen_scarcity and len(req.seen_scarcity) != 3:
        raise HTTPException(status_code=422, detail="seen_scarcity must be length 3 (or omitted)")
    result = _get_ecology().submit(
        personality_9d=req.personality_9d,
        run_id=req.run_id,
        session_id=req.session_id,
        outcome=req.outcome,
        seen_scarcity=req.seen_scarcity,
    )
    await _ecology_save()
    # (ii) 單一幣：真實玩家的生態 coins 累加進錢包。真人判準＝有 session_id + 真實冒險 outcome
    # （V2 submit(will, get_session_id(), get_session_id(), outcome) 的簽名）。
    # 2026-06-23 更正：舊條件 `if not req.run_id` 反掉——真人 run_id 非空(=session_id)，那條 credit
    # 對真玩家從不觸發；in-process replay/smoke 才 run_id 空、無 outcome → 不該 credit。
    if req.session_id and req.outcome:
        coins = int(result.get("coins", 0))
        if coins:
            _get_wallet().credit("ecology", coins, note="session %s" % req.session_id[:8])
            await _wallet_save()
        result["balance"] = _get_wallet().balance()
    return result


@app.get("/ecology/snapshot")
async def ecology_snapshot() -> dict[str, Any]:
    """Current archetype proportions + dynamic weights (no state change)."""
    return _get_ecology().snapshot()


@app.get("/ecology/assess")
async def ecology_assess() -> dict[str, Any]:
    """Grade the ecology's rotation L0–L3 via cycle_metrics over snapshot bins."""
    return _get_ecology().assess()


@app.get("/ecology/scarcity_variation")
async def ecology_scarcity_variation() -> dict[str, Any]:
    """乙 collection monitor：真人 live 提交面對的稀缺變異是否達標（鐵律 1）。read-only。"""
    return _get_ecology().collection_diagnostics()


# ── PVP / 地牢挑戰 API（最小 combat loop，Increment 1）─────────────────────────
# 3 派系 type-chart：地牢「剋制主人」（部署 counter(owner)），挑戰者帶剋制派系鑽破。
# H_counter 已證偽 → M 為 authored 剋制表。設計見 地牢counter-policy_L0L1介面_規劃_v1.md。

class PvpChallengeRequest(BaseModel):
    challenger_faction: str
    dungeon_id: str


@app.get("/pvp/dungeons")
async def pvp_dungeons() -> dict[str, Any]:
    """可挑戰地牢清單（顯示各地牢 deployed 派系 + 你的 Rank）。"""
    return _get_pvp().list_dungeons()


@app.post("/pvp/challenge")
async def pvp_challenge(req: PvpChallengeRequest) -> dict[str, Any]:
    """挑戰一座地牢：先扣門票（coin sink，F2 與 Rank 分離）→ authored M 判勝負 + Rank delta。"""
    wallet = _get_wallet()
    try:
        wallet.debit("pvp_ticket", wallet.params.ticket_cost,
                     note="challenge %s" % req.dungeon_id)
    except InsufficientFunds as exc:
        raise HTTPException(status_code=402, detail=str(exc))
    try:
        result = _get_pvp().challenge(req.challenger_faction, req.dungeon_id)
    except (ValueError, KeyError) as exc:
        # 請求無效（壞派系 / 地牢不存在）→ 退門票，不罰 malformed request。
        wallet.credit("pvp_ticket_refund", wallet.params.ticket_cost, note="challenge rejected")
        await _wallet_save()
        if isinstance(exc, KeyError):
            raise HTTPException(status_code=404, detail="dungeon not found: %s" % req.dungeon_id)
        raise HTTPException(status_code=422, detail=str(exc))
    await _pvp_save()
    await _wallet_save()
    result["ticket_charged"] = wallet.params.ticket_cost   # F2：固定 coin 扣款，不碰 Rank
    result["balance"] = wallet.balance()
    return result


# ── 玩家錢包 API（Increment 2 經濟；(ii) 單一幣）──────────────────────────────────
# source: 生態(後端 credit) + 存活(前端 credit)；sink: PvP 門票。防禦升級待玩家地牢(§7)。

class WalletCreditRequest(BaseModel):
    source: str
    amount: int
    note: str = ""


@app.get("/wallet")
async def wallet_get() -> dict[str, Any]:
    """錢包餘額 + 門票成本 + 近期帳目。"""
    return _get_wallet().state()


@app.post("/wallet/credit")
async def wallet_credit(req: WalletCreditRequest) -> dict[str, Any]:
    """前端 credit agnostic source（存活）；生態 coins 為後端權威 credit、不收此路。"""
    if req.amount < 0:
        raise HTTPException(status_code=422, detail="amount must be ≥0")
    if req.source not in _CLIENT_CREDIT_SOURCES:
        raise HTTPException(status_code=422,
                            detail="source must be one of %s" % sorted(_CLIENT_CREDIT_SOURCES))
    entry = _get_wallet().credit(req.source, req.amount, note=req.note)
    await _wallet_save()
    return {"entry": entry, "balance": _get_wallet().balance()}


# NOTE: /metrics/events is defined once above (post_metrics_events). A second
# duplicate definition used to live here but was dead code — FastAPI matches the
# first registered route, so this one never ran. Removed to avoid confusion.


# ===================================================================
# Entry Point — MUST stay at the very end (after all @app routes)
# ===================================================================

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8001,  # 8000 collides with the Godot AI MCP server under WSL mirrored networking
        log_level="info",
    )

