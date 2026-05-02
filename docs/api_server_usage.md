"""API Server Usage Documentation

## Quick Start

### Run the server locally

    ./venv/bin/python -m api.server

Server will start on `http://localhost:8000`

API documentation (Swagger) available at: `http://localhost:8000/docs`

---

## API Endpoints

### 1. Initialize Session

**Endpoint:** `POST /sessions/initialize`

**Request:**
```json
{
    "n_players": 10,
    "seed": 42
}
```

**Response:**
```json
{
    "session_id": "8ee3c2e1-a8cd-410d-abd8-e751420e95c9"
}
```

**Description:** Creates a new game session with specified number of players and optional random seed.

---

### 2. Step (Perform One Action)

**Endpoint:** `POST /sessions/{session_id}/step`

**Request:**
```json
{
    "action": "aggressive"
}
```

**Response (ResponseEnvelope):**
```json
{
    "api_version": "1.0.0",
    "kind": "step",
    "ok": true,
    "session_id": "8ee3c2e1-a8cd-410d-abd8-e751420e95c9",
    "tick": 0,
    "state_hash": "a1b2c3d4e5f6g7h8",
    "event": null,
    "choices": [],
    "personality_delta": {
        "vector": {},
        "norm": 0.0,
        "confidence": 0.0
    },
    "emotion": {
        "valence": 0.0,
        "arousal": 0.0,
        "dominance": 0.0,
        "intensity": 0.0
    },
    "world_state": {
        "scarcity": 0.5,
        "threat": 0.3,
        "noise": 0.2,
        "intel": 0.4
    },
    "player_state": {
        "health": null,
        "stamina": null,
        "stress": null,
        "utility": null,
        "alive": null,
        "personality": null,
        "memory_load": null
    },
    "result": {
        "selected_choice_id": "aggressive",
        "reward": 0.5,
        "utility_delta": 0.5,
        "risk_delta": 0.0,
        "terminated": false,
        "termination_reason": null
    },
    "warnings": [],
    "error": null,
    "extensions": {}
}
```

**Description:** Executes one step of the game engine. Returns ResponseEnvelope with:
- `tick`: Current step counter (increments each step)
- `state_hash`: Hash of current session state (useful for debugging/provenance)
- `result`: Step outcome with reward and utility changes
- All other ResponseEnvelope fields (frozen by contract)

---

### 3. Snapshot (Get Current State)

**Endpoint:** `GET /sessions/{session_id}/snapshot`

**Response (ResponseEnvelope with kind="snapshot"):**
```json
{
    "api_version": "1.0.0",
    "kind": "snapshot",
    "ok": true,
    "session_id": "8ee3c2e1-a8cd-410d-abd8-e751420e95c9",
    "tick": 1,
    "state_hash": "b2c3d4e5f6g7h8i9",
    "world_state": {
        "scarcity": 0.5,
        "threat": 0.3,
        "noise": 0.2,
        "intel": 0.4
    },
    ...
}
```

**Description:** Returns current game state without executing a step. Tick counter remains unchanged.

---

### 4. Reset Session

**Endpoint:** `POST /sessions/{session_id}/reset`

**Response (ResponseEnvelope with kind="reset"):**
```json
{
    "api_version": "1.0.0",
    "kind": "reset",
    "ok": true,
    "session_id": "8ee3c2e1-a8cd-410d-abd8-e751420e95c9",
    "tick": 0,
    ...
}
```

**Description:** Resets session to initial state. Tick counter returns to 0, world state resets to defaults.

---

### 5. Get State Hash

**Endpoint:** `GET /sessions/{session_id}/state-hash`

**Response:**
```json
{
    "state_hash": "a1b2c3d4e5f6g7h8"
}
```

**Description:** Returns current session state hash (for debugging/provenance verification).

---

## Contract Guarantees

### API Version Lock
- All responses use `api_version: "1.0.0"` (immutable)
- Breaking changes will increment version (backward compatibility maintained)
- Client must validate `api_version` matches expected version

### Session Isolation
- Each session maintains independent state
- Session IDs are immutable UUIDs
- Tick counter is independent per session

### Response Structure (ResponseEnvelope)
All endpoints return ResponseEnvelope with:
- `api_version`: API version (immutable)
- `kind`: Response type ("step", "snapshot", "reset", "error")
- `ok`: Boolean success flag
- `session_id`: Session identifier
- `tick`: Current step counter
- `state_hash`: Deterministic session state hash
- `world_state`, `player_state`: Game state (frozen schema)
- `result`: Optional step result
- `error`: Error details if `ok=False`

### State Hash Determinism
- Same session state → same hash
- Hash changes when tick increments
- Useful for debugging state mutations and regression testing

---

## Error Handling

### 404 Not Found
If session doesn't exist:
```json
{
    "detail": "Session {session_id} not found"
}
```

### 400 Bad Request
Invalid payload:
```json
{
    "detail": "Invalid request"
}
```

### 500 Internal Server Error
Server error (rare with locked contract):
```json
{
    "detail": "Internal server error"
}
```

---

## Example Usage (Python)

```python
import requests

BASE_URL = "http://localhost:8000"

# Initialize session
init_resp = requests.post(f"{BASE_URL}/sessions/initialize", json={
    "n_players": 10,
    "seed": 42
})
session_id = init_resp.json()["session_id"]

# Execute 5 steps
for i in range(5):
    step_resp = requests.post(
        f"{BASE_URL}/sessions/{session_id}/step",
        json={"action": "balanced"}
    )
    data = step_resp.json()
    print(f"Step {i}: tick={data['tick']}, reward={data['result']['reward']}")

# Get snapshot
snap_resp = requests.get(f"{BASE_URL}/sessions/{session_id}/snapshot")
print(f"Current state hash: {snap_resp.json()['state_hash']}")

# Reset session
reset_resp = requests.post(f"{BASE_URL}/sessions/{session_id}/reset")
print(f"Session reset, tick={reset_resp.json()['tick']}")
```

---

## Contract Lock Discipline

This API implements **Contract-First Development**:
1. **ResponseEnvelope schema is frozen** – no silent breaking changes
2. **Version is immutable** – upgrade requires new version number
3. **All fields are documented and validated** – unknown keys rejected
4. **Round-trip serialization is preserved** – stable JSON output

To modify the contract:
1. Update [api/schemas.py](../api/schemas.py) AND [docs/core_frontend_api_contract_v1.md](../docs/core_frontend_api_contract_v1.md)
2. Increment `API_VERSION` in [api/schemas.py](../api/schemas.py)
3. Add regression test in [tests/test_api_schemas.py](../tests/test_api_schemas.py)
4. Update this documentation

---

## Session Lifecycle

```
+--------+            +-------+            +-----------+            +-------+
| Create |  ------→  | Step  |  ------→  | Snapshot  |  ------→   | Reset |
+--------+            +-------+            +-----------+            +-------+
  (tick=0)           (tick++,            (tick unchanged)            (tick=0)
                      reward)
```

---

## Related Files

- **Server implementation:** [api/server.py](../api/server.py)
- **Schema definition:** [api/schemas.py](../api/schemas.py)
- **Contract spec:** [docs/core_frontend_api_contract_v1.md](../docs/core_frontend_api_contract_v1.md)
- **Tests:** [tests/test_api_server.py](../tests/test_api_server.py) (21 integration tests)
- **Schema tests:** [tests/test_api_schemas.py](../tests/test_api_schemas.py) (4 contract tests)
"""
