# API Contract Status

## Metadata

| Field | Value |
|---|---|
| Document ID | RB-01 |
| Owner | Backend Lead |
| Audience | Backend, Integration, QA |
| Update Frequency | Every 2-3 days |
| Dependency | runtime_bridge_matrix.md |

## Cross References

- ./runtime_bridge_matrix.md
- ./godot_python_sync.md
- ../04_phase_review/signoff_checklist.md

## Contract Inventory

| Endpoint | Contract Version | Source Schema | Target Client | Current Status | Backward Compatible | Break Risk | Owner |
|---|---|---|---|---|---|---|---|
| /personality/infer_sbert | v1 | infer request/response | Godot UI | Integrated | Yes | Medium | Backend Lead |
| /rl_sessions/initialize | v1 | session init schema | Godot Runtime | Prototype | Partial | High | Backend Lead |
| /rl_sessions/{id}/step | v1 | step request/response | Godot Runtime | Prototype | Partial | High | Integration Lead |

## Contract Change Log

| Date | Endpoint | Change Type | Impact Scope | Migration Needed | Approved By |
|---|---|---|---|---|---|
| 2026-05-27 | /rl_sessions/{id}/step | Baseline freeze candidate | Godot parser + demo loop | Yes | Integration Lead |

## Contract Validation Checklist

| Item | Status | Evidence | Notes |
|---|---|---|---|
| Request schema fixed | Done | [init route model](../../api/server.py#L389) | 已凍結最小欄位並通過合約測試 |
| Response schema fixed | Done | [step response model](../../api/server.py#L469) | 已最小化並通過合約測試 |
| Error contract defined | Done | [server.py](../../api/server.py) | 錯誤 taxonomy 已落實並通過測試 |
| Versioning policy applied | Fail | [api contract status](./api_contract_status.md) | 尚未設定 v2 升版規則 |
| Client fallback path exists | Partial | [DebugPanel](../../src/ui/DebugPanel.gd#L51) | 目前僅局部 UI 提示 |

## Phase 1 Contract Freeze Checklist

| Task ID | Goal | Pass/Fail 判準 | Evidence | Owner | Status |
|---|---|---|---|---|---|
| API-P1-01 | Freeze v1 init request | init request 必填欄位清單固定 + tests green | [init route model](../../api/server.py#L389) | Backend Lead | Done |
| API-P1-02 | Freeze v1 init response | init response 欄位固定 + Godot 解析通過 | [godot sync](./godot_python_sync.md) | Integration Lead | Done |
| API-P1-03 | Freeze v1 step response | step response 欄位固定 + UI 映射完成 | [step route](../../api/server.py#L469) | Integration Lead | Done |
| API-P1-04 | Error contract taxonomy | 定義 error code + retry policy | [server.py](../../api/server.py) | Backend Lead | Done |
| API-P1-05 | Versioning policy | v1 -> v2 升版規則定義 | [api contract status](./api_contract_status.md) | Backend Lead | Open |

## Minimum v1 Required Fields (Phase 1)

 | Endpoint | Required Request Fields | Required Response Fields | Notes |
 |---|---|---|---|
 | /rl_sessions/initialize | `n_players` (int), `n_rounds` (int), `burn_in` (int), `seed` (int|null), `personality_mode` (str) | `session_id` (str), `initial_snapshot` (object) with keys: `session_id`, `round`, `tick`, `warm` (bool), `cycle_level` (int), `s3_score` (float), `env_gamma` (float), `entropy` (float), `q_std` (float), `p_aggressive`, `p_defensive`, `p_balanced`, `pi_aggressive`, `pi_defensive`, `pi_balanced`, `q_mean_*`, `avg_reward`, `avg_utility`, `success_rate`, `risk_mean`, `stress_mean`, `world_scarcity`, `world_threat`, `world_noise`, `world_intel`, `phase` | 以 `init route model` 為基礎收斂 |
 | /rl_sessions/{id}/step | `action` (optional string) | `session_id` (str), `snapshot` (object) same shape as `initial_snapshot` above; alternatively ResponseEnvelope with `kind='step'`, `tick`, `state_hash`, `world_state`, `result` and `extensions` exposing `rl_*` metrics | 需與 Godot 映射一致 |

## Error Taxonomy (Phase 1)

Define standardized error codes and retryability for RL endpoints. Add entries to server error handling and document here.

| Code | HTTP | Description | Retryable | Notes |
|---|---:|---|---:|---|
| RL-INVALID-CONFIG | 400 | Request payload violates BL2 parameter lock or invalid types | No | Client must correct request |
| RL-SESSION-NOT-FOUND | 404 | Requested session_id does not exist | No | Non-retriable; caller should re-init session |
| RL-STEP-FAILED | 400 | Step execution failed due to session runtime (e.g., session complete) | No | Consider reset/re-init |
| RL-SERVER-ERROR | 500 | Unexpected server error (transient) | Yes (backoff) | Retry with exponential backoff; track occurrences |
| RL-BL2-PARAMETER-LOCK | 400 | BL2 anchor violation detected during init | No | Explicit about which param failed |

## Contract Validation Evidence

- Contract test run log: [logs/contract_tests_rl_sessions.log](logs/contract_tests_rl_sessions.log) — integration tests passed
- Schema snapshot: [docs/progress/01_runtime_bridge/api_contract_status.md](docs/progress/01_runtime_bridge/api_contract_status.md)
- PR: TBD (add PR link when available)


## Signoff Blockers

| Blocker ID | Description | Severity | Owner | ETA | Gate Impact |
|---|---|---|---|---|---|
| CBL-001 | step response 欄位未形成正式凍結契約 | High | Integration Lead | 2026-06-14 | Phase Gate |
| CBL-002 | 錯誤契約未標準化，重試策略不足 | Medium | Backend Lead | 2026-06-10 | Demo Gate |

## Revision History

| Date | Author | Change |
|---|---|---|
| 2026-05-27 | Copilot | Initial contract governance template |
| 2026-05-27 | Copilot | Added baseline endpoint evidence and blocker details |

## Future Expansion

- 增加 OpenAPI/JSON schema 自動比對輸出欄位。
