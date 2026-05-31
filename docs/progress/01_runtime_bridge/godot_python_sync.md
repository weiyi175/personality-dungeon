# Godot-Python Sync

## Metadata

| Field | Value |
|---|---|
| Document ID | RB-02 |
| Owner | Integration Engineer |
| Audience | Godot, Backend, QA |
| Update Frequency | Every 2-3 days |
| Dependency | api_contract_status.md |

## Cross References

- ./runtime_bridge_matrix.md
- ./api_contract_status.md
- ../03_technical_debt/technical_debt_matrix.md

## Field Mapping Matrix

| Domain Field | Python Source | API Payload Field | Godot Field | Sync Status | Drift Risk | Last Verified | Owner |
|---|---|---|---|---|---|---|---|
| Personality Vector | infer result | personality_vector | player_personality | Integrated | Medium | 2026-05-27 | Godot Lead |
| Session State | session engine snapshot | initial_snapshot.phase / snapshot.phase | runtime_state | In Progress | High | 2026-05-28 | Integration Lead |
| Event Result | session step output | snapshot.kind or derived result summary | current_event_result | Prototype | High | 2026-05-28 | Gameplay Lead |
| Risk Score | session engine snapshot | snapshot.risk_mean | player_risk | In Progress | Medium | 2026-05-28 | Integration Lead |
| Session Lifecycle | rl_sessions client | initial_snapshot / snapshot | RLSessionAPIClient | In Progress | High | 2026-05-28 | Integration Lead |

## Phase 1 Sync Targets

| Target ID | Field | Required Mapping | Pass/Fail 判準 | Evidence | Owner | Status |
|---|---|---|---|---|---|---|
| SY-P1-01 | Session State | `snapshot.phase` -> `runtime_state` | Godot 中可完整渲染 state | [runtime matrix](./runtime_bridge_matrix.md) | Integration Lead | In Progress |
| SY-P1-02 | Event Result | `event_outcome` -> `current_event_result` | UI 顯示一致且可追溯 | [playable loop review](../02_demo_review/playable_loop_review.md) | Gameplay Lead | In Progress |
| SY-P1-03 | Risk Score | `snapshot.risk_mean` -> `player_risk` | UI 顯示一致且無欄位缺失 | [player manager](../../src/core/PlayerManager.gd#L35) | Integration Lead | In Progress |
| SY-P1-04 | Session Bridge | `RLSessionAPIClient` -> `/rl_sessions/*` | initialize/step/snapshot 都可發出並解析 | [rl session client](../../src/core/RLSessionAPIClient.gd) | Integration Lead | In Progress |

## Phase 1 Verification Steps

1. 初始化 session 後，呼叫 step 回傳資料。
2. 檢查 `snapshot.phase`, `snapshot.risk_mean`, `snapshot.warm`, `snapshot.round` 是否在 UI 中可追蹤。
3. 若 `event_outcome` 尚未由 API 直出，先由 Godot 端暫以 step 結果摘要顯示，並標示為 prototype。
4. 保存 logs 與 UI 截圖，附在對應 issue/PR。

## Sync Validation Template

| Date | Scenario | Expected | Actual | Match | Evidence |
|---|---|---|---|---|---|
| 2026-05-27 | Infer -> personality update | vector applied to PlayerManager | signal emitted and chart updates | Yes | [player manager signal](../../src/core/PlayerManager.gd#L6) |
| 2026-05-28 | Init->Step x3 | stable state transitions | rl session client can initialize/step/snapshot; UI mapping still partial | Partial | [rl session client](../../src/core/RLSessionAPIClient.gd) |

## Drift Incident Log

| Incident ID | Symptom | Root Cause | Affected Layers | Severity | Fix Owner | Status |
|---|---|---|---|---|---|---|
| DR-001 | step payload 難以直接映射到前端回合展示 | 欄位語意尚未凍結 | API/Godot | High | Integration Lead | Open |
| DR-002 | 地牢事件回傳與 UI 文案層不一致 | 中間層轉換缺失 | Python/API/Godot | Medium | Gameplay Lead | Open |

## Baseline Evidence

- [DebugPanel API bridge](../../src/ui/DebugPanel.gd#L24)
- [PlayerManager vector apply](../../src/core/PlayerManager.gd#L35)
- [RLSessionAPIClient](../../src/core/RLSessionAPIClient.gd)
- [init endpoint](../../api/server.py#L389)
- [step endpoint](../../api/server.py#L439)

## Revision History

| Date | Author | Change |
|---|---|---|
| 2026-05-27 | Copilot | Initial sync mapping template |
| 2026-05-27 | Copilot | Added baseline sync status and drift incidents |

## Future Expansion

- 增加欄位自動對映檢查腳本輸出。
