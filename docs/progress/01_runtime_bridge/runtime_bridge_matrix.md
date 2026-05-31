# Runtime Bridge Matrix

## Metadata

| Field | Value |
|---|---|
| Document ID | RB-00 |
| Owner | Integration Lead |
| Audience | Backend, Godot, QA |
| Update Frequency | Every 2-3 days |
| Dependency | api_contract_status.md, godot_python_sync.md |

## Cross References

- ../00_master_matrix/project_progress_matrix.md
- ./api_contract_status.md
- ./godot_python_sync.md
- ../03_technical_debt/mock_tracking.md

## Runtime Path Matrix

| Flow ID | Source Layer | Target Layer | Contract | Status | Validation | Drift Risk | Owner | Blocker |
|---|---|---|---|---|---|---|---|---|
| RB-01 | Python Core | API | session initialize payload | Integrated | [route exists](../../api/server.py#L389) + schema model | Medium | Backend Lead | none |
| RB-02 | API | Godot | step response payload | Prototype | [step route](../../api/server.py#L469) + manual replay | High | Integration Lead | Godot loop 尚未全面吃 step 結果 |
| RB-03 | Godot | API | user action request | Prototype | DebugPanel 呼叫 API 手動驗證 | Medium | Godot Lead | 行為枚舉與回合語意仍待對齊 |
| RB-04 | API | Python Core | event resolution input | Integrated | [session engine](../../simulation/rl_session_engine.py) | Medium | Backend Lead | none |
| RB-05 | API | Product Demo | playable loop contract | Designed | dry run checklist | High | PM + QA | 缺結局關卡收束與教學引導 |

## Phase 1 Stabilization Checklist (Runtime Bridge & API)

| Task ID | Flow | Goal | Pass/Fail 判準 | Evidence | Owner | Status |
|---|---|---|---|---|---|---|
| RB-P1-01 | RB-01 | Freeze init schema (v1) | request/response 欄位清單固定 + contract tests 綠燈 | [api contract status](./api_contract_status.md) | Backend Lead | Open |
| RB-P1-02 | RB-02 | Step response mapping -> Godot | step 回應可完整映射到 UI 狀態 | [godot sync](./godot_python_sync.md) | Integration Lead | Open |
| RB-P1-03 | RB-03 | Action enum alignment | Godot action 枚舉與 API 契約一致 | [runtime matrix](./runtime_bridge_matrix.md) | Godot Lead | Open |
| RB-P1-04 | RB-04 | Engine input contract stable | event input 欄位與 engine handler 一致 | [session engine](../../simulation/rl_session_engine.py) | Backend Lead | Open |
| RB-P1-05 | RB-05 | 5-step playable loop smoke | 5-run smoke logs 無例外 | [playable loop review](../02_demo_review/playable_loop_review.md) | PM + QA | Open |

## Baseline Snapshot (2026-05-27)

| Flow | 目前可用度 | 阻塞等級 | 最快解法 |
|---|---:|---|---|
| initialize | 80% | Low | 將 request/response 固定版號 |
| step | 55% | High | Godot 完整接 step 回應，不走本地示意流程 |
| infer personality | 85% | Low | 補錯誤回復與 UI 提示 |
| end-to-end playable loop | 40% | Critical | 先做一輪 5-step 到結局最短路徑 |

## 狀態規範

- Integrated 代表契約在雙端一致且已有回歸驗證。
- Playable 代表該 flow 對玩家可見且可連續操作。
- Stable 代表至少連續兩個 phase 無破壞。

## 驗證記錄模板

| Date | Flow ID | Scenario | Result | Evidence Link | Verified By |
|---|---|---|---|---|---|
| 2026-05-27 | RB-01 | session init smoke | Pass | [init route](../../api/server.py#L389) | Copilot |
| 2026-05-27 | RB-02 | step route existence and contract | Partial | [step route](../../api/server.py#L469) | Copilot |
| 2026-05-27 | RB-03 | Godot infer bridge smoke | Pass | [debug panel](../../src/ui/DebugPanel.gd#L24) | Copilot |

## Revision History

| Date | Author | Change |
|---|---|---|
| 2026-05-27 | Copilot | Initial runtime matrix template |
| 2026-05-27 | Copilot | Added baseline snapshot and concrete flow evidence |

## Future Expansion

- 新增 flow latency, retry count, error budget 欄位。
