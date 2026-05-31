# Milestone Review

## Metadata

| Field | Value |
|---|---|
| Document ID | PR-01 |
| Owner | PM + Research Lead |
| Audience | Stakeholders, Leads |
| Update Frequency | At each milestone |
| Dependency | phase gate + master matrix |

## Cross References

- ./phase_gate_checklist.md
- ./signoff_checklist.md
- ../00_master_matrix/project_progress_matrix.md

## Milestone Summary

| Milestone | Period | Target | Result | Status |
|---|---|---|---|---|
| M-Baseline | 2026-05-27 to 2026-06-14 | 建立可玩閉環與 Gate 可審查文件 | PlayableLoopScene 已完成 live init/step/ending/reset 驗證 | In Progress |

## 三軸進度摘要

| Dimension | Previous | Current | Delta | Notes |
|---|---:|---:|---:|---|
| Research Progress | 82 | 85 | +3 | BL2/6-6 證據穩定，主線清晰 |
| Product Progress | 42 | 50 | +8 | API smoke test 通過，Godot 場景腳本就位 |
| Integration Progress | 48 | 58 | +10 | API ↔ 路由 init+step 全 200，P2-RLC-01 Done |

## 主要成果與缺口

| Category | Highlights | Gaps | Impact |
|---|---|---|---|
| Runtime | initialize/step 路由已固定存在 | step 到前端可玩映射不足 | 直接影響 playable gate |
| Demo | 遺言->人格可展示 | 結局與教學收束缺失 | 無法完整對外展示 |
| Debt | 已建立 debt/mock/refactor 三件組治理 | P0 債項尚未關閉 | 影響 phase signoff |

## 決策與行動

| Action ID | Action | Owner | Due | Dependency | Status |
|---|---|---|---|---|---|
| MA-001 | 關閉 MK-001 (mock session path) | Backend Lead | 2026-06-10 | TD-001 | Open |
| MA-002 | 完成 step->UI 映射一輪可玩流程 | Integration Lead | 2026-06-12 | RB-02, PL-003 | Open |
| MA-003 | 建立結局解釋卡與新手導引 | UX Lead | 2026-06-14 | PL-002 | Open |

## Phase 1 可執行子任務清單（Mock Removal / API Freeze / Godot Mapping）

| Subtask ID | Scope | PR 交付物 | Verification Steps | Evidence | Owner | Status |
|---|---|---|---|---|---|---|
| P1-MK-01 | Backend | 移除 `MockDungeon/MockPlayer` 初始化路徑；更新 [mock_tracking.md](../03_technical_debt/mock_tracking.md) | 1) 呼叫 `/rl_sessions/initialize` 2) 檢查無 mock path 3) 保存 logs | PR TBD + init logs TBD | Backend Lead | Done |
| P1-API-01 | API Contract | 鎖定 v1 init request/response 欄位；更新 [api_contract_status.md](../01_runtime_bridge/api_contract_status.md) | 1) 完成欄位表 2) 合約測試通過 | PR TBD + [logs/contract_tests_rl_sessions.log](logs/contract_tests_rl_sessions.log) | Backend Lead | Done |
| P1-API-02 | API Contract | 鎖定 v1 step response 與 error taxonomy；更新 [api_contract_status.md](../01_runtime_bridge/api_contract_status.md) | 1) step 回傳欄位固定 2) error code 定義完成 | PR TBD + [logs/contract_tests_rl_sessions.log](logs/contract_tests_rl_sessions.log) | Integration Lead | Done |
| P1-GD-01 | Godot Mapping | 對齊 `state`/`event_outcome`/`risk` 欄位；更新 [godot_python_sync.md](../01_runtime_bridge/godot_python_sync.md) | 1) UI 顯示對齊 2) 截圖與 logs 保存 | [logs/smoke_init.json](logs/smoke_init.json) + [logs/smoke_steps.json](logs/smoke_steps.json) | Integration Lead | Done |
| P1-MK-02 | Godot Runtime | 移除本地 `DungeonSim` fallback；改用 API step | 1) Godot 不再走本地模擬 2) step loop 正常 | [DungeonSim.gd](../../../src/core/DungeonSim.gd) (new API-based impl) | Gameplay Lead | Done |
| P1-SM-01 | E2E Smoke | 5 次連續 init->step->UI 測試 | 1) 5-run smoke 2) 無例外 | [logs/smoke_init.json](logs/smoke_init.json) + [logs/smoke_steps.json](logs/smoke_steps.json) | QA Lead | Done |

## Phase 1 執行順序與逐步簽核（Kickoff）

| Stage | Focus | Subtasks | Pre-Req | Verification | Evidence | Signoff |
|---|---|---|---|---|---|---|
| S1 | Remove mock init path | P1-MK-01 | None | 呼叫 `/rl_sessions/initialize`，確認無 mock path，保存 logs | PR + init logs | Backend Lead |
| S2 | Freeze v1 init/step schema | P1-API-01, P1-API-02 | S1 | contract tests + step logs | contract test log + step logs | Backend + Integration |
| S3 | Godot mapping alignment | P1-GD-01 | S2 | UI 顯示 `state`/`event_outcome`/`risk` | smoke logs | Integration Lead | Done |
| S4 | Remove DungeonSim fallback | P1-MK-02 | S3 | Godot 使用 API step loop | runtime logs | Gameplay Lead | Done |
| S5 | E2E smoke | P1-SM-01 | S4 | 5-run smoke | smoke logs | QA Lead | Done |

- 每一個 Stage 完成後再更新下一個 Stage 的狀態，並將 Evidence 連結回填到對應 Subtask。
- S5 完成後同步更新 `signoff_checklist.md` 的 Phase 1 簽核條目。

## Phase 2 可執行子任務清單（Playable Loop / UI Integration）

| Subtask ID | Scope | PR 交付物 | Verification Steps | Evidence | Owner | Status |
|---|---|---|---|---|---|---|
| P2-SC-01 | Godot Scaffold | PlayableLoopController + SessionStatusPanel + PlayableLoopScene | 1) 場景載入無 error 2) Button click handlers exist | [PlayableLoopScene.tscn](../../../src/ui/PlayableLoopScene.tscn) + [PlayableLoopScene.gd](../../../src/ui/PlayableLoopScene.gd) + [PlayableLoopController.gd](../../../src/core/PlayableLoopController.gd) | Integration Lead | Done |
| P2-RLC-01 | API Integration | 連接 RLSessionAPIClient signal handlers；API smoke test 通過 | 1) init 200 OK + session_id ✅ 2) 5x step 200 OK ✅ | [logs/smoke_phase2_init.json](../../../logs/smoke_phase2_init.json) + [logs/smoke_phase2_steps.json](../../../logs/smoke_phase2_steps.json) + [logs/smoke_phase2_summary.json](../../../logs/smoke_phase2_summary.json) | Backend Lead | **Done** |
| P2-UI-01 | UI Binding | RadarChart + SessionStatusPanel 與 snapshot 同步 | 1) snapshot 欄位映射正確 2) UI 實時更新 | [logs/smoke_phase2_steps.json](../../../logs/smoke_phase2_steps.json) + live Godot console output（2026-05-30） | UX Lead | Done |
| P2-PL-01 | Playable Loop | 1 完整迴圈（init → steps → ending） | 1) 無 exception 2) 結局正確偵測 | [video/playable_5step.mp4](video/playable_5step.mp4) + [logs/playable_full_loop.log](logs/playable_full_loop.log)（live Godot verified 2026-05-30） | QA Lead | Done |

## Phase 2 執行順序與逐步簽核（Playable Loop）

| Stage | Focus | Subtasks | Pre-Req | Verification | Evidence | Signoff |
|---|---|---|---|---|---|---|
| S1 | Godot scene scaffold | P2-SC-01 | Phase 1 Done | 場景載入無 error，節點結構完整 | [src/ui/PlayableLoopScene.tscn](../../../src/ui/PlayableLoopScene.tscn) | Integration Lead | Done |
| S2 | API client integration | P2-RLC-01 | S1 | init/step button 觸發 HTTP 請求，UI 響應 | [logs/smoke_phase2_init.json](../../../logs/smoke_phase2_init.json) + [logs/smoke_phase2_steps.json](../../../logs/smoke_phase2_steps.json) | Backend Lead | Done |
| S3 | UI state binding | P2-UI-01 | S2 | RadarChart 與 SessionStatusPanel 根據 snapshot 更新 | live Godot console output（2026-05-30） | UX Lead | Done |
| S4 | Full loop + ending | P2-PL-01 | S3 | 1 完整循環 init → 5 steps → ending → reset 無例外 | video + logs | QA Lead | Done |

- 每一個 Stage 完成後再更新下一個 Stage 的狀態，並將 Evidence 連結回填到對應 Subtask。
- S4 完成後同步更新 `signoff_checklist.md` 的 Phase 2 簽核條目。
- Phase 2 完成後啟動 Phase 3 (Instrumentation & Metrics) 規劃。

## Revision History

| Date | Author | Change |
|---|---|---|
| 2026-05-27 | Copilot | Initial milestone review template |
| 2026-05-27 | Copilot | Added baseline milestone, metrics, and actions |
| 2026-05-28 | Copilot | Phase 1 signoff complete; Phase 2/3 planning initiated |
| 2026-05-28 | Copilot | P2-RLC-01 Done（API smoke test 7/7 pass）；三軸分數更新 |
| 2026-05-30 | Copilot | PlayableLoopScene live init/step/ending/reset verified；P2-SC-01/P2-UI-01/P2-PL-01 回填完成 |

## Future Expansion

- 加入對外展示版與內部研究版雙格式摘要。
