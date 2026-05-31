# Playable Loop Review

## Metadata

| Field | Value |
|---|---|
| Document ID | DM-02 |
| Owner | Gameplay/UX Lead |
| Audience | UX, QA, PM |
| Update Frequency | Weekly |
| Dependency | demo_checklist.md |

## Cross References

- ./demo_checklist.md
- ../01_runtime_bridge/runtime_bridge_matrix.md
- ../04_phase_review/phase_gate_checklist.md

## Loop Stage Review

| Stage | 玩家行為 | 系統回應 | 可理解性 | 可持續性 | 阻塞風險 | Status | Evidence |
|---|---|---|---|---|---|---|---|
| 1. 輸入遺言 | 輸入文本 | 人格推斷結果回顯 | High | High | Low | Integrated | [debug panel](../../src/ui/DebugPanel.gd#L24) |
| 2. 進入地牢 | 點擊開始 | RLSessionAPIClient.initialize_session -> session initialize | Medium | Medium | Medium | **Integrated** | [smoke_phase2_init.json](../../logs/smoke_phase2_init.json)（API 200 OK ✅）；live Godot run 2026-05-30 |
| 3. 回合決策 | 選擇 action | RLSessionAPIClient.step_session + 風險更新 | Medium | Medium | High | **Integrated** | [smoke_phase2_steps.json](../../logs/smoke_phase2_steps.json)（5x step 200 OK ✅）；live Godot run 2026-05-30 |
| 4. 狀態演化 | 連續回合 | 人格/風險變化可視化 | Low | Medium | High | **Integrated** | [rl session client](../../src/core/RLSessionAPIClient.gd)；live snapshot updates verified |
| 5. 結局收束 | 到達終點或崩潰 | 結局說明畫面 | Low | Low | High | **Integrated** | live Godot ending/reset verified 2026-05-30 |

## 問題清單

| Issue ID | Stage | 問題描述 | 嚴重度 | Owner | ETA | Resolution |
|---|---|---|---|---|---|---|
| PL-001 | 3 | 回合結果文案可解釋性不足 | Medium | UX Lead | 2026-06-10 | Open |
| PL-002 | 5 | 結局收束未形成可展示流程 | High | Gameplay Lead | 2026-06-14 | Open |
| PL-003 | 3-4 | step 結果到 UI 的映射仍偏 debug 流 | High | Integration Lead | 2026-06-12 | Open |

## Verification Steps & Sample Commands

使用以下步驟驗證 playable loop 的基本可用性：

1. 初始化（呼叫 API，或從 Godot Client 觸發初始化）

```bash
# 範例：以 curl 初始化 session（本機開發用）
curl -s -X POST "http://localhost:8000/rl_sessions/initialize" -H "Content-Type: application/json" -d '{"player_id": "test", "seed": 42}' | jq '.' > logs/init.json
```

2. 執行一個 step，檢查回傳格式

```bash
curl -s -X POST "http://localhost:8000/rl_sessions/<session_id>/step" -H "Content-Type: application/json" -d '{"action": "explore"}' | jq '.' > logs/step.json
```

3. 執行 5 次連續 step 作為 smoke 測試，並保存 logs 與 UI 錄影

```bash
./venv/bin/python -m scripts/run_smoke.py --session-id <session_id> --steps 5 --out logs/playable_smoke.log
```

4. 驗證 UI 映射：確認 `RLSessionAPIClient.gd` 拿到的 `snapshot.phase`、`snapshot.risk_mean`、`snapshot.round` 可被 UI 或 debug 層正確摘要。
5. 若 `event_outcome` 尚未直出，先以 prototype 摘要欄位表示，並在 PR/issue 中標示為暫存映射。

## Baseline 判讀

- 目前可進行「完整可互動 demo」，init / step / snapshot update / ending / reset 已 live 驗證通過。
- 先前阻塞點已解除，Stage 5 由 Designed 升為 Integrated；剩餘工作改為展示文案與體驗優化。

## Signoff Snippet

| Role | Pass? | Notes | Date |
|---|---|---|---|
| UX Reviewer | Yes/No | | YYYY-MM-DD |
| QA Reviewer | Yes/No | | YYYY-MM-DD |

## Revision History

| Date | Author | Change |
|---|---|---|
| 2026-05-27 | Copilot | Initial playable loop review template |
| 2026-05-27 | Copilot | Added baseline stage evidence and issue backlog |
| 2026-05-28 | Copilot | Stage 2/3 升為 Integrated（API smoke test 通過）；Stage 4/5 仍待 Godot 手動驗證 |
| 2026-05-30 | Copilot | Live Godot init/step/ending/reset 驗證完成；Stage 2/3/4/5 回填為 Integrated |

## Future Expansion

- 新增新手旅程與進階玩家旅程雙視角審查。
