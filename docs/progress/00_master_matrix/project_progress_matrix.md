# Project Progress Matrix

## Metadata

| Field | Value |
|---|---|
| Document ID | PGM-00 |
| Owner | Project Lead |
| Maintainers | PM, Tech Lead, Research Lead |
| Audience | 全專案核心成員 |
| Update Frequency | Weekly |
| Source of Truth | SDD.md, 研發日誌.md, runtime/debt/demo docs |

## Cross References

- ../README.md
- ./system_layer_overview.md
- ../01_runtime_bridge/runtime_bridge_matrix.md
- ../02_demo_review/demo_gate.md
- ../03_technical_debt/technical_debt_matrix.md
- ../04_phase_review/phase_gate_checklist.md

## 評分標準

| 維度 | 權重 | 評分說明 |
|---|---:|---|
| Research Progress | 0.40 | 理論與實驗可重現度 |
| Product Progress | 0.35 | 玩家可用程度 |
| Integration Progress | 0.25 | 跨層串接與契約一致性 |

總分公式：

Score = 0.40 * Research + 0.35 * Product + 0.25 * Integration

## Master Matrix

| System Layer | Module | Status | Research Progress | Product Progress | Integration Progress | Demo Ready | Technical Debt Risk | Validation Evidence | Blockers | Owner | Next Gate |
|---|---|---|---:|---:|---:|---|---|---|---|---|---|
| 玩家系統 | 遺言輸入與狀態管理 | Prototype | 80 | 55 | 50 | Partial | Medium | [DebugPanel UI](../../src/ui/DebugPanel.gd#L24), [PlayerManager signal](../../src/core/PlayerManager.gd#L6) | 結局與死亡循環未接入完整玩家路徑 | Gameplay Engineer | Phase-Playable |
| Personality Engine | 9D 推斷與動態更新 | Integrated | 85 | 65 | 60 | Yes | Medium | [infer endpoint](../../api/server.py#L782), [set_personality_from_api](../../src/core/PlayerManager.gd#L35) | 玩家端可解釋性與教學文案不足 | Backend + UX | Phase-Bridge |
| 地牢系統 | 事件生成與風險演算 | Prototype | 80 | 45 | 50 | Partial | High | [event loader](../../dungeon/event_loader.py), [dungeon ai counter](../../dungeon/dungeon_ai.py#L17) | Godot 地牢流程尚未完全對接 API step | Gameplay + Integration | Phase-Playable |
| Runtime Bridge | Python-API-Godot 串接 | Prototype | 75 | 40 | 45 | Partial | High | [initialize route](../../api/server.py#L389), [step route](../../api/server.py#L469), [MockDungeon](../../api/server.py#L53) | mock 路徑仍存在且會影響產品驗收 | Integration Lead | Phase-Integration |
| Demo 系統 | Demo flow 與教學可理解性 | Designed | 60 | 35 | 30 | No | Medium | [Debug UI scene](../../src/ui/DebugUI.tscn#L38), [playable loop review](../02_demo_review/playable_loop_review.md) | 缺完整「輸入到結局」一輪可展示流程 | PM + UX | Phase-Demo |

## Baseline Snapshot (2026-05-27)

| 指標 | 估計值 | 判讀依據 |
|---|---:|---|
| Research Progress (global) | 85 | BL2/6/6 與 Runtime bridge 研究節點已形成穩定主線 |
| Product Progress (global) | 45 | UI 可互動但可玩閉環未完成 |
| Integration Progress (global) | 52 | API 路由存在且可串接，仍有 mock 依賴 |
| Demo Ready (global) | Partial | 可展示局部功能，仍非完整玩家旅程 |
| Technical Debt Risk (global) | High | mock 與 schema drift 對產品化有阻塞 |

加權總分：64.45

## 狀態解讀規則

| 規則 | 說明 |
|---|---|
| Status 不能超前於證據 | 無驗證證據不得標記 Integrated 以上 |
| Playable 至少需要三條成立 | 可完成 loop、玩家可理解、錯誤可恢復 |
| Stable 必須具回歸測試 | 需有固定 seed 與 gate pass 記錄 |

## 驗證證據索引

| Evidence ID | Type | Link | Last Verified | Verified By |
|---|---|---|---|---|
| EV-PGM-001 | API Trace | [runtime bridge matrix](../01_runtime_bridge/runtime_bridge_matrix.md) | 2026-05-27 | Copilot |
| EV-PGM-002 | Demo Dry Run | [demo checklist](../02_demo_review/demo_checklist.md) | 2026-05-27 | Copilot |
| EV-PGM-003 | Debt Review | [technical debt matrix](../03_technical_debt/technical_debt_matrix.md) | 2026-05-27 | Copilot |
| EV-PGM-004 | Route Evidence | [api server routes](../../api/server.py#L389) | 2026-05-27 | Copilot |
| EV-PGM-005 | Godot Bridge Evidence | [debug panel client load](../../src/ui/DebugPanel.gd#L24) | 2026-05-27 | Copilot |

## Revision History

| Date | Author | Change |
|---|---|---|
| 2026-05-27 | Copilot | Initial template and baseline rows |
| 2026-05-27 | Copilot | Added baseline snapshot, evidence links, and owner roles |

## Future Expansion

- 加入自動匯入 CI 測試與 smoke run 結果。
- 增加跨週趨勢圖欄位（progress delta）。
