# Technical Debt Matrix

## Metadata

| Field | Value |
|---|---|
| Document ID | TD-00 |
| Owner | Tech Lead |
| Audience | Maintainers, Integration, PM |
| Update Frequency | Weekly |
| Dependency | runtime and phase review docs |

## Cross References

- ../01_runtime_bridge/runtime_bridge_matrix.md
- ./mock_tracking.md
- ./refactor_risk.md
- ../04_phase_review/phase_gate_checklist.md

## Debt 類型字典

| 類型 | 定義 |
|---|---|
| mock code | 用於暫時替代正式邏輯 |
| duplicated logic | 同一邏輯散落多處 |
| inconsistent schema | 跨層欄位語意不一致 |
| debug-only flow | 僅為 debug 可用，非產品流程 |
| temporary API | 過渡端點或未版控契約 |
| hardcoded values | 未參數化的固定值 |

## Debt Matrix

| Debt ID | Type | Description | 爆炸機率 | 影響範圍 | 重構成本 | 阻塞產品化 | Severity | Owner | ETA | Status |
|---|---|---|---|---|---|---|---|---|---|---|
| TD-001 | mock code | server 仍含 MockDungeon/MockPlayer 路徑 | High | Runtime + Demo | Medium | Yes | High | Backend Lead | 2026-06-10 | Open |
| TD-002 | inconsistent schema | API/Godot 部分欄位與步進語意未完全一致 | High | Runtime | High | Yes | Critical | Integration Lead | 2026-06-14 | Open |
| TD-003 | debug-only flow | 目前主要展示入口仍偏 DebugUI | Medium | Demo | Medium | Yes | High | Gameplay Lead | 2026-06-12 | Open |
| TD-004 | temporary API governance | session contract 版本控管未完整落地 | Medium | API + QA | Medium | Partial | Medium | Backend Lead | 2026-06-08 | Open |

## Baseline Snapshot (2026-05-27)

| 指標 | 值 |
|---|---|
| Open Debts | 4 |
| P0 (Critical) | 1 |
| Product-blocking Debts | 3 |
| 建議本週優先關閉 | TD-001, TD-002 |

## 優先級算法

Priority Score = Severity Weight + Blocker Weight + Blast Radius Weight

| 等級 | 條件 |
|---|---|
| P0 | Critical 或 阻塞產品化=Yes 且爆炸機率=High |
| P1 | High 且影響範圍跨兩層以上 |
| P2 | Medium 且可在單一模組消化 |
| P3 | Low 與僅影響內部開發效率 |

## Revision History

| Date | Author | Change |
|---|---|---|
| 2026-05-27 | Copilot | Initial technical debt governance template |
| 2026-05-27 | Copilot | Added baseline debt register with ETA and ownership |

## Future Expansion

- 新增 debt burndown 圖與每週關閉率。
