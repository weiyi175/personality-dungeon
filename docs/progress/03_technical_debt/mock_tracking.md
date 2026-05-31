# Mock Tracking

## Metadata

| Field | Value |
|---|---|
| Document ID | TD-01 |
| Owner | Integration Lead |
| Audience | Backend, Gameplay, QA |
| Update Frequency | Weekly |
| Dependency | technical_debt_matrix.md |

## Cross References

- ./technical_debt_matrix.md
- ../01_runtime_bridge/runtime_bridge_matrix.md
- ../04_phase_review/signoff_checklist.md

## Mock Inventory

| Mock ID | Location | Purpose | Temporary Since | Removal Condition | Product Blocking | Owner | Status |
|---|---|---|---|---|---|---|---|
| MK-001 | [api/server.py](../../api/server.py) | dev smoke | 2026-05-27 (tracked) | real session path fully verified | Yes | Backend Lead | Closed |
| MK-002 | [src/core/DungeonSim.gd](../../src/core/DungeonSim.gd) | UI placeholder | 2026-05-27 (tracked) | API step loop integrated | Yes | Gameplay Lead | Open |

## Mock Removal Plan

| Plan ID | Mock ID | Step | Required Test | Target Date | Result |
|---|---|---|---|---|---|
| MR-001 | MK-001 | replace mock init | init e2e | 2026-06-10 | Done (2026-05-28) |
| MR-002 | MK-002 | bind step response | playable loop test | 2026-06-12 | Pending |
| MR-003 | MK-001 + MK-002 | run gate smoke (5-step) | gate dry run | 2026-06-14 | Pending |

## Phase 1 Gate Checklist (Mock Removal)

| Checklist ID | Requirement | Pass/Fail 判準 | Evidence | Owner | Status |
|---|---|---|---|---|---|
| MK-P1-01 | MK-001 removed | mock init path 不再可用 | [api/server.py](../../api/server.py) | Backend Lead | Done |
| MK-P1-02 | MK-002 removed | Godot 不再依賴本地 DungeonSim | [DungeonSim.gd](../../src/core/DungeonSim.gd) | Gameplay Lead | Open |
| MK-P1-03 | Gate smoke completed | 5-step smoke logs 無例外 | [playable loop review](../02_demo_review/playable_loop_review.md) | Integration Lead | Open |

## Baseline 結論

- 目前至少兩個 mock 直接阻塞產品級 gate。
- 建議先移除 MK-001，再移除 MK-002，最後再做 demo gate dry run。

## Revision History

| Date | Author | Change |
|---|---|---|
| 2026-05-27 | Copilot | Initial mock tracking template |
| 2026-05-27 | Copilot | Added concrete mock inventory and removal timeline |

## Future Expansion

- 增加 mock aging 指標，避免長期停留。
