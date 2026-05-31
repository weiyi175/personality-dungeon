# Demo Gate

## Metadata

| Field | Value |
|---|---|
| Document ID | DM-01 |
| Owner | PM |
| Audience | Reviewer, Product Owner, Leads |
| Update Frequency | Before each phase review |
| Dependency | demo_checklist.md |

## Cross References

- ./demo_checklist.md
- ../04_phase_review/phase_gate_checklist.md
- ../04_phase_review/signoff_checklist.md

## Gate 定義

| Gate ID | Gate 名稱 | 必要條件 | 通過標準 | 當前狀態 |
|---|---|---|---|---|
| DG-1 | Playable Entry | 完成玩家輸入到回合啟動 | End-to-end 可走通 | Partial |
| DG-2 | Runtime Consistency | API/Godot payload 對齊 | 無 blocker drift | Open |
| DG-3 | Demo Clarity | 玩家可解釋核心機制 | 問答通過率 >= 80% | Open |
| DG-4 | Recovery | 錯誤可提示與重試 | 中斷可恢復 | Open |

## Gate 評分卡

| Gate ID | Research | Product | Integration | Demo Ready | Result | Reviewer |
|---|---:|---:|---:|---|---|---|
| DG-1 | 80 | 50 | 45 | Partial | Pending | PM |
| DG-2 | 75 | 40 | 50 | Partial | Blocked | Integration Lead |
| DG-3 | 60 | 45 | 35 | No | Pending | UX Lead |
| DG-4 | 55 | 35 | 40 | No | Pending | QA Lead |

## Baseline Gate Notes (2026-05-27)

- DG-2 目前被 mock 路徑與 payload drift 阻塞。
- DG-3 主要缺口是「玩家能否理解結果原因」而非模型效果。
- DG-4 目前只有局部 UI 錯誤提示，尚未形成完整 recover 流程。

## Gate Signoff

| Role | Name | Decision | Date | Notes |
|---|---|---|---|---|
| PM | TBD | Approve/Reject | YYYY-MM-DD | |
| Tech Lead | TBD | Approve/Reject | YYYY-MM-DD | |
| Product Owner | TBD | Approve/Reject | YYYY-MM-DD | |

## Revision History

| Date | Author | Change |
|---|---|---|
| 2026-05-27 | Copilot | Initial demo gate template |
| 2026-05-27 | Copilot | Added baseline gate scores and blockers |

## Future Expansion

- 新增 gate trend，追蹤每次 review 的改善速度。
