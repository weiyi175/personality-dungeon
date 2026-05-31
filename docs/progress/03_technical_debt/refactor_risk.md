# Refactor Risk

## Metadata

| Field | Value |
|---|---|
| Document ID | TD-02 |
| Owner | Tech Lead |
| Audience | Architecture, Maintainers |
| Update Frequency | Weekly |
| Dependency | technical_debt_matrix.md |

## Cross References

- ./technical_debt_matrix.md
- ../00_master_matrix/module_dependency_graph.md
- ../04_phase_review/phase_gate_checklist.md

## Refactor Risk Matrix

| Refactor ID | Scope | Complexity | Regression Risk | Rollback Plan | Test Coverage Ready | Owner | Status |
|---|---|---|---|---|---|---|---|
| RF-001 | Runtime session bridge | High | High | keep legacy endpoint for 1 phase | Partial | Integration Lead | Planned |
| RF-002 | Godot state sync mapping | Medium | High | dual parser fallback | Partial | Gameplay Lead | Planned |
| RF-003 | Demo flow from debug to player loop | Medium | Medium | keep debug scene as fallback | Partial | UX Lead | Planned |

## Baseline Notes (2026-05-27)

- RF-001 是最高優先重構，因為直接影響 Gate 準入。
- RF-002 與 RF-001 耦合，建議同 sprint 處理。
- RF-003 可在 RF-001/002 穩定後進行，避免重工。

## 風險分級

| Risk Level | 判定 |
|---|---|
| Critical | 影響核心 loop 且無回滾方案 |
| High | 影響跨層契約，回滾成本高 |
| Medium | 單層改動，具替代路徑 |
| Low | 局部優化，可快速修正 |

## Revision History

| Date | Author | Change |
|---|---|---|
| 2026-05-27 | Copilot | Initial refactor risk template |
| 2026-05-27 | Copilot | Added baseline risk priorities and owners |

## Future Expansion

- 納入 refactor 後性能差異與事故率追蹤。
