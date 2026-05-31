# Phase Gate Checklist

## Metadata

| Field | Value |
|---|---|
| Document ID | PR-00 |
| Owner | PM |
| Audience | Leads, Review Board |
| Update Frequency | At each gate |
| Dependency | project matrix, demo gate, debt docs |

## Cross References

- ../00_master_matrix/project_progress_matrix.md
- ../02_demo_review/demo_gate.md
- ../03_technical_debt/technical_debt_matrix.md
- ./signoff_checklist.md

## Phase-by-Phase Acceptance Templates

以下為每個 Phase 的驗收模板（填寫後作為 Gate 判定依據）：

### Phase 1 — Runtime Bridge & API Stabilize

- Goal: 移除/隔離 Mock，鎖定 API schema，確保基本 session initialize/step 可用。
- Pass/Fail 判準:
	- API contract tests 全部通過
	- `/rl_sessions` initialize 與 step 可在真實 engine 上執行
	- 無公開/預設的 mock 路徑被啟用
- Required Evidence:
	1) Contract test run log
	2) PR（關閉 mock 的變更）
	3) Integration smoke logs (5 runs)
- Verification Steps:
	1) run contract tests
	2) run 5-step smoke on engine
	3) review PR diff for mock removal
- Signoff: Backend Lead, Integration Lead, QA Lead

### Phase 2 — Playable Loop Smoke

- Goal: 從 UI（Godot）發起最小可玩的回合迴圈（init -> 1 step -> show result）。
- Pass/Fail 判準:
	- 5 次連續 smoke run 無例外
	- UI 顯示與 API 回傳欄位一致
- Required Evidence:
	1) smoke logs + video
	2) UI screenshots
	3) ticket/PR 對應 mapping 修正
- Verification Steps: run smoke script, capture video, attach logs
- Signoff: Gameplay Lead, UX Lead, QA Lead

### Phase 3 — Instrumentation & Metrics

- Goal: 為 session 與 step 加入追蹤欄位（session_id, latency, event provenance），並導出到 metrics backend。
- Pass/Fail 判準:
	- 指標可在 dashboard 查到
	- 至少一個告警規則能被觸發測試
- Required Evidence: dashboard screenshot, metric export config, alert test run
- Signoff: SRE/Infra, QA

### Phase 4 — Demo Harden & UX Acceptance

- Goal: 擴展 playable loop 為 demo 流，補上教學與結局說明。
- Pass/Fail 判準: demo checklist 全部 Pass
- Required Evidence: demo video, checklist signed, UX notes
- Signoff: PM, UX, QA

### Phase 5 — Research Validation (adaptive/fate)

- Goal: 以可重現 protocol 驗證 `adaptive_counter` 與 `compute_death_threshold` 行為。
- Pass/Fail 判準: 實驗可重現、統計摘要達到預期收斂
- Required Evidence: experiment scripts, seeded runs, analysis notebook
- Signoff: Research Lead, Data Analyst

### Phase 6 — Regression Suite & Signoff Automation

- Goal: 把上面的驗收條件自動化成 CI gate（PR -> CI -> Gate check），並在合格後自動更新 `signoff_checklist.md`。
- Pass/Fail 判準: CI gate 可阻擋未通過 PR，自動產生簽核摘要
- Required Evidence: CI config, successful gate run, generated signoff artifact
- Signoff: CI Owner, PM

## Gate Criteria (summary)

| Criterion ID | 條件 | Required | Current | Evidence | Result |
|---|---|---|---|---|---|
| PG-01 | Runtime loop 可完成 | Yes | Partial | [runtime matrix](../01_runtime_bridge/runtime_bridge_matrix.md) | Pending |
| PG-02 | 無 mock blocking | Yes | No | [MockDungeon evidence](../../api/server.py#L53) | Blocked |
| PG-03 | Demo 可完整進行 | Yes | Partial | [demo checklist](../02_demo_review/demo_checklist.md) | Pending |
| PG-04 | API contract 固定 | Yes | Partial | [api contract status](../01_runtime_bridge/api_contract_status.md) | Pending |
| PG-05 | 回歸測試通過 | Yes | Partial | [tests backup note](../../SDD_12D_備份/tests) | Pending |

## Baseline Snapshot (2026-05-27)

| Gate Readiness | Score |
|---|---:|
| Runtime | 52 |
| Demo | 45 |
| Contract | 58 |
| Debt Control | 40 |

## Blocker Log

| Blocker ID | Description | Severity | Owner | ETA | Status |
|---|---|---|---|---|---|
| PGB-001 | step 流程尚未形成完整 playable loop | High | Integration Lead | 2026-06-14 | Open |
| PGB-002 | mock 路徑仍存在並影響 Gate 判定 | Critical | Backend Lead | 2026-06-10 | Open |
| PGB-003 | 結局與教學引導缺失，Demo 無法一輪完結 | High | Gameplay Lead | 2026-06-12 | Open |

## Revision History

| Date | Author | Change |
|---|---|---|
| 2026-05-27 | Copilot | Initial phase gate checklist template |
| 2026-05-27 | Copilot | Added baseline readiness and blocker details |

## Future Expansion

- 新增跨 gate 比較段落，記錄條件收斂速度。
