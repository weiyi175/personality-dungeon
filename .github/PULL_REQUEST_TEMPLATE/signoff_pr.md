<!-- PR 模板：用於提交需簽核的變更（例如關閉 mock、API contract 鎖定、demo hardening） -->

## 概要 / Summary
- Phase: <!-- Phase 1..6 -->
- Goal: <!-- e.g., remove mock routes, lock api schema -->
- Owner: <!-- name -->

## 相關 Issue
- Closes: #<!-- issue number -->

## 變更清單 / Changes
- 列出主要變更點與檔案

## 驗收清單（請在完成後一一勾選）
- [ ] Contract tests green (attach logs)
- [ ] Integration smoke (5 runs) attached
- [ ] Mock routes removed / disabled (link PR or diff)
- [ ] Update `docs/progress/01_runtime_bridge/api_contract_status.md`
- [ ] Update `docs/progress/04_phase_review/signoff_checklist.md` with evidence links

## Signoff（Reviewer 勾選）
- Backend Lead:  [ ] Approve  /  [ ] Reject  — Comment: _____ — Date: ___
- Integration Lead: [ ] Approve  /  [ ] Reject  — Comment: _____ — Date: ___
- QA Lead: [ ] Approve  /  [ ] Reject  — Comment: _____ — Date: ___

## Evidence Links
- Contract tests: 
- Smoke logs:
- Video / Screenshots:

---
_自動化提示：在 PR CI 綠燈時，請在 Signoff 表格填入 reviewer 與日期，系統將視為一個階段之簽核紀錄。_
