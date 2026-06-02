# Phase 06 SLA 與監控規格

**版本：** v1.0  
**生效日期：** 2026-06-01  
**Owner：** n1166

---

## 1. 回歸管線 SLA

| 管線 | 觸發時機 | 預期完成時間 | 最大允許失敗天數 | 逾期動作 |
|---|---|---:|---:|---|
| ci-smoke | PR 建立/更新 | ≤ 10 分鐘 | 0（單次失敗即阻擋合併）| Block PR merge |
| ci-regression | 每日 UTC 02:00 | ≤ 30 分鐘 | 連續 2 天 | 升級告警（見第 3 節）|

## 2. 夜間排程設定

`ci-regression.yml` 使用 GitHub Actions `schedule` trigger：

```yaml
on:
  schedule:
    - cron: "0 2 * * *"   # UTC 02:00 = 台灣時間 10:00
```

觸發條件：`main` branch 最新 commit。

## 3. 告警與通知

| 事件 | 通知方式 | 收件人/頻道 |
|---|---|---|
| ci-smoke 失敗 | GitHub PR check fail（原生阻擋）| PR author |
| ci-regression 失敗（第 1 天）| GitHub Actions email（內建）| n1166 |
| ci-regression 失敗（連續 2 天）| 升級：在 `docs/RL_Session/ALERT_RULES.md` 新增條目，並手動建立 GitHub Issue | n1166 + team |
| gate_pass=false（signoff_summary.json）| CI job exit code 1 → workflow failure | 同上 |

## 4. 升級流程

```
Day 1 失敗 → GitHub email 通知 → 人工 triage
Day 2 連續失敗 → 建立 GitHub Issue（label: regression-incident）
              → 在 ALERT_RULES.md 記錄事件
              → 本週內完成 root cause 分析
```

## 5. Artifact 保留政策

| Artifact | 保留天數 | 備註 |
|---|---:|---|
| ci-smoke-{run_id} | 7 天 | GitHub Actions 預設 |
| ci-regression-{run_id} | 30 天 | 需在 workflow 設定 `retention-days: 30` |
| artifacts/signoff_summary.json | 永久（版本控制）| 每次 regression 成功後更新 |

## 6. 修訂歷史

| 版本 | 日期 | 說明 |
|---|---|---|
| v1.0 | 2026-06-01 | 初版，Phase 06 CI 啟動 |
