# Phase 05 Validation Report (Signoff Version v1.0)

## Signoff Metadata

| Field | Value |
|---|---|
| Phase | Phase 5 — Research Validation |
| Report Version | v1.0 |
| Report Date | 2026-06-01 |
| Prepared By | Copilot (execution support) |
| Signoff State | Ready for Review |
| Scope Note | 本版覆蓋 smoke 規模（matrix 3 rows） |

## 1. 實驗目標與假設

- 目標：驗證 Phase 05 seeded experiment 流程具備可重現性與可簽核性。
- 假設 H1：同參數重跑（排除 generated_at）後，run payload 與 summary 應完全一致。
- 假設 H2：matrix smoke 的輸出數量應與 matrix 列數一致，且可生成固定欄位 summary CSV。

## 2. Protocol

| Item | Value |
|---|---|
| Date | 2026-06-01 |
| Python | /home/user/personality-dungeon/venv/bin/python |
| Single-seed smoke | seed=42, runs=10, n_steps=200 |
| Matrix smoke | scripts/experiments/matrix.csv（3 rows） |
| Matrix output dir | reports/experiments/phase05_matrix_smoke |
| Summary CSV | analysis/experiments_summary.csv |

## 3. 結果摘要

### 3.1 Reproducibility smoke（seed=42, N=10）

- 檔案：reports/experiments/repro_a.json, reports/experiments/repro_b.json
- 比對方式：移除 generated_at 後做 payload 等值比較。
- 結果：REPRO_PASS。

summary（A/B 一致）：

- reward_mean = 0.345291
- reward_std = 0.278632
- avg_latency_ms = 1.4887

### 3.2 Matrix smoke（3 rows）

- matrix 列數：3
- run JSON 數量：3
- summary CSV 列數：3
- 一致性：count_match=True

summary CSV（analysis/experiments_summary.csv）：

| file | seed | n_runs | n_steps | reward_mean | reward_std | avg_latency_ms |
|---|---:|---:|---:|---:|---:|---:|
| run_seed42_w0p08_burn50_rounds200.json | 42 | 1 | 200 | -0.164985 | 0.0 | 2.0986 |
| run_seed43_w0p10_burn50_rounds200.json | 43 | 1 | 200 | 0.774714 | 0.0 | 0.5964 |
| run_seed44_w0p12_burn50_rounds200.json | 44 | 1 | 200 | 0.558761 | 0.0 | 1.5213 |

## 4. Gate 檢核結果

| Gate | 定義 | Result |
|---|---|---|
| V5-01 | 可重現性（排除 generated_at） | PASS |
| V5-02 | 矩陣完整性（run 數 = matrix 列數） | PASS |
| V5-03 | 匯總一致性（summary 列數一致） | PASS |
| V5-04 | 研究可解釋性（有摘要與異常註記） | PASS |

## 5. 失敗案例與異常說明

- 初次 matrix 匯總時，reports/experiments 內已有既有 run_*.json，導致 run 計數大於 matrix 列數。
- 修正後改為隔離輸出目錄 reports/experiments/phase05_matrix_smoke，再次執行後一致性恢復正常。

## 6. 結論與下一步

- 結論：Phase 05 smoke 規模已達可簽核狀態。
- 限制：目前僅 smoke（3 rows），尚未覆蓋 full matrix（12-30 rows）。
- 建議下一步：
  1. 依 full matrix 規模重跑並維持隔離輸出目錄。
  2. 將 summary 產生與 count_match 驗證封裝成單一腳本，降低人工步驟。
  3. 將本流程接入 Phase 06 CI gate。

## 7. Signoff Decision Block

| Item | Value |
|---|---|
| Submission Date | 2026-06-01 |
| Gate Result | V5-01~V5-04 全部 PASS |
| Decision Proposal | Recommend Approve (Smoke Scope) |
| Approval Status | Completed ✅ |
| Effective Date | 2026-06-01 |

## 8. Approval Table

| Role | Name | Decision | Date | Notes |
|---|---|---|---|---|
| Research Lead | n1166 | Completed ✅ | 2026-06-01 | 假設與指標解釋確認 |
| Data Analyst | n1166 | Completed ✅ | 2026-06-01 | 資料一致性與欄位契約確認 |
| PM / Product | n1166 | Completed ✅ | 2026-06-01 | 同意推進 full matrix |
| Engineering Lead | n1166 | Completed ✅ | 2026-06-01 | 同意後續 CI 整合方向 |

## 9. 證據索引

- reports/experiments/run_seed42_r10.json
- reports/experiments/repro_a.json
- reports/experiments/repro_b.json
- reports/experiments/phase05_matrix_smoke/matrix_run_log.txt
- analysis/experiments_summary.csv
- scripts/experiments/matrix_phase05_full.csv
- reports/experiments/phase05_matrix_full/matrix_run_log.txt
- analysis/experiments_summary_full.csv
