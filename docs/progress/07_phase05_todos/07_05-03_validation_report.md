# 07_05-03 — 產出研究驗證報告

## Metadata

| Field | Value |
|---|---|
| Owner | n1166 |
| Target date | 2026-06-06 |
| Scope | 將可重現實驗結果轉為可簽核研究結論 |
| Dependency | 07_05-01、07_05-02 產物 |
| **封卷狀態** | **Closed ✅ (2026-06-01)** |

## 報告輸出規格

- 主報告：reports/experiments/validation_report.md（建議）或 validation_report.pdf。
- 附件：analysis/experiments_summary.csv。
- 附錄：reports/experiments/phase05_matrix_smoke/matrix_run_log.txt、代表性 run JSON。

## 報告章節模板

1. 實驗目標與假設。
2. Protocol（seed、矩陣欄位、執行環境、版本）。
3. 結果摘要（reward_mean、reward_std、latency）。
4. 可重現檢查（同 seed 重跑比對）。
5. 失敗案例與異常說明。
6. 結論與下一步決策（繼續擴矩陣或回退修正）。

## 驗證門檻（Pass/Fail）

| Gate ID | 條件 | Pass 定義 | Result |
|---|---|---|---|
| V5-01 | 可重現性 | 同參數重跑結果一致（排除 generated_at） | PASS (2026-06-01) |
| V5-02 | 矩陣完整性 | 所有矩陣列均成功產出 JSON | PASS (2026-06-01) |
| V5-03 | 匯總一致性 | summary CSV 列數與 run JSON 數量一致 | PASS (2026-06-01) |
| V5-04 | 研究可解釋性 | 主要指標有文字解釋與異常註記 | PASS (2026-06-01) |

## 證據索引

| Evidence ID | 路徑 | 說明 |
|---|---|---|
| E5-01 | reports/experiments/run_seed42_r10.json | 單 seed 基準 run |
| E5-02 | reports/experiments/phase05_matrix_smoke/matrix_run_log.txt | Smoke 矩陣執行 log（3 runs） |
| E5-03 | analysis/experiments_summary.csv | Smoke 彙總表 |
| E5-04 | reports/experiments/validation_report.md | 最終報告（Signoff v1.0） |
| E5-05 | scripts/experiments/matrix_phase05_full.csv | Full matrix 定義（12 rows） |
| E5-06 | reports/experiments/phase05_matrix_full/matrix_run_log.txt | Full 矩陣執行 log（12 runs） |
| E5-07 | analysis/experiments_summary_full.csv | Full 彙總表 |

## 簽核矩陣（Research / Product / Integration）

| Dimension | Reviewer | 檢查點 | Signoff |
|---|---|---|---|
| Research | Research Lead | 假設、指標、解釋是否成立 | Ready for Signoff |
| Product | PM / Gameplay | 成果是否支持下一階段需求 | Ready for Signoff |
| Integration | Engineering Lead | 腳本與產物可納入 CI/夜跑 | Ready for Signoff |

## Gate 判定

- 判定：Phase 05（07_05-01 ~ 07_05-03）已達到可簽核狀態。
- 限定範圍：本次為 smoke 規模（matrix 3 rows）；full 規模需在後續批次另行補件。

## 最終封卷摘要（Smoke + Full 合併結論）

**封卷日期：2026-06-01**

### 執行總覽

| 批次 | 矩陣定義 | rows | 輸出目錄 | count_match |
|---|---|---:|---|---|
| Smoke | scripts/experiments/matrix.csv | 3 | reports/experiments/phase05_matrix_smoke/ | ✅ |
| Full | scripts/experiments/matrix_phase05_full.csv | 12 | reports/experiments/phase05_matrix_full/ | ✅ |

### 可重現性結論

- seed=42、N=10 兩次重跑結果完全一致（排除 generated_at）：**REPRO_PASS**
- summary：reward_mean=0.345291、reward_std=0.278632、avg_latency_ms=1.4887

### Full Matrix 結論（12 runs，seed 42–53，w ∈ {0.05, 0.08, 0.10, 0.12}）

| 指標 | 最小值 | 最大值 | 說明 |
|---|---:|---:|---|
| reward_mean | -0.164985 | 1.046194 | seed 間差異顯著，無固定 w 主導 |
| avg_latency_ms | 0.5964 | 2.9459 | 全數低於 5 秒 smoke 門檻 |
| reward_std | 0.0 | 0.0 | 單 run per seed，std=0 為預期 |

### 異常記錄

- 初次 matrix 匯總時因輸出目錄未隔離，導致 run 計數大於 matrix 列數 → 已改為隔離目錄後修正。
- Full 矩陣 reward_std 均為 0.0，原因為每個 cell 只跑 N=1 run；若需要統計顯著性分析，需將 `--runs` 提升至 ≥5。

### Gate 全數 PASS

V5-01（可重現）、V5-02（矩陣完整）、V5-03（匯總一致）、V5-04（研究可解釋）均已通過。

### 封卷結論

- Phase 05 smoke + full 兩層驗證流程已完整執行並封卷。
- 後續接 Phase 06 CI Gate 時，可將 `run_matrix.py + count_match 驗證` 作為標準 smoke job。
- 若需要跨 w 統計比較，建議 full matrix 每 cell 升為 N≥5 runs，並補上 reward_mean 的 seed 間 std 欄位。

### 已簽核

| Dimension | Signoff |
|---|---|
| Research | Completed ✅ (2026-06-01) |
| Product | Completed ✅ (2026-06-01) |
| Integration | Completed ✅ (2026-06-01) |

## 預估工時

- 1-3 days（已完成）
