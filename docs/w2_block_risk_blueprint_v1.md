# W2 實驗藍圖 v1：A3 例行回歸 + 漂移偵測 + block-level 穩健性對比

## 1) 目標

W2 固定交付三件事：

1. A3 例行回歸（主線健康檢查）。
2. 漂移偵測（相對 W1 參考基準）。
3. block-level 穩健性對比，輸出 block risk report v1。

## 2) 輸入資料

1. `outputs/track_a_protocol_regression_summary.json`
2. `outputs/w1/d5_w1_review_summary_v1.json`
3. `outputs/pers_cal_baseline_gate60_summary.json`
4. `outputs/pers_cal_baseline_gate60_block102_161_summary.json`
5. `outputs/pers_cal_baseline_gate60_block162_221_summary.json`
6. `outputs/pers_cal_baseline_gate60_block222_281_summary.json`
7. B1/B2/B3 block gate summaries（由 `scripts/w2_block_risk_report_v1.py` 固定清單讀入）

## 3) 執行流程

### 3.1 A3 例行回歸

```bash
./venv/bin/python -m simulation.track_a_protocol_regression
```

### 3.2 W2 block risk 報告生成

```bash
./venv/bin/python scripts/w2_block_risk_report_v1.py
```

### 3.3 漂移偵測規則（W1 -> W2）

以 W1 D5 `mainline` 為 reference，A3 最新資料為 current：

1. `a3_overall_pass = true`
2. `|delta_l1| <= 1`
3. `|delta_healthy| <= 3`
4. `|delta_mean_s3| <= 0.02`

全部通過：`stable`；否則：`drift_detected`。

### 3.4 block-level 穩健性對比規則

以每個 block 的 baseline 作對照，檢查 event family gate 結果：

1. `overall_pass`
2. `new_l1`
3. `l1 / healthy`
4. `delta_mean_s3_vs_baseline`
5. `fairness_fail_count`

焦點對比：`b1_async_poisson_r008` 在 4 個 blocks 的通過率與 `new_l1` 表現。

## 4) 產物（block risk report v1）

1. `outputs/w2/block_risk_report_v1.json`
2. `outputs/w2/block_risk_report_v1.csv`
3. `outputs/w2/block_risk_report_v1.md`

## 5) 本輪比對摘要（2026-04-12）

### 5.1 A3 + 漂移

1. 漂移判定：`stable`
2. `delta_l1=0`, `delta_healthy=0`, `delta_mean_s3=0.000000`

### 5.2 block-level 焦點對比（b1 async_poisson r0.08）

| block | overall_pass | new_l1 | verdict |
| --- | --- | --- | --- |
| 42..101 | true | 0 | pass |
| 102..161 | false | 2 | fail |
| 162..221 | false | 1 | fail |
| 222..281 | false | 4 | fail |

通過率：`1/4 = 0.25`，判定：`not_block_robust`。

### 5.3 B2/B3 對比（block 42..101）

1. B2：`overall_pass=false`, `new_l1=2`, `l1=3`, `healthy=47`
2. B3：`overall_pass=false`, `new_l1=2`, `l1=2`, `healthy=49`

## 6) W2 決策輸入映射

1. 主線節奏：`keep_80_20`（A3 stable + no drift）。
2. Event line：`not_block_robust`（先做 block 診斷，不重啟）。

## 7) DoD（W2 當週）

1. A3 至少完成 1 次，且 summary 存檔。
2. block risk report v1 三份產物齊備（json/csv/md）。
3. 焦點機制（當前為 b1 poisson r0.08）完成跨 block 比對與通過率判定。
4. 週會結論可直接引用 `outputs/w2/block_risk_report_v1.md`。
