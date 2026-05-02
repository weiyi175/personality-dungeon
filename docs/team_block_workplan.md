# 組員分工工作表（Block 對應）

| Block | 主要任務 | 主要輸入 | 主要輸出 | 預估工時 |
|---|---|---|---|---|
| Block 1 核心模擬引擎 | 修正核心 loop、事件掛點、維持分層 invariants | SDD.md、core/ players/ dungeon/ evolution/ | 可重跑 simulation、無契約破壞變更 | 24-40h |
| Block 2 實驗與掃參 | seed stability、grid/refine、formal gate | baseline summary、protocol 參數 | outputs CSV/TSV/JSON 與 gate 結果 | 30-50h |
| Block 3 分析指標 | cycle metrics、stage3 guards、可視化 | timeseries CSV、analysis 規範 | 分析圖、判定報表、結論摘要 | 24-36h |
| Block 4 機制設計 | H1/H2/H3 新 family 最小實作與 ablation | closure 結論、失敗模式 | 新機制報告與可重現實驗 | 40-64h |
| Block 5 敘事文件 | runbook、週報、簡報、Q&A | 研發日誌、outputs、分析圖 | 對內外一致敘事文件 | 16-28h |

## 兩週建議排程

1. Week 1：Block1/2/3 建 baseline 與回歸檢核，Block5 同步整理。
2. Week 2：Block4 跑機制最小版，Block2/3 依 gate 做回測。
3. 週末 gate：確認輸出可重跑、指標可比較、結論可防守。