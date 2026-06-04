# P7-A：基線快照（Static Personality Snapshot Baseline）

> **目的**：在引入任何閉環機制之前，用現有靜態路徑（SBERT 推斷一次 → 固定人格向量 → RL session）建立可重現的基線分布，作為 P7-C 以後的對照組。

---

## 研究問題

1. 在靜態人格設定下，不同初始人格向量 P₀ 對策略分布 x(T) 的影響幅度是多少？
2. 基線的 reward 分布（mean、std、分位數）是否足夠穩定（seed 間變異 < 10%）？
3. 現有 `/rl_sessions/{id}/step` 回傳的哪些欄位將作為 P7-C 回饋的輸入訊號？

---

## 實驗設定（Protocol Lock）

| 參數 | 值 | 說明 |
|------|----|------|
| n_players | 4 | 與 Phase 02 一致 |
| n_rounds | 200 | burn-in 50 + tail 150 |
| burn_in | 50 | 前 50 輪不計入統計 |
| payoff_mode | `matrix_ab` | a=1.0, b=0.9（現有基準） |
| evolution_mode | `sampled` | 既有離散抽樣 |
| personality_mode | `static` | **禁止動態更新** |
| feedback_strength α | 0.0 | 確認退化行為正確 |
| seeds | {42, 43, 44, 45, 46} | 5 seeds × 3 人格組 = 15 runs |

### 人格組定義（3 組固定文字輸入）

| 組別 | 輸入文字 | 預期主導特徵 |
|------|----------|-------------|
| G-AGG | `"我喜歡冒險挑戰"` | impulsiveness ↑, risk_aversion ↓ |
| G-DEF | `"我謹慎保守行事"` | stability_seeking ↑, risk_aversion ↑ |
| G-BAL | `"我靈活應對局面"` | curiosity ↑, randomness ↑ |

> 每次 session 初始化前，先呼叫 `POST /personality/infer_sbert` 取得 P₀，記錄完整 9D 向量。

---

## 觀測指標

| 指標 | 欄位來源 | 計算視窗 |
|------|----------|----------|
| `reward_mean` | `step.reward` | tail（第 51~200 輪）|
| `reward_std` | `step.reward` | tail |
| `strategy_dist` | `step.strategy_distribution` | tail 平均 |
| `dominant_strategy` | argmax(strategy_dist) | tail 最後 10 輪 |
| `phase_distribution` | `step.phase` | 全程 |

---

## Gate 驗收標準

| Gate ID | 驗收條件 |
|---------|----------|
| G7A-01 | 15 runs 全數完成，無 HTTP 5xx |
| G7A-02 | reward_mean 在各 seed 間 std < 0.15（穩定基線確認）|
| G7A-03 | **序數約束**：$\text{mean}(p_{\text{agg}}[\text{G-AGG}]) > \text{mean}(p_{\text{agg}}[\text{G-DEF}])$ 且 $\text{mean}(p_{\text{def}}[\text{G-DEF}]) > \text{mean}(p_{\text{def}}[\text{G-AGG}])$（人格→策略影響可觀測）|
| G7A-04 | α=0.0 下，每次執行結果 bit-exact 等同於現有靜態路徑（回歸）|

> **G7A-03 改用序數約束理由**：n_players=4 時，argmax-dominant 受隨機 seed 主導，噪音遮蔽人格效果。改用平均比例排序更適合小樣本檢驗人格方向性影響。

> **若 G7A-03 失敗**（序數約束不成立）：代表靜態人格對策略選擇無方向性影響，需重新評估 P7-B 的映射設計方向，**不繼續推進 P7-C**。

---

## 產出物（Artifacts）

```
reports/experiments/p7a_baseline/
  run_{seed}_{personality_group}.json   # 每 run 的完整 step log
  p7a_baseline_summary.csv             # 15 rows：seed × group × 指標
  p7a_baseline_report.md               # 人工撰寫分析，含 G7A-01~04 結論
```

---

## 可重現指令（待 P7-C 實作後填寫）

```bash
# 預留位置：靜態基線掃描腳本
# ./venv/bin/python scripts/experiments/run_p7a_baseline.py \
#   --seeds 42 43 44 45 46 \
#   --groups G-AGG G-DEF G-BAL \
#   --out reports/experiments/p7a_baseline
```

---

## 依賴關係

- **上游**：Phase 01（API schema 凍結）✅、Phase 02（Playable Loop 閉環）✅
- **下游**：P7-B 需要 G7A-03 通過才能確定映射方向

---

*狀態：待執行 | 預計 Gate 目標日：TBD*
