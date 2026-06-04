# P7-C：單回合回饋注入（Single-Round Feedback Injection Validation）

> **目的**：在小規模環境（4 玩家 × 30 輪）中驗證「人格向量動態回饋→payoff 調制→人格更新」的完整迴圈注入點正確，schema 符合 P7-B 規格，且 α=0 行為嚴格退化至 P7-A 靜態路徑。

---

## 前置條件

| 條件 | 來源 |
|------|------|
| G7A-01 ~ G7A-04 全數通過 | P7-A |
| P7-B 映射規格確定（W 矩陣已校準）| P7-B |
| SDD CSV schema 已更新（含 `personality_vector`、`delta_u` 欄位）| 需先更新 SDD §4 |

---

## 實驗規模（刻意最小化）

| 參數 | 值 | 說明 |
|------|----|------|
| n_players | 4 | 最小族群 |
| n_rounds | 30 | burn-in 10 + tail 20；目的是驗證注入，不是觀測收斂 |
| burn_in | 10 | |
| seeds | {42, 43, 44} | 3 seeds × 3 α 值 = 9 runs |
| α 測試值 | {0.0, 0.3, 1.0} | 退化組 / 中等回饋 / 最強回饋 |
| η（人格學習率）| 0.05 | P7-B 草稿值，固定不掃描 |
| personality_mode | `dynamic` | 啟用人格動態更新 |
| payoff_mode | `matrix_ab` | a=1.0, b=0.9 |
| evolution_mode | `sampled` | |
| 初始人格輸入文字 | `"我靈活應對局面"` (G-BAL) | 選中性組，排除初始 bias 干擾 |

---

## 注入點規格（Injection Point Spec）

### 每 step 的執行順序

```
Step t 開始
  1. 讀取 Pₜ（當前人格向量，初始為 SBERT 推斷的 P₀）
  2. 計算 bₜ = W · Pₜ
  3. 計算 Δu(Pₜ) = α · clip(bₜ, -0.5, +0.5)
  4. 以 u_final = u_base + Δu 計算本輪策略選擇
  5. 執行博弈 → 得到 rₜ（reward）、sₜ（策略）
  6. 計算 ΔP(t) = η · (rₜ - r̄) · g(sₜ)
  7. Pₜ₊₁ = clip(Pₜ + ΔP, -1, +1)
  8. 將 Pₜ₊₁ 存入 player state，供 Step t+1 讀取
Step t 結束
```

### CSV 欄位驗收（新增至 FrameSnapshot）

| 欄位名 | 型別 | 語意 |
|--------|------|------|
| `personality_vector` | float[9] | 本步驟開始時的 Pₜ（注入前）|
| `delta_u` | float[3] | Δu(Pₜ)，三策略調制量 |
| `personality_updated` | float[9] | 本步驟結束後的 Pₜ₊₁ |
| `feedback_strength` | float | 本 session 使用的 α 值 |

---

## Gate 驗收標準

| Gate ID | 驗收條件 | 嚴格程度 |
|---------|----------|----------|
| G7C-01 | α=0.0 的 9 runs 中，每個 step 的 `delta_u` 均為 [0,0,0] | **阻斷**（未過不繼續）|
| G7C-02 | α=0.0 的 reward_mean 與 P7-A 同 seed 同人格組差值 < 0.001 | **阻斷** |
| G7C-03 | `personality_updated` 欄位值在 [-1,1] 範圍內（所有 run、所有 step）| **阻斷** |
| G7C-04 | α=1.0 的 `personality_vector` 在 step 2~30 中逐步改變（非全程靜止）| 警告 |
| G7C-05 | 新增欄位符合 P7-B §7 的 CSV schema，無缺值、無格式錯誤 | **阻斷** |

> **G7C-01 與 G7C-02 是最重要的回歸保護**；任何一項失敗，代表靜態退化邏輯有誤，立即停止後續實驗。

---

## 診斷輸出（Debug Artifacts）

除標準 summary CSV 外，需輸出：

```
reports/experiments/p7c_injection/
  run_{seed}_alpha{α}.json                 # 完整 step log（含 personality_vector）
  p7c_injection_summary.csv               # 9 rows：seed × α × 指標
  p7c_regression_check.txt               # G7C-01 / G7C-02 比對結果
  p7c_personality_trajectory_{seed}.csv  # 每 step 的 9D 向量時間序列（debug 用）
```

### personality_trajectory 格式

```csv
step, impulsiveness, assertiveness, optimism, risk_aversion, suspicion, endurance, randomness, stability_seeking, curiosity, alpha
1, 0.662, 0.724, 0.576, -0.922, -0.373, 0.324, 0.662, -0.471, 0.776, 0.3
2, 0.668, 0.731, 0.582, -0.915, -0.370, 0.320, 0.660, -0.468, 0.779, 0.3
...
```

---

## 已知風險與應對

| 風險 | 應對方式 |
|------|----------|
| SBERT 推斷在每 step 重新呼叫，latency 累積 | P7-C 只驗證注入正確性，SBERT 僅呼叫 1 次（session init），後續用 P_t 疊加更新 |
| ΔP 過小，人格幾乎不動（G7C-04 警告）| 調高 η 至 0.1，或檢查 g(s, trait) 表設計 |
| 回饋注入後 payoff 爆炸 | δ\_max=0.5 的 clamp 應已防護，若仍出現異常值，降低 α 至 0.1 後重測 |

---

## 依賴關係

- **上游**：P7-A（G7A-01~04）✅ 才開始、P7-B W 矩陣確定 ✅ 才開始
- **下游**：P7-D、P7-E 均以本實驗的 schema 驗收為前提

---

*狀態：待執行 | 依賴 P7-A Gate 結果*  
*建立日期：2026-06-03*
