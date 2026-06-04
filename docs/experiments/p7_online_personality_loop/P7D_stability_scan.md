# P7-D：迴圈穩定性掃描（Closed-Loop Stability Scan）

> **目的**：系統性掃描 feedback strength α，找出穩定區間、振盪臨界點、以及人格吸引子的初步特徵。

---

## 前置條件

| 條件 | 來源 |
|------|------|
| G7C-01 ~ G7C-05 全數通過 | P7-C ✅ |
| personality_update_enabled 機制驗證 | P7-C ✅ |

---

## 掃描矩陣設計

### 主要掃描軸

| 參數 | 掃描值 | 說明 |
|------|--------|------|
| α（feedback strength）| {0.1, 0.2, 0.3, 0.4, 0.5} | 5 點線性間隔 |
| η（personality learning rate）| 0.05 | 固定（P7-B 規格值） |
| seeds | {42, 43, 44} | 3 seeds 再現性驗證 |
| group | G-AGG | 單一人格組（P7-A PASS） |

**總計：5 × 1 × 3 = 15 runs**

### 固定參數（Protocol Lock）

| 參數 | 值 | 說明 |
|------|----|------|
| n_players | 4 | 與 P7-A/C 一致 |
| n_rounds | 300 | 比 P7-C 稍長，觀察收斂 |
| burn_in | 50 | 前 50 輪丟棄 |
| tail | 250 | tail 視窗統計 |
| payoff_mode | `matrix_ab`（a=1.0, b=0.9） | BL2 鎖定 |
| personality_mode | `static`（固定初始） | 與 P7-A/C 同 |
| lambda_alpha/beta/r/risk | 0.15/0.10/0.20/0.20 | B1 驗證值 |

---

## 穩定性指標定義

### 1. 人格變動幅度（Personality Amplitude）

$$A_P = \sqrt{\frac{1}{9} \sum_{i=1}^{9} (P_{i,\text{final}} - P_{i,\text{initial}})^2}$$

人格向量從初始到末尾的 L2 距離，單位為標準化人格空間。

### 2. 振盪得分（Oscillation Score）

計算人格變化速率的自相關峰值：

$$\text{OSC} = \max_{k \geq 20} \text{ACF}_{\|v(t)\|}[k]$$

其中 $v(t) = |P(t) - P(t-1)|$ 為步間人格變化。

- OSC < 0.3：穩定收斂（低振盪）
- 0.3 ≤ OSC < 0.6：弱振盪
- OSC ≥ 0.6：強週期振盪

### 3. 收斂時間（Convergence Time）

首次滿足 $v(t) < 0.01$ 且之後 20 輪內持續低變化的輪數 $t_c$。

### 4. 策略-人格耦合強度（Coupling Score）

計算人格向量範數與策略熵的 Pearson 相關係數：

$$\text{Coupling} = \text{Corr}(\|P(t)\|, H(x(t)))$$

其中 $H(x) = -\sum_s x_s \ln x_s$ 為策略分布熵。

---

## 穩定性分類標準

| 分類 | 判定條件 |
|------|----------|
| **S1 靜態** | $A_P < 0.05$ AND OSC < 0.2 |
| **S2 穩定動態** | $A_P \in [0.05, 0.20]$ AND OSC < 0.4 |
| **S3 邊界振盪** | $A_P \in [0.20, 0.50]$ AND OSC ∈ [0.4, 0.65] |
| **S4 發散崩潰** | $A_P > 0.50$ OR OSC > 0.65 |

**研究目標**：找到 α* 使系統進入 S2（穩定動態），同時保持高策略多樣性。

---

## Gate 驗收標準

| Gate ID | 驗收條件 |
|---------|----------|
| **G7D-01** | 15 runs 全數完成，無例外 |
| **G7D-02** | 所有 runs reward 與 personality 有限（無 NaN/inf） |
| **G7D-03** | α 遞增時，$A_P$ 遞增（單調劑量-反應） |
| **G7D-04** | 存在至少一個 α ∈ [0.1, 0.5] 使系統分類為 S2 或 S3 |
| **G7D-05** | 同一 α 跨 3 seeds，$A_P$ 標準差 < 0.08（再現性） |

---

## 預期結果

| 假設 | 期望 | 意義 |
|------|------|------|
| H-D1 | α 越大，$A_P$ 越大 | 回饋強度與人格變動成正相關 |
| H-D2 | ∃ α* ∈ (0.1, 0.4) 使 OSC < 0.3 | 存在穩定操作點 |
| H-D3 | seed 間相同 α 的 $A_P$ 差異小 | 人格軌跡再現性高 |
| H-D4 | Coupling ∈ [0.5, 0.8] | 人格與策略中等耦合 |

---

## 視覺化規格

分析階段將產出：

1. **A_P vs α 曲線**：α 遞增時人格幅度變化
2. **OSC vs α 曲線**：振盪得分隨 α 變化（找臨界點）
3. **t_c vs α**：收斂時間隨 α 變化
4. **Stability Heatmap**：α × seed 的 S1/S2/S3/S4 分類

---

## 產出物（Artifacts）

```
reports/experiments/p7d_stability_scan/
  run_a{alpha}_{seed}.json               # 15 個完整 step logs
  p7d_stability_summary.csv              # 15 rows：α × seed × 所有指標
  p7d_stability_matrix.json              # α × seed 的 S1/S2/S3/S4 分類
  p7d_gates.json                         # Gate 驗收結果
  p7d_stability_report.md                # 人工分析，含 α* 建議
```

---

## 可重現指令（待實作）

```bash
./venv/bin/python scripts/experiments/run_p7d_stability_scan.py \
  --alphas 0.1 0.2 0.3 0.4 0.5 \
  --seeds 42 43 44 \
  --group G-AGG \
  --learning-rate 0.05 \
  --out reports/experiments/p7d_stability_scan
```

---

## 與 P7-C 的差異

| 項目 | P7-C | P7-D |
|------|------|------|
| 目標 | 驗證機制 | 特徵化系統 |
| n_rounds | 200 | 300 |
| α 值 | 3 個 (0.0, 0.2, 0.5) | 5 個 (0.1~0.5) |
| seeds | 2 個 | 3 個 |
| 新增指標 | 無 | OSC, A_P, t_c, Coupling |
| 總 runs | 6 | 15 |

---

## 依賴關係

- **上游**：P7-C（G7C-01~05）✅ 完成
- **下游**：P7-E 使用本實驗找到的 α* 與吸引子特徵

---

*建立日期*：2026-06-03  
*更新日期*：2026-06-04  
*狀態*：快速版本規格確定，待實作  
*預計 Gate 目標日*：2026-06-04（同日完成）

