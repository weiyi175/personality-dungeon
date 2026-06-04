# P7-E：長期人格漂移（Long-Term Personality Drift）

> **目的**：使用 P7-D 找到的最佳 α*，在 200+ 輪的長期觀測下分析：
> 1. 人格向量是否收斂到某個「人格吸引子」（personality attractor）？
> 2. 不同初始人格（G-AGG / G-DEF / G-BAL）的軌跡是否殊途同歸，還是維持分異？
> 3. 社群層次（多玩家平均）人格是否出現集群效應？

---

## 前置條件

| 條件 | 來源 |
|------|------|
| G7D-01 ~ G7D-04 全數通過 | P7-D |
| α* 值已確定（P7-D 穩定性分類 S2 中的最大 α）| P7-D `p7d_stability_report.md` |

---

## 實驗設計

### 規模

| 參數 | 值 | 說明 |
|------|----|------|
| n_players | 4 | 同前 |
| n_rounds | 500 | burn-in 50 + observation 450（長期觀測）|
| burn_in | 50 | |
| α | α*（P7-D 決定）| 固定，不掃描 |
| η | η*（P7-D 決定）| 固定，不掃描 |
| seeds | {42, 43, 44, 45, 46, 47, 48, 49} | 8 seeds × 3 人格組 = 24 runs |
| 初始人格組 | G-AGG, G-DEF, G-BAL（各 8 seeds）| 測試吸引子依賴性 |

### 固定參數

| 參數 | 值 |
|------|----|
| payoff_mode | `matrix_ab`（a=1.0, b=0.9）|
| evolution_mode | `sampled` |
| personality_mode | `dynamic` |
| δ\_max | 0.5 |

---

## 分析維度

### 維度 1：人格吸引子分析

**問題**：長期運行後，$P_{500}$ 是否聚集在某個固定點附近？

**方法**：
- 繪製 24 runs 的 $P_{500}$（末端人格向量）散點圖（PCA 降至 2D）
- 計算 $\|P_{500} - P_0\|_2$（漂移量）
- 計算 $\text{std}(P_{500})$ across seeds（收斂穩健性）

**判斷標準**：
- **強吸引子**：同組 8 seeds 的 $P_{500}$ 聚集半徑 < 0.2
- **弱吸引子**：聚集半徑 0.2~0.5
- **無吸引子**：聚集半徑 > 0.5（隨機漂移）

---

### 維度 2：初始人格組間差異（殊途同歸 vs 保持分異）

**問題**：G-AGG、G-DEF、G-BAL 三組的末端 $P_{500}$ 是否彼此靠近？

**假設**：
- H-E1（殊途同歸）：三組最終吸引子相同 → 人格在長期動態中被「同化」
- H-E2（保持分異）：三組吸引子相異，差值 > 0.3 → 初始人格決定長期命運
- H-E3（分叉）：某些 seeds 收斂到同一吸引子，某些不同 → 存在相變臨界

**判斷方法**：
- 計算三組的末端 centroid 距離：$d_{AG-D} = \|\bar{P}^{(AGG)}_{500} - \bar{P}^{(DEF)}_{500}\|_2$
- 若 $d < 0.1$：殊途同歸；若 $d > 0.3$：保持分異

---

### 維度 3：人格-策略共演化軌跡

**觀測**：在 trait\_drift 最大的 run 中，人格向量的哪個 trait 變化最大？是否與策略分布的主導策略一致（即「高 IMP 玩家最終全走 aggressive」）？

**指標**：

| 指標 | 定義 |
|------|------|
| `dominant_trait_shift` | $\arg\max_i |P_{500,i} - P_{0,i}|$（變化最大的 trait）|
| `strategy_trait_alignment` | 主導策略 vs 預期的主導 trait 是否一致（見 P7-B §2.1 關聯表）|
| `cycle_phase` | tail 策略分布的循環模式（是否重現 B2-B5 的相位旋轉）|

---

### 維度 4：社群層次聚類（Multi-Player 平均）

**觀測**：4 位玩家的個體 $P_{500}$ 是否趨向相同（社會同質化），還是各自維持不同軌跡（個體分化）？

**方法**：
- 計算每個 run 的玩家間 $P_{500}$ 標準差（`inter_player_trait_std`）
- 對比 $P_0$ 的玩家間標準差（初始值應接近 0，因為同一文字輸入）
- 若 `inter_player_trait_std` 在 tail 窗口顯著增大 → 個體分化

---

## Gate 驗收標準

| Gate ID | 驗收條件 |
|---------|----------|
| G7E-01 | 24 runs 全數完成，`trait_saturation < 0.10`（不崩潰）|
| G7E-02 | G-AGG / G-DEF / G-BAL 三組的末端人格 centroid 距離 $d$ 可計算，並記錄 H-E1/H-E2/H-E3 結論 |
| G7E-03 | 產出人格軌跡 PCA 圖（`p7e_attractor_pca.png`），可視化辨識吸引子 |

> **研究封卷條件**：G7E-01~03 通過後，撰寫 `p7e_final_report.md`，總結三個維度的結論與對 P7 系列的整體評估。

---

## 產出物

```
reports/experiments/p7e_longterm_drift/
  run_{seed}_{personality_group}.json               # 每 run 完整 log（500 步）
  p7e_longterm_summary.csv                         # 24 rows：seed × group × 指標
  p7e_attractor_analysis.csv                       # 末端 P_500 向量 + centroid 距離
  p7e_final_report.md                              # 人工撰寫封卷結論
analysis/
  p7e_attractor_pca.png                            # PCA 2D 軌跡圖（24 runs）
  p7e_personality_trajectory_heatmap.png           # 9 traits × 500 steps heat map
  p7e_strategy_trait_alignment_table.md            # 策略-人格共演化對照表
```

---

## 整體 P7 系列封卷條件

P7-E `p7e_final_report.md` 完成後，在 `00_overview.md` 更新以下表格：

| 問題 | 結論（由 P7-E 填入）|
|------|---------------------|
| 閉環是否產生新的吸引子？ | TBD |
| α* 的穩定振盪是否觀測到循環動態？ | TBD |
| 初始人格是否影響長期命運？ | TBD |
| 社群層次是否出現同質化？ | TBD |

---

## 依賴關係

- **上游**：P7-D（G7D-01~04）✅ 才開始，α* 已確定
- **下游**：無（P7 系列最終階段）

---

*狀態：待執行 | 依賴 P7-D 的 α* 決定*  
*建立日期：2026-06-03*
