# Phase 1–21 實驗總結

> **Personality Dungeon — 目標 B：循環動態**
>
> 核心研究問題：在三策略 Rock-Paper-Scissors payoff 結構下，能否在有限族群 discrete sampling regime 中穩定產生 **Level 3（持久旋轉循環）**？

---

## 目錄

1. [五大敘述弧總覽](#五大敘述弧總覽)
2. [Phase 逐一詳述](#phase-逐一詳述)
3. [累積實驗統計](#累積實驗統計)
4. [關鍵洞察](#關鍵洞察)
5. [最終成果](#最終成果)

---

## 五大敘述弧總覽

| # | 敘述弧 | 涵蓋 Phase | 核心問題 | 結論 |
|---|---|---|---|---|
| 1 | **Replicator Exhaustion** | 1–9 | Broadcast replicator 與變體能否產生 L3？ | 否。~360 runs, 0 L3。均化機制是根本性瓶頸。 |
| 2 | **RL Breakthrough** | 10–12 | 獨立 RL + 異質性能否取代 replicator？ | E1 首次 48 genuine L3/60 runs，但需 wide α。 |
| 3 | **Measurement Calibration** | 13–15b | 分類器盲點、gate 規格錯誤 | CC1 修正 +7 L3 seeds。Gate entropy 降至合理值。 |
| 4 | **Confirmation & Lock** | 15b–17 | 修正後能否穩定確認？基線鎖定。 | BL1 鎖定（well\_mixed ε=0.02, 20000r, 4/6 L3）。 |
| 5 | **L3 Optimization** | 18–21 | 如何在 BL1 基礎上最大化 L3 品質？ | P21a mild\_cw 達 **6/6 L3 (100%)**，專案最佳。 |

---

## Phase 逐一詳述

### Arc 1 — Replicator Exhaustion（Phase 1–9）

#### Phase 1：Level 3 Bottleneck Exploration（早期基礎）

- **代碼**：W-series (W1/W2/W3.1–W3.3), B-series (B2–B5), C-series (C1–C2), T-series
- **機制**：基礎 replicator + 八種變體（topology, init, drift, island, weight jitter, world coupling, testament inheritance, leader policy）
- **配置**：players=300, rounds=3000–4000, seeds={45,47,49}
- **結果**：**0/230 L3**（6 substudies × ~35 runs）
- **結論**：Replicator 的 broadcast uniform update 根本無法突破 entropy lock。所有變體均達 plateau。
- **突破/驗證方向**：「為什麼本該產生 RPS 循環的偏微分動力學在 sampled discrete synchronous regime 下完全被吸收？」→ 建立 Layer 1/2/3 瓶頸分層理論。

#### Phase 2：B1 Tangential Projection Replicator

- **機制**：在 replicator operator 內，將增長向量分解為 radial + tangential，以 $(1+\alpha)$ 放大旋轉分量
- **配置**：tangential\_alpha ∈ {0.0, 0.3, 0.5, 1.0, 2.0}, 3 seeds
- **結果**：**0/15 L3**
- **突破/驗證方向**：驗證「Operator 內部的旋轉分量能否被保存？」→ 否。即使 3× 放大旋轉信號，Layer 1 sampling noise 的步間去相關（decorrelation）仍在每步完全吸收。**問題不在信號強度，而在 broadcast sampling 本質。**

#### Phase 3：A1 Asynchronous Replicator

- **機制**：破壞 synchronous update——每輪只 p 比例的 players 執行 replicator\_step()，製造權重異質性
- **配置**：async\_update\_fraction ∈ {0.1, 0.2, 0.5, 1.0}, 3 seeds
- **結果**：**0/12 L3**
- **突破/驗證方向**：驗證「Update scheduling 能否打破同向推進？」→ 否。共享 growth vector $g = f(\bar{x})$ 保證所有 updated player 往同方向移動，非同步只延遲但不改方向。

#### Phase 4：S1 Sampling Sharpness Power-law β

- **機制**：以冪律銳化策略抽樣 $P_\beta(s) \propto w_s^\beta$，控制 Layer 1 noise
- **配置**：sampling\_beta ∈ {0.5, 1.0, 2.0, 5.0, 50.0}, 3 seeds
- **結果**：**0/15 L3**
- **突破/驗證方向**：驗證「Sampling sharpness 是瓶頸嗎？」→ 否。Replicator 均化已使 $w \approx 1{:}1{:}1$，故 $w^\beta \approx w$。**瓶頸已上移至 weight homogeneity（Layer 2）。**

#### Phase 5：M1 Mutation Rate

- **機制**：每輪 Dirichlet 變異打破 weight homogeneity：$w' = (1-\eta)w_{\text{new}} + \eta \cdot u_i$, $u_i \sim \text{Dir}(1,1,1)$
- **配置**：mutation\_rate η ∈ {0.0, 0.001, 0.005, 0.01, 0.05, 0.10}, 3 seeds
- **結果**：**0/15 L3**（weight\_dispersion 成功注入 0→0.024，20× 增長；但 strategy\_entropy 完全不變）
- **突破/驗證方向**：驗證「Per-player heterogeneity 能突破 Layer 2 averaging 嗎？」→ 否。**Layer 2 (popularity averaging) 在 LLN 下將個體差異抹平。** 瓶頸確認為 Layer 2+3 聯合均化效應。

#### Phase 6：L2 Local-Group Growth

- **機制**：將 N=300 players 分成 local groups，每組獨立計算 growth vector（group\_size=5 時方差放大 ~7.7×）
- **配置**：local\_group\_size ∈ {5, 10, 20, 50}, 3 seeds
- **結果**：**0/12 L3**
- **突破/驗證方向**：驗證「破壞 LLN averaging 能打破 Layer 2 嗎？」→ 否。每個 group 的 growth vector 方向仍由全局 payoff matrix eigenstructure 決定，population-level entropy 不變。

#### Phase 7：H1 Payoff-Niche Heterogeneity

- **機制**：per-group payoff niche bonus $A^{(k)} = A + \varepsilon \cdot \text{diag}(e_{k \bmod 3})$，搭配 L2 per-group evolution
- **配置**：ε ∈ {0.02–0.30}, niche\_group\_size ∈ {0, 5, 10}, 3 seeds
- **結果**：**0/18 L3**（inter\_group\_growth\_cosine 從 1.000→0.563，但 entropy 不變）
- **突破/驗證方向**：驗證「打破 payoff eigenstructure 能突破 entropy lock 嗎？」→ 否。Replicator 的 uniform per-group weight update 仍使 population-level diversity 被「方向平均」抵銷。**Replicator Framework 宣告 exhaustion。**

#### Phase 8：（過渡 — Replicator Framework Exhaustion Declaration）

- Phase 1–7 累積結論的彙整。層 1 (sampling noise) → 層 2 (popularity averaging) → 層 3 (operator structure) 三層瓶頸模型確立。任何 replicator 變體都無法突破，必須跳出此框架。

#### Phase 9：C1 Local Pairwise Imitation on Structured Graph

- **機制**：完全移除 broadcast replicator，改用 per-player pairwise Fermi imitation on structured graph（lattice4, small\_world, ring4）
- **配置**：3 topologies × 2 parameter combos × 3 seeds = 21 runs, rounds=5000
- **結果**：**0/18 L3**（spatial\_clustering=1.000，退化 consensus）
- **突破/驗證方向**：驗證「Per-player pairwise imitation 能突破 entropy lock 嗎？」→ 否。**Fermi imitation 是 consensus-seeking mechanism**：凸組合使所有 players 在 50–100 rounds 內收斂到相同權重。必須跳出 imitation 框架。

---

### Arc 2 — RL Breakthrough（Phase 10–12）

#### Phase 10：D1 Independent Reinforcement Learning

- **機制**：完全移除社會資訊流。每位 player 維持獨立 Q-table，exponential recency update + Boltzmann softmax
- **配置**：3 α × 2 β × 2 topo + controls, 3 seeds = 42 runs, rounds=6000
- **結果**：**3/42 L3**（皆為噪音假陽性，staircase phase 旋轉不存在）
- **突破/驗證方向**：驗證「Independent RL 能打破 consensus 而維持 cycling？」→ 弱。**Specialization ≠ Diversity 困境**：高 β 致過度專精（entropy ↓），低 β 致隨機行為。Symmetric game 使 Q-value 收斂到相似 profile（低 q\_std）。但首次脫離 replicator 發現微弱旋轉信號。

#### Phase 11：E1 Heterogeneous RL ⭐

- **機制**：per-player $\alpha_i \sim U(\alpha_{lo}, \alpha_{hi})$ 與 $\beta_i \sim U(\beta_{lo}, \beta_{hi})$，製造 fast/slow learner 與 explorer/exploiter 共存
- **配置**：3 α ranges × 3 β ranges × 2 topo × 3 seeds = 60 runs, rounds=6000
- **結果**：**48/60 genuine L3 rotation**（16/18 active conditions 達 3/3 L3）
- **突破/驗證方向**：**專案歷史首次確認相位旋轉真實存在。** Phase angle plot 顯示穩定的 staircase continuous rotation。**α 異質化是驅動旋轉的核心機制。** Pass Gate 雖仍未通過（entropy gate 設定不可能，見 Phase 15），但動力學現象已確認。

#### Phase 12：F1 Heterogeneous Payoff

- **機制**：在 E1 基礎上為每位 player 注入 payoff bias $\delta_i[s] \sim U(-\varepsilon, +\varepsilon)$
- **配置**：α 範圍縮窄至 (0.02, 0.08), 5 ε values × 2 topo × 3 seeds = 30 runs
- **結果**：**0/30 L3**
- **突破/驗證方向**：**α range 縮窄是致命傷。** (0.02, 0.08) 只有 4× 倍差（vs E1 的 80×），破壞 fast/slow learner 極化。**教訓：wide α (0.005–0.40) 不可觸動。** 單變數 vs 多變數變更的方法論教訓。

---

### Arc 3 — Measurement/Classifier Calibration（Phase 13–15b）

#### Phase 13：E2 Lower β Ceiling

- **機制**：最保守單變數實驗——只降低 β，完全保留 E1 的 wide α (0.005, 0.40)
- **配置**：β ∈ {3.0, 5.0, 8.0, 10.0} × 2 topo × 3 seeds = 24 runs
- **結果**：**10/24 L3**（β=8/10 條件通過，β=3/5 被分類器漏判）
- **突破/驗證方向**：發現**測量瓶頸**——β=3 的旋轉比 E1 更強（R²=0.97, 累積旋轉 105 rad, 2.8× E1），但振幅太低，autocorrelation-based Stage 2 無法偵測。**分類器有 systematic false negative。**

#### Phase 14：CC1 Classifier Calibration

- **機制**：零模擬成本修正——新增 **phase rotation R² fallback** 作為 Stage 2/3 替代通過路徑
- **配置**：零新 runs，純回溯重評 E2 全部 24 seeds；r2\_threshold=0.85, min\_rotation=20.0
- **結果**：L3 從 10 → **17**（+7 seeds 被正確升級）
- **突破/驗證方向**：修正分類器盲點。**原 autocorrelation gate 對 low-β 平滑旋轉存在系統性 false negative。** R² fallback 不依賴振幅大小，直接量測旋轉的線性一致性。7 顆被誤判的 seeds 中，wm\_β3 seed45 有 R²=0.972、rotation=140.4 rad（≈22 圈），為專案歷史最強旋轉信號。

#### Phase 15：G1 Entropy Regularization Q-value Centering

- **機制**：兩版本——v1 reward bonus（失敗），v2 Q-centering $Q(s) \leftarrow Q(s) - \lambda \alpha (Q(s) - \bar{Q})$
- **配置**：entropy\_lambda sweep × β sweep, 48 runs
- **結果**：entropy 僅 +0.005，L3 維持但不改善，Pass Gate 仍 0/48
- **突破/驗證方向**：**發現 Pass Gate entropy ≥ 1.18 根本不可能達成**——三策略 Shannon entropy 理論上界為 $\ln(3) \approx 1.0986$，gate 需求 107.4% of theoretical max。此為**規格錯誤**而非動力學限制。

#### Phase 15b：Pass Gate Correction & Retroactive Assessment

- **修正**：entropy ≥ 1.06（96.5% of $\ln 3$），q\_std > 0.04，L3 ≥ 2，γ ≥ 0
- **結果**：**5 條件首次通過修正版 gate**（E2 β=3 × 2 topo, G1 β=3 × 2 λ）
- **突破/驗證方向**：**專案歷史首次 gate pass。** E2 β=3 +CC1 修正後，即使無 G1 Q-centering 也通過——證明基礎機制（wide α + low β）已足夠。為 baseline lock 奠定基礎。

---

### Arc 4 — Confirmation & Baseline Lock（Phase 16–17）

#### Phase 16：F1v2 Payoff Perturbation on E2-best Base

- **機制**：汲取 F1 教訓——保留 E2-best 的 wide α + β=3.0 base 不動，疊加 payoff perturbation（additive，非 substitution）
- **配置**：payoff\_epsilon ∈ {0.00–0.08} × 2 topo × 3 seeds = 30 runs
- **結果**：**17/30 L3**；lattice4 ε=0.04 完美 3/3 L3
- **突破/驗證方向**：驗證「additive perturbation 能在保持 L3 的前提下優化指標？」→ 能，但拓撲相關（lattice4 空間結構保護 cycling，well\_mixed 在高 ε 崩潰）。

#### Phase 16b：Extended Confirmation（12000r, 6 seeds）

- **結果**：lattice4 ε=0.04: 4/6 L3 (67%), well\_mixed ε=0.02: 5/6 L3 (83%)
- **突破/驗證方向**：**L3 cycling 非暫態效應，而是持久現象。** 12000 rounds 穩定通過。

#### Phase 17：BL1 Baseline Lock ⭐

- **設置**：well\_mixed ε=0.02 + wide α(0.005, 0.40) + β=3.0, 20000r × 6 seeds
- **結果**：**PASS 修正版 gate**：L3=4/6 (67%), entropy=1.0758, q\_std=0.04613
- **突破/驗證方向**：**正式鎖定基線。** 跨三個時域（6000r/12000r/20000r）entropy/q\_std 完全收斂（Δ<0.001）。L3 vs Non-L3 seeds 的唯一區分維度是**旋轉一致性（stab/turn/s3）**而非 entropy 絕對值。核心洞察：L3 是 **coherent rotational dynamics**。

---

### Arc 5 — L3 Optimization（Phase 18–21）

#### Phase 18b：Fixed ε Re-sampling Short Scout

- **機制**：固定 ε=0.02，定期重抽 payoff perturbation δ（resample interval R sweep）
- **配置**：R ∈ {0, 100, 500, 1000, 2000} × 3 seeds
- **結果**：R=1000 最佳：2/3 L3, turn=0.074（超強）
- **突破/驗證方向**：驗證「re-sampling 頻率如何影響 cycling？」→ 存在最優 R≈1000，給 Q-learning 足夠時間建立每段旋轉慣性。

#### Phase 18a：ε Annealing Short Scout

- **機制**：ε 從較大值線性衰減（ε\_start → ε\_end），搭配定期重抽（R=resample interval）
- **配置**：8 conditions × 3 seeds = 24 runs, rounds=6000
- **冠軍**：ε\_start=0.04→ε\_end=0.01, R=1000: **3/3 L3, s3=0.941, turn=0.053**
- **突破/驗證方向**：驗證「Dynamic ε 能同時最優化旋轉率和品質？」→ 能。初始大 ε 建立旋轉方向，後期衰減鎖定 cycling。

#### Phase 18a Extended Confirmation（12000r）

- **結果**：Annealing 5/6 L3 (83%) vs Control 5/6 (83%)；annealing s3=0.734 (+10% over control)
- **突破/驗證方向**：L3 率持平但旋轉品質顯著優於靜態基線。

#### Phase 20：Q-Rotation Bias Short Scout

- **機制**：每輪 Q-update 後注入微小旋轉偏置（δ\_rot），直接推動 Q 向下一策略偏移
- **配置**：δ\_rot ∈ {0, 0.001, 0.005, 0.01, 0.02} × 3 seeds
- **結果**：**0/15 L3**（所有 δ\_rot > 0 均劣於 control）
- **突破/驗證方向**：**NEGATIVE RESULT。** BL1 的循環來自 payoff landscape + 異質學習率的**自組織**，人工偏置反而 desynchronize 自然的 phase lag 結構。

#### Phase 21：Per-Strategy α Multipliers Short Scout

- **機制**：不同策略有不同學習速率倍率 $r_s$，使 $\alpha_i \leftarrow \alpha_i \times r_s$，製造自然 phase lag
- **配置**：5 multiplier 組合 × 3 seeds = 15 runs
- **冠軍**：mild\_cw [1.2, 1.0, 0.8]: 2/3 L3, s3=0.610, turn=0.035
- **突破/驗證方向**：驗證「非對稱學習速率能自然產生 phase lag？」→ 能，但需 CW 方向（r\_A > r\_B）。物理詮釋：Dove 快學、Betray 慢學 → 防止系統鎖到低熵均衡。

#### Phase 21a Extended Confirmation ⭐⭐

- **配置**：mild\_cw [1.2, 1.0, 0.8], 12000r × 6 seeds vs symmetric control
- **結果**：**6/6 L3 (100%)** vs control 5/6 (83%)
- **品質躍升**：s3: 0.863 (BL1) → **0.940** (+9%), turn: 0.032 → **0.054** (+69%), stab: 0.840 → **0.940** (+12%)
- **突破/驗證方向**：**專案最強結果。** 首次達成 100% L3 rate，消除所有退化路徑。Mild\_cw 列為 **BL2 候選**。

#### Phase 18×21 Combo Scout

- **機制**：同時應用 ε annealing + per-strategy α multiplier
- **結果**：s3=0.921 < P18a\_only (0.941)，**Non-additive**
- **結論**：兩種機制不疊加，分別使用效果更佳。關閉。

---

## 累積實驗統計

| 敘述弧 | Phases | 機制群 | 約 Runs | L3 Seeds | 結論 |
|---|---|---|---|---|---|
| Replicator Exhaustion | 1–9 | replicator 8 變體 + pairwise imitation | ~360 | 0 | 框架本質性失敗 |
| RL Breakthrough | 10–12 | independent RL + heterogeneity | ~132 | 48 (E1 only) | wide α 是核心 |
| Measurement Calibration | 13–15b | β sweep, CC1, gate correction | ~120 | 17 (post-CC1) | 修正測量瓶頸 |
| Confirmation & Lock | 16–17 | F1v2 confirmation, BL1 lock | ~48 | 26+ | BL1 鎖定 |
| L3 Optimization | 18–21 | ε annealing, Q-rot, α multiplier | ~75 | 到 6/6 (100%) | P21a 最強 |
| **合計** | **1–21** | **21 phases** | **~735** | | |

---

## 關鍵洞察

### 1. 相位旋轉是真實的

Phase 11 (E1) 首次產生穩定 staircase phase rotation。Phase 13 (E2) β=3 產生專案最強旋轉信號（cumulative 105 rad, R²=0.97）。Phase 14 (CC1) 揭露分類器盲點——7 顆真實 L3 seeds 被系統性誤判。

### 2. L3 與多樣性無必然聯繫

Phase 17 (BL1) 發現 L3 vs Non-L3 seeds 在 entropy/q\_std 上幾乎無差異。**唯一區分維度是相位旋轉一致性**（stab, turn\_strength, s3）。L3 是 **coherent rotational dynamics**，非多樣性指標。

### 3. Wide α 是不可觸動的核心條件

Phase 12 (F1) 的失敗證明 α range 從 (0.005, 0.40) 縮窄到 (0.02, 0.08) 會完全摧毀 cycling。後續所有成功的 Phase 都嚴格保留 wide α。

### 4. 非對稱性推動旋轉清晰性

Phase 21 (mild\_cw [1.2, 1.0, 0.8]) 的 CW 方向製造了自然 phase lag，首次達成 100% L3 rate。機制：快學策略先適應、慢學延遲 → 防止系統鎖在低熵均衡。

### 5. 測量方法必須與被測現象匹配

Phase 14 (CC1) 是零成本的純測量修正，卻回收了 7 顆被誤判的 L3 seeds。啟示：研究管線中測量層（analysis）的設計品質和被測動力學一樣重要。

---

## 最終成果

**冠軍條件：Phase 21a Extended Confirmation**

| 參數 | 值 |
|---|---|
| Topology | well\_mixed |
| α range | (0.005, 0.40) — wide |
| β | 3.0 |
| ε (payoff perturbation) | 0.02 static |
| α multipliers | mild\_cw [1.2, 1.0, 0.8] |
| Rounds | 12000 |
| Seeds | 6 (45, 47, 49, 51, 53, 55) |

| 指標 | BL1 (Phase 17) | P21a mild\_cw | 改進 |
|---|---|---|---|
| L3 rate | 4/6 (67%) | **6/6 (100%)** | +33pp |
| mean s3 | 0.863 | **0.940** | +9% |
| mean turn | 0.032 | **0.054** | +69% |
| stab | 0.840 | **0.940** | +12% |
| entropy | 1.076 | 1.077 | ≈ |
| q\_std | 0.046 | 0.046 | ≈ |
