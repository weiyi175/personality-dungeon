# C-family Local Sampling Blueprint

本文件是 B-series formal closure 後的第一份新主線實驗規劃。
定位不是立即升格為 SDD formal contract，而是先把後續最高優先的新 family 寫成可直接實作、可審核、可收斂的 blueprint。

核心問題固定為：

> 如果把 global sampled aggregation 改成 local interaction + local update，Level 3 是否才有機會存活？

這個問題直接對應目前最強的新假說：B4 / B3 / B5 / B2 已逐層證明，在同一個 well-mixed sampled framework 內，只增加局部異質性、局部方向差、切線催化、或空間相位分離，仍不足以穩定打開 Level 3 basin。下一步因此不再是往同一個 global sampled update 上疊 patch，而是直接測試：**global shared growth / shared update 本身是否才是致命瓶頸。**

---

## 1. Family 定位

### 1.1 為什麼這條線排第一

相較於 B1（切線投影 operator）或更強 topology 改造，C-family 有三個優勢：

1. 它直接對應目前最強的結構性假說：問題可能不在 local signal 是否存在，而在 local signal 是否每輪都被重新壓回單一 global sampled object
2. 它比 B1 的理論成本低，不必先重寫 replicator 幾何語意與正值性理論
3. 它比單做 lattice-topology 更乾淨，因為它不是只改 graph，而是同時改掉「global shared growth vector」這個核心機制

### 1.2 Family 結構

本 family 分成兩步，明確沿著「離開 global aggregation」的強度遞增：

1. **C1: local pairwise imitation / pairwise comparison**
2. **C2: mini-batch local replicator**

兩者的差異是：

| family | 保留 replicator 幾何程度 | 是否保留 global shared growth vector | 工程成本 | 科學訊號乾淨度 |
|---|---|---|---|---|
| **C1** | 低 | **完全移除** | 中 | **最高** |
| **C2** | 中高 | **移除，改為 player-local growth** | 中高 | 高 |

先做 C1 的理由很直接：它最乾淨地砍掉目前懷疑最有問題的物件，也就是 shared global growth / shared update。若 C1 仍完全 flat，才值得進 C2，看「保留 replicator 外形、但把 growth localize」是否有差。

---

## 2. 研究問題與假說

### 2.1 主問題

在目前 locked working point（`players=300`, `rounds=3000`, `seeds={45,47,49}`, `a=1.0`, `b=0.9`, `cross=0.20`）下，若更新規則不再依賴全族群共享的 sampled growth / shared normalize，而改成**局部互動、局部比較、局部更新**，系統是否能首次出現 robust 的 Level 3 basin candidate？

### 2.2 可檢驗的假說

**H_C_main**：
Level 3 在目前 sampled path 下之所以無法穩定存活，主因不是缺少局部異質性，而是局部訊號在每輪都被 global aggregation 壓回單一更新方向。只要讓玩家長時間只根據 local interaction + local update 演化，至少某些 conditions 應能出現 `>=1/3 seeds` 的 Level 3。

**若 C1 / C2 都 fail，則較強的反假說成立**：
即使移除 global shared growth vector，Level 3 仍無法打開。此時就更有理由懷疑問題在 sampled discrete update 的更底層結構，或需要更高成本的 operator 幾何修改（例如 B1）。

### 2.3 三種可能結果與解讀

1. **C1 pass**：global shared aggregation 幾乎可直接視為主要瓶頸；C-family 升格為主線
2. **C1 fail but C2 pass / weak positive**：純 pairwise imitation 太粗糙，但 replicator-like 的 local growth 有價值；說明「保留局部 growth object」是有效方向
3. **C1 / C2 全 fail**：即使切掉 global shared growth 仍不足；後續才有理由進 B1 或更深層的 sampling semantics 重構

---

## 3. Family 不變條件

這一節先鎖 blueprint-level invariants，避免實作時又回到 B-series 類型的小補丁。

1. **active cells 不得再使用單一 global sampled growth vector**
2. **active cells 不得把每輪所有玩家的策略權重直接同步成同一組共享 weights**
3. 第一輪只測 `payoff_mode=matrix_ab`, `a=1.0`, `b=0.9`, `matrix_cross_coupling=0.20`
4. 第一輪不啟用 events，避免把 sampling 機制變更與 event noise 混在一起
5. 第一輪 graph 固定為靜態、無權重、無方向、無 rewiring
6. 第一輪只允許 ring 與 2D torus lattice 兩種 graph
7. 第一輪不得改動主 timeseries CSV schema；新增訊息只寫入 summary / combined / decision / provenance / plots
8. analysis 不得反向依賴 simulation；所有 graph / local-update 診斷應產生為 CSV/TSV 或純函式輸出
9. 此文件只是 blueprint；**一旦 protocol 要正式實作，必須先把 locked contract 寫進 SDD.md，再改碼**
10. **更新時序一律同步（synchronous）**：每輪開始時先鎖住所有玩家的舊權重 $w(t)$，再根據舊權重同步計算所有玩家的新權重 $w(t+1)$。不得使用 asynchronous / random-order sequential update，以避免引入 update-order 對結果的不可控影響
11. **neighbor selection 一律 uniform random without replacement**：C1 每輪每個 player 從 $N(i)$ 中均勻隨機選取 1 個 model neighbor $j$；C2 不涉及 neighbor sampling（使用完整 $B_i$）
12. **初始化沿用既有 baseline**：所有玩家的初始權重必須與 well-mixed sampled baseline 相同（`init_bias=0.12` 對應的初始化方式），並使用相同 seed 保證可比
13. **per-round CSV row 由全族群平均產生**：每輪結束後，`p_*(t)` 為全體玩家本輪策略抽樣的平均比例，`w_*(t)` 為全體玩家更新後權重的平均值。這保證主 timeseries schema 不變，同時允許 local dynamics 在 player-level 進行

---

## 4. Graph 設計

### 4.1 第一輪 graph choices

第一輪只允許兩種 topology：

1. **ring-4**：每個 player 接左右各 2 個鄰居，總 degree = 4
2. **lattice-4**：`15 x 20` torus grid，每個 player 接上下左右 4 個鄰居

選 degree = 4 的理由：

1. 它是最小但不過稀的 locality
2. 比 degree = 8 更容易看出 local domain 是否真能維持
3. 若第一輪就用 degree = 8，可能過早把 graph 逼近 well-mixed

**第二輪才允許 degree = 8**，且只有在第一輪出現 weak positive 或 graph-local diagnostics 顯示 ring/lattice-4 太碎、但仍有局部訊號時才開。

### 4.2 玩家數與 graph 對應

第一輪固定 `players=300`，因此 lattice 形狀固定為 `15 x 20`。若未來要改玩家數，必須另外鎖定可整除的 grid shape；第一輪 blueprint 不允許自動 reshape 或 ragged lattice。

### 4.3 Graph 控制語意

研究上的主 control 仍是：

1. **well-mixed sampled baseline**（既有 sampled control）

graph 本身不作為第一輪主要 control。原因是本 family 的核心假說不是「哪個 graph 比較好」，而是「只要不再用 global shared aggregation，local update 是否就能保住 Level 3 訊號」。因此第一輪 control 必須是既有 global sampled path，而不是 graph-internal 對照。

**Control 運行方式**：使用既有 `simulation.run_simulation.simulate()` 以相同固定工作點執行，`popularity_mode=sampled`、`selection_strength=0.06`、`memory_kernel=3`、`init_bias=0.12`。condition 命名為 `control_wellmixed_sampled`。控制組用相同 seeds 執行，其結果即為 paired comparison 的 baseline。

---

## 5. C1：Local Pairwise Imitation / Pairwise Comparison

### 5.1 核心設計

C1 的目標是最乾淨地切掉 global shared growth vector。每個 player 不再讀全族群 growth，也不再與所有人共享同一個更新方向；他只看自己的局部鄰居，並根據 payoff 差異把自己的混合策略往鄰居靠近。

### 5.2 建議 update semantics

對每個 player $i$（每輪同步執行）：

1. 在 graph 上**均勻隨機**選一個 model neighbor $j \in N(i)$
2. 用 local interaction payoff 計算 $\pi_i, \pi_j$：每人的 payoff 是**與自身所有鄰居對局的平均報酬**（即 $\pi_i = \frac{1}{|N(i)|} \sum_{k \in N(i)} w_i^\top A_{\text{local}} s_k$，其中 $s_k$ 為鄰居 $k$ 本輪抽樣策略）
3. 用 pairwise comparison probability 決定 $i$ 對 $j$ 的模仿強度：

$$
q_{ij} = \frac{1}{1 + \exp[-\beta_{pair}(\pi_j - \pi_i)]}
$$

4. 令 $w_i$ 與 $w_j$ 分別為兩人的當前 mixed-strategy weights，更新為：

$$
w_i' = \operatorname{normalize}\big((1 - \mu q_{ij}) w_i + (\mu q_{ij}) w_j\big)
$$

其中：

- $\mu$：pairwise imitation strength，控制每次靠近鄰居的幅度
- $\beta_{pair}$：comparison sharpness，控制 payoff 差異對模仿機率的敏感度

### 5.3 第一輪固定與掃描

第一輪只掃一個主強度維度，避免把 operator 參數與 topology 掃描混在一起。

固定值：

1. `beta_pair = 8.0`
2. `players=300`
3. `rounds=3000`
4. `burn_in=1000`
5. `tail=1000`
6. `seeds={45,47,49}`
7. `graph_topology ∈ {ring4, lattice4}`

掃描值：

1. `pairwise_imitation_strength ∈ {0.10, 0.20, 0.35}`

總 active cells：

1. `2 topologies × 3 strengths = 6 cells`
2. 加 `1` 個既有 well-mixed sampled baseline
3. 共 `7 conditions × 3 seeds = 21 runs`

### 5.4 為什麼這樣鎖

1. 先不掃 degree，避免把「locality 長度」和「operator 強度」混在一起
2. 先固定 `beta_pair`，只掃 `mu`，避免兩個 operator 強度一起漂
3. ring 與 lattice 各保留一種最小 topology，足以區分「一維局部鏈」與「二維局部面」

### 5.5 C1 專屬診斷

第一輪至少必須新增：

1. `graph_topology`
2. `graph_degree`
3. `pairwise_imitation_strength`
4. `pairwise_beta`
5. `mean_pairwise_adoption_prob`
6. `mean_imitation_step_norm`
7. `mean_edge_strategy_distance`
8. `mean_neighbor_payoff_gap`
9. `mean_local_phase_spread`
10. `edge_disagreement_rate`
11. `representative_graph_phase_png`

這些指標的用意：

1. 確認 imitation 真的在發生，而不是 no-op
2. 確認 graph 上是否形成 local phase domains
3. 若 fail，分辨是「local interaction 根本沒形成差異」還是「形成了但仍不夠」

**診斷閾值參考**（用於判讀，非 hard gate）：

| 指標 | 「訊號未形成」 | 「訊號已形成但不足」 | 「強訊號」 |
|---|---|---|---|
| `mean_edge_strategy_distance` | < 0.03 | 0.03–0.08 | >= 0.08 |
| `mean_local_phase_spread` | < 0.05 rad | 0.05–0.15 rad | >= 0.15 rad |
| `mean_pairwise_adoption_prob` | < 0.05 or > 0.95 | 0.05–0.95 | 0.20–0.80 |
| `edge_disagreement_rate` | < 0.05 | 0.05–0.20 | >= 0.20 |

這些閾值在第一輪結束後可根據實際數據調整，但必須在 decision.md 中明確記錄調整原因。

---

## 6. C2：Mini-Batch Local Replicator

### 6.1 核心設計

C2 比 C1 更接近原本 replicator family，但把 global shared growth 改成 player-local mini-batch growth。直觀上，它測的是：如果保留 replicator-like exponential update 的外形，但每個 player 只根據自己的 local neighborhood 計算 growth，是否就能保住 Phase 3 訊號。

### 6.2 建議 update semantics

對每個 player $i$：

1. 取其 local batch $B_i = \{i\} \cup N(i)$
2. 由 batch 內玩家的 **當前權重**（而非抽樣策略）估計 local popularity：$x_i = \frac{1}{|B_i|} \sum_{k \in B_i} w_k$（即 batch 內權重向量的等權平均）
3. 用 local popularity 算 local payoff vector：

$$
u_i = A x_i
$$

4. 對 player $i$ 當前 weights $w_i$ 計算 local advantage：

$$
g_i = u_i - \langle w_i, u_i \rangle \mathbf{1}
$$

5. 用 replicator-like exponential update 回寫：

$$
w_i' = \operatorname{normalize}(w_i \odot \exp(k_{local} g_i))
$$

### 6.3 第一輪固定與掃描

固定值：

1. `players=300`
2. `rounds=3000`
3. `burn_in=1000`
4. `tail=1000`
5. `seeds={45,47,49}`
6. `graph_topology ∈ {ring4, lattice4}`
7. `local_batch = ego + 1-hop neighbors`

掃描值：

1. `local_selection_strength ∈ {0.04, 0.06, 0.08}`

總 active cells 同樣固定為 `6 cells`，再加 `1` 個 well-mixed sampled baseline，共 `21 runs`。

### 6.4 C2 與 C1 的分工

| 問題 | C1 | C2 |
|---|---|---|
| 去除 global shared growth | 最徹底 | 是 |
| 保留 replicator 家族形式 | 否 | 是 |
| 機制訊號是否乾淨 | 最高 | 高 |
| 與既有 runtime 相容性 | 中 | 較高 |

建議正式優先序：

1. 先做 C1
2. 若 C1 hard fail，再開 C2
3. 若 C1 已 pass，C2 反而降為解釋性對照，而不是主力線

### 6.5 C2 專屬診斷

第一輪至少必須新增：

1. `local_selection_strength`
2. `mean_local_growth_norm`
3. `mean_local_growth_cosine_vs_global`
4. `mean_player_growth_dispersion`
5. `mean_batch_phase_spread`
6. `mean_neighbor_popularity_dispersion`
7. `mean_local_update_step_norm`
8. `representative_local_growth_png`

**診斷閾值參考**（用於判讀，非 hard gate）：

| 指標 | 「訊號未形成」 | 「訊號已形成但不足」 | 「強訊號」 |
|---|---|---|---|
| `mean_local_growth_cosine_vs_global` | > 0.95 | 0.85–0.95 | <= 0.85 |
| `mean_player_growth_dispersion` | < 0.01 | 0.01–0.03 | >= 0.03 |
| `mean_batch_phase_spread` | < 0.05 rad | 0.05–0.15 rad | >= 0.15 rad |
| `mean_neighbor_popularity_dispersion` | < 0.02 | 0.02–0.06 | >= 0.06 |

---

## 7. Gate 設計

由於 C1/C2 都不再是 mean-field 可表達的 well-mixed ODE，第一輪不設 deterministic gate；整體 gate 結構沿 B2 的邏輯：**G0 + G2 only**。

### 7.1 G0：Degrade / Sanity Gate

這裡不要求 active runtime 精確退化成既有 sampled control，但至少必須通過 helper-level 與 runtime-level sanity。所有數值容差統一為 **±1e-9**（與 B-series G0 一致）。

**C1 G0**（共 4 項，全部必須通過）：

1. `pairwise_imitation_strength = 0` 時，$\|w_i' - w_i\|_\infty < 10^{-9}$
2. 若 $w_i = w_j$（逐元素相等），則 $\|w_i' - w_i\|_\infty < 10^{-9}$，不得因 payoff 差異產生偽旋轉
3. 更新後權重 $w_i'$ 每個分量 $> 0$ 且 $|\sum w_i' - 1| < 10^{-9}$
4. 對固定 seed，兩次獨立呼叫 `run_c1_cell` 產生 **bit-identical** 的 timeseries CSV

**C2 G0**（共 4 項，全部必須通過）：

1. 若 `local_batch = global_population`（即 graph 為 complete graph），local-growth helper 的輸出與 global-popularity growth 的 $\ell_\infty$ 差距 $< 10^{-9}$
2. `local_selection_strength = 0` 時，$\|w_i' - w_i\|_\infty < 10^{-9}$
3. 更新後權重 $w_i'$ 每個分量 $> 0$ 且 $|\sum w_i' - 1| < 10^{-9}$
4. 對固定 seed，兩次獨立呼叫 `run_c2_cell` 產生 **bit-identical** 的 timeseries CSV

若任一 G0 項目不通過，先修 runtime / helper，不得進正式 short scout。

### 7.2 G2：Short Scout

成功標準沿用既有 gate 精神，並與 SDD §4.6 Level 定義完全對齊。

**Per-seed Level 3 判定（全部必須滿足）**：

1. `amplitude >= 0.02`（`series=p` 下的振幅閾值）
2. Stage 2 通過：`corr >= 0.09` 或 `fft_power_ratio >= 8.0` 或 `p_value <= 0.05`
3. Stage 3 通過：`stage3_score >= 0.55`（`stage3_method=turning`, `eta=0.55`）且 `turn_strength >= 0.0`

**Per-condition pass 判定**：

1. `level3_seed_count >= 1`（即 $\geq 1/3$ seeds 達到 Level 3）
2. `mean_env_gamma ∈ [-5e-4, +5e-4]`（排除系統性發散型假陽性）
3. `phase_amplitude_stability >= 0.3`（排除末段振幅崩潰型假陽性）

**Overall scout pass**：至少 1 個 non-control condition 達到 per-condition pass。

**假陽性排除規則**：

1. 若 `level3_seed_count >= 1` 但 `mean_env_gamma` 超出 $\pm 5 \times 10^{-4}$，verdict 降為 `weak_positive`，不算 pass
2. 若 `level3_seed_count >= 1` 但 simplex tail 圖目視顯示策略比例持續單調趨近邊界（非循環軌跡），verdict 降為 `suspicious`，需人工覆核

### 7.3 Hard stop

**C1 或 C2 個別 family 的 hard stop**（三條件 AND，全部滿足則 closure）：

1. 所有 6 個 active cells 都 `level3_seed_count = 0`
2. `max(stage3_uplift_vs_control) < 0.02`（stage3_uplift 定義見 §7.5）
3. local mechanism diagnostics 全部低於 weak-positive 閾值（見 §7.4）

若三條件全部滿足，直接記為 `close_c1` 或 `close_c2`，在 decision.md 中寫明 closure rationale，不做 finer grid 或 longer confirm。

### 7.4 Weak positive 定義

若未達 per-condition pass，但出現以下任一組合，允許記為 `weak_positive`：

1. `level3_seed_count = 0`，但 `max(stage3_uplift_vs_control) >= 0.02`
2. `level3_seed_count = 0`，但 **C1**: `mean_edge_strategy_distance >= 0.08` 且 `mean_local_phase_spread >= 0.15 rad`；**C2**: `mean_local_growth_cosine_vs_global <= 0.85` 且 `mean_player_growth_dispersion >= 0.03`（上述值代表 graph-local signal 確實形成）
3. 恰好 1 個 cell 出現 `level3_seed_count = 1`，但 `mean_env_gamma ∈ [-5e-4, +5e-4]` 且 simplex tail 圖顯示循環（非單調趨邊界）

**Weak positive 後續動作**：允許 1 次 targeted follow-up（最多再掃 3 cells × 3 seeds = 9 runs），但不得超過此限制。若 follow-up 仍未達 pass，則 closure。

### 7.5 stage3_uplift_vs_control 計算定義

對每個 active condition $c$ 與其 per-seed 指標：

$$
\text{stage3\_uplift}(c) = \frac{1}{|S|} \sum_{s \in S} \big[ \text{stage3\_score}(c, s) - \text{stage3\_score}(\text{control}, s) \big]
$$

其中 $S = \{45, 47, 49\}$，control 為同 seed 的 well-mixed sampled baseline run。

此定義確保比較始終是同 seed 配對（paired comparison），避免 seed-level variance 干擾。

---

## 8. 第一輪正式 protocol 建議

### 8.1 C1 第一輪 locked candidate

建議命名：

1. `C1: local pairwise imitation / local pairwise comparison`

固定工作點：

1. `players=300`
2. `rounds=3000`
3. `burn_in=1000`
4. `tail=1000`
5. `seeds={45,47,49}`
6. `payoff_mode=matrix_ab`
7. `a=1.0`
8. `b=0.9`
9. `matrix_cross_coupling=0.20`
10. `series=p`
11. `eta=0.55`
12. `stage3_method=turning`
13. `phase_smoothing=1`
14. `enable_events = false`
15. `init_bias = 0.12`
16. `memory_kernel = 3`
17. `selection_strength = 0.06`（只用於 control；active cells 的 k 由 local operator 取代）

掃描：

1. `graph_topology ∈ {ring4, lattice4}`
2. `pairwise_imitation_strength ∈ {0.10, 0.20, 0.35}`
3. `pairwise_beta = 8.0`

總 runs：

1. `6 active cells + 1 baseline control = 21 runs`

### 8.2 C2 第一輪 locked candidate

只有在 C1 正式 closure 後才開。

固定工作點與上同，掃描改為：

1. `graph_topology ∈ {ring4, lattice4}`
2. `local_selection_strength ∈ {0.04, 0.06, 0.08}`
3. `local_batch = ego + one-hop neighbors`

同樣 `21 runs`。

---

## 9. Runtime 拆分建議

建議不要把 C-family 硬塞回既有 `simulation.run_simulation.simulate()` 主幹；這兩條線已經不只是加一個 operator 參數，而是改掉互動與更新的核心語意。

### 9.1 建議新增檔案

1. `simulation/c1_local_pairwise.py`
2. `simulation/c2_local_minibatch.py`
3. `evolution/local_graph.py`
4. `evolution/local_pairwise.py`
5. `evolution/local_replicator.py`

### 9.2 建議資料結構

```python
@dataclass(frozen=True)
class GraphSpec:
    topology: str          # ring4 | lattice4
    degree: int            # 4 for first round
    lattice_rows: int | None = None
    lattice_cols: int | None = None


@dataclass(frozen=True)
class C1CellConfig:
    condition: str
    graph_spec: GraphSpec
    pairwise_imitation_strength: float
    pairwise_beta: float
    players: int
    rounds: int
    seed: int
    a: float
    b: float
    cross: float
    burn_in: int
    tail: int
    init_bias: float
    memory_kernel: int
    out_dir: Path


@dataclass(frozen=True)
class C2CellConfig:
    condition: str
    graph_spec: GraphSpec
    local_selection_strength: float
    players: int
    rounds: int
    seed: int
    a: float
    b: float
    cross: float
    burn_in: int
    tail: int
    init_bias: float
    memory_kernel: int
    out_dir: Path
```

### 9.3 最小函式建議

1. `build_graph(players, graph_spec, seed) -> list[list[int]]`
2. `local_payoff_vector(player_index, neighbors, all_weights, a, b, cross) -> np.ndarray`
3. `pairwise_adoption_probability(payoff_i, payoff_j, beta_pair) -> float`
4. `pairwise_imitation_update(weights_i, weights_j, mu, q) -> np.ndarray`
5. `local_replicator_advantage(weights_i, local_popularity, a, b, cross) -> np.ndarray`
6. `local_replicator_update(weights_i, advantage_i, k_local) -> np.ndarray`
7. `run_c1_cell(config) -> dict`
8. `run_c2_cell(config) -> dict`
9. `run_c1_scout(...) -> dict`
10. `run_c2_scout(...) -> dict`

---

## 10. 輸出與 Decision 契約

第一輪建議沿用 B-series 風格：

1. `summary.tsv`
2. `combined.tsv`
3. `decision.md`
4. simplex 軌跡圖
5. phase-vs-amplitude 圖
6. graph-local diagnostics 圖

### 10.1 Summary / combined 至少必須新增

共同欄位：

1. `family` (`c1_local_pairwise` or `c2_local_minibatch`)
2. `graph_topology`
3. `graph_degree`
4. `local_update_mode`
5. `stage3_uplift_vs_control`
6. `level3_seed_count`
7. `mean_env_gamma`
8. `phase_amplitude_stability`

C1 專屬欄位：

1. `pairwise_imitation_strength`
2. `pairwise_beta`
3. `mean_pairwise_adoption_prob`
4. `mean_imitation_step_norm`
5. `mean_edge_strategy_distance`

C2 專屬欄位：

1. `local_selection_strength`
2. `mean_local_growth_norm`
3. `mean_local_growth_cosine_vs_global`
4. `mean_player_growth_dispersion`
5. `mean_batch_phase_spread`

### 10.2 Decision markdown 結構與必要欄位

decision.md 必須採用與 B-series 一致的結構化 per-condition 行格式：

```
condition: control_wellmixed_sampled  seed_count=3  level3_seed_count=0  mean_stage3_score=0.42  mean_env_gamma=-1.2e-4  verdict=control
condition: ring4_mu0.10  seed_count=3  level3_seed_count=0  stage3_uplift=+0.008  mean_env_gamma=+2.1e-4  mean_edge_strategy_distance=0.052  verdict=fail
...
```

每個 condition 行至少包含：

1. `condition`：condition 命名
2. `seed_count`：總 seed 數
3. `level3_seed_count`：達到 Level 3 的 seed 數
4. `stage3_uplift`（非 control）：vs control 的 paired uplift（見 §7.5）
5. `mean_env_gamma`：tail 區間的 envelope gamma
6. `phase_amplitude_stability`：tail 圖振幅穩定性
7. `short_scout_pass`：`yes` / `no`
8. `hard_stop_fail`：`yes` / `no`
9. `verdict`：`control` / `pass` / `weak_positive` / `fail`

C1 額外：`pairwise_imitation_strength`, `mean_edge_strategy_distance`, `mean_pairwise_adoption_prob`
C2 額外：`local_selection_strength`, `mean_local_growth_cosine_vs_global`, `mean_player_growth_dispersion`

文件末尾必須包含：

1. baseline control 指標摘要
2. 整體 verdict（`pass_c1` / `weak_positive_c1` / `close_c1` 等）
3. **mechanism signal 判讀**：显式寫明是「graph-local signal 未形成」還是「graph-local signal 已形成但仍不足以打開 Level 3」
4. 若硘到 `weak_positive`，必須列出建議的 follow-up cells

### 10.3 Provenance JSON

每個 seed run 必須輸出 `provenance.json`，記錄：

1. `family`：`c1_local_pairwise` 或 `c2_local_minibatch`
2. `condition`：condition 命名
3. `seed`：隨機種子
4. `graph_topology`、`graph_degree`
5. 所有 operator 參數（`pairwise_imitation_strength`, `pairwise_beta` 等）
6. `out_csv`：對應的 timeseries CSV 路徑
7. `cycle_level`、`stage3_score`、`turn_strength`、`env_gamma`
8. `git_hash`（若可取得）、`timestamp`

Summary / Combined TSV 的每一行必須包含 `provenance_json` 欄位，指向對應的 JSON 檔案路徑。

---

## 11. 最小測試集

建議新增：

1. `tests/test_local_graph.py`
2. `tests/test_c1_local_pairwise.py`
3. `tests/test_c2_local_minibatch.py`

### 11.1 Graph helper 測試

1. ring-4 每個 player 恰有 4 個鄰居，且鄰居關係對稱（$j \in N(i) \Leftrightarrow i \in N(j)$）
2. lattice-4 torus 每個 player 恰有 4 個鄰居，邊界 wrap 正確（例如 player 0 的鄰居包含 player 14 在 15x20 torus 上）
3. graph 建立對固定 seed 可重現（兩次呼叫結果 bit-identical）
4. 不允許自迴圈（$i \notin N(i)$）
5. `players=300` 時 ring4 與 lattice4 的總邊數分別為 600 與 600（因 degree=4，無向圖邊數 = $N \times d / 2$）

### 11.2 C1 helper-level 測試

1. `mu=0` 時 $\|w_i' - w_i\|_\infty < 10^{-12}$（精確 no-op）
2. 權重更新後三分量全為正且 $|\sum w_i' - 1| < 10^{-9}$
3. 若 $\|w_i - w_j\|_\infty < 10^{-12}$，則 $\|w_i' - w_i\|_\infty < 10^{-9}$
4. adoption probability 對 payoff gap 單調：$\pi_j - \pi_i > 0 \Rightarrow q > 0.5$；$\pi_j - \pi_i < 0 \Rightarrow q < 0.5$；$\pi_j = \pi_i \Rightarrow |q - 0.5| < 10^{-9}$
5. 極端 payoff gap 不導致數值溢出：`beta_pair=8`, $|\pi_j - \pi_i| = 100$ 時仍回傳有效 $q \in (0, 1)$

### 11.3 C2 helper-level 測試

1. `k_local=0` 時 $\|w_i' - w_i\|_\infty < 10^{-12}$（精確 no-op）
2. `local_batch=global_population` 時 local-growth 的 $\ell_\infty$ 差距 vs global-popularity 版本 $< 10^{-9}$
3. 權重更新後三分量全為正且 $|\sum w_i' - 1| < 10^{-9}$
4. advantage 向量的預期屬性：$\langle w_i, g_i \rangle = 0$（advantage vs current strategy 正交，容差 $10^{-9}$）

### 11.4 Harness / I/O 測試

1. smoke run（`players=30, rounds=120, seeds=[45]`，很小參數確保快速）會寫出 `summary.tsv`, `combined.tsv`, `decision.md`
2. summary 必須包含 `graph_topology`, `graph_degree`, `provenance_json` 及對應 family 專屬欄位
3. combined 每行必須包含 `condition`, `seed`, `cycle_level`, `stage3_score`, `verdict`
4. `provenance_json` 欄位指向的檔案必須實際存在且可解析
5. decision.md 包含至少一行 `verdict=control` 和至少一行 non-control verdict
6. control condition 的 `stage3_uplift` 欄位必須為空或不存在（control 不對自身算 uplift）
7. `n_strata=1` 那類 B3 退化語意在 C-family 不適用；因此必須改以 helper-level no-op 測試替代，不得假裝沿用舊測試模板

### 11.5 端對端回歸測試

1. `run_c1_scout`（或 `run_c2_scout`）對相同 seed 兩次執行，產生的 `combined.tsv` 數值欄位差異 $< 10^{-9}$（確保可重現）
2. control condition 的 `cycle_level` 與 `stage3_score` 必須與直接呼叫既有 `simulate()` 的結果一致（驗證 control 執行路徑未被意外修改）

---

## 12. 推薦實作順序

1. 先做 `evolution/local_graph.py`，把 ring4 / lattice4 graph helper 與測試補齊
2. 再做 C1 helper：pairwise probability + convex imitation update
3. 再做 `simulation/c1_local_pairwise.py` 的最小 smoke harness
4. 跑 C1 smoke 與 21-run short scout
5. 若 C1 formal closure，再做 C2 helper 與 harness
6. 跑 C2 smoke 與 21-run short scout

---

## 13. 明確不做的事

第一輪 blueprint 明確禁止以下事項：

1. 不同時掃 graph degree 與 operator strength
2. 不同時引入 events
3. 不加入 directed / weighted / rewired graph
4. 不把 C1 與 C2 混在同一個 harness 同時掃
5. 不在第一輪就做 small-world / random graph
6. 不在第一輪就做異步 update schedule 掃描
7. 不把 personality heterogeneity 再疊進 C-family 第一輪，避免與 B-series 的解釋混淆
8. 不使用 asynchronous / random-order update（見 §3 不變條件 #10）
9. 不在第一輪引入 weighted / heterogeneous beta_pair（所有 player 共用同一個 beta_pair）

---

## 14. 建議命名與命令模板

### 14.1 C1

```bash
./venv/bin/python -m simulation.c1_local_pairwise \
  --seeds 45,47,49 \
  --topologies ring4,lattice4 \
  --pairwise-imitation-strengths 0.10,0.20,0.35 \
  --pairwise-beta 8.0 \
  --players 300 --rounds 3000 \
  --burn-in 1000 --tail 1000 \
  --a 1.0 --b 0.9 --matrix-cross-coupling 0.20 \
  --out-root outputs/c1_local_pairwise_short_scout \
  --summary-tsv outputs/c1_local_pairwise_short_scout_summary.tsv \
  --combined-tsv outputs/c1_local_pairwise_short_scout_combined.tsv \
  --decision-md outputs/c1_local_pairwise_short_scout_decision.md
```

### 14.2 C2

```bash
./venv/bin/python -m simulation.c2_local_minibatch \
  --seeds 45,47,49 \
  --topologies ring4,lattice4 \
  --local-selection-strengths 0.04,0.06,0.08 \
  --players 300 --rounds 3000 \
  --burn-in 1000 --tail 1000 \
  --a 1.0 --b 0.9 --matrix-cross-coupling 0.20 \
  --out-root outputs/c2_local_minibatch_short_scout \
  --summary-tsv outputs/c2_local_minibatch_short_scout_summary.tsv \
  --combined-tsv outputs/c2_local_minibatch_short_scout_combined.tsv \
  --decision-md outputs/c2_local_minibatch_short_scout_decision.md
```

---

## 15. 正式驗收清單（Acceptance Checklist）

本節列出「從 blueprint 進入正式實作」以及「short scout 結束後的判讀」兩組驗收標準。

### 15.1 Implementation Readiness Gate

以下全部通過後才可開始正式 short scout（21 runs）：

| # | 項目 | 通過標準 |
|---|------|----------|
| IR-1 | SDD 契約已更新 | `SDD.md` 已包含 C1（或 C2）的 locked contract section，包含 graph spec、update semantics、output schema |
| IR-2 | G0 全項通過 | §7.1 的 4 項 G0 測試全部 PASS |
| IR-3 | Graph helper 測試通過 | §11.1 的 5 項測試全部 PASS |
| IR-4 | Helper-level 測試通過 | §11.2（C1）或 §11.3（C2）的全部項目 PASS |
| IR-5 | Smoke harness 測試通過 | §11.4 的 7 項 I/O 測試全部 PASS |
| IR-6 | 可重現性測試通過 | §11.5 的 2 項端對端回歸測試全部 PASS |
| IR-7 | 分層不變條件檢查 | `evolution/local_graph.py` 與 `evolution/local_pairwise.py`（或 `local_replicator.py`）不做 I/O、不 import plotting；`simulation/c1_local_pairwise.py` 不 import `analysis/` |

### 15.2 Short Scout Verdict Matrix

21-run scout 結束後，根據以下矩陣判讀整體 verdict：

| level3_seed_count (best cell) | stage3_uplift (best cell) | mechanism signal | verdict | 後續動作 |
|---|---|---|---|---|
| >= 1 seed | >= 0.02 | - | **pass** | 升格為主線，開 longer confirm（30 seeds） |
| >= 1 seed | >= 0.02 | - 但 `mean_env_gamma` 超出 $\pm 5 \times 10^{-4}$ | **weak_positive** | 開 targeted follow-up（最多 9 runs） |
| 0 | >= 0.02 | 強（見 §5.5/§6.5 閾值表） | **weak_positive** | 開 targeted follow-up（最多 9 runs） |
| 0 | < 0.02 | 強 | **weak_positive_marginal** | 僅記錄，不追加 runs，但在 C2 開始時參考 |
| 0 | < 0.02 | 弱或未形成 | **fail → closure** | 記為 `close_c1`/`close_c2`，寫入研發日誌 |

### 15.3 Longer Confirm Gate（僅在 pass → longer confirm 時適用）

若 short scout verdict = `pass`，開 30-seed longer confirm：

1. `seeds = 0:29`（與 SDD §4.6 protocol lock 一致）
2. **通過標準**：`>= 10/30 seeds` 達到 Level 3 且 `mean_env_gamma ∈ [-3e-4, +3e-4]`
3. **失敗**：`< 5/30 seeds` 達到 Level 3 → 降級為 `weak_positive`，記錄但不再追加
4. **邊界**：`5-9/30 seeds` → 人工審核 simplex 圖與 diagnostics，決定是否開第二輪

### 15.4 C-family 整體 Closure 條件

以下任一滿足即視為 C-family 整體 closure：

1. C1 hard fail **且** C2 hard fail（兩者的 hard stop 條件均滿足）
2. C1 pass 但 longer confirm fail **且** C2 也未能超過 C1 的 short scout 指標
3. 預算上限：C1 + C2 合計不超過 `42 + 18 = 60 runs`（21 base + 9 follow-up 每 family 各一次）加 longer confirm（若觸發）的 30 runs

### 15.5 Decision 文件歸檔規則

1. 每個 family 的 decision.md 必須在結束後 24h 內同步摘要至研發日誌
2. 研發日誌條目格式：`close_c1` 或 `pass_c1 → longer_confirm` + 一句話結論 + 指向 decision.md 的相對路徑
3. 若觸發 `weak_positive` follow-up，follow-up 的結果 append 在同一份 decision.md 末尾，不另開新檔

---

## 16. 一句話收束

C-family 的目的不是再替 existing operator 加一點局部補丁，而是第一次正面測試：**只要不再把每輪局部訊號重新壓回單一 global sampled update，Level 3 是否就有機會從 plateau 中存活下來。**