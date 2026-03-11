# SDD（Spec-Driven Development）—「個性地下城」研究型開發方法

本文件目標：把「研究問題」改寫成可驗證的規格（Spec），再以最小改動落到程式碼與輸出資料契約（CSV），讓研究迭代可重現、可回歸、可比較。

> 研究導向的 SDD 不是「先把產品做完再分析」，而是每一個假設都要有：**形式化定義 → 可觀測輸出 → 驗證指標 → 邊界條件 → 回歸測試**。

---

## 0) 本 repo 的分層（當作 SDD 的約束）

- `core/`：單回合博弈/引擎（不處理長期演化、不做 I/O）
- `dungeon/`：地下城 payoff 規則（leader），可切換 payoff mode
- `players/`：玩家抽樣策略 + 人格模型（之後擴充 12 維 personality vector）
- `evolution/`：長期動態（replicator dynamics），只做純計算
- `simulation/`：把以上組起來跑、輸出 CSV
- `analysis/`：純分析（不反向依賴 simulation；避免循環依賴）

SDD 的第一條：任何新功能都要先說清楚「落在哪一層」與「不該落在哪一層」。

---

## 1) 研究目標（Spec 的最高層）

### 1.1 研究主目標（方案 B）

- **目標 B：內部極限環／循環動態（meta-cycle）**
- 以最小結構實作 non-potential 的旋轉結構（例如 `matrix_ab`：$U = A x$）
- 在離散時間 + 有限族群抽樣下，仍能觀測到**結構性振盪**（非純噪音）

### 1.2 非目標（避免範圍膨脹）

- 不是要先做完整 UI / 內容型地下城關卡
- 不是要一次做完 12 維人格 + 事件系統 + Boss stackelberg
- 先把「循環動態」的最小可重現管線做穩：`simulation -> CSV -> analysis`

### 1.3 Spec 版本與變更規則（研究型 SDD 的「鎖死點」）

- 本文件（`SDD.md`）是**唯一**正式 Spec（source of truth）。
- 任何下列變更都視為「契約變更」，必須先改 Spec 再改碼，並補回歸測試：
  - CSV 欄位/意義
  - payoff 定義（含 time index）
  - 演化更新規則（權重定義/正規化方式）
  - 不變條件（invariants）

---

## 2) 數學規格（可執行的形式化定義）

本節是「程式行為應該等價於的數學定義」。

### 2.1 策略比例向量（simplex 狀態）

- 策略空間固定為三策略（MVP）：
  - `aggressive`, `defensive`, `balanced`
- 每一輪的族群比例：
  - $x(t) = (x_1(t), x_2(t), x_3(t))$
  - $x_i(t) \ge 0$，且 $\sum_i x_i(t) = 1$
- 在程式中，$x(t)$ 由上一輪選擇的策略直方圖正規化得到（`dungeon.DungeonAI.popularity`）。

### 2.2 Payoff 規格（兩種 mode）

#### (A) `payoff_mode = count_cycle`（計數版）

- 基本懲罰：
  - $r(s) = base\_reward - \gamma \cdot n_s$
- 可選循環交叉項：
  - $r(s) = base\_reward - \gamma \cdot n_s + \epsilon \cdot (n_{prev(s)} - n_{next(s)})$
- $n_*$ 取自**上一輪**人氣統計。

#### (B) `payoff_mode = matrix_ab`（理論版，研究主線）

- 定義策略順序為 `strategy_cycle = [aggressive, defensive, balanced]`。
- 上一輪比例向量：
  - $x = (x_A, x_D, x_B)$
- payoff：

$$
A = \begin{pmatrix}
0 & a & -b\\
-b & 0 & a\\
a & -b & 0
\end{pmatrix},\quad U = A x
$$

- 對策略 $i$ 的 reward：
  - $r_i = base\_reward + U_i$

> Spec 的關鍵：`matrix_ab` 必須使用「上一輪」的 $x$（而不是本輪剛抽樣的策略分布），否則時間索引會錯位。

### 2.3 演化更新（離散近似 replicator）

本 repo 的演化更新（權重更新）分成兩種模式（對應 CLI `--evolution-mode`）：

#### (A) `evolution_mode = sampled`（既有行為，有限族群抽樣）

- 由玩家本輪 `last_reward` 聚合成每個策略的平均 reward，並用

$$
\text{weight}(s) \propto \exp(k\,(\bar u_s - \bar u))
$$

其中 $k =$ `selection_strength`。

#### (B) `evolution_mode = mean_field`（研究用：確定性 mean-field）

- 目的：在不依賴有限族群抽樣噪音的情況下，產生可控、可重現的相位旋轉。
- 限制：目前僅定義在 `payoff_mode = matrix_ab`。
- 定義：令 $x(t)$ 為三策略 simplex 上的比例（由當下權重正規化得到）。
  - $x(t) = \text{Normalize}(w(t))$（把權重當作比例 mass 正規化到和為 1）
  - payoff 向量 $u(t)$：

$$
u(t) = A\,x(t-\ell),\quad \ell\in\{0,1\}\;\;\text{(對應 CLI `--payoff-lag`)}
$$

- replicator 映射（與本 repo 既有離散近似一致，採用乘法權重 + 指數成長）：

$$
w(t+1) \propto w(t)\odot \exp\big(k\,(u(t) - \bar u(t))\big),\quad \bar u(t)=\sum_i x_i(t)u_i(t)
$$

- 權重規格化同樣維持 `mean(weight)=1`。

- 權重規格：
  - 對每個策略 $s$：`weight(s) > 0`
  - 規格化：平均權重 `mean(weight) = 1`（方便比較與避免爆炸）

### 2.4 時間索引（非常重要：避免「看起來對」但其實錯位）

本 repo 的一輪更新（round $t$）在概念上是：

1. 玩家依當前權重抽樣本輪策略，得到本輪行為分布 $p(t)$。
2. 地下城對每位玩家計算 reward 時，使用的是**上一輪**人氣統計所導出的比例 $x(t-1)$：
   - 第 0 輪（$t=0$）因為沒有上一輪，視為 popularity 空集合，故 $x(-1)=(0,0,0)$。
3. 回合結束時才把本輪的 chosen strategies 寫回 popularity，供下一輪使用。
4. 演化更新（`replicator_step`）聚合本輪 `last_reward`，產生下一輪權重 $w(t+1)$。

因此在輸出資料上，`p_<strategy>` 記的是 $p(t)$，但 payoff 的 $x$ 其實是 $x(t-1)$。

補充（`evolution_mode = mean_field`）：

- 這個模式不依賴玩家抽樣來估計 per-strategy reward；它直接用 $u=A x$ 計算期望 payoff。
- `--payoff-lag` 明確控制 payoff 使用的時間索引：
  - `--payoff-lag 0`：$u(t)=A x(t)$（無 lag）
  - `--payoff-lag 1`：$u(t)=A x(t-1)$（一階 lag；對應我們在 Jacobian/ρ 分析用的 lagged 模型）

---

## 3) 不變條件（Invariants）

這些是「不管你怎麼掃參數，都必須成立」的規格。任何違反都視為 bug。

### 3.1 數值不變條件

- `strategy_distribution` 回傳必須滿足：
  - $0 \le p_s \le 1$
  - 若有效樣本數 $n>0$，則 $\sum_s p_s = 1$；若 $n=0$，則回傳全 0（此時總和為 0）
- `replicator_step` 回傳：
  - 所有權重為有限正數（非 `nan`/`inf`）
  - `mean(weight)=1`（允許 $10^{-9}$ 量級誤差）

### 3.2 分層不變條件（架構約束）

- `analysis/` 不能 import `simulation/`（避免循環依賴）
- `evolution/` 不做 I/O、不依賴 plotting
- `simulation/` 才負責 CSV schema（資料契約）

### 3.3 資料契約（CSV schema）

`simulation/run_simulation.py` 輸出的 timeseries CSV 至少包含：

- `round`
- `avg_reward`, `avg_utility`
- `p_<strategy>`：每輪策略比例
- `w_<strategy>`：每輪更新後的抽樣權重

SDD 的要求：
- 欄位名是契約（contract），更動必須在 Spec 先更新，並加回歸測試。

補充：CSV 內 `round=t` 的一列，代表「完成第 t 輪 step 後」的觀測值（此時玩家的 `last_*` 已更新）。

### 3.4 指標語意（避免 reward/utility 混用）

- `reward`：單回合即時回饋（每位玩家在該輪的 `last_reward`）。
- `utility`：累積效用（每位玩家的 `utility`，在每輪把 reward 累加）。

因此：
- `avg_reward` 是「該輪」族群平均 reward（若無玩家則空值）。
- `avg_utility` 是「到該輪為止」族群平均累積 utility。

---

## 4) 驗證標準（Verification / Definition of Done）

把「看起來有在震盪」改成可檢驗的指標。

### 4.1 最低驗收（MVP 研究版）

- 能用單一指令跑出 CSV：
  - `./venv/bin/python -m simulation.run_simulation --out outputs/timeseries.csv`
- 能切到理論 payoff：
  - `--payoff-mode matrix_ab --a <a> --b <b>`
- **可重現性（當指定 seed）**：
  - 同一組參數 + 同一個 `--seed` 重跑，輸出 CSV 應一致（做回歸比較用）
- analysis 端至少能：
  - 讀 CSV
  - 做最基本的摘要（末期比例、均值）

### 4.1.1 參數介面（以現行實作為準）

`simulation/run_simulation.py` 目前支援的 CLI 參數與預設：

- `--players`（預設 50）
- `--rounds`（預設 20）
- `--seed`（預設不指定；指定後使抽樣可重現）
- `--payoff-mode`（預設 `count_cycle`；可選 `matrix_ab`）
- `--popularity-mode`（預設 `sampled`；可選 `expected`，用權重期望值更新 popularity 以降噪）
- `--evolution-mode`（預設 `sampled`；可選 `mean_field`，用期望 payoff 的確定性更新以產生乾淨旋轉；目前僅支援 `matrix_ab`）
- `--payoff-lag`（預設 1；只在 `--evolution-mode mean_field` 使用，可選 0/1）
- `--gamma`（預設 0.1；`count_cycle` 使用）
- `--epsilon`（預設 0.0；`count_cycle` 使用）
- `--a`（預設 0.0；`matrix_ab` 使用）
- `--b`（預設 0.0；`matrix_ab` 使用）
- `--init-bias`（預設 0.0；初始對稱破缺，用來讓 deterministic/expected 動態離開均勻固定點；要求 $|bias|<1$）
- `--selection-strength`（預設 0.05）
- `--out`（預設 `outputs/timeseries.csv`）

### 4.2 循環動態的可觀測驗收（建議指標）

你已決定採用「兩階段」驗收策略（先振幅、再主頻），相位方向留到發表級：

- **第一階段（MVP / sanity check）**：振幅下限
- **第二階段（確認週期性）**：主頻存在
- **發表級（結構性循環）**：相位循環方向一致

指標列表（可逐步加嚴，不建議一開始就全開）：

- **振幅下限**：在 burn-in 之後，時間序列的峰谷差（peak-to-peak）> 閾值
- **主頻存在**：對時間序列做簡單 FFT 或自相關，存在顯著非零主頻
- **相位循環方向**：在 $(p_A, p_D, p_B)$ 的相圖投影中，繞行方向一致（例如 A→D→B→A）

（研究敘事加嚴：用來回答「為什麼 Stage3 常/不常成立」）

- **包絡衰減率（empirical envelope slope）**：估計振幅包絡是否衰減
  - 2D 投影：例如 $(u=p_A, v=p_D)$；中心 $c$ 為
    - `series=p`：$c=(1/3,1/3)$
    - `series=w`（normalize 後的權重 simplex）：$c=(1,1)$
  - 振幅代理：$A(t)=\max(|u_t-c_u|,|v_t-c_v|)$（L∞），抽取局部峰值 $A(t_k)$
  - 回歸：$\log A(t_k) = \beta + \gamma_{env} t_k$
  - 解讀：
    - $\gamma_{env}<0$：包絡衰減（stable focus / damped spiral）
    - $\gamma_{env}\approx 0$：近中性（更可能維持旋轉，使 Stage3 比例上升）
    - $\gamma_{env}>0$：包絡增長（通常會被非線性飽和成有限振盪）

命名注意：本 repo 的 payoff 參數也有 `--gamma`（`count_cycle` 懲罰強度）。為避免混淆，工具/報表使用 `env_gamma` / `mean_env_gamma` 表示包絡衰減率。

> 實務上，有限族群抽樣會讓 $p_s(t)$（策略比例）出現「看似週期」的偶然自相關。
> 因此回歸測試建議優先對 `w_<strategy>`（抽樣權重）做 Stage 1/2：
> - 在控制組（例如 `matrix_ab` 且 a=b=0）時，`w_*` 應為常數 1，能穩定排除假陽性。
> - 在循環組時，`w_*` 應呈現可觀測振幅與主頻。
> 指標函數本身仍是一般 time series 指標，可對 `p_*` 或 `w_*` 使用，但需要在 Spec/報告中明確記錄選用哪一組欄位。

Stage 3（發表級）在本 repo 的操作性定義：
- 先把 `w_<strategy>(t)` 正規化成 simplex 軌跡 $x(t)=w(t)/\sum_i w_i(t)$
- 固定策略順序為 `(aggressive, defensive, balanced)`
- 把 $x(t)$ 投影到 2D simplex（等邊三角形嵌入）後，對相鄰兩段位移向量計算局部轉向符號：
  - $\Delta p_t = p_{t+1}-p_t$
  - $\text{turn}_t = \Delta p_t \times \Delta p_{t+1}$（2D 叉積）
- 通過條件：
  - $|\sum_t \text{turn}_t|$ 夠大（避免只有抖動）
  - 且大多數 `turn_t` 的符號與總和符號一致（方向一致）

建議把「旋轉強度」也寫成可調門檻：

- 定義平均轉向強度：$\overline{|\text{turn}|} = \frac{1}{N}\sum_t |\text{turn}_t|$
- 方向一致性分數：$\text{score} = |\text{mean}(\text{sign}(\text{turn}_t))|$
- 通過條件：
  - $\text{score} \ge \eta$
  - 且 $\overline{|\text{turn}|} \ge \delta$

註：方向符號（`ccw/cw`）取決於策略順序與投影定義；因此 Spec 必須鎖死順序，否則「方向翻轉」會被誤判成 bug 或假陽性。

#### 建議的預設（可在研究中調整，但需記錄）

- burn-in：建議先用 `burn_in = rounds // 3`
- tail：若 rounds 很長，可只看最後一段（例如 `tail = 2000`）
- 主頻（自相關版本）搜尋範圍：`min_lag=2`、`max_lag=500`
- 主頻顯著門檻：先用相關係數 `corr_threshold=0.3`，之後可依噪音水平加嚴

### 4.3 正式分級判定（建議採用：cycle_level = 0/1/2/3）

為了讓研究敘事更嚴謹、也更利於 sweep 與回歸，本專案建議用分級框架而非單一布林：

- **Level 0：None（近似收斂/無顯著動態）**
  - 振幅不達標
- **Level 1：Oscillation（震盪存在）**
  - 條件：振幅下限達標（排除收斂到固定點）
- **Level 2：Periodic Oscillation（週期震盪）**
  - 條件：Level 1 + 主頻存在（排除純噪音大幅波動）
- **Level 3：Structured Cycle（結構性循環）**
  - 條件：Level 2 + 相位方向一致（具有方向性的支配循環）

此分級能描述中間狀態，例如「Level 2 週期震盪，但尚未達到結構性循環標準」。

#### 重要規則：同一時間窗（burn-in / tail）

三個指標必須在**相同 burn-in 與 tail 區間**上計算。

不得：
- 振幅用全資料
- 主頻只用後半段
- 相位用另一段

否則會造成統計不一致與誤判。

#### 實作對應（以 `analysis/` 純函數為準）

- Level 1：`analysis.cycle_metrics.assess_stage1_amplitude(...)`
- Level 2：`analysis.cycle_metrics.assess_stage2_frequency(...)`
- Level 3（預設 Stage3_v2 / turning-based）：`analysis.cycle_metrics.phase_direction_consistency_turning(...)`
  - 可用 `stage3_method="centroid"` 切回 `analysis.cycle_metrics.phase_direction_consistency(...)`（較敏感、較容易被噪音翻號）
  - 可用 `phase_smoothing=3/5/7...` 對 Stage3 輸入做移動平均降噪（turning-based 時適用；需為奇數）
- 分級輸出：`analysis.cycle_metrics.classify_cycle_level(...)`

加值（研究用，不改變 cycle_level 判定規則）：

- `env_gamma`（包絡衰減率）計算：`analysis/decay_rate.py`
- 多 seed 彙整時的 `env_gamma` / `mean_env_gamma` 報表欄位：`simulation/seed_stability.py`

### 4.4 控制組定義（False Positive Guard；建議列為硬性驗收）

**Control Baseline Spec**（控制組）

- 定義：`payoff_mode=matrix_ab` 且 `a=0`、`b=0`
- 理由：此時 $A=0$，$U=Ax=0$，所有策略 reward 相同，演化更新應保持中性。

**驗收條件（用來保護指標不對噪音過度敏感）**

- 以 `w_<strategy>` 的 time series 做 Level 判定時：
  - `cycle_level` 必須 = 0
  - 振幅不得達標（Stage 1 fail）
  - 主頻不得達標（Stage 2 fail；在 Level 設計上通常會因 Stage 1 fail 而不再計算）

此條稱為 **False Positive Guard**；若控制組過不了，後續所有循環結論都不可信。

### 4.5 研究級穩健性需求（Robustness Requirements；建議用於發表）

這些不是「單次跑」的 unit test，而是 sweep/實驗設計層的驗收：

1) **多 seed 穩定性**（同一參數，隨機性不應主導結論）

- 固定同一組參數（例如某組 `matrix_ab (a,b)`）
- 跑 $N$ 個不同 `--seed`（建議 $N=10$）
- 驗收：$\Pr(\text{cycle\_level} \ge 2) \ge 0.7$（門檻可調，但需先寫進報告）

（建議同時回報 near-neutral 程度，避免「看起來是 Level 3 但其實包絡快速衰減」）

- 附加回報：`mean_env_gamma` 接近 0（例如 $|\gamma_{env}| \le \tau$；$\tau$ 依 rounds/噪音調整）
- 可靠度：同時檢視 `env_gamma_n_peaks` 與 `env_gamma_r2`（峰值太少或 $R^2$ 太低時，不做強結論）

2) **參數鄰域穩健性**（避免「只在單點成立」的數值偶然）

- 在小區間內掃描（例如 $a\in[a_0-\Delta,a_0+\Delta]$、$b\in[b_0-\Delta,b_0+\Delta]$，建議 $\Delta=0.1$ 或依尺度調整）
- 驗收：區間內 `cycle_level >= 2` 的比例 $\ge 0.7$

備註：上述比例門檻是「研究敘事強度」的槓桿；越嚴格越有說服力，但需要更多計算與更長 time series。

### 4.6 有限尺寸臨界分析（Finite-size critical transition；下一階段主線）

當我們已能穩定產生 `P(Level3)` 隨控制參數變化的轉換曲線後，研究主線應由「找出是否能循環」升級為：

- **有限族群（finite N）下的臨界轉換**：`P(Level3)` 在臨界附近由 0 快速上升
- **臨界點如何隨 N 改變（finite-size scaling）**：例如 $k_{50}(N)$ 對 $1/N$ 的依賴
- **真正控制量（order parameter）**：優先用 $\epsilon = \rho - 1$ 而不是直接用 $k$

#### 4.6.1 定義：臨界指標與轉換點

- 定義 $P_{L3}(k;N)=\Pr(\text{cycle\_level}=3)$（由多 seed sweep 估計）
- 定義 $k_{50}(N)$：滿足 $P_{L3}(k;N)=0.5$ 的最小 $k$（可在相鄰 k 格點間做線性插值）
- 同理可定義 $k_{90}(N)$：$P_{L3}(k;N)=0.9$

#### 4.6.2 實驗輸入/輸出契約

輸入（由 `simulation/rho_curve.py` 產生）：

- 每個 (N,k) 一列，至少包含：`players, selection_strength, rho_minus_1, p_level_3, mean_env_gamma, n_seeds`

輸出（由 `analysis/rho_curve_scaling.py` 產生）：

- `*_summary.csv`：每個 N 的 `k50/k90/peak/plateau` 彙整
- 圖（建議至少三張）：
  - `k50 vs 1/N`
  - `P(Level3) vs (rho-1)`（curve collapse by N）
  - `mean_env_gamma vs (rho-1)`（檢查 $\gamma_{env}\approx 0$ 是否對應轉換）

#### 4.6.3 可重現命令（範例）

（1）先跑 sweep（例：主 sweep + N=200 延伸段）

- `./venv/bin/python -m simulation.rho_curve ... --out outputs/sweeps/rho_curve/<main>.csv`
- `./venv/bin/python -m simulation.rho_curve ... --players 200 --k-grid 1.0:1.4:0.02 --out outputs/sweeps/rho_curve/<n200_ext>.csv`

（2）做 finite-size scaling / collapse 分析

- `./venv/bin/python -m analysis.rho_curve_scaling \
  --in outputs/sweeps/rho_curve/<main>.csv \
  --in outputs/sweeps/rho_curve/<n200_ext>.csv \
  --outdir outputs/analysis/rho_curve \
  --prefix <tag>`

#### 4.6.4 本次結果的結論草稿（Discussion draft；以 order-parameter 觀點整理）

> 註：本小節屬於「論述草稿/示意」，數值可能隨 protocol lock 與資料更新而改變；最新的收斂結果與 per-N 評分請以 `研發日誌.md` 的「Rho-curve scaling：S120 iterative refinement 收斂」章節為準。

以下觀察以單一 payoff 組合（`matrix_ab`, `a=0.4`, `b≈0.2425406997`, `payoff_lag=1`）與固定判定門檻（例：`eta=0.55`）為前提，資料來自 `simulation/rho_curve.py` 的多 seed sweep，並由 `analysis/rho_curve_scaling.py` 彙整。

1) **以 $\epsilon=\rho-1$ 作為控制量，比直接用 $k$ 更乾淨**

- 在 `P(Level3) vs (rho-1)` 圖中，不同 N 的曲線形狀更一致；這支持「$\rho-1$ 才是有效控制量（order parameter / reduced control parameter）」的敘事，而 $k$ 只是間接旋鈕。

2) **有限尺寸轉換存在，且呈現明顯 finite-size / 非線性效應**

- 以 $k_{50}(N)$（$P(Level3)=0.5$）作為臨界點 proxy：
  - N=50：$k_{50}\approx 0.88$；峰值 $P_{max}\approx 0.77$
  - N=100：$k_{50}\approx 0.84$；且 $k_{90}\approx 1.00$、峰值 $P_{max}=1.0$
  - N=200：$k_{50}\approx 1.04$；峰值 $P_{max}\approx 0.83$（並出現平台區間）
  - N=1000：$k_{50}\approx 0.82$；且 $k_{90}\approx 0.96$、峰值 $P_{max}\approx 0.97$
  - N=2000：$k_{50}\approx 0.752$；且 $k_{90}\approx 0.82$、峰值 $P_{max}=1.0$
  - N=300 與 N=500：在本次「臨界帶」掃描的下界 $k_{min}=0.7$ 就已觀察到 $P(Level3)>0.5$，因此只能得到 $k_{50}\le 0.7$ 的**左截斷**上界；若要做更乾淨的 $k_{50}(N)$ 擬合，需把這兩個 N 的掃描範圍往更小 $k$ 延伸。
- 重要訊號：$k_{50}(N)$ **非單調**，且用簡單形式 $k_{50}(N)=k_\infty + c/N$ 的線性擬合解釋力很弱（在合併更多 N、並排除左截斷的 N=300/500 後，$R^2$ 仍很低）。這表示轉換位置受「有限尺寸噪音 × lag 非線性」的二階效應影響，不能用單調收斂故事簡化。

3) **Stage3 並非單純對應 $\rho>1$；而是 near-critical window 的機率現象**

- N=200 在較高 k 區間（例如 k≈1.34..1.40）呈現 `P(Level3)` 平台（約 0.83），顯示即使 $\rho-1$ 持續增大，`P(Level3)` 也不一定繼續上升。
- 因此 Stage3 成立機率更合理的敘事是：
  - 系統需落在「接近臨界（$\rho\approx 1$）且包絡近中性」的窗口
  - 有限抽樣噪音決定是否能在有限時間窗內維持方向一致性（Level3）

4) **$\gamma_{env}\approx 0$ 與轉換區間有一致性（但需搭配可靠度欄位解讀）**

- 在 $k_{50}$ 附近，`mean_env_gamma` 的量級普遍非常接近 0（本次資料在 $10^{-4}$ 量級內），符合「包絡近中性」的預期。
- 注意：在低振幅/低週期性區域，`env_gamma` 估計本身可能不穩（例如峰值數量不足或 $R^2$ 太低）。因此正式論述應搭配 `mean_env_gamma_r2` 與 `mean_env_gamma_n_peaks` 一起檢查，避免把估計噪音誤當成物理訊號。

結論（可用於段落收束）：
- 我們已經從「找會不會循環」轉向「有限尺寸臨界轉換」：`P(Level3)` 的上升更像是 near-critical（$\rho\approx 1$）窗口中的有限時間/有限樣本事件。
- 下一步應優先做：新增更多 N 的臨界帶掃描、在 $\epsilon=\rho-1$ 空間中檢查曲線 collapse、並用 $\gamma_{env}$（附帶可靠度）建立 envelope 臨界敘事。

#### 4.6.5 下一輪研究方向（Refinement-first；先把統計品質做乾淨）

本 repo 的 pipeline（sweep → merge → scaling → report）已足以支援研究級迭代；下一輪的瓶頸不在工具，而在 **統計設計**。
因此本階段的 Spec 重點是：**不要再擴大 sweep 範圍**，而要在 crossing 附近「加密 + 提高 seeds + 統一 protocol」，以降低 $k_{50}(N)$ 的隨機漂移與 segment heterogeneity。

**核心策略**

1) **Protocol lock（硬性要求）**：未來新增的任何 sweep segment，應盡可能固定下列要素一致：
  - `rounds`, `burn-in-frac` / `burn-in`, `tail`
  - `seeds`（至少固定 seed 數量；更理想是固定 seeds 的集合範圍）
  - `series`（p 或 w）
  - Stage1/2/3 的門檻（特別是 `corr_threshold` 與 `eta`）
  - `k-grid` 的步長（crossing 區要一致，避免解析度差異造成 crossing 偏移）

2) **Crossing zone densification（主要投入）**：對每個目標 $N$，以目前估到的 $k_{50,\mathrm{est}}(N)$ 為中心，只掃一個小 band（例如 $k_{50,\mathrm{est}}\pm 0.05$），並把解析度提高到 $\Delta k=0.005$。

3) **Seeds uplift（臨界帶加強）**：在 crossing band 內，建議 `seeds >= 30`（優先）以壓低 $P_{L3}$ 的 Bernoulli sampling noise（$\mathrm{sd}\approx\sqrt{p(1-p)/n}$）。

4) **Lowband 保持粗即可（定位用）**：low-k band 的目標主要是確認 $P_{L3}\approx 0$ 與解除 left-censoring；其 k-grid 不需要很細，但 protocol 仍需盡量一致（避免引入額外 heterogeneity）。

**建議的下一輪最小實驗設計（MVP for refinement）**

- `players-grid`: `100,200,300,500,1000`
- 對每個 $N$：`k-grid = (k50_est-0.05):(k50_est+0.05):0.005`（約 21 個 k 點）
- `seeds`: `0:29`（30 seeds）
- `rounds`: 建議先固定為 `4000`
- `series`: 建議與既有主線一致（目前多用 `series=p` 來定義 $P_{L3}$）
- `corr_threshold`: 建議鎖定在已校準的工作點（例如 `0.09`）

> 備註：band 的中心 $k50_{est}$ 來自上一輪 summary；因此本策略天然是「迭代式實驗設計」：先粗估 → 再在 crossing 附近加密。

實務面的「runbook、風險控管、檔案/迭代管理、自動選點與 refine 前後量化比較」請以 `研發日誌.md` 的 8.2/8.2.1 為準（Spec 只定義契約與一致性要求）。

#### 4.6.6 Crossing 估計的穩健性要求（避免插值噪音主導）

線性插值在 $\Delta k$ 粗、或 $P_{L3}(k;N)$ 非單調（噪音）時會明顯不穩；因此本階段建議至少滿足：

- **解析度要求**：crossing band 內 $\Delta k\le 0.005$（建議值；可依計算量調整）。
- **不確定性回報**（建議列為研究級輸出）：對每個 $N$ 回報 $k_{50}$ 的估計誤差（例如 bootstrap CI），避免只用點估計做 scaling fit。
- **單調性處理**（選配）：若單調性違反很嚴重，可在不改變資料契約前提下，在 analysis 端引入單調回歸/平滑（例如 isotonic regression 或 logistic fit）作為 crossing 的輔助估計，並同時回報「原始插值 vs 平滑估計」差異作為診斷。

#### 4.6.7 Scaling 模型比較（不預設 $1/N$ 一定正確）

在 stochastic finite-size 系統中，$k_{50}(N)$ 未必呈 $1/N$ 修正。除了基準模型外，建議同時比較：

- Model A（基準）：$k_{50}(N)=k_\infty + c/N$
- Model B（一般化）：$k_{50}(N)=k_\infty + c/N^{\beta}$（$\beta$ 由資料估計；不強制為 1）
- Model C（次高階修正）：$k_{50}(N)=k_\infty + c/N + d/N^2$

建議用 AIC/BIC（或交叉驗證）做模型選擇，並在有 $k_{50}$ 不確定性時用 weighted fit。

#### 4.6.8 主推論的優先順序（論文敘事建議）

在 4.6 階段，最具說服力的推論通常不是 $k_{50}$ 是否單調，而是：

1) **$P_{L3}$ 對 $\epsilon=\rho-1$ 的 curve collapse 是否成立**（支持「真正控制量是 $\rho-1$」的敘事）。
2) **$\gamma_{env}\approx 0$ 的 near-neutral band 是否與轉換區共定位**（支持 near-critical window 的敘事）。
3) **在統一 protocol + crossing band 加密後**，$k_{50}(N)$ 的非單調與低 $R^2$ 是否顯著改善；若改善，則先前現象多半是噪音/heterogeneity artifact。

#### 4.6.9 Refinement sweep：可複製命令模板（crossing band 加密；protocol 鎖死）

本節提供下一輪 refinement sweep 的「直接可跑」模板。

**鎖死 protocol（建議值）**

- `rounds=4000`
- `burn-in-frac=0.30`（等價 burn-in=1200）
- `tail=1000`
- `seeds=0:29`（30 seeds）
- crossing band：$k_{50,\mathrm{est}}\pm 0.05$
- crossing 解析度：$\Delta k=0.005$

**重要限制（目前工具行為）**

`simulation.rho_curve` 的 `--k-grid` 目前是「同一條 k-grid 套用到所有 players」；因此若每個 $N$ 都要用不同的 crossing band，建議以迴圈逐個 $N$ 跑（每個 $N$ 一個輸出 CSV）。

#### 4.6.10 收斂後（golden prefix）的診斷流程（Post-convergence diagnostics）

當 refinement 已達到「前後差分為 0」的收斂判定後，下一步不再追加 sweep，而是把資料集凍結為 **golden prefix**，並以「只讀 CSV 的 diagnostics」產生可回歸的圖與彙整表，用來支撐後續的物理解讀與模型比較。

**收斂判定（建議）**

- 使用 `analysis.validate_scaling` 比較前後兩個 prefix：
  - fit（例如 $k_\infty,c,R^2$）差分為 0
  - per-N $k_{50}$ 差分為 0

**必做 diagnostics（不跑 simulation；只讀已產生的 CSV）**

- 以 `analysis.rho_curve_viz` 產生：
  - `P(Level3) vs k` 疊圖（含 zoom）
  - `P(Level3)` 的 (N,k) heatmap（檢視 dip/局部非單調是否集中在少數 N）
  - $k_{50}$ 對多種座標變換（例如 $1/\sqrt{N}$、$\log N$）與 residual plot
  - collapse 試探（例如 $(k-k_\infty)N^\beta$ 的多組 \beta）
  - 產出 `*_diagnostics.csv`（彙整每個 N 的 k50、summary 分數與 sweeps 單調性指標）

命令模板（範例）：

```bash
PREFIX=<golden_prefix>
./venv/bin/python -m analysis.rho_curve_viz \
  --prefix "$PREFIX" \
  --summary outputs/analysis/rho_curve/${PREFIX}_summary.csv \
  --fit-json outputs/analysis/rho_curve/${PREFIX}_k50_fit.json \
  --compose outputs/_compose_cmds_${PREFIX}.sh \
  --outdir outputs/analysis/rho_curve \
  --eps 0.02 \
  --betas 0.5,0.8,1.0,1.5
```

**選配：不確定性（只重跑 analysis，不重跑 simulation）**

- 若需要把 $k_{50}$ 的不確定性納入 scaling/論述：可用 `analysis.rho_curve_scaling` 的 Bayesian k50（例如 logit-Laplace）輸出 credible interval。
- 為避免覆寫 golden prefix 的既有 `*_summary.csv`，建議用新的 `--prefix <golden_prefix>_bayesdiag` 產生 analysis-only 的結果檔。

**做法：用上一輪 summary 的 $k_{50}$ 當作 $k50_{est}$，逐 N 跑 crossing band**

（下例的中心值可直接替換成最新 summary 估計；建議把中心值先 round 到 0.005 的格點上，確保 band endpoints 落在同一解析度網格。）

備註：以下指令使用 bash 的 associative array / process substitution；請用 `bash` 執行（而非 `sh`）。另外建議使用 `./venv/bin/python`（專案 venv；避免環境不一致）。

```bash
# Refinement crossing bands (example centers from the latest merged summary; adjust as needed)
declare -A K50_CENTER=(
  [100]=0.840
  [200]=1.040
  [300]=0.340
  [500]=0.295
  [1000]=0.820
)

ROUNDS=4000
BURN_FRAC=0.30
TAIL=1000
SEEDS="0:29"
STEP=0.005
BAND=0.05

for N in 100 200 300 500 1000; do
  C=${K50_CENTER[$N]}
  KMIN=$(./venv/bin/python - <<PY
c=float("$C")
print(f"{c-$BAND:.3f}")
PY
)
  KMAX=$(./venv/bin/python - <<PY
c=float("$C")
print(f"{c+$BAND:.3f}")
PY
)
  OUT="outputs/sweeps/rho_curve/rho_curve_refine_crossing_a0p4_b0p2425407_lag1_eta055_N${N}_k${KMIN}_${KMAX}_s${STEP}_R${ROUNDS}_S30_tail${TAIL}.csv"

  ./venv/bin/python -m simulation.rho_curve \
    --payoff-mode matrix_ab --a 0.4 --b 0.2425406997 \
    --gamma 0.1 --epsilon 0.0 \
    --popularity-mode sampled --evolution-mode sampled --payoff-lag 1 \
    --players-grid "${N}" \
    --rounds "${ROUNDS}" --burn-in-frac "${BURN_FRAC}" --tail "${TAIL}" \
    --seeds "${SEEDS}" --jobs 0 \
    --series p \
    --k-grid "${KMIN}:${KMAX}:${STEP}" \
    --amplitude-threshold 0.02 --min-lag 2 --max-lag 500 --corr-threshold 0.09 \
    --eta 0.55 --stage3-method turning --phase-smoothing 1 \
    --resume \
    --out "${OUT}"
done
```

**可選：自動從 summary.csv 讀取 $k_{50}$ 作為中心（免手改 K50_CENTER）**

若你已經有最新的 `*_summary.csv`，可以用下面版本直接生成每個 $N$ 的中心值，並自動 round 到 `STEP=0.005` 的格點：

```bash
SUMMARY="outputs/analysis/rho_curve/rho_curve_merged_with_lowband_clean_summary.csv"
ROUNDS=4000
BURN_FRAC=0.30
TAIL=1000
SEEDS="0:29"
STEP=0.005
BAND=0.05

./venv/bin/python - <<PY | while read -r N KMIN KMAX; do
import csv

summary_path = "${SUMMARY}"
targets = [100, 200, 300, 500, 1000]
step = float("${STEP}")
band = float("${BAND}")

data = {}
with open(summary_path, newline="") as f:
  for row in csv.DictReader(f):
    try:
      n = int(float(row.get("players") or ""))
    except Exception:
      continue
    k50 = row.get("k50")
    if not k50:
      continue
    try:
      data[n] = float(k50)
    except Exception:
      pass

missing = [n for n in targets if n not in data]
if missing:
  raise SystemExit(f"Missing k50 for N={missing} in {summary_path}")

for n in targets:
  c = data[n]
  c = round(c / step) * step
  kmin = c - band
  kmax = c + band
  print(n, f"{kmin:.3f}", f"{kmax:.3f}")
PY
  OUT="outputs/sweeps/rho_curve/rho_curve_refine_crossing_a0p4_b0p2425407_lag1_eta055_N${N}_k${KMIN}_${KMAX}_s${STEP}_R${ROUNDS}_S30_tail${TAIL}.csv"

  ./venv/bin/python -m simulation.rho_curve \
  --payoff-mode matrix_ab --a 0.4 --b 0.2425406997 \
  --gamma 0.1 --epsilon 0.0 \
  --popularity-mode sampled --evolution-mode sampled --payoff-lag 1 \
  --players-grid "${N}" \
  --rounds "${ROUNDS}" --burn-in-frac "${BURN_FRAC}" --tail "${TAIL}" \
  --seeds "${SEEDS}" --jobs 0 \
  --series p \
  --k-grid "${KMIN}:${KMAX}:${STEP}" \
  --amplitude-threshold 0.02 --min-lag 2 --max-lag 500 --corr-threshold 0.09 \
  --eta 0.55 --stage3-method turning --phase-smoothing 1 \
  --resume \
  --out "${OUT}"
done
```

**下一步（analysis 端）**

- 把這些 refinement CSV 與既有主 sweep/critical band/lowband 一起丟進 `analysis.rho_curve_scaling` 做重新彙整。
- 若要嚴格比較「refinement 前後」的 $k_{50}(N)$ 品質，建議固定只用 protocol 一致的 segments 做一次對照版 summary。

可複製命令模板（把 refinement 併入 scaling；並重生 notes/paper 報表）：

```bash
./venv/bin/python -m analysis.rho_curve_scaling \
  --in outputs/sweeps/rho_curve/rho_curve_a0p4_b0p2425407_lag1_sampled_eta055_R4000_S30_N50_200_1000_k0p1_1p0_s0p02.csv \
  --in outputs/sweeps/rho_curve/rho_curve_a0p4_b0p2425407_lag1_sampled_eta055_R4000_S30_N200_k1p0_1p4_s0p02.csv \
  --in outputs/sweeps/rho_curve/rho_curve_critical_band_a0p4_b0p2425407_lag1_eta055_R2500_S10_N100_300_500_2000_k0p7_1p2_s0p02.csv \
  --in outputs/sweeps/rho_curve/rho_curve_lowband_corr0p09_clean_N300_500.csv \
  --in "outputs/sweeps/rho_curve/rho_curve_refine_crossing_a0p4_b0p2425407_lag1_eta055_N100*_R4000_S30_tail1000.csv" \
  --in "outputs/sweeps/rho_curve/rho_curve_refine_crossing_a0p4_b0p2425407_lag1_eta055_N200*_R4000_S30_tail1000.csv" \
  --in "outputs/sweeps/rho_curve/rho_curve_refine_crossing_a0p4_b0p2425407_lag1_eta055_N300*_R4000_S30_tail1000.csv" \
  --in "outputs/sweeps/rho_curve/rho_curve_refine_crossing_a0p4_b0p2425407_lag1_eta055_N500*_R4000_S30_tail1000.csv" \
  --in "outputs/sweeps/rho_curve/rho_curve_refine_crossing_a0p4_b0p2425407_lag1_eta055_N1000*_R4000_S30_tail1000.csv" \
  --outdir outputs/analysis/rho_curve \
  --prefix rho_curve_merged_with_lowband_clean_plus_refine

./venv/bin/python -m analysis.rho_curve_report \
  --summary outputs/analysis/rho_curve/rho_curve_merged_with_lowband_clean_plus_refine_summary.csv \
  --fit-json outputs/analysis/rho_curve/rho_curve_merged_with_lowband_clean_plus_refine_k50_fit.json \
  --in outputs/sweeps/rho_curve/rho_curve_a0p4_b0p2425407_lag1_sampled_eta055_R4000_S30_N50_200_1000_k0p1_1p0_s0p02.csv \
  --in outputs/sweeps/rho_curve/rho_curve_a0p4_b0p2425407_lag1_sampled_eta055_R4000_S30_N200_k1p0_1p4_s0p02.csv \
  --in outputs/sweeps/rho_curve/rho_curve_critical_band_a0p4_b0p2425407_lag1_eta055_R2500_S10_N100_300_500_2000_k0p7_1p2_s0p02.csv \
  --in outputs/sweeps/rho_curve/rho_curve_lowband_corr0p09_clean_N300_500.csv \
  --in "outputs/sweeps/rho_curve/rho_curve_refine_crossing_a0p4_b0p2425407_lag1_eta055_N100*_R4000_S30_tail1000.csv" \
  --in "outputs/sweeps/rho_curve/rho_curve_refine_crossing_a0p4_b0p2425407_lag1_eta055_N200*_R4000_S30_tail1000.csv" \
  --in "outputs/sweeps/rho_curve/rho_curve_refine_crossing_a0p4_b0p2425407_lag1_eta055_N300*_R4000_S30_tail1000.csv" \
  --in "outputs/sweeps/rho_curve/rho_curve_refine_crossing_a0p4_b0p2425407_lag1_eta055_N500*_R4000_S30_tail1000.csv" \
  --in "outputs/sweeps/rho_curve/rho_curve_refine_crossing_a0p4_b0p2425407_lag1_eta055_N1000*_R4000_S30_tail1000.csv" \
  --out outputs/analysis/rho_curve/rho_curve_merged_with_lowband_clean_plus_refine_report.md \
  --out-paper outputs/analysis/rho_curve/rho_curve_merged_with_lowband_clean_plus_refine_paper.md
```

### 4.7 發表級加嚴待辦（Parking lot：先記著，等穩定區域找到再做）

以下是「很有價值但成本較高」的統計/理論加嚴項目。先把需求寫死在 Spec，避免忘記或事後補洞。

補全原則（本節一致性約束）：

- `control baseline` 固定指 `matrix_ab` 且 $a=b=0$（無 payoff；應接近無動態）。
- 所有加嚴項目都必須維持 False Positive Guard：控制組不應被誤判為有循環/主頻/方向一致。

1) **振幅門檻標準化（Amplitude normalization）**

> 研發結果（已完成；2026-03-07）：Stage1 amplitude normalization 已端到端落地（指標→control cache→sweep CLI→測試），並在 sweep CSV 中保留可回溯的 provenance 欄位。

- 實作：`analysis/cycle_metrics.py`（`assess_stage1_amplitude` / `classify_cycle_level`）
- control cache builder：`analysis/amplitude_control_cache.py`
- 一鍵 baseline pipeline：`simulation/control_amp_baseline.py`
- sweep CLI/輸出：`simulation/rho_curve.py`
- 測試：`tests/test_amplitude_normalization.py`、`tests/test_amplitude_control_cache.py`、`tests/test_rho_curve_control_cache.py`

- 問題：固定振幅閾值會受 payoff/尺度影響（不同 $a/b$ 下 baseline 噪音水平會變）。
- 建議：把 Stage1 的門檻改成「相對於 control 參考尺度」：
  - **control-scaled（mean）**：$A \ge \eta\,\mu_{control}$
  - **control-scaled（std）**：$A \ge \eta\,\sigma_{control}$

其中 $A$ 是同一個視窗（burn-in/tail）上的 peak-to-peak amplitude（Stage1 目前定義不變）。$\eta$ 為可調 factor（越大越嚴格）。

重要注意（可實作邊界條件）：
- 若你選用的 `series=w` 且 control baseline 是 `matrix_ab, a=b=0`，則 `w_*` 在本模型下通常會**完全常數 1**，導致 $\mu_{control}=\sigma_{control}=0$。
  - 因此：**control-scaled amplitude gate 對 `w` 會退化**（除以 0 / 門檻為 0）。MVP 規格要求：當 control reference $\le 0$ 時必須視為無效設定並 raise（避免悄悄把門檻變成 0）。
  - 實務建議：若要做 control-scaled amplitude，優先對 `series=p` 做（control 下 `p` 有抽樣噪音，reference > 0），或改採其他「噪音地板」定義（非本 MVP）。

**落地點（介面參數；可直接照做）**

- 指標計算：`analysis.cycle_metrics.assess_stage1_amplitude(...)`
  - 新增參數（保留預設行為不變）：
    - `normalization`: `"none" | "control_mean" | "control_std"`
      - `"none"`：沿用現行 absolute threshold（`threshold`）
      - 其餘：啟用 control-scaled gate
    - `control_reference: float | None`
      - 當 `normalization != "none"` 時必填，且必須 `> eps`
      - 定義：$\mu_{control}$ 或 $\sigma_{control}$（取決於 normalization）
    - `threshold_factor: float`（對應 $\eta$；建議預設 3.0 以保守維持 False Positive Guard）
    - `eps: float`（避免數值退化；MVP：若 `control_reference <= eps` 直接 raise）
  - 輸出補強（便於報表/回溯）：
    - `Stage1AmplitudeResult.threshold` 回報 **effective threshold（raw units）**
    - 另回報：`normalization / threshold_factor / control_reference / normalized_amplitudes`

- 分級判定：`analysis.cycle_metrics.classify_cycle_level(...)`
  - 新增參數（預設皆不啟用，維持既有研究結果可回歸）：
    - `amplitude_normalization`
    - `amplitude_control_reference`
    - `amplitude_threshold_factor`

- sweep 彙整（下一步，不阻塞 MVP 實作）：`simulation.rho_curve`
- sweep 彙整（已完成）：`simulation.rho_curve`
  - 已支援 `--amplitude-normalization/--amplitude-control-cache/--amplitude-control-reference`，並在 sweep CSV 中寫入 `amplitude_threshold_effective`、`mean_max_amp_norm`、`amplitude_control_reference`、`amplitude_control_cache` 等欄位。
  - cache 產生器（MVP 工具；從 control 的 per-seed CSV 建 JSON；已完成）：

```bash
./venv/bin/python -m analysis.amplitude_control_cache \
  --in outputs/control_amp_per_seed.csv \
  --out outputs/control_baseline_stats.json \
  --series p --burn-in 1200 --tail 600
```

（輸入 CSV 需包含至少 `players,max_amp`；若含 `selection_strength` 且有多個值，需額外帶 `--selection-strength <k>` 過濾。）

補充：若輸入是 per-seed report 且沒有 `players` 欄，也可用 `--in 300:outputs/n300_report.csv` 這種寫法在參數裡指定 players。

一鍵流程（推薦；自動產生 per-seed CSV + JSON cache；固定 control baseline= `matrix_ab, a=b=0`）：

```bash
./venv/bin/python -m simulation.control_amp_baseline \
  --players-grid 100,300,500 \
  --rounds 4000 --seeds 0:29 \
  --series p --burn-in-frac 0.25 --tail 600 \
  --selection-strength 0.02 \
  --out-per-seed outputs/control/control_amp_per_seed.csv \
  --out-json outputs/control/control_baseline_stats.json
```
- 補全/優化（可選）：
  - 以 config 形式集中門檻（例如新增 `rho_curve_config.yaml`；避免散落 hardcode）。
  - control stats 的估計建議以獨立 control sweep 先行預跑，並存成 cache（例如 `outputs/control_baseline_stats.json`），供主 sweep 讀取。
  - 需要時可改成 per-N baseline（避免 finite-size bias 讓 normalization 本身引入系統性偏移）。
- 風險與緩解：
  - 風險：control stats seeds 太少 → normalization 不穩。
  - 緩解：固定 protocol（seeds/rounds/tail）並提高 control seeds；同時在報表回報 control stats 的 CI/有效樣本。
- DoD（最小可驗收；工程可回歸）
  - `analysis.cycle_metrics.assess_stage1_amplitude(...)` 支援 `normalization != "none"`，且：
    - 有效門檻：`threshold == threshold_factor * control_reference`
    - 若 `control_reference <= eps`（或非有限），必須 raise（避免門檻退化成 0）
  - 新增單元測試覆蓋 control-scaled gate 的 pass/fail 與退化 guard。
  - False Positive Guard（控制組不誤判）必須仍成立：
    - `series=w` + `matrix_ab a=b=0` 仍必須 Stage1 fail（既有 absolute threshold 路徑）
    - 若啟用 control-scaled gate，必須使用有效 reference（不允許 0 reference 悄悄放水）

2) **主頻顯著性以比值/檢定取代固定門檻（Frequency significance）**

> 研發結果（已完成；2026-03-07）：Stage2 已支援 method 可切換，且 sweep→scaling summary→report 全鏈路保留 method/參數 provenance（含多輸入一致性檢查）。

- 實作：`analysis/cycle_metrics.py`（Stage2 methods + prefilter + `Stage2FrequencyResult`）
- sweep CLI/輸出：`simulation/rho_curve.py`
- summary/provenance：`analysis/rho_curve_scaling.py`
- report 摘要與一致性警告：`analysis/rho_curve_report.py`
- 測試：`tests/test_cycle_metrics.py`、`tests/test_rho_curve_scaling_stage2_settings.py`

- 問題：`corr_threshold` 在不同 $T$ 與噪音下意義不同（tail 變長時更容易過閾）。
- 建議：Stage2 改成「method + 顯著性統計量」的可切換介面（保留現行作為 baseline），避免同一份 sweep 在不同 window/T 下不可比。

**落地點（介面參數；已完成）**

- 指標計算：`analysis.cycle_metrics.assess_stage2_frequency(...)` 新增：
  - `stage2_method`: `"autocorr_threshold" | "fft_power_ratio" | "permutation_p"`
    - `autocorr_threshold`：沿用現行 dominant autocorr + `corr_threshold`（baseline）
    - `fft_power_ratio`：對 window 序列計算 power spectrum（可先用純 Python DFT 或 numpy；MVP 允許先只在 analysis 用 numpy，若要維持零依賴則限縮到 critical band 才啟用）
      - 通過條件：$\frac{P(f^*)}{\overline{P}} \ge \kappa$
    - `permutation_p`：以 permutation/shuffle 建立虛無分佈（只在 pre-filter 通過時觸發，避免算量爆炸）
      - 統計量：以 `best_corr` 或 `power_ratio` 為 test statistic
      - 通過條件：`p_value <= alpha`
  - `stage2_prefilter`: bool（預設 True；先用便宜的 autocorr 粗篩）
  - `power_ratio_kappa: float`（預設 8.0；可調）
  - `permutation_alpha: float`（預設 0.05）
  - `permutation_resamples: int`（預設 200；可調；只在 critical band 使用）
  - `permutation_seed: int | None`（確保研究可重現）

- 輸出補強（讓報表可回溯）
  - `Stage2FrequencyResult` 或等價輸出必須能回報：
    - `method`（選用哪種顯著性法）
    - `statistic`（ratio / p-value / best_corr 等）
    - `effective_window_n`（實際樣本長度；避免不同 tail 混用）

- 報表：`analysis.rho_curve_report` 必須列出 Stage2 的方法與關鍵參數（kappa/alpha/resamples/seed）。
- 補全/優化（可選）：
  - permutation test 可用標準統計套件實作（例如 SciPy），但需權衡計算成本（只在疑似主頻時觸發；或限制 resamples）。
  - 與 (1) 連動：若 amplitude 改採 normalization，需在 Spec 明確 Stage 2 的輸入序列是否也要同步 normalize。
- 風險與緩解：
  - 風險：permutation 計算量過大。
  - 緩解：pre-filter（例如先用粗 autocorr 過濾），或只在 critical band 附近採用顯著性法。
- DoD（已完成；工程可回歸）
  - 介面參數與輸出欄位固定（method/參數/`Stage2FrequencyResult`），並在 sweep→summary→report 全鏈路保留 provenance。
  - 回歸測試覆蓋：
    - 控制/非週期序列不得誤判（False Positive Guard）
    - 手動構造的週期序列必須穩定通過

3) **理論對照 anchor（Theory anchor）**

> 研發結果（已完成；2026-03-07）：已落地 core deterministic anchors（避免 pytest flakiness），並提供可選 extended anchor runner（讀既有輸出資料）。

- runner：`analysis/theory_anchors.py`
- core regression tests：`tests/test_theory_anchors.py`

- 例：在某些對稱條件（如 $a=b$ 的特定設定/近零和）下，不應出現穩定方向性。
- 用途：讓實驗結論有理論預期可對照（也用來保護指標門檻調整不會偷偷改掉研究結論）。
- 落地點（已完成；見上）：
  - 以「參數點 + 預期結果」寫成固定清單（類 control suite；可放在 `tests/` 或研發日誌並固定引用）。
  - 建議區分 core anchors（必跑）與 extended anchors（選跑），避免日常迭代成本過高。
- 風險與緩解：
  - 風險：anchors 太多會拖慢回歸。
  - 緩解：只保留最小集合：一個 no-cycle、一個 cycle-hotspot。
- DoD（最小可驗收）：
  - 至少兩個 anchor：一個應該接近無動態（如 $a=b=0$），一個應該更容易出現旋轉/週期（研究主線熱點）。
  - 每次調整指標門檻（例如 `eta`/`corr_threshold`）都要重新跑 anchor 並記錄是否破壞預期。

4) **數值漂移防護（Numerical drift guard）**

> 研發結果（已完成；2026-03-06）：已集中化 simplex normalize helper，並在 Stage3/phase 前處理改用共用 guard（含 negative clip、non-finite fallback）。

- 實作：`analysis/simplex.py`、`analysis/cycle_metrics.py`
- 測試：`tests/test_simplex_drift_guard.py`

- 若後續引入會累積誤差的正規化（例如把某序列投影回 simplex），Spec 應要求：
  - **最低要求（規格）**：投影/正規化後必須維持 simplex 約束（非負且總和≈1；不允許產生 `nan/inf`）。
  - **現行實作（更強 guard）**：對每個 timestep 都做 normalize（而非只在 $|\sum-1|$ 超過門檻時才修正），並且：
    - 負值會先 clip 到 0 再 renorm
    - 非有限值（NaN/Inf）或總和≤0 會走 fallback（目前用 zeros/uniform）
  - 先前的「$|\sum-1| > 10^{-8}$ 才 renorm」可視為 *效能最佳化* 的一種做法，但不如現行作法保守。
- 落地點（已完成；見上）：
  - 任何會把序列投影/正規化到 simplex 的地方（例如 Stage 3 相關 preprocessing）都要共用同一個 helper（避免每個檔案各自做一版）。
- 補全/優化（可選）：
  - 記錄 drift event 的頻率與量級（避免 silent correction 掩蓋上游 bug）。
  - 避免過度截斷：必要時回報 pre/post normalization 的差異統計（作為診斷而非核心結論）。
- DoD（最小可驗收）：
  - 在單元測試中覆蓋：sum 稍微偏離 1 時會被拉回；含極小負值的輸入會被截斷後再歸一化；不產生 `nan/inf`。

5) **貝氏參數不確定性估計（Bayesian uncertainty estimation；新增）**

> 研發結果（部分完成；2026-03-07）：已落地 bootstrap CI baseline（stdlib-only）+ **Bayesian logistic regression（Laplace approximation；stdlib-only）**。完整 MCMC/VI（例如 PyMC/Pyro）仍保留為下一階段升級。

- bootstrap（已完成）：`analysis/rho_curve_scaling.py`（`--bootstrap-resamples/--bootstrap-seed` + summary 欄位）
- Bayesian（Laplace；已完成）：`analysis/rho_curve_scaling.py`（`--bayes-method laplace` + summary 欄位）
- report 顯示（已完成）：`analysis/rho_curve_report.py`
- 測試：`tests/test_rho_curve_scaling_bootstrap.py`、`tests/test_rho_curve_scaling_bayes_laplace.py`

- 問題：目前 $k_{50}$ 多為點估計，未顯式反映 seeds/rounds 的不確定性。
- 建議：用 Bayesian logistic regression 擬合 $P(L3)$ vs $k$，輸出 posterior 的 $k_{50}$ 分佈（例如 mean ± std，或 95% credible interval）。
- 落地點（Laplace 版本已做；完整 MCMC/VI 未做）：
  - `analysis.rho_curve_scaling`：以 binomial counts（由 `p_level_3,n_seeds` 還原）做 Binomial-logit 的 Laplace 近似，輸出 $k_{50}$ posterior samples 的 95% CrI。
  - `analysis.rho_curve_report`：回報 CrI 與 method（`bayes_method`）等 provenance。
- 風險與緩解：
  - 風險：MCMC/VI 成本高、依賴套件（例如 PyMC/Pyro）。
  - 緩解：只在 critical band（或少數目標 N）啟用；並保留 bootstrap CI 作為低成本基準。
- DoD（最小可驗收）：
  - 控制組的 posterior 應與 $P(L3)\approx 0$ 相容（不應產生「假 crossing」）。
  - 在主資料上，CI 能穩定輸出且與 bootstrap baseline 相容（不出現顯著自相矛盾）。

6) **敏感度分析（Sensitivity analysis；新增）**

> 研發結果（已完成；2026-03-07）：已提供跨 summary 的 k50 變異比較腳本，並可選擇性嵌入 notes/paper 報告作附錄。

- 腳本：`analysis/sensitivity.py`
- report 整合：`analysis/rho_curve_report.py --sensitivity-summary ...`
- 測試：`tests/test_sensitivity_script.py`、`tests/test_rho_curve_report_sensitivity.py`

- 問題：`eta`/`corr_threshold` 等超參數會影響結論，但目前缺系統性量化。
- 建議：對超參數做小範圍掃描（例如 `eta` 0.4–0.6），回報 $k_{50}$ 的變異與排名（哪些門檻最敏感）。
- 落地點（已完成；見上）：
  - 新增 analysis 腳本（例如 `analysis/sensitivity.py`）：輸入同一份 sweep data（或 summary + 需要的中間統計），輸出 sensitivity ranking 與簡圖。
  - 報表：把 sensitivity 結果收成一小段附錄/小節，避免主文過度膨脹。
- 風險與緩解：
  - 風險：超參數維度爆炸造成組合過多。
  - 緩解：先固定 2–3 個最關鍵門檻做單因子/小網格；必要時再上更進階的抽樣設計（LHS/Sobol）。
- DoD（最小可驗收）：
  - 至少覆蓋 3 個超參數（或 2 個超參數 + 2 個取值），輸出「最敏感」排名。
  - 主結論需附帶敏感度敘述（例如 variance < 10% 才可聲稱 robust；門檻可調但需記錄）。

建議實施順序（避免 over-engineering）：先做 (4) 數值基礎，再做 (1)(2) 統計核心，最後才做 (3)(5)(6) 驗證/進階。

---

## 5) 邊界條件（Boundary conditions）

SDD 需要在規格中先說清楚極端輸入怎麼處理。

- `players = 0`：允許。
  - `p_<strategy>` 皆為 0
  - `avg_utility = 0`
  - `avg_reward` 為空值（無可用 reward）
  - `w_<strategy>` 皆為 1（無觀測時回到中性權重）
- `rounds = 0`：
  - 允許輸出只有 header 的 CSV
- `selection_strength = 0`：權重必須回到均勻（全部 1，且 `mean=1`）。
- `payoff_mode = matrix_ab` 但策略數 != 3：
  - 必須明確 raise（避免錯誤數學假設）
- popularity 為空（第一輪）：
  - $x = (0,0,0)$，則 $U = 0$，reward = `base_reward`

---

## 6) 測試設計（Test design）

> 研究專案也要測試，但測試的目標是「保護規格與資料契約」，不是追求高覆蓋率。

建議測試層級：

### 6.1 單元測試（純函數/不含隨機）

- `evolution.replicator_dynamics.replicator_step`
  - selection_strength < 0 會 raise
  - 回傳權重全為正且 `mean=1`
  - 若所有玩家缺 reward：回傳全 1

- `dungeon.DungeonAI.evaluate`（以固定 popularity 驗證）
  - `matrix_ab` 對特定 x 的 $U$ 計算正確
  - `count_cycle` 對 n 的 penalty/bonus 正確

### 6.2 契約測試（CSV schema）

- 跑 1～3 輪的 simulation（小樣本）
- 檢查 CSV 欄位存在、列數正確、數值可 parse

### 6.3 隨機性控制（可選，但非常建議）

- 若要讓研究可重現：
  - `simulation/run_simulation.py` 已支援 `--seed`
  - `players.BasePlayer` 已支援可注入 RNG（避免所有玩家共用同一 RNG stream）

---

## 7) 模組規劃（把 Spec 映射到程式結構）

### 7.1 研究主線（循環動態）最小閉環

- `dungeon/`：定義 payoff 結構（`count_cycle` / `matrix_ab`）
- `evolution/`：定義更新規則（replicator 近似）
- `simulation/`：連起來、輸出 CSV
- `analysis/`：定義循環/收斂指標（純計算）

### 7.2 12 維 Personality Vector（下一階段，不阻塞主線）

- Spec 建議用「trait → decision policy」的可測映射：
  - trait 只影響「策略抽樣分佈」或「事件接受機率」
  - payoff 結構仍由 dungeon 決定
- 驗證方式：固定 dungeon 參數下，不同 trait 族群的 $p_s(t)$ 統計顯著差異

---

## 8) 實作步驟（SDD 迭代節奏）

每一個 step 都要能獨立驗證，並產出可比較的 CSV。

1. **鎖定契約**：確定 timeseries CSV 欄位與意義（本文件第 3.3）
2. **鎖定數學規格**：明確定義 $x(t)$ 與 payoff time index（本文件第 2 章）
3. **補齊最小指標**：在 `analysis/` 加 1 個循環指標函數（本文件第 4.2 擇一）
4. **加入回歸測試**：針對不變條件 + CSV schema（本文件第 6 章）
5. **參數掃描腳本化**：固定輸出命名規則，避免手動跑錯
6. **研究記錄**：每次 sweep 記錄 spec 版本 + 參數 + 指標結果（建議寫在 `outputs/` 的 metadata 或單一 markdown）

---

## 9) Review checklist（每次改動前先問自己）

- 這個改動的 Spec 是什麼？寫在哪一節？
- 我加的輸出/欄位是契約變更嗎？有更新契約與測試嗎？
- 這個行為能被 `analysis/` 的純函數指標驗證嗎？
- 是否破壞分層（analysis 反向依賴 simulation / evolution 做 I/O）？
- 邊界條件怎麼處理？有沒有 `nan/inf/負權重` 風險？

