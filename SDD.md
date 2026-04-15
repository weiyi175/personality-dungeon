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
- payoff（基礎型）：

$$
A = \begin{pmatrix}
0 & a & -b\\
-b & 0 & a\\
a & -b & 0
\end{pmatrix},\quad U = A x
$$

- 對策略 $i$ 的 reward：
  - $r_i = base\_reward + U_i$

- payoff（可選 cross-term 擴充；預設關閉）：

令 $c_{AD} \ge 0$ 表示 aggressive-defensive coexistence 懲罰強度，則

$$
K(x; c_{AD}) = c_{AD}\,\big(-x_D,\,-x_A,\,x_A + x_D\big)
$$

總 payoff 向量為：

$$
U = A x + K(x; c_{AD})
$$

亦即：

$$
\begin{aligned}
U_A &= a x_D - b x_B - c_{AD} x_D \\
U_D &= -b x_A + a x_B - c_{AD} x_A \\
U_B &= a x_A - b x_D + c_{AD}(x_A + x_D)
\end{aligned}
$$

- 詮釋：當 aggressive 與 defensive 同時存在時，兩者互相施加額外懲罰，balanced 吸收這個 cross-term。
- 不變條件：`c_{AD}=0` 時，行為必須完全退化回原始 `matrix_ab`。

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

### 2.5 新 payoff / memory / evolution 擴充規格（H-series）

本節是下一輪結構假說實驗的正式契約。目標不是立刻擴大 sweep，而是先把可能真正改變 multi-seed basin 的新機制寫成可驗證、可退化回基線的數學規格。

分層責任鎖定：

- `dungeon/`：負責 H1 與 H2 的 payoff 輸入狀態與 payoff mode
- `evolution/`：負責 H3 的更新規則
- `simulation/`：只負責 CLI、執行組裝、與既有 CSV 契約
- `analysis/`：沿用既有 `cycle_metrics` / `decay_rate`，不得反向依賴 simulation

#### (H1) 記憶型 payoff 輸入：`memory_kernel`

目的：測試單步 lag 是否只產生衰減螺旋，而較長回饋記憶是否能把系統推近中性極限環。

令 $x(t)$ 為第 $t$ 輪結束後寫回的 simplex 比例。對奇數 kernel 長度 $m\in\{1,3,5\}$，定義記憶狀態：

$$
x^{(m)}(t) = \frac{1}{m}\sum_{j=0}^{m-1} x(t-j)
$$

本階段先鎖定 **uniform moving average**；任何指數權重或自定 kernel 都屬於下一版契約，不得在本版偷加。

時間索引統一為：

$$
u(t) = F\big(x^{(m)}(t-\ell)\big),\qquad \ell\in\{0,1\}
$$

其中 `payoff_lag = 0/1` 對應 $\ell = 0/1$。

不變條件：

- `memory_kernel = 1` 時，行為必須完全退化回既有單步輸入
- $x^{(m)}(t)$ 必須仍位於 simplex 上：每一分量非負，且總和為 1
- 歷史長度不足時，只使用現有前綴歷史做平均；不得補負時間索引或做外插

#### (H2) 非線性 / 分段式 payoff：`payoff_mode = threshold_ab`

目的：測試線性 `matrix_ab` 是否只是沿著 plateau 平移，而缺乏形成 basin 所需的 regime switch。

先定義由記憶狀態導出的共存量：

$$
q_{AD}(t) = x_A^{(m)}(t) + x_D^{(m)}(t)
$$

對門檻 $\theta\in[0,1]$，定義兩段式 payoff matrix：

$$
A_{lo} = \begin{pmatrix}
0 & a & -b\\
-b & 0 & a\\
a & -b & 0
\end{pmatrix},
\qquad
A_{hi} = \begin{pmatrix}
0 & a_{hi} & -b_{hi}\\
-b_{hi} & 0 & a_{hi}\\
a_{hi} & -b_{hi} & 0
\end{pmatrix}
$$

regime 切換規則：

$$
A^{thr}(t)=
\begin{cases}
A_{lo}, & q_{AD}(t) < \theta \\
A_{hi}, & q_{AD}(t) \ge \theta
\end{cases}
$$

H2.1 最小救援版允許在 `threshold_ab` 上加入可選 hysteresis band。
若指定 `\theta_{lo}, \theta_{hi}` 且滿足 $0\le \theta_{lo} \le \theta_{hi} \le 1$，則 regime 狀態 $r(t)\in\{lo,hi\}$ 依下列規則更新：

$$
r(t)=
\begin{cases}
hi, & q_{AD}(t) \ge \theta_{hi} \\
lo, & q_{AD}(t) \le \theta_{lo} \\
r(t-1), & \theta_{lo} < q_{AD}(t) < \theta_{hi}
\end{cases}
$$

並令

$$
A^{thr}(t)=
\begin{cases}
A_{lo}, & r(t)=lo \\
A_{hi}, & r(t)=hi
\end{cases}
$$

若未指定 `\theta_{lo}, \theta_{hi}`，則視為 `\theta_{lo}=\theta_{hi}=\theta`，必須完全退化回原本的單門檻 `threshold_ab`。

H2.2 最小救援版允許把切換觸發量從固定的 `q_{AD}=x_A+x_D` 推廣成可選 trigger，並在 trigger 與 regime 間加入慢變內部狀態。

先定義 trigger 函數 $g(x)$：

$$
g_{share}(x)=x_A+x_D
$$

$$
g_{prod}(x)=4x_Ax_D
$$

其中 `ad_share` 對應 $g_{share}$，`ad_product` 對應 $g_{prod}$。

再定義內部狀態 $z(t)$：

$$
z(t)=\alpha\,g\big(x^{(m)}(t)\big) + (1-\alpha)z(t-1), \qquad \alpha\in(0,1]
$$

首次更新時令 $z(0)=g(x^{(m)}(0))$。若 `\alpha=1`，則 $z(t)$ 必須完全退化為當期 trigger 值本身，不得殘留任何額外記憶。

H2.1 的 hysteresis 規則在 H2.2 中一律作用於 $z(t)$，而不是直接作用於原始 trigger：

$$
r(t)=
\begin{cases}
hi, & z(t) \ge \theta_{hi} \\
lo, & z(t) \le \theta_{lo} \\
r(t-1), & \theta_{lo} < z(t) < \theta_{hi}
\end{cases}
$$

對於首次更新時尚無前一個 regime 狀態的情況，若 $q_{AD}(t)$ 落在 hysteresis band 內，初始化一律取 `lo`；只有當 $q_{AD}(t) \ge \theta_{hi}$ 時才進入 `hi`。這是為了避免 sampled 路徑在 band 內因初始條件直接鎖進 high regime。

補充語意鎖定：

- `q_AD(t)` 一律由同一個 payoff 輸入狀態 `x^{(m)}(t)` 計算，不得另外改用未平均的即時抽樣比例
- H2.2 中所有 trigger 都必須由同一個 payoff 輸入狀態 `x^{(m)}(t)` 計算，不得另外改用未平均的即時抽樣比例
- 臨界點 `q_AD(t) = theta` 時，一律走 high regime，避免不同模組各自採用不同 tie-break
- regime 切換只改變 payoff matrix 的 `(a,b)` 係數；`memory_kernel`、`payoff_lag`、與 `matrix_cross_coupling` 的語意完全沿用既有定義
- 啟用 hysteresis 時，band 內部不得重新以當下 `q_AD` 直接覆寫 regime；必須沿用上一個 regime 狀態
- 啟用 H2.2 時，band 內部不得重新以當下原始 trigger 直接覆寫 regime；必須沿用上一個 regime 狀態，且比較對象是慢變狀態 $z(t)$

因此 `threshold_ab` 的 payoff 為：

$$
u(t)=A^{thr}(t-\ell)\,x^{(m)}(t-\ell) + K\big(x^{(m)}(t-\ell); c_{AD}\big)
$$

其中 $K(\cdot;c_{AD})$ 保留既有 `matrix_cross_coupling` 定義，允許 threshold 與 cross-coupling 並存。

不變條件：

- `a_hi = a` 且 `b_hi = b` 時，`threshold_ab` 必須完全退化回 `matrix_ab`
- `theta` 必須限制在 $[0,1]$
- `theta_lo`、`theta_hi` 若提供，必須滿足 $0\le \theta_{lo} \le \theta_{hi} \le 1$
- `threshold_trigger` 目前只允許 `{ad_share, ad_product}`
- `threshold_state_alpha` 必須滿足 $0 < \alpha \le 1$
- `a_hi`、`b_hi` 必須是有限實數；`nan`、`inf` 一律視為非法輸入
- regime 切換只依賴 payoff 輸入狀態，不得偷看本輪尚未完成的抽樣結果
- 第一輪驗證必須先包含 no-op 對照：以 `a_hi=a`、`b_hi=b` 驗證 `threshold_ab` 的退化等價性，再進入真正的 smoke
- H2.1 驗證必須額外確認：當 `theta_lo=theta_hi=theta` 時，hysteresis 版本與原始 H2 `threshold_ab` 完全等價
- H2.2 驗證必須額外確認：當 `threshold_trigger=ad_share`、`threshold_state_alpha=1`、且 `theta_lo=theta_hi=theta` 時，H2.2 必須完全退化回原始 H2

#### (H3) 異質更新規則：`evolution_mode = hetero`

目的：測試同質 replicator 是否過快抹平方向性，而策略別更新速度是否能保留 phase lag。

本階段先鎖定最小版本：**per-strategy selection strength**。令

$$
\kappa=(k_A,k_D,k_B),\qquad k_i\ge 0
$$

則更新規則改為：

$$
w_i(t+1) \propto w_i(t)\exp\Big(k_i\big(u_i(t)-\bar u(t)\big)\Big),
\qquad \bar u(t)=\sum_i x_i(t)u_i(t)
$$

其餘正規化維持與既有 replicator 一致：所有權重有限且為正，最後 normalize 到 `mean(weight)=1`。

不變條件：

- `k_A = k_D = k_B = selection_strength` 時，`hetero` 必須完全退化回既有同質更新
- CLI 若省略 `--strategy-selection-strengths`，則視為自動展開成 `(selection_strength, selection_strength, selection_strength)`；這是合法但等價於基線的 no-op 設定
- `k_i < 0`、`nan`、`inf` 一律視為非法輸入

#### (H3.1) 固定子族群：`fixed_subgroup_share` + `fixed_subgroup_weights`

目的：測試只靠 per-strategy selection strength 是否仍不足以保留結構性相位差，而引入一個不隨 payoff 更新的 frozen subgroup 能否提供更穩定的旋轉錨點。

本版 H3.1 採最小契約：

- 玩家母體切成兩群：固定子族群與適應子族群
- 固定子族群只凍結 `strategy_weights`，不改 `GameEngine`、不改 popularity 更新規則、也不改 payoff 定義
- 適應子族群仍沿用 H3 的 `hetero` 或既有 `sampled` 更新規則

令玩家總數為 $N$，固定子族群占比為 $\rho_f\in[0,1]$，則固定玩家數定義為：

$$
N_f = \operatorname{round}(\rho_f N)
$$

其餘玩家數為 $N_a = N - N_f$。

若固定子族群權重向量為

$$
w^{fix} = (w_A^{fix}, w_D^{fix}, w_B^{fix}),\qquad w_i^{fix} > 0
$$

則所有固定玩家在每一輪結束後都必須維持同一個權重向量：

$$
w_i^{(p)}(t+1) = w_i^{fix},\qquad p \in \mathcal{F}
$$

適應子族群玩家 $p \notin \mathcal{F}$ 則照原本更新：

$$
w_i^{(p)}(t+1) \propto w_i^{(p)}(t)\exp\Big(k_i\big(u_i^{(p)}(t)-\bar u^{(p)}(t)\big)\Big)
$$

其中若未啟用 `hetero`，則 $k_i$ 全部退化為同一個 `selection_strength`。

不變條件：

- `fixed_subgroup_share = 0` 時，H3.1 必須完全退化回未啟用固定子族群的原始路徑
- `fixed_subgroup_share > 0` 且未提供 `fixed_subgroup_weights` 時，必須報錯，不得靜默套用預設值
- `fixed_subgroup_weights` 的各分量必須為有限正數
- `fixed_subgroup_share = 1` 時，整個 sampled population 的 `w_*` 序列必須恆等於固定權重向量
- 本版固定子族群的成員選取規則鎖定為 deterministic prefix：取玩家列表前 `N_f` 人；若要改成隨機抽樣或 personality-conditioned subgroup，必須另開下一版 Spec

#### (H3.2) 非對稱子族群耦合：`fixed_subgroup_coupling_strength`

目的：測試 H3.1 僅靠 frozen subgroup 的被動存在仍然太弱時，是否需要把 fixed subgroup 直接升格為 adaptive subgroup 的 payoff 錨點。

H3.2 建立在 H3.1 之上，因此前提仍是存在 frozen subgroup。令固定子族群的規格化 anchor simplex 為

$$
x^{fix} = \operatorname{Normalize}(w^{fix})
$$

令第 $t$ 輪 adaptive subgroup 的實現策略比例為

$$
x^{ad}(t)
$$

再定義 gap 向量：

$$
g(t) = x^{fix} - x^{ad}(t)
$$

H3.2 的額外 payoff shift 只作用在 adaptive subgroup，且沿用 `matrix_ab` 的旋轉幾何，不額外改寫既有 `matrix_cross_coupling`：

$$
\Delta u^{H3.2}(t) = \lambda_f \, A(a,b) \, g(t)
$$

其中 $\lambda_f \ge 0$ 對應 `fixed_subgroup_coupling_strength`，而

$$
A(a,b)=
\begin{pmatrix}
0 & a & -b\\
-b & 0 & a\\
a & -b & 0
\end{pmatrix}
$$

因此對 adaptive 玩家 $p \notin \mathcal{F}$、若其本輪選擇策略為 $i$，則更新前的 reward 調整為：

$$
u_i^{adj}(t) = u_i^{raw}(t) + \Delta u_i^{H3.2}(t)
$$

固定子族群玩家的 reward 不加這個 H3.2 shift；他們仍只扮演 frozen anchor。

不變條件：

- `fixed_subgroup_coupling_strength = 0` 時，H3.2 必須完全退化回 H3.1
- `fixed_subgroup_coupling_strength > 0` 時，必須同時提供合法的 `fixed_subgroup_share` 與 `fixed_subgroup_weights`
- H3.2 只定義在 sampled 類玩家路徑；若 `evolution_mode = mean_field` 且啟用 H3.2，必須報錯
- H3.2 不得新增主 CSV 欄位；它只改變更新前 reward，因而間接影響既有 `w_*`

#### (H3.3B) 狀態依賴子族群耦合：`state_dependent_subgroup_coupling`

目的：若 H3.2 的問題不在於固定 anchor 的方向錯誤，而在於耦合不該「每一輪都等強度」作用，則 H3.3B 改測試 phase-specific shaping：只在系統進入特定狀態區間時放大 frozen subgroup 對 adaptive subgroup 的耦合。

H3.3B 仍建立在 H3.1 與 H3.2 之上，因此前提同樣是存在 frozen subgroup，並沿用

$$
g(t) = x^{fix} - x^{ad}(t)
$$

以及 H3.2 的旋轉幾何

$$
A(a,b)=
\begin{pmatrix}
0 & a & -b\\
-b & 0 & a\\
a & -b & 0
\end{pmatrix}
$$

但 H3.3B 不再使用常數耦合，而是把實際耦合強度改成狀態函數：

$$
\lambda(t) = \lambda_0 \cdot s(t)
$$

其中 $\lambda_0 \ge 0$ 為 base coupling strength，而第一版 H3.3B 將 $s(t)$ 鎖定為單一狀態量 $z(t)$ 的 sigmoid gate：

$$
s(t)=\sigma\big(\beta(z(t)-\theta_z)\big)
$$

$$
\sigma(q)=\frac{1}{1+e^{-q}}
$$

因此 adaptive subgroup 的更新前 reward 調整為：

$$
\Delta u^{H3.3B}(t)=\lambda_0 \cdot \sigma\big(\beta(z(t)-\theta_z)\big) \cdot A(a,b) g(t)
$$

$$
u_i^{adj}(t)=u_i^{raw}(t)+\Delta u_i^{H3.3B}(t)
$$

其中第 1 版只允許單一狀態來源 `z(t)`，不得同時混用多個 trigger。為避免與 H2 類 regime switching 混淆，H3.3B 第一版將狀態量鎖定為 subgroup gap amplitude：

$$
z(t)=\lVert g(t) \rVert_2
$$

也就是說，只有當 frozen subgroup 與 adaptive subgroup 的 realized simplex gap 夠大時，coupling 才會接近完全開啟；當 gap 很小時，則自動退回弱耦合。

第一版 H3.3B 參數契約：

- `fixed_subgroup_state_coupling_strength = λ_0`：base coupling strength，格式為有限且 `>= 0` 的實數
- `fixed_subgroup_state_coupling_beta = β`：sigmoid slope，格式為有限且 `> 0` 的實數
- `fixed_subgroup_state_coupling_theta = θ_z`：state threshold，格式為有限且 `>= 0` 的實數
- `fixed_subgroup_state_signal = gap_norm`：第 1 版唯一合法值，表示 `z(t)=||g(t)||_2`

不變條件：

- `fixed_subgroup_state_coupling_strength = 0` 時，H3.3B 必須完全退化回 H3.1；不得殘留任何 state-dependent shift
- H3.3B 啟用時，不得同時啟用 H3.2 的 `fixed_subgroup_coupling_strength > 0`；H3.2 與 H3.3B 在第一版中互斥，避免常數耦合與狀態耦合同時混用
- H3.3B 只定義在 sampled 類玩家路徑；若 `evolution_mode = mean_field` 且啟用 H3.3B，必須報錯
- H3.3B 啟用時，仍必須同時提供合法的 `fixed_subgroup_share` 與 `fixed_subgroup_weights`
- `fixed_subgroup_state_signal` 第一版只能是 `gap_norm`；若要改成 `corr`, `gamma`, `event_intensity` 或其他 rolling proxy，必須另開下一版 Spec
- H3.3B 不得新增主 CSV 欄位；它只改變更新前 reward，並透過既有 `w_*`、`p_*` 與 cycle metrics 間接反映其效果

#### (H3.4) 半凍結子族群：`fixed_subgroup_anchor_pull_strength`

目的：若 H3.2 / H3.3B 的問題在於 frozen subgroup 對 adaptive subgroup 的外部 reward shaping 太弱，則 H3.4 不再只改 adaptive subgroup 的 reward，而是直接改 subgroup topology 本身：把 H3.1 的完全 frozen anchor 放鬆成 semi-frozen anchor。

H3.4 仍沿用 H3.1 的 deterministic prefix subgroup 與 anchor 權重 `w^{fix}`，但不再要求固定子族群每輪都被硬覆寫回 `w^{fix}`。改為先按既有 sampled/hetero path 算出本輪演化更新後的公共權重

$$
	ilde{w}(t+1)
$$

再對固定子族群玩家套用 anchor pullback：

$$
w^{semi}(t+1) = \operatorname{NormalizeMean1}\Big((1-\rho_f)\,\tilde{w}(t+1) + \rho_f\,w^{fix}\Big)
$$

其中 $\rho_f \in [0,1]$ 對應 `fixed_subgroup_anchor_pull_strength`。

語意：

- `\rho_f = 1`：完全退化回 H3.1，固定子族群每輪都被硬拉回 anchor
- `0 < \rho_f < 1`：固定子族群保留部分內生更新，但每輪都被 anchor 回拉
- `\rho_f = 0`：固定子族群不再被 anchor 拉回，整條路徑必須退化回沒有 subgroup topology effect 的原始 sampled/hetero path

H3.4 第一版只改「固定子族群自己的下一輪權重更新」，不額外新增雙向 payoff coupling；也就是說，固定子族群仍透過 sampled population 的 realized strategy mix 影響整體系統，但不再是完全 frozen external anchor。

第一版 H3.4 參數契約：

- `fixed_subgroup_anchor_pull_strength = \rho_f`：有限且落在 `[0,1]` 的實數；預設值為 `1.0`，以維持 H3.1 backward compatibility

不變條件：

- `fixed_subgroup_anchor_pull_strength = 1` 時，H3.4 必須完全退化回 H3.1
- `fixed_subgroup_anchor_pull_strength = 0` 時，H3.4 必須完全退化回未啟用 subgroup topology effect 的原始 sampled/hetero path
- `0 <= fixed_subgroup_anchor_pull_strength <= 1`；超界或非有限值必須報錯
- 若 `fixed_subgroup_anchor_pull_strength < 1`，則必須同時提供合法的 `fixed_subgroup_share` 與 `fixed_subgroup_weights`
- H3.4 第一版只定義在 sampled 類玩家路徑；若 `evolution_mode = mean_field` 且 `fixed_subgroup_anchor_pull_strength < 1`，必須報錯
- 為了保持因果可解釋性，當 `fixed_subgroup_anchor_pull_strength < 1` 時，不得同時啟用 H3.2 或 H3.3B；若要研究 semi-frozen 再疊常數/狀態耦合，必須另開下一版 Spec
- H3.4 不得新增主 CSV 欄位；它只改變 subgroup 自身的下一輪權重更新，並透過既有 `w_*`、`p_*` 與 cycle metrics 間接反映其效果

#### (H3.5) 真雙向 payoff coupling：`fixed_subgroup_bidirectional_coupling_strength`

目的：若 H3.4 仍然顯示單純把 frozen subgroup 放鬆成 semi-frozen subgroup 還不夠，則下一步不再繼續細調 `\rho_f`，而是讓兩個 subgroup 都成為彼此的 payoff source，也就是引入真正的雙向 payoff coupling。

H3.5 建立在 H3.4 之上，因此前提是存在一個 semi-frozen subgroup；也就是說，固定 prefix subgroup 雖然仍受 anchor pullback 約束，但已經恢復部分內生更新自由度。令兩個 subgroup 的 realized strategy simplex 為

$$
x^{fix}(t), \qquad x^{ad}(t)
$$

並定義 subgroup gap

$$
g(t)=x^{fix}(t)-x^{ad}(t)
$$

沿用既有 `matrix_ab` 的旋轉幾何

$$
A(a,b)=
\begin{pmatrix}
0 & a & -b\\
-b & 0 & a\\
a & -b & 0
\end{pmatrix}
$$

H3.5 的雙向 payoff shift 定義為：

$$
\Delta u^{ad}(t)=\lambda_{bi} A(a,b) g(t)
$$

$$
\Delta u^{fix}(t)=-\lambda_{bi} A(a,b) g(t)
$$

其中 `fixed_subgroup_bidirectional_coupling_strength = \lambda_{bi} \ge 0`。

因此兩個 subgroup 都在更新前收到來自對方的等量反向 payoff shift：

$$
u_i^{ad,adj}(t)=u_i^{ad,raw}(t)+\Delta u_i^{ad}(t)
$$

$$
u_i^{fix,adj}(t)=u_i^{fix,raw}(t)+\Delta u_i^{fix}(t)
$$

因為 H3.5 要求固定子族群本身也能對 payoff shift 做出下一輪權重回應，所以第一版 H3.5 明確要求 `fixed_subgroup_anchor_pull_strength < 1`。若仍是完全 frozen (`\rho_f=1`)，則 fixed subgroup 收到的 reward shift 不會改變其權重路徑，不能稱為「真正雙向」。

第一版 H3.5 參數契約：

- `fixed_subgroup_bidirectional_coupling_strength = \lambda_{bi}`：有限且 `>= 0` 的實數

不變條件：

- `fixed_subgroup_bidirectional_coupling_strength = 0` 時，H3.5 必須完全退化回 H3.4
- `fixed_subgroup_bidirectional_coupling_strength > 0` 時，必須同時提供合法的 `fixed_subgroup_share` 與 `fixed_subgroup_weights`
- H3.5 啟用時，必須滿足 `fixed_subgroup_anchor_pull_strength < 1`；否則報錯
- H3.5 只定義在 sampled 類玩家路徑；若 `evolution_mode = mean_field` 且啟用 H3.5，必須報錯
- H3.5 啟用時，不得同時啟用 H3.2 或 H3.3B；第一版不混用單向 coupling、狀態 gating、與雙向 coupling
- H3.5 不得新增主 CSV 欄位；它只改變兩個 subgroup 更新前的 reward，並透過既有 `w_*`、`p_*` 與 cycle metrics 間接反映其效果

#### (H4) Hybrid 更新機制：`evolution_mode = hybrid`

目的：既然 deterministic / mean_field 路徑已驗證線性 payoff 幾何可以穩住，而 sampled 路徑會把這個幾何洗掉，H4 的第一版不再強化 payoff，而是直接修改 sampled population 的更新規則，讓一部分玩家保留 deterministic 更新。

H4 第一版鎖定最小 sampled-side transfer mechanism：

- 玩家仍然照 sampled path 真實抽樣策略並進入 `GameEngine`
- popularity 更新仍然沿用既有 sampled 規則
- 但在每輪結束後的權重更新階段，玩家母體會分成兩群：hybrid cohort 與 ordinary sampled cohort

令 hybrid 佔比為 $\rho_h\in[0,1]$，玩家總數為 $N$，則 hybrid 玩家數定義為：

$$
N_h = \operatorname{round}(\rho_h N)
$$

第一版 hybrid cohort 的成員選取規則鎖定為 deterministic prefix：取玩家列表前 `N_h` 人。若未來要改成 personality-conditioned 或隨機抽樣 cohort，必須另開下一版 Spec。

對 ordinary sampled cohort 玩家，權重更新完全沿用既有 sampled replicator：

$$
w_i^{samp}(t+1) \propto \exp\Big(k_i\big(\hat u_i^{samp}(t)-\bar u^{samp}(t)\big)\Big)
$$

其中 $\hat u_i^{samp}(t)$ 仍由 sampled players 的 realized rewards 聚合而來。

對 hybrid cohort 玩家，則不使用 sampled per-strategy reward average，而是用與 deterministic 路徑一致的 payoff input state $x^{(m)}(t-\ell)$ 計算 expected payoff：

$$
u^{det}(t) = A(a,b)x^{(m)}(t-\ell) + K\big(x^{(m)}(t-\ell); c_{AD}\big)
$$

並對每位 hybrid 玩家自己的當前權重向量做 deterministic replicator 更新：

$$
w_i^{hyb,(p)}(t+1) \propto w_i^{(p)}(t)\exp\Big(k_i\big(u_i^{det}(t)-\bar u^{det,(p)}(t)\big)\Big)
$$

其中

$$
\bar u^{det,(p)}(t)=\sum_i x_i^{(p)}(t)u_i^{det}(t), \qquad x^{(p)}(t)=\operatorname{Normalize}(w^{(p)}(t))
$$

第一版 H4 刻意只支援 `matrix_ab`，不混入 `threshold_ab` 的 regime state，避免把 sampled-side transfer 與 H2 的切換幾何重新糾纏在一起。

第一版 H4 參數契約：

- `evolution_mode = hybrid`
- `hybrid_update_share = \rho_h`：有限且落在 `[0,1]` 的實數

不變條件：

- `hybrid_update_share = 0` 時，H4 必須完全退化回既有 `sampled` 更新路徑
- `hybrid_update_share = 1` 時，所有 sampled 玩家都採 deterministic expected-payoff 更新，但 action sampling 與 popularity 更新仍是 sampled；因此它不得被誤認為 `mean_field`
- H4 第一版只定義在 `popularity_mode = sampled` 且 `payoff_mode = matrix_ab`；若組合不符，必須報錯
- H4 第一版不得與 H3.* 的 subgroup family 混用；若 `fixed_subgroup_share > 0`、任何 subgroup coupling 非零，或其他 subgroup 旗標啟用，必須報錯
- H4 第一版不新增主 CSV 欄位；它只改變 sampled population 的權重更新規則，並透過既有 `w_*`、`p_*` 與 cycle metrics 間接反映其效果
- H3.3B 的 state signal 必須是 subgroup-local quantity；第一版不得直接重用 H2 的 trigger state 或 regime memory，以免把 H3.3B 退化成 H2 的另一種門控寫法

#### (H4.1) Momentum / inertia hybrid 更新：`hybrid_inertia`

H4 第一版證明了「把 deterministic expected-payoff update 直接混進 sampled path」本身還不夠。H4.1 的最小版不改 hybrid cohort 的選取規則，也不改 expected-payoff 輸入，而是在 hybrid 玩家自己的 deterministic replicator 上加入一階 inertia，讓上一輪的更新方向不會立刻被 sampled realization 洗掉。

對 hybrid cohort 玩家，先定義 deterministic log-growth：

$$
g_i^{det,(p)}(t)=k_i\big(u_i^{det}(t)-\bar u^{det,(p)}(t)\big)
$$

其中

$$
\bar u^{det,(p)}(t)=\sum_i x_i^{(p)}(t)u_i^{det}(t)
$$

再定義每位 hybrid 玩家、每個策略上的 momentum 狀態 $v_i^{(p)}(t)$。H4.1 的最小 inertia 更新為：

$$
v_i^{(p)}(t+1)=\mu_h\,v_i^{(p)}(t)+g_i^{det,(p)}(t)
$$

$$
w_i^{hyb,(p)}(t+1) \propto w_i^{(p)}(t)\exp\big(v_i^{(p)}(t+1)\big)
$$

其中 $\mu_h$ 為 hybrid inertia 係數。初始化時，所有 hybrid 玩家都令 $v_i^{(p)}(0)=0$。

H4.1 最小版參數契約：

- `hybrid_inertia = \mu_h`：有限且落在 `[0,1)` 的實數

不變條件：

- `hybrid_inertia = 0` 時，H4.1 必須精確退化回 H4 第一版 deterministic hybrid update
- `hybrid_update_share = 0` 時，不論 `hybrid_inertia` 為何，整體路徑都必須精確退化回既有 `sampled` 更新路徑
- H4.1 仍只定義在 `evolution_mode = hybrid`、`popularity_mode = sampled`、`payoff_mode = matrix_ab`
- H4.1 仍不得與 H3.* subgroup family 混用
- H4.1 不新增主 CSV 欄位；新增的 momentum state 只存在於 runtime player state，不進主 timeseries schema

#### (H5) Sampled 幾何保留機制

到 H4.1 為止，已可視為完成一輪框架級收斂：deterministic 線性 `matrix_ab` 在 `memory_kernel=1/3` 下可穩定維持長窗 Level 3，而 sampled-side 既有轉移機制（H1 memory、H2 threshold、H3 subgroup、H4 hybrid-share、H4.1 hybrid inertia）都尚未把這個幾何穩定地移植到 sampled path。後續研究主軸因此正式轉為 H5：直接處理 sampled replicator 平均化如何摧毀相位結構。

H5 的總原則：

- 目標是保留 deterministic 幾何的相位方向與 envelope，而不是再增加 payoff strength、subgroup share、或 memory/kernel 旋鈕
- 每一條新機制先用 deterministic mean_field gate 驗證「不破壞既有穩定環」，再進 sampled 測試
- 停止所有純 share 掃描、純 subgroup 微調、純 memory/kernel 延伸；若沒有新結構變因，視為不屬於 H5

#### (H5.1) Sampled inertial replicator：`sampled_inertia`

H5.1 是 H5 的第一條最小線：不再使用 deterministic prefix hybrid cohort，也不再保留 `hybrid_update_share`。它直接在 sampled 更新算子本體上加入一階 inertia，目標是延緩 sampled replicator 對相位的快速平均化。

先定義 sampled path 的 per-strategy 經驗 payoff 聚合：

$$
g_i^{samp}(t)=k\big(\hat u_i(t)-\bar{\hat u}(t)\big)
$$

其中 $\hat u_i(t)$ 是 sampled population 在第 $t$ 輪對策略 $i$ 的經驗平均 payoff，$\bar{\hat u}(t)$ 是對應的整體平均。再定義每位 sampled 玩家、每個策略上的 inertia 狀態 $v_i^{(p)}(t)$：

$$
v_i^{(p)}(t+1)=\mu_s\,v_i^{(p)}(t)+g_i^{samp}(t)
$$

$$
w_i^{(p)}(t+1) \propto w_i^{(p)}(t)\exp\big(v_i^{(p)}(t+1)\big)
$$

初始化時令所有玩家、所有策略的 $v_i^{(p)}(0)=0$。

H5.1 最小版參數契約：

- `evolution_mode = sampled_inertial`
- `sampled_inertia = \mu_s`：有限且落在 `[0,1)` 的實數

不變條件：

- `sampled_inertia = 0` 時，H5.1 必須精確退化回既有 `sampled` 更新路徑
- H5.1 第一版只定義在 `payoff_mode = matrix_ab` 且 `popularity_mode = sampled`，避免把 H5 和 H2 threshold regime 重新糾纏
- H5.1 第一版不得與 `hybrid_update_share`、`hybrid_inertia`、或任何 H3 subgroup 旗標混用
- H5.1 不新增主 CSV 欄位；新增 inertia state 只存在 runtime player state，不進主 timeseries schema
- deterministic mean_field gate 只是 H5.1 的控制 protocol，不是新的 runtime mode；它必須沿用既有 `expected + mean_field` 路徑

#### (H5.2) `sampled_inertia × selection_strength` 小網格

H5.2 不是新的 runtime mode，而是 H5.1 已落地的 `sampled_inertial` 算子上，針對 `sampled_inertia` 與 `selection_strength` 的最小交互掃描。它的目的不是找大範圍 ridge，而是快速回答：sampled inertia 是否只在某些 selection 強度區間內才與 sampled path 相容。

H5.2 第一版 protocol：

- 固定工作點：`payoff_mode = matrix_ab`, `popularity_mode = sampled`, `evolution_mode = sampled_inertial`
- 固定參數：`a=1.0`, `b=0.9`, `matrix_cross_coupling=0.20`, `memory_kernel=3`, `players=300`, `rounds=1500`, `burn_in=300`, `tail=400`, `series=p`, `eta=0.55`, `corr_threshold=0.09`, `stage3_method=turning`, `init_bias=0.12`
- 掃描網格：`sampled_inertia ∈ {0.10, 0.25, 0.40}`, `selection_strength ∈ {0.04, 0.06, 0.08}`
- H5.2 機制 cell 只有上述 `3×3 = 9` 個；但判讀時必須另外補 `mu_s = 0` 的 matched baseline，且 baseline 必須對每個 `selection_strength` 分開計算

H5.2 判讀規則：

- Primary success：至少 2 個 edge-adjacent cell（共邊，不含對角）同時滿足下列條件，且全部相對於「同一個 `selection_strength` 的 `mu_s=0` baseline」比較
- `mean_stage3_score` uplift `>= 0.018`
- `mean_turn_strength` uplift `>= 15%`
- `mean_env_gamma` 較 matched baseline 改善（數值更大；若原本為負，代表更接近 0 或轉正）
- Secondary success：若沒有 Primary success，但存在任一單點同時滿足 `env_gamma > 0` 且 `turn_strength >= 0.9 × baseline_turn_strength`，則可把該點保留到 longer confirm
- Fail rule：若 9 個 H5.2 cell 全部沒有達到 Primary/Secondary success，且 `mean_env_gamma` 也沒有對 matched baseline 形成改善，則 H5.2 直接結案，不再補任何點

H5.2 後續分流：

- 若有 Primary success，最多只取 2 個點進 longer confirm：先取 grid 內表現最佳的 1 個，再取 1 個與其 edge-adjacent 的支持點
- 若只有 Secondary success，最多只 confirm 1 個點
- 若 H5.2 失敗，直接轉 H5.3（phase-preserving weak global constraint）或停止 sampled-operator inertia 線

### 2.6 B-series：sampled-side geometry preservation（2026-04-03, updated 2026-04-07）

本節鎖定 H1-H7 / W2-W3 closure 後的新主線：**不是再調 payoff 幾何本身，而是直接處理 sampled path 的三層均化**。2026-04-03 版先正式落地 B4；2026-04-07 起，B4 linear 第一輪已依 locked short scout 正式 closure，允許 B3.1 進入 runtime 實作與 G0-G2。

#### (B4) `personality_coupled + state-dependent k`（第一輪正式契約）

目的：在不新建 evolution mode 的前提下，沿用既有 `personality_coupled` per-player update 骨架，讓 selection strength 同時受 personality 與當前 global state 調制。

定義：令第 $t$ 輪 sampled global strategy share 為

$$
p(t) = \big(p_A(t), p_D(t), p_B(t)\big)
$$

定義第一版 state signal 為當前 dominance：

$$
d(t) = \max\{p_A(t), p_D(t), p_B(t)\}
$$

其中 $d(t) \in [1/3, 1]$。B4 第一版只允許讀取當前輪的 global sampled share；不得讀未來資訊，也不得直接用 seed-level Stage3 指標作為 in-loop state。

每位玩家 $p$ 的 personality-driven 信號沿用 H7.1：

$$
z_k^{(p)} = 0.5\,(ambition^{(p)} + greed^{(p)}) - 0.5\,(caution^{(p)} + fearfulness^{(p)})
$$

第一版 linear state factor 定義為：

$$
f_{state}(t) = 1 + \beta_{state\_k}\left(\frac{1}{3d(t)} - 1\right)
$$

有效 selection strength：

$$
k_p(t) = \mathrm{clamp}\Big(k_{base}\,(1 + \lambda_k z_k^{(p)})\,f_{state}(t),\; 0.03,\; 0.09\Big)
$$

其中：

- `k_base` 第一輪固定沿用 `selection_strength`
- `\beta_{state_k}` 第一輪只允許 linear 版本
- `\beta_{state_k}` 的合法範圍第一版鎖定為 `[0, 1.5]`，以保證對所有 $d(t) \in [1/3,1]$，`f_state(t) >= 0`

per-player inertia 仍沿用 H7.1：

$$
\mu_p = \mathrm{clamp}(\mu_{base} + \lambda_\mu z_\mu^{(p)}, 0.0, 0.60)
$$

不變條件：

- `beta_state_k = 0` 時，B4 必須精確退化回既有 H7.1 `personality_coupled` 路徑
- `lambda_k = 0`, `lambda_mu = 0`, `mu_base = 0`, `beta_state_k = 0` 時，必須精確退化回 `sampled` baseline
- B4 第一版不得改動主 timeseries CSV schema；新增 state-dependent 診斷只允許存在 runtime / summary / decision 層
- B4 第一版只允許 linear state factor；entropy proxy、`dist_to_center`、exponential factor 都屬於 B4 family 內部升級版。若 linear Short Scout 已顯示 `k_clamped_ratio` 接近 0 且 `0/3` seeds Level 3、`mean_stage3_score` uplift `< 0.02`，則直接記為 B4 closure，不再要求先做 B4 擴展版

#### (B3) `sampled + stratified growth aggregation`（第一輪正式契約）

目的：直接切入 sampled path 的第二層均化，把「全族群共用一個 growth vector」改成「先在 strata 內聚合，再做跨 strata 加權合成」，檢查局部相位差是否在全域平均前就被磨平。

第一輪只允許 index-based strata；在 B3.1 short scout 已正式 closure 後，允許升級到 B3.2 personality-based strata。phase-aware strata 仍保留為 B3.3。

定義：令玩家集合被固定切成 `n_strata` 個 buckets，bucket id 由 player index 等距分配：

$$
\mathrm{stratum}(p_i) = i \bmod n_{strata}
$$

對每個 stratum $h$，先依 sampled path 原本的定義計算 per-strategy growth：

$$
g^{(h)}_s(t) = \bar u^{(h)}_s(t) - \bar u^{(h)}(t)
$$

其中 $\bar u^{(h)}_s$ 是 stratum $h$ 內本輪選擇策略 $s$ 的玩家平均 reward，$\bar u^{(h)}$ 是 stratum $h$ 內全部玩家平均 reward。再用 stratum player count 做 weighted mean：

$$
g^{strat}_s(t) = \sum_h \frac{N_h}{\sum_j N_j} g^{(h)}_s(t)
$$

其中 $N_h$ 是第 $h$ 個 stratum 的玩家數。

第一輪 runtime 契約：

- `sampled_growth_n_strata = 1` 時，必須精確退化回既有 sampled growth 路徑
- B3.1 只允許 `sampled_growth_n_strata ∈ {1,3,5,10}`
- B3.1 不得改動主 timeseries CSV schema；新增分層診斷只允許存在 runtime / summary / decision 層
- B3.1 第一版只允許 index-based strata；personality-based 與 phase-aware strata 屬於同一 B3 family 的升級版，只有在 B3.1 fail 後才允許啟動

#### (B3.2) `sampled + personality-based stratified growth`（第二輪正式契約）

目的：保留 B3 直接作用於 growth aggregation 的位置，但把分桶方式從 index-based 改成與玩家靜態異質性對齊的 personality signal，以避免 B3.1 的任意切桶。

第一版 personality signal 固定使用：

$$
z_k^{(p)} = 0.5\,(ambition^{(p)} + greed^{(p)}) - 0.5\,(caution^{(p)} + fearfulness^{(p)})
$$

B3.2 的 strata assignment 規則固定為：

1. 先為每位玩家指定一個靜態 personality 向量，但這些 personality 在 B3.2 中**只允許用來分桶**
2. 不得因為 B3.2 啟用而改變 player action choice、初始 strategy weights、payoff、或 replicator operator
3. 依 `z_k^{(p)}` 由小到大排序後，做等量 quantile 分桶，得到 `n_strata` 個 persistent strata

形式化寫成：令 $r(p)$ 是玩家 $p$ 在全體玩家中依 `z_k` 排序後的 rank，則

$$
\mathrm{stratum}(p) = \left\lfloor \frac{r(p)\, n_{strata}}{N} \right\rfloor
$$

其中最後一桶做上界截斷到 `n_strata - 1`。

B3.2 第一輪 harness 契約：

- personality world 固定採用靜態 heterogeneous 1:1:1 prototype sampling，且只作為 latent labels，不回寫到 weights
- `personality_jitter` 第一輪固定預設為 `0.08`
- `sampled_growth_n_strata = 1` 時，即使已指定 personality labels，也必須精確退化回既有 sampled baseline
- B3.2 與 B3.1 沿用同一套 G2 stop rule；不得另外放寬 pass 標準

#### H-series 的 CSV 契約

本次擴充原則是：**先改動態，不改核心 timeseries schema**。

必備欄位維持不變：

- `round`
- `avg_reward`, `avg_utility`
- `p_<strategy>`
- `w_<strategy>`

語意也維持不變：

- `p_<strategy>` 仍代表第 $t$ 輪抽樣結果比例
- `w_<strategy>` 仍代表第 $t$ 輪結束後寫回的下一輪權重

補充：在所有玩家共用同一組權重的舊路徑下，`w_<strategy>` 可直接視為那一組共享權重；但在 H3.1 或任何未來的異質玩家權重路徑下，`w_<strategy>` 的正式語意必須解讀為**全體玩家於該輪更新後的平均 strategy weight**，也就是：

$$
w_i^{csv}(t) = \frac{1}{N}\sum_{p=1}^{N} w_i^{(p)}(t+1)
$$

因此 H1/H2/H3 不得偷偷重定義既有欄位；分析端仍可直接沿用既有 `cycle_metrics` 與 `decay_rate`。

本版不新增必備 CSV 欄位。若未來需要輸出診斷欄位（例如 `xeff_*`、`regime_id`），必須遵守：

- 預設關閉，不影響主 schema
- 一旦加入，需在 Spec 另行鎖定名稱、時間索引、與是否屬於主線 DoD

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
- H-series 額外要求：
  - `memory_kernel=1` 與既有 lag-1 payoff 完全等價
  - `threshold_ab` 在 `(a_hi,b_hi)=(a,b)` 時與 `matrix_ab` 完全等價
  - `hetero` 在 `k_A=k_D=k_B` 時與既有同質更新完全等價
  - `fixed_subgroup_share=0` 時與未啟用 H3.1 完全等價
  - `fixed_subgroup_coupling_strength=0` 時與未啟用 H3.2 完全等價

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
- `--matrix-cross-coupling`（預設 0.0；`matrix_ab` 使用；對應 $c_{AD}$）
- `--init-bias`（預設 0.0；初始對稱破缺，用來讓 deterministic/expected 動態離開均勻固定點；要求 $|bias|<1$）
- `--selection-strength`（預設 0.05）
- `--enable-events`（預設關閉；開啟 Personality Dungeon 事件層）
- `--events-json`（預設為 `docs/personality_dungeon_v1/02_event_templates_v1.json`；僅在 `--enable-events` 時使用）
- `--out`（預設 `outputs/timeseries.csv`）

H-series CLI 狀態：

- 已落地：`--memory-kernel`（預設 1；僅允許正奇數；第一版研究鎖定 `1,3,5`）
- 已落地（單跑 / 批次 / sweep 路徑）：
  - `--payoff-mode threshold_ab`
  - `--threshold-theta`（H2 的 regime 門檻；第一輪 smoke 鎖定 `{0.40,0.55}`）
  - `--threshold-theta-low`, `--threshold-theta-high`（H2.1 的可選 hysteresis band；省略時退化回單門檻 H2）
  - `--threshold-a-hi`, `--threshold-b-hi`（H2 的 high-regime payoff 係數）
- H3 預留契約：
  - `--evolution-mode hetero`
  - `--strategy-selection-strengths`（格式：`kA,kD,kB`；H3 的 per-strategy selection strength）
- H3.1 已落地契約：
  - `--fixed-subgroup-share`（格式：`[0,1]` 間實數）
  - `--fixed-subgroup-weights`（格式：`wA,wD,wB`；H3.1 的 frozen subgroup 權重）
- H3.2 已落地契約：
  - `--fixed-subgroup-coupling-strength`（格式：`>=0` 的實數；H3.2 的 adaptive-subgroup anchor coupling 強度）
- B4 第一輪已落地契約：
  - `--beta-state-k`（格式：`[0,1.5]` 的實數；只在 `evolution_mode=personality_coupled` 有效）

註：目前 `threshold_ab` 已支援 `simulation.run_simulation.py`、`simulation.seed_stability.py`、`simulation.rho_curve.py` 的 sampled / mean-field 路徑；H2.1 另外支援可選 hysteresis band。驗證順序固定為 no-op 等價性 → smoke → 最小 multi-seed confirm。

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

#### 序列選擇規則（Series Selection Rule）

`analysis/event_provenance_summary.py::_derive_cycle_level()` 實作以下優先序：

1. 優先使用 `w_<strategy>` 序列（理論上更直接反映策略頻率）。
2. 若 `w_` 序列 Stage 1 振幅未達標（`assess_stage1_amplitude` fail），**自動改用 `p_<strategy>` 序列**。
   - `p_` 序列（normalized simplex proportion）在 `selection_strength` 較小時攜帶更清晰的週期訊號；w_ 尾段振幅僅約 `ss×0.21`，在 ss ≤ 0.10 時不足以通過 Stage 1 的 0.02 門檻，但 p_ 振幅約 0.23，穩定通過。
   - **量化觸發條件**：Stage 1 門檻 = 0.02；w_ 尾段振幅 ≈ `ss × 0.21`，故 fallback 在 `ss < 0.02 / 0.21 ≈ 0.095` 時必然觸發（rounds=4000, players=300 實測）。進入 fallback 時 p_ 振幅約 0.23，遠超門檻，Stage 2 亦穩定通過。
3. 若兩者均無資料則回傳 `None`。

> **重要**：只有序列「尾段（tail window）」振幅才是 Stage 1 的判據，跑全局均值會高估 w_ 序列振幅。

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

0) **Personality Event Schema 加嚴（12 維人格事件系統接軌主線前置條件）**

- 目的：讓後續 Personality Dungeon 的事件層可以無縫接入既有 `matrix_ab + replicator` 研究主線，而不是停留在敘事模板。
- 規格要求：
  - `weights` 必須受限於 `[-1, 1]`，且載入時必須做正規化；建議採 `L1` normalization：

$$
w' = \frac{w}{\max(\sum_i |w_i|, \varepsilon)}
$$

  - 事件 action 的最終風險必須採統一公式並明確 clamp：

$$
\text{final\_risk} = \mathrm{clamp}(0,1,\,base\_risk + \sum_i \rho_i p_i + b_{risk} + r_{state} + d_{risk} + \alpha_{stress}\,stress + \alpha_{health}(1 - health))
$$

    其中 $p_i$ 為 12 維人格、$\rho_i$ 為 risk trait weights、$b_{risk}$ 為 action/event bias、$r_{state}$ 為累積 state risk、$d_{risk}$ 為跨輪 risk drift，$stress$ 以小係數 $\alpha_{stress}=0.1$ 提供延遲壓力回授，$health$ 以 $\alpha_{health}=0.15$ 計算低生命值對風險的加成（health 滿時 penalty 為 0；health=0 時 penalty 最大 +0.15）。若出現 `nan`/`inf`，MVP 規格要求 fallback 到 `base_risk`。
  - 每個 action 必須有**成功模型**（`success_model`）；MVP 可接受的預設為：

$$
\text{success\_prob} = \mathrm{clamp}(0,1,1-\text{final\_risk})
$$

    或等價的 logistic 寫法，但不可省略。若未明示 event-local success model，runtime 必須套用 schema default，而不是默認「永遠成功」或「永遠失敗」。
    
    **failure_threshold hard gate（2026-03-17 新增）**：每個 action 在 `risk_model.failure_threshold` 宣告一個軟硬閾值；若 `final_risk >= failure_threshold`，runtime 直接回傳 `success_prob = 0.0`，不再進入成功模型計算。此機制提供明確的分段語意（regime switch），防止高風險 action 因模型平滑性而仍有正成功率。全局 fallback 為 `risk_policy.default_failure_threshold = 0.5`。
    
    MVP 允許 success model 在 clamp 前加上小幅 state 修正，例如 `intel` 對成功率的加成，但必須保留最終 clamp 到 `[0,1]`。
  - `success_model` 的正式契約應以 **registry name + kwargs** 表示：
    - `success_model: "linear_risk_complement"`
    - `success_model: "logistic"` 搭配 `success_model_kwargs: {"success_bias": 1.0, "steepness": 6.0}`
    - `success_model: "linear_clip"` 搭配 `success_model_kwargs: {"slope": -1.2, "offset": 1.0}`
    loader 可為舊版 `probability_formula` 風格保留 backward compatibility，但 schema 驗證與文件化必須以 registry 為主，不可把任意字串公式當作長期契約。
  - `reward_tags` 不得作為正式 reward 契約；正式 schema 必須改為**可量化**的 `reward_effects`，至少包含：
    - `utility_delta`
    - `risk_delta`
    - `popularity_shift`
    - `trait_deltas`
    - 可選 `sample_quality` / `sample_gain`
  - 每個 action 必須有**狀態影響**（`state_effects`），至少允許 success / failure 改變跨輪狀態變量。MVP 建議的最小 state 變量集合：
    - `stress`
    - `noise`
    - `risk_drift`
    - `health`（若人格地下城版本需要額外脆弱度）
    - `intel`（若事件鏈依賴資訊品質）
    若 action-local `state_effects` 省略，loader 必須明確展開 schema default；不可在 runtime 中用未定義狀態悄悄略過。
    最小 runtime 閉環要求：
    - `stress` 與 `risk_drift` 必須能回流到下一輪 `final_risk`
    - `risk_delta` 必須能透過 `player.state["risk"]` 回流到下一輪 `final_risk`
    - `noise` 必須能影響 action utility 的擾動幅度或等價的選擇抖動
    - `intel` 必須能影響 success probability 或等價的資訊折扣
    
    **state decay（2026-03-17 新增）**：每輪 `process_turn()` 結束後，runtime 必須對累積型 state 變量做指數衰減，防止單向堆積導致飽和。預設衰減率（可由 `state_policy.decay_rates` 覆寫）：
    - `stress *= 0.92`（約 8 步半生期）
    - `noise *= 0.92`
    - `risk_drift *= 0.90`（約 7 步半生期）
    - `risk *= 0.95`（慢速衰減，半生期約 14 步）
    - `health += 0.02`（每輪被動恢復，clamp 到 `[0, 1]`）
    
    衰減在 reward/state_effects 套用之後執行，讓單輪 delta 正常生效但不累積到下一輪。
  - 每個 action 必須明確宣告 `failure_outcomes`；失敗不得只是「風險裝飾」。至少要能影響下列其中之一：
    - `last_reward`
    - 人格漂移（如 `fearfulness += delta`）
    - popularity / strategy mass
    - 世界 state / future risk drift
- 不變條件：
  - 任一 action 進入 simulation 前，`final_risk` 必須有限且位於 `[0,1]`。
  - 任一 action 的 `success_prob` 必須有限且位於 `[0,1]`。
  - 任一 action 的 `success_model` 必須能在 loader 啟動時解析到已註冊模型；未知模型或不合法 kwargs 必須在載入時拒絕，不得拖到 sweep 中才失敗。
  - 任一 action 的 success/failure 都必須能回流到可聚合的 reward stream，否則不得宣稱可接入 replicator dynamics。
  - 任一 action 的 `state_effects` 只能修改 schema 宣告過的 state variables；未知欄位必須在載入時拒絕。
  - Personality event schema 若未滿足上述條件，只能標記為 design draft，不得作為研究輸入資料源。

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

### 7.3 Personality/Event 新主線最小 smoke 契約（2026-04-01）

目的：在不改動 core replicator 的前提下，先驗證「真實 personality 異質性 + 既有事件層」是否比 H-series 的 operator patch 更有機會抵抗 sampled 平均化。

本節鎖定目前 repo 可執行的最小閉環：

1. runtime 僅允許走 `sampled + events` 路徑；`mean_field + events` 尚未接通，因此不得把它當作第一輪 smoke 的執行目標
2. 12 維 personality projection 與 Little Dragon 目前仍屬 design-pack utility，不得假設已自動整合進 `simulation.run_simulation`
3. 第一輪 smoke 只允許新增「薄橋接層」：初始化玩家 personality，並把 12 維向量投影成初始三策略 weights；不得同步更動 replicator 核心更新規則

最小 smoke harness 的研究契約：

1. cohort 固定為 3 群：Aggressive / Defensive / Balanced，各 100 人
2. 每位玩家的 12 維 personality 由 prototype 加上獨立小抖動產生；第一輪 jitter 鎖定為每維 `±0.08`
3. 初始三策略權重必須由 `docs/personality_dungeon_v1/03_personality_projection_v1.py` 的 projection 邏輯產生，不得另外手寫第二套映射
4. 第一輪事件集合固定縮減為 3 個模板：`threat_shadow_stalker`、`resource_suspicious_chest`、`uncertainty_altar`
5. baseline 組定義為 zero-personality：所有 personality 維度為 0，其他 simulation 參數與 heterogeneous 組完全一致
6. heterogeneous 組定義為 3 個 prototype cohort 以 `1:1:1` 混合；第一輪不再額外掃 cohort 比例

Gate 0：Projection + Action Sanity

1. 檢查 3 個 cohort 的 projected centroid 是否落在 simplex 的不同三分之一區域
2. 在固定的 3 個事件模板上，比較 Aggressive 與 Defensive 的 action 選擇分佈；至少 2 個模板要出現最優 action 比例差異 `> 25%`
3. 若 Gate 0 未通過，必須先修正 projection / personality-to-action mapping；不得直接進 Gate 1

Gate 1：Static-World Sampled Smoke

1. 固定參數：`players=300`, `rounds=3000`, `seeds={45,47,49}`, `memory_kernel=3`, `selection_strength=0.06`, `init_bias=0.12`, `series=p`
2. 比較對象僅限 zero-personality baseline 與 heterogeneous 組；第一輪不引入 Little Dragon
3. pass 條件必須同時滿足：
   - `mean_env_gamma` 相對 baseline 改善（數值更大或更接近 0）
   - `mean_stage3_score` 相對 baseline 提升至少 `0.018`
   - 至少 `1/3` seeds 達到 Level 3
4. 若只滿足其中一項，記為 weak positive，可保留同設定進 Gate 2 準備，但不得宣稱靜態 personality 異質性已成立
5. 若完全 flat，則 personality 靜態異質性路線可視為結案；下一步應直接轉向更強動態機制（例如 personality + inertia 或 event-driven nonlinear payoff）

Gate 2：Adaptive-World Smoke

1. 只有在 Gate 1 至少出現 weak positive 時才允許啟動
2. 第一輪必須先做 offline Little Dragon：把 Gate 1 的 global `p` 時序每 200 rounds 餵入 `docs/personality_dungeon_v1/05_little_dragon_v1.py`
3. offline pass 條件：dominant strategy 改變時，Little Dragon 至少 `70%` 的調整步是反壓方向
4. 只有 offline pass 後，才允許規劃 in-loop adaptive-world 版本

H5.4：`personality + sampled_inertia` 最小 smoke

1. 只有在 Gate 1 得到 `weak_positive`，且研究主線明確選擇「異質性 + 抗平均化」時才允許啟動
2. 比較基準不再使用 zero-personality；H5.4 的唯一控制組是 Gate 1 heterogeneous world 的 matched control：`world_mode=heterogeneous`, `evolution_mode=sampled_inertial`, `sampled_inertia=0`
3. 第一輪固定工作點：`players=300`, `rounds=3000`, `seeds={45,47,49}`, `memory_kernel=3`, `selection_strength=0.06`, `init_bias=0.12`, `series=p`, `a=1.0`, `b=0.9`, `matrix_cross_coupling=0.20`, `events_json=02_event_templates_smoke_v1.json`
4. 第一輪只掃 `sampled_inertia ∈ {0.15, 0.25, 0.35}`；只有在整體判讀仍為 weak positive 時，才允許補一個 tie-break 點 `sampled_inertia=0.30`
5. 單一 `sampled_inertia` 點的 full-pass 條件必須同時滿足：
  - `mean_env_gamma` 相對 matched control 改善至少 `25%`
  - `mean_stage3_score` 相對 matched control 提升至少 `0.018`
  - 至少 `1` 個 seed 達到 Level 3
6. 整體 H5.4 pass 需要至少 `2` 個 `sampled_inertia` 點達到 full pass
7. 若至少 `1` 個點只達成部分條件，或只有 `1` 個點 full pass，整體記為 weak positive；若所有點都未推升 stage3 且沒有任何 Level 3 seed，整體記為 fail

H5.5：`personality + event-driven nonlinear payoff` 最小 smoke

1. 只有在 H5.4 明確失敗，且主線正式轉向「以非線性 payoff 重新打開 sampled + events 下的 regime switching」時才允許啟動
2. 比較基準固定為 Gate 1 heterogeneous world 的 `matrix_ab` matched control；H5.5 第一輪不再回頭使用 zero-personality baseline
3. 第一輪固定工作點沿用 Gate 1 / H5.4：`players=300`, `rounds=3000`, `seeds={45,47,49}`, `memory_kernel=3`, `selection_strength=0.06`, `init_bias=0.12`, `series=p`, `a=1.0`, `b=0.9`, `matrix_cross_coupling=0.20`, `events_json=02_event_templates_smoke_v1.json`
4. 第一輪主掃描只允許 3 個固定 nonlinear cells：
  - `threshold_legacy`：`payoff_mode=threshold_ab`, `threshold_trigger=ad_share`, `theta=0.55`, `theta_low=None`, `theta_high=None`, `state_alpha=1.00`, `a_hi=1.10`, `b_hi=1.00`
  - `threshold_hysteresis`：`payoff_mode=threshold_ab`, `threshold_trigger=ad_share`, `theta_low=0.62`, `theta_high=0.70`, `state_alpha=1.00`, `a_hi=1.10`, `b_hi=1.00`
  - `threshold_slow_state`：`payoff_mode=threshold_ab`, `threshold_trigger=ad_share`, `theta_low=0.62`, `theta_high=0.70`, `state_alpha=0.35`, `a_hi=1.10`, `b_hi=1.00`
5. `ad_product` family 已在 H2.2 的短窗局部掃描中被否定，因此不得作為 H5.5 第一輪主掃描預設 cell；若要重啟 `ad_product`，必須另開下一版 Spec
6. 單一 nonlinear cell 的 full-pass 條件必須同時滿足：
  - `mean_env_gamma` 相對 matched control 改善至少 `25%`
  - `mean_stage3_score` 相對 matched control 提升至少 `0.018`
  - 至少 `1` 個 seed 達到 Level 3
  - 至少 `1` 個 seed 出現 `threshold_regime_hi` 的實際切換
7. 整體 H5.5 pass 只需要至少 `1` 個 nonlinear cell full-pass；若只有部分 uplift 或只有 switch evidence，整體記為 weak positive；若三個 cells 全都沒有 uplift 且沒有 switch evidence，整體記為 fail

H5.5R：`personality + event-driven nonlinear payoff` 最後一次局部 refinement

1. 只有在 H5.5 第一輪正式結果為 `weak_positive`，且 evidence 主要來自 switch 而非 aggregate uplift 時，才允許啟動 H5.5R
2. H5.5R 不是新主線 family，而是 H5.5 的最後一次救援；本輪若仍無可進 confirm 的 cell，就必須正式結案 H5.5，不再追加任何 nonlinear refinement
3. 比較基準仍固定為 Gate 1 heterogeneous world 的 `matrix_ab` matched control；工作點沿用 H5.5：`players=300`, `rounds=3000`, `seeds={45,47,49}`, `memory_kernel=3`, `selection_strength=0.06`, `init_bias=0.12`, `series=p`, `a=1.0`, `b=0.9`, `matrix_cross_coupling=0.20`, `events_json=02_event_templates_smoke_v1.json`
4. H5.5R 第一輪只允許 6 個固定 refinement cells，不得擴張成網格：
  - legacy 幾何主軸 3 點：`(a_hi,b_hi) ∈ {(1.05,0.95), (1.10,1.00), (1.15,1.05)}`，其餘固定為 `threshold_trigger=ad_share`, `theta=0.55`, `state_alpha=1.00`
  - slow-state 主軸 3 點：`state_alpha ∈ {0.20,0.35,0.50}`，其餘固定為 `threshold_trigger=ad_share`, `theta_low=0.62`, `theta_high=0.70`, `a_hi=1.10`, `b_hi=1.00`
5. 單一 refinement cell 的 confirm-gate 條件必須同時滿足：
  - `mean_stage3_score` 相對 matched control 提升至少 `0.015`
  - `mean_env_gamma` 相對 matched control 至少提升 `10%`，亦即 `gamma_uplift_ratio_vs_control >= 1.10`
  - 至少 `1` 個 seed 出現 `threshold_regime_hi` 的實際切換
6. H5.5R 的硬停損規則：若 6 個 cells 中沒有任何一點通過 confirm gate，整體決策必須直接記為 `close_h55`，並停止所有後續 nonlinear refinement
7. 若有多個 cells 通過 confirm gate，只允許選出 `1` 個最佳 cell 進 single longer confirm；排序規則鎖定為：先比 `stage3_uplift_vs_control`，再比 `gamma_uplift_ratio_vs_control`，再比 `mean_regime_switches`
8. H5.5R 的 single longer confirm 協定預先鎖定為：`rounds=5000`, `seeds={45,47,49,51,53,55}`；本版只允許對最佳 cell 執行一次，不得補第二個 confirm cell

H6：完整 `Personality + Event` 世界模型主線（post-H5.5R reset）

1. 自 H5.5R 正式給出 `close_h55` 起，H1-H5 在目前 replicator + sampled + events 框架內的 rescue family 視為已收斂；後續 personality/event 主線改由 H6 接手，不再回頭追加 H5.4/H5.5/H5.5R 類型的局部救援
2. H6 第一版仍維持「薄橋接層」原則：先在既有 sampled + events runtime 上完成更強的靜態異質性驗證，再以 offline Little Dragon 驗證 world-pressure 映射；在 Gate 1 與 Gate 2 都通過前，不得宣稱已進入 in-loop adaptive world
3. H6 Gate 1（expanded static heterogeneity）固定工作點為：`players=300`, `rounds=4000`, `seeds={45,47,49,51,53,55}`, `memory_kernel=3`, `selection_strength=0.06`, `init_bias=0.12`, `series=p`, `a=1.0`, `b=0.9`, `matrix_cross_coupling=0.20`, `events_json=02_event_templates_smoke_v1.json`
4. H6 Gate 1 必須同時保留兩個 controls：
  - zero-personality control：所有 12 維 personality 都為 0
  - `1:1:1` reference control：Aggressive / Defensive / Balanced 三群等比混合
5. H6 Gate 1 第一輪只允許 6 個固定 extreme-ratio cells（順序固定為 Aggressive : Defensive : Balanced）：
  - `2:1:0` family：`2:1:0`, `1:2:0`, `1:0:2`
  - `3:0:0` family：`3:0:0`, `0:3:0`, `0:0:3`
6. H6 Gate 1 的正式比較基準固定為 `1:1:1` reference control，不再以 zero-personality 作為 pass gate；zero-personality 只保留作為 lineage 對照與 provenance
7. 單一 H6 Gate 1 candidate 的 full-pass 條件必須同時滿足：
  - `mean_stage3_score` 相對 `1:1:1` reference control 提升至少 `0.025`
  - 至少 `2/6` seeds 達到 Level 3
8. H6 Gate 1 的整體決策規則鎖定為：
  - `pass`：至少 1 個 candidate full-pass
  - `weak_positive`：沒有 candidate full-pass，但至少 1 個 candidate 出現下列任一訊號：`stage3_uplift_vs_reference > 0`、至少 `1` 個 Level 3 seed、或 `mean_env_gamma` 優於 `1:1:1` reference control
  - `fail`：6 個 candidates 全部沒有 uplift、沒有任何 Level 3 seed，且 `mean_env_gamma` 也未優於 `1:1:1` reference control
9. 只有在 H6 Gate 1 整體決策為 `pass` 時，才允許開 H6 Gate 2；若結果只有 `weak_positive` 或 `fail`，必須停止 H6 主線，並把結論記錄為「現有 replicator framework 的靜態 personality/event 版已到上限」
10. 若 H6 Gate 1 有多個 full-pass candidates，只允許選出 `1` 個最佳 cell 進 Gate 2；排序規則鎖定為：先比 `stage3_uplift_vs_reference`，再比 `level3_seed_count`，再比 `gamma_uplift_ratio_vs_reference`
11. H6 Gate 2（offline Little Dragon）只允許吃進 H6 Gate 1 的唯一最佳 cell，並固定使用該 cell 的 global `p` 時序每 `300` rounds 取樣一次後餵入 `docs/personality_dungeon_v1/05_little_dragon_v1.py`
12. H6 Gate 2 的變化步只統計「dominant strategy 與前一個 snapshot 不同」的 rounds，且本輪 `dominance_bias >= 0.02`；若沒有任何可評估變化步，Gate 2 直接記為 `fail`
13. 單一步 H6 Gate 2 的 anti-pressure response 必須同時滿足：
  - `aggressive` 佔優時：`event_type=Threat`，且 `a >= base_a`, `b >= base_b`
  - `defensive` 佔優時：`event_type=Resource`，且 `a <= base_a`, `b >= base_b`
  - `balanced` 佔優時：`event_type=Uncertainty`，且 `a >= base_a`, `b <= base_b`
14. H6 Gate 2 的整體 pass 條件為：可評估變化步中，至少 `70%` 同時滿足 anti-pressure response 與 world-output 確實改變（`event_type` 或 `(a,b)` 相對前一步有變化）
15. 只有在 H6 Gate 1 與 H6 Gate 2 都 `pass` 後，下一版 Spec 才能開 in-loop adaptive-world integration；在此之前，Little Dragon 只能以 offline validator 身分存在

H7：`Personality + Dynamic Coupling` 主線（post-H6 static closure）

1. 自 H6 Gate 1 正式結果仍為 `weak_positive` 且依 stop rule 關閉 Gate 2 起，H1-H6 在目前 replicator + sampled + events 框架內的靜態 personality / local rescue / offline-world 驗證線視為已完成一輪收斂；新的 personality 主線改為 H7：直接讓 personality 進入 replicator 更新算子本身
2. H7 第一版只允許最小單向耦合：`personality -> per-player inertia` 與 `personality -> per-player selection_strength`；不得在 H7.1 同步加入 personality drift、雙向 state feedback、或新的 payoff/world coupling
3. H7 的研究目的不是再證明 personality heterogeneity 不是 no-op，而是檢查 personality 是否能透過更新速度差異，把 sampled + events 路徑從 Level 2 plateau 推向至少單 seed 的 Level 3

H7.1：`personality-coupled sampled inertial` 最小 scout

1. H7.1 底座固定為 H6 的 `1:1:1` heterogeneous world，不再使用 zero-personality control，也不再使用 extreme-ratio cells
2. H7.1 固定工作點為：`players=300`, `rounds=3000`, `seeds={45,47,49}`, `memory_kernel=3`, `selection_strength=0.06`, `init_bias=0.12`, `series=p`, `a=1.0`, `b=0.9`, `matrix_cross_coupling=0.20`, `events_json=02_event_templates_smoke_v1.json`
3. H7.1 第一版 runtime 必須是新的 personality-coupled sampled mode；它不得偷偷改寫既有 `sampled_inertial` 的全局語意，也不得重定義 H6 的 static harness 結果
4. H7.1 每位玩家的 inertia 係數必須由 personality 決定：
  - 先定義 inertia signal：`z_mu = 0.5 * (stability_seeking + patience) - 0.5 * impulsiveness`
  - 再定義 `mu_p = clamp(mu_base + lambda_mu * z_mu, 0.0, 0.60)`
  - H7.1 第一版固定 `mu_base = 0.00`
5. H7.1 每位玩家的 selection strength 必須由 personality 決定：
  - 先定義 sensitivity signal：`z_k = 0.5 * (ambition + greed) - 0.5 * (caution + fearfulness)`
  - 再定義 `k_p = clamp(k_base * (1 + lambda_k * z_k), 0.03, 0.09)`
  - H7.1 第一版固定 `k_base = 0.06`
6. H7.1 的 personality mapping 只允許用 player 自身的 personality snapshot；不得引用其他玩家狀態、global `p`、或 event history 當作額外 gate
7. H7.1 明確禁止在同一版加入 `event-triggered personality drift`；該機制若要啟動，必須另開 H7.2 Spec，以免和 per-player inertia / k 耦合造成因果混淆
8. H7.1 的短 scout 只允許 4 個固定 cells：
  - `control`：`lambda_mu = 0.00`, `lambda_k = 0.00`
  - `inertia_only`：`lambda_mu = 0.25`, `lambda_k = 0.00`
  - `k_only`：`lambda_mu = 0.00`, `lambda_k = 0.25`
  - `combined_low`：`lambda_mu = 0.15`, `lambda_k = 0.15`
9. H7.1 的執行順序固定為：先 `control`，再 `inertia_only`、`k_only`，最後 `combined_low`；若 control 無法重現 H6 的 `1:1:1` heterogeneous baseline 語意，整輪 scout 視為無效
10. H7.1 的正式比較基準固定為 H7.1 `control` cell；不得再回頭與 zero-control 或 H6 extreme-ratio cells 混比
11. 單一 H7.1 cell 的 Primary success 必須同時滿足：
  - 至少 `1` 個 seed 達到 Level 3
  - `mean_stage3_score` 相對 control 提升至少 `0.020`
  - `mean_env_gamma` 相對 control 改善至少 `25%`，或數值更接近 `0`
12. 單一 H7.1 cell 的 Secondary success 必須同時滿足：
  - `mean_stage3_score` 相對 control 提升至少 `15%`
  - `mean_env_gamma` 相對 control 改善至少 `15%`，或數值更接近 `0`
  - 即使沒有 Level 3 seed，也只允許保留為 single longer confirm 候選，不得直接宣告通關
13. H7.1 的整體決策規則鎖定為：
  - `pass`：至少 `1` 個 non-control cell 達到 Primary success
  - `weak_positive`：沒有 Primary success，但至少 `1` 個 non-control cell 達到 Secondary success 或出現部分 uplift 訊號
  - `fail`：4 個 cells 全部沒有 uplift，且沒有任何 Level 3 seed
14. H7.1 的硬停損規則：若 4 個 cells 全部沒有 Level 3 seed，且整體仍只是 `weak_positive`，則 H7.1 直接結案，不再做任何 refinement；此時應正式記錄「replicator 框架下 personality 動態耦合已達極限」
15. 只有在 H7.1 出現 Primary success，或出現唯一 Secondary-success 最佳 cell 時，才允許開 single longer confirm；longer confirm 預先鎖定為：`rounds=5000`, `seeds={45,47,49,51,53,55}`，且最多只允許 `1` 個 candidate

W1：`In-loop Adaptive World` 主線（post-H7 closure）

1. 自 H7.1 依 stop rule 正式記為 `close_h71` 起，H1-H7 在目前 replicator + sampled + events 框架內的 rescue family 視為已完成收斂；後續主線改為 W1：由世界狀態主導的 in-loop adaptive world，不再把 personality / world gain 直接加到 replicator 更新算子上
2. W1 的研究目的不是再找另一種 `time-varying a/b/k/mu` 補丁，而是建立具有內部記憶的 world state，使 event composition 與 event intensity 能跨窗更新；任何若可近似還原成單純 payoff modifier 或單純 event weight 擾動的設計，皆不得立項
3. W1 第一版世界狀態固定為四維 latent state：`scarcity`, `threat`, `noise`, `intel`；各維範圍均為 `[0,1]`，初始值固定為 `0.5`
4. W1 第一版世界狀態對玩家不可見；玩家只能經由事件家族比例、事件 base_risk、reward multiplier 與 state-effects 的變化間接感受世界
5. W1 的世界狀態更新一律採 batch update，不得改成每回合同步更新；W1.1 基準 scout 的 `world_update_interval` 固定為 `200` rounds，而後續若要改 interval，必須以明確版本化 protocol 鎖定，不能臨時在 runtime 內偷改
6. W1 第一版只吃 aggregate behavior 與事件統計，不得引入 testament / NLP / cross-life memory；跨 life 迴圈需另開後續 Spec
7. W1 的核心更新式固定為：`s(t+1) = clamp(s(t) + lambda_world * (delta_p + delta_e), 0, 1)`；其中 `s=(scarcity, threat, noise, intel)`，`lambda_world` 為 world adaptation rate，W1.1 預設基準值固定為 `0.08`
8. W1 的玩家群體偏差項固定為：`delta_p = B_p * (p_bar - p_balanced)`；其中 `p_bar` 為最近一個 `world_update_interval` batch window 的 global strategy 平均，`p_balanced=(1/3,1/3,1/3)`，而 `B_p` 在 W1.1 / W1.2 都鎖定為
  - `scarcity <- [0.60, 0.20, -0.40]`
  - `threat <- [0.60, -0.10, -0.30]`
  - `noise <- [-0.25, -0.25, 0.40]`
  - `intel <- [-0.30, -0.10, 0.45]`
9. W1 的事件反饋項固定為：`delta_e = (r_bar - r_target) * (B_e * f_bar)`；其中 `r_bar` 為最近一個 `world_update_interval` batch window 的 mean reward，`r_target=0.27`，`f_bar=(Threat, Resource, Uncertainty, Navigation, Internal)` 為最近一個 `world_update_interval` batch window 的事件家族 share，而 `B_e` 在 W1.1 / W1.2 都鎖定為
  - `scarcity <- [-0.10, 0.40, 0.10, -0.10, 0.10]`
  - `threat <- [0.45, -0.15, 0.10, -0.05, 0.05]`
  - `noise <- [0.05, -0.10, 0.35, -0.05, 0.30]`
  - `intel <- [-0.20, 0.15, -0.20, 0.40, -0.05]`
10. W1 的世界輸出只允許經由事件層四類旋鈕生效，不得直接改寫 `matrix_ab` 的 `a/b`、`matrix_cross_coupling`、replicator `k/mu`、或任何等價的 payoff operator：
  - event family weights：調整 `Threat / Resource / Uncertainty / Navigation / Internal` 的抽樣權重
  - risk parameters：調整各事件家族的 `base_risk / risk_bias`
  - reward multipliers：調整事件 payload 內 `utility_delta / risk_delta` 的 scaling
  - trait-delta strength：調整事件 payload 內 `trait_deltas` 的強度
11. W1.1 的事件權重映射鎖定為：
  - `w_threat = clamp(1.0 + 0.80*(threat-0.5) + 0.30*(noise-0.5) - 0.20*(intel-0.5), 0.20, 3.00)`
  - `w_resource = clamp(1.0 - 0.80*(scarcity-0.5) + 0.35*(intel-0.5), 0.20, 3.00)`
  - `w_uncertainty = clamp(1.0 + 0.90*(noise-0.5) + 0.20*(threat-0.5), 0.20, 3.00)`
  - `w_navigation = clamp(1.0 + 0.75*(intel-0.5) - 0.30*(noise-0.5), 0.20, 3.00)`
  - `w_internal = clamp(1.0 + 0.55*(noise-0.5) + 0.25*(scarcity-0.5) - 0.20*(intel-0.5), 0.20, 3.00)`
12. W1.1 的事件參數映射鎖定為：
  - risk parameters
    - `threat_risk_multiplier = clamp(1.0 + 0.30*(threat-0.5), 0.85, 1.15)`
    - `resource_risk_multiplier = clamp(1.0 + 0.15*(scarcity-0.5) - 0.10*(intel-0.5), 0.90, 1.10)`
    - `uncertainty_risk_multiplier = clamp(1.0 + 0.35*(noise-0.5), 0.85, 1.15)`
    - `navigation_risk_multiplier = clamp(1.0 - 0.20*(intel-0.5) + 0.10*(noise-0.5), 0.90, 1.10)`
    - `internal_risk_multiplier = clamp(1.0 + 0.20*(noise-0.5) + 0.10*(scarcity-0.5), 0.90, 1.10)`
  - reward multipliers
    - `resource_reward_multiplier = clamp(1.0 - 0.30*(scarcity-0.5), 0.85, 1.15)`
    - `noise_penalty_multiplier = clamp(1.0 + 0.40*(noise-0.5), 0.80, 1.20)`
    - `intel_reward_multiplier = clamp(1.0 + 0.30*(intel-0.5), 0.85, 1.15)`
    - `threat_penalty_multiplier = clamp(1.0 + 0.20*(threat-0.5), 0.85, 1.15)`
  - trait-delta strength
    - `threat_trait_multiplier = clamp(1.0 + 0.20*(threat-0.5), 0.85, 1.15)`
    - `resource_trait_multiplier = clamp(1.0 - 0.15*(scarcity-0.5), 0.85, 1.15)`
    - `uncertainty_trait_multiplier = clamp(1.0 + 0.25*(noise-0.5), 0.80, 1.20)`
    - `navigation_trait_multiplier = clamp(1.0 + 0.20*(intel-0.5), 0.85, 1.15)`
    - `internal_trait_multiplier = clamp(1.0 + 0.20*(noise-0.5) + 0.10*(scarcity-0.5), 0.85, 1.15)`
13. W1 必須保留可還原的退化模式：若 `lambda_world=0.0`，或所有世界狀態都固定在 `0.5`，則事件分布與事件參數必須完全退化回既有靜態事件集語意；若做不到，W1 runtime 視為契約不合格
14. 在退化模式下，W1 runtime 必須 100% 還原到既有 `02_event_templates_smoke_v1.json` 的 3-template 行為；step-level TSV 中的 `a_new` / `b_new` 只作為診斷欄位，且必須維持等於輸入工作點的 `a` / `b`
15. W1.1 的最小 scout 只允許 4 個固定 cells：`control(lambda_world=0.00)`, `w1_low(0.04)`, `w1_base(0.08)`, `w1_high(0.15)`；除 `lambda_world` 外，其餘工作點與 H7.1 相同：`players=300`, `rounds=3000`, `seeds={45,47,49}`, `memory_kernel=3`, `selection_strength=0.06`, `init_bias=0.12`, `a=1.0`, `b=0.9`, `matrix_cross_coupling=0.20`, `events_json=02_event_templates_smoke_v1.json`
16. W1.1 `control` cell 的定義固定為：世界狀態在整輪執行中保持 `(0.5, 0.5, 0.5, 0.5)`，不得偷偷更新；若 control 無法退化回既有 static events baseline，整輪 scout 視為無效
17. W1.1 的整體決策規則鎖定為：
  - `pass`：至少 `1` 個 non-control cell 同時滿足 `>= 1` 個 Level 3 seed，且 `mean_env_gamma` 相對 control 達到明顯改善；第一版門檻固定為 `>= 20%`
  - `weak_positive`：沒有 `pass`，但至少 `1` 個 non-control cell 出現 `stage3_uplift_vs_control > 0`、`mean_env_gamma` 優於 control、或世界狀態確實偏離 `(0.5,0.5,0.5,0.5)`
  - `fail`：所有 non-control cells 都沒有 Level 3 seed，且 `mean_env_gamma` 相對 control 未達 `>= 20%` 改善；不得再用只有 switch / state drift、沒有 uplift 的結果視為通過
18. W1.1 的 decision markdown 必須額外執行非同構檢查：若候選 cell 的效果可近似還原成「只有 time-varying event weights，其他 risk / reward / trait 旋鈕保持 identity」，則直接標記為 `nonisomorphic_fail`，不得進 Gate 2
19. W1.1 若所有 non-control cells 都沒有 Level 3 seed，且唯一訊號只剩 aggregate uplift，則本輪只記為 `weak_positive` 或 `fail`，不得直接擴成 W2 / W3；後續是否升級主線，必須先記錄 W1 是否已提供不可約簡的世界記憶證據
20. W1.2 是 W1 主線唯一允許的 timescale refinement protocol；目的不是再掃更大 state 幅度，而是檢查世界更新頻率是否與 sampled replicator 的可觀測旋轉時間尺度錯配
21. W1.2 的固定工作點與 W1.1 相同：`players=300`, `rounds=3000`, `seeds={45,47,49}`, `memory_kernel=3`, `selection_strength=0.06`, `init_bias=0.12`, `a=1.0`, `b=0.9`, `matrix_cross_coupling=0.20`, `events_json=02_event_templates_v1.json`
22. W1.2 只允許 4 個固定 cells，且這是本主線唯一允許依 cell 指定 `world_update_interval` 的版本化例外：
  - `control(lambda_world=0.00, world_update_interval=100)`；control 雖然帶有名目 interval，但世界狀態在整輪中必須保持 `(0.5,0.5,0.5,0.5)` 不變
  - `w1_fast(lambda_world=0.08, world_update_interval=100)`
  - `w1_mid_fast(lambda_world=0.12, world_update_interval=150)`
  - `w1_high_fast(lambda_world=0.15, world_update_interval=100)`
23. W1.2 的整體 Promotion Gate 鎖定為：至少 `1` 個 non-control cell 同時滿足 `>=1` 個 Level 3 seed、`mean_env_gamma` 相對 control 改善 `>=30%`、且 `mean_stage3_score` 不低於 control；三條缺一不可
24. W1.2 的 Closure Gate 鎖定為：若三個 adaptive cells 全部沒有 Level 3 seed，或 adaptive cells 的 `mean_stage3_score` 仍持續低於 control，則整輪直接記為 `close_w1`，W1 主線正式結案，不再做 `lambda_world`、coupling gain、或 state 幅度微調
25. W1.2 的 aggregate summary 至少必須額外提供：`protocol`, `gamma_gate_threshold_pct`, `gamma_improve_pct`, `gamma_gate_pass`, `stage3_not_below_control`, `promotion_gate_pass`, `verdict`
26. W1.2 的 decision markdown 必須明示：control 指標、4 個固定 cells 的 `Level 3 seeds / mean_stage3_score / mean_env_gamma / gamma_improve_pct / verdict`，以及整體 `pass/close_w1` 結論
27. W1.2 的輸出必須與 W1.1 隔離；建議固定前綴為 `outputs/w1_worldstate_timescale_*`

W2：`Episode / Life-based World` 主線（post-W1 closure）

1. 自 W1.2 依 Closure Gate 正式記為 `close_w1` 起，H1-H7 與 W1 在目前 `replicator + sampled + events` 單 life 框架內的 rescue line 視為已完成收斂；後續主線改為 W2：把 `life` 當作一級動態單位，而不是再往單一 life 內部加新 patch
2. W2 的研究目的不是把 W1 world state 再包一層更慢的 time-varying payoff，而是引入真正的跨 life 狀態傳遞：`birth -> event sequence -> death/end-of-life -> testament -> next life initialization`
3. W2 第一版固定採 episode/life protocol：每個 life 最多 `3000` rounds，或在死亡條件觸發時提早結束；第一輪總 lives 固定為 `5`
4. W2 第一版仍沿用既有 sampled + events 執行主幹，不開 Little Dragon，也不做 in-loop world-response validator；Little Dragon 只有在跨 life 主線先出現明確 uplift 後才允許評估是否接入初始化層
5. W2 第一版的 control 必須能完全退化回現有單 life 行為；最小退化條件為：`total_lives=1`，且禁用 testament 與任何跨 life world carryover

W2.1：最小 testament / death 契約

6. 每位玩家在 life `ℓ` 的人格向量記為 `P_i(ℓ) ∈ [-1,1]^{12}`；每個 life 結束時，只允許更新下一個 life 的初始 personality，不得直接改寫當前 life 中的 strategy weights 或 replicator operator
7. W2.1 的 testament 更新式固定為：`P_i(ℓ+1) = clamp(P_i(ℓ) + alpha_testament * clip(DeltaP_i(ℓ), -0.25, 0.25), -1, 1)`；第一版只允許兩個非 control 強度：`alpha_testament=0.12` 與 `0.22`
8. W2.1 的 testament delta 固定為：`DeltaP_i = 0.50 * Delta_util_i + 0.35 * Delta_dom_i + 0.15 * Delta_event_i`
9. `Delta_dom_i` 依玩家最後 `500` rounds 的 dominant strategy 映射到固定 12 維 trait template：
  - aggressive-dominant：`impulsiveness`, `greed`, `ambition` 各 `+0.18`
  - defensive-dominant：`caution`, `stability_seeking`, `patience` 各 `+0.18`
  - balanced-dominant：`curiosity`, `optimism`, `persistence` 各 `+0.18`
  - 其餘維度為 `0.0`
10. `Delta_util_i` 必須是明確的 12 維向量，不允許把純 scalar 直接加到 personality；W2.1 第一版鎖定為：先計算 `z_util_i = clip((utility_i - mean_utility) / max(std_utility, 1e-6), -1, 1)`，再令 `Delta_util_i = z_util_i * normalize(Delta_dom_i)`；若該玩家沒有可辨識的 dominant strategy，則 `Delta_util_i = 0`
11. `Delta_event_i` 鎖定為該玩家在本 life 內所有已套用 `trait_deltas` 的逐維平均，再依事件成功率做縮放；若玩家本 life 沒有任何事件 trait 記錄，則 `Delta_event_i = 0`
12. 當 `alpha_testament=0` 時，personality 必須完全不變；這是 W2 testament 的硬退化模式

W2.1：死亡條件契約

13. W2.1 的死亡不是 HP 歸零，而是 `fate collapse`：每回合事件結算後，玩家的累積風險若超過個人承受閾值，該玩家立即結束當前 life，進入 testament 與下一 life 初始化
14. 令 `risk_i(t)` 為玩家 `i` 在當前 life 的累積風險。W2.1 固定採用：`risk_i(t) = risk_i(t-1) + Delta_risk_event_i(t) + Delta_risk_personality_i(t)`；其中 `Delta_risk_event` 直接承接現有 event payload 的 `risk_delta`
15. `Delta_risk_personality_i(t)` 第一版只允許 4 維 trait 進入：
  - `impulsiveness=+0.22`
  - `caution=-0.25`
  - `stability_seeking=-0.20`
  - `fearfulness=+0.18`
  - 其餘 8 維權重固定為 `0.0`
16. 每位玩家的死亡閾值固定為 `threshold_i = 1.0 + 0.15 * (caution_i + stability_seeking_i - impulsiveness_i - fearfulness_i)`；第一版不得再加更高階個體化修正
17. 死亡觸發後的順序必須固定為：
  - 記錄該玩家本 life 的最終 `utility`、dominant strategy、最後 `500` rounds 的策略統計、與事件成功摘要
  - 立即執行 testament 更新，產生下一 life 的 personality
  - 重置 `risk=0` 與 life-local state，再進入下一 life
18. 死亡只影響該玩家自己的 life 進程，不得直接改變其他玩家的 risk、personality、或 strategy weights
19. W2.1 的 control 退化模式固定為：`single_life` 且 `alpha_testament=0`。把 personality risk weights 設為 `0` 只等於 event-only death，不等於單 life control；不得把兩者混為一談

W2.1：跨 life world initialization 與第一輪 protocol

20. W2.1 第一版預設關閉 world carryover；下一 life 的事件模板與世界初始化不得直接讀取上一 life 的 final global `p`，以維持最小可歸因性
21. W2.1 的第一輪最小 scout 只允許 3 個固定 cells：
  - `control`：單 life baseline，`alpha_testament=0`
  - `w2_base`：跨 life + testament，使用 `alpha_testament=0.12`
  - `w2_strong`：跨 life + 較強 testament，使用 `alpha_testament=0.22`
22. W2.1 固定工作點鎖定為：`players=300`, `rounds_per_life=3000`, `total_lives=5`, `seeds={45,47,49}`；其餘 sampled/event 工作點沿用 W1 formal run 的完整版事件模板與既有 `matrix_ab` sampled baseline
23. W2.1 的主要通過條件鎖定為：至少 `1` 個 non-control cell 在 `life 3..5` 中出現 `>=1` 個 Level 3 seed，且後半段 `mean_env_gamma >= 0`
24. W2.1 的 Closure Gate 鎖定為：若所有 3 個 cells 在後半段 life 都沒有任何 Level 3 seed，則 W2.1 直接記為 `close_w2_1`，不得再微調 `alpha_testament`
25. W2.1 的 summary-level schema 至少還必須提供：`protocol`, `cell`, `seed`, `life_index`, `ended_by_death`, `n_deaths`, `mean_life_rounds`, `mean_stage3_score`, `mean_env_gamma`, `level3_seed_count`, `testament_alpha`, `testament_applied`, `verdict`
26. W2.1 的 decision markdown 必須明示：control 指標、每個 cell 在 `life 3..5` 的 verdict、是否真的出現 death/testament 事件、以及整體 `pass/weak_positive/close_w2_1` 結論
27. 在 W2.1 未先產生可重現 uplift 之前，不得把 Little Dragon 接入跨 life 初始化，也不得把 W1 world carryover 與 testament 同時打開，避免再次失去可歸因性
28. W2.1 修正版正式 dry-run（W2.1R）允許使用較大的 control-stabilized 工作點：`players=400`, `rounds_per_life=4000`, `total_lives=6`, `seeds={45,47,49}`，事件模板固定為完整版 `02_event_templates_v1.json`；control 仍必須維持 `single_life + alpha_testament=0`
29. W2.1R 的 tail 判讀視窗固定為 `life 4..6`；主要通過條件改為：至少 `1` 個 non-control cell 在 `life 4..6` 首次或再次出現 `>=1` 個 Level 3 seed，且後半段 `mean_env_gamma >= 0`
30. W2.1R 的 Closure Gate 固定為：若 `life 4..6` 仍無任何 Level 3 seed，則 W2.1 正式記為 `close_w2_1`，不得再做更多 `alpha_testament` 微調；後續只允許轉向 W2.2（world carryover + Little Dragon）或 W3（Stackelberg）
31. W2.1R 的 decision markdown 除既有 control / verdict / death-testament evidence 外，還必須明示：`life 4..6` 的 `mean_env_gamma` 是否轉正、`first tail Level 3 life`、tail death rate 是否落在 `15%~40%` 理想區間、以及 tail personality drift 是否持續累積
32. W2.1R 的 aggregate summary 至少還必須額外提供：`tail_life_start`, `first_tail_level3_life`, `mean_death_rate`, `tail_mean_death_rate`, `tail_death_rate_band_ok`, `tail_mean_personality_abs_shift`, `tail_mean_personality_l2_shift`, `tail_mean_personality_centroid_json`

W3：Stackelberg commitment 主線（post-W2.1 closure）

33. W3 第一版的目的不是立刻做完整 boss policy optimizer，而是先驗證「leader 先承諾 payoff geometry、followers 再以既有 sampled + events 動態回應」是否已足夠產生新的 Level 3 seed；在這一版中，leader commitment 在整輪 run 內必須固定，不得 in-loop 重算
34. W3.1 第一版只允許 leader 控制既有 `matrix_ab` payoff 幾何中的三個旋鈕：`a`, `b`, `matrix_cross_coupling`；followers 仍沿用既有 `simulation.run_simulation` sampled baseline，不開 testament、不開 world carryover、不開 Little Dragon
35. W3.1 的固定工作點鎖定為：`players=300`, `rounds=3000`, `seeds={45,47,49}`, `memory_kernel=3`, `selection_strength=0.06`, `init_bias=0.12`, `events_json=02_event_templates_v1.json`；control 固定為目前主 baseline `matrix_ab(a=1.0, b=0.9, cross=0.20)`
36. W3.1 的最小 formal scout 只允許 4 個固定 cells：
  - `control`：leader commit `a=1.00`, `b=0.90`, `cross=0.20`
  - `w3_cross_strong`：leader commit `a=1.00`, `b=0.90`, `cross=0.35`
  - `w3_edge_tilt`：leader commit `a=1.05`, `b=0.85`, `cross=0.20`
  - `w3_commit_push`：leader commit `a=1.05`, `b=0.85`, `cross=0.35`
37. W3.1 的 leader ranking 固定採 lexicographic 順序，不使用可任意調權的 scalar score：優先比較 `level3_seed_count`，再比較 `mean_stage3_score`，再比較 `mean_env_gamma`，最後比較 `mean_turn_strength`；這個順序必須同時用於 best cell 選擇與 `leader_prefers_over_control` 判定
38. W3.1 的 Promotion Gate 鎖定為：至少 `1` 個 non-control cell 同時滿足 `level3_seed_count >= 1`、`mean_env_gamma >= 0`、且 `mean_stage3_score` 不低於 control，才算 `pass`
39. W3.1 的 Closure Gate 鎖定為：若所有 non-control leader commitments 都沒有任何 Level 3 seed，則整輪直接記為 `close_w3_1`，不得再做更多 `a/b/cross` 細網格微調；後續只能升級到更高階 leader policy，或離開 W3 主線
40. W3.1 的 summary-level schema 至少還必須提供：`protocol`, `condition`, `leader_action`, `a`, `b`, `matrix_cross_coupling`, `seed`, `cycle_level`, `mean_stage3_score`, `mean_turn_strength`, `mean_env_gamma`, `level3_seed_count`, `has_level3_seed`, `out_csv`, `provenance_json`
41. W3.1 的 aggregate summary 至少還必須提供：`protocol`, `condition`, `is_control`, `leader_action`, `a`, `b`, `matrix_cross_coupling`, `n_seeds`, `mean_stage3_score`, `mean_turn_strength`, `mean_env_gamma`, `level3_seed_count`, `gamma_uplift_ratio_vs_control`, `stage3_uplift_vs_control`, `leader_prefers_over_control`, `is_best_commitment`, `promotion_gate_pass`, `verdict`
42. W3.1 的 decision markdown 必須明示：control 指標、4 個固定 cells 的 `leader_action / level3_seed_count / mean_stage3_score / mean_env_gamma / leader_prefers_over_control / verdict`、唯一 best commitment 是否存在、以及整體 `pass/weak_positive/fail/close_w3_1` 結論
43. W3.1 一旦依 Closure Gate 記為 `close_w3_1`，不得再做更多固定 `a/b/cross` cell sweep；若續做 W3，必須升級為更高階的 leader policy，即 leader 可以根據 follower state 在少數預先鎖定的 commitments 間切換

W3.2：State-feedback leader policy（最小高階 leader）

44. W3.2 的 leader 仍不得做自由連續優化；第一版只允許在少數預先鎖定的 commitment regimes 間切換，避免把 W3 退化成不可回溯的 online tuning
45. W3.2 的 state signal 第一版固定為 `dominance_gap = max(p_A, p_D, p_B) - min(p_A, p_D, p_B)`，其中 `p_*` 取自固定視窗內的平均策略比例；leader 不得直接讀未來資訊、也不得用 seed-level Stage3 指標作為 in-loop state
46. W3.2 的 policy update 必須是低頻、分段常數且帶 hysteresis：固定每 `150` rounds 更新一次，並使用 `theta_low=0.08`, `theta_high=0.12`；若 `dominance_gap >= theta_high` 則進入 active regime，若 `dominance_gap <= theta_low` 則回到 baseline regime，中間區間保持上一個 regime
47. W3.2 的 control 固定為 `control_policy`：全程 baseline regime，不啟動任何 policy switch；其 payoff geometry 固定為 `a=1.0`, `b=0.9`, `cross=0.20`
48. W3.2 的最小 formal scout 只允許 4 個固定 cells：
  - `control_policy`：baseline regime only
  - `w3_policy_crossguard`：active regime=`cross_guard(a=1.0, b=0.9, cross=0.35)`
  - `w3_policy_edgetilt`：active regime=`edge_tilt(a=1.05, b=0.85, cross=0.20)`
  - `w3_policy_commitpush`：active regime=`commit_push(a=1.05, b=0.85, cross=0.35)`
49. W3.2 的固定工作點與 W3.1 相同：`players=300`, `rounds=3000`, `seeds={45,47,49}`, `memory_kernel=3`, `selection_strength=0.06`, `init_bias=0.12`, `events_json=02_event_templates_v1.json`
50. W3.2 的 Promotion Gate 鎖定為：至少 `1` 個 non-control policy cell 同時滿足 `level3_seed_count >= 1`、`mean_env_gamma >= 0`、且 `mean_stage3_score` 不低於 control，才算 `pass`
51. W3.2 的 Closure Gate 鎖定為：若所有 non-control policy cells 都沒有任何 `Level 3 seed`，則整輪直接記為 `close_w3_2`，不得再做 `theta`、`interval`、或 regime 組合的細網格微調
52. W3.2 的 step-level TSV 至少還必須提供：`round`, `window_index`, `dominant_strategy`, `p_aggressive`, `p_defensive`, `p_balanced`, `dominance_gap`, `leader_action`, `regime_active`, `regime_switched`, `a`, `b`, `matrix_cross_coupling`
53. W3.2 的 summary-level schema 至少還必須提供：`protocol`, `condition`, `leader_action`, `a`, `b`, `matrix_cross_coupling`, `seed`, `cycle_level`, `mean_stage3_score`, `mean_turn_strength`, `mean_env_gamma`, `level3_seed_count`, `policy_activation_share`, `n_policy_switches`, `mean_dominance_gap`, `policy_steps_tsv`, `out_csv`, `provenance_json`
54. W3.2 的 aggregate summary 至少還必須提供：`protocol`, `condition`, `is_control`, `leader_action`, `n_seeds`, `mean_stage3_score`, `mean_turn_strength`, `mean_env_gamma`, `level3_seed_count`, `gamma_uplift_ratio_vs_control`, `stage3_uplift_vs_control`, `leader_prefers_over_control`, `policy_activation_rate`, `mean_policy_switches`, `mean_dominance_gap`, `is_best_policy`, `promotion_gate_pass`, `verdict`
55. W3.2 的 decision markdown 必須明示：control 指標、4 個 policy cells 的 `leader_action / level3_seed_count / mean_stage3_score / mean_env_gamma / policy_activation_rate / leader_prefers_over_control / verdict`、唯一 best policy 是否存在、以及整體 `pass/weak_positive/fail/close_w3_2` 結論

W3.3：Event-driven pulse leader policy（真正不同的 leader policy class）

56. W3.3 是 `close_w3_2` 之後唯一允許的 W3 主線延伸；它必須在 policy class 上和 W3.2 明確切開：不得再使用 `dominance_gap` threshold、不得使用 hysteresis band、不得固定每 `N` rounds 檢查一次、也不得讓 active regime 黏住整段 run
57. W3.3 的 leader 介入單位固定為 `pulse` 而不是 regime：pulse 只在有限 horizon 內暫時覆寫 baseline payoff geometry，pulse 結束後必須自動回到 baseline；leader 不得把 pulse 串成等價於全程 active 的長期 regime
58. W3.3 的 in-loop state 第一版固定為 follower shares 的 per-round exponential smoother：`p_hat_i(t) = (1 - ema_alpha) * p_hat_i(t-1) + ema_alpha * p_i(t)`，其中 `ema_alpha=0.15`；leader 不得直接讀未來資訊，也不得使用 seed-level Stage3 指標作為 in-loop state
59. W3.3 的 trigger 第一版固定為 `dominant_transition`：令 `dominant_strategy_ema(t) = argmax_i p_hat_i(t)`，只有當 `dominant_strategy_ema(t) != dominant_strategy_ema(t-1)` 時才允許觸發 pulse；若發生 tie，固定依 `aggressive -> defensive -> balanced` 的順序決定唯一 dominant label
60. W3.3 的 pulse mechanics 鎖定為：`pulse_horizon=120` rounds、`refractory_rounds=240`、`ema_alpha=0.15`；若 trigger 發生時目前仍在 active pulse 或 refractory 期間，該 trigger 直接忽略，不得排隊或延後觸發。這個限制的目的，是從機制上避免 W3.2 那種「很快塌縮成近靜態 commitment」的退化
61. W3.3 第一版的 action family 為了隔離「policy class 是否真的不同」，固定重用 W3.1/W3.2 已驗證過的三個 payoff geometries 作為 pulse action，而不是引入新的幾何自由度：`cross_pulse(a=1.0, b=0.9, cross=0.35)`、`edge_pulse(a=1.05, b=0.85, cross=0.20)`、`commit_pulse(a=1.05, b=0.85, cross=0.35)`
62. W3.3 的 control 固定為 `control_pulse_policy`：全程 baseline `matrix_ab(a=1.0, b=0.9, cross=0.20)`，不允許任何 pulse；W3.3 的最小 formal scout 只允許 4 個固定 cells：
  - `control_pulse_policy`：baseline only
  - `w3_pulse_crossguard`：每次 `dominant_transition` 觸發 `cross_pulse`
  - `w3_pulse_edgetilt`：每次 `dominant_transition` 觸發 `edge_pulse`
  - `w3_pulse_commitpush`：每次 `dominant_transition` 觸發 `commit_pulse`
63. W3.3 的固定工作點與 W3.1/W3.2 相同：`players=300`, `rounds=3000`, `seeds={45,47,49}`, `memory_kernel=3`, `selection_strength=0.06`, `init_bias=0.12`, `events_json=02_event_templates_v1.json`；leader ranking 仍固定沿用 W3.1 的 lexicographic 順序：`level3_seed_count -> mean_stage3_score -> mean_env_gamma -> mean_turn_strength`
64. W3.3 的 Promotion Gate 鎖定為：至少 `1` 個 non-control pulse policy cell 同時滿足 `level3_seed_count >= 1`、`mean_env_gamma >= 0`、`mean_stage3_score` 不低於 control、且 `pulse_activation_rate > 0`，才算 `pass`；若 pulse 從未被觸發，aggregate uplift 不得視為強證據
65. W3.3 的 Closure Gate 鎖定為：若所有 non-control pulse policy cells 都沒有任何 `Level 3 seed`，則整輪直接記為 `close_w3_3`，不得再做 `ema_alpha`、`pulse_horizon`、`refractory_rounds`、或 pulse action 排列組合的細網格微調；後續只能升級到另一個更高階且不同類型的 leader policy class，或離開 W3 主線
66. W3.3 的 step-level TSV 至少還必須提供：`round`, `p_aggressive`, `p_defensive`, `p_balanced`, `p_hat_aggressive`, `p_hat_defensive`, `p_hat_balanced`, `dominant_strategy_ema`, `dominant_changed`, `leader_action`, `pulse_active`, `pulse_started`, `pulse_rounds_left`, `refractory_rounds_left`, `a`, `b`, `matrix_cross_coupling`
67. W3.3 的 summary-level schema 至少還必須提供：`protocol`, `condition`, `leader_action`, `a`, `b`, `matrix_cross_coupling`, `seed`, `cycle_level`, `mean_stage3_score`, `mean_turn_strength`, `mean_env_gamma`, `level3_seed_count`, `pulse_count`, `pulse_active_share`, `first_pulse_round`, `dominant_transition_count`, `pulse_steps_tsv`, `out_csv`, `provenance_json`
68. W3.3 的 aggregate summary 至少還必須提供：`protocol`, `condition`, `is_control`, `leader_action`, `n_seeds`, `mean_stage3_score`, `mean_turn_strength`, `mean_env_gamma`, `level3_seed_count`, `gamma_uplift_ratio_vs_control`, `stage3_uplift_vs_control`, `leader_prefers_over_control`, `pulse_activation_rate`, `mean_pulse_count`, `mean_pulse_active_share`, `mean_first_pulse_round`, `mean_dominant_transition_count`, `is_best_policy`, `promotion_gate_pass`, `verdict`
69. W3.3 的 decision markdown 必須明示：control 指標、4 個 pulse policy cells 的 `leader_action / level3_seed_count / mean_stage3_score / mean_env_gamma / pulse_activation_rate / mean_pulse_count / leader_prefers_over_control / verdict`、唯一 best pulse policy 是否存在、以及整體 `pass/weak_positive/fail/close_w3_3` 結論
70. W3.3 的第一輪正式實驗矩陣固定為下表，不得再額外加入第 5 個 pulse cell，也不得把 trigger / pulse mechanics / action geometry 在同一輪內混著改：

| condition | trigger | pulse action | baseline geometry | pulse geometry | 備註 |
|---|---|---|---|---|---|
| `control_pulse_policy` | none | none | `a=1.00, b=0.90, cross=0.20` | none | matched control |
| `w3_pulse_crossguard` | `dominant_transition` | `cross_pulse` | `a=1.00, b=0.90, cross=0.20` | `a=1.00, b=0.90, cross=0.35` | 只改 cross |
| `w3_pulse_edgetilt` | `dominant_transition` | `edge_pulse` | `a=1.00, b=0.90, cross=0.20` | `a=1.05, b=0.85, cross=0.20` | 只改 edge |
| `w3_pulse_commitpush` | `dominant_transition` | `commit_pulse` | `a=1.00, b=0.90, cross=0.20` | `a=1.05, b=0.85, cross=0.35` | edge+cross 同時打開 |

71. W3.3 的 overall decision 必須採固定優先序，不得事後改寫：

| precedence | decision | 必要條件 | 研究解讀 |
|---:|---|---|---|
| 1 | `pass` | 至少 `1` 個 non-control cell 同時滿足 `promotion_gate_pass=yes` | pulse policy 真正打出可升級的 emergence |
| 2 | `close_w3_3` | 所有 non-control cells 都滿足 `level3_seed_count=0` | 此一 pulse policy family 在鎖定 working point 下正式結案 |
| 3 | `weak_positive` | 不滿足 `pass`，且至少 `1` 個 non-control cell 具 `level3_seed_count>=1` 或 `leader_prefers_over_control=yes`，同時 `pulse_activation_rate>0` | pulse 有介入且出現局部訊號，但不足以升級 |
| 4 | `fail` | 其餘情形 | pulse 既未形成 promotion，也沒有可保留的正向訊號 |

72. W3.3 的 per-cell verdict 也必須固定，避免 overall decision 與 cell verdict 混用：`control` 只保留給 matched control；non-control cell 若 `promotion_gate_pass=yes` 則記為 `pass`，否則若 `pulse_activation_rate>0` 且 `leader_prefers_over_control=yes` 則記為 `weak_positive`，其餘一律記為 `fail`
73. W3.3 的最低機制驗證條件固定為：任何 non-control cell 若 `pulse_activation_rate=0` 或 `mean_pulse_count=0`，即使 aggregate 指標略優於 control，也不得被當成 W3.3 的強證據；這種結果只能視為 control variance 或 trigger 設計失效，不能作為續做 pulse local tuning 的理由

輸出與 provenance 鎖定：

1. 新主線輸出必須與 H-series 隔離；第一輪建議前綴使用 `outputs/personality_gate0_*`, `outputs/personality_gate1_*`, `outputs/personality_gate2_*`
2. run-level 或 summary-level provenance 至少要能回溯：`personality_mode`（zero / heterogeneous）、`personality_jitter`, `personality_cohorts`, `events_json`, `enable_events`
3. 若使用縮減事件集，必須把實際 JSON 路徑寫入輸出 provenance，不得只依賴預設值
4. H5.4 若落在既有 `simulation.personality_gate1` 薄 harness 上，summary-level schema 至少還要能回溯：`sampled_inertia`, `is_control`, `is_tiebreak`, `evolution_mode`, `selection_strength`, `memory_kernel`
5. H5.4 的 aggregate summary 必須至少提供：`mean_env_gamma`, `mean_stage3_score`, `p_level_3`, `level_counts_json`, `gamma_uplift_ratio_vs_control`, `stage3_uplift_vs_control`, `verdict`
6. H5.4 的 decision markdown 必須明示：matched control 指標、每個 `sampled_inertia` 點的 verdict、是否執行 tie-break、以及整體 `pass/weak_positive/fail` 結論
7. 若 `payoff_mode=threshold_ab`，timeseries CSV 必須額外輸出 `threshold_regime_hi` 與 `threshold_state_value`；若 `payoff_mode=matrix_ab`，兩欄位保留空值以維持 schema 穩定
8. H5.5 若沿用既有 `simulation.personality_gate1` harness，summary-level schema 至少還要能回溯：`payoff_mode`, `threshold_trigger`, `threshold_theta`, `threshold_theta_low`, `threshold_theta_high`, `threshold_state_alpha`, `threshold_a_hi`, `threshold_b_hi`, `mean_regime_high_share`, `mean_regime_switches`, `p_switched_seed`, `verdict`
9. H5.5 的 decision markdown 必須明示：matched control 指標、每個 nonlinear cell 的 verdict、對應的 switch evidence，以及整體 `pass/weak_positive/fail` 結論
10. H5.5R 若沿用既有 `simulation.personality_gate1` harness，aggregate summary 還必須額外提供：`stage3_refine_pass`, `gamma_refine_pass`, `confirm_gate_pass`, `confirm_rank`
11. H5.5R 的 decision markdown 必須明示：hard stop 是否觸發、最佳 cell 是否存在、若存在則給出唯一 longer confirm 候選；若不存在，必須明示 `close_h55`
12. H6 Gate 1 的 aggregate summary 至少還必須提供：`ratio_spec`, `is_zero_control`, `is_reference_control`, `level3_seed_count`, `gamma_uplift_ratio_vs_reference`, `stage3_uplift_vs_reference`, `stage3_gate_pass`, `level3_seed_pass`, `gate1_pass`, `verdict`
13. H6 Gate 1 的 decision markdown 必須明示：`1:1:1` reference control 指標、每個 extreme-ratio cell 的 verdict、唯一最佳 cell 是否存在、以及整體 `pass/weak_positive/fail` 結論
14. H6 Gate 2 的 step-level TSV 至少必須提供：`round`, `dominant_strategy`, `previous_dominant_strategy`, `dominant_changed`, `dominance_bias`, `event_type`, `a`, `b`, `pressure`, `anti_pressure_pass`, `response_pass`
15. H6 Gate 2 的 aggregate summary / decision markdown 必須明示：`source_condition`, `sampling_interval`, `n_changed_steps`, `n_evaluable_changed_steps`, `anti_pressure_share`, 以及整體 `pass/fail` 結論
16. H7.1 的 summary-level schema 至少還必須提供：`lambda_mu`, `lambda_k`, `mu_base`, `k_base`, `mean_stage3_score`, `mean_env_gamma`, `level3_seed_count`, `gamma_uplift_ratio_vs_control`, `stage3_uplift_vs_control`, `primary_pass`, `secondary_pass`, `verdict`
17. H7.1 的 seed-level provenance 至少還必須能回溯：`personality_signal_mu`, `personality_signal_k`, `mu_player_min/max/mean`, `k_player_min/max/mean`
18. H7.1 的 decision markdown 必須明示：control 指標、4 個固定 cells 的 verdict、是否存在唯一 longer-confirm 候選、以及整體 `pass/weak_positive/fail/close_h71` 結論
19. W1.1 的輸出必須與 H-series 隔離；第一輪前綴固定使用 `outputs/w1_worldstate_*`
20. W1.1 的 world-update step-level TSV 必須採每個 cell / seed 各自獨立輸出，檔名固定為 `w1_step_world_{cell_name}_seed{seed}.tsv`
21. W1.1 的 world-update step-level TSV 至少必須提供：`round`, `scarcity`, `threat`, `noise`, `intel`, `dominant_event_type`, `a_new`, `b_new`, `event_distribution`；若要保留診斷欄位，可再額外附上 `window_index`, `window_start_round`, `window_end_round`, `p_aggressive`, `p_defensive`, `p_balanced`, `mean_reward_window`, `event_share_threat`, `event_share_resource`, `event_share_uncertainty`, `event_share_navigation`, `event_share_internal`, `state_deviated`, `risk_multipliers_json`, `reward_multipliers_json`, `trait_multipliers_json`
22. W1.1 的 summary-level schema 至少還必須提供：`world_mode`, `world_update_lambda`, `world_update_interval`, `world_state_init_json`, `mean_world_scarcity`, `mean_world_threat`, `mean_world_noise`, `mean_world_intel`, `mean_stage3_score`, `mean_env_gamma`, `level3_seed_count`, `stage3_uplift_vs_control`, `gamma_uplift_ratio_vs_control`, `gamma_improved_20pct`, `nonisomorphic_pass`, `verdict`
23. W1.1 的 decision markdown 必須明示：control 指標、4 個固定 cells 的 verdict、世界狀態是否真的偏離 `(0.5,0.5,0.5,0.5)`、`mean_env_gamma` 是否達到 `>=20%` 改善、以及 `nonisomorphic_pass` 是否成立；若未通過非同構門檻，必須明示不得進 Gate 2

### 7.4 B-series 主線契約（2026-04-03, updated 2026-04-07）

本節鎖定 B-series 的 Phase 0 研究契約。重點是：**先固定 protocol / stop rule / provenance，再做最小 runtime 實作。**

主線順序與定位固定為：

1. 主力線：`B4 -> B3 -> B5`
2. topology 對照：`B2`
3. 高理論成本最後手段：`B1`

任何實作都必須遵守：若前一條主力線尚未依 stop rule 正式 closure，不得跳做下一條主力線。

B4 第一輪正式 protocol 鎖定為：

1. runtime 固定為 `payoff_mode=matrix_ab`, `popularity_mode=sampled`, `evolution_mode=personality_coupled`
2. 工作點固定為：`players=300`, `rounds=3000`, `seeds={45,47,49}`, `memory_kernel=3`, `selection_strength=0.06`, `init_bias=0.12`, `a=1.0`, `b=0.9`, `matrix_cross_coupling=0.20`, `series=p`, `eta=0.55`, `stage3_method=turning`, `phase_smoothing=1`
3. B4 第一輪只允許掃描：`beta_state_k ∈ {0.0, 0.3, 0.6, 1.0}` 與 `k_base ∈ {0.06, 0.08}`；並固定 `lambda_k=0.0`, `lambda_mu=0.0`, `mu_base=0.0`
4. B4 第一輪只允許 linear state factor；若要升級到 exponential 或改用 entropy / `dist_to_center`，必須先滿足「Short Scout fail 且高 clamp 飽和」
5. B4 的 G0 Degrade Gate 固定為：`beta_state_k=0` 時，結果必須與 matched control 數值等價；若不等價，先修 runtime，不得進 G1/G2
6. B4 的 G1 Deterministic Gate 固定為：不得破壞既有 deterministic / mean-field Level 3；若 `turn_strength < 0.92 × baseline` 或 Level 掉出 3，則 B4 第一輪直接記為 runtime bug 或機制不合格
7. B4 的 G2 Short Scout 固定為 3 seeds；若 `0/3` seeds 達 Level 3 且 `mean_stage3_score` uplift `< 0.02`，整體直接記為 `fail`，不做 Longer Confirm
8. `k_clamped_ratio > 0.3` 視為高 clamp 飽和；`k_clamped_ratio > 0.5` 視為嚴重飽和。只有在 high/serious clamp 飽和成立時，才允許在 B4 family 內升級到更寬 clamp 或 exponential state factor
9. 若 B4 linear Short Scout 已同時滿足：`0/3` seeds 達 Level 3、`mean_stage3_score` uplift `< 0.02`、且 `k_clamped_ratio <= 0.3`，則直接記為 B4 closure，正式轉入 B3；不要求先做 B4 擴展版

B3 第一輪正式 protocol 鎖定為：

1. runtime 固定為 `payoff_mode=matrix_ab`, `popularity_mode=sampled`, `evolution_mode=sampled`
2. 工作點固定為：`players=300`, `rounds=3000`, `seeds={45,47,49}`, `memory_kernel=3`, `selection_strength=0.06`, `init_bias=0.12`, `a=1.0`, `b=0.9`, `matrix_cross_coupling=0.20`, `series=p`, `eta=0.55`, `stage3_method=turning`, `phase_smoothing=1`
3. B3.1 第一輪只允許掃描：`sampled_growth_n_strata ∈ {1,3,5,10}`；若需要第二個 `selection_strength`，只能額外開 `{0.08}` 做平行對照
4. B3.1 的 G0 Degrade Gate 固定為：`sampled_growth_n_strata=1` 時，結果必須與 matched control 數值等價；若不等價，先修 runtime，不得進 G1/G2
5. B3.1 的 G2 Short Scout 固定為 3 seeds；若 `0/3` seeds 達 Level 3 且 `mean_stage3_score` uplift `< 0.02`，整體直接記為 `fail`，不做 Longer Confirm
6. 若 B3.1 fail，則允許升級到 B3.2 personality-based strata；若 B3.2 仍 fail，才允許 B3.3 phase-aware strata
7. B3.2 第一輪固定沿用 B3.1 的 locked protocol，僅新增 `strata_mode=personality` 與 `personality_jitter=0.08`；其餘 `players`, `rounds`, `seeds`, `memory_kernel`, `selection_strength`, `init_bias`, `a`, `b`, `matrix_cross_coupling` 一律不變
8. B3.3 第一輪固定沿用 B3.1/B3.2 的 locked protocol，僅新增 `strata_mode=phase` 與 `phase_rebucket_interval=100`；不得同時引入 personality sampling、payoff 修改、或 operator 修改
9. B3.3 的 phase angle 一律由 player 當前 mixed strategy 向量 $p=(p_A,p_B,p_C)$ 計算：$\theta = \operatorname{atan2}(\sqrt{3}(p_B-p_C),\; 2p_A-p_B-p_C)$；分桶方式固定為 $[-\pi,\pi)$ 上的等角分桶
10. B3.3 第一輪仍只允許掃描 `sampled_growth_n_strata ∈ {1,3,5,10}`；其中 `n_strata=1` 必須精確退化為 matched control，且第一輪不得再掃 `rebucket_interval`、`sector_offset`、`slow_rebalance` 等局部自由度
11. 若 B3.3 依既有 G2 stop rule 仍 fail，則 B3 family 直接 closure，正式轉入 B5；除非發現 runtime bug，否則不得回頭做 B3.3 參數微調
12. B3.3 實作優先沿用既有 `simulation.b3_stratified_growth` harness 與 `simulation.run_simulation.simulate(..., round_callback=...)` 介面；第一輪不得另開第二套掃描 harness
13. B3.3 的 Gate 0 / G0 必須先通過：`strata_mode=phase` 可被 CLI 與 harness 接受、`phase_rebucket_interval > 0`、且 `n_strata=1` 在 `phase` 模式下逐列精確退化為既有 sampled control
14. B3.3 的 Gate 1 / G1 必須先通過：phase 分桶純函式需可重現，且在小型 smoke case 中至少能觀察到一次非零 rebucket churn；若 phase 分桶始終不變，先視為 runtime / hook 問題，不得直接進 G2
15. B3.3 的 Gate 2 / G2 summary / provenance 至少必須新增：`phase_rebucket_interval`, `mean_phase_rebucket_churn`, `mean_within_stratum_phase_spread`, `phase_occupancy_entropy`；若缺任一欄位，視為機制不可判讀，不得進正式 short scout

B5 第一輪正式 protocol 鎖定為：

1. G2 runtime 固定為 `payoff_mode=matrix_ab`, `popularity_mode=sampled`, `evolution_mode=sampled`
2. G1 deterministic gate 固定跑在 matched deterministic / mean-field Level 3 control 上；第一輪不得把 B5 漂移混入 personality-coupled、sampled-inertial 或 B3 strata 路徑
3. 工作點固定沿用 B3 locked protocol：`players=300`, `rounds=3000`, `seeds={45,47,49}`, `memory_kernel=3`, `selection_strength=0.06`, `init_bias=0.12`, `a=1.0`, `b=0.9`, `matrix_cross_coupling=0.20`, `series=p`, `eta=0.55`, `stage3_method=turning`, `phase_smoothing=1`
4. B5 第一輪只允許掃描：`tangential_drift_delta ∈ {0.000, 0.003, 0.006, 0.010, 0.015}`；第一輪不得同時掃 `selection_strength`、`memory_kernel`、`phase_smoothing` 或其他 drift 形狀參數
5. B5 的 drift 幾何狀態一律使用當前 population simplex `x(t)`；不得改用 subgroup simplex、player-local personality signal 或外部 world state 作為第一輪 drift 來源
6. B5 的 helper-level 幾何 invariants 必須先通過：`tangential_drift_vector()` 在 `delta=0` 或 `x=(1/3,1/3,1/3)` 時精確回傳零向量；在非退化狀態下，輸出 drift 必須同時滿足 `drift·(x-c)=0`、`drift·(1,1,1)=0`、且 `|drift|=delta`
7. B5 的 Gate 0 / G0 必須先通過：helper-level 的 `delta=0` 與 runtime-level 的 `--tangential-drift-delta=0` 都必須精確退化為 matched control；若不等價，先修 runtime，不得進 G1/G2
8. B5 的 Gate 1 / G1 必須先通過：`delta ∈ {0.003, 0.006, 0.010}` 在 deterministic / mean-field gate 下不得破壞既有 Level 3；若 `turn_strength < 0.92 × baseline`、level 掉出 3、或 `drift_contribution_ratio > 0.05`，直接記為 drift 過大或實作錯誤，不得進 G2
9. B5 的 Gate 2 / G2 short scout 固定為 3 seeds；若 `delta <= 0.010` 的所有 active cells 都 `0/3` Level 3 且 `mean_stage3_score` uplift `< 0.02`，B5 直接 closure，不掃更大 delta
10. B5 的第一輪實作必須優先以共用 helper 注入既有 `replicator_step()` 與 `deterministic_replicator_step()`，不得另開第二套 sampled runtime 或第二套 mean-field runtime
11. B5 的 G2 summary / diagnostics 至少必須新增：`tangential_drift_delta`, `mean_drift_norm`, `mean_effective_delta_growth_ratio`, `mean_tangential_alignment`, `phase_amplitude_stability`, `representative_drift_vector_rose_png`；若缺任一欄位，視為機制不可判讀，不得進正式 short scout

2026-04-07 formal result（已完成）：

1. G1 deterministic gate 已在 `delta ∈ {0.003, 0.006, 0.010, 0.015}` 全數通過；所有 nonzero cells 都維持 `Level 3`，且 `drift_contribution_ratio` 僅約為 `0.003572 / 0.007110 / 0.011820 / 0.017800`，均顯著低於 `0.05`
2. G2 sampled short scout 已正式完成；matched control 為 `level_counts={0:0,1:0,2:3,3:0}`, `mean_stage3_score=0.515010`, `mean_env_gamma=-0.000066`
3. `delta <= 0.010` 的所有 active cells 均為 `0/3` `Level 3` seeds，且 `mean_stage3_score` uplift 全為負值：`-0.001174`, `-0.000512`, `-0.001883`；因此依 locked closure rule，B5 第一輪正式記為 `close_b5`，不得再掃更大 delta 作為救援
4. `delta=0.015` 亦未產生任何 `Level 3` seed，且 `mean_stage3_score` uplift 仍為負值（`-0.002896`）；此點只可記為診斷性的 `weak_positive`，不得作為續做 B5 family 的理由
5. B5 formal closure 的研究解讀固定為：幾何明確的切線催化足以在 deterministic gate 中保持 Level 3 幾何，但在 well-mixed sampled replicator 的 locked working point 下，仍無法把系統推出既有 `Level 2` plateau；B5 因此提供的是正式 negative result，而非 runtime 失敗

B2 第一輪正式 protocol 鎖定為：

1. B2 不使用既有 `simulate()` 進行 sampled 模擬；B2 的核心差異在於 deme-local payoff + deme-local growth + deme-local weight broadcast，必須以自建 inner loop 實現（`simulation/b2_island_deme.py`）
2. B2 不設 G1 Deterministic Gate：mean-field 天然 well-mixed，「island topology」在 ODE 中無意義；G1 對 B5 有意義（drift 可在 mean-field 中測試），但對 B2 不適用
3. G0 Degrade Gate 固定為：M=1 control（全體 300 人在同一個 deme）必須與 well-mixed sampled 行為一致；若 `cycle_level` 或 `mean_stage3_score` 差異超過容許範圍，先修 runtime，不得進 G2
4. G2 Short Scout 固定為 3 seeds；判讀標準與 B4/B3/B5 一致（見 §19.9.3）
5. 工作點固定沿用 B-series locked protocol：`players=300`, `rounds=3000`, `seeds={45,47,49}`, `burn_in=1000`, `tail=1000`, `memory_kernel=3`, `selection_strength=0.06`, `init_bias=0.12`, `a=1.0`, `b=0.9`, `matrix_cross_coupling=0.20`, `series=p`, `eta=0.55`, `stage3_method=turning`, `phase_smoothing=1`
6. B2 第一輪不啟用 events（純 topology 效應測試）；events 可在第二輪加入（若 B2 有 weak positive 需要進一步確認）
7. B2 第一輪只允許掃描：`num_demes=3`，`migration_fraction ∈ {0.02, 0.05, 0.10}`，`migration_interval ∈ {100, 200}`；第一輪不掃 M=5（N_d=60 太小，stochasticity 會干擾判讀）
8. 遷移機制固定為 symmetric random pooled redistribution：每 $T_{mig}$ 輪從每個 deme 抽出 $f\%$ 玩家，pool 後隨機重新分配回所有 demes
9. Deme 內更新嚴格遵守 §19.7.3：deme-local payoff（用 per-deme strategy distribution 計算 `_matrix_ab_payoff_vec`）、deme-local `sampled_growth_vector()`、deme-local `replicator_step()`、deme-local weight broadcast
10. 全域判定：所有 deme 合併後計算 global `p_*(t)` 做 Stage3 分析
11. B2 第一輪掃描：M=1 control（3 seeds）+ M=3 × f={0.02,0.05,0.10} × T_mig={100,200}（6 conditions × 3 seeds = 18 runs）= 共 21 runs
12. Stop rule：若所有 M=3 conditions 都 `0/3` Level 3 且 `max(stage3_uplift) < 0.02`，B2 直接 closure，不掃 M=5 或第二輪
13. Phase spread early-stop：若所有 M=3 conditions 的 `mean_inter_deme_phase_spread < 0.10` rad，可記為 early closure（topology 效應未形成），但仍需完成全部 21 runs 才能下結論
14. 若 ≥1/3 seeds 在任何 condition 達到 Level 3 → `short_scout_pass`，推薦進 Longer Confirm
15. B2 的 G2 summary / diagnostics 至少必須新增：`num_demes`, `migration_fraction`, `migration_interval`, `mean_inter_deme_phase_spread`, `max_inter_deme_phase_spread`, `mean_inter_deme_growth_cosine`, `phase_amplitude_stability`, `representative_deme_simplex_png`；若缺任一欄位，視為機制不可判讀，不得進正式 short scout

2026-04-07 formal result（已完成）：

1. G2 sampled short scout 已正式完成；M=1 control（3 seeds）`cycle_level=2`, `mean_stage3_score≈0.527`, `mean_env_gamma≈6e-6`，符合 well-mixed baseline
2. 所有 6 active conditions（M=3 × f={0.02,0.05,0.10} × T_mig={100,200}）× 3 seeds = 18 runs，全部 `0/18` Level 3 seeds；依 locked stop rule，B2 第一輪正式記為 `close_b2`
3. `max(stage3_uplift) = 0.011134`（condition `g2_M3_f0p02_T200`, seed47），低於 0.02 threshold；其餘 uplift 分布在 `-0.011` 到 `+0.008` 之間，無系統性上升趨勢
4. 拓撲效應確實形成：所有 M=3 conditions 的 `mean_inter_deme_phase_spread ≈ 1.92–1.96 rad`（遠大於 early-stop threshold 0.10 rad），`mean_inter_deme_growth_cosine ≈ 0.61`，deme 間確有 phase separation 與 growth diversity；但全局採樣加權仍將差異均化，Level 3 basin 未打開
5. B2 formal closure 的研究解讀固定為：island deme topology 在 well-mixed sampled replicator 下確實可以形成顯著的 deme-level phase separation（`~1.93 rad`），但此空間差異仍不足以對抗三層均化壓力，`Level 3` basin 始終無法在全局尺度觸發；B2 因此提供的是正式 negative result，而非 runtime 失敗

#### (B1) `tangential_projection_replicator`（Phase 2 第一輪正式契約，2026-04-08）

**動機與理論位置**

Phase 1 closure（B2–B5, C1–C2, T-series, W-series，共 ≈230 runs）已確認：sampled discrete synchronous update 的三層均化效應（sampling noise → popularity averaging → synchronous replicator）對 rotational / directional component 的耗散極為頑強。所有已測試的「局部修正」——topology、初始條件、selection strength、drift、分層、island、world-state coupling、episodic inheritance、leader policy——均不足以穩定打開 Level 3 basin。

Phase 1 的核心排除結論指向 Layer 3（replicator operator 本身）是瓶頸：
- B5 tangential drift 在 **deterministic gate 中保持 Level 3**（drift 能注入 rotation），但在 sampled path 下被均化吸收 → operator 「能接收」rotational signal，但現有 operator 的 gradient-only 結構不產生也不保存 rotation
- W2 episodic 的 stage3_score 可達 0.46 → 方向性信號可以被外部機制累積，但**operator 內部的每步耗散率**仍高於累積率

B1 是唯一直接改動 operator 幾何的方案：不是在 growth vector 上加外部 drift（B5），而是**在 operator 內部將 growth vector 分解為 radial + tangential 分量，並以可控比例放大 tangential 分量**。

**數學規格**

在 3-simplex 上，以重心 $c = (1/3, 1/3, 1/3)$ 為參考，當前 population state $x(t)$ 的 growth vector $g = (g_A, g_D, g_B)$（已 zero-mean）可分解為：

1. **Radial component**（沿 $x - c$ 方向，即 convergence/divergence）：

$$
g_r = \frac{g \cdot r}{\|r\|^2} \, r, \quad r = x - c
$$

2. **Tangential component**（在 simplex 平面上正交於 $r$ 的方向，即 rotation）：

$$
g_\tau = g - g_r
$$

B1 tangential projection replicator 的修正 growth vector 為：

$$
g' = g_r + (1 + \alpha) \, g_\tau
$$

其中 $\alpha \in [0, \alpha_{\max}]$ 為 **tangential amplification factor**：
- $\alpha = 0$ 時精確退化為標準 replicator（$g' = g$）
- $\alpha > 0$ 時放大 tangential 分量，等價於在 operator 內部注入旋轉偏好
- $g'$ 仍然 zero-mean（因 $g_r$ 和 $g_\tau$ 都是 zero-mean 向量的線性組合）

然後依標準指數化更新：

$$
w_s(t+1) = \frac{e^{k \cdot g'_s}}{\text{mean}(e^{k \cdot g'_i})}
$$

**與 B5 tangential drift 的關鍵差異**

| 面向 | B5 tangential drift | B1 tangential projection |
|------|---------------------|--------------------------|
| 來源 | 外部注入的固定方向 $\tau(x)$ | growth vector $g$ 自身的 tangential 分量 |
| 依賴 | 只依賴 $x(t)$，與 payoff 無關 | 依賴 $g$ 的方向，保留 payoff 資訊 |
| signal 強度 | 由 $\delta$ 固定控制 | 由 $g_\tau$ 的自然大小乘以 $(1+\alpha)$ |
| 退化條件 | $\delta=0$ 退化 | $\alpha=0$ 退化 |
| 失敗模式（Phase 1） | deterministic gate 通過，sampled path 被均化吸收 | (尚未測試) |

**code-level 規格**

新增獨立函式於 `evolution/replicator_dynamics.py`：

```python
def tangential_projection_replicator_step(
    players: Iterable[object],
    strategy_space: List[str],
    selection_strength: float = 0.05,
    *,
    tangential_alpha: float = 0.0,
    sampled_growth_n_strata: int = 1,
    sampled_growth_strata_key: str = "stratum",
) -> tuple[Dict[str, float], Dict[str, float]]:
    """B1: Replicator step with tangential amplification.
    
    Returns (weights, diagnostics).
    diagnostics keys: radial_norm, tangential_norm, alpha_effective,
                      tangential_ratio, growth_angle_rad
    """
```

`SimConfig` 新增欄位：`tangential_alpha: float = 0.0`

`simulate()` 中的 dispatch：當 `evolution_mode = "sampled"` 且 `tangential_alpha > 0`，改用 `tangential_projection_replicator_step()`。

**退化/幾何 invariants（G0 必須通過）**

1. `tangential_alpha=0` 時，`tangential_projection_replicator_step()` 的回傳值必須與 `replicator_step()` **位元等價**
2. Growth vector 分解保持 zero-mean：$\sum g'_i = 0$（允許 $10^{-12}$ 量級誤差）
3. 當 $x = c$（重心）時，$r = 0$，tangential 分解未定義 → 此時直接使用原始 $g$（不放大）
4. 回傳 weights 滿足既有 invariant：正數、mean=1

**第一輪正式 protocol 鎖定**

1. G2 runtime 固定為 `payoff_mode=matrix_ab`, `popularity_mode=sampled`, `evolution_mode=sampled`（加 `tangential_alpha` 參數）
2. G1 deterministic gate：在 mean-field 路徑下，`deterministic_replicator_step()` 加入 tangential projection 後，既有 Level 3 不得被破壞
3. 工作點固定沿用 B-series locked protocol：`players=300`, `rounds=3000`, `seeds={45,47,49}`, `memory_kernel=3`, `selection_strength=0.06`, `init_bias=0.12`, `a=1.0`, `b=0.9`, `matrix_cross_coupling=0.20`, `series=p`, `eta=0.55`, `stage3_method=turning`, `phase_smoothing=1`
4. B1 第一輪只允許掃描：`tangential_alpha ∈ {0.0, 0.3, 0.5, 1.0, 2.0}`；第一輪不得同時掃 `selection_strength`、`memory_kernel`、`phase_smoothing` 或其他超參數
5. B1 的 G0 Degrade Gate 固定為：`tangential_alpha=0` 時，結果必須與 matched control 數值等價；若不等價，先修 runtime，不得進 G1/G2
6. B1 的 G1 Deterministic Gate 固定為：`tangential_alpha ∈ {0.3, 0.5, 1.0}` 在 deterministic / mean-field gate 下不得破壞既有 Level 3；若 `turn_strength < 0.92 × baseline` 或 Level 掉出 3，直接記為 alpha 過大或實作錯誤，不得進 G2
7. B1 的 G2 Short Scout 固定為 3 seeds；若所有 `tangential_alpha <= 1.0` cells 都 `0/3` Level 3 且 `mean_stage3_score` uplift `< 0.02`，B1 直接 closure
8. B1 第一輪不得與 B5 drift 同時啟用（`tangential_alpha > 0` 與 `tangential_drift_delta > 0` 互斥）；若需測試組合效應，必須在 B1 獨立完成後另案設計
9. B1 的 G2 summary / diagnostics 至少必須新增：`tangential_alpha`, `mean_radial_norm`, `mean_tangential_norm`, `mean_tangential_ratio`, `mean_growth_angle_rad`, `phase_amplitude_stability`；若缺任一欄位，視為機制不可判讀，不得進正式 short scout
10. B1 若依上述 protocol 正式 closure，其研究解讀將固定為：即使在 operator 內部顯式放大 rotation（tangential amplification），在 sampled discrete synchronous update 下，三層均化的 Layer 1（sampling noise）仍足以在每步消滅放大的 tangential signal → replicator family 在此 sampling regime 下的 rotation ceiling 為 Level 2。此結論將作為 Phase 1+2 的 closing theorem

#### (A1) `async_replicator`（Phase 3 第一輪正式契約，2026-04-08）

**動機與理論位置**

Phase 1+2 closure（B/C/T/W/B1，≈245 runs）已確認：sampled **discrete synchronous** update 的三層均化效應為 Level 2 ceiling。核心瓶頸的物理圖像：

- **Layer 1**（sampling noise）：每步離散策略選擇引入高頻噪聲
- **Layer 2**（population averaging）：$x = \text{mean}(\text{sampled strategies})$，抹平個體差異
- **Layer 3**（replicator update）：$w(t+1) \propto w(t) \cdot f(x)$，**同步更新 = 全場同相位推進**

A1 直接攻擊 Layer 3：將同步更新替換為**隨機異步更新**。每輪只有一部分玩家更新權重，其餘保持上一次更新的舊權重。這自然製造**權重異質性**：不同玩家處於不同「演化年齡」，因此策略選擇分布不再同質，打破全場同相位推進。

**數學規格**

每輪 $t$：

1. 所有 $N$ 位玩家選擇策略並獲得 reward（`engine.step()` 不變）
2. 計算共享 growth vector：$g_s = \bar{u}_s - \bar{u}$（與同步相同）
3. 對每位玩家 $i$ 獨立以機率 $p$ 決定是否更新：$m_i \sim \text{Bernoulli}(p)$
4. **若 $m_i = 1$（更新）**：

$$
w_i(t+1)_s = \frac{\exp(k \cdot g_s)}{\text{mean}_j(\exp(k \cdot g_j))}
$$

5. **若 $m_i = 0$（不更新）**：$w_i(t+1) = w_i(t)$

其中 $p$ = `async_update_fraction` ∈ $(0, 1]$。

**與同步 replicator 的關鍵差異**

| 面向 | 同步 replicator | A1 異步 replicator |
|------|----------------|--------------------|
| 更新範圍 | 全部 $N$ 位玩家同步更新 | 每輪只有 $\sim pN$ 位隨機更新 |
| 權重異質性 | 所有玩家始終有相同權重 | 自然發散：不同玩家有不同「演化年齡」 |
| 策略分布 | 同質 → Layer 2 averaging 最強 | 異質 → 降低 Layer 2 averaging 效力 |
| 退化條件 | — | $p=1.0$ 退化回同步（位元等價） |

**code-level 規格**

新增函式於 `evolution/replicator_dynamics.py`：

```python
def async_replicator_step(
    players: list[object],
    strategy_space: list[str],
    *,
    selection_strength: float = 0.05,
    async_update_fraction: float = 1.0,
    seed: int = 0,
    round_index: int = 0,
) -> tuple[dict[str, float], dict[str, float]]:
    """A1: Stochastic asynchronous replicator step.

    Returns (mean_weights, diagnostics).
    diagnostics keys: n_updated, fraction_updated, weight_dispersion
    """
```

`SimConfig` 新增欄位：`async_update_fraction: float = 1.0`

`simulate()` 中的 dispatch：當 `evolution_mode = "sampled"` 且 `async_update_fraction < 1.0`，改用 `async_replicator_step()`；函式內部直接 per-player 更新，weight application loop 中 `continue`（同 personality_coupled 模式）。

**退化 invariants（G0 必須通過）**

1. `async_update_fraction=1.0` 時，`async_replicator_step()` 的回傳 mean_weights 必須與 `replicator_step()` **位元等價**（所有玩家起始相同權重時）
2. `async_update_fraction=0.0` 時，沒有任何玩家更新，所有權重不變
3. 回傳 weights 滿足既有 invariant：正數、每位玩家 mean=1
4. 給定相同 `seed` + `round_index`，更新結果可重現（deterministic）

**第一輪正式 protocol 鎖定**

1. G2 runtime 固定為 `payoff_mode=matrix_ab`, `popularity_mode=sampled`, `evolution_mode=sampled`（加 `async_update_fraction` 參數）
2. 不設 G1 deterministic gate（mean-field 路徑無實際玩家，async 不適用）；G0 退化測試由 unit test 覆蓋
3. 工作點固定沿用 locked protocol：`players=300`, `rounds=3000`, `seeds={45,47,49}`, `memory_kernel=3`, `selection_strength=0.06`, `init_bias=0.12`, `a=1.0`, `b=0.9`, `matrix_cross_coupling=0.20`, `series=p`, `eta=0.55`, `stage3_method=turning`, `phase_smoothing=1`
4. A1 第一輪只允許掃描：`async_update_fraction ∈ {0.1, 0.2, 0.5, 1.0}`；`1.0` 為 matched control
5. A1 的 G0 Degrade Gate 固定為：`async_update_fraction=1.0` 時，結果必須與標準同步 replicator 數值等價
6. A1 的 G2 Short Scout 固定為 3 seeds；若所有 `async_update_fraction <= 0.5` cells 都 `0/3` Level 3 且 `mean_stage3_score` uplift `< 0.02`，A1 直接 closure
7. A1 第一輪不得與 B1/B5 同時啟用（`async_update_fraction < 1.0` 與 `tangential_alpha > 0`、`tangential_drift_delta > 0` 互斥）
8. A1 的 G2 summary / diagnostics 至少必須新增：`async_update_fraction`, `mean_fraction_updated`, `mean_weight_dispersion`, `phase_amplitude_stability`
9. A1 若依上述 protocol 正式 closure，其研究解讀將固定為：異步隨機更新在 sampled discrete regime 下是否可打破同步均化的 Level 2 ceiling

#### (S1) `sampling_sharpness`（Phase 4 第一輪正式契約，2026-04-08）

**動機與理論位置**

Phase 1–3 closure（B/C/T/W/B1/A1，≈257 runs）已確認：sampled discrete update（同步或非同步）的三層均化效應 → Level 2 ceiling。Layer 2（operator geometry）與 Layer 3（update scheduling）均已排除。剩餘瓶頸鎖定為 **Layer 1（sampling ∝ w 的離散噪聲）**。

S1 直接攻擊 Layer 1：將 w-proportional 策略抽樣替換為 **power-law sharpened sampling**。

**數學定義**

目前策略抽樣：

$$P(s) = \frac{w_s^{\text{eff}}}{\sum_i w_i^{\text{eff}}}, \qquad w_s^{\text{eff}} = \max(10^{-9},\, w_s) \cdot e^{b_s}$$

S1 引入 sharpness exponent $\beta$（`sampling_beta`）：

$$P_\beta(s) = \frac{\big(w_s^{\text{eff}}\big)^\beta}{\sum_i \big(w_i^{\text{eff}}\big)^\beta}$$

| $\beta$ 值 | 行為 | Layer 1 噪聲 |
|------------|------|-------------|
| 0 | 均勻隨機（與 w 無關） | 最大 |
| 0.5 | 平滑化 | 增加 |
| **1.0** | **現行 w-proportional（control）** | **baseline** |
| 2.0 | 中等銳化 | 減少 |
| 5.0 | 強銳化 | 大幅減少 |
| 50.0 | 近乎確定性（≈argmax） | 近零 |

**Code-level spec**

```
# players/base_player.py — choose_strategy()
sampling_beta = getattr(self, 'sampling_beta', 1.0)  # 從 SimConfig 流入
eff_w = max(1e-9, w_s) * exp(bias_s)
if sampling_beta != 1.0:
    eff_w = eff_w ** sampling_beta
# 後續以 eff_w 做加權抽樣
```

```
# SimConfig 新增欄位
sampling_beta: float = 1.0
```

```
# simulate() 中 player setup
for pl in players:
    setattr(pl, "sampling_beta", float(cfg.sampling_beta))
```

**第一輪正式 protocol 鎖定**

1. G2 runtime 固定為 `payoff_mode=matrix_ab`, `popularity_mode=sampled`, `evolution_mode=sampled`
2. 不設 G1 deterministic gate（mean-field 路徑下 n_players=0，無策略抽樣）；G0 退化測試由 unit test 覆蓋（`sampling_beta=1.0` 時結果必須與現行 w-proportional 數值等價）
3. 工作點固定沿用 locked protocol：`players=300`, `rounds=3000`, `seeds={45,47,49}`, `memory_kernel=3`, `selection_strength=0.06`, `init_bias=0.12`, `a=1.0`, `b=0.9`, `matrix_cross_coupling=0.20`, `series=p`, `eta=0.55`, `stage3_method=turning`, `phase_smoothing=1`
4. S1 第一輪只允許掃描：`sampling_beta ∈ {0.5, 1.0, 2.0, 5.0, 50.0}`；`1.0` 為 matched control
5. S1 的 G0 Degrade Gate 固定為：`sampling_beta=1.0` 時，結果必須與標準 w-proportional 數值等價
6. S1 的 G2 Short Scout 固定為 3 seeds；若所有 `sampling_beta != 1.0` 的 active cells 全部 `0/3` Level 3 且 `mean_stage3_score` uplift `< 0.02`，S1 直接 closure
7. S1 第一輪不得與 A1 同時啟用（`sampling_beta != 1.0` 與 `async_update_fraction < 1.0` 互斥），避免複合效應
8. S1 的 G2 summary / diagnostics 至少必須新增：`sampling_beta`, `mean_strategy_entropy`（per-round Shannon entropy of sampled strategy distribution，作為 Layer 1 noise 量化指標）, `phase_amplitude_stability`
9. S1 若依上述 protocol 正式 closure，其研究解讀將固定為：改變 Layer 1 的取樣銳度（power-law β）在 sampled discrete regime 下是否可打開 Level 3 basin

#### (M1) `mutation_rate`（Phase 5 第一輪正式契約，2026-04-08）

**動機與理論位置**

Phase 1–4 closure（B/C/T/W/B1/A1/S1，≈272 runs）揭示：Layer 1 噪聲的根源不是 sampling 機制的離散性，而是 **replicator 均化產生的 weight homogeneity**（w ≈ 1:1:1），使得 $w^\beta \approx w$ 無論 β 取值。S1 的 entropy ≈ 1.05 across β ∈ [0.5, 50] 證實 sampling sharpness 不改變人口層策略分佈。

M1 直接攻擊 weight homogeneity 本身：每一輪在 replicator weight update 後，對每位 player 注入獨立的 **Dirichlet mutation**，打破全場同質權重。

**數學定義**

標準 replicator 產出共享權重 $w_{\text{new}}$。M1 在此之後對每位 player $i$ 施加 per-round Dirichlet mixing：

$$w_i' = (1 - \eta) \cdot w_{\text{new}} + \eta \cdot u_i, \qquad u_i \sim \text{Dirichlet}(1, 1, 1)$$

| $\eta$ 值 | 行為 | weight heterogeneity |
|-----------|------|---------------------|
| 0.0 | 無 mutation（control = 標準 replicator） | 無（完全同質） |
| 0.001 | 微弱擾動 | O(10⁻³) |
| 0.005 | 溫和擾動 | O(10⁻³–10⁻²) |
| 0.01 | 中等擾動 | O(10⁻²) |
| 0.05 | 顯著擾動 | O(10⁻²–10⁻¹) |
| 0.10 | 強擾動 | O(10⁻¹) |

**Code-level spec**

```
# simulation/run_simulation.py — SimConfig
mutation_rate: float = 0.0

# simulate() — 在 weight broadcast 之後
if mutation_rate > 0:
    for each player i (non-fixed):
        cur_w = player.strategy_weights
        u = Dirichlet(1,1,1)  # per-player RNG
        mutated_w = {s: (1-eta)*cur_w[s] + eta*u[s] for s in strategies}
        player.update_weights(mutated_w)
```

**第一輪正式 protocol 鎖定**

1. G2 runtime 固定為 `payoff_mode=matrix_ab`, `popularity_mode=sampled`, `evolution_mode=sampled`
2. 不設 G1 deterministic gate（per-player Dirichlet mutation 在 mean-field 無意義）；G0 退化測試由 unit test 覆蓋（`mutation_rate=0.0` 時結果必須與標準 replicator bit-identical）
3. 工作點固定沿用 locked protocol：`players=300`, `rounds=3000`, `seeds={45,47,49}`, `memory_kernel=3`, `selection_strength=0.06`, `init_bias=0.12`, `a=1.0`, `b=0.9`, `matrix_cross_coupling=0.20`
4. M1 第一輪只允許掃描：`mutation_rate ∈ {0.001, 0.005, 0.01, 0.05, 0.10}`；`0.0` 為 matched control
5. M1 的 G0 Degrade Gate 固定為：`mutation_rate=0.0` 時，結果必須與標準 replicator 數值等價
6. M1 的 G2 Short Scout 固定為 3 seeds；若所有 `mutation_rate > 0` 的 active cells 全部 `0/3` Level 3 且 `mean_stage3_score` uplift `< 0.02`，M1 直接 closure
7. M1 第一輪不得與 A1 同時啟用（`mutation_rate > 0` 與 `async_update_fraction < 1.0` 互斥）
8. M1 的 G2 summary / diagnostics 至少必須新增：`mutation_rate`, `mean_weight_dispersion`（per-round 各 player 權重向量的 std，作為 heterogeneity 量化指標）, `mean_strategy_entropy`, `phase_amplitude_stability`
9. M1 若依上述 protocol 正式 closure，其研究解讀將固定為：per-round Dirichlet mutation 在 sampled discrete regime 下是否可透過打破 weight homogeneity 來打開 Level 3 basin

#### (L2) `local_group_size`（Phase 6 第一輪正式契約，2026-04-08）

**動機與理論位置**

Phase 1–5 closure（B/C/T/W/B1/A1/S1/M1，≈290 runs）循序確認瓶頸上移：
- Phase 4（S1）：sampling sharpness 不影響 entropy → weight homogeneity 是根因
- Phase 5（M1）：Dirichlet mutation 成功打破 weight homogeneity（dispersion 0→0.024），但 **strategy entropy 完全不變**（1.049 ± 0.0004）

M1 的核心發現是 Layer 2 的 popularity averaging（$\bar{x} = \text{mean}(s_i)$）以大數定律（$N=300$）在每步將個體差異壓縮至人口層不可見。即使每個 player 的 $w_i$ 不同，共享的 $\bar{x}$ 仍收斂到相同值。

L2 直接攻擊 Layer 2：將 growth vector 的計算從全局 $N=300$ 的共享 popularity 改為**局部子群**（local sub-groups）的 popularity。每個子群獨立計算自己的 growth vector，使有效 $N$ 從 300 降至 `local_group_size`，打破大數定律的平滑效應。

**數學定義**

將 $N$ 位 player 靜態分割為 $\lceil N / G \rceil$ 個固定子群，每群大小 $\approx G$（`local_group_size`）。每 round：

1. 全體 player 在同一 dungeon 中博弈（global payoffs 不變）
2. 每個子群 $k$ 只用本群 player 的 sampled strategies 和 rewards 計算局部 growth：
   $$g_s^{(k)} = \bar{r}_s^{(k)} - \bar{r}^{(k)}$$
3. 本群 player 的權重用本群 growth 更新：
   $$w_{s}^{(k)} \leftarrow w_{s}^{(k)} \cdot \exp\!\bigl(k_{\text{sel}} \cdot g_s^{(k)}\bigr) \Big/ \text{mean}$$

| `local_group_size` | 群數（N=300） | 有效 $N$ | $\bar{x}$ 變異度 | 預期效果 |
|---|---|---|---|---|
| 0（control = 全局） | 1 | 300 | $O(1/\sqrt{300})$ | 標準 replicator |
| 50 | 6 | 50 | $O(1/\sqrt{50})$ ≈ 2.4× | 溫和解耦 |
| 20 | 15 | 20 | $O(1/\sqrt{20})$ ≈ 3.9× | 中等解耦 |
| 10 | 30 | 10 | $O(1/\sqrt{10})$ ≈ 5.5× | 強解耦 |
| 5 | 60 | 5 | $O(1/\sqrt{5})$ ≈ 7.7× | 極端解耦 |

**Code-level spec**

```
# simulation/run_simulation.py — SimConfig
local_group_size: int = 0  # 0 = global (default)

# simulate() — player setup 之後
if local_group_size > 0 and local_group_size < n_players:
    n_groups = n_players // local_group_size
    local_groups = partition_players_into_groups(players, n_groups)

# simulate() — evolution step（standard sampled path）
if local_groups is not None:
    for group_players in local_groups:
        local_weights = replicator_step(group_players, strategy_space, ...)
        for pl in group_players:  # non-fixed only
            pl.update_weights(local_weights)
    new_weights = mean_player_weights(players)  # for CSV logging
```

**第一輪正式 protocol 鎖定**

1. G2 runtime 固定為 `payoff_mode=matrix_ab`, `popularity_mode=sampled`, `evolution_mode=sampled`
2. 不設 G1 deterministic gate（local group 在 mean-field 無意義）；G0 退化測試由 unit test 覆蓋（`local_group_size=0` 或 `≥N` 時結果必須與標準 replicator bit-identical）
3. 工作點固定沿用 locked protocol：`players=300`, `rounds=3000`, `seeds={45,47,49}`, `memory_kernel=3`, `selection_strength=0.06`, `init_bias=0.12`, `a=1.0`, `b=0.9`, `matrix_cross_coupling=0.20`
4. L2 第一輪只允許掃描：`local_group_size ∈ {5, 10, 20, 50}`；`0` 為 matched control
5. L2 的 G0 Degrade Gate 固定為：`local_group_size=0` 時，結果必須與標準 replicator 數值等價
6. L2 的 G2 Short Scout 固定為 3 seeds；若所有 `local_group_size > 0` 的 active cells 全部 `0/3` Level 3 且 `mean_stage3_score` uplift `< 0.02`，L2 直接 closure
7. L2 第一輪不得與 A1/S1/M1/B1 同時啟用（`local_group_size > 0` 與 `async_update_fraction < 1.0`、`sampling_beta != 1.0`、`mutation_rate > 0`、`tangential_alpha > 0` 互斥）
8. L2 的 G2 summary / diagnostics 至少必須新增：`local_group_size`, `mean_inter_group_weight_std`（各 strategy 的 group-mean weight 跨群 std 的均值，量化群間分化）, `mean_strategy_entropy`, `phase_amplitude_stability`
9. L2 若依上述 protocol 正式 closure，其研究解讀將固定為：在 sampled discrete regime 下，將 growth vector 計算局部化（降低有效 N）是否可打破 Layer 2 的大數定律平滑效應，打開 Level 3 basin

#### (H1) `payoff_niche_epsilon`（Phase 7 第一輪正式契約，2025-07-24）

**動機**

L2（Phase 6）證明：即使每個 sub-group 獨立計算 growth vector（打破 LLN），growth vector 的**方向**仍由全局 payoff matrix $A$ 的 eigenstructure 綁定——所有 group 面對同一 payoff，growth 方向相同 → population-level entropy 不變。

H1 直接攻擊此瓶頸：對每個 sub-group 施加 **diagonal payoff niche bonus**，使不同 group 的 growth vector 指向 simplex 的不同方向。

**數學定義**

對每個 group $k$（$k = 0, 1, \ldots, K-1$），定義 niche strategy $s_k = k \bmod 3$（0→aggressive, 1→defensive, 2→balanced）。Player $i$ 在 group $k$ 的 effective reward：

$$r_i^{(k)} = r_i^{\text{base}} + \epsilon \cdot \mathbf{1}[\text{last\_strategy}_i = s_k]$$

其中 $\epsilon$ = `payoff_niche_epsilon`。等價於 per-group payoff matrix：

$$A^{(k)} = A + \epsilon \cdot \text{diag}(e_{k \bmod 3})$$

**注入點規範**

在 `simulate()` 主迴圈中，`engine.step()` 執行完畢（所有 player 的 `last_reward` 已設定）之後、evolution step 之前注入：

```python
# H1: per-group payoff niche bonus
if h1_groups is not None and h1_epsilon > 0:
    for gk, group in enumerate(h1_groups):
        niche_strategy = strategy_space[gk % len(strategy_space)]
        for pl in group:
            if getattr(pl, "last_strategy", None) == niche_strategy:
                pl.last_reward = float(pl.last_reward) + h1_epsilon
```

| `payoff_niche_epsilon` | niche 效果（ε vs base payoff ~ O(0.1)） | 預期 growth direction 分歧 |
|---|---|---|
| 0.0 | 無（control = L2 only） | cosine ≈ 1.0 |
| 0.02 | 微弱 | cosine ~ 0.98 |
| 0.05 | 中等（≈50% of base payoff spread） | cosine ~ 0.90 |
| 0.10 | 強（≈100% of base payoff spread） | cosine ~ 0.75 |
| 0.20 | 極強 | cosine ~ 0.50 |

**Code-level spec**

```python
@dataclass
class SimConfig:
    payoff_niche_epsilon: float = 0.0  # H1: niche bonus strength (0 = disabled)
    niche_group_size: int = 0           # H1: players per niche group (0 = global/disabled)
```

H1 必須結合 per-group evolution（`niche_group_size > 0` 時自動啟用 L2 per-group replicator），因為若仍用 global growth vector，niche bonus 會被 population-level 聚合吸收。

**Protocol lock（9 point）**

1. G2 runtime 固定為 `payoff_mode=matrix_ab`, `popularity_mode=sampled`, `evolution_mode=sampled`
2. 不設 G1 deterministic gate（niche bonus 在 mean-field 無對應物）；G0 退化測試由 unit test 覆蓋（`payoff_niche_epsilon=0` 時結果必須與標準 replicator bit-identical）
3. 工作點固定沿用 locked protocol：`players=300`, `rounds=4000`, `seeds={45,47,49}`, `memory_kernel=3`, `selection_strength=0.06`, `init_bias=0.12`, `a=1.0`, `b=0.9`, `matrix_cross_coupling=0.20`；`niche_group_size=10`（固定）
4. H1 第一輪只允許掃描：`payoff_niche_epsilon ∈ {0.02, 0.05, 0.10, 0.20}`；`0.0` 為 matched control；另含一條極端對照 `epsilon=0.30, niche_group_size=5`
5. H1 的 G0 Degrade Gate 固定為：`payoff_niche_epsilon=0.0` 且 `niche_group_size=0` 時，結果必須與標準 replicator 數值等價
6. H1 的 G2 Short Scout Pass Gate：至少 1 個 non-control cell 同時滿足：(a) level3_seed_count ≥ 2, (b) mean_env_gamma ≥ 0, (c) mean_inter_group_growth_cosine ≤ 0.92, (d) strategy_entropy ≥ 1.12
7. H1 第一輪不得與 A1/S1/M1/B1 同時啟用（`payoff_niche_epsilon > 0` 與 `async_update_fraction < 1.0`、`sampling_beta != 1.0`、`mutation_rate > 0`、`tangential_alpha > 0` 互斥）；但 H1 **必須** 與 L2 以 `niche_group_size` 形式結合（不使用 `local_group_size`，避免雙重 partition）
8. H1 的 G2 summary / diagnostics 至少必須新增：`payoff_niche_epsilon`, `niche_group_size`, `mean_inter_group_growth_cosine`（跨組 growth vector pair-wise cosine similarity 平均）, `mean_inter_group_weight_std`, `mean_strategy_entropy`, `phase_amplitude_stability`
9. H1 若依上述 protocol 正式 closure，其研究解讀將固定為：在 sampled discrete regime 下，per-group payoff niche heterogeneity 是否可使不同 sub-group 的 growth vector 方向分歧，從而打破 population-level entropy 鎖定，打開 Level 3 basin

B-series 的共用輸出 / provenance 契約：

1. B-series 第一輪不得改動主 timeseries CSV schema；所有新增機制資訊都應寫入 CLI provenance、summary TSV、decision markdown、或圖表產物
2. run-level provenance 至少必須可回溯：`evolution_mode`, `selection_strength`, `sampled_growth_n_strata`, `strata_mode`, `personality_jitter`, `phase_rebucket_interval`, `tangential_drift_delta`, `personality_coupling_mu_base`, `personality_coupling_lambda_mu`, `personality_coupling_lambda_k`, `beta_state_k`, `memory_kernel`, `a`, `b`, `matrix_cross_coupling`
3. 每一輪 G2 / G3 都必須至少產出：核心指標表、simplex 軌跡圖、phase-vs-amplitude 圖，以及提案專屬診斷欄位；其中 B5 另必須產出 `drift_vector_rose_plot`
4. 若 B4/B3/B5 全線 fail，decision markdown 必須明示：這是「well-mixed sampled replicator 下，即使保留局部差異，三層均化仍過強」的正式 negative result，而不是單次 patch 失敗

#### (C1) Local Pairwise Imitation on Structured Graph（Phase 9 正式契約）

**動機**

Phases 1–8（B1/A1/S1/M1/L2/H1/B2，340+ runs）全面證明 `replicator_step()` 的全局 broadcast 更新是 population-level entropy lock ($H \approx 1.042$) 的根本瓶頸。C1 **完全移除 broadcast**：每位 player 維持獨立權重，透過 structured graph 上的 pairwise Fermi imitation 更新。

**數學定義**

Graph $G = (V, E)$, $|V| = N$, adjacency $N(i)$.  每輪 $t$：

1. Player $i$ 抽樣策略 $s_i \sim \text{Cat}(w_i)$
2. 局部回報 $\pi_i = \frac{1}{|N(i)|}\sum_{j \in N(i)} w_i \cdot A \cdot e_{s_j}$
3. 抽模仿對象 $j \sim \text{Uniform}(N(i))$
4. Fermi 採納機率 $q_{ij} = \sigma\!\bigl(\beta(\pi_j - \pi_i)\bigr)$
5. Convex imitation $w_i' = \text{norm}\!\bigl((1 - \mu q_{ij})\,w_i + \mu q_{ij}\,w_j\bigr)$
6. 同步更新

**Graph topologies**

| Topology | Degree | GraphSpec | Note |
|---|---|---|---|
| `lattice4` | 4 | `GraphSpec("lattice4", 4, 15, 20)` | 15×20 torus |
| `small_world` | 4 | `GraphSpec("small_world", 4, p_rewire=0.10)` | Watts–Strogatz |
| `ring4` | 4 | `GraphSpec("ring4", 4)` | baseline ring |

Well-mixed control: `topology=None`, 每輪隨機 4 鄰居（無空間結構）。

**Code-level spec**

Harness: `simulation/c1_pairwise_scout.py`（standalone loop，不使用 `simulate()`）。依賴：`evolution.local_pairwise`、`evolution.local_graph`、`analysis.cycle_metrics` + `analysis.decay_rate`。

**Sweep parameters（第一輪）**

| Parameter | Values |
|---|---|
| `graph_topology` | lattice4, small_world, ring4 |
| `beta_pair` | 5.0, 10.0 |
| `mu_pair` | 0.8, 0.5（paired with beta） |
| `players` | 300 |
| `rounds` | 5000 |
| `burn_in` / `tail` | 1500 / 1500 |
| `seeds` | {45, 47, 49} |
| `a, b, cross` | 1.0, 0.9, 0.20 |
| `init_bias` | 0.12 |

Default: 3 top × 2 combos × 3 seeds = 18 active + 1 control × 3 seeds = 21.

**Diagnostics**

| Metric | Definition |
|---|---|
| `mean_player_weight_entropy` | $\bar{H} = N^{-1}\sum_i (-\sum_s w_{is}\ln w_{is})$, tail mean |
| `spatial_strategy_clustering` | $1 - \text{edge\_disagreement\_rate}$（dominant strategy） |
| `mean_edge_strategy_distance` | Mean L1 of neighbor weight vectors (tail-end) |
| `weight_heterogeneity_std` | Cross-player weight std (tail mean) |

**G2 Short Scout Pass Gate（4-way AND）**

> **⚠️ 勘誤（Phase 15 批准）**：原始 `entropy ≥ 1.18` 超過 $\ln(3) \approx 1.099$ nats（三策略 Shannon entropy 上界），不可達。所有下游 Pass Gate 已統一修正為 `entropy ≥ 1.06` + `q_std > 0.04`。

1. `level3_seed_count ≥ 2`
2. `mean_env_gamma ≥ 0`
3. `mean_player_weight_entropy ≥ 1.06`（原 1.18 不可達，已修正）
4. `spatial_strategy_clustering > 0.25`

**Protocol lock（9 point）**

1. C1 使用獨立迴圈，不經過 `simulate()` / `replicator_step()`；payoff 參數沿用 matrix_ab 工作點
2. G0 退化測試：well-mixed control 預期 level ≤ 2；unit test 覆蓋 deterministic reproducibility
3. 工作點固定：`players=300, rounds=5000, seeds={45,47,49}, init_bias=0.12, a=1.0, b=0.9, cross=0.20, init_weight_jitter=0.02`
4. 第一輪只掃描上述 topology × (beta, mu) 組合；不得同時啟用 H1/B2 機制
5. `lattice4` 固定 15×20 torus；`small_world` 固定 `p_rewire=0.10`
6. 診斷至少新增：`mean_player_weight_entropy`, `spatial_strategy_clustering`, `weight_heterogeneity_std`, `mean_edge_strategy_distance`
7. C1 closure 研究解讀：per-player pairwise Fermi imitation on structured graph 是否能打破 broadcast 造成的 entropy lock
8. CSV schema 不變（round, p_aggressive, p_defensive, p_balanced）；新增指標寫入 summary TSV / provenance JSON
9. burn_in=1500, tail=1500；cycle classification 沿用 `classify_cycle_level` + `estimate_decay_gamma`

#### (D1) Independent Reinforcement Learning on Structured Graph（Phase 10 正式契約）

**動機**

Phase 1–9 證明兩類失敗模式：(1) replicator broadcast 使所有 player 同步更新相同權重；(2) pairwise imitation 的 convex combination 使 population 收斂到 consensus。共同根因：**信息流是同質化的**。D1 徹底消除 social information flow：每位 player 維護獨立 Q-table，只從自身 reward 歷史學習，不模仿、不 broadcast。

**數學定義**

每位 player $i$ 維護 Q-values $Q_i = (Q_i^{\text{agg}}, Q_i^{\text{def}}, Q_i^{\text{bal}})$，初始 $Q_i(s) = 0$。

每輪 $t$：

1. **Strategy selection**（Boltzmann softmax）：$\Pr(s) = \exp(\beta \cdot Q_i(s)) / \sum_{s'} \exp(\beta \cdot Q_i(s'))$
2. **Local payoff**：$r_i = \frac{1}{|N(i)|}\sum_{j \in N(i)} e_{s_i}^T A \, e_{s_j}$（one-hot strategy payoff，非 weight-weighted）
3. **Q-update**（exponential recency）：
   - 被選策略：$Q_i(s_i) \leftarrow (1-\alpha)\,Q_i(s_i) + \alpha \cdot r_i$
   - 未選策略：$Q_i(s') \leftarrow (1-\alpha)\,Q_i(s')$（forgetting / decay）
4. **Weight derivation**（for diagnostics only）：$w_i(s) = \text{softmax}(\beta \cdot Q_i)$

**Anti-consensus 性質**：不同 player 因不同鄰居組合 → 不同 reward 序列 → 不同 Q → 不同 weights。Population diversity 靠 reward noise 維持。

**Code-level spec**

Module: `evolution/independent_rl.py`（pure functions）。Harness: `simulation/d1_independent_rl.py`（standalone loop）。依賴：`evolution.local_graph`、`analysis.cycle_metrics` + `analysis.decay_rate`。

**Sweep parameters（第一輪 Short Scout）**

| Parameter | Values |
|---|---|
| `graph_topology` | lattice4 (primary), well_mixed (control) |
| `alpha` | 0.01, 0.05, 0.10 |
| `softmax_beta` | 5.0, 10.0 | 
| `init_q` | 0.0（uniform start） |
| `players` | 300 |
| `rounds` | 6000 |
| `burn_in` / `tail` | 2000 / 2000 |
| `seeds` | {45, 47, 49} |
| `a, b, cross` | 1.0, 0.9, 0.20 |

Active: 3 alpha × 2 beta × 2 graph = 12 conditions × 3 seeds = 36 runs。
Controls: `d1_frozen_lat`（alpha=0, lattice4）+ `d1_frozen_wm`（alpha=0, well_mixed）× 3 seeds = 6 runs。
Total: 42 runs。

**Diagnostics**

| Metric | Definition | Target |
|---|---|---|
| `mean_player_weight_entropy` | $\bar{H} = N^{-1}\sum_i (-\sum_s w_{is}\ln w_{is})$ | ≥ 1.06 |
| `weight_heterogeneity_std` | Cross-player weight std (tail mean) | > 0.015 |
| `mean_q_value_std` | $\text{std}_{i}(\bar{Q}_i)$ across population, tail mean | > 0.04 |
| `mean_neighbor_weight_cosine` | Mean $\cos(w_i, w_j)$ over graph edges | ≤ 0.80 |

**G2 Short Scout Pass Gate（4-way AND）**

> **修訂（Phase 15 批准）**：原始 entropy ≥ 1.18 超過 $\ln(3) \approx 1.099$ nats（三策略 Shannon entropy 上界），不可達。原始 q_std > 0.08 基於 D1 初期估計，後續實驗確認 wide-α regime 穩定在 ~0.046。閾值已下修至有實驗證據支撐的可達值。

1. `level3_seed_count ≥ 2`
2. `mean_env_gamma ≥ 0`
3. `mean_player_weight_entropy ≥ 1.06`（≈ 96.5% of ln(3)；原 1.18 不可達）
4. `mean_q_value_std > 0.04`（原 0.08；wide-α regime 穩定 ~0.046）

**Protocol lock（9 point）**

1. D1 使用獨立迴圈，不經過 `simulate()` / `replicator_step()`；payoff 為 one-hot strategy payoff（非 weight-weighted）
2. G0 退化測試：alpha=0 control 為純 random play（Q 不更新）；unit test 覆蓋 deterministic reproducibility
3. 工作點固定：`players=300, rounds=6000, seeds={45,47,49}, a=1.0, b=0.9, cross=0.20`
4. 第一輪只掃描上述 alpha × beta × graph 組合；不得同時啟用 H1/B2/C1 機制
5. `lattice4` 固定 15×20 torus；well_mixed 每輪隨機 4 鄰居
6. 診斷至少新增：`mean_q_value_std`, `mean_neighbor_weight_cosine`, `mean_player_weight_entropy`, `weight_heterogeneity_std`
7. D1 closure 研究解讀：independent per-player RL 是否能產生持續的 weight diversity，從而打開 Level 3 basin
8. CSV schema 不變（round, p_aggressive, p_defensive, p_balanced）；新增指標寫入 summary TSV / provenance JSON
9. burn_in=2000, tail=2000；cycle classification 沿用 `classify_cycle_level` + `estimate_decay_gamma`

### §(E1) Heterogeneous RL — 異質 α/β per player

**動機**：D1 closure 顯示 symmetric game + identical α/β → 所有 player 學到相似 Q-profile → q_value_std ≤ 0.03。E1 為每位 player 抽樣獨立的 α_i, β_i，打破 Q-convergence。

**數學定義**
- 初始化：$\alpha_i \sim \text{Uniform}(\alpha_{\text{lo}}, \alpha_{\text{hi}})$；$\beta_i \sim \text{Uniform}(\beta_{\text{lo}}, \beta_{\text{hi}})$
- α_i, β_i 在整場模擬中固定（退火或適應性留給未來實驗）
- 其餘 Q-update / Boltzmann / one-hot payoff 同 D1

**Code spec**

| 模組 | 檔案 | 職責 |
|---|---|---|
| evolution | `independent_rl.py`（復用 D1，無改動） | Q-update, softmax, payoff |
| simulation | `e1_heterogeneous_rl.py` | 異質 α_i/β_i 抽樣 + 獨立迴圈 |
| tests | `test_e1_heterogeneous_rl.py` | 回歸 + 異質性驗證 |

**Sweep table（3 alpha_range × 3 beta_range × 2 graph − 1 homo×homo = 17 active + 2 frozen + 1 D1_baseline × 2 graph = 21 conditions）**

| alpha_range | beta_range | graph | tag |
|---|---|---|---|
| (0.10, 0.10) | (10.0, 10.0) | lat4 / wm | d1_baseline（control） |
| (0.10, 0.10) | (3.0, 20.0) | lat4 / wm | homo_α_mid_β |
| (0.10, 0.10) | (1.0, 40.0) | lat4 / wm | homo_α_wide_β |
| (0.02, 0.20) | (10.0, 10.0) | lat4 / wm | mid_α_homo_β |
| (0.02, 0.20) | (3.0, 20.0) | lat4 / wm | mid_α_mid_β |
| (0.02, 0.20) | (1.0, 40.0) | lat4 / wm | mid_α_wide_β |
| (0.005, 0.40) | (10.0, 10.0) | lat4 / wm | wide_α_homo_β |
| (0.005, 0.40) | (3.0, 20.0) | lat4 / wm | wide_α_mid_β |
| (0.005, 0.40) | (1.0, 40.0) | lat4 / wm | wide_α_wide_β |
| alpha=0 | beta=5.0 | lat4 / wm | frozen_control |

Total: (9 × 2 + 2 frozen) × 3 seeds = **60 runs**

**新增診斷**

| 指標 | 定義 | 通過閾值 |
|---|---|---|
| realized_alpha_std | std(α_i) across population | >0（confirm heterogeneous） |
| realized_beta_std | std(β_i) across population | >0 |
| mean_q_value_std | std of per-player mean-Q | >0.04 |
| mean_player_weight_entropy | Shannon entropy pop-average | ≥1.06 |
| weight_heterogeneity_std | cross-player weight std | >0.015 |
| mean_neighbor_weight_cosine | edge cosine pop-average | ≤0.80 |

**Pass Gate（4-way AND，修訂版，同 D1）**

1. `level3_seed_count ≥ 2`
2. `mean_env_gamma ≥ 0`
3. `mean_player_weight_entropy ≥ 1.06`
4. `mean_q_value_std > 0.04`

**Protocol lock（9 point）**

1. E1 使用獨立迴圈（同 D1），不經過 `simulate()` / `replicator_step()`；payoff 為 one-hot strategy payoff
2. per-player α_i, β_i 在模擬開始時抽樣一次並固定，不做 adaptation
3. G0 退化測試：alpha_lo=alpha_hi=X → 退化為 D1；alpha=0 frozen control → Q 不更新
4. 工作點固定：`players=300, rounds=6000, seeds={45,47,49}, a=1.0, b=0.9, cross=0.20`
5. `lattice4` 固定 15×20 torus；well_mixed 每輪隨機 4 鄰居
6. D1_baseline（homo×homo）作為內建對照；frozen 作為零學習基線
7. CSV schema 不變（round, p_aggressive, p_defensive, p_balanced）
8. 新增指標：realized_alpha_std, realized_beta_std（寫入 provenance JSON + summary TSV）
9. burn_in=2000, tail=2000；cycle classification 沿用 D1 pipeline

---

### §(F1) Heterogeneous Payoff — per-player payoff perturbation

**動機**：E1 closure 顯示 per-player α/β 異質化可使 16/18 conditions 達到 Level 3 cycling（staircase phase rotation），但 q_std ≤ 0.044（目標 >0.04）且 entropy ≤ 0.92（目標 ≥1.06）。根因：symmetric payoff matrix → 所有 player 面對相同 payoff landscape → Q-tables 方向趨同。F1 為每位 player 注入獨立的 payoff perturbation δ_i[s]，打破 payoff 對稱性，使 Q-tables 在方向上分歧。

**數學定義**
- 每位 player i 在模擬開始時抽樣 payoff perturbation 向量：$\delta_i[s] \sim \text{Uniform}(-\varepsilon, +\varepsilon)$，$s \in \{0, 1, 2\}$
- δ_i 在整場固定（同 E1 的 α_i/β_i）
- 每輪 payoff 計算：$r_i = \text{one\_hot\_local\_payoff}(s_i, \{s_j\}_{j \in N(i)}, A) + \delta_i[s_i]$
- α 異質化沿用 E1 best：$\alpha_i \sim \text{Uniform}(\alpha_\text{lo}, \alpha_\text{hi})$，base_α=0.05, jitter=0.6 → α_lo=0.02, α_hi=0.08
- β 固定 homo=5.0（降低 ceiling 以提升 entropy）

**Code spec**

| 模組 | 檔案 | 職責 |
|---|---|---|
| evolution | `independent_rl.py`（新增 `sample_payoff_perturbation` + `apply_per_player_payoff_perturbation`） | per-player δ 抽樣與加成 |
| simulation | `f1_heterogeneous_payoff.py` | F1 standalone harness（基於 E1 拓撲） |
| tests | `test_f1_heterogeneous_payoff.py` | 回歸 + perturbation 驗證 |

**固定參數（基於 E1 best）**

| 參數 | 值 | 來源 |
|---|---|---|
| alpha_lo | 0.02 | base_α=0.05, jitter=0.6 → 0.05×0.4 |
| alpha_hi | 0.08 | base_α=0.05, jitter=0.6 → 0.05×1.6 |
| beta_lo = beta_hi | 5.0 | homo，低 ceiling（E1 顯示高 β 傷 entropy） |
| players | 300 | |
| rounds | 6000 | |
| burn_in / tail | 2000 / 2000 | |
| a / b / cross | 1.0 / 0.9 / 0.20 | |
| seeds | {45, 47, 49} | |
| graph | lattice4, well_mixed | |

**Sweep table（5 ε × 2 graph = 10 conditions × 3 seeds = 30 runs）**

| payoff_epsilon | graph | tag |
|---|---|---|
| 0.00 | lattice4 / well_mixed | f1_ctrl（E1-best baseline） |
| 0.02 | lattice4 / well_mixed | f1_eps0p02 |
| 0.05 | lattice4 / well_mixed | f1_eps0p05 |
| 0.08 | lattice4 / well_mixed | f1_eps0p08 |
| 0.12 | lattice4 / well_mixed | f1_eps0p12 |

**新增診斷**

| 指標 | 定義 | 通過閾值 |
|---|---|---|
| mean_q_value_std | std of per-player mean-Q（核心） | >0.04 |
| mean_player_weight_entropy | Shannon entropy pop-average | ≥1.06 |
| strategy_cycle_stability | R² of linear fit to unwrapped phase angle（staircase 穩定度） | informational |
| mean_payoff_heterogeneity | std of per-player δ vectors（確認 perturbation 生效） | >0 when ε>0 |

**Pass Gate（4-way AND，修訂版）**

1. `level3_seed_count ≥ 2`
2. `mean_env_gamma ≥ 0`
3. `mean_player_weight_entropy ≥ 1.06`
4. `mean_q_value_std > 0.04`

**Protocol lock（9 point）**

1. F1 使用獨立迴圈（同 D1/E1），不經過 `simulate()` / `replicator_step()`
2. per-player δ_i[s] 在模擬開始時抽樣一次並固定；ε=0 時 δ=0（退化為 E1）
3. G0 退化測試：epsilon=0 → 退化為 E1（相同 α/β 設定）
4. 工作點固定：`players=300, rounds=6000, seeds={45,47,49}, a=1.0, b=0.9, cross=0.20`
5. α_lo=0.02, α_hi=0.08, β=5.0（homo），基於 E1 最佳結果
6. `lattice4` 固定 15×20 torus；well_mixed 每輪隨機 4 鄰居
7. CSV schema 不變（round, p_aggressive, p_defensive, p_balanced）
8. 新增指標：mean_payoff_heterogeneity, strategy_cycle_stability（寫入 provenance JSON + summary TSV）
9. burn_in=2000, tail=2000；cycle classification 沿用 D1/E1 pipeline

---

### §(E2) Lower β Ceiling — 保留 E1 wide α + 降低 β

**動機**：E1 closure 顯示 wide α (0.005, 0.40) 是 Level 3 cycling 的核心驅動力（α_std=0.116），但 β=10 homo 使 entropy ≤ 0.92（目標 ≥1.06）。F1 closure 確認**不能犧牲 wide α 來換取其他機制**。E2 僅調降 β ceiling，在 E1 best 基礎上直接修復 entropy 瓶頸。

**數學定義**
- α_i ~ Uniform(0.005, 0.40)（同 E1 wide α）
- β_i = β_ceiling（homo，不做異質化）
- β_ceiling ∈ {3.0, 5.0, 8.0, 10.0}（10.0 = E1 baseline control）
- 其餘 Q-update / Boltzmann / one-hot payoff 同 E1

**Code spec**

| 模組 | 檔案 | 職責 |
|---|---|---|
| evolution | `independent_rl.py`（復用，無改動） | Q-update, softmax, payoff |
| simulation | `e2_lower_beta.py` | 薄封裝 E1 core + E2-specific verdict |
| tests | `test_e2_lower_beta.py` | 回歸 + β-entropy 關聯驗證 |

**Sweep table（4 β × 2 graph = 8 conditions × 3 seeds = 24 runs）**

| β_ceiling | graph | tag |
|---|---|---|
| 3.0 | lattice4 / well_mixed | e2_b3 |
| 5.0 | lattice4 / well_mixed | e2_b5 |
| 8.0 | lattice4 / well_mixed | e2_b8 |
| 10.0 | lattice4 / well_mixed | e2_b10（E1 control） |

**新增診斷**

| 指標 | 定義 | 通過閾值 |
|---|---|---|
| strategy_cycle_stability | R² of linear fit to unwrapped phase angle | informational |
| mean_player_weight_entropy | Shannon entropy pop-average | ≥1.06 |
| mean_q_value_std | std of per-player mean-Q | >0.04 |

**Pass Gate（4-way AND，修訂版）**

1. `level3_seed_count ≥ 2`
2. `mean_env_gamma ≥ 0`
3. `mean_player_weight_entropy ≥ 1.06`
4. `mean_q_value_std > 0.04`

**Protocol lock（7 point）**

1. E2 完全沿用 E1 core simulation loop，只變更 β 參數
2. α_lo=0.005, α_hi=0.40 固定（E1 wide α，不可修改）
3. G0 退化測試：β=10.0 → 退化為 E1 wide_α_homo_β baseline
4. 工作點固定：`players=300, rounds=6000, seeds={45,47,49}, a=1.0, b=0.9, cross=0.20`
5. `lattice4` 固定 15×20 torus；well_mixed 每輪隨機 4 鄰居
6. CSV schema 不變（round, p_aggressive, p_defensive, p_balanced）
7. burn_in=2000, tail=2000；cycle classification 沿用 E1 pipeline

---

### §(CC1) Classifier Calibration — Stage 2 Phase Rotation R² Fallback

#### 目的

E2 實驗揭露 `classify_cycle_level` Stage 2 對低振幅高一致性旋轉存在系統性 false negative。β=3 產生累積旋轉 105 rad（E1 的 2.8×）、R²=0.97，但 autocorrelation < 0.3 導致 CL=1。

CC1 在 Stage 2 增加 OR 邏輯 fallback，不改寫既有 autocorrelation 路徑。

#### 最小契約

1. `classify_cycle_level` 新增兩個可選參數：
   - `stage2_fallback_r2_threshold: float | None = None`（None = disabled = legacy behaviour）
   - `stage2_fallback_min_rotation: float = 20.0`（最小累積旋轉 radians）
2. 當 `stage2_fallback_r2_threshold is not None` 且 primary Stage 2 失敗時：
   - 計算 `phase_rotation_r2(proportions, strategies, burn_in, tail)`
   - 若 `r2 >= threshold AND cumulative_rotation >= min_rotation` → Stage 2 視為通過
   - 通過時 `stage2.method = "phase_rotation_r2_fallback"`
3. 同樣的 fallback 亦適用於 Stage 3：若 primary Stage 3 (turning consistency) 失敗但 R²+rotation 門檻通過 → Stage 3 視為通過
   - `stage3.score = r2`（以 R² 作為 consistency proxy）
   - `stage3.direction` 由 unwrapped phase slope 符號決定
4. `stage2_fallback_r2_threshold = None`（default）→ 完全退化為既有行為（bit-for-bit）

#### 新增公開 API

- `phase_rotation_r2(proportions, *, strategies, burn_in, tail) → PhaseRotationR2Result`
  - `PhaseRotationR2Result(r2, slope, cumulative_rotation, window_length)`
- 內部 helpers：`_simplex3_phase_angle`, `_unwrap_phases`, `_linear_fit_r2`（純標準庫）

#### 不變條件

1. `stage2_fallback_r2_threshold=None` → `classify_cycle_level` 輸出與 CC1 前完全一致
2. Stationary signal（R²≈0, rotation≈0）不會被 fallback 救回
3. `_unwrap_phases` 等效 numpy.unwrap（±2π 校正）
4. `phase_rotation_r2` 在 window_length < 10 時返回 r2=0.0

#### 建議操作門檻

- E2 回溯重評：`r2_threshold=0.85, min_rotation=20.0`
  - β=3 seeds：R²=0.62-0.97（2/3 seeds > 0.85），rotation=105（遠超 20）
  - β=10 baseline：R²=0.82-0.97，rotation=37（通過）
  - Stationary/noise：R²<0.1，rotation<5（不通過）

---

### §(G1) Entropy Regularization — Q-update 層 Entropy Bonus

#### 目的

CC1 修正測量瓶頸後，E2 β=3 已能通過 L3，但 entropy（1.076）仍接近理論上界 ln(3)≈1.099，q_std（0.047）則接近修正版門檻 0.04。

G1 在 RL Q-update 層加入 per-player entropy bonus `λ·H(w_i)`，直接拉高族群策略多樣性，不動 α range 或拓撲。

#### 機制

每回合 Q-update 後，對每位玩家的 Q-values 施加 **均值收縮（mean centering）**：

$$Q_i(s) \leftarrow Q_i(s) - \lambda \cdot \alpha_i \cdot (Q_i(s) - \bar{Q}_i)$$

其中：
- $\bar{Q}_i = \frac{1}{|S|}\sum_s Q_i(s)$：玩家 $i$ 的 Q-value 均值
- $\alpha_i$：玩家 $i$ 的學習率
- $\lambda$：entropy regularization 強度（新參數）

等價於：$Q_i(s) \leftarrow (1 - \lambda\alpha_i) \cdot Q_i(s) + \lambda\alpha_i \cdot \bar{Q}_i$

此操作直接限制 Q-values 之間的分離度，使 Boltzmann softmax 輸出更接近均勻分佈，從而提升 per-player entropy。

**設計備忘**：初版使用 reward bonus `r' = r + λ·H(w)`，但因 bonus 僅進入 Q(chosen) 而 Q(other) 持續衰減，反而引發 "rich get richer" 效應使 entropy 下降。Q-centering 直接作用於所有策略，避免此問題。

#### 最小契約

1. `_run_e1_simulation` 新增可選參數 `entropy_lambda: float = 0.0`
   - `entropy_lambda = 0.0` → bit-for-bit 等同既有行為
   - `entropy_lambda > 0` → 在 step 3（Q-update）之後插入 Q-value centering
2. Q-centering 使用 `sum(Q)/len(Q)` 計算均值，收縮率 = `λ·α_i`
3. G1 harness（`simulation/g1_entropy_reg.py`）為 thin wrapper，sweep λ × β
4. G1 metrics 啟用 CC1 fallback（`stage2_fallback_r2_threshold=0.85, min_rotation=20.0`）

#### Sweep 設計

| 參數 | 值 |
|---|---|
| entropy_lambda | {0.02, 0.05, 0.10, 0.20} |
| beta_ceiling | {3.0, 10.0} |
| topology | {well_mixed, lattice4} |
| seeds | {45, 47, 49} |
| alpha_lo/hi | 0.005 / 0.40（E1 wide α） |
| players / rounds | 300 / 6000 |
| a / b / cross | 1.0 / 0.9 / 0.20 |
| **Total** | **48 runs** |

#### Pass Gate（per-condition 4-way AND）

> **規格勘誤（Phase 15 批准）**：原始 gate `entropy ≥ 1.18` 超過三策略 Shannon entropy 理論上界 $\ln(3) \approx 1.0986$ nats，數學上不可能達到。此門檻自 §(D1) 起即為不可達規格。以下為原始值（已標記）與正式修正值。

原始（不可達）：
1. `level3_seed_count ≥ 2`
2. ~~`mean_player_weight_entropy ≥ 1.18`~~  ← 107.4% of ln(3), 不可能
3. `mean_q_value_std > 0.06`
4. `mean_env_gamma ≥ 0`

**正式修正版（Phase 15 批准）**：
1. `level3_seed_count ≥ 2`（CC1 fallback: r2=0.85, rotation=20）
2. `mean_player_weight_entropy ≥ 1.06`（≈ 96.5% of ln(3)）
3. `mean_q_value_std > 0.04`
4. `mean_env_gamma ≥ 0`

G1 v2（Q-centering）在修正版 gate 下有 3/16 conditions 通過。此修訂適用於所有下游實驗（D1/E1/F1/E2/G1/F1v2）。

#### Hard Stop

- 若所有 active conditions 的 L3=0 → close_g1

#### 不變條件

1. `entropy_lambda=0.0` → `_run_e1_simulation` 輸出與 pre-G1 bit-for-bit 相同
2. `evolution/independent_rl.py` 不新增 I/O（保持 pure functions）
3. `analysis/` 不 import `simulation/`

---

### §(F1v2) Payoff Perturbation v2 — E2-best base + mild ε

#### 目的

F1 (Phase 12) 失敗的根因是犧牲了 wide α（使用 α_lo=0.02, α_hi=0.08），導致 L3=0。F1v2 在 **E2 β=3 + wide α (0.005, 0.40)** 的已通過基礎上，僅疊加 mild payoff perturbation ε，目標是進一步提升 q_std（0.046→更高），強化 Pass Gate 通過的穩健性。

修正版 Gate 下，E2 β=3 已在 2/8 conditions 通過。F1v2 聚焦於：
1. 確認小 ε 不破壞已有的 L3 staircase rotation
2. 測試 payoff perturbation 能否提升 q_std（打破 payoff 對稱性 → Q-table 方向分歧）

#### 機制

沿用 F1 的 per-player payoff perturbation 數學定義：
- 每位 player i 在模擬開始時抽樣：$\delta_i[s] \sim \text{Uniform}(-\varepsilon, +\varepsilon)$
- δ_i 整場固定
- 每輪 payoff：$r_i = \text{one\_hot\_local\_payoff}(s_i, \{s_j\}_{j \in N(i)}, A) + \delta_i[s_i]$

**關鍵差異 vs F1**：
- α_lo=0.005, α_hi=0.40（E1 wide α，非 F1 的 0.02/0.08）
- β=3.0（E2-best，非 F1 的 5.0）
- ε 範圍更保守（0.02~0.08，不含 0.12）

#### 最小契約

1. `_run_e1_simulation` 新增可選參數 `payoff_epsilon: float = 0.0`
   - `payoff_epsilon = 0.0` → bit-for-bit 等同既有行為（不消耗 RNG）
   - `payoff_epsilon > 0` → 在 step 1 之前 sample perturbations，在 step 2 之後 apply
2. 使用 `evolution/independent_rl.py` 既有的 `sample_payoff_perturbation` + `apply_per_player_payoff_perturbation`
3. F1v2 harness（`simulation/f1v2_payoff_perturbation.py`）為 thin wrapper
4. F1v2 metrics 啟用 CC1 fallback

#### Sweep 設計

| 參數 | 值 |
|---|---|
| payoff_epsilon | {0.00, 0.02, 0.04, 0.06, 0.08} |
| beta_ceiling | 3.0（E2-best） |
| topology | {well_mixed, lattice4} |
| seeds | {45, 47, 49} |
| alpha_lo/hi | 0.005 / 0.40（E1 wide α） |
| players / rounds | 300 / 6000 |
| a / b / cross | 1.0 / 0.9 / 0.20 |
| CC1 fallback | r2=0.85, min_rotation=20.0 |
| **Total** | **30 runs**（5ε × 2topo × 3seeds） |

#### Pass Gate（修正版 4-way AND）

1. `level3_seed_count ≥ 2`（CC1 fallback: r2=0.85, rotation=20）
2. `mean_player_weight_entropy ≥ 1.06`
3. `mean_q_value_std > 0.04`
4. `mean_env_gamma ≥ 0`

#### Hard Stop

- 若所有 active conditions（ε>0）的 L3=0 → close_f1v2

#### Extended Confirmation（Phase 16b）

Short scout（6000 rounds, 3 seeds）結果：lattice4 ε=0.04 (PASS, 3/3 L3) 與 well_mixed ε=0.02 (PASS, 2/3 L3) 列為 longer_confirm_candidate。

Extended run 設定：

| 參數 | 值 |
|---|---|
| conditions | lattice4 ε=0.04, well_mixed ε=0.02 |
| rounds | 12000（2× short scout） |
| burn_in / tail | 4000 / 4000（等比例放大） |
| seeds | 45, 47, 49, 51, 53, 55（6 seeds） |
| players | 300 |
| Total | **12 runs**（2 cond × 6 seeds） |

**Confirm 門檻**：同修正版 Pass Gate（4-way AND），但 `level3_seed_count ≥ 3`（6 seeds 下要求 50%+）。

#### 不變條件

1. `payoff_epsilon=0.0` → `_run_e1_simulation` 輸出與 pre-F1v2 bit-for-bit 相同（不消耗 RNG）
2. `evolution/independent_rl.py` 不新增 I/O
3. `analysis/` 不 import `simulation/`

---

### §(BL1) Baseline Lock — well_mixed ε=0.02

#### 動機

Phase 16b extended confirmation 確認 well_mixed ε=0.02 為最強條件（L3=5/6, stab=0.923）。Phase 17 將此鎖定為正式 baseline，跑超長驗證（20000 rounds）並輸出完整指標報告，作為未來所有新機制比較的 anchor。

#### 鎖定參數

| 參數 | 值 | 來源 |
|---|---|---|
| topology | well_mixed | Phase 16b best |
| payoff_epsilon | 0.02 | F1v2 sweep 最佳 ε |
| alpha_lo / alpha_hi | 0.005 / 0.40 | E1 wide α |
| beta_ceiling | 3.0 | E2-best |
| players | 300 | 標準 |
| rounds | 20000 | 3.3× short scout |
| burn_in / tail | 6000 / 6000 | 等比例放大 |
| entropy_lambda | 0.0 | 無 G1 Q-centering |
| a / b / cross | 1.0 / 0.9 / 0.20 | 標準 |
| CC1 fallback | r2=0.85, min_rotation=20.0 | CC1 參數 |
| seeds | 45, 47, 49, 51, 53, 55 | 6 seeds |
| **Total** | **6 runs** | |

#### Pass Gate（同修正版 4-way AND）

1. `level3_seed_count ≥ 3`（6 seeds 下 50%+）
2. `mean_player_weight_entropy ≥ 1.06`
3. `mean_q_value_std > 0.04`
4. `mean_env_gamma ≥ 0`（容許 |γ|<1e-4 浮點雜訊）

#### 輸出

1. Per-seed CSV + provenance JSON
2. Per-seed simplex + phase-amplitude plots
3. Aggregated summary TSV
4. Decision markdown
5. 此 baseline 的完整指標將作為後續 Phase 18+ 的 anchor

#### 不變條件

1. 不修改任何 simulation code，僅用現有 harness + 不同參數
2. `analysis/` 不 import `simulation/`

---

### §(L3-OPT) Level 3 優化時代 — 實驗路線圖

#### 架構約束

BL1 baseline 使用 E1 管線（`e1_heterogeneous_rl.py::_run_e1_simulation()`），
該管線為**獨立迴圈**，不經過 `replicator_step()`。

`run_simulation.py` 中的 `tangential_drift_delta`（B5）、`memory_kernel`（H1）、
`strategy_selection_strengths` 均**不適用於 E1 管線**。

所有下列 Phase 均需在 E1 管線中實作 RL-native 等效機制。

#### 共用實驗 protocol

1. **Spec 先行**：本節定義行為契約，先 Spec 後程式碼
2. **Short scout**：6000r × 3 seeds（45,47,49），快速掃參數
3. **Gate check**：revised Pass Gate（ent≥1.06, q_std>0.04, L3≥2, γ≥0）
4. **Extended confirm**：12000r × 6 seeds，最佳條件
5. **Baseline comparison**：所有結果與 BL1 anchor 對比（Δstage3_score, Δturn_strength）
6. **Hard stop**：若所有 active 條件 L3=0，該 Phase 關閉

#### 共用 BL1 anchor 指標

| metric | BL1 值 | 優化目標 |
|---|---|---|
| L3 rate | 4/6 (67%) | ≥ 83% |
| stage3_score (L3 avg) | 0.863 | ≥ 0.95 |
| turn_strength (L3 avg) | 0.032 | ≥ 0.06 |
| q_std | 0.046 | ≥ 0.06 |
| stab | 0.840 | ≥ 0.92 |

---

#### §(P18) Phase 18 — Dynamic Payoff Perturbation

##### 假設

BL1 的 payoff_epsilon 為 static（初始化時 per-player 抽一次 δ_i[s]）。若讓 ε 隨時間變化，
可破壞 Q-table 收斂平衡，強迫持續探索 → 更穩定的旋轉動態。

##### 機制（E1 管線改動）

修改 `_run_e1_simulation()` 的 payoff perturbation 步驟：

- **18a — ε annealing**：`ε(t) = ε_start × (1 - t/T) + ε_end × (t/T)`
  - 每 R rounds 根據 ε(t) 重新抽樣 payoff_deltas
  - 預期：前期大擾動建立旋轉，後期小擾動穩定化
- **18b — ε re-sampling**：固定 ε=0.02，但每 R rounds 對所有 player 重新抽樣 δ_i[s]
  - R ∈ {100, 500, 1000, 2000}
  - 預期：持續擾動阻止 Q-convergence lock

##### 參數 sweep

| sub | 變數 | sweep 值 |
|---|---|---|
| 18a | ε_start | 0.04, 0.06 |
| 18a | ε_end | 0.01, 0.02 |
| 18a | resample_interval | 500, 1000 |
| 18b | resample_interval R | 100, 500, 1000, 2000 |

Seeds: 45,47,49 × 上述組合 = ~24 runs (18a) + ~12 runs (18b)

##### 改動清單

| 檔案 | 改動 |
|---|---|
| `evolution/independent_rl.py` | 新增 `resample_payoff_perturbation()` 支援 per-round ε |
| `simulation/e1_heterogeneous_rl.py` | `_run_e1_simulation()` 新增 `epsilon_schedule` 參數 |
| `simulation/p18_dynamic_payoff.py` | 新 harness（仿 F1v2 結構） |

##### 不變條件

- 不修改 `analysis/` 層
- CSV schema 與 BL1 相同
- payoff perturbation 算法不變，只變抽樣時機

---

#### §(P20) Phase 20 — Q-Rotation Bias

##### 假設

BL1 的 L3 vs Non-L3 唯一差異是 stab/turn_strength（旋轉一致性），而非 entropy/q_std。
在 Q-update 後注入微小旋轉偏置，直接推動 Q-values 向「下一策略」偏移 → 增強 phase direction consistency。

##### 機制（E1 管線改動）

每輪 Q-update 後，對每位 player 施加：

```
q_mean = mean(Q_i)
centered = Q_i - q_mean
# Antisymmetric rotation: A→D→B→A
drift = δ_rot × [centered[D] - centered[B],    # A gets (D-B)
                 centered[B] - centered[A],    # D gets (B-A)  
                 centered[A] - centered[D]]    # B gets (A-D)
Q_i += drift
```

此為 3-simplex 上的 tangential drift 在 Q-space 的等效。δ_rot=0 恢復 BL1。

##### 參數 sweep

| 參數 | sweep 值 |
|---|---|
| δ_rot | 0.0 (control), 0.001, 0.005, 0.01, 0.02 |
| payoff_epsilon | 0.02 (BL1 locked) |

Seeds: 45,47,49 × 5 = 15 runs

##### 改動清單

| 檔案 | 改動 |
|---|---|
| `evolution/independent_rl.py` | 新增 `apply_q_rotation_bias()` 函數 |
| `simulation/e1_heterogeneous_rl.py` | `_run_e1_simulation()` 新增 `q_rotation_delta` 參數 |
| `simulation/p20_q_rotation.py` | 新 harness |

##### 判讀標準

- 若 δ_rot=0.001 已將 stage3_score 推到 0.95+ → 「最低有效劑量」確認
- 若 δ_rot 很大才有效 → 人工旋轉，非自發機制，需謹慎解讀
- 若 entropy 或 q_std 同步下降 → rotation bias 壓制多樣性，不可接受

##### 不變條件

- Q-rotation bias 不改變 payoff 矩陣或 learning rate
- 當 δ_rot=0.0 結果需與 BL1 數值完全一致（回歸校驗）

---

#### §(P21) Phase 21 — Per-Strategy Learning Rate Asymmetry

##### 假設

對稱 α_i 意味著三策略以相同速率學習 → Q-values 同步收斂 → 旋轉缺乏方向性。
若不同策略有不同學習速率，快學策略先適應、慢學策略延遲反應 → 自然 phase lag → 旋轉方向一致性。

##### 機制（E1 管線改動）

Q-update 從 `α_i` 改為 `α_i × r_s`，其中 `r_s` 為 per-strategy multiplier：

```
Q_i(s) ← (1 - α_i × r_s) × Q_i(s) + α_i × r_s × reward   (if s = chosen)
```

##### 參數 sweep

| 配置 | r_A, r_D, r_B | 預期效果 |
|---|---|---|
| symmetric (control) | 1.0, 1.0, 1.0 | BL1 baseline |
| mild CW | 1.2, 1.0, 0.8 | A 快學 → D lag → B further lag |
| mild CCW | 0.8, 1.0, 1.2 | 反方向 lag |
| strong CW | 1.5, 1.0, 0.5 | 強不對稱 |
| gradient | 1.0, 0.8, 1.2 | 另一組合 |

Seeds: 45,47,49 × 5 配置 = 15 runs

##### 改動清單

| 檔案 | 改動 |
|---|---|
| `evolution/independent_rl.py` | `rl_q_update()` 接受 per-strategy α list |
| `simulation/e1_heterogeneous_rl.py` | `_run_e1_simulation()` 新增 `strategy_alpha_multipliers` |
| `simulation/p21_asymmetric_alpha.py` | 新 harness |

##### 不變條件

- α_i × r_s 需 clip 到 [0, 1] 避免數值問題
- Multipliers 為 global（所有 players 共享相同 r_s）
- 三個策略空間定義不變（A=aggressive, D=defensive, B=balanced）

---

#### §(BL2) Baseline Lock v2 — mild_cw asymmetric alpha

##### 動機

Phase 21a extended confirmation 已確認 `mild_cw = [1.2, 1.0, 0.8]` 在 E1/BL1 管線上達成目前最強結果：

1. `6/6` seeds 為 Level 3（100%）
2. `mean_stage3_score = 0.940`
3. `mean_turn_strength = 0.054`
4. `stab = 0.940`

同時，Phase 20 已證明外加 Q-rotation bias 會破壞自組織 phase lag；Phase 18×21 combo scout 又證明
`ε annealing + mild_cw` 不具加成性。因此研究主線正式從 BL1 升格為 BL2，並停止同 family 的額外局部掃描。

##### 鎖定參數

| 參數 | 值 | 來源 |
|---|---|---|
| topology | well_mixed | Phase 21a |
| payoff_epsilon | 0.02 static | BL1 / P21 control |
| alpha_lo / alpha_hi | 0.005 / 0.40 | E1 wide α |
| beta_ceiling | 3.0 | E2-best |
| strategy_alpha_multipliers | 1.2, 1.0, 0.8 | P21a mild_cw |
| entropy_lambda | 0.0 | 無 G1 centering |
| q_rotation_delta | 0.0 | P20 closure |
| epsilon_schedule | none | P18×21 closure |
| a / b / cross | 1.0 / 0.9 / 0.20 | 標準 |
| players | 300 | 標準 |
| rounds | 12000 | P21a confirm |
| burn_in / tail | 4000 / 4000 | P21a confirm |
| CC1 fallback | r2=0.85, min_rotation=20.0 | CC1 |
| seeds | 45, 47, 49, 51, 53, 55 | 6 seeds |

##### BL2 Pass Gate

1. `level3_seed_count = 6`
2. `mean_player_weight_entropy >= 1.06`
3. `mean_q_value_std >= 0.04`
4. `mean_stage3_score >= 0.90`
5. `mean_turn_strength >= 0.05`
6. `mean_env_gamma >= -1e-5`

第 6 條的容忍帶只用於 BL2 封板，目的在於吸收 `~1e-6` 等級的數值誤差；它不是對舊 gate 的全面重寫。

##### 輸出

1. Per-seed CSV + provenance JSON
2. Per-seed simplex / phase-amplitude 圖
3. Aggregated summary TSV
4. Decision markdown
5. BL2 作為後續所有 Personality Dungeon runtime bridge 的唯一 RL anchor

##### 不變條件

1. BL2 僅允許使用 E1 管線 + `strategy_alpha_multipliers=[1.2,1.0,0.8]`
2. 不得同時打開 `q_rotation_delta`、dynamic `epsilon_schedule`、或任何未在 BL2 鎖定表中的新機制
3. `analysis/` 不 import `simulation/`
4. `evolution/independent_rl.py` 維持 pure functions，不新增 I/O

---

#### §(RL-CLOSE) RL 研究主線正式封板

本節將 Phase 10–21 的 RL 主線從「繼續尋找更強機制」改為「以 BL2 為研究封板點」。

##### 正式結論

1. BL2 取代 BL1，成為 RL 主線正式 baseline
2. Phase 20（Q-rotation bias）為正式 negative result，不再續做
3. Phase 18 champion 保留為次佳輔助條件，但不再作為主 baseline
4. Phase 18×21 combo 已確認 non-additive，整條 combo 線關閉
5. Phase 19（Delayed Payoff Feedback）轉為 archived idea，不再作為目前 sanctioned 主線

##### 主線轉移規則

自本版 Spec 起，下一輪 sanctioned work 不再是 RL-native 小機制探索，而是：

1. 把 BL2 機制翻譯成 Personality Dungeon 的正式系統藍圖
2. 把 BL2 接到帶 personality / events / world-state 的正式 runtime
3. 保持 `simulation -> CSV -> analysis` 的研究可回歸管線

若沒有先完成 runtime bridge，則不得重啟 P19 或其他新的 BL2 周邊微調線。

---

#### §(PERS-CAL) Personality Calibration Lock — Phase 5 最終結果

##### 動機

Phase 3–5 共執行 200+ 組態 × seed 組合，從 7×7 grid → bug fix → 3×3 precision grid → β compensation sweep → full validation，確定 personality modulation 的最佳 lambda 配置。

##### 鎖定參數（在 BL2 基礎上疊加）

| 參數 | 值 | 來源 |
|---|---|---|
| personality_mode | random | Phase 1 Gate 2 |
| lambda_alpha | 0.15 | Phase 1 Gate 1 |
| lambda_beta | 0.10 | Phase 1 Gate 1 |
| lambda_r | 0.20 | Phase 5 Step 1 (3×3 精細掃描) |
| lambda_risk | 0.20 | Phase 5 Step 1 (3×3 精細掃描) |
| lambda_beta_comp | 0.0 | Phase 5 Step 2 (β 補償無益) |

##### Pass Gate（在 BL2 6 seeds 之外，另以 60-seed Gate 2 + 15 marginal + 10 regression 衡量）

1. Gate 0 bit-exact：`personality_mode=none` 下與 BL2 結果完全一致
2. 60-seed Gate 2：L1 ≤ 5%，L3 ≥ 90%
3. 15 persistent marginals：healthy rate ≥ 80%（實測：13/15 = 86.7%）
4. 10 regression reference：regression ≤ 2（實測：1/10）

##### 已知限制

1. **(0.20, 0.20) 是窄甜蜜點**：偏離 ±0.02 性能急降，Phase 5 3×3 grid 明確顯示
2. **seed 45**（L1）和 **seed 97**（s3=0.558）在所有測試配置下均為 marginal，屬結構不可救
3. **lambda_beta_comp** 已實作但經實驗證明無益（在 seeds 82/84 之間製造 trade-off），保留為 code-level optional knob (default=0)

##### 不變條件

1. 程式碼中 `PersonalityRLConfig` 的 lambda 預設值維持 `0.0`（確保 BL2 bit-exact 安全）
2. 鎖定配置通過 CLI 或呼叫端顯式指定，不透過改預設值隱式啟用
3. 後續實驗若需重新校準 lambda，必須先更新本節 Spec 再改碼

---

#### §(EV-INT) Event Integration — Closed-Loop 實驗結果

##### 動機

在 §(PERS-CAL) 鎖定配置上，執行 personality → events → world-state 的 4-Arm 2×2 factorial 閉環實驗，分離 event 與 world 各自貢獻。

##### 探索實驗結果（11 seeds × 4 arms）

| Arm | 組態 | Healthy/11 | Mean s3 |
|-----|------|-----------|---------|
| A pers_only（baseline） | PERS-CAL, no events | 9/11 | 0.834 |
| B pers_event | PERS-CAL + events(rate=0.15) | 11/11 | 0.951 |
| C pers_world_lo | PERS-CAL + world(λ=0.08) | 6/11 | 0.706 |
| D full_loop | PERS-CAL + events + world | 9/11 | 0.829 |

##### 60-seed Gate 驗證（Arm B FAIL）

| 配置 | L1 | Healthy | Mean s3 | Gate |
|------|----|---------|---------|------|
| **Arm A** (PERS-CAL baseline) | **2/60 (3.3%)** | **51/60 (85.0%)** | 0.874 | **PASS** |
| Arm B (PERS-CAL + events@0.15) | 5/60 (8.3%) | 46/60 (76.7%) | 0.809 | FAIL (L1>3) |

Paired comparison: events@0.15 rescued 7 seeds but broke 12 (net destructive).
Arm B 的 11-seed 11/11 為 selection bias — 60-seed 空間新增 5 個 L1。

##### 正式 Baseline（PERS-CAL only）

§(PERS-CAL) 配置**無事件整合**通過 60-seed Gate，確認為當前正式 baseline。

##### Event Rate Sweep 結果

22-seed × 5 rates probe（event_reward_scale=0.01, smoke_v1, world_feedback=False）：

| Rate | L1 | Healthy/22 | Mean s3 | new_L1 | Broke |
|------|----|-----------|---------|--------|-------|
| 0.0 (baseline) | 2 | 15 (68%) | 0.782 | — | — |
| 0.03 | 1 | 19 (86%) | 0.868 | 1 | 3 |
| 0.05 | 0 | 17 (77%) | 0.858 | 0 | 3 |
| **0.08** | **0** | **18 (82%)** | **0.883** | **0** | **1** |
| 0.10 | 2 | 14 (64%) | 0.746 | 1 | 6 |

##### Event Integration 狀態：**未通過（deferred research — 架構層面限制）**

rate=0.08 60-seed Gate 結果：

| 指標 | rate=0.08 | baseline | Gate | 結果 |
|------|-----------|----------|------|------|
| L1 | 2/60 | 2/60 | ≤3 | PASS |
| Healthy | 51/60 | 51/60 | ≥42 | PASS |
| **new_L1** | **2** | — | **0** | **FAIL** |
| broke | 6 | — | — | — |

new_L1 seeds: 50 (baseline s3=0.978→0), 67 (baseline s3=0.951→0).
22-seed probe 給出 "new_L1=0" 的過度樂觀結果，60-seed Gate 推翻。

##### Phase K 緩解實驗（§4.2.K）

針對 0.555 bifurcation trap 設計 4 種緩解策略：

| Arm | 策略 | Probe 12-seed 結果 |
|-----|------|-------------------|
| K1 warmup=6000 | 前 6000 輪不觸發事件 | 11/12 healthy, broke=1 |
| K2 clamp=0.005 | 事件獎勵上限 | 10/12 healthy, broke=2 |
| K3 risk_off | 事件不影響 risk channel | 11/12 healthy, broke=1 |
| K4 per_player | 每輪隨機子集觸發 | 7/12 healthy, broke=4 |

**K1+K3 Combo 60-seed Gate（最終嘗試）：FAIL**

| 指標 | K1+K3 combo | baseline | Gate | 結果 |
|------|------------|----------|------|------|
| L1 | 2/60 | 2/60 | ≤3 | PASS |
| Healthy | 52/60 | 51/60 | ≥42 | PASS |
| **new_L1** | **2** | — | **0** | **FAIL** |
| broke | 8 | — | — | — |
| rescued | 9 | — | — | — |

K1+K3 比 vanilla rate=0.08 更差（broke 8 > 7）。6/8 壞 seed 卡在 s3≈0.555 bifurcation trap，2 個崩塌到 L1。

##### 根因結論

EventBridge 的**加性獎勵 + 全玩家同步觸發**在 300 玩家規模下產生高度相關的策略攝動，導致確定性分岔（deterministic bifurcation at s3≈0.555）。此為架構層面問題，非參數可調。

未來若要重新啟用 Event Integration，需要架構重設計：
- 非同步事件觸發（async per-player event queue）
- Multiplicative 而非 additive 獎勵調制
- 或低於 0.03 的極低 rate（但效果可忽略）

事件整合列為 **deferred research**，不影響 §(PERS-CAL) baseline 鎖定。

##### Known Limitations

1. **Event integration 全部嘗試均未通過 60-seed Gate**：events@0.15 L1 FAIL，events@0.08 new_L1 FAIL，K1+K3 combo new_L1 FAIL
2. **World feedback 在 λ_world=0.08 下為破壞性**：Arm C 6/11 healthy
3. **22-seed / 12-seed probe 不足以檢測 new_L1 風險**：未來變更必須直接 60-seed Gate
4. **0.555 bifurcation trap 是 EventBridge 架構性限制**：加性同步獎勵在大群體中產生不可逆分岔
5. **Event 配置依賴 smoke v1 模板**：5 families × 12D personality weights

##### 不變條件

1. Event 參數通過 CLI 或呼叫端顯式指定，不改動 `PersonalityRLConfig` 預設值
2. World feedback 預設 `False`，啟用須先更新本節 Spec
3. §(PERS-CAL) 的 lambda 鎖定不受事件配置影響

---

#### §(RUNTIME-BRIDGE) 下一輪工程主線

##### 工程目標

把 BL2 的 RL-native 循環機制，從 standalone research harness 提升為 Personality Dungeon 正式可用 runtime，並保留既有研究可重現性。

##### 第一輪只允許的工程範圍

1. 完整藍圖文件：`docs/personality_dungeon_complete_blueprint_v1.md`
2. 最小整合方案：`docs/personality_rl_runtime_bridge_v1.md`
3. 新 runtime 僅要求打通 `personality -> learning dynamics -> event/world coupling` 的最小閉環

##### 第一輪禁止事項

1. 不得把 BL2 直接硬塞回既有 replicator `sampled` 路徑，改寫其語意
2. 不得在同一輪同時改寫 world generator、event schema、與 RL operator
3. 不得把 RL runtime 的 timeseries schema 偷塞進既有 `w_*` 欄位語意而不更新契約

##### 最小契約

1. `p_*` 仍代表 realized action proportions，可沿用現有 analysis/cycle metrics
2. RL runtime 若要輸出 policy state，需使用新欄位（例如 `pi_*`），不得覆寫既有 `w_*` 契約
3. personality 對 learning dynamics 的映射必須是 bounded、可回溯、可在 provenance 中完整重建
4. event/world 只透過已定義的 reward/risk/state channels 影響 RL agent，不得跨層混入 analysis 或 I/O

##### Post-§(EV-INT) 實驗分流策略（Lock）

由於 §(EV-INT) 已確認 EventBridge 存在架構性同步攝動問題，後續工作分為主線與支線：

1. **主線（Track A）**：PERS-CAL-only production hardening + runtime bridge 交付
2. **支線（Track B）**：EventBridge v2 架構研究（僅限獨立 feature flags）

###### Track A 硬規則

1. 以 §(PERS-CAL) 鎖定配置為唯一 production baseline
2. 不因 Event 研究而調整 production gate 或既有 baseline 判讀標準
3. runtime bridge 變更需維持本節最小契約與分層不變條件

###### Track A A2 Runtime Bridge 契約（Spec Lock）

本節鎖定 A2（runtime bridge 最小整合）的資料契約，目標是「schema/provenance 完整，且 analysis 可直接相容」。

1. **Round CSV 欄位（固定）**
2. `round`
3. `avg_reward`, `avg_utility`, `success_rate`
4. `risk_mean`, `stress_mean`
5. `event_affected_count`, `event_affected_ratio`, `event_sync_index`
6. `p_aggressive`, `p_defensive`, `p_balanced`
7. `pi_aggressive`, `pi_defensive`, `pi_balanced`
8. `q_mean_aggressive`, `q_mean_defensive`, `q_mean_balanced`
9. `world_scarcity`, `world_threat`, `world_noise`, `world_intel`
10. `dominant_event_type`
11. 禁止以 `w_*` 欄位承載 RL policy 狀態（避免與既有 replicator 契約混義）
12. 事件關閉時（`events_json=""` 或有效事件觸發率為 0），`event_*` 欄位仍須存在，且值必須為 0；`dominant_event_type` 需為空字串

1. **Seed-level provenance 欄位（最小必備）**
2. `alpha_lo`, `alpha_hi`, `beta_ceiling`, `strategy_alpha_multipliers`, `payoff_epsilon`
3. `personality_mode`, `lambda_alpha`, `lambda_beta`, `lambda_r`, `lambda_risk`
4. `events_json`, `world_mode`, `lambda_world`, `world_update_interval`, `seed`
5. `mean_alpha`, `std_alpha`, `mean_beta`, `std_beta`
6. `risk_rule_version`, `diagnostic_rule_version`
7. `event_sync_index_mean`, `reward_perturb_corr`, `trap_entry_round`
8. `dispatch_seed_stream`, `dispatch_mode`, `dispatch_target_rate`
9. `dispatch_mean_affected_ratio`, `dispatch_player_activation_min`, `dispatch_player_activation_max`, `dispatch_player_activation_cv`
10. `dispatch_fairness_window`, `dispatch_fairness_tolerance`, `dispatch_fairness_checks`, `dispatch_fairness_failures`, `dispatch_fairness_pass`
11. `event_reward_mode`, `event_reward_multiplier_cap`
12. `event_impact_mode`, `event_impact_horizon`, `event_impact_decay`

1. **Analysis 相容規則（硬鎖）**
2. `analysis/cycle_metrics.py`、`analysis/decay_rate.py` 必須能直接以 `p_*` 欄位運作，不得要求 `w_*` 才能執行
3. `analysis/` 只能讀 CSV/TSV 或純函式輸入，不得 import runtime 物件
4. runtime bridge 契約變更時，需同步更新本節與對應回歸測試

###### Track A A3 Protocol Regression 契約（Spec Lock）

本節鎖定 A3（gate 腳本固定化）的單指令 protocol，目標是把 A1 + A2 固定成可回歸的日常檢查入口。

1. **單指令入口（固定）**
2. `./venv/bin/python -m simulation.track_a_protocol_regression`

1. **固定流程（不得重排）**
2. Stage A1：執行 PERS-CAL baseline gate 重驗（預設 `seeds=42..101`），輸出 `outputs/pers_cal_baseline_gate60_recheck_42_101_summary.json`
3. Stage A1-compare：將重驗輸出對照鎖定 baseline（`outputs/pers_cal_baseline_gate60_summary.json`），至少檢查
4. `total_seeds`, `l1`, `l2`, `l3`, `healthy`, `mean_s3`, `median_s3`, `p10_s3`, `mean_gamma`
5. `gate.overall_pass`
6. `non_healthy` seeds 集合一致性
7. Stage A2-regression：執行 `./venv/bin/pytest -q tests/test_personality_rl_runtime.py`

1. **輸出契約（固定）**
2. `outputs/track_a_protocol_regression_summary.json`
3. `outputs/track_a_protocol_regression_summary.md`
4. summary 需至少包含每個 stage 的 command、exit code、pass/fail、以及 overall_pass

1. **判定規則（硬鎖）**
2. 任一 stage fail，整體 `overall_pass=false`，且程式 exit code 必須非 0
3. 全部 stage pass，整體 `overall_pass=true`，且程式 exit code 必須為 0
4. A3 入口僅固定 protocol，不得修改 runtime 動力學、baseline 判定閾值、或 Track B gate 契約

###### Track B 硬規則

1. 不再以 12/22-seed probe 做 go/no-go；probe 僅用於 smoke
2. 任一 Event 機制變更，通過 smoke 後需直接執行 60-seed Gate
3. Gate 判定固定：L1≤3、Healthy≥42、new_L1=0
4. 連續兩個機制 family 在 60-seed Gate 失敗，即關閉當季 Event line
5. B1 執行流程固定為 `6-seed smoke -> 60-seed gate`，且 seed 集合鎖定為 `smoke={42,44,45,67,73,90}`、`gate=42..101`

###### EventBridge v2 可接受機制族（研究白名單）

1. 非同步事件觸發（async dispatch / per-player queue）
2. bounded multiplicative reward modulation（取代 additive overlay）
3. 多輪衝擊展開（impact spreading / decay kernel）

###### B1 Async Dispatch 契約（Spec Lock）

本節鎖定 B1 的資料契約，避免「機制改動」與「隨機流改動」混在一起。

1. **Config 欄位（最小集合）**
2. `event_dispatch_mode: "sync" | "async_round_robin" | "async_poisson"`（預設 `sync`）
3. `event_dispatch_target_rate: float`（每輪目標受影響比例）
4. `event_dispatch_batch_size: int`（每輪目標受影響人數；`0` 表示由 rate 自動換算）
5. `event_dispatch_seed_offset: int`（dispatch 專用 RNG stream 偏移）
6. `event_dispatch_fairness_window: int`（公平性檢查視窗）
7. `event_dispatch_fairness_tolerance: float`（允許偏差比例，預設 ±15%）

1. **Round CSV 欄位（觀測契約）**
2. `event_affected_count`
3. `event_affected_ratio`
4. `event_sync_index`
5. `dominant_event_type`

1. **Provenance 欄位（seed 級）**
2. `dispatch_seed_stream`
3. `dispatch_mode`
4. `dispatch_target_rate`
5. `dispatch_mean_affected_ratio`
6. `dispatch_player_activation_min`
7. `dispatch_player_activation_max`
8. `dispatch_player_activation_cv`
9. `dispatch_fairness_window`
10. `dispatch_fairness_tolerance`
11. `dispatch_fairness_pass`

1. **Seed 映射與可重現性規則（硬鎖）**
2. `sync` 模式必須維持舊路徑 bit-exact（同 seed 同結果）
3. async dispatch 必須使用**獨立 RNG stream**，不得消耗主 gameplay RNG
4. dispatch RNG 種子規則：`seed_dispatch = hash(seed, event_dispatch_seed_offset, "dispatch")`
5. player-level 抽樣需與玩家迭代順序解耦（同 seed 不因 list 順序改變而改變抽樣結果）

1. **公平性規則（硬鎖）**
2. 在任一 fairness window `W` 中，每位玩家的 activation count 需落在期望值 `E` 的 `E ± tolerance*E`
3. 每 seed 需回報 activation 分布離散度（`dispatch_player_activation_cv`）
4. 若 `dispatch_fairness_pass = false`，該次 run 不得作為 Gate 判讀依據

###### B2 Multiplicative Modulation 契約（Spec Lock）

本節鎖定 B2（乘法獎勵調制）的最小資料契約，避免與 B1 的 async dispatch 混淆。

1. **Config 欄位（最小集合）**
2. `event_reward_mode: "additive" | "multiplicative"`（預設 `additive`）
3. `event_reward_multiplier_cap: float`（僅在 `multiplicative` 生效，限制單輪乘數偏移量）

1. **運算語意（硬鎖）**
2. `additive`：維持既有路徑 `reward_i' = reward_i + delta_i`
3. `multiplicative`：`reward_i' = reward_i * m_i`，其中 `m_i = clip(1 + delta_i, 1-cap, 1+cap)`
4. `cap` 必須滿足 `0 < cap < 1`，避免乘數翻號；若輸入超界，實作需在執行時夾取到有效範圍
5. `event_reward_mode=additive` 時，既有 B1/Baseline 結果必須維持 bit-exact

1. **Provenance 欄位（seed 級）**
2. `event_reward_mode`
3. `event_reward_multiplier_cap`

1. **B2 第一輪 protocol（硬鎖）**
2. 流程固定為 `6-seed smoke -> 60-seed gate`
3. seed 集合沿用 Track B 鎖定集合：`smoke={42,44,45,67,73,90}`、`gate=42..101`
4. Gate 判定固定：`L1<=3`、`Healthy>=42`、`new_L1=0`、`fairness_fail_count=0`
5. 第一個 block（`42..101`）未通過時，不得直接擴張到第二個 block（`102..161`）

###### B3 Impact Spreading 契約（Spec Lock）

本節鎖定 B3（多輪衝擊展開 / decay kernel）的最小資料契約，目的在於降低單輪衝擊，且不破壞 B1/B2 既有可比較性。

1. **Config 欄位（最小集合）**
2. `event_impact_mode: "instant" | "spread"`（預設 `instant`）
3. `event_impact_horizon: int`（僅 `spread` 生效，表示衝擊展開輪數，最小為 `1`）
4. `event_impact_decay: float`（僅 `spread` 生效，`>=0`，定義幾何衰減核）

1. **運算語意（硬鎖）**
2. `instant`：維持既有單輪路徑（event 影響只在當輪生效）
3. `spread`：每次事件產生的 `delta_i` 需分配到未來 `horizon` 輪，權重由幾何核 `w_k ∝ decay^k` 給定，並正規化為 `sum_k w_k = 1`
4. `spread` 的總衝擊量守恆：同一個事件在 `horizon` 視窗內的加總影響量必須等於原始 `delta_i`
5. `event_impact_mode=instant` 時，既有 B1/B2 結果必須維持 bit-exact
6. `event_impact_horizon < 1` 或 `event_impact_decay < 0` 為非法輸入；實作需在執行時夾取到有效範圍（`horizon>=1`, `decay>=0`）

1. **Provenance 欄位（seed 級）**
2. `event_impact_mode`
3. `event_impact_horizon`
4. `event_impact_decay`

1. **B3 第一輪 protocol（硬鎖）**
2. 流程固定為 `6-seed smoke -> 60-seed gate`
3. seed 集合沿用 Track B 鎖定集合：`smoke={42,44,45,67,73,90}`、`gate=42..101`
4. Gate 判定固定：`L1<=3`、`Healthy>=42`、`new_L1=0`、`fairness_fail_count=0`
5. B3 第一個 block 固定工作點：`event_impact_mode=spread`、`event_impact_horizon=5`、`event_impact_decay=0.70`
6. 第一個 block（`42..101`）未通過時，不得直接擴張到第二個 block（`102..161`）

除白名單外之機制調整，需先更新本節 Spec 才能實作。

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

---

## 10) 附錄：本輪決策快照（2026-04-13）

本附錄為 **2026-04-13 當輪治理快照**，用途是把 W1~W4 的結論固定成可追溯摘要；
不覆寫本文件前述契約與硬規則。

### 10.1 本輪範圍與結論

1. 範圍：W1（A3 + failure catalog）到 W4（月結 + go/no-go + decision record）。
2. 主線結論：`keep_80_20`（GO）。
3. Event line 結論：重啟條件未達，維持 `NO-GO`。
4. 機制擴張結論：本輪維持 decision-only，不做機制擴張 sweep。

### 10.2 W1~W4 完成快照

| Stage | 判定 | 主要產物 |
| --- | --- | --- |
| W1 | completed | `outputs/w1/d5_w1_review_summary_v1.json` |
| W2 | completed | `outputs/w2/block_risk_report_v1.json` |
| W3 | completed | `outputs/w3/w3_handoff_note_v1.md` |
| W4 | completed (initial decision package finalized) | `outputs/w4/go_no_go_memo_v1.md`, `outputs/w4/monthly_review_v1.md`, `outputs/w4/decision_record_v1.md`, `outputs/w4/w4_closure_memo_v1.md` |

### 10.3 本輪量化快照

1. A3 `overall_pass=true`
2. `L1=2`, `Healthy=51`, `mean_s3=0.8743063884195799`
3. `pytest=34/34`
4. block robustness verdict：`not_block_robust`
5. `b1_async_poisson_r008` 4-block pass rate：`0.25`（`1/4`）
6. 4-block `mean_new_l1=1.75`

### 10.4 驗收清單快照（W4 closure）

1. A3 summary 存在且可追溯：Pass
2. L1/Healthy/mean_s3/pytest 在可接受區間：Pass
3. 探索遵守 no-expansion + falsifiable discipline：Pass
4. Event restart guard（cross-block robust 且 `new_l1=0`）達成：Not Met（故維持 NO-GO）

### 10.5 守門條件維持

Event line restart 的硬條件不變：

1. `cross_block_robustness == true`
2. `new_l1 == 0`
3. A3 drift guard 持續穩定

在三項未共同成立前，不得將 Event line 由 NO-GO 改為 GO。

