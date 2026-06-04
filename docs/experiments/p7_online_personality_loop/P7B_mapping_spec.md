# P7-B：人格→Payoff 映射數學規格（Personality-to-Payoff Mapping Spec）

> **本文件是純數學 Spec，不含模擬執行。** 任何後續實作（P7-C）必須以本文件為準；若映射定義變更，必須先修改本文件，再改碼。

---

## 1. 核心問題

給定玩家在 step t 的人格向量 $P_t \in [-1, 1]^9$，如何將其轉換為對三策略（aggressive / defensive / balanced）的 payoff 調制項 $\Delta u(P_t) \in \mathbb{R}^3$？

---

## 2. 人格→策略 Bias 線性投影（f: ℝ⁹ → ℝ³）

### 2.1 投影矩陣 W（設計依據：直覺關聯表）

定義投影矩陣 $W \in \mathbb{R}^{3 \times 9}$，其中行對應策略 {aggressive, defensive, balanced}，列對應 9 個 trait：

$$
W = \begin{pmatrix}
w_{A,\cdot}\\
w_{D,\cdot}\\
w_{B,\cdot}
\end{pmatrix}
$$

**初始 W 設計值（待 P7-A G7A-03 結果校準）：**

| | IMP | ASS | OPT | RAV | SUS | END | RND | STB | CUR |
|-|-----|-----|-----|-----|-----|-----|-----|-----|-----|
| w\_A（aggressive）| +0.4 | +0.4 | +0.2 | −0.4 | −0.2 | −0.1 | +0.1 | −0.3 | +0.2 |
| w\_D（defensive）| −0.3 | −0.3 | −0.1 | +0.4 | +0.3 | +0.3 | −0.1 | +0.4 | −0.2 |
| w\_B（balanced） | −0.1 | −0.1 | +0.2 | 0.0 | −0.1 | −0.2 | +0.2 | −0.1 | +0.4 |

> **不變條件**：每行 $w_s$ 的 L1 norm ≤ 1.0（避免 payoff 爆炸）。

### 2.2 原始 bias 向量

$$
b_t = W \cdot P_t \in \mathbb{R}^3
$$

其中 $P_t$ 為 SBERT+MLP v7 推斷的 9D 向量，值域 $[-1, 1]$。

---

## 3. Payoff 調制函數

### 3.1 調制項定義

$$
\Delta u(P_t) = \alpha \cdot \text{clip}(b_t,\; -\delta_{max},\; +\delta_{max})
$$

其中：
- $\alpha \in [0, 1]$：回饋強度（feedback strength），P7-D 的主要掃描參數
- $\delta_{max} = 0.5$：clamp 上限（防止單一 trait 主導 payoff）
- $\text{clip}$ 保持 $\Delta u$ 的符號語意

### 3.2 最終 Payoff 計算

$$
u_s^{final}(t) = u_s^{base}(t) + \Delta u_s(P_t)
$$

其中 $u_s^{base}(t)$ 為現有 `matrix_ab` payoff（$Ax(t-1)$，$A$ 定義見 SDD §2.2）。

**時間索引規則**（延續 SDD §2.4 的約定）：
- $u^{base}$ 使用 $x(t-1)$（上一輪族群比例）
- $P_t$ 使用本輪 step 開始時的最新人格向量
- $\Delta u$ 在本輪策略選擇前注入

---

## 4. 人格更新規則（Personality Update Rule）

> 本節定義 $P_t \to P_{t+1}$ 的更新語意。

### 4.1 Reward-driven 更新（ΔP）

$$
\Delta P_i(t) = \eta \cdot (r_t - \bar{r}) \cdot g_i(s_t)
$$

其中：
- $\eta$：人格學習率（learning rate），固定為 0.05（待 P7-D 校準）
- $r_t$：玩家本輪 reward
- $\bar{r}$：本輪所有玩家的 reward 均值
- $g_i(s_t)$：策略 $s_t$ 對 trait $i$ 的「強化方向」（查表，下列）

### 4.2 策略→trait 強化方向表 g(s, trait)

| 策略 s | 被強化的 trait（g = +1）| 被弱化的 trait（g = −1）| 無關（g = 0）|
|--------|------------------------|------------------------|--------------|
| aggressive | IMP, ASS, OPT, CUR | RAV, STB, END | RND, SUS |
| defensive  | RAV, STB, END, SUS | IMP, ASS, OPT | RND, CUR |
| balanced   | RND, CUR, OPT | SUS | IMP, ASS, RAV, STB, END |

### 4.3 更新後 clamp

$$
P_{t+1,i} = \text{clip}(P_{t,i} + \Delta P_i(t),\; -1.0,\; +1.0)
$$

**不變條件**：任何更新後，$P_t$ 的所有分量必須維持在 $[-1, 1]$。

---

## 5. 不變條件（Invariants，須在 P7-C 回歸測試中驗證）

| ID | 條件 |
|----|------|
| INV-7B-01 | $\alpha = 0$ 時，$u^{final} \equiv u^{base}$（靜態退化） |
| INV-7B-02 | $\eta = 0$ 時，$P_{t+1} \equiv P_t$（人格不更新）|
| INV-7B-03 | 任意 $P_t$ 與 $\alpha$，$\|\Delta u\|_\infty \le \delta_{max}$（payoff 有界）|
| INV-7B-04 | 更新後 $P_{t+1,i} \in [-1, 1]$（人格有界）|
| INV-7B-05 | $W$ 的每行 L1 norm ≤ 1.0 |

---

## 6. 開放問題（需 P7-A 結果再決定）

1. **W 校準**：若 P7-A G7A-03 顯示現有 W 設計值無法產生策略分異，需以 P7-A 的 reward 差值回推合適的 $w_s$ 係數。
2. **非線性投影**：目前採線性投影；若線性 $b_t$ 分布過於集中，可考慮加入 tanh 壓縮：$b_t = \tanh(W \cdot P_t)$。
3. **共享 vs 個體 W**：目前假設所有玩家共用同一 W；若要建模不同「個性類型」的玩家，可為每組人格類型定義不同 W。

---

## 7. 與現有 SDD 的邊界

| 項目 | 現有 SDD 定義 | P7-B 擴充 |
|------|--------------|-----------|
| payoff 計算層 | `dungeon/dungeon_ai.py` | 新增 `personality_bias()` 方法（P7-C 實作） |
| CSV 欄位 | FrameSnapshot 28 欄 | 新增 `personality_vector`（9 floats）、`delta_u`（3 floats）|
| API 回傳 | `/step` response | 新增 `personality_state` 欄位（P7-C Gate）|

> **契約變更原則**：上表中任何欄位變更，必須同步更新 SDD §4.x 並補回歸測試，方可實作。

---

*狀態：草稿規格 | 依賴 P7-A G7A-03 結果進行 W 校準*  
*建立日期：2026-06-03*
