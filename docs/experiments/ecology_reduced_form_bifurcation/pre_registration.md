# Pre-registration — 生態 neg-freq 引擎的 reduced-form 定向分岔（雙動力學 gain-sweep）

> **狀態**: DRAFT v2（pre-reg，未跑、未 commit）
> **日期**: 2026-06-21（v2 修：解析錨 bug 修正 + 改雙極點 dichotomy 設計）
> **作者**: Claude Opus 4.8 + User
> **類型**: reduced-form 模擬控制（causal-isolation control），非真人實驗
> **關係**: P7-H（已封存）的旁支；遊戲層 PvP/經濟決策的前置 control。**不碰 production code、不碰 live game。**

---

## §0 一句話：這份 pre-reg 回答什麼

把「定向競技壓力會不會把 neg-frequency 多樣性引擎壓成 monoculture、以及在什麼閾值」這個問題，
**從 PvP（會污染皇冠結果的第二算子）抽離**，化成單純形上的一個可掃增益 `g`。
產出 = **定向壓力預算 g\***（任何未來經濟旋鈕只要 *effective* `g < g*` 就不會把生態壓成單一文化）
**＋一個機制發現**：g\* 是否被有限尺寸效應**侵蝕還是保護**，取決於選擇動力學有沒有吸收邊界——
而 production（可重選 authoring + neg-freq 再播種）落在「保護」那一極。沒有這個數，
`rank_stake`/`offset_ratio`/source-sink 表全是盲調（parking C 🔴）。

**這不是 headline-paper，是 control。** 它同時是任何未來「真人在競技階梯下會不會同質化」行為實驗的**必備 null**
（沒有它就無法把「數學本來就會在那個 g 分岔」與「人把它推過 g\*」分開歸因）。

---

## §1 背景與動機（為何 reduced-form，附 code 出處）

決策鏈（本 session 裁定，verify-don't-assert）：

1. **H_counter 已證偽** → combat counter-matrix 不可測、M 只能 authored（[api/pvp_manager.py:5](../../../api/pvp_manager.py#L5)）。
2. **PvP 的 3-RPS 正是生態於 2026-06-19 親手退役的 payoff 形式**——退役理由＝可被「搶當第一個選死派系」exploit、q→0 爆衝、whiplash（[api/ecology_tracker.py:156-160](../../../api/ecology_tracker.py#L156-L160) docstring 逐字）。在 PvP 重建它＝重蓋一個已知較差形式。
3. **PvP 與生態共用同一 latent state**（archetype；[api/pvp_manager.py:11,22](../../../api/pvp_manager.py#L11)）→ Increment 2 把惰性顯示階梯「武裝」成共用 state 上的**第二 selection operator** → Rank churn 會漏進多樣性 observable，污染剛封存的皇冠結果。
4. **生態 production 不跑 replicator**——`submit()` 用 advantage 更新動態權重/coin 分，但 `_recent.append(i)` 存的是**真實提交的 archetype**，advantage **不回饋下一個選擇**（[api/ecology_tracker.py:183-211](../../../api/ecology_tracker.py#L183-L211)；亦見 [scripts/experiments/ecology_9d_coverage.py:5](../../../scripts/experiments/ecology_9d_coverage.py#L5)「step 0 已證生態層不跑 replicator」）。

**裁決**：把 directional 壓力當成抽象 `g·d_i`，在**獨立 reduced-form harness** 裡掃。
**不**把 `g·d_i` 加進 production `_fitness`（那裡無閉環、只會偏置 live game）。
這從源頭消掉污染（沒有第二算子），且 reduced-form 是 B（行為版）的前置 control，永不浪費。

---

## §2 模型（formal）— 共享 fitness、**兩種動力學**

**狀態**：3 個生態 archetype 上的近期佔比 `q = [q_agg, q_def, q_bal]`，`Σq=1`，活在 2-單純形上。
（vertices＝aggressive 莽 / defensive 穩 / balanced 平衡；topology 同 [ecology_bstep0](../../../scripts/experiments/ecology_bstep0_driven_response.py#L36) 的 `_VERT`。）

**共享適應度**（production neg-freq + 定向注入）：

```
f_i(q) = (1/K − q_i)        +   g · d_i
         └─ 向心 / 穩定 ─┘       └─ 離心 / 定向 ─┘
```

- `1/K − q_i`：production 負頻率依賴向心力，`K=3`（向心目標＝均勻 1/3）；**有界** `q_i∈[0,1] → f∈[−2/3, +1/3]`（[api/ecology_tracker.py:153-163](../../../api/ecology_tracker.py#L153-L163)）。
- `g · d_i`：離心定向力。`d` ＝單位「優勢 build」偏置（primary `d=e_agg`，§6）；`g≥0` ＝掃描增益。
- ⚠ 記號分離：`K=3`＝archetype 數（neg-freq 的「1/N」指此，非人數）；`N_pop`/`W`＝有限族群/窗（下）。

**選擇映射**（共享，production-faithful）：`resp_i(q) = softplus(lam·f_i) / Σ_j softplus(lam·f_j)`，
`softplus(x)=ln(1+e^x)`、`lam`＝選擇銳度（production 錨 `2.0`，[api/ecology_tracker.py:98,165-166](../../../api/ecology_tracker.py#L98)）。
**關鍵性質**：`softplus>0` 處處 → resp 永遠給每個 archetype 正機率（軟地板、無零吸收）。

### 兩種動力學（兩極點，**這是 v2 的核心**）

| | **Arm S — sampling（primary / faithful）** | **Arm R — replicator（reference / idealized）** |
|---|---|---|
| 更新 | per-submission boxcar：窗 `deque(maxlen=W)`（mirror [_proportions](../../../api/ecology_tracker.py#L143) production `window=50`），每步抽 1 筆 ∝ `resp(q_est)`、推入 | `dq_i/dt = q_i·(a_i − ā)`，`a_i=softplus(lam·f_i)`（沿用 [run_closed_replicator](../../../scripts/experiments/ecology_bstep0_driven_response.py#L78)，payoff 核換掉） |
| 不動點 | `q* = resp(q*)`（自洽，**非** f-equal） | `a_i` 相等 ⟺ `f_i` 相等（f-equal） |
| 邊界 | **軟地板**：softplus>0 → minority 缺席時 `f→+1/3`（最大稀缺獎勵）→ **被 neg-freq 再播種**，**無確定 monoculture** | **吸收**：`q_i=0 → dq_i=0`，minority 一旦觸 0 即死（neg-freq 再播種被 `q_i×` 因子殺掉） |
| 噪聲源 | 有限窗 `W` | 有限族群 `N_pop`（optional demographic noise） |
| 忠實度 | **高**：production = 離散稀缺-響應 authoring，無「因常見而模仿」因子（§1.4） | **低**：`q_i×` imitation + 吸收邊界移除了 production 真有的再播種 |

> **為何 v2 改雙極點**（裁定 2026-06-21）：v1 的解析錨用了 **replicator 的 f-equal FP**，卻把族群更新指定成 **sampling**——
> 兩者不動點不同（f-equal 點 `resp=(⅓,⅓,⅓)≠q*` for g>0），自相矛盾。修正後發現兩動力學在邊界**質性相反**
> （吸收 vs 再播種），且這個對比本身就是本實驗最強的結果（§3–§4）。故**兩極都跑**，sampling 給實務 verdict、replicator 量化「吸收理想化低估多少」。

**Apparatus 失真軸**：`W`（窗，Arm S；production 50）、`N_pop`（族群，Arm R）、`lam`（銳度；production 2.0）。
連續確定極限 = `W,N→∞` + 平均場 → §3 的共享 landmark。

---

## §3 共享解析 landmark：`g=1` = 定向增益 = neg-freq fitness span

neg-freq fitness 的動態範圍：`f_i∈[+1/3 (q_i=0), −2/3 (q_i=1)]`，**span = 1/3−(−2/3) = 1**。
`g=1` ＝定向 bonus 剛好抵掉一個消失中 minority 的**最大稀缺獎勵**。在兩種動力學裡都是有意義的確定 landmark，
但 g>1 之後行為相反：

**Arm R（replicator）— 硬吸收，`g*_det,R = 1`、與 lam 無關**
內部 FP（f-equal，選擇單調故與 lam 無關）：`q_agg=1/3+2g/3，q_def=q_bal=1/3−g/3`。
存在 ⟺ `g<1`；`g=1` 時 `q_def=q_bal→0` 撞**吸收**邊界 → monoculture（transcritical boundary collision）。
→ `g*_det,R = 1`（解析、乾淨、lam 無關）。

**Arm S（sampling）— 軟地板，確定極限「無 monoculture」**
`q*=resp(q*)`，softplus>0 → **所有 g 都有內部解**（minority 隨 g↑ 漸小但**永不為 0**；`g→∞` 時 `q*_agg→1` 漸近）。
`g=1` 仍是 landmark：g<1 時在小 `q_def` 處 `f_def→+1/3 > f_agg` → minority **自我修正**（地板穩定、遠離 0）；
g>1 時 `f_agg>f_def` → minority 衰向小（非零）值。**確定極限不分岔。**

**★ Dichotomy（headline）：有限尺寸效應符號相反**
- **Arm R + finite-N**：demographic 噪聲在 g→1 把縮小的 minority 推進吸收邊界 → **`g*_emp,R < 1`（侵蝕）**，N→∞ 時 →1⁻。
- **Arm S + finite-W**：minority 缺席→最大稀缺獎勵→**再播種**對抗滅絕 → 滅絕被**推遲到 g>1** → **`g*_emp,S > 1`（保護）**，標度 heuristic ~`O(W/lam)`。

⇒ **「finite-size 一律 destabilize」的直覺是錯的**；符號取決於有無吸收邊界。**production 是 Arm S → 有限尺寸保護**。
這個符號翻轉**就是** metric-defined vs real：「預算」取決於你假設哪種 selection 理想化。

> 聲明：以上為 analytic/heuristic（design-fact，讀碼+推導），**未模擬**——正是要 pre-register 的預測，由 harness 確認。
> 特別地 `g*_emp,S ~ O(W/lam)` 的標度是 heuristic，§6 掃描範圍須先用數值 floor `r*(g;lam)` 校準、不寫死。

---

## §4 假設（confirmatory / exploratory；雙臂對偶）

**Confirmatory（預先承諾，主結果）**

- **C1（★ dichotomy = headline）**：`sign(g*_emp − 1)` 在兩臂**相反**——
  Arm R（finite-N，production-ish `N_pop` 小）**< 1**（侵蝕）；Arm S（finite-W，`W=50`）**> 1**（保護）。預先承諾異號。
- **C2（尺寸單調）**：Arm R `g*_emp → 1⁻` 當 `N_pop→∞`；Arm S `g*_emp` 隨 `W` **單調遞增**（W→∞ 無確定崩壞）。
- **C3（銳度）**：兩臂 `g*_emp` 皆隨 `lam` **遞減**（更銳選擇 → 更早失守）。

**Exploratory（不預設，報告即可）**

- E1：Arm S 的 intrinsic 攪動率 `ρ`（1−ρ 從固定偏好抽，mirror [run_open](../../../scripts/experiments/ecology_bstep0_driven_response.py#L57)）。
- E2：bandwagon 反 d，`d_i ∝ (q_i−1/3)`（rich-get-richer，直接對撞 neg-freq）。
- E3：per-vertex 對稱性（clean 模型對稱，finite-W/N 可能破缺）。
- E4：Arm S kernel（boxcar vs EMA `q←(1−α)q+α·cohort`）對 `g*` 的影響。

---

## §5 Observable / 主結果（雙臂統一報法）

**主 observable**：聯合 archetype 正規化 Shannon 熵 `H_norm(q) = −Σ q_i ln q_i / ln K ∈ [0,1]`
（1＝均勻共存、0＝monoculture），現成於 [_metrics](../../../scripts/experiments/ecology_bstep0_driven_response.py#L110) `mean_diversity`。

**g\* 操作化（預先承諾，兩臂同一套，跨硬/軟門檻可比）**：
- **Primary（軟，兩臂通用）**：`g*` ＝穩態 `⟨H_norm⟩(g)` 曲線的**最陡下降拐點**（threshold-free，符合分岔語意）。
- **Robustness 1**：`⟨H_norm⟩(g)` 跌破 **0.5** 的最小 `g`（預先固定門檻；全曲線一併報告）。
- **Robustness 2（滅絕事件）**：`min_q < 0.02` 或 `mono_frac > 0.15`（[classify](../../../scripts/experiments/ecology_bstep0_driven_response.py#L123)）——
  Arm R 是吸收滅絕、Arm S 是 finite-W 隨機滅絕，**同指標、不同機制**，分臂報。

**雙帳（核心交付）**：每組參數同時報 `g*_emp`（有限尺寸隨機）與共享 landmark `1`，
頭條量 = **兩臂的 `g*_emp − 1` 對偶**（Arm R 為負、Arm S 為正）＝符號翻轉證據。

**自檢（per arm）**：
- 兩臂 `g=0`：穩定 center、`H_norm≈1`。
- Arm R `N→∞`：內部 FP 投影 Jacobian 最大實部過零落在 `g≈1`（重用 [jacobian_at/tangent_eigs](../../../scripts/experiments/ecology_replicator_probe.py#L75)）。
- Arm S `W=∞`：數值解 `q*=resp(q*)` 對大 `g` 仍 `min q* > 0`（驗「無確定 monoculture」）。
- 任一自檢不符 → 先修 harness 再跑掃描（不報為發現）。

---

## §6 設計與參數（預先承諾網格）

**共享固定**：`K=3`；`d` primary `=e_agg`（莽＝e2e 熱門/優勢 build 候選；clean 對稱故頂點選擇不影響 primary，見 E3）；
seeds `{0,1,2,3,4}`；初值 `q0=1/3±微擾`（[ecology_bstep0:61](../../../scripts/experiments/ecology_bstep0_driven_response.py#L61)）；burn 前 20%。

**Arm S（sampling）**
- `g`：**範圍不寫死**——先數值解確定 floor `r*(g;lam)`（`q*=resp(q*)`），掃描區間設成**包夾 `r*·W ≈ 1`**（heuristic `g* ~ O(W/lam)`，production 預估 O(10s)）；log-spaced，bisection 細化 `g*`。
- `lam`：`{0.5, 1, 2(錨), 4, 8}`；`W`：`{10, 20, 50(錨), 100, 200, ∞(確定自洽解)}`；`T=50,000` 筆/seed（≈1000 個 W=50 窗）。

**Arm R（replicator）**
- `g`：`0.0→1.4`，步長 `0.02`（錨在 1）；bisection 細化 `g*`。
- `lam`：同上；`N_pop`：`{30, 100, 300, ∞(確定 ODE)}`；`dt=0.05`、`T=3000`、burn 1000（沿用 run_closed_replicator）。

**注**：`eta=0.2`（production 動態權重 EMA，[:100](../../../api/ecology_tracker.py#L100)）兩臂皆**不入模**——production 權重只調 coin 分/顯示、不入族群閉環。故意不引入，避假精度。

---

## §7 分析計畫

1. 各臂 × 各參數 × seed：跑穩態，取 `⟨H_norm⟩`、`min_q`、`mono_frac`；bisection 求 primary `g*` ＋ robustness `g*`；跨 seed 給均值＋95% 區間。
2. **C1 dichotomy 檢定**：比較 `g*_emp,R(N_pop=30,lam=2)` 與 `g*_emp,S(W=50,lam=2)`，確認前者 `<1`、後者 `>1`（CI 不跨 1，異號）。
3. **C2/C3 單調性**：Arm R `g*(N_pop)`、Arm S `g*(W)`、兩臂 `g*(lam)` 的 Spearman ρ（符號預先承諾）。
4. 自檢（§5）：Arm R eigenvalue-crossing `≈1`；Arm S `W=∞` 大 g 仍 `min q*>0`。偏離即視 harness bug 須查、不報為發現。
5. Exploratory E1–E4 另表、明標 exploratory。

---

## §8 決策規則 / 可證偽 / 結局預先綁定

- **C1 成立（Arm R `g*<1` 且 Arm S `g*>1`，異號）**：確認 dichotomy → 機制發現「有限尺寸效應符號由吸收邊界 vs 再播種決定」；**production-faithful 預算 = Arm S `g*_emp,S`**（>1，遊戲經濟定向壓力上界，較 replicator 寬鬆）。
- **C1 不成立（兩臂同號 / Arm S 也 <1）**：neg-freq 再播種**未**提供保護 → 推翻「軟地板保護」論證 → 須回查（是 softplus 飽和？lam 太大？W 太小？）。預先登記為**真結果非失敗**：意味 production 引擎比預期脆弱、經濟預算須收緊。
- **Arm S 掃到 `r*·W≈1` 的上界仍無 collapse**：與 finite-W 滅絕論證矛盾 → 視 harness/標度 bug 須查，不報為「無條件穩定」。
- 任一結局**不**據此立刻動 PvP 行為儀器或 coin→Rank 接線——那是 B，需另一份 pre-reg。

---

## §9 Harness retarget 清單（具體，逐符號；本份 doc 不寫 code）

新檔：`scripts/experiments/ecology_reduced_form_bifurcation.py`（**新檔，不改 production `api/`**）。

**共用核（兩臂同）**
- payoff 核：刪 `A@q`（反對稱 RPS，退役形式）→ `f = (1/K − q) + g*d`（`fitness_fn(q,g,d)`）。
- 選擇：`softmax(β·f)` → `resp = softplus(lam*f); resp/=resp.sum()`。
- 重用：`to2d`/`_VERT`/`winding`/`_metrics`（含 `mean_diversity`=H_norm、`mono_frac`、`min_q`）、sweep+plot+JSON 骨架（[ecology_bstep0 main](../../../scripts/experiments/ecology_bstep0_driven_response.py#L135)）。

**Arm S driver（sampling，primary）**
- 由 [run_open](../../../scripts/experiments/ecology_bstep0_driven_response.py#L57) 改：per-cohort EMA → **per-submission boxcar** `deque(maxlen=W)`，每步抽 1 筆 ∝ resp(q_est)、推入、重算 q_est（mirror production [_proportions](../../../api/ecology_tracker.py#L143)）。EMA 版留 E4。
- 數值 floor `r*(g;lam)`：fixed-point iterate `q←resp(q)` 至收斂（給 §6 掃描範圍 + §5 自檢）。

**Arm R driver（replicator，reference）**
- 由 [run_closed_replicator](../../../scripts/experiments/ecology_bstep0_driven_response.py#L78) 改：payoff 核換成共用 `fitness_fn`（成長率用 `a_i=softplus(lam*f_i)`）；保留 optional `N_pop` demographic noise。
- 確定 FP 自檢：Arm R 用 §3 閉式 `q_def=1/3−g/3`；eigenvalue 用 [jacobian_at/tangent_eigs](../../../scripts/experiments/ecology_replicator_probe.py#L75)（⚠ 推廣成吃 `fitness_fn` callable，現簽名只吃線性矩陣 `A`；`interior_fixed_point(A)` 不可直接用，改閉式）。

**掃描 / 輸出**
- 巢狀 `arm × g × lam × (W|N_pop)`；bisection 求 g*。
- `reports/ecology/reduced_form_bifurcation.json` + `.png`（左＝兩臂 `H_norm(g)`；中＝`g*_emp−1` 雙臂對偶（dichotomy 圖）；右＝代表性單純形軌跡，沿用 ax[2] 樣式）。

---

## §10 不在範圍 / 明確延後（非殺）

- **真人「在競技階梯下會不會同質化」行為宣稱**＝B，需另建 directional ladder + raw-vs-effective 雙帳 + 招募，且**以本 reduced-form 的 g\*（Arm S）為 null**。延後。
- **parking C 的 coin→Rank 字面抵銷接線**＝污染向量，研究裡由抽象 `g` 取代；遊戲裡由**非 archetype-coupled** 的 sink（門票/防禦升級）取代（R3/C 並行軌，另份 game-spec doc）。
- **production code 改動**：本實驗一律新檔 sim、**零 `api/` 改動、零 live-game 偏置**。

---

## §11 Provenance / 與專案關係

- 上游：生態「逐利→多樣性」閉環（neg-freq `1/N−q`，[api/ecology_tracker.py](../../../api/ecology_tracker.py)）；Exp B 驅動響應 harness（[ecology_bstep0_driven_response.py](../../../scripts/experiments/ecology_bstep0_driven_response.py)）。
- 本命題對齊：裝置限制／metric-defined vs real mechanism（記憶 `apparatus-limits-static-personality`、`personality-ecology-layer`、`trait-blast-radius-expA`）。
- 下游：g\*（Arm S）＝遊戲經濟（parking C 🔴）定向壓力預算 + 任何 PvP 行為實驗（B）的 null；dichotomy ＝獨立機制發現。
- **v1→v2 修訂史**：v1 解析錨誤用 replicator FP 配 sampling 動力學（自相矛盾）；v2 修正並改雙極點，把 bug 升級成 headline dichotomy。
