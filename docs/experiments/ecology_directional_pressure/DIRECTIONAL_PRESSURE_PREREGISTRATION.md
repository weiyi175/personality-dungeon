# 生態多樣性引擎在 directional 壓力下的分岔 — 預先註冊（Pre-registration）

> **狀態（2026-06-21）**：**DRAFT — 待用戶 ratify 後鎖定**。一旦開始執行 §6 的 confirmatory g-sweep，§3–§7 的假設、判定鏈、grid、種子數、停止規則 **不得修改**；偏離記入 §10。
> **研究代號**：ECO-DP（Ecology Directional-Pressure bifurcation）
> **論文定位**：機制級**因果隔離** — *Where a directional pressure breaks a negative-frequency-dependent coexistence engine: a bifurcation of the live ecology operator, causally isolated from a competitive PvP ladder.*
> **這份取代的決策**：把「PvP Rank 結算零和 vs 通膨（A/B/C）」這個二階記帳問題，重定為「directional selection 注入生態 fitness 的可掃 gain」這個一階 dynamics 問題（討論串 2026-06-21）。
> **判定對象（live 算子，不重寫）**：[api/ecology_tracker.py](../../../api/ecology_tracker.py) `_fitness`(L153–163) / `_advantage`(L165–166) / EMA 權重更新(L187–192) / `_proportions`(L143–151)。
> **前置決策（已鎖，本研究承接不重議）**：生態 M vs combat M **不統一**（共用 taxonomy/topology、不共用增益；唯一約束 ecology payoff a≥b；parking_lot D ★）；faction = Option 4+（PC1-PC2 3×120° argmax；parking_lot A）；現役 `_fitness` = 負頻率依賴 `1/N−q`、lam=2（parking_lot D，2026-06-20 換掉 RPS Path-B 殘留）。

---

## 0. 探索 / 確認分界 + 復用聲明（誠實聲明）

- **這是純模擬 confirmatory**，判定算子＝**現役生態 code 本身**（非另寫的理想 replicator），故結論直接適用於 live 系統的 latent dynamics。
- **新增物只有兩件**：(a) `_fitness` 加一個 directional 項 `+ g·d_i`（g=0 嚴格還原現役引擎）；(b) 一個薄的 **player-response driver**（logit/replicator 選擇模型），把「玩家逐 coin 而選 archetype」這條回饋閉合成可模擬的動力系統。
- **關鍵範圍限制（決定 ② 為何延後）**：① 在「玩家以 logit 響應 coin 誘因」這個**假設的 response model 之下**回答「算子會不會分岔」。它**不**回答「真人是否真的這樣響應」——那是 ②（行為版），① 是 ② 的必備 baseline（§11）。
- **承接的既有結果，不重跑**：Exp B 已建立**已退役 RPS 算子**的 `maxRe ⟺ sign(b−a)`（parking_lot D 🟢）；本研究對象是**現役 neg-freq 算子**，其在 directional 擾動下的穩定性是**新的觀測對象**，與 Exp B 不同算子、不可互相替代。

---

## 1. 背景與目的

系統有**兩個 selection operator 作用在同一個 latent archetype state**（3 原型 Aggressive/Defensive/Balanced，共用 `personality_to_archetype_soft`）：

1. **生態 coins＝負頻率依賴**：`fitness_i = 1/N − q_i`（稀有→正、普及→負），bounded、向 simplex 重心的 **centripetal restoring force** → coexistence。這是反同質化引擎，coin 是它的可觀測支付（`score=base·weight`, `coins=score_to_coins`）。
2. **PvP Rank＝競技壓力**：目前惰性（vs static house，只動單一 solo Rank，地牢 Rank 靜態）。Increment 2 會把它「武裝」成作用在 population 上的第二算子——這正是觸發本裁決的精確操作。

**核心機制問題**：neg-freq 多樣性引擎能在多強的 directional 同質化壓力下守住 coexistence？**directional 壓力不需要住在 PvP**——它的最乾淨投放點是生態 fitness 本身：

```
fitness_i = (1/N − q_i)  +  g · d_i        # 第一項=neg-freq centripetal；第二項=constant/directional centrifugal
```

`d` = 固定 directional bias（指定某原型為「就是比較強的 build」，frequency-independent）；`g` = 強度旋鈕。`g=0` 還原現役引擎。掃 `g` → 量重心 coexistence 固定點何時失穩、塌向某 vertex（monoculture）= 一個 transcritical/saddle-node 分岔，臨界值 **g\***。

**為何不武裝 PvP 來問**：(i) 把 directional 力放進 fitness＝單旋鈕、**不開第二 observable**、不污染生態量測（保護已封存的多樣性結果）；(ii) 它是任何未來 PvP 行為實驗的**必備 control**——沒有它，PvP 看到的同質化無法歸因（是 dynamics 本來會塌、還是真人把它推過 g\*）。

---

## 2. 設計

- **型態**：純模擬，g-sweep × 種子網格，含 negative control（g=0）與 positive control（g≫g\*）。
- **動力系統**：每步 (a) player-response driver 依當前 `advantage_i`（含 `g·d_i`）抽 archetype，餵生態 `submit` 流；(b) 生態 EMA 權重照現役 code 更新；(c) `_proportions()` 滑動窗給狀態 `q`。跑到 stationary。
- **directional bias d（鎖定）**：one-hot 取 **Defensive 為 dominant build**：`d = (0, 1, 0)`（ARCHETYPES 序 `[aggressive, defensive, balanced]`，[ecology_tracker.py:30](../../../api/ecology_tracker.py#L30)；`d_dom−d̄ = 2/3`）。
  - **為何 Defensive（實證錨）**：唯一資料背書的 type-chart 邊＝「穩剋莽」，靠 `recklessness→survival ρ=−0.98`（[pvp_manager.py:28](../../../api/pvp_manager.py#L28)）→ defensive(穩)＝存活最優 build；aggressive(莽)＝存活最差；balanced(野)＝設計上刻意稀有。現實的「一個 build 就是比較強」directional 壓力指向**穩**。實驗因此問有實證意義的問題：neg-freq 引擎擋不擋得住向**真選擇會偏好的存活最優 build** 塌陷。
  - **g\* 不受 d 選擇影響**：`1/N−q` 對原型置換對稱，one-hot 取哪 vertex 都 `Δd=2/3`，解析閾值不動（§3）。d 只改語意、不改數學。保持 one-hot＝閉式 g\* + 最乾淨的「單一 build 主導」假設。
  - **caveat**：driver 在 archetype 空間抽樣，故投影不對稱（s_bal 難達，parking_lot D 🟡）**不咬 g\*=1**；若 ② 改在 personality 空間抽樣，該不對稱回來、破壞對稱性（標記給 §11）。
- **provenance**：每 run 寫 `*_provenance.json`（沿用 L3/Phase-1 慣例）。

---

## 3. 條件清單（鎖定）

| 條件 | g | 角色 | 預期 |
|---|---|---|---|
| **NC** negative control | `0.0` | 純 neg-freq，無 directional | stationary entropy ≈ `log 3`（max），**0 fixation** |
| **PC** positive control | `5.0`（≫ 解析 g\*≈1） | directional 壓倒 neg-freq | **全種子 fixation → argmax(d)=Defensive** |
| **treatment sweep** | `{0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0}` | bracket 解析 g\*≈1 | entropy 在某 g 穿過門檻 → 定 g\*_apparatus |

- **每 g × 10 seeds**（連續整數 `50–59`，避免挑種子；同種子集跨 g 重用利於配對比較）。
- **解析 g\* 預測（pre-registered，invasion-fitness 推導）**：monoculture vertex（q_dom→1）穩定 ⟺ resident dom 抵抗 rare 入侵 ⟺ 在 vertex 上 `fitness_dom > fitness_rare`：
  - `fitness_dom|vertex = (1/N − 1) + g·1 = −2/3 + g`
  - `fitness_rare|vertex = (1/N − 0) + g·0 = 1/3`
  - ⟹ 穩定 ⟺ `−2/3 + g > 1/3` ⟺ **g\* = 1.0（raw-fitness 單位）**。（重心鄰域 neg-freq restoring 斜率 = −1，給 coexistence 端的 centripetal 對照。）
- **核心待測**：apparatus（EMA eta=0.2 + softplus lam=2 + 窗=50 + 離散 binning）把 g\* 從 ~1.0 推到哪、往哪個方向、推多少。

---

## 4. 假設與判定（confirmatory，可證偽形式）

> 預先界定「什麼觀測推翻每條假設」，杜絕事後移動球門。

### H-NC — negative control（g=0 守住 coexistence）
- **陳述**：g=0 下 fixation 率 = **0/10**；stationary entropy 95% CI **不涵蓋** fixation floor。
- **角色**：若 g=0 也塌 → response driver / 生態算子接線壞，**先排查再解讀 sweep**（等同 L3 的「Arm A 不到 1.0 即 generator 異常」）。

### H-PC — positive control（g=5 強制 monoculture）
- **陳述**：g=5 下 fixation 率 = **10/10**，且收斂頂點 = argmax(d)=Defensive。
- **角色**：證明 directional 項真的有牽引力、driver 對它敏感。任一不塌 = 接線未生效，sweep 不可解讀。

### H1-bif — 主假設（存在有限 g\*，coexistence→monoculture 分岔）
- **陳述**：存在有限 `g* ∈ (0, 5)`，stationary entropy 隨 g 單調穿過鎖定門檻（§5）由「高熵 coexistence」轉「低熵 fixation」。
- **證偽**：(a) entropy 全 grid 維持高（無 transition）⇒ 此 directional 形式**打不破** neg-freq 引擎（更強的反同質化主張）；(b) transition 出現在 g→0⁺ ⇒ 引擎**平凡脆弱**。兩者都翻轉「存在非平凡 g\*」並須改寫主張。

### H2-faithful — 主假設（apparatus 對 g\* 的扭曲＝本論文機制貢獻）
- **陳述**：報告 `g*_apparatus / g*_analytic`（解析≈1.0），以**描述性**呈現（**非** go/no-go）。
- **判準**：偏離歸因到 softplus(lam) 的增益重映射、EMA(eta) 的遲滯、有限窗(50) 的取樣噪聲、離散 binning、**response driver β**——逐項以關掉/改值的 ablation 佐證。
- **β 的雙面性（鎖定認知）**：vertex 穩定閾值 g\*=1 **對 β 不變**（vertex 處 fitness 符號條件，與 driver 無關）；但**可觀測 fixation g\*（max_q≥0.95）對 β 不是**——β↑ → interior 固定點 q_dom↑ → 更低 g 跨 0.95 → `g*_apparatus` 被往下壓。故 β 是 apparatus 旋鈕、入 §6 ablation，不是自由 nuisance。
- **理由**：分岔的**存在**是教科書；但**在帶這些非理想性的現役儀器裡 g\* 落在哪、被扭曲多少**，正是本專案「metric-defined vs real mechanism」主命題下的真貢獻；同時校驗生態這台儀器對 directional 擾動是否忠實。
- **★ finite-size 扭曲的*符號*依 response driver 邊界行為（2026-06-22 addendum；併入自重複草稿 reduced_form_bifurcation，已撤）**：上述 β 效應是**確定/mean-field**的（β↑→q_dom↑→可觀測 g\* 往下）。另有一層**有限尺寸**效應，其**符號取決於 driver 邊界**：(a) **吸收邊界**（replicator 式 `q_i×`）→ demographic 噪聲在 g→1 把縮小的 minority 推進吸收 → g\* **往下**（finite-N 侵蝕）；(b) **軟地板**（softmax/softplus 式：minority 缺席→最大稀缺獎勵→再播種）→ finite-W 滅絕被**推遲** → g\* **往上**（finite-W 保護）。**現役算子 driver 是 softmax/softplus（軟地板）→ 預測 finite-W 把 g\* 往上推、部分抵消 β 的往下效應** → §6 ablation 應分離這兩個反向效應。獨立 g=0 κ-sweep（遊戲軌，harness `scripts/experiments/ecology_reduced_form_bifurcation.py`）已證軟地板下 whiplash 由 α（重選率/慣性）主導、neg-freq 引擎在 production per-submission 序列化下穩健，佐證軟地板再播種的保護性。

### 機制診斷（探索性，報告不作 confirmatory 判定）
- **遲滯（hysteresis）**：g 上掃 vs 下掃的 g\* 是否不同 → 分岔型態（乾淨 transcritical vs 有遲滯的 saddle-node）= phase portrait 診斷。
- time-to-fixation vs g；窗大小 ∈ {25,50,100} 對 g\* 的敏感度。

---

## 5. Operationalization（逐字鎖定）

- **算子（不重寫）**：直接 import 現役 `EcologyTracker`，唯一改動＝`_fitness` 回傳加 `g·d_i`（以 param 注入，g=0 為預設＝現役行為）。其餘 `lam=2`、`eta=0.2`、`base=100`、`window=50`、`tau=0.6` 全鎖現役預設（[api/ecology_tracker.py](../../../api/ecology_tracker.py) L92–103）。
- **response driver（鎖定）**：每步玩家以 `P(i) ∝ softmax(β · advantage_i)` 抽 archetype，`advantage_i` ＝現役 `_advantage(_fitness(q)+g·d)`。**`β = 2`（central）**＝錨定到算子的 `lam=2`（[ecology_tracker.py:98](../../../api/ecology_tracker.py#L98)），令 driver 判別銳度＝算子 fitness→advantage 銳度、兩層互不壓制；落在 weak-selection 帶（中等 gap 偏好比 ~1.5×，響應但不 argmax）。β 同列 §6 ablation `{1,2,4}`（見 §4 H2 β 雙面性）。
- **observable**：`q = _proportions()`（滑動窗，現役 L143–151）。`H(q) = −Σ q_i ln q_i`，最大 `ln 3 ≈ 1.0986`。
- **fixation 判準（鎖定）**：stationary 窗內 `max_i q_i ≥ 0.95`。
- **stationary 窗（鎖定）**：`rounds=3000`、`burn_in=1000`、`tail=1000`（與 L3 同窗，利於跨研究比較）。
- **g\* 定位**：treatment grid 上 fixation 率首次由 0 升至 ≥0.5 的相鄰 g 區間，以 stationary entropy 對 g 的 logistic 擬合取半飽和點為 `g*_apparatus`（連續估計，非格點）。
- 任一參數最終綁定源為 confirmatory runner config（隨本 pre-reg 提交）；衝突以本文件為準、記 §10。

---

## 6. 樣本

- NC(g=0) + PC(g=5) + 7 個 treatment g × 各 10 seeds（50–59）= **90 runs** × 3000 rounds。
- ablation（H2-faithful）：在 g\* 鄰近 3 個 g 上，分別關 softplus（線性 advantage）/ 設 eta=1（無 EMA 遲滯）/ 窗∈{25,100} / **β∈{1,4}**（central 2 已在主 sweep），各 10 seeds。算力與 L3 Phase-2 同量級，預估 < 2 機時。

## 7. 停止規則與容忍

- **停止**：90 confirmatory runs + ablation 滿額即停；**不期中偷看**，僅監控完成計數。
- **control 容忍**：NC 允許至多 **1/10** fixation（尾端孤點），≥2/10 ⇒ H-NC 推翻、排查 driver。PC 要求 10/10，任一不塌即 root-cause。
- **合格 run**：完成全 rounds、無 NaN、provenance 完整；崩潰同種子重跑一次，再失敗記 §10。

## 8. 分析執行

- 輸出 per-(g,seed) `fixation`(bool) / `stationary_entropy` / `top_q` / `time_to_fixation` 至 `reports/experiments/ecology_directional_pressure/sweep_combined.tsv`。
- confirmatory 報告：H-NC/H-PC 控制表、H1-bif（entropy-vs-g 曲線 + g\* logistic 擬合 + CI）、H2-faithful（`g*_app/g*_analytic` + ablation 歸因表）。
- 機制診斷：hysteresis（上/下掃疊圖）、time-to-fixation-vs-g。
- **產出**：`reports/experiments/ecology_directional_pressure/eco_dp_analysis.json` + 報告 md。

## 9. 預期結果與論文映射

| 結果 | 主張 |
|---|---|
| NC 守 coexistence + PC 全塌 + 存在有限 g\* + 報告 `g*_app/g*_analytic` 與 ablation 歸因 | **因果主張成立**：neg-freq 引擎被 directional 壓力在 g\* 處分岔；apparatus 把 g\* 從解析 ~1 扭曲量化；儀器忠實性受檢 |
| NC 自塌 或 PC 不塌 | 控制失效 → driver/接線異常，**先排查再解讀 sweep**（不得逕報分岔） |
| entropy 全 grid 維持高（無 g\*） | 負結果**升級**：此 directional 形式打不破 neg-freq 引擎 → 更強反同質化主張，改寫 |

---

## 10.（③，並行）PvP 隔離契約 — 把「牆」寫成可檢驗 invariant

① 之所以乾淨，靠的是 PvP **不碰生態 observable**。③＝Increment 2 的 PvP 以 game feature 落地，但用以下契約把它牆在多樣性量測之外：

- **C1 因果隔離 invariant（可測）**：任何 PvP challenge / 結算 **不得**呼叫生態 `submit()`、不得 mutate `_recent`/`_proportions`/`_weights`。落地測試＝跑一批 PvP 對戰前後，斷言生態 submission 數與 `_proportions()` **位元級不變**。這把「隔離」從口號變成 CI 可擋的斷言。
- **C2 coin sink 去耦**：③ 下 coin sink ＝**門票（配對成本）＋ 防禦升級**，**不**做 coin→Rank 抵銷。
- **C3 因此 parking_lot C 的記帳問題 moot-by-deferral**：既然沒有 coin↔Rank 耦合，「零和守恆 vs 通膨 / A/B/C」「offset_ratio / coin_per_rank」**當前無對象可選**——不是選了某案，是這個 fork 在 ③ 下不存在。Rank ledger 退化成簡單帳（沿用 v1 vs-house，或真玩家間零和——與 coins 正交，之後獨立決定）。
- **C4 RPS 保留為活機制**：authored 剋制表（faction Option 4+）續作玩家面 counter-play，只是**從不寫進多樣性 observable**。生態 M / combat M 不統一（已決）在此自動成立。

> ③ 與 ① **正交、可並行**：① 在生態/sim 層，③ 在 PvP/game 層，C1 invariant 保證兩者不爭同一 observable。唯一被犧牲的是 parking C 那條 coin→Rank 字面接線（它本就是污染向量）。

---

## 11.（②，延後）行為版 PvP 儀器 — gate 條件

- **②＝把 directional 壓力改由真玩家在競技階梯上的選擇驅動**，回答「真人會不會被推過 g\*」。它需要另建 directional stat-ladder（RPS 是 cyclic、非 directional，不適任此角色）+ raw/effective 雙帳。
- **Gate（pre-commit）**：**不得**在 ① 報出有限、定位良好的 `g*_apparatus` 之前啟動 ②。理由：① 是 ② 的必備 baseline——缺它，②的同質化觀測無法在「dynamics 本會塌」與「真人推過 g\*」之間歸因（causal isolation 失效）。
- ① 出 g\* 後，再 hypothesis-driven 決定值不值得為 ② 付「新階梯 + 雙帳 + 收真人 + finite-size 風險」的成本。

---

## 附：本 doc 與既有決策的關係（不重議）
- 承接：生態 M/combat M 不統一、faction Option 4+、現役 neg-freq `_fitness`（parking_lot A/D）。
- 重定：parking_lot C「Rank 零和 vs 通膨 / A/B/C」→ 在 ③ 下 moot-by-deferral（§10 C3）；directional 研究問題移入生態 fitness（§1）。
- 待 ratify 後鎖定本文件、建 runner、跑 §6。
