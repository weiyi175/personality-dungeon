# 乙 — 真人 authoring 稀缺-響應銳度 β 的行為量測（Pre-registration）

> **狀態**：DRAFT v1（2026-06-23）。這是 [DIRECTIONAL_PRESSURE_PREREGISTRATION.md](DIRECTIONAL_PRESSURE_PREREGISTRATION.md)（canonical，①）
> 明文延後的 **② arm**。①在「玩家以 `P(i)∝softmax(β·advantage)` 響應稀缺」這個**假設的 response model 之下**
> 算出 g\*(β)（§5 鎖定的 driver、§4b 的 g\*(β) 曲線）。**② = 把那個 β 從假設變成實測。**
> 分析器與證偽律已先行造好並驗證：`scripts/experiments/ecology_beta_fit.py`（+ power `ecology_beta_power.py`），
> 報告 `reports/experiments/ecology_directional_pressure/BETA_INSTRUMENT.md`。

## 0. 探索 / 確認分界

- **Confirmatory（本 pre-reg 鎖定）**：H0（β=0，真人不理 live 稀缺）的證偽；β 點估計 + CI；以及 β 落點對
  g\*(β) robustness 預算的映射判定。分析模型、排除律、樣本量、停止律**逐項預先鎖定於下**。
- **Exploratory（報告但不作 confirmatory 判定）**：α intrinsic 截距的精確值（[[real-human-intrinsic-archetype-dist]]
  已知 balanced-thin 穩健、agg/def 排序不穩）；response 的函數形式（logit vs probit vs rank，見 §7 validity）。

## 1. 背景與目的 — 為何需要 ②，且為何「現有資料」不夠

- ECO-DP 的唯一開放經驗量＝**真實 β**。g\*(β) 是 robustness 預算的**遞減函數**（β=2→g\*≈2.06、β→∞→解析 1）。
  ① 全程把 β 當**假設輸入**；機制結論因此是 conditional-on-β。② 量出 β，才把 g\*(β) 從一條曲線釘成一個數。
- **現有 live 資料存在但無資訊量**（2026-06-23 驗證，BETA_INSTRUMENT.md §4）：`ecology_state.json` 有 **n=208**
  真人 V2-live 提交（非 sim——先前「n=0/205 sim」是 run_id 慣例誤判，已更正），估出
  **β̂=0.22, 95%CI [−1.6, 2.0]**——CI 橫跨 0 到 g\* 錨點 2.06，**連 H0 都拒不掉**。
- **失能主因（② 要解的）**：(a) 稀缺幾乎沒變動（scarcity_std 0.074，落在 power 分析的 low-variation 失能區）；
  (b) 疑 pseudo-replication（208 session 未必獨立真人，疑開發期重複跑）。② 的設計**就是針對這兩點**。

## 2. 假設與判定（confirmatory，可證偽形式）

模型（conditional logit，分離 intrinsic 與稀缺響應）：`P(author archetype j | adv) = softmax(α_j + β·adv_j)`，
`adv_j = softplus(lam·(1/N − q_before_j))`，α_balanced≡0 基準。

- **H0（null，主證偽對象）**：**β = 0** — 真人 authoring **不理** live 稀缺，純 intrinsic。
  - 判準：95%CI 是否含 0。CI 排除 0 且 β̂>0 ⟹ 拒 H0（真人會追稀缺）。
- **H1（方向）**：**β > 0** — 顯示某原型稀缺時，真人更傾向 author 該原型。
- **H2（量化 → g\* 映射，主貢獻）**：β 落在哪個區間，決定「real」反同質化預算：
  - β̂ 顯著 < 1 → robustness 預算 **> 解析 2×**（g\* 大、機制很耐操）；
  - β̂ ≈ 2（① 的錨）→ 預算 ≈ 2.06；
  - β̂ 顯著 > 4 → 預算逼近解析下限（g\*→1，機制脆）。
  - 判準：β̂ 的 CI 是否把上述相鄰區間分開（見 §4 樣本量）。

## 3. 設計 — 三條鐵律（由 power 分析鎖定，BETA_INSTRUMENT.md §6B）

1. **稀缺必須隨時間變動**（這是現有 n=208 失能的真主因，不是 n）。收集期間生態狀態 q 必須有真實漂移，
   使不同真人面對不同稀缺向量。兩個門檻**別混**：
   - **識別性硬 gate**：scarcity_std ≥ 0.06（< 0.02 → 估計器判 UNIDENTIFIED、不報 β）。低於此整批作廢。
   - **power 設計假設**：§4 的 n target（300 拒 H0 / 1000 釘區間）是在**高變動 std≈0.22** 下算的。
     若實收只到 mid（≈0.12），n 須上調（resolve 需 ~3000，見 §6B）；只到 low（≈0.06）則 power 嚴重不足、
     幾乎拒不掉 H0。**故設計目標 = 把 std 推到 ~0.2**，而非僅過識別性 gate。
   induction 手段（§5 apparatus）：跨時段/跨批收集讓 q 自然漂移、或主動輪換顯示的稀缺型。
2. **獨立受試**：以 `participant_id` 追蹤；分析**以 participant 為 cluster**（避免 pseudo-replication）。
   排除無歸屬的 pilot/dev session（即 P7-H 那 54 筆 `cycle_index=-1` 類的東西）。
3. **樣本量 + 停止律**：見 §4。預先鎖定 target 與 stopping，不看資料調整。

## 4. 樣本量 / 停止律（power 鎖定）

power 模擬（`ecology_beta_power.py`，真實 intrinsic α、高稀缺變動）：

| 目標 | 條件 | 預先鎖定 target |
|---|---|---|
| **拒 H0**（β=0），最低可發表 | 高變動（std≈0.22） | **n ≥ 300** 真人 live（97–100% power @ β≈2） |
| **釘進 g\*(β) 區間**（分辨 β=1/2/4） | 高變動 | **n ≈ 1000**（CI 半寬 ~±0.3） |

- **主停止律**：收滿 **n=1000** 獨立-加權真人 live 提交**或** β̂ 的 95%CI 半寬 ≤ 0.35（先到先停）。
- **最低出場律**：若資源受限，n=300 達標即可作 H0 confirmatory（H2 量化降為 exploratory）。
- **識別性 gate（硬條件）**：分析前先檢 scarcity_std ≥ 0.06；未達 → 判 UNIDENTIFIED、不報 β 點估計（不偽造數字）。

## 5. Operationalization（逐字鎖定）+ apparatus 前置

- **量測管線（鎖定）**：`ecology_beta_fit.load_real_submissions`（真人判準 = `session_id` ∧ `outcome` 雙非空）
  → `fit_beta`（scipy BFGS + analytic gradient，CI by inv-Hessian）→ verdict guard
  `INSUFFICIENT_N(<30) / UNIDENTIFIED(scarcity_std<0.02) / OK`。**這些閾值此刻鎖定，不事後調。**
- **apparatus 前置（收集前要 build，gate 在此）**：
  1. **稀缺漂移 induction**：確保收集期 q 有變動（鐵律 1）。
     - ✅ **後端 monitor 已建**（2026-06-23）：`GET /ecology/scarcity_variation`（+ `ecology_beta_fit.py --monitor`）
       即時報真人 live 的 advantage-std 對 gate 0.06 / target 0.2 → 收集中可隨時驗鐵律 1。現況 std=0.074「過 gate
       但偏低、power 不足」。最小 induction＝跨足夠長時段/分波收集讓窗自然演化（由 monitor 確認達標）。
     - ⏳ **強化 induction（顯示輪換/隨機化）延後**：會讓顯示≠真 q → 動到 coin 誠實性/F6，是**設計/firewall 抉擇**，
       不單方面寫碼；要做需先過 firewall 重審（與 [[economy-architecture-r3c-ii]] 對齊）。
  2. **`seen_scarcity` passthrough**：記人**實際看到**的稀缺快照（消除 display-vs-submit 漂移 + 確認 softplus(adv)
     link → 解鎖 β 的**絕對** g\*(β) 刻度；否則 β 只當單調指標，§7）。
     - ✅ **後端已建**（2026-06-23）：`/ecology/submit` 收 optional `seen_scarcity`（len 3）→ 落檔 `EcologySubmission`；
       β-instrument 優先採用、退回 q_before 代理並回報 seen/proxy 筆數。純記錄、不入評分、F-safe。
     - ⏳ **前端側待做**（需 Godot 驗）：V2 author 前把顯示的快照隨 submit 帶上。現全部走 q_before 代理（低流量足夠）。

## 6. 威脅與混淆（預先聲明處理）

- **pseudo-replication**：以 participant cluster；report ICC / cluster-robust CI。單人多 cycle 不當獨立樣本。
- **intrinsic α 混淆**：模型內含 α_j 截距吸收（[[real-human-intrinsic-archetype-dist]] 的 balanced-thin 等），β 為**淨**稀缺響應。
- **模型 mis-specification（validity envelope，已測）**：BETA_INSTRUMENT.md §6A——真人若走 probit/q-linear/rank
  響應，估計器仍 100% 偵測到「β>0 顯著」、null 不誤報，**但絕對 β 是 model-specific**（rank DGP 膨脹）。
  ⇒ **未確認 link 前，β 報為稀缺響應的單調偵測器/指標**；H0 證偽**穩健**，H2 的絕對 g\* 映射**待 link 確認**（§5 前置 2）。
- **display ≠ submit 漂移**：現用 q_before 代理，低流量足夠；高流量或要絕對刻度 → seen_scarcity。

## 7. 與 ECO-DP / firewall 的關係

- **F-safe**：② 是**純觀測**——不新增算子、不碰 Rank/coin、不改提交流（只可選**加記** seen_scarcity）→ 不觸任何 firewall invariant（[[economy-architecture-r3c-ii]] 的 F1–F6/S1 全不動）。
- **接合**：② 的 β̂ 直接代入 ① 的 g\*(β)（ECO_DP_RESULTS.md §4b）→ 把 conditional-on-β 結論釘成定量。
- **與 PvP 行為版（研究軌 B）區隔**：本 ② 量的是**生態 authoring 對 neg-freq 稀缺**的響應；
  「競技階梯下是否同質化」是另一條、被 firewall 隔離在外、另需 pre-reg（game-spec §7）。**勿混。**

## 8. 執行協定 — pilot→confirm + 雙實驗隔離（2026-06-23 鎖定）

> 兩層分界：**產品迭代↔研究量測**用「時間分相 + config 版本」隔；**乙↔PvP-同質化(B')**用「firewall」隔。

### 8.1 三相 + 凍結 gate
1. **Pilot（探索；其數據不進 confirmatory 推論）**：自由調遊戲/裝置（手感、coin 數值、**稀缺 induction**、UI）。
   出場 gate（全滿足才可凍結）：① monitor `/ecology/scarcity_variation` 報 **`meets_target`（advantage-std ≥ 0.20）**；
   ② 裝置 config 連續穩定 ≥ 3 個收集日無改動；③ 估計器在 pilot 資料上跑出**合理形狀**（verdict OK、α 截距與 [[real-human-intrinsic-archetype-dist]] 同量級）。
2. **凍結**：鎖第 8.3 的「凍結清單」，指派一個 **config_version 字串**（如 `confirm-2026-07-a`）。
3. **Confirmatory（凍結；不動裝置/模型）**：照 §4 固定停止律（n≥300 拒 H0 / ~1000 釘 g\*(β) 區間或 CI 半寬≤0.35）收。
   **禁 optional stopping**（不可「β 一顯著就停」）。要再調裝置＝**跑完本波→改→重凍結→下一波**，不同 config **不可混池**。

### 8.2 產品線可動 vs 研究線凍結
- **隨時可動（產品/裝置，pilot 或波次間）**：UI/手感、`starting_balance`/`ticket_cost`/`defense_cost`、稀缺 induction 機制、招募、session 長度。
- **confirmatory 窗內凍結（量測）**：β 模型（conditional logit）、advantage 重建（`lam`）、真人判準（session_id ∧ outcome）、verdict 閾值（INSUFFICIENT<30 / UNIDENTIFIED<0.02）、H0/停止律/排除律、**及裝置 config**（窗內不動）。

### 8.3 標記規範（zero backend change：走既有 `outcome` dict 傳遞）
前端每筆 `/ecology/submit` 的 `outcome` 內**附帶**：
- `study_phase`：`"pilot"` | `"confirm"`（pilot 一律排除於 confirmatory 分析）。
- `config_version`：字串；confirmatory 分析**只取單一 config_version**，跨版本不池化。
- （沿用既有）`participant_id`：以 participant 為 cluster，防 pseudo-replication（§6）。
分析端：confirmatory 子集 = `study_phase=="confirm"` ∧ 指定 `config_version` ∧ 真人判準（session+outcome）。
*（此規範只需前端在 outcome 塞欄位 + 分析端 filter；`outcome` 本就原樣落檔，後端零改動。）*

### 8.4 雙實驗隔離（乙 ↔ B' PvP-同質化）— 靠 firewall，可同時在同一玩家身上跑
- **F5 observable 分離**：乙 讀 will-authoring → archetype 分佈（ecology store）；B' 讀 PvP Rank/loadout（pvp store）。兩條 ledger 永不混流。
- **F1（PvP 零讀 will）+ F4（PvP 不呼 ecology.submit）**：玩家同時打 PvP 也進不了乙 的生態量測。
- **store 標記**：ecology 真人 = `session_id` ∧ `outcome`（**非** run_id 號——run_id 慣例 store-specific，見 [[real-human-intrinsic-archetype-dist]] 踩過的坑）；PvP 記錄走 pvp_state。**勿跨 store 套 run_id 判準。**
- **優先序**：乙 先（instrument 已備、β 餵 g\*(β)；B' 的 null＝reduced-form g\* 需先有 β）；B' 另開 pre-reg、待 Increment 3 競技迴路成熟。

### 8.5 執行 runbook（照表即可直接實驗）
- [ ] **R0** 確保賺幣路徑通：玩冒險會 credit 生態 coins（已修）；若要存活幣，先接 `credit_survival`（目前是死路）。
- [ ] **R1 pilot**：開放真人玩，自由調 induction/手感/coin；每日看 `GET /ecology/scarcity_variation`。
- [ ] **R2 凍結 gate**：monitor `meets_target` + config 穩定≥3 日 + pilot 估計器形狀合理 → 通過。
- [ ] **R3 凍結**：鎖 §8.2 量測 + 指派 `config_version`；前端起在 `outcome` 寫 `study_phase="confirm"` + 該 `config_version`。
- [ ] **R4 confirmatory 收集**：收到 n≥300（拒 H0 最低）或續至 ~1000 / CI 半寬≤0.35；**窗內不動裝置/模型，不 optional-stop**。
- [ ] **R5 分析**：`ecology_beta_fit`（confirmatory 子集）→ verdict + β̂ + CI；對照 §2 判 H0/H2，代入 g\*(β)。
- [ ] **R6**（可選）接 `seen_scarcity` 前端 → 確認 softplus link → 解鎖 β 絕對刻度（否則 β 為單調指標）。
- [ ] **R7**（之後）另開 B' pre-reg（PvP-同質化），firewall 已保證與乙 隔離。

## 9. Provenance

- 上游：① canonical pre-reg（假設 β）、g\*(β) 曲線（ECO_DP_RESULTS.md §4b）、β-instrument + power（BETA_INSTRUMENT.md）。
- 對齊記憶：[[beta-instrument]]、[[eco-dp-directional-pressure-reframe]]、[[real-human-intrinsic-archetype-dist]]、[[economy-architecture-r3c-ii]]。
- 主張：① 給了「若 β 是 X 則機制 robustness 是 Y」的完整地圖；② 是去把真實的 X 量出來的**唯一**待辦，
  且其失能模式（稀缺不變、pseudo-replication）已由本輪驗證具體定位、由三鐵律對症。
