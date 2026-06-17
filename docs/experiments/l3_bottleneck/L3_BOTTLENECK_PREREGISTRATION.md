# L3 Bottleneck 負結果 — 預先註冊（Pre-registration）

> **鎖定狀態**：本文件於 confirmatory 擴充收案開始前定稿。一旦開始執行 §6 的 confirmatory seeds，下列假設、判定鏈、條件清單、種子數與停止規則 **不得修改**。任何偏離須在最終報告的「Deviations」段落（§10）逐條明列。
>
> **定稿日期**：2026-06-18
> **研究代號**：L3-BN（Level-3 bottleneck negative result）
> **論文定位**：機制級**因果隔離**（負結果 + mean-field 正控制）— *Sampling destroys RPS rotation: a mean-field-vs-sampled isolation of where Level-3 dies in discretized replicator dynamics*
> **判定鏈（鎖定）**：[analysis/cycle_metrics.py](../../../analysis/cycle_metrics.py) `classify_cycle_level`（L1345）／`phase_direction_consistency_turning`（pass 判準 L1082）
> **exploratory 來源（已完成）**：[level3_bottleneck_phase1_closure.md](../../../level3_bottleneck_phase1_closure.md)（B/C/T 三系列，≈120 runs，3 seeds/cell）

---

## 0. 本研究的探索 / 確認分界（誠實聲明）

- **Phase 1（exploratory，已完成，不可改寫）**：B2/B3/B4/B5、C1/C2、T-series 與 T follow-up 之短掃描（每 cell 3 seeds，≈120 runs）。此階段為**假設生成**，所有結論（0/120 達 L3、三層均化機制）在此被視為 exploratory，**不**作為本論文的 confirmatory 證據。
- **Phase 2（confirmatory，本文件鎖定）**：在 Phase 1 已生成、且**事前選定**的「最有資訊量 / 最接近 L3」cell 上，把種子由 3 擴充至 10，於**鎖定的判定鏈與分析計畫**下檢驗負結果是否守得住。
- 此分界等同 P7-H 的「offline pipeline 驗證 → pre-reg 真人 confirmatory」結構（見 [P7H_PREREGISTRATION.md](../p7_online_personality_loop/P7H_PREREGISTRATION.md) §0–§3）。混用兩階段的證據視為偏差。

---

## 1. 背景與目的

近中性三策略（Aggressive / Defensive / Balanced）replicator 系統在 personality-event 回饋下可從靜止吸引子（Level 0）進入**結構性極限環（Level 2）**，此正向結果跨 30 runs（seeds 1–10 × σ∈{0.005,0.05,0.20}）穩健（[docs/paper_draft_v1.md](../../../docs/paper_draft_v1.md) Abstract，Pr=30/30）。Level 2 = 有振幅、有週期，但**相空間轉向無單一方向性**。

**Level 3** 要求相空間運動具**單向旋轉一致性**（sustained RPS rotation）。Phase 1 在 30+ 條件、≈120 runs 下取得 **0 個 L3 seed**，並提出機制假說：sampled discrete synchronous replicator update 的**三層均化**（sampling noise → popularity averaging → synchronous update）系統性耗散 rotational/directional component，使 L2 plateau 成為強吸引子。

**研究問題（confirmatory）**：在 Phase 1 中**最接近 L3、機制訊號最強**的 sampled cell 上擴充種子，L3 是否仍**結構性不可達**、stage-3 一致性是否**統計上不可與機率水平 0.50 區分**；且此不可達性是否**特屬於離散抽樣層**——亦即其**確定性 mean-field 對應系統是否反而達到 consistency=1.0 的完美 L3 旋轉**？

**因果升級（2026-06-18，Phase-1 資料中發現正控制）**：B5 generator 同時含兩個 gate——`g1_mean_field`（確定性 mean-field、無抽樣）跨全部 delta×seed **穩定 cycle_level=3、consistency=1.0000**（三種子位元級相同 ⇒ 確定性確認）；其抽樣孿生 `g2_sampled` 塌至 consistency≈0.50、cycle_level=2（[outputs/b5_tangential_drift_short_scout_summary.tsv](../../../outputs/b5_tangential_drift_short_scout_summary.tsv)）。二者**唯一差異為 mean-field→sampled**。故本研究主張由「有界負結果（各 patch 皆打不開 L3）」升級為**因果隔離**：**旋轉的 L3 結構在連續極限證明存在（1.0），被抽樣+均化離散層證明性摧毀（→機率水平）**。耗散歸因於離散化層，非 payoff 缺旋轉結構。

---

## 2. 設計

- **型態**：純模擬、條件×種子網格、與 control（well-mixed sampled）並列。
- **正向對照（locked anchor，不重跑）**：paper_draft_v1 之 L0→L2（30/30 robust）作為「機制能達到的上限」基準，劃定「L2 能、L3 不能」的邊界主張。
- **每 run 流程**：依各條件的 evolution operator 跑 `rounds` 輪 → 取 `burn_in`/`tail` 窗 → 經**鎖定判定鏈**（§5）輸出 `cycle_level`、`stage3_score`(=consistency)、`turn_strength` 及機制診斷量。
- **provenance**：每 run 寫 `*_provenance.json`（沿用 Phase 1 慣例，見 [outputs/t_series_short_scout/](../../../outputs/) 之 `seed*_provenance.json`）。

---

## 3. 條件清單（鎖定）— 含 C2-uniform 剔除裁定

### 3.1 有資訊量的條件家族（Phase 1 → Phase 2 候選）

| 家族 | 機制 | Phase 1 cell 數 | Phase 1 L3 | 機制訊號 | 入 Phase 2 |
|---|---|---|---|---|---|
| **B-series** | well-mixed 上的結構 patch（B2 島嶼、B3 分層、B4 振幅依賴 k、B5 切線 drift） | 21 active | 0 | patch 被三層均化吸收 | 取最高 stage3_score cell ×1 |
| **C1** | local Fermi pairwise imitation | 6 active | 0 | **過度同質化**（edge_strategy_distance=0.000） | 是（最乾淨的「反向失敗」） |
| **T-series** | random-init local minibatch replicator（**真正的 local 測試**） | 6 active | 0 | **domain 成形**（batch_phase_spread≈1.55 rad、player_growth_dispersion≈0.26–0.33） | 是（**最接近 L3 的 best-hope**） |
| **T follow-up** | 同上 + 強化 k_local=0.12 | 2 active | 0 | uplift 仍 ≤0 | 是（selection-pressure 反證） |

### 3.2 ⚠ C2-uniform 剔除裁定（鎖定，不可逆）

C2（local minibatch，**uniform init**）於 Phase 1 量得 `mean_local_growth_cosine_vs_global = 1.000000`、`mean_batch_phase_spread = 0.000000`（精確 0；見 [outputs/c2_local_minibatch_short_scout_decision.md](../../../outputs/c2_local_minibatch_short_scout_decision.md)）。在 uniform init 下 `w_i(0)` 全等 ⇒ `x_local ≡ x_global` ⇒ `g_local ≡ g_global` **恆等**——此 config **按建構退化為 global replicator**，並非對 local replicator 的有效檢驗。

**裁定**：C2-uniform **自「已測試介入」計數中剔除**，僅以**方法學警示**呈現（「天真的 local-replicator 離散化會退化成 global，是該領域實作的隱藏陷阱」）。對 local replicator 的**唯一有效檢驗為 T-series（random init）**。此剔除使 Phase 1 的有效條件數小於 closure 文件的「30+」表述，confirmatory 不得引用 C2-uniform 作為負證據。

### 3.3 Phase 2 confirmatory cell 選取（已鎖定 → [phase2_cell_selection.json](phase2_cell_selection.json)）

選取規則：**keystone = B5 generator 內 mean-field vs sampled 配對對照**；**breadth = 各機制相異的 informative 家族中，Phase-1 平均 `stage3_score`(=consistency) 最高的單一 sampled cell**。規則以 Phase-1 凍結資料計算、對審稿透明、避免事後挑 cell。選取已寫入 `phase2_cell_selection.json` 並隨本 pre-reg git 提交（時間戳即鎖定證據）。

**Arm A — 正控制（mean-field，確定性，不需擴充種子）**

| cell | 系統 | Phase-1 level | consistency | 角色 |
|---|---|---|---|---|
| `g1_mean_field_delta0p000` | 確定性 mean-field（無抽樣） | **3** | **1.0000** | 證明旋轉結構在連續極限存在 ⇒ 負結果是離散化效應 |

> mean-field 為確定性（seeds 45/47/49 位元級相同），擴充種子無意義；其再現性改以 delta∈{0,.003,.006,.010,.015} 全為 consistency=1.0、L3 呈現。

**Arm B — sampled 處理組（各擴充 3→10 seeds）**

| cell | 家族 | Phase-1 mean / max consistency | Phase-1 L3 | 角色 |
|---|---|---|---|---|
| `g2_sampled_delta0p000` | B5 切線 drift（**Arm A 的抽樣孿生**） | 0.5150 / 0.5410 | 0/3 | **keystone** 配對：同 generator，僅 mean-field→sampled |
| `beta0p60_k0p08` | B4 振幅依賴 k | 0.5306 / **0.5488** | 0/3 | 全 Phase-1 **最接近 0.55 bar** 的 sampled cell |
| `t_lattice4_minibatch` | T local-minibatch（random init，**唯一有效 local 測試**） | 0.5061 / 0.5255 | 0/3 | domain 成形（batch_phase_spread≈1.55 rad）但 consistency 仍卡 chance |
| `g2_c1_small_world_b10p0_m0p5` | C1 local pairwise Fermi | 0.5109 / 0.5219 | 0/3 | 過度同質化分支（最高 mean 之 C1 cell） |

**理由**：keystone 給因果歸因（離散層殺死旋轉）；breadth 三 cell 證明負結果橫跨 B4/T/C1 三機制、非 B5 特例。負結果若連「最接近 bar 的 best-hope cell」在 N=10 下都守住，對全網格結論最難推翻。

---

## 4. 假設與判定（confirmatory，逐字對齊判定鏈）

> 負結果的 confirmatory 假設以**可證偽形式**陳述：預先界定「**什麼觀測會推翻負結果**」，杜絕事後移動球門。

### H0-neg — 主假設（L3 結構性不可達）
- **陳述**：在 §3.3 選定的 4 個 best-hope cell（各 10 seeds，共 40 runs）中，L3-seed 率維持 **0/10**（容忍見停止規則 §7）。
- **L3 判準（鎖定）**：`cycle_level == 3` ⟺ stage-1 ∧ stage-2 ∧ stage-3 全通過（§5）。
- **證偽條件**：任一 cell 出現 **≥ 2/10** L3 seeds（見 §7 容忍邏輯）⇒ H0-neg 於該 cell 被推翻，須降級主張並回報。

### H1-chance — 主假設（一致性卡在機率水平）
- **陳述**：各 confirmatory cell 之 `stage3_score`(=consistency) 分布的母體平均**統計上不可與 0.50 區分**。
- **檢定**：對每 cell 的 10 個 consistency 值作**單樣本雙尾 t 檢定 vs μ₀=0.50**；一併報告平均、95% CI、與 0.50 之差的 Cohen's d。
- **判準**：**不**作 go/no-go 顯著性反推；以「CI 是否涵蓋 0.50」與效應量呈現「一致性 ≈ 機率水平 = 無方向性旋轉」。
- **理由**：consistency=0.50 為相空間轉向符號無偏（coin-flip）之期望值，即**噪聲式擺動而非 rotation**；plateau 坐在 0.50 是比「未達某門檻」更強的機制陳述。

### H2-threshold-robust — 主假設（結論不依賴 eta=0.55）
- **陳述**：負結果對 stage-3 門檻 `eta` 的選擇穩健。
- **分析**：在 confirmatory 資料上對 `eta ∈ {0.55, 0.60, 0.65, 0.70}` 重算 L3-seed 率，並報告完整 consistency 分布直方圖。
- **判準**：若 L3-seed 率在全 sweep 維持 0（或 §7 容忍內），則「唯一 L3 例（paper_draft_v1 seed45, consistency=0.5611）為分布尾端孤點、僅高於 0.55 約 1 點、高於機率水平約 6 點」之描述成立。

### H-PC — 正控制（mean-field 達成完美 L3 旋轉）
- **陳述**：Arm A 之 `g1_mean_field` 維持 `cycle_level==3` 且 `consistency==1.0`（確定性，全 delta）。
- **判準**：確定性系統，預期位元級再現；任何 delta 出現 cycle_level<3 即視為 generator/判定鏈異常，須排查後始能解讀 Arm B。
- **角色**：與 H0-neg/H1-chance 構成**因果對照**——同 generator 下 mean-field（consistency=1.0, L3）vs sampled（consistency≈0.50, L2）。此對照是論文機制歸因的脊椎；缺正控制則負結果僅為「未觀察到」，有正控制則為「離散層摧毀了確證存在的旋轉」。

### 機制診斷（探索性，報告但不作 confirmatory 判定）
報告各 cell 的 `mean_batch_phase_spread`、`mean_player_growth_dispersion`、`mean_edge_strategy_distance`、`mean_local_growth_cosine_vs_global`、`spatial_autocorrelation_d1`、`init_weight_dispersion`，以支撐「三層均化耗散 rotational component」之機制圖像（domain 成形但相干壽命不足）。

---

## 5. Operationalization（逐字鎖定判定鏈）

L3 判定由 [analysis/cycle_metrics.py](../../../analysis/cycle_metrics.py) `classify_cycle_level` 級聯決定（L1345）：

```
Stage 1（振幅）：assess_stage1_amplitude，amplitude_threshold=0.02
   未過 → Level 0
Stage 2（週期）：assess_stage2_frequency，stage2_method="autocorr_threshold"，corr_threshold=0.09
   未過 → Level 1
   ⚠ corr_threshold 僅 gate L1→L2；本研究全部 confirmatory cell 早已是 L2/L3（Stage 2 已過），
     故 corr_threshold 對「L2→L3」之負結果問題 **非綁定**，鎖 0.09（L2-lineage 慣例）不影響結論。
Stage 3（單向旋轉）：phase_direction_consistency_turning，stage3_method="turning"
   未過 → Level 2
全過 → Level 3
```

**Stage 3 pass 判準（命門，cycle_metrics.py:1082，逐字）**：
```python
passed = (consistency >= float(eta)) and (turn_strength >= float(min_turn_strength))
```
- `score`(報表 `stage3_score`) **＝ consistency**：相空間連續轉折叉積 `cross(Δp_t, Δp_{t+1})` 之符號與淨旋轉方向一致的比例 ∈ [0,1]（cycle_metrics.py:1078–1082）。**0.50 = 機率水平（方向無偏）**。
- `turn_strength` ＝ `mean(|cross|)`。
- **鎖定參數**：`eta = 0.55`、`min_turn_strength = 0.0`（⇒ 僅 `consistency ≥ 0.55` 綁定；此為**寬鬆**判準，只看方向不看旋轉幅度，連此都不過則負結果更硬）。
- **鎖定窗**：`burn_in = 1000`、`tail = 1000`、`rounds = 3000`、`stage3_method="turning"`、`phase_smoothing=1`、`strategies=("aggressive","defensive","balanced")`。（與 Phase 1 T follow-up 一致，見 [outputs/t_series_followup_summary.tsv](../../../outputs/t_series_followup_summary.tsv) 之 rounds/burn_in/tail 欄。）
- **相空間輸入**：策略**權重序列 `w_*`**（非比例 `p_*`），以利有限族群抽樣下的迴歸穩定（cycle_metrics.py:421–423 註）。
- 任一參數的最終綁定來源為 confirmatory 執行的 runner config（隨 `phase2_cell_selection.json` 一併提交）；本節數值若與該 config 衝突，以本 pre-reg 為準並記為偏差。

---

## 6. 樣本（confirmatory seeds）

- **Arm A（mean-field 正控制）**：`g1_mean_field`，確定性，**不擴充種子**；以既有 delta∈{0,.003,.006,.010,.015}×{45,47,49}（15 runs，全 consistency=1.0、L3）作再現性呈現。
- **Arm B（4 sampled 處理 cell）**：每 cell 種子由 **3 → 10**（新增 7/cell）。
  - 種子：既有 `45/47/49` + 新增 `50,51,52,53,54,55,56`（連續整數，避免挑種子；同種子集亦用於各 cell 的配對 control）。
  - 總 Arm B runs：4 × 10 = **40**（12 既有 + 28 新增）；含配對 control 另 40。
- **算力預算**：新增 ~28 sampled runs × 3000 rounds（+ control），與 Phase 1（≈120 runs）同量級，預估 < 2 機時。

---

## 7. 停止規則與容忍

- **停止**：4 個 confirmatory cell 各滿 10 合格 seeds（共 40 runs）即停止；**不做期中偷看 p 值**，僅監控完成計數。
- **L3-seed 容忍（事前鎖定）**：以 0/10 為負結果預期；允許**至多 1/10** 為「分布尾端孤點」（與 paper_draft_v1 單例性質一致），仍維持負結論但須在報告標注該 seed。**≥ 2/10** ⇒ H0-neg 於該 cell 被推翻，須降級主張、回報並啟動 root-cause。
- **合格 run 條件**：完成全 `rounds`、judge 鏈無 NaN、provenance 完整。中途崩潰之 run 以同種子重跑一次，仍失敗則記入 Deviations。

---

## 8. 分析執行（鎖定）

- **判定**：所有 run 經 §5 鎖定鏈分類，輸出 per-seed `cycle_level` / `stage3_score` / `turn_strength` / 機制診斷至 `phase2_combined.tsv`。
- **confirmatory 報告**：H0-neg（L3-seed 率表）、H1-chance（每 cell consistency vs 0.50 之 t 檢定 + CI + d）、H2-threshold-robust（eta sweep 表 + consistency 直方圖）。
- **正向對照**：引用 paper_draft_v1 之 L2 30/30 結果（不重跑），呈現「L2 能 / L3 不能」邊界。
- **產出**：`reports/experiments/l3_bottleneck/phase2_confirmatory_analysis.json` 與報告 markdown。

---

## 9. 預期結果與論文映射

| confirmatory 結果 | 對論文主張 |
|---|---|
| Arm A mean-field 維持 L3/1.0 **且** Arm B 4 cell 全 0–1/10 L3 + consistency CI 涵蓋 0.50 + eta sweep 全 0 | **因果主張成立**：旋轉結構於連續極限存在、被離散抽樣層摧毀；橫跨 B4/B5/T/C1 四機制；C2-uniform 退化為方法學警示；paper_draft_v1 L2 為正向邊界 |
| Arm A 未達 L3/1.0 | 正控制失效 → generator/判定鏈異常，**先排查再解讀 Arm B**（不得逕報負結果） |
| 任一 Arm B cell ≥ 2/10 L3 | 負結果**降級**：該機制存在可達 L3 的窗，改寫為「條件性可達」並 root-cause（哪一層均化被繞過） |

---

## 10. 偏離記錄（Deviations）

> confirmatory 執行/分析過程中任何與本文件不符之處，於此逐條記錄（事件、原因、影響）。預設為空。
