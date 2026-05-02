# Track B v2.1 長週期修復實驗藍圖（Program Blueprint v1.1）

版本：v1.1（含 2026-04-22 Stage-1 Tier 治理增補）  
日期：2026-04-19  
適用範圍：Personality Dungeon / Track B v2.1（B3 impact spreading 定向修復主線）

## 1. 問題定義與立案理由

### 1.1 核心痛點

目前 B3 已確認以下現象反覆出現：

- 目標 seed 可修復（例如 47/79/86 可達 new_L1=0）
- 但 full gate 會產生新破壞（L1/new_L1 反向升高）
- 即「局部命中、全域拒收」

這代表問題不是單點調參不足，而是搜尋目標與全域風險不一致。

### 1.2 為何需要長週期專案化

單輪實驗可找到可行點，但無法穩定產出可鎖定參數。要達成 new_L1=0 且維持全域穩健，必須同時具備：

1. 守門式 objective（把全域風險內生化）
2. 穩定的兩階段評估管線（避免高成本與過擬合）
3. 一致的 gate 決策與 artifact 治理
4. 固定節奏的回歸與停損機制

本文件將上述內容轉為可執行專案計畫。

---

## 2. 成功標準（Program Success Criteria）

### 2.1 最終成功標準（Hard Gate）

候選參數必須同時滿足：

- gate60（42..101）：L1 <= 3
- gate60（42..101）：Healthy >= 42
- gate60（42..101）：new_L1 = 0
- fairness_fail_count = 0
- gate.invariant_overall_pass = true
- pre-A3 / post-A3：overall_pass = true

### 2.2 穩健成功標準（Robustness Gate）

在 42..101 通過後，至少再通過一個 block 重複 gate（例如 102..161），且仍維持 new_L1=0。

### 2.3 專案完成標準（Closure）

- 產生 lock artifact（最佳參數 + 指標 + provenance）
- 研發日誌完成可追溯記錄
- fail-lock 或 pass-lock 決策明確
- 可由單指令重現核心評估鏈

---

## 3. 不變條件與治理原則

### 3.1 SDD 與分層不變條件

- 先 Spec 後程式碼
- analysis 不得 import simulation
- evolution 不做 I/O
- simulation 維護 CSV 契約

### 3.2 執行環境不變條件

所有 Python 指令固定使用：

- ./venv/bin/python -m ...

### 3.3 協議鎖定

- gate 決策規則不放寬
- new_L1=0 為硬門檻
- smoke 僅作流程驗證，不作最終 go/no-go

---

## 4. 工作流設計（四條工作流並行）

| Workstream | 目標 | 主要輸出 |
|---|---|---|
| WS-A Objective Engineering | 守門式多目標與 penalty 穩定化 | NSGA-II/GA/MOPSO 腳本、objective 契約、欄位字典 |
| WS-B Evaluation Pipeline | 兩階段搜尋與重排流程 | proxy run 結果、rerank 報告、top-k 清單 |
| WS-C Compute Orchestration | 20 邏輯核心的效率化排程 | run manifest、資源監控、成本摘要 |
| WS-D Governance & Regression | 研究決策與回歸可追溯 | lock/fail-lock、日誌、決策 memo |

---

## 5. 實驗策略（Guarded + Two-Stage）

### 5.1 Stage-1（Guarded Fast Search）

目標：快速淘汰全域風險高的候選。

設定建議：

- 核心 seeds：47,79,86
- 哨兵 seeds：先用 3~5 顆高風險（例：46,58,80,101,72）
- objective 優先序：
  1) hard_new_l1_violation
  2) new_l1
  3) sentinel_floor_violation_count
  4) broken_count_selected
  5) max min_s3_selected
  6) max s3_sum_selected

輸出：Pareto top-k（建議 k=5）

### 5.2 Stage-2（Expanded Rerank）

目標：用更大 seed 集重排，過濾 Stage-1 局部優化偏差。

設定建議：

- 在 Stage-1 top-k 上，擴增哨兵至 8~12 seeds
- 保持相同 objective 結構
- 只對 rerank top-3 進 hill-climb

### 5.3 Stage-3（Full Gate Promotion）

目標：只讓真正候選進正式晉升。

流程：

1. full gate60（42..101）
2. post-A3 回歸
3. 若通過，進 block repeat（102..161）
4. 通過後才可 promotion 為 lock

### 5.4 搜尋器組合（GA/PSO + Hill Climbing 納入正式策略）

本專案將搜尋器視為可切換策略層，固定採「全域搜尋 + 局部爬山」架構。

優先順序：

1. NSGA-II + hill climbing（主線）
2. GA + hill climbing（備援 A）
3. MOPSO + hill climbing（備援 B）

切換條件：

- NSGA-II 若連續 2 輪出現 Pareto 過度塌縮（pareto_size 很小且候選重複度高），切換到 GA。
- GA 若在相同計算預算下多樣性不足（重複 key 過高），切換到 MOPSO。
- MOPSO 若收斂品質不穩定（候選波動大、重跑差異大），回到 NSGA-II 並縮小參數域。

共同規則（不因演算法改變）：

- objective 與 gate 契約完全一致
- 所有演算法最終都要經過 hill climbing 與 full gate promotion
- new_L1=0 為唯一硬門檻，不可用平均指標替代

### 5.5 優先調整參數方式（先機制，再演算法）

#### A. 機制參數優先序（B3）

P0（最優先，先調）：

1. event_impact_horizon
2. event_impact_decay
3. impact_spread_local_mass / neighbor_mass

P1（第二層）：

1. impact_spread_neighbor_hop
2. impact_spread_memory_kernel

P2（守門層）：

1. sentinel seeds 組成
2. sentinel_s3_floor
3. s3_healthy_threshold

#### B. 搜尋器超參數優先序（GA/PSO/NSGA）

NSGA-II + HC：

1. pop_size
2. mutation_rate
3. generations
4. hill_topk / hill_steps / hill_neighbors

GA + HC：

1. population_size
2. crossover_rate
3. mutation_rate
4. elitism_ratio
5. hill_topk / hill_steps / hill_neighbors

MOPSO + HC：

1. swarm_size
2. inertia_w
3. c1 / c2
4. velocity_clamp
5. mutation_injection_rate
6. hill_topk / hill_steps / hill_neighbors

調參紀律：

- 每輪只允許調整單一優先層（P0 或 P1 或 P2）
- 若 P0 尚未穩定，不進入演算法超參數細調
- 演算法超參數調整不可改寫 gate 契約

### 5.6 Stage-1 Tier 治理增補（2026-04-22）

本節為 Stage-1 的執行鎖定增補，目的為避免 sentinel 震盪、look-ahead bias 與目標維度失控。

#### 5.6.1 採納矩陣（本輪決議）

| 項目 | 決議 | 說明 |
|---|---|---|
| Tier 輸入整輪固定（run-level freeze） | 採納 | 同一輪 Stage-1 全候選共用同一組 Tier |
| failure set 來源鎖定（source lock） | 採納 | 只允許使用上一輪完整 gate60/Stage-3 產物 |
| Tier 更新順序固定 | 採納 | 固定 7 步驟順序，避免邏輯衝突 |
| Tier-2a hard cap（<=4） | 採納 | 避免 Stage-1 objective 退化為過度保守 |
| 當輪 NewL1 保留位 | 採納 | Tier-2a 至少 1 席由當輪 NewL1 的最高風險 seed 取得 |
| seed cooldown（anti-lock-in） | 採納 | 連續 3 輪進 Tier-2a，下一輪強制 Tier-2b 1 輪 |
| Promotion 節流（每輪最多 1 顆） | 採納 | 避免 Tier-1 同步膨脹 |
| Tier-1 核心 seed 保護（86,61） | 部分採納 | 僅綁定當前主線，主線改版時需重簽核心集合 |
| broke 權重自適應（wb） | 部分採納 | 只允許在回合邊界按觸發條件遞增，且需完整記錄 |
| recency 衰減權重 | 部分採納 | 作為可切換策略，預設關閉（避免立即破壞歷史可比性） |
| Tier 監控指標輸出 | 採納 | 納入每輪報告，驗證機制是否有效 |

#### 5.6.2 一致性鎖定（必要）

1. 在每一輪 t，先計算 Tier-1/Tier-2a/Tier-2b/Tier-3，接著整輪固定；不得在同輪中途改 Tier。
2. F_t 僅能由上一輪完整 gate60 或 Stage-3 結果產生；不得使用本輪 Stage-1/ hill-climb 中間評估。
3. 同一輪內所有候選必須共用同一組 Tier，避免 NSGA-II 評估基準漂移。

#### 5.6.3 Tier 更新順序（固定 7 步）

1. 更新 F_acc（最近 N 輪視窗，預設 N=3）。
2. 計算 I_t(s)。
3. 先做 Promotion 判定。
4. 若 Tier-1 已滿（cap=5），先執行 demote。
5. 再組 Tier-2a（含當輪 NewL1 保留位）。
6. 套用 cooldown。
7. 剩餘集合進 Tier-2b。

Promotion 必須先於 Tier-2a 選擇，cooldown 必須晚於 Tier-2a 組合，順序不得改動。

#### 5.6.4 分數與記憶規則

定義（預設使用未衰減頻率）：

$$
freq_{L1}^{(t)}(s)=\sum_{i=t-N+1}^{t}\mathbf{1}[s\in NewL1_i],\quad
freq_{broke}^{(t)}(s)=\sum_{i=t-N+1}^{t}\mathbf{1}[s\in Broke_i]
$$

評分函數：

$$
I_t(s)=4\cdot\mathbf{1}[s\in NewL1_t]+2\cdot freq_{L1}^{(t)}(s)+w_b\cdot freq_{broke}^{(t)}(s)
$$

其中 $w_b\in[0.5,1.0]$，預設 $w_b=0.5$。

recency 衰減（可選，預設關閉）可使用最近三輪權重 [1.0, 0.7, 0.4]，僅在專案簽核後啟用。

#### 5.6.5 Tier 組成與邊界條件

1. Tier-2a hard cap：len(Tier-2a) <= 4。
2. 當輪 NewL1 保留位：若 NewL1_t 非空，先選 argmax(I_t(s)) among NewL1_t 進 Tier-2a。
3. cooldown：seed 連續 K=3 輪在 Tier-2a，下一輪強制 Tier-2b（1 輪）。
4. Promotion 條件：同一 seed 連續 >=2 輪排名前 2 且 freq_L1 >= 2。
5. Promotion 節流：每輪最多 1 個 seed 由 Tier-2 升 Tier-1。
6. Tier-1 cap：5；若滿員需先 demote，再允許 promotion。
7. demote tie-break（固定）：argmin(I_t(s), freq_L1, entered_round)，若仍同分則取 seed id 較小者。
8. 核心 seed 保護：當前主線核心 seed=86,61，不可 demote。

#### 5.6.6 wb 自適應防護（部分採納）

允許的唯一自適應規則：

1. 僅能在回合邊界調整，不得在回合中調整。
2. 若連續兩輪滿足「new_L1 未下降」且「broke 上升」，則 wb := min(1.0, wb+0.1)。
3. 每次 wb 變更需寫入 run manifest 與 decision memo。
4. 若需要嚴格 A/B 可比性，wb 必須固定不調整。

#### 5.6.7 監控指標（每輪必輸出）

每輪輸出下列治理指標：

1. tier2a_overlap_rate
2. new_seed_injection
3. promotion_count
4. cooldown_triggered
5. tier1_churn_count

判讀建議：

1. overlap 過低：可能追噪音。
2. overlap 過高：可能過度鎖定。
3. new_seed_injection=0：NewL1 保留位規則失效。

#### 5.6.8 分層不變條件（再次鎖定）

本增補僅調整「資料如何進入優化器」，不改動：

1. objective 的 hard-first 結構。
2. 60-seed gate 與 post-A3 硬判定。
3. Stage-3 promotion 規則。

---

## 6. 計算資源規劃（20 Logical Processors）

### 6.1 核心策略

- workers 固定 20（強制）
- OMP_NUM_THREADS 固定 1（強制）
- OPENBLAS_NUM_THREADS 固定 1（強制）
- MKL_NUM_THREADS 固定 1（強制）
- Stage-1 需確保 pending evaluations >= 20，避免 CPU 閒置
- Stage-2/3 才提高 seed 覆蓋與完整評估窗

### 6.2 參數建議

- Stage-1（吃滿 20 核）：pop_size 20，generations 1~2
- Stage-2：top-k rerank + hill_topk=3，rerank 候選批次並行到 20
- Stage-3：top-2 或 top-3 並行 full gate（可並行時盡量並行）

### 6.3 滿載策略（20 邏輯處理器）

1. 單輪搜尋至少保證第一波可同時 dispatch 20 個 candidate eval。
2. 若某階段候選數 < 20，改採 block-parallel rerank（多 block 同時）補滿併發。
3. 若 Stage-3 只剩單一候選，允許非滿載，優先保證決策品質與可重現性。
4. 所有 batch 任務統一採 xargs -P 20 或等價併發控制，不做超額 oversubscribe。

### 6.4 成本控管守則

若單代 wall time 超出預期門檻，優先調整順序：

1. 降低 Stage-1 seed 覆蓋（保留核心 + 高頻哨兵）
2. 降低 generations，不先降 objective 維度
3. 僅在 Stage-2 擴展 seed
4. 若仍超時，先減少 gate seeds，再減少 hill 鄰域數

---

## 7. 里程碑與時程（建議 6 週）

### 里程碑 M1（第 1 週）：目標函數與欄位契約凍結

交付：

- Guarded objective 文件化
- CLI 參數凍結
- 指標欄位定義完成

### 里程碑 M2（第 2~3 週）：兩階段搜尋上線

交付：

- Stage-1 + Stage-2 可重跑流程
- top-k 產出與排序報告
- wall time 成本報告

### 里程碑 M3（第 4 週）：全域 gate 驗證與 block repeat

交付：

- gate60 + post-A3 報告
- 至少一個 block repeat 結果
- pass 或 fail-lock 決策

### 里程碑 M4（第 5~6 週）：鎖定/回退與封板

交付：

- lock artifact 或 fail-lock
- 研發日誌整合
- 決策 memo 與下一輪假說

---

## 8. 風險登錄與對策

| 風險 | 現象 | 對策 |
|---|---|---|
| R1 局部過擬合 | 核心 seed 好、gate 退化 | 強制 Stage-2 擴展 rerank |
| R2 計算成本爆炸 | 候選評估過慢 | 兩階段 seed 覆蓋分離 |
| R3 指標漂移 | objective 與 gate 不一致 | objective 契約凍結 + 每週審核 |
| R4 決策延遲 | 長時間無 go/no-go | 設置固定 stop condition |
| R5 文檔脫節 | 實驗與日誌不一致 | 每輪必出 summary + memo |

---

## 9. 決策閘門（Go/No-Go）

### Gate-G1（Stage-1 完成）

- 條件：可產生穩定 Pareto top-k
- 否則：縮 seed / 縮世代，重啟 Stage-1

### Gate-G2（Stage-2 完成）

- 條件：top-3 於擴展 seed 上無明顯 new_L1 回升
- 否則：調整哨兵組成後再 rerank

### Gate-G3（Stage-3 完成）

- 條件：gate60 + post-A3 全通過，new_L1=0
- 否則：輸出 fail-lock，回到 Stage-1（不允許直接 promotion）

### Gate-G4（Robustness 完成）

- 條件：至少一個額外 block 仍 new_L1=0
- 否則：標記為 conditional，禁止最終封板

---

## 10. Artifact 與命名規範

建議統一使用以下路徑族群：

- outputs/track_b_v2/b3_impact_spreading_v2/program_runs/
- outputs/track_b_v2/b3_impact_spreading_v2/program_reports/
- outputs/track_b_v2/b3_impact_spreading_v2/program_locks/

每輪至少輸出：

1. nsga2_hill_summary.json
2. all_evaluations_sorted.json
3. gate60_summary.json（top 候選）
4. post_a3_summary.json
5. decision_memo.md

---

## 11. 團隊協作建議（合理專案配合）

### 11.1 角色分工（最小配置）

- Research Owner：定義假說、批准 objective 變更
- Runtime Owner：維護 simulation / gate CLI
- Analysis Owner：維護排序、診斷與報告
- QA Owner：維護 pre/post-A3 與回歸測試

### 11.2 會議節奏

- 每日 15 分鐘：昨日結果、今日計畫、阻塞點
- 每週 1 次：Go/No-Go 與里程碑審核
- 每輪實驗後：24 小時內完成 decision memo

### 11.3 文件同步節奏

每個 milestone 完成後同步更新：

1. 研發日誌
2. lock/fail-lock
3. 本藍圖的進度區塊

---

## 12. 立即執行清單（Next 7 Days）

1. 固定 Stage-1 哨兵首版（3 核心 + 5 哨兵）
2. 跑 NSGA-II + HC 的 Stage-1 guarded search（pop=20, gen 1~2）
3. 追加 GA + HC 與 MOPSO + HC 各一輪同預算基準（A/B 對照）
4. 產生 top-k 與 CPU 使用率/成本摘要
5. 對 top-k 執行 Stage-2 擴展 rerank（盡量維持 20 併發）
6. 選 top-1~3 進 gate60 + post-A3，形成第一版 go/no-go memo

---

## 13. 與既有文件的關聯

本文件為長週期專案治理層，不替代既有實驗藍圖與協議文件：

- ../track_b_v2_1/track_b_v2_blueprint_v2_1.md
- ../phase_based_coupling_blueprint_v2.md
- ../../SDD.md
- ../../研發日誌.md

本文件用途是把「實驗方法」升級為「可持續專案節奏」，確保問題可在多輪中被收斂，而非單次偶然命中。
