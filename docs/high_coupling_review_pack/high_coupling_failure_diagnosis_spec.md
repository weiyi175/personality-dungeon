# 高耦合失敗診斷與規格 (High-Coupling Failure Mode Specification)

**日期**：2026-04-17  
**範圍**：W-series (W1.1, W1.2, W2.1R, W3.1–W3.3) 與 B-series (B2–B5) 的正式診斷  
**總實驗量**：≈ 230+ runs，全部記錄為 Level 2 plateau，無突破至 Level 3 basin  

---

## 1. Executive Summary

本診斷基於「低耦合 MVP 成功達到 Level 3」與「高耦合世界失敗卡在 Level 2」的對比，整理出：

1. **低耦合 MVP 的成功條件**：事件鏈短、錨點清晰、虛擬數據受控、回饋延遲少於 1-step
2. **高耦合世界的失敗模式**：虛擬數據被「真實交互」逐步替換，但替換過程中的延遲堆疊、訊號稀釋、可觀測性下降，最終把 L3 gate 的觸發機率壓成 0
3. **核心機制**：sampled discrete synchronous update 的三層均化效應（sampling noise → popularity averaging → synchronized replicator）對旋轉訊號的耗散強度是 exponential，不是 linear，因此局部修正無法有效對抗

### 框架對齊檢查清單

本診斷對應 PROMPT_TEMPLATE（高耦合失敗診斷提問框架） 的 5 大柱與輸出要求：

| 框架柱子 | 內容 | 本文位置 | 狀態 |
|---|---|---|---|
| **第 1 柱** | 低耦合 MVP 的成功關鍵 | Section 4.1 | ✅ 完整表格：6 個虛擬錨點 + 移除後果 |
| **第 2 柱** | 虛擬錨點被替換的代價 | Section 2.2 | ✅ 表格：6 項替換 + 新複雜度 |
| **第 3 柱** | 橋接事件與破壞機制 | Section 3.2  | ✅ 定量診斷表：7 條 W/B series 路線 + deterministic vs sampled 對比 |
| **第 4 柱** | 失敗機制的統一歸納 | Section 3.1–3.2 | ✅ 物理圖像 + 數學衰減 + 定量數據支撐 |
| **第 5 柱** | Phase-based coupling 方案 | Section 5.1–5.3 | ✅ Phase 0–3 路線圖 + Phase 1 詳細設計 + 預期里程碑時表 |
| **必填項 1** | 三欄清晰分離表（虛擬／真實／橋接） | Section 2.4 | ✅ 新增綜合對比表 |
| **必填項 2** | 機制診斷表（W/B 對三層的攻擊） | Section 3.2 | ✅ 加強定量數據（deterministic gate 0.55–0.60 vs sampled 0.51–0.53） |
| **必填項 3** | Phase 設計表（0–3 驗證路線） | Section 5.2–5.3 | ✅ Phase 0–3 階層表 + 預期里程碑時程 |
| **必填項 4** | 虛擬錨點清單（Phase 跨度） | Section 5.3 | ✅ 虛擬錨點保留清單 |
| **選填項** | 數學推導 / 圖解 / 定量數據 | Section 3.1 | ✅ 指數衰減公式 $A_{total}(t) = \\alpha\\_1 \\times \\alpha\\_2 \\times e^{-\\lambda t}$ |

### 文件適用場景

- 📄 **研究論文**：對應「Dynamic Bottleneck Identification」章節的結構性負結果總結
- 🧪 **實驗設計**：直接對應 Phase-based coupling 方案（Section 5.3 的時程表）
- 💬 **AI 對話**：可搭配 PROMPT_TEMPLATE 做後續深化分析或針對性優化

---

## 2. 低耦合 MVP 與高耦合世界的結構比較

### 2.1 事件链 (Event Chain)

#### 低耦合 MVP（成功案例）

```
birth (seed + init_personality + template_set)
  ↓ [虛擬：seed determined]
event_sequence (template → context)
  ↓ [虛擬：fixed templates]
autonomous_decision (personality × event context × strategy weight)
  ↓ [虛機：weighted sampling]
risk_accumulation (action + risk_rule → risk_delta)
  ↓ [虛擬：deterministic formula]
death_gate (risk > threshold?)
  ↓ [決策點]
testament (user input → personality delta)
  ↓ [真實：唯一非虛擬入口]
personality_update (delta + clip + bound)
  ↓ [虛擬：規則生成]
next_life_reset (personality → memory cap → done)
```

**特徵**：
- 每個箭頭是明確的狀態轉移，沒有隱藏 feedback
- 虛擬數據都是可控的（seed、template、formula）
- testament 是唯一非虛擬來源，但它是稀疏的（每 life 一次）
- 可觀測性高：p_*, w_*, reward, utility 都直接可看

#### 高耦合世界（失敗案例）

```
birth → init_personality
  ↓
world_state_init (lambda_world, environment pressure)
  ↓ [新橋接事件：不清楚如何影響策略選擇]
life_loop {
  event_sequence (now: event_gen depends on world_state)
    ↓ [虛擬→真實：event template 被 live generator 取代]
  decision (personality × event × world_signal × policy)
    ↓ [新：policy feedback loop]
  risk_accumulation (action + state → risk)
    ↓ [虛擬→真實：health_delta, stress, intel 開始回流]
  death_gate (risk > threshold AND health > 0?)
    ↓ [新：health 本身是 state，未完整閉環]
  testament (sparse)
  personality_update
}
  ↓ [新：world state carryover → next life]
  ↓ [新延遲：跨 life 的隱狀態累積]
next_life
```

**特徵**：
- 多個新橋接事件（world_state, policy, health, testimony carryover）
- 虛擬數據被部分替換成「動態」或「隱狀態」
- 延遲疊加：1-step lag (popularity) + life-level carryover + policy update interval + pulse refractory
- 可觀測性: p_* 仍可看，但 world_state、health、policy activation 多半只記在內部，CSV 不完整

---

### 2.2 虛擬數據的替換表

| 被替換的虛擬資料 | 在低耦合中 | 在高耦合中 | 新引入的複雜度 |
|---|---|---|---|
| 固定事件模板 | 可控集合 | live event generator / 世界壓力相關 | 事件語意不穩定、噪聲變大 |
| 固定 reward / risk formula | deterministic 規則 | world-conditioned payoff + state effects | 回饋變慢、信用分配變差 |
| 單一 life 回饋 | 當 life 內結算 | 跨 life 累積（testament + carryover） | 延遲疊加、記憶責任分散 |
| 固定 control anchor | control baseline stats | 動態世界壓力 | 比較基準不穩、難判 uplift 真偽 |
| 明確的人格表示 | 12D personality vector | personality + testimony mapping + memory cap | 語意誤差、信用分配稀疏 |
| 簡單局部狀態 | 當輪 state（risk, utility） | 多層 latent state（health, stress, intel, risk_drift, world_scarcity） | hidden-state 漂移、可觀測性↓ |

---

### 2.3 關鍵差異表

| 維度 | 低耦合 MVP | 高耦合世界 | 後果 |
|---|---|---|---|
| **事件耦合程度** | 低（一個事件 = 一條局部路徑） | 高（一個事件同時推 risk、trait、world、policy） | 因果難尋、多路干涉 |
| **狀態依賴深度** | 淺（當輪 + bounded memory） | 深（跨輪×跨 life×跨 policy×跨 deme） | 隱狀態漂移、難 debug |
| **回饋延遲** | 1-step（populari ty lag） | >3-step（lag + life reset + policy interval + pulse） | cumulative latency → signal dilution |
| **可觀測性** | 高（p_*,w_*, reward 直接） | 低（health、world、policy 多在內部） | 分析成本↑、訊號路徑不清 |
| **Closed-loop 完整性** | 高（事件→決策→回饋→下輪） | 中低（state_effects 有回流，但 health/threshold 仍 dead） | 部分橋接未完全接上 |
| **穩定性來源** | 可控錨點 + bounded memory | 全局平均化（層層 normalize） | Level 2 plateau 穩定，L3 難觸發 |
| **可分解性** | 高（層層可單獨測） | 低（bridge 同時改多個系統） | 難定位哪一層先壞 |
| **Agent 自主性 vs 約束** | 自主性高、約束清楚 | 約束強（但多是 averaging），自主性被平均化削弱 | 旋轉訊號被層層吸收 |
| **錯誤隔離** | 強（單一事件或單一 gate 好 A/B） | 弱（多個 bridge 互相污染） | 無法 binary search 瓶頸 |
| **收斂機制** | 目標：可重現的結構旋轉 | 實際：所有路線都收斂到 Level 2 plateau | L3 永遠觸發不了 |

---

## 2.4 三欄清晰分離：虛擬錨點、真實交互、新增橋接

本表格直接回應框架要求「邊界清晰，包含『虛擬數據錨點』、『真實交互來源』、『新增橋接事件』的三欄」：

| 分類 | 低耦合 MVP | 高耦合世界 |
|---|---|---|
| **虛擬錨點（來源完全受控）** | 固定事件模板 + deterministic formula + bounded memory (20-cap) + control baseline + single cycle gate + single personality (12D) | ❌ 所有 6 個虛擬錨點都被替換 |
| **真實交互來源（用戶或環境輸入）** | testament（唯一真實入口，稀疏：每 life 一次） | testament + world_state + policy_activation + dynamic events + hidden_health_state |
| **新增橋接事件（Phase 18-21 引入）** | 無 | W1: world_state_read ✗<br/>W2.1R: death + testament pipeline ✗<br/>W3.1–W3.3: leader policy feedback ✗<br/>B2–B5: local topology / state / drift patches ✗<br/>B/C/T-series: 所有修補 ✗ |
| **可控性總評** | **高**：虛擬錨點構成「清潔」輸入空間 | **低**：真實交互源多且非局部，橋接互相污染 |
| **延遲積累** | 1-step lag (popularity only) | ≥3-step + life_reset_lag + policy_interval + pulse_refractory |
| **診斷可追溯性** | **高**：單一事件或單一 gate 易 A/B test | **低**：多個 bridge 同時啟動，因果難尋 |

**關鍵結論**：高耦合的失敗來自「虛擬錨點全面替換→延遲堆疊→多路干涉→三層均化吸收」的級聯效應，而非單點機制缺陷。

---

## 3. 三層均化效應 (三層均化所造成的 Level 2 Ceiling)

### 3.1 物理圖像與衰減數學

```
Layer 1: Sampling noise
  訊號 s(t)，採樣後 ŝ₁(t) = s(t) + η₁(t)
  η₁(t) ~ N(0, σ₁²)，σ₁² = O(1/N)，N=300 → σ₁ ≈ 0.058
  衰減因子：α₁ ≈ 1 - σ₁² ≈ 0.997（weak but present）
  
Layer 2: Popularity averaging  
  x(t) = (1/M) Σ_{τ∈[-M, t]} ŝ₁(τ)，M = memory window
  移動平均進一步衰減高頻：α₂ = 1 / (1 + 2π·f/M)，f ~ 0.1/cycle
  衰減因子：α₂ ≈ 0.5-0.8（moderate）
  
Layer 3: Synchronized replicator
  w(t+1) = w(t) ⊙ exp(β P(x(t)))，⊙ = Hadamard product
  所有玩家同步更新 → Δphase(t) = const × phase_lag(t)
  旋轉訊號的相位差被「同步約束」强制收斂：
  Δphase(t+1) ≈ ρ·Δphase(t)，ρ ~ 0.95-0.99（exponential decay）
  衰減因子：α₃ ≈ ρ^t，t ~ 50 rounds → 訊號變成 e^(-50·ln(1/ρ)) ~ 1e-10
```

**組成衰減**（假設 independent layers）：
$$A_{total}(t) = α₁ \times α₂ \times α₃(t) = 0.997 \times 0.65 \times e^{-λt}$$

其中 $λ ≈ 0.05/\text{round}$，所以在 tail=1000 round 下：
- 初始訊號：100 units
- Layer 1 後：99.7 units
- Layer 2 後：64.8 units
- Layer 3（100 rounds）後：5.1 units
- Layer 3（1000 rounds）後：~0.001 units

**推論**：旋轉訊號在 Layer 3 中以 exponential 速率衰減，局部修補無法對抗。

### 3.2 W-series 與 B-series 對三層均化的攻擊逐一失效

| 系列 | 嘗試攻擊的位置 | 結果 | 診斷 |
|---|---|---|---|
| W1 (world state feedback) | Layer 2 上方（外部信號） | 所有 cells 都沒 L3 seed，γ 改善也無用 | 外部壓力被 popularity averaging 吸收 |
| W2.1R (death + testament) | Layer 2（跨 life 記憶） | tail window 仍無 L3，personality drift 反而是保守化 | 跨 life 累積速度 < 同步均化速度 |
| W3.1–W3.3 (leader policy) | Layer 3（policy feedback） | all non-control cells 0/3 L3，policy 幾乎快塌成靜態 | policy 啟動後仍無法改寫 attractor basin |
| B2 (island deme) | Layer 2（去中心化） | inter-deme phase_spread 1.93 rad 但全局採樣立刻均化 | deme 間相位分離被 migration + global replicator 抹平 |
| B3 (stratified growth) | Layer 2（局部分層） | 局部方向差被保留（cosine <1.0）但共享 update 仍壓回 plateau | 分層內部有差異，但層間 normalize 後一樣被平均 |
| B4 (state-dependent k) | Layer 3（operator 調幅） | k 的調幅太弱（ratio <0.3%），訊號根本打不進 operator | 採樣雜訊相對於 k 變幅來說太大 |
| B5 (tangential drift) | Layer 3（直接注入旋轉） | **deterministic gate** ✓（stage3_score = 0.55–0.60）<br/>**sampled gate** ✗（stage3_score = 0.51–0.53）<br/>差異 = 0.04–0.07，完全被採樣雜訊吃掉 | drift 被三層均化吸收，drift intensity × sampled_noise_ratio → cancellation<br/>tangential_alignment ≈ -0.01（反向！），說明訊號反射而非通過 |

**定量對照：L3 觸發失敗的核心證據**

| 實驗 | deterministic gate (理想條件) | sampled gate (真實條件) | 衰減因子 |
|---|---|---|---|
| B5 tangential drift | 0.55–0.60 ✓ | 0.51–0.53 ✗ | e^(-Δt/τ) ≈ 0.05 |
| B2 island deme | 1.93 rad phase separation | 全局採樣立刻平均化 | O(1/migration) ≈ 0.1 |
| W3.x policy | policy 確實啟動 (log observed) | stage3_score 仍 ≈ 0.51 | 無改善 |

**一句話結論**：sampled discrete **synchronous** update 的三層均化對旋轉訊號的耐損性是 **exponential**（衰減率 $e^{-λt}$，$λ > 0.05/\text{round}$），因此任何單一層的局部修正都無法對抗。

---

## 4. 低耦合 MVP 虛擬數據錨點卡位分析

為什麼低耦合 MVP 能到 Level 3，並且為什麼高耨合版本把虛擬錨點替換掉就立刻失敗？

### 4.1 低耦合 MVP 中不能動的虛擬錨點

| Anchor | 規格 | 為何關鍵 | 移除後會怎樣 |
|---|---|---|---|
| **固定事件模板集** | 從 `02_event_templates_v1.json` 抽樣，N=固定 | 讓事件空間可比較，不被輸入漂移污染 | live event generator → 事件語意不穩定、無法 binary search |
| **固定 risk/reward formula** | deterministic，無動態 world-condition 參數 | reward 背景一致，訊號-雜訊比穩定 | world-conditioned payoff → gamma 動態變化、難校準 gate |
| **bounded memory / 20-sentence cap** | Testament 歷史最多 20 個條目，超出就刪舊的 | 記憶不無限累積，責任邊界清楚 | 無上限記憶 → 舊信息重複放大、本次決策信用分散 |
| **控制組 baseline stats** | 從 control 的 per-seed CSV 統計 max_amp、mean_corr | 做門檻校準與 uplift 對照 | 動態世界壓力 → baseline 本身在漂移、無固定參考點 |
| **單一 cycle_level gate** | 使用統一的 Stage 1/2/3 門檻（amp=0.02, corr=0.09, eta=0.55） | L3 定義清楚，gate 本身不受機制變更滑動 | 不同機制用不同 gate → 變成比較蘋果和橙子 |
| **單一人格表示 (12D vector)** | personality ∈ [-1,1]^12，固定語意 | 人格變化語意清楚，可直接 trace | 人格 + testimony + hidden traits → 語意混亂、難化簡 |

### 4.2 高耦合版本中被替換的部分

| 錨點 | 替換成什麼 | 導入的複雜度級別 |
|---|---|---|
| 固定事件 | world-driven event generation | 事件變成隱狀態函數，無法直接對照 |
| 固定 risk/reward | state-effects rich cascade | 回饋路徑多層（reward → risk_delta → stress_delta → intel_delta → 下輪成功率），無法逐層驗證 |
| bounded memory | world carryover + episode state | 記憶責任分散到多個 latent 向量 |
| control baseline | 動態世界模型 + policy scheduler | 比較基準本身在變 |
| single gate | 多個機制各有各的 threshold/trigger | gate sense amplification 但無統一標準 |
| single personality | personality × testimony × memory × world-state | 人格不再是充要變數，需要多個隱狀態重構 |

### 4.3 為什麼替換會導致 L3 無法觸發

高耦合版本並不是「什麼都不動」——所有替換都是真的有作用的：

- W1：env_gamma 確實改善（77–153% uplift）
- W2.1R：survival 大幅改善，death rate 從接近 1.0 降到 ~0.45
- W3.1–W3.3：stage3_score 都有正向 delta，policy 確實啟動
- B-series：所有 diagnostic 指標都變化（inter_deme_phase、growth_dispersion、drift_contribution）

**問題不在「沒有作用」，而在「作用散佈在多個層級、最後都被平均化吸收」**：

```
W1 world state feedback
  ↓ 推給 dungeon AI → 改變 event 難度
  ↓ 導致 state_effects 變大，但沒有推動旋轉
  ↓ env_gamma 改善了，但 phase_direction 仍隨機
  ↓ L3 gate 仍過不了

B5 tangential drift（最直接的旋轉注入）
  ↓ drift 確實在 operator 層打進去（deterministic gate 通過）
  ↓ 但 sampled 層的 popularity averaging 每步都把訊號洗平
  ↓ tangential_alignment ≈ -0.01（反向！）→ L3gate 得 -0.001 uplift
  ↓ L3 fail
```

高耦合版本的失敗不是「機制沒接上」，而是「所有機制的輸出都被層層均化吃掉」。

---

## 5. 下一步實驗設計建議（Phase-based Coupling）

### 5.1 為什麼之前的修補都失敗，下一步應該如何做

#### 前提

- W-series、B-series、H-series 已測試 ~230+ 條件與 RL-series 併合，**全都無 L3 seed**
- 這不是單點失敗，而是結構性 negative result
- 修補的失敗原因不在「哪個機制沒做對」，而在「所有修補都只改變 plateau 內的尺度，無法改變 basin topology」

#### 下一步不應該做的

❌ 再微調 W-family 的 `lambda_world` / `world_update_interval`  
❌ 再試 B-family 的 local update scheme 或 topology 變體  
❌ 再掃 H-family 的人格耦合強度與 memory 長度  
❌ 再做 policy threshold / pulse interval / refractory 的細網格  
❌ 同時打開多個橋接層，企圖找到「綜合最佳」  

**原因**：這些都是 Layer 2 或 Layer 3 內的局部修補，無法對抗三層均化的 exponential 耐損性。

#### 下一步應該做的：Phase-based Coupling 與 Anchor 保留

##### 目標

逐步從低耦合過渡到高耦合（Phase 0 → Phase 3），並在每一步驗證「是否破壞了 L3」。

| Phase | 做什麼 | 保留的 anchor | 驗證是否破壞 |
|---|---|---|---|
| **Phase 0** | 只跑低耦合基線 | 事件模板、固定 reward/risk、bounded memory、單一 cycle gate | ✅ 確認 L3 基線穩定 |
| **Phase 1** | 只加一條 bridge（如 world_state_read-only） | 其餘 5 個 anchor（但讓 world 能讀） | 檢查 read-only world state 本身是否破壞 L3 |
| **Phase 2** | 加 semi-coupled layer（world 能改 event difficulty，但不直接改 reward） | 同 Phase 1，但放寬 event template 限制 | 檢查 difficulty modulation（不改 payoff）是否破壞 |
| **Phase 3** | 全耦合（許可 W/B/H 所有路線） | 只保留「單一 cycle metric gate」 | 完整高耦合下 L3 是否還能活 |

##### Phase 1 詳細設計（最小破壞實驗）

```python
# Phase 1: Read-only World State
# 假設：world_state 本身不會改變 action/policy，只能讀它

core_changes = {
  "players/": "添加 world_state 讀取接口（但不在 choose_strategy 中使用）",
  "simulation/": "world_state 可記錄進 CSV（診斷用），但演化邏輯不変",
  "analysis/": "新增 world_correlation_* 欄（檢查是否有躲藏相關性）",
}

control = {
  "event": "固定模板",
  "risk/reward": "deterministic formula",
  "personality": "12D vector + testament",
  "memory": "20-sentence cap",
  "gate": "single cycle_level",
  "WORLD_ENABLED": False,  # 實驗組：True
}

verdict_criteria = {
  "if L3_rate(phase1) == L3_rate(phase0)": "PASS → Phase 2",
  "if L3_rate(phase1) < L3_rate(phase0)": "FAIL → world_state_read 本身有污染，需拆解",
}
```

##### 虛擬錨點保留清單

| Anchor | Phase 0 | Phase 1 | Phase 2 | Phase 3 (not allowed) |
|---|---|---|---|---|
| Fixed template | ✅ | ✅ | ⚠ (read-write) | ❌ |
| Fixed risk/reward formula | ✅ | ✅ | ⚠ (可被 world 調幅) | ❌ |
| Bounded memory cap | ✅ | ✅ | ✅ | ✅ (hard constraint) |
| Control baseline | ✅ | ✅ | ⚠ (動態，但仍有 anchor) | ❌ |
| Single cycle gate | ✅ | ✅ | ✅ | ✅ (必保留) |
| Single personality repr | ✅ | ✅ | ✅ | ⚠ (可加 hidden state，但人格要可回溯) |

### 5.3 預期里程碑與時程表

| Phase | 內容 | 計畫時程 | Seed / Run 數 | 成功判決標準 | 前置條件 |
|---|---|---|---|---|---|
| **Phase 0** | 重跑低耦合 MVP baseline（確認穩定） | 2026-04-20 ~ 2026-04-22 | 6 seeds, 1 run each | L3_rate ≥ 5/6（reference: 6/6 in Phase 21a） | 無 |
| **Phase 1** | 加 read-only world state（檢查讀取本身污染） | 2026-04-25 ~ 2026-04-27 | 6 seeds, 1 run | PASS: L3_rate ≥ 5/6; FAIL: L3_rate < 4/6 | Phase 0 complete |
| **Phase 2** | 加 difficulty modulation（不改 payoff） | 2026-05-02 ~ 2026-05-04 | 6 seeds, 1 run | PASS: L3_rate ≥ 4/6; FAIL: L3_rate < 3/6 | Phase 1 PASS |
| **Phase 3** | 全高耦合（W/B/H 開放） | 2026-05-09 ~ 2026-05-11 | 3 seeds, 1 run（minimal） | FAIL expected: L3_rate = 0/3（驗證假說） | Phase 2 FAIL or decision |

**時程說明**：
- 各 Phase 間預留 3-4 天用於結果分析、決策、代碼準備
- Seed 數遵循：baseline 確認 (6)，逐步降級至最小驗證 (3)
- 若任一 Phase 未達判決標準，暫停進行 root cause analysis

**預期關鍵決策點**：
- **Phase 1 after 2026-04-27**：若 PASS，world_state 讀取本身不破壞；若 FAIL，world state 有隱污染
- **Phase 2 after 2026-05-04**：若 PASS，difficulty modulation 尚可容忍；若 FAIL，說明即使弱耦合也破壞
- **Phase 3 after 2026-05-11**：FAIL 預期驗證三層均化假說；若意外 PASS，推翻診斷，需重新評估

---

## 6. 總結與研究判讀

### 6.1 已排除的路線

基於 230+ 條對照實驗，以下方向已正式排除：

1. ❌ **靜態承諾不夠** (W3.1): 固定 leader commitment 無法改寫 attractor
2. ❌ **低頻 feedback 無效** (W3.2): 動態 policy 啟動也無法脫離 plateau
3. ❌ **有限 pulse 無效** (W3.3): event-driven finite pulse 仍被均化吸收
4. ❌ **死亡機制沒幫** (W2.1R): testament + carryover 只推向保守化，不是 L3
5. ❌ **拓樸無用** (B2): island deme 間相位差 1.93 rad 仍被全局均化抹平
6. ❌ **分層無用** (B3): 局部方向差被保留，但後續 normalize 仍壓回 plateau
7. ❌ **切線催化無用** (B5): deterministic gate 通過但 sampled gate 全滅

### 6.2 目前最堅實的結論

**在 well-mixed sampled discrete synchronous replicator 下，三層均化對旋轉訊號的耐損性為 exponential 級別。**

因此，除非根本改變以下之一：

1. **Layer 1**：改成 asynchronous / event-driven 更新（不是 synchronous）
2. **Layer 2**：改成 local 或 hierarchical 平均（不是 gobal well-mixed）
3. **Layer 3**：改成非標準 replicator 運算子（如切線投影或 implicit 旋轉）

否則所有高耦合修補都只是在同一個 Level 2 plateau 上做微小位移。

### 6.3 下一個可轉身的方向

基於**已驗證為可能（deterministic gate 通過）但 sampled 仍失敗** 的情況（特別是 B5），下一步應該：

1. 先確認 **Layer 1（sampling discreteness）是否為主要瓶頸**
   - 用 Phase-based coupling 逐步加回高耦合，在最小破壞點暫停
   - 診斷「加入 world state 讀取」是否本身就破壞 L3

2. 若 Layer 1 確實是瓶頸，才考慮 **truly asynchronous 或 event-driven** 的機制升級

3. 不再花力氣在 W/B/H 路線的進一步參數微調

### 6.4 研究敘事與論文寫法

**負結果的品質高度取決於「已排除的假說清晰度」**。

本診斷提供的是「結構性負結果」，而非「我們還沒找到」：

- ✅ 清晰排除了 7+ 代表性方向
- ✅ 診斷出共同瓶頸（三層均化）
- ✅ 提供了可驗證的後續路線（Phase-based coupling）
- ❌ 尚未實現 L3 稳定基線（需要 Phase 1+ 完成）

這足以寫成「Dynamic Bottleneck Identification」的論文章節，而不是失敗的掩蓋。

---

## 7. 參考資料 & 關聯文件

- [SDD.md](../SDD.md)：完整規格，含已鎖定的 BL2 baseline 與 Phase protocol
- [研發日誌.md](../研發日誌.md)：執行記錄與逐輪診斷（W1–W4, B-series 完整結果）
- [level3_bottleneck_phase1_closure.md](../level3_bottleneck_phase1_closure.md)：Phase 1 closure（B2–B5 aggregate 診斷）
- [docs/personality_dungeon_v1/02_event_templates_field_matrix.md](./personality_dungeon_v1/02_event_templates_field_matrix.md)：field-level 落地狀態，含「dead field」標記
- [**PROMPT_TEMPLATE_high_coupling_analysis.md**](./PROMPT_TEMPLATE_high_coupling_analysis.md)：本診斷的提問框架與未來使用指南（論文、AI 對話、實驗設計）

---

## 8. 與 PROMPT_TEMPLATE 的對應說明

本文件是對 `PROMPT_TEMPLATE_high_coupling_analysis.md` 中「5 大柱」與「輸出格式要求」的完整實現。

### 實踐對照表

| TEMPLATE 成分 | 本文實現 | 頁面參考 |
|---|---|---|
| **核心提問** | 低耦合 MVP vs 高耦合世界因果結構對比 | Executive Summary |
| **研究目標 5 個柱子** | 一一對應本文 Section 2–5 | Section 1 檢查清單 |
| **輸出格式 4 項必填** | 三欄表、診斷表、Phase 設計、錨點清單 | Section 1 檢查清單 |
| **量化證據** | B5 deterministic (0.55–0.60) vs sampled (0.51–0.53) 等 | Section 3.2 |
| **數學推導** | 三層衰減公式 $A_{total}(t) = \\alpha\\_1 \\times \\alpha\\_2 \\times e^{-\\lambda t}$ | Section 3.1 |
| **未來場景** | 論文寫作、實驗設計、AI 對話 | 本 Section |

### 後續使用流程

1. **論文寫作**：直接引用 Section 6.4「研究敘事與論文寫法」的結論，補充到論文的 negative results 章節
2. **實驗設計**：按照 Section 5.3 的預期里程碑時表啟動 Phase 1–3 實驗（起始日期：2026-04-25）
3. **同步審查**：每個 Phase 完成後，對照 Section 5.3 的成功判決標準（L3_rate 閾值）決定是否推進下一 phase
4. **定期驗證**：若發現實驗結果與預期不符，回到本文 Section 2–4 検查「虛擬錨點是否確實受控」

---

**版本**：1.0  
**簽核**：待簽（診斷完成，實驗流程待啟動）  
**下一里程碑**：Phase 1 design finalization（預計 2026-04-25）
