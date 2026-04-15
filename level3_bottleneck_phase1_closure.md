# Level 3 Bottleneck Exploration — Phase 1 Closure

> **日期**：2026-04-08
> **涵蓋範圍**：B-series (B2–B5)、C-family (C1–C2)、T-series (含 follow-up)
> **總實驗量**：≈ 120 runs across 30+ conditions

---

## 1. 全系列比較總覽

### 1.1 系列層級彙總

| 系列 | 策略方向 | 條件數 | 總跑數 | Level 3 成功 | 最佳 uplift | 判定 |
|------|---------|--------|--------|-------------|------------|------|
| **B2** Island Deme | well-mixed + 島嶼拓樸 | 6 active | 18 | 0/18 | +0.004 | **close_b2** |
| **B3.1** Personality Strata | well-mixed + 靜態分層 | 3 active | 9 | 0/9 | +0.001 | **close_b3.1** |
| **B3.2** Phase Smoke | 相位感知分層(前導) | 2 active | 6 | 0/6 | — | weak → **close** |
| **B3.3** Phase Strata | 相位感知分層(正式) | — | — | 0 | — | **close_b3.3** |
| **B4** State-dep K | well-mixed + 振幅依賴 k | 6 active | 18 | 0/18 | +0.014 | **close_b4** |
| **B5** Tangential Drift | well-mixed + 切線催化 | 4 active | 12 | 0/12 | −0.001 | **close_b5** |
| **C1** Pairwise Imitation | local pairwise (Fermi) | 6 active | 18 | 0/18 | +0.005 | **close_c1** |
| **C2** Mini-batch Replicator | local minibatch (uniform init) | 6 active | 18 | 0/18 | +0.008 | **close_c2** |
| **T** Topology × Update 2×2 | local + random init + topology | 4 active | 12 | 0/12 | −0.006 | weak_positive |
| **T follow-up** strong k_local | local minibatch k=0.12 | 2 active | 6 | 0/6 | −0.005 | **close_t** |

**結論**：所有系列加總 **0 Level 3 seeds**，跨越 30+ 條件與 ≈120 runs。

### 1.2 核心診斷指標比較

| 指標 | B-series | C1 Pairwise | C2 Minibatch (uniform) | T Minibatch (random init) |
|------|----------|-------------|----------------------|--------------------------|
| edge_strategy_distance | N/A | **0.000** (完全收斂) | 0.000 | >0 (有空間差異) |
| cosine_vs_global | N/A | N/A | **1.000** (退化) | **0.05–0.10** (解耦) |
| player_growth_dispersion | N/A | N/A | 0.08 | **0.22–0.32** |
| batch_phase_spread | N/A | **0.000** | **0.000** | **1.54–1.63 rad** |
| spatial_autocorrelation_d1 | N/A | ~0.0 | ~0.0 | **−0.014 ~ −0.018** |
| init_weight_dispersion | 0.0 | 0.0 | 0.0 | **0.382** |
| inter_deme_phase_spread (B2) | **1.93 rad** | N/A | N/A | N/A |
| drift_contribution_ratio (B5) | **0.003–0.032** | N/A | N/A | N/A |
| k_clamped_ratio (B4) | **0.000–0.002** | N/A | N/A | N/A |
| growth_dispersion (B3) | **0.17–0.22** | N/A | N/A | N/A |
| **Level 3** | **0** | **0** | **0** | **0** |

---

## 2. 機制解讀

### 2.1 B-series：well-mixed 框架內的結構 patch 全部失敗

| 子系列 | 修正手段 | 為何失敗 |
|--------|---------|---------|
| B2 | 島嶼拓樸 + 週期遷移 | deme 間相位擴散存在 (1.93 rad) 但遷移平均化立刻抹平 |
| B3 | personality / phase-aware 分層 | 層間 cosine < 1.0 但 global averaging 仍主導 |
| B4 | 振幅依賴 selection strength | k 的調幅不足以改變 operator 本身結構 |
| B5 | 外部切線 drift | drift 被三層均化吸收，tangential alignment ≈ −0.01（與 tangent 正交） |

### 2.2 C-family：local update 的兩種極端

| 子系列 | 更新機制 | 為何失敗 |
|--------|---------|---------|
| C1 | Fermi pairwise imitation | **過度同質化**——模仿鄰居 → 邊際策略距離 = 0.000，共識比 well-mixed 更快 |
| C2 | Local minibatch replicator (uniform init) | **數學退化**——`w_i(0)` 全相同 → `x_local ≡ x_global` → `g_local = g_global` always |

### 2.3 T-series：random init 打破退化，但仍不夠

T-series 是最具資訊量的實驗：

1. **Random Dirichlet init 成功打破 C2 退化**：cosine_vs_global 從 1.000 降至 0.05–0.10
2. **Local phase domain 確實形成**：batch_phase_spread ≈ 1.55 rad，player_growth_dispersion > 0.2
3. **Spatial structure 存在**：spatial_autocorrelation_d1 微負（鄰居反相關，暗示 domain boundary）
4. **但 domain 壽命不足**：所有 seed 最終收斂回 Level 2 plateau，stage3_uplift 反而為負
5. **加強 k_local（0.08→0.12）無效**：uplift 反而更負，顯示不是 selection pressure 不足的問題
6. **Topology 無顯著差異**：lattice4 與 small-world 表現幾乎相同

**核心洞見**：局部 phase domain 可以被激活（這是正面的），但 sampled discrete synchronous update 在每一步都系統性地稀釋/打散這些局部方向性信號，domain 的相干壽命無法累積成全局可見的 Level 3 rotation。

---

## 3. 統一物理圖像

Level 2 plateau 的「三層均化」耗散機制：

```
Layer 1: Sampling noise     — 離散策略選擇 ∝ w，引入高頻噪聲
Layer 2: Popularity averaging — x = mean(sampled strategies)，抹平個體差異
Layer 3: Replicator update   — w(t+1) ∝ w(t) · f(x)，同步更新 = 全場同相位推進
```

- **B-series** 嘗試在 Layer 3 上方做 patch（改 k、加 drift、加分層）→ 被 Layer 1+2 吸收
- **C1** 移除 Layer 3 換成 pairwise → 變成更強的 Layer 2（共識機制）
- **C2** 移除 Layer 3 換成 local replicator → 均勻 init 下退化回 global（Layer 3 等價）
- **T-series** 同時攻擊 Layer 2（topology）和初始條件（random init）→ 成功製造 local domain 但 Layer 1 的 sampling 抖動仍在每步稀釋相干性

**一句話結論**：sampled discrete synchronous update 的三層均化效應對 rotational / directional component 的耗散極為頑強，任何已測試的「局部修正」（topology、初始條件、selection strength、drift、分層、island）均不足以穩定打開 Level 3 basin。

---

## 4. 後續方向建議

基於 Phase 1 的完整 negative result，突破 Level 2 需要更根本性的改變：

| 優先級 | 方向 | 為何可能有效 | 需要的改動 |
|--------|------|------------|-----------|
| **1** | B1 切線投影 Replicator | 全新 operator，直接在連續流場中注入 rotational component | 新 evolution module |
| **2** | Asynchronous / event-driven update | 打破同步 averaging，讓 leading players 帶動 rotation | 新 simulation loop |
| **3** | 持續 per-round mutation / noise | T-series 顯示一次性 random init 不夠，需要持續注入異質性 | mutation_rate 已實裝 |
| **4** | 非權重平均 strategy sampling | 改變 Layer 1 本身（如 Thompson sampling 或 boltzmann selection） | core/strategy.py 重寫 |

---

## 5. 實驗產物索引

| 檔案 | 內容 |
|------|------|
| `outputs/b2_island_deme_short_scout_decision.md` | B2 closure |
| `outputs/b3_personality_strata_short_scout_decision.md` | B3.1 closure |
| `outputs/b3_phase_strata_short_scout_decision.md` | B3.3 closure |
| `outputs/b4_state_k_short_scout_decision.md` | B4 closure |
| `outputs/b5_tangential_drift_short_scout_decision.md` | B5 closure |
| `outputs/c1_local_pairwise_short_scout_decision.md` | C1 closure |
| `outputs/c2_local_minibatch_short_scout_decision.md` | C2 closure |
| `outputs/t_series_short_scout_decision.md` | T-series scout + closure |
| `outputs/t_series_followup_decision.md` | T follow-up closure (hard-stop) |
| `outputs/t_series_short_scout_summary.tsv` | T-series per-seed data |
| `outputs/t_series_short_scout_combined.tsv` | T-series aggregated data |
| `outputs/t_series_followup_summary.tsv` | Follow-up per-seed data |
| `outputs/t_series_followup_combined.tsv` | Follow-up aggregated data |

---

*Phase 1 Closure 標誌著 B/C/T 三大結構性探索階段的結束。*
*此 negative result 的品質與完整度足以作為未來 paper Discussion 的基礎。*
