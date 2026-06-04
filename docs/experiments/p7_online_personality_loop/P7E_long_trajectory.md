# P7-E 長期人格軌跡特徵化（Long-term Trajectory Characterization）

**規格版本**: P7-E v1.0  
**制定日期**: 2025-06-04  
**基於**: P7-D 結果（α* = 0.2/0.3，S2 穩定區域確認）  
**目標**: 驗證人格是否存在吸引子結構，量化長期可塑性窗口

---

## 1. 科學問題

基於 P7-A/C/D 的發現：

1. **P7-A**: 人格影響策略選擇 (6-7pp 效應)
2. **P7-C**: 人格反饋導致確定性漂移 (ΔP = α·η·(r-r̄)·g_i(s))
3. **P7-D**: α ∈ [0.2, 0.4] 產生穩定動態 (S2 分類)

**P7-E 核心問題**：
- 在 300 rounds 內，人格已漂移 0.1～0.2；在 1000 rounds 內，人格會不會：
  - (a) 繼續單調漂移直到飽和？
  - (b) 收斂至某個固定點（吸引子）？
  - (c) 進入週期軌道（環路）？
  - (d) 崩潰（如 S4 表現的不穩定性）？

**次要問題**：
- 人格何時「鎖定」（不再改變）？稱為 **策略-人格耦合窗口**
- 不同 seed 是否收斂至相同吸引子（同調性）？

---

## 2. 實驗設計

### 2.1 參數固定（Protocol Lock）

```
n_players = 4              # 保持 P7-D 規模
n_rounds = 1000            # P7-D 的 3.3 倍（300→1000）
burn_in = 100              # 10% of n_rounds (vs. P7-D 的 50)
tail_start = 100           # 從 round 100 開始記錄尾部指標
tail_window = 900          # 最後 900 rounds

personality_mode = "static"
personality_update_enabled = True
personality_learning_rate η = 0.05  # P7-D 固定值
lambda_alpha = 0.15, lambda_beta = 0.10, lambda_r = 0.20, lambda_risk = 0.20
```

### 2.2 參數掃描

**α 選擇** (基於 P7-D):

| α | 選擇理由 | 預期穩定性 |
|---|---------|-----------|
| **0.2** | P7-D 的 100% S2；最穩定 | S2 → S2 or fixed point |
| **0.3** | P7-D 的 67% S2；驗證強度 | S2 → oscillation or attractor |

**種子** (擴展):

| seed | 理由 |
|------|------|
| 42 | P7-D baseline，穩定 |
| 43 | P7-D baseline，穩定 |
| 44 | P7-D 異常點，驗證可重現性 |
| 101 | 新 seed，打破 P7-D seed 對稱 |
| 102 | 新 seed，提高統計效力 |

**全組合**:
```
2 α × 5 seeds = 10 runs
預期執行時間: 50-60 分鐘 (1000 rounds × 10)
```

### 2.3 數據記錄與分析視窗

**Per-round 記錄** (step_log):
```json
{
  "round": int,
  "p_aggressive": float,
  "p_defensive": float,
  "p_balanced": float,
  "avg_reward": float,
  "strategy_entropy": float,
  "personality_vector": dict,        // NEW: 整個 9D 向量
  "personality_norm": float,         // ||P(t) - P(0)||
  "personality_velocity": float,     // |dP/dt|
  "max_personality_component": float // max(|P_i|)
}
```

**分析視窗**:
1. **全序列** (round 0～1000): 整體漂移趨勢
2. **燃盡期** (round 0～100): 初期調整
3. **尾部** (round 100～1000): 穩態行為分析

---

## 3. 新增分析指標

### 3.1 人格吸引子檢測

**定義**: 人格在尾部 900 rounds 內是否趨向某個穩定狀態

#### 方法 A: 方差衰減指標 (Variance Decay Index, VDI)

```python
def compute_vdi(trajectory, window_size=100):
    """
    將尾部分割成 N 個 window，檢查方差是否單調衰減
    """
    windows = [trajectory[i:i+window_size] 
               for i in range(0, len(trajectory), window_size)]
    variances = [np.var(w) for w in windows]
    
    # VDI = mean(variance slope)
    # 如果 VDI < 0（負斜率），表示收斂
    slopes = np.diff(variances)
    vdi = np.mean(slopes)
    
    return vdi, variances
```

**解釋**:
- VDI < -0.01: 強收斂特徵（人格趨向固定點）
- -0.01 ≤ VDI ≤ 0.01: 穩態（無明顯漂移，可能在環路上）
- VDI > 0.01: 持續漂移或發散

#### 方法 B: 自相關時間常數 (ACT)

```python
def compute_acf_timescale(trajectory, max_lag=200):
    """
    ACF 的「相關時間」：多少 rounds 後自相關降至 e^(-1)?
    反映軌跡的「記憶長度」
    """
    acf = compute_acf(trajectory, max_lag)
    
    # 找第一個 ACF < e^(-1) ≈ 0.37 的 lag
    timescale = np.argmax(np.array(acf) < 0.37)
    
    return timescale
```

**解釋**:
- ACT < 50 rounds: 高度隨機（每 50 steps 遺忘過去）
- 50 ≤ ACT < 200: 中等相關（動態演化）
- ACT ≥ 200: 強相關（軌跡具有長記憶，可能在大尺度環路上）

### 3.2 吸引子維度估計 (Attractor Dimensionality)

```python
def estimate_attractor_dim(trajectory_9d, embedding_dim=3, delay=10):
    """
    使用時間延遲嵌入法（Time Delay Embedding）估計吸引子的拓樸維度
    """
    # 將 trajectory[t] 轉換為延遲向量
    # X[i] = [P(i), P(i+delay), P(i+2*delay), ...]
    
    embedded = np.array([
        trajectory_9d[i:i+embedding_dim*delay:delay]
        for i in range(len(trajectory_9d) - embedding_dim*delay)
    ])
    
    # 估計最近鄰距離的相關維數 (Grassberger-Procaccia)
    # d_corr = lim(epsilon->0) [ log(C(epsilon)) / log(epsilon) ]
    # 其中 C(epsilon) = 相距 < epsilon 的點對比例
    
    correlation_sum = compute_correlation_sum(embedded)
    d_est = estimate_correlation_dimension(correlation_sum)
    
    return d_est
```

**解釋**:
- d_est ≈ 0: 固定點（吸引子是一個點）
- 0 < d_est < 1: 週期軌道（環路）
- 1 < d_est < 2: 複雜動態（可能混沌，但在 RL 中罕見）

### 3.3 人格可塑性窗口 (Plasticity Window)

**定義**: 人格 **停止顯著變化** 的時刻

```python
def find_plasticity_window(step_log, threshold=0.01, window_size=50):
    """
    找出第一次出現 window_size 連續 rounds，其中 |dP/dt| < threshold 的時刻
    """
    velocities = [abs(step_log[i+1]["personality_norm"] 
                       - step_log[i]["personality_norm"])
                  for i in range(len(step_log)-1)]
    
    for i in range(len(velocities) - window_size):
        if all(v < threshold for v in velocities[i:i+window_size]):
            return i  # 人格在 round i 時「鎖定」
    
    return None  # 未鎖定
```

**指標**:
- `plasticity_window`: 人格鎖定的 round（若無則為 None）
- `plasticity_ratio`: plasticity_window / n_rounds（百分比）

**解釋**:
- 若 plasticity_window = 300，表示在 300 rounds 後人格已固定（耦合窗口 300 rounds）
- 若 plasticity_window = None，表示 1000 rounds 內仍在漂移

### 3.4 策略-人格耦合強度 (Strategy-Personality Coupling)

**定義**: 人格變化與獎勵（或策略分布）的相關性

```python
def compute_coupling_strength(step_log, window_size=100):
    """
    在 tail 內逐個 100-round window 計算相關性，取平均
    """
    personality_velocities = [step_log[i]["personality_velocity"] for i in range(len(step_log))]
    rewards = [step_log[i]["avg_reward"] for i in range(len(step_log))]
    
    windows = [(personality_velocities[i:i+window_size], rewards[i:i+window_size])
               for i in range(burn_in, len(step_log) - window_size, window_size)]
    
    correlations = [pearsonr(pv, r)[0] for pv, r in windows]
    
    return {
        "mean_correlation": np.mean(correlations),
        "std_correlation": np.std(correlations),
        "trend": "strengthening" if np.polyfit(range(len(correlations)), correlations, 1)[0] > 0 else "weakening"
    }
```

**解釋**:
- mean_correlation ≈ 0.5: 人格漂移與策略學習強耦合（好徵兆）
- mean_correlation ≈ 0: 人格漂移與策略解耦（人格自主演化）
- trend = "weakening": 耦合隨時間減弱（人格逐漸自治）

---

## 4. Gate 定義（P7-E）

### G7E-01: 執行完整性

**標準**: 10/10 runs 完成，無例外終止

```python
G7E-01 = (len(results) == 10)
```

### G7E-02: 數值穩定性

**標準**: 所有 runs 的 reward 和人格值皆有限

```python
G7E-02 = all(np.isfinite(r.step_log[-1]["avg_reward"]) 
             for r in results)
```

### G7E-03: 吸引子存在性

**標準**: 至少 1 個 run 展示 VDI < -0.01（收斂跡象）

```python
G7E-03 = any(r.vdi < -0.01 for r in results)
```

### G7E-04: 可塑性窗口一致性

**標準**: 同 α 的多個 seed，plasticity_window 標準差 < 150 rounds

```python
for alpha in [0.2, 0.3]:
    pw_list = [r.plasticity_window for r in results if r.alpha == alpha and r.plasticity_window is not None]
    if len(pw_list) >= 2 and np.std(pw_list) < 150:
        G7E-04 = True
```

### G7E-05: 種子一致性

**標準**: 同 α 跨 seeds，最終人格向量的 cosine 相似度 > 0.8

```python
for alpha in [0.2, 0.3]:
    final_vectors = [r.final_personality_vector for r in results if r.alpha == alpha]
    if len(final_vectors) >= 2:
        similarity = cosine_similarity(final_vectors[0], final_vectors[1])
        if similarity > 0.8:
            G7E-05 = True
```

---

## 5. 實作計劃

### 5.1 腳本結構

```
scripts/experiments/run_p7e_long_trajectory.py
├── def compute_vdi(trajectory, window_size=100)
├── def compute_acf_timescale(trajectory, max_lag=200)
├── def estimate_attractor_dim(trajectory_9d, embedding_dim=3)
├── def find_plasticity_window(step_log, threshold=0.01)
├── def compute_coupling_strength(step_log, window_size=100)
├── def run_single(seed, alpha, personality_vector) -> RunResult
├── def check_gates(results) -> Dict[str, bool]
└── main()
```

### 5.2 輸出結構

```
reports/experiments/p7e_long_trajectory/
├── p7e_long_trajectory_summary.csv       # 10 runs × 10 metrics
├── p7e_gates.json                        # G7E-01~05
├── run_a0.2_42_attractor.json            # 含 VDI, ACT, dim_est 等
├── run_a0.2_42_trajectory.csv            # 完整 step_log（1000 rows）
├── run_a0.3_44_trajectory.csv
├── ... (10 files)
└── p7e_long_trajectory_report.md         # 綜合分析與吸引子特徵化
```

### 5.3 預期時間表

| 階段 | 工作 | 預期時間 |
|------|------|--------|
| 1 | 實作新指標函式與 run_single | 30 min |
| 2 | 執行 10 runs (1000 rounds each) | 60 min |
| 3 | 分析與視覺化吸引子結構 | 30 min |
| 4 | 撰寫 P7-E 報告 | 20 min |
| **Total** | | **~140 min (2.3 hrs)** |

---

## 6. 預期結果與後續

### 6.1 最可能的 outcomes

**情境 A: 固定點吸引子**
- VDI < -0.01（強收斂）
- plasticity_window ≈ 300～400 rounds
- final personality ~ constant across tail
- **推進**: P7-F 可研究「吸引子的人格特徵是什麼？是否與 W 矩陣相關？」

**情節 B: 週期軌道吸引子**
- ACT > 150（強相關）
- 人格呈週期性振盪（週期～50～100 rounds）
- VDI ≈ 0（無衰減）
- **推進**: P7-F 轉向「週期特徵」分析（傅立葉分解）

**情節 C: 持續漂移（無吸引子）**
- plasticity_window = None（未鎖定）
- VDI > 0（持續發散）
- 1000 rounds 後人格仍在變化
- **推進**: 人格可塑性遠高於預期，可能需 α 調降或 N_rounds 擴展

### 6.2 決策樹（下一階段）

```
P7-E 完成
├─ 情境 A (固定點)
│  └─> P7-F: "吸引子人格空間特徵化"
│      (探索 {α, seed} 空間中有多少個不同吸引子)
├─ 情節 B (週期軌道)
│  └─> P7-F: "週期動力學特徵化"
│      (傅立葉分析、Lyapunov 指數估計)
└─ 情節 C (持續漂移)
   └─> P7-E' 重設計 (α 調降到 0.1，或 n_rounds 擴至 3000)
```

---

## 7. 文件與版本控制

- **規格**: `docs/experiments/p7_online_personality_loop/P7E_long_trajectory.md` ← **本文件**
- **日誌**: `研發日誌.md` (新增 P7-E 小節)
- **SDD**: `SDD.md` (更新 §5.3 的 P7-E 定義)
- **腳本**: `scripts/experiments/run_p7e_long_trajectory.py` (待實作)
- **報告**: `reports/experiments/p7e_long_trajectory_report.md` (待執行後生成)
