# P7-F 吸引子人格空間映射（Attractor Personality Space Mapping）

**規格版本**: P7-F v1.0  
**制定日期**: 2025-06-04  
**基於**: P7-E 結果（靜態固定點吸引子確認，α-FP 參數族發現）  
**目標**: 完整刻畫吸引子在 9D 人格空間中的位置與演化特徵

---

## 1. 科學問題

基於 P7-A/C/D/E 的發現：

1. **P7-E 發現**: 人格在 ~100 rounds 內聚斂至靜態固定點
2. **參數相依性**: 不同 α 導致不同吸引子位置 (α=0.2→||P||≈0.14, α=0.3→||P||≈0.32)
3. **缺失信息**: 
   - 吸引子在 9D 空間中的具體座標是什麼？
   - α 與吸引子位置的函數關係是否線性？
   - 吸引子是否位於某個低維不變子空間？
   - 不同 α 的吸引子之間距離多遠？

**P7-F 核心問題**：
- **問題 1**: 繪製 α-FP curve（參數化吸引子族）
- **問題 2**: 確定吸引子所在的不變子空間維度
- **問題 3**: 檢測參數空間中的分岔點（bifurcation）

---

## 2. 實驗設計

### 2.1 參數固定（Protocol Lock）

```
n_players = 4              # 保持 P7-D/E 規模
n_rounds = 200             # P7-E 的 1/5（100 rounds 後已達吸引子）
burn_in = 50               # 25% of n_rounds
tail_start = 50
tail_window = 150

personality_mode = "static"
personality_update_enabled = True
personality_learning_rate η = 0.05  # 固定
lambda_alpha = 0.15, lambda_beta = 0.10, lambda_r = 0.20, lambda_risk = 0.20
```

### 2.2 參數掃描

**α 細密掃描** (基於 P7-D 的穩定域):

| α 值 | 選擇理由 | P7-D 分類 |
|------|---------|----------|
| 0.10 | S1/S2 邊界 | 主要 S1 |
| 0.15 | 插值點 | 預期 S1→S2 過渡 |
| **0.20** | P7-E 主要 | 100% S2 |
| 0.25 | 插值點 | 預期 S2 中點 |
| **0.30** | P7-E 次要 | 67% S2 |
| 0.35 | 插值點 | 預期 S2→S4 過渡 |
| 0.40 | S2/S4 邊界 | 主要 S4 |

**種子選擇** (簡化為 3):

| seed | 理由 |
|------|------|
| 42 | P7-D/E baseline，穩定 |
| 43 | P7-D/E baseline，穩定 |
| 44 | P7-D 異常點，驗證健壯性 |

**全組合**:
```
7 α × 3 seeds = 21 runs
預期執行時間: 2.5-3 hours (200 rounds × 21)
```

### 2.3 數據記錄與分析

**Per-round 記錄** (步驟日誌):
```json
{
  "round": int,
  "p_aggressive": float,
  "p_defensive": float,
  "p_balanced": float,
  "avg_reward": float,
  "personality_vector": dict,        // 完整 9D 向量
  "personality_norm": float,         // ||P(t) - P(0)||
  "phase": str
}
```

**吸引子指標** (從尾部計算):
```python
attractor_vector = mean(personality_vector[tail_start:])  # 尾部平均
attractor_distance = ||attractor_vector||  # 到初始的距離
attractor_norm = ||attractor_vector||_2 / sqrt(9)  # 正規化
```

---

## 3. 新增分析指標

### 3.1 吸引子座標與距離

**定義**:
```python
def compute_attractor_coordinates(step_log, tail_start=50):
    """
    計算吸引子在 9D 空間中的座標（尾部平均）
    """
    tail_trajectories = [step_log[i]["personality_vector"] for i in range(tail_start, len(step_log))]
    attractor = {
        trait: np.mean([t[trait] for t in tail_trajectories])
        for trait in PERSONALITY_TRAITS
    }
    return attractor

def compute_attractor_distance(initial_personality, attractor):
    """
    計算吸引子到初始位置的歐幾里得距離
    """
    delta = np.array([attractor[t] - initial_personality.get(t, 0.0) for t in PERSONALITY_TRAITS])
    distance = np.linalg.norm(delta)
    return distance
```

**預期結果** (基於 P7-E):
```
α=0.2: distance ≈ 0.14
α=0.3: distance ≈ 0.32
```

### 3.2 吸引子間距離矩陣

**定義**: 計算同組 3 seeds 的吸引子之間的距離

```python
def compute_pairwise_distances(attractors):
    """
    attractors: list of 3 attractor vectors (for same α)
    Returns: 3×3 distance matrix (symmetric, diagonal=0)
    """
    n = len(attractors)
    distances = np.zeros((n, n))
    for i in range(n):
        for j in range(i+1, n):
            d = np.linalg.norm(
                np.array([attractors[i][t] for t in PERSONALITY_TRAITS]) -
                np.array([attractors[j][t] for t in PERSONALITY_TRAITS])
            )
            distances[i, j] = d
            distances[j, i] = d
    return distances
```

**預期結果** (基於 P7-E cosine_sim ≥ 0.99):
```
同 α 的 3 seeds 吸引子距離應 < 0.05（非常接近）
```

### 3.3 不變子空間維度估計

**定義**: 使用 SVD 估計吸引子所在的低維子空間

```python
def estimate_invariant_subspace_dim(attractors_across_alpha, threshold=0.95):
    """
    attractors_across_alpha: 7α × 3 seeds = 21 個吸引子向量
    
    步驟：
    1. 堆積 21 個吸引子向量為 21×9 矩陣
    2. 計算 SVD，取奇異值 σ_i
    3. 找最小 k 使得 sum(σ_i^2 for i<k) / sum(all σ_i^2) > threshold
    
    k = 吸引子所在子空間的維度
    """
    U, S, Vt = np.linalg.svd(attractors_matrix, full_matrices=False)
    cumsum = np.cumsum(S**2) / np.sum(S**2)
    dim = np.argmax(cumsum > threshold) + 1
    return dim, S[:5]  # 返回維度和前 5 個奇異值
```

**預期結果** (假設):
```
如果吸引子主要沿著 1-2 個主軸變化 → dim ≤ 2
如果均勻分布 → dim ≈ 9（無降維）
```

### 3.4 分岔點偵測（Bifurcation Analysis）

**定義**: 檢測人格軌跡是否存在突然改變

```python
def detect_bifurcation(all_attractors_by_alpha):
    """
    比較相鄰 α 值的吸引子
    若 ||attractor(α+Δα) - attractor(α)|| 存在不成比例的跳躍 → bifurcation
    """
    alphas = sorted(all_attractors_by_alpha.keys())
    bifurcation_points = []
    
    for i in range(len(alphas)-1):
        alpha1, alpha2 = alphas[i], alphas[i+1]
        # 計算每個 seed 的吸引子變化
        delta_alphas = alpha2 - alpha1
        
        for seed_idx in range(3):
            att1 = all_attractors_by_alpha[alpha1][seed_idx]
            att2 = all_attractors_by_alpha[alpha2][seed_idx]
            delta_att = np.linalg.norm(att2 - att1)
            rate_of_change = delta_att / delta_alphas
            
            # 若變化率明顯變大 → bifurcation
            if rate_of_change > 0.5:  # 閾值可調
                bifurcation_points.append((alpha1, alpha2, rate_of_change))
    
    return bifurcation_points
```

**預期結果** (假設):
```
若無分岔 → 光滑的 α-FP 曲線（預期）
若有分岔 → 在某個 α* 處出現分類改變
```

---

## 4. Gate 定義（P7-F）

### G7F-01: 執行完整性

**標準**: 21/21 runs 完成，無例外

```python
G7F-01 = (len(results) == 21)
```

### G7F-02: 數值穩定性

**標準**: 所有 runs 的吸引子座標皆有限

```python
G7F-02 = all(all(np.isfinite(v) for v in r.attractor_coordinates.values()) 
             for r in results)
```

### G7F-03: 同 α 聚斂性

**標準**: 同 α 的 3 seeds 吸引子距離平均 < 0.05

```python
for alpha in alphas:
    dists = compute_pairwise_distances(
        [r.attractor_coordinates for r in results if r.alpha == alpha]
    )
    mean_dist = np.mean(dists[np.triu_indices_from(dists, k=1)])
    if mean_dist > 0.05:
        G7F-03 = False
```

### G7F-04: α 線性性

**標準**: 吸引子距離 (||attractor||) 對 α 的擬合 R² > 0.95

```python
alpha_list = sorted(set(r.alpha for r in results))
norm_list = [np.mean([r.attractor_distance for r in results if r.alpha == a])
             for a in alpha_list]

fit = np.polyfit(alpha_list, norm_list, 1)  # 1 次多項式
y_pred = np.polyval(fit, alpha_list)
r_squared = 1 - np.sum((norm_list - y_pred)**2) / np.sum((np.array(norm_list) - np.mean(norm_list))**2)

G7F-04 = (r_squared > 0.95)
```

### G7F-05: 維度估計可信性

**標準**: 估計的吸引子子空間維度 ≤ 3（低維確認）

```python
dim, singular_values = estimate_invariant_subspace_dim(all_attractors)
G7F-05 = (dim <= 3)
```

---

## 5. 實作計劃

### 5.1 腳本結構

```
scripts/experiments/run_p7f_attractor_mapping.py
├── def compute_attractor_coordinates(step_log, tail_start, TRAITS)
├── def compute_attractor_distance(initial, attractor)
├── def compute_pairwise_distances(attractors)
├── def estimate_invariant_subspace_dim(attractors_matrix, threshold)
├── def detect_bifurcation(all_attractors_by_alpha)
├── def run_single(seed, alpha, personality_vector) -> RunResult
├── def check_gates(results) -> Dict[str, bool]
└── main()
```

### 5.2 輸出結構

```
reports/experiments/p7f_attractor_mapping/
├── p7f_attractor_mapping_summary.csv       # 21 runs × 15 metrics
├── p7f_gates.json                          # G7F-01~05
├── p7f_attractor_coordinates.json          # 21 個吸引子的 9D 座標
├── p7f_subspace_analysis.json              # SVD 結果（維度、奇異值）
├── p7f_bifurcation_analysis.json           # 分岔點檢測結果
├── run_a0.10_42.json                       # Per-run metadata
├── run_a0.10_42_trajectory.csv             # 200-round trajectory
├── ... (21 run pairs)
└── p7f_attractor_mapping_report.md         # 綜合分析與吸引子特徵化
```

### 5.3 預期時間表

| 階段 | 工作 | 預期時間 |
|------|------|--------|
| 1 | 實作吸引子分析函式 | 20 min |
| 2 | 執行 21 runs (200 rounds each) | 120 min |
| 3 | SVD 與分岔分析 | 15 min |
| 4 | 繪製 α-FP curve 與視覺化 | 20 min |
| 5 | 撰寫 P7-F 報告 | 25 min |
| **Total** | | **~200 min (3.3 hrs)** |

---

## 6. 預期結果與科學意義

### 6.1 最可能的結果

**情境 A: 1D 線性吸引子族**
- ||attractor|| ∝ α（完美線性）
- subspace_dim = 1（所有吸引子位於某條直線上）
- 無分岔點
- **意義**: 人格空間中存在單參數化的吸引子族，反映系統的簡樸結構

**情節 B: 2D 平面吸引子族**
- ||attractor|| ∝ α²（非線性），或多個主軸方向
- subspace_dim = 2
- 可能在某個 α* 附近出現拐點（但非分岔）
- **意義**: 吸引子更複雜，涉及 2 個獨立自由度

**情節 C: 高維分散（無低維結構）**
- subspace_dim ≈ 9（無明顯降維）
- 吸引子均勻分布在 9D 空間中
- **意義**: 人格空間的吸引子軌跡充分利用維度（可能需要重新考慮模型）

### 6.2 科學影響

- **若 A**: 進行 P7-G（單參數吸引子沿著該方向的微擾分析）
- **若 B**: 進行 P7-G（識別 2D 子空間的基向量，分析耦合效應）
- **若 C**: 反思人格空間與反饋機制的設計（可能維度過高）

---

## 7. 與其他階段的銜接

```
P7-D (穩定性)  ─────────┐
                         ├──→ P7-F (吸引子映射) ─────→ P7-G (微擾/分岔)
P7-E (軌跡)  ──────────┘

P7-F 整合 P7-D 的穩定域識別與 P7-E 的吸引子驗證，
進一步推進系統動力學的完整刻畫
```

---

## 8. 檢查清單

- [ ] 實作 `compute_attractor_coordinates()` 與 distance 計算
- [ ] 實作 `estimate_invariant_subspace_dim()` (SVD)
- [ ] 實作 `detect_bifurcation()` 邏輯
- [ ] 執行 21 runs，驗證 G7F-01/02
- [ ] 驗證 G7F-03（同 α 聚斂）
- [ ] 驗證 G7F-04（α 線性性）
- [ ] 驗證 G7F-05（維度估計）
- [ ] 繪製 α-FP curve（2D 投影）
- [ ] 撰寫 P7-F 報告與科學結論
- [ ] 更新研發日誌與 SDD
