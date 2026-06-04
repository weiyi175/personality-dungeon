# P7-G 吸引子穩定性與微擾分析（Perturbation Analysis on Attractors）

**規格版本**: P7-G v1.0  
**制定日期**: 2025-06-04  
**基於**: P7-F 結果（1D 不變子空間確認，dim=1, σ₁ 解釋 99.93% 方差）  
**目標**: 驗證 1D 子空間穩定性，量化系統動力學（Lyapunov 指數、恢復時間）

---

## 1. 科學問題

### 從 P7-F 到 P7-G 的邏輯

P7-F 發現：
```
✓ 吸引子完全位於 1D 子空間
✓ 主方向捕捉 99.93% 方差
✓ 線性 α-FP 曲線 (R²=1.0)
```

P7-G 新問題：
```
1. 1D 子空間為何穩定？
   → 垂直于 1D 的擾動是否衰減？衰減速度多快？

2. 系統的混沌性如何？
   → Lyapunov 指數是多少？系統是否可預測？

3. 吸引子的盆地是什麼樣？
   → 擾動大小與恢復時間的關係？

4. 主方向在人格空間中的含義？
   → 第 1 主特徵向量的成分解釋？
```

### P7-G 核心目標

```
目標 1: 驗證 1D 子空間的吸引性
  標準: 垂直于 1D 的擾動應快速衰減回到 1D 子空間
  測量: 擾動恢復時間 τ、指數衰減率 λ

目標 2: 估計 Lyapunov 指數
  標準: 計算沿 1D 主方向的最大 Lyapunov 指數 λ_max
  解釋: 若 λ_max < 0 → 穩定吸引子; λ_max ≈ 0 → 邊界; λ_max > 0 → 混沌

目標 3: 刻畫恢復動力學
  標準: 測量擾動大小 ε 對恢復時間的依賴
  預期: 若線性系統 → τ ∝ log(1/ε); 若非線性 → τ 可能更複雜

目標 4: 識別不變子空間的特徵
  標準: 提取 SVD 主方向，在 9D 人格空間中解釋
```

---

## 2. 實驗設計

### 2.1 基準運行（Base Run）

```yaml
protocol:
  n_players: 4
  n_rounds: 200
  burn_in: 50
  tail_start: 50
  
personality:
  mode: "static"
  group: "G-AGG"  # ("我喜歡冒險挑戰") [來自 P7-F]
  update_enabled: true
  learning_rate: 0.05
  feedback_strength: α = 0.2  # [S2 穩定域中心]
  
dynamics:
  lambda_alpha: 0.15
  lambda_beta: 0.10
  lambda_r: 0.20
  lambda_risk: 0.20

seeds:
  base: [42, 43, 44]  # [來自 P7-F]
  perturbations: +ε, -ε for each axis
```

**選擇理由**:
- α=0.2: P7-D/E/F 都驗證此值在 S2 穩定域中心
- G-AGG: 保持與 P7-F 一致的人格初始向量
- n_rounds=200: 足以觀察擾動衰減與吸引子返回
- base_seeds={42,43,44}: P7-F 已驗證其行為

### 2.2 微擾實驗設計

#### 2.2.1 擾動軸選擇

9 個人格特徵軸（選 8 個垂直于 1D 主軸的方向）：

```
P7-F SVD 主方向: v₁ (第 1 特徵向量，捕捉 99.93% 方差)
垂直于 v₁ 的子空間: V_perp (8 維)

微擾軸:
  axis_1: 沿 SVD 第 2 主成分方向 (σ₂ 方向) [驗證邊界穩定性]
  axis_2: 沿 SVD 第 3 主成分方向 (σ₃ 方向) [更弱的次要方向]
  axis_3-9: 沿 personality 空間的 orthonormal 基 (通過 Gram-Schmidt)
```

#### 2.2.2 擾動幅度

```
ε ∈ {0.005, 0.010, 0.020}  # 三個尺度

選擇理由:
  ε=0.005 (小): 線性政權，檢測局部穩定性
  ε=0.010 (中): 典型擾動尺度（P7-F 吸引子距離的 ~10%）
  ε=0.020 (大): 測試非線性，邊界效應

應用方式:
  - positive perturbation: P(0) = P_init + ε × e_i
  - negative perturbation: P(0) = P_init - ε × e_i
```

#### 2.2.3 實驗矩陣

```
總 runs 計算:
  Base runs: 3 seeds × 1 (無擾動) = 3 runs
  Perturbed runs: 3 seeds × 8 axes × 3 ε × 2 (±) = 144 runs
  總計: 3 + 144 = 147 runs

執行計劃:
  Phase 1: 3 base runs (驗證 α=0.2 下的吸引子位置)
  Phase 2: 24 runs × 3 ε sizes (single axis sweep, 選主要軸 axis_1,2)
  Phase 3: 完整 8 軸 × 3 ε × 2 (±) = 48 runs per seed = 144 runs (optional if Phase 2 結果充分)
  
完整執行: 3 + 144 = 147 runs
簡化版本: 3 + 48 = 51 runs (僅 axis_1,2 + 3 ε)
```

### 2.3 指標定義

#### 2.3.1 吸引子返回（Attractor Recovery）

```python
def compute_recovery_time(trajectory, baseline_attractor, threshold=0.01):
    """
    測量擾動軌跡何時返回到吸引子。
    
    Args:
        trajectory: list of personality vectors (length n_rounds)
        baseline_attractor: 無擾動時的吸引子位置 (9D)
        threshold: 距離閾值 (predefined, 預設 0.01)
    
    Returns:
        recovery_time (int rounds), recovery_success (bool)
    
    邏輯:
        distance_t = ||trajectory[t] - baseline_attractor||
        return first t where distance_t < threshold
        if never happens, return_time = n_rounds, success = False
    """
```

**預期結果**:
```
若系統線性穩定:
  τ ≈ 50-100 rounds (recovery time 較短)
  
若非線性:
  τ 依賴 ε，可能 τ ∝ log(1/ε)
```

#### 2.3.2 Lyapunov 指數估計

```python
def compute_local_lyapunov_exponent(trajectory_perturbed, trajectory_base, dt=1):
    """
    計算局部最大 Lyapunov 指數。
    
    Theory:
        λ_max ≈ (1/T) × Σ_t log(||Δx(t)||/||Δx(0)||)
        其中 Δx(t) = x_perturbed(t) - x_base(t)
    
    邏輯:
        1. 計算每 round 的擾動范數 ||Δx(t)||
        2. 計算比例 ||Δx(t)||/||Δx(0)||
        3. 取平均 log(ratio) 得 λ_max per round
        4. 若 λ_max < 0 → 穩定; λ_max > 0 → 不穩定
    """
```

**預期結果**:
```
若吸引子穩定:
  λ_max < 0 (負指數表示指數衰減)
  
若邊界混沌:
  λ_max ≈ 0 或略大於 0
```

#### 2.3.3 擾動衰減指標

```python
def compute_perturbation_decay_metrics(perturbation_history):
    """
    計算擾動如何隨時間衰減。
    
    指標:
    1. decay_rate: 指數衰減速率 (rounds^-1)
    2. half_life: 擾動幅度減半所需 rounds 數
    3. final_residual: 尾部軌跡的平均擾動幅度
    """
    
    # 尾部定義: tail = [50:200] (150 rounds, 同 P7-F)
    tail_perturbations = [||Δx(t)|| for t in range(50, 200)]
    
    # 指數衰減擬合: ||Δx(t)|| = A × exp(-λ × t)
    # 估計 λ 與 A
    
    if tail_perturbations mean < 1e-6:
        decay_rate = np.inf  # 快速衰減
        half_life = 0
    else:
        decay_rate, half_life = fit_exponential_decay(tail_perturbations)
    
    return {
        "decay_rate": decay_rate,
        "half_life": half_life,
        "final_residual": np.mean(tail_perturbations)
    }
```

#### 2.3.4 恢復時間與擾動幅度的關係

```python
def compute_recovery_dependency(results_by_epsilon):
    """
    分析 τ(ε) 關係。
    
    預期模式 1 (線性): τ ~ constant (快速恢復，不依賴 ε)
    預期模式 2 (對數): τ ~ α × log(1/ε)
    預期模式 3 (冪律): τ ~ ε^(-β)
    """
```

---

## 3. Gate 定義（P7-G）

### G7G-01: 執行完整性

**標準**: 預定義的 runs (至少 Phase 1+2) 全部完成

```python
if execution_level == "full":
    required_runs = 3 + 144  # 147
elif execution_level == "medium":
    required_runs = 3 + 48   # 51
else:
    required_runs = 3        # Phase 1 only

G7G-01 = (len(results) >= required_runs * 0.95)  # 允許 5% 失敗
```

### G7G-02: 基準穩定性

**標準**: Base runs (無擾動) 應聚斂到與 P7-F 一致的吸引子

```python
# 比較 P7-G base runs 與 P7-F results at α=0.2
p7f_attractor_alpha02 = ...  # from P7-F

p7g_base_attractors = [r.attractor for r in results if r.epsilon == 0 and r.seed in {42,43,44}]

cosine_similarities = [cosine_sim(att, p7f_attractor_alpha02) for att in p7g_base_attractors]

G7G-02 = (mean(cosine_similarities) > 0.99)  # 與 P7-F 高度一致
```

### G7G-03: 擾動衰減性

**標準**: 所有擾動應在合理時間內衰減

```python
recovery_times = [r.recovery_time for r in results if r.is_perturbed]

G7G-03 = (
    (mean(recovery_times) < 150 * n_rounds/200) AND  # 平均恢復 < 150 rounds
    (all(rt < n_rounds for rt in recovery_times))     # 所有 runs 內 n_rounds 恢復
)
```

### G7G-04: Lyapunov 穩定性

**標準**: 最大 Lyapunov 指數應為負（穩定）

```python
lyapunov_exponents = [r.lambda_max for r in results if r.is_perturbed]

G7G-04 = (
    (mean(lyapunov_exponents) < 0) AND  # 平均為負
    (quantile(lyapunov_exponents, 0.75) < 0)  # 75% 分位數為負
)
```

### G7G-05: 軸方向一致性

**標準**: 不同 seeds 在相同軸、相同 ε 下的恢復時間應相似

```python
for axis in axes:
    for epsilon in epsilons:
        times = [r.recovery_time for r in results 
                 if r.axis == axis and r.epsilon == epsilon]
        
        if len(times) >= 2:
            cv = std(times) / mean(times)  # 變異係數
            if cv > 0.5:  # 變異過大
                G7G-05 = False
                break

G7G-05 = all_axis_epsilon_consistent
```

---

## 4. 數據記錄與分析

### 4.1 Per-Run 記錄

```python
class PerturbationRunResult:
    seed: int
    alpha: float = 0.2  # 固定
    group: str = "G-AGG"
    
    # 擾動配置
    is_perturbed: bool
    perturbed_axis: Optional[int]  # 0-7 (8 個軸), None if base
    epsilon: float  # 擾動幅度, 0.0 if base
    perturbation_direction: str  # "positive" or "negative"
    
    # 基準吸引子
    initial_personality: Dict[str, float]
    baseline_attractor: Dict[str, float]  # 來自 base run
    
    # 擾動初始條件
    perturbed_initial_state: Dict[str, float]
    initial_perturbation_magnitude: float  # ||perturbed - base||
    
    # 軌跡記錄
    step_log: List[Dict]  # 完整 200 round 日誌
    
    # 分析指標
    recovery_time: int  # rounds 直到返回吸引子
    recovery_success: bool
    
    lyapunov_exponent: float  # λ_max
    decay_rate: float  # 指數衰減率
    half_life: int  # 擾動幅度減半時間
    
    final_attractor_distance: float  # ||trajectory[-1] - baseline_attractor||
```

### 4.2 Step Log 記錄

```json
{
  "round": int,
  "personality_vector": dict,  # 完整 9D
  "personality_norm": float,
  "perturbation_magnitude": float,  // ||P(t) - P_base(t)||
  "distance_to_baseline_attractor": float,
  "phase": str
}
```

---

## 5. 實作計劃

### 5.1 腳本結構

```
scripts/experiments/run_p7g_perturbation_analysis.py
├── def load_p7f_results() -> Dict[float, List[Dict]]
│   # 加載 P7-F 吸引子座標與主特徵向量
│
├── def compute_perturbation_axes(all_attractors_p7f) -> np.ndarray
│   # 計算 SVD 主方向，生成正交軸基
│
├── def compute_recovery_time(trajectory, baseline, threshold) -> Tuple[int, bool]
│   # 測量擾動恢復時間
│
├── def compute_lyapunov_exponent(traj_pert, traj_base) -> float
│   # Lyapunov 指數估計
│
├── def compute_decay_metrics(perturbation_history) -> Dict
│   # 衰減速率與半衰期
│
├── def run_single(seed, axis, epsilon, direction) -> PerturbationRunResult
│   # 單次擾動 run
│
├── def check_gates(results) -> Dict[str, bool]
│   # G7G-01~05 驗證
│
└── main()
    # 執行所有 runs、數據整理、報告生成
```

### 5.2 輸出結構

```
reports/experiments/p7g_perturbation_analysis/
├── p7g_perturbation_summary.csv
│   # axes × epsilons × directions × seeds, ~150 rows
│
├── p7g_lyapunov_analysis.json
│   # λ_max per axis, per epsilon, statistics
│
├── p7g_recovery_times.json
│   # recovery_time vs (axis, epsilon, direction, seed)
│
├── p7g_decay_rates.json
│   # decay_rate, half_life per perturbation
│
├── p7g_perturbation_axes.json
│   # 8 個正交軸的坐標 (9D)
│
├── p7g_axes_interpretation.json
│   # 每個軸的人格特徵權重解釋
│
├── p7g_gates.json
│   # G7G-01~05 結果
│
├── run_seed42_axis0_eps0005_pos.json + trajectory.csv
├── ... (各 perturbation run)
│
└── p7g_perturbation_analysis_report.md
    # 完整分析報告
```

---

## 6. 預期結果與科學意義

### 6.1 情景 A: 完全穩定（最可能）

```
特徵:
  - λ_max < -0.01 (強負, 指數穩定)
  - τ(ε) 獨立于 ε (快速恢復，不超過 50 rounds)
  - 垂直于 1D 軸的擾動快速衰減 (τ << 100)

意義:
  ✓ 1D 子空間為全局吸引子
  ✓ 系統完全可預測，無混沌
  ✓ 邊界盆地大，魯棒性高
```

### 6.2 情景 B: 邊界穩定

```
特徵:
  - λ_max ≈ 0 (邊界，弱穩定)
  - τ(ε) ∝ log(1/ε) (非線性恢復)
  - 軸方向間恢復時間差異大

意義:
  ⚠️ 系統接近混沌邊界
  ⚠️ 需要謹慎參數選擇
  ✓ 但仍整體穩定
```

### 6.3 情景 C: 局部不穩定

```
特徵:
  - λ_max > 0 (某些軸方向不穩定)
  - 擾動沿某些方向增長
  - 恢復失敗率 > 10%

意義:
  ✗ 反饋機制設計需調整
  ✗ 部分參數範圍不適用
```

---

## 7. 實驗流程與時間表

| 階段 | 工作 | 預期時間 |
|------|------|--------|
| 1 | 加載 P7-F 數據、計算正交軸 | 5 min |
| 2 | 實行 Phase 1 (3 base runs) | 10 min |
| 3 | 實行 Phase 2 (48 perturbation runs) | 2 hrs |
| 4 | 實行 Phase 3 (144 perturbation runs, optional) | 4-6 hrs |
| 5 | 分析 (Lyapunov, decay rates, recovery times) | 20 min |
| 6 | 繪製視覺化與撰寫報告 | 30 min |
| **Total (Phase 1+2)** | | **~2.5 hrs** |
| **Total (Phase 1-3 full)** | | **~7 hrs** |

---

## 8. 與 SDD 的對應

### 8.1 新增規約

```
Section: 4.7 吸引子穩定性（Attractor Stability）

定義:
  - 1D 不變子空間 V₁: P7-F SVD 主方向
  - 吸引子函數 P*(α): ||P*(α)|| = 0.4565 × α
  - 局部 Lyapunov 指數 λ_max: 衡量穩定性

條件:
  - 若 λ_max < 0: 指數穩定
  - 若 λ_max ≈ 0: 邊界穩定（Lyapunov 穩定但非漸近)
  - 若 λ_max > 0: 不穩定 (違反設計要求)

Protocol lock (P7-G):
  - α = 0.2 (S2 穩定域中心)
  - n_rounds = 200
  - perturbation_axes = 8 (垂直于 P*(α))
  - perturbation_scales = 3 (ε ∈ {0.005, 0.010, 0.020})
```

---

## 9. 檢查清單

- [ ] 加載 P7-F 吸引子座標與 SVD 結果
- [ ] 計算 8 個正交軸基 (Gram-Schmidt orthogonalization)
- [ ] 驗證軸的正交性 (dot products 應 ≈ 0)
- [ ] 執行 3 個 base runs，確認與 P7-F 一致
- [ ] 執行 Phase 2 微擾 runs (48 runs)
- [ ] 計算所有 Lyapunov 指數
- [ ] 計算所有恢復時間
- [ ] 分析 τ(ε) 依賴性
- [ ] 繪製 λ_max vs axis, λ_max vs epsilon 圖
- [ ] 解釋主軸方向的人格含義
- [ ] 撰寫 P7-G 報告與科學結論
- [ ] 更新研發日誌

---

## 10. 銜接與後續

```
P7-F (空間映射) ──┐
                 ├──> P7-G (穩定性分析) ✓ 設計完成
                 │
           動力學刻畫 ──┐
                        ├──> P7-H? (系統設計優化或實驗外推)
           Lyapunov分析  │
           恢復動力學  ──┘
```

**P7-H 預期方向** (可選):
```
1. 參數空間擴展 (α > 0.4 進入混沌區？)
2. 人格向量非線性效應 (不同初始條件的吸引子族)
3. 多人玩家互動的集體人格動力學
```

---

**規格狀態**: ✅ 完成 (v1.0)  
**審批**: 待執行前確認
