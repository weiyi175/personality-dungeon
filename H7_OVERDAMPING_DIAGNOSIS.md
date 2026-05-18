# H7 過度阻尼（Over-damped）診斷報告

**日期**：2026-05-15  
**結論**：H7.2 的失敗證實了系統並非被「參數上限」限制，而是被**算子級別的過度阻尼機制**錮住了。

---

## 一、問題症狀回顧

| 實驗 | 規模 | L3 成功率 | 結論 |
|---|---|---|---|
| coupling_grid_pilot | 162 runs | 0.0% (全 L2) | control ≈ random（無對照牆） |
| c_power_aggressive_scan | 216 runs | 0.0% (全 L2) | 高功率掃描無效 |
| H7.2 (k-clamp 放寬) | 72 runs | 1-6% (按 SS) | random 反而更容易 L3 |

**統一診斷**：問題不在參數的上限值，而在**動力學導數本身**。

---

## 二、三層過度阻尼根源

### 2.1 第一層：Inertia（μ）設計過於保守

#### 當前實作（`personality_coupling.py` L46-50）

```python
def resolve_personality_coupling(...):
    signal_mu = 0.5 * (stability + endurance) - 0.5 * impulsiveness
    mu_value = clamp(mu_base + lambda_mu * signal_mu, mu_lower, mu_upper)
    # 其中 mu_base=0.05, lambda_mu=0.10
```

#### 數值分析

- **signal_mu 範圍**：假設 personality traits ∈ [-0.4, 0.4]，signal_mu ≈ [-0.4, 0.4]
- **μ 計算**：`μ = clamp(0.05 + 0.10 * [-0.4, 0.4], [0.0, 0.60])`
  - **最小值**：clamp(0.05 - 0.04, ...) = **0.01**
  - **最大值**：clamp(0.05 + 0.04, ...) = **0.09**
  - **有效範圍**：μ ∈ [0.01, 0.09] — **極其窄小！**

#### 物理意涵

在 `inertial_growth_step` 中（`evolution/replicator_dynamics.py` L498-500）：

```python
velocity[s] = momentum * prev_velocity[s] + strength[s] * growth[s]
            = μ * v(t-1) + k * g(t)
```

當 μ ∈ [0.01, 0.09] 時：
- 大部分情況下 μ ≈ 0.05（中位值）
- 這意味著上一步的速度只被保留 5%，95% 被新增的 growth 主導
- **結果**：系統是「接近無記憶的」，相位無法累積

#### 對比健康系統

如果我們希望系統有「明確的相位記憶」，μ 應該在 [0.2, 0.4] 範圍，使得上一步貢獻 20-40%。

### 2.2 第二層：Selection Strength（k）在 personality_coupled 模式下被過度約束

#### 當前實作（`run_simulation.py` L2172-2176）

```python
params = resolve_personality_coupling(
    ...,
    k_base=float(cfg.selection_strength),  # <-- 這裡關鍵！
    lambda_k=float(cfg.personality_coupling_lambda_k),  # = 0.20
    ...,
    k_lower=float(cfg.personality_coupling_k_lower),   # = 0.05 (H7.2)
    k_upper=float(cfg.personality_coupling_k_upper),   # = 0.25 (H7.2)
)
```

#### 數值分析

`k_raw = k_base * (1 + lambda_k * signal_k)`

其中 `k_base = cfg.selection_strength`（外部傳入），signal_k ≈ [-0.4, 0.4]

**情況 1：SS=0.10 時**
- `k_raw_min = 0.10 * (1 - 0.20*0.4) = 0.10 * 0.92 = 0.092`
- `k_raw_max = 0.10 * (1 + 0.20*0.4) = 0.10 * 1.08 = 0.108`
- **clamp 後**（[0.05, 0.25]）：k ∈ [0.092, 0.108]
- **有效範圍**：只有 1.6% 的變化空間！

**情況 2：SS=0.20 時**
- `k_raw_min = 0.20 * 0.92 = 0.184`
- `k_raw_max = 0.20 * 1.08 = 0.216`
- **clamp 後**：k ∈ [0.184, 0.216]
- **有效範圍**：仍然只有 1.7%

#### 物理意涵

在 `inertial_growth_step` 中（L514）：

```python
velocity[s] = μ * prev_velocity[s] + k * growth[s]
```

當 k 的變化幅度 < 2% 時，personality 對選擇強度的調製**幾乎沒有影響**。

**這意味著 personality_coupled 實際上是「假 coupling」** — personality 被計算了，但對動力學的影響微乎其微。

### 2.3 第三層：Exponential Growth 的衰減特性

#### 當前實作（`evolution/replicator_dynamics.py` L509）

```python
out[s] = max(1e-6, weights[s] * exp(velocity[s]))
```

#### 分析

當 velocity[s] 很小（如 < 0.01）時，exp(velocity) ≈ 1.0，權重幾乎不變。

結合 velocity = μ * v_prev + k * g 的低數值，系統趨向於：
1. 速度記憶短暫（μ 小）
2. 每一步的權重變化幅度小（k 小）
3. exp 函數在小輸入時的變化更小
4. **淨結果**：權重在 simplex 上爬行緩慢，遠達不到 L3 的旋轉速度要求

---

## 三、為何 H7.2「放寬 k-clamp」也失敗了？

H7.2 把 k_clamp 從 [0.03, 0.09] 放寬到 [0.05, 0.25]，預期能增加 k 的有效範圍。

但實際上：
- `k_raw` 的計算方式沒變：仍然是 `selection_strength * (1 + lambda_k * signal_k)`
- `selection_strength` 本身仍然從外部 cfg 傳入（SS=0.10, 0.15, 0.20）
- clamp 的下限從 0.03 → 0.05，對於 `k_raw ≈ 0.08~0.12` 的計算結果，**沒有實質改變**
- **結論**：放寬 clamp 的上限對這個系統基本無效，因為 `k_raw` 的計算本身就很小

---

## 四、建議的「外科手術」改造

### 選項 A：激進提升 μ 基準

**改動**：直接修改 `personality_coupling.py` 中的 μ 計算

```python
# 舊版本
mu_value = clamp(mu_base + lambda_mu * signal_mu, mu_lower, mu_upper)

# 新版本（激進）
# 方法 1：提高 mu_base
mu_base_new = 0.20  # 從 0.05 → 0.20

# 方法 2：或者改變計算形式（乘法而非加法）
mu_value = clamp(mu_base * (1.0 + 2.0 * lambda_mu * signal_mu), mu_lower, mu_upper)
# 這樣當 lambda_mu=0.10 時，有效範圍變成：
# min: 0.05 * (1 - 2*0.10*0.4) = 0.05 * 0.92 = 0.046 → clamp(0.0, 0.60) = 0.046
# max: 0.05 * (1 + 2*0.10*0.4) = 0.05 * 1.08 = 0.054 → clamp(0.0, 0.60) = 0.054
# 仍然太小！不行。

# 方法 3：直接設置為固定的較高值
mu_value = 0.25  # 忽視 personality signal，使用固定高慣性
```

### 選項 B：重新設計 k 的計算機制

```python
# 舊版本
k_raw = k_base * (1.0 + lambda_k * signal_k)

# 新版本（激進）
# 方法 1：將 signal_k 映射到更寬的 k 範圍
# 假設我們希望 k ∈ [0.05, 0.50]（寬度 10 倍），而 signal_k ∈ [-0.5, 0.5]
k_value_new = 0.05 + 0.45 * (0.5 + signal_k) / 1.0
# 當 signal_k = -0.5 時，k = 0.05
# 當 signal_k = +0.5 時，k = 0.50
# clamp(result, 0.05, 0.50)

# 方法 2：或者使用非線性映射（sigmoid-like）
def k_from_signal(signal_k):
    return 0.05 + 0.45 / (1.0 + exp(-4.0 * signal_k))
# 這會在 signal_k=0 附近形成快速轉變
```

### 選項 C：改變 inertial_growth_step 的非線性

```python
# 舊版本（evolution/replicator_dynamics.py L509）
out[s] = max(1e-6, weights[s] * exp(velocity[s]))

# 新版本（激進）
# 使用更陡峭的增長函數
out[s] = max(1e-6, weights[s] * (1.0 + 2.0 * velocity[s] + velocity[s]**2))
# 這是泰勒級數的二階逼近，但保留了更多非線性
# 或者直接使用平方項放大：
out[s] = max(1e-6, weights[s] * exp(velocity[s] * 3.0))  # 3x 放大
```

---

## 五、建議的即時測試方案

### 5.1 快速驗證（單點測試）

```bash
# 修改 personality_coupling.py，將 mu_base 從 0.05 提高到 0.30
# 或將 personality_coupled 模式下的 k 計算改為：
# k = clamp(0.1 + 0.4 * signal_k, 0.05, 0.50)

cd /home/user/personality-dungeon
PYTHONPATH=/home/user/personality-dungeon ./venv/bin/python scripts/run_structural_breakthrough.py \
  --mu-base 0.30 \
  --lambda-mu 0.05
```

### 5.2 對照實驗（決定是否值得投入）

```bash
# Test 1: High inertia, standard k
./venv/bin/python scripts/run_structural_breakthrough.py \
  --mu-base 0.30 --lambda-mu 0.05 --lambda-k 0.10 \
  --rounds 6000 --seeds 45,47,49,51,53,55 \
  > /tmp/test1_high_mu.log 2>&1

# Test 2: Standard mu, high-range k
# 需要修改 personality_coupling.py 的 k 計算邏輯
```

---

## 六、決策樹

**Q1**：你是否想要**激進改動** `personality_coupling.py` 的算子設計？
- **是** → 建議先執行 5.1 的「High inertia」測試，看是否能點亮 L3
- **否** → 考慮在 payoff 矩陣層面進行改造（更底層但影響範圍更廣）

**Q2**：如果 High inertia 測試成功，你是否願意進一步改造 k 的計算？
- **是** → 執行選項 B
- **否** → 保持高 μ，接受 k 的「假 coupling」特性

---

## 七、風險評估

| 改動 | 風險 | 缺點 |
|---|---|---|
| 提高 μ | 低（只改數值，不改邏輯） | 可能導致系統過度振盪（低阻尼）|
| 改造 k 計算 | 中（改邏輯，需全面測試） | 破壞既有 personality_coupled 契約 |
| 改造 exp 函數 | 高（影響全系統） | 需要重新驗證所有既有實驗的可重現性 |

---

## 總結

**當前症狀**：系統被過度阻尼鎖死，即使放寬參數上限也無效。

**根本原因**：
1. μ 太小（< 0.09）→ 無相位記憶
2. k 的有效變化 < 2% → personality 影響可忽略
3. exponential growth 的衰減特性進一步削弱效果

**下一步**：執行「High inertia 測試」，驗證提高 μ 是否能點亮 L3。如果成功，再考慮改造 k 的計算機制。
