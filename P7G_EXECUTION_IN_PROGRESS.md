# P7-G 吸引子穩定性與微擾分析 — 執行進行中

**執行狀態**: 🔄 進行中  
**開始時間**: 2025-06-04 (執行中)  
**預計完成**: ~2.5 小時  
**Target**: 39 runs (3 base + 36 perturbation)

---

## 執行配置

```yaml
α: 0.2 (S2 穩定域中心)
Seeds: [42, 43, 44]
Perturbation scales: [0.005, 0.010, 0.020]
Phase: 2 (2 main axes × 3 seeds × 3 scales × 2 directions)

Phase breakdown:
  Phase 1: 3 base runs (無擾動)
  Phase 2: 36 perturbation runs
  Total: 39 runs
  
Rounds per run: 200
Total rounds: 39 × 200 = 7,800
```

---

## 預期輸出結構

執行完成後，以下文件將生成：

### 核心分析文件

```
reports/experiments/p7g_perturbation_analysis/

✓ p7g_perturbation_summary.csv
  - 39 rows × 12 columns (metrics per run)
  - Columns: seed, is_perturbed, axis, epsilon, direction, 
             recovery_time, recovery_success, lyapunov_exponent, 
             decay_rate, half_life, final_perturbation_magnitude, 
             final_distance_to_baseline

✓ p7g_perturbation_axes.json
  - 8 orthonormal axes (each 9D)
  - Generated via Gram-Schmidt orthogonalization

✓ p7g_lyapunov_analysis.json
  - Per-axis Lyapunov exponent statistics
  - mean, std, min, max, values

✓ p7g_gates.json
  - Gate check results (G7G-01 ~ G7G-05)
```

### Per-Run 文件

```
✓ run_seed42_axis0_eps0.005_positive.json (metadata)
✓ run_seed42_axis0_eps0.005_positive_trajectory.csv (200 rows)
✓ run_seed42_axis0_eps0.005_negative.json
✓ run_seed42_axis0_eps0.005_negative_trajectory.csv
✓ run_seed42_axis0_eps0.010_positive.json + trajectory.csv
✓ run_seed42_axis0_eps0.010_negative.json + trajectory.csv
✓ run_seed42_axis0_eps0.020_positive.json + trajectory.csv
✓ run_seed42_axis0_eps0.020_negative.json + trajectory.csv
✓ run_seed42_axis1_eps0.005_positive.json + trajectory.csv
... (36 perturbation run pairs total)

Plus 3 base runs (no perturbation):
✓ run_seed42_base.json + trajectory.csv
✓ run_seed43_base.json + trajectory.csv
✓ run_seed44_base.json + trajectory.csv
```

---

## 預期關鍵指標

### 基準穩定性 (Phase 1 Base Runs)

```
Expected attractor (α=0.2) based on P7-F:
  ||attractor|| ≈ 0.0913 (平均)
  
Consistency check:
  ✓ Cosine similarity vs P7-F > 0.99 (high consistency)
```

### 恢復時間 (Recovery Time)

```
Expected distribution:
  ε=0.005: τ ~ 30-50 rounds
  ε=0.010: τ ~ 40-70 rounds
  ε=0.020: τ ~ 60-100 rounds
  
All: τ < 150 rounds (system stable)
```

### Lyapunov 指數

```
Expected results:
  λ_max < 0 (所有軸方向都穩定)
  
Typical range:
  λ ∈ [-0.05, 0] (weakly stable to marginal)
  
Gate threshold:
  mean(λ) < 0 AND quantile_75(λ) < 0
```

### 衰減速率 (Decay Rate)

```
Expected pattern:
  Exponential decay: ||ΔP(t)|| = A × exp(-λ × t)
  
Half-life: t_half = ln(2) / λ
  Typical: t_half ~ 20-50 rounds
```

---

## Gate 檢驗標準

| Gate | 標準 | 預期結果 |
|------|------|--------|
| **G7G-01** | 39/39 runs 完成 | ✓ PASS |
| **G7G-02** | cosine_sim > 0.99 vs P7-F | ✓ PASS |
| **G7G-03** | mean(τ) < 150 rounds | ✓ PASS |
| **G7G-04** | mean(λ) < 0 | ✓ PASS |
| **G7G-05** | cross-seed CV < 50% | ✓ PASS |

---

## 進度檢查點

### ✓ 已完成
- [x] 規格文檔 (P7G_perturbation_analysis.md)
- [x] 執行腳本 (run_p7g_perturbation_analysis.py)
- [x] 語法檢查通過
- [x] 執行啟動

### 🔄 進行中
- [ ] Phase 1: 3 base runs (預計 15 min)
- [ ] Phase 2: 36 perturbation runs (預計 2 hrs)
- [ ] 分析與 Gate 檢驗 (預計 20 min)

### 待執行
- [ ] P7-G 完整分析報告撰寫
- [ ] 研發日誌更新 (P7-G 結果章節)
- [ ] 決定 P7-H 方向 (可選)

---

## 實時監控命令

```bash
# 檢查輸出目錄
ls -lah reports/experiments/p7g_perturbation_analysis/

# 檢查進度 (tail output)
tail -f <output_file>

# 驗證 Gate 結果
cat reports/experiments/p7g_perturbation_analysis/p7g_gates.json
```

---

## 執行完成後立即行動清單

1. **檢查 Gate 結果** (1 min)
   ```bash
   cat reports/experiments/p7g_perturbation_analysis/p7g_gates.json | jq .
   ```

2. **驗證關鍵指標** (5 min)
   ```bash
   # Recovery times
   csvcut -c recovery_time reports/experiments/p7g_perturbation_analysis/p7g_perturbation_summary.csv | describe
   
   # Lyapunov exponents
   cat reports/experiments/p7g_perturbation_analysis/p7g_lyapunov_analysis.json | jq '.[] | .mean'
   ```

3. **生成快速摘要** (10 min)
   - 確認所有軸的 λ_max < 0
   - 驗證恢復時間分佈
   - 檢查 cross-seed 一致性

4. **撰寫 P7-G 分析報告** (30-60 min)
   - 執行摘要
   - Gate 檢驗結果
   - Lyapunov 分析
   - 恢復動力學
   - 科學結論

5. **更新研發日誌** (10 min)
   - 加入 P7-G 執行結果
   - 更新進度狀態

---

## 相關檔案

| 檔案 | 狀態 |
|------|------|
| P7G_perturbation_analysis.md | ✅ 規格完成 |
| run_p7g_perturbation_analysis.py | ✅ 實現完成 |
| READY_FOR_P7G_EXECUTION.md | ✅ 準備清單 |
| p7g_perturbation_summary.csv | 🔄 執行中... |
| p7g_gates.json | 🔄 執行中... |
| p7g_perturbation_analysis_report.md | ⏳ 待撰寫 |

---

**狀態**: 🔄 **進行中** — 預計 ~2.5 小時完成

監控進度中...
