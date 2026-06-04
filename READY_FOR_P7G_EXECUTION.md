# P7-G 執行準備清單

**規格版本**: P7-G v1.0 (2025-06-04)  
**狀態**: ✅ 設計完成，準備執行  
**審批**: 已通過語法檢查，可立即執行

---

## 📋 已完成的準備工作

### 1. 規格文檔 ✅
- 檔案: `docs/experiments/p7_online_personality_loop/P7G_perturbation_analysis.md`
- 內容: 科學問題、實驗設計、指標定義、Gate 標準、時間預算
- 版本: v1.0

### 2. 執行腳本 ✅
- 檔案: `scripts/experiments/run_p7g_perturbation_analysis.py` (~750 行)
- 語法檢查: ✓ 通過
- 核心函數:
  ```
  ✓ load_p7f_attractor()          # 加載 P7-F 吸引子數據
  ✓ compute_perturbation_axes()   # 計算 8 個正交軸 (Gram-Schmidt)
  ✓ compute_recovery_time()       # 測量擾動恢復時間
  ✓ compute_lyapunov_exponent()   # 計算 Lyapunov 指數
  ✓ compute_decay_metrics()       # 計算衰減速率與半衰期
  ✓ run_base() & run_perturbed()  # 執行 runs
  ✓ check_gates()                 # G7G-01~05 驗證
  ```

### 3. 研發日誌 ✅
- 已添加 P7-G 設計章節 (~300 行)
- 包含: 實驗設計、指標定義、預期結果、執行計劃

### 4. 待辦事項 ✅
- P7-G 設計標記為完成
- P7-G Phase 1+2 執行標記為準備中

---

## 🚀 立即執行指令

### 推薦: Phase 1+2 (中等規模，~2.5 小時)

```bash
cd /home/user/personality-dungeon

./venv/bin/python scripts/experiments/run_p7g_perturbation_analysis.py \
    --alpha 0.2 \
    --seeds 42 43 44 \
    --perturbation-scales 0.005 0.010 0.020 \
    --phase 2 \
    --out reports/experiments/p7g_perturbation_analysis
```

**執行詳情**:
```
Phase 1: 3 base runs (無擾動)
Phase 2: 36 perturbation runs (2 主要軸 × 3 scales × 2 directions × 3 seeds)
Total: 39 runs × 200 rounds = 7,800 total rounds
Time: ~2-2.5 hours
```

### 可選: 完整 Phase 1-3 (完整規模，~7 小時)

```bash
./venv/bin/python scripts/experiments/run_p7g_perturbation_analysis.py \
    --alpha 0.2 \
    --seeds 42 43 44 \
    --perturbation-scales 0.005 0.010 0.020 \
    --phase 3 \
    --out reports/experiments/p7g_perturbation_analysis
```

**執行詳情**:
```
Phase 1: 3 base runs
Phase 2: 36 perturbation runs (2 軸)
Phase 3: 96 additional runs (6 軸)
Total: 135 runs × 200 rounds = 27,000 total rounds
Time: ~6-7 hours
```

---

## 📊 P7-G 核心目標

| 目標 | 方法 | 預期結果 |
|------|------|--------|
| **驗證 1D 穩定性** | 測量垂直擾動衰減 | τ_recovery < 50 rounds |
| **計算 Lyapunov 指數** | 跟蹤擾動增長速率 | λ_max < 0 (穩定) |
| **刻畫恢復動力學** | 分析 τ(ε) 依賴性 | τ 獨立于 ε (線性系統) |
| **識別主方向** | 提取 SVD 軸坐標 | 9D 人格空間的幾何解釋 |

---

## 📈 預期產出 (Phase 1+2)

```
reports/experiments/p7g_perturbation_analysis/

✓ p7g_perturbation_summary.csv (39 rows × 12 metrics)
  - seed, is_perturbed, perturbation_axis, epsilon, direction
  - recovery_time, recovery_success, lyapunov_exponent, decay_rate, ...

✓ p7g_perturbation_axes.json (8 軸 × 9D)
  - axis_0, axis_1, ..., axis_7 (Gram-Schmidt 正交化)

✓ p7g_lyapunov_analysis.json
  - per-axis λ_max 統計 (mean, std, min, max)

✓ p7g_gates.json
  - G7G-01~05 檢驗結果

✓ p7g_gates.json 狀態範例
  {
    "G7G-01": true,                    # 執行完整性
    "G7G-02": true,                    # 基準穩定性 (vs P7-F)
    "G7G-03": true,                    # 擾動衰減
    "G7G-04": true,                    # Lyapunov 穩定性
    "G7G-05": true                     # 軸方向一致性
  }

✓ run_seed42_axis0_eps0.005_positive.json + trajectory.csv
  ... (36 perturbation run pairs)
```

---

## 🔄 與 P7-F 的銜接

### P7-F → P7-G 數據流

```
P7-F outputs/
├─ p7f_attractor_coordinates.json
│  └─ [P7-G loads] → baseline attractors
├─ p7f_subspace_analysis.json
│  └─ [P7-G loads] → SVD results (for axis computation)
└─ p7f_perturbation_axes.json (已生成的 8 軸)
   └─ [P7-G uses] → perturbation directions

P7-G verification:
  ✓ load_p7f_attractor(alpha=0.2) → 3 baseline attractors
  ✓ compute_perturbation_axes(p7f_attractors) → 8 orthonormal axes
  ✓ run_base() → 確認 α=0.2 下的吸引子位置
  ✓ run_perturbed() → 3 seeds × 8 axes × 3 scales × 2 directions
```

---

## ⏰ 時間估算

| 階段 | 項目 | 耗時 |
|------|------|-----|
| 準備 | 加載 P7-F, 計算正交軸 | 5 min |
| Phase 1 | 3 base runs (200 rounds each) | 15 min |
| Phase 2 | 36 perturbation runs (200 rounds each) | 2 hrs |
| 分析 | Lyapunov, decay rates, recovery times | 20 min |
| 報告 | 視覺化與結論撰寫 | 15 min |
| **Total Phase 1+2** | | **~2.5 hrs** |

---

## 🎯 P7-G 成功指標

### 全 Gate 通過的徵兆

✅ **執行完整** (G7G-01)
```
39 runs 全部完成，無超時或崩潰
```

✅ **基準一致** (G7G-02)
```
base run 吸引子 vs P7-F α=0.2 吸引子
Cosine similarity > 0.99 (高度一致)
```

✅ **快速恢復** (G7G-03)
```
mean(recovery_time) < 150 rounds
recovery_success rate > 90%
```

✅ **穩定指標** (G7G-04)
```
mean(λ_max) < 0 (負指數，指數衰減)
quantile_75(λ_max) < 0 (多數軸穩定)
```

✅ **一致性** (G7G-05)
```
同軸同 ε 的跨種子恢復時間
變異係數 CV < 0.5 (低變異性)
```

---

## 📝 後續步驟

### 執行後任務

1. **立即** (執行完成後 ~1 hr)
   - [ ] 檢查 Gate 結果
   - [ ] 快速審查 Lyapunov 統計
   - [ ] 驗證恢復時間分佈

2. **短期** (執行後 1-2 hr)
   - [ ] 撰寫 P7-G 完整分析報告
   - [ ] 繪製 λ_max vs axis, λ_max vs epsilon 圖
   - [ ] 解釋各軸方向的人格含義

3. **長期** (執行後 2-3 hr)
   - [ ] 更新研發日誌 P7-G 結果章節
   - [ ] 評估是否需要 P7-H (參數空間擴展 or 多人集體動力學)
   - [ ] 撰寫 P7 完整系列的最終總結

---

## 🔗 相關檔案位置

| 檔案 | 路徑 |
|------|------|
| 規格 | `docs/experiments/p7_online_personality_loop/P7G_perturbation_analysis.md` |
| 腳本 | `scripts/experiments/run_p7g_perturbation_analysis.py` |
| P7-F 吸引子 | `reports/experiments/p7f_attractor_mapping/p7f_attractor_coordinates.json` |
| P7-F SVD | `reports/experiments/p7f_attractor_mapping/p7f_subspace_analysis.json` |
| 研發日誌 | `研發日誌.md` (P7-G 章節已加入) |

---

## ✅ 檢查清單 (執行前確認)

- [x] 規格文檔完成 (P7G_perturbation_analysis.md)
- [x] 執行腳本完成 (run_p7g_perturbation_analysis.py)
- [x] 語法檢查通過
- [x] P7-F 輸出文件存在 (p7f_attractor_coordinates.json)
- [x] venv 環境可用
- [x] 輸出目錄可寫

**狀態**: ✅ **準備完成，可立即執行**

---

**推薦執行時間**: 立即開始 Phase 1+2 (預計 2.5 小時內完成)

**命令**:
```bash
cd /home/user/personality-dungeon && ./venv/bin/python scripts/experiments/run_p7g_perturbation_analysis.py --phase 2
```
