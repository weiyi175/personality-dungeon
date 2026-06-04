# P7 系列實驗數據彙整索引

> **統一管理**：P7-A 至 P7-H 所有階段的實驗數據、報告、分析結果

**更新日期**: 2025-06-04  
**組織方式**: 按階段分資料夾 (`p7a_*/`, `p7b_*/`, ..., `p7h_*/`)  
**數據根目錄**: `/reports/experiments/`

---

## 📊 P7 階段概覽

| 階段 | 名稱 | 狀態 | 數據夾 | 運行數 | 主要產出 |
|------|------|------|--------|--------|--------|
| **P7-A** | 基線耦合 | ✅ 完成 | `p7a_baseline/` | 18 | 基線效應確認 (6-7 pp) |
| **P7-B** | 人格分組 | ⏸️ 規劃中 | `p7b_personality_groups/` | - | （待執行） |
| **P7-C** | 反饋驗證 | ✅ 完成 | `p7c_feedback_injection/` | 8 | α 線性調控驗證 |
| **P7-D** | 穩定性掃描 | ✅ 完成 | `p7d_stability_scan/` | 15 | S1/S2/S4 三域識別 |
| **P7-E** | 軌跡特徵化 | ✅ 完成 | `p7e_long_trajectory/` | 10 | 1D 固定點吸引子確認 |
| **P7-F** | 空間映射 | ✅ 完成 | `p7f_attractor_mapping/` | 21 | α-FP 曲線 (R²=1.0) |
| **P7-G** | 穩定性分析 | ✅ 完成 | `p7g_perturbation_analysis/` | 39 | 邊界穩定、盆地 ε_c≈0.007 |
| **P7-H** | 相空間探索 | 🔄 準備中 | `p7h_landscape_explorer/` | - | （待執行）|

---

## 📁 各階段數據結構

### P7-A: 基線耦合 (`p7a_baseline/`)

**目標**: 驗證人格-策略耦合存在性  
**參數**: 4 seeds × 3 人格分組 = 12 runs (實驗時擴至 18)  
**時間**: 500 rounds/run

**包含文件**:
```
p7a_baseline/
├── p7a_baseline_report.md          ← 完整分析報告
├── p7a_baseline_summary.csv        ← 運行摘要統計
├── p7a_gates.json                  ← Gate 檢驗結果
├── run_42_G-AGG.json               ├─ seed 42 × 3 personality groups
├── run_42_G-BAL.json               │
├── run_42_G-DEF.json               │
├── run_43_G-AGG.json               ├─ seed 43 × 3 personality groups
├── run_43_G-BAL.json               │
├── run_43_G-DEF.json               │
├── run_44_G-AGG.json               ├─ seed 44 × 3 personality groups
├── run_44_G-BAL.json               │
├── run_44_G-DEF.json               │
├── run_45_G-AGG.json               ├─ seed 45 × 3 personality groups
├── run_45_G-BAL.json               │
├── run_45_G-DEF.json               │
├── run_46_G-AGG.json               ├─ seed 46 × 3 personality groups
├── run_46_G-BAL.json               │
└── run_46_G-DEF.json               │
```

**主要指標**: 人格-策略相關性, coupling_pp (pp = percentage points)

---

### P7-B: 人格分組 (`p7b_personality_groups/`)

**目標**: 識別人格空間中的自然分組 (計劃中)  
**狀態**: 🔄 未執行 (規格已定, 待資源分配)

**計劃內容**:
```
p7b_personality_groups/
├── p7b_design_spec.md              ← 設計規格
├── p7b_clustering_report.md        ← (待執行) 聚類分析
└── p7b_group_profiles.json         ← (待執行) 人格分組檔案
```

---

### P7-C: 反饋驗證 (`p7c_feedback_injection/`)

**目標**: 驗證人格反饋強度 α 的線性調控效應  
**參數**: 3 α 值 {0.0, 0.2, 0.5} × 2 seeds = 6 runs (實驗時擴至 8)  
**時間**: 250 rounds/run

**包含文件**:
```
p7c_feedback_injection/
├── p7c_feedback_report.md          ← 完整分析報告
├── p7c_feedback_summary.csv        ← 運行摘要統計
├── p7c_gates.json                  ← Gate 檢驗結果
├── run_a0.00_42.json               ├─ α=0.0 (無反饋)
├── run_a0.00_43.json               │
├── run_a0.20_42.json               ├─ α=0.2 (標準反饋)
├── run_a0.20_43.json               │
├── run_a0.50_42.json               ├─ α=0.5 (強反饋)
└── run_a0.50_43.json               │
```

**主要指標**: personality_velocity, personality_shift, α-依賴性

---

### P7-D: 穩定性掃描 (`p7d_stability_scan/`)

**目標**: 識別系統穩定域，α-穩定性映射  
**參數**: 5 α {0.1, 0.2, 0.3, 0.4, 0.5} × 3 seeds = 15 runs  
**時間**: 300 rounds/run

**包含文件**:
```
p7d_stability_scan/
├── p7d_stability_summary.csv       ← 運行摘要 (15 rows)
├── p7d_gates.json                  ← Gate 檢驗結果 (4/5 PASS)
├── run_a0.10_42.json               ├─ α=0.1
├── run_a0.10_43.json               │
├── run_a0.10_44.json               │
├── run_a0.20_42.json               ├─ α=0.2 (最穩定)
├── run_a0.20_43.json               │
├── run_a0.20_44.json               │
├── run_a0.30_42.json               ├─ α=0.3
├── run_a0.30_43.json               │
├── run_a0.30_44.json               │
├── run_a0.40_42.json               ├─ α=0.4
├── run_a0.40_43.json               │
├── run_a0.40_44.json               │
├── run_a0.50_42.json               ├─ α=0.5 (崩潰域)
├── run_a0.50_43.json               │
└── run_a0.50_44.json               │
```

**主要指標**: A_P (人格變化幅度), OSC (振盪), stability_class (S1/S2/S4)

---

### P7-E: 軌跡特徵化 (`p7e_long_trajectory/`)

**目標**: 驗證吸引子存在性，量化人格可塑性窗口  
**參數**: 2 α {0.2, 0.3} × 5 seeds = 10 runs  
**時間**: 1000 rounds/run

**包含文件**:
```
p7e_long_trajectory/
├── p7e_long_trajectory_summary.csv ← 運行摘要 (10 rows)
├── p7e_gates.json                  ← Gate 檢驗結果 (4/5 PASS)
├── run_a0.20_42.json               ├─ α=0.2 (最穩定)
├── run_a0.20_42_trajectory.csv     │ ├─ 完整軌跡 (1000 rows)
├── run_a0.20_43.json               │
├── run_a0.20_43_trajectory.csv     │
├── run_a0.20_101.json              │
├── run_a0.20_101_trajectory.csv    │
├── run_a0.20_102.json              │
├── run_a0.20_102_trajectory.csv    │
├── run_a0.30_42.json               ├─ α=0.3 (中等強度)
├── run_a0.30_42_trajectory.csv     │
├── run_a0.30_43.json               │
├── run_a0.30_43_trajectory.csv     │
├── run_a0.30_101.json              │
├── run_a0.30_101_trajectory.csv    │
├── run_a0.30_102.json              │
└── run_a0.30_102_trajectory.csv    │
```

**主要指標**: VDI (方差衰減), ACT (自相關時間常數), plasticity_window (可塑性窗口)

---

### P7-F: 空間映射 (`p7f_attractor_mapping/`)

**目標**: 完整刻畫 α-吸引子參數空間映射  
**參數**: 7 α {0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4} × 3 seeds = 21 runs  
**時間**: 200 rounds/run

**包含文件**:
```
p7f_attractor_mapping/
├── p7f_attractor_mapping_summary.csv  ← 運行摘要 (21 rows)
├── p7f_attractor_coordinates.json     ← 吸引子座標 (9D × 21 vectors)
├── p7f_subspace_analysis.json         ← SVD 子空間分析
├── p7f_bifurcation_analysis.json      ← 分岔點偵測
├── p7f_gates.json                     ← Gate 檢驗結果 (4/5 PASS)
├── p7h_principal_vectors.json         ← (P7-H 用) 主向量 v₁, v₂
│
├── run_a0.10_42.json                  ├─ α=0.1
├── run_a0.10_42_trajectory.csv        │
├── run_a0.10_43.json                  │
├── run_a0.10_43_trajectory.csv        │
├── run_a0.10_44.json                  │
├── run_a0.10_44_trajectory.csv        │
│
├── run_a0.15_42.json                  ├─ α=0.15
├── run_a0.15_42_trajectory.csv        │
├── run_a0.15_43.json                  │
├── run_a0.15_43_trajectory.csv        │
├── run_a0.15_44.json                  │
├── run_a0.15_44_trajectory.csv        │
│
├── [類似結構 α=0.2, 0.25, 0.3, 0.35, 0.4...]
│
└── run_a0.40_44_trajectory.csv        └─ α=0.4 (最後一個)
```

**主要指標**: ||attractor||, attractor_distance, 1D 子空間佔有率 (99.93%)

---

### P7-G: 穩定性分析 (`p7g_perturbation_analysis/`)

**目標**: 驗證 1D 子空間的吸引子穩定性，量化 Lyapunov 指數  
**參數**: α=0.2 固定, 2 擾動軸 × 3 scales × 2 directions × 3 seeds = 39 runs  
**時間**: 200 rounds/run

**包含文件**:
```
p7g_perturbation_analysis/
├── P7G_ANALYSIS_REPORT.md          ← 完整分析報告
├── p7g_perturbation_summary.csv    ← 運行摘要 (39 rows)
├── p7g_perturbation_axes.json      ← 擾動軸定義 (2-8 軸)
├── p7g_lyapunov_analysis.json      ← Lyapunov 指數分析
├── p7g_gates.json                  ← Gate 檢驗結果 (3/5 PASS)
│
├── run_00_axis00_eps0005_pos_seed42.json  ├─ Phase 2 完整結果
├── run_01_axis00_eps0005_pos_seed43.json  │ (36 runs)
├── ... (33 more runs)                     │
│
└── (Phase 3 data 若已執行)
```

**主要指標**: recovery_time (恢復時間), λ_max (Lyapunov), decay_rate (衰減速率), ε_critical (臨界擾動 ≈0.007)

---

### P7-H: 相空間探索 (`p7h_landscape_explorer/`)

**目標**: 定位多吸引子結構，探索相空間分割  
**參數**: 41×41 = 1,681 個網格點 (α=0.2, 主軸 v₁/v₂ 展開)  
**狀態**: 🔄 準備執行 (June 5, 2025 預計)

**預期輸出**:
```
p7h_landscape_explorer/
├── P7H_LAUNCH_CHECKLIST.md         ← 執行前清單
├── p7h_grid_scan_phase1.json       ← Phase 1: 1,681 點分類
├── p7h_phase1_summary.json         ← Phase 1 統計摘要
├── p7h_landscape_heatmap.png       ← 熱力圖可視化
├── p7h_attractor_map.json          ← 吸引子映射 (phase 1+2)
├── p7h_bifurcation_landscape.json  ← 分岔景觀 (phase 2+3)
│
└── [運行軌跡檔]                     (Phase 1: 1,681 trajectory CSVs)
```

**主要指標**: attractor_classification, basin_of_attraction, bifurcation_boundaries

---

## 🔧 快速訪問指南

### 按用途查找

**想找穩定性數據?**  
→ `p7d_stability_scan/` (α-穩定性映射)  
→ `p7g_perturbation_analysis/` (Lyapunov 指數)

**想看吸引子軌跡?**  
→ `p7e_long_trajectory/*_trajectory.csv` (1000 rounds 長軌跡)  
→ `p7f_attractor_mapping/*_trajectory.csv` (200 rounds, 21 runs)

**想查 Gate 結果?**  
→ 各文件夾內的 `p7*_gates.json`

**想讀完整分析報告?**  
→ `p7a_baseline_report.md`, `p7c_feedback_report.md`, `P7G_ANALYSIS_REPORT.md`

### 按數據類型查找

| 類型 | 位置 | 檔案模式 |
|------|------|--------|
| **運行結果** | `p7*/run_*.json` | 個別運行的完整數據 |
| **軌跡數據** | `p7e/`, `p7f/`, `p7h/` | `run_*_trajectory.csv` |
| **摘要統計** | `p7*/p7*_summary.csv` | 聚合統計 (1 row/run) |
| **Gate 檢驗** | `p7*/p7*_gates.json` | 階段性驗收標準 |
| **分析報告** | `p7*/p7*_report.md` | 完整科學分析 |
| **中間產物** | `p7f/, p7g/` | `.json` 分析文件 |

---

## 📈 數據管道示意

```
P7-A (基線)
  ↓
P7-C (反饋驗證)
  ↓
P7-D (穩定性掃描) ──→ SVD 分析 ──→ P7-H 主軸
  ↓
P7-E (長軌跡)
  ↓
P7-F (空間映射) ─────→ SVD 提取 ──→ P7-H 網格生成
  ↓
P7-G (穩定性分析)
  ↓
P7-H (相空間探索) ──→ (決策) ──→ P7-I (根本性研究) / 應用設計
```

---

## 📋 常見查詢

**Q: P7 系列共有多少個實驗運行?**  
A: P7-A(18) + P7-C(8) + P7-D(15) + P7-E(10) + P7-F(21) + P7-G(39) = **111 運行** (不含 P7-B, P7-H)

**Q: 哪個階段確認了 1D 吸引子?**  
A: P7-E (軌跡驗證) 首次確認，P7-F (空間映射) 完全驗證 (R²=1.0)

**Q: 哪個 α 值最穩定?**  
A: α=0.2 (S2 穩定域中心)，根據 P7-D 和 P7-E 確認

**Q: 吸引子盆地多大?**  
A: 根據 P7-G，臨界擾動幅度 ε_critical ≈ 0.007

**Q: P7-B 何時執行?**  
A: 待資源分配 (設計已完成，待排期)

---

## 🔗 關鍵文檔連結

- 📄 [研發日誌](../../研發日誌.md) - 完整的 P7 進度記錄
- 📋 [SDD.md](../../SDD.md) - 系統設計規格
- 📊 [P7H 決策框架](../../P7H_DECISION_FRAMEWORK.md) - 下一階段決策依據
- ⚙️ [P7H 執行清單](../../P7H_LAUNCH_CHECKLIST.md) - 執行前檢查清單

---

**維護者**: Research Pipeline  
**最後更新**: 2025-06-04 16:30  
**版本**: P7-INDEX v1.0
