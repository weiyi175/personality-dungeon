# P7 實驗數據彙整 - 快速指南

> 一站式查詢 P7-A 到 P7-H 的實驗數據、報告與分析結果

**組織位置**: `/home/user/personality-dungeon/reports/experiments/`

---

## 📊 實驗階段一覽

### ✅ 已完成的階段

| 階段 | 資料夾 | 運行數 | 主要發現 | 快速訪問 |
|------|--------|--------|--------|---------|
| **P7-A** | `p7a_baseline/` | 18 | 基線耦合 6-7 pp ✓ | [報告](p7a_baseline/p7a_baseline_report.md) |
| **P7-C** | `p7c_feedback_injection/` | 8 | α 線性調控 ✓ | [報告](p7c_feedback_injection/p7c_feedback_report.md) |
| **P7-D** | `p7d_stability_scan/` | 15 | S1/S2/S4 三域 ✓ | [數據](p7d_stability_scan/p7d_stability_summary.csv) |
| **P7-E** | `p7e_long_trajectory/` | 10 | 1D 固定點吸引子 ✓ | [數據](p7e_long_trajectory/p7e_long_trajectory_summary.csv) |
| **P7-F** | `p7f_attractor_mapping/` | 21 | R²=1.0 線性映射 ✓ | [座標](p7f_attractor_mapping/p7f_attractor_coordinates.json) |
| **P7-G** | `p7g_perturbation_analysis/` | 39 | 邊界穩定 ε_c≈0.007 ✓ | [報告](p7g_perturbation_analysis/P7G_ANALYSIS_REPORT.md) |

### 🔄 規劃中的階段

| 階段 | 資料夾 | 狀態 | 概述 |
|------|--------|------|------|
| **P7-B** | `p7b_personality_groups/` | 📋 規劃 | [設計規格](p7b_personality_groups/p7b_design_spec.md) |
| **P7-H** | `p7h_landscape_explorer/` | 🚀 就緒 | [實驗設計](p7h_landscape_explorer/P7H_EXPERIMENT_DESIGN.md) |

---

## 🗂️ 資料夾結構

```
reports/experiments/
│
├── P7_EXPERIMENTS_INDEX.md          ← 【主索引】完整文檔與文件位置
│
├── p7a_baseline/                    ✅ 完成
│   ├── p7a_baseline_report.md
│   ├── p7a_baseline_summary.csv
│   ├── p7a_gates.json
│   └── run_*.json (18 files)
│
├── p7b_personality_groups/          🔄 規劃中
│   └── p7b_design_spec.md
│
├── p7c_feedback_injection/          ✅ 完成
│   ├── p7c_feedback_report.md
│   ├── p7c_feedback_summary.csv
│   ├── p7c_gates.json
│   └── run_*.json (8 files)
│
├── p7d_stability_scan/              ✅ 完成
│   ├── p7d_stability_summary.csv
│   ├── p7d_gates.json
│   └── run_*.json (15 files)
│
├── p7e_long_trajectory/             ✅ 完成
│   ├── p7e_long_trajectory_summary.csv
│   ├── p7e_gates.json
│   └── run_*[.json + _trajectory.csv] (20 files)
│
├── p7f_attractor_mapping/           ✅ 完成
│   ├── p7f_attractor_mapping_summary.csv
│   ├── p7f_attractor_coordinates.json
│   ├── p7f_subspace_analysis.json
│   ├── p7f_bifurcation_analysis.json
│   ├── p7f_gates.json
│   ├── p7h_principal_vectors.json   ← (P7-H 用)
│   └── run_*[.json + _trajectory.csv] (42 files)
│
├── p7g_perturbation_analysis/       ✅ 完成
│   ├── P7G_ANALYSIS_REPORT.md
│   ├── p7g_perturbation_summary.csv
│   ├── p7g_perturbation_axes.json
│   ├── p7g_lyapunov_analysis.json
│   └── p7g_gates.json
│
└── p7h_landscape_explorer/          🚀 準備就緒
    ├── P7H_EXPERIMENT_DESIGN.md
    ├── P7H_LAUNCH_CHECKLIST.md       (待建立)
    └── [Phase 1+2+3 產出] (待執行)
```

---

## 🔍 快速查詢

### 按主題查找

**想了解穩定性?**
- 📊 數據: [p7d_stability_scan/](p7d_stability_scan/) - 15 個α值的穩定性掃描
- 📊 數據: [p7g_perturbation_analysis/](p7g_perturbation_analysis/) - Lyapunov 指數
- 📄 分析: [P7G_ANALYSIS_REPORT.md](p7g_perturbation_analysis/P7G_ANALYSIS_REPORT.md)

**想看吸引子結構?**
- 📊 座標: [p7f_attractor_coordinates.json](p7f_attractor_mapping/p7f_attractor_coordinates.json) - 21 個吸引子位置
- 📊 子空間: [p7f_subspace_analysis.json](p7f_attractor_mapping/p7f_subspace_analysis.json) - 99.93% 方差 1D 子空間
- 📈 軌跡: `p7e_long_trajectory/*_trajectory.csv` - 完整 1000 round 軌跡

**想查反饋效應?**
- 📊 數據: [p7c_feedback_summary.csv](p7c_feedback_injection/p7c_feedback_summary.csv) - α 效應
- 📄 分析: [p7c_feedback_report.md](p7c_feedback_injection/p7c_feedback_report.md)

**想看 Gate 結果?**
- 各資料夾內的 `p7*_gates.json`

### 按檔案類型查找

| 檔案類型 | 位置 | 內容 |
|---------|------|------|
| 完整報告 | `p7*/p7*_report.md` | 科學分析、圖表、結論 |
| 摘要統計 | `p7*/p7*_summary.csv` | 聚合結果 (1 row/run) |
| 運行結果 | `p7*/run_*.json` | 個別運行的完整數據 |
| 軌跡數據 | `p7e/`, `p7f/`, `p7h/` | `run_*_trajectory.csv` (時間序列) |
| 分析文件 | `p7*/p7*_[xyz].json` | SVD、分岔、Lyapunov 等 |
| Gate 檢驗 | `p7*/p7*_gates.json` | 階段驗收標準結果 |

---

## 📈 數據流程與依賴關係

```
       ┌─────────────────────────────────────┐
       │         P7-A: 基線耦合 (18)        │
       │  ✓ 確認人格-策略耦合 6-7 pp       │
       └────────────┬────────────────────────┘
                    │
       ┌────────────▼────────────┐
       │  P7-C: 反饋驗證 (8)    │
       │  ✓ α 線性調控驗證      │
       └────────────┬────────────┘
                    │
       ┌────────────▼──────────────────────┐
       │  P7-D: 穩定性掃描 (15)           │
       │  ✓ S1/S2/S4 三域識別             │
       │  └→ SVD 分析 ──→ P7-H 主軸       │
       └────────────┬──────────────────────┘
                    │
       ┌────────────▼──────────────────────┐
       │  P7-E: 軌跡特徵化 (10)           │
       │  ✓ 1D 固定點吸引子確認            │
       └────────────┬──────────────────────┘
                    │
       ┌────────────▼──────────────────────────┐
       │  P7-F: 空間映射 (21)                │
       │  ✓ α-FP 曲線 (R²=1.0)             │
       │  ✓ 99.93% 方差 1D 子空間            │
       │  └→ SVD 提取 ──→ P7-H 網格生成      │
       └────────────┬──────────────────────────┘
                    │
       ┌────────────▼────────────────────────┐
       │  P7-G: 穩定性分析 (39)             │
       │  ✓ 邊界穩定, ε_critical ≈ 0.007  │
       └────────────┬────────────────────────┘
                    │
       ┌────────────▼───────────────────────────┐
       │  P7-H: 相空間探索 (1,681)            │
       │  🚀 就緒，待執行 June 5             │
       │  → 1D 網格掃描，多吸引子定位        │
       └──────────────────────────────────────┘
```

---

## 🚀 即將執行

### P7-H 相空間景觀探索

**狀態**: 🟢 所有準備完成  
**預計執行**: June 5, 2025 (09:00-17:00)  
**預計完成**: June 6-9, 2025  

**執行方式**:
```bash
cd /home/user/personality-dungeon
./venv/bin/python scripts/experiments/run_p7h_landscape_explorer.py \
    --alpha 0.2 --seed 42 --phase 1 --out reports/experiments/p7h_landscape_explorer
```

**預期發現**:
- 多吸引子結構映射
- 相空間分割邊界
- 吸引子盆地大小估計

---

## 📋 統計總覽

| 項目 | 數值 |
|------|------|
| **總完成運行** | 111 (A+C+D+E+F+G) |
| **總 rounds** | ~17,700 (包含所有階段) |
| **最大網格點** | 1,681 (P7-H Phase 1) |
| **總參數掃描** | α: 14 個值, seeds: 5+ 個 |
| **軌跡數據** | 41 個 trajectory CSVs (~200K 行) |
| **分析報告** | 6 個完整 MD 報告 |
| **JSON 分析** | 15+ 個專項分析文件 |

---

## 🔗 相關文檔

- **主索引**: [P7_EXPERIMENTS_INDEX.md](P7_EXPERIMENTS_INDEX.md) - 完整文檔說明
- **研發日誌**: ../../研發日誌.md - 實驗進度與決策記錄
- **系統設計**: ../../SDD.md - 系統規格與架構
- **P7-H 決策**: ../../P7H_DECISION_FRAMEWORK.md - 下一階段決策依據
- **P7-H 清單**: ../../P7H_LAUNCH_CHECKLIST.md - 執行前檢查清單

---

## 💡 使用建議

1. **快速概覽**  
   → 讀本文件 (2 分鐘)

2. **詳細查詢**  
   → 查 [P7_EXPERIMENTS_INDEX.md](P7_EXPERIMENTS_INDEX.md) (5-10 分鐘)

3. **深入分析**  
   → 開各階段的 `p7*_report.md` (20-60 分鐘)

4. **數據查詢**  
   → 用 CSV/JSON 查看工具開啟 `p7*_summary.csv` 或 `run_*.json`

5. **執行 P7-H**  
   → 參考 [P7H_EXPERIMENT_DESIGN.md](p7h_landscape_explorer/P7H_EXPERIMENT_DESIGN.md) 中的「一鍵啟動」

---

**彙整日期**: 2025-06-04  
**版本**: P7-INDEX-QUICKSTART v1.0  
**維護者**: Research Pipeline
