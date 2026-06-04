# P7-H: 相空間景觀探索 (Phase Space Landscape Explorer)

## 概述

**階段**: P7-H  
**目標**: 完整探索 α=0.2 條件下的相空間，定位多吸引子結構，探索盆地邊界  
**狀態**: 🔄 準備就緒，待執行  
**預計執行**: June 5-9, 2025  
**預計時間**: Phase 1+2 ≈ 5-6 小時

---

## 核心問題

基於 P7-F 發現的 1D 簡樸性（99.93% 方差由第 1 主軸捕捉），P7-H 深化探索：

1. **1D 簡樸性的邊界在哪裡?**
   - 相空間是否存在多個吸引子?
   - 若有，如何分布？

2. **吸引子盆地的大小與形狀?**
   - 從主吸引子逃逸需要多大的擾動?
   - 盆地邊界的光滑度如何?

3. **相空間的分割結構?**
   - 不同吸引子間的分界線在哪裡?
   - 是否存在邊界混沌或分形?

---

## 實驗設計

### Phase 1: 粗網格掃描 (1,681 點)

**網格定義** (2D 平面，v₁-v₂ 展開):
```
中心: P₀(α=0.2) ≈ [0.0913, 0, 0, ...]  (來自 P7-F)

展開軸:
  v₁: 第 1 主成分 (99.93% 方差)
  v₂: 第 2 主成分 (0.065% 方差)

網格:
  v₁ 方向: [-0.020, +0.020], 41 點 (Δ = 0.001)
  v₂ 方向: [-0.020, +0.020], 41 點 (Δ = 0.001)
  總點數: 41 × 41 = 1,681

初始化:
  P_init = P₀ + ε₁·v₁ + ε₂·v₂  (其他 7D 沿 P₀)
```

**分類方案**:
```
執行 200 rounds 後，根據最終人格向量 P_final：

主吸引子區 (Attractor A):
  distance(P_final, A_center) < 0.010
  
次級吸引子區 (Attractor B):
  distance(P_final, A_center) ∈ [0.100, 0.120]

遠程區 (Remote):
  distance(P_final, A_center) > 0.150

過渡區 (Transition):
  其他
```

**產出**:
```
p7h_grid_scan_phase1.json:
  {
    "metadata": { "alpha": 0.2, "n_points": 1681, "n_rounds": 200 },
    "grid": [
      { "grid_idx": 0, "eps1": -0.020, "eps2": -0.020, 
        "classification": "remote", "distance": 0.285, "stability": "S4" },
      { "grid_idx": 1, "eps1": -0.019, "eps2": -0.020, 
        "classification": "main", "distance": 0.008, "stability": "S2" },
      ...
    ]
  }

p7h_phase1_summary.json:
  {
    "main_attractor": { "count": 850, "fraction": 50.6%, "center": [...] },
    "secondary_attractor": { "count": 280, "fraction": 16.7%, "center": [...] },
    "remote": { "count": 450, "fraction": 26.8% },
    "transition": { "count": 121, "fraction": 7.2% },
    "boundary_points": 300
  }
```

### Phase 2: 邊界精細掃描 (50-100 新增點)

**觸發條件** (基於 Phase 1 結果):
- 若邊界清晰且相鄰點分類一致 → 跳過 Phase 2
- 若邊界複雜或存在孤立點 → 執行 Phase 2

**掃描策略**:
- 在 Phase 1 邊界點周圍增加 20-30 倍解析度 (Δ = 0.00005)
- 目標：定位精確的盆地邊界

### Phase 3: Lyapunov 譜分析 (可選)

**目標** (若 Phase 1+2 表明存在複雜結構):
- 計算代表性軌跡的 Lyapunov 指數譜
- 驗證是否存在邊界混沌

---

## 執行前檢查清單

- [x] **環境準備**
  - [x] venv 環境確認
  - [x] 依賴完整 (numpy, scipy, sklearn, joblib)
  - [x] 儲存空間充足 (> 10 GB)

- [x] **數據依賴**
  - [x] P7-F SVD 分析完成 (v₁, v₂ 提取)
  - [x] 基準吸引子已定位 (P₀ 座標已知)
  - [x] SBERT 模型快取 (paraphrase-multilingual-MiniLM-L12-v2)
  - [x] MLP v7 模型驗證

- [x] **腳本準備**
  - [x] `scripts/experiments/run_p7h_landscape_explorer.py` 實現完成
  - [x] 語法檢查通過
  - [x] API 對接驗證 (RLSessionEngine, personality inference)

---

## 一鍵啟動

```bash
# Phase 1: 1,681 點網格掃描 (3-6 小時)
cd /home/user/personality-dungeon && \
./venv/bin/python scripts/experiments/run_p7h_landscape_explorer.py \
    --alpha 0.2 \
    --seed 42 \
    --phase 1 \
    --resolution 0.001 \
    --radius 0.020 \
    --out reports/experiments/p7h_landscape_explorer

# Phase 2: 邊界精細掃描 (若需要，~2-3 小時)
./venv/bin/python scripts/experiments/run_p7h_landscape_explorer.py \
    --phase 2 \
    --out reports/experiments/p7h_landscape_explorer

# Phase 3: Lyapunov 譜 (若需要，~1-2 小時)
./venv/bin/python scripts/experiments/run_p7h_landscape_explorer.py \
    --phase 3 \
    --out reports/experiments/p7h_landscape_explorer
```

---

## 成功標準

✅ **完全成功**:
- Phase 1: 1,681 個點無例外完成
- ≥ 70% 點分類明確 (非 transition)
- 邊界清晰 (相鄰點分類一致 > 80%)
- 發現 2-3 個主要吸引子區域

✅ **基本成功**:
- Phase 1: ≥ 95% 點成功完成
- 60-70% 點分類明確
- 邊界複雜但可追蹤

⚠️ **需改進**:
- < 80% 點成功完成
- 無明確吸引子結構
- 邊界完全不清晰

---

## 後續決策樹

```
IF (邊界清晰 ∧ 次級吸引子明確):
  ✓ 升級 Phase 2+3
  → 完整描述相空間
  → June 6-9 完成全景

ELSE IF (邊界模糊 ∨ 結構複雜):
  ⚠️ Phase 2 精細掃描
  → 提高解析度
  → 或發現分形結構

ELSE IF (無清晰結構):
  ✗ 需要參數調優
  → 嘗試其他 α 值
  → 或進入 P7-I 根本研究
```

---

## 預期產出

```
reports/experiments/p7h_landscape_explorer/
├── P7H_LAUNCH_CHECKLIST.md          ← 執行前檢查
├── p7h_grid_scan_phase1.json        ← 1,681 點分類
├── p7h_phase1_summary.json          ← 統計摘要
├── p7h_landscape_heatmap.png        ← 熱力圖
├── p7h_attractor_classification.json ← 吸引子映射
├── p7h_gates.json                   ← Gate 檢驗
└── [軌跡 CSVs]                       (可選，1,681 × 200 rows)
```

---

**版本**: P7-H 設計規格 v1.0  
**狀態**: 🔄 準備就緒，待執行授權  
**預計執行**: June 5, 2025 (09:00)  
**預計完成**: June 9, 2025 (17:00)
