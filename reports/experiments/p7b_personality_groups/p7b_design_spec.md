# P7-B: 人格分組特徵化 (Personality Group Characterization)

## 概述

**階段**: P7-B (介於 P7-A 基線與 P7-C 反饋驗證之間的計劃研究)  
**目標**: 在 9D 人格空間中識別自然分組，探索人格的內在結構與聚類特徵  
**狀態**: 🔄 規劃中 (未執行)  
**預計時間**: TBD (待資源分配)

---

## 核心問題

1. **人格空間是否存在自然聚類?**
   - 是否能識別出相對離散的人格分組?
   - 分組之間的邊界是否明確?

2. **人格分組與遊戲行為的關聯性**
   - 不同分組在基線耦合上的表現是否有差異?
   - 是否存在「人格-行為型」?

3. **聚類的穩定性與魯棒性**
   - 跨 seeds 的聚類結果一致性如何?
   - 聚類邊界對初始條件的敏感性?

---

## 實驗設計

**Phase 1: 無監督聚類**

| 參數 | 值 |
|------|-----|
| 數據來源 | P7-A 所有 18 runs 的最終人格向量 |
| 聚類方法 | K-means, GMM, Hierarchical (對比分析) |
| K 範圍 | 2 ~ 8 (確定最優分組數) |
| 降維可視化 | t-SNE, UMAP, PCA (3D/2D) |
| 評估指標 | Silhouette, Davies-Bouldin, Calinski-Harabasz |

**Phase 2: 分組特徵提取**

| 項目 | 內容 |
|------|------|
| 分組中心 | 各分組的平均人格向量 |
| 分散度 | 組內標準差、相對大小 |
| 特徵向量 | PCA 主成分，解釋 K 個分組的差異 |
| 邊界分析 | 邊界點與分組中心的距離分布 |

**Phase 3: 行為映射**

| 映射 | 內容 |
|------|------|
| 人格 → 策略 | 不同分組的平均策略特徵 |
| 人格 → 收益 | 不同分組的平均 payoff |
| 人格 → 穩定性 | 與後期 P7-D 穩定域的對應關係 |

---

## 預期輸出

```
p7b_personality_groups/
├── p7b_design_spec.md               ← 本文件
├── p7b_clustering_analysis.md       ← 聚類分析完整報告
├── p7b_clustering_summary.json      ← 聚類結果摘要
│
├── p7b_group_profiles.json          ← 人格分組檔案
│   ├── group_0: {center, size, members, characteristics}
│   ├── group_1: ...
│   └── group_N: ...
│
├── p7b_silhouette_analysis.json     ← 不同 K 值的評估指標
├── p7b_pca_variance.json            ← PCA 方差解釋
│
├── visualization/
│   ├── pca_2d_scatter.png
│   ├── tsne_2d_scatter.png
│   ├── dendrogram_hierarchical.png
│   └── silhouette_plot.png
│
└── p7b_gates.json                   ← Gate 檢驗結果
    ├── G7B-01: 聚類完整性 (N clusters identified)
    ├── G7B-02: 分組一致性 (cross-seed silhouette > 0.5)
    ├── G7B-03: 行為相關性 (personality-strategy correlation)
    └── G7B-04: 邊界穩定性 (boundary consistency)
```

---

## Gate 標準 (建議)

| Gate | 標準 | 預期結果 |
|------|------|--------|
| **G7B-01** | 最優聚類數 K 明確 | 2 ≤ K ≤ 6 |
| **G7B-02** | 平均 Silhouette ≥ 0.5 | ✓ 分組良好 |
| **G7B-03** | 跨 methods 一致性 | K-means vs GMM 差異 < 20% |
| **G7B-04** | 行為相關性 R² ≥ 0.6 | 人格-策略可解釋 |

---

## 備註

- **為何放在 P7-A 之後？** P7-B 需要 P7-A 的完整數據作為輸入
- **為何不在 P7-C 之前？** P7-B 是可選的分析，主線進程是 P7-C (反饋驗證)
- **與 P7 主線的關係**: P7-B 是「橫向補充」，不阻擋主線進展

---

**版本**: P7-B 設計規格 v0.1  
**狀態**: 🔄 待執行  
**更新日期**: 2025-06-04
