# Grain 分析報告（9D→3-archetype 是否藏多樣性）

```
======================================================================
  GRAIN 分析：9D will-space → 3-archetype 投影是否藏多樣性  (N=199)
======================================================================

[M1] PCA 有效維度（原始單位、置中、未標準化）
   PC 變異占比 : PC1=38.2%  PC2=20.1%  PC3=12.6%  PC4= 7.3%  PC5= 6.2%  PC6= 5.3%  PC7= 4.1%  PC8= 3.2%  PC9= 3.1%
   累積 top-2 : 58.3%   top-3 : 70.9%
   participation ratio（有效維度）: 4.60 / 9
   PC1 主導 : stability_seeking+0.54  risk_aversion+0.51  impulsiveness-0.48
   PC2 主導 : optimism-0.58  suspicion+0.54  endurance-0.34

[M2] GRAIN 盲區：9D 變異可從 archetype 投影回收的比例
   精確可見子空間 (score 差, 2D) R² : 46.2%   → 盲區 53.8%
   操作 soft 權重 (softmax 後) R²   : 36.8%   → 盲區 63.2%
   ★ 隱藏多樣性（盲區，取精確值）  : 53.8%  ← 生態層看不到的 will 變異

[M3] per-feature 可見度（從 3-soft 回收各特徵的 R²；低＝被藏）
   endurance        R²=  6.8%  (std=0.334)  ← 幾乎不可見
   suspicion        R²= 15.3%  (std=0.392)
   randomness       R²= 15.6%  (std=0.411)
   optimism         R²= 20.3%  (std=0.379)
   assertiveness    R²= 34.6%  (std=0.331)
   curiosity        R²= 36.3%  (std=0.293)
   stability_seeking R²= 49.4%  (std=0.477)
   impulsiveness    R²= 53.6%  (std=0.461)
   risk_aversion    R²= 62.4%  (std=0.500)

[M4] 軸對齊：損失是「2D 太少」還是「archetype 軸選錯」（兩者皆 2D，公平比）
   最佳線性 2D (PCA top-2) 可回收     : 58.3%   (3D: 70.9%)
   archetype 可見 2D (score 差) 可回收 : 46.2%
   → 軸選擇損失 (best-2D − archetype) : +12.1 pt（archetype 軸偏向 PC1、欠讀 PC2）
     降維不可逆損失 (100% − best-2D) : 41.7 pt（4.6 維攤不進 2D）

[M5] grain 粗度：hard argmax 3-箱的變異拆解
   箱大小 : aggressive=88  defensive=97  balanced=14
   within-bin 變異占比 : 74.3%   between-bin : 25.7%
   → hard 標籤丟掉 74.3% 的 9D 變異（箱內連續差異）

----------------------------------------------------------------------
裁決：⚠ grain 藏多樣性：54% 的 will 變異落在 archetype 投影盲區。 生態層的 diversity/monoculture ≠ will-space 真實多樣性。
======================================================================
```
