# P7-H 執行前最後檢查清單 & 啟動指令

**狀態**: 🟢 **全部就緒，可隨時啟動**  
**日期**: 2025-06-04  
**預計執行**: June 5 早上

---

## ✅ 後端集成確認

### 已驗證的依賴

| 模塊 | 位置 | 狀態 | API |
|------|------|------|-----|
| RLSessionEngine | `simulation/rl_session_engine.py` | ✓ | `step()` → FrameSnapshot |
| 個性推斷 | `api/personality_sbert_inference.py` | ✓ | `infer_personality_vector_sbert(text)` |
| SVD 向量 | `reports/experiments/p7f_attractor_mapping/` | ✓ | `p7h_principal_vectors.json` |
| P7-F 吸引子 | `reports/experiments/p7f_attractor_mapping/` | ✓ | `p7f_attractor_coordinates.json` |

### 環境確認

```bash
✓ Python venv 活躍
✓ SBERT 模型緩存就位
✓ MLP 模型加載 (mlp_v7_mlp.joblib)
✓ NumPy/SciPy 依賴完整
✓ 儲存空間充足 (1,681 runs × 200 rounds ≈ 500 MB)
```

---

## 📋 執行前最終檢查清單

### 技術檢查

```
後端集成:
  ✓ RLSessionEngine 初始化可用
  ✓ 個性推斷 API 可用
  ✓ SVD 向量已加載
  ✓ P7-H 腳本語法通過
  ✓ 主特徵向量已提取 (v₁, v₂)

數據準備:
  ✓ P7-F 吸引子座標完整 (21 runs × 9D)
  ✓ P7-F SVD 分析完成 (σ₁=99.93%, σ₂=0.06%)
  ✓ 基準吸引子 P₀(α=0.2) 已定位
  ✓ 2D 網格配置已設計 (41×41 = 1,681 點)

參數驗證:
  ✓ α = 0.2 (P7-D/E/F 驗證的最穩定值)
  ✓ n_rounds = 200 (充分達到吸引子)
  ✓ burn_in = 50 (10% 標準)
  ✓ 3 seeds {42, 43, 44} (P7-G 驗證的健壯性種子)
```

### 資源檢查

```
計算資源:
  ✓ CPU: 標準 venv 環境充足
  ✓ RAM: 預計峰值 ~2-4 GB (可管理)
  ✓ 儲存: 剩餘空間 > 10 GB (充足)
  ✓ 執行時間預算: 5-6 小時 (含 Phase 1+2)

GPU (若可用):
  ○ SBERT 推斷已優化 (CPU 模式主要)
  ○ 若有 CUDA 可自動加速
```

### 文檔檢查

```
規格完整性:
  ✓ P7H_landscape_explorer.md (設計文檔)
  ✓ P7H_DECISION_FRAMEWORK.md (決策框架)
  ✓ P7H_READINESS_NOTICE.md (準備就緒通知)
  ✓ run_p7h_landscape_explorer.py (執行腳本)
  ✓ p7h_principal_vectors.json (主向量數據)

進度追蹤:
  ✓ 研發日誌已更新 (P7-A ~ P7-G 完整記錄)
  ✓ 進度清單已更新
  ✓ 決策框架已確立
```

---

## 🚀 Phase 1 執行指令 (一鍵啟動)

### 快速啟動

```bash
cd /home/user/personality-dungeon && \
./venv/bin/python scripts/experiments/run_p7h_landscape_explorer.py \
    --alpha 0.2 \
    --seed 42 \
    --phase 1 \
    --resolution 0.001 \
    --radius 0.020 \
    --out reports/experiments/p7h_landscape_explorer
```

**預期執行時間**: 3-6 小時  
**預期輸出**: 
- `p7h_grid_scan_phase1.json` (1,681 點分類)
- `p7h_phase1_summary.json` (統計摘要)
- `p7h_landscape_heatmap.png` (可視化)

### 監控進度

```bash
# 在另一個終端監控
tail -f reports/experiments/p7h_landscape_explorer/*.log

# 或檢查已完成的點數
ls reports/experiments/p7h_landscape_explorer/run_grid_*.csv | wc -l
```

### 失敗恢復

若執行中斷，可從上一個完成的網格點恢復：

```bash
# 檢查已完成的最高點
ls reports/experiments/p7h_landscape_explorer/ | grep "grid_" | tail -1

# 修改腳本，從該點繼續 (進階用法)
# 或重新啟動 (時間允許情況下)
```

---

## 📊 Phase 1 完成後的決策樹

### 決策點 1: 邊界清晰度？

```
IF 邊界清晰 (主吸引子区域 > 50%, 邊界銳化):
  ✓ 升級 Phase 2 邊界精細掃描
  → 時間: 1.5 小時
  → 輸出: p7h_boundary_scan_summary.csv

ELSE IF 邊界模糊 (多個區域混淆):
  ⚠️ 需要提高解析度
  → 改用 Δε = 0.0005 (更細)
  → 時間: 額外 3-4 小時

ELSE IF 完全分散 (無清晰結構):
  ✗ 系統複雜性超預期
  → 進入 P7-I 深度探索
  → 或調整參數策略
```

### 決策點 2: 次級吸引子？

```
IF 發現明確的次級吸引子 (聚點在距 ≈ 0.11):
  🎉 核心發現確認！
  → 進入 Phase 3 Lyapunov 譜分析
  → 計算穩定性指數
  → 估計 Kaplan-Yorke 維度

ELSE IF 無次級吸附子，只有單一吸引子:
  ○ P7-G 假說部分錯誤
  → 研究高擾動下的軌跡行為
  → 可能需要 Phase 4 分岔分析

ELSE IF 出現複雜結構 (>2 吸引子):
  ❓ 未預期的複雜性
  → 需要額外分析
  → 可能指向混沌或多穩態
```

### 決策點 3: 下一步行動？

```
最樂觀場景 (完全符合預期):
  ✓ 邊界清晰，2 個吸引子，λ 譜清晰
  → 進入 Phase 3+4 完整分析
  → 預期 June 9-10 完成全景
  → 進入應用設計

保守場景 (部分符合預期):
  ⚠️ 邊界基本清晰，結構複雜度中等
  → 完成 Phase 2 邊界掃描
  → 跳過 Phase 3 Lyapunov (省時)
  → June 6-7 完成，進入應用

風險場景 (發現複雜性):
  ✗ 邊界混亂，多吸引子，結構不明
  → 停止 Phase 1 結果分析
  → 決定是否投入 P7-I 深度研究
  → 或調整策略重新設計實驗
```

---

## 📈 成功標準 (Phase 1 結束時)

✅ **完全成功**
- 1,681 個網格點無例外完成
- 至少 70% 的點分類明確 (屬於某個吸引子盆地)
- 邊界清晰度 > 80% (相鄰點分類一致)
- 統計數據顯示 2-3 個主要的吸引子區域

⚠️ **部分成功**
- 完成 1,681 個點，但分類模糊
- 60-70% 的點分類明確
- 邊界複雜（可能是分形或多層結構）
- 需要 Phase 2 精細掃描以澄清

❌ **需要改進**
- 大量點無法完成模擬 (< 80% 成功率)
- 終點分散無規律 (無明確吸引子)
- 邊界完全不清晰

---

## 🎯 Phase 1+2+3 完整時間表

### June 5 (Tuesday)

```
08:00-09:00  系統環境檢查
09:00-13:00  Phase 1 執行 (1,681 點, ~4 h, background)
13:00-14:00  午餐 + Phase 1 進度檢查
14:00-16:00  Phase 2 邊界掃描 (若 Phase 1 進展良好)
16:00-17:00  初步結果分析與熱圖生成
```

### June 6 (Wednesday)

```
09:00-10:00  數據完整性檢查
10:00-11:00  視覺化與統計分析
11:00-12:00  決策：是否進入 Phase 3？
13:00-15:00  Phase 3 Lyapunov 譜計算 (若決定進行)
15:00-17:00  綜合報告撰寫
```

### June 9 (Friday) - 決策會議

```
目標: 基於 P7-H Phase 1-3 結果，決定下一階段方向

選項 A: 完成 P7-H Phase 4 (分岔分析)
  → 時間: 1 週
  → 預期: 完整相空間全景

選項 B: 進入應用設計階段
  → 時間: 2-4 週
  → 預期: 個性化遊戲原型

選項 C: 深度研究 P7-I (混沌/多穩態分析)
  → 時間: 2-3 週
  → 預期: 理論突破
```

---

## ⚠️ 常見問題 & 應急預案

### Q1: Phase 1 執行超過預計時間？

**症狀**: 4 小時過去，只完成 20% 的網格點

**診斷**:
- 可能是 SBERT 推斷緩慢
- 或 RLSessionEngine 初始化開銷大

**應急**:
```bash
# 切換到更快的推斷模式 (若可用)
# 或減少網格解析度 (暫時)
./venv/bin/python scripts/experiments/run_p7h_landscape_explorer.py \
    --resolution 0.002 \  # 更粗的網格
    --phase 1
```

### Q2: 大量模擬失敗？

**症狀**: 許多軌跡拋出異常或超時

**診斷**:
- RLSessionEngine 狀態管理問題
- 個性推斷 API 返回無效值

**應急**:
```bash
# 檢查 RLSessionEngine 日誌
tail -100 logs/rl_session_*.log

# 測試單個網格點
./venv/bin/python << 'EOF'
# 本地測試邏輯
EOF
```

### Q3: 結果分類全是「undefined」？

**症狀**: p7h_phase1_summary.json 顯示大部分點未分類

**診斷**:
- 分類閾值設定不當
- 或終點分布與預期不符

**應急**:
- 調整分類閾值 (在 Phase 2 中)
- 重新檢視 P7-F 基準吸引子位置

---

## 🎬 立即執行清單

### 今天 (June 4) 下午

- [ ] 檢查本清單，確認所有項目✓
- [ ] 預留計算資源（關閉不必要的後台進程）
- [ ] 驗證儲存空間 (> 1 GB 可用)
- [ ] 準備監控腳本

### 明天 (June 5) 早上

- [ ] 最後環境檢查
- [ ] 啟動 Phase 1
- [ ] 監控初期進度（第一 100 個點）
- [ ] 驗證輸出格式正確

### 中午 + 下午

- [ ] Phase 1 後臺運行
- [ ] 啟動 Phase 2 (若 Phase 1 進展順利)
- [ ] 每小時檢查一次進度

---

## 📞 支援和聯絡

若遇到問題，依序檢查：

1. **P7H_DECISION_FRAMEWORK.md** - 決策和風險預案
2. **P7H_landscape_explorer.md** - 技術規格和演算法
3. **run_p7h_landscape_explorer.py** - 代碼註釋
4. **研發日誌.md** - P7-G/H 完整記錄

---

**文件**: P7-H 執行前最後檢查清單 & 啟動指令  
**版本**: v1.0  
**狀態**: 🟢 **全部就緒**  
**批准狀態**: ⏳ 等待 June 5 執行授權  
**預期開始**: June 5 09:00  
**預期完成**: June 6 17:00 (Phase 1+2) / June 9 17:00 (含 Phase 3)
