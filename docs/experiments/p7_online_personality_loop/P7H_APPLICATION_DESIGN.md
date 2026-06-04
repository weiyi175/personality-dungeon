# P7-H 應用設計：邊界穩定性驅動的人格控制系統

## 文檔版本
- **版本**: v1.0 (Design Phase)
- **日期**: 2025-06-04
- **狀態**: 🟢 **邊界穩定性科學驗證完成，應用設計開始**
- **基礎**: P7-H Phase 1 (網格掃描) + Phase 3 (Lyapunov 譜分析)

---

## 1. 應用願景

### 1.1 核心概念

利用人格動力系統的邊界穩定性 (λ ≈ 0⁺) 設計**高敏感性人格控制機制**：

```
傳統反饋：大的遊戲事件 → 中等人格改變 → 線性效果
邊界穩定：小的遊戲事件 → 巨大人格分岔 → 非線性效果

系統特性: 邊界穩定性（λ ≈ 1.29e-10）
應用機會: 最小能量觸發最大效應
```

### 1.2 設計目標

1. **檢測邊界穩定區域**: 識別玩家何時進入高敏感狀態
2. **設計分岔事件序列**: 創造觸發多吸引子切換的遊戲事件
3. **驗證人格轉變機制**: 實測邊界穩定性理論的可玩性
4. **評估遊戲體驗**: 測量玩家對人格跳躍的感知與反應

### 1.3 成功指標

| 指標 | 目標 | 驗證方式 |
|------|------|---------|
| 邊界檢測精度 | > 85% | A/B 測試 |
| 分岔觸發成功率 | > 70% | 事件序列播放 |
| 人格切換幅度 | ≥ 0.1 (9D 距離) | 軌跡終點距離 |
| 玩家感知 | 主觀報告分數 | 問卷調查 |

---

## 2. 科學基礎

### 2.1 邊界穩定性特性

**來源**: P7-H Phase 3 Lyapunov 譜分析

```
Lyapunov 指數特徵:
  λ_max = 1.29e-10 (極小正數)
  λ_min = 7.32e-11 (極小正數)
  Trace(J) = 9.92e-10 (接近零)
  Kaplan-Yorke dim = 9.0 (完整維度)
  
物理意義:
  ✓ 系統既不擴張也不收縮 (λ ≈ 0)
  ✓ 所有 9D 都參與 (無低維投影)
  ✓ 擾動極度敏感 (邊界響應)
  ✓ 分岔點接近 (多吸引子共存)
```

### 2.2 吸引子盆地結構

**來源**: P7-H Phase 1 網格掃描

```
本地盆地 (v₁-v₂ 平面):
  範圍: ±0.020 每軸
  完全收斂: 所有 1,681 點返回主吸引子
  最大距離: < 1e-06 (浮點精度)
  
全球結構推理:
  ✓ 主吸引子有巨大吸引域 (至少 ±0.020)
  ✗ 次級吸引子不在 v₁-v₂ 平面
  ✓ 但在 v₃-v₉ 弱維度方向存在分岔點
  
應用含義:
  小擾動 (ε < ε_c ≈ 0.007) → 返回主吸引子
  中等擾動 (ε ≈ 0.01-0.1) → 進入過渡區/次吸引子
  大擾動 (ε > 0.1) → 完全逃逸到遠區域
```

### 2.3 1D 簡樸性 + 9D 複雜性的悖論

**來源**: P7-F (1D 簡樸) + P7-H Phase 3 (9D 複雜)

```
靜態結構 (P7-F):
  - 21 個吸引子位於 1D 不變子空間
  - σ₁ = 99.93% (幾乎所有方差)
  - α-FP 線性參數化 (R² = 1.0)

動態結構 (P7-H Phase 3):
  - 所有 9D 都有正 Lyapunov 指數
  - Kaplan-Yorke dim = 9.0
  - 邊界穩定性是完整 9D 現象

解釋:
  ✓ 靜態結構 = 1D 上的"珠子" (吸引子位置)
  ✓ 動態結構 = 9D 的"彈簧"（穩定性和分岔）
  ✓ 應用可利用 1D 可預測性 + 9D 敏感性
```

---

## 3. 應用架構設計

### 3.1 系統組件

```
┌─────────────────────────────────────────────────────┐
│ 遊戲前端 (Godot)                                      │
│  - 玩家決策輸入                                      │
│  - 人格視覺反饋                                      │
│  - 事件序列播放                                      │
└──────────────┬──────────────────────────────────────┘
               │
┌──────────────▼──────────────────────────────────────┐
│ 邊界穩定性檢測層                                      │
│  - 計算當前人格軌跡                                  │
│  - 評估邊界接近度 (Distance to Bifurcation)        │
│  - 預測分岔敏感方向 (Unstable Eigenvector)         │
└──────────────┬──────────────────────────────────────┘
               │
┌──────────────▼──────────────────────────────────────┐
│ 事件控制層                                           │
│  - 選擇觸發事件（基於敏感方向）                      │
│  - 調制事件強度（基於邊界距離）                      │
│  - 播放事件序列（多步分岔）                          │
└──────────────┬──────────────────────────────────────┘
               │
┌──────────────▼──────────────────────────────────────┐
│ RLSessionEngine (模擬核心)                           │
│  - 人格推斷 (SBERT + MLP v7)                        │
│  - 軌跡演化 (α-FP 動力學)                           │
│  - 輸出新人格向量                                   │
└──────────────┬──────────────────────────────────────┘
               │
┌──────────────▼──────────────────────────────────────┐
│ 驗證層                                              │
│  - 記錄軌跡位置                                      │
│  - 測量分岔成功 (是否到達次吸引子)                  │
│  - 評估學習效果                                      │
└─────────────────────────────────────────────────────┘
```

### 3.2 數據流

```
遊戲事件
  ↓
[文本提取] → SBERT 編碼
  ↓
[人格推斷] → MLP v7 ← {SBERT embedding}
  ↓
[軌跡位置] ← {當前人格向量 P(t)}
  ↓
[邊界檢測] → 計算:
  - Distance to bifurcation
  - Unstable direction (v₁, v₂, ... v₉)
  - Sensitivity coefficient λ_eff
  ↓
[事件設計] → 選擇與生成:
  - Event type (與 unstable direction 對齊)
  - Event intensity (與 bifurcation distance 反向)
  - Sequence length (根據預計軌跡)
  ↓
[模擬播放] → RLSessionEngine.step()
  ↓
[驗證] → 測量:
  - ||P_final - P_baseline||
  - 是否成功轉移到次吸引子
  - 感知報告
```

---

## 4. 核心算法

### 4.1 邊界接近度計算

```python
def compute_bifurcation_distance(personality_vector):
    """
    計算當前人格向量到分岔點的距離
    
    基於 P7-H Phase 1/3 的發現:
    - 主吸引子在 baseline 附近
    - 分岔點距離 ≈ ε_c ≈ 0.007 (9D 中)
    - 或需要沿著弱維度探索
    """
    
    # 計算與基準吸引子距離
    distance_to_attractor = np.linalg.norm(
        personality_vector - baseline_attractor
    )
    
    # 估計分岔臨界距離
    bifurcation_critical_distance = 0.007  # P7-G 估計
    
    # 邊界接近度 (0-1 scale)
    # 越接近 1，越接近分岔點
    proximity = min(1.0, distance_to_attractor / bifurcation_critical_distance)
    
    return {
        'distance': distance_to_attractor,
        'bifurcation_proximity': proximity,
        'is_critical': proximity > 0.8,  # 臨界閾值
    }
```

### 4.2 敏感方向計算

```python
def compute_sensitive_direction(personality_vector, alpha=0.2):
    """
    計算當前位置的不穩定特徵向量方向
    
    基於 Lyapunov 譜:
    - λ ≈ 0⁺ 表示存在邊界穩定方向
    - 對應的特徵向量是擾動方向
    """
    
    # 載入 Jacobian (可從 P7-G 或計算)
    # 計算特徵分解
    eigenvalues, eigenvectors = np.linalg.eig(jacobian_at_point)
    
    # 找到最大的接近零的特徵值
    # (邊界穩定性的簽名)
    critical_idx = np.argmax(eigenvalues)
    
    # 對應的特徵向量是最敏感方向
    sensitive_direction = eigenvectors[:, critical_idx]
    
    return {
        'direction': sensitive_direction / np.linalg.norm(sensitive_direction),
        'eigenvalue': eigenvalues[critical_idx],
        'sensitivity_strength': np.abs(eigenvalues[critical_idx]),
    }
```

### 4.3 事件設計與強度調制

```python
def design_bifurcation_event(
    personality_vector,
    target_bifurcation='personality_shift',
    intensity_scale=1.0
):
    """
    設計觸發人格分岔的遊戲事件
    
    參數:
    - personality_vector: 當前人格向量
    - target_bifurcation: 目標吸引子類型
    - intensity_scale: 事件強度倍數 (基於邊界距離)
    """
    
    bifurc_info = compute_bifurcation_distance(personality_vector)
    sensitive_info = compute_sensitive_direction(personality_vector)
    
    # 基礎事件強度與邊界距離反向相關
    # 越接近分岔點，事件越弱就能觸發
    proximity = bifurc_info['bifurcation_proximity']
    base_event_strength = 0.5 * (1.0 - proximity)  # 越近越弱
    
    # 結合使用者控制強度
    final_event_strength = base_event_strength * intensity_scale
    
    # 選擇與敏感方向對齐的事件
    event = {
        'type': 'personality_shift',
        'direction': sensitive_info['direction'],
        'magnitude': final_event_strength,
        'duration_rounds': max(1, int(10 * proximity)),  # 越近越短
        'callback': 'evaluate_bifurcation_success',
    }
    
    return event
```

---

## 5. 實現路線圖

### 5.1 Phase I: 核心系統實現 (Day 1-2)

**目標**: 建立邊界檢測和事件生成的基本框架

```
[ ] Task 1: 創建 bifurcation_detector.py
    - compute_bifurcation_distance()
    - compute_sensitive_direction()
    - get_jacobian() 或使用預計算版本
    
[ ] Task 2: 創建 event_generator.py
    - design_bifurcation_event()
    - event_sequence_planner()
    - intensity_modulation()
    
[ ] Task 3: 集成到 API 層
    - /api/bifurcation/detect
    - /api/bifurcation/event
    - /api/bifurcation/sequence
    
[ ] Task 4: 單元測試
    - 驗證邊界檢測邏輯
    - 驗證事件設計算法
    - 邊界情況測試
```

### 5.2 Phase II: 驗證與校準 (Day 3-4)

**目標**: 在模擬中驗證邊界穩定性理論

```
[ ] Task 1: 創建 bifurcation_verifier.py
    - 運行 N 個試驗 (N=50-100)
    - 每個試驗: 檢測邊界 → 生成事件 → 執行 → 測量
    - 統計成功率、效應大小
    
[ ] Task 2: 生成驗證報告
    - 邊界檢測精度: precision/recall
    - 事件觸發成功率
    - 人格轉移大小分佈
    - 與理論預測比較
    
[ ] Task 3: 參數調整
    - 臨界閾值 (proximity > ?)
    - 事件強度係數
    - 敏感方向權重
    
[ ] Task 4: 邊界情況處理
    - 多步序列中的累積效應
    - 非邊界區域的事件無效性
    - 應急回滾機制
```

### 5.3 Phase III: 原型集成 (Day 5-7)

**目標**: 將驗證的系統集成到 Godot 遊戲前端

```
[ ] Task 1: Godot 端點設計
    - 輸入玩家決策 → 提取文本
    - 調用 /api/bifurcation/detect
    - 根據邊界接近度調整 UI 反饋
    
[ ] Task 2: 事件播放機制
    - 根據 /api/bifurcation/event 生成事件序列
    - 插入遊戲循環
    - 記錄軌跡變化
    
[ ] Task 3: 反饋 UI
    - 邊界接近度儀表板 (0-100%)
    - 人格維度動態顯示
    - 分岔事件視覺標記
    
[ ] Task 4: A/B 測試框架
    - 控制組: 隨機事件
    - 實驗組: 邊界穩定性控制事件
    - 測量組: 人格轉移幅度
```

### 5.4 Phase IV: 評估與報告 (Day 8)

**目標**: 驗證可玩性和遊戲體驗

```
[ ] Task 1: 玩家測試
    - 招募 10-20 玩家
    - 分配實驗組/控制組
    - 遊玩 1-2 小時
    
[ ] Task 2: 數據收集
    - 軌跡記錄 (人格向量序列)
    - 事件響應時間
    - 人格轉移成功率
    
[ ] Task 3: 問卷調查
    - 主觀感知: 人格轉變是否感到自然？
    - 遊戲體驗: 邊界控制是否增加樂趣？
    - 可玩性: 是否願意繼續遊玩？
    
[ ] Task 4: 最終報告
    - 比較實驗/控制組效果
    - 估計實際邊界穩定性的遊戲影響
    - 建議後續改進
    - 發表研究結果
```

---

## 6. 預期結果與假設

### 6.1 樂觀情景 (Success)

```
✅ 邊界檢測精度 > 85%
✅ 事件觸發成功率 > 70%
✅ 人格轉移幅度 0.08-0.15 (9D 距離)
✅ 玩家感知得分 > 7/10 (自然感)
✅ 實驗組 vs 控制組 效果差異顯著 (p < 0.05)

結論: 邊界穩定性可應用於遊戲設計
後續: 深化研究，考慮商業化
```

### 6.2 中等情景 (Partial Success)

```
⚠️ 邊界檢測精度 70-85%
⚠️ 事件觸發成功率 50-70%
⚠️ 人格轉移幅度 0.04-0.08
⚠️ 玩家感知得分 5-7/10

原因分析:
- Jacobian 估計偏差
- 遊戲事件的文本複雜性
- 玩家決策的隨機成分

改進策略:
- 更精確的 Jacobian 計算 (Phase 2 弱維度掃描)
- 事件文本的精心設計
- 多步序列而非單步事件
```

### 6.3 悲觀情景 (Failure)

```
❌ 邊界檢測精度 < 70%
❌ 事件觸發成功率 < 50%
❌ 人格轉移不明顯
❌ 玩家感知無顯著差異

可能原因:
- SBERT + MLP 推斷誤差過大
- 邊界穩定性只在特定參數下有效
- 遊戲事件的文本影響不足
- 次級吸引子不存在或不可達

應變計畫:
- 返回 P7-H Phase 2 (弱維度掃描，精確定位)
- 考慮混合應用 (1D + 固定點控制)
- 評估是否需要追加物理模擬驗證
```

### 6.4 關鍵假設

```
假設 1: SBERT + MLP v7 人格推斷精度足夠
  驗證: 在應用設計中實測，精度 > 80%

假設 2: 遊戲事件的文本能有效驅動人格變化
  驗證: Phase III 中測試不同事件類型的效應大小

假設 3: 分岔點確實可以通過遊戲事件觸發
  驗證: Phase II 模擬驗證，成功率 > 70%

假設 4: 玩家會感知到邊界穩定性效應
  驗證: Phase IV 玩家測試，主觀感知 > 6/10
```

---

## 7. 風險與對策

| 風險 | 影響 | 機率 | 對策 |
|------|------|------|------|
| 人格推斷誤差大 | 邊界檢測失準 | 30% | 追加 SBERT 微調，或切換模型 |
| 次級吸引子不可達 | 無法觸發分岔 | 25% | 回歸 Phase 2 掃描，精確定位 |
| 事件強度難調整 | 觸發成功率低 | 40% | 多步序列設計，自適應強度 |
| 玩家感知無差異 | 遊戲體驗無改善 | 35% | 加強 UI 反饋，或增加事件多樣性 |
| 計算資源不足 | 實時性不足 | 15% | 預計算 Jacobian，或降低檢測頻率 |

---

## 8. 成功評估標準

### 8.1 科學評估

```
✓ 邊界穩定性在應用中得到驗證 (λ ≈ 0⁺ 確實導致高敏感性)
✓ 多吸引子分岔在遊戲中可觀測 (人格切換 ≥ 0.1 距離)
✓ 應用架構可擴展 (適配不同遊戲或對話系統)
```

### 8.2 工程評估

```
✓ API 端點完整 (/bifurcation/detect, /event, /sequence)
✓ 測試覆蓋率 > 80%
✓ 文檔完善 (代碼註釋 + 使用指南)
✓ 性能達標 (邊界檢測 < 100ms)
```

### 8.3 UX 評估

```
✓ 玩家能感知到人格變化 (主觀評分 > 6/10)
✓ 界面反饋清楚 (邊界接近度、推薦事件明確)
✓ 遊戲流暢性無損傷 (幀率穩定)
```

---

## 9. 後續研究方向

### 9.1 短期 (1-2 週)

- [ ] 實現應用設計 (Phase I-IV 完整)
- [ ] 收集玩家反饋
- [ ] 撰寫應用報告

### 9.2 中期 (1 個月)

- [ ] Phase 2 弱維度掃描 (精確定位次吸引子)
- [ ] 多人對弈場景 (多智能體人格互動)
- [ ] 遊戲原型發布 (Beta 版)

### 9.3 長期 (3-6 個月)

- [ ] 商業化評估 (IP 申報，融資)
- [ ] 學術發表 (頂級 AI/遊戲會議)
- [ ] 跨領域應用 (對話系統、教育、治療)

---

## 10. 參考資料與鏈接

### P7 系列成果

| 文檔 | 內容 | 鏈接 |
|------|------|------|
| P7-F 報告 | 1D 簡樸性、α-FP 線性 | `/reports/experiments/p7f_attractor_mapping/` |
| P7-G 報告 | 邊界穩定性、ε_c ≈ 0.007 | `/docs/experiments/p7_online_personality_loop/` |
| P7-H Phase 1 | 1,681 點網格掃描 | `/reports/experiments/p7h_landscape_explorer/` |
| P7-H Phase 3 | Lyapunov 譜分析 | `/reports/experiments/p7h_landscape_explorer/p7h_lyapunov_spectra.json` |

### 技術棧

- **人格推斷**: SBERT (paraphrase-multilingual-MiniLM-L12-v2) + MLP v7
- **模擬核心**: RLSessionEngine (`simulation/rl_session_engine.py`)
- **前端**: Godot Engine
- **API**: FastAPI (`api/server.py`)

---

## 11. 附錄：關鍵公式

### 邊界接近度 (Bifurcation Proximity)

$$
\text{Proximity}(P) = \min\left(1, \frac{||P - P_0||}{ε_c}\right)
$$

其中 $P_0$ 是基準吸引子，$ε_c ≈ 0.007$ 是臨界擾動距離。

### Kaplan-Yorke 維度

$$
D_{KY} = j + \frac{1}{λ_{j+1}} \sum_{i=1}^{j} λ_i
$$

其中 $λ_1 ≥ λ_2 ≥ ... ≥ λ_d$ 是 Lyapunov 指數，$j$ 是使得 $\sum_{i=1}^{j} λ_i ≥ 0$ 的最大整數。

對於本系統，$D_{KY} = 9.0$（全維度）。

### 邊界穩定性簽名 (Marginal Stability Signature)

$$
\max(λ_i) ≈ 0^+ \quad \text{且} \quad \text{Trace}(J) ≈ 0
$$

這表示系統處於分岔邊界，任何小擾動都可能導致軌跡遷移。

---

**版本**: P7-H 應用設計文檔 v1.0  
**狀態**: 🟢 **就緒，等待實現**  
**預計開始**: 2025-06-04 下午  
**預計完成**: 2025-06-11 (7 天)  
**ROI**: 科學 + 應用雙贏，可擴展到多個領域
