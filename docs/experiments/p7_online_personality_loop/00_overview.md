# P7 實驗系列：在線人格推斷迴圈（Online Personality-RL Closed Loop）

> **研究問題**：若將 RL Session 每步輸出的人格向量即時回饋為下一步的 payoff 調制項，系統動態是否會產生新的吸引子、振盪模式或不穩定區間？

**SDD 分層歸屬**

| 層 | 本系列涉及的異動 |
|----|-----------------|
| `players/` | 人格向量從靜態初始改為動態更新（P7-C 以後） |
| `dungeon/` | payoff 調制函數 Δu（P7-B 規格化，P7-C 注入） |
| `simulation/` | 新增「personality\_feedback」欄位到 CSV schema（P7-C Gate） |
| `analysis/` | 人格軌跡指標（P7-D、P7-E 新增） |
| `api/` | `/rl_sessions/{id}/step` 回傳欄位擴充（P7-C Gate） |

> **分層不變條件**：`analysis/` 不得 import `simulation/`；`evolution/` 不做 I/O。

---

## 實驗家族地圖

```
P7-A  基線快照
  └─► P7-B  人格→payoff 映射規格（數學 Spec，無模擬）
        └─► P7-C  單回合回饋注入（小規模驗證）
              ├─► P7-D  迴圈穩定性掃描（參數矩陣）
              └─► P7-E  長期人格漂移（200+ 輪軌跡）
```

每個子實驗都有**獨立 Gate**；若上游 Gate 失敗，下游實驗暫停。

---

## 核心概念：閉環機制

### 現況（靜態路徑）

```
文字輸入 → SBERT → 人格向量 P₀ (固定)
                                ↓
RL Session: init → step → step → ...
payoff 固定使用 matrix_ab(a,b)
```

### 目標（動態閉環）

```
文字輸入 → SBERT → 人格向量 P₀
                          ↓
  ┌─────── RL Session Step t ──────────┐
  │  策略選擇  →  payoff(Pₜ)  →  結果  │
  │                                    │
  │  Pₜ₊₁ = Pₜ + α · ΔP(結果, Pₜ)   │
  └────────────────────────────────────┘
        ↑____ 回饋 Pₜ₊₁ 到下一步 ____↑
```

- **α**（feedback strength）是主要控制參數
- α = 0 退化回現有靜態路徑（回歸安全保障）
- ΔP 的計算規格由 **P7-B** 定義

---

## 人格特徵向量（固定，9D Enneagram）

| trait | 縮寫 | 與策略的預期關聯 |
|-------|------|-----------------|
| impulsiveness | IMP | ↑ → aggressive 偏好 ↑ |
| assertiveness | ASS | ↑ → aggressive 偏好 ↑ |
| optimism | OPT | ↑ → aggressive / balanced 偏好 ↑ |
| risk_aversion | RAV | ↑ → defensive 偏好 ↑ |
| suspicion | SUS | ↑ → defensive 偏好 ↑ |
| endurance | END | ↑ → defensive / balanced 偏好 ↑ |
| randomness | RND | ↑ → 均等策略分布 |
| stability_seeking | STB | ↑ → defensive 偏好 ↑ |
| curiosity | CUR | ↑ → balanced / aggressive 偏好 ↑ |

---

## 分階段 Gate 摘要

| 階段 | Gate | 驗收核心 |
|------|------|----------|
| P7-A | G7A-01 ~ G7A-03 | 靜態路徑基線 reward 分布鎖定 |
| P7-B | G7B-01 ~ G7B-04 | 映射函數 Spec 完整、invariants 通過 |
| P7-C | G7C-01 ~ G7C-05 | 回饋注入後 schema 正確、α=0 行為不變 |
| P7-D | G7D-01 ~ G7D-04 | 找到穩定 α* 區間、振盪可量化 |
| P7-E | G7E-01 ~ G7E-03 | 長期軌跡可重現，吸引子可辨識 |

---

## 檔案清單

| 檔案 | 描述 |
|------|------|
| `00_overview.md` | ← 本文件 |
| `P7A_baseline_snapshot.md` | 基線快照實驗規格 |
| `P7B_mapping_spec.md` | 人格→payoff 映射數學規格 |
| `P7C_feedback_injection.md` | 單回合回饋注入小規模驗證 |
| `P7D_stability_scan.md` | 迴圈穩定性掃描（參數矩陣） |
| `P7E_longterm_drift.md` | 長期人格漂移（200+ 輪軌跡） |

---

*建立日期：2026-06-03*  
*狀態：規劃中（尚未開始實作）*
