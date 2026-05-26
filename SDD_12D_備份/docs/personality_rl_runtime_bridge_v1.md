# Personality RL Runtime Bridge v1

本文件定義下一輪工程主線的最小整合方案。
目標不是重寫整個 repo，而是用最小侵入方式，把已被 BL2 證實的 RL 循環機制，接入 Personality Dungeon 的正式 runtime。

---

## 1. 問題定義

目前最強結果存在於 `simulation/e1_heterogeneous_rl.py` 的 standalone loop。
但正式地城 runtime 主要建立在 `simulation/run_simulation.py` 的 sampled / mean_field / personality_coupled 路徑上。

這兩條線的核心差異不是參數，而是動力學物件不同：

1. `run_simulation.py` 以共享權重與 replicator-style update 為中心
2. E1 / BL2 以 per-player Q-table、Boltzmann action choice、與 local one-hot payoff 為中心

因此，下一輪工程的根本任務不是調參，而是建立一條正式的 RL runtime bridge。

---

## 2. 設計原則

### 2.1 要保留的東西

1. BL2 的核心機制不變：wide α、β=3、static ε=0.02、mild_cw `[1.2,1.0,0.8]`
2. `evolution/independent_rl.py` 維持 pure functions
3. event/world 仍由 `dungeon/` 與 `simulation/` 負責組裝與 I/O
4. `analysis/` 仍只吃 CSV/TSV，不依賴 runtime 物件

### 2.2 不要做的事

1. 不要把 BL2 直接塞成 `evolution_mode='sampled'` 的新分支
2. 不要覆寫既有 `w_*` 欄位的語意
3. 不要在第一輪同時改動 event schema 與 RL operator
4. 不要先做完整 UI / 內容池，再回頭補動力學

---

## 3. 建議架構

### 3.1 新增一條正式 RL runtime，而不是硬改舊 sampled path

第一版建議新增：

1. `players/rl_player.py`
2. `simulation/personality_rl_runtime.py`
3. `tests/test_personality_rl_runtime.py`

理由：

1. `players/` 應承接 per-player state，而不是把所有 RL state 都塞在 simulation local variables
2. `simulation/` 應負責組裝 event loader、world state、life loop、輸出與 provenance
3. 這樣可以讓舊的 replicator runtime 保持穩定，避免語意污染

### 3.2 `players/rl_player.py` 的最小責任

建議最小資料結構：

```python
@dataclass
class RLPlayer:
    player_id: int
    personality: dict[str, float]
    q_values: list[float]
    alpha: float
    beta: float
    strategy_alpha_multipliers: list[float]
    payoff_bias: list[float]
    cumulative_utility: float = 0.0
    cumulative_risk: float = 0.0
    stress: float = 0.0
```

這個物件只保存狀態，不做 I/O。

### 3.3 `simulation/personality_rl_runtime.py` 的最小責任

第一版建議它負責：

1. 初始化 RL players
2. 讀入 event templates
3. 每輪生成事件 / action candidates
4. 用 RL policy 選動作
5. 結算 reward / risk / utility
6. 更新 Q-table
7. 寫 round-level timeseries 與 seed-level provenance

它不應該直接實作 analysis，也不應把 event schema 細節散落到多個 helper。

---

## 4. 最小資料流

建議每輪的正式資料流固定如下：

1. 由 world state 產生本輪 event bundle
2. 每個 RL player 讀取 event bundle 與自身 state
3. 將 candidate actions 嵌入三策略 simplex
4. 由 `Q_i + personality bias + event-local utility` 形成 action score
5. 取樣 action
6. 寫回 realized action family 與對應的 macro strategy
7. 結算 reward / risk / success / failure
8. 執行 BL2-compatible Q-update
9. 聚合 round summary 並輸出

這個資料流的關鍵是：

1. 事件層提供 action context
2. RL 層提供動態選擇機制
3. personality 提供穩定偏置
4. world 提供長期壓力結構

---

## 5. BL2 映射到正式 runtime 的最小方案

### 5.1 第一版參數鎖定

第一版不要搜尋，直接鎖定 BL2：

1. `alpha_lo=0.005`, `alpha_hi=0.40`
2. `beta=3.0`
3. `payoff_epsilon=0.02`
4. `strategy_alpha_multipliers=[1.2,1.0,0.8]`
5. `a=1.0`, `b=0.9`, `cross=0.20`

### 5.2 personality 只做 bounded modulation

第一版建議：

1. 由 personality 決定 player-specific `alpha_i` 與 `beta_i` 的初始化位置
2. 由 personality 在 BL2 的 `strategy_alpha_multipliers` 上做小幅偏移
3. 由 personality 影響 risk response 與 action scoring
4. 不讓 personality 直接改 payoff matrix

### 5.3 event/world 只做 context，不直接代替 RL

第一版建議事件與世界只影響：

1. reward modifiers
2. risk modifiers
3. candidate-action geometry
4. observable information level

不直接做：

1. 直接覆寫 Q 值
2. 直接改 analysis gate
3. 直接指定 agent 下一步策略

---

## 6. 輸出契約

### 6.1 round-level timeseries CSV

第一版建議欄位：

1. `round`
2. `avg_reward`
3. `avg_utility`
4. `success_rate`
5. `risk_mean`
6. `stress_mean`
7. `p_aggressive`
8. `p_defensive`
9. `p_balanced`
10. `pi_aggressive`
11. `pi_defensive`
12. `pi_balanced`
13. `q_mean_aggressive`
14. `q_mean_defensive`
15. `q_mean_balanced`
16. `world_scarcity`
17. `world_threat`
18. `world_noise`
19. `world_intel`
20. `dominant_event_type`

其中：

1. `p_*` 是 realized action proportions，可直接餵給既有 cycle metrics
2. `pi_*` 是 mean policy probabilities，是 RL runtime 的新狀態欄位
3. 不再使用 `w_*` 來承載 RL policy，避免污染既有契約

### 6.2 seed-level provenance

至少需要：

1. `alpha_lo`, `alpha_hi`, `beta_ceiling`
2. `strategy_alpha_multipliers`
3. `payoff_epsilon`
4. `personality_mode`
5. `events_json`
6. `world_mode`
7. `seed`
8. `mean_alpha`, `std_alpha`
9. `mean_beta`, `std_beta`
10. `risk_rule_version`

### 6.3 per-player snapshot TSV

建議在每個 seed 結束時額外輸出：

1. `player_id`
2. `personality_json`
3. `alpha`
4. `beta`
5. `r_a`, `r_d`, `r_b`
6. `q_a`, `q_d`, `q_b`
7. `utility_final`
8. `risk_final`
9. `dominant_strategy`

這份輸出會是日後診斷 personality 是否真的映到動力學的關鍵。

---

## 7. 最小實作步驟

### Step 1

新增 `players/rl_player.py`，把 RL state 正式放進 players layer。

### Step 2

新增 `simulation/personality_rl_runtime.py`，先在沒有 world adaptation 的固定事件流下跑通 BL2。

### Step 3

讓 runtime 能吃既有 `02_event_templates_v1.json`，但只開最小 smoke subset。

### Step 4

輸出新的 round CSV / provenance / snapshot TSV，並確認 `p_*` 能直接進既有 analysis。

### Step 5

只有在 Step 1–4 都穩定後，才把 Little Dragon world feedback 接進 in-loop。

---

## 8. 最小測試集

建議第一輪測試只鎖四組。

### 8.1 退化測試

1. 關閉 event/world 特效時，RL runtime 應退化為 BL2 近似條件
2. `strategy_alpha_multipliers=[1,1,1]` 時，結果應退化回對稱 control

### 8.2 契約測試

1. `p_*` 每輪總和必須為 1
2. `pi_*` 每輪總和必須為 1
3. CSV 不得出現 `w_*` 與 `pi_*` 混義

### 8.3 personality 映射測試

1. 高 impulsiveness / greed / ambition 的 player，其 aggressive-side 更新速度高於 baseline
2. 高 caution / fearfulness 的 player，其 risk sensitivity 高於 baseline

### 8.4 I/O 測試

1. 會寫出 timeseries CSV
2. 會寫出 provenance JSON
3. 會寫出 per-player snapshot TSV

---

## 9. 第一輪成功條件

最小整合方案的成功，不是直接再追求 100% L3，而是先滿足這四件事：

1. BL2 的核心旋轉幾何在正式 runtime 裡沒有被破壞
2. personality 確實會改變 per-player learning dynamics
3. event/world 的影響是透過正式 channel 進來，而不是 ad hoc patch
4. 輸出仍可回到現有 analysis pipeline 比較

只要這四件事成立，下一輪工程主線就算真正建立起來了。