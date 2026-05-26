# Personality Dungeon Complete Blueprint v1

本文件是研究主線封板後的第一份整合藍圖。
定位不是取代既有 `docs/personality_dungeon_v1/` design pack，而是把目前已驗證的 RL 結果、既有 12 維 personality 設計、events/world 原型，整理成一份可以直接指導下一輪工程的總藍圖。

核心原則只有一句：

> Personality Dungeon 不是把研究結果包一層皮，而是把 `personality -> learning dynamics -> event/world coupling` 做成同一個可回歸系統。

---

## 1. 產品定位

### 1.1 一句話版本

玩家不直接控制角色，而是透過有限記憶的遺言、長期的人格塑形、與多次生命循環，逐步培養一個會在地城壓力下形成穩定行為模式的代理人。

### 1.2 產品核心

1. 玩家互動單位不是「每回合指令」，而是「人格干預」
2. 系統主體不是固定腳本角色，而是「帶人格與學習動力的 agent」
3. 地城不是單純關卡，而是「會對群體行為作出反應的壓力場」
4. 死亡不是失敗畫面，而是人格與世界共同重寫的節點

### 1.3 研究結果在產品中的角色

目前已證實的 BL2 循環不是附加特效，而是產品核心可玩性來源：

1. 它讓人格不是靜態屬性，而是可觀察的動態偏好結構
2. 它讓世界壓力可以對玩家塑造出的 agent 形成可預期、可變形、但不僵死的反應
3. 它讓多 life 的遺言干預有長期可比較的結果，而不是一次性隨機波動

---

## 2. 系統總覽

完整系統拆成六層。

### 2.1 Personality Layer

狀態是 12 維人格向量 `P ∈ [-1, 1]^12`。

它的責任不是直接做出動作，而是提供三類穩定偏置：

1. 策略偏置：傾向 aggressive / defensive / balanced 哪個區域
2. 學習偏置：更新速度、探索程度、衝擊後的恢復與固著
3. 風險偏置：在同樣事件下如何放大或縮小 risk / stress / hesitation

### 2.2 Learning-Dynamics Layer

這一層是 BL2 的正式落點。

每位 agent 都有自己的：

1. `Q_i(s)`
2. `alpha_i`
3. `beta_i`
4. `strategy_alpha_multipliers_i` 或其 personality-conditioned 變體
5. per-player payoff bias / risk memory / shock memory

這一層負責把 personality 的穩定偏好，轉成真正會在多輪內累積的行為動力。

### 2.3 Strategy Layer

策略層維持 3-strategy simplex：

1. aggressive
2. defensive
3. balanced

這不是最終遊戲的全部動作空間，而是上位 latent policy manifold。
事件中的具體動作先映到這三類，再由學習層決定當下的行動機率。

### 2.4 Event Layer

事件是可選動作集合，而不是固定地圖格。

每個事件至少要提供：

1. 風險壓力
2. 報酬期望
3. 資訊價值
4. 節奏屬性
5. 與三策略的幾何關聯

### 2.5 World Layer

世界層不是直接指定玩家該做什麼，而是控制事件分佈與壓力形狀。

建議第一版固定四個世界狀態軸：

1. scarcity
2. threat
3. noise
4. intel

世界層的責任是改變 event mix、reward/risk 係數、與可觀測資訊密度，而不是直接改寫 agent 內部 Q-table。

### 2.6 Meta Layer

Meta layer 管理：

1. 遺言輸入
2. 多 life 傳承
3. 世界適應器（Little Dragon）
4. 研究輸出與 replay / provenance

---

## 3. 玩家生命循環

建議完整循環固定如下：

1. 出生：初始化人格向量、Q-state、世界狀態
2. 事件段：agent 在一串事件中自主選擇動作
3. 風險累積：risk / stress / utility / memory 同步更新
4. 死亡或存活結束：本 life 收束
5. 遺言輸入：玩家只留下有限文字干預
6. 人格更新：把文字轉成 bounded personality delta
7. 下一 life：帶著更新後人格重新進入新的地城壓力場

這個循環中，玩家的真正操作是「在死亡之間塑形」，不是「在回合之中按鍵」。

---

## 4. Personality 到 Learning Dynamics 的正式映射

這一節是整份藍圖最重要的部分。

### 4.1 第一層映射：12D -> 3-strategy simplex anchor

沿用既有 `docs/personality_dungeon_v1/01_personality_dimensions_v1.json` 與 `03_personality_projection_v1.py`：

1. aggressive 主群：impulsiveness, greed, ambition
2. defensive 主群：caution, suspicion, fearfulness
3. balanced 主群：persistence, stability_seeking, patience
4. optimism, randomness, curiosity 作為二階修飾

這一層的輸出不是直接行動，而是 agent 在策略空間中的先驗位置。

### 4.2 第二層映射：personality -> RL parameters

建議把人格拆成四個中介訊號，再映到學習動力參數。

| latent signal | 來源人格 | 作用 |
|---|---|---|
| `z_drive` | impulsiveness, greed, ambition, optimism | 提高反應速度與 aggressive 更新強度 |
| `z_guard` | caution, suspicion, fearfulness, stability_seeking | 提高 defensive 反應與風險抑制 |
| `z_temporal` | persistence, patience, stability_seeking | 控制長期記憶、更新平滑、抗噪性 |
| `z_noise` | randomness, curiosity, optimism | 控制探索與非常規轉向 |

對應到 RL 參數時，第一版建議只開四個出口：

| RL parameter | 建議來源 | 說明 |
|---|---|---|
| `alpha_i` | `z_drive - z_temporal` | 快學與慢學的個體異質性 |
| `beta_i` | `z_guard + z_temporal - z_noise` | exploit / explore 張力 |
| `r_A, r_D, r_B` | strategy prior + bounded personality offsets | P21 的正式產品化入口 |
| `risk_sensitivity_i` | `z_guard + fearfulness - optimism` | 決定同事件下 risk 累積速度 |

### 4.3 P21 在產品中的角色

P21 的 `mild_cw = [1.2, 1.0, 0.8]` 不應被視為純研究技巧，而應被視為世界中三類行為反應速度的基線幾何。

因此第一版建議：

1. 全域 baseline 使用 BL2 的 `mild_cw`
2. personality 只在其上做小幅、bounded 的 per-player 修飾
3. 不讓 personality 直接推翻 BL2 幾何，只允許在其附近形成個體差異

也就是：

`r_s(player) = r_s(BL2 baseline) + small personality-conditioned offset`

而不是讓每個 player 都有完全自由的三軸倍率。

這樣做的理由是：

1. 保住已驗證的旋轉結構
2. 讓 personality 影響「誰更容易先學到哪種反應」
3. 避免一開始就把產品 runtime 送回無結構的搜尋空間

### 4.4 不應直接映射的項目

第一版不建議讓 personality 直接控制：

1. payoff matrix `a / b / cross`
2. analysis gate thresholds
3. world-level event schema 結構
4. Q-rotation bias 之類已證實會破壞自組織的外加力

---

## 5. Event 與 World 的耦合方式

### 5.1 Event -> agent

事件對 agent 的影響應經過三條明確 channel：

1. reward channel：改變本輪 reward / utility
2. risk channel：改變 risk / stress / death pressure
3. information channel：改變不確定性與可觀測訊號

### 5.2 Agent -> event choice

agent 不直接選 strategy label，而是對事件提供的 action candidates 做評分。
建議流程：

1. 先把 action 嵌入 aggressive / defensive / balanced 的混合向量
2. 再以當前 RL policy `pi_i` 對 action 做先驗加權
3. 最後由 personality / world state / immediate utility 補上局部偏置

### 5.3 World -> event distribution

world layer 的主要責任是控制 event mix，而不是替 agent 做決策。

第一版建議世界只動四種東西：

1. 各 event family 的出現機率
2. reward / risk multiplier
3. 信息透明度
4. life 段落中的壓力節奏

### 5.4 Population -> world feedback

Little Dragon 應讀取的是群體行為統計，而不是單一 seed 的局部噪音。

建議第一版只讀：

1. `p_aggressive / p_defensive / p_balanced`
2. Stage 3 方向一致性摘要
3. risk / death distribution
4. dominant event family shares

world feedback 的目標不是消滅 BL2 旋轉，而是讓旋轉在不同 pressure regime 下換形但不塌陷。

---

## 6. 正式 runtime 藍圖

### 6.1 第一版 runtime 應有的模組責任

| layer | 建議責任 |
|---|---|
| `players/` | 保存 per-player personality、Q-table、risk 狀態、life memory |
| `evolution/` | 保持 RL update、softmax、payoff helpers 的 pure functions |
| `dungeon/` | 事件載入、事件效果、世界狀態到事件分佈的映射 |
| `simulation/` | 組裝一場 run、輸出 CSV/TSV/decision/provenance |
| `analysis/` | 只讀輸出，不依賴 runtime 內部資料結構 |

### 6.2 第一版不要做的事

1. 不要把所有舊 replicator mode 與新 RL runtime 混成同一套隱式語意
2. 不要讓 event engine 直接改寫 analysis 指標
3. 不要讓 personality 更新與 world update 在同一函式中互相纏繞

---

## 7. 研究可回歸要求

完整藍圖必須保留研究管線，而不是把產品與研究切開。

第一版至少要保留：

1. round-level timeseries
2. per-life summary
3. per-seed provenance
4. personality snapshot
5. world-state snapshot
6. event provenance

只有這樣，產品 runtime 裡的 emergent behavior 才能回到既有 cycle metrics 管線做比較。

---

## 8. 建議實作順序

1. 先把 BL2 變成正式 runtime anchor
2. 再把 12D personality 映到 RL parameters，而不是先擴大內容池
3. 接著把 event layer 接進 RL action scoring
4. 最後才讓 Little Dragon 進入 in-loop world adaptation

如果順序反過來，會很容易回到「世界很豐富，但 agent 動力是假的」的舊問題。

---

## 9. v1 成功條件

Blueprint v1 不要求一次把整個遊戲做完；它要求的是這三件事同時成立：

1. 玩家的遺言真的能改變人格，再經由 learning dynamics 影響多輪行為
2. 世界事件與壓力真的能改變 agent 的旋轉結構，而不是只改表面數值
3. 整個系統仍能輸出可比較、可回歸、可分析的研究資料

做到這三件事，Personality Dungeon 才算從研究原型走到產品藍圖。