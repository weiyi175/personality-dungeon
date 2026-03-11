personality-dungeon/
│
├── README.md
├── requirements.txt
├── config/
│   ├── game_config.yaml
│   ├── evolution_config.yaml
│   └── risk_config.yaml
│
├── core/                       # 核心博弈引擎
│   ├── game_engine.py
│   ├── state.py
│   ├── strategy.py
│   ├── utility.py
│   ├── equilibrium.py
│   └── stackelberg.py
│
├── players/                    # 玩家人格模型
│   ├── base_player.py
│   ├── personality_traits.py
│   ├── strategy_library.py
│   └── adaptive_player.py
│
├── dungeon/                    # 地下城控制邏輯（GAI）
│   ├── dungeon_ai.py
│   ├── popularity_penalty.py
│   ├── rule_mutator.py
│   └── event_generator.py
│
├── evolution/                  # 演化博弈層
│   ├── replicator_dynamics.py
│   ├── population_model.py
│   └── mutation.py
│
├── simulation/
│   ├── run_simulation.py
│   ├── experiment_runner.py
│   └── logger.py
│
├── analysis/
│   ├── metrics.py
│   ├── visualization.py
│   └── convergence_test.py
│
├── api/                        # 若未來對外開放
│   ├── server.py
│   └── schemas.py
│
└── tests/
    ├── test_engine.py
    ├── test_evolution.py
    └── test_equilibrium.py

1️⃣ core/ —— 核心博弈引擎層
定位

負責「單回合博弈邏輯」，不涉及人格、不涉及演化、不涉及長期族群變化。

它只回答：

在當前規則下，行動 → 報酬如何計算？

應該負責的功能
① Game 流程控制

回合開始

玩家提交策略

計算 payoff

更新 utility

② 狀態管理（state.py）

當前回合數

玩家行動紀錄

地下城規則參數

③ 策略抽象（strategy.py）

定義策略類型（Aggressive / Safe / Random 等）

可擴充策略空間

④ 效用函數（utility.py）

reward 計算方式

風險懲罰

popularity penalty

⑤ 均衡分析（equilibrium.py）

Nash equilibrium 檢測

Best response 計算

⑥ Stackelberg 模型（stackelberg.py）

地下城先手

玩家後手

求解 leader-follower equilibrium

這層不應該做：

不更新族群比例

不改變人格權重

不做長期演化

2️⃣ players/ —— 玩家人格模型層
定位

定義「玩家是什麼」。

應該負責的功能
① BasePlayer

utility 累積

策略選擇接口

② personality_traits.py

風險偏好

攻擊傾向

合作傾向

模仿傾向

③ strategy_library.py

可選策略集合

每種策略對應 payoff 風險特性

④ adaptive_player.py

根據歷史結果調整策略概率

Reinforcement Learning 簡化版本

這層不應該做：

不計算整體族群比例

不決定地下城規則

不處理 Nash

3️⃣ dungeon/ —— 地下城 AI 層（Leader）
定位

這是 Stackelberg 模型中的「領導者」。

負責：

調整規則，使某些策略不再過度主導。

應該負責的功能
① dungeon_ai.py

計算策略報酬

加入 popularity penalty

控制風險係數 gamma

② popularity_penalty.py

統計策略使用頻率

動態懲罰熱門策略

③ rule_mutator.py

改變 payoff 結構

調整 reward scaling

④ event_generator.py

隨機事件

改變局部 payoff matrix

這層不應該做：

不改變玩家人格

不更新族群權重

不做 replicator dynamics

4️⃣ evolution/ —— 演化博弈層
定位

負責長期族群動態。

回答：

哪些人格會變多？哪些會滅絕？

應該負責的功能
① replicator_dynamics.py

根據 utility 更新族群比例

經典公式：

𝑥
𝑖
˙
=
𝑥
𝑖
(
𝑢
𝑖
−
𝑢
ˉ
)
x
i
	​

˙
	​

=x
i
	​

(u
i
	​

−
u
ˉ
)
② population_model.py

人格分布

策略比例向量

③ mutation.py

少量隨機人格變異

避免固定點

這層不應該做：

不計算單回合 reward

不決定地下城規則

5️⃣ simulation/ —— 控制與實驗層
定位

負責把所有模組串起來。

回答：

整個系統跑 1000 回合會發生什麼？

應該負責的功能
① run_simulation.py

初始化玩家

初始化地下城

執行多回合

呼叫 evolution 更新族群

② experiment_runner.py

改變 gamma

改變 mutation rate

做參數掃描

③ logger.py

記錄策略比例

記錄平均 utility

輸出 CSV

整體邏輯分層圖
players ──┐
           │
           ▼
         core  ←── dungeon
           │
           ▼
        evolution
           │
           ▼
       simulation
關鍵設計原則
1️⃣ 單一責任原則

每層只負責一種時間尺度：

層級	時間尺度
core	單回合
dungeon	單回合規則
players	個體決策
evolution	長期族群
simulation	整體實驗
2️⃣ 不要交叉引用

core 不 import evolution

players 不 import dungeon

evolution 不 import core 計算細節

目前你該怎麼做？

現在只建議：

先實作：

core/game_engine.py

players/base_player.py

dungeon/dungeon_ai.py

simulation/run_simulation.py

演化層先不要動。

如果你願意，我可以幫你畫出：

「MVP 版最簡依賴關係圖」

讓你清楚第一階段應該寫到哪裡停手。

「MVP 版最簡依賴關係圖」

以下是你目前階段應採用的 MVP 最簡依賴關係圖。

目標：
只實作「一輪動態博弈可以跑起來」，不包含演化層、不包含均衡分析。

✅ MVP 模組依賴圖（最小可運行版本）
simulation/run_simulation.py
            │
            ▼
     core/game_engine.py
        ▲              ▲
        │              │
players/base_player.py  dungeon/dungeon_ai.py
📦 每個模組在 MVP 階段負責什麼？
1️⃣ players/base_player.py

責任：

保存 utility

提供 choose_strategy()

提供 update_utility()

不應：

計算 reward

統計族群

修改規則

2️⃣ dungeon/dungeon_ai.py

責任：

計算 reward

根據 popularity 給懲罰

不應：

呼叫玩家方法

更新玩家人格

控制回合流程

3️⃣ core/game_engine.py

責任：

控制單回合流程

讓玩家選策略

向 dungeon 要 reward

更新玩家 utility

不應：

管理長期族群

改變人格權重

進行 replicator dynamics

4️⃣ simulation/run_simulation.py

責任：

初始化玩家

初始化 dungeon

建立 GameEngine

跑多回合

不應：

計算 reward

修改策略邏輯

🚫 MVP 階段不要碰的資料夾

evolution/

analysis/

equilibrium.py

stackelberg.py

mutation.py

這些都是第二階段。

🧠 執行流程（單回合）
GameEngine.step()

1. 玩家選策略
2. Dungeon 計算 reward
3. 玩家更新 utility
4. Dungeon 更新 popularity
🧱 MVP 階段你應該只建立這 4 個檔案
core/game_engine.py
players/base_player.py
dungeon/dungeon_ai.py
simulation/run_simulation.py

其他資料夾保持空著即可。

📌 為什麼這樣設計？

因為現在目標是：

確保整個專案架構是「可執行」的
而不是「完整的研究架構」

如果這四個檔案能成功跑 100 回合：

✔ import 沒問題
✔ 模組責任清楚
✔ CLI 可執行
✔ 架構穩定

然後才開始加：

replicator dynamics

Nash 分析

Stackelberg 模型