# SDD：遺言輸入與伺服器連線設計

> 對應分支：`feature/phase4-todos`
> 最後更新：2026-06-08

---

## 1. 完整信號流程圖

```
╔══════════════════════════════════════════════════════╗
║  玩家輸入遺言文字（最多 20 字）                        ║
╚══════════════════════════════════════════════════════╝
                        │
                  按「分析遺言中的人格」
                        ↓
┌─────────────────────────────────────┐
│  WillInputPanel                     │
│  _on_analyse_pressed()              │
│  → emit will_submitted(text)        │
└─────────────────────────────────────┘
                        │
                        ↓
┌─────────────────────────────────────┐
│  DungeonLifecycleScene              │
│  _on_will_submitted(text)           │
│  → _controller.submit_will(text)    │
└─────────────────────────────────────┘
                        │
                        ↓
┌─────────────────────────────────────┐
│  DungeonLifecycleController         │
│  submit_will(text)                  │
│  → state = INFERRING                │
│  → _infer_client.infer(text)        │
└─────────────────────────────────────┘
                        │
                        ↓
┌─────────────────────────────────────┐
│  PersonalityInferClient             │
│  infer(text)                        │
│  → POST /personality/infer_sbert    │  ←─── 第一次網路呼叫
└─────────────────────────────────────┘
                        │
              後端 SBERT + MLP 推論
                        │
              {"vector": {trait: float × 9}}
                        ↓
┌─────────────────────────────────────┐
│  PersonalityInferClient             │
│  _on_completed()                    │
│  → emit infer_completed(Array[9])   │
└─────────────────────────────────────┘
                        │
                        ↓
┌──────────────────────────────────────────────────────┐
│  DungeonLifecycleController                          │
│  _on_infer_completed(vector)                         │
│  ① _apply_sub_critical() → _will_personality        │
│  ② _compute_recklessness() → R, intensity, cadence  │
│  ③ state = WRITING_WILL（等待玩家確認）               │
│  ④ emit infer_preview_ready(_will_personality)       │
│  ⑤ emit will_dynamics_ready(R, intensity, cadence)   │
└──────────────────────────────────────────────────────┘
          │                          │
          ↓                          ↓
┌──────────────────┐      ┌─────────────────────────┐
│ WillInputPanel   │      │ WillInputPanel           │
│ show_analysis()  │      │ show_dynamics()          │
│ → 雷達圖更新      │      │ → 魯莽度 / 強度 / 節奏   │
│ → 接近度顯示      │      │   文字顯示               │
└──────────────────┘      └─────────────────────────┘

                    玩家確認後按「確認，開始新的一生」
                                   │
                        emit will_confirmed()
                                   ↓
┌──────────────────────────────────────────────────────┐
│  DungeonLifecycleController                          │
│  confirm_will()                                      │
│  → state = INFERRING                                 │
│  → POST /rl_sessions/initialize                      │  ←─── 第二次網路呼叫
│    payload: {                                        │
│      fixed_personality_vector: _will_personality,   │
│      space_a_events_enabled: true,                  │
│      n_players, n_rounds, burn_in, seed             │
│    }                                                 │
└──────────────────────────────────────────────────────┘
                        │
              後端建立 RL session
                        │
              {session_id, snapshot:{bifurcation_proximity, ...}}
                        ↓
┌─────────────────────────────────────┐
│  DungeonLifecycleController         │
│  _on_session_initialized()          │
│  → state = RUNNING                  │
│  → 自動步進開始                      │
└─────────────────────────────────────┘
```

---

## 2. 伺服器連線設定

### 設定位置

```
src/core/AppConfig.gd  （Project → Autoload，名稱 "AppConfig"）

const API_BASE_URL: String = "http://172.31.143.82:8000"
```

**為何只需改一個地方？**

所有 HTTP client 的 `api_base_url` 屬性在 `_ready()` 裡都有：

```gdscript
func _ready() -> void:
    if api_base_url.is_empty():
        api_base_url = AppConfig.API_BASE_URL
```

若 Inspector 沒有手動覆寫，就自動使用 `AppConfig.API_BASE_URL`。
換伺服器 IP 只需改 `AppConfig.gd` 第 10 行，其他 client 不需動。

### 目前 IP 說明

`172.31.143.82` 是 WSL2 的 Windows 端 IP（透過 `ip route` 取得）。
重開機後 IP 可能變動，需重新確認並更新 `AppConfig.gd`。

---

## 3. API 端點規格

### 3-1. `POST /personality/infer_sbert`

遺言文字 → 9D Space-A 人格向量

**Request**
```json
{ "text": "我勇敢面對未知" }
```

限制：`text` 長度 ≤ 20 字（後端硬性 400 錯誤，Godot 端 TextEdit 同步限制）。

**Response 200**
```json
{
  "request_id": "0119cde9...",
  "text": "我勇敢面對未知",
  "vector": {
    "impulsiveness":       0.620,
    "assertiveness":       0.058,
    "optimism":           -0.026,
    "risk_aversion":      -0.919,
    "suspicion":           0.044,
    "endurance":          -0.050,
    "randomness":          0.430,
    "stability_seeking":  -0.510,
    "curiosity":           0.710
  },
  "model": "sbert-v7-mlp",
  "temperature": 0.0
}
```

**Godot 端轉換**（`PersonalityInferClient._on_completed`）：
依 `FEATURE_NAMES` 固定順序（impulsiveness → curiosity）把 dict 轉為 `Array[float]`。

---

### 3-2. `POST /rl_sessions/initialize`

建立 RL session，注入遺言人格為族群初始狀態

**Request**
```json
{
  "n_players": 4,
  "n_rounds": 200,
  "burn_in": 50,
  "seed": 42381,
  "fixed_personality_vector": [0.812, 0.736, 0.618, -0.914, -0.426, 0.293, 0.703, -0.465, 0.817],
  "space_a_events_enabled": true
}
```

注意：`fixed_personality_vector` 是**縮放後**的 Space-A 向量，非原始 SBERT 輸出（見第 4 節）。

**Response 200**
```json
{
  "session_id": "a3f9...",
  "snapshot": {
    "round": 0,
    "phase": "burn-in",
    "bifurcation_proximity": 0.44,
    "mean_personality": [0.812, 0.736, ...],
    ...
  }
}
```

---

## 4. Sub-Critical 縮放公式

### 問題

SBERT 輸出值域約 `[-0.5, 0.5]`（以 0 為中心），
而 Space-A 的 `BASELINE_ATTRACTOR` ≈ `[0.772, 0.724, 0.611, ...]`。

原始 SBERT 向量離 baseline 距離 ≈ 1.9（≈ 17× ε_c_app），
直接使用 → 初始 `bifurcation_proximity` 飽和至 100% → 無法量測 H1 效果。

### 解法：正規化方向縮放

```
offset = SBERT_vector - BASELINE_ATTRACTOR        # 方向向量
P₀ = BASELINE_ATTRACTOR
     + (offset / |offset|) × ε_c_app × WILL_HEADROOM
```

參數值：
- `ε_c_app = 0.11`（bifurcation 臨界半徑，來自 `bifurcation_detector.py`）
- `WILL_HEADROOM = 0.5` → 初始 proximity ≈ 44%

效果：
- 保留遺言的「**方向性**」（哪個特質更突出）
- 距離固定在安全範圍，proximity 不飽和
- 不同遺言的起始 proximity 幾乎相同（~44%），命運差異改由事件參數承載

實作位置：`DungeonLifecycleController._apply_sub_critical()`

---

## 5. 魯莽度 → 事件強度/節奏

### 為何起始 Proximity 無法區分遺言命運

後端事件方向 `compute_sensitive_direction()` 永遠回傳常數 ±V1
（`bifurcation_detector.py:151`，這是已驗證的 H1 實驗裝置核心，不可更動）。
縮放公式又讓所有遺言起始距離相同。
因此「命運差異」只能由**事件參數**承載。

### 魯莽度公式

```
raw = (impulsiveness + randomness + curiosity
       - risk_aversion - stability_seeking - endurance) / 6.0

R = sigmoid(3 × raw)    ∈ [0, 1]
```

### 事件參數映射

```
intensity_scale = 0.6 + 1.2 × R          # 事件推力倍率
cadence         = round(14 - 8 × R)      # 每 N 回合觸發一次（clamp [6, 14]）
```

### 實測崩壞時間對照（同起點 proximity ≈ 44%）

| 遺言類型 | R    | 強度倍率 | 事件節奏   | 到達 0.8 需幾回合 |
|---------|------|---------|-----------|----------------|
| 謹慎保守 | 0.11 | ×0.74   | 每 13 回合 | ~39 回合        |
| 熱情樂觀 | 0.32 | ×0.99   | 每 11 回合 | ~22 回合        |
| 勇敢冒險 | 0.51 | ×1.21   | 每 10 回合 | ~20 回合        |
| 衝動混亂 | 0.69 | ×1.43   | 每  8 回合 | ~16 回合        |

謹慎 vs 衝動：2.4× 崩壞速度差距。

實作位置：`DungeonLifecycleController._compute_recklessness()`

---

## 6. 涉及的 Godot 檔案一覽

| 檔案 | 職責 |
|-----|-----|
| `src/core/AppConfig.gd` | 伺服器 IP/port（Autoload，所有 client 的統一來源） |
| `src/core/PersonalityInferClient.gd` | `POST /personality/infer_sbert` 封裝；dict → Array 轉換 |
| `src/core/RLSessionAPIClient.gd` | `POST /rl_sessions/*` 封裝（initialize、step、apply-event） |
| `src/core/EventChoiceClient.gd` | `POST /event/choose` 封裝（被動事件，獨立 HTTPRequest） |
| `src/core/DungeonLifecycleController.gd` | 主狀態機；縮放、魯莽度計算；信號協調 |
| `src/ui/WillInputPanel.gd` | 遺言輸入 UI；20 字限制；雷達/魯莽度預覽顯示 |
| `src/ui/DungeonLifecycleScene.gd` | 信號串接；畫面切換（遺言→遊戲→崩壞→問卷） |
