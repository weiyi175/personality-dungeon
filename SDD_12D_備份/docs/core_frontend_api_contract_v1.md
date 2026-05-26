# Core / Frontend API Contract v1

本文件定義 Python Core Engine 與 Godot Frontend 之間的第一版穩定輸出契約。

核心原則只有一條：**核心內部機制可以改，對外輸出格式不能破壞**。

本文件是 [SDD.md](../SDD.md) 與 [docs/personality_rl_runtime_bridge_v1.md](personality_rl_runtime_bridge_v1.md) 的 companion spec。若未來需要破壞相容性，應先規劃新版本契約，不得直接在 v1 上改名或刪欄位。

---

## 1. 目標

1. 讓前端只依賴穩定的 JSON schema，不直接碰 Python 內部物件。
2. 讓核心能自由重構 decision engine、event system、memory、governance 與 simulation 細節。
3. 讓回歸測試可以跨版本比較輸出，而不是比較內部類別實作。
4. 讓 future UI / scene / animation 層只需要根據固定欄位渲染。

---

## 2. 非目標

1. 不鎖死 HTTP、WebSocket、stdin/stdout、IPC 或 gRPC 等傳輸方式。
2. 不鎖死內部資料庫、快取、random seed、排程器或事件生成演算法。
3. 不把整個世界狀態 dump 成無限制的大物件。
4. 不在 v1 內解決 N 維策略空間或新 family 擴充。

---

## 3. 設計原則

### 3.1 穩定優先

1. 任何前端需要長期依賴的欄位，都必須先進 spec，再進實作。
2. v1 內允許新增欄位，但不得刪欄位、改欄位名、改欄位意義。
3. 如果真的需要破壞性改動，應升到 v2，不要偷偷覆寫 v1。

### 3.2 結構優先於內部表示

1. 前端只看 response schema，不看 core class structure。
2. 內部如果從 list 改成 dataclass、從 dataclass 改成 pure dict、從同步改成非同步，都不應影響此契約。
3. 所有複雜內部狀態，應透過穩定 projection 映射成 v1 欄位。

### 3.3 Additive only

1. v1 的穩定欄位只能做 additive change。
2. 新資訊先放進 `extensions`，確認需要長期存在後再進下一版正式欄位。
3. 不要用同名欄位承接不同語意。

---

## 4. API Surface

本 spec 只鎖最小公開 surface。

### 4.1 `reset`

用途：建立新 session，回傳初始 frame。

輸出：回傳一個完整的 `reset response`，其結構與 `step response` 一致。

### 4.2 `step`

用途：消費一個 action，推進核心一個 tick，並回傳新的 frame。

輸出：回傳一個完整的 `step response`。

### 4.3 `snapshot`

用途：在不推進 tick 的情況下，取得目前 state 的完整快照。

輸出：回傳一個完整的 `snapshot response`。

### 4.4 `error`

用途：任一操作失敗時，仍回傳可解析的錯誤 response。

輸出：結構仍需符合本 spec 的頂層 envelope。

---

## 5. 頂層 Response Envelope

所有 v1 response 都必須包含下列頂層欄位：

1. `api_version`
2. `kind`
3. `ok`
4. `session_id`
5. `tick`
6. `state_hash`
7. `event`
8. `choices`
9. `personality_delta`
10. `emotion`
11. `world_state`
12. `player_state`
13. `result`
14. `warnings`
15. `error`
16. `extensions`

### 5.1 欄位語意

1. `api_version`：固定格式 `MAJOR.MINOR`，例如 `"1.0"`。
2. `kind`：`"reset" | "step" | "snapshot" | "error"`。
3. `ok`：是否成功。成功為 `true`，失敗為 `false`。
4. `session_id`：單一 session 的穩定識別字串。
5. `tick`：目前 frame 所屬的 tick。`reset` 時通常為 `0`。
6. `state_hash`：可選的 canonical hash，用於回歸與一致性檢查。
7. `event`：當前可視事件或場景資訊。
8. `choices`：當前可選動作清單。
9. `personality_delta`：本 tick 對人格向量造成的變化。
10. `emotion`：本 tick 的情緒摘要。
11. `world_state`：本 tick 的世界壓力摘要。
12. `player_state`：本 tick 的玩家可視狀態。
13. `result`：本次 action 的結果摘要。
14. `warnings`：非致命警告清單。
15. `error`：錯誤資訊；成功時為 `null`。
16. `extensions`：保留給實驗性欄位，不屬於穩定契約。

### 5.2 固定預設值

若某欄位在某次 response 中沒有實際內容，也必須保留欄位本身，並使用下列預設：

1. `event = null`
2. `choices = []`
3. `personality_delta = {}`
4. `emotion = {}`
5. `world_state = {}`
6. `player_state = {}`
7. `result = null`
8. `warnings = []`
9. `error = null`
10. `extensions = {}`

這個規則的目的，是讓前端 parser 不需要因為成功 / 失敗 / 初始 frame 而切換不同解析路徑。

---

## 6. 欄位 schema

### 6.1 `event`

`event` 是前端要渲染的主要內容載體。v1 建議固定欄位如下：

1. `event_id`：穩定識別字串。
2. `event_type`：事件型別，例如 `combat`、`choice`、`story`、`system`。
3. `title`：短標題。
4. `summary`：可直接顯示在 UI 的摘要文字。
5. `tags`：字串陣列，用於分類與圖層判斷。
6. `severity`：0 到 1 的浮點數，表示事件強度。
7. `phase`：可選字串，用於表示事件所處階段。

規格要求：

1. `event_id`、`event_type`、`title`、`summary` 必須存在。
2. `tags` 必須存在，但可為空陣列。
3. `severity` 應維持在 `[0, 1]`。

### 6.2 `choices`

`choices` 是前端可以送給玩家點擊或 AI 決策的候選動作清單。每個元素建議固定欄位如下：

1. `choice_id`：穩定識別字串。
2. `label`：UI 顯示文字。
3. `strategy`：`"aggressive" | "defensive" | "balanced" | "custom"`。
4. `enabled`：是否可選。
5. `selected`：本 tick 是否被選中。
6. `description`：可選長說明。
7. `preview`：可選預覽物件，用來顯示預期結果。

`preview` 若存在，建議包含：

1. `expected_reward`
2. `expected_risk`
3. `expected_personality_delta`
4. `confidence`

規格要求：

1. `choice_id` 必須唯一。
2. `enabled` 與 `selected` 必須是 boolean。
3. `strategy` 若不是三策略之一，必須明確標成 `custom`，不要偽裝成既有類別。

### 6.3 `personality_delta`

`personality_delta` 是本 tick 對人格向量的變化摘要。

v1 的 canonical personality basis 使用固定 12 維：

1. `impulsiveness`
2. `greed`
3. `ambition`
4. `caution`
5. `suspicion`
6. `fearfulness`
7. `persistence`
8. `stability_seeking`
9. `patience`
10. `optimism`
11. `randomness`
12. `curiosity`

建議欄位如下：

1. `vector`：物件，內含上述 12 維的 delta。
2. `norm`：delta 的摘要大小。
3. `confidence`：0 到 1 的浮點數。

規格要求：

1. 未變動的維度可以不出現在 `vector` 中。
2. 若沒有人格變化，`personality_delta` 應為空物件 `{}`。
3. 若出現 delta，對應值應可正負，但每一維都應維持在合理的 bounded range；超出範圍時應先由 adapter 截斷再輸出。

### 6.4 `emotion`

`emotion` 是可視化用的情緒摘要，不是核心內部全狀態。

v1 建議欄位如下：

1. `valence`：範圍 `[-1, 1]`。
2. `arousal`：範圍 `[0, 1]`。
3. `dominance`：範圍 `[0, 1]`。
4. `intensity`：範圍 `[0, 1]`。

規格要求：

1. 四個欄位都應可直接數值化。
2. 若 core 尚未產生有效情緒估計，必須回傳預設值，不可省略欄位。

### 6.5 `world_state`

`world_state` 是前端可視的世界壓力摘要。v1 鎖定四軸：

1. `scarcity`
2. `threat`
3. `noise`
4. `intel`

規格要求：

1. 四個軸都必須存在。
2. 每個值都應維持在 `[0, 1]`。
3. 若某軸尚未啟用，值仍應存在，但可為 0.0。

### 6.6 `player_state`

`player_state` 是單一代理在當前 tick 的可視狀態摘要。

v1 建議欄位如下：

1. `health`：範圍 `[0, 1]`。
2. `stamina`：範圍 `[0, 1]`。
3. `stress`：範圍 `[0, 1]`。
4. `utility`：可為任意浮點數。
5. `alive`：boolean。
6. `personality`：完整 12 維人格快照。
7. `memory_load`：可選，範圍 `[0, 1]`。

規格要求：

1. `personality` 必須與 `personality_delta` 使用同一套 canonical basis。
2. `health`、`stamina`、`stress` 都應是 normalized 值，不應把內部原始尺度直接外露。
3. `alive=false` 時，`result.terminated` 應同步為 `true`。

### 6.7 `result`

`result` 是本次 action 造成的結果摘要。

v1 建議欄位如下：

1. `selected_choice_id`：本次實際採用的 choice id。
2. `reward`：本次回饋。
3. `utility_delta`：本次 utility 變化。
4. `risk_delta`：本次風險變化。
5. `terminated`：本次 step 是否結束該 session 或 life。
6. `termination_reason`：若 terminated，則說明原因。
7. `state_delta`：可選，表示可視 state 的改變。

規格要求：

1. `selected_choice_id` 應對應到 `choices` 中的一個 `choice_id`，或在 no-op 情況下為 `null`。
2. `terminated=false` 時，`termination_reason` 應為 `null`。
3. `state_delta` 若存在，應只包含本 tick 發生變化的欄位。

### 6.8 `error`

`error` 只在 `ok=false` 時使用。

建議欄位如下：

1. `code`：穩定機器可讀錯誤碼。
2. `message`：人類可讀訊息。
3. `retryable`：boolean。
4. `details`：可選，補充診斷資訊。

規格要求：

1. 成功 response 的 `error` 必須是 `null`。
2. 失敗 response 的 `error.code` 必須存在。
3. 前端不應依賴 `message` 做邏輯判斷，只能用 `code`。

---

## 7. 版本與相容性規則

### 7.1 v1 允許的變更

1. 新增 `extensions` 內的實驗性欄位。
2. 新增可選欄位，但不能改既有欄位的語意。
3. 增加更完整的前端渲染資訊，但不得移除舊欄位。

### 7.2 v1 禁止的變更

1. 不得刪除任何頂層穩定欄位。
2. 不得更名任何頂層穩定欄位。
3. 不得把既有欄位改成另一種語意。
4. 不得把同一欄位在不同版本內混用為不同單位或不同範圍。

### 7.3 升版條件

如果未來需要下列改動，應直接規劃 v2：

1. 3 策略 simplex 改成 N 維策略空間。
2. world_state 由四軸改成不同基底。
3. personality basis 大改。
4. response 必須破壞舊 parser 才能表達新機制。

---

## 8. 不變條件

1. `api_version = "1.0"` 的 response 必須能被舊版 parser 安全解析。
2. 同一類 response 的頂層鍵集合必須固定。
3. `choices` 的順序必須由 core 明確定義，前端不得自行重排後再拿去做邏輯判斷。
4. `event`、`world_state`、`player_state` 的欄位不得依賴前端臨時補值。
5. `extensions` 內的資料不屬於穩定契約，前端必須忽略未知鍵。
6. 所有 response 必須是可 JSON 序列化的純資料，不能回傳 Python 物件實例。

---

## 9. 範例 Response

```json
{
  "api_version": "1.0",
  "kind": "step",
  "ok": true,
  "session_id": "session_20260429_0001",
  "tick": 12,
  "state_hash": "sha256:7ce1f2a8b0d3e1d1",
  "event": {
    "event_id": "evt_00091",
    "event_type": "choice",
    "title": "門後的回聲",
    "summary": "你聽見門後傳來微弱回聲，三種反應都可以選擇。",
    "tags": ["story", "decision"],
    "severity": 0.42,
    "phase": "active"
  },
  "choices": [
    {
      "choice_id": "c_aggressive",
      "label": "直接推門",
      "strategy": "aggressive",
      "enabled": true,
      "selected": false,
      "preview": {
        "expected_reward": 0.25,
        "expected_risk": 0.58,
        "confidence": 0.71
      }
    },
    {
      "choice_id": "c_defensive",
      "label": "先觀察再進入",
      "strategy": "defensive",
      "enabled": true,
      "selected": true,
      "preview": {
        "expected_reward": 0.12,
        "expected_risk": 0.18,
        "confidence": 0.83
      }
    },
    {
      "choice_id": "c_balanced",
      "label": "緩慢推進",
      "strategy": "balanced",
      "enabled": true,
      "selected": false,
      "preview": {
        "expected_reward": 0.19,
        "expected_risk": 0.31,
        "confidence": 0.77
      }
    }
  ],
  "personality_delta": {
    "vector": {
      "caution": 0.01,
      "suspicion": 0.02,
      "fearfulness": 0.00,
      "patience": 0.01
    },
    "norm": 0.024,
    "confidence": 0.88
  },
  "emotion": {
    "valence": -0.08,
    "arousal": 0.41,
    "dominance": 0.36,
    "intensity": 0.29
  },
  "world_state": {
    "scarcity": 0.34,
    "threat": 0.57,
    "noise": 0.22,
    "intel": 0.63
  },
  "player_state": {
    "health": 0.82,
    "stamina": 0.59,
    "stress": 0.24,
    "utility": 1.87,
    "alive": true,
    "memory_load": 0.31,
    "personality": {
      "impulsiveness": 0.18,
      "greed": 0.11,
      "ambition": 0.07,
      "caution": 0.23,
      "suspicion": 0.19,
      "fearfulness": 0.14,
      "persistence": 0.25,
      "stability_seeking": 0.21,
      "patience": 0.29,
      "optimism": 0.16,
      "randomness": 0.09,
      "curiosity": 0.31
    }
  },
  "result": {
    "selected_choice_id": "c_defensive",
    "reward": 0.12,
    "utility_delta": 0.04,
    "risk_delta": -0.03,
    "terminated": false,
    "termination_reason": null
  },
  "warnings": [],
  "error": null,
  "extensions": {}
}
```

---

## 10. 實作映射建議

這份 spec 的責任歸屬應該如下：

1. `core/`：只負責計算內部狀態與 transition，不直接承擔 API 版本管理。
2. `api/schemas.py`：只負責定義與驗證此 spec 的公開 schema。
3. `api/server.py`：只負責把 core 結果序列化成此 spec 的 response。
4. `simulation/`：只在需要批次或研究輸出時，重用同一套 schema projection。
5. `analysis/`：只能讀輸出結果，不得反向依賴 runtime 物件。

---

## 11. v1 驗收標準

1. 前端只要讀 response，就能完成畫面渲染，不需要碰 Python core object。
2. 核心內部更換演算法後，只要輸出 schema 不變，前端不需要改碼。
3. 所有 response 都能被 JSON round-trip。
4. `choices`、`event`、`world_state`、`player_state` 的欄位在成功與失敗路徑上都能保持解析一致。
5. 若要加入新欄位，先更新 spec 與回歸測試，再改實作。

---

## 12. 後續建議

如果這份 v1 草案被接受，下一步應該做兩件事：

1. 把這份 spec 轉成 `api/schemas.py` 的正式 schema。
2. 補一個最小的 contract test，確認 `reset` / `step` / `snapshot` 的輸出鍵集合固定不變。
