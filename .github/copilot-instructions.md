# Copilot Workspace Instructions (Personality Dungeon)

本專案採用研究型 SDD（Spec-Driven Development）。所有變更必須以 [SDD.md](../SDD.md) 與 [研發日誌.md](../研發日誌.md) 的定義為準，優先維持可重現、可回歸、可比較。

## 必須遵守（Hard rules）

### 1) SDD 流程（先 Spec、後程式碼）
- 若要變更任何「行為契約」，必須先更新 [SDD.md](../SDD.md) 再改碼：
  - CSV 欄位/語意、payoff 定義與時間索引、演化更新規則、invariants（不變條件）。
- 修改程式碼前，先確認變更落在哪一層（`core/` / `dungeon/` / `players/` / `evolution/` / `simulation/` / `analysis/`），且不跨層放錯職責。

### 2) 分層不變條件（Architecture invariants）
- `analysis/` 不得 import `simulation/`（避免循環依賴）；分析只能讀 CSV 或純函式輸入。
- `evolution/` 不做 I/O、不依賴 plotting。
- `simulation/` 才負責 I/O 與 CSV schema（資料契約）。

### 3) 一律在 venv 中執行（Execution invariants）
- 任何要在終端機執行的 Python 指令，一律使用專案 venv：
  - 使用 `./venv/bin/python -m ...`
  - 或沿用既有腳本慣例：設定 `PYTHON_BIN=./venv/bin/python`
- 不要使用系統 `python` 或 `python3`（避免環境不一致與不可重現）。
- 測試/格式化同理：例如 `./venv/bin/pytest -q`。

### 4) 不要破壞既有研究管線
- 尊重既有 CLI 介面、既有 outputs/ 命名慣例、以及 SDD 4.6 的 protocol lock。
- 若調整 cycle metrics（Stage1/2/3）閾值或語意，必須在 Spec/日誌中同步更新，並更新相對應回歸測試。

## 建議做法（Strong defaults）
- 任何「研發進度」判讀以 [研發日誌.md](../研發日誌.md) 與 `outputs/` 既有產物為準；若不確定，先讀檔再下結論。
- 優先提供可重現命令（使用 venv 路徑、固定 seed、寫明 burn-in/tail 視窗語意）。
