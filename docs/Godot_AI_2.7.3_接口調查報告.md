# Godot AI 2.7.3 — 接口調查報告

> 調查日期：2026-06-14 ｜ 環境：WSL2 (Claude Code) + Windows Godot 專案
> 專案路徑：`/mnt/c/Users/n1166/personality-dungeon`（Godot 端）

## 1. 這個外掛是什麼

`addons/godot_ai/`（plugin.cfg：name="Godot AI", **version=2.7.3**）是
[hi-godot/godot-ai](https://github.com/hi-godot/godot-ai) 的官方外掛。它把一個
**MCP（Model Context Protocol）伺服器**嵌進 Godot 編輯器，讓 Claude Code / VSCode
等 AI 客戶端可以**直接操作正在運行的 Godot 編輯器**（不是改檔案而已，是對 live editor 下指令）。

### 連線架構（你提到的 WS 9500 / HTTP 8000）

```
Claude Code / VSCode ──HTTP──▶ Python FastMCP 伺服器 ──WebSocket──▶ Godot 編輯器外掛
  (MCP client)        :8000/mcp   (uv 啟動)              :9500        (addons/godot_ai)
```

- **HTTP 8000** — MCP 客戶端的接口，固定 `http://127.0.0.1:8000/mcp`
- **WebSocket 9500** — Python 伺服器 ↔ Godot 編輯器之間的內部通道
- Python 伺服器由外掛在「啟用 plugin + 開著編輯器」時用 `uv` 自動拉起

## 2. 2.7.3 / 2.7.x 更新內容（官方 release notes）

| 版本 | 日期 | 重點 |
|------|------|------|
| **2.7.3** | 06-14 | 新增 **structured log cursor reads**（結構化日誌游標讀取，可增量抓 log）|
| 2.7.2 | 06-12 | 新增 **Godot ClassDB API 內省**（讓 AI 查得到引擎類別資訊）|
| 2.7.1 | 06-11 | 升級 fastmcp 至 `>=3.0.0,<3.5.0`；log 讀取中**夾帶編輯器錯誤上下文** |
| 2.7.0 | 06-10 | **安全強化（含破壞性變更）**：所有檔案操作做路徑封閉驗證；`--allow-host` 控管 LAN 信任邊界；設定檔強制 0600 權限；遙測可關閉。錯誤碼從泛用 `INVALID_PARAMS` 改為區分 `MISSING_REQUIRED_PARAM` / `VALUE_OUT_OF_RANGE` |

對我們最有用的是 2.7.x 整條線：**log/錯誤可被 AI 增量讀取 + ClassDB 內省**，等於除錯時我能直接讀到編輯器報的錯。

## 3. 能做哪些工作 — 完整工具清單（共 ~40 tools / 120+ ops）

從 `addons/godot_ai/tool_catalog.gd` 取得（這是 `domains.py` 的鏡像，CI 會驗證兩者一致）。
4 個 **Core 工具永遠開啟**，其餘 21 個 domain 可在 dock 勾選關閉。

**Core（不可關）**：`editor_state`、`node_get_properties`、`scene_get_hierarchy`、`session_activate`

| Domain | 工具 |
|--------|------|
| scene | `scene_open`、`scene_save`、`scene_manage` |
| node | `node_create`、`node_find`、`node_set_property`、`node_manage` |
| script | `script_create`、`script_attach`、`script_patch`、`script_manage` |
| signal | `signal_manage`（接線/斷線）|
| project | `project_run`、`project_manage` |
| game | `game_manage`（執行中遊戲控制）|
| editor | `editor_screenshot`、`editor_reload_plugin`、`logs_read`、`editor_manage` |
| testing | `test_run`、`test_manage` |
| filesystem | `filesystem_manage` |
| resource | `resource_manage` |
| ui / theme | `ui_manage`、`theme_manage` |
| material / particle | `material_manage`、`particle_manage` |
| animation | `animation_create`、`animation_manage` |
| camera / environment* | `camera_manage`（*environment 由 handler 提供）|
| audio | `audio_manage` |
| input_map | `input_map_manage` |
| autoload | `autoload_manage` |
| api | `api_manage`（ClassDB 內省，2.7.2 新增）|
| batch | `batch_execute`（一次跑多步）|
| client | `client_manage` |

> 對照 `addons/godot_ai/handlers/` 還有 physics_shape、curve、texture、control_draw_recipe
> 等更細的 handler 支撐這些工具。

### 對我們這個專案（personality-dungeon）特別實用的場景
- **`editor_screenshot` + `game_manage`** — 取代現在 headless walk-strip 截圖流程
  （見記憶 `godot-screenshot-harness`），可直接截編輯器/執行畫面，不必再對付
  Windows Godot 殭屍鎖 `.godot` 的問題。
- **`logs_read`（2.7.3 增量游標）+ editor 錯誤上下文** — DungeonLifecycleSceneV2 跑
  full-flow smoke 時，後端 fallback 到 stale 值（記憶 `backend-restart-required`）
  這類問題，AI 能直接讀到編輯器 log 判斷。
- **`scene_get_hierarchy` / `node_*` / `signal_manage`** — 直接驗證
  ProximityMeter / EventChoicePanel 的 z-layer 與 set_displacement 接線
  （記憶 `v2-delta-radar-wiring`），不必人工開編輯器點。

## 4. 目前狀態：**接口尚未連線（我這回合無法 live 呼叫）**

實測結果：

| 檢查項 | 結果 |
|--------|------|
| HTTP `127.0.0.1:8000/mcp` | ❌ 沒在 listen（`curl` 空回應）|
| WebSocket `9500` | ❌ 沒在 listen |
| Godot / python 伺服器 process | ❌ 找不到 |
| `claude mcp list` | ❌ **No MCP servers configured** |

原因：MCP 伺服器只有在「**Godot 編輯器開著 + plugin 已啟用**」時才會被 `uv` 拉起；
目前編輯器沒開，所以 8000/9500 都是空的。而且 Claude Code 端也還沒註冊這個 MCP server。

**重要限制**：Claude Code 的 MCP server 是在「**啟動 session 時**」載入的。即使現在用
`claude mcp add` 註冊，這個進行中的 session 也不會出現新工具 — 需要**重開一個
Claude Code session** 才會掛上。所以這回合我能做完整調查，但無法實際對 Godot 下指令。

## 5. 要讓我能實際驅動 Godot，請完成 3 步

1. **裝 uv**（Python 伺服器靠它，Windows PowerShell）：
   ```powershell
   powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
   ```
2. **開 Godot 編輯器** → Project Settings → Plugins → 啟用 **Godot AI** →
   在 **Godot AI dock** 選 **Claude Code** 按 **Configure**。
   （這會把 `godot-ai` 寫進 `~/.claude.json` 的 `mcpServers`，type=http，
   url=`http://127.0.0.1:8000/mcp`。）此時 8000/9500 應開始 listen。
3. **重開 Claude Code session**（讓它載入新註冊的 MCP server）。

> 手動註冊等價指令（dock 的 Configure 按鈕就是跑這個）：
> ```bash
> claude mcp add --scope user --transport http godot-ai http://127.0.0.1:8000/mcp
> ```
> 注意：dock 也支援 VSCode、Cursor、Codex、Antigravity 等 16+ 客戶端一鍵設定
> （見 `addons/godot_ai/clients/`）。

完成後 `claude mcp list` 會看到 `godot-ai`，我就能在新 session 直接呼叫上面那 ~40 個工具。

## 來源
- 本機外掛原始碼：`addons/godot_ai/`（plugin.cfg、tool_catalog.gd、clients/claude_code.gd、handlers/）
- [github.com/hi-godot/godot-ai](https://github.com/hi-godot/godot-ai) 及其 releases
- [Godot AI — Godot Asset Library (asset 5050)](https://godotengine.org/asset-library/asset/5050)
