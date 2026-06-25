# 友人 Web 測試 — 部署 / 測試 SOP

> 讓朋友用瀏覽器（手機/電腦）連到本機後端測試前端。封閉測試、零帳號、隔離不碰 production。
> 建立於 2026-06-25（Phase 1 隔離後端 + Phase 2 Godot web client）。更新 2026-06-26（補：網路外曝風險、測試資料、狀態速查、端點地圖）。
> 本檔為**權威版本**。`addons/godot_ai/README.md` 末尾有同內容備忘，但會在外掛升級時被覆蓋。

## 0. 架構速覽
- **兩份 repo（不同檔！）**
  - WSL `/home/user/personality-dungeon` = Python 後端 + venv + **production 資料(ecology 211)**。後端、tunnel 都在這跑。
  - Windows `/mnt/c/Users/n1166/personality-dungeon` = Godot 前端（project.godot、`src/*.gd`）。**編輯/匯出在這。**
- **隔離**：playtest 後端用 env 把四個 store 改寫到 `reports/playtest_2026/`，**不碰 production 211**（細節見 §7）。
- **連線**：友人開 `https://<tunnel>/play/` → 後端**同時** serve 遊戲（`/play` 靜態）+ API（`/pvp` 等），**同源免 CORS**；tunnel 換網址自動跟著對。
- **後端 port = 8001**（`api/server.py` 尾段 `uvicorn.run(host="0.0.0.0", port=8001)`；8000 留給 Godot AI MCP server）。
  - ⚠️ `host="0.0.0.0"` = **綁所有網卡**，所以不只 tunnel，**同網段/Windows 主機也能直接打 `http://<本機IP>:8001`**（見 §6 風險）。
- **`/play` 靜態掛載條件**：`api/server.py` 只有在 `web_build/` 目錄存在時才 `app.mount("/play", StaticFiles(html=True))`；可用 `WEB_BUILD_DIR` env 改來源目錄（預設 `web_build/`，已 gitignore）。掛在所有 API route 之後。

## 1. 開機（每次測試開始）— 在 WSL
```bash
cd /home/user/personality-dungeon
bash scripts/run_playtest_server.sh                                  # 隔離後端（防呆/detach/帶隔離 env）
~/.local/bin/cloudflared tunnel --url http://localhost:8001 --no-autoupdate > logs/cloudflared.log 2>&1 &
grep -oE "https://[a-z0-9-]+\.trycloudflare\.com" logs/cloudflared.log | head -1   # 取得網址
```
- **友人連結 = `<上面網址>/play/`**（一定要帶 `/play/`，根目錄 `/` 會回 404 是正常）。
- ⚠️ **絕不手打 `python -m api.server`** —— 沒帶隔離 env 會去讀寫 production（`reports/experiments/p7h_real_study`、`reports/ecology` 等預設路徑）。一律用 script。
- **確認都起來了**（可選）：
  ```bash
  lsof -tiTCP:8001 -sTCP:LISTEN          # 有 PID = 後端在跑
  pgrep -af 'cloudflared tunnel --url'   # 有行 = 通道在跑
  curl -s -o /dev/null -w "play=%{http_code}\n" http://localhost:8001/play/   # 200 = /play 已掛
  ```

## 2. 部署 / 更新前端（首次，或改了前端 .gd 之後）
**A. Godot 編輯器（Windows）匯出 Web**
- Project ▸ Export ▸ 選 **Web** preset ▸ **Export Project…** ▸ 取消 **Export With Debug** ▸ 存到 `web_build\index.html`（覆蓋）。
- 關鍵 preset 設定（已設好，別動）：Thread Support **OFF**、Export Mode=**all resources**、非資源過濾器 `*.json`、`experimental_virtual_keyboard` **ON**。

**B. 同步到後端（WSL）**——⚠️ **每次 Export 後都要做，否則朋友看到舊版**
```bash
cd /home/user/personality-dungeon
rm -rf web_build && mkdir -p web_build
cp /mnt/c/Users/n1166/personality-dungeon/web_build/* web_build/
```
- Windows 端匯出落在 `/mnt/c/.../web_build/`，但後端 serve 的是 **WSL 端**的 `web_build/`，兩者不同檔，必須 copy。
- StaticFiles 即時讀盤 → **改前端、copy 完免重啟後端**（只有改 `api/`、`simulation/` 才要重啟，見 §9）。

**C. 驗證**
```bash
curl -s -o /dev/null -w "%{http_code}\n" https://<tunnel>/play/          # 200
curl -s -I https://<tunnel>/play/index.pck | grep -i content-length      # 確認是新 build 大小
```

## 3. 簡單測試 SOP（4 項煙霧測試）— 開 `https://<tunnel>/play/`
| # | 測什麼 | 通過長相 |
|---|---|---|
| 1 亂碼 | UI 文字 | **正常中文（楷體）**，非豆腐框 |
| 2 手機鍵盤 | 點遺言輸入框 | **跳出虛擬鍵盤** |
| 4 主流程 | 寫遺言 → 分析人格 | **雷達圖跑出 9D**，無紅字 |
| 4 PvP | 進 PvP | dungeons 清單出；**raid 前先點「你的地牢部署」**（否則 422） |
| 3 iOS | iOS Safari | ⚠️ 可能偶發重載（記憶體）；建議用 **Android Chrome / 桌機** |

> ⚠️ 看起來像舊版 = 瀏覽器快取 → **硬重新整理**（朋友也要；或用無痕視窗）。

## 4. 換 tunnel 網址（重啟後）
quick tunnel 每次重啟 **URL 會變**（隨機子網域）。**只需重發新的 `/play/` 連結**，不必重匯出、不必改任何設定（web 走同源 `window.location.origin`，tunnel 換網址自動跟著對）。
（只有「在 Godot 編輯器內」開發要連 tunnel 時，才改 `config.cfg` 那一行。）

## 5. 關機（睡前）— 在 WSL
```bash
kill $(cat logs/playtest_server.pid)        # 停後端
pkill -f 'cloudflared tunnel --url'          # 關通道（你互動式 shell 打是安全的）
```
- **確認都關了**：`lsof -tiTCP:8001 -sTCP:LISTEN`（無輸出）、`pgrep -af cloudflared`（無輸出）。
- 測完務必關通道——通道開著等於本機後端一直曝在公網（見 §6）。

## 6. 🔒 網路外曝風險 & 安全須知
cloudflared quick tunnel 會把本機 8001 暴露到**公開網際網路**。開測前要知道：
- **完全公開、零帳號**：拿到網址的任何人都進得來，沒有登入/驗證。隨機子網域**只是「靠難猜」**——一旦被截圖、轉傳、貼到群組就等於公開。
- **所有 API 端點全部可達**（見 §10 的端點地圖），可能被亂打、灌資料、或被自動掃描器命中。
- **後端跑在你自己的機器上**：`host=0.0.0.0` 綁所有網卡，所以除了 tunnel，**同網段/Windows 主機也能直接連 `http://<本機IP>:8001`**。api.server 若有 bug/漏洞，攻擊面會落到你本機。
- **資料隔離但仍會被寫**：playtest 只寫 `reports/playtest_2026/`（不碰 production 211），但朋友送的遺言文字等會存在你機器上。
- **Cloudflare 是中間人**：所有流量經 Cloudflare 代理，內容可被其記錄/快取（quick tunnel 本就是給臨時測試、非正式生產，可能不穩或被限流，且 gzip 會打架 → 見 §9）。
- **降風險做法**：
  - 只在測試時段開、**測完立刻 `pkill cloudflared`**。
  - 連結只私下給信任的朋友、別公開貼。
  - 同台機器測試時段內，別在 8001 同時跑含 production 的服務。
  - 想要更正規（固定網址 + 帳號保護）→ 見 §11 named tunnel。

## 7. 測試資料：位置 / 隔離 / 清空重來
- **隔離機制**：`scripts/run_playtest_server.sh` 用 env 覆蓋四個 OUT_DIR（`api/server.py` 讀這些 env，預設值指向 production）：
  | env | playtest 寫到 | production 預設（手打才會誤碰） |
  |---|---|---|
  | `P7H_OUT_DIR` | `reports/playtest_2026/p7h` | `reports/experiments/p7h_real_study` |
  | `ECOLOGY_OUT_DIR` | `reports/playtest_2026/ecology` | `reports/ecology` |
  | `PVP_OUT_DIR` | `reports/playtest_2026/pvp` | `reports/pvp` |
  | `WALLET_OUT_DIR` | `reports/playtest_2026/wallet` | `reports/wallet` |
- **現有檔**：`reports/playtest_2026/{p7h,ecology,pvp,wallet}/*.json`（p7h 含 survey / player_test / ab_test）。
- **要從乾淨狀態重開一輪**（清空 playtest 資料，**不影響 production**）：
  ```bash
  pkill -f 'api.server'                      # 先停後端，避免邊清邊寫
  rm -rf reports/playtest_2026               # 砍掉所有 playtest 資料
  bash scripts/run_playtest_server.sh        # 重啟（startup 會從空夾載入 = fresh 生態）
  ```

## 8. 狀態檢查 & 指令速查（cheat sheet）
```bash
# 看狀態
lsof -tiTCP:8001 -sTCP:LISTEN                                   # 後端 PID
pgrep -af 'cloudflared tunnel --url'                            # 通道
grep -oE "https://[a-z0-9-]+\.trycloudflare\.com" logs/cloudflared.log | tail -1   # 目前網址
tail -f logs/playtest_server.log                               # 後端即時 log
tail -f logs/cloudflared.log                                   # 通道即時 log

# 重啟後端（改了 api/ 或 simulation/ 後）
kill $(cat logs/playtest_server.pid); bash scripts/run_playtest_server.sh

# cloudflared 安裝/版本（已裝於 ~/.local/bin，目前 v2026.6.1）
~/.local/bin/cloudflared --version
```

## 9. Troubleshooting（已踩過的坑）
| 症狀 | 根因 | 修法 |
|---|---|---|
| 整片豆腐 / 亂碼 | web 無系統字型 fallback | 嵌中文字型（已嵌 `assets/fonts/kaiu.ttf` + `project.godot` `gui/theme/custom_font`） |
| API 回 200 卻 `result!=SUCCESS` / 紅字「HTTP 200」 | Cloudflare gzip × Godot web HTTPRequest | 所有 `HTTPRequest` 設 `accept_gzip=false`（已套 ~21 處，遍佈 `src/core/*Client.gd`）|
| 手機鍵盤不彈 | export preset 關著 | `html/experimental_virtual_keyboard=true` |
| iOS 偶發重新載入 | Safari 記憶體壓力（wasm 大）| 已用最小字型 kaiu + GL Compatibility + 單執行緒；難根治 → 建議非 iOS |
| `/pvp/raid HTTP 422 "deploy a dungeon first"` | 正常規則 | 先在 PvP 點「你的地牢部署」任一派系。**非 bug** |
| 瀏覽器看 JSON 中文亂碼 | 缺 charset | 後端已 `application/json; charset=utf-8`（UTF8JSONResponse）|
| 改 `api/`/`simulation/` 沒生效 | 後端無 `--reload` | `kill` + `bash scripts/run_playtest_server.sh` 重啟 |
| 啟動 script 說「port 8001 已被佔用」 | 已有 server 在跑（重啟會 bind 失敗 Errno 98 並清掉 log）| 照提示 `kill <PID>` 再重跑；或本來就在跑就別重啟 |
| `/play/` 回 404 / 沒掛載 | `web_build/` 不存在（後端啟動時才檢查）| 先做 §2-B 的 copy 把 `web_build/` 補上，再重啟後端 |
| `cloudflared: command not found` | 不在 PATH | 用全路徑 `~/.local/bin/cloudflared`（已裝） |

## 10. 端點地圖（後端 serve 什麼，公網都打得到）
> 以下全部掛在同一個 tunnel 網域下（同源）。風險評估見 §6。
- **遊戲靜態**：`/play/`（StaticFiles，`html=True`）
- **遊戲流程**：`/sessions/*`、`/rl_sessions/*`（含 `step` / `snapshot` / `reset` / `apply-event` / `personality` / `info`）
- **人格分析**：`/personality/infer`、`/personality/infer_sbert`
- **事件**：`/event/choose`
- **玩家測試 / 問卷**：`/player-test/*`、`/survey/*`
- **生態**：`/ecology/submit`、`/ecology/snapshot`、`/ecology/assess`、`/ecology/scarcity_variation`
- **PvP**：`/pvp/dungeons`、`/pvp/deploy`、`/pvp/defense/upgrade`、`/pvp/raid`、`/pvp/challenge`
- **錢包**：`/wallet`、`/wallet/credit`
- **分流 / A-B**：`/bifurcation/*`、`/bifurcation/ab-test/*`
- **指標**：`/metrics/events`

## 11. 升級路線（延後）
- **固定網址 + 保護**：Cloudflare named tunnel（免費帳號 + 一個網域）→ 不再換 URL，可加 Access 驗證。
- requirements pin / 搬 VM：目前自機 venv 即可，將來搬機再做。
