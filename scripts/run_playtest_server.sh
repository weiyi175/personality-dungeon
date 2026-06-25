#!/usr/bin/env bash
# run_playtest_server.sh — 啟動「封閉朋友測試」後端（Phase 1）。
#
# 重點：把四個 store 的 OUT_DIR 全部改指向 reports/playtest_2026/，
# 與 production（reports/experiments/p7h_real_study + reports/ecology 等）完全隔離。
# 測試資料不碰 production 211；playtest 生態刻意 fresh 空的（startup 從空夾載入）。
#
# 用法：bash scripts/run_playtest_server.sh
# 停止：kill $(cat logs/playtest_server.pid)   或   pkill -f 'api.server'

# 不論從哪裡呼叫，一律切到 repo 根（讓相對路徑/venv 都對）。
cd "$(dirname "$0")/.." || exit 1

# 防呆：8001 已有 server 在跑就別重啟 —— 重複啟動會 bind 失敗（Errno 98），
# 而且 `>` 會先清掉現有 server 的 log。先擋下來，避免假性「啟動成功」。
EXISTING=$(lsof -tiTCP:8001 -sTCP:LISTEN 2>/dev/null || true)
if [ -n "$EXISTING" ]; then
  echo "⚠ port 8001 已被 PID $EXISTING 佔用 —— playtest server 已在跑，不重複啟動。"
  echo "  要重啟：kill $EXISTING && bash scripts/run_playtest_server.sh"
  exit 1
fi

export P7H_OUT_DIR=reports/playtest_2026/p7h
export ECOLOGY_OUT_DIR=reports/playtest_2026/ecology
export PVP_OUT_DIR=reports/playtest_2026/pvp
export WALLET_OUT_DIR=reports/playtest_2026/wallet

mkdir -p "$P7H_OUT_DIR" "$ECOLOGY_OUT_DIR" "$PVP_OUT_DIR" "$WALLET_OUT_DIR" logs

# setsid + nohup：把 server 丟進**新 session**、忽略 SIGHUP，這樣關掉終端機
# / 啟動它的 shell 結束後，server 仍續跑（友人測試常跨小時）。少了這個，
# server 會在啟動它的 shell 被回收時一起死掉。
setsid nohup ./venv/bin/python -m api.server > logs/playtest_server.log 2>&1 < /dev/null &
SVR_PID=$!
echo "$SVR_PID" > logs/playtest_server.pid
echo "playtest server PID $SVR_PID（已 detach，關終端機不會死）"
