# Infer SBERT Health Check

This folder contains a minimal health check for the offline personality inference
endpoint: /personality/infer_sbert.

## Location

docs/infer_sbert_check

## Files

- run_infer_sbert_check.sh (WSL/Linux)
- run_infer_sbert_check.ps1 (Windows PowerShell)

## Prerequisites

- The API server is running on port 8000.

## Start Server (WSL)

cd /home/user/personality-dungeon
./venv/bin/python -m api.server

Keep this running in a separate terminal while you run the check.

## Usage (WSL/Linux)

bash docs/infer_sbert_check/run_infer_sbert_check.sh

## Usage (Windows PowerShell)

.\docs\infer_sbert_check\run_infer_sbert_check.ps1

## Optional Overrides

You can override the endpoint or text using environment variables. Keep text length <= 20 characters.

API_URL=http://127.0.0.1:8000/personality/infer_sbert \
TEXT="quick test" \
bash docs/infer_sbert_check/run_infer_sbert_check.sh

## Success Output

A JSON response with a "vector" field indicates success.

## Connection Map

| Target | Purpose | Endpoint | Used by PlayableLoopScene? | Notes |
|---|---|---|---|---|
| PlayableLoopScene | 可玩迴圈 UI 與回合流程 | RL Session API (/rl_sessions/*) | Yes | 對應 [src/ui/PlayableLoopScene.tscn](../../src/ui/PlayableLoopScene.tscn) 與 [src/core/RLSessionAPIClient.gd](../../src/core/RLSessionAPIClient.gd) |
| RL Session | 初始化 / step / snapshot | http://localhost:8000 或 WSL IP :8000 | Yes | 這是 PlayableLoopScene 真正要連的服務 |
| Ollama | 文字推論 / 模型檢查 | http://<host>:11434 | No | 只用於 LLM / 模型健康檢查，不提供 /rl_sessions/* |
| infer_sbert | 離線人格向量推論 | /personality/infer_sbert | No | 對應本資料夾的健康檢查，不是 PlayableLoopScene 主流程 |

### Quick Rule

- PlayableLoopScene 走 RL Session，不走 infer_sbert。
- 11434 是 Ollama，不是 RL Session API。
- 如果要驗證 PlayableLoopScene，請確認 api_base_url 指向 RL Session 服務，而不是 Ollama。

### PlayableLoopScene Memo

PlayableLoopScene 只看 RL Session：`/rl_sessions/initialize`、`/rl_sessions/{id}/step`、`/rl_sessions/{id}/snapshot`。
Ollama 只管 `11434` 上的模型推論；`infer_sbert` 只管 `/personality/infer_sbert` 的人格向量檢查。
看到 `11434` 就先不要拿來測 PlayableLoopScene，看到 `infer_sbert` 也不要把它當成回合流程。
