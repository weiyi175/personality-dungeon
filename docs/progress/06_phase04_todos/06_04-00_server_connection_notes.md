# 06_04-00 — Server Connection Notes (WSL + Windows)

目的：紀錄 WSL API 伺服器啟動、Windows 連線方式、以及實測成功輸出，方便後續重現。

## 1) WSL 啟動 API

在 WSL 中執行（專案 venv）：

```bash
cd /home/user/personality-dungeon
./venv/bin/python -m api.server
```

成功啟動的範例輸出：

```text
INFO:     Started server process [1982953]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
INFO:     172.31.128.1:52550 - "POST /rl_sessions/initialize HTTP/1.1" 200 OK
```

## 2) Windows 取得 WSL IP

在 PowerShell 取得 WSL IP（取第一個 IPv4）：

```powershell
wsl hostname -I
```

範例輸出：

```text
172.31.143.82 100.104.143.27 fd7a:115c:a1e0::bb39:8f1b
```

## 3) Windows 連線驗證（RL Session API）

將 RL_API_BASE_URL 指到 WSL IP，再執行測試腳本：

```powershell
$env:RL_API_BASE_URL="http://172.31.143.82:8000"
& C:\Users\n1166\.local\bin\python3.11.exe c:\Users\n1166\personality-dungeon\scripts\send_metrics_test.py
```

成功輸出範例（200 + session_id）：

```text
200
{"session_id":"0de31832-8fed-4706-8ece-86b6835756a4","initial_snapshot":{"session_id":"0de31832-8fed-4706-8ece-86b6835756a4","round":0,"tick":0,"warm":false,"cycle_level":0,"s3_score":0.0,"env_gamma":0.0,"entropy":1.0986122886681096,"q_std":0.0,"p_aggressive":0.25,"p_defensive":0.5,"p_balanced":0.25,"pi_aggressive":0.3333333333333333,"pi_defensive":0.3333333333333333,"pi_balanced":0.3333333333333333,"q_mean_aggressive":0.0,"q_mean_defensive":0.0,"q_mean_balanced":0.0,"avg_reward":0.0,"avg_utility":0.0,"success_rate":0.0,"risk_mean":0.0,"stress_mean":0.0,"world_scarcity":0.0,"world_threat":0.0,"world_noise":0.0,"world_intel":1.0,"phase":"burn-in"}}
```

## 4) 常見問題

- 404 Not Found：通常是打錯主機或 API 未啟動。請確認 WSL server 正在跑，且 RL_API_BASE_URL 指向 WSL IP。
- WSL IP 變動：WSL 重啟可能更換 IP，請重新執行 `wsl hostname -I` 更新。
- Windows/WSL 互通：Windows 直接用 127.0.0.1 可能會連到本機而不是 WSL 服務，務必改用 WSL IP。
