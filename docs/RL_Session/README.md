# RL Session API Connection Check

This folder contains a minimal connectivity check for the backend used by PlayableLoopScene.

The target is the RL Session API, not Ollama and not infer_sbert.

## What it checks

- The base URL is reachable.
- POST /rl_sessions/initialize responds successfully.
- The response contains a session_id.

## Files

- check_rl_session_api.py - sends a sample initialize request and prints the response.

## Start the server

If the API server runs inside WSL, start it from the project root with:

```bash
cd /home/user/personality-dungeon
./venv/bin/python -m api.server
```

Keep that terminal open while you run the check.

By default the script targets http://127.0.0.1:8000.

If Godot or the test script cannot reach localhost across Windows/WSL boundaries, point RL_API_BASE_URL to the reachable host IP on port 8000.

### Verified connection setup

- Godot / Windows currently uses `http://172.31.143.82:8000`
- WSL-local connectivity check keeps using `http://127.0.0.1:8000`
- If WSL restarts and the IP changes, update Godot's `api_base_url` to the new host IP

## Run the check

From WSL/Linux:

```bash
cd /home/user/personality-dungeon
python3 docs/RL_Session/check_rl_session_api.py
```

From Windows PowerShell:

```powershell
python .\docs\RL_Session\check_rl_session_api.py
```

## Optional overrides

You can override the target host and request payload with environment variables:

- RL_API_BASE_URL - API host, default http://127.0.0.1:8000
- N_PLAYERS - default 4
- N_ROUNDS - default 200
- BURN_IN - default 50
- SEED - default 42

Example:

```bash
RL_API_BASE_URL=http://127.0.0.1:8000 \
N_PLAYERS=4 \
N_ROUNDS=200 \
BURN_IN=50 \
SEED=42 \
python3 docs/RL_Session/check_rl_session_api.py
```

## Success criteria

The check is successful when the server returns HTTP 200 and the JSON body includes a session_id.

## Godot metrics smoke test

Quick start:

```powershell
& "C:\Program Files (x86)\Godot_v4.6.2-stable_win64\Godot_v4.6.2-stable_win64.exe" --headless --path "C:\Users\n1166\personality-dungeon" "res://src/ui/GodotMetricsSmoke.tscn"
```

Verified output:

```text
[godot_metrics_smoke] sending 5 events to http://172.31.143.82:8000
[godot_metrics_smoke] result=0 http=200 body={"ok":true,"received":5}
```

This scene is defined in [src/ui/GodotMetricsSmoke.tscn](../../src/ui/GodotMetricsSmoke.tscn) and sends five `godot_smoke_metric` events to the WSL API.

## Controlled batch

To run the phase 2 controlled batch runner once or multiple times:

```powershell
python .\scripts\run_phase2_batch.py --runs 3
```

The runner repeats `scripts/phase2_headless_test.gd`, writes a JSON summary to [reports/phase2_batch_summary.json](../../reports/phase2_batch_summary.json), and exits non-zero if any run fails.

## Alert checks

Run the alert checker after smoke/batch output is available:

```powershell
python .\scripts\check_rl_session_alerts.py
```

The first-pass rules are documented in [docs/RL_Session/ALERT_RULES.md](ALERT_RULES.md).

## Results & signoff

The verified outcomes are summarized in [docs/RL_Session/RESULTS.md](RESULTS.md).

Example request shape:

```json
{
  "n_players": 4,
  "n_rounds": 200,
  "burn_in": 50,
  "seed": 42
}
```

## Notes

- This check is for /rl_sessions/* only.
- Do not use http://localhost:11434; that is Ollama.
- Do not use /personality/infer_sbert; that is a different inference endpoint.