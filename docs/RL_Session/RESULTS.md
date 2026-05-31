# RL Session Results & Signoff

## What was verified

- Headless Godot smoke test runs successfully and posts metrics to the WSL API.
- Controlled phase 2 batch runner completes successfully.
- Alert checks pass on the current metrics artifacts.

## Verified commands

### Godot smoke test

```powershell
& "C:\Program Files (x86)\Godot_v4.6.2-stable_win64\Godot_v4.6.2-stable_win64.exe" --headless --path "C:\Users\n1166\personality-dungeon" "res://src/ui/GodotMetricsSmoke.tscn"
```

Observed output:

```text
[godot_metrics_smoke] sending 5 events to http://172.31.143.82:8000
[godot_metrics_smoke] result=0 http=200 body={"ok":true,"received":5}
```

### Controlled batch

```powershell
python .\scripts\run_phase2_batch.py --runs 3
```

The most recent validated batch run was `--runs 1` and returned:

- `runs=1`
- `passed=1`
- `failed=0`

## Analysis snapshot

The regenerated metrics summary currently contains 7 session buckets, including:

- `godot_smoke` with 10 events
- `b94a0a76-b39a-45ae-b44f-ea2873c728be` with 145 events
- `60e3c5e0-785b-4ac6-9d4a-b192d84b9a91` with 103 events and `avg_latency_ms=0.02`

Current alert check status:

```text
ALERT CHECK PASSED
events=266 rows=7
```

## Signoff

The RL Session metrics pipeline is now end-to-end verifiable across:

- Godot headless smoke generation
- WSL API metrics ingestion
- JSONL persistence
- CSV summary generation
- Alert rule validation

Next work item after this signoff: controlled experiment expansion or additional alert coverage.
