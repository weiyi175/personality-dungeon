"""Simple metrics summarizer: reads logs/metrics_*.jsonl and writes CSV summary.

Usage:
  python analysis/metrics_summary.py --out summary.csv
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


def load_events(log_dir: Path) -> list[dict[str, Any]]:
    events = []
    for p in sorted(log_dir.glob("metrics_*.jsonl")):
        with p.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    events.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return events


def summarize(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_session = defaultdict(lambda: {
        "session_id": None,
        "n_events": 0,
        "latencies": [],
        "error_count": 0,
        "final_phase": None,
        "last_ts": 0,
    })

    for ev in events:
        sid = ev.get("session_id", "<no-session>")
        s = by_session[sid]
        s["session_id"] = sid
        s["n_events"] += 1
        if "latency_ms" in ev:
            try:
                s["latencies"].append(float(ev.get("latency_ms", 0)))
            except Exception:
                pass
        # count explicit error events
        if ev.get("event_type") == "error" or ev.get("error"):
            s["error_count"] += 1
        # track final phase by timestamp
        ts = int(ev.get("timestamp", 0))
        if ts >= s["last_ts"]:
            s["last_ts"] = ts
            phase = ev.get("phase") or ev.get("rl_phase") or ev.get("event_type")
            s["final_phase"] = phase

    rows = []
    for sid, s in by_session.items():
        lat_avg = None
        if s["latencies"]:
            lat_avg = sum(s["latencies"]) / len(s["latencies"]) if s["latencies"] else None
        rows.append({
            "session_id": sid,
            "n_events": s["n_events"],
            "avg_latency_ms": round(lat_avg, 2) if lat_avg is not None else "",
            "error_count": s["error_count"],
            "final_phase": s["final_phase"],
        })

    return rows


def write_csv(rows: list[dict[str, Any]], out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["session_id", "n_events", "avg_latency_ms", "error_count", "final_phase"])
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--logs", default="logs", help="logs directory")
    parser.add_argument("--out", default="analysis/metrics_summary.csv", help="output CSV path")
    args = parser.parse_args()

    log_dir = Path(args.logs)
    events = load_events(log_dir)
    if not events:
        print("No events found in", log_dir)
        return 1
    rows = summarize(events)
    write_csv(rows, Path(args.out))
    print(f"Wrote summary to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
