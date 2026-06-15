#!/usr/bin/env python3
"""Relabel iteration-study pilot participant_ids → gate-compliant P-codes (run-level).

2026-06-15 v2:第一批 dev01-13(真人盲測)在前一次 relabel 後被 live 後端覆寫回 devNN
(見記憶 backend-restart-required),且期間新招了 P04/P05/P14-P19。本次在**後端停止**下
重做,把 dev 群映到不撞號的 P20+ 空段,並做 run-level 去污染:

  簡單映射(整個 pid → 一個 P 碼):
    dev01→P20 dev02→P21 dev03→P22 dev05→P24 dev07→P25 dev08→P26
    dev09→P27 dev10→P28 dev11→P29 dev12→P30 dev13→P31
  run-level(同一 pid 兩個 run 需分開處理):
    dev04:首輪(最早 started_at,iterated,確認盲測)→P23;次輪(reset)→隔離。
    P19 :首輪→保留 P19;次輪(同臂 reset 重複)→隔離。

只改 p7h_player_test_sessions.json 的 participant_id。survey(keyed by session_id)、
ab_test_sessions(legacy,無 pid)不需動。預設 dry-run;--apply 才寫(先備份)。

⚠️ 執行前後端必須是停止狀態,否則 in-memory tracker 會再次覆寫(見記憶)。
"""
from __future__ import annotations

import argparse
import json
import shutil
import time
from collections import Counter, defaultdict
from pathlib import Path

SESSIONS = Path("reports/experiments/p7h_real_study/p7h_player_test_sessions.json")
BAK_DIR = Path("reports/experiments/p7h_real_study/_relabel_bak")

SIMPLE_MAP = {
    "dev01": "P20", "dev02": "P21", "dev03": "P22", "dev05": "P24",
    "dev07": "P25", "dev08": "P26", "dev09": "P27", "dev10": "P28",
    "dev11": "P29", "dev12": "P30", "dev13": "P31",
}
# run-level:pid → (first_run 新碼, other_runs 新碼)
RUNLEVEL = {
    "dev04": ("P23", "EXCL_dev04_run2_reset"),   # 救首輪 iterated,隔離次輪 reset
    "P19":   ("P19", "EXCL_P19_dup_reset"),       # 留首輪,隔離同臂重複次輪
}


def first_run_of(sessions, pid):
    """回傳該 pid 最早 started_at 的 run_id。"""
    runs = defaultdict(list)
    for v in sessions.values():
        if isinstance(v, dict) and v.get("participant_id") == pid:
            runs[v.get("run_id", "")].append(v.get("started_at") or 0)
    return min(runs, key=lambda r: min(runs[r])) if runs else None


def decide(v, first_runs):
    """回傳該 session 的新 participant_id;None=不動。"""
    pid = v.get("participant_id", "")
    if pid in SIMPLE_MAP:
        return SIMPLE_MAP[pid]
    if pid in RUNLEVEL:
        keep_code, other_code = RUNLEVEL[pid]
        is_first = v.get("run_id", "") == first_runs[pid]
        new = keep_code if is_first else other_code
        return None if new == pid else new        # P19 首輪維持原碼 → 不動
    return None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args()

    data = json.load(open(SESSIONS))
    sessions = data.get("sessions", {})

    # collision guard:目標新碼不得已存在於現有資料(P19 例外:本來就是它)
    existing = {v.get("participant_id") for v in sessions.values() if isinstance(v, dict)}
    targets = set(SIMPLE_MAP.values()) | {c for pair in RUNLEVEL.values() for c in pair}
    clash = (targets & existing) - {"P19"}
    if clash:
        print(f"⚠️ 撞號,中止:目標碼已存在於資料 {sorted(clash)}")
        return

    first_runs = {pid: first_run_of(sessions, pid) for pid in RUNLEVEL}

    changes = Counter()
    plan = defaultdict(lambda: defaultdict(int))   # old_pid → new_pid → #sessions
    for v in sessions.values():
        if not isinstance(v, dict):
            continue
        new = decide(v, first_runs)
        if new is not None:
            plan[v.get("participant_id")][new] += 1
            changes[new] += 1

    print(f"檔案:{SESSIONS}")
    print(f"first-run(dev04)={first_runs.get('dev04')}  first-run(P19)={first_runs.get('P19')}\n")
    print(f"{'OLD':>8} → {'NEW':<24} (#sessions)")
    for old in sorted(plan):
        for new, n in plan[old].items():
            tag = "  ← 隔離" if new.startswith("EXCL") else ""
            print(f"{old:>8} → {new:<24} ({n}){tag}")

    if not args.apply:
        print("\n[DRY-RUN] 未寫入。加 --apply 實際執行(會先備份)。"
              "\n預期 apply 後標準 analyzer:NAIVE pilot runs=20 (iterated=10 reset=10)。")
        return

    BAK_DIR.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    bak = BAK_DIR / f"p7h_player_test_sessions.{ts}.json"
    shutil.copy2(SESSIONS, bak)
    print(f"\n備份 → {bak}")

    n = 0
    for v in sessions.values():
        if isinstance(v, dict):
            new = decide(v, first_runs)
            if new is not None:
                v["participant_id"] = new
                n += 1
    json.dump(data, open(SESSIONS, "w"), ensure_ascii=False, indent=2)
    print(f"已寫入:{n} sessions participant_id 更新。")
    print("驗證:./venv/bin/python scripts/experiments/analyze_iteration_study.py")


if __name__ == "__main__":
    main()
