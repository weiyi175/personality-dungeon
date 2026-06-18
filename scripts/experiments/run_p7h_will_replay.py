#!/usr/bin/env python
"""P7-H 遺言 replay（sim 穩健性複製）——把迭代研究污染 session 的真人遺言救出來重跑。

把 `p7h_player_test_sessions.json` 內帶 `run_id`（人格迭代研究污染、被 run_id→force
experiment 灌進 P7-H experiment 臂）的真人遺言抽出，**用儲存的原始 SBERT 向量**重跑
P7-H apparatus，每個遺言**跑兩臂（paired：control + experiment，同 RL seed）**，
量 H1 主要 DV `max_proximity`。

定位：**sim 穩健性複製**（is_human=false 語意；不可併入 confirmatory 26/26）。
乾淨保證：**只打無狀態計算端點** `/rl_sessions/{initialize,step,apply-event}`，
**完全不碰 `/player-test/*`** → confirmatory 主檔零變更、無 is_human 污染。
結果客戶端自寫獨立檔 `reports/experiments/p7h_real_study/p7h_will_replay_sim.json`。

Usage:
    python scripts/experiments/run_p7h_will_replay.py [--base http://127.0.0.1:8001] [--limit N]
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SESSIONS = ROOT / "reports/experiments/p7h_real_study/p7h_player_test_sessions.json"
OUT = ROOT / "reports/experiments/p7h_real_study/p7h_will_replay_sim.json"

# ── 常數（與 DungeonLifecycleController / run_p7h_will_sim 一致）─────────────────
BASELINE = [0.772, 0.724, 0.611, -0.923, -0.432, 0.289, 0.697, -0.471, 0.811]
EPSILON_C_APP = 0.11
WILL_HEADROOM = 0.5
INTENSITY_MIN, INTENSITY_MAX = 0.6, 1.8
CADENCE_CALM, CADENCE_RECKLESS = 14, 6
N_PLAYERS = 4
N_ROUNDS = 200
BURN_IN = 50
COLLAPSE_PROXIMITY = 0.8
FAILURE_THRESHOLD = 3
TRAIT_KEYS = [
    "impulsiveness", "assertiveness", "optimism", "risk_aversion",
    "suspicion", "endurance", "randomness", "stability_seeking", "curiosity",
]

BASE = "http://127.0.0.1:8001"


# ── HTTP ─────────────────────────────────────────────────────────────────────
def _post(path: str, payload: dict) -> dict:
    data = json.dumps(payload).encode()
    req = urllib.request.Request(
        BASE + path, data=data,
        headers={"Content-Type": "application/json"}, method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            return json.loads(r.read().decode())
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"POST {path} → HTTP {e.code}: {e.read().decode()[:200]}") from e


# ── DLC 邏輯 ──────────────────────────────────────────────────────────────────
def apply_sub_critical(vector: list[float], headroom: float = WILL_HEADROOM) -> list[float]:
    n = min(len(vector), len(BASELINE))
    offset = [vector[i] - BASELINE[i] for i in range(n)]
    norm = math.sqrt(sum(d * d for d in offset))
    if norm < 1e-6:
        return list(BASELINE)
    scale = EPSILON_C_APP * headroom / norm
    return [BASELINE[i] + offset[i] * scale for i in range(n)]


def compute_recklessness(vector: list[float]) -> float:
    v = dict(zip(TRAIT_KEYS, vector))
    raw = (v["impulsiveness"] + v["randomness"] + v["curiosity"]
           - v["risk_aversion"] - v["stability_seeking"] - v["endurance"]) / 6.0
    return 1.0 / (1.0 + math.exp(-3.0 * raw))


def lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


# ── 抽取真人遺言（去重）───────────────────────────────────────────────────────
def load_real_wills() -> list[dict]:
    with open(SESSIONS) as f:
        sessions = json.load(f)["sessions"]
    seen: set[str] = set()
    wills: list[dict] = []
    for s in sessions.values():
        if not s.get("run_id", ""):          # 只取迭代研究污染 session
            continue
        text = (s.get("will_text") or "").strip()
        vec = s.get("will_sbert_vector")
        if not text or not vec or len(vec) != 9:
            continue
        if text in seen:
            continue
        seen.add(text)
        wills.append({
            "will_text": text,
            "will_sbert_vector": vec,
            "src_participant_id": s.get("participant_id"),
        })
    return wills


# ── 單臂模擬（只打無狀態 RL 端點，不碰 /player-test/*）─────────────────────────
def simulate_arm(raw_vector: list[float], group: str, seed: int) -> dict:
    scaled = apply_sub_critical(raw_vector)
    recklessness = compute_recklessness(raw_vector)
    intensity = lerp(INTENSITY_MIN, INTENSITY_MAX, recklessness)
    cadence = int(round(lerp(float(CADENCE_CALM), float(CADENCE_RECKLESS), recklessness)))
    cadence = max(CADENCE_RECKLESS, min(CADENCE_CALM, cadence))

    init = _post("/rl_sessions/initialize", {
        "n_players": N_PLAYERS,
        "n_rounds": N_ROUNDS,
        "burn_in": BURN_IN,
        "seed": seed,
        "fixed_personality_vector": scaled,
        "space_a_events_enabled": True,
    })
    rl_sid = init["session_id"]
    snap0 = init.get("initial_snapshot", {})
    prox = float(snap0.get("bifurcation_proximity", 0.0))
    max_proximity = prox
    failure_count = 0
    rounds_survived = 0
    collapsed = False

    for r in range(1, N_ROUNDS + 1):
        step = _post(f"/rl_sessions/{rl_sid}/step", {})
        snap = step.get("snapshot", step)
        rounds_survived = r
        prox = float(snap.get("bifurcation_proximity", prox))
        max_proximity = max(max_proximity, prox)

        if cadence > 0 and r % cadence == 0:
            ev = _post(f"/rl_sessions/{rl_sid}/apply-event", {
                "group": group,                       # experiment→aligned, control→random
                "intensity_scale": intensity,
            })
            prox = float(ev.get("bifurcation_proximity", prox))
            max_proximity = max(max_proximity, prox)

        if prox >= COLLAPSE_PROXIMITY:
            failure_count += 1
            if failure_count >= FAILURE_THRESHOLD:
                collapsed = True
                break
        if snap.get("phase", "") == "ended":
            break

    return {
        "group": group,
        "seed": seed,
        "recklessness": recklessness,
        "intensity": intensity,
        "cadence": cadence,
        "max_proximity": max_proximity,
        "rounds_survived": rounds_survived,
        "collapsed": collapsed,
        "collapse_reason": "proximity+passive_failure" if collapsed else "max_rounds",
        # 問卷留空（sim 無主觀問卷）
        "survey": None,
        "is_human": False,
    }


def main() -> int:
    global BASE
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=BASE)
    ap.add_argument("--limit", type=int, default=0, help="只跑前 N 個遺言（0=全部）")
    args = ap.parse_args()
    BASE = args.base

    wills = load_real_wills()
    if args.limit > 0:
        wills = wills[:args.limit]
    print(f"抽出 {len(wills)} 個不重複真人遺言 → paired replay（每個跑 control+experiment）")
    print(f"後端 {BASE}　輸出 {OUT.relative_to(ROOT)}\n")

    runs: list[dict] = []
    t0 = time.time()
    for i, w in enumerate(wills):
        seed = i * 7 + 42                              # paired：同遺言兩臂同 seed
        # provenance：遺言是真人寫的（will_author_is_human），但 session 是 sim replay
        # （is_human=False，見 simulate_arm）。兩者刻意分開，杜絕把 sim run 誤當人類 session。
        rec = {"will_text": w["will_text"],
               "will_author_is_human": True,
               "will_source": "iteration_study",
               "src_participant_id": w["src_participant_id"],
               "will_sbert_vector": w["will_sbert_vector"]}
        for group in ("control", "experiment"):
            res = simulate_arm(w["will_sbert_vector"], group, seed)
            run = {**rec, **res}
            runs.append(run)
        c = runs[-2]["max_proximity"]; e = runs[-1]["max_proximity"]
        print(f"[{i+1:3d}/{len(wills)}] R={runs[-1]['recklessness']:.2f}  "
              f"ctrl_maxprox={c:.3f}  exp_maxprox={e:.3f}  「{w['will_text'][:24]}」")

    # ── 摘要（paired H1）──────────────────────────────────────────────────────
    ctrl = [r for r in runs if r["group"] == "control"]
    exp = [r for r in runs if r["group"] == "experiment"]
    import statistics as st
    c_mp = [r["max_proximity"] for r in ctrl]
    e_mp = [r["max_proximity"] for r in exp]
    deltas = [e - c for c, e in zip(c_mp, e_mp)]      # paired diff（同遺言同 seed）
    summary = {
        "n_wills": len(wills),
        "n_runs": len(runs),
        "design": "paired (each will run through control AND experiment with same RL seed)",
        "positioning": "SIM robustness replication of P7-H H1 on real human wills; "
                       "is_human=false; NOT part of confirmatory 26/26 (pre-reg excludes sim).",
        "control_max_proximity_mean": round(st.mean(c_mp), 5),
        "experiment_max_proximity_mean": round(st.mean(e_mp), 5),
        "paired_delta_mean": round(st.mean(deltas), 5),
        "paired_delta_sd": round(st.pstdev(deltas), 5) if len(deltas) > 1 else 0.0,
        "experiment_collapse_rate": f"{sum(r['collapsed'] for r in exp)}/{len(exp)}",
        "control_collapse_rate": f"{sum(r['collapsed'] for r in ctrl)}/{len(ctrl)}",
    }
    OUT.write_text(json.dumps({
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source": str(SESSIONS.relative_to(ROOT)),
        "summary": summary,
        "runs": runs,
    }, ensure_ascii=False, indent=2))

    print("\n" + "=" * 60)
    print("P7-H 遺言 SIM REPLAY（穩健性，非 confirmatory）")
    print("=" * 60)
    for k, v in summary.items():
        print(f"  {k}: {v}")
    print(f"\n  ✓ 寫入 {OUT.relative_to(ROOT)}（{time.time()-t0:.1f}s）")
    print("  ⚠ is_human=false sim；不可併入 confirmatory 26/26。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
