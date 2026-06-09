#!/usr/bin/env python
"""P7-H: 遺言模擬器 — 模擬多位玩家各自留下遺言，跑完整遊戲流程，存入 player-test JSON。

複製 DungeonLifecycleController 的邏輯（sub-critical scaling、recklessness、cadence）
並直接打後端 HTTP API，不需要 Godot。

Usage:
    python scripts/experiments/run_p7h_will_sim.py [--n-sessions N] [--base URL]
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BASE = "http://127.0.0.1:8000"

# ── 複製 DungeonLifecycleController 的常數 ────────────────────────────────────
BASELINE = [0.772, 0.724, 0.611, -0.923, -0.432, 0.289, 0.697, -0.471, 0.811]
EPSILON_C_APP = 0.11
WILL_HEADROOM = 0.5
INTENSITY_MIN, INTENSITY_MAX = 0.6, 1.8
CADENCE_CALM, CADENCE_RECKLESS = 14, 6
N_PLAYERS = 4
N_ROUNDS = 200
BURN_IN = 50
COLLAPSE_PROXIMITY = 0.8
FAILURE_THRESHOLD = 3  # CollapseScreen 預設

TRAIT_KEYS = [
    "impulsiveness", "assertiveness", "optimism", "risk_aversion",
    "suspicion", "endurance", "randomness", "stability_seeking", "curiosity",
]

# 多樣化的遺言文字（涵蓋不同人格方向）
WILL_TEXTS = [
    "我想留下美好的回憶給所有人",
    "衝吧讓生命盡情燃燒",
    "謹慎走好每一步不要犯錯",
    "探索每個未知的角落",
    "保持平靜接受一切結果",
    "我會永遠保護我愛的人",
    "隨心所欲才是人生的意義",
    "相信未來一切都會更好",
    "懷疑一切只相信自己",
    "耐心等待才能得到最好的",
    "勇敢面對所有的挑戰",
    "愛是唯一值得追求的事",
    "不要回頭只管往前走",
    "深思熟慮再做每個決定",
    "笑著離開這個世界",
    "我的遺憾是沒有更瘋狂",
    "願世界充滿好奇與探索",
    "穩定才是最大的幸福",
    "讓隨機決定我的命運",
    "帶著傷疤繼續前行",
]


# ── HTTP helpers ──────────────────────────────────────────────────────────────

def _post(path: str, payload: dict) -> dict:
    data = json.dumps(payload).encode()
    req = urllib.request.Request(
        BASE + path, data=data,
        headers={"Content-Type": "application/json"}, method="POST"
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return json.loads(r.read().decode())
    except urllib.error.HTTPError as e:
        body = e.read().decode()
        raise RuntimeError(f"POST {path} → HTTP {e.code}: {body[:200]}") from e


def _get(path: str) -> dict:
    with urllib.request.urlopen(BASE + path, timeout=30) as r:
        return json.loads(r.read().decode())


# ── 複製 DLC 邏輯 ─────────────────────────────────────────────────────────────

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


# ── 單局遊戲流程 ──────────────────────────────────────────────────────────────

def run_session(will_text: str, session_idx: int, run_id: str) -> dict:
    print(f"\n[{session_idx}] 遺言：「{will_text}」")

    # 1. SBERT 推論
    infer = _post("/personality/infer_sbert", {"text": will_text})
    raw_vector = [infer["vector"][k] for k in TRAIT_KEYS]

    # 2. Sub-critical scaling（與 DLC 完全相同）
    scaled = apply_sub_critical(raw_vector)
    recklessness = compute_recklessness(raw_vector)
    intensity = lerp(INTENSITY_MIN, INTENSITY_MAX, recklessness)
    cadence = int(round(lerp(float(CADENCE_CALM), float(CADENCE_RECKLESS), recklessness)))
    cadence = max(CADENCE_RECKLESS, min(CADENCE_CALM, cadence))
    print(f"    魯莽度={recklessness:.2f}  強度={intensity:.2f}  節奏={cadence}回合")

    # 3. AB-test 分組
    pt_sid = f"wsim_{run_id}_{session_idx:04d}"
    assign = _post("/bifurcation/ab-test/assign", {"session_id": pt_sid})
    group = assign.get("group", "experiment")

    # 4. Player-test 開始記錄
    _post("/player-test/start", {
        "session_id": pt_sid,
        "group": group,
        "player_alias": f"wsim_{run_id}_{session_idx:04d}",
    })

    # 5. RL session 初始化（使用縮放後的遺言人格種子）
    init = _post("/rl_sessions/initialize", {
        "n_players": N_PLAYERS,
        "n_rounds": N_ROUNDS,
        "burn_in": BURN_IN,
        "seed": session_idx * 7 + 42,
        "fixed_personality_vector": scaled,
        "space_a_events_enabled": True,
    })
    rl_sid = init["session_id"]
    init_snap = init.get("initial_snapshot", {})
    print(f"    RL session={rl_sid[:8]}  group={group}")

    # 6. 遊戲迴圈
    prev_pv = init_snap.get("mean_personality", scaled[:]) or scaled[:]
    prox_before = float(init_snap.get("bifurcation_proximity", 0.0))
    max_proximity = prox_before
    failure_count = 0
    round_num = 0

    for r in range(1, N_ROUNDS + 1):
        # step — 回傳 {"session_id": ..., "snapshot": {...}}
        step_resp = _post(f"/rl_sessions/{rl_sid}/step", {})
        snap = step_resp.get("snapshot", step_resp)
        round_num = r
        prox = float(snap.get("bifurcation_proximity", 0.0))
        max_proximity = max(max_proximity, prox)
        mean_pv = snap.get("mean_personality", prev_pv)

        is_event = (cadence > 0 and r % cadence == 0)

        # player-test step 記錄
        _post("/player-test/step", {
            "session_id": pt_sid,
            "action_text": f"rl_step_{r}",
            "personality_before": prev_pv[:9],
            "personality_after": mean_pv[:9] if len(mean_pv) == 9 else prev_pv[:9],
            "proximity_before": prox_before,
            "proximity_after": prox,
            "event_type": "personality_shift" if is_event else "none",
            "response_time_ms": 0.0,
        })

        # apply-event
        if is_event:
            ev = _post(f"/rl_sessions/{rl_sid}/apply-event", {
                "group": group,
                "intensity_scale": intensity,
            })
            prox = float(ev.get("bifurcation_proximity", prox))
            max_proximity = max(max_proximity, prox)

        if len(mean_pv) == 9:
            prev_pv = mean_pv[:]
        prox_before = prox

        # 崩壞條件（模擬 failure_threshold = FAILURE_THRESHOLD）
        if prox >= COLLAPSE_PROXIMITY:
            failure_count += 1
            if failure_count >= FAILURE_THRESHOLD:
                print(f"    崩壞於第 {r} 回合  最高接近度={max_proximity:.3f}")
                break

        phase = snap.get("phase", "")
        if phase == "ended":
            break

    # 7. Player-test 結束
    end_result = _post("/player-test/end", {"session_id": pt_sid})
    saved_steps = end_result.get("n_steps", 0)
    saved_prox = end_result.get("max_proximity", 0.0)
    print(f"    ✓ 存檔 {saved_steps} 步  max_prox={saved_prox:.3f}")

    return {
        "will_text": will_text,
        "group": group,
        "recklessness": recklessness,
        "max_proximity": max_proximity,
        "rounds_survived": round_num,
        "n_steps_saved": saved_steps,
    }


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-sessions", type=int, default=len(WILL_TEXTS),
                    help="要跑幾局（預設用完所有遺言文字）")
    ap.add_argument("--base", default="http://127.0.0.1:8000",
                    help="後端 URL")
    ap.add_argument("--delay", type=float, default=0.1,
                    help="每局之間的間隔秒數")
    args = ap.parse_args()

    global BASE
    BASE = args.base.rstrip("/")

    # 健康檢查
    try:
        _get("/openapi.json")
    except Exception as e:
        print(f"ERROR: 後端無法連線 ({e})")
        return 1

    n = min(args.n_sessions, len(WILL_TEXTS))
    run_id = datetime.utcnow().strftime("%Y%m%d%H%M%S")
    print(f"=== P7-H 遺言模擬器 — 跑 {n} 局  run_id={run_id} ===")
    print(f"後端: {BASE}")

    results = []
    for i, text in enumerate(WILL_TEXTS[:n]):
        try:
            r = run_session(text, i, run_id)
            results.append(r)
        except Exception as e:
            print(f"    ✗ 失敗：{e}")
        if args.delay > 0:
            time.sleep(args.delay)

    print(f"\n=== 完成 {len(results)}/{n} 局 ===")
    if results:
        avg_prox = sum(r["max_proximity"] for r in results) / len(results)
        avg_rounds = sum(r["rounds_survived"] for r in results) / len(results)
        ctrl = [r for r in results if r["group"] == "control"]
        exp  = [r for r in results if r["group"] == "experiment"]
        print(f"平均 max_proximity={avg_prox:.3f}  平均存活={avg_rounds:.0f}回合")
        if ctrl:
            print(f"控制組 (n={len(ctrl)})  avg_prox={sum(r['max_proximity'] for r in ctrl)/len(ctrl):.3f}")
        if exp:
            print(f"實驗組 (n={len(exp)})  avg_prox={sum(r['max_proximity'] for r in exp)/len(exp):.3f}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
