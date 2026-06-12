#!/usr/bin/env python3
"""人格生態評分 — 遺言 replay / 金幣欄位驗證 / 評判標準優化。

由對話中那段 ~20 行的「46 筆真人遺言 replay」擴增而來。三件事：
  1. 把真人遺言（p7h 的 will_sbert_vector）+ 合成遺言 跑過生態評分。
  2. 驗證 `coins` 欄位正確寫入落檔（save→load round-trip + coins==score_to_coins(score)）。
  3. 報告 score/coins/archetype 分佈，並用分位數提出「優化後的金幣區間」供拍板。

用法：
  ./venv/bin/python scripts/experiments/ecology_will_replay.py
  ./venv/bin/python scripts/experiments/ecology_will_replay.py --n-synth 120 --seed 7
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from api.ecology_tracker import (  # noqa: E402
    ARCHETYPES,
    COIN_BRACKETS,
    EcologyTracker,
    personality_to_archetype_soft,
    score_to_coins,
)

_P7H = "reports/experiments/p7h_real_study/p7h_player_test_sessions.json"

# 合成遺言的三個原型錨點（9D，FEATURE_NAMES 序）：刻意偏向各原型，加噪音覆蓋空間。
_ANCHORS = {
    "aggressive": [0.9, 0.8, 0.1, -0.6, 0.0, 0.0, 0.1, -0.3, 0.2],
    "defensive": [-0.2, -0.1, 0.0, 0.8, 0.6, 0.7, -0.1, 0.7, 0.0],
    "balanced": [0.0, 0.1, 0.7, 0.0, 0.0, 0.1, 0.0, 0.0, 0.7],
}


def load_real_wills(path: str) -> list[list[float]]:
    try:
        d = json.load(open(path))
    except FileNotFoundError:
        return []
    return [s["will_sbert_vector"] for s in d.get("sessions", {}).values()
            if s.get("is_human") and len(s.get("will_sbert_vector", [])) == 9]


def make_synthetic(n: int, rng: np.random.Generator) -> list[list[float]]:
    out = []
    for _ in range(n):
        base = np.array(_ANCHORS[rng.choice(ARCHETYPES)])
        out.append((base + rng.normal(0, 0.25, 9)).clip(-1, 1).tolist())
    return out


def replay(wills: list[list[float]]) -> EcologyTracker:
    t = EcologyTracker()
    for w in wills:
        t.submit(personality_9d=w)
    return t


def verify_coins_written(t: EcologyTracker) -> tuple[bool, str]:
    """save→load round-trip，並確認每筆 coins 都存在且 == score_to_coins(score)。"""
    with tempfile.TemporaryDirectory() as td:
        t.save(td)
        raw = json.load(open(Path(td) / "ecology_state.json"))
        subs = raw["submissions"]
        # (a) JSON 裡每筆都有 coins 欄位
        missing = [i for i, s in enumerate(subs) if "coins" not in s]
        if missing:
            return False, f"{len(missing)} 筆落檔缺 coins 欄位"
        # (b) coins 與 score 區間一致
        bad = [(s["score"], s["coins"]) for s in subs
               if s["coins"] != score_to_coins(s["score"])]
        if bad:
            return False, f"{len(bad)} 筆 coins 與 score_to_coins 不一致，例：{bad[:3]}"
        # (c) reload 後 dataclass 仍帶 coins
        t2 = EcologyTracker()
        t2.load(td)
        if any(getattr(s, "coins", None) is None for s in t2._submissions):
            return False, "reload 後 coins 遺失"
        if len(t2._submissions) != len(subs):
            return False, "reload 後筆數不符"
    return True, f"{len(subs)}/{len(subs)} 筆 coins 正確寫入並可 round-trip"


def histogram(coins: list[int]) -> dict[int, int]:
    h = {c: 0 for c in [10, 25, 50, 100, 200]}
    for c in coins:
        h[c] = h.get(c, 0) + 1
    return h


def quantile_brackets(scores: np.ndarray, tiers: list[int]) -> list[tuple[float, int]]:
    """純分位數均分：每檔人數均衡（但最高檔語意被稀釋）。"""
    n = len(tiers)
    qs = [np.percentile(scores, 100 * k / n) for k in range(n)]
    return list(reversed(list(zip([round(float(q), 1) for q in qs], tiers))))


def semantic_brackets(scores: np.ndarray, tiers: list[int],
                      top_threshold: float = 100.0) -> list[tuple[float, int]]:
    """保留最高檔語意（門檻固定 = 「搭上熱潮」分數，稀有），
    其餘檔對 < top_threshold 的分數做分位數均衡。"""
    top = tiers[-1]
    lower = tiers[:-1]
    sub = scores[scores < top_threshold]
    if len(sub) == 0:
        sub = scores
    n = len(lower)
    qs = [np.percentile(sub, 100 * k / n) for k in range(n)]
    lo_brackets = list(zip([round(float(q), 1) for q in qs], lower))
    brackets = [(top_threshold, top)] + list(reversed(lo_brackets))
    return brackets


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-synth", type=int, default=90, help="合成遺言數（增加分數覆蓋）")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--p7h", default=_P7H)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    real = load_real_wills(args.p7h)
    synth = make_synthetic(args.n_synth, rng)
    wills = real + synth
    print(f"遺言：真人 {len(real)} + 合成 {len(synth)} = {len(wills)}")

    t = replay(wills)
    subs = t._submissions
    scores = np.array([s.score for s in subs])
    coins = [s.coins for s in subs]
    from collections import Counter
    arch = Counter(s.archetype for s in subs)

    # ── 1) 金幣欄位寫入驗證 ──────────────────────────────────────────────────
    ok, msg = verify_coins_written(t)
    print(f"\n[1] coins 欄位寫入驗證：{'✅ ' if ok else '❌ '}{msg}")

    # ── 2) 分佈 ────────────────────────────────────────────────────────────
    print("\n[2] 分佈")
    print(f"   原型：{dict(arch)}")
    print(f"   score：min={scores.min():.1f} p25={np.percentile(scores,25):.1f} "
          f"p50={np.percentile(scores,50):.1f} p75={np.percentile(scores,75):.1f} "
          f"p90={np.percentile(scores,90):.1f} max={scores.max():.1f}")
    cur_hist = histogram(coins)
    print(f"   現行金幣區間落點：{cur_hist}")
    print(f"   現行區間 COIN_BRACKETS={COIN_BRACKETS}")

    # ── 3) 評判標準優化（兩種方案）──────────────────────────────────────────
    tiers = [10, 25, 50, 100, 200]
    opt_q = quantile_brackets(scores, tiers)
    opt_s = semantic_brackets(scores, tiers, top_threshold=100.0)
    print("\n[3] 評判標準優化（兩方案）")
    print(f"   現行：{[(b[0],b[1]) for b in COIN_BRACKETS]}  落點 {histogram(coins)}")
    print(f"   方案 Q（純分位數均分，各檔均衡但稀釋『搭熱潮』語意）：")
    print(f"       {opt_q}  落點 {histogram([_apply(opt_q,s) for s in scores])}")
    print(f"   方案 S（★建議：保留 200=≥100『搭上熱潮』稀有，其餘均衡）：")
    print(f"       {opt_s}  落點 {histogram([_apply(opt_s,s) for s in scores])}")
    print("   註：200 檔在 replay 為 0（混合遺言生態不集中，無人衝破 100）；上線後生態")
    print("       一旦傾斜，搭上稀有剋星者分數會 >100 觸發，正是要獎勵的旋轉行為。")
    return 0 if ok else 1


def _apply(brackets: list[tuple[float, int]], score: float) -> int:
    for lo, coins in brackets:
        if score >= lo:
            return coins
    return brackets[-1][1]


if __name__ == "__main__":
    raise SystemExit(main())
