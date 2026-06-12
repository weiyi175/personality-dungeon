"""Tests for the personality ecology tracker (meta-layer, Path B rotation)."""

from __future__ import annotations

import math

from api.ecology_tracker import (
    ARCHETYPES,
    EcologyParams,
    EcologyTracker,
    personality_to_archetype_soft,
    score_to_coins,
)

# 9D 特徵順序：impulsiveness, assertiveness, optimism, risk_aversion,
#               suspicion, endurance, randomness, stability_seeking, curiosity
_AGG = [0.9, 0.8, 0.1, -0.6, 0.0, 0.0, 0.0, -0.3, 0.0]   # 衝動+強勢、低趨避 → aggressive
_DEF = [-0.2, -0.1, 0.0, 0.8, 0.6, 0.7, 0.0, 0.7, 0.0]    # 趨避+耐力+求穩 → defensive
_BAL = [0.0, 0.0, 0.7, 0.0, 0.0, 0.0, 0.0, 0.0, 0.7]      # 樂觀+好奇、低極端 → balanced


def test_projection_directions():
    assert ARCHETYPES == ["aggressive", "defensive", "balanced"]
    assert _argmax(personality_to_archetype_soft(_AGG)) == 0
    assert _argmax(personality_to_archetype_soft(_DEF)) == 1
    assert _argmax(personality_to_archetype_soft(_BAL)) == 2


def test_soft_proportions_sum_to_one():
    soft = personality_to_archetype_soft(_AGG)
    assert len(soft) == 3
    assert math.isclose(sum(soft), 1.0, abs_tol=1e-9)
    assert all(p >= 0 for p in soft)


def test_cyclic_dominance_is_nontransitive():
    """路 B 核心：當某原型壟斷時，favored 的是它的『剋星』，且三者循環。

    agg 壟斷 → balanced 吃香；bal 壟斷 → defensive 吃香；def 壟斷 → aggressive 吃香。
    這就是 RPS 旋轉力（非遞移），與靜態稀有度（路 A）本質不同。
    """
    t = EcologyTracker()
    fit_when_agg = t._fitness([1.0, 0.0, 0.0])
    fit_when_def = t._fitness([0.0, 1.0, 0.0])
    fit_when_bal = t._fitness([0.0, 0.0, 1.0])
    assert _argmax(fit_when_agg) == 2  # aggressive 壟斷 → balanced 最高
    assert _argmax(fit_when_bal) == 1  # balanced 壟斷 → defensive 最高
    assert _argmax(fit_when_def) == 0  # defensive 壟斷 → aggressive 最高


def test_score_to_coins_brackets():
    # v2 區間（方案 S）：200=≥100；100=≥82；50=≥76；25=≥64；else 10。
    assert score_to_coins(150) == 200
    assert score_to_coins(100) == 200
    assert score_to_coins(90) == 100
    assert score_to_coins(80) == 50
    assert score_to_coins(70) == 25
    assert score_to_coins(50) == 10
    assert score_to_coins(0) == 10


def test_score_to_coins_monotone_nondecreasing():
    prev = 0
    for sc in range(0, 220, 2):
        c = score_to_coins(sc)
        assert c >= prev, "coins 不應隨 score 上升而下降"
        prev = c


def test_submit_returns_coins():
    t = EcologyTracker()
    out = t.submit(personality_9d=_AGG)
    assert "coins" in out and out["coins"] in (10, 25, 50, 100, 200)


def test_submit_returns_score_and_updates_ecology():
    t = EcologyTracker()
    out = t.submit(personality_9d=_AGG, run_id="r1", session_id="s1",
                   outcome={"max_proximity": 0.9})
    assert out["archetype"] == "aggressive"
    assert out["score"] > 0
    assert out["n_submissions"] == 1
    assert math.isclose(sum(out["db_proportions"]), 1.0, abs_tol=1e-9)
    assert set(out["advantage"]) == set(ARCHETYPES)


def test_rare_counter_scores_higher_than_common_type():
    """壟斷型再上傳一筆，分數應低於『剋制壟斷型的稀有原型』上傳的分數。"""
    t = EcologyTracker(EcologyParams(window=50, eta=1.0))  # eta=1 讓權重立刻反映
    for _ in range(20):                # 生態被 aggressive 壟斷
        t.submit(personality_9d=_AGG)
    score_more_agg = t.submit(personality_9d=_AGG)["score"]
    score_counter = t.submit(personality_9d=_BAL)["score"]  # balanced 剋 aggressive
    assert score_counter > score_more_agg


def test_assess_detects_rotation_in_synthetic_series():
    """注入合成旋轉的佔比快照 → classify_cycle_level 應判 level ≥ 1。"""
    t = EcologyTracker()
    # 直接灌入旋轉中的快照（繞過 submit，模擬一個已旋轉的生態）。
    from api.ecology_tracker import EcologySnapshot
    n = 60
    for k in range(n):
        phase = 2 * math.pi * k / 12.0
        a = (1 + math.cos(phase)) / 3.0
        b = (1 + math.cos(phase - 2 * math.pi / 3)) / 3.0
        c = (1 + math.cos(phase - 4 * math.pi / 3)) / 3.0
        s = a + b + c
        t._snapshots.append(EcologySnapshot(
            bin_index=k, ts=float(k),
            counts={}, proportions=[a / s, b / s, c / s], weights=[1, 1, 1],
        ))
    res = t.assess()
    assert res["level"] >= 1, res


def test_assess_insufficient_bins():
    t = EcologyTracker()
    res = t.assess()
    assert res["level"] == 0
    assert res["n_bins"] == 0


def test_save_load_roundtrip(tmp_path):
    t = EcologyTracker(EcologyParams(window=30, bin_every=5))
    for v in [_AGG, _DEF, _BAL] * 4:
        t.submit(personality_9d=v)
    t.save(tmp_path)

    t2 = EcologyTracker()
    assert t2.load(tmp_path) is True
    assert len(t2._submissions) == len(t._submissions)
    assert len(t2._snapshots) == len(t._snapshots)
    assert t2._weights == t._weights
    assert t2.params.window == 30
    assert t2.params.bin_every == 5


def _argmax(xs):
    return max(range(len(xs)), key=lambda i: xs[i])
