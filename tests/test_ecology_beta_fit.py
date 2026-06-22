"""β-instrument 測試：估計器在合成資料上回收已知 β、且在資料不足/不可識別時誠實判錯。

驗證三件事，對應 estimator 的三條 verdict：
  1. OK            — 足夠且變動的稀缺下，β̂ 的 95%CI 覆蓋真值（自我回收）。
  2. INSUFFICIENT_N — 真人筆數過少（現狀 n=2）→ 不偽造數字。
  3. UNIDENTIFIED   — 稀缺零變異（所有人面對同生態）→ β 與截距共線。
"""
from __future__ import annotations

import numpy as np
import pytest

from scripts.experiments import ecology_beta_fit as bf


@pytest.mark.parametrize("true_beta", [0.0, 1.0, 2.0, 4.0])
def test_self_recovery_covers_true_beta(true_beta):
    """足量 + 變動稀缺 → CI 覆蓋真 β。"""
    rng = np.random.default_rng(7)
    alpha = np.array([0.3, 0.2, 0.0])
    q = bf.random_q_states(2000, rng)
    data = bf.simulate(alpha, true_beta, q, rng)
    r = bf.fit_beta(data, min_n=10)
    assert r.verdict == "OK", r.note
    assert r.beta_ci[0] <= true_beta <= r.beta_ci[1], (
        f"true {true_beta} not in CI {r.beta_ci}")


def test_point_estimate_close_at_large_n():
    rng = np.random.default_rng(3)
    alpha = np.array([0.4, 0.1, 0.0])
    q = bf.random_q_states(8000, rng)
    data = bf.simulate(alpha, 2.0, q, rng)
    r = bf.fit_beta(data, min_n=10)
    assert r.verdict == "OK"
    assert abs(r.beta - 2.0) < 0.4          # 大樣本點估計收斂到真值附近


def test_insufficient_n_verdict():
    rng = np.random.default_rng(0)
    q = bf.random_q_states(5, rng)
    data = bf.simulate(np.zeros(3), 2.0, q, rng)
    r = bf.fit_beta(data, min_n=30)
    assert r.verdict == "INSUFFICIENT_N"
    assert r.beta is None                    # 不偽造數字


def test_unidentified_when_scarcity_constant():
    """所有真人面對同一生態狀態 → adv 跨筆零變異 → β 不可識別。"""
    rng = np.random.default_rng(1)
    q_fixed = np.array([0.5, 0.3, 0.2])
    q = np.tile(q_fixed, (200, 1))           # 每筆都同一稀缺
    data = bf.simulate(np.array([0.3, 0.2, 0.0]), 2.0, q, rng)
    r = bf.fit_beta(data, min_n=30)
    assert r.verdict == "UNIDENTIFIED"


def test_advantage_reconstruction_matches_tracker():
    """advantage 重建須複刻 ecology_tracker 的 fitness=1/N−q、softplus(lam··)。"""
    q = [0.6, 0.3, 0.1]
    adv = bf.reconstruct_advantage(q, lam=2.0)
    # 手算：fitness = 1/3 − q；softplus(2·fitness)
    fit = np.array([1/3 - x for x in q])
    expect = np.where(2*fit > 0, 2*fit + np.log1p(np.exp(-2*fit)), np.log1p(np.exp(2*fit)))
    assert np.allclose(adv, expect)
    # 稀有(q=.1)的 advantage > 普及(q=.6)的 advantage
    assert adv[2] > adv[0]


def test_load_real_submissions_filter(tmp_path):
    """真人判準 = session_id+outcome 雙非空（2026-06-23 更正後）：
    V2-live（有 session+outcome）計入、smoke/replay（缺其一）剔除。"""
    import json
    state = {
        "params": {"lam": 2.0},
        "submissions": [
            # 真人 V2-live：run_id==session_id==uuid、帶冒險 outcome → 計入
            {"run_id": "u1", "session_id": "u1", "archetype": "defensive",
             "outcome": {"rounds_survived": 81},
             "score_components": {"q_before": [0.4, 0.35, 0.25]}},
            {"run_id": "u2", "session_id": "u2", "archetype": "aggressive",
             "outcome": {"rounds_survived": 200},
             "score_components": {"q_before": [0.5, 0.3, 0.2]}},
            # smoke artifact：無 session、無 outcome → 剔除
            {"run_id": "", "session_id": "", "archetype": "balanced",
             "outcome": {}, "score_components": {"q_before": [0.48, 0.42, 0.1]}},
            # 有 session 但無 outcome（半殘）→ 剔除
            {"run_id": "u3", "session_id": "u3", "archetype": "defensive",
             "outcome": {}, "score_components": {"q_before": [0.4, 0.4, 0.2]}},
        ],
    }
    p = tmp_path / "ecology_state.json"
    p.write_text(json.dumps(state))
    data = bf.load_real_submissions(p)
    assert data.n_real == 2          # 只有兩筆真人 live
    assert data.n_artifact == 2      # 兩筆非真實-live 被剔除
    assert data.n_total == 4


def test_power_dgps_detector_wiring():
    """power 模組的 DGP 接線守護：真響應 → β̂>0 顯著；null → CI 含 0。"""
    from scripts.experiments import ecology_beta_power as bp

    rng = np.random.default_rng(2)
    q = bf.random_q_states(800, rng, concentration=bp.SCARCITY_REGIMES["high"])
    # 真響應（softmax β=3）→ 顯著偵測到
    chosen = bp.dgp_softmax(q, bp.ALPHA_REAL, 3.0, np.random.default_rng(2))
    adv = np.stack([bf.reconstruct_advantage(qt) for qt in q])
    data = bf.BetaData(adv=adv, chosen=np.asarray(chosen), n_real=len(adv), n_total=len(adv))
    res = bf.fit_beta(data, min_n=10)
    assert res.verdict == "OK" and res.beta > 0 and res.beta_ci[0] > 0
    # null（不理稀缺）→ CI 含 0
    chosen0 = bp.dgp_noresponse(q, bp.ALPHA_REAL, 0.0, np.random.default_rng(2))
    data0 = bf.BetaData(adv=adv, chosen=np.asarray(chosen0), n_real=len(adv), n_total=len(adv))
    res0 = bf.fit_beta(data0, min_n=10)
    assert res0.verdict == "OK" and res0.beta_ci[0] <= 0 <= res0.beta_ci[1]


def test_beta_zero_means_ignores_scarcity():
    """β=0 模擬 → 估出的 β̂ 應 ~0（CI 含 0）：不理稀缺、純 intrinsic。"""
    rng = np.random.default_rng(5)
    q = bf.random_q_states(3000, rng)
    data = bf.simulate(np.array([0.5, 0.3, 0.0]), 0.0, q, rng)
    r = bf.fit_beta(data, min_n=10)
    assert r.verdict == "OK"
    assert r.beta_ci[0] <= 0.0 <= r.beta_ci[1]
