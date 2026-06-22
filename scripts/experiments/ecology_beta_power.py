"""β-instrument 的 validity（mis-spec robustness）+ power/design 模擬。

在花成本收集真人之前先回答兩個決策問題：
  A. **validity**：人若不完全照估計器假設的 softmax(α+β·adv) 響應，β̂ 還可信嗎？
     → 用幾種 plausible 的「真實響應 DGP」生成資料、套同一估計器，檢查它是否仍是
        scarcity-response 的**有效偵測器**（真有響應 → β̂>0 顯著；無響應 → β̂≈0、不誤報）。
        絕對 β 是 model-specific summary，跨 DGP 不可比；可比的是**號誌/顯著性/單調性**。
  B. **power/design**：給定真實 intrinsic α=[.48/.43/.09]，要收多少 n、稀缺要多大變動，
     才能 (i) reject β=0、(ii) 把 CI 收到能分辨 g*(β) 在意的 β 區間（1 vs 2 vs 4）？

依賴估計器 scripts/experiments/ecology_beta_fit.py。純模擬、零 apparatus 改動。
"""
from __future__ import annotations

import argparse
import math

import numpy as np

from scripts.experiments import ecology_beta_fit as bf

_NARCH = bf._NARCH

# 真實 intrinsic 偏好（甲：[.48/.43/.09]）→ β=0、balanced 基準下的 logit 截距。
P_INTRINSIC = np.array([0.48, 0.43, 0.09])
ALPHA_REAL = np.log(P_INTRINSIC / P_INTRINSIC[bf._REF])   # [α_agg, α_def, 0]

# 稀缺變動 regime：Dirichlet concentration（高→近均勻低變動；低→高變動）。
SCARCITY_REGIMES = {"low": 20.0, "mid": 5.0, "high": 1.0}


# ── 各種「真實響應」DGP（A：validity）──────────────────────────────────────────
# 都吃 author-time 生態 q（T,N），輸出每筆 authored archetype index。
# 共識：稀缺 s_j = 1/N − q_j（稀有→大）。各 DGP 用不同 link/covariate 表「追稀缺」。

def _choose(p: np.ndarray, rng) -> np.ndarray:
    return np.array([rng.choice(_NARCH, p=p[t]) for t in range(len(p))])


def dgp_softmax(q, alpha, strength, rng):
    """估計器自家模型：P=softmax(α + β·softplus(lam·s))。strength=β。"""
    adv = np.stack([bf.reconstruct_advantage(qt) for qt in q])
    u = alpha[None, :] + strength * adv
    p = np.exp(u - u.max(1, keepdims=True)); p /= p.sum(1, keepdims=True)
    return _choose(p, rng)


def dgp_probit(q, alpha, strength, rng):
    """mis-spec：響應走 probit（高斯 utility 噪音）而非 logit，covariate 仍 adv。"""
    adv = np.stack([bf.reconstruct_advantage(qt) for qt in q])
    u = alpha[None, :] + strength * adv + rng.normal(0, 1.0, size=adv.shape)
    return u.argmax(1)


def dgp_qlinear(q, alpha, strength, rng):
    """mis-spec：人對佔比 q 本身線性反應（−q，普及就避開），不經 softplus(adv)。"""
    qa = np.asarray(q)
    u = alpha[None, :] + strength * (-qa) + rng.gumbel(0, 1.0, size=qa.shape)
    return u.argmax(1)


def dgp_rank(q, alpha, strength, rng):
    """mis-spec：人只認稀缺的**排序**（最稀有 +1、中 0、最普及 −1），不認量級。"""
    qa = np.asarray(q)
    s = (1.0 / _NARCH) - qa
    ranks = np.argsort(np.argsort(s, axis=1), axis=1) - 1.0   # {-1,0,1}
    u = alpha[None, :] + strength * ranks + rng.gumbel(0, 1.0, size=qa.shape)
    return u.argmax(1)


def dgp_noresponse(q, alpha, strength, rng):
    """null：完全不理稀缺，純 intrinsic（strength 被忽略）。"""
    p = np.exp(alpha - alpha.max()); p /= p.sum()
    return rng.choice(_NARCH, size=len(q), p=p)


DGPS = {"softmax": dgp_softmax, "probit": dgp_probit,
        "qlinear": dgp_qlinear, "rank": dgp_rank, "noresponse": dgp_noresponse}


def _fit_from_q(q, chosen):
    adv = np.stack([bf.reconstruct_advantage(qt) for qt in q])
    data = bf.BetaData(adv=adv, chosen=np.asarray(chosen), n_real=len(adv), n_total=len(adv))
    return bf.fit_beta(data, min_n=10)


def run_validity(n=1500, reps=120, seed=0) -> None:
    """A：各 DGP（真有響應 strength>0，除 noresponse）→ 估計器是否正確偵測。"""
    rng = np.random.default_rng(seed)
    q = bf.random_q_states(n, rng, concentration=SCARCITY_REGIMES["high"])
    print(f"=== A. Validity（n={n}, reps={reps}, high-var scarcity）===")
    print(f"{'DGP':<11}{'真響應?':<9}{'β̂ median':<12}{'P(β̂>0 顯著)':<14}{'P(誤報null)':<12}verdict")
    for name, dgp in DGPS.items():
        truth = name != "noresponse"
        betas, sig_pos, false_pos = [], 0, 0
        for r in range(reps):
            rr = np.random.default_rng(seed * 1000 + r)
            qq = bf.random_q_states(n, rr, concentration=SCARCITY_REGIMES["high"])
            chosen = dgp(qq, ALPHA_REAL, 2.0, rr)
            res = _fit_from_q(qq, chosen)
            if res.verdict != "OK":
                continue
            betas.append(res.beta)
            excl0 = not (res.beta_ci[0] <= 0 <= res.beta_ci[1])
            if excl0 and res.beta > 0:
                sig_pos += 1
            if excl0:                      # null DGP 下這就是誤報
                false_pos += 1
        med = float(np.median(betas)) if betas else float("nan")
        if truth:
            verdict = "✓偵測到響應" if sig_pos / reps > 0.9 else "⚠ power 不足"
            print(f"{name:<11}{'是':<9}{med:<12.3f}{sig_pos/reps:<14.2f}{'—':<12}{verdict}")
        else:
            verdict = "✓不誤報" if false_pos / reps < 0.1 else "✗ 假陽性過高"
            print(f"{name:<11}{'否(null)':<9}{med:<12.3f}{'—':<14}{false_pos/reps:<12.2f}{verdict}")
    print("註：絕對 β 跨 DGP 不可比（不同 link/covariate）；可比的是「真響應→β̂>0 顯著、"
          "null→不誤報」。確立 β-instrument 為 scarcity-response 的有效偵測器 + 單調指標。")


# ── B：power / design sweep ───────────────────────────────────────────────────

def run_power(reps=200, seed=1) -> None:
    ns = [100, 300, 1000, 3000]
    betas = [0.5, 1.0, 2.0, 4.0]
    print(f"\n=== B. Power/design（intrinsic α=[.48/.43/.09], softmax DGP, reps={reps}）===")
    # 先報各 regime 達到的稀缺變異（解讀 power 用）。
    rng0 = np.random.default_rng(seed)
    print("scarcity regime 達成的 median scarcity_std：", {
        k: round(bf.scarcity_variation(
            np.stack([bf.reconstruct_advantage(qt)
                      for qt in bf.random_q_states(2000, rng0, concentration=c)])), 4)
        for k, c in SCARCITY_REGIMES.items()})
    for regime, conc in SCARCITY_REGIMES.items():
        print(f"\n-- scarcity={regime} (conc={conc}) --")
        print(f"{'true β':<8}{'n':<7}{'P(OK)':<8}{'P(reject β=0)':<15}{'median CI半寬':<14}")
        for tb in betas:
            for n in ns:
                ok = rej0 = 0
                halfws = []
                for r in range(reps):
                    rr = np.random.default_rng(seed * 9999 + r)
                    q = bf.random_q_states(n, rr, concentration=conc)
                    chosen = dgp_softmax(q, ALPHA_REAL, tb, rr)
                    res = _fit_from_q(q, chosen)
                    if res.verdict != "OK":
                        continue
                    ok += 1
                    halfws.append((res.beta_ci[1] - res.beta_ci[0]) / 2)
                    if not (res.beta_ci[0] <= 0 <= res.beta_ci[1]):
                        rej0 += 1
                hw = float(np.median(halfws)) if halfws else float("nan")
                print(f"{tb:<8}{n:<7}{ok/reps:<8.2f}{rej0/reps:<15.2f}{hw:<14.3f}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["validity", "power", "both"], default="both")
    ap.add_argument("--reps", type=int, default=150)
    args = ap.parse_args()
    if args.mode in ("validity", "both"):
        run_validity(reps=args.reps)
    if args.mode in ("power", "both"):
        run_power(reps=args.reps)


if __name__ == "__main__":
    main()
