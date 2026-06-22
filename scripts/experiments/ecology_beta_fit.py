"""β-instrument：從 live ecology 上傳回收真人對「稀缺」的 authoring 響應銳度 β。

研究軌 ECO-DP 的唯一開放經驗量＝真實 β（softmax 響應銳度）。甲（2026-06-22）證：
現有資料**不能** bound β——真人從沒走過 live ecology（>200 筆 subs 幾乎全 sim/replay）。
本估計器**不偽造數字**：它在有真人資料時回收 β，沒有時誠實判 INSUFFICIENT/UNIDENTIFIED。
一插真人就能算——這就是「讓 apparatus 收集就緒」的那塊 scaffolding。

模型（conditional logit，把 intrinsic 偏好和稀缺響應分離）:
    人在 author 前看到稀缺 → 每原型的 advantage adv_j = softplus(lam·(1/N − q_before_j))
    P(author archetype j | adv) = softmax(α_j + β · adv_j)
  - α_j = intrinsic archetype 偏好（甲：真人 [.48/.43/.09]，balanced 薄）；balanced 設 0 為基準。
  - β   = 稀缺響應銳度（ECO-DP 的開放量；g*(β) 曲線的橫軸）。β=0 ⟺ 純 intrinsic、不理稀缺。

可識別性：β 要可估，真人必須面對**變動的**稀缺（adv 跨筆要有 within-archetype 變異），
否則 adv_j 退化成 j 的固定函數 → 與 α_j 截距共線 → β 不可識別（估計器會判 UNIDENTIFIED）。

純離線：讀 reports/ecology/ecology_state.json，零 backend/前端改動 → 零 firewall 風險。
設計脈絡見 docs/experiments/ecology_directional_pressure/DIRECTIONAL_PRESSURE_PREREGISTRATION.md。
"""
from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

ARCHETYPES = ["aggressive", "defensive", "balanced"]
_NARCH = len(ARCHETYPES)
_REF = 2  # balanced = 截距基準（α_balanced ≡ 0），與甲的「balanced 薄」對齊

DEFAULT_LAM = 2.0           # 與 EcologyParams.lam 一致（ecology_tracker）
MIN_N = 30                  # 真人筆數下限（低於此 β 估計無意義）
MIN_SCARCITY_STD = 0.02     # adv 跨筆 within-archetype 變異下限（低於此 β 不可識別）


def softplus(x: np.ndarray) -> np.ndarray:
    # 數值安全 softplus，與 ecology_tracker._softplus 同形。
    return np.where(x > 0, x + np.log1p(np.exp(-x)), np.log1p(np.exp(x)))


def reconstruct_advantage(q_before: list[float], lam: float = DEFAULT_LAM) -> np.ndarray:
    """從 author-time 的生態佔比 q_before 重建人看到的 advantage 向量。

    複刻 ecology_tracker：fitness_i = 1/N − q_i；advantage_i = softplus(lam·fitness_i)。
    """
    q = np.asarray(q_before, dtype=float)
    fitness = (1.0 / _NARCH) - q
    return softplus(lam * fitness)


# ── 資料載入 ──────────────────────────────────────────────────────────────────

@dataclass
class BetaData:
    adv: np.ndarray        # (T, N) 每筆 author-time 的 advantage 向量
    chosen: np.ndarray     # (T,) 所選 archetype index
    n_real: int
    n_total: int
    n_artifact: int = 0    # run_id 空但 session_id 也空（程式/smoke 成對提交，非真人 session）


def load_real_submissions(state_path: str | Path, lam: float = DEFAULT_LAM) -> BetaData:
    """讀 ecology_state.json，只取**真正的真人 session**。

    篩選＝run_id 空（P7-H 清洗律排 sim/replay）**且 session_id 非空**。後者是 2026-06-23
    驗證踩到的坑：曾有 2 筆 run_id 空但 session_id 空、ts 僅差 ~21ms 的成對提交——那是
    程式/smoke 觸發、不是真人 author-under-scarcity。只看 run_id 會把這種 artifact 當真人。
    """
    data = json.loads(Path(state_path).read_text())
    subs = data.get("submissions", [])
    # params.lam 若存在則用它（換 lam 校準後 advantage 重建須一致）。
    lam = float(data.get("params", {}).get("lam", lam))
    adv_rows, chosen, n_artifact = [], [], 0
    for s in subs:
        if s.get("run_id"):          # 非空 = sim/replay，排除
            continue
        if not s.get("session_id"):  # run_id 空但無 session = 程式/smoke artifact
            n_artifact += 1
            continue
        q_before = s.get("score_components", {}).get("q_before")
        arch = s.get("archetype")
        if q_before is None or arch not in ARCHETYPES:
            continue
        adv_rows.append(reconstruct_advantage(q_before, lam))
        chosen.append(ARCHETYPES.index(arch))
    adv = np.asarray(adv_rows, dtype=float) if adv_rows else np.zeros((0, _NARCH))
    return BetaData(adv=adv, chosen=np.asarray(chosen, dtype=int),
                    n_real=len(adv_rows), n_total=len(subs), n_artifact=n_artifact)


# ── 估計器（conditional logit MLE）────────────────────────────────────────────

def _unpack(theta: np.ndarray) -> np.ndarray:
    """theta=[α_agg, α_def, β] → 完整 α 向量（α_balanced=0）。"""
    alpha = np.zeros(_NARCH)
    alpha[0], alpha[1] = theta[0], theta[1]
    return alpha


def _utilities(theta: np.ndarray, adv: np.ndarray) -> np.ndarray:
    alpha = _unpack(theta)
    beta = theta[2]
    return alpha[None, :] + beta * adv          # (T, N)


def _log_softmax(u: np.ndarray) -> np.ndarray:
    m = u.max(axis=1, keepdims=True)
    z = u - m
    return z - np.log(np.exp(z).sum(axis=1, keepdims=True))


def neg_loglik(theta: np.ndarray, adv: np.ndarray, chosen: np.ndarray) -> float:
    logp = _log_softmax(_utilities(theta, adv))
    return -logp[np.arange(len(chosen)), chosen].sum()


def neg_loglik_grad(theta: np.ndarray, adv: np.ndarray, chosen: np.ndarray) -> np.ndarray:
    u = _utilities(theta, adv)
    p = np.exp(_log_softmax(u))                 # (T, N)
    onehot = np.zeros_like(p)
    onehot[np.arange(len(chosen)), chosen] = 1.0
    resid = onehot - p                          # (T, N)
    g_alpha = resid.sum(axis=0)[:2]             # ∂/∂α_agg, ∂/∂α_def
    g_beta = (resid * adv).sum()                # ∂/∂β
    return -np.concatenate([g_alpha, [g_beta]])


def _hessian_fd(theta, adv, chosen, eps=1e-5):
    """negative-loglik Hessian（對 analytic gradient 做中央差分）→ 協方差 = inv(H)。"""
    n = len(theta)
    H = np.zeros((n, n))
    for i in range(n):
        tp, tm = theta.copy(), theta.copy()
        tp[i] += eps
        tm[i] -= eps
        H[:, i] = (neg_loglik_grad(tp, adv, chosen) - neg_loglik_grad(tm, adv, chosen)) / (2 * eps)
    return 0.5 * (H + H.T)


def scarcity_variation(adv: np.ndarray) -> float:
    """adv 跨筆的 within-archetype 變異（取各 archetype std 的中位數）。
    ~0 ⟹ 所有真人面對同一稀缺 → β 與截距共線、不可識別。"""
    if len(adv) < 2:
        return 0.0
    return float(np.median(adv.std(axis=0)))


@dataclass
class FitResult:
    verdict: str           # "OK" | "INSUFFICIENT_N" | "UNIDENTIFIED" | "NONCONVERGED"
    n_real: int
    n_total: int
    n_artifact: int = 0
    beta: float | None = None
    beta_se: float | None = None
    beta_ci: tuple[float, float] | None = None
    alpha_agg: float | None = None
    alpha_def: float | None = None
    scarcity_std: float | None = None
    loglik: float | None = None
    note: str = ""


def fit_beta(data: BetaData, *, min_n: int = MIN_N,
             min_scarcity_std: float = MIN_SCARCITY_STD) -> FitResult:
    adv, chosen = data.adv, data.chosen
    scar = scarcity_variation(adv)
    base = FitResult(verdict="", n_real=data.n_real, n_total=data.n_total,
                     n_artifact=data.n_artifact, scarcity_std=scar)

    if data.n_real < min_n:
        base.verdict = "INSUFFICIENT_N"
        base.note = (f"真人 session 筆數 {data.n_real} < {min_n}：β 不可估。"
                     f" 需收集（乙）；現有 {data.n_total} 筆中其餘為 sim/replay（run_id 非空）"
                     f"，另剔除 {data.n_artifact} 筆無 session 的程式/smoke artifact。")
        return base
    if scar < min_scarcity_std:
        base.verdict = "UNIDENTIFIED"
        base.note = (f"稀缺變異 {scar:.4f} < {min_scarcity_std}：真人幾乎都面對同一生態狀態，"
                     " β 與 intrinsic 截距共線、不可識別。需稀缺隨時間變動的收集設計。")
        return base

    theta0 = np.array([0.0, 0.0, 1.0])
    res = minimize(neg_loglik, theta0, args=(adv, chosen), jac=neg_loglik_grad,
                   method="BFGS")
    if not res.success:
        base.verdict = "NONCONVERGED"
        base.note = f"MLE 未收斂：{res.message}"
        return base

    theta = res.x
    cov = np.linalg.inv(_hessian_fd(theta, adv, chosen))
    se = float(np.sqrt(max(cov[2, 2], 0.0)))
    beta = float(theta[2])
    base.verdict = "OK"
    base.beta = beta
    base.beta_se = se
    base.beta_ci = (beta - 1.96 * se, beta + 1.96 * se)
    base.alpha_agg = float(theta[0])
    base.alpha_def = float(theta[1])
    base.loglik = float(-res.fun)
    base.note = "β 已回收（β=0 ⟺ 不理稀缺、純 intrinsic；β↑ ⟺ 越追稀缺）。"
    return base


# ── 合成自我回收（驗證估計器 + 估算所需樣本量）─────────────────────────────────

def simulate(alpha: np.ndarray, beta: float, q_states: np.ndarray,
             rng: np.random.Generator, lam: float = DEFAULT_LAM) -> BetaData:
    """給定真 (α, β) 與一組 author-time 生態狀態 q_states (T,N)，模擬 authored archetype。"""
    adv = np.stack([reconstruct_advantage(q, lam) for q in q_states])
    u = alpha[None, :] + beta * adv
    m = u.max(axis=1, keepdims=True)
    p = np.exp(u - m)
    p /= p.sum(axis=1, keepdims=True)
    chosen = np.array([rng.choice(_NARCH, p=p[t]) for t in range(len(adv))])
    return BetaData(adv=adv, chosen=chosen, n_real=len(adv), n_total=len(adv))


def random_q_states(n: int, rng: np.random.Generator, concentration: float = 1.0) -> np.ndarray:
    """模擬隨時間變動的稀缺：Dirichlet 抽生態佔比（變動 → β 可識別）。"""
    return rng.dirichlet([concentration] * _NARCH, size=n)


def _self_recovery(true_beta: float, n: int, seed: int = 0,
                   alpha=(0.3, 0.2)) -> FitResult:
    rng = np.random.default_rng(seed)
    alpha_vec = np.array([alpha[0], alpha[1], 0.0])
    q = random_q_states(n, rng)
    data = simulate(alpha_vec, true_beta, q, rng)
    return fit_beta(data, min_n=10)   # 合成驗證放寬 min_n


# ── 報告 ──────────────────────────────────────────────────────────────────────

def _fmt(r: FitResult) -> str:
    lines = [
        f"verdict      : {r.verdict}",
        f"n_real       : {r.n_real}  (total subs {r.n_total}, artifacts dropped {r.n_artifact})",
        f"scarcity_std : {r.scarcity_std:.4f}" if r.scarcity_std is not None else "scarcity_std : —",
    ]
    if r.verdict == "OK":
        lines += [
            f"β (response) : {r.beta:.3f}  ±{r.beta_se:.3f}  95%CI [{r.beta_ci[0]:.3f}, {r.beta_ci[1]:.3f}]",
            f"α_aggressive : {r.alpha_agg:.3f}   α_defensive : {r.alpha_def:.3f}   (α_balanced≡0)",
            f"loglik       : {r.loglik:.3f}",
        ]
    lines.append(f"note         : {r.note}")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description="β-instrument：回收真人稀缺-響應銳度")
    ap.add_argument("--state", default="reports/ecology/ecology_state.json",
                    help="ecology_state.json 路徑")
    ap.add_argument("--lam", type=float, default=DEFAULT_LAM)
    ap.add_argument("--min-n", type=int, default=MIN_N)
    ap.add_argument("--self-test", action="store_true",
                    help="跑合成自我回收（驗證估計器 + 樣本量）")
    args = ap.parse_args()

    if args.self_test:
        print("=== 合成自我回收（true β → β̂）===")
        for true_b in (0.0, 1.0, 2.0, 4.0):
            for n in (200, 1000):
                r = _self_recovery(true_b, n, seed=1)
                ok = (r.verdict == "OK" and r.beta_ci[0] <= true_b <= r.beta_ci[1])
                got = f"{r.beta:.2f} CI[{r.beta_ci[0]:.2f},{r.beta_ci[1]:.2f}]" if r.beta is not None else r.verdict
                print(f"  true β={true_b:<4} n={n:<5} → β̂={got:<28} {'✓覆蓋' if ok else '✗'}")
        return

    data = load_real_submissions(args.state, lam=args.lam)
    r = fit_beta(data, min_n=args.min_n)
    print(f"=== β-fit on {args.state} ===")
    print(_fmt(r))


if __name__ == "__main__":
    main()
