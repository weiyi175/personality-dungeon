#!/usr/bin/env python3
"""B-step 0 — 開放被驅動的跨玩家生態:合成玩家回應模型,掃回應率 ρ。

問題:把「真人會不會追 payoff 梯度」這個賭注變成可量化門檻,並檢驗
「開放被驅動系統能否維持 substrate(封閉)守不住的 L3 旋轉」。

模型(離散世代):
  狀態 q = 近期玩家 build 在 3 個 macro 策略 [expanding, contracting, exploring]
           上的分布(= ecology 滑動窗 q(t))。
  循環 payoff A(反對稱 RPS,expanding>exploring>contracting>expanding):
           fitness_i(q) = (A q)_i  → 稀有剋星分數高。
  每代來 N 個玩家,每人:
     以機率 ρ「回應」→ 從 softmax(β·fitness(q)) 抽策略(追高分)
     以機率 1−ρ「intrinsic」→ 從固定偏好 p_intrinsic 抽(想寫啥寫啥)= 攪動
  q 更新(開放、被持續驅動):q(t+1) = (1−α)q(t) + α·cohort_dist(t),
           cohort 用 N 人多項抽樣(含 demographic noise)。

對照組(A 的詛咒):封閉純 replicator(同 payoff、無 intrinsic 注入)。

輸出:reports/ecology/bstep0_driven_response.json + .png
用法:./venv/bin/python scripts/experiments/ecology_bstep0_driven_response.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

STRATS = ["expanding", "contracting", "exploring"]
# 反對稱 RPS:exp 剋 expl、con 剋 exp、expl 剋 con。A[i][j]=i 對 j 的 payoff。
A = np.array([[0.0, -1.0, 1.0],
              [1.0, 0.0, -1.0],
              [-1.0, 1.0, 0.0]])
CENTER = np.ones(3) / 3.0
_VERT = np.array([[np.cos(np.deg2rad(a)), np.sin(np.deg2rad(a))] for a in (90.0, 210.0, 330.0)])


def to2d(q):
    return q @ _VERT


def softmax(z):
    z = z - z.max()
    e = np.exp(z)
    return e / e.sum()


def winding(traj2d, c0):
    v = traj2d - c0
    ang = np.arctan2(v[:, 1], v[:, 0])
    d = np.diff(ang)
    d = (d + np.pi) % (2 * np.pi) - np.pi
    return d  # per-step signed angular increments (radians)


def run_open(rho, *, beta=4.0, alpha=0.25, N=30, T=3000, burn=1000,
             p_intrinsic=None, seed=0):
    rng = np.random.default_rng(seed)
    p_int = np.array(p_intrinsic) if p_intrinsic is not None else CENTER.copy()
    q = CENTER + rng.normal(0, 0.03, 3)
    q = np.clip(q, 1e-6, 1); q /= q.sum()
    traj = []
    for t in range(T):
        f = A @ q
        resp = softmax(beta * f)
        mix = rho * resp + (1.0 - rho) * p_int     # 期望 cohort 分布
        mix = np.clip(mix, 1e-9, 1); mix /= mix.sum()
        counts = rng.multinomial(N, mix)            # 有限 N → demographic noise
        cohort = counts / N
        q = (1.0 - alpha) * q + alpha * cohort
        q = np.clip(q, 1e-9, 1); q /= q.sum()
        traj.append(q.copy())
    traj = np.array(traj)
    return _metrics(traj, burn)


def run_closed_replicator(*, dt=0.05, T=3000, burn=1000, N=None, seed=0):
    """A 的詛咒對照:封閉純 replicator(反對稱 RPS=中心)。N 給值則加有限族群雜訊。"""
    rng = np.random.default_rng(seed)
    q = CENTER + rng.normal(0, 0.03, 3)
    q = np.clip(q, 1e-6, 1); q /= q.sum()
    traj = []
    for t in range(T):
        f = A @ q
        phi = q @ f
        q = q + dt * q * (f - phi)
        if N:                                       # 可選 demographic noise
            q = rng.multinomial(N, np.clip(q, 1e-9, 1) / q.sum()) / N
        q = np.clip(q, 1e-9, 1); q /= q.sum()
        traj.append(q.copy())
    return _metrics(np.array(traj), burn)


def _metrics(traj, burn):
    post = traj[burn:]
    c0 = to2d(CENTER)
    p2 = to2d(post)
    dist = np.linalg.norm(p2 - c0, axis=1)
    dincr = winding(p2, c0)
    net = float(np.sum(dincr))
    abss = float(np.sum(np.abs(dincr)))
    # 旋轉相干性:1=單向乾淨繞,0=來回抖動
    coherence = abs(net) / abss if abss > 1e-9 else 0.0
    # 是否衰減:後半 churn / 前半 churn
    half = len(dist) // 2
    sustain = float(dist[half:].mean() / dist[:half].mean()) if dist[:half].mean() > 1e-9 else 1.0
    minq = float(np.min(np.min(post, axis=1)))
    mono_frac = float(np.mean(np.max(post, axis=1) > 0.8))   # 近單一文化時間比
    ent = -np.sum(post * np.log(post + 1e-12), axis=1) / np.log(3)
    return {
        "churn": float(dist.mean()),               # 平均離中心距離(0=不動)
        "rot_rate_rev_per_gen": net / (2 * np.pi) / len(post),
        "coherence": float(coherence),
        "sustain_ratio": sustain,                   # ~1=不衰減,<<1=阻尼死
        "min_q": minq,                              # →0=逼近滅絕
        "mono_frac": mono_frac,                     # 近單一文化的時間占比
        "mean_diversity": float(ent.mean()),        # 正規化 Shannon,1=均勻
        "traj2d": p2[::5].tolist(),
    }


def classify(m):
    if m["churn"] < 0.04:
        return "不動 (static / 死在固定點)"
    if m["mono_frac"] > 0.15 or m["min_q"] < 0.02:
        return "塌陷傾向 (collapse-prone / 近單一文化)"
    if m["sustain_ratio"] < 0.6:
        return "阻尼死 (damped → 鬆弛回中心)"
    if m["coherence"] > 0.5 and m["churn"] > 0.06:
        return "★ 持續相干旋轉 (sustained coherent churn = L3-like)"
    return "持續但不相干 churn (noisy, 無明確旋轉)"


def main():
    seeds = [0, 1, 2, 3, 4]
    rhos = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    intrinsics = {"uniform": CENTER.tolist(), "skewed_real(.45/.45/.10)": [0.45, 0.45, 0.10]}

    results = {}
    for iname, pint in intrinsics.items():
        print(f"\n===== intrinsic = {iname} =====")
        print(f"{'rho':>4} {'churn':>6} {'rot/gen':>8} {'coher':>6} {'sustain':>7} "
              f"{'min_q':>6} {'mono%':>6} {'div':>5}  verdict")
        rows = []
        for rho in rhos:
            ms = [run_open(rho, p_intrinsic=pint, seed=s) for s in seeds]
            agg = {k: float(np.mean([m[k] for m in ms])) for k in ms[0] if k != "traj2d"}
            agg["verdict"] = classify(agg)
            agg["rho"] = rho
            agg["traj2d"] = ms[0]["traj2d"]
            rows.append(agg)
            print(f"{rho:>4.1f} {agg['churn']:>6.3f} {agg['rot_rate_rev_per_gen']:>8.4f} "
                  f"{agg['coherence']:>6.2f} {agg['sustain_ratio']:>7.2f} {agg['min_q']:>6.3f} "
                  f"{agg['mono_frac']*100:>5.1f}% {agg['mean_diversity']:>5.2f}  {agg['verdict']}")
        results[iname] = rows

    # A 的詛咒對照
    print("\n===== 對照:封閉純 replicator (A 的詛咒) =====")
    cl = [run_closed_replicator(seed=s) for s in seeds]
    clm = {k: float(np.mean([m[k] for m in cl])) for k in cl[0] if k != "traj2d"}
    cl_n = [run_closed_replicator(N=30, seed=s) for s in seeds]
    cln = {k: float(np.mean([m[k] for m in cl_n])) for k in cl_n[0] if k != "traj2d"}
    print(f"  determ : churn={clm['churn']:.3f} sustain={clm['sustain_ratio']:.2f} "
          f"min_q={clm['min_q']:.3f} → {classify(clm)}")
    print(f"  finite-N(30): churn={cln['churn']:.3f} sustain={cln['sustain_ratio']:.2f} "
          f"min_q={cln['min_q']:.3f} mono%={cln['mono_frac']*100:.1f} → {classify(cln)}")

    out = {"params": {"beta": 4.0, "alpha": 0.25, "N": 30, "T": 3000, "burn": 1000, "seeds": seeds},
           "payoff": A.tolist(),
           "sweep": {k: [{kk: vv for kk, vv in r.items() if kk != "traj2d"} for r in rows]
                     for k, rows in results.items()},
           "closed_baseline": {"deterministic": {k: v for k, v in clm.items() if k != "traj2d"},
                               "finite_N30": {k: v for k, v in cln.items() if k != "traj2d"}}}
    Path("reports/ecology").mkdir(parents=True, exist_ok=True)
    json.dump(out, open("reports/ecology/bstep0_driven_response.json", "w"), indent=2)

    # 圖:左=ρ 掃描指標,右=代表性 ρ 的單純形軌跡
    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 3, figsize=(17, 5.2))
        for iname, rows in results.items():
            xs = [r["rho"] for r in rows]
            ax[0].plot(xs, [r["churn"] for r in rows], "o-", label=f"churn [{iname}]")
            ax[0].plot(xs, [r["coherence"] for r in rows], "s--", label=f"coherence [{iname}]")
        ax[0].set_xlabel("rho (player response rate)"); ax[0].set_title("churn & rotation coherence vs rho")
        ax[0].axhline(0, color="k", lw=0.4); ax[0].legend(fontsize=7); ax[0].grid(alpha=0.3)
        for iname, rows in results.items():
            xs = [r["rho"] for r in rows]
            ax[1].plot(xs, [r["min_q"] for r in rows], "o-", label=f"min_q [{iname}]")
            ax[1].plot(xs, [r["mono_frac"] for r in rows], "^--", label=f"mono_frac [{iname}]")
        ax[1].set_xlabel("rho"); ax[1].set_title("collapse risk vs rho (min_q↓ / mono_frac↑ = bad)")
        ax[1].axhline(0.02, color="r", lw=0.4, ls=":"); ax[1].legend(fontsize=7); ax[1].grid(alpha=0.3)
        # 代表性軌跡(uniform intrinsic 的幾個 ρ)
        tri = np.vstack([_VERT, _VERT[0]]); ax[2].plot(tri[:, 0], tri[:, 1], "k-", lw=1)
        for v, n in zip(_VERT, STRATS):
            ax[2].annotate(n, v * 1.13, ha="center", fontsize=8)
        for r in results["uniform"]:
            if r["rho"] in (0.2, 0.5, 0.9):
                t = np.array(r["traj2d"])
                ax[2].plot(t[:, 0], t[:, 1], lw=0.5, alpha=0.7, label=f"rho={r['rho']}")
        ax[2].plot(*to2d(CENTER), "r*", ms=12)
        ax[2].set_aspect("equal"); ax[2].axis("off"); ax[2].legend(fontsize=8)
        ax[2].set_title("simplex trajectories (uniform intrinsic)")
        fig.tight_layout(); fig.savefig("reports/ecology/bstep0_driven_response.png", dpi=125)
        print("\nsaved reports/ecology/bstep0_driven_response.png")
    except Exception as e:
        print("plot skipped:", e)


if __name__ == "__main__":
    main()
