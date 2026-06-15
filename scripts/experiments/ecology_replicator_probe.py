#!/usr/bin/env python3
"""(b) 決定性實驗 — 鎖定 ecology payoff 在 replicator 下到底繞不繞圈。

pilot-independent。完全不碰真人資料：直接用 SDD §11.2 BL2 鎖定的 (a,b,cross)
建非遞移 payoff A，跑標準 replicator ODE：

    q̇_i = q_i · (f_i − φ),   f_i = (A q)_i,   φ = qᵀ A q

從多組『內點』初始（三策略都在）出發，問三件事：
  1. 內點不動點 q* 在哪、Jacobian 特徵值（線性化分類：旋進/旋出/中心）。
  2. 實現軌跡繞不繞 q*（帶號角速度的累積繞數 winding）。
  3. 到 q* 的距離趨勢（旋進=收斂 / 旋出=發散到邊環 / 中心=守恆閉軌 / 塌到邊頂）。

輸出：reports/ecology/replicator_probe.json + replicator_probe.png（單純形軌跡）。

用法：./venv/bin/python scripts/experiments/ecology_replicator_probe.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from evolution.independent_rl import STRATEGY_SPACE, strategy_payoff_matrix  # noqa: E402

# 鎖定值（與 api/ecology_tracker.EcologyParams 預設一致）。
A_PARAMS = dict(a=1.0, b=0.9, cross=0.20)
DT = 0.01
T_MAX = 3000.0           # 長積分，給旋進/旋出充分發展
N_STEPS = int(T_MAX / DT)
SAMPLE_EVERY = 25        # 軌跡落點抽樣（省記憶體/JSON）

# 單純形 2D 嵌入（正三角，頂點在 90/210/330 度），給角度與繪圖用。
_VERT = np.array([[np.cos(np.deg2rad(a)), np.sin(np.deg2rad(a))]
                  for a in (90.0, 210.0, 330.0)])


def to2d(q: np.ndarray) -> np.ndarray:
    return q @ _VERT


def fitness(q: np.ndarray, A: np.ndarray) -> np.ndarray:
    return A @ q


def replicator_rhs(q: np.ndarray, A: np.ndarray) -> np.ndarray:
    f = fitness(q, A)
    phi = q @ f
    return q * (f - phi)


def interior_fixed_point(A: np.ndarray) -> np.ndarray | None:
    """解 A q* = c·1, Σq*=1。零對角 RPS 型一般有唯一內點解。"""
    n = A.shape[0]
    # [A | -1] [q; c] = 0 ; Σq = 1  → 增廣求解
    M = np.zeros((n + 1, n + 1))
    M[:n, :n] = A
    M[:n, n] = -1.0
    M[n, :n] = 1.0
    rhs = np.zeros(n + 1)
    rhs[n] = 1.0
    try:
        sol = np.linalg.solve(M, rhs)
    except np.linalg.LinAlgError:
        return None
    q = sol[:n]
    if np.any(q < -1e-9) or np.any(q > 1 + 1e-9):
        return None
    return np.clip(q, 0, 1)


def jacobian_at(q: np.ndarray, A: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """replicator RHS 在 q 的數值 Jacobian，投影到 Σ=0 切空間後取特徵值。"""
    n = A.shape[0]
    J = np.zeros((n, n))
    for j in range(n):
        dq = np.zeros(n)
        dq[j] = eps
        J[:, j] = (replicator_rhs(q + dq, A) - replicator_rhs(q - dq, A)) / (2 * eps)
    return J


def tangent_eigs(J: np.ndarray) -> np.ndarray:
    """投影到 Σδ=0 切平面（單純形約束）後的特徵值。"""
    n = J.shape[0]
    # 切平面正交基（去掉 [1,1,1]/√n 方向）
    ones = np.ones(n) / np.sqrt(n)
    B, _ = np.linalg.qr(np.eye(n) - np.outer(ones, ones))
    B = B[:, :n - 1]              # n-1 個切向基
    Jt = B.T @ J @ B
    return np.linalg.eigvals(Jt)


def integrate(q0: np.ndarray, A: np.ndarray, qstar: np.ndarray) -> dict:
    """RK4 積分一條軌跡，回傳繞數、距離趨勢、最終落點。"""
    q = q0.copy()
    c0 = to2d(qstar)
    prev_ang = np.arctan2(*(to2d(q) - c0)[::-1])
    winding = 0.0
    d0 = np.linalg.norm(to2d(q) - c0)
    traj = [q.copy()]
    dist_series = [d0]

    for step in range(N_STEPS):
        k1 = replicator_rhs(q, A)
        k2 = replicator_rhs(q + 0.5 * DT * k1, A)
        k3 = replicator_rhs(q + 0.5 * DT * k2, A)
        k4 = replicator_rhs(q + DT * k3, A)
        q = q + (DT / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        q = np.clip(q, 0, 1)
        q = q / q.sum()

        v = to2d(q) - c0
        ang = np.arctan2(v[1], v[0])
        dθ = ang - prev_ang
        if dθ > np.pi:
            dθ -= 2 * np.pi
        elif dθ < -np.pi:
            dθ += 2 * np.pi
        winding += dθ
        prev_ang = ang

        if step % SAMPLE_EVERY == 0:
            traj.append(q.copy())
            dist_series.append(np.linalg.norm(v))

    d_end = np.linalg.norm(to2d(q) - c0)
    return {
        "q0": q0.tolist(),
        "q_end": q.tolist(),
        "winding_revolutions": winding / (2 * np.pi),
        "dist_start": float(d0),
        "dist_end": float(d_end),
        "dist_ratio_end_start": float(d_end / d0) if d0 > 1e-9 else None,
        "traj2d": [to2d(p).tolist() for p in traj],
        "min_component_end": float(q.min()),
    }


def classify(runs: list[dict], eig_real: float) -> str:
    avg_ratio = np.mean([r["dist_ratio_end_start"] for r in runs
                         if r["dist_ratio_end_start"] is not None])
    avg_rev = np.mean([abs(r["winding_revolutions"]) for r in runs])
    min_comp = np.mean([r["min_component_end"] for r in runs])

    rotating = avg_rev >= 0.75   # 至少繞了大半圈才算「有旋轉成分」
    if min_comp < 1e-3:
        return ("塌到邊/頂點 (collapse to edge/vertex)"
                + (" — 旋出到 heteroclinic 邊環" if rotating else " — 單調塌陷,無旋轉"))
    if not rotating:
        return "無旋轉:直接趨向內點不動點 (node/degenerate, no winding)"
    if avg_ratio < 0.7:
        return "螺旋進內點不動點 (spiral-IN / stable focus) — 不繞圈,旋轉被阻尼掉"
    if avg_ratio > 1.4:
        return "螺旋外擴 (spiral-OUT) → heteroclinic 邊環"
    return "閉軌/中心 (closed orbit / center) — 守恆旋轉"


def main() -> None:
    A = np.array(strategy_payoff_matrix(**A_PARAMS))
    qstar = interior_fixed_point(A)
    print("STRATEGY_SPACE:", STRATEGY_SPACE)
    print("payoff A:\n", A)
    print("interior fixed point q*:", None if qstar is None else qstar.round(4))

    eigs = None
    if qstar is not None:
        eigs = tangent_eigs(jacobian_at(qstar, A))
        print("tangent-space eigenvalues at q*:", np.round(eigs, 5))
        print("  → max Re =", round(float(eigs.real.max()), 5),
              "| has rotation (Im≠0):", bool(np.any(np.abs(eigs.imag) > 1e-6)))

    if qstar is None:
        print("無內點不動點;payoff 非 RPS 型,直接判定不旋轉。")
        Path("reports/ecology").mkdir(parents=True, exist_ok=True)
        json.dump({"params": A_PARAMS, "interior_fp": None,
                   "verdict": "no interior FP → payoff cannot host interior rotation"},
                  open("reports/ecology/replicator_probe.json", "w"), indent=2)
        return

    # 多組內點初始：靠近 q*、三個偏角、近重心。
    inits = [
        qstar + np.array([0.05, -0.03, -0.02]),
        qstar + np.array([-0.04, 0.05, -0.01]),
        np.array([0.6, 0.3, 0.1]),
        np.array([0.2, 0.6, 0.2]),
        np.array([0.34, 0.33, 0.33]),
        np.array([0.45, 0.45, 0.10]),
    ]
    inits = [np.clip(x, 1e-3, 1) for x in inits]
    inits = [x / x.sum() for x in inits]

    runs = [integrate(q0, A, qstar) for q0 in inits]
    for r in runs:
        print(f"  q0={np.round(r['q0'],3)} → rev={r['winding_revolutions']:+.2f} "
              f"dist {r['dist_start']:.3f}→{r['dist_end']:.3f} "
              f"(×{r['dist_ratio_end_start']:.2f}) min_comp_end={r['min_component_end']:.4f}")

    verdict = classify(runs, float(eigs.real.max()))
    print("\nVERDICT:", verdict)

    out = {
        "params": A_PARAMS,
        "payoff_matrix": A.tolist(),
        "strategy_space": list(STRATEGY_SPACE),
        "interior_fp": qstar.tolist(),
        "tangent_eigenvalues": [[float(e.real), float(e.imag)] for e in eigs],
        "max_real_part": float(eigs.real.max()),
        "has_imaginary": bool(np.any(np.abs(eigs.imag) > 1e-6)),
        "runs": [{k: v for k, v in r.items() if k != "traj2d"} for r in runs],
        "verdict": verdict,
    }
    Path("reports/ecology").mkdir(parents=True, exist_ok=True)
    json.dump(out, open("reports/ecology/replicator_probe.json", "w"), indent=2)

    # 繪圖：單純形三角 + 各軌跡 + q*。
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(7, 6.5))
        tri = np.vstack([_VERT, _VERT[0]])
        ax.plot(tri[:, 0], tri[:, 1], "k-", lw=1)
        for v, name in zip(_VERT, STRATEGY_SPACE):
            ax.annotate(name, v * 1.12, ha="center", va="center", fontsize=11)
        cs = to2d(qstar)
        ax.plot(*cs, "r*", ms=15, label=f"q* {qstar.round(2)}")
        for r in runs:
            t = np.array(r["traj2d"])
            ax.plot(t[:, 0], t[:, 1], lw=0.8, alpha=0.8)
            ax.plot(t[0, 0], t[0, 1], "o", ms=4, color="green")
            ax.plot(t[-1, 0], t[-1, 1], "s", ms=5, color="black")
        ax.set_title(f"Replicator on locked ecology payoff (a={A_PARAMS['a']},"
                     f"b={A_PARAMS['b']},cross={A_PARAMS['cross']})\n{verdict}",
                     fontsize=9)
        ax.set_aspect("equal"); ax.axis("off"); ax.legend(loc="upper right", fontsize=8)
        fig.tight_layout()
        fig.savefig("reports/ecology/replicator_probe.png", dpi=130)
        print("saved reports/ecology/replicator_probe.png")
    except Exception as e:
        print("plot skipped:", e)


if __name__ == "__main__":
    main()
