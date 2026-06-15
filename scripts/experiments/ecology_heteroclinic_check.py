#!/usr/bin/env python3
"""(i) 驗證 b>a 是『邊界 heteroclinic 環』而非『內點極限環』。

數學預期（Zeeman/Hofbauer，3 策略線性 payoff replicator）：內點 FP 只能 center/
sink/source，內部無孤立極限環。b>a → 內點 FP 變 source（Re>0）→ 軌跡螺旋外擴、
貼三頂點繞、每圈週期無上限拉長（在頂點/鞍點附近停滯）。本腳本量三件事證實：
  1. 內點 FP 特徵值 Re>0（source）。
  2. min(q) 隨時間 →0（貼邊/逼近頂點）。
  3. 逐圈週期單調拉長（heteroclinic 的指紋；內點極限環會是固定週期）。

用法：./venv/bin/python scripts/experiments/ecology_heteroclinic_check.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from evolution.independent_rl import strategy_payoff_matrix  # noqa: E402
from scripts.experiments.ecology_replicator_probe import (  # noqa: E402
    interior_fixed_point, jacobian_at, tangent_eigs, to2d, replicator_rhs, _VERT, STRATEGY_SPACE,
)

DT = 0.005
T_MAX = 6000.0
N_STEPS = int(T_MAX / DT)


def run_bgta(a: float, b: float, cross: float = 0.2) -> dict:
    A = np.array(strategy_payoff_matrix(a=a, b=b, cross=cross))
    qstar = interior_fixed_point(A)
    eigs = tangent_eigs(jacobian_at(qstar, A))
    c0 = to2d(qstar)

    q = qstar + np.array([0.02, -0.01, -0.01])     # 內點微擾起跑
    q = np.clip(q, 1e-9, 1); q /= q.sum()

    prev_ang = np.arctan2(*(to2d(q) - c0)[::-1])
    cum = 0.0
    loop_times = []          # 每完成 2π 的時刻
    last_loop_t = 0.0
    min_comp_series = []
    t = 0.0
    for step in range(N_STEPS):
        k1 = replicator_rhs(q, A); k2 = replicator_rhs(q + 0.5*DT*k1, A)
        k3 = replicator_rhs(q + 0.5*DT*k2, A); k4 = replicator_rhs(q + DT*k3, A)
        q = np.clip(q + (DT/6)*(k1+2*k2+2*k3+k4), 0, 1); q /= q.sum()
        t += DT
        v = to2d(q) - c0
        ang = np.arctan2(v[1], v[0])
        dθ = ang - prev_ang
        dθ -= 2*np.pi*(dθ > np.pi); dθ += 2*np.pi*(dθ < -np.pi)
        cum += dθ; prev_ang = ang
        if abs(cum) >= 2*np.pi*(len(loop_times)+1):
            loop_times.append(t - last_loop_t); last_loop_t = t
        if step % 200 == 0:
            min_comp_series.append((round(t, 1), float(q.min())))
    return {
        "a": a, "b": b, "cross": cross,
        "interior_fp": qstar.tolist(),
        "eig": [[float(e.real), float(e.imag)] for e in eigs],
        "max_real": float(eigs.real.max()),
        "n_loops_completed": len(loop_times),
        "loop_periods": [round(x, 1) for x in loop_times],
        "period_ratio_last_first": round(loop_times[-1]/loop_times[0], 2) if len(loop_times) >= 2 else None,
        "min_comp_end": float(q.min()),
        "q_end": q.round(4).tolist(),
        "min_comp_trace": min_comp_series[::5],
    }


def main() -> None:
    print("=== b>a (swap of locked: a=0.9, b=1.0) ===")
    r = run_bgta(a=0.9, b=1.0)
    print("interior FP:", np.round(r["interior_fp"], 3))
    print("eigenvalues:", np.round(np.array(r["eig"]), 4),
          "→ max Re =", round(r["max_real"], 4),
          "(>0 = SOURCE ⇒ spiral OUT)" if r["max_real"] > 0 else "(≤0)")
    print(f"loops completed: {r['n_loops_completed']}")
    print(f"per-loop periods: {r['loop_periods']}")
    print(f"  → last/first period ratio = {r['period_ratio_last_first']}  "
          f"(>>1 ⇒ period lengthening = heteroclinic, NOT fixed-period interior limit cycle)")
    print(f"min(q) end = {r['min_comp_end']:.2e}  q_end={r['q_end']}  "
          f"(→0 ⇒ trajectory hugs boundary/vertex)")
    print("min(q) over time (t, min_q):")
    for t, m in r["min_comp_trace"]:
        print(f"    t={t:>7}  min_q={m:.2e}")

    Path("reports/ecology").mkdir(parents=True, exist_ok=True)
    json.dump(r, open("reports/ecology/heteroclinic_check.json", "w"), indent=2)

    # 軌跡圖（外擴貼邊）。
    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        A = np.array(strategy_payoff_matrix(a=0.9, b=1.0, cross=0.2))
        qstar = np.array(r["interior_fp"]); c0 = to2d(qstar)
        q = qstar + np.array([0.02, -0.01, -0.01]); q = np.clip(q, 1e-9, 1); q /= q.sum()
        pts = [to2d(q)]
        for _ in range(int(5000/DT)):
            k1 = replicator_rhs(q, A); k2 = replicator_rhs(q+0.5*DT*k1, A)
            k3 = replicator_rhs(q+0.5*DT*k2, A); k4 = replicator_rhs(q+DT*k3, A)
            q = np.clip(q+(DT/6)*(k1+2*k2+2*k3+k4), 0, 1); q /= q.sum()
            pts.append(to2d(q))
        pts = np.array(pts[::20])
        fig, ax = plt.subplots(figsize=(6.5, 6))
        tri = np.vstack([_VERT, _VERT[0]]); ax.plot(tri[:, 0], tri[:, 1], "k-", lw=1)
        for v, n in zip(_VERT, STRATEGY_SPACE):
            ax.annotate(n, v*1.12, ha="center", fontsize=11)
        ax.plot(pts[:, 0], pts[:, 1], lw=0.5, alpha=0.7, color="C3")
        ax.plot(*c0, "r*", ms=14)
        ax.set_title("b>a (a=0.9,b=1.0): spiral-OUT to boundary heteroclinic cycle\n"
                     "interior FP is a SOURCE; no interior limit cycle (Hofbauer n=3)", fontsize=9)
        ax.set_aspect("equal"); ax.axis("off"); fig.tight_layout()
        fig.savefig("reports/ecology/heteroclinic_check.png", dpi=130)
        print("saved reports/ecology/heteroclinic_check.png")
    except Exception as e:
        print("plot skipped:", e)


if __name__ == "__main__":
    main()
