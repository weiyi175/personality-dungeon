#!/usr/bin/env python3
"""Reduced-form 生態動力學 harness — 共用核 + κ-arm（遊戲調參，exploratory）。

兩個 sweep 共用同一個核，但問不同問題：
  • κ-arm（本檔已實作，g=0）：R3/C 經濟調參。directional 算子已搬出 PvP（g=0），
    只剩 neg-freq 引擎。掃「多樣性 coin → PvP 戰力」耦合 κ × 阻尼 α × 玩家雜訊 σ，
    量 whiplash，找「爽（高 κ）又不翻（不 whiplash）」的甜區。決定 game-spec 的 (ii)/(iii)。
  • g-arm（DEFERRED，gated on pre-reg 最終點頭）：研究軌 confirmatory g-sweep。
    canonical pre-reg = docs/experiments/ecology_directional_pressure/
    DIRECTIONAL_PRESSURE_PREREGISTRATION.md（判定 live EcologyTracker，非另寫 Arm R/S）。
    本 session 的 dichotomy（finite-size 扭曲符號）已折進其 H2-faithful。本檔留 stub。

── 模型（κ-arm，供 review）─────────────────────────────────────────────────────
狀態 q=[q_agg,q_def,q_bal]，Σ=1，2-單純形，K=3。
fitness（g=0）：    f_i(q) = 1/K − q_i              # 純 neg-freq（稀缺→高），有界 [−2/3,+1/3]
玩家選擇（probit random-utility ＝ 使用者提的「期望值 + 高斯雜訊」）：
   每代來 N 個玩家，每人 archetype i 的效用
       U_i = (1+κ)·f_i(q)  +  ε_i ,   ε_i ~ N(0, σ²)
   選 argmax_i U_i。  (1+κ)＝coin 價值（κ=0 純聲望=(iii)；κ↑ coin 買戰力=(ii)）。
   等價：除以 (1+κ) → 有效雜訊 σ_eff = σ/(1+κ) → 有效選擇銳度 ∝ (1+κ)/σ。
族群更新（含阻尼）：   q(t+1) = (1−α)·q(t) + α·cohort,   cohort=N 人 argmax 計數/N
   α＝adaptation/turnover 率＝阻尼旋鈕（parking D）：α 小=阻尼重=小步=穩；α=1=整代翻=易 overshoot。
有限 N → demographic noise（暫態滅絕來源）。

預測（待本實驗驗）：g=0 下高 (1+κ)/σ + 高 α → 對滯後訊號 overshoot → whiplash（非穩態 monoculture）。
觀測 churn/coherence/sustain_ratio/mono_frac/diversity（_metrics，沿用 bstep0 的 whiplash 偵測）。

用法：  ./venv/bin/python scripts/experiments/ecology_reduced_form_bifurcation.py
輸出：  reports/ecology/reduced_form_kappa_sweep.json + .png
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

K = 3
CENTER = np.ones(K) / K
# 單純形 2D 嵌入 + 指標：adapted from ecology_bstep0_driven_response.py（同一套 whiplash 偵測）
_VERT = np.array([[np.cos(np.deg2rad(a)), np.sin(np.deg2rad(a))] for a in (90.0, 210.0, 330.0)])
ARCHE = ["aggressive", "defensive", "balanced"]


def to2d(q: np.ndarray) -> np.ndarray:
    return q @ _VERT


def _winding(traj2d: np.ndarray, c0: np.ndarray) -> np.ndarray:
    v = traj2d - c0
    ang = np.arctan2(v[:, 1], v[:, 0])
    d = np.diff(ang)
    return (d + np.pi) % (2 * np.pi) - np.pi


def neg_freq_fitness(q: np.ndarray, g: float = 0.0, d: np.ndarray | None = None) -> np.ndarray:
    """f_i = (1/K − q_i) + g·d_i。κ-arm 用 g=0（directional 算子已搬出 PvP）。"""
    f = (1.0 / K) - q
    if g and d is not None:
        f = f + g * d
    return f


def probit_cohort(f: np.ndarray, kappa: float, sigma: float, n_pop: int,
                  rng: np.random.Generator) -> np.ndarray:
    """N 個玩家每人 U_i=(1+κ)f_i+N(0,σ²)、選 argmax → 計數/N。回 cohort 分布。"""
    util = (1.0 + kappa) * f[None, :] + rng.normal(0.0, sigma, size=(n_pop, K))
    picks = np.argmax(util, axis=1)
    counts = np.bincount(picks, minlength=K)
    return counts / n_pop


def run_sim(*, kappa: float, alpha: float, sigma: float, n_pop: int = 30,
            g: float = 0.0, T: int = 3000, burn: int = 1000, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    q = CENTER + rng.normal(0, 0.03, K)
    q = np.clip(q, 1e-6, 1); q /= q.sum()
    traj = np.empty((T, K))
    for t in range(T):
        f = neg_freq_fitness(q, g=g)
        cohort = probit_cohort(f, kappa, sigma, n_pop, rng)
        q = (1.0 - alpha) * q + alpha * cohort
        q = np.clip(q, 1e-9, 1); q /= q.sum()
        traj[t] = q
    return _metrics(traj, burn)


def _metrics(traj: np.ndarray, burn: int) -> dict:
    post = traj[burn:]
    c0 = to2d(CENTER)
    p2 = to2d(post)
    dist = np.linalg.norm(p2 - c0, axis=1)
    dincr = _winding(p2, c0)
    net, abss = float(np.sum(dincr)), float(np.sum(np.abs(dincr)))
    coherence = abs(net) / abss if abss > 1e-9 else 0.0       # 1=單向乾淨繞,0=來回抖
    half = len(dist) // 2
    sustain = float(dist[half:].mean() / dist[:half].mean()) if dist[:half].mean() > 1e-9 else 1.0
    ent = -np.sum(post * np.log(post + 1e-12), axis=1) / np.log(K)
    return {
        "churn": float(dist.mean()),                  # 平均離中心距離（0=不動）
        "coherence": float(coherence),                # 旋轉相干性
        "sustain_ratio": sustain,                     # ~1=不衰減,<<1=阻尼死
        "min_q": float(np.min(np.min(post, axis=1))),  # →0=逼近暫態滅絕
        "mono_frac": float(np.mean(np.max(post, axis=1) > 0.8)),  # 近單一文化時間占比
        "mean_diversity": float(ent.mean()),          # 正規化 Shannon,1=均勻共存
    }


def classify(m: dict) -> str:
    if m["churn"] < 0.04:
        return "static-coexist (穩態共存/不動;高多樣)"
    if m["mono_frac"] > 0.15 or m["min_q"] < 0.02:
        return "★WHIPLASH-collapse (暫態近單一文化)"
    if m["coherence"] > 0.5 and m["churn"] > 0.06:
        return "★WHIPLASH-rotation (持續相干旋轉)"
    if m["sustain_ratio"] < 0.6:
        return "damped→coexist (阻尼回中心;OK)"
    return "noisy-coexist (抖動但不塌;OK)"


def _agg(rows: list[dict]) -> dict:
    return {k: float(np.mean([r[k] for r in rows])) for k in rows[0]}


def self_checks() -> bool:
    """模型行為自檢（g=0）。任一失敗 → 先修 harness 再信掃描。"""
    seeds = [0, 1, 2]
    ok = True
    print("=== self-checks (g=0) ===")
    # 1) 低銳度 + 近確定 (大 N)：穩態共存、高多樣、無暫態崩
    m1 = _agg([run_sim(kappa=0.0, alpha=0.25, sigma=0.5, n_pop=4000, seed=s) for s in seeds])
    c1 = m1["mean_diversity"] > 0.9 and m1["mono_frac"] < 0.02
    print(f"  [1] κ=0,α=.25,σ=.5,N=4000  div={m1['mean_diversity']:.3f} mono={m1['mono_frac']:.3f} churn={m1['churn']:.3f} -> {'PASS' if c1 else 'FAIL'} (期望穩態共存)")
    ok &= c1
    # 2) 高銳度 + 整代翻 + 有限 N：whiplash 現形（churn 起來或暫態崩）
    m2 = _agg([run_sim(kappa=8.0, alpha=1.0, sigma=0.2, n_pop=30, seed=s) for s in seeds])
    c2 = (m2["churn"] > 0.06) or (m2["mono_frac"] > 0.15)
    print(f"  [2] κ=8,α=1.0,σ=.2,N=30    churn={m2['churn']:.3f} mono={m2['mono_frac']:.3f} coher={m2['coherence']:.3f} -> {'PASS' if c2 else 'FAIL'} (期望 whiplash 現形)")
    ok &= c2
    # 3) 重阻尼：近靜止於中心（churn 低、高多樣）
    m3 = _agg([run_sim(kappa=2.0, alpha=0.05, sigma=0.5, n_pop=30, seed=s) for s in seeds])
    c3 = m3["churn"] < 0.06 and m3["mean_diversity"] > 0.85
    print(f"  [3] κ=2,α=.05,σ=.5,N=30    churn={m3['churn']:.3f} div={m3['mean_diversity']:.3f} -> {'PASS' if c3 else 'FAIL'} (期望重阻尼→近靜止)")
    ok &= c3
    print(f"  self-checks: {'ALL PASS' if ok else 'SOME FAIL'}\n")
    return ok


def kappa_sweep(*, sigma: float = 0.5, n_pop: int = 30,
                kappas=(0.0, 0.5, 1.0, 2.0, 4.0, 8.0),
                alphas=(0.1, 0.25, 0.5, 1.0),
                seeds=(0, 1, 2, 3, 4)) -> dict:
    """主掃描：κ（coin→戰力耦合）× α（阻尼）於固定 σ,N。找 whiplash 邊界 / 甜區。"""
    print(f"=== κ × α sweep (g=0, σ={sigma}, N={n_pop}, seeds={list(seeds)}) ===")
    hdr = "κ \\ α"
    print(f"{hdr:>6}", *[f"{a:>22.2f}" for a in alphas])
    grid = []
    for kap in kappas:
        cells = []
        line = [f"{kap:>6.1f}"]
        for al in alphas:
            m = _agg([run_sim(kappa=kap, alpha=al, sigma=sigma, n_pop=n_pop, seed=s) for s in seeds])
            v = classify(m)
            cells.append({"kappa": kap, "alpha": al, **m, "verdict": v})
            tag = "WHIP" if "WHIPLASH" in v else ("stat" if "static" in v else "ok")
            line.append(f"{tag:>4} d{m['mean_diversity']:.2f} m{m['mono_frac']:.2f}".rjust(22))
        grid.append(cells)
        print(*line)
    return {"sigma": sigma, "n_pop": n_pop, "kappas": list(kappas),
            "alphas": list(alphas), "seeds": list(seeds), "grid": grid}


def sigma_sensitivity(*, kappa=2.0, alpha=0.5, n_pop=30,
                      sigmas=(0.2, 0.35, 0.5, 0.8, 1.2), seeds=(0, 1, 2, 3, 4)) -> list[dict]:
    print(f"\n=== σ sensitivity @ κ={kappa}, α={alpha}, N={n_pop} ===")
    rows = []
    for sg in sigmas:
        m = _agg([run_sim(kappa=kappa, alpha=alpha, sigma=sg, n_pop=n_pop, seed=s) for s in seeds])
        rows.append({"sigma": sg, **m, "verdict": classify(m)})
        print(f"  σ={sg:>4.2f}  div={m['mean_diversity']:.3f} mono={m['mono_frac']:.3f} "
              f"churn={m['churn']:.3f} coher={m['coherence']:.3f} -> {classify(m)}")
    return rows


def _div_at(kappa: float, alpha: float, sigma: float, n_pop: int, seeds: list[int]) -> float:
    return _agg([run_sim(kappa=kappa, alpha=alpha, sigma=sigma, n_pop=n_pop, seed=s)
                 for s in seeds])["mean_diversity"]


def bisect_alpha_star(kappa: float, *, div_floor: float = 0.9, sigma: float = 0.5,
                      n_pop: int = 30, lo: float = 0.4, hi: float = 1.0, iters: int = 14,
                      seeds=range(8)) -> tuple[float, str]:
    """二分搜尋 whiplash onset α*：mean_diversity 跌破 div_floor 的 α（div 隨 α 遞減）。
    回 (α*, 說明)。α<α* → 多樣性 ≥ floor（安全）。stochastic 目標用多 seed 平均降噪。"""
    seeds = list(seeds)
    if _div_at(kappa, lo, sigma, n_pop, seeds) < div_floor:
        return lo, f"unsafe even at α={lo}"
    if _div_at(kappa, hi, sigma, n_pop, seeds) >= div_floor:
        return hi, f"safe even at α={hi} (range 內無 onset)"
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        if _div_at(kappa, mid, sigma, n_pop, seeds) >= div_floor:
            lo = mid
        else:
            hi = mid
    return round(0.5 * (lo + hi), 3), "onset"


def fine_alpha_grid(*, sigma=0.5, n_pop=30, kappas=(0.0, 1.0, 2.0, 4.0, 8.0),
                    alphas=(0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0), seeds=(0, 1, 2, 3, 4)) -> dict:
    print(f"\n=== fine α grid (g=0, σ={sigma}, N={n_pop}) — mean_diversity ===")
    hdr = "κ \\ α"
    print(f"{hdr:>5}", *[f"{a:>6.2f}" for a in alphas])
    grid = []
    for kap in kappas:
        row, cells = [f"{kap:>5.1f}"], []
        for al in alphas:
            m = _agg([run_sim(kappa=kap, alpha=al, sigma=sigma, n_pop=n_pop, seed=s) for s in seeds])
            cells.append({"kappa": kap, "alpha": al, **m})
            row.append(f"{m['mean_diversity']:>6.2f}")
        grid.append(cells); print(*row)
    return {"sigma": sigma, "n_pop": n_pop, "kappas": list(kappas), "alphas": list(alphas), "grid": grid}


def alpha_star_curve(*, sigma=0.5, n_pop=30, kappas=(0.0, 0.5, 1.0, 2.0, 4.0, 8.0)) -> list[dict]:
    print(f"\n=== α* (whiplash onset) via bisection (σ={sigma}, N={n_pop}) ===")
    print("  κ    α*(div≥0.9)  α*(div≥0.8)   解讀")
    rows = []
    for kap in kappas:
        a9, n9 = bisect_alpha_star(kap, div_floor=0.9, sigma=sigma, n_pop=n_pop)
        a8, n8 = bisect_alpha_star(kap, div_floor=0.8, sigma=sigma, n_pop=n_pop)
        rows.append({"kappa": kap, "alpha_star_div90": a9, "note90": n9,
                     "alpha_star_div80": a8, "note80": n8})
        print(f"  {kap:>4.1f}   {a9:>6.3f} ({n9[:12]:<12})  {a8:>6.3f} ({n8[:12]:<12})")
    return rows


def low_sigma_stress(*, n_pop=30, combos=((2.0, 0.8), (8.0, 0.8), (2.0, 1.0), (8.0, 1.0)),
                     sigmas=(0.1, 0.2, 0.3, 0.5), seeds=(0, 1, 2, 3, 4)) -> list[dict]:
    print(f"\n=== 高 α 低 σ 壓力測（危險角 N={n_pop}）===")
    rows = []
    for kap, al in combos:
        print(f"  κ={kap}, α={al}:")
        for sg in sigmas:
            m = _agg([run_sim(kappa=kap, alpha=al, sigma=sg, n_pop=n_pop, seed=s) for s in seeds])
            rows.append({"kappa": kap, "alpha": al, "sigma": sg, **m, "verdict": classify(m)})
            print(f"     σ={sg:>4.2f}  div={m['mean_diversity']:.3f} mono={m['mono_frac']:.3f} "
                  f"churn={m['churn']:.3f} -> {classify(m)}")
    return rows


def refine() -> None:
    out_dir = Path("reports/ecology"); out_dir.mkdir(parents=True, exist_ok=True)
    fg = fine_alpha_grid()
    ac = alpha_star_curve()
    ls = low_sigma_stress()
    out = {"fine_alpha_grid": fg, "alpha_star_curve": ac, "low_sigma_stress": ls,
           "method": "bisection on mean_diversity threshold (8-seed avg), 14 iters"}
    out_json = out_dir / "reduced_form_kappa_refine.json"
    json.dump(out, open(out_json, "w"), ensure_ascii=False, indent=2)
    print(f"\nsaved {out_json}")


def _plot(sweep: dict, out_png: Path) -> None:
    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print("plot skipped:", e); return
    ks, als = sweep["kappas"], sweep["alphas"]
    div = np.array([[c["mean_diversity"] for c in row] for row in sweep["grid"]])
    mono = np.array([[c["mono_frac"] for c in row] for row in sweep["grid"]])
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    for a, M, ttl, cm in ((ax[0], div, "mean_diversity (1=共存好)", "viridis"),
                          (ax[1], mono, "mono_frac (高=whiplash崩壞)", "magma")):
        im = a.imshow(M, origin="lower", aspect="auto", cmap=cm)
        a.set_xticks(range(len(als))); a.set_xticklabels(als)
        a.set_yticks(range(len(ks))); a.set_yticklabels(ks)
        a.set_xlabel("α (阻尼; 小=重阻尼)"); a.set_ylabel("κ (coin→戰力耦合)")
        a.set_title(ttl); fig.colorbar(im, ax=a)
    fig.suptitle(f"R3/C κ-sweep (g=0, σ={sweep['sigma']}, N={sweep['n_pop']})")
    fig.tight_layout(); fig.savefig(out_png, dpi=125)
    print(f"saved {out_png}")


def main() -> None:
    out_dir = Path("reports/ecology"); out_dir.mkdir(parents=True, exist_ok=True)
    ok = self_checks()
    sweep = kappa_sweep()
    sig = sigma_sensitivity()
    out = {"self_checks_pass": ok, "model": "probit U_i=(1+κ)f_i+N(0,σ²), q'=(1-α)q+α·cohort, g=0",
           "kappa_alpha_sweep": sweep, "sigma_sensitivity": sig}
    out_json = out_dir / "reduced_form_kappa_sweep.json"
    json.dump(out, open(out_json, "w"), ensure_ascii=False, indent=2)
    print(f"\nsaved {out_json}")
    _plot(sweep, out_dir / "reduced_form_kappa_sweep.png")


# ── g-arm（研究軌 confirmatory）— DEFERRED，gated on pre-reg 最終點頭 ────────────
def g_sweep_research(*args, **kwargs):  # noqa: D401
    """DEFERRED：研究軌 g-sweep。canonical = 判定 live EcologyTracker，見
    docs/experiments/ecology_directional_pressure/DIRECTIONAL_PRESSURE_PREREGISTRATION.md。
    共用核（neg_freq_fitness/_metrics/to2d）已就位，待 pre-reg 最終點頭再實作。"""
    raise NotImplementedError("g-arm gated on canonical pre-reg final nod (DIRECTIONAL_PRESSURE_PREREGISTRATION.md)")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "refine":
        refine()
    else:
        main()
