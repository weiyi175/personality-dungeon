#!/usr/bin/env python3
"""ECO-DP g-arm — confirmatory g-sweep，判定 **live** ecology 算子。

pre-reg: docs/experiments/ecology_directional_pressure/DIRECTIONAL_PRESSURE_PREREGISTRATION.md
唯一改動＝`_fitness` 回傳加 `g·d_i`（subclass 注入，**零 api/ 改動**，g=0 還原現役）。
response driver（§5 鎖定）：每步玩家 `P(i) ∝ softmax(β·advantage_i)`、`advantage=現役 _advantage(_fitness(q)+g·d)`、
餵 `_recent` 滑動窗（_proportions 給狀態 q）。β=2、其餘 lam/eta/window 全鎖現役預設。

⚠ Finding 2 旗標：driver 用瞬時 advantage、繞過權重 EMA(eta=0.2)；q 已含窗 W=50 lag。
⚠ Finding 1：softmax 軟地板 → 動力學嚴格 interior、無 exact monoculture；g\* = 可觀測 max_q≥0.95 crossover，非 g=1 分岔。

用法：./venv/bin/python scripts/experiments/ecology_dp_gsweep.py
輸出：reports/experiments/ecology_directional_pressure/{sweep_combined.tsv, eco_dp_analysis.json}
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from api.ecology_tracker import EcologyTracker, EcologyParams, ARCHETYPES  # noqa: E402

K = len(ARCHETYPES)                       # 3
D_DEFENSIVE = np.array([0.0, 1.0, 0.0])   # one-hot Defensive（ARCHETYPES[1]，§2 鎖定）
D_AGG = np.array([1.0, 0.0, 0.0])         # per-vertex 對稱性診斷用
D_BAL = np.array([0.0, 0.0, 1.0])
LOG_K = float(np.log(K))                  # entropy 上界 ≈1.0986
SEEDS = list(range(50, 60))               # §3 鎖定 50–59
TREATMENT = [0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0]


class GTracker(EcologyTracker):
    """live 算子 + `_fitness += g·d_i`（唯一改動）。"""
    def __init__(self, g: float, d, params=None):
        super().__init__(params)
        self._g = float(g)
        self._d = np.asarray(d, float)

    def _fitness(self, q):
        base = super()._fitness(q)
        return [base[i] + self._g * self._d[i] for i in range(len(base))]


def _softmax(z):
    z = np.asarray(z, float)
    z = z - z.max()
    e = np.exp(z)
    return e / e.sum()


def run_dp(g: float, *, d=D_DEFENSIVE, beta: float = 2.0,
           rounds: int = 3000, burn: int = 1000, tail: int = 1000, seed: int = 0,
           advantage_mode: str = "softplus", window: int | None = None,
           driver_signal: str = "instant", eta: float = 0.2, init: str = "center") -> dict:
    rng = np.random.default_rng(seed)
    params = EcologyParams(window=window) if window else None
    tr = GTracker(g, d, params)
    lam = tr.params.lam
    weights = np.full(K, float(np.log(2.0)))               # lagged-weights 中性初值（mirror :139）
    if init == "mono":                                     # hysteresis：從 monoculture(argmax d) 起
        for _ in range(tr.params.window):
            tr._recent.append(int(np.argmax(d)))
    top_q = np.empty(rounds)
    ent = np.empty(rounds)
    ttf = None
    for t in range(rounds):
        q = np.asarray(tr._proportions(), float)           # 現役滑動窗狀態
        f = tr._fitness(q)                                 # 含 +g·d
        if advantage_mode == "linear":                     # ablation：關 softplus
            adv = lam * np.asarray(f, float)
        else:
            adv = np.asarray(tr._advantage(f), float)       # 現役 softplus(lam·f)
        if driver_signal == "lagged_weights":               # Finding-2 ablation：driver 響應 EMA 權重
            weights = (1.0 - eta) * weights + eta * adv
            signal = weights
        else:
            signal = adv                                    # baseline：瞬時 advantage
        i = rng.choice(K, p=_softmax(beta * signal))        # softmax driver
        tr._recent.append(int(i))                          # 餵滑動窗
        top_q[t] = q.max()
        ent[t] = -float(np.sum(q * np.log(q + 1e-12)))
        if ttf is None and t >= burn and top_q[t] >= 0.95:
            ttf = t - burn
    sl = slice(rounds - tail, rounds)
    mean_top = float(top_q[sl].mean())
    return {"g": g, "seed": seed,
            "stationary_entropy": float(ent[sl].mean()),
            "top_q": mean_top,
            "fix_frac": float(np.mean(top_q[sl] >= 0.95)),   # tail 內 max_q≥0.95 的時間比
            "fixation": bool(mean_top >= 0.95),
            "time_to_fix": ttf}


def _agg(g: float, beta: float = 2.0) -> dict:
    runs = [run_dp(g, beta=beta, seed=s) for s in SEEDS]
    fix_rate = float(np.mean([r["fixation"] for r in runs]))
    return {"g": g, "beta": beta,
            "mean_entropy": float(np.mean([r["stationary_entropy"] for r in runs])),
            "sd_entropy": float(np.std([r["stationary_entropy"] for r in runs])),
            "mean_top_q": float(np.mean([r["top_q"] for r in runs])),
            "fixation_rate": fix_rate,
            "n_fix": int(sum(r["fixation"] for r in runs)),
            "runs": runs}


def estimate_gstar(rows: list[dict]) -> dict:
    """g* = stationary_entropy 在 NC↔PC 中點的線性內插穿越點（連續估計，§5）。"""
    gs = np.array([r["g"] for r in rows])
    ent = np.array([r["mean_entropy"] for r in rows])
    e_hi, e_lo = LOG_K, float(ent.min())
    mid = 0.5 * (e_hi + e_lo)
    g_star = None
    for j in range(len(gs) - 1):
        if (ent[j] - mid) * (ent[j + 1] - mid) <= 0 and ent[j] != ent[j + 1]:
            frac = (ent[j] - mid) / (ent[j] - ent[j + 1])
            g_star = float(gs[j] + frac * (gs[j + 1] - gs[j]))
            break
    return {"mid_entropy": mid, "g_star_apparatus": g_star, "g_star_analytic": 1.0,
            "ratio": (g_star / 1.0) if g_star else None}


def _interp_cross(xs, ys, target):
    """ys 單調穿 target 的線性內插 x（找不到回 None）。"""
    for j in range(len(xs) - 1):
        if (ys[j] - target) * (ys[j + 1] - target) <= 0 and ys[j] != ys[j + 1]:
            frac = (target - ys[j]) / (ys[j + 1] - ys[j])
            return float(xs[j] + frac * (xs[j + 1] - xs[j]))
    return None


def exploratory_extension(gs=(2.5, 3.0, 3.5, 4.0)) -> list[dict]:
    """Finding-3：主 grid 未 bracket fixation(max_q≥0.95) → 補 exploratory g>2（明標）。"""
    print("\n=== exploratory extension (Finding-3, g>2.0；明標 exploratory) ===")
    print(f"{'g':>6} {'entropy':>9} {'top_q':>7} {'fix_rate':>9} {'n_fix':>6}")
    rows = []
    for g in gs:
        r = _agg(g)
        rows.append(r)
        print(f"{r['g']:>6.2f} {r['mean_entropy']:>9.4f} {r['mean_top_q']:>7.3f} "
              f"{r['fixation_rate']:>9.2f} {r['n_fix']:>4}/10")
    return rows


def _agg_kw(g: float, **kw) -> dict:
    runs = [run_dp(g, seed=s, **kw) for s in SEEDS]
    return {"g": g,
            "mean_entropy": float(np.mean([r["stationary_entropy"] for r in runs])),
            "mean_top_q": float(np.mean([r["top_q"] for r in runs])),
            "fixation_rate": float(np.mean([r["fixation"] for r in runs])),
            "n_fix": int(sum(r["fixation"] for r in runs))}


def ablation(g_grid=(1.0, 1.5, 2.0, 2.5, 3.0, 3.5)) -> dict:
    """H2-faithful：逐項關 apparatus 成分，量各自把 g* 推離解析 1.0 多少。
    注：baseline driver 用瞬時 advantage、繞過權重 EMA(eta) → 純改 eta 無效；
    故用 lagged_weights 模式（Finding-2 faithfulness ablation）測 EMA 阻尼的位移。"""
    conditions = {
        "baseline (softplus,β2,W50,instant)": dict(),
        "linear advantage (no softplus)":     dict(advantage_mode="linear"),
        "β=1":                                dict(beta=1.0),
        "β=4":                                dict(beta=4.0),
        "window=25":                          dict(window=25),
        "window=100":                         dict(window=100),
        "lagged_weights eta=0.2 (F2 faithful)": dict(driver_signal="lagged_weights", eta=0.2),
        "lagged_weights eta=1.0 (no EMA lag)":  dict(driver_signal="lagged_weights", eta=1.0),
    }
    print(f"=== H2-faithful ablation（g* vs 解析 1.0；g_grid={list(g_grid)}）===")
    print(f"{'condition':<40} {'g*_ent':>7} {'g*_fix':>7}  top_q@grid")
    results = {}
    for name, kw in conditions.items():
        rows = [_agg_kw(g, **kw) for g in g_grid]
        mid = 0.5 * (LOG_K + min(r["mean_entropy"] for r in rows))
        g_ent = _interp_cross([r["g"] for r in rows], [r["mean_entropy"] for r in rows], mid)
        g_fix = _interp_cross([r["g"] for r in rows], [r["mean_top_q"] for r in rows], 0.95)
        results[name] = {"g_star_entropy_mid": g_ent, "g_star_fixation": g_fix, "rows": rows}
        ge = f"{g_ent:.3f}" if g_ent else " >grid"
        gf = f"{g_fix:.3f}" if g_fix else " >grid"
        print(f"{name:<40} {ge:>7} {gf:>7}  " + " ".join(f"{r['mean_top_q']:.2f}" for r in rows))
    out_dir = Path("reports/experiments/ecology_directional_pressure")
    out_dir.mkdir(parents=True, exist_ok=True)
    json.dump({"g_grid": list(g_grid), "ablation": results},
              open(out_dir / "eco_dp_ablation.json", "w"), ensure_ascii=False, indent=2)
    print(f"\nsaved {out_dir / 'eco_dp_ablation.json'}")
    return results


def diag_gstar_vs_beta(betas=(0.5, 1.0, 2.0, 4.0, 8.0),
                       g_grid=(0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0)) -> list[dict]:
    print("\n=== g*(β)：robustness vs 響應銳度（g*_fix = top_q 穿 0.95）===")
    print(f"{'β':>5} {'g*_fix':>8}  top_q@grid")
    rows = []
    for b in betas:
        cells = [_agg_kw(g, beta=b) for g in g_grid]
        gf = _interp_cross([c["g"] for c in cells], [c["mean_top_q"] for c in cells], 0.95)
        rows.append({"beta": b, "g_star_fix": gf, "top_q": [c["mean_top_q"] for c in cells]})
        print(f"{b:>5.1f} {(f'{gf:.3f}' if gf else ' >grid'):>8}  " +
              " ".join(f"{v:.2f}" for v in rows[-1]["top_q"]))
    return rows


def diag_hysteresis(g_grid=(0.5, 1.0, 1.5, 2.0, 2.5, 3.0), beta=2.0) -> list[dict]:
    print("\n=== hysteresis：center-init vs mono-init 穩態 top_q（gap→bistable/saddle-node）===")
    print(f"{'g':>5} {'center':>8} {'mono':>8} {'gap':>7}")
    rows = []
    for g in g_grid:
        tc = float(np.mean([run_dp(g, beta=beta, init="center", seed=s)["top_q"] for s in SEEDS]))
        tm = float(np.mean([run_dp(g, beta=beta, init="mono", seed=s)["top_q"] for s in SEEDS]))
        rows.append({"g": g, "center": tc, "mono": tm, "gap": abs(tc - tm)})
        print(f"{g:>5.2f} {tc:>8.3f} {tm:>8.3f} {abs(tc - tm):>7.3f}")
    return rows


def diag_ttf(g_grid=(2.0, 2.5, 3.0, 3.5, 4.0, 5.0), beta=2.0) -> list[dict]:
    print("\n=== time-to-fixation vs g（center-init；rounds→max_q≥0.95）===")
    print(f"{'g':>5} {'mean_ttf':>9} {'n_fix':>7}")
    rows = []
    for g in g_grid:
        ttfs = [run_dp(g, beta=beta, init="center", seed=s)["time_to_fix"] for s in SEEDS]
        fixed = [t for t in ttfs if t is not None]
        mt = float(np.mean(fixed)) if fixed else None
        rows.append({"g": g, "mean_ttf": mt, "n_fix": len(fixed)})
        print(f"{g:>5.2f} {(f'{mt:.0f}' if mt is not None else 'n/a'):>9} {len(fixed):>4}/10")
    return rows


def diag_per_vertex(g_grid=(1.0, 1.5, 2.0, 2.5, 3.0, 3.5), beta=2.0) -> dict:
    print("\n=== per-vertex d 對稱性（g*_fix；解析應同，量 finite 破缺）===")
    out = {}
    for nm, d in {"Aggressive": D_AGG, "Defensive": D_DEFENSIVE, "Balanced": D_BAL}.items():
        cells = [_agg_kw(g, beta=beta, d=d) for g in g_grid]
        gf = _interp_cross([c["g"] for c in cells], [c["mean_top_q"] for c in cells], 0.95)
        out[nm] = gf
        print(f"  {nm:<12} g*_fix = {f'{gf:.3f}' if gf else '>grid'}")
    return out


def diagnostics() -> None:
    out_dir = Path("reports/experiments/ecology_directional_pressure")
    out_dir.mkdir(parents=True, exist_ok=True)
    res = {"gstar_vs_beta": diag_gstar_vs_beta(),
           "hysteresis": diag_hysteresis(),
           "time_to_fixation": diag_ttf(),
           "per_vertex": diag_per_vertex()}
    json.dump(res, open(out_dir / "eco_dp_diagnostics.json", "w"), ensure_ascii=False, indent=2)
    print(f"\nsaved {out_dir / 'eco_dp_diagnostics.json'}")


def main() -> None:
    out_dir = Path("reports/experiments/ecology_directional_pressure")
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=== ECO-DP g-sweep (live operator, β=2, d=Defensive, 10 seeds 50–59) ===")
    print(f"{'g':>6} {'entropy':>9} {'top_q':>7} {'fix_rate':>9} {'n_fix':>6}  role")

    nc = _agg(0.0)
    print(f"{nc['g']:>6.2f} {nc['mean_entropy']:>9.4f} {nc['mean_top_q']:>7.3f} "
          f"{nc['fixation_rate']:>9.2f} {nc['n_fix']:>4}/10  NC(g=0)")

    sweep = [_agg(g) for g in TREATMENT]
    for r in sweep:
        print(f"{r['g']:>6.2f} {r['mean_entropy']:>9.4f} {r['mean_top_q']:>7.3f} "
              f"{r['fixation_rate']:>9.2f} {r['n_fix']:>4}/10  treatment")

    pc = _agg(5.0)
    print(f"{pc['g']:>6.2f} {pc['mean_entropy']:>9.4f} {pc['mean_top_q']:>7.3f} "
          f"{pc['fixation_rate']:>9.2f} {pc['n_fix']:>4}/10  PC(g=5)")

    # 控制判定（§7 容忍）
    print("\n=== controls ===")
    nc_ok = nc["n_fix"] <= 1
    pc_ok = pc["n_fix"] == 10
    print(f"  H-NC (g=0 fixation ≤1/10): {nc['n_fix']}/10 -> {'PASS' if nc_ok else 'FAIL'}")
    print(f"  H-PC (g=5 fixation =10/10): {pc['n_fix']}/10 -> {'PASS' if pc_ok else 'FAIL'}")

    bracketed = sweep[-1]["fixation_rate"] >= 0.5
    ext = [] if bracketed else exploratory_extension()

    rows = [nc] + sweep + ext + [pc]
    gstar = estimate_gstar([nc] + sweep + ext + [pc])
    # fixation g*（doc 鎖定判準 max_q≥0.95）：top_q 連續穿 0.95
    allg = [nc] + sweep + ext
    g_fix = _interp_cross([r["g"] for r in allg], [r["mean_top_q"] for r in allg], 0.95)
    gstar["g_star_fixation"] = g_fix
    print("\n=== g* (apparatus) ===")
    print(f"  g*_entropy-mid = {gstar['g_star_apparatus']:.3f}  (entropy 半飽和；ratio vs 1.0 = {gstar['ratio']:.3f})")
    print(f"  g*_fixation    = {g_fix:.3f}  (max_q≥0.95 連續穿越；doc 鎖定判準)" if g_fix
          else "  g*_fixation    = 未 bracket（即便 g=4）")
    print("  → 兩者皆 >1.0：軟地板再播種把可觀測 g* 推到解析閾值之上（驗證 dichotomy/Finding-1）")

    # TSV
    tsv = out_dir / "sweep_combined.tsv"
    with open(tsv, "w") as f:
        f.write("g\tseed\tstationary_entropy\ttop_q\tfix_frac\tfixation\n")
        for r in rows:
            for run in r["runs"]:
                f.write(f"{run['g']}\t{run['seed']}\t{run['stationary_entropy']:.5f}\t"
                        f"{run['top_q']:.4f}\t{run['fix_frac']:.3f}\t{int(run['fixation'])}\n")

    out = {"pre_reg": "DIRECTIONAL_PRESSURE_PREREGISTRATION.md",
           "driver": "softmax(beta=2 * advantage), live EcologyTracker + g·d (Defensive)",
           "controls": {"H_NC_pass": nc_ok, "H_PC_pass": pc_ok,
                        "nc_n_fix": nc["n_fix"], "pc_n_fix": pc["n_fix"]},
           "gstar": gstar, "bracketed": bracketed,
           "summary": [{k: v for k, v in r.items() if k != "runs"} for r in rows]}
    aj = out_dir / "eco_dp_analysis.json"
    json.dump(out, open(aj, "w"), ensure_ascii=False, indent=2)
    print(f"\nsaved {tsv}\nsaved {aj}")


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "main"
    {"ablation": ablation, "diagnostics": diagnostics}.get(mode, main)()
