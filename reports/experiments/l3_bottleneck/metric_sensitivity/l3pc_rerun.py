"""L3-BN positive-control PROTOCOL-2 RERUN (post-hoc supplementary cross-protocol check).

Re-runs the paper_draft_v1 L0->L2 positive control (seeds 1-10 x sigma in
{0.005,0.05,0.20} = 30 runs) under the EXACT protocol-2 decision chain as executed
by the confirmatory runner (simulation/b5_tangential_drift.py::_seed_metrics):
  series = p_* proportions; burn_in=1000; tail=1000; rounds=3000;
  amplitude_threshold=0.02; corr_threshold=0.09; eta=0.55; min_turn_strength=0.0;
  stage3_method="turning"; phase_smoothing=1; strategies=(agg,def,bal).

This is NOT part of the locked confirmatory (pre-reg locked the positive control as
"not re-run"). It is a supplementary check that can only strengthen the positive
boundary. Records p_-level (binding) and w_-level (pre-reg-literal cross-check).
"""
import csv, json, subprocess, sys, statistics as st
from pathlib import Path

ROOT = Path("/home/user/personality-dungeon")
sys.path.insert(0, str(ROOT))
from analysis.cycle_metrics import classify_cycle_level  # noqa: E402

SEEDS = list(range(1, 11))
SIGMAS = [0.005, 0.05, 0.20]
ROUNDS = 3000
OUTDIR = ROOT / "reports/experiments/l3_bottleneck/phase2_positive_control_protocol2"
OUTDIR.mkdir(parents=True, exist_ok=True)
TMP = Path("/tmp/l3pc_runs")
TMP.mkdir(exist_ok=True)


def run_sim(seed: int, sigma: float) -> Path:
    out_csv = TMP / f"pc_s{seed}_sig{sigma}.csv"
    cmd = [
        str(ROOT / "venv/bin/python"), "-m", "simulation.run_simulation",
        "--enable-events",
        "--events-json", "docs/personality_dungeon_v1/02_event_templates_v1.json",
        "--popularity-mode", "expected",
        "--seed", str(seed), "--rounds", str(ROUNDS), "--players", "300",
        "--payoff-mode", "matrix_ab", "--a", "0.8", "--b", "0.9",
        "--matrix-cross-coupling", "0.0",
        "--selection-strength", str(sigma), "--init-bias", "0.12",
        "--event-failure-threshold", "0.72", "--event-health-penalty", "0.10",
        "--out", str(out_csv),
    ]
    r = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"sim failed seed={seed} sigma={sigma} rc={r.returncode}\n{r.stderr[-2000:]}")
    return out_csv


def classify_series(rows, prefix):
    series = {
        "aggressive": [float(x[f"{prefix}aggressive"]) for x in rows],
        "defensive":  [float(x[f"{prefix}defensive"]) for x in rows],
        "balanced":   [float(x[f"{prefix}balanced"]) for x in rows],
    }
    c = classify_cycle_level(
        series, burn_in=1000, tail=1000, amplitude_threshold=0.02,
        corr_threshold=0.09, eta=0.55, min_turn_strength=0.0,
        stage3_method="turning", phase_smoothing=1,
    )
    s3 = float(c.stage3.score) if c.stage3 is not None else 0.0
    return c.level, s3


from concurrent.futures import ThreadPoolExecutor  # noqa: E402

conds = [(seed, sigma) for sigma in SIGMAS for seed in SEEDS]
print(f"launching {len(conds)} sims in parallel (10 workers)...", flush=True)
with ThreadPoolExecutor(max_workers=10) as ex:
    csv_paths = list(ex.map(lambda sc: run_sim(*sc), conds))
print("all sims done; classifying...", flush=True)

results = []
for (seed, sigma), csvp in zip(conds, csv_paths):
    rows = list(csv.DictReader(open(csvp)))
    p_level, p_s3 = classify_series(rows, "p_")
    w_level, w_s3 = classify_series(rows, "w_")
    results.append({"seed": seed, "sigma": sigma,
                    "p_cycle_level": p_level, "p_stage3_score": p_s3,
                    "w_cycle_level": w_level, "w_stage3_score": w_s3})
    print(f"seed={seed:2d} sigma={sigma:<6} -> p_level={p_level} p_s3={p_s3:.4f} | w_level={w_level}", flush=True)

# aggregate
p_l2plus = sum(1 for r in results if r["p_cycle_level"] >= 2)
p_l3 = sum(1 for r in results if r["p_cycle_level"] == 3)
p_s3s = [r["p_stage3_score"] for r in results]
by_sigma = {}
for sigma in SIGMAS:
    sub = [r for r in results if r["sigma"] == sigma]
    by_sigma[str(sigma)] = {
        "n": len(sub),
        "L2plus": sum(1 for r in sub if r["p_cycle_level"] >= 2),
        "L3": sum(1 for r in sub if r["p_cycle_level"] == 3),
        "p_stage3_mean": st.mean(r["p_stage3_score"] for r in sub),
        "p_stage3_max": max(r["p_stage3_score"] for r in sub),
    }

summary = {
    "label": "L0->L2 positive control re-run under protocol-2 (p_* chain) — supplementary",
    "n_runs": len(results),
    "conditions": {"seeds": SEEDS, "sigmas": SIGMAS, "rounds": ROUNDS,
                   "popularity_mode": "expected", "init_bias": 0.12,
                   "a": 0.8, "b": 0.9, "ft": 0.72, "hp": 0.10, "players": 300},
    "protocol2_chain": {"series": "p_*", "burn_in": 1000, "tail": 1000,
                        "amplitude_threshold": 0.02, "corr_threshold": 0.09,
                        "eta": 0.55, "min_turn_strength": 0.0,
                        "stage3_method": "turning", "phase_smoothing": 1},
    "p_L2plus": p_l2plus, "p_L3": p_l3,
    "p_stage3_mean": st.mean(p_s3s), "p_stage3_min": min(p_s3s), "p_stage3_max": max(p_s3s),
    "by_sigma": by_sigma,
    "w_literal_L2plus": sum(1 for r in results if r["w_cycle_level"] >= 2),
    "w_literal_note": "pre-reg-literal w_* series fails stage-1 at low sigma (amp<0.02); binding chain is p_*",
    "runs": results,
}
(OUTDIR / "positive_control_protocol2.json").write_text(json.dumps(summary, indent=2))

# summary TSV (matches confirmatory per-cell TSV columns)
with open(OUTDIR / "positive_control_protocol2_summary.tsv", "w", newline="") as f:
    w = csv.writer(f, delimiter="\t")
    w.writerow(["condition", "seed", "sigma", "cycle_level", "stage3_score", "w_cycle_level", "w_stage3_score"])
    for r in results:
        w.writerow(["l0_l2_positive_control", r["seed"], r["sigma"],
                    r["p_cycle_level"], f"{r['p_stage3_score']:.6f}",
                    r["w_cycle_level"], f"{r['w_stage3_score']:.6f}"])

print("\n" + "=" * 72)
print(f"POSITIVE CONTROL under protocol-2 (p_* chain): L>=2 = {p_l2plus}/{len(results)}  | L3 = {p_l3}/{len(results)}")
print(f"stage3 consistency: mean={st.mean(p_s3s):.4f} min={min(p_s3s):.4f} max={max(p_s3s):.4f}")
for s, v in by_sigma.items():
    print(f"  sigma={s}: L>=2 {v['L2plus']}/{v['n']}  L3 {v['L3']}/{v['n']}  s3_mean={v['p_stage3_mean']:.4f} s3_max={v['p_stage3_max']:.4f}")
print(f"w_-literal L>=2 = {summary['w_literal_L2plus']}/{len(results)} (degenerate at low sigma; see note)")
print("=" * 72)
print("written:", OUTDIR)
