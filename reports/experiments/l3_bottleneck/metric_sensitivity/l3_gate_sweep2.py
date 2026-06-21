"""Scale-calibrated gate sweep + genuine-rotation yardstick (Arm A mean-field).

turn_strengths are ~1e-6, so sweep min_turn_strength at the RELEVANT scale.
Anchor against Arm A (consistency=1.0 = proven-genuine rotation): if b4/t magnitudes
are negligible vs Arm A, their w_-L3 is a magnitude-blind-gate artifact.
"""
import csv, sys, statistics as st
from pathlib import Path
ROOT = Path("/home/user/personality-dungeon"); sys.path.insert(0, str(ROOT))
from analysis.cycle_metrics import classify_cycle_level

P2 = ROOT / "reports/experiments/l3_bottleneck/phase2"
CELLS = [
    ("ArmA_meanfield", "b5_scout/g1_mean_field_delta0p000"),
    ("t_minibatch",    "t_scout/t_lattice4_minibatch"),
    ("b4_state_k",     "b4_scout/beta0p60_k0p08"),
    ("b5_keystone",    "b5_scout/g2_sampled_delta0p000"),
]
MTS_SWEEP = [0.0, 1e-8, 3e-8, 1e-7, 3e-7, 1e-6, 3e-6, 1e-5]

def metrics(rows, prefix):
    if f"{prefix}aggressive" not in rows[0].keys(): return None
    series = {s: [float(r[f"{prefix}{s}"]) for r in rows] for s in ("aggressive","defensive","balanced")}
    c = classify_cycle_level(series, burn_in=1000, tail=1000, amplitude_threshold=0.02,
                             corr_threshold=0.09, eta=0.0, min_turn_strength=0.0,
                             stage3_method="turning", phase_smoothing=1)
    if c.stage3 is None: return {"reached": False, "consistency": 0.0, "turn_strength": 0.0}
    return {"reached": True, "consistency": float(c.stage3.score), "turn_strength": float(c.stage3.turn_strength)}

data_w, data_p = {}, {}
for label, sub in CELLS:
    sfs = sorted((P2/sub).glob("seed*.csv"))
    data_w[label] = [m for sf in sfs if (m:=metrics(list(csv.DictReader(open(sf))), "w_"))]
    data_p[label] = [m for sf in sfs if (m:=metrics(list(csv.DictReader(open(sf))), "p_"))]

print("=== turn_strength yardstick (w_ vs p_) ===")
print(f"  {'cell':<16} {'w_cons':>7} {'w_turnstr':>11} | {'p_cons':>7} {'p_turnstr':>11}")
for label,_ in CELLS:
    w, p = data_w[label], data_p[label]
    wc = f"{st.mean(r['consistency'] for r in w):.3f}" if w else "--"
    wt = f"{st.mean(r['turn_strength'] for r in w):.2e}" if w else "--"
    pc = f"{st.mean(r['consistency'] for r in p):.3f}" if p else "--"
    pt = f"{st.mean(r['turn_strength'] for r in p):.2e}" if p else "--"
    print(f"  {label:<16} {wc:>7} {wt:>11} | {pc:>7} {pt:>11}")

# ratio of b4/t magnitude to genuine ArmA magnitude
arm_w = st.mean(r['turn_strength'] for r in data_w["ArmA_meanfield"])
print(f"\n  genuine-rotation magnitude (ArmA w_ turn_strength) = {arm_w:.2e}")
for lab in ("b4_state_k","t_minibatch","b5_keystone"):
    m = st.mean(r['turn_strength'] for r in data_w[lab])
    print(f"    {lab:<14} w_ turn_strength / ArmA = {m/arm_w:.2e}  ({m:.2e})")

def l3(recs, eta, mts):
    return sum(1 for r in recs if r["reached"] and r["consistency"]>=eta and r["turn_strength"]>=mts)

print("\n=== SWEEP 1 (scale-calibrated): min_turn_strength (eta=0.55) -> w_ L3 ===")
labs = [l for l,_ in CELLS]
print("  " + f"{'mts':>8} | " + " ".join(f"{l:>14}" for l in labs))
for mts in MTS_SWEEP:
    print(f"  {mts:>8.0e} | " + " ".join(f"{l3(data_w[l],0.55,mts):>12}/{len(data_w[l])}" if data_w[l] else f"{'--':>14}" for l in labs))
