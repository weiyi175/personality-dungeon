import csv, sys, statistics as st
from pathlib import Path
ROOT = Path("/home/user/personality-dungeon"); sys.path.insert(0, str(ROOT))
from analysis.cycle_metrics import classify_cycle_level

P2 = ROOT / "reports/experiments/l3_bottleneck/phase2"
CELLS = [
    ("ArmA b5 g1_mean_field", "b5_scout/g1_mean_field_delta0p000"),
    ("ArmB b5 g2_sampled (neg keystone)", "b5_scout/g2_sampled_delta0p000"),
    ("ArmB b4 beta0p60_k0p08 (neg)", "b4_scout/beta0p60_k0p08"),
    ("ArmB t_lattice4_minibatch (neg)", "t_scout/t_lattice4_minibatch"),
    ("ArmB c1 small_world locked (neg)", "c1_scout/g2_c1_small_world_b10p0_m0p5"),
    ("c1 small_world native (neg)", "c1_native_scout/g2_c1_small_world_b10p0_m0p5"),
]

def amp(series):  # tail-1000 peak-to-peak per strategy, max over strategies
    return max(max(v[-1000:]) - min(v[-1000:]) for v in series.values())

def classify(rows, prefix, tail=1000):
    cols = rows[0].keys()
    if f"{prefix}aggressive" not in cols:
        return None, None, None, "no_cols"
    series = {s: [float(r[f"{prefix}{s}"]) for r in rows] for s in ("aggressive","defensive","balanced")}
    c = classify_cycle_level(series, burn_in=1000, tail=tail, amplitude_threshold=0.02,
                             corr_threshold=0.09, eta=0.55, min_turn_strength=0.0,
                             stage3_method="turning", phase_smoothing=1)
    s3 = float(c.stage3.score) if c.stage3 is not None else 0.0
    return c.level, s3, amp(series), "ok"

print(f"{'cell':<38} {'n':>2} | {'p_L3':>5} {'p_s3':>6} {'p_amp':>6} | {'w_L3':>5} {'w_s3':>6} {'w_amp':>6}")
print("-"*100)
for label, sub in CELLS:
    seedfiles = sorted((P2/sub).glob("seed*.csv"))
    if not seedfiles:
        print(f"{label:<38} -- none"); continue
    rec = {"p":[[],[],[]], "w":[[],[],[]]}
    note_w = ""
    for sf in seedfiles:
        rows = list(csv.DictReader(open(sf)))
        for pfx,key in (("p_","p"),("w_","w")):
            lv,s3,a,note = classify(rows, pfx)
            if note=="no_cols": note_w="(no w_ cols)"; continue
            rec[key][0].append(lv); rec[key][1].append(s3); rec[key][2].append(a)
    def fmt(k):
        lvs,s3s,amps = rec[k]
        if not lvs: return f"{'--':>5} {'--':>6} {'--':>6}"
        l3=sum(1 for x in lvs if x==3)
        return f"{l3:>3}/{len(lvs)} {st.mean(s3s):>6.3f} {st.mean(amps):>6.3f}"
    print(f"{label:<38} {len(seedfiles):>2} | {fmt('p')} | {fmt('w')} {note_w}")
