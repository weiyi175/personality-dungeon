"""b4 weight-vs-proportion phase relationship: is b4 genuinely winding in w_ but not p_?

Per seed, tail window: project (aggressive, defensive) deviations from tail-mean,
phase = atan2, unwrap, measure:
  net_turns  = (angle[-1]-angle[0]) / 2pi          (signed winding number)
  total_turns= sum|d angle| / 2pi                  (total angular path)
  coherence  = |net| / total                       (fraction of turning that is coherent)
Genuine rotation => large |net_turns| AND high coherence. Frozen/artifact => ~0 net.
"""
import csv, sys, math, statistics as st
from pathlib import Path
ROOT = Path("/home/user/personality-dungeon"); sys.path.insert(0, str(ROOT))

P2 = ROOT / "reports/experiments/l3_bottleneck/phase2"
CELLS = [
    ("ArmA_meanfield", "b5_scout/g1_mean_field_delta0p000"),
    ("b4_state_k",     "b4_scout/beta0p60_k0p08"),
    ("t_minibatch",    "t_scout/t_lattice4_minibatch"),
    ("b5_keystone",    "b5_scout/g2_sampled_delta0p000"),
]

def winding(rows, prefix, tail=1000):
    if f"{prefix}aggressive" not in rows[0].keys(): return None
    n = len(rows); begin = max(0, n - tail)
    a = [float(r[f"{prefix}aggressive"]) for r in rows][begin:]
    d = [float(r[f"{prefix}defensive"]) for r in rows][begin:]
    ma, md = sum(a)/len(a), sum(d)/len(d)
    ang = [math.atan2(dd-md, aa-ma) for aa, dd in zip(a, d)]
    # unwrap
    un = [ang[0]]
    for x in ang[1:]:
        dx = x - (un[-1] % (2*math.pi)) if False else x
        # simple unwrap
        prev = un[-1]; cand = x
        while cand - prev > math.pi: cand -= 2*math.pi
        while cand - prev < -math.pi: cand += 2*math.pi
        un.append(cand)
    net = (un[-1] - un[0]) / (2*math.pi)
    total = sum(abs(un[i+1]-un[i]) for i in range(len(un)-1)) / (2*math.pi)
    coh = abs(net)/total if total > 0 else 0.0
    return net, total, coh

print(f"{'cell':<16} | {'w_ net_turns':>12} {'w_ tot_turns':>12} {'w_ coher':>8} | {'p_ net_turns':>12} {'p_ tot':>7} {'p_ coher':>8}")
print("-"*100)
for label, sub in CELLS:
    sfs = sorted((P2/sub).glob("seed*.csv"))
    wn, wt_, wc, pn, pt, pc = [], [], [], [], [], []
    for sf in sfs:
        rows = list(csv.DictReader(open(sf)))
        rw = winding(rows, "w_"); rp = winding(rows, "p_")
        if rw: wn.append(rw[0]); wt_.append(rw[1]); wc.append(rw[2])
        if rp: pn.append(rp[0]); pt.append(rp[1]); pc.append(rp[2])
    def m(xs): return st.mean(xs) if xs else float('nan')
    wcol = (f"{m(wn):>12.2f} {m(wt_):>12.2f} {m(wc):>8.3f}" if wn else f"{'-- no w_ --':>34}")
    pcol = (f"{m(pn):>12.3f} {m(pt):>7.1f} {m(pc):>8.3f}" if pn else f"{'--':>28}")
    print(f"{label:<16} | {wcol} | {pcol}")
