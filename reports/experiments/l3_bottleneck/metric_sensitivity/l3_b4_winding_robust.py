"""b4 p_ winding robustness probe — Case A vs Case B.

Q: would a winding-number classifier call b4's OBSERVABLE (p_) trajectory a positive?
Decisive test = window-length scaling: genuine steady rotation => net_turns grows
~linearly with window (constant angular velocity); jitter/drift => no steady advance.
Plus: per-seed sign consistency, per-quarter steadiness, separation from matched
sampled null (b5 keystone) and positive anchor (ArmA).
"""
import csv, sys, math, statistics as st
from pathlib import Path
ROOT = Path("/home/user/personality-dungeon"); sys.path.insert(0, str(ROOT))

P2 = ROOT / "reports/experiments/l3_bottleneck/phase2"
CELLS = {
    "ArmA": "b5_scout/g1_mean_field_delta0p000",
    "b4":   "b4_scout/beta0p60_k0p08",
    "t":    "t_scout/t_lattice4_minibatch",
    "b5":   "b5_scout/g2_sampled_delta0p000",
}

def net_winding(a, d):
    """signed net turns + total angular path (turns) over the given segment."""
    ma, md = sum(a)/len(a), sum(d)/len(d)
    ang = [math.atan2(dd-md, aa-ma) for aa, dd in zip(a, d)]
    un = [ang[0]]
    for x in ang[1:]:
        cand = x; prev = un[-1]
        while cand - prev > math.pi: cand -= 2*math.pi
        while cand - prev < -math.pi: cand += 2*math.pi
        un.append(cand)
    net = (un[-1]-un[0])/(2*math.pi)
    tot = sum(abs(un[i+1]-un[i]) for i in range(len(un)-1))/(2*math.pi)
    return net, tot

def series(rows, prefix):
    return ([float(r[f"{prefix}aggressive"]) for r in rows],
            [float(r[f"{prefix}defensive"]) for r in rows])

def load(cell):
    out = []
    for sf in sorted((P2/CELLS[cell]).glob("seed*.csv")):
        out.append(list(csv.DictReader(open(sf))))
    return out

WINDOWS = [250, 500, 1000, 2000]

print("=== (A) per-seed p_ net_turns (sign consistency) — tail=1000 ===")
for cell in CELLS:
    runs = load(cell)
    nets = []
    for rows in runs:
        a, d = series(rows, "p_"); n = len(a)
        nets.append(net_winding(a[n-1000:], d[n-1000:])[0])
    same = sum(1 for x in nets if (x < 0) == (st.mean(nets) < 0))
    print(f"  {cell:<5} mean={st.mean(nets):+.2f}  per-seed={[round(x,1) for x in nets]}  same-sign={same}/{len(nets)}")

print("\n=== (B) WINDOW-LENGTH SCALING of p_ net_turns (decisive) ===")
print("    genuine rotation => |net| grows ~linearly with window; rate (turns/1000) ~const")
print(f"  {'cell':<5} | " + " ".join(f"w={w:<5}" for w in WINDOWS) + " | rate/1000 across windows")
for cell in CELLS:
    runs = load(cell)
    row = {}
    for w in WINDOWS:
        nets = []
        for rows in runs:
            a, d = series(rows, "p_"); n = len(a)
            if n >= w: nets.append(net_winding(a[n-w:], d[n-w:])[0])
        row[w] = st.mean(nets)
    rates = [row[w]/w*1000 for w in WINDOWS]
    print(f"  {cell:<5} | " + " ".join(f"{row[w]:+6.2f} " for w in WINDOWS) + f" | " + " ".join(f"{r:+.2f}" for r in rates))

print("\n=== (C) per-quarter p_ net_turns (steadiness; last 1000 in 4x250) — mean over seeds ===")
for cell in CELLS:
    runs = load(cell)
    quarts = [[] for _ in range(4)]
    for rows in runs:
        a, d = series(rows, "p_"); n = len(a)
        base = n-1000
        for q in range(4):
            seg_a = a[base+q*250: base+(q+1)*250]; seg_d = d[base+q*250: base+(q+1)*250]
            quarts[q].append(net_winding(seg_a, seg_d)[0])
    qm = [st.mean(q) for q in quarts]
    print(f"  {cell:<5} quarters={[round(x,2) for x in qm]}  (same-sign quarters: {sum(1 for x in qm if (x<0)==(st.mean(qm)<0))}/4)")

print("\n=== (D) separation: b4 vs matched null b5, vs positive ArmA (p_ rate turns/1000, tail=2000) ===")
def rate2000(cell):
    runs = load(cell); nets=[]
    for rows in runs:
        a,d=series(rows,"p_"); n=len(a)
        nets.append(net_winding(a[n-2000:], d[n-2000:])[0]/2000*1000)
    return st.mean(nets), st.pstdev(nets)
for cell in ("ArmA","b4","t","b5"):
    m,s = rate2000(cell); print(f"  {cell:<5} rate={m:+.3f} ± {s:.3f} turns/1000")
