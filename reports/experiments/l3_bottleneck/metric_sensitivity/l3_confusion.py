"""Phase-1 winding re-scan -> turning-vs-winding confusion matrix (observable p_ series).

Both metrics on the SAME series (p_) so the matrix isolates the METRIC, not the series.
- Turning verdict: classify_cycle_level protocol-2 chain (eta=0.55, min_turn=0) -> L3 == level 3.
- Winding verdict (per-seed steady rotation): tail-1000 split into 4 quarters (250 ea);
  winding+ iff 4/4 quarters same sign AND min|quarter net turns| >= 0.30
  (=> a non-trivial, steady rotation present in EVERY sub-window; calibrated on anchors:
   ArmA min|q|=0.66, b4 min|q|=0.70 pass; t 3/4 & b5 2/4 fail).

Critical quadrant = Turning- / Winding+ : rotation the step-local metric misses.
"""
import csv, sys, math, statistics as st
from pathlib import Path
from collections import defaultdict
ROOT = Path("/home/user/personality-dungeon"); sys.path.insert(0, str(ROOT))
from analysis.cycle_metrics import classify_cycle_level

# Phase-1 L3-BN families (B2/B3/B4/B5/C1/C2/T) — the ~120-run population
SCOUT_DIRS = [
    ("B2", "outputs/b2_island_deme_short_scout"),
    ("B3", "outputs/b3_stratified_growth_short_scout"),
    ("B3", "outputs/b3_personality_strata_short_scout"),
    ("B3", "outputs/b3_phase_strata_short_scout"),
    ("B4", "outputs/b4_state_k_short_scout"),
    ("B5", "outputs/b5_tangential_drift_short_scout"),
    ("C1", "outputs/c1_pairwise_scout"),
    ("C1", "outputs/c1_local_pairwise_short_scout"),
    ("C2", "outputs/c2_local_minibatch_short_scout"),
    ("T",  "outputs/t_series_short_scout"),
]
FLOOR = 0.30  # min |net turns| per quarter

def pseries(rows):
    return ([float(r["p_aggressive"]) for r in rows], [float(r["p_defensive"]) for r in rows])

def net_winding(a, d):
    ma, md = sum(a)/len(a), sum(d)/len(d)
    ang = [math.atan2(dd-md, aa-ma) for aa, dd in zip(a, d)]
    un=[ang[0]]
    for x in ang[1:]:
        c=x
        while c-un[-1]>math.pi: c-=2*math.pi
        while c-un[-1]<-math.pi: c+=2*math.pi
        un.append(c)
    return (un[-1]-un[0])/(2*math.pi)

def turning_L3(rows):
    s={k:[float(r[f"p_{k}"]) for r in rows] for k in ("aggressive","defensive","balanced")}
    c=classify_cycle_level(s, burn_in=1000, tail=1000, amplitude_threshold=0.02, corr_threshold=0.09,
                           eta=0.55, min_turn_strength=0.0, stage3_method="turning", phase_smoothing=1)
    return int(c.level)==3

def winding_pos(rows):
    a,d=pseries(rows); n=len(a); base=n-1000
    if base<0: base=0
    qs=[net_winding(a[base+q*250:base+(q+1)*250], d[base+q*250:base+(q+1)*250]) for q in range(4)]
    same = all(q>0 for q in qs) or all(q<0 for q in qs)
    return (same and min(abs(q) for q in qs)>=FLOOR), qs

confusion = defaultdict(int)   # (turning, winding) -> seed count
quad_TmWp = []                 # Turning- / Winding+ cells
by_family = defaultdict(lambda: defaultdict(int))
ncells=0; nseeds=0

for fam, d in SCOUT_DIRS:
    dd=ROOT/d
    if not dd.exists(): continue
    for cond in sorted([p for p in dd.iterdir() if p.is_dir()]):
        seeds=sorted(cond.glob("seed*.csv"))
        if not seeds: continue
        ncells+=1
        cell_tw=[]
        for sf in seeds:
            rows=list(csv.DictReader(open(sf)))
            if "p_aggressive" not in rows[0]: continue
            tL3=turning_L3(rows); wPos,qs=winding_pos(rows)
            confusion[(tL3,wPos)]+=1; nseeds+=1
            by_family[fam][(tL3,wPos)]+=1
            cell_tw.append((tL3,wPos,qs,sf.name))
            if (not tL3) and wPos:
                quad_TmWp.append(f"{fam} {cond.name}/{sf.name} q={[round(x,2) for x in qs]}")

print(f"scanned {ncells} cells, {nseeds} seeds (Phase-1, p_ series)\n")
print("=== CONFUSION MATRIX (seed-level, p_) ===")
print(f"                Winding+    Winding-")
print(f"  Turning+    {confusion[(True,True)]:>8}   {confusion[(True,False)]:>8}")
print(f"  Turning-    {confusion[(False,True)]:>8}   {confusion[(False,False)]:>8}")
print(f"\n  CRITICAL Turning-/Winding+ = {confusion[(False,True)]} seeds")
for q in quad_TmWp: print("    -", q)
print("\n=== by family (T+/W+ , T+/W- , T-/W+ , T-/W-) ===")
for fam in ["B2","B3","B4","B5","C1","C2","T"]:
    f=by_family[fam]
    print(f"  {fam:<3} {f[(True,True)]:>3} {f[(True,False)]:>3} {f[(False,True)]:>3} {f[(False,False)]:>3}")
