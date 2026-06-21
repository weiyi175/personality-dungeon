"""Consolidate all metric-sensitivity findings into supplementary data JSON."""
import csv, sys, math, json, statistics as st
from pathlib import Path
ROOT = Path("/home/user/personality-dungeon"); sys.path.insert(0, str(ROOT))
from analysis.cycle_metrics import classify_cycle_level

P2 = ROOT / "reports/experiments/l3_bottleneck/phase2"
OUT = ROOT / "reports/experiments/l3_bottleneck/metric_sensitivity"
OUT.mkdir(parents=True, exist_ok=True)
CELLS = {
    "ArmA_meanfield": "b5_scout/g1_mean_field_delta0p000",
    "b4_state_k":     "b4_scout/beta0p60_k0p08",
    "t_minibatch":    "t_scout/t_lattice4_minibatch",
    "b5_keystone":    "b5_scout/g2_sampled_delta0p000",
    "c1_smallworld":  "c1_scout/g2_c1_small_world_b10p0_m0p5",
}

def load(cell): return [list(csv.DictReader(open(sf))) for sf in sorted((P2/CELLS[cell]).glob("seed*.csv"))]
def has(rows, pfx): return f"{pfx}aggressive" in rows[0].keys()
def ser(rows, pfx): return ([float(r[f"{pfx}aggressive"]) for r in rows], [float(r[f"{pfx}defensive"]) for r in rows])

def classify(rows, pfx):
    if not has(rows, pfx): return None
    s = {k: [float(r[f"{pfx}{k}"]) for r in rows] for k in ("aggressive","defensive","balanced")}
    c = classify_cycle_level(s, burn_in=1000, tail=1000, amplitude_threshold=0.02, corr_threshold=0.09,
                             eta=0.0, min_turn_strength=0.0, stage3_method="turning", phase_smoothing=1)
    if c.stage3 is None: return {"level_at_eta55": int(c.level), "consistency": 0.0, "turn_strength": 0.0}
    cons = float(c.stage3.score)
    return {"consistency": cons, "turn_strength": float(c.stage3.turn_strength),
            "L3_at_eta55_mts0": int(cons >= 0.55)}

def winding(a, d):
    ma, md = sum(a)/len(a), sum(d)/len(d)
    ang = [math.atan2(dd-md, aa-ma) for aa, dd in zip(a, d)]
    un = [ang[0]]
    for x in ang[1:]:
        cand = x
        while cand-un[-1] > math.pi: cand -= 2*math.pi
        while cand-un[-1] < -math.pi: cand += 2*math.pi
        un.append(cand)
    net = (un[-1]-un[0])/(2*math.pi); tot = sum(abs(un[i+1]-un[i]) for i in range(len(un)-1))/(2*math.pi)
    return net, (abs(net)/tot if tot else 0.0)

data = {"description": "L3-BN metric-sensitivity: turning-consistency vs winding-number; p_ vs w_.",
        "protocol2_chain": "burn_in=1000,tail=1000,eta=0.55,min_turn_strength=0,stage3=turning,phase_smoothing=1",
        "cells": {}}
for cell in CELLS:
    runs = load(cell); rec = {"n": len(runs)}
    for pfx, key in (("p_","p"), ("w_","w")):
        cl = [classify(r, pfx) for r in runs]
        if cl[0] is None: rec[key] = "no_columns"; continue
        cons = [c["consistency"] for c in cl]; ts = [c["turn_strength"] for c in cl]
        l3 = sum(c.get("L3_at_eta55_mts0", 0) for c in cl)
        def _tail_wind(r):
            a, d = ser(r, pfx); n = len(a); return winding(a[n-1000:], d[n-1000:])
        winds = [_tail_wind(r) if has(r, pfx) else (0,0) for r in runs]
        # window scaling on p_/w_
        scale = {}
        for w in (250,500,1000,2000):
            nets=[]
            for r in runs:
                a,d = ser(r,pfx); n=len(a)
                if n>=w: nets.append(winding(a[n-w:], d[n-w:])[0])
            scale[w] = round(st.mean(nets),3)
        rec[key] = {
            "L3_count_turning_eta55": f"{l3}/{len(runs)}",
            "consistency_mean": round(st.mean(cons),4),
            "turn_strength_mean": st.mean(ts),
            "winding_net_turns_mean": round(st.mean(n for n,_ in winds),3),
            "winding_net_turns_perseed": [round(n,2) for n,_ in winds],
            "winding_coherence_mean": round(st.mean(c for _,c in winds),3),
            "winding_net_by_window": scale,
        }
    data["cells"][cell] = rec

(OUT/"metric_sensitivity_data.json").write_text(json.dumps(data, indent=2))
print("wrote", OUT/"metric_sensitivity_data.json")
print(json.dumps({c: {"p_L3": data["cells"][c].get("p",{}).get("L3_count_turning_eta55") if isinstance(data["cells"][c].get("p"),dict) else data["cells"][c].get("p"),
                      "p_wind": data["cells"][c].get("p",{}).get("winding_net_turns_mean") if isinstance(data["cells"][c].get("p"),dict) else None,
                      "w_L3": data["cells"][c].get("w",{}).get("L3_count_turning_eta55") if isinstance(data["cells"][c].get("w"),dict) else data["cells"][c].get("w")}
                  for c in CELLS}, indent=2))
