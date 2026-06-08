#!/usr/bin/env python
"""P7-H: complete the live "200 run" over HTTP through the CORRECTED apparatus.

Stands in for the Godot client and drives a 4-agent RL session for n_rounds=200
against the FastAPI backend, but routes proximity / events through the validated
Space-A apparatus instead of computing them on the raw Space-B display scale
(the bypass that saturated the DV at 1.0).

Per round it records two proximities for direct contrast:
  - corrected : POST /bifurcation/b/detect  (Space-B in → b_to_a → Space-A DV)
  - naive_bug : POST /bifurcation/detect     (Space-B treated AS Space-A → saturates)

Events are injected every --event-every rounds via the authoritative, persisting
Space-A path POST /rl_sessions/{id}/apply-event (group=experiment, v1-aligned).

Primary DV = max corrected proximity over the session (P7-H H1).

Outputs:
  reports/experiments/p7h_live200_http/p7h_live200_trajectory.json
  reports/experiments/p7h_live200_http/p7h_live200_summary.txt
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _post(base: str, path: str, payload: dict) -> dict:
    data = json.dumps(payload).encode()
    req = urllib.request.Request(
        base + path, data=data, headers={"Content-Type": "application/json"}, method="POST"
    )
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read().decode())


def _get(base: str, path: str) -> dict:
    with urllib.request.urlopen(base + path, timeout=30) as r:
        return json.loads(r.read().decode())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:8000")
    ap.add_argument("--n-players", type=int, default=4)      # live "4-agent" config
    ap.add_argument("--n-rounds", type=int, default=200)     # the "200 run"
    ap.add_argument("--burn-in", type=int, default=0)
    ap.add_argument("--event-every", type=int, default=10)
    ap.add_argument("--intensity-scale", type=float, default=1.0)
    ap.add_argument("--personality-mode", default="random_9persona")
    ap.add_argument("--sub-critical-headroom", type=float, default=None,
                    help="Seed the population near baseline with proximity headroom "
                         "(0,1] — the validated H1 regime. Overrides personality_mode "
                         "to static server-side. Omit to keep the saturated live config.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="reports/experiments/p7h_live200_http")
    args = ap.parse_args()

    base = args.base.rstrip("/")

    # 0) sanity: the corrected route must be loaded (else server is stale).
    paths = _get(base, "/openapi.json")["paths"]
    if "/bifurcation/b/detect" not in paths:
        print("ERROR: /bifurcation/b/detect not in server routes — restart the backend.")
        return 2

    # 1) initialize the live session through the validated config.
    init_payload = {
        "n_players": args.n_players,
        "n_rounds": args.n_rounds,
        "burn_in": args.burn_in,
        "seed": args.seed,
        "personality_mode": args.personality_mode,
        "space_a_events_enabled": True,
    }
    if args.sub_critical_headroom is not None:
        init_payload["sub_critical_headroom"] = args.sub_critical_headroom
    init = _post(base, "/rl_sessions/initialize", init_payload)
    sid = init["session_id"]
    print(f"session {sid}  n_players={args.n_players}  n_rounds={args.n_rounds}  "
          f"mode={args.personality_mode}  event_every={args.event_every}")

    rows: list[dict] = []
    events_applied = 0
    t0 = time.time()

    for r in range(1, args.n_rounds + 1):
        _post(base, f"/rl_sessions/{sid}/step", {})                 # advance one round
        pers = _get(base, f"/rl_sessions/{sid}/personality")
        pv_b = pers["mean_personality_space_b"]                     # what Godot displays
        prox_a_direct = pers["bifurcation"]["bifurcation_proximity"]  # Space-A truth

        # corrected Godot path: Space-B in → b_to_a → Space-A proximity
        corrected = _post(base, "/bifurcation/b/detect",
                          {"personality_vector": pv_b})["bifurcation"]["bifurcation_proximity"]
        # the bypass bug: feed the Space-B vector straight into the Space-A route
        naive = _post(base, "/bifurcation/detect",
                      {"personality_vector": pv_b})["bifurcation"]["bifurcation_proximity"]

        # inject a persisting, v1-aligned Space-A event on the cadence
        if args.event_every > 0 and r % args.event_every == 0:
            _post(base, f"/rl_sessions/{sid}/apply-event",
                  {"group": "experiment", "intensity_scale": args.intensity_scale})
            events_applied += 1

        rows.append({
            "round": r,
            "prox_corrected": corrected,
            "prox_naive_bug": naive,
            "prox_a_direct": prox_a_direct,
            "displacement": pers["personality_displacement"],
        })

    elapsed = time.time() - t0

    corr = [x["prox_corrected"] for x in rows]
    naive = [x["prox_naive_bug"] for x in rows]
    disp = [x["displacement"] for x in rows]
    # consistency: b/detect (Space-B in) must equal the Space-A direct proximity.
    max_route_gap = max(abs(x["prox_corrected"] - x["prox_a_direct"]) for x in rows)

    max_prox = max(corr)                       # PRIMARY DV (H1)
    n_naive_saturated = sum(1 for v in naive if v >= 0.999)
    n_corr_saturated = sum(1 for v in corr if v >= 0.999)

    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    (out / "p7h_live200_trajectory.json").write_text(json.dumps({
        "session_id": sid,
        "config": vars(args),
        "events_applied": events_applied,
        "max_proximity_dv": max_prox,
        "max_route_gap": max_route_gap,
        "rows": rows,
    }, indent=2))

    lines = [
        "P7-H LIVE 200-RUN (HTTP, corrected apparatus)",
        "=" * 52,
        f"session        : {sid}",
        f"rounds         : {args.n_rounds}   players: {args.n_players}   "
        f"mode: {args.personality_mode}",
        f"events applied : {events_applied} (every {args.event_every}, v1-aligned, persisting)",
        f"elapsed        : {elapsed:.1f}s",
        "",
        "PROXIMITY (corrected vs the bypass bug)",
        f"  corrected  max_proximity (DV) = {max_prox:.4f}   "
        f"min={min(corr):.4f}  mean={sum(corr)/len(corr):.4f}",
        f"  naive bug  max={max(naive):.4f}  min={min(naive):.4f}  "
        f"mean={sum(naive)/len(naive):.4f}",
        f"  saturated@1.0 rounds:  corrected={n_corr_saturated}/{len(corr)}   "
        f"naive_bug={n_naive_saturated}/{len(naive)}",
        "",
        f"route consistency: max |b/detect − Space-A direct| = {max_route_gap:.2e} "
        f"({'OK' if max_route_gap < 1e-9 else 'MISMATCH'})",
        f"displacement DV  : final={disp[-1]:.5f}  max={max(disp):.5f}",
    ]
    summary = "\n".join(lines)
    (out / "p7h_live200_summary.txt").write_text(summary + "\n")
    print("\n" + summary)
    print(f"\nsaved: {out/'p7h_live200_trajectory.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
