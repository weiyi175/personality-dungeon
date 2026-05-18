#!/usr/bin/env python3
"""H8.3 F3 Threshold Characterization

SDD §H8.3 — Pure incubation + relock study: quantify the "Topological Hardness"
of the Arnold Tongue attractor via the F3 threshold.

Context:
  H8.1a/H8.2 established:
    G1 DEAD  — no external intervention accelerates AT locking (A_res is the ONLY key)
    G2 PASS  — strong hysteresis: T_unlock / T_lock_1 ≈ 0.03–0.10
    G3 AMBIGUOUS — path memory real but confounded by F3 threshold effect
    F3 PATTERN — very short T_lock_1 → slow relock (attractor not deep enough);
                 long T_lock_1 → fast relock (path memory activates past threshold)

Design:
  Single condition: c3_pure
    LOCK_PHASE (A_res, t=0) → UNLOCK_DELAY(100) → THAWING(A_herd) → RELOCK_PHASE → RELOCKED
  No herding, no ramp — pure resonance-amplitude evolution.
  T_lock_1 sampled across wide natural distribution (C1-equivalent first phase).
  n_reps=10 maximises complete cycles for F3 scatter characterisation.

Gates:
  G0  AT_SURVIVES  : ≥2/3 seeds have ≥1 rep with lock_1_reached
  G3  PATH_MEMORY  : ≥2/3 seeds (with ≥1 complete cycle) have median T_lock_2 < T_lock_1
  F3  THRESHOLD    : complete cycles split at median T_lock_1;
                     upper-half median(T_lock_2/T_lock_1) < lower-half median

Seeds:    {47, 97, 99}
A_res:    {47: 0.10, 97: 0.07, 99: 0.13}   (H8.1a validated)
n_reps:   10
n_rounds: 30,000   (30k covers T_lock_1≈20k + T_unlock≈6k + T_lock_2≈9k worst case)

Usage:
  PYTHONPATH=/home/user/personality-dungeon ./venv/bin/python -u \\
      scripts/run_w83_f3_characterize.py
  PYTHONPATH=/home/user/personality-dungeon ./venv/bin/python -u \\
      scripts/run_w83_f3_characterize.py --resume
  PYTHONPATH=/home/user/personality-dungeon ./venv/bin/python -u \\
      scripts/run_w83_f3_characterize.py --seeds 97 --reps 3 --n-rounds 5000
"""
from __future__ import annotations

import argparse
import collections
import copy as _copy
import csv as _csv
import json
import random
import statistics
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[1]

from simulation.run_simulation import SimConfig, simulate
from players.base_player import DEFAULT_PERSONALITY_KEYS

# ──────────────────────────────────────────────────────────────────────────────
# Constants (SDD §H8.3)
# ──────────────────────────────────────────────────────────────────────────────

OUTPUT_DIR = ROOT / "outputs" / "h83_f3_characterize"

SEEDS       = [47, 97, 99]
N_REPLICATES = 10
N_ROUNDS    = 30_000

A_HERD           = 0.30   # thawing amplitude
LOCK_WINDOW      = 500
LOCK_THRESHOLD   = 0.80
ESCAPE_THRESHOLD = 0.50
UNLOCK_DELAY     = 100

# Golden Point (H7.3 locked)
MU_BASE            = 0.30
LAMBDA_MU          = 0.05
LAMBDA_K           = 0.20
SELECTION_STRENGTH = 0.15
K_LOWER            = 0.05
K_UPPER            = 0.25
GAMMA_BASE         = 0.16
SYNERGY_POWER      = 3.2

# Per-seed AT config (H8.1a validated)
SEED_CONFIG: dict[int, dict[str, Any]] = {
    47: {"A_res": 0.10, "target": "balanced"},
    97: {"A_res": 0.07, "target": "balanced"},
    99: {"A_res": 0.13, "target": "defensive"},
}

G3_SEEDS_NEEDED = 2   # ≥ this many seeds must satisfy median T_lock_2 < T_lock_1


# ──────────────────────────────────────────────────────────────────────────────
# H8.3 — PureNoiseController  (state machine: no HERDING / no COOLING)
# ──────────────────────────────────────────────────────────────────────────────

class PureNoiseController:
    """Additive noise with pure A_res incubation + relock loop.

    State machine (c3_pure):
      LOCK_PHASE → UNLOCK_DELAY → THAWING → RELOCK_PHASE → RELOCKED

    Amplitude:
      THAWING  →  A_herd
      all else →  A_res

    T_lock_1 = absolute round when LOCK_WINDOW first satisfied (from t=0)
    T_unlock = rounds from THAWING start to first escape
    T_lock_2 = rounds from escape to second lock completion
    """

    def __init__(
        self,
        *,
        A_res: float,
        A_herd: float,
        target_corner: str,
        noise_base_seed: int,
    ) -> None:
        self.A_res       = A_res
        self.A_herd      = A_herd
        self._target_key = f"p_{target_corner}"
        self._noise_base = noise_base_seed

        self.state: str = "LOCK_PHASE"

        self.t_lock_1    : int | None = None
        self.t_thaw_start: int | None = None
        self.t_escape    : int | None = None
        self.t_lock_2    : int | None = None

        self.T_lock_1: int | None = None
        self.T_unlock: int | None = None
        self.T_lock_2: int | None = None

        self.lock_1_reached: bool = False
        self.escape_reached: bool = False
        self.lock_2_reached: bool = False

        self._pwin: collections.deque[float] = collections.deque(maxlen=LOCK_WINDOW)

    def _current_A(self) -> float:
        return self.A_herd if self.state == "THAWING" else self.A_res

    def _window_locked(self) -> bool:
        return (
            len(self._pwin) >= LOCK_WINDOW
            and all(v > LOCK_THRESHOLD for v in self._pwin)
        )

    def make_callback(self) -> Callable:
        target_key = self._target_key
        noise_base = self._noise_base
        keys       = list(DEFAULT_PERSONALITY_KEYS)
        ctrl       = self

        def _cb(
            t: int,
            cfg: Any,
            players: list,
            dungeon: Any,
            step_records: list,
            row: dict,
        ) -> None:
            p_target = float(row.get(target_key, 0.0))
            ctrl._pwin.append(p_target)

            if ctrl.state == "LOCK_PHASE":
                if not ctrl.lock_1_reached and ctrl._window_locked():
                    ctrl.lock_1_reached = True
                    ctrl.t_lock_1       = t
                    ctrl.T_lock_1       = t
                    ctrl.state          = "UNLOCK_DELAY"

            elif ctrl.state == "UNLOCK_DELAY":
                if ctrl.t_lock_1 is not None and t >= ctrl.t_lock_1 + UNLOCK_DELAY:
                    ctrl.state        = "THAWING"
                    ctrl.t_thaw_start = t
                    ctrl._pwin.clear()

            elif ctrl.state == "THAWING":
                if not ctrl.escape_reached and p_target < ESCAPE_THRESHOLD:
                    ctrl.escape_reached = True
                    ctrl.t_escape       = t
                    ctrl.T_unlock       = t - ctrl.t_thaw_start  # type: ignore[operator]
                    ctrl.state          = "RELOCK_PHASE"
                    ctrl._pwin.clear()

            elif ctrl.state == "RELOCK_PHASE":
                if not ctrl.lock_2_reached and ctrl._window_locked():
                    ctrl.lock_2_reached = True
                    ctrl.t_lock_2       = t
                    ctrl.T_lock_2       = t - ctrl.t_escape  # type: ignore[operator]
                    ctrl.state          = "RELOCKED"

            # RELOCKED: terminal state

            A = ctrl._current_A()
            if A <= 0.0:
                return
            n = len(players)
            for idx, player in enumerate(players):
                rng = random.Random(noise_base * 10_000_000 + t * n + idx)
                if not hasattr(player, "personality"):
                    continue
                for key in keys:
                    if key in player.personality:
                        cur = float(player.personality[key])
                        player.personality[key] = max(
                            -1.0, min(1.0, cur + rng.uniform(-A, A))
                        )

        return _cb

    def metrics(self) -> dict[str, Any]:
        return {
            "state_final"   : self.state,
            "T_lock_1"      : self.T_lock_1,
            "T_unlock"      : self.T_unlock,
            "T_lock_2"      : self.T_lock_2,
            "lock_1_reached": self.lock_1_reached,
            "escape_reached": self.escape_reached,
            "lock_2_reached": self.lock_2_reached,
        }


# ──────────────────────────────────────────────────────────────────────────────
# CSV helper
# ──────────────────────────────────────────────────────────────────────────────

def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = _csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


# ──────────────────────────────────────────────────────────────────────────────
# Core run
# ──────────────────────────────────────────────────────────────────────────────

def run_one(
    *,
    seed: int,
    rep: int,
    out_dir: Path,
    resume: bool = False,
    n_rounds: int = N_ROUNDS,
    seed_cfg: dict | None = None,
) -> dict[str, Any]:
    if seed_cfg is None:
        seed_cfg = SEED_CONFIG
    scfg   = seed_cfg[seed]
    A_res  = float(scfg["A_res"])
    target = str(scfg["target"])

    sim_seed   = seed * 100 + rep
    noise_seed = sim_seed + 500_000

    subdir = out_dir / "c3_pure"
    subdir.mkdir(parents=True, exist_ok=True)
    out_csv = subdir / f"seed{seed}_rep{rep}.csv"

    ctrl   = PureNoiseController(
        A_res           = A_res,
        A_herd          = A_HERD,
        target_corner   = target,
        noise_base_seed = noise_seed,
    )
    rnd_cb = ctrl.make_callback()

    if resume and out_csv.exists():
        with out_csv.open(newline="") as f:
            stored_rows = list(_csv.DictReader(f))
        for t, row in enumerate(stored_rows):
            rnd_cb(t, None, [], None, [], row)
    else:
        cfg = SimConfig(
            n_players                      = 300,
            n_rounds                       = n_rounds,
            seed                           = sim_seed,
            payoff_mode                    = "matrix_ab",
            popularity_mode                = "sampled",
            gamma                          = GAMMA_BASE,
            epsilon                        = 0.0,
            a                              = 1.0,
            b                              = 0.9,
            matrix_cross_coupling          = 0.20,
            init_bias                      = 0.5,
            evolution_mode                 = "personality_coupled",
            payoff_lag                     = 1,
            selection_strength             = float(SELECTION_STRENGTH),
            enable_events                  = False,
            events_json                    = None,
            out_csv                        = out_csv,
            memory_kernel                  = 1,
            synergy_type                   = "nonlinear",
            synergy_gamma                  = float(GAMMA_BASE),
            synergy_nonlinear_type         = "power",
            synergy_nonlinear_power        = float(SYNERGY_POWER),
            personality_coupling_mu_base   = float(MU_BASE),
            personality_coupling_lambda_mu = float(LAMBDA_MU),
            personality_coupling_lambda_k  = float(LAMBDA_K),
            personality_coupling_mu_lower  = 0.0,
            personality_coupling_mu_upper  = 0.60,
            personality_coupling_k_lower   = float(K_LOWER),
            personality_coupling_k_upper   = float(K_UPPER),
            synergy_pulse_t_start          = None,
            synergy_pulse_duration         = None,
            synergy_pulse_delta_gamma      = 0.0,
        )
        _strategy_space, rows = simulate(cfg, round_callback=rnd_cb)
        _write_csv(out_csv, rows)

    m = ctrl.metrics()
    m.update({
        "seed"    : seed,
        "rep"     : rep,
        "A_res"   : A_res,
        "target"  : target,
        "sim_seed": sim_seed,
        "csv_path": str(out_csv.relative_to(ROOT)),
    })
    return m


# ──────────────────────────────────────────────────────────────────────────────
# Gate analysis
# ──────────────────────────────────────────────────────────────────────────────

def _median(vals: list) -> float | None:
    clean = [v for v in vals if v is not None]
    return statistics.median(clean) if clean else None


def compute_gates(
    all_metrics: list[dict[str, Any]],
    seeds: list[int],
) -> dict[str, Any]:
    # ── G0 ────────────────────────────────────────────────────────────────────
    g0_detail: dict[str, Any] = {}
    g0_pass_seeds: list[int]  = []
    for s in seeds:
        runs = [m for m in all_metrics if m["seed"] == s]
        lock_count = sum(1 for r in runs if r["lock_1_reached"])
        med_T1     = _median([r["T_lock_1"] for r in runs if r["lock_1_reached"]])
        seed_pass  = lock_count > 0
        g0_detail[str(s)] = {
            "lock_count": lock_count,
            "n_reps"    : len(runs),
            "med_T_lock_1": med_T1,
            "passes"    : seed_pass,
        }
        if seed_pass:
            g0_pass_seeds.append(s)
    g0_pass = len(g0_pass_seeds) >= 2

    # ── G3 ────────────────────────────────────────────────────────────────────
    g3_detail: dict[str, Any] = {}
    g3_pass_seeds: list[int]  = []
    for s in seeds:
        complete = [m for m in all_metrics if m["seed"] == s and m["lock_2_reached"]]
        med_1 = _median([r["T_lock_1"] for r in complete])
        med_2 = _median([r["T_lock_2"] for r in complete])
        seed_pass = (
            med_1 is not None and med_2 is not None and med_2 < med_1
        )
        g3_detail[str(s)] = {
            "n_complete"  : len(complete),
            "med_T_lock_1": med_1,
            "med_T_lock_2": med_2,
            "med_ratio"   : round(med_2 / med_1, 3) if med_1 and med_2 else None,
            "passes"      : seed_pass,
        }
        if seed_pass:
            g3_pass_seeds.append(s)
    g3_pass = len(g3_pass_seeds) >= G3_SEEDS_NEEDED

    # ── F3: threshold characterisation ────────────────────────────────────────
    complete_all = [m for m in all_metrics if m["lock_2_reached"]]
    f3_detail: dict[str, Any] = {"n_complete": len(complete_all)}
    f3_pass = False

    if len(complete_all) >= 4:
        t1_vals  = sorted(r["T_lock_1"] for r in complete_all)
        split    = statistics.median(t1_vals)
        lower    = [r["T_lock_2"] / r["T_lock_1"]
                    for r in complete_all if r["T_lock_1"] <= split]
        upper    = [r["T_lock_2"] / r["T_lock_1"]
                    for r in complete_all if r["T_lock_1"] > split]
        med_lower = _median(lower)
        med_upper = _median(upper)
        f3_pass   = (
            med_lower is not None and med_upper is not None
            and med_upper < med_lower
        )
        f3_detail.update({
            "split_T_lock_1"        : split,
            "n_lower"               : len(lower),
            "n_upper"               : len(upper),
            "med_ratio_lower_half"  : round(med_lower, 3) if med_lower else None,
            "med_ratio_upper_half"  : round(med_upper, 3) if med_upper else None,
            "passes"                : f3_pass,
        })
    else:
        f3_detail["passes"] = False
        f3_detail["note"]   = f"insufficient complete cycles ({len(complete_all)} < 4)"

    # ── Verdict ───────────────────────────────────────────────────────────────
    if not g0_pass:
        verdict = "G0_FAIL"
    elif g3_pass and f3_pass:
        verdict = "PASS"
    elif g3_pass or f3_pass:
        verdict = "PARTIAL"
    else:
        verdict = "FAIL"

    return {
        "gates": {
            "g0_at_survives": g0_pass,
            "g3_path_memory": g3_pass,
            "f3_threshold"  : f3_pass,
        },
        "verdict"         : verdict,
        "g0_passing_seeds": g0_pass_seeds,
        "g0_detail"       : g0_detail,
        "g3_detail"       : g3_detail,
        "f3_detail"       : f3_detail,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Scatter plot
# ──────────────────────────────────────────────────────────────────────────────

def _save_scatter(all_metrics: list[dict], out_dir: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        return

    complete = [m for m in all_metrics if m["lock_2_reached"]]
    escape   = [m for m in all_metrics if m["lock_1_reached"] and m["escape_reached"]]
    if not complete:
        return

    seed_colors  = {47: "#1f77b4", 97: "#ff7f0e", 99: "#2ca02c"}
    seed_markers = {47: "o", 97: "s", 99: "^"}

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # ── Left: T_lock_1 vs T_lock_2 (G3/F3) ───────────────────────────────────
    ax = axes[0]
    seen = set()
    for r in complete:
        s   = r["seed"]
        lbl = f"seed{s}" if s not in seen else ""
        seen.add(s)
        ax.scatter(r["T_lock_1"], r["T_lock_2"],
                   color=seed_colors[s], marker=seed_markers[s], s=80, zorder=3,
                   label=lbl)
        ax.annotate(f"r{r['rep']}", (r["T_lock_1"], r["T_lock_2"]),
                    textcoords="offset points", xytext=(5, 3),
                    fontsize=6.5, color=seed_colors[s])

    xmax = max(r["T_lock_1"] for r in complete) * 1.15
    ax.plot([0, xmax], [0, xmax], "k--", alpha=0.4, label="T₂=T₁ (no memory)")
    ax.fill_between([0, xmax], [0, 0], [0, xmax], alpha=0.07, color="green")

    # F3 split line
    t1_all = [r["T_lock_1"] for r in complete]
    if len(t1_all) >= 4:
        split = statistics.median(t1_all)
        ax.axvline(split, color="red", linestyle=":", alpha=0.6,
                   label=f"F3 split T₁={split:.0f}")

    handles, labels = ax.get_legend_handles_labels()
    ax.legend(dict(zip(labels, handles)).values(),
              dict(zip(labels, handles)).keys(), fontsize=8)
    ax.set_xlabel("T_lock_1 (rounds)")
    ax.set_ylabel("T_lock_2 (rounds)")
    ax.set_title("F3 Threshold: T_lock_1 vs T_lock_2\n(below diagonal = path memory ✓)")
    ax.grid(alpha=0.3)

    # ── Right: T_lock_1 vs ratio (F3 threshold curve) ─────────────────────────
    ax2 = axes[1]
    seen2 = set()
    for r in complete:
        s     = r["seed"]
        ratio = r["T_lock_2"] / r["T_lock_1"]
        lbl   = f"seed{s}" if s not in seen2 else ""
        seen2.add(s)
        ax2.scatter(r["T_lock_1"], ratio,
                    color=seed_colors[s], marker=seed_markers[s], s=80, zorder=3,
                    label=lbl)

    ax2.axhline(1.0, color="k", linestyle="--", alpha=0.4, label="ratio=1 (no change)")
    ax2.fill_between([0, max(r["T_lock_1"] for r in complete) * 1.15],
                     0, 1, alpha=0.07, color="green", label="faster relock zone")

    if len(t1_all) >= 4:
        ax2.axvline(split, color="red", linestyle=":", alpha=0.6,
                    label=f"F3 split T₁={split:.0f}")

    # log-scale y if range is large
    ratios = [r["T_lock_2"] / r["T_lock_1"] for r in complete]
    if max(ratios) / (min(ratios) + 1e-9) > 10:
        ax2.set_yscale("log")

    handles2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(dict(zip(labels2, handles2)).values(),
               dict(zip(labels2, handles2)).keys(), fontsize=8)
    ax2.set_xlabel("T_lock_1 (rounds)")
    ax2.set_ylabel("T_lock_2 / T_lock_1 (ratio)")
    ax2.set_title("F3 Threshold Curve\n(ratio < 1 → path memory, ratio > 1 → F3 region)")
    ax2.grid(alpha=0.3)

    fig.suptitle("H8.3 F3 Threshold Characterization (c3_pure, n_reps=10)",
                 fontsize=12, fontweight="bold")
    out = out_dir / "h83_f3_scatter.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Scatter plot → {out.relative_to(ROOT)}")


# ──────────────────────────────────────────────────────────────────────────────
# Progress display
# ──────────────────────────────────────────────────────────────────────────────

def _fmt(m: dict[str, Any]) -> str:
    parts = []
    if m["lock_1_reached"]:
        parts.append(f"T₁={m['T_lock_1']}")
        if m["escape_reached"]:
            parts.append(f"Tu={m['T_unlock']}")
            if m["lock_2_reached"]:
                ratio = m["T_lock_2"] / m["T_lock_1"]
                parts.append(f"T₂={m['T_lock_2']} ({ratio:.2f}×)")
    else:
        parts.append("no_lock")
    return "  " + "  ".join(parts)


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="H8.3 F3 Threshold Characterization",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--resume", action="store_true",
                        help="replay existing CSVs without re-simulating")
    parser.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    parser.add_argument("--reps", type=int, default=N_REPLICATES,
                        help=f"replicates per seed (default: {N_REPLICATES})")
    parser.add_argument("--n-rounds", type=int, default=N_ROUNDS,
                        help=f"simulation length (default: {N_ROUNDS})")
    parser.add_argument("--a-res-override", nargs=2, metavar=("SEED", "A_RES"),
                        action="append", default=[],
                        help="override A_res for a seed, e.g. --a-res-override 97 0.07")
    args = parser.parse_args()

    seeds    : list[int] = args.seeds
    n_reps   : int       = args.reps
    n_rounds : int       = args.n_rounds

    seed_cfg = _copy.deepcopy(SEED_CONFIG)
    for seed_str, a_res_str in args.a_res_override:
        sid = int(seed_str)
        if sid not in seed_cfg:
            raise ValueError(f"--a-res-override: unknown seed {sid}")
        seed_cfg[sid]["A_res"] = float(a_res_str)
        print(f"[override] seed {sid}: A_res → {float(a_res_str)}")

    total   = len(seeds) * n_reps
    out_dir = OUTPUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / "summary.json"

    print(
        f"H8.3 F3 Threshold — {total} runs  "
        f"(c3_pure  seeds={seeds}  reps={n_reps}  n_rounds={n_rounds})"
    )

    all_metrics: list[dict[str, Any]] = []
    done = 0

    for seed in seeds:
        for rep in range(n_reps):
            done += 1
            print(f"[{done:>3}/{total}] seed={seed}  rep={rep} …", end=" ", flush=True)
            m = run_one(
                seed     = seed,
                rep      = rep,
                out_dir  = out_dir,
                resume   = args.resume,
                n_rounds = n_rounds,
                seed_cfg = seed_cfg,
            )
            all_metrics.append(m)
            print(_fmt(m), flush=True)

    # ── Gate analysis ─────────────────────────────────────────────────────────
    gate_result = compute_gates(all_metrics, seeds)
    complete_all = [m for m in all_metrics if m["lock_2_reached"]]

    summary = {
        **gate_result,
        "n_runs"  : len(all_metrics),
        "n_complete_cycles": len(complete_all),
        "config"  : {
            "seeds"           : seeds,
            "seed_config"     : {str(k): v for k, v in seed_cfg.items()},
            "n_replicates"    : n_reps,
            "n_rounds"        : n_rounds,
            "A_herd"          : A_HERD,
            "lock_window"     : LOCK_WINDOW,
            "lock_threshold"  : LOCK_THRESHOLD,
            "escape_threshold": ESCAPE_THRESHOLD,
            "unlock_delay"    : UNLOCK_DELAY,
        },
        "runs": all_metrics,
    }
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False))

    # ── Scatter plot ──────────────────────────────────────────────────────────
    _save_scatter(all_metrics, out_dir)

    # ── Print summary ─────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"H8.3 F3 Threshold — verdict: {gate_result['verdict']}")
    print(f"{'='*60}")
    for key, label in [
        ("g0_at_survives", "G0  AT_SURVIVES"),
        ("g3_path_memory", "G3  PATH_MEMORY"),
        ("f3_threshold"  , "F3  THRESHOLD"),
    ]:
        v = gate_result["gates"][key]
        print(f"  {'✓' if v else '✗'} {label}")

    print(f"\n  Complete cycles: {len(complete_all)}/{total}")
    print("\n  G3 detail:")
    for s, det in gate_result["g3_detail"].items():
        ratio_str = f"{det['med_ratio']:.3f}" if det["med_ratio"] is not None else "–"
        print(f"    seed{s}: {det['n_complete']} complete  "
              f"T₁={det['med_T_lock_1']}  T₂={det['med_T_lock_2']}  ratio={ratio_str}")

    print("\n  F3 detail:")
    fd = gate_result["f3_detail"]
    if "split_T_lock_1" in fd:
        print(f"    split at T₁={fd['split_T_lock_1']:.0f}")
        print(f"    lower half ({fd['n_lower']}): med ratio = {fd['med_ratio_lower_half']}")
        print(f"    upper half ({fd['n_upper']}): med ratio = {fd['med_ratio_upper_half']}")
    else:
        print(f"    {fd.get('note', '–')}")

    print(f"\n  Summary → {summary_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
