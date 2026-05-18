#!/usr/bin/env python3
"""H8.4 Controlled-T₁ G3 Confirmation Experiment

SDD §H8.4 — Controlled incubation study to confirm G3 (path memory) and
characterize the F3 threshold via deterministic T1_target.

Background:
  H8.3 showed F3 threshold at T₁≈4800 with *natural* (random) T₁ distribution.
  G3 failed because many runs had T₁ < 4800, pulling up the global median ratio.
  H8.4 fixes T1_target deterministically and asks:
    "After exactly T1_target rounds at A_res, does T₂ < T1_target?"

Protocol per run:
  1. INCUBATING: A_res for exactly T1_target rounds (no lock detection)
  2. FORCED_ESCAPE: A_herd for ESCAPE_ROUNDS rounds (guaranteed disruption)
  3. RELOCK: A_res until lock detected; T₂ measured from escape_end

Metric:
  ratio = T₂ / T1_target
  G3_controlled: ≥2/3 conditions (T1_target × seed) show median ratio < 1.0
  F3_controlled: median ratio decreases monotonically as T1_target increases

Seeds:
  Core:        [47, 97]   (balanced, H8.1a validated)
  Exploratory: [99]  with A_res=0.07  (fixes H8.3's instability)

T1_targets: [2000, 4000, 6000, 10000]
n_reps:     5 per (seed, T1_target) → 40 core runs + 20 seed99 exploratory
n_rounds:   40000  (10000 + 500 + 25000 + 4500 buffer)

Usage:
  PYTHONPATH=/home/user/personality-dungeon ./venv/bin/python -u \\
      scripts/run_w84_controlled_t1.py
  PYTHONPATH=/home/user/personality-dungeon ./venv/bin/python -u \\
      scripts/run_w84_controlled_t1.py --seeds 47 97 --reps 5 --n-rounds 40000
  PYTHONPATH=/home/user/personality-dungeon ./venv/bin/python -u \\
      scripts/run_w84_controlled_t1.py --seeds 99 --a-res-override 0.07
"""
from __future__ import annotations

import argparse
import collections
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
# Constants
# ──────────────────────────────────────────────────────────────────────────────

OUTPUT_DIR = ROOT / "outputs" / "h84_controlled_t1"

SEEDS_CORE   = [47, 97]
SEEDS_ALL    = [47, 97, 99]
N_REPS       = 5
N_ROUNDS     = 40_000
T1_TARGETS   = [2000, 4000, 6000, 10_000]

A_HERD          = 0.30   # forced escape amplitude
ESCAPE_ROUNDS   = 500    # rounds of A_herd during FORCED_ESCAPE phase
LOCK_WINDOW     = 500
LOCK_THRESHOLD  = 0.80

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
    99: {"A_res": 0.13, "target": "defensive"},   # overridden for H8.4 exploratory
}

# seed99 exploratory: override A_res to match seed97 (fixes 6/10 no-lock in H8.3)
SEED99_A_RES_OVERRIDE = 0.07


# ──────────────────────────────────────────────────────────────────────────────
# ControlledT1Controller  (state machine: INCUBATING → FORCED_ESCAPE → RELOCK)
# ──────────────────────────────────────────────────────────────────────────────

class ControlledT1Controller:
    """Fixed-T1_target incubation + forced escape + relock measurement.

    State machine:
      INCUBATING   (t=0 … T1_target-1)   →  A_res applied
      FORCED_ESCAPE (t=T1_target … T1_target+ESCAPE_ROUNDS-1)  →  A_herd applied
      RELOCK       (t=T1_target+ESCAPE_ROUNDS …)  →  A_res applied, measuring T₂
      RELOCKED     (terminal)

    T₂ = rounds from RELOCK start to lock detection.
    ratio = T₂ / T1_target.
    """

    def __init__(
        self,
        *,
        A_res: float,
        A_herd: float,
        T1_target: int,
        escape_rounds: int,
        target_corner: str,
        noise_base_seed: int,
    ) -> None:
        self.A_res          = A_res
        self.A_herd         = A_herd
        self.T1_target      = T1_target
        self.escape_rounds  = escape_rounds
        self._target_key    = f"p_{target_corner}"
        self._noise_base    = noise_base_seed

        self.state: str     = "INCUBATING"
        self.t_escape_start : int | None = None
        self.t_relock_start : int | None = None
        self.T_lock_2       : int | None = None
        self.lock_2_reached : bool       = False
        self._pwin: collections.deque[float] = collections.deque(maxlen=LOCK_WINDOW)

    # ------------------------------------------------------------------
    def _current_A(self) -> float:
        return self.A_herd if self.state == "FORCED_ESCAPE" else self.A_res

    def _window_locked(self) -> bool:
        return (
            len(self._pwin) >= LOCK_WINDOW
            and all(v > LOCK_THRESHOLD for v in self._pwin)
        )

    # ------------------------------------------------------------------
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

            if ctrl.state == "INCUBATING":
                # Transition: at t == T1_target, switch to FORCED_ESCAPE
                if t >= ctrl.T1_target:
                    ctrl.state          = "FORCED_ESCAPE"
                    ctrl.t_escape_start = t
                    ctrl._pwin.clear()

            elif ctrl.state == "FORCED_ESCAPE":
                # Transition: after ESCAPE_ROUNDS of A_herd, enter RELOCK
                if t >= ctrl.t_escape_start + ctrl.escape_rounds:  # type: ignore[operator]
                    ctrl.state          = "RELOCK"
                    ctrl.t_relock_start = t
                    ctrl._pwin.clear()

            elif ctrl.state == "RELOCK":
                ctrl._pwin.append(p_target)
                if not ctrl.lock_2_reached and ctrl._window_locked():
                    ctrl.lock_2_reached = True
                    ctrl.T_lock_2       = t - ctrl.t_relock_start  # type: ignore[operator]
                    ctrl.state          = "RELOCKED"

            # RELOCKED: terminal — still apply noise (maintains state)

            # ── apply personality noise ─────────────────────────────────
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

    # ------------------------------------------------------------------
    def metrics(self) -> dict[str, Any]:
        ratio = (self.T_lock_2 / self.T1_target) if self.T_lock_2 is not None else None
        return {
            "state_final"   : self.state,
            "T1_target"     : self.T1_target,
            "T_lock_2"      : self.T_lock_2,
            "ratio"         : round(ratio, 4) if ratio is not None else None,
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
    T1_target: int,
    out_dir: Path,
    n_rounds: int = N_ROUNDS,
    a_res_override: float | None = None,
) -> dict[str, Any]:
    scfg   = SEED_CONFIG[seed]
    A_res  = float(a_res_override if a_res_override is not None else scfg["A_res"])
    target = str(scfg["target"])

    sim_seed   = seed * 100 + rep + T1_target   # unique per (seed, rep, T1_target)
    noise_seed = sim_seed + 700_000

    subdir  = out_dir / f"t1_{T1_target:05d}"
    subdir.mkdir(parents=True, exist_ok=True)
    out_csv = subdir / f"seed{seed}_rep{rep}.csv"

    ctrl   = ControlledT1Controller(
        A_res           = A_res,
        A_herd          = A_HERD,
        T1_target       = T1_target,
        escape_rounds   = ESCAPE_ROUNDS,
        target_corner   = target,
        noise_base_seed = noise_seed,
    )
    rnd_cb = ctrl.make_callback()

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
        "seed"         : seed,
        "rep"          : rep,
        "A_res"        : A_res,
        "target"       : target,
        "sim_seed"     : sim_seed,
        "csv_path"     : str(out_csv.relative_to(ROOT)),
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
    t1_targets: list[int],
) -> dict[str, Any]:
    """
    G3_controlled: ≥2/3 of (seed, T1_target) conditions with n≥3 runs
                   show median ratio < 1.0
    F3_controlled: median ratio strictly decreases from smallest to largest T1_target
                   (across all seeds pooled)
    """
    # ── per-cell analysis ─────────────────────────────────────────────────────
    cell_detail: dict[str, Any] = {}
    g3_pass_count = 0
    g3_total_count = 0

    for s in seeds:
        for t1 in t1_targets:
            runs = [m for m in all_metrics
                    if m["seed"] == s and m["T1_target"] == t1]
            complete = [r for r in runs if r["lock_2_reached"]]
            rats     = [r["ratio"] for r in complete if r["ratio"] is not None]
            med_r    = _median(rats)
            cell_key = f"seed{s}_T1{t1}"
            cell_detail[cell_key] = {
                "n_complete"  : len(complete),
                "n_total"     : len(runs),
                "med_ratio"   : round(med_r, 3) if med_r is not None else None,
                "passes_g3"   : (med_r is not None and med_r < 1.0),
            }
            if len(runs) >= 3:
                g3_total_count += 1
                if med_r is not None and med_r < 1.0:
                    g3_pass_count += 1

    g3_pass = (g3_total_count > 0 and
               g3_pass_count / g3_total_count >= 2 / 3)

    # ── F3_controlled: monotonic decrease across T1_targets ───────────────────
    pooled_med: list[tuple[int, float]] = []
    for t1 in t1_targets:
        runs     = [m for m in all_metrics if m["T1_target"] == t1 and m["lock_2_reached"]]
        rats     = [r["ratio"] for r in runs if r["ratio"] is not None]
        med_r    = _median(rats)
        if med_r is not None:
            pooled_med.append((t1, med_r))

    f3_monotone = False
    if len(pooled_med) >= 3:
        vals = [v for _, v in pooled_med]
        # strict monotone decrease check (allow 1 non-monotone step)
        drops = sum(1 for a, b in zip(vals, vals[1:]) if b < a)
        f3_monotone = drops >= len(vals) - 1

    f3_detail = {
        "pooled_medians": [
            {"T1_target": t1, "med_ratio": round(r, 3)} for t1, r in pooled_med
        ],
        "monotone_decreasing": f3_monotone,
        "passes": f3_monotone,
    }

    # ── Verdict ───────────────────────────────────────────────────────────────
    if not any(m["lock_2_reached"] for m in all_metrics):
        verdict = "G0_FAIL"
    elif g3_pass and f3_monotone:
        verdict = "PASS"
    elif g3_pass or f3_monotone:
        verdict = "PARTIAL"
    else:
        verdict = "FAIL"

    return {
        "gates": {
            "g3_controlled": g3_pass,
            "f3_controlled": f3_monotone,
        },
        "verdict"        : verdict,
        "g3_pass_count"  : g3_pass_count,
        "g3_total_count" : g3_total_count,
        "cell_detail"    : cell_detail,
        "f3_detail"      : f3_detail,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Scatter / line plot
# ──────────────────────────────────────────────────────────────────────────────

def _save_plots(all_metrics: list[dict], out_dir: Path, seeds: list[int]) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        return

    complete = [m for m in all_metrics if m["lock_2_reached"]]
    if not complete:
        return

    t1s   = np.array([m["T1_target"]    for m in complete], float)
    rats  = np.array([m["ratio"]        for m in complete], float)
    seed_arr = np.array([m["seed"]      for m in complete], int)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("H8.4 Controlled-T₁ — path memory confirmation", fontsize=12)

    # left: scatter T1_target vs ratio, color by seed
    ax = axes[0]
    colors = {47: "#e41a1c", 97: "#377eb8", 99: "#4daf4a"}
    for s in sorted(set(seed_arr)):
        mask = seed_arr == s
        lab  = f"seed{s} (A_res={SEED_CONFIG[s]['A_res']:.2f})"
        ax.scatter(t1s[mask], rats[mask], color=colors.get(s, "grey"),
                   label=lab, alpha=0.7, s=50)
    ax.axhline(1.0, color="k", lw=1, ls="--", label="ratio=1 (T₂=T₁)")
    ax.axvline(4800, color="grey", lw=1, ls=":", alpha=0.7, label="H8.3 F3≈4800")
    ax.set_xlabel("T1_target (fixed incubation rounds)")
    ax.set_ylabel("ratio = T₂ / T1_target")
    ax.set_title("Scatter: ratio vs T1_target")
    ax.legend(fontsize=8)

    # right: pooled median ratio per T1_target
    ax2 = axes[1]
    t1_uniq = sorted(set(t1s.astype(int)))
    med_per_t1 = []
    for t1 in t1_uniq:
        mask = t1s == t1
        if mask.sum() > 0:
            med_per_t1.append((t1, float(np.median(rats[mask]))))
    if med_per_t1:
        xs, ys = zip(*med_per_t1)
        ax2.plot(xs, ys, "ko-", lw=2, ms=8, label="pooled median ratio")
        for x, y in zip(xs, ys):
            ax2.annotate(f"{y:.2f}", (x, y), textcoords="offset points",
                         xytext=(0, 8), ha="center", fontsize=9)
    ax2.axhline(1.0, color="k", lw=1, ls="--")
    ax2.axvline(4800, color="grey", lw=1, ls=":", alpha=0.7, label="H8.3 F3≈4800")
    ax2.set_xlabel("T1_target")
    ax2.set_ylabel("median ratio")
    ax2.set_title("Pooled median ratio vs T1_target")
    ax2.legend(fontsize=8)

    plt.tight_layout()
    out_path = out_dir / "h84_controlled_t1_plot.png"
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  Plot → {out_path}")


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="H8.4 Controlled-T₁ G3 Confirmation")
    p.add_argument("--seeds",  nargs="+", type=int, default=SEEDS_CORE,
                   help="seeds to run (default: 47 97)")
    p.add_argument("--t1-targets", nargs="+", type=int, default=T1_TARGETS,
                   help="T1_target values (default: 2000 4000 6000 10000)")
    p.add_argument("--reps",   type=int,   default=N_REPS,
                   help="replicates per cell (default: 5)")
    p.add_argument("--n-rounds", type=int, default=N_ROUNDS,
                   help="total rounds per run (default: 40000)")
    p.add_argument("--a-res-override", type=float, default=None,
                   help="override A_res for ALL seeds (useful for seed99 at 0.07)")
    p.add_argument("--include-seed99", action="store_true",
                   help="add seed99 with A_res=0.07 as exploratory condition")
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    seeds      = list(args.seeds)
    t1_targets = list(args.t1_targets)
    n_reps     = args.reps
    n_rounds   = args.n_rounds

    # Build run list
    run_list: list[dict[str, Any]] = []

    for seed in seeds:
        a_res_ov = args.a_res_override
        for t1 in t1_targets:
            for rep in range(n_reps):
                run_list.append(dict(seed=seed, rep=rep, T1_target=t1, a_res_override=a_res_ov))

    if args.include_seed99 and 99 not in seeds:
        for t1 in t1_targets:
            for rep in range(n_reps):
                run_list.append(dict(seed=99, rep=rep, T1_target=t1,
                                     a_res_override=SEED99_A_RES_OVERRIDE))

    total = len(run_list)
    print(f"H8.4 Controlled-T₁ — {total} runs  "
          f"(seeds={seeds}  T1_targets={t1_targets}  reps={n_reps}  "
          f"n_rounds={n_rounds})")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    all_metrics: list[dict] = []
    for idx, kw in enumerate(run_list, 1):
        s, rep, t1 = kw["seed"], kw["rep"], kw["T1_target"]
        print(f"[{idx:3d}/{total}] seed={s}  T1={t1:>6}  rep={rep} … ", end="", flush=True)
        m = run_one(
            seed           = s,
            rep            = rep,
            T1_target      = t1,
            out_dir        = OUTPUT_DIR,
            n_rounds       = n_rounds,
            a_res_override = kw["a_res_override"],
        )
        all_metrics.append(m)
        # progress line
        if m["lock_2_reached"]:
            print(f"  T₂={m['T_lock_2']:>6}  ratio={m['ratio']:.3f}")
        else:
            print(f"  no_relock  ({m['state_final']})")

    # ── gates ─────────────────────────────────────────────────────────────────
    all_seeds_present = sorted(set(m["seed"] for m in all_metrics))
    gates_result = compute_gates(all_metrics, all_seeds_present, t1_targets)
    g = gates_result["gates"]

    # ── save summary ──────────────────────────────────────────────────────────
    summary = {
        **gates_result,
        "n_runs"          : total,
        "n_complete_cycles": sum(1 for m in all_metrics if m["lock_2_reached"]),
        "config": {
            "seeds"         : seeds,
            "t1_targets"    : t1_targets,
            "n_reps"        : n_reps,
            "n_rounds"      : n_rounds,
            "A_herd"        : A_HERD,
            "escape_rounds" : ESCAPE_ROUNDS,
            "lock_window"   : LOCK_WINDOW,
            "lock_threshold": LOCK_THRESHOLD,
        },
        "runs": all_metrics,
    }
    summary_path = OUTPUT_DIR / "summary.json"
    with summary_path.open("w") as f:
        json.dump(summary, f, indent=2)

    # ── plot ──────────────────────────────────────────────────────────────────
    _save_plots(all_metrics, OUTPUT_DIR, all_seeds_present)

    # ── per-cell table ────────────────────────────────────────────────────────
    print()
    print("=" * 70)
    print(f"H8.4 Controlled-T₁ — verdict: {gates_result['verdict']}")
    print("=" * 70)
    sym = {True: "✓", False: "✗"}
    print(f"  {sym[g['g3_controlled']]} G3_controlled  PATH_MEMORY "
          f"({gates_result['g3_pass_count']}/{gates_result['g3_total_count']} cells)")
    print(f"  {sym[g['f3_controlled']]} F3_controlled  MONOTONE_DECREASE")
    print()
    print("  Per-cell detail  (med ratio = T₂/T1_target):")
    print(f"  {'cell':>20}  {'n_ok':>5}  {'med_ratio':>9}  G3")
    for cell, d in gates_result["cell_detail"].items():
        p = "✓" if d["passes_g3"] else "✗"
        mr = f"{d['med_ratio']:.3f}" if d["med_ratio"] is not None else "  -  "
        print(f"  {cell:>20}  {d['n_complete']:>2}/{d['n_total']:<2}  {mr:>9}  {p}")
    print()
    print("  F3 pooled medians:")
    for entry in gates_result["f3_detail"]["pooled_medians"]:
        print(f"    T1={entry['T1_target']:>6}  med_ratio={entry['med_ratio']:.3f}")
    print()
    print(f"  Summary → {summary_path}")


if __name__ == "__main__":
    main()
