#!/usr/bin/env python3
"""H8.2 Ramp-Cooling Protocol Experiment

SDD §H8.2 — Fix G1 from H8.1a by replacing trigger+hard-switch with
scheduled linear ramp-down cooling (simulated annealing approach).

Root cause of G1 FAIL in H8.1a:
  - Trigger fires on noise peak (not stable AT convergence)
  - 200-round A_herd herding period disperses the system after trigger
  - Net effect: C2 locks 1.7–2.2× SLOWER than C1

H8.2 hypothesis:
  - Gradual cooling (A_herd → A_res over T_COOL rounds) allows system to
    explore the strategy landscape early (high noise) then settle (low noise)
  - Mechanism: simulated annealing — high early exploration escapes local
    basins, then cooling allows convergence into the global AT attractor
  - Expected: C2_ramp locks ≤ 0.90 × C1_ctrl (at least 10% faster)

Conditions:
  c1_ctrl  : fixed A = A_res from t=0               [baseline, replicates H8.1a C1]
  c2_ramp  : A(t) = A_herd*(1-t/T_COOL) + A_res*(t/T_COOL), then A_res  [annealing]
  c3_ramp  : c2_ramp + after first lock → UNLOCK_DELAY → THAW → RELOCK   [path memory]

Seeds:        {47, 97, 99}
A_res:        {47: 0.10, 97: 0.07, 99: 0.13}  (H8.1a validated values)
n_rounds:     25000
n_reps:       5
T_COOL:       2000  (ramp duration; after this, A = A_res)

Personality variance snapshots recorded at each phase transition (mechanism probe).

Gates:
  G0  AT_SURVIVES       : ≥2/3 seeds lock in c1_ctrl within 25000 rounds
  G1  RAMP_ACCELERATES  : median(T_lock_c2) / median(T_lock_c1) ≤ 0.90
  G2  HYSTERESIS        : median(T_unlock) / median(T_lock_1) ≤ 0.20 (c3_ramp)
  G3  PATH_MEMORY       : ≥2/3 seeds have median T_lock_2 < T_lock_1 (c3_ramp)

Usage:
  PYTHONPATH=/home/user/personality-dungeon ./venv/bin/python -u \\
      scripts/run_w82_ramp_cooling.py
  PYTHONPATH=/home/user/personality-dungeon ./venv/bin/python -u \\
      scripts/run_w82_ramp_cooling.py --conditions c1_ctrl c2_ramp --reps 3
  PYTHONPATH=/home/user/personality-dungeon ./venv/bin/python -u \\
      scripts/run_w82_ramp_cooling.py --t-cool 1000 --seeds 47 97
"""
from __future__ import annotations

import argparse
import collections
import copy as _copy
import csv as _csv
import json
import math
import random
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[1]

from simulation.run_simulation import SimConfig, simulate
from players.base_player import DEFAULT_PERSONALITY_KEYS

# ──────────────────────────────────────────────────────────────────────────────
# Constants (SDD §H8.2 locked)
# ──────────────────────────────────────────────────────────────────────────────

OUTPUT_DIR = ROOT / "outputs" / "h82_ramp_cooling"

SEEDS = [47, 97, 99]
N_REPLICATES = 5
N_ROUNDS = 25_000

A_HERD   = 0.30    # starting amplitude for ramp (same as H8.1)
T_COOL   = 2_000   # ramp duration in rounds (A_herd → A_res linear)

LOCK_WINDOW       = 500    # rolling window for stable-lock detection
LOCK_THRESHOLD    = 0.80   # all LOCK_WINDOW values must exceed this
ESCAPE_THRESHOLD  = 0.50   # first p_target < this → escape confirmed
UNLOCK_DELAY      = 100    # rounds after first lock before THAW (c3_ramp)

# Golden Point (H7.3 locked)
MU_BASE            = 0.30
LAMBDA_MU          = 0.05
LAMBDA_K           = 0.20
SELECTION_STRENGTH = 0.15
K_LOWER            = 0.05
K_UPPER            = 0.25
GAMMA_BASE         = 0.16
SYNERGY_POWER      = 3.2

# Per-seed AT config (H8.1a validated: seed97 A_res reduced to 0.07)
SEED_CONFIG: dict[int, dict[str, Any]] = {
    47: {"A_res": 0.10, "target": "balanced"},
    97: {"A_res": 0.07, "target": "balanced"},
    99: {"A_res": 0.13, "target": "defensive"},
}

CONDITION_SUBDIRS = {
    "c1_ctrl" : "c1_ctrl",
    "c2_ramp" : "c2_ramp",
    "c3_ramp" : "c3_ramp",
}

# Gate thresholds
G1_RAMP_THRESHOLD  = 0.90   # ratio ≤ this → ramp accelerates locking
G2_HYSTER_RATIO    = 0.20   # T_unlock / T_lock_1 ≤ this
G3_SEEDS_NEEDED    = 2      # ≥ this many seeds need T_lock_2 < T_lock_1


# ──────────────────────────────────────────────────────────────────────────────
# H8.2 — RampNoiseController
# ──────────────────────────────────────────────────────────────────────────────

class RampNoiseController:
    """Additive per-round noise with linear ramp-down cooling protocol.

    State machines:
      c1_ctrl : [LOCK_PHASE]       constant A_res from t=0
      c2_ramp : COOLING → LOCK_PHASE
      c3_ramp : COOLING → LOCK_PHASE → UNLOCK_DELAY → THAWING → RELOCK_PHASE → RELOCKED

    Amplitude by state:
      COOLING   →  A(t) = A_herd*(1 - t/T_COOL) + A_res*(t/T_COOL)   (linear ramp)
      THAWING   →  A_herd   (instant reheat)
      all else  →  A_res

    T_lock_1 is measured from t=0 for all conditions (enables fair G1 comparison).

    Personality variance snapshots are recorded at each state transition
    (mechanism probe: does cooling concentrate or spread personalities?).
    """

    def __init__(
        self,
        *,
        condition: str,
        A_res: float,
        A_herd: float,
        t_cool: int,
        target_corner: str,
        noise_base_seed: int,
    ) -> None:
        self.condition   = condition
        self.A_res       = A_res
        self.A_herd      = A_herd
        self.t_cool      = t_cool
        self._target_key = f"p_{target_corner}"
        self._noise_base = noise_base_seed

        # Initial state
        self.state: str = "LOCK_PHASE" if condition == "c1_ctrl" else "COOLING"

        # Timestamps (absolute rounds)
        self.t_lock_1    : int | None = None
        self.t_thaw_start: int | None = None
        self.t_escape    : int | None = None
        self.t_lock_2    : int | None = None

        # Timing metrics (gate analysis)
        self.T_lock_1: int | None = None  # rounds from t=0 to first lock
        self.T_unlock: int | None = None  # rounds from thaw_start to escape
        self.T_lock_2: int | None = None  # rounds from escape to second lock

        # Status flags
        self.lock_1_reached: bool = False
        self.escape_reached: bool = False
        self.lock_2_reached: bool = False

        # Rolling lock-detection window
        self._pwin: collections.deque[float] = collections.deque(maxlen=LOCK_WINDOW)

        # Personality variance snapshots at phase transitions
        # Each entry: {"state_entered": str, "t": int, "pers_var": float, "pers_mean": float}
        self._pers_snaps: list[dict[str, Any]] = []

    # ── helpers ───────────────────────────────────────────────────────────────

    def _current_A(self, t: int) -> float:
        if self.state == "COOLING":
            frac = min(1.0, t / self.t_cool) if self.t_cool > 0 else 1.0
            return self.A_herd * (1.0 - frac) + self.A_res * frac
        elif self.state == "THAWING":
            return self.A_herd
        else:
            return self.A_res

    def _window_locked(self) -> bool:
        return (
            len(self._pwin) >= LOCK_WINDOW
            and all(v > LOCK_THRESHOLD for v in self._pwin)
        )

    @staticmethod
    def _pers_stats(players: list) -> tuple[float, float]:
        """Return (population variance, mean) of per-player mean personality."""
        if not players:
            return 0.0, 0.0
        vals = []
        for p in players:
            if hasattr(p, "personality") and p.personality:
                v = list(p.personality.values())
                vals.append(sum(v) / len(v))
        if not vals:
            return 0.0, 0.0
        mean = sum(vals) / len(vals)
        var  = sum((x - mean) ** 2 for x in vals) / len(vals)
        return var, mean

    def _snap(self, state_name: str, t: int, players: list) -> None:
        var, mean = self._pers_stats(players)
        self._pers_snaps.append({
            "state_entered": state_name,
            "t"            : t,
            "pers_var"     : round(var, 6),
            "pers_mean"    : round(mean, 6),
        })

    # ── callback ──────────────────────────────────────────────────────────────

    def make_callback(self) -> Callable:
        """Return a RoundCallback compatible with simulate(round_callback=...)."""
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

            # ── State machine ─────────────────────────────────────────────────
            if ctrl.state == "COOLING":
                if t >= ctrl.t_cool:
                    ctrl.state = "LOCK_PHASE"
                    ctrl._pwin.clear()  # fresh window after ramp ends
                    ctrl._snap("LOCK_PHASE", t, players)

            elif ctrl.state == "LOCK_PHASE":
                if not ctrl.lock_1_reached and ctrl._window_locked():
                    ctrl.lock_1_reached = True
                    ctrl.t_lock_1       = t
                    ctrl.T_lock_1       = t  # from t=0 for all conditions
                    if ctrl.condition == "c3_ramp":
                        ctrl.state = "UNLOCK_DELAY"
                        ctrl._snap("UNLOCK_DELAY", t, players)

            elif ctrl.state == "UNLOCK_DELAY":
                if ctrl.t_lock_1 is not None and t >= ctrl.t_lock_1 + UNLOCK_DELAY:
                    ctrl.state        = "THAWING"
                    ctrl.t_thaw_start = t
                    ctrl._pwin.clear()
                    ctrl._snap("THAWING", t, players)

            elif ctrl.state == "THAWING":
                if not ctrl.escape_reached and p_target < ESCAPE_THRESHOLD:
                    ctrl.escape_reached = True
                    ctrl.t_escape       = t
                    ctrl.T_unlock       = t - ctrl.t_thaw_start  # type: ignore[operator]
                    ctrl.state          = "RELOCK_PHASE"
                    ctrl._pwin.clear()
                    ctrl._snap("RELOCK_PHASE", t, players)

            elif ctrl.state == "RELOCK_PHASE":
                if not ctrl.lock_2_reached and ctrl._window_locked():
                    ctrl.lock_2_reached = True
                    ctrl.t_lock_2       = t
                    ctrl.T_lock_2       = t - ctrl.t_escape  # type: ignore[operator]
                    ctrl.state          = "RELOCKED"

            # RELOCKED: terminal state

            # ── Noise injection ───────────────────────────────────────────────
            A = ctrl._current_A(t)
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
            "pers_snaps"    : self._pers_snaps,
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
    condition: str,
    seed: int,
    rep: int,
    out_dir: Path,
    resume: bool = False,
    n_rounds: int = N_ROUNDS,
    t_cool: int = T_COOL,
    seed_cfg: dict | None = None,
) -> dict[str, Any]:
    if seed_cfg is None:
        seed_cfg = SEED_CONFIG
    scfg   = seed_cfg[seed]
    A_res  = float(scfg["A_res"])
    target = str(scfg["target"])

    sim_seed   = seed * 100 + rep
    noise_seed = sim_seed + 500_000

    subdir = out_dir / CONDITION_SUBDIRS[condition]
    subdir.mkdir(parents=True, exist_ok=True)
    out_csv = subdir / f"seed{seed}_rep{rep}.csv"

    ctrl = RampNoiseController(
        condition       = condition,
        A_res           = A_res,
        A_herd          = A_HERD,
        t_cool          = t_cool,
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
        "seed"     : seed,
        "rep"      : rep,
        "condition": condition,
        "A_res"    : A_res,
        "target"   : target,
        "sim_seed" : sim_seed,
        "csv_path" : str(out_csv.relative_to(ROOT)),
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
    """Compute G0–G3 gate verdicts for H8.2."""

    def _runs(cond: str, s: int) -> list[dict]:
        return [m for m in all_metrics if m["condition"] == cond and m["seed"] == s]

    # ── G0: AT survives in c1_ctrl ────────────────────────────────────────────
    g0_detail: dict[str, Any] = {}
    g0_pass_seeds: list[int]  = []
    for s in seeds:
        c1 = _runs("c1_ctrl", s)
        lock_count = sum(1 for r in c1 if r.get("lock_1_reached"))
        med_T      = _median([r["T_lock_1"] for r in c1 if r.get("lock_1_reached")])
        seed_pass  = lock_count > 0
        g0_detail[str(s)] = {
            "lock_count"   : lock_count,
            "n_reps"       : len(c1),
            "med_T_lock"   : med_T,
            "passes"       : seed_pass,
        }
        if seed_pass:
            g0_pass_seeds.append(s)
    g0_pass = len(g0_pass_seeds) >= 2

    # ── G1: Ramp accelerates locking (c2_ramp vs c1_ctrl) ────────────────────
    g1_detail: dict[str, Any] = {}
    g1_seed_pass: list[bool]  = []
    for s in g0_pass_seeds:
        c1_runs = _runs("c1_ctrl", s)
        c2_runs = _runs("c2_ramp", s)
        med_a = _median([r["T_lock_1"] for r in c1_runs if r.get("lock_1_reached")])
        med_b = _median([r["T_lock_1"] for r in c2_runs if r.get("lock_1_reached")])
        if med_a is not None and med_b is not None and med_a > 0:
            ratio     = med_b / med_a
            seed_pass = ratio <= G1_RAMP_THRESHOLD
        else:
            ratio     = None
            seed_pass = False
        g1_detail[str(s)] = {
            "med_T_c1" : med_a,
            "med_T_c2" : med_b,
            "ratio"    : round(ratio, 3) if ratio is not None else None,
            "threshold": G1_RAMP_THRESHOLD,
            "passes"   : seed_pass,
        }
        g1_seed_pass.append(seed_pass)
    g1_pass = bool(g1_seed_pass) and all(g1_seed_pass)

    # ── G2: Hysteresis (c3_ramp) ──────────────────────────────────────────────
    g2_detail: dict[str, Any] = {}
    g2_seed_pass: list[bool]  = []
    for s in g0_pass_seeds:
        c3 = _runs("c3_ramp", s)
        med_lock   = _median([r["T_lock_1"] for r in c3 if r.get("lock_1_reached")])
        med_unlock = _median([r["T_unlock"]  for r in c3 if r.get("escape_reached")])
        if med_lock is not None and med_lock > 0 and med_unlock is not None:
            ratio     = med_unlock / med_lock
            seed_pass = ratio <= G2_HYSTER_RATIO
        else:
            ratio     = None
            seed_pass = False
        g2_detail[str(s)] = {
            "med_T_lock_1" : med_lock,
            "med_T_unlock" : med_unlock,
            "ratio"        : round(ratio, 3) if ratio is not None else None,
            "passes"       : seed_pass,
        }
        g2_seed_pass.append(seed_pass)
    g2_pass = bool(g2_seed_pass) and all(g2_seed_pass)

    # ── G3: Path memory (c3_ramp) ─────────────────────────────────────────────
    g3_detail: dict[str, Any] = {}
    g3_pass_seeds: list[int]  = []
    for s in seeds:
        c3    = _runs("c3_ramp", s)
        med_1 = _median([r["T_lock_1"] for r in c3 if r.get("lock_1_reached")])
        med_2 = _median([r["T_lock_2"] for r in c3 if r.get("lock_2_reached")])
        seed_pass = med_1 is not None and med_2 is not None and med_2 < med_1
        g3_detail[str(s)] = {
            "med_T_lock_1"     : med_1,
            "med_T_lock_2"     : med_2,
            "n_complete_cycles": sum(1 for r in c3 if r.get("lock_2_reached")),
            "passes"           : seed_pass,
        }
        if seed_pass:
            g3_pass_seeds.append(s)
    g3_pass = len(g3_pass_seeds) >= G3_SEEDS_NEEDED

    # ── Verdict ───────────────────────────────────────────────────────────────
    if not g0_pass:
        verdict = "G0_FAIL"
    elif g1_pass and g2_pass and g3_pass:
        verdict = "PASS"
    elif any([g1_pass, g2_pass, g3_pass]):
        verdict = "PARTIAL"
    else:
        verdict = "FAIL"

    return {
        "gates": {
            "g0_at_survives"    : g0_pass,
            "g1_ramp_accelerates": g1_pass,
            "g2_hysteresis"     : g2_pass,
            "g3_path_memory"    : g3_pass,
        },
        "verdict"         : verdict,
        "g0_passing_seeds": g0_pass_seeds,
        "g0_detail"       : g0_detail,
        "g1_detail"       : g1_detail,
        "g2_detail"       : g2_detail,
        "g3_detail"       : g3_detail,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Progress display
# ──────────────────────────────────────────────────────────────────────────────

def _fmt_run(m: dict[str, Any]) -> str:
    parts = []
    if m.get("lock_1_reached"):
        parts.append(f"T_lock={m['T_lock_1']}")
    else:
        parts.append("no_lock")
    if m.get("escape_reached"):
        parts.append(f"T_unlock={m['T_unlock']}")
    if m.get("lock_2_reached"):
        parts.append(f"T_lock2={m['T_lock_2']}")
    n_snaps = len(m.get("pers_snaps", []))
    if n_snaps:
        parts.append(f"snaps={n_snaps}")
    return "  " + ("  ".join(parts) if parts else "running…")


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="H8.2 Ramp-Cooling Protocol Experiment",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="re-read existing CSVs to recover measurements without re-simulating",
    )
    parser.add_argument(
        "--conditions", nargs="+",
        choices=["c1_ctrl", "c2_ramp", "c3_ramp"],
        default=["c1_ctrl", "c2_ramp", "c3_ramp"],
    )
    parser.add_argument(
        "--seeds", nargs="+", type=int, default=SEEDS,
    )
    parser.add_argument(
        "--reps", type=int, default=N_REPLICATES,
        help=f"replicates per (condition, seed) (default: {N_REPLICATES})",
    )
    parser.add_argument(
        "--n-rounds", type=int, default=N_ROUNDS,
        help=f"simulation length in rounds (default: {N_ROUNDS})",
    )
    parser.add_argument(
        "--t-cool", type=int, default=T_COOL,
        help=f"ramp duration in rounds (default: {T_COOL})",
    )
    parser.add_argument(
        "--a-res-override", nargs=2, metavar=("SEED", "A_RES"), action="append",
        default=[],
        help="override A_res for a seed, e.g. --a-res-override 97 0.07",
    )
    args = parser.parse_args()

    conditions : list[str] = args.conditions
    seeds      : list[int] = args.seeds
    n_reps     : int       = args.reps
    n_rounds   : int       = args.n_rounds
    t_cool     : int       = args.t_cool

    seed_cfg = _copy.deepcopy(SEED_CONFIG)
    for seed_str, a_res_str in args.a_res_override:
        sid = int(seed_str)
        if sid not in seed_cfg:
            raise ValueError(f"--a-res-override: unknown seed {sid}")
        seed_cfg[sid]["A_res"] = float(a_res_str)
        print(f"[override] seed {sid}: A_res → {float(a_res_str)}")

    total   = len(conditions) * len(seeds) * n_reps
    out_dir = OUTPUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / "summary.json"

    print(
        f"H8.2 Ramp-Cooling — {total} runs  "
        f"(conditions={conditions}  seeds={seeds}  reps={n_reps}  "
        f"T_cool={t_cool}  n_rounds={n_rounds})"
    )

    all_metrics: list[dict[str, Any]] = []
    done = 0

    for condition in conditions:
        for seed in seeds:
            for rep in range(n_reps):
                done += 1
                print(
                    f"[{done:>3}/{total}] {condition}  seed={seed}  rep={rep} …",
                    end=" ",
                    flush=True,
                )
                m = run_one(
                    condition = condition,
                    seed      = seed,
                    rep       = rep,
                    out_dir   = out_dir,
                    resume    = args.resume,
                    n_rounds  = n_rounds,
                    t_cool    = t_cool,
                    seed_cfg  = seed_cfg,
                )
                all_metrics.append(m)
                print(_fmt_run(m), flush=True)

    # ── Gate analysis ─────────────────────────────────────────────────────────
    gate_result = compute_gates(all_metrics, seeds)

    summary = {
        **gate_result,
        "n_runs" : len(all_metrics),
        "config" : {
            "seeds"           : seeds,
            "seed_config"     : {str(k): v for k, v in seed_cfg.items()},
            "n_replicates"    : n_reps,
            "n_rounds"        : n_rounds,
            "t_cool"          : t_cool,
            "A_herd"          : A_HERD,
            "lock_window"     : LOCK_WINDOW,
            "lock_threshold"  : LOCK_THRESHOLD,
            "escape_threshold": ESCAPE_THRESHOLD,
            "unlock_delay"    : UNLOCK_DELAY,
            "g1_threshold"    : G1_RAMP_THRESHOLD,
        },
        "runs": all_metrics,
    }
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False))

    # ── Print summary ─────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"H8.2 Ramp-Cooling — verdict: {gate_result['verdict']}")
    print(f"{'='*60}")
    gate_labels = {
        "g0_at_survives"     : "G0  AT_SURVIVES",
        "g1_ramp_accelerates": "G1  RAMP_ACCELERATES (threshold ≤0.90)",
        "g2_hysteresis"      : "G2  HYSTERESIS",
        "g3_path_memory"     : "G3  PATH_MEMORY",
    }
    for key, label in gate_labels.items():
        v    = gate_result["gates"][key]
        mark = "✓" if v else "✗"
        print(f"  {mark} {label}")

    # G1 detail
    print("\n  G1 detail (c2_ramp / c1_ctrl T_lock_1 median ratio):")
    for s, det in gate_result["g1_detail"].items():
        ratio_str = f"{det['ratio']:.3f}" if det["ratio"] is not None else "–"
        print(f"    seed{s}: ratio={ratio_str}  c1={det['med_T_c1']}  c2={det['med_T_c2']}")

    # G3 detail
    print("\n  G3 detail (c3_ramp complete cycles):")
    for s, det in gate_result["g3_detail"].items():
        print(
            f"    seed{s}: T_lock_1={det['med_T_lock_1']}  T_lock_2={det['med_T_lock_2']}"
            f"  complete={det['n_complete_cycles']}/{ n_reps}"
        )

    print(f"\n  Summary → {summary_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
