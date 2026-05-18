#!/usr/bin/env python3
"""H8.1 Phase Trap & Hysteresis Experiment

SDD §H8.0 (NoiseController) + §H8.1 (Reversible Phase Trap & Hysteresis)

Conditions:
  C1 baseline    : fixed A = A_res (additive per-round)  — AT structure calibration
  C2 phase_trap  : A = A_herd until p_target > 0.90, then A = A_res
  C3 unlock_loop : C2 + after lock → wait 100 r → A_herd → escape detected → A_res (relock)

Seeds:           {47, 97, 99}
A_resonance:     {47: 0.10, 97: 0.13, 99: 0.13}  (from H7.7 AT structure)
Target corners:  {47: balanced, 97: balanced, 99: defensive}
n_rounds:        10000 (C3 needs three full phases)
Replicates:      3 per (condition, seed)  →  81 runs total

Gates:
  G0 AT_SURVIVES_ADDITIVE   : ≥2/3 seeds lock in C1 within 10000 rounds
  G1 PHASE_TRAP_ACCELERATES : median T_lock_B ≤ 0.60 × median T_lock_A (G0-passing seeds)
  G2 HYSTERESIS             : median T_unlock / T_lock_1 ≤ 0.20 (C3, G0-passing seeds)
  G3 CUMULATIVE_LOCK_BIAS   : ≥2/3 seeds have median T_lock_2 < median T_lock_1 (C3)

NOTE: G0 is a safety valve — if G0 FAIL, execution stops; G1–G3 are skipped.

Usage:
  PYTHONPATH=/home/user/personality-dungeon ./venv/bin/python -u \\
      scripts/run_w81_phase_trap.py
  PYTHONPATH=/home/user/personality-dungeon ./venv/bin/python -u \\
      scripts/run_w81_phase_trap.py --resume
  PYTHONPATH=/home/user/personality-dungeon ./venv/bin/python -u \\
      scripts/run_w81_phase_trap.py --conditions c1 --seeds 47 --reps 1
"""
from __future__ import annotations

import argparse
import collections
import csv as _csv
import json
import random
import statistics
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[1]

from simulation.run_simulation import SimConfig, simulate
from players.base_player import DEFAULT_PERSONALITY_KEYS

# ──────────────────────────────────────────────────────────────────────────────
# Constants (all locked per SDD §H8.1)
# ──────────────────────────────────────────────────────────────────────────────

OUTPUT_DIR = ROOT / "outputs" / "h81_phase_trap"

SEEDS = [47, 97, 99]
N_REPLICATES = 3
N_ROUNDS = 10_000

A_HERD = 0.30            # herding amplitude (non-resonant, high)
THETA_TRIGGER = 0.90     # p_target_corner > this → trigger phase switch
LOCK_WINDOW = 500        # rolling window size for stable-lock detection
LOCK_THRESHOLD = 0.80    # all LOCK_WINDOW values must exceed this for "locked"
ESCAPE_THRESHOLD = 0.50  # first p_target < this → escape confirmed
TRIGGER_TIMEOUT = 2_000  # rounds; if no trigger by this round, mark timeout
UNLOCK_DELAY = 100       # rounds after first lock before starting thaw (C3)

# Golden Point (H7.3 locked)
MU_BASE           = 0.30
LAMBDA_MU         = 0.05
LAMBDA_K          = 0.20
SELECTION_STRENGTH = 0.15
K_LOWER           = 0.05
K_UPPER           = 0.25
GAMMA_BASE        = 0.16
SYNERGY_POWER     = 3.2

# Per-seed Arnold Tongue config (H7.7)
SEED_CONFIG: dict[int, dict[str, Any]] = {
    47: {"A_res": 0.10, "target": "balanced"},
    97: {"A_res": 0.13, "target": "balanced"},
    99: {"A_res": 0.13, "target": "defensive"},
}

CONDITION_SUBDIRS = {
    "c1": "c1_baseline",
    "c2": "c2_phase_trap",
    "c3": "c3_unlock",
}

# Gate thresholds
G1_RATIO_THRESHOLD   = 0.60   # T_lock_B ≤ 0.60 × T_lock_A
G2_HYSTERESIS_RATIO  = 0.20   # T_unlock / T_lock_1 ≤ 0.20
G3_SEEDS_NEEDED      = 2      # ≥ this many seeds must satisfy T_lock_2 < T_lock_1


# ──────────────────────────────────────────────────────────────────────────────
# H8.0 — NoiseController
# ──────────────────────────────────────────────────────────────────────────────

class NoiseController:
    """Additive per-round noise injection with closed-loop amplitude state machine.

    Implements the H8.0 NoiseController protocol (SDD §H8.0).

    State transitions:
      C1 : [LOCK_PHASE]  (constant A_res, no herding)
      C2 : HERDING → LOCK_PHASE
      C3 : HERDING → LOCK_PHASE → UNLOCK_DELAY → THAWING → RELOCK_PHASE

    Amplitude by state:
      HERDING, THAWING  →  A_herd
      all other states  →  A_res

    The callback (make_callback()) is called by simulate() AFTER rows.append(row),
    so personality changes take effect at round t+1.

    RNG seeding (deterministic, per-round × per-player):
      random.Random(noise_base_seed * 10_000_000 + t * n_players + player_idx)
    """

    def __init__(
        self,
        *,
        condition: str,
        A_res: float,
        A_herd: float,
        target_corner: str,
        noise_base_seed: int,
    ) -> None:
        self.condition      = condition
        self.A_res          = A_res
        self.A_herd         = A_herd
        self._target_key    = f"p_{target_corner}"
        self._noise_base    = noise_base_seed

        # State machine initial state
        self.state: str = "LOCK_PHASE" if condition == "c1" else "HERDING"

        # Absolute-round timestamps
        self.t_trigger   : int | None = None
        self.t_lock_1    : int | None = None   # round when first lock window completes
        self.t_thaw_start: int | None = None   # round when THAWING begins
        self.t_escape    : int | None = None   # round when escape detected
        self.t_lock_2    : int | None = None   # round when second lock window completes

        # Derived measurements (gate analysis uses these)
        self.T_trigger: int | None = None     # absolute trigger round (C2/C3)
        self.T_lock_1 : int | None = None     # rounds from phase-start to first lock
        self.T_unlock : int | None = None     # rounds from thaw_start to escape (C3)
        self.T_lock_2 : int | None = None     # rounds from escape to second lock (C3)

        # Status flags
        self.trigger_timeout : bool = False
        self.lock_1_reached  : bool = False
        self.escape_reached  : bool = False
        self.lock_2_reached  : bool = False

        # Rolling deque: holds last LOCK_WINDOW p_target values
        self._pwin: collections.deque[float] = collections.deque(maxlen=LOCK_WINDOW)

    # ── internal helpers ───────────────────────────────────────────────────────

    def _current_A(self) -> float:
        return self.A_herd if self.state in ("HERDING", "THAWING") else self.A_res

    def _window_locked(self) -> bool:
        return (
            len(self._pwin) >= LOCK_WINDOW
            and all(v > LOCK_THRESHOLD for v in self._pwin)
        )

    # ── public API ─────────────────────────────────────────────────────────────

    def make_callback(self) -> Callable:
        """Return a RoundCallback compatible with simulate(round_callback=...)."""
        target_key  = self._target_key
        noise_base  = self._noise_base
        keys: list[str] = list(DEFAULT_PERSONALITY_KEYS)
        ctrl = self   # captured reference; not a closure over mutable locals

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

            # ── State machine ────────────────────────────────────────────────
            if ctrl.state == "HERDING":
                # Trigger check (must happen before timeout check, has priority)
                if (
                    not ctrl.trigger_timeout
                    and p_target > THETA_TRIGGER
                    and ctrl.t_trigger is None
                ):
                    ctrl.state        = "LOCK_PHASE"
                    ctrl.t_trigger    = t
                    ctrl.T_trigger    = t
                    ctrl._pwin.clear()     # reset: measure lock from trigger
                # Timeout check (only if still in HERDING with no trigger)
                elif (
                    not ctrl.trigger_timeout
                    and t >= TRIGGER_TIMEOUT
                    and ctrl.t_trigger is None
                ):
                    ctrl.trigger_timeout = True
                    # State stays HERDING; no further measurements taken

            elif ctrl.state == "LOCK_PHASE":
                if not ctrl.lock_1_reached and ctrl._window_locked():
                    ctrl.lock_1_reached = True
                    ctrl.t_lock_1       = t
                    # T_lock_1: from t=0 (C1) or from trigger round (C2/C3)
                    phase_start       = 0 if ctrl.condition == "c1" else ctrl.t_trigger
                    ctrl.T_lock_1     = t - phase_start   # type: ignore[operator]
                    if ctrl.condition == "c3":
                        ctrl.state = "UNLOCK_DELAY"

            elif ctrl.state == "UNLOCK_DELAY":
                # C3 only: wait UNLOCK_DELAY rounds after lock confirmation, then thaw
                if ctrl.t_lock_1 is not None and t >= ctrl.t_lock_1 + UNLOCK_DELAY:
                    ctrl.state        = "THAWING"
                    ctrl.t_thaw_start = t
                    ctrl._pwin.clear()     # reset: not used further for this purpose

            elif ctrl.state == "THAWING":
                # Escape: first round where p_target drops below threshold
                if not ctrl.escape_reached and p_target < ESCAPE_THRESHOLD:
                    ctrl.escape_reached = True
                    ctrl.t_escape       = t
                    ctrl.T_unlock       = t - ctrl.t_thaw_start   # type: ignore[operator]
                    ctrl.state          = "RELOCK_PHASE"
                    ctrl._pwin.clear()     # reset: measure second lock from escape

            elif ctrl.state == "RELOCK_PHASE":
                if not ctrl.lock_2_reached and ctrl._window_locked():
                    ctrl.lock_2_reached = True
                    ctrl.t_lock_2       = t
                    ctrl.T_lock_2       = t - ctrl.t_escape   # type: ignore[operator]
                    ctrl.state          = "RELOCKED"

            # RELOCKED: terminal state, no transitions

            # ── Additive noise injection ─────────────────────────────────────
            A = ctrl._current_A()
            if A <= 0.0:
                return   # no-op: preserves ctrl condition exactly (H8.0-D1)

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
        """Summary dict of all timing measurements and status flags."""
        return {
            "state_final"     : self.state,
            "trigger_timeout" : self.trigger_timeout,
            "T_trigger"       : self.T_trigger,
            "T_lock_1"        : self.T_lock_1,
            "T_unlock"        : self.T_unlock,
            "T_lock_2"        : self.T_lock_2,
            "lock_1_reached"  : self.lock_1_reached,
            "escape_reached"  : self.escape_reached,
            "lock_2_reached"  : self.lock_2_reached,
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
    seed_cfg: dict | None = None,
) -> dict[str, Any]:
    """Run one (condition, seed, rep) combination and return a metrics dict.

    With resume=True, an existing CSV is replayed through the NoiseController
    state machine to recover timing measurements without re-simulating.
    """
    if seed_cfg is None:
        seed_cfg = SEED_CONFIG
    scfg    = seed_cfg[seed]
    A_res   = float(scfg["A_res"])
    target  = str(scfg["target"])

    # Replicate seed isolation: sim_seed = seed * 100 + rep
    sim_seed   = seed * 100 + rep
    noise_seed = sim_seed + 500_000   # separate RNG domain from simulation

    subdir = out_dir / CONDITION_SUBDIRS[condition]
    subdir.mkdir(parents=True, exist_ok=True)
    out_csv = subdir / f"seed{seed}_rep{rep}.csv"

    ctrl = NoiseController(
        condition       = condition,
        A_res           = A_res,
        A_herd          = A_HERD,
        target_corner   = target,
        noise_base_seed = noise_seed,
    )
    rnd_cb = ctrl.make_callback()

    if resume and out_csv.exists():
        # Replay stored CSV through the state machine to recover measurements.
        # The callback only reads row[target_key] for state transitions;
        # the players list is empty so noise injection is a no-op (harmless).
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
        "seed"      : seed,
        "rep"       : rep,
        "condition" : condition,
        "A_res"     : A_res,
        "target"    : target,
        "sim_seed"  : sim_seed,
        "csv_path"  : str(out_csv.relative_to(ROOT)),
    })
    return m


# ──────────────────────────────────────────────────────────────────────────────
# Gate analysis
# ──────────────────────────────────────────────────────────────────────────────

def _median(vals: list) -> float | None:
    clean = [v for v in vals if v is not None]
    return statistics.median(clean) if clean else None


def compute_gates(all_metrics: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute G0–G3 gate verdicts from the full list of run metrics.

    Gate logic follows SDD §H8.1 exactly.
    G0 must pass before G1–G3 are evaluated.
    """

    def _runs(cond: str, s: int) -> list[dict]:
        return [m for m in all_metrics if m["condition"] == cond and m["seed"] == s]

    # ── G0: AT structure survives in additive model (C1 baseline) ─────────────
    g0_detail: dict[str, Any] = {}
    g0_pass_seeds: list[int]  = []

    for s in SEEDS:
        c1 = _runs("c1", s)
        lock_count = sum(1 for r in c1 if r.get("lock_1_reached"))
        med_T      = _median([r["T_lock_1"] for r in c1 if r.get("lock_1_reached")])
        seed_pass  = lock_count > 0
        g0_detail[str(s)] = {
            "lock_count": lock_count,
            "n_reps"    : len(c1),
            "med_T_lock_A": med_T,
            "passes"    : seed_pass,
        }
        if seed_pass:
            g0_pass_seeds.append(s)

    g0_pass = len(g0_pass_seeds) >= 2   # ≥2/3 seeds

    # ── G1: Phase Trap accelerates locking (C2 vs C1, G0-passing seeds) ───────
    g1_detail: dict[str, Any] = {}
    g1_seed_pass: list[bool]  = []

    for s in g0_pass_seeds:
        c1_runs = _runs("c1", s)
        c2_runs = _runs("c2", s)
        med_a = _median([r["T_lock_1"] for r in c1_runs if r.get("lock_1_reached")])
        med_b = _median([
            r["T_lock_1"] for r in c2_runs
            if r.get("lock_1_reached") and not r.get("trigger_timeout")
        ])
        if med_a is not None and med_b is not None and med_a > 0:
            ratio      = med_b / med_a
            seed_pass  = ratio <= G1_RATIO_THRESHOLD
        else:
            ratio     = None
            seed_pass = False
        g1_detail[str(s)] = {
            "med_T_lock_A": med_a,
            "med_T_lock_B": med_b,
            "ratio"       : round(ratio, 3) if ratio is not None else None,
            "passes"      : seed_pass,
        }
        g1_seed_pass.append(seed_pass)

    g1_pass = bool(g1_seed_pass) and all(g1_seed_pass)

    # ── G2: Hysteresis — T_unlock << T_lock_1 (C3, G0-passing seeds) ─────────
    g2_detail: dict[str, Any] = {}
    g2_seed_pass: list[bool]  = []

    for s in g0_pass_seeds:
        c3 = _runs("c3", s)
        med_lock   = _median([r["T_lock_1"] for r in c3 if r.get("lock_1_reached")])
        med_unlock = _median([r["T_unlock"]  for r in c3 if r.get("escape_reached")])
        if (
            med_lock   is not None and med_lock > 0
            and med_unlock is not None
        ):
            ratio     = med_unlock / med_lock
            seed_pass = ratio <= G2_HYSTERESIS_RATIO
        else:
            ratio     = None
            seed_pass = False
        g2_detail[str(s)] = {
            "med_T_lock_1": med_lock,
            "med_T_unlock": med_unlock,
            "ratio"       : round(ratio, 3) if ratio is not None else None,
            "passes"      : seed_pass,
        }
        g2_seed_pass.append(seed_pass)

    g2_pass = bool(g2_seed_pass) and all(g2_seed_pass)

    # ── G3: Cumulative lock bias — T_lock_2 < T_lock_1 (C3, all seeds) ────────
    g3_detail: dict[str, Any] = {}
    g3_pass_seeds: list[int]  = []

    for s in SEEDS:
        c3 = _runs("c3", s)
        med_1 = _median([r["T_lock_1"] for r in c3 if r.get("lock_1_reached")])
        med_2 = _median([r["T_lock_2"] for r in c3 if r.get("lock_2_reached")])
        seed_pass = (
            med_1 is not None and med_2 is not None and med_2 < med_1
        )
        g3_detail[str(s)] = {
            "med_T_lock_1": med_1,
            "med_T_lock_2": med_2,
            "passes"      : seed_pass,
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
            "g0_at_survives"         : g0_pass,
            "g1_phase_trap_accelerates": g1_pass,
            "g2_hysteresis"          : g2_pass,
            "g3_cumulative_lock_bias": g3_pass,
        },
        "verdict"          : verdict,
        "g0_passing_seeds" : g0_pass_seeds,
        "g0_detail"        : g0_detail,
        "g1_detail"        : g1_detail,
        "g2_detail"        : g2_detail,
        "g3_detail"        : g3_detail,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Progress display
# ──────────────────────────────────────────────────────────────────────────────

def _fmt_run(m: dict[str, Any]) -> str:
    parts = []
    if m.get("trigger_timeout"):
        parts.append("TRIGGER_TIMEOUT")
    if m.get("lock_1_reached"):
        parts.append(f"T_lock={m['T_lock_1']}")
    elif m["condition"] != "c1":
        parts.append("no_lock")
    if m.get("escape_reached"):
        parts.append(f"T_unlock={m['T_unlock']}")
    if m.get("lock_2_reached"):
        parts.append(f"T_lock2={m['T_lock_2']}")
    return "  " + ("  ".join(parts) if parts else "running…")


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="H8.1 Phase Trap & Hysteresis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="re-read existing CSVs instead of re-simulating",
    )
    parser.add_argument(
        "--conditions", nargs="+", choices=["c1", "c2", "c3"],
        default=["c1", "c2", "c3"],
        help="which conditions to run (default: all three)",
    )
    parser.add_argument(
        "--seeds", nargs="+", type=int, default=SEEDS,
        help="which seeds to run (default: 47 97 99)",
    )
    parser.add_argument(
        "--reps", type=int, default=N_REPLICATES,
        help="replicates per (condition, seed) (default: 3)",
    )
    parser.add_argument(
        "--n-rounds", type=int, default=N_ROUNDS,
        help=f"rounds per simulation (default: {N_ROUNDS})",
    )
    parser.add_argument(
        "--a-res-override", nargs=2, metavar=("SEED", "A_RES"), action="append",
        default=[],
        help="override A_res for a seed, e.g. --a-res-override 97 0.07",
    )
    args = parser.parse_args()

    conditions: list[str] = args.conditions
    seeds: list[int]       = args.seeds
    n_reps: int            = args.reps
    n_rounds_cli: int      = args.n_rounds

    # Apply per-seed A_res overrides (mutate a local copy of SEED_CONFIG)
    import copy as _copy
    seed_cfg: dict = _copy.deepcopy(SEED_CONFIG)
    for seed_str, a_res_str in args.a_res_override:
        sid = int(seed_str)
        if sid not in seed_cfg:
            raise ValueError(f"--a-res-override: unknown seed {sid}")
        seed_cfg[sid]["A_res"] = float(a_res_str)
        print(f"[override] seed {sid}: A_res → {float(a_res_str)}")
    total = len(conditions) * len(seeds) * n_reps

    out_dir = OUTPUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / "summary.json"

    print(
        f"H8.1 Phase Trap — {total} runs  "
        f"(conditions={conditions}  seeds={seeds}  reps={n_reps})"
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
                    n_rounds  = n_rounds_cli,
                    seed_cfg  = seed_cfg,
                )
                all_metrics.append(m)
                print(_fmt_run(m), flush=True)

    # ── Gate analysis ─────────────────────────────────────────────────────────
    gate_result = compute_gates(all_metrics)

    summary = {
        **gate_result,
        "n_runs": len(all_metrics),
        "config": {
            "seeds"           : seeds,
            "seed_config"     : {str(k): v for k, v in seed_cfg.items()},
            "n_replicates"    : n_reps,
            "n_rounds"        : n_rounds_cli,
            "A_herd"          : A_HERD,
            "theta_trigger"   : THETA_TRIGGER,
            "lock_window"     : LOCK_WINDOW,
            "lock_threshold"  : LOCK_THRESHOLD,
            "escape_threshold": ESCAPE_THRESHOLD,
            "trigger_timeout" : TRIGGER_TIMEOUT,
            "unlock_delay"    : UNLOCK_DELAY,
        },
        "runs": all_metrics,
    }
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False))

    # ── Print summary ─────────────────────────────────────────────────────────
    print(f"\n{'='*52}")
    print(f"H8.1 Phase Trap — verdict: {gate_result['verdict']}")
    print(f"{'='*52}")
    gate_labels = {
        "g0_at_survives"          : "G0  AT_SURVIVES_ADDITIVE",
        "g1_phase_trap_accelerates": "G1  PHASE_TRAP_ACCELERATES",
        "g2_hysteresis"           : "G2  HYSTERESIS",
        "g3_cumulative_lock_bias" : "G3  CUMULATIVE_LOCK_BIAS",
    }
    for key, label in gate_labels.items():
        v = gate_result["gates"][key]
        mark = "✓" if v else "✗"
        print(f"  {mark} {label}")
    if gate_result["verdict"] == "G0_FAIL":
        print("\n  ⚠ G0 FAIL: additive model AT structure not confirmed.")
        print("    Diagnose before running G1–G3.")
        print("    Possible fallback: try 'reset per-round' noise model (re-spec H8.1a).")
    print(f"\n  Summary → {summary_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
