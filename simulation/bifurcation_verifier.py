"""
Bifurcation verifier for P7-H Phase II.

Validates the bifurcation detection and event generation system using a
lightweight personality dynamics model (exponential decay toward attractor).

Protocol per trial
------------------
1. Start at a controlled initial position (defined by starting_proximity).
2. Apply an event sequence (N steps from event_sequence_planner / design_bifurcation_event).
3. Measure peak proximity reached during the event phase.
4. Run M rounds of free dynamics (decay toward baseline) and measure decay curve.
5. Record: peak_proximity, displacement, recovery_rounds, success flag.

"Success" definition
--------------------
A trial is "successful" if the event sequence drives bifurcation_proximity above
the CRITICAL_PROXIMITY threshold (0.8) during the event phase.  This is a
reachability test—it verifies that the event system can push the personality
into the high-sensitivity zone.

Aligned vs. random comparison
------------------------------
Each trial is mirrored with a random-direction event of the same magnitude to
quantify the advantage of using the sensitive direction (v1) vs. noise.

Usage
-----
    verifier = BifurcationVerifier(n_trials=50)
    results = verifier.run_all_trials()
    verifier.save_report(results, "reports/experiments/p7h_bifurcation_verification")
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from simulation.bifurcation_detector import (
    BASELINE_ATTRACTOR,
    CRITICAL_PROXIMITY,
    EPSILON_C,
    FEATURE_NAMES,
    compute_bifurcation_distance,
)
from simulation.event_generator import (
    design_bifurcation_event,
    event_sequence_planner,
    intensity_modulation,
)

# Personality dynamics decay rate (per round), calibrated to P7-H landscape explorer
_DECAY_RATE: float = 0.95


# ── Data classes ─────────────────────────────────────────────────────────────

@dataclass
class TrialResult:
    trial_id: int
    starting_proximity: float
    n_event_steps: int
    intensity_scale: float
    event_target: str

    # Event phase metrics
    peak_proximity: float
    peak_displacement: float
    final_event_proximity: float

    # Recovery phase metrics (free dynamics after events stop)
    proximity_after_10_rounds: float
    proximity_after_50_rounds: float
    proximity_after_100_rounds: float

    # Aligned vs random comparison
    aligned_peak_proximity: float
    random_peak_proximity: float
    alignment_advantage: float  # aligned - random

    # Success flags
    reached_critical: bool  # peak_proximity > CRITICAL_PROXIMITY


@dataclass
class VerificationReport:
    metadata: dict[str, Any]
    summary: dict[str, Any]
    parameter_sweep: list[dict[str, Any]]
    trials: list[dict[str, Any]]


# ── Dynamics model ────────────────────────────────────────────────────────────

def _free_dynamics(pv: np.ndarray, n_rounds: int) -> list[float]:
    """Simulate free personality dynamics (exponential decay toward baseline).

    Returns a list of bifurcation_proximity values, one per round.
    """
    cur = pv.copy()
    proximities = []
    for _ in range(n_rounds):
        cur = _DECAY_RATE * cur + (1.0 - _DECAY_RATE) * BASELINE_ATTRACTOR
        proximities.append(compute_bifurcation_distance(cur)["bifurcation_proximity"])
    return proximities


def _apply_event_sequence(
    start_pv: np.ndarray,
    n_steps: int,
    intensity_scale: float,
    target: str = "personality_shift",
    rng: np.random.RandomState | None = None,
    use_random_direction: bool = False,
) -> tuple[np.ndarray, list[float]]:
    """Apply N event steps and return (final_pv, proximity_trace).

    If use_random_direction=True, events are random unit vectors (control arm).
    """
    cur = start_pv.copy()
    proximities = []

    for _ in range(n_steps):
        bifurc = compute_bifurcation_distance(cur)
        proximity = bifurc["bifurcation_proximity"]
        magnitude = intensity_modulation(proximity) * intensity_scale

        if use_random_direction:
            assert rng is not None
            direction = rng.randn(9)
            direction /= np.linalg.norm(direction)
        else:
            ev = design_bifurcation_event(cur, target=target, intensity_scale=intensity_scale)
            direction = np.array(ev["direction"])
            magnitude = ev["magnitude"]

        cur = cur + direction * magnitude
        proximities.append(compute_bifurcation_distance(cur)["bifurcation_proximity"])

    return cur, proximities


# ── Core verifier ─────────────────────────────────────────────────────────────

class BifurcationVerifier:
    """Run verification trials for the bifurcation detection + event system."""

    def __init__(
        self,
        n_trials: int = 50,
        seed: int = 42,
        free_dynamics_rounds: int = 100,
    ) -> None:
        self.n_trials = n_trials
        self.rng = np.random.RandomState(seed)
        self.free_dynamics_rounds = free_dynamics_rounds

    # ── Single trial ──────────────────────────────────────────────────────────

    def run_trial(
        self,
        trial_id: int,
        starting_proximity: float,
        n_event_steps: int,
        intensity_scale: float,
        event_target: str = "personality_shift",
    ) -> TrialResult:
        """Run one verification trial."""

        # Place starting personality at the requested proximity level
        start_pv = self._position_at_proximity(starting_proximity)

        # ── Aligned event sequence ────────────────────────────────────────────
        aligned_pv, aligned_trace = _apply_event_sequence(
            start_pv,
            n_steps=n_event_steps,
            intensity_scale=intensity_scale,
            target=event_target,
        )
        aligned_peak = max(aligned_trace) if aligned_trace else 0.0

        # ── Random-direction control arm ──────────────────────────────────────
        _, random_trace = _apply_event_sequence(
            start_pv,
            n_steps=n_event_steps,
            intensity_scale=intensity_scale,
            rng=self.rng,
            use_random_direction=True,
        )
        random_peak = max(random_trace) if random_trace else 0.0

        # ── Free dynamics after aligned sequence ──────────────────────────────
        recovery = _free_dynamics(aligned_pv, self.free_dynamics_rounds)

        def _at(rounds: int) -> float:
            idx = min(rounds - 1, len(recovery) - 1)
            return recovery[idx] if recovery else 0.0

        final_event_proximity = aligned_trace[-1] if aligned_trace else 0.0
        peak_displacement = float(
            np.linalg.norm(aligned_pv - start_pv)
        )

        return TrialResult(
            trial_id=trial_id,
            starting_proximity=starting_proximity,
            n_event_steps=n_event_steps,
            intensity_scale=intensity_scale,
            event_target=event_target,
            peak_proximity=aligned_peak,
            peak_displacement=peak_displacement,
            final_event_proximity=final_event_proximity,
            proximity_after_10_rounds=_at(10),
            proximity_after_50_rounds=_at(50),
            proximity_after_100_rounds=_at(100),
            aligned_peak_proximity=aligned_peak,
            random_peak_proximity=random_peak,
            alignment_advantage=aligned_peak - random_peak,
            reached_critical=aligned_peak > CRITICAL_PROXIMITY,
        )

    # ── Parameter sweep ───────────────────────────────────────────────────────

    def run_all_trials(self) -> VerificationReport:
        """Run the full verification suite.

        Sweeps:
          starting_proximity ∈ {0.0, 0.3, 0.5, 0.7}
          n_event_steps      ∈ {1, 3, 5, 7}
          intensity_scale    ∈ {0.5, 1.0, 2.0}
        """
        param_grid = [
            (sp, ns, iscale)
            for sp    in [0.0, 0.3, 0.5, 0.7]
            for ns    in [1, 3, 5, 7]
            for iscale in [0.5, 1.0, 2.0]
        ]

        trials: list[TrialResult] = []
        t_start = time.time()

        for tid, (sp, ns, iscale) in enumerate(param_grid):
            trial = self.run_trial(
                trial_id=tid,
                starting_proximity=sp,
                n_event_steps=ns,
                intensity_scale=iscale,
            )
            trials.append(trial)

        # Fill remaining n_trials with random starting points
        base_count = len(param_grid)
        remaining = max(0, self.n_trials - base_count)
        for extra in range(remaining):
            sp = float(self.rng.uniform(0.0, 0.8))
            ns = int(self.rng.choice([3, 5, 7]))
            iscale = float(self.rng.choice([0.5, 1.0, 2.0]))
            trial = self.run_trial(
                trial_id=base_count + extra,
                starting_proximity=sp,
                n_event_steps=ns,
                intensity_scale=iscale,
            )
            trials.append(trial)

        elapsed = time.time() - t_start
        summary = self._compute_summary(trials, elapsed)
        sweep_table = self._parameter_sweep_table(trials)

        return VerificationReport(
            metadata={
                "n_trials": len(trials),
                "decay_rate": _DECAY_RATE,
                "epsilon_c": EPSILON_C,
                "critical_proximity_threshold": CRITICAL_PROXIMITY,
                "free_dynamics_rounds": self.free_dynamics_rounds,
                "elapsed_seconds": round(elapsed, 3),
            },
            summary=summary,
            parameter_sweep=sweep_table,
            trials=[asdict(t) for t in trials],
        )

    # ── Rollback safety check ─────────────────────────────────────────────────

    def check_rollback_safety(
        self,
        n_check: int = 20,
    ) -> dict[str, Any]:
        """Verify that proximity reliably drops back toward 0 after events stop.

        This is the "emergency rollback" test: confirms free dynamics are
        contracting (λ < 0 in the real dynamics sense) so the system won't
        drift permanently unless a secondary attractor exists.
        """
        recovery_rates = []
        for _ in range(n_check):
            # Push to near critical
            sp = float(self.rng.uniform(0.5, 1.0))
            pv = self._position_at_proximity(sp)
            recovery = _free_dynamics(pv, 50)
            if sp > 0:
                rate = (recovery[-1] - sp) / sp  # negative = contracting
                recovery_rates.append(rate)

        return {
            "n_checks": n_check,
            "mean_recovery_rate": float(np.mean(recovery_rates)),
            "all_contracting": all(r < 0 for r in recovery_rates),
            "half_life_estimate_rounds": self._estimate_half_life(),
        }

    def _estimate_half_life(self) -> float:
        """Estimate rounds to halve proximity after events stop (at sp=0.9)."""
        pv = self._position_at_proximity(0.9)
        recovery = _free_dynamics(pv, 200)
        start = recovery[0]
        for i, v in enumerate(recovery):
            if v <= start * 0.5:
                return float(i + 1)
        return float(len(recovery))

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _position_at_proximity(self, proximity: float) -> np.ndarray:
        """Create a personality vector at the requested bifurcation proximity.

        Displaces baseline along v1 by the corresponding 9D distance.
        """
        from simulation.bifurcation_detector import V1
        target_distance = proximity * EPSILON_C
        return BASELINE_ATTRACTOR + target_distance * V1

    def _compute_summary(
        self, trials: list[TrialResult], elapsed: float
    ) -> dict[str, Any]:
        peak_prox = [t.peak_proximity for t in trials]
        reached = [t.reached_critical for t in trials]
        advantage = [t.alignment_advantage for t in trials]
        prox_100 = [t.proximity_after_100_rounds for t in trials]

        return {
            "total_trials": len(trials),
            "success_rate": float(np.mean(reached)),
            "n_reached_critical": int(sum(reached)),
            "mean_peak_proximity": float(np.mean(peak_prox)),
            "std_peak_proximity": float(np.std(peak_prox)),
            "mean_alignment_advantage": float(np.mean(advantage)),
            "mean_proximity_after_100_rounds": float(np.mean(prox_100)),
            "elapsed_seconds": round(elapsed, 3),
        }

    def _parameter_sweep_table(
        self, trials: list[TrialResult]
    ) -> list[dict[str, Any]]:
        """Aggregate by (n_event_steps, intensity_scale) grid."""
        from collections import defaultdict
        buckets: dict[tuple, list[TrialResult]] = defaultdict(list)
        for t in trials:
            buckets[(t.n_event_steps, t.intensity_scale)].append(t)

        rows = []
        for (ns, iscale), group in sorted(buckets.items()):
            rows.append({
                "n_event_steps": ns,
                "intensity_scale": iscale,
                "n_trials": len(group),
                "success_rate": float(np.mean([t.reached_critical for t in group])),
                "mean_peak_proximity": float(np.mean([t.peak_proximity for t in group])),
                "mean_alignment_advantage": float(
                    np.mean([t.alignment_advantage for t in group])
                ),
            })
        return rows

    # ── Report I/O ────────────────────────────────────────────────────────────

    def save_report(
        self,
        report: VerificationReport,
        out_dir: str | Path,
    ) -> Path:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)

        report_path = out / "p7h_bifurcation_verification.json"
        with open(report_path, "w") as f:
            json.dump(
                {
                    "metadata": report.metadata,
                    "summary": report.summary,
                    "parameter_sweep": report.parameter_sweep,
                    "trials": report.trials,
                },
                f,
                indent=2,
            )

        # Compact human-readable summary
        summary_path = out / "p7h_verification_summary.txt"
        with open(summary_path, "w") as f:
            m = report.summary
            f.write("P7-H BIFURCATION VERIFIER — SUMMARY\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Trials run          : {m['total_trials']}\n")
            f.write(f"Success rate        : {m['success_rate']*100:.1f}%  "
                    f"({m['n_reached_critical']}/{m['total_trials']} reached critical)\n")
            f.write(f"Mean peak proximity : {m['mean_peak_proximity']:.3f}\n")
            f.write(f"Alignment advantage : +{m['mean_alignment_advantage']:.3f}  "
                    "(aligned v1 vs random direction)\n")
            f.write(f"Proximity after 100 rounds: {m['mean_proximity_after_100_rounds']:.4f}\n")
            f.write(f"Elapsed             : {m['elapsed_seconds']:.2f}s\n\n")
            f.write("PARAMETER SWEEP (n_steps × intensity)\n")
            f.write("-" * 50 + "\n")
            f.write(f"{'steps':>5}  {'scale':>5}  {'success%':>8}  {'mean_peak':>9}  "
                    f"{'adv':>6}\n")
            for row in report.parameter_sweep:
                f.write(
                    f"{row['n_event_steps']:>5}  {row['intensity_scale']:>5.1f}  "
                    f"{row['success_rate']*100:>7.1f}%  "
                    f"{row['mean_peak_proximity']:>9.3f}  "
                    f"{row['mean_alignment_advantage']:>+6.3f}\n"
                )

        return report_path
