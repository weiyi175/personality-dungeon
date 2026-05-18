from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis.cycle_metrics import classify_cycle_level
from core.stackelberg import StackelbergCommitment
from simulation.personality_gate0 import DEFAULT_FULL_EVENTS_JSON
from simulation.run_simulation import SimConfig, _write_timeseries_csv, simulate


DEFAULT_SEEDS = [45, 47, 49, 51, 53, 55]


def _yes_no(v: bool) -> str:
    return "yes" if v else "no"


def _format_float(v: float) -> str:
    return f"{float(v):.6f}"


def _dominant_from_shares(aggressive: float, defensive: float, balanced: float) -> str:
    items = [
        ("aggressive", float(aggressive)),
        ("defensive", float(defensive)),
        ("balanced", float(balanced)),
    ]
    # Tie-break order is fixed to keep deterministic behavior.
    items.sort(key=lambda x: (-x[1], ["aggressive", "defensive", "balanced"].index(x[0])))
    return items[0][0]


def _ema_step(previous: float | None, current: float, alpha: float) -> float:
    if previous is None:
        return float(current)
    return (1.0 - float(alpha)) * float(previous) + float(alpha) * float(current)


@dataclass
class RollingPoint:
    round_end: int
    level: int
    passed: bool


class BasePulseAdapter:
    def __init__(
        self,
        *,
        condition: str,
        seed: int,
        baseline_commitment: StackelbergCommitment,
        pulse_commitment: StackelbergCommitment,
        pulse_horizon: int,
        refractory_rounds: int,
    ) -> None:
        self.condition = condition
        self.seed = int(seed)
        self.baseline_commitment = baseline_commitment
        self.pulse_commitment = pulse_commitment
        self.pulse_horizon = int(pulse_horizon)
        self.refractory_rounds = int(refractory_rounds)
        self.pulse_rounds_left = 0
        self.refractory_rounds_left = 0
        self.pulse_count = 0
        self.first_pulse_round: int | None = None
        self.step_rows: list[dict[str, Any]] = []

    def _advance(self) -> None:
        if self.pulse_rounds_left > 0:
            self.pulse_rounds_left -= 1
            if self.pulse_rounds_left == 0 and self.refractory_rounds > 0:
                self.refractory_rounds_left = self.refractory_rounds
            return
        if self.refractory_rounds_left > 0:
            self.refractory_rounds_left -= 1

    def _should_start_pulse(self, round_index: int, row: Mapping[str, Any]) -> bool:
        raise NotImplementedError

    def __call__(
        self,
        round_index: int,
        _cfg: SimConfig,
        _players: list[object],
        dungeon: Any,
        _step_records: list[dict[str, object]],
        row: dict[str, Any],
    ) -> None:
        self._advance()
        pulse_started = False
        if (
            self.pulse_rounds_left == 0
            and self.refractory_rounds_left == 0
            and self._should_start_pulse(round_index, row)
        ):
            self.pulse_rounds_left = self.pulse_horizon
            self.pulse_count += 1
            pulse_started = True
            if self.first_pulse_round is None:
                self.first_pulse_round = int(round_index) + 1

        pulse_active = self.pulse_rounds_left > 0
        selected = self.pulse_commitment if pulse_active else self.baseline_commitment
        dungeon.a = float(selected.a)
        dungeon.b = float(selected.b)
        dungeon.matrix_cross_coupling = float(selected.matrix_cross_coupling)

        self.step_rows.append(
            {
                "condition": self.condition,
                "seed": int(self.seed),
                "round": int(round_index) + 1,
                "pulse_active": _yes_no(pulse_active),
                "pulse_started": _yes_no(pulse_started),
                "pulse_rounds_left": int(self.pulse_rounds_left),
                "refractory_rounds_left": int(self.refractory_rounds_left),
                "a": _format_float(float(selected.a)),
                "b": _format_float(float(selected.b)),
                "matrix_cross_coupling": _format_float(float(selected.matrix_cross_coupling)),
            }
        )


class DynamicTriggerPulseAdapter(BasePulseAdapter):
    def __init__(
        self,
        *,
        condition: str,
        seed: int,
        baseline_commitment: StackelbergCommitment,
        pulse_commitment: StackelbergCommitment,
        pulse_horizon: int,
        refractory_rounds: int,
        ema_alpha: float,
    ) -> None:
        super().__init__(
            condition=condition,
            seed=seed,
            baseline_commitment=baseline_commitment,
            pulse_commitment=pulse_commitment,
            pulse_horizon=pulse_horizon,
            refractory_rounds=refractory_rounds,
        )
        self.ema_alpha = float(ema_alpha)
        self.p_hat_aggressive: float | None = None
        self.p_hat_defensive: float | None = None
        self.p_hat_balanced: float | None = None
        self.prev_dominant: str | None = None
        self.dominant_transition_count = 0

    def _should_start_pulse(self, _round_index: int, row: Mapping[str, Any]) -> bool:
        p_aggressive = float(row.get("p_aggressive") or 0.0)
        p_defensive = float(row.get("p_defensive") or 0.0)
        p_balanced = float(row.get("p_balanced") or 0.0)
        self.p_hat_aggressive = _ema_step(self.p_hat_aggressive, p_aggressive, self.ema_alpha)
        self.p_hat_defensive = _ema_step(self.p_hat_defensive, p_defensive, self.ema_alpha)
        self.p_hat_balanced = _ema_step(self.p_hat_balanced, p_balanced, self.ema_alpha)
        dominant = _dominant_from_shares(
            aggressive=float(self.p_hat_aggressive),
            defensive=float(self.p_hat_defensive),
            balanced=float(self.p_hat_balanced),
        )
        changed = self.prev_dominant is not None and dominant != self.prev_dominant
        if changed:
            self.dominant_transition_count += 1
        self.prev_dominant = dominant
        return bool(changed)


class PeriodicPulseAdapter(BasePulseAdapter):
    def __init__(
        self,
        *,
        condition: str,
        seed: int,
        baseline_commitment: StackelbergCommitment,
        pulse_commitment: StackelbergCommitment,
        pulse_horizon: int,
        refractory_rounds: int,
        periodic_interval: int,
    ) -> None:
        super().__init__(
            condition=condition,
            seed=seed,
            baseline_commitment=baseline_commitment,
            pulse_commitment=pulse_commitment,
            pulse_horizon=pulse_horizon,
            refractory_rounds=refractory_rounds,
        )
        self.periodic_interval = int(periodic_interval)

    def _should_start_pulse(self, round_index: int, _row: Mapping[str, Any]) -> bool:
        if self.periodic_interval <= 0:
            return False
        round_no = int(round_index) + 1
        return round_no % self.periodic_interval == 0


def _extract_series(rows: list[dict[str, Any]]) -> dict[str, list[float]]:
    return {
        "aggressive": [float(r["p_aggressive"]) for r in rows],
        "defensive": [float(r["p_defensive"]) for r in rows],
        "balanced": [float(r["p_balanced"]) for r in rows],
    }


def _window_eval(series3: Mapping[str, list[float]]) -> dict[str, dict[str, float | int]]:
    def _one(burn: int, tail: int) -> dict[str, float | int]:
        cyc = classify_cycle_level(
            series3,
            burn_in=burn,
            tail=tail,
            amplitude_threshold=0.02,
            corr_threshold=0.09,
            eta=0.55,
            stage3_method="turning",
            phase_smoothing=1,
            min_lag=2,
            max_lag=500,
        )
        return {
            "level": int(cyc.level),
            "stage3_score": float(cyc.stage3.score if cyc.stage3 else 0.0),
        }

    return {
        "burn1000_tail1000": _one(1000, 1000),
        "burn2000_tail2000": _one(2000, 2000),
    }


def _rolling_dropout_round(series3: Mapping[str, list[float]], *, sustain_windows: int = 5) -> int | None:
    window = 1000
    step = 20
    roll: list[RollingPoint] = []
    n = min(len(series3["aggressive"]), len(series3["defensive"]), len(series3["balanced"]))
    for end in range(window, n + 1, step):
        seg = {
            "aggressive": series3["aggressive"][end - window : end],
            "defensive": series3["defensive"][end - window : end],
            "balanced": series3["balanced"][end - window : end],
        }
        cyc = classify_cycle_level(
            seg,
            burn_in=0,
            tail=None,
            amplitude_threshold=0.02,
            corr_threshold=0.09,
            eta=0.55,
            stage3_method="turning",
            phase_smoothing=1,
            min_lag=2,
            max_lag=500,
        )
        roll.append(RollingPoint(round_end=end, level=int(cyc.level), passed=bool(cyc.level >= 3)))

    if not roll:
        return None
    flags = [1 if p.passed else 0 for p in roll]
    try:
        first_pass_idx = flags.index(1)
    except ValueError:
        return None
    k = max(1, int(sustain_windows))
    for i in range(first_pass_idx + 1, max(first_pass_idx + 1, len(flags) - k + 1)):
        if flags[i] == 0 and all(flags[j] == 0 for j in range(i, i + k)):
            return int(roll[i].round_end)
    return None


def _write_tsv(path: Path, *, rows: list[dict[str, Any]]) -> None:
    import csv

    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def _run_single(
    *,
    condition: str,
    seed: int,
    rounds: int,
    players: int,
    selection_strength: float,
    init_bias: float,
    memory_kernel: int,
    pulse_horizon: int,
    refractory_rounds: int,
    ema_alpha: float,
    periodic_interval: int,
    out_root: Path,
) -> dict[str, Any]:
    baseline = StackelbergCommitment(
        condition="control_pulse_policy",
        leader_action="baseline_pulse_policy",
        a=1.00,
        b=0.90,
        matrix_cross_coupling=0.20,
        description="Baseline payoff geometry",
    )
    pulse = StackelbergCommitment(
        condition="pulse_commitpush",
        leader_action="commit_pulse",
        a=1.05,
        b=0.85,
        matrix_cross_coupling=0.35,
        description="Finite edge+cross pulse",
    )

    out_dir = out_root / condition
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / f"seed{seed}.csv"
    out_steps = out_dir / f"pulse_steps_seed{seed}.tsv"

    cfg = SimConfig(
        n_players=int(players),
        n_rounds=int(rounds),
        seed=int(seed),
        payoff_mode="matrix_ab",
        popularity_mode="sampled",
        gamma=0.1,
        epsilon=0.0,
        a=float(baseline.a),
        b=float(baseline.b),
        matrix_cross_coupling=float(baseline.matrix_cross_coupling),
        init_bias=float(init_bias),
        evolution_mode="sampled",
        payoff_lag=1,
        selection_strength=float(selection_strength),
        enable_events=True,
        events_json=DEFAULT_FULL_EVENTS_JSON,
        out_csv=out_csv,
        memory_kernel=int(memory_kernel),
    )

    if condition == "control":
        adapter = None
    elif condition == "periodic":
        adapter = PeriodicPulseAdapter(
            condition=condition,
            seed=int(seed),
            baseline_commitment=baseline,
            pulse_commitment=pulse,
            pulse_horizon=int(pulse_horizon),
            refractory_rounds=int(refractory_rounds),
            periodic_interval=int(periodic_interval),
        )
    elif condition == "dynamic":
        adapter = DynamicTriggerPulseAdapter(
            condition=condition,
            seed=int(seed),
            baseline_commitment=baseline,
            pulse_commitment=pulse,
            pulse_horizon=int(pulse_horizon),
            refractory_rounds=int(refractory_rounds),
            ema_alpha=float(ema_alpha),
        )
    else:
        raise ValueError(f"Unknown condition: {condition}")

    strategy_space, rows = simulate(cfg, round_callback=adapter)
    _write_timeseries_csv(out_csv, strategy_space=strategy_space, rows=rows)
    if adapter is not None:
        _write_tsv(out_steps, rows=adapter.step_rows)

    series3 = _extract_series(rows)
    window = _window_eval(series3)
    return {
        "condition": condition,
        "seed": int(seed),
        "csv": str(out_csv.relative_to(ROOT)),
        "steps_tsv": str(out_steps.relative_to(ROOT)) if adapter is not None else "",
        "window_eval": window,
        "dropout_round": _rolling_dropout_round(series3),
        "pulse_count": int(adapter.pulse_count) if adapter is not None else 0,
        "first_pulse_round": int(adapter.first_pulse_round) if adapter is not None and adapter.first_pulse_round is not None else None,
        "dominant_transition_count": int(adapter.dominant_transition_count) if isinstance(adapter, DynamicTriggerPulseAdapter) else None,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare periodic pulse vs dynamic-trigger pulse")
    parser.add_argument("--seeds", type=str, default=",".join(str(s) for s in DEFAULT_SEEDS))
    parser.add_argument("--rounds", type=int, default=3000)
    parser.add_argument("--players", type=int, default=300)
    parser.add_argument("--selection-strength", type=float, default=0.10)
    parser.add_argument("--init-bias", type=float, default=0.12)
    parser.add_argument("--memory-kernel", type=int, default=3)
    parser.add_argument("--pulse-horizon", type=int, default=120)
    parser.add_argument("--refractory-rounds", type=int, default=240)
    parser.add_argument("--ema-alpha", type=float, default=0.15)
    parser.add_argument("--periodic-interval", type=int, default=300)
    parser.add_argument("--out-dir", type=str, default="outputs/actionB_pulse_compare")
    args = parser.parse_args()

    seeds = [int(x.strip()) for x in args.seeds.split(",") if x.strip()]
    out_root = ROOT / args.out_dir
    out_root.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    for condition in ("control", "periodic", "dynamic"):
        for seed in seeds:
            print(f"[run] condition={condition} seed={seed}", flush=True)
            rows.append(
                _run_single(
                    condition=condition,
                    seed=seed,
                    rounds=int(args.rounds),
                    players=int(args.players),
                    selection_strength=float(args.selection_strength),
                    init_bias=float(args.init_bias),
                    memory_kernel=int(args.memory_kernel),
                    pulse_horizon=int(args.pulse_horizon),
                    refractory_rounds=int(args.refractory_rounds),
                    ema_alpha=float(args.ema_alpha),
                    periodic_interval=int(args.periodic_interval),
                    out_root=out_root,
                )
            )

    aggregate: dict[str, dict[str, Any]] = {}
    for condition in ("control", "periodic", "dynamic"):
        sub = [r for r in rows if r["condition"] == condition]
        short_levels = [int(r["window_eval"]["burn1000_tail1000"]["level"]) for r in sub]
        long_levels = [int(r["window_eval"]["burn2000_tail2000"]["level"]) for r in sub]
        short_s3 = [float(r["window_eval"]["burn1000_tail1000"]["stage3_score"]) for r in sub]
        long_s3 = [float(r["window_eval"]["burn2000_tail2000"]["stage3_score"]) for r in sub]
        dropout = [int(r["dropout_round"]) for r in sub if r["dropout_round"] is not None]
        pulse_counts = [int(r["pulse_count"]) for r in sub]
        aggregate[condition] = {
            "n": len(sub),
            "short_level3_count": sum(1 for x in short_levels if x >= 3),
            "long_level3_count": sum(1 for x in long_levels if x >= 3),
            "short_s3_mean": float(sum(short_s3) / max(len(short_s3), 1)),
            "long_s3_mean": float(sum(long_s3) / max(len(long_s3), 1)),
            "dropout_round_median": (None if not dropout else int(sorted(dropout)[len(dropout) // 2])),
            "pulse_count_mean": float(sum(pulse_counts) / max(len(pulse_counts), 1)),
        }

    payload = {
        "generated_at": datetime.now().isoformat(),
        "scope": {
            "seeds": seeds,
            "rounds": int(args.rounds),
            "players": int(args.players),
            "selection_strength": float(args.selection_strength),
            "memory_kernel": int(args.memory_kernel),
            "pulse_horizon": int(args.pulse_horizon),
            "refractory_rounds": int(args.refractory_rounds),
            "ema_alpha": float(args.ema_alpha),
            "periodic_interval": int(args.periodic_interval),
        },
        "rows": rows,
        "aggregate": aggregate,
    }

    out_json = out_root / "option2_pulse_compare_summary.json"
    out_md = out_root / "option2_pulse_compare_onepager.md"
    out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    lines: list[str] = []
    lines.append("# Option 2 Pulse Compare")
    lines.append("")
    lines.append("| condition | n | short L3 count | long L3 count | short s3 mean | long s3 mean | dropout median | pulse count mean |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for condition in ("control", "periodic", "dynamic"):
        a = aggregate[condition]
        d = "None" if a["dropout_round_median"] is None else str(a["dropout_round_median"])
        lines.append(
            f"| {condition} | {a['n']} | {a['short_level3_count']} | {a['long_level3_count']}"
            f" | {a['short_s3_mean']:.3f} | {a['long_s3_mean']:.3f} | {d} | {a['pulse_count_mean']:.2f} |"
        )
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"written: {out_json.relative_to(ROOT)}")
    print(f"written: {out_md.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
