from __future__ import annotations

import json
from pathlib import Path

import pytest

from simulation import pers_cal_baseline_gate60 as gate60


def test_parse_seeds_range_and_csv() -> None:
    assert gate60.parse_seeds("42..45") == [42, 43, 44, 45]
    assert gate60.parse_seeds("42,44,46") == [42, 44, 46]


def test_summarize_outcomes_gate_pass() -> None:
    outcomes = [
        gate60.SeedOutcome(seed=42, level=3, s3=0.91, turn=200.0, gamma=0.0, elapsed_sec=1.0),
        gate60.SeedOutcome(seed=43, level=3, s3=0.88, turn=180.0, gamma=0.0, elapsed_sec=1.0),
        gate60.SeedOutcome(seed=44, level=1, s3=0.00, turn=0.0, gamma=0.0, elapsed_sec=1.0),
    ]
    summary = gate60.summarize_outcomes(
        outcomes,
        healthy_threshold=0.80,
        gate_max_l1=1,
        gate_min_healthy=2,
    )
    assert summary["l1"] == 1
    assert summary["healthy"] == 2
    assert summary["gate"]["overall_pass"] is True


def test_summarize_outcomes_gate_fail() -> None:
    outcomes = [
        gate60.SeedOutcome(seed=42, level=1, s3=0.00, turn=0.0, gamma=0.0, elapsed_sec=1.0),
        gate60.SeedOutcome(seed=43, level=1, s3=0.00, turn=0.0, gamma=0.0, elapsed_sec=1.0),
        gate60.SeedOutcome(seed=44, level=3, s3=0.79, turn=0.0, gamma=0.0, elapsed_sec=1.0),
    ]
    summary = gate60.summarize_outcomes(
        outcomes,
        healthy_threshold=0.80,
        gate_max_l1=1,
        gate_min_healthy=2,
    )
    assert summary["gate"]["l1_pass"] is False
    assert summary["gate"]["healthy_pass"] is False
    assert summary["gate"]["overall_pass"] is False


def test_main_writes_summary_json(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    def _fake_evaluate_seed(cfg, *, seed: int, burn_in: int, tail: int):  # noqa: ANN001
        return gate60.SeedOutcome(
            seed=seed,
            level=3,
            s3=0.90,
            turn=220.0,
            gamma=0.0,
            elapsed_sec=0.01,
        )

    monkeypatch.setattr(gate60, "evaluate_seed", _fake_evaluate_seed)

    out_json = tmp_path / "summary.json"
    rc = gate60.main(
        [
            "--seeds",
            "42,43,44",
            "--n-players",
            "30",
            "--n-rounds",
            "100",
            "--burn-in",
            "20",
            "--tail",
            "40",
            "--gate-max-l1",
            "0",
            "--gate-min-healthy",
            "3",
            "--out-json",
            str(out_json),
        ]
    )

    assert rc == 0
    assert out_json.exists()
    payload = json.loads(out_json.read_text(encoding="utf-8"))
    assert payload["total_seeds"] == 3
    assert payload["gate"]["overall_pass"] is True
