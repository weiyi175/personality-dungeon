from __future__ import annotations

import json
from pathlib import Path

from simulation import b1_async_dispatch_gate as b1


def test_parse_seeds() -> None:
    assert b1.parse_seeds("42..44") == [42, 43, 44]
    assert b1.parse_seeds("42,45") == [42, 45]


def test_main_stops_when_smoke_fairness_fails(monkeypatch, tmp_path: Path) -> None:
    calls: list[str] = []

    def _fake_run_stage(**kwargs):  # noqa: ANN003
        stage = kwargs["stage_name"]
        calls.append(stage)
        if "Smoke" in stage:
            return [], {
                "fairness_fail_count": 1,
                "gate": {"overall_pass": False},
            }
        raise AssertionError("gate stage should not run when smoke fails")

    monkeypatch.setattr(b1, "run_stage", _fake_run_stage)

    smoke_json = tmp_path / "smoke.json"
    gate_json = tmp_path / "gate.json"
    rc = b1.main(
        [
            "--smoke-seeds", "42",
            "--gate-seeds", "42",
            "--smoke-out-json", str(smoke_json),
            "--gate-out-json", str(gate_json),
        ]
    )

    assert rc == 2
    assert calls == ["B1 Smoke (fixed flow stage 1/2)"]
    assert smoke_json.exists()
    assert not gate_json.exists()


def test_main_runs_gate_after_smoke_pass(monkeypatch, tmp_path: Path) -> None:
    calls: list[str] = []

    baseline_json = tmp_path / "baseline.json"
    baseline_json.write_text(
        json.dumps(
            {
                "outcomes": [
                    {"seed": 42, "level": 3, "s3": 0.95},
                ]
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    def _fake_run_stage(**kwargs):  # noqa: ANN003
        stage = kwargs["stage_name"]
        calls.append(stage)
        if "Smoke" in stage:
            return [], {
                "fairness_fail_count": 0,
                "gate": {"overall_pass": False},
            }
        return [], {
            "fairness_fail_count": 0,
            "gate": {"overall_pass": True},
        }

    monkeypatch.setattr(b1, "run_stage", _fake_run_stage)

    smoke_json = tmp_path / "smoke.json"
    gate_json = tmp_path / "gate.json"
    rc = b1.main(
        [
            "--smoke-seeds", "42",
            "--gate-seeds", "42",
            "--baseline-summary-json", str(baseline_json),
            "--smoke-out-json", str(smoke_json),
            "--gate-out-json", str(gate_json),
        ]
    )

    assert rc == 0
    assert calls == [
        "B1 Smoke (fixed flow stage 1/2)",
        "B1 Gate60 (fixed flow stage 2/2)",
    ]
    assert smoke_json.exists()
    assert gate_json.exists()
