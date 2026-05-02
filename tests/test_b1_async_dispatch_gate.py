from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

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


def test_main_writes_phase_metadata_to_summaries(monkeypatch, tmp_path: Path) -> None:
    baseline_json = tmp_path / "baseline.json"
    baseline_json.write_text(
        json.dumps({"outcomes": [{"seed": 42, "level": 3, "s3": 0.95}]}, ensure_ascii=False),
        encoding="utf-8",
    )

    fake_outcome = SimpleNamespace(
        seed=42,
        level=3,
        s3=0.9,
        turn=1.0,
        gamma=0.0,
        elapsed_sec=0.1,
        dispatch_fairness_pass=True,
        event_sync_index_mean=0.0,
        reward_perturb_corr=0.0,
        trap_entry_round=None,
        event_neutrality_max_abs_mean=0.0,
        event_neutrality_pass=True,
        event_trigger_guard_check_count=0,
        event_trigger_guard_block_count=0,
        event_trigger_guard_block_rate=0.0,
        event_trigger_guard_pass=True,
    )

    def _fake_run_stage(**kwargs):  # noqa: ANN003
        stage = kwargs["stage_name"]
        if "Smoke" in stage:
            return [fake_outcome], {
                "fairness_fail_count": 0,
                "gate": {"overall_pass": False},
            }
        return [fake_outcome], {
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
            "--phase-id", "p0",
            "--bridge-id", "none",
            "--bridge-count", "0",
            "--anchor-profile-id", "lowcoupling_baseline_v2",
            "--smoke-out-json", str(smoke_json),
            "--gate-out-json", str(gate_json),
        ]
    )

    assert rc == 0
    smoke_payload = json.loads(smoke_json.read_text(encoding="utf-8"))
    gate_payload = json.loads(gate_json.read_text(encoding="utf-8"))

    for payload in (smoke_payload, gate_payload):
        assert payload["phase_id"] == "p0"
        assert payload["bridge_id"] == "none"
        assert payload["bridge_count"] == 0
        assert payload["anchor_profile_id"] == "lowcoupling_baseline_v2"
        assert payload["phase_run_metadata"]["phase_id"] == "p0"
        assert payload["outcomes"][0]["phase_id"] == "p0"
        assert payload["outcomes"][0]["bridge_count"] == 0


def test_summarize_stage_flags_payoff_static_invariant() -> None:
    outcomes = [
        SimpleNamespace(
            seed=42,
            level=3,
            s3=0.91,
            gamma=0.0,
            dispatch_fairness_pass=True,
            event_neutrality_pass=True,
            event_trigger_guard_pass=True,
            event_neutrality_max_abs_mean=0.0,
            event_trigger_guard_block_rate=0.0,
            readonly_leak_score=0.0,
            difficulty_index_mean=0.5,
            event_difficulty_multiplier_mean=1.03,
            payoff_static_score=0.0,
        ),
        SimpleNamespace(
            seed=43,
            level=3,
            s3=0.92,
            gamma=0.0,
            dispatch_fairness_pass=True,
            event_neutrality_pass=True,
            event_trigger_guard_pass=True,
            event_neutrality_max_abs_mean=0.0,
            event_trigger_guard_block_rate=0.0,
            readonly_leak_score=0.0,
            difficulty_index_mean=0.5,
            event_difficulty_multiplier_mean=1.06,
            payoff_static_score=1e-4,
        ),
    ]

    summary = b1.summarize_stage(
        outcomes,
        healthy_threshold=0.8,
        gate_max_l1=3,
        gate_min_healthy=1,
        readonly_leak_threshold=1e-6,
        payoff_static_threshold=1e-9,
        require_async_update=False,
        queue_overflow_max=0,
        phase_lag_min=0.0,
        baseline_map={
            42: {"level": 3, "s3": 0.91, "healthy": True},
            43: {"level": 3, "s3": 0.92, "healthy": True},
        },
    )

    assert summary["payoff_static_fail_count"] == 1
    assert summary["gate"]["invariant_payoff_static_pass"] is False
    assert summary["gate"]["invariant_overall_pass"] is False


def test_main_accepts_async_queue_gate_flags(monkeypatch, tmp_path: Path) -> None:
    baseline_json = tmp_path / "baseline.json"
    baseline_json.write_text(
        json.dumps({"outcomes": [{"seed": 42, "level": 3, "s3": 0.95}]}, ensure_ascii=False),
        encoding="utf-8",
    )

    seen_kwargs: list[dict[str, object]] = []

    def _fake_run_stage(**kwargs):  # noqa: ANN003
        seen_kwargs.append(kwargs)
        return [], {"fairness_fail_count": 0, "gate": {"overall_pass": True}}

    monkeypatch.setattr(b1, "run_stage", _fake_run_stage)

    rc = b1.main(
        [
            "--smoke-seeds", "42",
            "--gate-seeds", "42",
            "--baseline-summary-json", str(baseline_json),
            "--replicator-update-mode", "async_per_player",
            "--replicator-async-minibatch", "16",
            "--replicator-async-jitter", "0.20",
            "--event-queue-mode", "per_player",
            "--event-queue-cap", "8",
            "--event-queue-drain-rate", "1.0",
            "--require-async-update",
            "--queue-overflow-max", "0",
            "--phase-lag-min", "0.0",
            "--smoke-out-json", str(tmp_path / "smoke.json"),
            "--gate-out-json", str(tmp_path / "gate.json"),
        ]
    )

    assert rc == 0
    assert len(seen_kwargs) == 2
    assert seen_kwargs[0]["require_async_update"] is True
    assert seen_kwargs[0]["queue_overflow_max"] == 0
    assert float(seen_kwargs[0]["phase_lag_min"]) == 0.0


def test_main_accepts_b2_modulation_flags(monkeypatch, tmp_path: Path) -> None:
    baseline_json = tmp_path / "baseline.json"
    baseline_json.write_text(
        json.dumps({"outcomes": [{"seed": 42, "level": 3, "s3": 0.95}]}, ensure_ascii=False),
        encoding="utf-8",
    )

    seen_cfg: list[object] = []

    def _fake_build_cfg(**kwargs):  # noqa: ANN003
        seen_cfg.append(kwargs)
        return object()

    def _fake_run_stage(**kwargs):  # noqa: ANN003
        return [], {"fairness_fail_count": 0, "gate": {"overall_pass": True}}

    monkeypatch.setattr(b1, "build_b1_config", _fake_build_cfg)
    monkeypatch.setattr(b1, "run_stage", _fake_run_stage)

    rc = b1.main(
        [
            "--smoke-seeds", "42",
            "--gate-seeds", "42",
            "--baseline-summary-json", str(baseline_json),
            "--event-reward-mode", "multiplicative",
            "--event-reward-multiplier-cap", "0.20",
            "--event-modulation-mode", "multiplicative_v2",
            "--event-modulation-gain", "0.12",
            "--event-modulation-log-center", "0.0",
            "--event-modulation-zero-mean",
            "--event-modulation-floor", "0.85",
            "--event-modulation-ceiling", "1.15",
            "--smoke-out-json", str(tmp_path / "smoke.json"),
            "--gate-out-json", str(tmp_path / "gate.json"),
        ]
    )

    assert rc == 0
    assert len(seen_cfg) == 1
    cfg = seen_cfg[0]
    assert cfg["event_modulation_mode"] == "multiplicative_v2"
    assert float(cfg["event_modulation_gain"]) == 0.12
    assert float(cfg["event_modulation_log_center"]) == 0.0
    assert bool(cfg["event_modulation_zero_mean"]) is True
    assert float(cfg["event_modulation_floor"]) == 0.85
    assert float(cfg["event_modulation_ceiling"]) == 1.15


def test_main_accepts_b3_extended_impact_kernel_flags(monkeypatch, tmp_path: Path) -> None:
    baseline_json = tmp_path / "baseline.json"
    baseline_json.write_text(
        json.dumps({"outcomes": [{"seed": 42, "level": 3, "s3": 0.95}]}, ensure_ascii=False),
        encoding="utf-8",
    )

    seen_cfg: list[object] = []

    def _fake_build_cfg(**kwargs):  # noqa: ANN003
        seen_cfg.append(kwargs)
        return object()

    def _fake_run_stage(**kwargs):  # noqa: ANN003
        return [], {"fairness_fail_count": 0, "gate": {"overall_pass": True}}

    monkeypatch.setattr(b1, "build_b1_config", _fake_build_cfg)
    monkeypatch.setattr(b1, "run_stage", _fake_run_stage)

    rc = b1.main(
        [
            "--smoke-seeds", "42",
            "--gate-seeds", "42",
            "--baseline-summary-json", str(baseline_json),
            "--event-impact-mode", "spread",
            "--event-impact-horizon", "7",
            "--event-impact-decay", "0.82",
            "--impact-spread-kernel-id", "hierarchical_v2",
            "--impact-spread-local-mass", "0.65",
            "--impact-spread-neighbor-mass", "0.35",
            "--impact-spread-neighbor-hop", "2",
            "--impact-spread-memory-kernel", "3",
            "--smoke-out-json", str(tmp_path / "smoke.json"),
            "--gate-out-json", str(tmp_path / "gate.json"),
        ]
    )

    assert rc == 0
    assert len(seen_cfg) == 1
    cfg = seen_cfg[0]
    assert cfg["event_impact_mode"] == "spread"
    assert int(cfg["event_impact_horizon"]) == 7
    assert float(cfg["event_impact_decay"]) == 0.82
    assert cfg["impact_spread_kernel_id"] == "hierarchical_v2"
    assert float(cfg["impact_spread_local_mass"]) == 0.65
    assert float(cfg["impact_spread_neighbor_mass"]) == 0.35
    assert int(cfg["impact_spread_neighbor_hop"]) == 2
    assert int(cfg["impact_spread_memory_kernel"]) == 3
