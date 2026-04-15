from __future__ import annotations

import runpy

import pytest


def test_script_entry_delegates_to_main(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, bool] = {"called": False}

    def _fake_main() -> int:
        captured["called"] = True
        return 0

    monkeypatch.setattr("simulation.pers_cal_baseline_gate60.main", _fake_main)

    with pytest.raises(SystemExit) as exc_info:
        runpy.run_path("scripts/pers_cal_baseline_gate60.py", run_name="__main__")

    assert exc_info.value.code == 0
    assert captured["called"] is True
