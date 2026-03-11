from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from analysis.amplitude_control_cache import build_amplitude_control_cache_from_csv


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else [])
        w.writeheader()
        w.writerows(rows)


def test_build_cache_groups_by_players_and_computes_std(tmp_path: Path) -> None:
    p = tmp_path / "control_per_seed.csv"
    rows = [
        {"players": 100, "seed": 0, "max_amp": 0.10},
        {"players": 100, "seed": 1, "max_amp": 0.14},
        {"players": 200, "seed": 0, "max_amp": 0.20},
        {"players": 200, "seed": 1, "max_amp": 0.22},
        {"players": 200, "seed": 2, "max_amp": 0.18},
    ]
    _write_csv(p, rows)

    cache = build_amplitude_control_cache_from_csv(
        [p],
        series="p",
        burn_in=1200,
        tail=600,
        ddof=1,
    )
    obj = cache.to_json_obj()
    assert obj["series"] == "p"
    assert obj["burn_in"] == 1200
    assert obj["tail"] == 600
    assert "100" in obj["by_players"] and "200" in obj["by_players"]
    assert abs(obj["by_players"]["100"]["mean_max_amp"] - 0.12) < 1e-12
    assert obj["by_players"]["100"]["std_max_amp"] > 0.0


def test_build_cache_requires_filter_when_multiple_selection_strengths(tmp_path: Path) -> None:
    p = tmp_path / "control_mixed_k.csv"
    rows = [
        {"players": 100, "seed": 0, "selection_strength": 0.1, "max_amp": 0.10},
        {"players": 100, "seed": 1, "selection_strength": 0.2, "max_amp": 0.12},
    ]
    _write_csv(p, rows)

    with pytest.raises(ValueError, match=r"multiple selection_strength"):
        build_amplitude_control_cache_from_csv(
            [p],
            series="p",
            burn_in=0,
            tail=100,
        )

    cache = build_amplitude_control_cache_from_csv(
        [p],
        series="p",
        burn_in=0,
        tail=100,
        selection_strength=0.1,
    )
    obj = cache.to_json_obj()
    assert abs(obj["by_players"]["100"]["mean_max_amp"] - 0.10) < 1e-12


def test_json_is_serializable(tmp_path: Path) -> None:
    p = tmp_path / "control.csv"
    _write_csv(p, [{"players": 100, "seed": 0, "max_amp": 0.10}])
    cache = build_amplitude_control_cache_from_csv([p], series="w", burn_in=1, tail=None)
    s = json.dumps(cache.to_json_obj())
    assert "by_players" in s


def test_build_cache_supports_per_input_players_override(tmp_path: Path) -> None:
    p100 = tmp_path / "n100.csv"
    p200 = tmp_path / "n200.csv"
    _write_csv(p100, [{"seed": 0, "max_amp": 0.10}, {"seed": 1, "max_amp": 0.12}])
    _write_csv(p200, [{"seed": 0, "max_amp": 0.20}])

    cache = build_amplitude_control_cache_from_csv(
        [(p100, 100), (p200, 200)],
        series="p",
        burn_in=0,
        tail=100,
    )
    obj = cache.to_json_obj()
    assert "100" in obj["by_players"]
    assert "200" in obj["by_players"]
