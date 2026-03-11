"""Build control cache JSON for Stage1 amplitude normalization.

This module produces a JSON file compatible with `simulation.rho_curve --amplitude-control-cache`.

Input expectation (CSV): per-seed or per-run rows that include `max_amp`.
Required fields:
- max_amp (float)
Optional fields:
- players (int)  (if missing, use --players override)
- selection_strength (float) (if present, must be unique unless filtered)
- series (str), burn_in (int), tail (int or empty)

Output JSON shape:
{
  "series": "p"|"w",
  "burn_in": 1200,
  "tail": 600,
  "by_players": {
    "100": {"mean_max_amp": 0.01, "std_max_amp": 0.003},
    ...
  }
}
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path


def _parse_tail(s: str) -> int | None:
    v = str(s).strip().lower()
    if v in ("", "none", "null"):
        return None
    return int(v)


def _as_float(x: object) -> float:
    v = float(x)
    if not math.isfinite(v):
        raise ValueError(f"Non-finite float value: {x!r}")
    return v


def _as_int(x: object) -> int:
    return int(str(x).strip())


def _mean_std(values: list[float], *, ddof: int = 1) -> tuple[float, float]:
    if not values:
        return (0.0, 0.0)
    mean = sum(values) / float(len(values))
    if len(values) <= ddof:
        return (float(mean), 0.0)
    var = sum((x - mean) ** 2 for x in values) / float(len(values) - ddof)
    return (float(mean), float(math.sqrt(max(0.0, var))))


@dataclass(frozen=True, slots=True)
class ControlCache:
    series: str
    burn_in: int
    tail: int | None
    by_players: dict[int, dict[str, float]]

    def to_json_obj(self) -> dict:
        return {
            "series": str(self.series),
            "burn_in": int(self.burn_in),
            "tail": (int(self.tail) if self.tail is not None else None),
            "by_players": {
                str(int(k)): {
                    "mean_max_amp": float(v.get("mean_max_amp", 0.0)),
                    "std_max_amp": float(v.get("std_max_amp", 0.0)),
                }
                for k, v in sorted(self.by_players.items(), key=lambda kv: int(kv[0]))
            },
        }


def build_amplitude_control_cache_from_csv(
    paths: list[Path] | list[tuple[Path, int | None]],
    *,
    series: str,
    burn_in: int,
    tail: int | None,
    players_override: int | None = None,
    selection_strength: float | None = None,
    ddof: int = 1,
) -> ControlCache:
    """Build control cache from per-seed CSV rows.

    Each row contributes one `max_amp` sample to the (players) bucket.
    """
    if str(series) not in ("p", "w"):
        raise ValueError("series must be 'p' or 'w'")
    if int(burn_in) < 0:
        raise ValueError("burn_in must be >= 0")

    by_players_samples: dict[int, list[float]] = {}
    seen_selection_strengths: set[float] = set()

    for item in paths:
        if isinstance(item, tuple):
            path, per_file_players_override = item
        else:
            path, per_file_players_override = item, None
        with path.open(newline="") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames is None:
                continue
            for row in reader:
                if not row:
                    continue

                # Optional filter: selection_strength must match.
                if "selection_strength" in row and row.get("selection_strength", "") not in (None, ""):
                    k = _as_float(row["selection_strength"])
                    seen_selection_strengths.add(float(k))
                    if selection_strength is not None and abs(float(k) - float(selection_strength)) > 1e-12:
                        continue

                if "max_amp" not in row:
                    raise ValueError(f"Input CSV missing required column 'max_amp': {path}")
                amp = _as_float(row["max_amp"])

                players: int
                if "players" in row and row.get("players", "") not in (None, ""):
                    players = _as_int(row["players"])
                else:
                    local_override = per_file_players_override if per_file_players_override is not None else players_override
                    if local_override is None:
                        raise ValueError(
                            f"Input CSV missing 'players' and no --players override provided: {path}"
                        )
                    players = int(local_override)

                by_players_samples.setdefault(int(players), []).append(float(amp))

    if not by_players_samples:
        raise ValueError("No usable rows found (check filters / inputs).")

    if selection_strength is None and len(seen_selection_strengths) > 1:
        ks = ", ".join(f"{k:.8g}" for k in sorted(seen_selection_strengths))
        raise ValueError(
            "Input contains multiple selection_strength values; set --selection-strength to filter one. "
            f"Seen: {ks}"
        )

    by_players: dict[int, dict[str, float]] = {}
    for players, samples in by_players_samples.items():
        mean, std = _mean_std(samples, ddof=int(ddof))
        by_players[int(players)] = {"mean_max_amp": float(mean), "std_max_amp": float(std)}

    return ControlCache(series=str(series), burn_in=int(burn_in), tail=tail, by_players=by_players)


def main() -> None:
    p = argparse.ArgumentParser(description="Build control cache JSON for Stage1 amplitude normalization")
    p.add_argument(
        "--in",
        dest="inputs",
        type=str,
        action="append",
        required=True,
        help="Input CSV path. You may also use 'PLAYERS:path.csv' to override players for that input.",
    )
    p.add_argument("--out", type=Path, required=True, help="Output JSON path")
    p.add_argument("--series", choices=["p", "w"], required=True)
    p.add_argument("--burn-in", type=int, required=True)
    p.add_argument(
        "--tail",
        type=str,
        required=True,
        help="Tail length (int) or 'none' to indicate full post-burn window.",
    )
    p.add_argument(
        "--players",
        type=int,
        default=None,
        help="Override players when input CSV lacks a 'players' column (single-N input).",
    )
    p.add_argument(
        "--selection-strength",
        type=float,
        default=None,
        help="If input has selection_strength column, filter rows to this value.",
    )
    p.add_argument(
        "--ddof",
        type=int,
        default=1,
        help="Std ddof (1 = sample std, 0 = population std).",
    )
    args = p.parse_args()

    parsed_inputs: list[Path] | list[tuple[Path, int | None]]
    tmp: list[tuple[Path, int | None]] = []
    for spec in list(args.inputs):
        s = str(spec)
        players_hint: int | None = None
        path_str = s
        if ":" in s:
            left, right = s.split(":", 1)
            try:
                players_hint = int(left)
                path_str = right
            except Exception:
                players_hint = None
                path_str = s
        tmp.append((Path(path_str), players_hint))
    parsed_inputs = tmp

    cache = build_amplitude_control_cache_from_csv(
        parsed_inputs,
        series=str(args.series),
        burn_in=int(args.burn_in),
        tail=_parse_tail(str(args.tail)),
        players_override=(int(args.players) if args.players is not None else None),
        selection_strength=(float(args.selection_strength) if args.selection_strength is not None else None),
        ddof=int(args.ddof),
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        json.dump(cache.to_json_obj(), f, ensure_ascii=False, indent=2, sort_keys=True)
        f.write("\n")


if __name__ == "__main__":
    main()
