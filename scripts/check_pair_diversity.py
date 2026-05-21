from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = REPO_ROOT / "outputs" / "personality_text_pairs.tsv"
DEFAULT_OUTPUT = REPO_ROOT / "outputs" / "pair_diversity_report.json"

PERSONALITY_BASIS: tuple[str, ...] = (
    "impulsiveness",
    "assertiveness",
    "optimism",
    "risk_aversion",
    "suspicion",
    "endurance",
    "randomness",
    "stability_seeking",
    "curiosity",
)


@dataclass(frozen=True)
class Record:
    text: str
    length: int
    source: str
    vector: dict[str, float]


def _normalize_text(text: str) -> str:
    return text.replace("\r", " ").replace("\n", " ").replace("\t", " ").strip()


def _safe_float(value: str | None) -> float:
    if value is None or value == "":
        return 0.0
    return float(value)


def _normalize_source(value: str | None) -> str:
    source = _normalize_text(value or "")
    return source or "unknown"


def _load_tsv(path: Path) -> list[Record]:
    records: list[Record] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            text = _normalize_text(row.get("text", ""))
            if not text:
                continue
            vector = {trait: _safe_float(row.get(trait)) for trait in PERSONALITY_BASIS}
            records.append(
                Record(
                    text=text,
                    length=int(float(row.get("length", len(text)))),
                    source=_normalize_source(row.get("source")),
                    vector=vector,
                )
            )
    return records


def _load_jsonl(path: Path) -> list[Record]:
    records: list[Record] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            text = _normalize_text(str(row.get("text", "")))
            if not text:
                continue
            vector_obj = row.get("vector", {})
            vector = {trait: float(vector_obj.get(trait, 0.0)) for trait in PERSONALITY_BASIS}
            records.append(
                Record(
                    text=text,
                    length=int(row.get("length", len(text))),
                    source=_normalize_source(str(row.get("source", ""))),
                    vector=vector,
                )
            )
    return records


def _load_records(path: Path) -> list[Record]:
    if not path.exists():
        return []
    if path.suffix.lower() == ".jsonl":
        return _load_jsonl(path)
    return _load_tsv(path)


def _parse_source_filter(value: str | None) -> set[str] | None:
    if value is None:
        return None
    sources = {item.strip() for item in value.split(",") if item.strip()}
    return sources or None


def _length_bucket(length: int) -> str:
    if 1 <= length <= 5:
        return "1-5"
    if 6 <= length <= 12:
        return "6-12"
    if 13 <= length <= 20:
        return "13-20"
    if length <= 0:
        return "empty"
    return ">20"


def _mean_std(values: list[float]) -> dict[str, float]:
    if not values:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
    if len(values) == 1:
        value = float(values[0])
        return {"mean": value, "std": 0.0, "min": value, "max": value}
    return {
        "mean": float(statistics.mean(values)),
        "std": float(statistics.pstdev(values)),
        "min": float(min(values)),
        "max": float(max(values)),
    }


def _build_report(records: list[Record], *, input_path: Path) -> dict[str, Any]:
    total_rows = len(records)
    unique_texts = len({record.text for record in records})
    duplicate_rows = total_rows - unique_texts
    length_mismatches = sum(1 for record in records if record.length != len(record.text))
    length_buckets = Counter(_length_bucket(record.length) for record in records)
    source_counts = Counter(record.source for record in records)

    per_trait_values: dict[str, list[float]] = defaultdict(list)
    out_of_range_counts: dict[str, int] = {trait: 0 for trait in PERSONALITY_BASIS}
    for record in records:
        for trait in PERSONALITY_BASIS:
            value = record.vector.get(trait, 0.0)
            per_trait_values[trait].append(value)
            if not (-1.0 <= value <= 1.0):
                out_of_range_counts[trait] += 1

    per_trait_stats = {trait: _mean_std(values) for trait, values in per_trait_values.items()}
    bucket_ratios = {bucket: (count / total_rows if total_rows else 0.0) for bucket, count in length_buckets.items()}

    gate = {
        "row_count_ge_1000": total_rows >= 1000,
        "all_trait_std_ge_0_3": all(stats["std"] >= 0.3 for stats in per_trait_stats.values()),
        "length_bucket_coverage_ge_0_2": all(
            bucket_ratios.get(bucket, 0.0) >= 0.2 for bucket in ("1-5", "6-12", "13-20")
        ),
        "no_length_mismatch": length_mismatches == 0,
        "no_out_of_range_values": all(count == 0 for count in out_of_range_counts.values()),
        "overall_pass": False,
    }
    gate["overall_pass"] = all(value for key, value in gate.items() if key != "overall_pass")

    return {
        "input_path": str(input_path),
        "row_count": total_rows,
        "unique_text_count": unique_texts,
        "duplicate_row_count": duplicate_rows,
        "length_mismatch_count": length_mismatches,
        "length_buckets": dict(length_buckets),
        "length_bucket_ratios": bucket_ratios,
        "source_counts": dict(source_counts),
        "per_trait_stats": per_trait_stats,
        "out_of_range_counts": out_of_range_counts,
        "gate": gate,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Check A-stage diversity for personality weak labels")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="Input TSV or JSONL path")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Report JSON path")
    parser.add_argument("--source-filter", default=None, help="Comma-separated source names to include")
    args = parser.parse_args(argv)

    records = _load_records(args.input)
    source_filter = _parse_source_filter(args.source_filter)
    if source_filter is not None:
        records = [record for record in records if record.source in source_filter]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    report = _build_report(records, input_path=args.input)
    report["source_filter"] = sorted(source_filter) if source_filter is not None else None
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["gate"]["overall_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
