from __future__ import annotations

import argparse
import json
import sys
from enum import Enum
from pathlib import Path
from typing import Iterable, List

from pydantic import ValidationError

from simulation_schema import (
    MultiRunSummary,
    validate_legacy_summary_payload,
    validate_provenance_payload,
)


def safe_print(message: str) -> None:
    try:
        print(message)
    except BrokenPipeError:
        raise SystemExit(0)


MULTI_RUN_REQUIRED_KEYS = {
    "schema_version",
    "summary_type",
    "protocol_name",
    "protocol_version",
    "generated_at_utc",
    "source_glob",
    "total_runs",
    "runs",
    "aggregate",
}

PROVENANCE_HINT_KEYS = {
    "seed",
    "condition",
    "dispatch_mode",
    "world_mode",
    "config",
    "tail_diagnostics",
    "round_diagnostics",
    "diagnostic_rule_version",
}

SUMMARY_HINT_KEYS = {
    "total_seeds",
    "l1",
    "l2",
    "l3",
    "outcomes",
    "seeds",
    "stage",
    "gate",
}


class PayloadType(str, Enum):
    MULTI_RUN_SUMMARY = "multi_run_summary"
    LEGACY_SUMMARY = "legacy_summary"
    PROVENANCE = "provenance"
    UNKNOWN = "unknown"


def is_multi_run_summary_payload(payload: dict) -> bool:
    return MULTI_RUN_REQUIRED_KEYS.issubset(set(payload.keys()))


def is_provenance_payload(payload: dict) -> bool:
    return bool(PROVENANCE_HINT_KEYS.intersection(set(payload.keys())))


def detect_payload_type(payload: dict) -> PayloadType:
    if is_multi_run_summary_payload(payload):
        return PayloadType.MULTI_RUN_SUMMARY
    if is_provenance_payload(payload):
        return PayloadType.PROVENANCE
    # Treat legacy/alternative summary JSON as summary candidates so they are
    # validated against the summary contract and fail with actionable details.
    summary_hint_count = len(SUMMARY_HINT_KEYS.intersection(set(payload.keys())))
    if summary_hint_count >= 2:
        return PayloadType.LEGACY_SUMMARY
    return PayloadType.UNKNOWN


def extract_extra_forbidden_fields(exc: ValidationError) -> list[str]:
    fields: list[str] = []
    for err in exc.errors():
        if err.get("type") == "extra_forbidden":
            loc = err.get("loc")
            if isinstance(loc, tuple) and loc:
                fields.append(str(loc[-1]))
    return sorted(set(fields))


def iter_json_files(inputs: List[str], recursive: bool) -> Iterable[Path]:
    for raw in inputs:
        path = Path(raw)
        if path.is_file() and path.suffix.lower() == ".json":
            yield path
            continue

        if path.is_dir():
            pattern = "**/*.json" if recursive else "*.json"
            yield from path.glob(pattern)
            continue

        # Also support glob patterns passed directly.
        yield from Path().glob(raw)


def validate_file(path: Path, *, strict_type: bool) -> tuple[str, PayloadType, str]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        return (f"[INVALID] {path}: JSON parse error: {exc}", PayloadType.UNKNOWN, "invalid")

    if not isinstance(payload, dict):
        return (
            f"[INVALID] {path}: top-level JSON must be an object",
            PayloadType.UNKNOWN,
            "invalid",
        )

    payload_type = detect_payload_type(payload)

    if payload_type == PayloadType.UNKNOWN:
        if strict_type:
            return (
                f"[INVALID] {path}: unknown JSON type (not summary/provenance)",
                payload_type,
                "invalid",
            )
        return (f"[SKIP] {path}: unknown JSON type", payload_type, "skip")

    if payload_type == PayloadType.PROVENANCE:
        try:
            validate_provenance_payload(payload)
            return (f"[OK][PROVENANCE] {path}", payload_type, "ok")
        except ValidationError as exc:
            extra_fields = extract_extra_forbidden_fields(exc)
            if extra_fields:
                return (
                    f"[INVALID][PROVENANCE] {path}: validation failed\n"
                    f"extra_forbidden fields: {', '.join(extra_fields)}\n{exc}",
                    payload_type,
                    "invalid",
                )
            return (
                f"[INVALID][PROVENANCE] {path}: validation failed\n{exc}",
                payload_type,
                "invalid",
            )
        except Exception as exc:  # noqa: BLE001
            return (
                f"[INVALID][PROVENANCE] {path}: unexpected validation error: {exc}",
                payload_type,
                "invalid",
            )

    if payload_type == PayloadType.LEGACY_SUMMARY:
        try:
            validate_legacy_summary_payload(payload)
            return (f"[OK][LEGACY_SUMMARY] {path}", payload_type, "ok")
        except ValidationError as exc:
            extra_fields = extract_extra_forbidden_fields(exc)
            if extra_fields:
                return (
                    f"[INVALID][LEGACY_SUMMARY] {path}: validation failed\n"
                    f"extra_forbidden fields: {', '.join(extra_fields)}\n{exc}",
                    payload_type,
                    "invalid",
                )
            return (
                f"[INVALID][LEGACY_SUMMARY] {path}: validation failed\n{exc}",
                payload_type,
                "invalid",
            )
        except Exception as exc:  # noqa: BLE001
            return (
                f"[INVALID][LEGACY_SUMMARY] {path}: unexpected validation error: {exc}",
                payload_type,
                "invalid",
            )

    if payload_type != PayloadType.MULTI_RUN_SUMMARY:
        if strict_type:
            return (
                f"[INVALID] {path}: unsupported payload type {payload_type.value}",
                payload_type,
                "invalid",
            )
        return (f"[SKIP] {path}: unsupported payload type {payload_type.value}", payload_type, "skip")

    try:
        MultiRunSummary.model_validate(payload)
    except ValidationError as exc:
        extra_fields = extract_extra_forbidden_fields(exc)
        if extra_fields:
            return (
                f"[INVALID] {path}: schema validation failed\n"
                f"extra_forbidden fields: {', '.join(extra_fields)}\n{exc}"
                ,
                payload_type,
                "invalid",
            )
        return (f"[INVALID] {path}: schema validation failed\n{exc}", payload_type, "invalid")
    except Exception as exc:  # noqa: BLE001
        return (
            f"[INVALID] {path}: unexpected validation error: {exc}",
            payload_type,
            "invalid",
        )

    return (f"[OK][SUMMARY] {path}", payload_type, "ok")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate B1 multi-run summary JSON files with Pydantic schema."
    )
    parser.add_argument(
        "inputs",
        nargs="+",
        help="JSON files, directories, or glob patterns to validate.",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="When an input is a directory, search recursively for JSON files.",
    )
    parser.add_argument(
        "--strict-type",
        action="store_true",
        help="Treat non-MultiRunSummary JSON files as invalid instead of skipped.",
    )
    parser.add_argument(
        "--only-type",
        choices=[
            PayloadType.LEGACY_SUMMARY.value,
            PayloadType.PROVENANCE.value,
            PayloadType.MULTI_RUN_SUMMARY.value,
        ],
        help=(
            "Validate only one payload type. Useful for CI gates per pipeline: "
            "legacy_summary, provenance, or multi_run_summary."
        ),
    )
    args = parser.parse_args()

    files = list(iter_json_files(args.inputs, recursive=args.recursive))
    unique_files = sorted({p.resolve() for p in files if p.exists() and p.is_file()})

    if not unique_files:
        safe_print("No JSON files found.")
        return 2

    failed = 0
    skipped = 0
    filtered_out = 0
    counts_by_type = {
        PayloadType.MULTI_RUN_SUMMARY: {"ok": 0, "invalid": 0, "skip": 0},
        PayloadType.LEGACY_SUMMARY: {"ok": 0, "invalid": 0, "skip": 0},
        PayloadType.PROVENANCE: {"ok": 0, "invalid": 0, "skip": 0},
        PayloadType.UNKNOWN: {"ok": 0, "invalid": 0, "skip": 0},
    }

    only_type: PayloadType | None = None
    if args.only_type is not None:
        only_type = PayloadType(args.only_type)

    for file_path in unique_files:
        if only_type is not None:
            try:
                payload = json.loads(file_path.read_text(encoding="utf-8"))
            except Exception:
                payload = None

            detected = (
                detect_payload_type(payload)
                if isinstance(payload, dict)
                else PayloadType.UNKNOWN
            )

            if detected != only_type:
                safe_print(
                    f"[SKIP][FILTER] {file_path}: detected {detected.value}, "
                    f"only-type={only_type.value}"
                )
                filtered_out += 1
                skipped += 1
                counts_by_type[detected]["skip"] += 1
                continue

        result, payload_type, status = validate_file(file_path, strict_type=args.strict_type)
        safe_print(result)
        counts_by_type[payload_type][status] += 1

        if status == "invalid":
            failed += 1
        elif status == "skip":
            skipped += 1

    safe_print("\n=== Stage 3 Health Report ===")
    if only_type is not None:
        safe_print(f"only_type: {only_type.value}")
    safe_print(f"filtered_out: {filtered_out}")
    for payload_type in (
        PayloadType.PROVENANCE,
        PayloadType.LEGACY_SUMMARY,
        PayloadType.MULTI_RUN_SUMMARY,
        PayloadType.UNKNOWN,
    ):
        bucket = counts_by_type[payload_type]
        safe_print(
            f"{payload_type.value}: ok={bucket['ok']} invalid={bucket['invalid']} skip={bucket['skip']}"
        )

    if failed > 0:
        safe_print(
            f"Validation failed: {failed}/{len(unique_files)} file(s) invalid, "
            f"{skipped} skipped."
        )
        return 1

    passed = len(unique_files) - skipped
    safe_print(f"Validation passed: {passed} valid, {skipped} skipped.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
