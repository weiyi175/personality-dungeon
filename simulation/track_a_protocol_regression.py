from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from numbers import Number
from pathlib import Path
from typing import Any

from simulation.pers_cal_baseline_gate60 import main as pers_cal_gate_main


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BASELINE_JSON = REPO_ROOT / "outputs" / "pers_cal_baseline_gate60_summary.json"
DEFAULT_RECHECK_JSON = REPO_ROOT / "outputs" / "pers_cal_baseline_gate60_recheck_42_101_summary.json"
DEFAULT_SUMMARY_JSON = REPO_ROOT / "outputs" / "track_a_protocol_regression_summary.json"
DEFAULT_SUMMARY_MD = REPO_ROOT / "outputs" / "track_a_protocol_regression_summary.md"

_A1_COMPARE_KEYS = [
    "total_seeds",
    "l1",
    "l2",
    "l3",
    "healthy",
    "mean_s3",
    "median_s3",
    "p10_s3",
    "mean_gamma",
]


@dataclass(frozen=True)
class StageResult:
    name: str
    command: str
    exit_code: int
    passed: bool
    details: dict[str, Any]


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _non_healthy_seed_set(summary: dict[str, Any]) -> list[int]:
    rows = summary.get("non_healthy", [])
    seeds: list[int] = []
    for row in rows:
        if isinstance(row, dict) and "seed" in row:
            seeds.append(int(row["seed"]))
    return sorted(seeds)


def _values_equal(a: Any, b: Any, *, float_tol: float) -> bool:
    if isinstance(a, Number) and isinstance(b, Number):
        return abs(float(a) - float(b)) <= float(float_tol)
    return a == b


def _compare_a1_recheck(
    baseline_summary: dict[str, Any],
    recheck_summary: dict[str, Any],
    *,
    float_tol: float,
) -> dict[str, Any]:
    mismatches: dict[str, dict[str, Any]] = {}

    for key in _A1_COMPARE_KEYS:
        base_val = baseline_summary.get(key)
        new_val = recheck_summary.get(key)
        if not _values_equal(base_val, new_val, float_tol=float_tol):
            mismatches[key] = {
                "baseline": base_val,
                "recheck": new_val,
            }

    base_gate = bool(baseline_summary.get("gate", {}).get("overall_pass", False))
    new_gate = bool(recheck_summary.get("gate", {}).get("overall_pass", False))
    if base_gate != new_gate:
        mismatches["gate.overall_pass"] = {
            "baseline": base_gate,
            "recheck": new_gate,
        }

    baseline_non_healthy = _non_healthy_seed_set(baseline_summary)
    recheck_non_healthy = _non_healthy_seed_set(recheck_summary)
    if baseline_non_healthy != recheck_non_healthy:
        mismatches["non_healthy_seeds"] = {
            "baseline": baseline_non_healthy,
            "recheck": recheck_non_healthy,
        }

    return {
        "keys_compared": list(_A1_COMPARE_KEYS),
        "baseline_overall_pass": base_gate,
        "recheck_overall_pass": new_gate,
        "baseline_non_healthy": baseline_non_healthy,
        "recheck_non_healthy": recheck_non_healthy,
        "float_tolerance": float(float_tol),
        "mismatches": mismatches,
        "passed": len(mismatches) == 0,
    }


def _run_pytest(*, python_bin: str, pytest_target: str) -> tuple[int, str, str]:
    proc = subprocess.run(
        [python_bin, "-m", "pytest", "-q", pytest_target],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    return int(proc.returncode), str(proc.stdout), str(proc.stderr)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_markdown(path: Path, summary: dict[str, Any]) -> None:
    lines: list[str] = []
    lines.append("# Track A Protocol Regression Summary")
    lines.append("")
    lines.append(f"- protocol: {summary['protocol']}")
    lines.append(f"- timestamp_utc: {summary['timestamp_utc']}")
    lines.append(f"- overall_pass: {summary['overall_pass']}")
    lines.append("")
    lines.append("## Stages")
    lines.append("")

    for stage in summary.get("stages", []):
        lines.append(f"### {stage['name']}")
        lines.append("")
        lines.append(f"- passed: {stage['passed']}")
        lines.append(f"- exit_code: {stage['exit_code']}")
        lines.append(f"- command: {stage['command']}")
        details = stage.get("details", {})
        if stage["name"] == "A1 Compare":
            lines.append(f"- compared_keys: {', '.join(details.get('keys_compared', []))}")
            lines.append(f"- mismatch_count: {len(details.get('mismatches', {}))}")
        if stage["name"] == "A2 Runtime Regression":
            lines.append(f"- pytest_target: {details.get('pytest_target')}")
        lines.append("")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Track A A3 protocol regression: A1 gate recheck + A2 runtime pytest"
    )
    parser.add_argument("--seeds", default="42..101")
    parser.add_argument("--baseline-summary-json", default=str(DEFAULT_BASELINE_JSON))
    parser.add_argument("--recheck-out-json", default=str(DEFAULT_RECHECK_JSON))
    parser.add_argument("--pytest-target", default="tests/test_personality_rl_runtime.py")
    parser.add_argument("--python-bin", default=sys.executable)
    parser.add_argument("--float-tol", type=float, default=1e-12)
    parser.add_argument("--summary-json", default=str(DEFAULT_SUMMARY_JSON))
    parser.add_argument("--summary-md", default=str(DEFAULT_SUMMARY_MD))
    args = parser.parse_args(argv)

    baseline_summary_path = Path(args.baseline_summary_json)
    recheck_summary_path = Path(args.recheck_out_json)

    stage_results: list[StageResult] = []

    # Stage A1: fixed baseline gate recheck.
    gate_cmd = (
        f"{args.python_bin} -m simulation.pers_cal_baseline_gate60 "
        f"--seeds {args.seeds} --out-json {recheck_summary_path}"
    )
    try:
        gate_exit = int(
            pers_cal_gate_main(
                [
                    "--seeds",
                    str(args.seeds),
                    "--out-json",
                    str(recheck_summary_path),
                ]
            )
        )
    except Exception as exc:  # pragma: no cover - defensive fallback
        gate_exit = 99
        stage_results.append(
            StageResult(
                name="A1 Gate Recheck",
                command=gate_cmd,
                exit_code=gate_exit,
                passed=False,
                details={"error": str(exc)},
            )
        )
    else:
        stage_results.append(
            StageResult(
                name="A1 Gate Recheck",
                command=gate_cmd,
                exit_code=gate_exit,
                passed=(gate_exit == 0 and recheck_summary_path.exists()),
                details={
                    "seeds": str(args.seeds),
                    "recheck_out_json": str(recheck_summary_path),
                    "recheck_json_exists": recheck_summary_path.exists(),
                },
            )
        )

    # Stage A1-compare: fixed baseline consistency check.
    compare_cmd = (
        "compare recheck summary against locked baseline "
        f"({baseline_summary_path} vs {recheck_summary_path})"
    )
    compare_exit = 0
    compare_payload: dict[str, Any]
    if not baseline_summary_path.exists():
        compare_exit = 3
        compare_payload = {
            "passed": False,
            "error": f"Missing baseline summary JSON: {baseline_summary_path}",
        }
    elif not recheck_summary_path.exists():
        compare_exit = 4
        compare_payload = {
            "passed": False,
            "error": f"Missing recheck summary JSON: {recheck_summary_path}",
        }
    else:
        base_summary = _load_json(baseline_summary_path)
        recheck_summary = _load_json(recheck_summary_path)
        compare_payload = _compare_a1_recheck(
            base_summary,
            recheck_summary,
            float_tol=float(args.float_tol),
        )
        if not bool(compare_payload.get("passed", False)):
            compare_exit = 1

    stage_results.append(
        StageResult(
            name="A1 Compare",
            command=compare_cmd,
            exit_code=compare_exit,
            passed=bool(compare_payload.get("passed", False)),
            details=compare_payload,
        )
    )

    # Stage A2: fixed runtime regression test command.
    pytest_cmd = f"{args.python_bin} -m pytest -q {args.pytest_target}"
    pytest_exit, pytest_stdout, pytest_stderr = _run_pytest(
        python_bin=str(args.python_bin),
        pytest_target=str(args.pytest_target),
    )
    stage_results.append(
        StageResult(
            name="A2 Runtime Regression",
            command=pytest_cmd,
            exit_code=pytest_exit,
            passed=(pytest_exit == 0),
            details={
                "pytest_target": str(args.pytest_target),
                "stdout": pytest_stdout,
                "stderr": pytest_stderr,
            },
        )
    )

    overall_pass = all(stage.passed for stage in stage_results)
    summary = {
        "protocol": "track_a_a3_protocol_regression_v1",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "inputs": {
            "seeds": str(args.seeds),
            "baseline_summary_json": str(baseline_summary_path),
            "recheck_out_json": str(recheck_summary_path),
            "pytest_target": str(args.pytest_target),
            "python_bin": str(args.python_bin),
            "float_tol": float(args.float_tol),
        },
        "stages": [asdict(stage) for stage in stage_results],
        "overall_pass": bool(overall_pass),
    }

    summary_json_path = Path(args.summary_json)
    summary_md_path = Path(args.summary_md)
    _write_json(summary_json_path, summary)
    _write_markdown(summary_md_path, summary)

    print("=" * 80)
    print("Track A Protocol Regression")
    print(f"overall_pass={overall_pass}")
    print(f"summary_json={summary_json_path}")
    print(f"summary_md={summary_md_path}")
    print("=" * 80)

    return 0 if overall_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())