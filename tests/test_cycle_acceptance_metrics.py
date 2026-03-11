from __future__ import annotations

import csv
import sys
from pathlib import Path


def _run_simulation_to_csv(
	*,
	tmp_path: Path,
	args: list[str],
	out_name: str = "timeseries.csv",
) -> Path:
	from simulation import run_simulation

	out_csv = tmp_path / out_name
	argv = ["prog", *args, "--out", str(out_csv)]
	sys.argv = argv
	run_simulation.main()
	assert out_csv.exists()
	return out_csv


def _read_weight_series(out_csv: Path) -> dict[str, list[float]]:
	series = {"aggressive": [], "defensive": [], "balanced": []}
	with out_csv.open("r", encoding="utf-8", newline="") as f:
		reader = csv.DictReader(f)
		for row in reader:
			for s in series:
				series[s].append(float(row[f"w_{s}"]))
	return series


def test_cycle_metrics_pass_for_matrix_ab_cycle_regime(tmp_path: Path) -> None:
	from analysis import cycle_metrics

	# A deterministic, known-good configuration that exhibits oscillatory weights.
	out_csv = _run_simulation_to_csv(
		tmp_path=tmp_path,
		args=[
			"--players",
			"60",
			"--rounds",
			"220",
			"--seed",
			"123",
			"--payoff-mode",
			"matrix_ab",
			"--a",
			"0.12",
			"--b",
			"0.08",
			"--selection-strength",
			"0.9",
		],
	)

	weights = _read_weight_series(out_csv)

	# Stage 1: amplitude screen.
	stage1 = cycle_metrics.assess_stage1_amplitude(
		weights,
		burn_in=80,
		tail=120,
		threshold=0.03,
		aggregation="any",
	)
	assert stage1.passed, stage1

	# Stage 2: dominant frequency screen (autocorrelation proxy).
	stage2 = cycle_metrics.assess_stage2_frequency(
		weights,
		burn_in=80,
		tail=120,
		min_lag=2,
		max_lag=80,
		corr_threshold=0.25,
		aggregation="any",
	)
	assert stage2.passed, stage2


def test_cycle_metrics_fail_for_zero_matrix_control(tmp_path: Path) -> None:
	from analysis import cycle_metrics

	# Negative control: A=0 => U=0 always, rewards equal, weights stay constant at 1.
	out_csv = _run_simulation_to_csv(
		tmp_path=tmp_path,
		args=[
			"--players",
			"60",
			"--rounds",
			"220",
			"--seed",
			"123",
			"--payoff-mode",
			"matrix_ab",
			"--a",
			"0.0",
			"--b",
			"0.0",
			"--selection-strength",
			"0.9",
		],
	)

	weights = _read_weight_series(out_csv)

	stage1 = cycle_metrics.assess_stage1_amplitude(
		weights,
		burn_in=80,
		tail=120,
		threshold=0.03,
		aggregation="any",
	)
	assert not stage1.passed, stage1

	stage2 = cycle_metrics.assess_stage2_frequency(
		weights,
		burn_in=80,
		tail=120,
		min_lag=2,
		max_lag=80,
		corr_threshold=0.25,
		aggregation="any",
	)
	assert not stage2.passed, stage2


def test_cycle_metrics_stage3_direction_consistent_ccw(tmp_path: Path) -> None:
	from analysis import cycle_metrics

	# Stronger regime chosen via small grid scan for robust stage3 direction.
	out_cycle = _run_simulation_to_csv(
		tmp_path=tmp_path,
		out_name="cycle.csv",
		args=[
			"--players",
			"80",
			"--rounds",
			"260",
			"--seed",
			"123",
			"--payoff-mode",
			"matrix_ab",
			"--a",
			"0.35",
			"--b",
			"0.08",
			"--selection-strength",
			"2.5",
		],
	)
	weights_cycle = _read_weight_series(out_cycle)

	stage3 = cycle_metrics.assess_stage3_direction(
		weights_cycle,
		burn_in=90,
		tail=140,
		min_abs_net_rotation=0.02,
		consistency_threshold=0.65,
	)
	assert stage3.passed, stage3
	assert stage3.direction == "ccw", stage3
	assert stage3.net_rotation > 0.0

	# Negative control under the same runtime settings: A=0 => no direction.
	out_control = _run_simulation_to_csv(
		tmp_path=tmp_path,
		out_name="control.csv",
		args=[
			"--players",
			"80",
			"--rounds",
			"260",
			"--seed",
			"123",
			"--payoff-mode",
			"matrix_ab",
			"--a",
			"0.0",
			"--b",
			"0.0",
			"--selection-strength",
			"2.5",
		],
	)
	weights_control = _read_weight_series(out_control)

	stage3_control = cycle_metrics.assess_stage3_direction(
		weights_control,
		burn_in=90,
		tail=140,
		min_abs_net_rotation=0.02,
		consistency_threshold=0.65,
	)
	assert not stage3_control.passed, stage3_control
	assert stage3_control.direction is None


def test_classify_cycle_level_reaches_level3_with_turning_stage3(tmp_path: Path) -> None:
	from analysis.cycle_metrics import classify_cycle_level

	# Use the same robust regime as the Stage3 unit test, but validate the full
	# graded classifier reaches Level 3 under turning-based Stage3.
	out_cycle = _run_simulation_to_csv(
		tmp_path=tmp_path,
		out_name="cycle_level3.csv",
		args=[
			"--players",
			"80",
			"--rounds",
			"260",
			"--seed",
			"123",
			"--payoff-mode",
			"matrix_ab",
			"--a",
			"0.35",
			"--b",
			"0.08",
			"--selection-strength",
			"2.5",
		],
	)
	weights = _read_weight_series(out_cycle)

	res = classify_cycle_level(
		weights,
		burn_in=90,
		tail=140,
		amplitude_threshold=0.01,
		min_lag=2,
		max_lag=80,
		corr_threshold=0.2,
		eta=0.65,
		min_turn_strength=0.0,
		normalize_for_phase=True,
		stage3_method="turning",
		phase_smoothing=1,
	)
	assert res.level == 3, res
	assert res.stage3 is not None and res.stage3.passed
