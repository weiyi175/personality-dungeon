import sys
from pathlib import Path

from analysis.cycle_metrics import classify_cycle_level
from analysis.visualization import load_timeseries_csv


def test_control_baseline_matrix_ab_zero_zero_is_level_0(tmp_path: Path):
	"""False Positive Guard.

	For payoff_mode=matrix_ab with a=b=0, all players receive the same reward.
	Replicator update should keep weights at 1, so no oscillation and cycle_level=0.

	We classify on the w_* time series (weights), which should be exactly constant.
	"""

	out_csv = tmp_path / "control.csv"

	argv_backup = sys.argv[:]
	try:
		sys.argv = [
			"simulation.run_simulation",
			"--players",
			"50",
			"--rounds",
			"50",
			"--seed",
			"123",
			"--payoff-mode",
			"matrix_ab",
			"--a",
			"0.0",
			"--b",
			"0.0",
			"--selection-strength",
			"0.05",
			"--out",
			str(out_csv),
		]
		import simulation.run_simulation as sim

		sim.main()
	finally:
		sys.argv = argv_backup

	ts = load_timeseries_csv(out_csv)
	res = classify_cycle_level(
		ts.weights,
		burn_in=0,
		# thresholds deliberately small but >0
		amplitude_threshold=1e-6,
		corr_threshold=0.3,
		eta=0.6,
	)
	assert res.level == 0
	assert res.stage1.passed is False
	assert res.stage2 is None
	assert res.stage3 is None
