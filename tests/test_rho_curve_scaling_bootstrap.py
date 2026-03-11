import csv
from pathlib import Path

from analysis import rho_curve_scaling


def _write_sweep_csv(path: Path) -> None:
	path.parent.mkdir(parents=True, exist_ok=True)
	fieldnames = [
		"players",
		"selection_strength",
		"rho_minus_1",
		"p_level_3",
		"mean_env_gamma",
		"n_seeds",
	]
	with path.open("w", newline="") as f:
		w = csv.DictWriter(f, fieldnames=fieldnames)
		w.writeheader()
		# A simple increasing curve that crosses 0.5 around k~0.25.
		for k, p3 in ((0.1, 0.1), (0.2, 0.4), (0.3, 0.7), (0.4, 0.9)):
			w.writerow(
				{
					"players": 100,
					"selection_strength": k,
					"rho_minus_1": 0.01 * k,
					"p_level_3": p3,
					"mean_env_gamma": 0.0,
					"n_seeds": 30,
				}
			)


def test_scaling_bootstrap_writes_ci_columns(tmp_path: Path) -> None:
	sweep = tmp_path / "sweep.csv"
	_write_sweep_csv(sweep)
	outdir = tmp_path / "out"
	rho_curve_scaling.main(
		[
			"--in",
			str(sweep),
			"--outdir",
			str(outdir),
			"--prefix",
			"t",
			"--bootstrap-resamples",
			"50",
			"--bootstrap-seed",
			"0",
		]
	)

	summary = outdir / "t_summary.csv"
	with summary.open(newline="") as f:
		r = csv.DictReader(f)
		rows = list(r)
	assert rows
	row = rows[0]
	assert row["k50_boot_ci_low"].strip() != ""
	assert row["k50_boot_ci_high"].strip() != ""
