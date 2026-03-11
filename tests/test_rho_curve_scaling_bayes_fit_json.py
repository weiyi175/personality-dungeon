import csv
import json
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


def test_scaling_writes_bayes_fit_json(tmp_path: Path) -> None:
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
			"--bayes-method",
			"laplace",
			"--bayes-draws",
			"2000",
			"--bayes-seed",
			"0",
		]
	)

	bayes_path = outdir / "t_bayes_fit.json"
	assert bayes_path.exists()
	obj = json.loads(bayes_path.read_text(encoding="utf-8"))
	assert obj["bayes_method"] == "laplace"
	assert obj["bayes_draws"] == 2000
	assert "per_N" in obj and obj["per_N"], "per_N should not be empty"

	row0 = obj["per_N"][0]
	assert int(row0["players"]) == 100
	assert row0["k50_bayes_ci_low"] is not None
	assert row0["k50_bayes_ci_high"] is not None
	assert float(row0["k50_bayes_ci_low"]) < float(row0["k50_bayes_ci_high"])
	assert int(row0["k50_bayes_n_eff"]) > 200
