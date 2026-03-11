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
		"stage2_method",
		"stage2_prefilter",
		"power_ratio_kappa",
		"permutation_alpha",
		"permutation_resamples",
		"permutation_seed",
	]
	with path.open("w", newline="") as f:
		w = csv.DictWriter(f, fieldnames=fieldnames)
		w.writeheader()
		for n in (100, 300):
			for k, p3 in ((0.1, 0.0), (0.2, 0.2), (0.3, 0.8)):
				w.writerow(
					{
						"players": n,
						"selection_strength": k,
						"rho_minus_1": 0.01 * k,
						"p_level_3": p3,
						"mean_env_gamma": 0.0,
						"n_seeds": 10,
						"stage2_method": "fft_power_ratio",
						"stage2_prefilter": 1,
						"power_ratio_kappa": 8.0,
						"permutation_alpha": 0.05,
						"permutation_resamples": 200,
						"permutation_seed": 123,
					}
				)


def test_scaling_summary_includes_stage2_settings(tmp_path: Path) -> None:
	sweep = tmp_path / "sweep.csv"
	_write_sweep_csv(sweep)

	outdir = tmp_path / "out"
	rho_curve_scaling.main(["--in", str(sweep), "--outdir", str(outdir), "--prefix", "t"])

	summary = outdir / "t_summary.csv"
	assert summary.exists()
	with summary.open(newline="") as f:
		r = csv.DictReader(f)
		assert r.fieldnames is not None
		for k in (
			"stage2_method",
			"stage2_prefilter",
			"power_ratio_kappa",
			"permutation_alpha",
			"permutation_resamples",
			"permutation_seed",
		):
			assert k in r.fieldnames
		rows = list(r)
		assert rows, "summary should have at least one row"
		for row in rows:
			assert row["stage2_method"] == "fft_power_ratio"
			assert row["stage2_prefilter"] == "1"
			assert row["power_ratio_kappa"] == "8.0"
			assert row["permutation_alpha"] == "0.05"
			assert row["permutation_resamples"] == "200"
			assert row["permutation_seed"] == "123"
