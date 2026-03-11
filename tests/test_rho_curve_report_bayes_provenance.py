import csv
import json
from pathlib import Path

from analysis import rho_curve_report


def _write_min_summary(path: Path, *, k50: float) -> None:
	path.parent.mkdir(parents=True, exist_ok=True)
	fieldnames = [
		"players",
		"k_min",
		"p3_at_k_min",
		"k_first_positive",
		"rho_minus_1_first_positive",
		"k50",
		"rho_minus_1_50",
		"k50_censored_low",
		"k90",
		"rho_minus_1_90",
		"p3_max",
		"k_at_max",
		"rho_minus_1_at_max",
		"plateau_k_start",
		"plateau_k_end",
		"corr_rho_minus_1_p_level_3",
	]
	with path.open("w", newline="") as f:
		w = csv.DictWriter(f, fieldnames=fieldnames)
		w.writeheader()
		w.writerow(
			{
				"players": 100,
				"k_min": 0.1,
				"p3_at_k_min": 0.0,
				"k_first_positive": 0.2,
				"rho_minus_1_first_positive": 0.0,
				"k50": k50,
				"rho_minus_1_50": 0.0,
				"k50_censored_low": "false",
				"k90": 0.3,
				"rho_minus_1_90": 0.0,
				"p3_max": 1.0,
				"k_at_max": 0.4,
				"rho_minus_1_at_max": 0.0,
				"plateau_k_start": 0.4,
				"plateau_k_end": 0.4,
				"corr_rho_minus_1_p_level_3": 0.9,
			}
		)


def test_rho_curve_report_embeds_bayes_provenance(tmp_path: Path) -> None:
	summary = tmp_path / "t_summary.csv"
	_write_min_summary(summary, k50=0.2)

	bayes_fit = tmp_path / "t_bayes_fit.json"
	bayes_fit.write_text(
		json.dumps(
			{
				"bayes_method": "laplace",
				"bayes_prior_sigma": 2.0,
				"bayes_draws": 500,
				"bayes_seed": 123,
				"bayes_assume_increasing": True,
				"per_N": [
					{
						"players": 100,
						"k50_bayes_mean": 0.21,
						"k50_bayes_ci_low": 0.18,
						"k50_bayes_ci_high": 0.25,
						"k50_bayes_n_eff": 500,
					}
				],
			},
			indent=2,
		),
		encoding="utf-8",
	)

	notes_out = tmp_path / "notes.md"
	paper_out = tmp_path / "paper.md"

	rho_curve_report.main(
		[
			"--summary",
			str(summary),
			"--out",
			str(notes_out),
			"--out-paper",
			str(paper_out),
		]
	)

	notes = notes_out.read_text(encoding="utf-8")
	paper = paper_out.read_text(encoding="utf-8")
	for text in (notes, paper):
		assert "Bayesian k50 provenance:" in text
		assert str(bayes_fit) in text
		assert "method: laplace" in text
