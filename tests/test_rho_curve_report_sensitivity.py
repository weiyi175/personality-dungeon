import csv
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


def test_rho_curve_report_embeds_sensitivity(tmp_path: Path) -> None:
	main_summary = tmp_path / "main_summary.csv"
	alt_summary = tmp_path / "alt_summary.csv"
	_write_min_summary(main_summary, k50=0.2)
	_write_min_summary(alt_summary, k50=0.3)

	notes_out = tmp_path / "notes.md"
	paper_out = tmp_path / "paper.md"

	rho_curve_report.main(
		[
			"--summary",
			str(main_summary),
			"--out",
			str(notes_out),
			"--out-paper",
			str(paper_out),
			"--sensitivity-summary",
			f"alt:{alt_summary}",
		]
	)

	notes = notes_out.read_text(encoding="utf-8")
	paper = paper_out.read_text(encoding="utf-8")
	assert "Sensitivity report" in notes
	assert "Per-N variability" in notes
	assert "Appendix: sensitivity" in paper
	assert "k50 variability across runs" in paper
