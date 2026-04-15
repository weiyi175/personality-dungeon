from __future__ import annotations

import csv
from pathlib import Path

from simulation.b2_island_deme import run_b2_scout


def test_b2_scout_writes_outputs_and_diagnostics(tmp_path: Path) -> None:
	out_root = tmp_path / "b2"
	summary_tsv = tmp_path / "b2_summary.tsv"
	combined_tsv = tmp_path / "b2_combined.tsv"
	decision_md = tmp_path / "b2_decision.md"

	result = run_b2_scout(
		seeds=[45, 47],
		num_demes_list=[3],
		migration_fractions=[0.05],
		migration_intervals=[100],
		out_root=out_root,
		summary_tsv=summary_tsv,
		combined_tsv=combined_tsv,
		decision_md=decision_md,
		players=30,
		rounds=120,
		burn_in=30,
		tail=60,
		memory_kernel=3,
	)

	assert Path(result["summary_tsv"]).exists()
	assert Path(result["combined_tsv"]).exists()
	assert Path(result["decision_md"]).exists()

	# Check summary TSV has expected columns
	with summary_tsv.open(newline="", encoding="utf-8") as handle:
		summary_rows = list(csv.DictReader(handle, delimiter="\t"))
		assert summary_rows
		for row in summary_rows:
			assert row["mean_inter_deme_phase_spread"] != ""
			assert row["max_inter_deme_phase_spread"] != ""
			assert row["mean_inter_deme_growth_cosine"] != ""
			assert row["phase_amplitude_stability"] != ""
			assert row["num_demes"] != ""

	# Check combined TSV: 1 control (M=1) + 1 active (M=3)
	with combined_tsv.open(newline="", encoding="utf-8") as handle:
		rows = list(csv.DictReader(handle, delimiter="\t"))
		assert len(rows) == 2
		conditions = {row["condition"] for row in rows}
		assert "g2_M1_control" in conditions
		for row in rows:
			assert row["mean_inter_deme_phase_spread"] != ""
			assert row["max_inter_deme_phase_spread"] != ""
			assert row["mean_inter_deme_growth_cosine"] != ""
			assert row["phase_amplitude_stability"] != ""
			assert Path(row["representative_deme_simplex_png"]).exists()
			assert Path(row["representative_phase_amplitude_png"]).exists()
		# Active condition should have a verdict
		active = [r for r in rows if r["condition"] != "g2_M1_control"]
		assert len(active) == 1
		assert active[0]["verdict"] in {"pass", "weak_positive", "fail"}
		assert active[0]["short_scout_pass"] in {"yes", "no"}
		assert active[0]["hard_stop_fail"] in {"yes", "no"}

	# Check decision markdown
	decision_text = decision_md.read_text(encoding="utf-8")
	assert "mean_inter_deme_phase_spread" in decision_text
	assert "mean_inter_deme_growth_cosine" in decision_text
	assert "phase_amplitude_stability" in decision_text
	assert "topology" in decision_text.lower() or "B2" in decision_text


def test_b2_m1_control_has_zero_spread(tmp_path: Path) -> None:
	"""M=1 control should have zero inter-deme phase spread (single deme)."""
	out_root = tmp_path / "b2c"
	summary_tsv = tmp_path / "b2c_summary.tsv"
	combined_tsv = tmp_path / "b2c_combined.tsv"
	decision_md = tmp_path / "b2c_decision.md"

	result = run_b2_scout(
		seeds=[45],
		num_demes_list=[3],
		migration_fractions=[0.05],
		migration_intervals=[100],
		out_root=out_root,
		summary_tsv=summary_tsv,
		combined_tsv=combined_tsv,
		decision_md=decision_md,
		players=30,
		rounds=80,
		burn_in=20,
		tail=40,
		memory_kernel=3,
	)

	with combined_tsv.open(newline="", encoding="utf-8") as handle:
		rows = list(csv.DictReader(handle, delimiter="\t"))
		control = next(r for r in rows if r["condition"] == "g2_M1_control")
		# M=1 has no inter-deme spread
		assert float(control["mean_inter_deme_phase_spread"]) == 0.0
		assert float(control["max_inter_deme_phase_spread"]) == 0.0
