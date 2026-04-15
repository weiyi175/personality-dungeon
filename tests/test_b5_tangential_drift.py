from __future__ import annotations

import csv
from pathlib import Path

from simulation.b5_tangential_drift import run_b5_scout


def test_b5_scout_writes_g1_and_g2_diagnostics(tmp_path: Path) -> None:
	out_root = tmp_path / "b5"
	summary_tsv = tmp_path / "b5_summary.tsv"
	combined_tsv = tmp_path / "b5_combined.tsv"
	decision_md = tmp_path / "b5_decision.md"

	result = run_b5_scout(
		seeds=[45, 47],
		tangential_drift_deltas=[0.0, 0.003],
		out_root=out_root,
		summary_tsv=summary_tsv,
		combined_tsv=combined_tsv,
		decision_md=decision_md,
		players=30,
		rounds=120,
		burn_in=30,
		tail=60,
		memory_kernel=3,
		enable_events=False,
	)

	assert Path(result["summary_tsv"]).exists()
	assert Path(result["combined_tsv"]).exists()
	assert Path(result["decision_md"]).exists()

	with summary_tsv.open(newline="", encoding="utf-8") as handle:
		summary_rows = list(csv.DictReader(handle, delimiter="\t"))
		assert summary_rows
		for row in summary_rows:
			assert row["drift_contribution_ratio"] != ""
			assert row["mean_drift_norm"] != ""
			assert row["mean_effective_delta_growth_ratio"] != ""
			assert row["mean_tangential_alignment"] != ""
			assert row["phase_amplitude_stability"] != ""

	with combined_tsv.open(newline="", encoding="utf-8") as handle:
		rows = list(csv.DictReader(handle, delimiter="\t"))
		assert len(rows) == 4
		assert {row["condition"] for row in rows} == {
			"g1_mean_field_delta0p000",
			"g1_mean_field_delta0p003",
			"g2_sampled_delta0p000",
			"g2_sampled_delta0p003",
		}
		for row in rows:
			assert row["drift_contribution_ratio"] != ""
			assert row["mean_drift_norm"] != ""
			assert row["mean_effective_delta_growth_ratio"] != ""
			assert row["mean_tangential_alignment"] != ""
			assert row["phase_amplitude_stability"] != ""
			assert Path(row["representative_simplex_png"]).exists()
			assert Path(row["representative_phase_amplitude_png"]).exists()
			assert Path(row["representative_drift_vector_rose_png"]).exists()
		g1_active = next(row for row in rows if row["condition"] == "g1_mean_field_delta0p003")
		g2_active = next(row for row in rows if row["condition"] == "g2_sampled_delta0p003")
		assert g1_active["g1_gate_pass"] in {"yes", "no"}
		assert g1_active["g1_drift_gate_pass"] in {"yes", "no"}
		assert g2_active["g1_gate_pass"] in {"yes", "no"}
		assert g2_active["verdict"] in {"pass", "weak_positive", "fail", "blocked_by_g1"}

	decision_text = decision_md.read_text(encoding="utf-8")
	assert "drift_contribution_ratio" in decision_text
	assert "mean_effective_delta_growth_ratio" in decision_text
	assert "mean_tangential_alignment" in decision_text
	assert "phase_amplitude_stability" in decision_text
	assert "drift_rose_png" in decision_text