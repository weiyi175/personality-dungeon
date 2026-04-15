from __future__ import annotations

import csv
from pathlib import Path

from simulation.b4_state_k import run_b4_scout


def test_b4_scout_writes_outputs(tmp_path: Path) -> None:
	out_root = tmp_path / "b4"
	summary_tsv = tmp_path / "b4_summary.tsv"
	combined_tsv = tmp_path / "b4_combined.tsv"
	decision_md = tmp_path / "b4_decision.md"

	result = run_b4_scout(
		seeds=[45, 47],
		beta_state_ks=[0.0, 0.6],
		k_bases=[0.06],
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

	with combined_tsv.open(newline="", encoding="utf-8") as handle:
		rows = list(csv.DictReader(handle, delimiter="\t"))
		assert len(rows) == 2
		assert {row["condition"] for row in rows} == {"beta0p00_k0p06", "beta0p60_k0p06"}
		assert any(row["is_control"] == "yes" for row in rows)
		assert any(row["verdict"] in {"pass", "weak_positive", "fail"} for row in rows if row["is_control"] == "no")
		for row in rows:
			assert Path(row["representative_simplex_png"]).exists()
			assert row["mean_k_clamped_ratio"] != ""