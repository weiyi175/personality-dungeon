from pathlib import Path

from analysis import sensitivity


def _write_summary(path: Path, *, label: str, k50: float) -> None:
	path.parent.mkdir(parents=True, exist_ok=True)
	path.write_text(
		"stage2_method,stage2_prefilter,power_ratio_kappa,permutation_alpha,permutation_resamples,permutation_seed,players,k_min,p3_at_k_min,k50_boot_mean,k50_boot_std,k50_boot_ci_low,k50_boot_ci_high,k_first_positive,rho_minus_1_first_positive,k50,rho_minus_1_50,k50_censored_low,k90,rho_minus_1_90,p3_max,k_at_max,rho_minus_1_at_max,plateau_k_start,plateau_k_end,corr_rho_minus_1_p_level_3\n"
		+ f"autocorr_threshold,1,,,,,100,0.1,0.0,,,,,,,{k50},0.0,false,0.3,0.0,1.0,0.4,0.0,0.4,0.4,0.9\n",
		encoding="utf-8",
	)


def test_sensitivity_build_report(tmp_path: Path) -> None:
	a = tmp_path / "a.csv"
	b = tmp_path / "b.csv"
	_write_summary(a, label="a", k50=0.2)
	_write_summary(b, label="b", k50=0.3)
	runs = [
		sensitivity.Run(label="a", path=a, by_n=sensitivity.load_k50_map(a)),
		sensitivity.Run(label="b", path=b, by_n=sensitivity.load_k50_map(b)),
	]
	text = sensitivity.build_report(runs)
	assert "Per-N variability" in text
	assert "N=100" in text
