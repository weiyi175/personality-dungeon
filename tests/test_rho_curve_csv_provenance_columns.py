from simulation.rho_curve import SWEEP_CSV_FIELDNAMES


def test_rho_curve_sweep_csv_includes_protocol_and_stage3_provenance_columns():
	# Protocol
	for name in ("payoff_mode", "gamma", "epsilon", "seeds", "k_grid", "players_grid"):
		assert name in SWEEP_CSV_FIELDNAMES

	# Stage3 settings
	for name in ("stage3_method", "phase_smoothing", "stage3_window", "stage3_step", "stage3_quantile"):
		assert name in SWEEP_CSV_FIELDNAMES

	# Sanity: legacy columns remain present
	for name in ("players", "selection_strength", "series", "rounds", "burn_in", "tail"):
		assert name in SWEEP_CSV_FIELDNAMES
