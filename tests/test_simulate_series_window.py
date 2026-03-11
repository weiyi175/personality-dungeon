import pytest

from pathlib import Path

from simulation.run_simulation import SimConfig, simulate, simulate_series_window


@pytest.mark.parametrize("series", ["p", "w"])
def test_simulate_series_window_matches_simulate_rows(series: str):
	cfg = SimConfig(
		n_players=30,
		n_rounds=50,
		seed=123,
		payoff_mode="matrix_ab",
		popularity_mode="sampled",
		gamma=0.1,
		epsilon=0.0,
		a=1.0,
		b=1.2,
		init_bias=0.0,
		evolution_mode="sampled",
		payoff_lag=1,
		selection_strength=0.02,
		out_csv=Path("outputs") / "_ignored.csv",
	)

	burn_in = 10
	tail = 20

	series_window = simulate_series_window(cfg, series=series, burn_in=burn_in, tail=tail)
	strategy_space, rows = simulate(cfg)

	# The window semantics should match analysis.cycle_metrics._slice_series
	begin = max(burn_in, cfg.n_rounds - tail)
	window_rows = rows[begin:]

	assert set(series_window.keys()) == set(strategy_space)
	for s in strategy_space:
		if series == "p":
			expected = [float(r[f"p_{s}"]) for r in window_rows]
		else:
			expected = [float(r[f"w_{s}"]) for r in window_rows]
		assert series_window[s] == pytest.approx(expected, abs=0.0)
