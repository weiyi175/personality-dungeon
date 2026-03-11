import json

import pytest

from simulation.rho_curve import _load_amplitude_control_cache, _resolve_amplitude_control_reference


def test_load_amplitude_control_cache_and_resolve_reference(tmp_path):
	path = tmp_path / "control_cache.json"
	path.write_text(
		json.dumps(
			{
				"series": "p",
				"burn_in": 10,
				"tail": 20,
				"by_players": {
					"100": {"mean_max_amp": 0.01, "std_max_amp": 0.003},
					"200": {"mean_max_amp": 0.02, "std_max_amp": 0.006},
				},
			}
		),
		encoding="utf-8",
	)
	cache = _load_amplitude_control_cache(
		path,
		expected_series="p",
		expected_burn_in=10,
		expected_tail=20,
	)
	assert cache[100]["mean_max_amp"] == pytest.approx(0.01)
	assert cache[200]["std_max_amp"] == pytest.approx(0.006)

	ref_mean = _resolve_amplitude_control_reference(
		players=100,
		normalization="control_mean",
		override_reference=None,
		cache=cache,
	)
	assert ref_mean == pytest.approx(0.01)

	ref_std = _resolve_amplitude_control_reference(
		players=200,
		normalization="control_std",
		override_reference=None,
		cache=cache,
	)
	assert ref_std == pytest.approx(0.006)

	ref_override = _resolve_amplitude_control_reference(
		players=200,
		normalization="control_mean",
		override_reference=0.123,
		cache=cache,
	)
	assert ref_override == pytest.approx(0.123)


def test_load_amplitude_control_cache_rejects_mismatch(tmp_path):
	path = tmp_path / "control_cache.json"
	path.write_text(
		json.dumps(
			{
				"series": "w",
				"burn_in": 10,
				"tail": 20,
				"by_players": {"100": {"mean_max_amp": 0.01}},
			}
		),
		encoding="utf-8",
	)
	with pytest.raises(ValueError):
		_load_amplitude_control_cache(
			path,
			expected_series="p",
			expected_burn_in=10,
			expected_tail=20,
		)
