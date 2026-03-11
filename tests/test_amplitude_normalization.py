import pytest

from analysis.cycle_metrics import assess_stage1_amplitude


def test_stage1_control_scaled_threshold_blocks_control_scale_amplitude():
	# Raw amplitude is 0.05; with factor=3 and control_reference=0.05,
	# effective threshold is 0.15 -> should fail.
	props = {
		"aggressive": [0.0, 0.05],
		"defensive": [0.0, 0.0],
		"balanced": [0.0, 0.0],
	}
	res = assess_stage1_amplitude(
		props,
		threshold=0.0,  # ignored under normalization != none
		aggregation="any",
		normalization="control_mean",
		control_reference=0.05,
		threshold_factor=3.0,
	)
	assert res.passed is False
	assert res.threshold == pytest.approx(0.15)
	assert res.normalization == "control_mean"
	assert res.threshold_factor == pytest.approx(3.0)
	assert res.control_reference == pytest.approx(0.05)
	assert res.normalized_amplitudes is not None
	assert res.normalized_amplitudes["aggressive"] == pytest.approx(1.0)


def test_stage1_control_scaled_threshold_allows_super_control_amplitude():
	props = {
		"aggressive": [0.0, 0.2],  # amp 0.2
		"defensive": [0.0, 0.0],
		"balanced": [0.0, 0.0],
	}
	res = assess_stage1_amplitude(
		props,
		threshold=0.0,
		aggregation="any",
		normalization="control_mean",
		control_reference=0.05,
		threshold_factor=3.0,
	)
	assert res.passed is True
	assert res.threshold == pytest.approx(0.15)
	assert res.normalized_amplitudes is not None
	assert res.normalized_amplitudes["aggressive"] == pytest.approx(4.0)


def test_stage1_control_scaled_requires_positive_reference():
	props = {"aggressive": [0.0, 0.2]}
	with pytest.raises(ValueError):
		assess_stage1_amplitude(
			props,
			normalization="control_mean",
			control_reference=0.0,
			threshold_factor=3.0,
		)
