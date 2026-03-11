import math

import pytest

from analysis.rho_curve_scaling import RhoCurvePoint, bayesian_k50_laplace


def _make_seq(*, players: int = 200, n_seeds: int = 30):
    # Synthetic binomial counts that roughly follow a logistic with k50 ~ 0.3.
    # We encode them via p_level_3=y/n to match the scaling pipeline input.
    ks = [0.20, 0.25, 0.30, 0.35, 0.40]
    ys = [0, 2, 15, 26, 30]  # out of 30
    seq = []
    for k, y in zip(ks, ys):
        p = float(y) / float(n_seeds)
        seq.append(
            RhoCurvePoint(
                players=int(players),
                k=float(k),
                rho_minus_1=0.0,
                p_level_3=float(p),
                mean_env_gamma=float("nan"),
                n_seeds=int(n_seeds),
            )
        )
    return seq


def test_bayesian_k50_laplace_smoke():
    seq = _make_seq()
    mean, std, lo, hi, n_eff = bayesian_k50_laplace(seq, prior_sigma=10.0, draws=4000, seed=123)

    assert mean is not None and lo is not None and hi is not None
    assert std is not None
    assert n_eff > 500  # should retain most draws

    # Expected k50 is around 0.3 for this synthetic curve.
    assert 0.20 < float(mean) < 0.40
    assert float(lo) < float(mean) < float(hi)

    # CI should cover ~0.30 (not too strict to avoid flaky due to randomness).
    assert float(lo) <= 0.30 <= float(hi)


def test_bayesian_k50_laplace_rejects_degenerate_all_failures():
    seq = [
        RhoCurvePoint(players=100, k=0.1, rho_minus_1=0.0, p_level_3=0.0, mean_env_gamma=0.0, n_seeds=20),
        RhoCurvePoint(players=100, k=0.2, rho_minus_1=0.0, p_level_3=0.0, mean_env_gamma=0.0, n_seeds=20),
        RhoCurvePoint(players=100, k=0.3, rho_minus_1=0.0, p_level_3=0.0, mean_env_gamma=0.0, n_seeds=20),
    ]
    with pytest.raises(ValueError):
        bayesian_k50_laplace(seq, draws=1000, seed=0)
