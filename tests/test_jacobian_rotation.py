from __future__ import annotations

import cmath

from analysis.jacobian_rotation import (
	_durand_kerner_roots,
	lagged_discrete_update,
	ode_jacobian_at_uniform,
	uv_to_simplex,
)


def test_durand_kerner_quartic_real_roots() -> None:
	# (z-1)(z-2)(z-3)(z-4) = z^4 - 10 z^3 + 35 z^2 - 50 z + 24
	coeffs = [24.0, -50.0, 35.0, -10.0, 1.0]
	roots = _durand_kerner_roots([complex(c) for c in coeffs], max_iter=400, tol=1e-14)
	got = sorted([round(r.real, 6) for r in roots], key=float)
	assert got == [1.0, 2.0, 3.0, 4.0]
	assert all(abs(r.imag) < 1e-6 for r in roots)


def test_lagged_update_has_uniform_fixed_point() -> None:
	# For any (a,b), uniform makes all strategies equal payoff => no selection gradient.
	p = (1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0)
	for (a, b, k) in [(0.1, 0.0, 2.5), (0.35, 0.08, 1.0), (0.2, 0.2, 0.7)]:
		p2 = lagged_discrete_update(p_cur=p, p_prev=p, a=a, b=b, selection_strength=k)
		assert all(abs(p2[i] - p[i]) < 1e-12 for i in range(3))


def test_ode_jacobian_is_pure_rotation_when_a_equals_b() -> None:
	# For antisymmetric matrix (a=b), the replicator ODE around the interior equilibrium
	# should have a complex conjugate pair with ~0 real part (center / pure rotation).
	rep = ode_jacobian_at_uniform(a=0.2, b=0.2, h=2e-6)
	lam1, lam2 = rep.eigvals
	assert abs(lam1.real) < 1e-5
	assert abs(lam2.real) < 1e-5
	assert abs(lam1.imag) > 1e-6
	assert abs(lam2.imag) > 1e-6
	assert abs(lam1 - lam2.conjugate()) < 1e-5


def test_uv_to_simplex_is_valid_simplex_point() -> None:
	x = uv_to_simplex(0.4, 0.4)
	assert all(v > 0 for v in x)
	assert abs(sum(x) - 1.0) < 1e-12
