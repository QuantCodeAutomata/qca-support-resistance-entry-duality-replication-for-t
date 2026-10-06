import numpy as np
import pytest
from src.roots_and_crossing import solve_roots, compute_kappas, crossing_matrix, validate_crossing_inverse

R, MU, SIGMA = 0.05, 0.01, 0.2


def test_roots_ordering():
    a1, a2 = solve_roots(R, MU, SIGMA)
    assert a1 < 0 < 1 < a2


def test_roots_satisfy_ode():
    a1, a2 = solve_roots(R, MU, SIGMA)
    for alpha in [a1, a2]:
        residual = 0.5 * SIGMA**2 * alpha * (alpha - 1) + MU * alpha - R
        assert abs(residual) < 1e-14, f'ODE residual too large: {residual}'


def test_kappas_sum_to_one():
    a1, a2 = solve_roots(R, MU, SIGMA)
    k1, k2 = compute_kappas(a1, a2)
    assert abs(k1 + k2 - 1.0) < 1e-14
    assert k1 > 0 and k2 > 0


def test_crossing_matrix_inverse():
    a1, a2 = solve_roots(R, MU, SIGMA)
    for beta in [0.3, 0.5, 0.9, -0.3, -0.5]:
        for z in [0.5, 1.0, 1.5, 3.0]:
            assert validate_crossing_inverse(a1, a2, beta, z)


def test_crossing_matrix_zero_beta():
    # T_0(z) should be identity
    a1, a2 = solve_roots(R, MU, SIGMA)
    T = crossing_matrix(a1, a2, 0.0, 1.0)
    assert np.allclose(T, np.eye(2), atol=1e-13)


def test_crossing_matrix_shape():
    a1, a2 = solve_roots(R, MU, SIGMA)
    T = crossing_matrix(a1, a2, 0.5, 1.0)
    assert T.shape == (2, 2)


def test_different_parameter_sets():
    for r, mu, sigma in [(0.05, 0.01, 0.2), (0.10, 0.02, 0.3), (0.03, 0.005, 0.15)]:
        a1, a2 = solve_roots(r, mu, sigma)
        assert a1 < 0 < 1 < a2
        assert validate_crossing_inverse(a1, a2, 0.5, 1.5)
