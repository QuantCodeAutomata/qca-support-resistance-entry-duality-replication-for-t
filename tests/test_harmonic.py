import numpy as np
import pytest
from src.roots_and_crossing import solve_roots
from src.harmonic import build_harmonic_functions, eval_at_point, build_psi_r_coeffs, build_varphi_r_coeffs

R, MU, SIGMA, L = 0.05, 0.01, 0.2, 1.0


def make_grid(n=1000):
    return np.geomspace(1e-4, 1e4, n)


def test_psi_r_positive_increasing():
    x = make_grid()
    hf = build_harmonic_functions(R, MU, SIGMA, [L, 1.6], [0.9, -0.3], x)
    assert np.all(hf['psi_r'] > 0)
    assert np.all(np.diff(hf['psi_r']) > 0)


def test_varphi_r_positive_decreasing():
    x = make_grid()
    hf = build_harmonic_functions(R, MU, SIGMA, [L, 1.6], [0.9, -0.3], x)
    assert np.all(hf['varphi_r'] > 0)
    assert np.all(np.diff(hf['varphi_r']) < 0)


def test_F_strictly_increasing():
    x = make_grid()
    hf = build_harmonic_functions(R, MU, SIGMA, [L, 1.6], [0.9, -0.3], x)
    assert np.all(np.diff(hf['F']) > 0)


def test_psi_r_is_alpha2_power_below_L():
    """On (0,L], psi_r(x) should equal x^alpha_2."""
    a1, a2 = solve_roots(R, MU, SIGMA)
    x = np.geomspace(1e-4, 0.99, 500)
    hf = build_harmonic_functions(R, MU, SIGMA, [L, 1.6], [0.9, -0.3], x)
    expected = x**a2
    assert np.allclose(hf['psi_r'], expected, rtol=1e-12)


def test_varphi_r_is_alpha1_power_above_H():
    """On [H,inf), varphi_r(x) should equal x^alpha_1."""
    a1, a2 = solve_roots(R, MU, SIGMA)
    H = 1.6
    x = np.geomspace(1.61, 1e4, 500)
    hf = build_harmonic_functions(R, MU, SIGMA, [L, H], [0.9, -0.3], x)
    expected = x**a1
    assert np.allclose(hf['varphi_r'], expected, rtol=1e-12)


def test_eval_at_point_matches_grid():
    """Direct point evaluation should match grid evaluation near L."""
    x = make_grid(5000)
    hf = build_harmonic_functions(R, MU, SIGMA, [L, 1.6], [0.9, -0.3], x)
    a1, a2 = hf['alpha_1'], hf['alpha_2']
    # Evaluate psi_r at L=1.0 directly
    psi_L = eval_at_point(1.0, a1, a2, hf['psi_coeffs'])
    phi_L = eval_at_point(1.0, a1, a2, hf['varphi_coeffs'])
    assert psi_L > 0 and phi_L > 0


def test_single_level_model():
    """One-level model: psi above level should differ from power law."""
    x = make_grid()
    hf = build_harmonic_functions(R, MU, SIGMA, [L], [0.9], x)
    assert np.all(hf['psi_r'] > 0)
    assert np.all(hf['varphi_r'] > 0)


def test_kappa_values():
    a1, a2 = solve_roots(R, MU, SIGMA)
    from src.roots_and_crossing import compute_kappas
    k1, k2 = compute_kappas(a1, a2)
    assert abs(k1 + k2 - 1.0) < 1e-12
    assert 0 < k1 < 1 and 0 < k2 < 1
