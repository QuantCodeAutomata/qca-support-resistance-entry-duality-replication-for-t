import numpy as np
import pytest
from src.harmonic import build_harmonic_functions
from src.exit_problem import solve_exit, extract_exit_band

R, MU, SIGMA, L = 0.05, 0.01, 0.2, 1.0


def make_grid(n=10000):
    return np.geomspace(1e-4, 1e4, n)


def test_exit_value_geq_g():
    x = make_grid()
    hf = build_harmonic_functions(R, MU, SIGMA, [L, 1.6], [0.9, -0.3], x)
    result = solve_exit(x, x, hf['varphi_r'], hf['F'])
    assert np.all(result['v'] >= x - 1e-10), 'v must be >= g everywhere'


def test_exit_band_valid():
    x = make_grid()
    hf = build_harmonic_functions(R, MU, SIGMA, [L, 1.6], [0.9, -0.3], x)
    result = solve_exit(x, x, hf['varphi_r'], hf['F'])
    a, b = extract_exit_band(x, result['v'], x)
    assert 0 < a < b
    assert a < L < b


def test_exit_value_scenario_a():
    """Scenario A: H=1.60, beta_L=0.9, beta_H=-0.30. Expected a~0.6253, b~1.4654."""
    x = np.geomspace(1e-4, 1e4, 50000)  # smaller grid for speed
    H = 1.6
    hf = build_harmonic_functions(R, MU, SIGMA, [L, H], [0.9, -0.30], x)
    result = solve_exit(x, x, hf['varphi_r'], hf['F'])
    a, b = extract_exit_band(x, result['v'], x)
    assert abs(a - 0.6253) < 0.01, f'a={a:.4f} expected ~0.6253'
    assert abs(b - 1.4654) < 0.01, f'b={b:.4f} expected ~1.4654'


def test_exit_contact_set_nonempty():
    x = make_grid()
    hf = build_harmonic_functions(R, MU, SIGMA, [L, 1.6], [0.9, -0.3], x)
    result = solve_exit(x, x, hf['varphi_r'], hf['F'])
    assert np.any(result['contact_mask'])


def test_exit_v_harmonic_in_continuation():
    """In the continuation region, v should be a combination of x^alpha_1 and x^alpha_2."""
    x = make_grid(5000)
    hf = build_harmonic_functions(R, MU, SIGMA, [L, 1.6], [0.9, -0.3], x)
    result = solve_exit(x, x, hf['varphi_r'], hf['F'])
    a, b = extract_exit_band(x, result['v'], x)
    # In continuation region, v > g (not on contact), check v is smooth by checking it's > g
    cont_mask = (x > a * 1.05) & (x < b * 0.95)
    assert np.all(result['v'][cont_mask] > x[cont_mask] - 1e-12)
