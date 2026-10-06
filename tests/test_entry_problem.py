import numpy as np
import pytest
from src.harmonic import build_harmonic_functions, eval_at_point
from src.exit_problem import solve_exit, extract_exit_band
from src.entry_problem import (solve_entry_bruteforce, solve_entry_theorem,
                                classify_entry_contact, compute_ratio_window,
                                compute_hitting_time_formula)

R, MU, SIGMA, L = 0.05, 0.01, 0.2, 1.0


def make_small_grid(n=5000):
    return np.geomspace(1e-4, 1e4, n)


def get_setup(H, beta_L, beta_H, n=5000):
    x = make_small_grid(n)
    hf = build_harmonic_functions(R, MU, SIGMA, [L, H], [beta_L, beta_H], x)
    res = solve_exit(x, x, hf['varphi_r'], hf['F'])
    W = res['v'] - x
    return x, hf, res, W


def test_entry_u_geq_W():
    x, hf, res, W = get_setup(1.6, 0.9, -0.3)
    entry = solve_entry_bruteforce(x, W, hf['varphi_r'], hf['F'])
    assert np.all(entry['u'] >= W - 1e-10), 'u must be >= W everywhere'


def test_entry_u_nonnegative():
    x, hf, res, W = get_setup(1.6, 0.9, -0.3)
    entry = solve_entry_bruteforce(x, W, hf['varphi_r'], hf['F'])
    assert np.all(entry['u'] >= -1e-12)


def test_theorem_matches_bruteforce_beta_L_positive():
    """For beta_L>0>beta_H, theorem entry should match brute-force."""
    x, hf, res, W = get_setup(1.6, 0.9, -0.3, n=5000)
    a1, a2 = hf['alpha_1'], hf['alpha_2']
    psi_L = eval_at_point(L, a1, a2, hf['psi_coeffs'])
    phi_L = eval_at_point(L, a1, a2, hf['varphi_coeffs'])
    F_L = psi_L / phi_L
    K_at_FL = np.interp(F_L, hf['F'], res['K_g'])
    v_L = phi_L * K_at_FL
    W_L = v_L - L

    u_theory = solve_entry_theorem(x, np.array([W_L]), np.array([phi_L]),
                                    np.array([F_L]), hf['varphi_r'], hf['F'])
    bf = solve_entry_bruteforce(x, W, hf['varphi_r'], hf['F'])

    compare_mask = (x >= 0.3) & (x <= 3.0)
    max_err = np.max(np.abs(u_theory[compare_mask] - bf['u'][compare_mask]))
    # Tolerance reflects grid-discretization error at n=5000 (geomspace F-step ~0.015 near L)
    assert max_err < 1e-2, f'Max error too large: {max_err:.2e}'


def test_all_negative_skews_zero_entry():
    """Both skews negative: P_+ empty, entry value should be ~0."""
    x, hf, res, W = get_setup(1.5, -0.5, -0.5)
    bf = solve_entry_bruteforce(x, W, hf['varphi_r'], hf['F'])
    compare_mask = (x >= 0.3) & (x <= 6.0)
    assert np.all(np.abs(bf['u'][compare_mask]) < 1e-8), 'Entry value should be ~0 for all-negative skews'


def test_ratio_window_classification():
    # Scenario with contact {L,H}: H=1.5, beta_L=0.9, beta_H=0.9
    x, hf, res, W = get_setup(1.5, 0.9, 0.9)
    a1, a2 = hf['alpha_1'], hf['alpha_2']
    psi_L = eval_at_point(L, a1, a2, hf['psi_coeffs'])
    phi_L = eval_at_point(L, a1, a2, hf['varphi_coeffs'])
    F_L = psi_L / phi_L
    K_FL = np.interp(F_L, hf['F'], res['K_g'])
    W_L = phi_L * K_FL - L

    psi_H = eval_at_point(1.5, a1, a2, hf['psi_coeffs'])
    phi_H = eval_at_point(1.5, a1, a2, hf['varphi_coeffs'])
    F_H = psi_H / phi_H
    K_FH = np.interp(F_H, hf['F'], res['K_g'])
    W_H = phi_H * K_FH - 1.5

    rw = compute_ratio_window(psi_L, psi_H, phi_L, phi_H, W_L, W_H)
    # rho should be in window for contact {L,H}
    assert rw['underline_rho'] <= rw['rho'] <= rw['overline_rho'], \
        f'Expected {rw["rho"]:.4f} in [{rw["underline_rho"]:.4f}, {rw["overline_rho"]:.4f}]'


def test_hitting_time_formula():
    """For beta_L<0, beta_H>0: u_theory from hitting-time formula."""
    x, hf, res, W = get_setup(3.0, -0.5, 0.9)
    a1, a2 = hf['alpha_1'], hf['alpha_2']
    H = 3.0
    psi_H = eval_at_point(H, a1, a2, hf['psi_coeffs'])
    phi_H = eval_at_point(H, a1, a2, hf['varphi_coeffs'])
    F_H = psi_H / phi_H
    K_FH = np.interp(F_H, hf['F'], res['K_g'])
    W_H = phi_H * K_FH - H

    u_hit = compute_hitting_time_formula(x, H, W_H, hf['psi_r'], psi_H, hf['varphi_r'], phi_H)
    bf = solve_entry_bruteforce(x, W, hf['varphi_r'], hf['F'])

    compare_mask = (x >= 0.3) & (x <= 6.0)
    max_err = np.max(np.abs(u_hit[compare_mask] - bf['u'][compare_mask]))
    # Tolerance reflects grid-discretization error at n=5000 over the wider [0.3, 6] window
    assert max_err < 1e-2, f'Hitting-time formula error too large: {max_err:.2e}'
