# Custom — Context7 found no library equivalent (piecewise r-harmonic functions for multi-skew BM)
import numpy as np
from typing import List, Tuple

from src.roots_and_crossing import solve_roots, compute_kappas, crossing_matrix


def build_psi_r_coeffs(alpha_1: float, alpha_2: float,
                        levels: List[float], betas: List[float]) -> List[Tuple[float, float, float, float]]:
    """
    Build piecewise coefficient representation of psi_r (increasing r-harmonic function).
    levels: sorted list of skew levels [L, H, ...]
    betas: corresponding skew coefficients
    Returns list of (x_low, x_high, A, B) tuples for each interval,
    where psi_r(x) = A*x^alpha_1 + B*x^alpha_2 on (x_low, x_high).
    Intervals: (0, L], (L, H], ..., [last_level, inf)
    """
    assert len(levels) == len(betas)
    levels_betas = sorted(zip(levels, betas), key=lambda t: t[0])
    sorted_levels = [lv for lv, b in levels_betas]
    sorted_betas = [b for lv, b in levels_betas]

    # Start: on (0, L], [A,B] = [0,1]
    intervals = []
    A, B = 0.0, 1.0

    x_low = 0.0
    for z, beta in zip(sorted_levels, sorted_betas):
        x_high = z
        intervals.append((x_low, x_high, A, B))
        # Cross upward: use T_{-beta}(z)
        T = crossing_matrix(alpha_1, alpha_2, -beta, z)
        coeff = T @ np.array([A, B])
        A, B = coeff[0], coeff[1]
        x_low = z

    # Final interval [last_level, inf)
    intervals.append((x_low, np.inf, A, B))
    return intervals


def build_varphi_r_coeffs(alpha_1: float, alpha_2: float,
                           levels: List[float], betas: List[float]) -> List[Tuple[float, float, float, float]]:
    """
    Build piecewise coefficient representation of varphi_r (decreasing r-harmonic function).
    Intervals built from top (H, inf) downward.
    Returns list of (x_low, x_high, A, B) tuples.
    """
    assert len(levels) == len(betas)
    levels_sorted = sorted(zip(levels, betas), key=lambda t: t[0], reverse=True)

    # Start: on [H_max, inf), [A,B] = [1,0]
    intervals = []
    A, B = 1.0, 0.0

    x_high = np.inf
    for z, beta in levels_sorted:
        x_low = z
        intervals.append((x_low, x_high, A, B))
        # Cross downward: use T_{beta}(z)
        T = crossing_matrix(alpha_1, alpha_2, beta, z)
        coeff = T @ np.array([A, B])
        A, B = coeff[0], coeff[1]
        x_high = z

    # Final interval (0, z_min]
    intervals.append((0.0, x_high, A, B))
    intervals.reverse()  # Order from low to high x
    return intervals


def eval_piecewise(x: np.ndarray, alpha_1: float, alpha_2: float,
                   intervals: List[Tuple[float, float, float, float]]) -> np.ndarray:
    """
    Evaluate piecewise harmonic function f(x) = A*x^alpha_1 + B*x^alpha_2
    on the given grid. intervals is list of (x_low, x_high, A, B).
    """
    result = np.zeros_like(x, dtype=float)
    for (x_low, x_high, A, B) in intervals:
        if x_high == np.inf:
            mask = x >= x_low
        elif x_low == 0.0:
            mask = x <= x_high
        else:
            mask = (x > x_low) & (x <= x_high)
        if np.any(mask):
            xm = x[mask]
            result[mask] = A * xm**alpha_1 + B * xm**alpha_2
    return result


def eval_at_point(x_pt: float, alpha_1: float, alpha_2: float,
                  intervals: List[Tuple[float, float, float, float]]) -> float:
    """
    Evaluate piecewise harmonic function at a single point x_pt.
    Uses exact interval coefficients to avoid grid-interpolation errors.
    """
    for (x_low, x_high, A, B) in intervals:
        if x_high == np.inf:
            if x_pt >= x_low:
                return A * x_pt**alpha_1 + B * x_pt**alpha_2
        elif x_low == 0.0:
            if x_pt <= x_high:
                return A * x_pt**alpha_1 + B * x_pt**alpha_2
        else:
            if x_low < x_pt <= x_high:
                return A * x_pt**alpha_1 + B * x_pt**alpha_2
    # Edge: x_pt exactly at leftmost boundary
    x_low, x_high, A, B = intervals[0]
    return A * x_pt**alpha_1 + B * x_pt**alpha_2


def build_harmonic_functions(r: float, mu: float, sigma: float,
                              levels: List[float], betas: List[float],
                              x_grid: np.ndarray) -> dict:
    """
    Build all harmonic functions and F-transform for a multi-skew BM model.
    Returns dict with keys: alpha_1, alpha_2, kappa_1, kappa_2,
    psi_r, varphi_r, F (all evaluated on x_grid),
    psi_coeffs, varphi_coeffs (interval coefficient lists for direct evaluation).
    """
    alpha_1, alpha_2 = solve_roots(r, mu, sigma)
    kappa_1, kappa_2 = compute_kappas(alpha_1, alpha_2)

    psi_coeffs = build_psi_r_coeffs(alpha_1, alpha_2, levels, betas)
    varphi_coeffs = build_varphi_r_coeffs(alpha_1, alpha_2, levels, betas)

    psi_r = eval_piecewise(x_grid, alpha_1, alpha_2, psi_coeffs)
    varphi_r = eval_piecewise(x_grid, alpha_1, alpha_2, varphi_coeffs)
    F = psi_r / varphi_r

    # Validate monotonicity and positivity
    assert np.all(psi_r > 0), 'psi_r must be positive'
    assert np.all(varphi_r > 0), 'varphi_r must be positive'
    assert np.all(np.diff(psi_r) > 0), 'psi_r must be strictly increasing'
    assert np.all(np.diff(varphi_r) < 0), 'varphi_r must be strictly decreasing'
    assert np.all(np.diff(F) > 0), 'F must be strictly increasing'

    return {
        'alpha_1': alpha_1, 'alpha_2': alpha_2,
        'kappa_1': kappa_1, 'kappa_2': kappa_2,
        'psi_r': psi_r, 'varphi_r': varphi_r, 'F': F,
        'psi_coeffs': psi_coeffs, 'varphi_coeffs': varphi_coeffs,
    }
