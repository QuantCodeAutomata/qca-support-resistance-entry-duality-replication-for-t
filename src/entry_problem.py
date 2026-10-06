# Custom — Context7 found no library for finite-hull entry theorem on multi-skew BM
import numpy as np
from typing import List, Tuple, Dict

from src.lcm import compute_lcm, compute_lcm_finite_points, upper_hull_indices
from src.harmonic import eval_at_point


def solve_entry_bruteforce(x_grid: np.ndarray, W: np.ndarray,
                            varphi_r: np.ndarray, F: np.ndarray,
                            tol_contact: float = 1e-10) -> Dict:
    """
    Compute entry value u via brute-force LCM of W/varphi_r over full grid.
    W(x): entry reward (exit premium), W = v - x.
    Returns dict with u (entry value), K_W (LCM in transformed space), contact_mask.
    """
    H_W = W / varphi_r  # transformed reward
    K_W = compute_lcm(F, H_W, include_origin=True)
    u = varphi_r * K_W
    contact_mask = np.abs(u - W) <= tol_contact
    return {'u': u, 'K_W': K_W, 'contact_mask': contact_mask}


def solve_entry_theorem(x_grid: np.ndarray, W_support: np.ndarray,
                         varphi_r_support: np.ndarray, F_support: np.ndarray,
                         varphi_r_grid: np.ndarray, F_grid: np.ndarray) -> np.ndarray:
    """
    Compute theorem-based entry value using only finite support-point set.
    W_support: W values at upward-skew support points (direct evaluation, NOT grid interpolation)
    varphi_r_support: varphi_r values at support points
    F_support: F values at support points
    Returns u_theory on x_grid.
    """
    h_support = W_support / varphi_r_support
    # Build finite hull: (0,0) + support points
    support_F = np.concatenate([[0.0], F_support])
    support_h = np.concatenate([[0.0], h_support])
    K_theory = compute_lcm_finite_points(support_F, support_h, F_grid)
    u_theory = varphi_r_grid * K_theory
    return u_theory


def classify_entry_contact(support_levels: List[float],
                            W_support: np.ndarray,
                            varphi_r_support: np.ndarray,
                            F_support: np.ndarray,
                            tol: float = 1e-10) -> List[float]:
    """
    Determine which support points are in the entry contact set.
    A support level z is in Gamma if it lies on the upper hull of
    {(0,0)} union {(F(z_j), W(z_j)/varphi_r(z_j))}.
    Returns list of levels in Gamma.
    """
    h_support = W_support / varphi_r_support
    support_F_all = np.concatenate([[0.0], F_support])
    support_h_all = np.concatenate([[0.0], h_support])
    hull_idx = upper_hull_indices(support_F_all, support_h_all)
    # Indices >= 1 correspond to support levels (0 is origin)
    gamma = [support_levels[i-1] for i in hull_idx if i > 0]
    return gamma


def compute_ratio_window(psi_r_L: float, psi_r_H: float,
                          varphi_r_L: float, varphi_r_H: float,
                          W_L: float, W_H: float) -> Dict:
    """
    Compute ratio-window classification for two-support case.
    rho = W(L)/W(H), underline_rho = psi_r(L)/psi_r(H), overline_rho = varphi_r(L)/varphi_r(H).
    Returns dict with rho, underline_rho, overline_rho, and predicted contact set.
    """
    rho = W_L / W_H
    underline_rho = psi_r_L / psi_r_H
    overline_rho = varphi_r_L / varphi_r_H
    if rho < underline_rho:
        predicted = 'H'
    elif rho > overline_rho:
        predicted = 'L'
    else:
        predicted = 'L,H'
    return {
        'rho': rho,
        'underline_rho': underline_rho,
        'overline_rho': overline_rho,
        'predicted': predicted,
    }


def compute_hitting_time_formula(x_grid: np.ndarray, H_level: float,
                                  W_H: float, psi_r_grid: np.ndarray,
                                  psi_r_H: float, varphi_r_grid: np.ndarray,
                                  varphi_r_H: float) -> np.ndarray:
    """
    Compute the single-point hitting-time formula for entry:
    u(x) = W(H) * E_x[e^{-r*tau_H}]
    where E_x[e^{-r*tau_H}] = psi_r(x)/psi_r(H) for x <= H
                              = varphi_r(x)/varphi_r(H) for x >= H
    """
    u = np.where(
        x_grid <= H_level,
        W_H * psi_r_grid / psi_r_H,
        W_H * varphi_r_grid / varphi_r_H,
    )
    return u
