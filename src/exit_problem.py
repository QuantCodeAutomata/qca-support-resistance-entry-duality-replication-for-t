# Custom — Context7 found no library for transformed-space LCM exit problem on multi-skew BM
import numpy as np
from typing import Tuple, Dict

from src.lcm import compute_lcm


def solve_exit(x_grid: np.ndarray, g: np.ndarray,
               varphi_r: np.ndarray, F: np.ndarray,
               tol_contact: float = 1e-10) -> Dict:
    """
    Solve the exit problem: v(x) = sup_tau E_x[e^{-r*tau} g(S_tau)].
    Method: F-transform + least concave majorant in transformed space.
    g(x) = x (reward function, typically the asset price).

    Returns dict with:
      v: exit value function on x_grid
      K_g: LCM values in transformed space (H_i = g_i/varphi_r_i over F_i)
      contact_mask: boolean mask where v(x) ≈ g(x) (exit region)
    """
    # Transform reward: H_i = g(x_i) / varphi_r(x_i)
    H = g / varphi_r  # transformed reward
    Y = F  # already sorted since F is strictly increasing

    # Compute LCM of H over Y, anchored at origin (F(0+) -> 0)
    K_g = compute_lcm(Y, H, include_origin=True)

    # Recover exit value
    v = varphi_r * K_g

    # Contact set: where LCM touches H, i.e., v ≈ g
    contact_mask = np.abs(v - g) <= tol_contact

    return {'v': v, 'K_g': K_g, 'contact_mask': contact_mask}


def extract_exit_band(x_grid: np.ndarray, v: np.ndarray, g: np.ndarray,
                      tol_contact: float = 1e-10) -> Tuple[float, float]:
    """
    Extract exit band boundaries (a, b) from exit value function.
    Continuation region: {v > g} = (a, b)
    a: lower boundary (sell below a), b: upper boundary (sell above b).
    Uses linear interpolation at transition points.
    Returns (a, b).
    """
    diff = v - g  # positive in continuation region

    # Find leftmost index where v > g (transition from exit to continuation)
    cont_mask = diff > tol_contact
    if not np.any(cont_mask):
        return (np.nan, np.nan)

    first_cont = np.where(cont_mask)[0][0]
    last_cont = np.where(cont_mask)[0][-1]

    # Interpolate lower boundary a
    if first_cont > 0:
        i0, i1 = first_cont - 1, first_cont
        a = np.interp(0, [diff[i0], diff[i1]], [x_grid[i0], x_grid[i1]])
    else:
        a = x_grid[first_cont]

    # Interpolate upper boundary b
    if last_cont < len(x_grid) - 1:
        i0, i1 = last_cont, last_cont + 1
        b = np.interp(0, [diff[i0], diff[i1]], [x_grid[i0], x_grid[i1]])
    else:
        b = x_grid[last_cont]

    return (a, b)
