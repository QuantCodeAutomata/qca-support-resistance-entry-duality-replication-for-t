# Custom — Context7 found no library equivalent (multi-skew BM characteristic ODE)
import numpy as np
from typing import Tuple


def solve_roots(r: float, mu: float, sigma: float) -> Tuple[float, float]:
    """Solve 0.5*sigma^2*alpha*(alpha-1) + mu*alpha - r = 0.
    Returns (alpha_1, alpha_2) with alpha_1 < 0 < 1 < alpha_2."""
    # Quadratic: 0.5*sigma^2 * alpha^2 + (mu - 0.5*sigma^2) * alpha - r = 0
    a_coef = 0.5 * sigma**2
    b_coef = mu - 0.5 * sigma**2
    c_coef = -r
    disc = b_coef**2 - 4*a_coef*c_coef
    alpha_1 = (-b_coef - np.sqrt(disc)) / (2*a_coef)
    alpha_2 = (-b_coef + np.sqrt(disc)) / (2*a_coef)
    assert alpha_1 < 0 < 1 < alpha_2, f"Root ordering violated: {alpha_1}, {alpha_2}"
    return alpha_1, alpha_2


def compute_kappas(alpha_1: float, alpha_2: float) -> Tuple[float, float]:
    """Compute kappa_1=(alpha_2-1)/(alpha_2-alpha_1) and kappa_2=(1-alpha_1)/(alpha_2-alpha_1)."""
    delta = alpha_2 - alpha_1
    kappa_1 = (alpha_2 - 1) / delta
    kappa_2 = (1 - alpha_1) / delta
    assert abs(kappa_1 + kappa_2 - 1) < 1e-12
    return kappa_1, kappa_2


def crossing_matrix(alpha_1: float, alpha_2: float, beta: float, z: float) -> np.ndarray:
    """T_beta(z) = 1/((alpha_2-alpha_1)*(1-beta)) * [[...]] as defined in the paper.
    Top-left: (alpha_2-alpha_1)-(alpha_1+alpha_2)*beta
    Top-right: -2*alpha_2*beta*z^(alpha_2-alpha_1)
    Bottom-left: 2*alpha_1*beta*z^(alpha_1-alpha_2)
    Bottom-right: (1+beta)*alpha_2 - (1-beta)*alpha_1"""
    delta = alpha_2 - alpha_1
    denom = delta * (1 - beta)
    T = np.array([
        [delta - (alpha_1 + alpha_2)*beta, -2*alpha_2*beta*z**(alpha_2 - alpha_1)],
        [2*alpha_1*beta*z**(alpha_1 - alpha_2), (1+beta)*alpha_2 - (1-beta)*alpha_1]
    ]) / denom
    return T


def validate_crossing_inverse(alpha_1: float, alpha_2: float, beta: float, z: float,
                               tol: float = 1e-12) -> bool:
    """Validate that T_beta(z) @ T_{-beta}(z) = I numerically."""
    T_pos = crossing_matrix(alpha_1, alpha_2, beta, z)
    T_neg = crossing_matrix(alpha_1, alpha_2, -beta, z)
    prod = T_pos @ T_neg
    return np.allclose(prod, np.eye(2), atol=tol)
