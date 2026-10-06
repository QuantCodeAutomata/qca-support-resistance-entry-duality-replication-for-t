# Custom — Context7 found no library equivalent for LCM in F-space optimal stopping
import numpy as np
from typing import Tuple


def upper_hull_indices(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Compute indices of upper convex hull of points (x_i, y_i) sorted by x.
    This gives the least concave majorant (LCM) support points.
    Uses Andrew's monotone chain for upper hull.
    Returns array of indices into (x, y) that form the upper hull.
    """
    n = len(x)
    hull = []
    for i in range(n):
        while len(hull) >= 2:
            j, k = hull[-2], hull[-1]
            # cross product: if left turn or collinear, remove k
            cross = (x[k] - x[j]) * (y[i] - y[j]) - (y[k] - y[j]) * (x[i] - x[j])
            if cross >= 0:
                hull.pop()
            else:
                break
        hull.append(i)
    return np.array(hull)


def compute_lcm(Y: np.ndarray, H: np.ndarray,
                include_origin: bool = True) -> np.ndarray:
    """
    Compute least concave majorant K(y) of discrete function (Y_i, H_i).
    Y must be strictly increasing (sorted).
    If include_origin=True, prepend (0,0) as anchor point.
    Returns K values at each Y_i by linear interpolation from upper hull vertices.
    """
    if include_origin:
        Yx = np.concatenate([[0.0], Y])
        Hx = np.concatenate([[0.0], H])
    else:
        Yx = Y.copy()
        Hx = H.copy()

    hull_idx = upper_hull_indices(Yx, Hx)
    hull_Y = Yx[hull_idx]
    hull_H = Hx[hull_idx]

    # Interpolate K values at original Y points
    # For Y < hull_Y[0]: extrapolate flat left (should not occur if origin included)
    # For Y > hull_Y[-1]: flat right at hull_H[-1] (LCM is flat after last hull point)
    K = np.interp(Y, hull_Y, hull_H, left=hull_H[0], right=hull_H[-1])
    return K


def compute_lcm_finite_points(support_F: np.ndarray, support_h: np.ndarray,
                               query_Y: np.ndarray) -> np.ndarray:
    """
    Compute least concave majorant over a FINITE set of support points.
    support_F: F-values at support points (support_F[0] should be 0 for origin)
    support_h: h-values at support points
    query_Y: dense grid of F-values at which to evaluate K
    Returns K values at query_Y by piecewise-linear interpolation from hull.
    """
    assert support_F[0] == 0.0, 'First support point must be origin (0,0)'
    hull_idx = upper_hull_indices(support_F, support_h)
    hull_Y = support_F[hull_idx]
    hull_H = support_h[hull_idx]
    K = np.interp(query_Y, hull_Y, hull_H, left=hull_H[0], right=hull_H[-1])
    return K
