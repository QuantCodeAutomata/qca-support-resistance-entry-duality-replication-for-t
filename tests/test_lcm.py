import numpy as np
import pytest
from src.lcm import upper_hull_indices, compute_lcm, compute_lcm_finite_points


def test_upper_hull_simple():
    x = np.array([0.0, 1.0, 2.0, 3.0])
    y = np.array([0.0, 1.0, 0.5, 0.0])
    idx = upper_hull_indices(x, y)
    # Hull should be [0, 1, 3] (peak at 1, then down to 3)
    assert 0 in idx and 1 in idx


def test_lcm_majorizes_input():
    np.random.seed(42)
    Y = np.sort(np.random.uniform(0, 10, 100))
    H = np.random.uniform(-1, 2, 100)
    K = compute_lcm(Y, H, include_origin=False)
    assert np.all(K >= H - 1e-14), 'LCM must majorize H everywhere'


def test_lcm_is_concave():
    Y = np.linspace(0.1, 5.0, 500)
    H = np.sin(Y)  # non-concave function
    K = compute_lcm(Y, H, include_origin=False)
    # Check concavity: second differences <= 0
    slopes = np.diff(K) / np.diff(Y)
    assert np.all(slopes[1:] <= slopes[:-1] + 1e-12), 'LCM must be concave'


def test_lcm_of_concave_function_is_itself():
    Y = np.linspace(0.01, 5.0, 500)
    H = -Y**2 + 5  # concave function
    K = compute_lcm(Y, H, include_origin=False)
    assert np.allclose(K, H, atol=1e-10), 'LCM of concave function should equal itself'


def test_lcm_with_origin():
    Y = np.array([1.0, 2.0, 3.0])
    H = np.array([2.0, 3.0, 2.0])
    K = compute_lcm(Y, H, include_origin=True)
    assert np.all(K >= H - 1e-14)


def test_lcm_finite_points_single():
    support_F = np.array([0.0, 2.0])
    support_h = np.array([0.0, 1.5])
    query_Y = np.linspace(0, 3.0, 100)
    K = compute_lcm_finite_points(support_F, support_h, query_Y)
    # Should be linear from (0,0) to (2,1.5), then flat at 1.5
    assert np.isclose(K[0], 0.0, atol=1e-12)
    mid_idx = np.searchsorted(query_Y, 2.0)
    assert np.isclose(K[mid_idx], 1.5, atol=1e-10)
    assert np.all(K[mid_idx:] <= 1.5 + 1e-12)


def test_lcm_flat_after_last_hull_point():
    Y = np.linspace(0.1, 10.0, 1000)
    H = np.where(Y <= 3.0, Y, 3.0 - (Y - 3.0)**2 / 10)  # decreasing after 3
    K = compute_lcm(Y, H, include_origin=False)
    # After the peak, K should be flat or concavely decreasing
    assert np.all(np.diff(K) <= 1e-12 + np.abs(np.diff(Y)))
