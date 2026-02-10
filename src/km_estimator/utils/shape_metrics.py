"""Shape comparison metrics for KM curves."""

from bisect import bisect_right
from collections.abc import Sequence

import numpy as np


def _build_lookup(curve: Sequence[tuple[float, float]]) -> tuple[list[float], list[float]]:
    """Build sorted lookup arrays for step-function interpolation."""
    ordered = sorted(curve, key=lambda p: p[0])
    times = [float(t) for t, _ in ordered]
    survivals = [float(s) for _, s in ordered]
    return times, survivals


def _survival_at(lookup: tuple[list[float], list[float]], t: float) -> float:
    """Step-function lookup using bisect."""
    times, survivals = lookup
    if not times:
        return 1.0
    idx = bisect_right(times, t) - 1
    if idx < 0:
        return survivals[0]
    return survivals[idx]


def dtw_distance(
    curve1: Sequence[tuple[float, float]],
    curve2: Sequence[tuple[float, float]],
) -> float:
    """
    Compute Dynamic Time Warping distance between two curves.

    DTW finds the optimal alignment between two sequences, making it robust to:
    - Different sampling densities
    - Time warping/stretching
    - Local variations in timing

    Uses only the survival (y) values for comparison since KM curves
    may have different time sampling.

    Args:
        curve1: First curve as list of (time, survival) tuples
        curve2: Second curve as list of (time, survival) tuples

    Returns:
        Normalized DTW distance (lower is better, 0 = identical)
    """
    if not curve1 or not curve2:
        return float("inf")

    s1 = np.array([p[1] for p in curve1])
    s2 = np.array([p[1] for p in curve2])

    n, m = len(s1), len(s2)

    # Initialize DTW matrix with infinity
    dtw_matrix = np.full((n + 1, m + 1), np.inf)
    dtw_matrix[0, 0] = 0

    # Fill the matrix
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = abs(s1[i - 1] - s2[j - 1])
            dtw_matrix[i, j] = cost + min(
                dtw_matrix[i - 1, j],  # insertion
                dtw_matrix[i, j - 1],  # deletion
                dtw_matrix[i - 1, j - 1],  # match
            )

    # Normalize by path length
    return float(dtw_matrix[n, m] / max(n, m))


def frechet_distance(
    curve1: Sequence[tuple[float, float]],
    curve2: Sequence[tuple[float, float]],
) -> float:
    """
    Compute discrete Frechet distance (maximum deviation) between curves.

    The Frechet distance is the maximum distance that needs to be traveled
    between the two curves when traversing them optimally. It's useful for
    detecting the worst-case deviation.

    Args:
        curve1: First curve as list of (time, survival) tuples
        curve2: Second curve as list of (time, survival) tuples

    Returns:
        Frechet distance (maximum deviation in survival values)
    """
    if not curve1 or not curve2:
        return float("inf")

    n, m = len(curve1), len(curve2)
    ca = np.full((n, m), 0.0)

    for i in range(n):
        for j in range(m):
            d = abs(curve1[i][1] - curve2[j][1])
            if i == 0 and j == 0:
                ca[i, j] = d
            elif i > 0 and j == 0:
                ca[i, j] = max(ca[i - 1, 0], d)
            elif i == 0 and j > 0:
                ca[i, j] = max(ca[0, j - 1], d)
            else:
                ca[i, j] = max(min(ca[i - 1, j], ca[i - 1, j - 1], ca[i, j - 1]), d)

    return float(ca[n - 1, m - 1])


def area_between_curves(
    curve1: Sequence[tuple[float, float]],
    curve2: Sequence[tuple[float, float]],
) -> float:
    """
    Compute the area between two curves using trapezoidal integration.

    This metric captures the overall difference between curves,
    weighting larger deviations over longer time periods more heavily.

    Args:
        curve1: First curve as list of (time, survival) tuples
        curve2: Second curve as list of (time, survival) tuples

    Returns:
        Normalized area between curves (0 = identical)
    """
    if not curve1 or not curve2:
        return float("inf")

    # Get all unique time points from both curves
    times1 = set(p[0] for p in curve1)
    times2 = set(p[0] for p in curve2)
    all_times = sorted(times1 | times2)

    if len(all_times) < 2:
        return 0.0

    lookup1 = _build_lookup(curve1)
    lookup2 = _build_lookup(curve2)

    # Calculate area using trapezoidal rule
    area = 0.0
    for i in range(len(all_times) - 1):
        t1, t2 = all_times[i], all_times[i + 1]
        dt = t2 - t1

        s1_at_t1 = _survival_at(lookup1, t1)
        s1_at_t2 = _survival_at(lookup1, t2)
        s2_at_t1 = _survival_at(lookup2, t1)
        s2_at_t2 = _survival_at(lookup2, t2)

        # Average absolute difference over interval
        diff1 = abs(s1_at_t1 - s2_at_t1)
        diff2 = abs(s1_at_t2 - s2_at_t2)
        area += (diff1 + diff2) / 2 * dt

    # Normalize by total time range
    time_range = all_times[-1] - all_times[0]
    return area / time_range if time_range > 0 else 0.0


def rmse(
    curve1: Sequence[tuple[float, float]],
    curve2: Sequence[tuple[float, float]],
) -> float:
    """
    Compute Root Mean Square Error between curves at common time points.

    Args:
        curve1: First curve as list of (time, survival) tuples
        curve2: Second curve as list of (time, survival) tuples

    Returns:
        RMSE value (0 = identical)
    """
    if not curve1 or not curve2:
        return float("inf")

    lookup2 = _build_lookup(curve2)

    squared_errors = []
    for t, s1 in curve1:
        s2 = _survival_at(lookup2, t)
        squared_errors.append((s1 - s2) ** 2)

    return float(np.sqrt(np.mean(squared_errors))) if squared_errors else 0.0


def max_error(
    curve1: Sequence[tuple[float, float]],
    curve2: Sequence[tuple[float, float]],
) -> float:
    """
    Compute maximum absolute error between curves.

    Args:
        curve1: First curve as list of (time, survival) tuples
        curve2: Second curve as list of (time, survival) tuples

    Returns:
        Maximum absolute error
    """
    if not curve1 or not curve2:
        return float("inf")

    lookup2 = _build_lookup(curve2)

    errors = []
    for t, s1 in curve1:
        s2 = _survival_at(lookup2, t)
        errors.append(abs(s1 - s2))

    return float(max(errors)) if errors else 0.0
