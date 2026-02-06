"""Overlap resolution: enforce step function, fill gaps."""

import numpy as np

from .axis_calibration import AxisMapping

MAX_CANDIDATES_PER_X = 9
MEDIAN_ATTRACTION_WEIGHT = 0.08
SMOOTHNESS_WEIGHT = 0.15
UPWARD_MOVE_PENALTY = 8.0
ALLOWED_UPWARD_MOVE_PX = 2


def enforce_step_function(
    points: list[tuple[float, float]],
    time_tolerance: float = 0.001,
) -> list[tuple[float, float]]:
    """
    Convert digitized points to proper step function format.

    KM curves are step functions: survival stays constant between events,
    then drops vertically at event times. This function ensures the output
    follows this pattern by:
    1. Sorting by time
    2. For each survival drop, adding a horizontal segment to the drop point
    3. Then adding the vertical drop

    Args:
        points: List of (time, survival) tuples
        time_tolerance: Minimum time difference to consider points distinct

    Returns:
        List of (time, survival) tuples in proper step function format
    """
    if len(points) < 2:
        return points

    # Sort by time
    sorted_pts = sorted(points, key=lambda p: p[0])

    # Remove near-duplicates (same time within tolerance)
    deduped: list[tuple[float, float]] = [sorted_pts[0]]
    for t, s in sorted_pts[1:]:
        prev_t, prev_s = deduped[-1]
        if abs(t - prev_t) > time_tolerance:
            deduped.append((t, s))
        elif s < prev_s:  # Same time but lower survival - keep the drop
            deduped[-1] = (t, s)

    if len(deduped) < 2:
        return deduped

    # Build step function: for each drop, add horizontal then vertical
    # Track minimum survival seen so far (KM curves are monotonically non-increasing)
    step_points: list[tuple[float, float]] = [deduped[0]]
    min_survival = deduped[0][1]

    for i in range(1, len(deduped)):
        curr_t, curr_s = deduped[i]
        prev_t, prev_s = step_points[-1]

        # Clamp any survival increase to previous level (noise correction)
        if curr_s > min_survival:
            curr_s = min_survival

        # If survival dropped, add the step shape
        if curr_s < prev_s:
            # Horizontal segment: carry previous survival to current time
            step_points.append((curr_t, prev_s))
            # Vertical drop: drop to current survival at current time
            step_points.append((curr_t, curr_s))
            min_survival = curr_s
        else:
            # Survival stayed same - extend horizontal
            step_points.append((curr_t, curr_s))

    return step_points


def _select_y_candidates(ys: list[int], max_candidates: int = MAX_CANDIDATES_PER_X) -> list[int]:
    """Select representative y candidates for one x-column."""
    arr = np.asarray(ys, dtype=np.float32)
    unique = np.unique(arr.astype(np.int32))
    if unique.size <= max_candidates:
        return unique.tolist()

    quantiles = np.linspace(0.05, 0.95, max_candidates)
    sampled = np.rint(np.quantile(arr, quantiles)).astype(np.int32)
    candidates = np.unique(sampled)
    return candidates.tolist()


def _transition_cost(prev_y: int, curr_y: int, x_gap: int) -> float:
    """
    Transition penalty between adjacent x-columns.

    The graph is left-to-right in x, with strong penalty for y decreases
    (which would imply survival increasing in KM semantics).
    """
    upward_move = max(0, prev_y - curr_y - ALLOWED_UPWARD_MOVE_PX)
    smooth = abs(curr_y - prev_y) / max(1, x_gap)
    return upward_move * UPWARD_MOVE_PENALTY + smooth * SMOOTHNESS_WEIGHT


def _trace_curve_shortest_path(
    x_to_ys: dict[int, list[int]],
) -> list[tuple[int, int]]:
    """Find a smooth monotone path using shortest-path dynamic programming."""
    if not x_to_ys:
        return []

    x_values = sorted(x_to_ys.keys())
    candidates_per_x = [_select_y_candidates(x_to_ys[x]) for x in x_values]
    medians = [float(np.median(x_to_ys[x])) for x in x_values]

    costs: list[np.ndarray] = []
    back_ptr: list[np.ndarray] = []

    first_candidates = np.asarray(candidates_per_x[0], dtype=np.int32)
    first_cost = (
        np.abs(first_candidates.astype(np.float32) - medians[0])
        * MEDIAN_ATTRACTION_WEIGHT
    )
    costs.append(first_cost)
    back_ptr.append(np.full(first_candidates.shape[0], -1, dtype=np.int32))

    for i in range(1, len(x_values)):
        prev_candidates = np.asarray(candidates_per_x[i - 1], dtype=np.int32)
        curr_candidates = np.asarray(candidates_per_x[i], dtype=np.int32)
        prev_cost = costs[i - 1]
        x_gap = x_values[i] - x_values[i - 1]

        curr_cost = np.full(curr_candidates.shape[0], np.inf, dtype=np.float32)
        curr_parent = np.full(curr_candidates.shape[0], -1, dtype=np.int32)

        data_cost = (
            np.abs(curr_candidates.astype(np.float32) - medians[i])
            * MEDIAN_ATTRACTION_WEIGHT
        )
        for curr_idx, curr_y in enumerate(curr_candidates):
            best_cost = np.inf
            best_parent = -1
            for prev_idx, prev_y in enumerate(prev_candidates):
                total = prev_cost[prev_idx] + _transition_cost(int(prev_y), int(curr_y), x_gap)
                if total < best_cost:
                    best_cost = total
                    best_parent = prev_idx
            curr_cost[curr_idx] = best_cost + data_cost[curr_idx]
            curr_parent[curr_idx] = best_parent

        costs.append(curr_cost)
        back_ptr.append(curr_parent)

    path_indices = [0] * len(x_values)
    path_indices[-1] = int(np.argmin(costs[-1]))
    for i in range(len(x_values) - 1, 0, -1):
        path_indices[i - 1] = int(back_ptr[i][path_indices[i]])

    traced: list[tuple[int, int]] = []
    for x, idx, candidates in zip(x_values, path_indices, candidates_per_x):
        traced.append((x, int(candidates[idx])))

    # Hard monotonic enforcement for KM semantics in pixel coordinates.
    monotone: list[tuple[int, int]] = []
    max_y = traced[0][1]
    for px, py in traced:
        if py < max_y:
            py = max_y
        else:
            max_y = py
        monotone.append((px, py))

    return monotone


def _fill_small_gaps(
    points: list[tuple[int, int]],
    gap_threshold: int,
) -> list[tuple[int, int]]:
    """Fill small x-gaps by interpolation to keep curve continuity."""
    if len(points) < 2:
        return points

    filled: list[tuple[int, int]] = []
    for i, (px, py) in enumerate(points):
        filled.append((px, py))
        if i >= len(points) - 1:
            continue
        next_px, next_py = points[i + 1]
        gap = next_px - px
        if 1 < gap <= gap_threshold:
            for fill_x in range(px + 1, next_px):
                ratio = (fill_x - px) / gap
                fill_y = int(py + ratio * (next_py - py))
                # Keep monotone in pixel space.
                fill_y = max(fill_y, py)
                filled.append((fill_x, fill_y))

    return filled


def resolve_overlaps(
    raw_curves: dict[str, list[tuple[int, int]]],
    mapping: AxisMapping,
) -> dict[str, list[tuple[int, int]]]:
    """Resolve overlaps by tracing each curve with shortest-path continuity constraints."""
    clean: dict[str, list[tuple[int, int]]] = {}
    x0, _, x1, _ = mapping.plot_region
    x_range = x1 - x0
    gap_threshold = max(1, int(x_range * 0.05))  # 5% of x-range

    for name, pixels in raw_curves.items():
        if len(pixels) == 0:
            clean[name] = []
            continue

        # 1) Group by x.
        x_to_ys: dict[int, list[int]] = {}
        for px, py in pixels:
            x_to_ys.setdefault(px, []).append(py)

        # 2) Trace smooth monotone path through ambiguous overlap regions.
        traced = _trace_curve_shortest_path(x_to_ys)

        # 3) Fill small gaps for continuity.
        clean[name] = _fill_small_gaps(traced, gap_threshold)

    return clean
