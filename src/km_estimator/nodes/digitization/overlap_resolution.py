"""Overlap resolution: enforce step function, fill gaps."""

import numpy as np

from .axis_calibration import AxisMapping


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


def resolve_overlaps(
    raw_curves: dict[str, list[tuple[int, int]]],
    mapping: AxisMapping,
) -> dict[str, list[tuple[int, int]]]:
    """Clean curves: median y per x, enforce monotonic, fill small gaps."""
    clean: dict[str, list[tuple[int, int]]] = {}
    x0, _, x1, _ = mapping.plot_region
    x_range = x1 - x0
    gap_threshold = max(1, int(x_range * 0.05))  # 5% of x-range

    for name, pixels in raw_curves.items():
        if len(pixels) == 0:
            clean[name] = []
            continue

        # 1. Group by x, take median y (handles line thickness)
        x_to_ys: dict[int, list[int]] = {}
        for px, py in pixels:
            x_to_ys.setdefault(px, []).append(py)

        step_curve: list[tuple[int, int]] = []
        for px in sorted(x_to_ys.keys()):
            ys = x_to_ys[px]
            median_y = int(np.median(ys))
            step_curve.append((px, median_y))

        # 2. Enforce monotonic y (survival curves never increase)
        # In pixel coords: y should never decrease (y=0 is top, higher y = lower survival)
        enforced: list[tuple[int, int]] = []
        max_y = 0
        for px, py in step_curve:
            if py >= max_y:
                max_y = py
                enforced.append((px, py))
            # else: skip impossible increase in survival

        # 3. Fill small gaps with linear interpolation
        filled: list[tuple[int, int]] = []
        for i, (px, py) in enumerate(enforced):
            filled.append((px, py))
            if i < len(enforced) - 1:
                next_px, next_py = enforced[i + 1]
                gap = next_px - px
                if 1 < gap <= gap_threshold:
                    for fill_x in range(px + 1, next_px):
                        ratio = (fill_x - px) / gap
                        fill_y = int(py + ratio * (next_py - py))
                        filled.append((fill_x, fill_y))

        clean[name] = filled

    return clean
