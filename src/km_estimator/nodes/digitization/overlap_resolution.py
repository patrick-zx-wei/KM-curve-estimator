"""Overlap resolution: enforce step function, fill gaps."""

import numpy as np

from .axis_calibration import AxisMapping


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
