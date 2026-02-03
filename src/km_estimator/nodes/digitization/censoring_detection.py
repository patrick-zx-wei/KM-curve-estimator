"""Censoring mark detection via template matching."""

import cv2
import numpy as np
from numpy.typing import NDArray
from scipy.spatial import cKDTree

from .axis_calibration import AxisMapping


def detect_censoring(
    image: NDArray,
    curves: dict[str, list[tuple[int, int]]],
    mapping: AxisMapping,
) -> dict[str, list[float]]:
    """Find + symbols near curve paths, return x-coordinates."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Create + templates at multiple sizes
    templates: list[NDArray] = []
    for size in [5, 7, 9, 11]:
        t = np.zeros((size, size), dtype=np.uint8)
        mid = size // 2
        t[mid, :] = 255  # horizontal bar
        t[:, mid] = 255  # vertical bar
        templates.append(t)

    # Find all + matches
    matches: list[tuple[int, int]] = []
    for template in templates:
        result = cv2.matchTemplate(gray, template, cv2.TM_CCOEFF_NORMED)
        locs = np.where(result > 0.6)
        for py, px in zip(*locs):
            cx = px + template.shape[1] // 2
            cy = py + template.shape[0] // 2
            matches.append((cx, cy))

    # Deduplicate (within 5px)
    unique: list[tuple[int, int]] = []
    for m in matches:
        is_dup = False
        for u in unique:
            if abs(m[0] - u[0]) < 5 and abs(m[1] - u[1]) < 5:
                is_dup = True
                break
        if not is_dup:
            unique.append(m)

    if not unique:
        return {name: [] for name in curves}

    # Build KDTree for each curve for O(log n) nearest neighbor lookup
    censoring: dict[str, list[float]] = {name: [] for name in curves}
    max_dist = 15

    curve_trees: dict[str, cKDTree | None] = {}
    for name, pixels in curves.items():
        if pixels:
            curve_trees[name] = cKDTree(pixels)
        else:
            curve_trees[name] = None

    for mx, my in unique:
        best_curve: str | None = None
        best_dist = max_dist

        for name, tree in curve_trees.items():
            if tree is None:
                continue
            dist, _ = tree.query([mx, my], k=1)
            if dist < best_dist:
                best_dist = dist
                best_curve = name

        if best_curve is not None:
            x_real, _ = mapping.px_to_real(mx, my)
            censoring[best_curve].append(x_real)

    # Sort each curve's marks
    for name in censoring:
        censoring[name] = sorted(censoring[name])

    return censoring
