"""Censoring mark detection via robust multi-template matching."""

import cv2
import numpy as np
from numpy.typing import NDArray
from scipy.spatial import cKDTree

from .axis_calibration import AxisMapping

TEMPLATE_SIZES = (5, 7, 9, 11)
PLUS_THRESHOLD = 0.56
X_THRESHOLD = 0.56
T_THRESHOLD = 0.58
TICK_THRESHOLD = 0.60
EDGE_MIN_AREA = 5
EDGE_MAX_AREA = 80
DEDUP_RADIUS_PX = 4.5


def _build_mark_templates() -> list[tuple[NDArray[np.uint8], float]]:
    """Build common censor-mark templates across multiple scales."""
    templates: list[tuple[NDArray[np.uint8], float]] = []
    for size in TEMPLATE_SIZES:
        mid = size // 2

        plus = np.zeros((size, size), dtype=np.uint8)
        plus[mid, :] = 255
        plus[:, mid] = 255
        templates.append((plus, PLUS_THRESHOLD))

        cross = np.zeros((size, size), dtype=np.uint8)
        np.fill_diagonal(cross, 255)
        np.fill_diagonal(np.fliplr(cross), 255)
        templates.append((cross, X_THRESHOLD))

        tmark = np.zeros((size, size), dtype=np.uint8)
        tmark[1, :] = 255
        tmark[1:, mid] = 255
        templates.append((tmark, T_THRESHOLD))

        tick = np.zeros((size, size), dtype=np.uint8)
        tick[:, mid] = 255
        templates.append((tick, TICK_THRESHOLD))
    return templates


def _edge_mark_candidates(gray: NDArray[np.uint8]) -> list[tuple[int, int, float]]:
    """Extract small high-frequency candidates likely to be censor marks."""
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    edges = cv2.Canny(blur, 35, 110)
    kernel = np.ones((2, 2), dtype=np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=1)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(edges, connectivity=8)
    candidates: list[tuple[int, int, float]] = []
    for i in range(1, num_labels):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area < EDGE_MIN_AREA or area > EDGE_MAX_AREA:
            continue
        x, y, w, h = (
            int(stats[i, cv2.CC_STAT_LEFT]),
            int(stats[i, cv2.CC_STAT_TOP]),
            int(stats[i, cv2.CC_STAT_WIDTH]),
            int(stats[i, cv2.CC_STAT_HEIGHT]),
        )
        if w > 18 or h > 18:
            continue
        cx, cy = centroids[i]
        # Prefer compact-ish components over elongated axis fragments.
        aspect = float(max(w, h)) / float(max(1, min(w, h)))
        score = float(np.clip(1.6 - 0.25 * aspect, 0.2, 0.95))
        candidates.append((int(round(cx)), int(round(cy)), score))
    return candidates


def _deduplicate_candidates(
    candidates: list[tuple[int, int, float]],
    radius: float = DEDUP_RADIUS_PX,
) -> list[tuple[int, int, float]]:
    """Deduplicate candidates by score-aware radius suppression."""
    if not candidates:
        return []
    ordered = sorted(candidates, key=lambda item: item[2], reverse=True)
    kept: list[tuple[int, int, float]] = []
    for cand in ordered:
        cx, cy, score = cand
        keep = True
        for kx, ky, _ in kept:
            if (cx - kx) ** 2 + (cy - ky) ** 2 <= radius * radius:
                keep = False
                break
        if keep:
            kept.append((cx, cy, score))
    return kept


def _template_candidates(gray: NDArray[np.uint8]) -> list[tuple[int, int, float]]:
    """Collect candidate marks from multi-shape template matching."""
    candidates: list[tuple[int, int, float]] = []
    for template, threshold in _build_mark_templates():
        result = cv2.matchTemplate(gray, template, cv2.TM_CCOEFF_NORMED)
        ys, xs = np.where(result >= threshold)
        offset_x = template.shape[1] // 2
        offset_y = template.shape[0] // 2
        for py, px in zip(ys, xs):
            score = float(result[py, px])
            candidates.append((int(px + offset_x), int(py + offset_y), score))
    return candidates


def _filter_axis_proximal(
    candidates: list[tuple[int, int, float]],
    mapping: AxisMapping,
) -> list[tuple[int, int, float]]:
    """Remove candidates hugging axis borders where tick labels/axes create false positives."""
    if not candidates:
        return []
    x0, y0, x1, y1 = mapping.plot_region
    x_margin = max(3, int((x1 - x0) * 0.015))
    y_margin = max(3, int((y1 - y0) * 0.015))
    kept: list[tuple[int, int, float]] = []
    for cx, cy, score in candidates:
        if cx <= x0 + x_margin:
            continue
        if cy >= y1 - y_margin:
            continue
        if cy <= y0 + y_margin and cx >= x1 - x_margin:
            continue
        kept.append((cx, cy, score))
    return kept


def detect_censoring(
    image: NDArray,
    curves: dict[str, list[tuple[int, int]]],
    mapping: AxisMapping,
) -> dict[str, list[float]]:
    """Find censoring marks near traced curves, return x-coordinates."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    template_cands = _template_candidates(gray)
    edge_cands = _edge_mark_candidates(gray)
    merged = _deduplicate_candidates(template_cands + edge_cands)
    merged = _filter_axis_proximal(merged, mapping)

    if not merged:
        return {name: [] for name in curves}

    # Build KDTree for each curve for nearest-neighbor lookup.
    censoring: dict[str, list[float]] = {name: [] for name in curves}

    # Adaptive distance threshold: ~2% of plot width, but at least 10 pixels
    x0, _, x1, _ = mapping.plot_region
    roi_width = x1 - x0
    max_dist = max(9, int(roi_width * 0.018))

    curve_trees: dict[str, cKDTree | None] = {}
    for name, pixels in curves.items():
        if pixels:
            curve_trees[name] = cKDTree(pixels)
        else:
            curve_trees[name] = None

    for mx, my, score in merged:
        best_curve: str | None = None
        best_dist = float(max_dist)

        for name, tree in curve_trees.items():
            if tree is None:
                continue
            dist, _ = tree.query([mx, my], k=1)
            score_adjusted_dist = float(dist) / max(0.35, score)
            if score_adjusted_dist < best_dist:
                best_dist = score_adjusted_dist
                best_curve = name

        if best_curve is not None:
            x_real, _ = mapping.px_to_real(mx, my)
            censoring[best_curve].append(x_real)

    # Sort and deduplicate each curve's marks.
    for name in censoring:
        xs = sorted(censoring[name])
        deduped: list[float] = []
        for x_val in xs:
            if not deduped or abs(x_val - deduped[-1]) > 1e-3:
                deduped.append(float(x_val))
        censoring[name] = deduped

    return censoring
