"""Curve isolation via k-medoids clustering."""

import cv2
import numpy as np
from numpy.typing import NDArray

from km_estimator.models import CurveInfo, PlotMetadata, ProcessingError, ProcessingStage
from km_estimator.utils import cv_utils

from .axis_calibration import AxisMapping

# Common color names to RGB (normalized 0-1)
COLOR_MAP: dict[str, tuple[float, float, float]] = {
    "red": (0.84, 0.15, 0.16),
    "blue": (0.12, 0.47, 0.71),
    "green": (0.17, 0.63, 0.17),
    "black": (0.0, 0.0, 0.0),
    "orange": (1.0, 0.50, 0.05),
    "purple": (0.58, 0.40, 0.74),
    "brown": (0.55, 0.34, 0.29),
    "pink": (0.89, 0.47, 0.76),
    "gray": (0.5, 0.5, 0.5),
    "grey": (0.5, 0.5, 0.5),
    "cyan": (0.09, 0.75, 0.81),
    "magenta": (0.89, 0.10, 0.89),
    "yellow": (0.74, 0.74, 0.13),
}

MAX_KMEDOIDS_FIT_SAMPLES = 15_000
LABEL_ASSIGNMENT_CHUNK_SIZE = 100_000
POSITION_FEATURE_WEIGHT = 1.0
COLOR_FEATURE_WEIGHT = 2.5
LOW_SIGNAL_MASK_MULTIPLIER = 3
MIN_PRIMARY_MASK_DENSITY = 0.0002
MIN_COMPONENT_AREA_RATIO = 0.000003

MIN_COVERAGE_RATIO = 0.8
MAX_START_OFFSET_RATIO = 0.15
MAX_END_GAP_RATIO = 0.15
MIN_COLOR_SEPARATION = 0.15
MAX_MASK_DENSITY = 0.25
SPARSE_INTERPOLATION_GAP_RATIO = 0.03
SPARSE_INTERPOLATION_ABS_MAX_GAP = 28
SPARSE_MIN_POINTS_FOR_COMPLETION = 12
BORDER_LINE_OCCUPANCY_RATIO = 0.65
BORDER_LINE_THICKNESS = 1
BORDER_MARGIN_RATIO = 0.02
TOP_BAND_RESCUE_RATIO = 0.28
TOP_BAND_STRICT_RATIO = 0.18
TOP_BAND_RESCUE_MIN_PIXELS = 24
TOP_BAND_RESCUE_MAX_DENSITY = 0.30
TOP_BAND_EDGE_LOW = 14
TOP_BAND_EDGE_HIGH = 60
TOP_BAND_MIN_SAT_FOR_COLOR_KEEP = 24
TOP_BORDER_PROTECT_ROWS = 3
BORDER_GRAY_SAT_MAX = 55
BORDER_GRAY_VAL_MAX = 245
COLOR_CALIBRATION_ITERS = 2
COLOR_CALIBRATION_BLEND = 0.38
COLOR_CALIBRATION_MIN_SAMPLES = 28
COLOR_CALIBRATION_MIN_SAT = 28
COLOR_CALIBRATION_TRIM_QUANTILE = 18.0


def parse_curve_color(color_description: str | None) -> tuple[float, float, float] | None:
    """Extract RGB from color description like 'solid blue' or 'dashed red'."""
    if not color_description:
        return None
    desc = color_description.lower()
    for color_name, rgb in COLOR_MAP.items():
        if color_name in desc:
            return rgb
    return None


def _parse_color(curve: CurveInfo) -> tuple[float, float, float] | None:
    """Extract expected RGB for one curve."""
    return parse_curve_color(curve.color_description)


def _color_distance(c1: tuple[float, float, float], c2: tuple[float, float, float]) -> float:
    """Euclidean distance between two RGB colors."""
    return float(np.sqrt(sum((a - b) ** 2 for a, b in zip(c1, c2))))


def _rgb01_to_lab(rgb: tuple[float, float, float]) -> tuple[float, float, float]:
    """Convert normalized RGB (0-1) to OpenCV LAB coordinates."""
    r = int(np.clip(round(rgb[0] * 255), 0, 255))
    g = int(np.clip(round(rgb[1] * 255), 0, 255))
    b = int(np.clip(round(rgb[2] * 255), 0, 255))
    pixel = np.array([[[b, g, r]]], dtype=np.uint8)
    lab = cv2.cvtColor(pixel, cv2.COLOR_BGR2LAB)[0, 0].astype(np.float32)
    return float(lab[0]), float(lab[1]), float(lab[2])


def _lab_distance(c1: tuple[float, float, float], c2: tuple[float, float, float]) -> float:
    """Euclidean distance in LAB color space."""
    return float(np.sqrt(sum((a - b) ** 2 for a, b in zip(c1, c2))))


def _assign_to_nearest_medoids(
    pixels: NDArray,
    medoids: NDArray,
    chunk_size: int = LABEL_ASSIGNMENT_CHUNK_SIZE,
) -> NDArray:
    """Assign each pixel to nearest medoid in bounded-memory chunks."""
    n_points = pixels.shape[0]
    labels = np.empty(n_points, dtype=np.int32)

    for start in range(0, n_points, chunk_size):
        end = min(start + chunk_size, n_points)
        chunk = pixels[start:end]
        d2 = np.sum((chunk[:, None, :] - medoids[None, :, :]) ** 2, axis=2)
        labels[start:end] = np.argmin(d2, axis=1).astype(np.int32)

    return labels


def _remove_small_components(mask: NDArray, min_area: int) -> NDArray:
    """Remove tiny connected components from binary mask."""
    if min_area <= 1:
        return mask

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    cleaned = np.zeros_like(mask)

    for label_idx in range(1, num_labels):
        area = int(stats[label_idx, cv2.CC_STAT_AREA])
        if area >= min_area:
            cleaned[labels == label_idx] = 255

    return cleaned


def _adaptive_curve_mask(roi: NDArray, roi_area: int) -> NDArray:
    """Build adaptive curve mask for low-signal/noisy images."""
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    sat = hsv[:, :, 1].astype(np.float32)
    val = hsv[:, :, 2].astype(np.float32)

    sat_threshold = int(max(5, np.percentile(sat, 45)))
    val_low = int(max(20, np.percentile(val, 8)))
    val_high = int(min(245, np.percentile(val, 98)))

    relaxed = cv2.inRange(
        hsv,
        np.array([0, max(5, sat_threshold - 8), val_low], dtype=np.uint8),
        np.array([180, 255, val_high], dtype=np.uint8),
    )

    kernel_size = 3 if min(roi.shape[:2]) < 1400 else 5
    kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
    relaxed = cv2.morphologyEx(relaxed, cv2.MORPH_CLOSE, kernel)
    relaxed = cv2.morphologyEx(relaxed, cv2.MORPH_OPEN, kernel)

    min_component_area = max(4, int(roi_area * MIN_COMPONENT_AREA_RATIO))
    return _remove_small_components(relaxed, min_component_area)


def _sparse_curve_mask(roi: NDArray, roi_area: int) -> NDArray:
    """Edge-assisted rescue mask for sparse/degraded curve traces."""
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3, 3), 0)

    p85 = float(np.percentile(blur, 85))
    high = int(np.clip(max(35, p85 * 0.6), 35, 160))
    low = int(max(12, high * 0.4))
    edges = cv2.Canny(blur, low, high)

    dark_mask = cv2.inRange(gray, 0, 245)
    sparse = cv2.bitwise_and(edges, dark_mask)

    kernel = np.ones((3, 3), dtype=np.uint8)
    sparse = cv2.dilate(sparse, kernel, iterations=1)
    sparse = cv2.morphologyEx(sparse, cv2.MORPH_CLOSE, kernel)

    min_component_area = max(4, int(roi_area * MIN_COMPONENT_AREA_RATIO))
    return _remove_small_components(sparse, min_component_area)


def _top_band_curve_mask(roi: NDArray) -> NDArray:
    """Extra rescue pass focused on the top region where early KM segments live."""
    h, w = roi.shape[:2]
    if h < 12 or w < 12:
        return np.zeros((h, w), dtype=np.uint8)

    top_h = max(8, int(h * TOP_BAND_RESCUE_RATIO))
    strict_h = max(6, int(h * TOP_BAND_STRICT_RATIO))
    top_roi = roi[:top_h, :]
    top_hsv = cv2.cvtColor(top_roi, cv2.COLOR_BGR2HSV)
    top_gray = cv2.cvtColor(top_roi, cv2.COLOR_BGR2GRAY)
    top_blur = cv2.GaussianBlur(top_gray, (3, 3), 0)

    # Keep faint but colored traces.
    color_relaxed = cv2.inRange(
        top_hsv,
        np.array([0, 4, 18], dtype=np.uint8),
        np.array([180, 255, 255], dtype=np.uint8),
    )
    sat_keep = cv2.inRange(
        top_hsv[:, :, 1],
        TOP_BAND_MIN_SAT_FOR_COLOR_KEEP,
        255,
    )

    # Keep thin edges that often survive antialiasing better than color.
    edges = cv2.Canny(top_blur, TOP_BAND_EDGE_LOW, TOP_BAND_EDGE_HIGH)
    edge_dilate = cv2.dilate(edges, np.ones((2, 2), dtype=np.uint8), iterations=1)
    non_bg = cv2.inRange(top_gray, 0, 249)
    rescue = cv2.bitwise_or(cv2.bitwise_and(color_relaxed, non_bg), edge_dilate)
    rescue = cv2.bitwise_or(rescue, sat_keep)

    # Use close-only in top band to avoid eroding thin early segments.
    kernel = np.ones((3, 3), dtype=np.uint8)
    rescue = cv2.morphologyEx(rescue, cv2.MORPH_CLOSE, kernel)
    if strict_h > 0:
        strict_slice = rescue[:strict_h, :]
        strict_slice = cv2.morphologyEx(strict_slice, cv2.MORPH_CLOSE, kernel)
        rescue[:strict_h, :] = strict_slice
    rescue = _remove_small_components(
        rescue,
        min_area=max(3, int((top_h * w) * MIN_COMPONENT_AREA_RATIO)),
    )

    out = np.zeros((h, w), dtype=np.uint8)
    out[:top_h, :] = rescue
    return out


def _merge_top_band_rescue(base_mask: NDArray, roi: NDArray) -> NDArray:
    """Merge top-band rescue mask only when it is sparse enough to be credible."""
    top_mask = _top_band_curve_mask(roi)
    h, w = roi.shape[:2]
    top_h = max(8, int(h * TOP_BAND_RESCUE_RATIO))
    top_count = int(np.count_nonzero(top_mask[:top_h, :]))
    top_density = top_count / max(1, top_h * w)
    base_top_count = int(np.count_nonzero(base_mask[:top_h, :]))
    weak_top_coverage = base_top_count < max(TOP_BAND_RESCUE_MIN_PIXELS // 2, int(0.0003 * top_h * w))
    if (
        top_count >= TOP_BAND_RESCUE_MIN_PIXELS
        and top_density <= TOP_BAND_RESCUE_MAX_DENSITY
    ) or weak_top_coverage:
        return cv2.bitwise_or(base_mask, top_mask)
    return base_mask


def _extract_curve_mask(
    roi: NDArray,
    min_pixels: int,
) -> NDArray:
    """Extract curve mask with low-signal rescue fallback."""
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    roi_area = roi.shape[0] * roi.shape[1]

    primary = cv2.inRange(
        hsv,
        np.array([0, 10, 50], dtype=np.uint8),
        np.array([180, 255, 230], dtype=np.uint8),
    )
    primary_count = int(np.count_nonzero(primary))
    primary_density = primary_count / max(1, roi_area)

    low_signal = (
        primary_count < max(min_pixels * LOW_SIGNAL_MASK_MULTIPLIER, 1)
        or primary_density < MIN_PRIMARY_MASK_DENSITY
    )
    if not low_signal:
        merged = _merge_top_band_rescue(primary, roi)
        return _suppress_border_lines(merged, roi=roi)

    adaptive = _adaptive_curve_mask(roi, roi_area)
    sparse = _sparse_curve_mask(roi, roi_area)
    adaptive_count = int(np.count_nonzero(adaptive))
    adaptive_density = adaptive_count / max(1, roi_area)
    sparse_count = int(np.count_nonzero(sparse))
    sparse_density = sparse_count / max(1, roi_area)
    combined = cv2.bitwise_or(adaptive, sparse)
    combined_count = int(np.count_nonzero(combined))
    combined_density = combined_count / max(1, roi_area)

    best = primary
    best_count = primary_count
    if adaptive_count > best_count and adaptive_density <= MAX_MASK_DENSITY:
        best = adaptive
        best_count = adaptive_count
    if sparse_count > best_count and sparse_density <= MAX_MASK_DENSITY:
        best = sparse
        best_count = sparse_count
    if combined_count > best_count and combined_density <= MAX_MASK_DENSITY:
        best = combined

    best = _merge_top_band_rescue(best, roi)
    return _suppress_border_lines(best, roi=roi)


def _suppress_border_lines(mask: NDArray, roi: NDArray | None = None) -> NDArray:
    """Remove axis-like border lines that contaminate curve clustering."""
    if mask.ndim != 2:
        return mask
    h, w = mask.shape
    if h < 10 or w < 10:
        return mask

    margin_y = max(2, int(h * BORDER_MARGIN_RATIO))
    margin_x = max(2, int(w * BORDER_MARGIN_RATIO))

    cleaned = mask.copy()
    sat: NDArray | None = None
    val: NDArray | None = None
    if roi is not None and roi.shape[:2] == mask.shape:
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        sat = hsv[:, :, 1]
        val = hsv[:, :, 2]

    row_occ = np.count_nonzero(cleaned, axis=1) / max(1, w)
    remove_rows = np.where(row_occ >= BORDER_LINE_OCCUPANCY_RATIO)[0]
    for row in remove_rows:
        if row <= margin_y or row >= h - margin_y - 1:
            y0 = max(0, row - BORDER_LINE_THICKNESS)
            y1 = min(h, row + BORDER_LINE_THICKNESS + 1)
            if sat is None or val is None:
                cleaned[y0:y1, :] = 0
            else:
                region = cleaned[y0:y1, :]
                present = region > 0
                if np.any(present):
                    border_like = (
                        (sat[y0:y1, :] <= BORDER_GRAY_SAT_MAX)
                        & (val[y0:y1, :] <= BORDER_GRAY_VAL_MAX)
                    )
                    if y0 <= TOP_BORDER_PROTECT_ROWS:
                        # Preserve likely curve pixels near top border.
                        border_like &= (sat[y0:y1, :] <= TOP_BAND_MIN_SAT_FOR_COLOR_KEEP)
                    region[present & border_like] = 0

    col_occ = np.count_nonzero(cleaned, axis=0) / max(1, h)
    remove_cols = np.where(col_occ >= BORDER_LINE_OCCUPANCY_RATIO)[0]
    for col in remove_cols:
        if col <= margin_x or col >= w - margin_x - 1:
            x0 = max(0, col - BORDER_LINE_THICKNESS)
            x1 = min(w, col + BORDER_LINE_THICKNESS + 1)
            if sat is None or val is None:
                cleaned[:, x0:x1] = 0
            else:
                region = cleaned[:, x0:x1]
                present = region > 0
                if np.any(present):
                    border_like = (
                        (sat[:, x0:x1] <= BORDER_GRAY_SAT_MAX)
                        & (val[:, x0:x1] <= BORDER_GRAY_VAL_MAX)
                    )
                    if x0 <= max(2, margin_x):
                        border_like &= (sat[:, x0:x1] <= TOP_BAND_MIN_SAT_FOR_COLOR_KEEP)
                    region[present & border_like] = 0

    return cleaned


def _coverage_issue(
    points: list[tuple[int, int]],
    plot_x0: int,
    plot_x1: int,
) -> bool:
    """Check if a curve misses substantial start/end coverage."""
    if not points:
        return True

    xs = [p[0] for p in points]
    min_x = min(xs)
    max_x = max(xs)
    x_range = max(1, plot_x1 - plot_x0)
    coverage_ratio = (max_x - min_x) / x_range

    starts_too_late = min_x > plot_x0 + int(x_range * MAX_START_OFFSET_RATIO)
    ends_too_early = max_x < plot_x1 - int(x_range * MAX_END_GAP_RATIO)
    return coverage_ratio < MIN_COVERAGE_RATIO or starts_too_late or ends_too_early


def _all_curves_have_distinct_colors(
    meta: PlotMetadata,
) -> tuple[bool, list[tuple[str, tuple[float, float, float]]]]:
    """Return whether all curves have parseable and sufficiently distinct colors."""
    parsed: list[tuple[str, tuple[float, float, float]]] = []
    for curve in meta.curves:
        rgb = _parse_color(curve)
        if rgb is None:
            return False, []
        parsed.append((curve.name, rgb))

    for i in range(len(parsed)):
        for j in range(i + 1, len(parsed)):
            if _color_distance(parsed[i][1], parsed[j][1]) < MIN_COLOR_SEPARATION:
                return False, []
    return True, parsed


def _complete_curve_topology(
    points: list[tuple[int, int]],
    plot_x0: int,
    plot_x1: int,
) -> list[tuple[int, int]]:
    """Fill moderate x-gaps while preserving KM-like monotone topology in pixels."""
    if len(points) < SPARSE_MIN_POINTS_FOR_COMPLETION:
        return points

    x_to_ys: dict[int, list[int]] = {}
    for x, y in points:
        x_to_ys.setdefault(int(x), []).append(int(y))
    xs = sorted(x_to_ys.keys())
    if len(xs) < 2:
        return points

    x_range = max(1, plot_x1 - plot_x0)
    max_gap = min(
        SPARSE_INTERPOLATION_ABS_MAX_GAP,
        max(3, int(x_range * SPARSE_INTERPOLATION_GAP_RATIO)),
    )

    dense: list[tuple[int, int]] = []
    prev_x = xs[0]
    prev_y = int(np.median(x_to_ys[prev_x]))
    dense.append((prev_x, prev_y))

    for curr_x in xs[1:]:
        curr_y = int(np.median(x_to_ys[curr_x]))
        gap = curr_x - prev_x
        if 1 < gap <= max_gap:
            for x in range(prev_x + 1, curr_x):
                ratio = (x - prev_x) / gap
                y = int(round(prev_y + ratio * (curr_y - prev_y)))
                dense.append((x, y))
        dense.append((curr_x, curr_y))
        prev_x, prev_y = curr_x, curr_y

    monotone: list[tuple[int, int]] = []
    max_y = dense[0][1]
    for x, y in dense:
        if y < max_y:
            y = max_y
        else:
            max_y = y
        monotone.append((x, y))
    return monotone


def _assign_by_expected_color(
    roi: NDArray,
    xs: NDArray,
    ys: NDArray,
    named_colors: list[tuple[str, tuple[float, float, float]]],
    x0: int,
    y0: int,
) -> dict[str, list[tuple[int, int]]]:
    """
    Assign pixels to curves by nearest expected RGB color.

    This enforces identity priors from MMPU colors and is robust to overlaps/crossings.
    """
    roi_lab = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB).astype(np.float32)
    pixel_lab = roi_lab[ys, xs]
    expected_lab = np.asarray([_rgb01_to_lab(rgb) for _, rgb in named_colors], dtype=np.float32)
    expected_lab = _calibrate_expected_labs_from_pixels(
        roi=roi,
        xs=xs,
        ys=ys,
        expected_lab=expected_lab,
    )

    d2 = np.sum((pixel_lab[:, None, :] - expected_lab[None, :, :]) ** 2, axis=2)
    nearest = np.argmin(d2, axis=1).astype(np.int32)

    # Prevent complete curve collapse in noisy scenes: ensure each curve gets a seed set.
    n_curves = max(1, len(named_colors))
    min_seed_pixels = max(1, xs.shape[0] // (50 * n_curves))
    for curve_idx in range(n_curves):
        if np.any(nearest == curve_idx):
            continue
        top_idx = np.argpartition(d2[:, curve_idx], min_seed_pixels - 1)[:min_seed_pixels]
        nearest[top_idx] = curve_idx

    curves: dict[str, list[tuple[int, int]]] = {}
    for curve_idx, (curve_name, _) in enumerate(named_colors):
        curve_mask = nearest == curve_idx
        curve_xs = xs[curve_mask] + x0
        curve_ys = ys[curve_mask] + y0
        curves[curve_name] = list(zip(curve_xs.tolist(), curve_ys.tolist()))

    return curves


def _calibrate_expected_labs_from_pixels(
    roi: NDArray,
    xs: NDArray,
    ys: NDArray,
    expected_lab: NDArray[np.float32],
) -> NDArray[np.float32]:
    """
    Adapt legend-derived LAB targets to image-specific color shifts.

    Robustly updates each expected color from high-saturation inlier pixels so
    JPEG compression and antialiasing don't break color-based assignment.
    """
    n_curves = int(expected_lab.shape[0])
    if n_curves == 0 or xs.size < max(120, n_curves * 40):
        return expected_lab

    roi_lab = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB).astype(np.float32)
    roi_hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    pixel_lab = roi_lab[ys, xs]
    pixel_sat = roi_hsv[ys, xs, 1].astype(np.float32)

    calibrated = expected_lab.astype(np.float32, copy=True)
    min_keep = max(6, COLOR_CALIBRATION_MIN_SAMPLES // 3)
    for _ in range(COLOR_CALIBRATION_ITERS):
        d2 = np.sum((pixel_lab[:, None, :] - calibrated[None, :, :]) ** 2, axis=2)
        nearest = np.argmin(d2, axis=1).astype(np.int32)
        for curve_idx in range(n_curves):
            assigned = np.where(nearest == curve_idx)[0]
            if assigned.size < COLOR_CALIBRATION_MIN_SAMPLES:
                continue
            saturated = assigned[pixel_sat[assigned] >= COLOR_CALIBRATION_MIN_SAT]
            sample_ids = (
                saturated
                if saturated.size >= (COLOR_CALIBRATION_MIN_SAMPLES // 2)
                else assigned
            )
            sample = pixel_lab[sample_ids]
            if sample.shape[0] >= 12:
                low = np.percentile(sample, COLOR_CALIBRATION_TRIM_QUANTILE, axis=0)
                high = np.percentile(sample, 100.0 - COLOR_CALIBRATION_TRIM_QUANTILE, axis=0)
                inlier_mask = np.logical_and(sample >= low, sample <= high).all(axis=1)
                if int(np.count_nonzero(inlier_mask)) >= min_keep:
                    sample = sample[inlier_mask]
            observed = np.mean(sample, axis=0)
            calibrated[curve_idx] = (
                calibrated[curve_idx] * (1.0 - COLOR_CALIBRATION_BLEND)
                + observed * COLOR_CALIBRATION_BLEND
            )

    return calibrated


def isolate_curves(
    image: NDArray,
    meta: PlotMetadata,
    mapping: AxisMapping,
    artifact_robust: bool = False,
) -> dict[str, list[tuple[int, int]]] | ProcessingError:
    """Extract curve pixels and cluster by color/position."""
    x0, y0, x1, y1 = mapping.plot_region
    roi = image[y0:y1, x0:x1]

    if roi.size == 0:
        return ProcessingError(
            stage=ProcessingStage.DIGITIZE,
            error_type="empty_roi",
            recoverable=False,
            message="Plot region is empty",
        )

    # Calculate ROI dimensions for adaptive thresholds
    roi_height, roi_width = roi.shape[:2]
    roi_area = roi_width * roi_height

    # Adaptive minimum pixel threshold: 0.005% of ROI area, but at least 5 pixels
    min_pixels = max(5, int(roi_area * 0.00005))

    roi_for_mask = roi
    if artifact_robust:
        roi_for_mask = cv_utils.preprocess_plot_roi_for_digitization(roi)
    roi_for_color = roi
    if artifact_robust and roi_for_mask.shape[:2] == roi.shape[:2]:
        roi_for_color = cv2.addWeighted(roi, 0.65, roi_for_mask, 0.35, 0)

    # Extract curve mask with fallback for sparse/degraded cases.
    mask = _extract_curve_mask(roi_for_mask, min_pixels)

    # Get pixel coordinates
    ys, xs = np.where(mask > 0)

    if len(xs) < min_pixels:
        return ProcessingError(
            stage=ProcessingStage.DIGITIZE,
            error_type="no_curve_pixels",
            recoverable=False,
            message="No curve pixels detected in plot region",
            details={"roi_shape": roi.shape, "mask_sum": int(mask.sum())},
        )

    if len(xs) < len(meta.curves):
        return ProcessingError(
            stage=ProcessingStage.DIGITIZE,
            error_type="insufficient_curve_pixels",
            recoverable=False,
            message=(
                f"Detected only {len(xs)} curve pixels for {len(meta.curves)} curves"
            ),
            details={"n_pixels": int(len(xs)), "n_curves": int(len(meta.curves))},
        )

    expected_names = [curve.name for curve in meta.curves]
    all_colors_ok, named_colors = _all_curves_have_distinct_colors(meta)

    # Fast path: if colors are distinct and assignment has full coverage, skip clustering.
    if all_colors_ok:
        color_first = _assign_by_expected_color(roi_for_color, xs, ys, named_colors, x0, y0)
        min_curve_pixels = max(5, min_pixels // max(1, len(expected_names)))
        color_empty = [
            curve_name
            for curve_name in expected_names
            if len(color_first.get(curve_name, [])) < min_curve_pixels
        ]
        color_issues = [
            curve_name
            for curve_name in expected_names
            if _coverage_issue(color_first.get(curve_name, []), x0, x1)
        ]
        if not color_empty and not color_issues:
            for curve_name, points in list(color_first.items()):
                if _coverage_issue(points, x0, x1) or len(points) < SPARSE_MIN_POINTS_FOR_COMPLETION:
                    color_first[curve_name] = _complete_curve_topology(points, x0, x1)
            return color_first

    # Build feature matrix: (x, y, R, G, B) with stronger color weighting.
    pixels = np.column_stack([
        (xs / roi.shape[1]) * POSITION_FEATURE_WEIGHT,
        (ys / roi.shape[0]) * POSITION_FEATURE_WEIGHT,
        (roi_for_color[ys, xs, 2] / 255) * COLOR_FEATURE_WEIGHT,  # R (BGR order)
        (roi_for_color[ys, xs, 1] / 255) * COLOR_FEATURE_WEIGHT,  # G
        (roi_for_color[ys, xs, 0] / 255) * COLOR_FEATURE_WEIGHT,  # B
    ]).astype(np.float32)

    k = len(meta.curves)
    if k == 0:
        return ProcessingError(
            stage=ProcessingStage.DIGITIZE,
            error_type="zero_curves",
            recoverable=False,
            message="PlotMetadata has zero curves",
        )

    # k-medoids clustering
    try:
        from sklearn_extra.cluster import KMedoids

        fit_pixels = pixels
        if pixels.shape[0] > MAX_KMEDOIDS_FIT_SAMPLES:
            rng = np.random.default_rng(42)
            sample_idx = rng.choice(
                pixels.shape[0], size=MAX_KMEDOIDS_FIT_SAMPLES, replace=False
            )
            fit_pixels = pixels[sample_idx]

        kmedoids = KMedoids(n_clusters=k, random_state=42, max_iter=100)
        kmedoids.fit(fit_pixels)

        cluster_centers = getattr(kmedoids, "cluster_centers_", None)
        if cluster_centers is not None:
            medoids = np.asarray(cluster_centers, dtype=np.float32)
        else:
            medoid_indices = getattr(kmedoids, "medoid_indices_", None)
            if medoid_indices is None:
                return ProcessingError(
                    stage=ProcessingStage.DIGITIZE,
                    error_type="clustering_failed",
                    recoverable=True,
                    message="KMedoids did not expose medoid centers",
                )
            medoids = fit_pixels[np.asarray(medoid_indices, dtype=np.int32)]

        if fit_pixels.shape[0] == pixels.shape[0]:
            labels = np.asarray(kmedoids.labels_, dtype=np.int32)
        else:
            labels = _assign_to_nearest_medoids(pixels, medoids)
    except ImportError:
        return ProcessingError(
            stage=ProcessingStage.DIGITIZE,
            error_type="missing_dependency",
            recoverable=False,
            message="sklearn_extra not installed (needed for KMedoids)",
        )
    except Exception as e:
        return ProcessingError(
            stage=ProcessingStage.DIGITIZE,
            error_type="clustering_failed",
            recoverable=True,
            message=str(e),
        )

    # Compute average cluster color in LAB for robust matching under noise.
    cluster_colors_lab: dict[int, tuple[float, float, float]] = {}
    roi_lab = cv2.cvtColor(roi_for_color, cv2.COLOR_BGR2LAB).astype(np.float32)
    for cluster_id in range(k):
        cluster_mask = labels == cluster_id
        if cluster_mask.any():
            cluster_lab = roi_lab[ys[cluster_mask], xs[cluster_mask]]
            avg_lab = tuple(float(v) for v in np.mean(cluster_lab, axis=0))
            cluster_colors_lab[cluster_id] = avg_lab

    # Match clusters to curves by color similarity
    # Fall back to median x-position if color info unavailable
    curve_colors = [(i, _parse_color(c)) for i, c in enumerate(meta.curves)]
    has_colors = any(c is not None for _, c in curve_colors)

    curves: dict[str, list[tuple[int, int]]] = {}
    used_clusters: set[int] = set()

    if has_colors and cluster_colors_lab:
        # Match by color: for each curve with color, find best matching cluster
        for curve_idx, expected_color in curve_colors:
            if expected_color is None:
                continue

            expected_lab = _rgb01_to_lab(expected_color)
            best_cluster: int | None = None
            best_dist = float("inf")

            for cluster_id, actual_lab in cluster_colors_lab.items():
                if cluster_id in used_clusters:
                    continue
                dist = _lab_distance(expected_lab, actual_lab)
                if dist < best_dist:
                    best_dist = dist
                    best_cluster = cluster_id

            if best_cluster is not None:
                used_clusters.add(best_cluster)
                curve_info = meta.curves[curve_idx]
                cluster_mask = labels == best_cluster
                cluster_xs = xs[cluster_mask] + x0
                cluster_ys = ys[cluster_mask] + y0
                curves[curve_info.name] = list(zip(cluster_xs.tolist(), cluster_ys.tolist()))

    # Assign remaining clusters to remaining curves by median x
    remaining_clusters = [c for c in range(k) if c not in used_clusters]
    remaining_curves = [i for i, _ in curve_colors if meta.curves[i].name not in curves]

    if remaining_clusters and remaining_curves:
        # Sort remaining clusters by median x
        cluster_median_x = []
        for cluster_id in remaining_clusters:
            cluster_mask = labels == cluster_id
            if cluster_mask.any():
                median_x = float(np.median(xs[cluster_mask]))
            else:
                median_x = float("inf")
            cluster_median_x.append((cluster_id, median_x))
        cluster_median_x.sort(key=lambda x: x[1])

        for (cluster_id, _), curve_idx in zip(cluster_median_x, remaining_curves):
            curve_info = meta.curves[curve_idx]
            cluster_mask = labels == cluster_id
            cluster_xs = xs[cluster_mask] + x0
            cluster_ys = ys[cluster_mask] + y0
            curves[curve_info.name] = list(zip(cluster_xs.tolist(), cluster_ys.tolist()))

    # Coverage enforcement: if clustering splits curves into partial time windows,
    # fall back to color-prior assignment when MMPU colors are available.
    coverage_issues = [
        curve_name
        for curve_name in expected_names
        if _coverage_issue(curves.get(curve_name, []), x0, x1)
    ]

    if coverage_issues and all_colors_ok:
        color_guided = _assign_by_expected_color(roi_for_color, xs, ys, named_colors, x0, y0)
        min_curve_pixels = max(5, min_pixels // max(1, len(expected_names)))
        color_empty = [
            curve_name
            for curve_name in expected_names
            if len(color_guided.get(curve_name, [])) < min_curve_pixels
        ]
        color_issues = [
            curve_name
            for curve_name in expected_names
            if _coverage_issue(color_guided.get(curve_name, []), x0, x1)
        ]
        if len(color_issues) < len(coverage_issues) and not color_empty:
            curves = color_guided

    for curve_name, points in list(curves.items()):
        if _coverage_issue(points, x0, x1) or len(points) < SPARSE_MIN_POINTS_FOR_COMPLETION:
            curves[curve_name] = _complete_curve_topology(points, x0, x1)

    return curves
