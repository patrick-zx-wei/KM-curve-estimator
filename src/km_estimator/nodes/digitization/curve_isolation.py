"""Curve isolation via k-medoids clustering."""

import cv2
import numpy as np
from numpy.typing import NDArray

from km_estimator.models import CurveInfo, PlotMetadata, ProcessingError, ProcessingStage

from .axis_calibration import AxisMapping

# Common color names to RGB (normalized 0-1)
COLOR_MAP: dict[str, tuple[float, float, float]] = {
    "red": (1.0, 0.0, 0.0),
    "blue": (0.0, 0.0, 1.0),
    "green": (0.0, 0.5, 0.0),
    "black": (0.0, 0.0, 0.0),
    "orange": (1.0, 0.65, 0.0),
    "purple": (0.5, 0.0, 0.5),
    "brown": (0.6, 0.3, 0.0),
    "pink": (1.0, 0.75, 0.8),
    "gray": (0.5, 0.5, 0.5),
    "grey": (0.5, 0.5, 0.5),
    "cyan": (0.0, 1.0, 1.0),
    "magenta": (1.0, 0.0, 1.0),
    "yellow": (1.0, 1.0, 0.0),
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
        return primary

    adaptive = _adaptive_curve_mask(roi, roi_area)
    adaptive_count = int(np.count_nonzero(adaptive))
    adaptive_density = adaptive_count / max(1, roi_area)

    if adaptive_count > primary_count and adaptive_density >= primary_density * 0.9:
        return adaptive
    return primary


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
    pixel_rgb = np.column_stack([
        roi[ys, xs, 2] / 255,  # R
        roi[ys, xs, 1] / 255,  # G
        roi[ys, xs, 0] / 255,  # B
    ]).astype(np.float32)
    expected_rgb = np.asarray([rgb for _, rgb in named_colors], dtype=np.float32)

    d2 = np.sum((pixel_rgb[:, None, :] - expected_rgb[None, :, :]) ** 2, axis=2)
    nearest = np.argmin(d2, axis=1).astype(np.int32)

    curves: dict[str, list[tuple[int, int]]] = {}
    for curve_idx, (curve_name, _) in enumerate(named_colors):
        curve_mask = nearest == curve_idx
        curve_xs = xs[curve_mask] + x0
        curve_ys = ys[curve_mask] + y0
        curves[curve_name] = list(zip(curve_xs.tolist(), curve_ys.tolist()))

    return curves


def isolate_curves(
    image: NDArray,
    meta: PlotMetadata,
    mapping: AxisMapping,
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

    # Extract curve mask with fallback for sparse/degraded cases.
    mask = _extract_curve_mask(roi, min_pixels)

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

    # Build feature matrix: (x, y, R, G, B) with stronger color weighting.
    pixels = np.column_stack([
        (xs / roi.shape[1]) * POSITION_FEATURE_WEIGHT,
        (ys / roi.shape[0]) * POSITION_FEATURE_WEIGHT,
        (roi[ys, xs, 2] / 255) * COLOR_FEATURE_WEIGHT,  # R (BGR order)
        (roi[ys, xs, 1] / 255) * COLOR_FEATURE_WEIGHT,  # G
        (roi[ys, xs, 0] / 255) * COLOR_FEATURE_WEIGHT,  # B
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

    # Compute average color for each cluster
    cluster_colors: dict[int, tuple[float, float, float]] = {}
    for cluster_id in range(k):
        cluster_mask = labels == cluster_id
        if cluster_mask.any():
            avg_r = float(np.mean(roi[ys[cluster_mask], xs[cluster_mask], 2] / 255))
            avg_g = float(np.mean(roi[ys[cluster_mask], xs[cluster_mask], 1] / 255))
            avg_b = float(np.mean(roi[ys[cluster_mask], xs[cluster_mask], 0] / 255))
            cluster_colors[cluster_id] = (avg_r, avg_g, avg_b)

    # Match clusters to curves by color similarity
    # Fall back to median x-position if color info unavailable
    curve_colors = [(i, _parse_color(c)) for i, c in enumerate(meta.curves)]
    has_colors = any(c is not None for _, c in curve_colors)

    curves: dict[str, list[tuple[int, int]]] = {}
    used_clusters: set[int] = set()

    if has_colors and cluster_colors:
        # Match by color: for each curve with color, find best matching cluster
        for curve_idx, expected_color in curve_colors:
            if expected_color is None:
                continue

            best_cluster: int | None = None
            best_dist = float("inf")

            for cluster_id, actual_color in cluster_colors.items():
                if cluster_id in used_clusters:
                    continue
                dist = _color_distance(expected_color, actual_color)
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
    expected_names = [curve.name for curve in meta.curves]
    coverage_issues = [
        curve_name
        for curve_name in expected_names
        if _coverage_issue(curves.get(curve_name, []), x0, x1)
    ]

    all_colors_ok, named_colors = _all_curves_have_distinct_colors(meta)
    if coverage_issues and all_colors_ok:
        color_guided = _assign_by_expected_color(roi, xs, ys, named_colors, x0, y0)
        color_issues = [
            curve_name
            for curve_name in expected_names
            if _coverage_issue(color_guided.get(curve_name, []), x0, x1)
        ]
        if len(color_issues) < len(coverage_issues):
            curves = color_guided

    return curves
