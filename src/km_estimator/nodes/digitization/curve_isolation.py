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


def _parse_color(curve: CurveInfo) -> tuple[float, float, float] | None:
    """Extract RGB from color_description like 'solid blue' or 'dashed red'."""
    if not curve.color_description:
        return None
    desc = curve.color_description.lower()
    for color_name, rgb in COLOR_MAP.items():
        if color_name in desc:
            return rgb
    return None


def _color_distance(c1: tuple[float, float, float], c2: tuple[float, float, float]) -> float:
    """Euclidean distance between two RGB colors."""
    return float(np.sqrt(sum((a - b) ** 2 for a, b in zip(c1, c2))))


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

    # Convert to HSV for color filtering
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    # Mask: exclude white background and black axes/grid
    # Keep pixels with some saturation or mid-range value
    lower_bound = np.array([0, 10, 50])
    upper_bound = np.array([180, 255, 230])
    mask = cv2.inRange(hsv, lower_bound, upper_bound)

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

    # Build feature matrix: (x, y, R, G, B) normalized
    pixels = np.column_stack([
        xs / roi.shape[1],
        ys / roi.shape[0],
        roi[ys, xs, 2] / 255,  # R (BGR order)
        roi[ys, xs, 1] / 255,  # G
        roi[ys, xs, 0] / 255,  # B
    ])

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

        kmedoids = KMedoids(n_clusters=k, random_state=42, max_iter=100)
        labels = kmedoids.fit_predict(pixels)
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

    return curves
