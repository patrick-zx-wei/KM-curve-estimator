"""Curve isolation via k-medoids clustering."""

import cv2
import numpy as np
from numpy.typing import NDArray

from km_estimator.models import PlotMetadata, ProcessingError, ProcessingStage

from .axis_calibration import AxisMapping


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

    # Convert to HSV for color filtering
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    # Mask: exclude white background and black axes/grid
    # Keep pixels with some saturation or mid-range value
    lower_bound = np.array([0, 10, 50])
    upper_bound = np.array([180, 255, 230])
    mask = cv2.inRange(hsv, lower_bound, upper_bound)

    # Get pixel coordinates
    ys, xs = np.where(mask > 0)

    if len(xs) < 10:
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

    # Group pixels by cluster, map to curve names
    curves: dict[str, list[tuple[int, int]]] = {}
    for i, curve_info in enumerate(meta.curves):
        cluster_mask = labels == i
        cluster_xs = xs[cluster_mask] + x0  # back to full image coords
        cluster_ys = ys[cluster_mask] + y0
        curves[curve_info.name] = list(zip(cluster_xs.tolist(), cluster_ys.tolist()))

    return curves
