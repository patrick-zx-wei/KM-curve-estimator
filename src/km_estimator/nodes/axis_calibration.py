"""Axis calibration via Hough transform and tick detection."""

from dataclasses import dataclass

import cv2
import numpy as np
from numpy.typing import NDArray

from km_estimator.models import (
    AxisConfig,
    PlotMetadata,
    ProcessingError,
    ProcessingStage,
    RiskTable,
)


@dataclass
class AxisMapping:
    """Pixel to real-unit coordinate mapping."""

    plot_region: tuple[int, int, int, int]  # (x0, y0, x1, y1) in pixels
    x_axis: AxisConfig
    y_axis: AxisConfig

    def px_to_real(self, px: int, py: int) -> tuple[float, float]:
        x0, y0, x1, y1 = self.plot_region
        # x: left→right = start→end
        x_ratio = (px - x0) / (x1 - x0) if x1 != x0 else 0
        # y: top→bottom = end→start (inverted in image coords)
        y_ratio = (py - y0) / (y1 - y0) if y1 != y0 else 0

        # For log scale, interpolate in log space then exponentiate
        if self.x_axis.scale == "log" and self.x_axis.start > 0 and self.x_axis.end > 0:
            log_start = np.log10(self.x_axis.start)
            log_end = np.log10(self.x_axis.end)
            x_real = 10 ** (log_start + x_ratio * (log_end - log_start))
        else:
            x_real = self.x_axis.start + x_ratio * (self.x_axis.end - self.x_axis.start)

        if self.y_axis.scale == "log" and self.y_axis.start > 0 and self.y_axis.end > 0:
            log_start = np.log10(self.y_axis.start)
            log_end = np.log10(self.y_axis.end)
            y_real = 10 ** (log_end - y_ratio * (log_end - log_start))
        else:
            y_real = self.y_axis.end - y_ratio * (self.y_axis.end - self.y_axis.start)

        return (x_real, y_real)

    def real_to_px(self, x: float, y: float) -> tuple[int, int]:
        x0, y0, x1, y1 = self.plot_region

        # For log scale, compute ratio in log space
        if self.x_axis.scale == "log" and x > 0 and self.x_axis.start > 0 and self.x_axis.end > 0:
            log_start = np.log10(self.x_axis.start)
            log_end = np.log10(self.x_axis.end)
            log_range = log_end - log_start
            x_ratio = (np.log10(x) - log_start) / log_range if log_range != 0 else 0
        else:
            x_range = self.x_axis.end - self.x_axis.start
            x_ratio = (x - self.x_axis.start) / x_range if x_range != 0 else 0
        px = int(x0 + x_ratio * (x1 - x0))

        if self.y_axis.scale == "log" and y > 0 and self.y_axis.start > 0 and self.y_axis.end > 0:
            log_start = np.log10(self.y_axis.start)
            log_end = np.log10(self.y_axis.end)
            log_range = log_end - log_start
            y_ratio = (log_end - np.log10(y)) / log_range if log_range != 0 else 0
        else:
            y_range = self.y_axis.end - self.y_axis.start
            y_ratio = (self.y_axis.end - y) / y_range if y_range != 0 else 0
        py = int(y0 + y_ratio * (y1 - y0))

        return (px, py)


def _build_ink_mask(gray: NDArray[np.uint8]) -> NDArray[np.uint8]:
    """Create a robust dark-ink mask for axis extent refinement."""
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    _, mask_otsu = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    _, mask_fixed = cv2.threshold(blur, 160, 255, cv2.THRESH_BINARY_INV)
    mask = cv2.bitwise_or(mask_otsu, mask_fixed)
    mask = cv2.medianBlur(mask, 3)
    return mask


def _find_axis_segment_1d(
    signal: NDArray[np.float32],
    anchor_idx: int,
    min_len: int,
) -> tuple[int, int] | None:
    """
    Find a dominant active segment in a 1D axis signal.

    Prefer a segment containing the axis anchor; otherwise choose the longest active segment.
    """
    n = int(signal.size)
    if n <= 0:
        return None
    q80 = float(np.percentile(signal, 80.0))
    q90 = float(np.percentile(signal, 90.0))
    thr = max(0.01, 0.55 * q80, 0.35 * q90)
    active = (signal >= thr).astype(np.uint8)
    if np.count_nonzero(active) == 0:
        return None

    # Bridge short gaps to handle dashed/aliased axis strokes.
    kernel = np.ones(5, dtype=np.uint8)
    bridged = np.convolve(active, kernel, mode="same") > 0
    arr = bridged.astype(np.uint8)

    segments: list[tuple[int, int]] = []
    start = None
    for idx, v in enumerate(arr):
        if v and start is None:
            start = idx
        elif not v and start is not None:
            segments.append((start, idx - 1))
            start = None
    if start is not None:
        segments.append((start, n - 1))
    if not segments:
        return None

    filtered = [seg for seg in segments if (seg[1] - seg[0] + 1) >= max(3, int(min_len))]
    if not filtered:
        filtered = segments

    anchor = int(np.clip(anchor_idx, 0, n - 1))
    containing = [seg for seg in filtered if seg[0] <= anchor <= seg[1]]
    if containing:
        # Pick the widest segment containing the anchor.
        containing.sort(key=lambda s: (s[1] - s[0], -s[0]), reverse=True)
        return containing[0]

    # Fallback: widest segment.
    filtered.sort(key=lambda s: (s[1] - s[0], -s[0]), reverse=True)
    return filtered[0]


def _refine_plot_extents_from_axis_ink(
    gray: NDArray[np.uint8],
    x_axis_y: int,
    y_axis_x: int,
    top_border_y: int,
    right_border_x: int,
) -> tuple[int, int]:
    """
    Refine top/right plot borders using ink continuity along detected axis bands.

    This corrects cases where Hough/projection picks a truncated right/top border.
    """
    h, w = gray.shape
    ink = _build_ink_mask(gray)
    row_band = max(2, int(round(h * 0.008)))
    col_band = max(2, int(round(w * 0.008)))

    # Horizontal axis span along x-axis row neighborhood.
    ry0 = max(0, int(x_axis_y) - row_band)
    ry1 = min(h, int(x_axis_y) + row_band + 1)
    row_patch = ink[ry0:ry1, :]
    if row_patch.size > 0:
        row_signal = (np.mean(row_patch, axis=0) / 255.0).astype(np.float32, copy=False)
        seg = _find_axis_segment_1d(
            row_signal,
            anchor_idx=int(y_axis_x),
            min_len=max(8, int(w * 0.28)),
        )
        if seg is not None:
            _, seg_end = seg
            right_border_x = max(int(right_border_x), int(seg_end))

    # Vertical axis span along y-axis column neighborhood.
    cx0 = max(0, int(y_axis_x) - col_band)
    cx1 = min(w, int(y_axis_x) + col_band + 1)
    col_patch = ink[:, cx0:cx1]
    if col_patch.size > 0:
        col_signal = (np.mean(col_patch, axis=1) / 255.0).astype(np.float32, copy=False)
        seg = _find_axis_segment_1d(
            col_signal,
            anchor_idx=int(x_axis_y),
            min_len=max(8, int(h * 0.28)),
        )
        if seg is not None:
            seg_start, _ = seg
            top_border_y = min(int(top_border_y), int(seg_start))

    return int(top_border_y), int(right_border_x)


def calibrate_axes(image: NDArray[np.uint8], meta: PlotMetadata) -> AxisMapping | ProcessingError:
    """Detect plot region via Hough lines, build coordinate mapping."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape

    # Detect edges and lines
    edges = cv2.Canny(gray, 45, 140)
    lines = cv2.HoughLinesP(
        edges, 1, np.pi / 180, threshold=80, minLineLength=max(w, h) // 5, maxLineGap=8
    )

    x_axis_y: int | None = None  # y-pixel of horizontal x-axis
    y_axis_x: int | None = None  # x-pixel of vertical y-axis
    top_border_y: int | None = None
    right_border_x: int | None = None

    if lines is not None:
        horizontals: list[tuple[int, int]] = []  # (y, length)
        verticals: list[tuple[int, int]] = []  # (x, length)

        for line in lines:
            coords = np.asarray(line, dtype=np.int32).reshape(-1)
            if coords.size < 4:
                continue
            lx1, ly1, lx2, ly2 = (int(coords[0]), int(coords[1]), int(coords[2]), int(coords[3]))
            line_len = int(np.hypot(lx2 - lx1, ly2 - ly1))
            if abs(ly2 - ly1) < 8 and line_len >= int(w * 0.35):  # horizontal
                horizontals.append(((ly1 + ly2) // 2, line_len))
            elif abs(lx2 - lx1) < 8 and line_len >= int(h * 0.35):  # vertical
                verticals.append(((lx1 + lx2) // 2, line_len))

        # X-axis: bottom-most strong horizontal in lower half of image.
        for y_pos, _ in sorted(horizontals, key=lambda item: (item[0], item[1]), reverse=True):
            if y_pos > h * 0.5:
                x_axis_y = y_pos
                break

        # Y-axis: left-most strong vertical in left half of image.
        for x_pos, _ in sorted(verticals, key=lambda item: (item[0], -item[1])):
            if x_pos < w * 0.5:
                y_axis_x = x_pos
                break

        if x_axis_y is not None:
            top_candidates = [
                (y_pos, length)
                for y_pos, length in horizontals
                if y_pos < x_axis_y - max(6, int(h * 0.03))
            ]
            if top_candidates:
                top_border_y = min(top_candidates, key=lambda item: item[0])[0]

        if y_axis_x is not None:
            right_candidates = [
                (x_pos, length)
                for x_pos, length in verticals
                if x_pos > y_axis_x + max(6, int(w * 0.03))
            ]
            if right_candidates:
                right_border_x = max(right_candidates, key=lambda item: item[0])[0]

    # Projection fallback for missing border detections.
    if x_axis_y is None or y_axis_x is None or top_border_y is None or right_border_x is None:
        edge_row_density = np.count_nonzero(edges, axis=1) / max(1, w)
        edge_col_density = np.count_nonzero(edges, axis=0) / max(1, h)

        row_floor = max(0.015, float(np.percentile(edge_row_density, 80)))
        col_floor = max(0.015, float(np.percentile(edge_col_density, 80)))

        if x_axis_y is None:
            lower_slice = edge_row_density[int(h * 0.55) :]
            if lower_slice.size > 0 and float(np.max(lower_slice)) >= row_floor:
                x_axis_y = int(int(h * 0.55) + int(np.argmax(lower_slice)))

        if y_axis_x is None:
            left_slice = edge_col_density[: int(w * 0.45)]
            if left_slice.size > 0 and float(np.max(left_slice)) >= col_floor:
                y_axis_x = int(np.argmax(left_slice))

        if top_border_y is None:
            upper_slice = edge_row_density[: int(h * 0.45)]
            if upper_slice.size > 0 and float(np.max(upper_slice)) >= row_floor:
                top_border_y = int(np.argmax(upper_slice))

        if right_border_x is None:
            right_start = int(w * 0.55)
            right_slice = edge_col_density[right_start:]
            if right_slice.size > 0 and float(np.max(right_slice)) >= col_floor:
                right_border_x = int(right_start + int(np.argmax(right_slice)))

    # Percentage fallback when lines/projections are weak.
    if x_axis_y is None:
        x_axis_y = int(h * 0.85)
    if y_axis_x is None:
        y_axis_x = int(w * 0.1)
    if top_border_y is None:
        top_border_y = int(h * 0.05)
    if right_border_x is None:
        right_border_x = int(w * 0.95)

    # Refine top/right extents from axis-band ink continuity to avoid truncated plot widths.
    top_border_y, right_border_x = _refine_plot_extents_from_axis_ink(
        gray=gray,
        x_axis_y=int(x_axis_y),
        y_axis_x=int(y_axis_x),
        top_border_y=int(top_border_y),
        right_border_x=int(right_border_x),
    )

    # Sanity clamping to avoid degenerate regions.
    top_border_y = int(np.clip(top_border_y, 0, h - 2))
    x_axis_y = int(np.clip(x_axis_y, top_border_y + 2, h - 1))
    y_axis_x = int(np.clip(y_axis_x, 0, w - 2))
    right_border_x = int(np.clip(right_border_x, y_axis_x + 2, w - 1))

    # Validate non-zero region
    if right_border_x <= y_axis_x or x_axis_y <= top_border_y:
        return ProcessingError(
            stage=ProcessingStage.DIGITIZE,
            error_type="invalid_plot_region",
            recoverable=False,
            message=(
                f"Invalid plot region: ({y_axis_x}, {top_border_y}, {right_border_x}, {x_axis_y})"
            ),
            details={"image_size": (w, h)},
        )

    plot_region = (y_axis_x, top_border_y, right_border_x, x_axis_y)

    return AxisMapping(
        plot_region=plot_region,
        x_axis=meta.x_axis,
        y_axis=meta.y_axis,
    )


def calculate_anchors_from_risk_table(
    risk_table: RiskTable | None,
    curve_names: list[str],
) -> dict[str, list[tuple[float, float]]]:
    """
    Calculate approximate survival anchors from number-at-risk data.

    These anchors are LOWER BOUNDS (typically 5-15% below actual survival)
    because number-at-risk decreases from both events AND censoring,
    while survival probability only decreases from events.

    Args:
        risk_table: Risk table with time points and patient counts per group
        curve_names: Names of curves to calculate anchors for

    Returns:
        Dict mapping curve name to list of (time, survival_lower_bound) tuples
    """
    if risk_table is None:
        return {}

    anchors: dict[str, list[tuple[float, float]]] = {}

    for group in risk_table.groups:
        # Match group name to curve names (case-insensitive)
        matched_name = None
        for name in curve_names:
            if name.lower() == group.name.lower():
                matched_name = name
                break
        if matched_name is None:
            continue

        if not group.counts or group.counts[0] <= 0:
            continue

        initial_count = group.counts[0]
        anchors[matched_name] = [
            (time, count / initial_count)
            for time, count in zip(risk_table.time_points, group.counts)
            if count >= 0
        ]

    return anchors


def validate_against_anchors(
    digitized_curves: dict[str, list[tuple[float, float]]],
    anchors: dict[str, list[tuple[float, float]]],
    tolerance: float = 0.05,
) -> list[str]:
    """
    Validate that digitized curves are above anchor points (within tolerance).

    Since anchors are lower bounds (derived from number-at-risk which includes
    censoring), the actual survival curve should be ABOVE these values.

    Args:
        digitized_curves: Dict of curve name to list of (time, survival) points
        anchors: Dict of curve name to list of (time, survival_lower_bound) points
        tolerance: How far below anchor is acceptable (default 5%)

    Returns:
        List of warning messages for curves that fall below anchors
    """
    warnings: list[str] = []

    for curve_name, anchor_points in anchors.items():
        if curve_name not in digitized_curves:
            continue

        curve = digitized_curves[curve_name]
        if not curve:
            continue

        for anchor_time, anchor_survival in anchor_points:
            # Find closest digitized point by time
            closest = min(curve, key=lambda p: abs(p[0] - anchor_time))
            time_diff = abs(closest[0] - anchor_time)

            # Only validate if we have a point close enough in time
            if time_diff > anchor_time * 0.1 + 1:  # 10% of time + 1 unit tolerance
                continue

            if closest[1] < anchor_survival - tolerance:
                warnings.append(
                    f"{curve_name} at t={anchor_time:.1f}: digitized survival "
                    f"{closest[1]:.3f} below anchor {anchor_survival:.3f}"
                )

    return warnings


def validate_axis_bounds(
    digitized_curves: dict[str, list[tuple[float, float]]],
    y_axis: AxisConfig,
) -> list[str]:
    """
    Validate that digitized points respect y-axis bounds.

    This catches issues with truncated y-axes (e.g., starting at 0.3 instead of 0).

    Args:
        digitized_curves: Dict of curve name to list of (time, survival) points
        y_axis: Y-axis configuration with start/end values

    Returns:
        List of warning messages for out-of-bounds points
    """
    warnings: list[str] = []
    tolerance = 0.02  # Allow 2% out of bounds for measurement noise

    for curve_name, points in digitized_curves.items():
        below_count = 0
        above_count = 0

        for x, y in points:
            if y < y_axis.start - tolerance:
                below_count += 1
            if y > y_axis.end + tolerance:
                above_count += 1

        if below_count > 0:
            warnings.append(f"{curve_name}: {below_count} points below y_axis.start={y_axis.start}")
        if above_count > 0:
            warnings.append(f"{curve_name}: {above_count} points above y_axis.end={y_axis.end}")

    return warnings


def validate_axis_config(axis: AxisConfig, axis_name: str) -> list[str]:
    """
    Validate that axis configuration is sensible.

    Checks:
    - start < end
    - tick_values are within [start, end]
    - tick_values are in increasing order

    Args:
        axis: Axis configuration to validate
        axis_name: Name for error messages (e.g., "x_axis", "y_axis")

    Returns:
        List of warning messages for invalid configurations
    """
    warnings: list[str] = []

    # Check start < end
    if axis.start >= axis.end:
        warnings.append(f"{axis_name}: start ({axis.start}) >= end ({axis.end})")

    if axis.tick_values:
        # Check tick values are in range
        out_of_range = [v for v in axis.tick_values if v < axis.start - 0.01 or v > axis.end + 0.01]
        if out_of_range:
            warnings.append(f"{axis_name}: tick values out of range: {out_of_range}")

        # Check tick values are increasing
        if axis.tick_values != sorted(axis.tick_values):
            warnings.append(f"{axis_name}: tick values not in increasing order")

    return warnings
