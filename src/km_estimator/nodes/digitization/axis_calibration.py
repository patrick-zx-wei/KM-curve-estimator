"""Axis calibration via Hough transform and tick detection."""

from dataclasses import dataclass

import cv2
import numpy as np
from numpy.typing import NDArray

from km_estimator.models import AxisConfig, PlotMetadata, ProcessingError, ProcessingStage, RiskTable


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


def calibrate_axes(
    image: NDArray, meta: PlotMetadata
) -> AxisMapping | ProcessingError:
    """Detect plot region via Hough lines, build coordinate mapping."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape

    # Detect edges and lines
    edges = cv2.Canny(gray, 50, 150)
    lines = cv2.HoughLinesP(
        edges, 1, np.pi / 180, threshold=100, minLineLength=w // 4, maxLineGap=10
    )

    x_axis_y: int | None = None  # y-pixel of horizontal x-axis
    y_axis_x: int | None = None  # x-pixel of vertical y-axis

    if lines is not None:
        horizontals: list[int] = []
        verticals: list[int] = []

        for line in lines:
            lx1, ly1, lx2, ly2 = line[0]
            if abs(ly2 - ly1) < 10:  # horizontal (y nearly same)
                horizontals.append((ly1 + ly2) // 2)
            elif abs(lx2 - lx1) < 10:  # vertical (x nearly same)
                verticals.append((lx1 + lx2) // 2)

        # X-axis: bottom-most horizontal in lower 30% of image
        for y_pos in sorted(horizontals, reverse=True):
            if y_pos > h * 0.7:
                x_axis_y = y_pos
                break

        # Y-axis: left-most vertical in left 30% of image
        for x_pos in sorted(verticals):
            if x_pos < w * 0.3:
                y_axis_x = x_pos
                break

    # Fallback: use image edge percentages
    if x_axis_y is None:
        x_axis_y = int(h * 0.85)
    if y_axis_x is None:
        y_axis_x = int(w * 0.1)

    # Estimate plot region
    plot_x1 = int(w * 0.95)  # right edge
    plot_y0 = int(h * 0.05)  # top edge

    # Validate non-zero region
    if plot_x1 <= y_axis_x or x_axis_y <= plot_y0:
        return ProcessingError(
            stage=ProcessingStage.DIGITIZE,
            error_type="invalid_plot_region",
            recoverable=False,
            message=f"Invalid plot region: ({y_axis_x}, {plot_y0}, {plot_x1}, {x_axis_y})",
            details={"image_size": (w, h)},
        )

    plot_region = (y_axis_x, plot_y0, plot_x1, x_axis_y)

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
            warnings.append(
                f"{curve_name}: {below_count} points below y_axis.start={y_axis.start}"
            )
        if above_count > 0:
            warnings.append(
                f"{curve_name}: {above_count} points above y_axis.end={y_axis.end}"
            )

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
        out_of_range = [
            v for v in axis.tick_values
            if v < axis.start - 0.01 or v > axis.end + 0.01
        ]
        if out_of_range:
            warnings.append(f"{axis_name}: tick values out of range: {out_of_range}")

        # Check tick values are increasing
        if axis.tick_values != sorted(axis.tick_values):
            warnings.append(f"{axis_name}: tick values not in increasing order")

    return warnings
