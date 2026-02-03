"""Axis calibration via Hough transform and tick detection."""

from dataclasses import dataclass

import cv2
import numpy as np
from numpy.typing import NDArray

from km_estimator.models import AxisConfig, PlotMetadata, ProcessingError, ProcessingStage


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
        x_real = self.x_axis.start + x_ratio * (self.x_axis.end - self.x_axis.start)
        # y: top→bottom = end→start (inverted in image coords)
        y_ratio = (py - y0) / (y1 - y0) if y1 != y0 else 0
        y_real = self.y_axis.end - y_ratio * (self.y_axis.end - self.y_axis.start)
        # Log scale transform
        if self.x_axis.scale == "log":
            x_real = 10**x_real
        if self.y_axis.scale == "log":
            y_real = 10**y_real
        return (x_real, y_real)

    def real_to_px(self, x: float, y: float) -> tuple[int, int]:
        x0, y0, x1, y1 = self.plot_region
        # Inverse log if needed
        if self.x_axis.scale == "log" and x > 0:
            x = np.log10(x)
        if self.y_axis.scale == "log" and y > 0:
            y = np.log10(y)
        # x
        x_range = self.x_axis.end - self.x_axis.start
        x_ratio = (x - self.x_axis.start) / x_range if x_range != 0 else 0
        px = int(x0 + x_ratio * (x1 - x0))
        # y (inverted)
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
