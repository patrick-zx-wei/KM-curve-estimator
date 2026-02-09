"""Debug artifact writers for digitization_v2."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from numpy.typing import NDArray

from km_estimator.utils import cv_utils

from .axis_map import PlotModel
from .path_trace import TraceResult
from .probability_map import EvidenceCube


def _safe_write(path: Path, image: NDArray[np.uint8]) -> None:
    cv_utils.save_image(image, path)


def _ambiguity_to_color(ambiguity: NDArray[np.float32]) -> NDArray[np.uint8]:
    scaled = np.clip(ambiguity * 255.0, 0, 255).astype(np.uint8)
    return cv2.applyColorMap(scaled, cv2.COLORMAP_TURBO)


def _draw_tick_overlay(
    image: NDArray[np.uint8],
    plot_model: PlotModel,
) -> NDArray[np.uint8]:
    """Draw pixel positions for x/y ticks with labels."""
    out = image.copy()
    x0, y0, x1, y1 = plot_model.plot_region
    cv2.rectangle(out, (x0, y0), (x1 - 1, y1 - 1), (255, 220, 0), 2, cv2.LINE_AA)

    font = cv2.FONT_HERSHEY_SIMPLEX
    fs = 0.38
    thick = 1

    # X ticks at y-axis start baseline (expected from metadata mapping).
    y_base = float(plot_model.mapping.y_axis.start)
    for xv in plot_model.mapping.x_axis.tick_values:
        px, py = plot_model.mapping.real_to_px(float(xv), y_base)
        px = int(np.clip(px, x0, x1 - 1))
        py = int(np.clip(py, y0, y1 - 1))
        cv2.drawMarker(
            out,  # type: ignore[arg-type]
            (px, py),
            color=(255, 255, 0),
            markerType=cv2.MARKER_CROSS,
            markerSize=10,
            thickness=1,
            line_type=cv2.LINE_AA,
        )
        cv2.putText(
            out,
            f"x={xv:g}",
            (px + 4, min(out.shape[0] - 4, py + 14)),
            font,
            fs,
            (255, 255, 0),
            thick,
            cv2.LINE_AA,
        )

    # Y ticks at x-axis start baseline (expected from metadata mapping).
    x_base = float(plot_model.mapping.x_axis.start)
    for yv in plot_model.mapping.y_axis.tick_values:
        px, py = plot_model.mapping.real_to_px(x_base, float(yv))
        px = int(np.clip(px, x0, x1 - 1))
        py = int(np.clip(py, y0, y1 - 1))
        cv2.drawMarker(
            out,  # type: ignore[arg-type]
            (px, py),
            color=(0, 165, 255),
            markerType=cv2.MARKER_TILTED_CROSS,
            markerSize=10,
            thickness=1,
            line_type=cv2.LINE_AA,
        )
        cv2.putText(
            out,
            f"y={yv:g}",
            (min(out.shape[1] - 80, px + 6), max(12, py - 4)),
            font,
            fs,
            (0, 165, 255),
            thick,
            cv2.LINE_AA,
        )

    # Refined tick anchors used by tick-guided calibration.
    for px, xv in plot_model.x_tick_anchors:
        px = int(np.clip(px, x0, x1 - 1))
        py = int(np.clip(plot_model.mapping.real_to_px(float(xv), y_base)[1], y0, y1 - 1))
        cv2.circle(out, (px, py), 3, (0, 255, 255), -1, cv2.LINE_AA)
        cv2.putText(
            out,
            f"x*={xv:g}",
            (px + 4, max(12, py - 8)),
            font,
            fs,
            (0, 255, 255),
            thick,
            cv2.LINE_AA,
        )
    for py, yv in plot_model.y_tick_anchors:
        py = int(np.clip(py, y0, y1 - 1))
        px = int(np.clip(plot_model.mapping.real_to_px(x_base, float(yv))[0], x0, x1 - 1))
        cv2.circle(out, (px, py), 3, (0, 255, 0), -1, cv2.LINE_AA)
        cv2.putText(
            out,
            f"y*={yv:g}",
            (min(out.shape[1] - 88, px + 8), min(out.shape[0] - 6, py + 10)),
            font,
            fs,
            (0, 255, 0),
            thick,
            cv2.LINE_AA,
        )

    return out


def write_debug_artifacts(
    image: NDArray[np.uint8],
    plot_model: PlotModel,
    evidence: EvidenceCube,
    trace: TraceResult,
    out_dir: Path,
    prefix: str,
) -> list[str]:
    """Persist standard debug overlays for failure triage."""
    warnings: list[str] = []
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
        x0, y0, x1, y1 = plot_model.plot_region
        roi = image[y0:y1, x0:x1].copy()

        # Axis mask overlay.
        axis_overlay = roi.copy()
        axis_red = np.zeros_like(roi)
        axis_red[:, :, 2] = cv2.bitwise_or(plot_model.axis_mask, plot_model.tick_mask)
        axis_overlay = cv2.addWeighted(axis_overlay, 0.78, axis_red, 0.32, 0)
        _safe_write(out_dir / f"{prefix}_axis_mask_overlay.png", axis_overlay)

        # Ambiguity map.
        amb_color = _ambiguity_to_color(evidence.ambiguity_map)
        _safe_write(out_dir / f"{prefix}_ambiguity_map.png", amb_color)

        # Text/line penalty maps.
        text_map = np.clip(evidence.text_penalty_map * 255.0, 0, 255).astype(np.uint8)
        text_region_map = np.clip(evidence.text_region_penalty_map * 255.0, 0, 255).astype(np.uint8)
        line_map = np.clip(evidence.line_penalty_map * 255.0, 0, 255).astype(np.uint8)
        _safe_write(out_dir / f"{prefix}_text_penalty.png", text_map)
        _safe_write(out_dir / f"{prefix}_text_region_penalty.png", text_region_map)
        _safe_write(out_dir / f"{prefix}_line_penalty.png", line_map)

        ui_overlay = roi.copy()
        ui_red = np.zeros_like(roi)
        ui_red[:, :, 2] = cv2.max(text_map, line_map)
        ui_overlay = cv2.addWeighted(ui_overlay, 0.78, ui_red, 0.32, 0)
        _safe_write(out_dir / f"{prefix}_ui_penalty_overlay.png", ui_overlay)

        # Tick pixel anchor overlay (full image coordinates).
        tick_overlay = _draw_tick_overlay(image, plot_model)
        _safe_write(out_dir / f"{prefix}_tick_pixel_overlay.png", tick_overlay)

        # Arm overlays.
        palette = [
            (45, 105, 255),
            (50, 205, 50),
            (255, 165, 0),
            (255, 20, 147),
            (0, 215, 255),
        ]
        traced = roi.copy()
        for idx, name in enumerate(sorted(trace.pixel_curves)):
            color = palette[idx % len(palette)]
            for px, py in trace.pixel_curves[name]:
                rx = int(px - x0)
                ry = int(py - y0)
                if 0 <= rx < traced.shape[1] and 0 <= ry < traced.shape[0]:
                    traced[ry, rx] = color
        _safe_write(out_dir / f"{prefix}_trace_overlay.png", traced)
    except Exception as exc:  # pragma: no cover - debug path should not fail pipeline
        warnings.append(f"W_DEBUG_ARTIFACT_WRITE_FAILED:{exc}")
    return warnings
