"""Shared plot model for digitization_v2.

This module is the single source of truth for:
- plot bounds
- pixel/value transforms
- x-column grid
- axis/tick penalty masks
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal

import cv2
import numpy as np
from numpy.typing import NDArray

from km_estimator.models import PlotMetadata, ProcessingError, RawOCRTokens
from km_estimator.nodes.digitization.axis_calibration import AxisMapping, calibrate_axes

CurveDirection = Literal["downward", "upward", "unknown"]

AXIS_THICKNESS_RATIO = 0.010
TICK_EXTENT_RATIO = 0.015
TICK_SEARCH_RATIO = 0.035
TICK_SCORE_MIN = 0.020
TICK_MAX_SHIFT_RATIO = 0.08


@dataclass(frozen=True)
class PlotModel:
    """Shared geometric model consumed by all digitization_v2 stages."""

    mapping: AxisMapping
    x_grid: NDArray[np.int32]
    axis_mask: NDArray[np.uint8]  # ROI-local mask (0/255)
    tick_mask: NDArray[np.uint8]  # ROI-local mask (0/255)
    x_tick_anchors: tuple[tuple[int, float], ...]  # global px -> value
    y_tick_anchors: tuple[tuple[int, float], ...]  # global py -> value
    tick_calibration_confidence: float
    curve_direction: CurveDirection
    direction_confidence: float
    warning_codes: tuple[str, ...]

    @property
    def plot_region(self) -> tuple[int, int, int, int]:
        return self.mapping.plot_region

    @property
    def width(self) -> int:
        x0, _, x1, _ = self.mapping.plot_region
        return max(1, x1 - x0)

    @property
    def height(self) -> int:
        _, y0, _, y1 = self.mapping.plot_region
        return max(1, y1 - y0)

    @staticmethod
    def _axis_value_transform(value: float, scale: str) -> float:
        if scale == "log" and value > 0:
            return float(np.log10(value))
        return float(value)

    @staticmethod
    def _axis_value_inverse(value: float, scale: str) -> float:
        if scale == "log":
            return float(10 ** value)
        return float(value)

    @staticmethod
    def _interp_extrap(x: float, xp: NDArray[np.float32], fp: NDArray[np.float32]) -> float:
        if xp.size == 0:
            return float(fp[0]) if fp.size else 0.0
        if xp.size == 1:
            return float(fp[0])
        if x <= float(xp[0]):
            den = float(xp[1] - xp[0])
            if abs(den) < 1e-9:
                return float(fp[0])
            slope = float(fp[1] - fp[0]) / den
            return float(fp[0] + slope * (x - float(xp[0])))
        if x >= float(xp[-1]):
            den = float(xp[-1] - xp[-2])
            if abs(den) < 1e-9:
                return float(fp[-1])
            slope = float(fp[-1] - fp[-2]) / den
            return float(fp[-1] + slope * (x - float(xp[-1])))
        return float(np.interp(x, xp, fp))

    @classmethod
    def _coalesce_anchor_pairs(
        cls,
        anchors: tuple[tuple[int, float], ...],
    ) -> list[tuple[int, float]]:
        by_px: dict[int, list[float]] = {}
        for px, val in anchors:
            by_px.setdefault(int(px), []).append(float(val))
        pairs = [(px, float(np.mean(np.asarray(vals, dtype=np.float32)))) for px, vals in by_px.items()]
        pairs.sort(key=lambda p: p[0])
        return pairs

    @classmethod
    def _coalesce_value_pairs(
        cls,
        anchors: tuple[tuple[int, float], ...],
        scale: str,
    ) -> list[tuple[float, int]]:
        by_val: dict[float, list[int]] = {}
        for px, val in anchors:
            key = round(cls._axis_value_transform(float(val), scale), 8)
            by_val.setdefault(key, []).append(int(px))
        pairs = [(float(k), int(round(float(np.mean(np.asarray(v, dtype=np.float32)))))) for k, v in by_val.items()]
        pairs.sort(key=lambda p: p[0])
        return pairs

    def _px_to_axis_from_anchors(
        self,
        px: int,
        anchors: tuple[tuple[int, float], ...],
        scale: str,
        fallback: float,
    ) -> float:
        if len(anchors) < 2:
            return float(fallback)
        pairs = self._coalesce_anchor_pairs(anchors)
        if len(pairs) < 2:
            return float(fallback)
        xp = np.asarray([p for p, _ in pairs], dtype=np.float32)
        fp = np.asarray(
            [self._axis_value_transform(v, scale) for _, v in pairs],
            dtype=np.float32,
        )
        transformed = self._interp_extrap(float(px), xp, fp)
        return self._axis_value_inverse(transformed, scale)

    def _axis_to_px_from_anchors(
        self,
        value: float,
        anchors: tuple[tuple[int, float], ...],
        scale: str,
        fallback: int,
    ) -> int:
        if len(anchors) < 2:
            return int(fallback)
        pairs = self._coalesce_value_pairs(anchors, scale=scale)
        if len(pairs) < 2:
            return int(fallback)
        xp = np.asarray([v for v, _ in pairs], dtype=np.float32)
        fp = np.asarray([px for _, px in pairs], dtype=np.float32)
        transformed = self._axis_value_transform(float(value), scale)
        px = self._interp_extrap(float(transformed), xp, fp)
        return int(round(px))

    def px_to_real(self, px: int, py: int) -> tuple[float, float]:
        x_lin, y_lin = self.mapping.px_to_real(px, py)
        x_real = self._px_to_axis_from_anchors(
            px=int(px),
            anchors=self.x_tick_anchors,
            scale=self.mapping.x_axis.scale,
            fallback=float(x_lin),
        )
        y_real = self._px_to_axis_from_anchors(
            px=int(py),
            anchors=self.y_tick_anchors,
            scale=self.mapping.y_axis.scale,
            fallback=float(y_lin),
        )
        return (float(x_real), float(y_real))

    def real_to_px(self, x: float, y: float) -> tuple[int, int]:
        px_lin, py_lin = self.mapping.real_to_px(x, y)
        px = self._axis_to_px_from_anchors(
            value=float(x),
            anchors=self.x_tick_anchors,
            scale=self.mapping.x_axis.scale,
            fallback=px_lin,
        )
        py = self._axis_to_px_from_anchors(
            value=float(y),
            anchors=self.y_tick_anchors,
            scale=self.mapping.y_axis.scale,
            fallback=py_lin,
        )
        x0, y0, x1, y1 = self.mapping.plot_region
        px = int(np.clip(px, x0, x1 - 1))
        py = int(np.clip(py, y0, y1 - 1))
        return (px, py)


def _direction_from_text(text: str) -> CurveDirection | None:
    lowered = text.lower()
    if any(
        token in lowered
        for token in ("incidence", "cumulative", "event-free", "hazard", "probability of event")
    ):
        return "upward"
    if any(token in lowered for token in ("survival", "overall survival", "progression-free", "pfs", "os")):
        return "downward"
    return None


def infer_curve_direction(
    meta: PlotMetadata,
    ocr_tokens: RawOCRTokens | None,
) -> tuple[CurveDirection, float, list[str]]:
    """Infer expected curve direction from text first, metadata second."""
    warnings: list[str] = []

    text_candidates: list[str] = []
    if meta.title:
        text_candidates.append(meta.title)
    if meta.y_axis.label:
        text_candidates.append(meta.y_axis.label)
    text_candidates.extend(meta.annotations)
    if ocr_tokens is not None:
        text_candidates.extend(ocr_tokens.axis_labels)
        text_candidates.extend(ocr_tokens.annotations)
        if ocr_tokens.title:
            text_candidates.append(ocr_tokens.title)

    votes: list[CurveDirection] = []
    for chunk in text_candidates:
        vote = _direction_from_text(chunk)
        if vote is not None:
            votes.append(vote)

    if votes:
        n_up = sum(1 for v in votes if v == "upward")
        n_down = sum(1 for v in votes if v == "downward")
        if n_up == n_down:
            # Keep deterministic behavior: fall back to metadata direction
            # rather than unknown/no-constraint tracing.
            fallback = meta.curve_direction if meta.curve_direction in ("upward", "downward") else "unknown"
            warnings.append("W_DIRECTION_AMBIGUOUS_TEXT")
            warnings.append(f"I_DIRECTION_FALLBACK_METADATA:{fallback}")
            if fallback != "unknown":
                return fallback, 0.62, warnings
            return "unknown", 0.45, warnings
        if n_up > n_down:
            return "upward", min(0.95, 0.60 + 0.07 * n_up), warnings
        return "downward", min(0.95, 0.60 + 0.07 * n_down), warnings

    if meta.curve_direction in ("upward", "downward"):
        warnings.append("W_DIRECTION_FROM_METADATA_ONLY")
        return meta.curve_direction, 0.60, warnings

    warnings.append("W_DIRECTION_UNKNOWN")
    return "unknown", 0.35, warnings


def _tick_search_radius(
    expected_positions: list[int],
    idx: int,
    axis_len: int,
) -> int:
    if not expected_positions:
        return max(3, int(round(axis_len * TICK_SEARCH_RATIO)))
    local = max(3, int(round(axis_len * TICK_SEARCH_RATIO)))
    left_gap = None
    right_gap = None
    cur = expected_positions[idx]
    if idx > 0:
        left_gap = abs(cur - expected_positions[idx - 1])
    if idx < len(expected_positions) - 1:
        right_gap = abs(expected_positions[idx + 1] - cur)
    gaps = [g for g in (left_gap, right_gap) if g is not None and g > 1]
    if gaps:
        local = min(local, max(3, int(round(0.45 * min(gaps)))))
    return local


def _build_dark_mask(gray: NDArray[np.uint8]) -> NDArray[np.uint8]:
    h, w = gray.shape
    min_dim = max(9, min(h, w))
    block = int(max(9, min(51, (min_dim // 6) | 1)))
    if block % 2 == 0:
        block += 1
    mask = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        block,
        8,
    )
    mask = cv2.medianBlur(mask, 3)
    return mask


def _score_x_tick(
    dark_mask: NDArray[np.uint8],
    x_pos: int,
    y_axis_row: int,
    tick_extent: int,
) -> float:
    _, w = dark_mask.shape
    x0 = max(0, x_pos - 1)
    x1 = min(w - 1, x_pos + 1)
    y0 = max(0, y_axis_row - tick_extent)
    y1 = max(0, y_axis_row - 1)
    if y1 < y0:
        return 0.0
    patch = dark_mask[y0: y1 + 1, x0: x1 + 1]
    if patch.size == 0:
        return 0.0
    return float(np.mean(patch)) / 255.0


def _score_y_tick(
    dark_mask: NDArray[np.uint8],
    y_pos: int,
    x_axis_col: int,
    tick_extent: int,
) -> float:
    h, w = dark_mask.shape
    x0 = min(w - 1, x_axis_col + 1)
    x1 = min(w - 1, x_axis_col + tick_extent)
    y0 = max(0, y_pos - 1)
    y1 = min(h - 1, y_pos + 1)
    if x1 < x0:
        return 0.0
    patch = dark_mask[y0: y1 + 1, x0: x1 + 1]
    if patch.size == 0:
        return 0.0
    return float(np.mean(patch)) / 255.0


def _refine_positions_1d(
    expected_positions: list[int],
    axis_len: int,
    scorer: Callable[[int], float],
) -> tuple[list[int], float, float]:
    if not expected_positions:
        return [], 0.0, 0.0
    refined: list[int] = []
    scores: list[float] = []
    shifts: list[float] = []
    for idx, expected in enumerate(expected_positions):
        radius = _tick_search_radius(expected_positions, idx, axis_len)
        best_pos = int(expected)
        best_score = -1.0
        lo = max(0, int(expected) - radius)
        hi = min(axis_len - 1, int(expected) + radius)
        for pos in range(lo, hi + 1):
            score = float(scorer(pos))
            if score > best_score:
                best_score = score
                best_pos = pos
        if best_score < TICK_SCORE_MIN:
            best_pos = int(expected)
            best_score = float(scorer(best_pos))
        refined.append(int(best_pos))
        scores.append(float(best_score))
        shifts.append(float(abs(best_pos - int(expected))))
    mean_score = float(np.mean(np.asarray(scores, dtype=np.float32))) if scores else 0.0
    med_shift = float(np.median(np.asarray(shifts, dtype=np.float32))) if shifts else 0.0
    return refined, mean_score, med_shift


def _refine_tick_anchors(
    image: NDArray[np.uint8],
    mapping: AxisMapping,
) -> tuple[tuple[tuple[int, float], ...], tuple[tuple[int, float], ...], float, list[str]]:
    warnings: list[str] = []
    x0, y0, x1, y1 = mapping.plot_region
    roi = image[y0:y1, x0:x1]
    if roi.size == 0:
        warnings.append("W_TICK_CALIB_EMPTY_ROI")
        return (), (), 0.0, warnings
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    dark_mask = _build_dark_mask(gray)

    width = max(1, x1 - x0)
    height = max(1, y1 - y0)
    tick_extent_x = max(2, int(round(width * TICK_EXTENT_RATIO)))
    tick_extent_y = max(2, int(round(height * TICK_EXTENT_RATIO)))

    # Local axis coordinates inside the ROI.
    x_axis_col = int(np.clip(mapping.real_to_px(mapping.x_axis.start, mapping.y_axis.start)[0] - x0, 0, width - 1))
    y_axis_row = int(np.clip(mapping.real_to_px(mapping.x_axis.start, mapping.y_axis.start)[1] - y0, 0, height - 1))

    x_expected: list[int] = []
    x_values: list[float] = []
    for xv in mapping.x_axis.tick_values:
        px, _ = mapping.real_to_px(float(xv), mapping.y_axis.start)
        x_expected.append(int(np.clip(px - x0, 0, width - 1)))
        x_values.append(float(xv))

    y_expected: list[int] = []
    y_values: list[float] = []
    for yv in mapping.y_axis.tick_values:
        _, py = mapping.real_to_px(mapping.x_axis.start, float(yv))
        y_expected.append(int(np.clip(py - y0, 0, height - 1)))
        y_values.append(float(yv))

    x_refined, x_score, x_shift = _refine_positions_1d(
        expected_positions=x_expected,
        axis_len=width,
        scorer=lambda pos: _score_x_tick(dark_mask, pos, y_axis_row=y_axis_row, tick_extent=tick_extent_y),
    )
    y_refined, y_score, y_shift = _refine_positions_1d(
        expected_positions=y_expected,
        axis_len=height,
        scorer=lambda pos: _score_y_tick(dark_mask, pos, x_axis_col=x_axis_col, tick_extent=tick_extent_x),
    )

    x_anchors = tuple((int(x0 + pos), float(val)) for pos, val in zip(x_refined, x_values))
    y_anchors = tuple((int(y0 + pos), float(val)) for pos, val in zip(y_refined, y_values))

    x_quality = 0.0
    y_quality = 0.0
    if x_values:
        x_shift_den = max(1.0, float(width) * TICK_MAX_SHIFT_RATIO)
        x_quality = float(np.clip((x_score / 0.25) * (1.0 - min(1.0, x_shift / x_shift_den)), 0.0, 1.0))
        warnings.append(f"I_TICK_CAL_X:{len(x_values)}:{x_score:.3f}:{x_shift:.1f}")
    else:
        warnings.append("W_TICK_CAL_X_NO_TICKS")
    if y_values:
        y_shift_den = max(1.0, float(height) * TICK_MAX_SHIFT_RATIO)
        y_quality = float(np.clip((y_score / 0.25) * (1.0 - min(1.0, y_shift / y_shift_den)), 0.0, 1.0))
        warnings.append(f"I_TICK_CAL_Y:{len(y_values)}:{y_score:.3f}:{y_shift:.1f}")
    else:
        warnings.append("W_TICK_CAL_Y_NO_TICKS")

    confidence = float(np.clip(0.5 * (x_quality + y_quality), 0.0, 1.0))
    if len(x_anchors) < 2:
        warnings.append("W_TICK_CAL_X_WEAK")
    if len(y_anchors) < 2:
        warnings.append("W_TICK_CAL_Y_WEAK")
    return x_anchors, y_anchors, confidence, warnings


def _draw_tick_mask(
    mask: NDArray[np.uint8],
    mapping: AxisMapping,
    thickness: int,
    x_ticks_px: tuple[int, ...] | None = None,
    y_ticks_py: tuple[int, ...] | None = None,
) -> NDArray[np.uint8]:
    """Draw tick-adjacent penalty regions in ROI-local coordinates."""
    x0, y0, x1, y1 = mapping.plot_region
    width = max(1, x1 - x0)
    height = max(1, y1 - y0)
    tick_extent_x = max(2, int(round(width * TICK_EXTENT_RATIO)))
    tick_extent_y = max(2, int(round(height * TICK_EXTENT_RATIO)))

    # X ticks: vertical little bands near bottom axis.
    x_row = int(np.clip(mapping.real_to_px(mapping.x_axis.start, mapping.y_axis.start)[1] - y0, 0, height - 1))
    x_tick_local = (
        [int(np.clip(px - x0, 0, width - 1)) for px in x_ticks_px]
        if x_ticks_px is not None
        else [
            int(np.clip(mapping.real_to_px(float(xv), mapping.y_axis.start)[0] - x0, 0, width - 1))
            for xv in mapping.x_axis.tick_values
        ]
    )
    for rx in x_tick_local:
        ry = x_row
        cv2.rectangle(
            mask,
            (max(0, rx - thickness), max(0, ry - tick_extent_y)),
            (min(width - 1, rx + thickness), min(height - 1, ry + tick_extent_y)),
            color=255,
            thickness=-1,
        )

    # Y ticks: horizontal little bands near left axis.
    y_col = int(np.clip(mapping.real_to_px(mapping.x_axis.start, mapping.y_axis.start)[0] - x0, 0, width - 1))
    y_tick_local = (
        [int(np.clip(py - y0, 0, height - 1)) for py in y_ticks_py]
        if y_ticks_py is not None
        else [
            int(np.clip(mapping.real_to_px(mapping.x_axis.start, float(yv))[1] - y0, 0, height - 1))
            for yv in mapping.y_axis.tick_values
        ]
    )
    for ry in y_tick_local:
        rx = y_col
        cv2.rectangle(
            mask,
            (max(0, rx - tick_extent_x), max(0, ry - thickness)),
            (min(width - 1, rx + tick_extent_x), min(height - 1, ry + thickness)),
            color=255,
            thickness=-1,
        )
    return mask


def build_plot_model(
    image: NDArray[np.uint8],
    meta: PlotMetadata,
    ocr_tokens: RawOCRTokens | None,
) -> PlotModel | ProcessingError:
    """Build the shared plot geometry and masks for digitization_v2."""
    mapping = calibrate_axes(image, meta)
    if isinstance(mapping, ProcessingError):
        return mapping

    x0, y0, x1, y1 = mapping.plot_region
    width = max(1, x1 - x0)
    height = max(1, y1 - y0)
    min_dim = max(1, min(width, height))
    axis_thickness = max(2, int(round(min_dim * AXIS_THICKNESS_RATIO)))

    axis_mask = np.zeros((height, width), dtype=np.uint8)
    # Draw strong penalties on the two principal axes.
    cv2.line(axis_mask, (0, height - 1), (width - 1, height - 1), color=255, thickness=axis_thickness)
    cv2.line(axis_mask, (0, 0), (0, height - 1), color=255, thickness=axis_thickness)
    axis_mask = cv2.dilate(axis_mask, np.ones((3, 3), dtype=np.uint8), iterations=1)

    x_tick_anchors: tuple[tuple[int, float], ...] = ()
    y_tick_anchors: tuple[tuple[int, float], ...] = ()
    tick_cal_conf = 0.0
    tick_warnings: list[str] = []
    rx, ry, conf, refine_warnings = _refine_tick_anchors(image, mapping)
    if len(rx) >= 2:
        x_tick_anchors = rx
    if len(ry) >= 2:
        y_tick_anchors = ry
    tick_cal_conf = max(tick_cal_conf, float(conf))
    tick_warnings.extend(refine_warnings)

    tick_mask = np.zeros((height, width), dtype=np.uint8)
    tick_mask = _draw_tick_mask(
        tick_mask,
        mapping,
        thickness=max(1, axis_thickness // 2),
        x_ticks_px=tuple(px for px, _ in x_tick_anchors),
        y_ticks_py=tuple(py for py, _ in y_tick_anchors),
    )

    direction, direction_confidence, warning_codes = infer_curve_direction(meta, ocr_tokens)
    warning_codes.extend(tick_warnings)
    x_grid = np.arange(x0, x1, dtype=np.int32)
    if x_grid.size == 0:
        x_grid = np.asarray([x0], dtype=np.int32)

    return PlotModel(
        mapping=mapping,
        x_grid=x_grid,
        axis_mask=axis_mask,
        tick_mask=tick_mask,
        x_tick_anchors=x_tick_anchors,
        y_tick_anchors=y_tick_anchors,
        tick_calibration_confidence=float(tick_cal_conf),
        curve_direction=direction,
        direction_confidence=float(direction_confidence),
        warning_codes=tuple(warning_codes),
    )
