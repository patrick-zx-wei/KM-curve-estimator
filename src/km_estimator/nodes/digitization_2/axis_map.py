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
TICK_SCORE_DELTA_MIN = 0.010
TICK_EDGE_HIT_REJECT_FRAC = 0.45
TICK_GAIN_REJECT_MAX = 0.015
TICK_SHIFT_REJECT_RATIO = 0.020
TICK_SCORE_REJECT_MAX = 0.20
TICK_PEAK_MIN_DIST_RATIO = 0.035
TICK_PEAK_QUANTILE = 75.0
TICK_PEAK_REL_THRESHOLD = 0.70
TICK_DETECT_BAND_RATIO = 0.025
TICK_DETECT_MIN_COUNT = 2
TICK_ASSIGN_TOL_RATIO = 0.06


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


def _detect_tick_candidates_x(
    dark_mask: NDArray[np.uint8],
    axis_row: int,
    axis_col: int,
) -> tuple[list[int], float]:
    """Independently detect x-axis tick pixel positions (columns)."""
    h, w = dark_mask.shape
    min_dim = max(1, min(h, w))
    kernel_h = max(3, int(round(h * 0.018)))
    vert_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, kernel_h))
    vert = cv2.morphologyEx(dark_mask, cv2.MORPH_OPEN, vert_kernel)

    band = max(2, int(round(min_dim * TICK_DETECT_BAND_RATIO)))
    y0 = max(0, int(axis_row) - band)
    y1 = min(h - 1, int(axis_row) + band)
    if y1 <= y0:
        return [], 0.0

    profile = np.sum(vert[y0: y1 + 1, :], axis=0).astype(np.float32) / 255.0

    # Exclude y-axis spine neighborhood to avoid counting the axis as a tick.
    excl = max(2, int(round(w * 0.015)))
    c0 = max(0, int(axis_col) - excl)
    c1 = min(w, int(axis_col) + excl + 1)
    profile[c0:c1] = 0.0

    smooth = _smooth_1d(profile.astype(np.float32, copy=False))
    q = float(np.percentile(smooth, 80.0)) if smooth.size else 0.0
    threshold = max(1.0, q * 0.65)
    min_dist = max(2, int(round(float(w) * TICK_PEAK_MIN_DIST_RATIO)))
    peaks = _find_peaks_1d(smooth, threshold=threshold, min_distance=min_dist)
    mean_score = float(np.mean(np.asarray([smooth[p] for p in peaks], dtype=np.float32))) if peaks else 0.0
    return peaks, mean_score


def _detect_tick_candidates_y(
    dark_mask: NDArray[np.uint8],
    axis_col: int,
    axis_row: int,
) -> tuple[list[int], float]:
    """Independently detect y-axis tick pixel positions (rows)."""
    h, w = dark_mask.shape
    min_dim = max(1, min(h, w))
    kernel_w = max(3, int(round(w * 0.018)))
    horiz_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_w, 1))
    horiz = cv2.morphologyEx(dark_mask, cv2.MORPH_OPEN, horiz_kernel)

    band = max(2, int(round(min_dim * TICK_DETECT_BAND_RATIO)))
    x0 = max(0, int(axis_col) - band)
    x1 = min(w - 1, int(axis_col) + band)
    if x1 <= x0:
        return [], 0.0

    profile = np.sum(horiz[:, x0: x1 + 1], axis=1).astype(np.float32) / 255.0

    # Exclude x-axis spine neighborhood.
    excl = max(2, int(round(h * 0.015)))
    r0 = max(0, int(axis_row) - excl)
    r1 = min(h, int(axis_row) + excl + 1)
    profile[r0:r1] = 0.0

    smooth = _smooth_1d(profile.astype(np.float32, copy=False))
    q = float(np.percentile(smooth, 80.0)) if smooth.size else 0.0
    threshold = max(1.0, q * 0.65)
    min_dist = max(2, int(round(float(h) * TICK_PEAK_MIN_DIST_RATIO)))
    peaks = _find_peaks_1d(smooth, threshold=threshold, min_distance=min_dist)
    mean_score = float(np.mean(np.asarray([smooth[p] for p in peaks], dtype=np.float32))) if peaks else 0.0
    return peaks, mean_score


def _monotonic_subset_match(
    expected: list[int],
    detected: list[int],
) -> list[int]:
    """
    Assign an ordered subset of detected positions to expected positions.

    Both inputs must be sorted and len(detected) >= len(expected) >= 2.
    """
    n = len(expected)
    m = len(detected)
    if n < 2 or m < n:
        return expected

    inf = 1e18
    cost = np.full((n, m), inf, dtype=np.float64)
    prev = np.full((n, m), -1, dtype=np.int32)

    for j in range(m):
        cost[0, j] = abs(float(expected[0]) - float(detected[j]))

    for i in range(1, n):
        best_val = inf
        best_idx = -1
        for j in range(i, m):
            candidate_prev = cost[i - 1, j - 1]
            if candidate_prev < best_val:
                best_val = candidate_prev
                best_idx = j - 1
            if best_idx >= 0 and best_val < inf:
                cost[i, j] = best_val + abs(float(expected[i]) - float(detected[j]))
                prev[i, j] = best_idx

    j_end = int(np.argmin(cost[n - 1, n - 1:])) + (n - 1)
    if not np.isfinite(cost[n - 1, j_end]):
        return expected

    out = [0] * n
    j = j_end
    for i in range(n - 1, -1, -1):
        out[i] = int(detected[j])
        if i > 0:
            j = int(prev[i, j])
            if j < 0:
                return expected
    return out


def _assign_ticks_to_values(
    expected_positions: list[int],
    detected_positions: list[int],
    axis_len: int,
) -> tuple[list[int], float, float]:
    """
    Match independently detected ticks to value-tick sequence.

    Returns (assigned_positions, match_ratio, median_shift).
    """
    if not expected_positions:
        return [], 0.0, 0.0
    expected = [int(v) for v in expected_positions]
    detected = sorted(set(int(v) for v in detected_positions))
    if len(detected) < TICK_DETECT_MIN_COUNT:
        return expected, 0.0, 0.0

    assigned = expected.copy()
    hits = 0
    tol = max(3, int(round(float(axis_len) * TICK_ASSIGN_TOL_RATIO)))

    if len(detected) >= len(expected) and len(expected) >= 2:
        assigned = _monotonic_subset_match(expected=expected, detected=detected)
        hits = len(assigned)
    else:
        # Sparse detection: snap only when candidate is plausibly close.
        n = len(expected)
        m = len(detected)
        if n >= 2 and m >= 1:
            idxs = np.round(np.linspace(0, m - 1, n)).astype(int).tolist()
            for i, j in enumerate(idxs):
                cand = int(detected[int(np.clip(j, 0, m - 1))])
                if abs(cand - expected[i]) <= tol:
                    assigned[i] = cand
                    hits += 1

    shifts = np.abs(np.asarray(assigned, dtype=np.float32) - np.asarray(expected, dtype=np.float32))
    med_shift = float(np.median(shifts)) if shifts.size else 0.0
    match_ratio = float(hits) / float(max(1, len(expected)))
    return assigned, match_ratio, med_shift


def _independent_tick_anchors(
    image: NDArray[np.uint8],
    mapping: AxisMapping,
) -> tuple[tuple[tuple[int, float], ...], tuple[tuple[int, float], ...], float, list[str]]:
    """
    Independent tick detection -> axis association -> value assignment.

    1) Detect tick candidates in pixel space without value labels.
    2) Associate candidates to x/y axes by geometry.
    3) Match to ordered axis tick values.
    """
    warnings: list[str] = []
    x0, y0, x1, y1 = mapping.plot_region
    roi = image[y0:y1, x0:x1]
    if roi.size == 0:
        return (), (), 0.0, ["W_TICK_CALIB_EMPTY_ROI"]

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    dark_mask = _build_dark_mask(gray)
    width = max(1, x1 - x0)
    height = max(1, y1 - y0)

    axis_col = int(np.clip(mapping.real_to_px(mapping.x_axis.start, mapping.y_axis.start)[0] - x0, 0, width - 1))
    axis_row = int(np.clip(mapping.real_to_px(mapping.x_axis.start, mapping.y_axis.start)[1] - y0, 0, height - 1))

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

    x_detected, x_raw_score = _detect_tick_candidates_x(
        dark_mask=dark_mask,
        axis_row=axis_row,
        axis_col=axis_col,
    )
    y_detected, y_raw_score = _detect_tick_candidates_y(
        dark_mask=dark_mask,
        axis_col=axis_col,
        axis_row=axis_row,
    )
    warnings.append(f"I_TICK_DETECT_RAW_X:{len(x_detected)}:{x_raw_score:.3f}")
    warnings.append(f"I_TICK_DETECT_RAW_Y:{len(y_detected)}:{y_raw_score:.3f}")

    x_assigned, x_match_ratio, x_shift = _assign_ticks_to_values(
        expected_positions=x_expected,
        detected_positions=x_detected,
        axis_len=width,
    )
    y_assigned, y_match_ratio, y_shift = _assign_ticks_to_values(
        expected_positions=y_expected,
        detected_positions=y_detected,
        axis_len=height,
    )

    x_anchors = tuple((int(x0 + px), float(val)) for px, val in zip(x_assigned, x_values))
    y_anchors = tuple((int(y0 + py), float(val)) for py, val in zip(y_assigned, y_values))

    if x_values:
        warnings.append(
            f"I_TICK_MATCH_X:{len(x_values)}:{x_match_ratio:.2f}:{x_shift:.1f}:{len(x_detected)}"
        )
    if y_values:
        warnings.append(
            f"I_TICK_MATCH_Y:{len(y_values)}:{y_match_ratio:.2f}:{y_shift:.1f}:{len(y_detected)}"
        )
    if x_match_ratio < 0.5:
        warnings.append("W_TICK_MATCH_X_WEAK")
    if y_match_ratio < 0.5:
        warnings.append("W_TICK_MATCH_Y_WEAK")

    x_shift_den = max(1.0, float(width) * TICK_MAX_SHIFT_RATIO)
    y_shift_den = max(1.0, float(height) * TICK_MAX_SHIFT_RATIO)
    x_quality = float(np.clip(0.6 * x_match_ratio + 0.4 * (1.0 - min(1.0, x_shift / x_shift_den)), 0.0, 1.0))
    y_quality = float(np.clip(0.6 * y_match_ratio + 0.4 * (1.0 - min(1.0, y_shift / y_shift_den)), 0.0, 1.0))
    confidence = float(np.clip(0.5 * (x_quality + y_quality), 0.0, 1.0))
    return x_anchors, y_anchors, confidence, warnings


def _smooth_1d(values: NDArray[np.float32]) -> NDArray[np.float32]:
    if values.size < 5:
        return values
    kernel = np.asarray([1.0, 2.0, 3.0, 2.0, 1.0], dtype=np.float32)
    kernel /= float(np.sum(kernel))
    padded = np.pad(values, (2, 2), mode="edge")
    smooth = np.convolve(padded, kernel, mode="valid")
    return smooth.astype(np.float32, copy=False)


def _find_peaks_1d(
    signal: NDArray[np.float32],
    threshold: float,
    min_distance: int,
) -> list[int]:
    n = int(signal.size)
    if n < 3:
        return []
    candidates: list[int] = []
    for i in range(1, n - 1):
        v = float(signal[i])
        if v < threshold:
            continue
        if v >= float(signal[i - 1]) and v >= float(signal[i + 1]):
            candidates.append(i)
    if not candidates:
        return []

    candidates.sort(key=lambda idx: float(signal[idx]), reverse=True)
    selected: list[int] = []
    for idx in candidates:
        if all(abs(int(idx) - int(j)) >= min_distance for j in selected):
            selected.append(int(idx))
    selected.sort()
    return selected


def _detect_tick_positions_1d(
    axis_len: int,
    expected_count: int,
    scorer: Callable[[int], float],
) -> tuple[list[int], float]:
    if axis_len <= 0:
        return [], 0.0
    raw = np.asarray([float(scorer(pos)) for pos in range(axis_len)], dtype=np.float32)
    smoothed = _smooth_1d(raw)
    q = float(np.percentile(smoothed, TICK_PEAK_QUANTILE)) if smoothed.size else 0.0
    threshold = max(TICK_SCORE_MIN, q * TICK_PEAK_REL_THRESHOLD)
    min_dist = max(2, int(round(float(axis_len) * TICK_PEAK_MIN_DIST_RATIO)))
    peaks = _find_peaks_1d(smoothed, threshold=threshold, min_distance=min_dist)
    if not peaks:
        return [], 0.0

    if expected_count >= 2 and len(peaks) > expected_count:
        peaks = sorted(peaks, key=lambda idx: float(smoothed[idx]), reverse=True)[:expected_count]
        peaks.sort()

    mean_score = float(np.mean(np.asarray([smoothed[idx] for idx in peaks], dtype=np.float32)))
    return peaks, mean_score


def _refine_positions_1d(
    expected_positions: list[int],
    axis_len: int,
    scorer: Callable[[int], float],
) -> tuple[list[int], float, float, float, float]:
    if not expected_positions:
        return [], 0.0, 0.0, 0.0, 0.0
    refined: list[int] = []
    scores: list[float] = []
    shifts: list[float] = []
    gains: list[float] = []
    boundary_hits = 0
    tie_eps = 1e-6
    for idx, expected in enumerate(expected_positions):
        radius = _tick_search_radius(expected_positions, idx, axis_len)
        lo = max(0, int(expected) - radius)
        hi = min(axis_len - 1, int(expected) + radius)
        expected_pos = int(np.clip(int(expected), lo, hi))
        expected_score = float(scorer(expected_pos))
        best_pos = int(expected_pos)
        best_score = float(expected_score)
        lo = max(0, int(expected) - radius)
        hi = min(axis_len - 1, int(expected) + radius)
        for pos in range(lo, hi + 1):
            score = float(scorer(pos))
            if score > best_score + tie_eps:
                best_score = score
                best_pos = pos
            elif abs(score - best_score) <= tie_eps:
                # Deterministic tie-break: keep candidate closest to expected.
                if abs(int(pos) - int(expected_pos)) < abs(int(best_pos) - int(expected_pos)):
                    best_pos = int(pos)
        if best_score < TICK_SCORE_MIN:
            best_pos = int(expected_pos)
            best_score = float(expected_score)
        elif (best_score - expected_score) < TICK_SCORE_DELTA_MIN:
            # Avoid drifting to weak local maxima when improvement is negligible.
            best_pos = int(expected_pos)
            best_score = float(expected_score)
        refined.append(int(best_pos))
        scores.append(float(best_score))
        shifts.append(float(abs(best_pos - int(expected_pos))))
        gains.append(float(best_score - expected_score))
        if radius > 0 and (best_pos <= lo or best_pos >= hi):
            boundary_hits += 1
    mean_score = float(np.mean(np.asarray(scores, dtype=np.float32))) if scores else 0.0
    med_shift = float(np.median(np.asarray(shifts, dtype=np.float32))) if shifts else 0.0
    edge_hit_frac = float(boundary_hits) / float(max(1, len(refined)))
    mean_gain = float(np.mean(np.asarray(gains, dtype=np.float32))) if gains else 0.0
    return refined, mean_score, med_shift, edge_hit_frac, mean_gain


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

    x_refined, x_score, x_shift, x_edge_frac, x_gain = _refine_positions_1d(
        expected_positions=x_expected,
        axis_len=width,
        scorer=lambda pos: _score_x_tick(dark_mask, pos, y_axis_row=y_axis_row, tick_extent=tick_extent_y),
    )
    y_refined, y_score, y_shift, y_edge_frac, y_gain = _refine_positions_1d(
        expected_positions=y_expected,
        axis_len=height,
        scorer=lambda pos: _score_y_tick(dark_mask, pos, x_axis_col=x_axis_col, tick_extent=tick_extent_x),
    )

    x_detected, x_detect_score = _detect_tick_positions_1d(
        axis_len=width,
        expected_count=len(x_expected),
        scorer=lambda pos: _score_x_tick(dark_mask, pos, y_axis_row=y_axis_row, tick_extent=tick_extent_y),
    )
    y_detected, y_detect_score = _detect_tick_positions_1d(
        axis_len=height,
        expected_count=len(y_expected),
        scorer=lambda pos: _score_y_tick(dark_mask, pos, x_axis_col=x_axis_col, tick_extent=tick_extent_x),
    )

    if len(x_detected) == len(x_expected) and len(x_expected) >= 2:
        detect_shift = float(
            np.median(
                np.abs(
                    np.asarray(x_detected, dtype=np.float32) - np.asarray(x_expected, dtype=np.float32)
                )
            )
        )
        warnings.append(f"I_TICK_DETECT_X:{len(x_detected)}:{x_detect_score:.3f}:{detect_shift:.1f}")
        if x_detect_score >= x_score + 0.02 and detect_shift <= float(width) * 0.12:
            x_refined = [int(v) for v in x_detected]
            x_score = float(x_detect_score)
            warnings.append("I_TICK_CAL_X_USED_PEAKS")

    if len(y_detected) == len(y_expected) and len(y_expected) >= 2:
        detect_shift = float(
            np.median(
                np.abs(
                    np.asarray(y_detected, dtype=np.float32) - np.asarray(y_expected, dtype=np.float32)
                )
            )
        )
        warnings.append(f"I_TICK_DETECT_Y:{len(y_detected)}:{y_detect_score:.3f}:{detect_shift:.1f}")
        if y_detect_score >= y_score + 0.02 and detect_shift <= float(height) * 0.12:
            y_refined = [int(v) for v in y_detected]
            y_score = float(y_detect_score)
            warnings.append("I_TICK_CAL_Y_USED_PEAKS")

    x_refined_anchors = tuple((int(x0 + pos), float(val)) for pos, val in zip(x_refined, x_values))
    y_refined_anchors = tuple((int(y0 + pos), float(val)) for pos, val in zip(y_refined, y_values))
    x_expected_anchors = tuple((int(x0 + pos), float(val)) for pos, val in zip(x_expected, x_values))
    y_expected_anchors = tuple((int(y0 + pos), float(val)) for pos, val in zip(y_expected, y_values))

    x_reject = (
        x_edge_frac >= TICK_EDGE_HIT_REJECT_FRAC
        and x_gain <= TICK_GAIN_REJECT_MAX
        and x_shift >= float(width) * TICK_SHIFT_REJECT_RATIO
        and x_score <= TICK_SCORE_REJECT_MAX
    )
    y_reject = (
        y_edge_frac >= TICK_EDGE_HIT_REJECT_FRAC
        and y_gain <= TICK_GAIN_REJECT_MAX
        and y_shift >= float(height) * TICK_SHIFT_REJECT_RATIO
        and y_score <= TICK_SCORE_REJECT_MAX
    )

    x_anchors = x_expected_anchors if x_reject else x_refined_anchors
    y_anchors = y_expected_anchors if y_reject else y_refined_anchors

    x_quality = 0.0
    y_quality = 0.0
    if x_values:
        x_shift_den = max(1.0, float(width) * TICK_MAX_SHIFT_RATIO)
        x_quality = float(np.clip((x_score / 0.25) * (1.0 - min(1.0, x_shift / x_shift_den)), 0.0, 1.0))
        warnings.append(f"I_TICK_CAL_X:{len(x_values)}:{x_score:.3f}:{x_shift:.1f}:{x_edge_frac:.2f}:{x_gain:.3f}")
        if x_reject:
            x_quality = 0.0
            warnings.append("W_TICK_CAL_X_REJECTED_UNSTABLE")
    else:
        warnings.append("W_TICK_CAL_X_NO_TICKS")
    if y_values:
        y_shift_den = max(1.0, float(height) * TICK_MAX_SHIFT_RATIO)
        y_quality = float(np.clip((y_score / 0.25) * (1.0 - min(1.0, y_shift / y_shift_den)), 0.0, 1.0))
        warnings.append(f"I_TICK_CAL_Y:{len(y_values)}:{y_score:.3f}:{y_shift:.1f}:{y_edge_frac:.2f}:{y_gain:.3f}")
        if y_reject:
            y_quality = 0.0
            warnings.append("W_TICK_CAL_Y_REJECTED_UNSTABLE")
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
    rx_ind, ry_ind, conf_ind, ind_warnings = _independent_tick_anchors(image, mapping)
    rx_ref, ry_ref, conf_ref, ref_warnings = _refine_tick_anchors(image, mapping)
    tick_warnings.extend(ind_warnings)
    tick_warnings.extend(ref_warnings)

    use_refine = (
        conf_ref > conf_ind + 0.08
        and len(rx_ref) >= 2
        and len(ry_ref) >= 2
    )
    if use_refine:
        x_tick_anchors = rx_ref
        y_tick_anchors = ry_ref
        tick_cal_conf = float(conf_ref)
        tick_warnings.append("I_TICK_CAL_SOURCE:refine_fallback")
    else:
        if len(rx_ind) >= 2:
            x_tick_anchors = rx_ind
        elif len(rx_ref) >= 2:
            x_tick_anchors = rx_ref
            tick_warnings.append("W_TICK_CAL_X_FALLBACK_REFINE")

        if len(ry_ind) >= 2:
            y_tick_anchors = ry_ind
        elif len(ry_ref) >= 2:
            y_tick_anchors = ry_ref
            tick_warnings.append("W_TICK_CAL_Y_FALLBACK_REFINE")

        tick_cal_conf = float(
            max(
                conf_ind,
                conf_ref if (len(x_tick_anchors) >= 2 and len(y_tick_anchors) >= 2) else 0.0,
            )
        )
        tick_warnings.append("I_TICK_CAL_SOURCE:independent_detect")

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
