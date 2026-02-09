"""Shared evidence cube and per-arm score maps for digitization_v5."""

from __future__ import annotations

from dataclasses import dataclass
import os

import cv2
import numpy as np
from numpy.typing import NDArray

from .axis_map import PlotModel
from .legend_color import ArmColorModel

RIDGE_WEIGHT = 0.30
EDGE_WEIGHT = 0.20
COLOR_WEIGHT = 0.52
AXIS_PENALTY_WEIGHT = 0.85
AXIS_PENALTY_WEIGHT_UPWARD = 0.45
AXIS_PENALTY_WEIGHT_UNKNOWN = 0.65
TEXT_PENALTY_WEIGHT = 0.58
TEXT_REGION_PENALTY_WEIGHT = 0.42
LINE_PENALTY_WEIGHT = 0.68
FRAME_PENALTY_WEIGHT = 0.70
HORIZONTAL_SUPPORT_WEIGHT = 0.42
HORIZONTAL_SUPPORT_MIN = 0.14
HORIZONTAL_SUPPORT_KERNEL_RATIO = 0.012
FRAME_LINE_DENSITY_MIN = 0.11
FRAME_BAND_RATIO = 0.012
FRAME_TOP_X_EXEMPT_RATIO = 0.06
COLOR_GOOD_DISTANCE = 25.0
COLOR_ANTI_WEIGHT = 2.90
COLOR_STRICT_RELIABILITY = 0.30
COLOR_STRICT_MIN_LIKELIHOOD = 0.75
COLOR_STRICT_QUANTILE = 90.0
COLOR_STRICT_OFF_PENALTY = 1.55
COLOR_STRICT_MIN_DENSITY = 0.0002
COLOR_STRICT_MIN_COLUMN_COVERAGE = 0.10
COLOR_HARD_LOCK_RELIABILITY = 0.45
COLOR_HARD_LOCK_MIN_LIKELIHOOD = 0.65
COLOR_HARD_LOCK_MIN_COLUMN_COVERAGE = 0.08
COLOR_RMSE_RELAX_BASE = 12.0
COLOR_RMSE_RELAX_RANGE = 24.0
COLOR_RMSE_RELAX_MAX = 0.20
GRAY_DYNAMIC_MIN_DIFF = 5.0
GRAY_DYNAMIC_MAX_DIFF = 12.0
GRAY_DYNAMIC_MAX_DIFF_LOW_RELIABILITY = 14.0
GRAY_DYNAMIC_SEED_MIN = 18
GRAY_DYNAMIC_SEED_QUANTILE = 90.0
GRAY_DYNAMIC_SEED_MIN_SCORE = 0.72
GRAY_DYNAMIC_MIN_DENSITY = 0.0002
APPLY_DYNAMIC_GRAY_GATE_DEFAULT = False
HSV_SAT_MIN = 56.0
HSV_SAT_MIN_LOW_RELIABILITY = 46.0
HSV_STRICT_MIN_DENSITY = 0.00025
HSV_STRICT_MIN_COLUMN_COVERAGE = 0.08
HSV_HUE_MIN_THR = 8.0
HSV_HUE_MAX_THR = 16.0
HSV_HUE_MAX_THR_LOW_RELIABILITY = 22.0
HSV_HUE_SEED_MIN = 18
HSV_HUE_SEED_QUANTILE = 88.0
HSV_HUE_SEED_MIN_SCORE = 0.62
CANDIDATE_AXIS_THRESH = 0.25
CANDIDATE_AXIS_THRESH_UPWARD = 0.55
CANDIDATE_AXIS_THRESH_UNKNOWN = 0.40
CANDIDATE_TEXT_THRESH = 0.35
CANDIDATE_TEXT_REGION_THRESH = 0.42
CANDIDATE_LINE_THRESH = 0.55
CANDIDATE_RIDGE_THRESH = 0.24
COMPONENT_MIN_AREA_RATIO = 0.00003
COMPONENT_START_STRIP_RATIO = 0.14
COMPONENT_START_BAND_RATIO = 0.22
COMPONENT_XSPAN_KEEP_RATIO = 0.10
COMPONENT_XMIN_KEEP_RATIO = 0.40
PRIMARY_COMPONENT_MIN_XSPAN_RATIO = 0.05
HSL_KMEDOIDS_MAX_POINTS = 2600
HSL_KMEDOIDS_MAX_ITERS = 6
HSL_KMEDOIDS_KNN_K = 12
HSL_BG_LIGHTNESS_MAX = 0.96
HSL_BG_SATURATION_MIN = 0.05
HSL_CLUSTER_WEIGHT = np.asarray([1.0, 1.6, 1.6], dtype=np.float32)
CONSENSUS_BLEND_WEIGHT = 0.30
LINE_HOUGH_MIN_RATIO = 0.70
LINE_HOUGH_ANGLE_DEG = 8.0
HARDPOINT_WINDOW_BASE_RATIO = 0.020
HARDPOINT_WINDOW_EXTRA_RATIO = 0.030
HARDPOINT_WINDOW_DIST_RATIO = 0.10
HARDPOINT_WINDOW_MIN_DENSITY = 0.00015
HARDPOINT_WINDOW_MIN_COLUMN_COVERAGE = 0.18
HARDPOINT_WINDOW_OFF_PENALTY = 2.40
LOCAL_COLOR_KERNEL = 3
LOCAL_COLOR_BLEND_WEIGHT = 0.80
MIN_CANDIDATE_SATURATION = 18.0
ARM_EXCLUSIVE_MIN_MARGIN = 0.06
ARM_EXCLUSIVE_MIN_DENSITY = 0.00012
ARM_EXCLUSIVE_MIN_COLUMN_COVERAGE = 0.08


def _normalize01(arr: NDArray[np.float32]) -> NDArray[np.float32]:
    lo = float(np.min(arr))
    hi = float(np.max(arr))
    if hi <= lo + 1e-9:
        return np.zeros_like(arr, dtype=np.float32)
    return (arr - lo) / (hi - lo)


def _local_mean_lab(roi_lab: NDArray[np.float32], k: int = LOCAL_COLOR_KERNEL) -> NDArray[np.float32]:
    """Compute local-average Lab values for robust anti-aliased color matching."""
    kk = max(1, int(k))
    if kk % 2 == 0:
        kk += 1
    return cv2.blur(roi_lab, (kk, kk), borderType=cv2.BORDER_REPLICATE).astype(np.float32)


def _mask_column_coverage(mask: NDArray[np.bool_]) -> float:
    if mask.size <= 0:
        return 0.0
    return float(np.mean(np.any(mask, axis=0).astype(np.float32)))


def _hardpoint_corridor_mask(
    shape: tuple[int, int],
    guides: tuple[tuple[int, int], ...] | None,
) -> NDArray[np.bool_] | None:
    """Build a per-column hardpoint corridor mask from anchor guides."""
    if not guides or len(guides) < 2:
        return None
    h, w = shape
    by_col: dict[int, list[int]] = {}
    for x, y in guides:
        cx = int(np.clip(int(x), 0, max(0, w - 1)))
        cy = int(np.clip(int(y), 0, max(0, h - 1)))
        by_col.setdefault(cx, []).append(cy)
    if len(by_col) < 2:
        return None

    cols = np.asarray(sorted(by_col), dtype=np.int32)
    vals = np.asarray(
        [float(np.median(np.asarray(by_col[c], dtype=np.float32))) for c in cols.tolist()],
        dtype=np.float32,
    )
    xs = np.arange(w, dtype=np.float32)
    target = np.interp(xs, cols.astype(np.float32), vals, left=float(vals[0]), right=float(vals[-1]))
    nearest = np.min(np.abs(xs[:, None] - cols[None, :].astype(np.float32)), axis=1)

    base = max(2.0, float(h) * HARDPOINT_WINDOW_BASE_RATIO)
    extra = max(2.0, float(h) * HARDPOINT_WINDOW_EXTRA_RATIO)
    dist_scale = max(3.0, float(w) * HARDPOINT_WINDOW_DIST_RATIO)
    grow = np.clip(nearest / dist_scale, 0.0, 1.0)
    band = base + (grow * extra)

    ys = np.arange(h, dtype=np.float32)[:, None]
    in_band = np.abs(ys - target[None, :]) <= band[None, :]
    span = (xs >= float(cols[0])) & (xs <= float(cols[-1]))
    # Outside hardpoint span, do not hard-lock the arm mask.
    mask = np.where(span[None, :], in_band, True)
    return mask.astype(np.bool_)


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _ridge_response(gray: NDArray[np.uint8]) -> NDArray[np.float32]:
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    lap = cv2.Laplacian(blur, cv2.CV_32F, ksize=3)
    ridge = np.abs(lap).astype(np.float32)
    return _normalize01(ridge)


def _edge_response(gray: NDArray[np.uint8]) -> NDArray[np.float32]:
    edges = cv2.Canny(gray, 35, 110).astype(np.float32) / 255.0
    edges = cv2.GaussianBlur(edges, (3, 3), 0)
    return _normalize01(edges.astype(np.float32))


def _frame_penalty(gray: NDArray[np.uint8]) -> tuple[NDArray[np.float32], bool, bool]:
    """Detect and penalize top/right full-box frame lines."""
    h, w = gray.shape
    if h <= 0 or w <= 0:
        return np.zeros_like(gray, dtype=np.float32), False, False

    band = max(1, int(round(min(h, w) * FRAME_BAND_RATIO)))
    ink = (gray < 130).astype(np.uint8)
    top_density = float(np.mean(ink[:band, :])) if band < h else 0.0
    right_density = float(np.mean(ink[:, max(0, w - band):])) if band < w else 0.0
    has_top = top_density >= FRAME_LINE_DENSITY_MIN
    has_right = right_density >= FRAME_LINE_DENSITY_MIN

    pen = np.zeros((h, w), dtype=np.uint8)
    if has_top:
        x_exempt = max(0, int(round(w * FRAME_TOP_X_EXEMPT_RATIO)))
        pen[:band, x_exempt:] = 255
    if has_right:
        pen[:, max(0, w - band):] = 255

    pen = cv2.GaussianBlur((pen.astype(np.float32) / 255.0).astype(np.float32), (5, 5), 0)
    pen = _normalize01(pen.astype(np.float32))
    return pen, has_top, has_right


def _horizontal_support(mask: NDArray[np.bool_]) -> NDArray[np.float32]:
    """Score horizontal continuity to suppress vertical/text-like structures."""
    h, w = mask.shape
    if h <= 0 or w <= 0:
        return np.zeros_like(mask, dtype=np.float32)
    k = max(5, int(round(w * HORIZONTAL_SUPPORT_KERNEL_RATIO)))
    if k % 2 == 0:
        k += 1
    kernel = np.ones((1, k), dtype=np.float32)
    src = mask.astype(np.float32)
    conv = cv2.filter2D(src, ddepth=-1, kernel=kernel, borderType=cv2.BORDER_REPLICATE)
    support = conv / float(k)
    return np.clip(support, 0.0, 1.0).astype(np.float32)


def _prune_curve_components(
    mask: NDArray[np.bool_],
    direction: str,
) -> tuple[NDArray[np.bool_], int, int]:
    """
    Keep only connected components that look like plausible curve fragments.

    Heuristics:
    - survives if connected to start region (left strip + expected start-y band)
    - or has broad x-span and starts near the left side
    """
    h, w = mask.shape
    if h <= 0 or w <= 0:
        return mask, 0, 0

    src = (mask.astype(np.uint8) * 255)
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(src, connectivity=8)
    if n_labels <= 1:
        return mask, 0, 0

    min_area = max(4, int(round(h * w * COMPONENT_MIN_AREA_RATIO)))
    x_start_max = max(1, int(round(w * COMPONENT_START_STRIP_RATIO)))
    y_band = max(2, int(round(h * COMPONENT_START_BAND_RATIO)))
    keep = np.zeros((h, w), dtype=np.uint8)
    kept = 0
    dropped = 0

    for idx in range(1, n_labels):
        x = int(stats[idx, cv2.CC_STAT_LEFT])
        y = int(stats[idx, cv2.CC_STAT_TOP])
        cw = int(stats[idx, cv2.CC_STAT_WIDTH])
        ch = int(stats[idx, cv2.CC_STAT_HEIGHT])
        area = int(stats[idx, cv2.CC_STAT_AREA])
        if area < min_area or cw <= 0 or ch <= 0:
            dropped += 1
            continue

        x_min = x
        x_max = x + cw - 1
        y_min = y
        y_max = y + ch - 1
        x_span_ratio = float(cw) / float(max(1, w))
        starts_left = x_min <= x_start_max

        if direction == "upward":
            in_start_band = y_max >= (h - y_band)
        else:
            in_start_band = y_min <= y_band

        keep_comp = (starts_left and in_start_band) or (
            x_span_ratio >= COMPONENT_XSPAN_KEEP_RATIO
            and x_min <= int(round(w * COMPONENT_XMIN_KEEP_RATIO))
        )

        if keep_comp:
            keep[labels == idx] = 255
            kept += 1
        else:
            dropped += 1

    if kept == 0:
        return mask, 0, dropped
    return keep.astype(bool), kept, dropped


def _select_primary_component(
    mask: NDArray[np.bool_],
    direction: str,
) -> tuple[NDArray[np.bool_], bool, float]:
    """
    Keep a single most plausible component to avoid arm hopping.
    Returns: selected_mask, used_primary_selector, x_span_ratio_of_selected
    """
    h, w = mask.shape
    if h <= 0 or w <= 0:
        return mask, False, 0.0

    src = (mask.astype(np.uint8) * 255)
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(src, connectivity=8)
    if n_labels <= 1:
        return mask, False, 0.0

    x_start_max = max(1, int(round(w * COMPONENT_START_STRIP_RATIO)))
    y_band = max(2, int(round(h * COMPONENT_START_BAND_RATIO)))
    best_idx = -1
    best_score = -1e9
    best_xspan = 0.0
    for idx in range(1, n_labels):
        x = int(stats[idx, cv2.CC_STAT_LEFT])
        y = int(stats[idx, cv2.CC_STAT_TOP])
        cw = int(stats[idx, cv2.CC_STAT_WIDTH])
        ch = int(stats[idx, cv2.CC_STAT_HEIGHT])
        area = int(stats[idx, cv2.CC_STAT_AREA])
        if area <= 0 or cw <= 0 or ch <= 0:
            continue
        x_span = float(cw) / float(max(1, w))
        if x_span < PRIMARY_COMPONENT_MIN_XSPAN_RATIO:
            continue
        y_min = y
        y_max = y + ch - 1
        starts_left = x <= x_start_max
        if direction == "upward":
            start_band = y_max >= (h - y_band)
        else:
            start_band = y_min <= y_band
        area_ratio = float(area) / float(max(1, h * w))
        score = (
            2.2 * x_span
            + 0.8 * min(1.0, area_ratio * 180.0)
            + (0.7 if starts_left else 0.0)
            + (0.7 if start_band else 0.0)
        )
        if score > best_score:
            best_score = score
            best_idx = idx
            best_xspan = x_span

    if best_idx < 0:
        return mask, False, 0.0

    out = np.zeros_like(src, dtype=np.uint8)
    out[labels == best_idx] = 255
    return out.astype(bool), True, float(best_xspan)


def _text_penalty(gray: NDArray[np.uint8]) -> NDArray[np.float32]:
    """Approximate text-like regions as a soft penalty map."""
    adaptive = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        25,
        6,
    )
    _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    k_w = max(7, int(round(gray.shape[1] * 0.018)))
    k_h = max(5, int(round(gray.shape[0] * 0.014)))
    blackhat_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k_w, k_h))
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, blackhat_kernel)
    _, bh_bin = cv2.threshold(blackhat, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    merged = cv2.bitwise_or(adaptive, otsu)
    merged = cv2.bitwise_or(merged, bh_bin)

    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(merged, connectivity=8)
    out = np.zeros_like(gray, dtype=np.uint8)
    h, w = gray.shape
    area_floor = max(8, int((h * w) * 0.00002))
    area_ceil = max(area_floor + 1, int((h * w) * 0.004))
    for idx in range(1, n_labels):
        area = int(stats[idx, cv2.CC_STAT_AREA])
        if area < area_floor or area > area_ceil:
            continue
        bw = int(stats[idx, cv2.CC_STAT_WIDTH])
        bh = int(stats[idx, cv2.CC_STAT_HEIGHT])
        if bw <= 0 or bh <= 0:
            continue
        aspect = float(max(bw, bh)) / float(max(1, min(bw, bh)))
        fill = float(area) / float(max(1, bw * bh))
        # Text glyph-ish blobs: compact-ish and not large bars.
        if aspect > 10.0:
            continue
        if fill < 0.12 or fill > 0.92:
            continue
        out[labels == idx] = 255
    out = cv2.dilate(out, np.ones((2, 2), dtype=np.uint8), iterations=1)
    out = cv2.GaussianBlur((out.astype(np.float32) / 255.0).astype(np.float32), (5, 5), 0)
    return _normalize01(out.astype(np.float32))


def _text_region_penalty(text_penalty: NDArray[np.float32]) -> NDArray[np.float32]:
    """
    Expand local text detections into UI regions (legend/annotation blocks)
    so tracing does not jump to labels and boxed callouts.
    """
    h, w = text_penalty.shape
    if h <= 0 or w <= 0:
        return np.zeros_like(text_penalty, dtype=np.float32)

    bin_mask = (text_penalty > 0.25).astype(np.uint8) * 255
    if not np.any(bin_mask):
        return np.zeros_like(text_penalty, dtype=np.float32)

    k_w = max(9, int(round(w * 0.030)))
    k_h = max(7, int(round(h * 0.025)))
    dil = cv2.dilate(bin_mask, cv2.getStructuringElement(cv2.MORPH_RECT, (k_w, k_h)), iterations=1)

    out = np.zeros_like(bin_mask, dtype=np.uint8)
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(dil, connectivity=8)
    area_floor = max(30, int((h * w) * 0.00025))
    area_ceil = max(area_floor + 1, int((h * w) * 0.09))
    pad_x = max(2, int(round(w * 0.012)))
    pad_y = max(2, int(round(h * 0.012)))
    for idx in range(1, n_labels):
        area = int(stats[idx, cv2.CC_STAT_AREA])
        if area < area_floor or area > area_ceil:
            continue
        x = int(stats[idx, cv2.CC_STAT_LEFT])
        y = int(stats[idx, cv2.CC_STAT_TOP])
        bw = int(stats[idx, cv2.CC_STAT_WIDTH])
        bh = int(stats[idx, cv2.CC_STAT_HEIGHT])
        if bw <= 0 or bh <= 0:
            continue
        x0 = max(0, x - pad_x)
        y0 = max(0, y - pad_y)
        x1 = min(w - 1, x + bw - 1 + pad_x)
        y1 = min(h - 1, y + bh - 1 + pad_y)
        cv2.rectangle(out, (x0, y0), (x1, y1), color=255, thickness=-1)

    out = cv2.GaussianBlur((out.astype(np.float32) / 255.0).astype(np.float32), (5, 5), 0)
    return _normalize01(out.astype(np.float32))


def _straight_line_penalty(
    gray: NDArray[np.uint8],
    axis_mask: NDArray[np.uint8],
) -> tuple[NDArray[np.float32], int]:
    """
    Penalize long straight horizontal/vertical lines (grid/frame/boxed UI)
    without suppressing short KM step segments.
    """
    h, w = gray.shape
    if h <= 0 or w <= 0:
        return np.zeros_like(gray, dtype=np.float32), 0

    edges = cv2.Canny(gray, 40, 130)
    min_len = max(10, int(round(min(h, w) * 0.42)))
    raw_lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180.0,
        threshold=max(18, int(round(0.07 * max(h, w)))),
        minLineLength=min_len,
        maxLineGap=max(3, int(round(min(h, w) * 0.018))),
    )

    horiz_min = float(max(12, int(round(w * LINE_HOUGH_MIN_RATIO))))
    vert_min = float(max(12, int(round(h * LINE_HOUGH_MIN_RATIO))))
    out = np.zeros_like(gray, dtype=np.uint8)
    kept = 0
    if raw_lines is not None:
        for line in raw_lines:
            x1, y1, x2, y2 = [int(v) for v in line[0]]
            dx = float(x2 - x1)
            dy = float(y2 - y1)
            length = float(np.hypot(dx, dy))
            if length <= 1.0:
                continue
            angle = abs(float(np.degrees(np.arctan2(dy, dx))))
            is_horizontal = angle <= LINE_HOUGH_ANGLE_DEG or angle >= (180.0 - LINE_HOUGH_ANGLE_DEG)
            is_vertical = abs(angle - 90.0) <= LINE_HOUGH_ANGLE_DEG
            if is_horizontal and length >= horiz_min:
                cv2.line(out, (x1, y1), (x2, y2), color=255, thickness=2, lineType=cv2.LINE_AA)
                kept += 1
            elif is_vertical and length >= vert_min:
                cv2.line(out, (x1, y1), (x2, y2), color=255, thickness=2, lineType=cv2.LINE_AA)
                kept += 1

    if np.any(axis_mask):
        axis_dil = cv2.dilate(axis_mask, np.ones((5, 5), dtype=np.uint8), iterations=1)
        out[axis_dil > 0] = 0

    out_f = cv2.GaussianBlur((out.astype(np.float32) / 255.0).astype(np.float32), (5, 5), 0)
    out_f = _normalize01(out_f.astype(np.float32))
    return out_f, int(kept)


def _color_likelihood(
    roi_lab: NDArray[np.float32],
    reference_lab: tuple[float, float, float] | None,
    reliability: float,
    good_dist: float | None = None,
) -> NDArray[np.float32]:
    if reference_lab is None or reliability <= 0.0:
        return np.zeros(roi_lab.shape[:2], dtype=np.float32)

    gd = good_dist if good_dist is not None else COLOR_GOOD_DISTANCE
    ref = np.asarray(reference_lab, dtype=np.float32)
    dist = np.linalg.norm(roi_lab - ref[None, None, :], axis=2).astype(np.float32)
    # Saturating positive-only color contribution.
    likelihood = np.clip((gd - dist) / gd, 0.0, 1.0)
    return likelihood.astype(np.float32)


def _reference_gray_from_lab(reference_lab: tuple[float, float, float] | None) -> float | None:
    if reference_lab is None:
        return None
    lab = np.asarray(
        [[[
            int(np.clip(round(reference_lab[0]), 0, 255)),
            int(np.clip(round(reference_lab[1]), 0, 255)),
            int(np.clip(round(reference_lab[2]), 0, 255)),
        ]]],
        dtype=np.uint8,
    )
    bgr = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    return float(gray[0, 0])


def _reference_hsv_from_lab(
    reference_lab: tuple[float, float, float] | None,
) -> tuple[float, float, float] | None:
    if reference_lab is None:
        return None
    lab = np.asarray(
        [[[
            int(np.clip(round(reference_lab[0]), 0, 255)),
            int(np.clip(round(reference_lab[1]), 0, 255)),
            int(np.clip(round(reference_lab[2]), 0, 255)),
        ]]],
        dtype=np.uint8,
    )
    bgr = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[0, 0].tolist()
    return float(h), float(s), float(v)


def _hue_distance(
    h_channel: NDArray[np.float32],
    center: float,
) -> NDArray[np.float32]:
    diff = np.abs(h_channel - float(center))
    return np.minimum(diff, 180.0 - diff).astype(np.float32)


def _dynamic_hue_gate(
    hue_channel: NDArray[np.float32],
    sat_channel: NDArray[np.float32],
    candidate_mask: NDArray[np.bool_],
    color_mix: NDArray[np.float32],
    ref_hue: float | None,
    reliability: float,
) -> tuple[float | None, float, int, float, float]:
    """
    Compute dynamic hue center/threshold from high-confidence color seeds.

    Returns:
      center_hue, threshold, seed_count, hue_mad, seed_score_threshold
    """
    if ref_hue is None:
        return None, HSV_HUE_MIN_THR, 0, 0.0, HSV_HUE_SEED_MIN_SCORE

    sat_min = HSV_SAT_MIN if reliability >= COLOR_STRICT_RELIABILITY else HSV_SAT_MIN_LOW_RELIABILITY
    seed_cand = candidate_mask & (sat_channel >= sat_min)
    seed_scores = color_mix[seed_cand]
    if seed_scores.size <= 0:
        cap = HSV_HUE_MAX_THR if reliability >= COLOR_STRICT_RELIABILITY else HSV_HUE_MAX_THR_LOW_RELIABILITY
        return float(ref_hue), float(cap), 0, 0.0, HSV_HUE_SEED_MIN_SCORE

    q = float(np.percentile(seed_scores, HSV_HUE_SEED_QUANTILE))
    seed_thr = max(HSV_HUE_SEED_MIN_SCORE, q)
    seed_mask = seed_cand & (color_mix >= seed_thr)
    seed_hue = hue_channel[seed_mask]

    if seed_hue.size < HSV_HUE_SEED_MIN:
        alt_thr = max(0.54, float(np.percentile(seed_scores, 78.0)))
        alt_mask = seed_cand & (color_mix >= alt_thr)
        alt_h = hue_channel[alt_mask]
        if alt_h.size > seed_hue.size:
            seed_hue = alt_h
            seed_thr = alt_thr

    seed_n = int(seed_hue.size)
    if seed_n <= 0:
        cap = HSV_HUE_MAX_THR if reliability >= COLOR_STRICT_RELIABILITY else HSV_HUE_MAX_THR_LOW_RELIABILITY
        return float(ref_hue), float(cap), seed_n, 0.0, seed_thr

    deltas = ((seed_hue - float(ref_hue) + 90.0) % 180.0) - 90.0
    center_delta = float(np.median(deltas))
    center = (float(ref_hue) + center_delta) % 180.0
    mad = float(np.median(np.abs(deltas - center_delta)))
    cap = HSV_HUE_MAX_THR if reliability >= COLOR_STRICT_RELIABILITY else HSV_HUE_MAX_THR_LOW_RELIABILITY
    thr = float(np.clip((1.9 * mad) + 3.0, HSV_HUE_MIN_THR, cap))
    return float(center), thr, seed_n, mad, seed_thr


def _dynamic_gray_gate(
    gray_f: NDArray[np.float32],
    candidate_mask: NDArray[np.bool_],
    color_mix: NDArray[np.float32],
    ref_gray: float | None,
    reliability: float,
) -> tuple[float | None, float, int, float, float]:
    """
    Compute dynamic grayscale center/threshold from high-confidence arm seeds.

    Returns:
      center_gray, threshold, seed_count, seed_mad, seed_threshold
    """
    if gray_f.shape != candidate_mask.shape or gray_f.shape != color_mix.shape:
        return ref_gray, GRAY_DYNAMIC_MIN_DIFF, 0, 0.0, GRAY_DYNAMIC_SEED_MIN_SCORE

    cand_vals = color_mix[candidate_mask]
    if cand_vals.size <= 0:
        return ref_gray, GRAY_DYNAMIC_MIN_DIFF, 0, 0.0, GRAY_DYNAMIC_SEED_MIN_SCORE

    seed_q = float(np.percentile(cand_vals, GRAY_DYNAMIC_SEED_QUANTILE))
    seed_thr = max(GRAY_DYNAMIC_SEED_MIN_SCORE, seed_q)
    seed_mask = candidate_mask & (color_mix >= seed_thr)
    seed_gray = gray_f[seed_mask]

    if seed_gray.size < GRAY_DYNAMIC_SEED_MIN:
        seed_thr2 = max(0.62, float(np.percentile(cand_vals, 82.0)))
        seed_mask2 = candidate_mask & (color_mix >= seed_thr2)
        seed_gray2 = gray_f[seed_mask2]
        if seed_gray2.size > seed_gray.size:
            seed_mask = seed_mask2
            seed_gray = seed_gray2
            seed_thr = seed_thr2

    seed_count = int(seed_gray.size)
    if seed_count < GRAY_DYNAMIC_SEED_MIN:
        return ref_gray, GRAY_DYNAMIC_MIN_DIFF, seed_count, 0.0, seed_thr

    center = float(np.median(seed_gray))
    mad = float(np.median(np.abs(seed_gray - center)))
    cap = GRAY_DYNAMIC_MAX_DIFF if reliability >= COLOR_STRICT_RELIABILITY else GRAY_DYNAMIC_MAX_DIFF_LOW_RELIABILITY
    thr = float(np.clip((2.5 * mad) + 2.0, GRAY_DYNAMIC_MIN_DIFF, cap))
    return center, thr, seed_count, mad, seed_thr


def _pairwise_sqdist(a: NDArray[np.float32], b: NDArray[np.float32]) -> NDArray[np.float32]:
    aa = np.sum(a * a, axis=1, keepdims=True)
    bb = np.sum(b * b, axis=1, keepdims=True).T
    out = aa + bb - (2.0 * np.dot(a, b.T))
    return np.maximum(out, 0.0).astype(np.float32)


def _init_medoids_farthest(
    x: NDArray[np.float32],
    k: int,
) -> NDArray[np.int32]:
    n = int(x.shape[0])
    if n <= 0:
        return np.zeros((0,), dtype=np.int32)
    k = max(1, min(k, n))
    medoids = [0]
    if k == 1:
        return np.asarray(medoids, dtype=np.int32)
    d2 = np.sum((x - x[0:1]) ** 2, axis=1)
    for _ in range(1, k):
        idx = int(np.argmax(d2))
        medoids.append(idx)
        d2 = np.minimum(d2, np.sum((x - x[idx:idx + 1]) ** 2, axis=1))
    return np.asarray(medoids, dtype=np.int32)


def _pam_kmedoids(
    x: NDArray[np.float32],
    k: int,
    max_iters: int,
) -> tuple[NDArray[np.int32], NDArray[np.int32]]:
    n = int(x.shape[0])
    if n <= 0:
        return np.zeros((0,), dtype=np.int32), np.zeros((0,), dtype=np.int32)
    medoids = _init_medoids_farthest(x, k)
    dist = _pairwise_sqdist(x, x)
    labels = np.argmin(dist[:, medoids], axis=1).astype(np.int32)

    for _ in range(max_iters):
        changed = False
        for j in range(len(medoids)):
            members = np.where(labels == j)[0]
            if members.size == 0:
                continue
            sub = dist[np.ix_(members, members)]
            costs = np.sum(sub, axis=1)
            best_local = int(np.argmin(costs))
            new_medoid = int(members[best_local])
            if new_medoid != int(medoids[j]):
                medoids[j] = new_medoid
                changed = True
        new_labels = np.argmin(dist[:, medoids], axis=1).astype(np.int32)
        if not changed and np.array_equal(new_labels, labels):
            break
        labels = new_labels
    return medoids.astype(np.int32), labels


def _build_hsl_partition(
    roi_bgr: NDArray[np.uint8],
    candidate_mask: NDArray[np.bool_],
    axis_penalty: NDArray[np.float32],
    text_penalty: NDArray[np.float32],
    n_clusters: int,
) -> tuple[
    list[NDArray[np.float32]],
    NDArray[np.float32],
    list[tuple[float, float, float]],
    list[str],
]:
    """
    Partition foreground color with HSL + K-medoids and compute kNN consensus.

    Returns:
      cluster_likelihood_maps, consensus_map, cluster_lab_centers, warnings
    """
    warnings: list[str] = []
    h, w = candidate_mask.shape
    if n_clusters <= 0:
        return [], np.zeros((h, w), dtype=np.float32), [], warnings

    roi_hls = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HLS).astype(np.float32)
    roi_lab = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    h_chan = (roi_hls[:, :, 0] / 180.0).astype(np.float32)
    l_chan = (roi_hls[:, :, 1] / 255.0).astype(np.float32)
    s_chan = (roi_hls[:, :, 2] / 255.0).astype(np.float32)

    fg_mask = (
        candidate_mask
        & (axis_penalty < 0.70)
        & (text_penalty < 0.70)
        & ((s_chan >= HSL_BG_SATURATION_MIN) | (l_chan <= HSL_BG_LIGHTNESS_MAX))
    )
    ys, xs = np.where(fg_mask)
    n_fg = int(ys.size)
    if n_fg < max(120, n_clusters * 60):
        warnings.append(f"W_HSL_KMEDOIDS_SPARSE:{n_fg}")
        return [], np.zeros((h, w), dtype=np.float32), [], warnings

    feats = np.stack([h_chan[ys, xs], s_chan[ys, xs], l_chan[ys, xs]], axis=1).astype(np.float32)
    if n_fg > HSL_KMEDOIDS_MAX_POINTS:
        step = int(np.ceil(float(n_fg) / float(HSL_KMEDOIDS_MAX_POINTS)))
        sel = np.arange(0, n_fg, step, dtype=np.int32)
    else:
        sel = np.arange(0, n_fg, dtype=np.int32)

    ys_s = ys[sel]
    xs_s = xs[sel]
    x_s = feats[sel]
    mu = np.mean(x_s, axis=0, keepdims=True)
    sigma = np.std(x_s, axis=0, keepdims=True)
    sigma = np.where(sigma < 1e-4, 1.0, sigma)
    x_norm = ((x_s - mu) / sigma) * HSL_CLUSTER_WEIGHT[None, :]

    k = max(1, min(n_clusters, x_norm.shape[0]))
    medoid_idx, labels_s = _pam_kmedoids(x_norm, k=k, max_iters=HSL_KMEDOIDS_MAX_ITERS)
    if medoid_idx.size == 0:
        warnings.append("W_HSL_KMEDOIDS_FAILED")
        return [], np.zeros((h, w), dtype=np.float32), [], warnings

    medoids = x_norm[medoid_idx]
    medoid_lab = [tuple(float(v) for v in roi_lab[ys_s[i], xs_s[i]]) for i in medoid_idx.tolist()]

    x_all = ((feats - mu) / sigma) * HSL_CLUSTER_WEIGHT[None, :]
    d_all = np.sqrt(_pairwise_sqdist(x_all.astype(np.float32), medoids.astype(np.float32)))
    sigma_d = np.percentile(d_all, 40, axis=0)
    sigma_d = np.clip(sigma_d, 0.08, 0.65).astype(np.float32)

    cluster_like_maps: list[NDArray[np.float32]] = []
    for j in range(k):
        like = np.exp(-((d_all[:, j] / sigma_d[j]) ** 2)).astype(np.float32)
        m = np.zeros((h, w), dtype=np.float32)
        m[ys, xs] = like
        cluster_like_maps.append(_normalize01(m))

    # kNN consensus on sampled foreground pixels.
    if x_norm.shape[0] >= max(30, HSL_KMEDOIDS_KNN_K + 1):
        d2 = _pairwise_sqdist(x_norm.astype(np.float32), x_norm.astype(np.float32))
        np.fill_diagonal(d2, np.inf)
        k_nn = min(HSL_KMEDOIDS_KNN_K, x_norm.shape[0] - 1)
        nn_idx = np.argpartition(d2, kth=k_nn, axis=1)[:, :k_nn]
        nn_d2 = np.take_along_axis(d2, nn_idx, axis=1)
        nn_w = 1.0 / (nn_d2 + 1e-10)
        same = np.where(labels_s[nn_idx] == labels_s[:, None], 1.0, -1.0).astype(np.float32)
        raw = np.sum(same * nn_w, axis=1) / np.maximum(1e-9, np.sum(nn_w, axis=1))
        raw01 = np.clip(0.5 * (raw + 1.0), 0.0, 1.0).astype(np.float32)
        cons = np.zeros((h, w), dtype=np.float32)
        cons[ys_s, xs_s] = raw01
        cons = cv2.GaussianBlur(cons, (7, 7), 0)
        fill = float(np.median(raw01))
        cons = np.where(cons > 0, cons, fill).astype(np.float32)
        consensus_map = _normalize01(cons)
    else:
        consensus_map = np.full((h, w), 0.5, dtype=np.float32)
        warnings.append("W_HSL_KNN_CONSENSUS_SPARSE")

    warnings.append(f"I_HSL_KMEDOIDS_USED:{k}:{x_norm.shape[0]}")
    return cluster_like_maps, consensus_map, medoid_lab, warnings


def _assign_clusters_to_arms(
    arm_names: list[str],
    color_models: dict[str, ArmColorModel],
    cluster_lab: list[tuple[float, float, float]],
) -> dict[str, int | None]:
    assigned: dict[str, int | None] = {name: None for name in arm_names}
    if not cluster_lab:
        return assigned

    pairs: list[tuple[float, str, int]] = []
    for name in arm_names:
        ref = color_models[name].reference_lab() if name in color_models else None
        if ref is None:
            continue
        ref_arr = np.asarray(ref, dtype=np.float32)
        for idx, center in enumerate(cluster_lab):
            dist = float(np.linalg.norm(ref_arr - np.asarray(center, dtype=np.float32)))
            pairs.append((dist, name, idx))
    pairs.sort(key=lambda x: (x[0], x[1], x[2]))

    used_names: set[str] = set()
    used_clusters: set[int] = set()
    for _, name, idx in pairs:
        if name in used_names or idx in used_clusters:
            continue
        assigned[name] = idx
        used_names.add(name)
        used_clusters.add(idx)

    leftovers = [idx for idx in range(len(cluster_lab)) if idx not in used_clusters]
    for name in arm_names:
        if assigned[name] is None and leftovers:
            assigned[name] = leftovers.pop(0)
    return assigned


@dataclass(frozen=True)
class EvidenceCube:
    ridge_map: NDArray[np.float32]
    edge_map: NDArray[np.float32]
    text_penalty_map: NDArray[np.float32]
    text_region_penalty_map: NDArray[np.float32]
    line_penalty_map: NDArray[np.float32]
    axis_penalty_map: NDArray[np.float32]
    structure_map: NDArray[np.float32]
    overlap_consensus_map: NDArray[np.float32]
    candidate_mask: NDArray[np.bool_]
    arm_candidate_masks: dict[str, NDArray[np.bool_]]
    arm_score_maps: dict[str, NDArray[np.float32]]
    ambiguity_map: NDArray[np.float32]
    warning_codes: tuple[str, ...]


def build_evidence_cube(
    image: NDArray[np.uint8],
    plot_model: PlotModel,
    color_models: dict[str, ArmColorModel],
    hardpoint_guides: dict[str, tuple[tuple[int, int], ...]] | None = None,
) -> EvidenceCube:
    """Compute shared evidence once, then derive per-arm score maps."""
    warnings: list[str] = []
    x0, y0, x1, y1 = plot_model.plot_region
    roi = image[y0:y1, x0:x1]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray_f = gray.astype(np.float32)
    roi_hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV).astype(np.float32)
    hue_chan = roi_hsv[:, :, 0]
    sat_chan = roi_hsv[:, :, 1]
    roi_lab = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB).astype(np.float32)
    roi_lab_local = _local_mean_lab(roi_lab, k=LOCAL_COLOR_KERNEL)

    ridge = _ridge_response(gray)
    edge = _edge_response(gray)
    text_pen_raw = _text_penalty(gray)
    text_region_pen = _text_region_penalty(text_pen_raw)
    text_pen = np.maximum(text_pen_raw, 0.75 * text_region_pen).astype(np.float32)
    # Colorful pixels (high saturation) cannot be text — text is always black/gray.
    # Suppress both text penalties at saturated pixels to prevent dense KM step
    # patterns and censoring marks from being misclassified as text blobs.
    sat_not_text = sat_chan >= 50.0
    text_pen = np.where(sat_not_text, text_pen * 0.10, text_pen).astype(np.float32)
    text_region_pen = np.where(sat_not_text, text_region_pen * 0.10, text_region_pen).astype(np.float32)
    frame_pen, has_top_frame, has_right_frame = _frame_penalty(gray)
    prelim_mask = (
        ((ridge > max(0.10, CANDIDATE_RIDGE_THRESH * 0.7)) | sat_not_text)
        & (text_pen < min(0.65, CANDIDATE_TEXT_THRESH + 0.20))
        & (text_region_pen < min(0.70, CANDIDATE_TEXT_REGION_THRESH + 0.20))
        & (frame_pen < 0.65)
    )
    h_support = _horizontal_support(prelim_mask.astype(np.bool_))

    axis_pen = cv2.bitwise_or(plot_model.axis_mask, plot_model.tick_mask)
    axis_pen_f = (axis_pen.astype(np.float32) / 255.0).astype(np.float32)
    axis_pen_f = cv2.GaussianBlur(axis_pen_f, (5, 5), 0)
    axis_pen_f = _normalize01(axis_pen_f)
    line_pen, line_count = _straight_line_penalty(gray, axis_mask=axis_pen)

    direction = plot_model.curve_direction
    axis_weight = AXIS_PENALTY_WEIGHT
    candidate_axis_thresh = CANDIDATE_AXIS_THRESH
    axis_pen_for_structure = axis_pen_f
    if direction == "upward":
        h = axis_pen_f.shape[0]
        row_rel = np.linspace(0.0, 1.0, h, dtype=np.float32)[:, None]
        # Upward (cumulative incidence) curves are expected near the x-axis early;
        # reduce axis suppression in the lower band while still discouraging true-axis capture.
        soften = 0.35 + 0.65 * (1.0 - row_rel)
        axis_pen_for_structure = (axis_pen_f * soften).astype(np.float32)
        axis_weight = AXIS_PENALTY_WEIGHT_UPWARD
        candidate_axis_thresh = CANDIDATE_AXIS_THRESH_UPWARD
        warnings.append("I_AXIS_PENALTY_SOFTENED_UPWARD")
    elif direction == "unknown":
        axis_weight = AXIS_PENALTY_WEIGHT_UNKNOWN
        candidate_axis_thresh = CANDIDATE_AXIS_THRESH_UNKNOWN

    structure_base = (
        RIDGE_WEIGHT * ridge
        + EDGE_WEIGHT * edge
        + HORIZONTAL_SUPPORT_WEIGHT * h_support
        - axis_weight * axis_pen_for_structure
        - TEXT_PENALTY_WEIGHT * text_pen
        - TEXT_REGION_PENALTY_WEIGHT * text_region_pen
        - LINE_PENALTY_WEIGHT * line_pen
        - FRAME_PENALTY_WEIGHT * frame_pen
    ).astype(np.float32)
    structure_map = _normalize01(structure_base)

    candidate_mask = (
        ((ridge > CANDIDATE_RIDGE_THRESH) | sat_not_text)
        & (axis_pen_f < candidate_axis_thresh)
        & (text_pen < CANDIDATE_TEXT_THRESH)
        & (text_region_pen < CANDIDATE_TEXT_REGION_THRESH)
        & (line_pen < CANDIDATE_LINE_THRESH)
        & (frame_pen < 0.40)
        & (h_support > HORIZONTAL_SUPPORT_MIN)
        & (sat_chan >= MIN_CANDIDATE_SATURATION)
    )
    cand_density = float(np.mean(candidate_mask))
    if cand_density < 0.003:
        warnings.append(f"W_RIDGE_CANDIDATES_SPARSE:{cand_density:.4f}")
        relaxed = (
            (ridge > max(0.08, CANDIDATE_RIDGE_THRESH * 0.60))
            & (axis_pen_f < min(0.85, candidate_axis_thresh + 0.25))
            & (text_pen < min(0.90, CANDIDATE_TEXT_THRESH + 0.30))
            & (text_region_pen < min(0.92, CANDIDATE_TEXT_REGION_THRESH + 0.30))
            & (line_pen < min(0.92, CANDIDATE_LINE_THRESH + 0.30))
            & (frame_pen < 0.75)
        )
        relaxed_density = float(np.mean(relaxed))
        warnings.append(f"I_RIDGE_CANDIDATES_RELAXED:{relaxed_density:.4f}")
        if relaxed_density >= 0.001:
            candidate_mask = relaxed.astype(np.bool_)
        else:
            # Last fallback: highest-ridge pixels, still respecting axis/tick suppression.
            ridge_thr = float(np.quantile(ridge, 0.92))
            fallback = (
                (ridge >= ridge_thr)
                & (axis_pen_f < min(0.92, candidate_axis_thresh + 0.35))
                & (text_pen < 0.95)
                & (frame_pen < 0.85)
            )
            candidate_mask = fallback.astype(np.bool_)
            warnings.append(f"W_RIDGE_CANDIDATES_FALLBACK:{float(np.mean(candidate_mask)):.4f}")

    arm_names = sorted(color_models)
    cluster_like_maps, overlap_consensus_map, cluster_lab, cluster_warnings = _build_hsl_partition(
        roi_bgr=roi,
        candidate_mask=candidate_mask,
        axis_penalty=axis_pen_f,
        text_penalty=text_pen,
        n_clusters=max(1, len(arm_names)),
    )
    warnings.extend(cluster_warnings)
    if has_top_frame or has_right_frame:
        warnings.append(
            f"I_FRAME_PENALTY_APPLIED:{int(has_top_frame)}:{int(has_right_frame)}"
        )
    line_density = float(np.mean(line_pen > 0.35))
    if line_count > 0:
        warnings.append(f"I_STRAIGHT_LINE_PENALTY:{line_count}:{line_density:.4f}")
    cluster_assignment = _assign_clusters_to_arms(arm_names, color_models, cluster_lab)

    arm_maps: dict[str, NDArray[np.float32]] = {}
    arm_candidate_masks: dict[str, NDArray[np.bool_]] = {}
    for arm_name in arm_names:
        model = color_models[arm_name]
        ref_lab = model.reference_lab()
        color_like_local = _color_likelihood(
            roi_lab=roi_lab_local,
            reference_lab=ref_lab,
            reliability=model.reliability,
        )
        color_like_raw = _color_likelihood(
            roi_lab=roi_lab,
            reference_lab=ref_lab,
            reliability=model.reliability,
        )
        color_like = np.clip(
            LOCAL_COLOR_BLEND_WEIGHT * color_like_local
            + (1.0 - LOCAL_COLOR_BLEND_WEIGHT) * color_like_raw,
            0.0,
            1.0,
        ).astype(np.float32)
        cluster_like = np.zeros_like(color_like, dtype=np.float32)
        cluster_idx = cluster_assignment.get(arm_name)
        if isinstance(cluster_idx, int) and 0 <= cluster_idx < len(cluster_like_maps):
            cluster_like = cluster_like_maps[cluster_idx]
        color_mix = np.maximum(color_like, 0.85 * cluster_like).astype(np.float32)
        color_term = (
            COLOR_WEIGHT
            * np.clip(0.25 + 0.75 * model.reliability, 0.15, 1.0)
            * color_mix
        ).astype(np.float32)
        anti_color_penalty = (
            COLOR_ANTI_WEIGHT
            * np.clip(model.reliability, 0.0, 1.0)
            * (1.0 - color_mix)
        ).astype(np.float32)
        base = (structure_base + color_term - anti_color_penalty).astype(np.float32)
        color_relax = 0.0
        if ref_lab is not None:
            seed_vals = color_mix[candidate_mask]
            if seed_vals.size > 0:
                seed_thr = max(0.55, float(np.percentile(seed_vals, 70.0)))
                seed_mask = candidate_mask & (color_mix >= seed_thr)
                seed_count = int(np.count_nonzero(seed_mask))
                if seed_count >= 24:
                    ref_arr = np.asarray(ref_lab, dtype=np.float32)
                    dist_map = np.linalg.norm(
                        roi_lab_local - ref_arr[None, None, :],
                        axis=2,
                    ).astype(np.float32)
                    seed_dist = dist_map[seed_mask]
                    color_rmse = float(np.sqrt(np.mean(np.square(seed_dist))))
                    color_relax = float(
                        np.clip(
                            (color_rmse - COLOR_RMSE_RELAX_BASE) / COLOR_RMSE_RELAX_RANGE,
                            0.0,
                            1.0,
                        )
                    )
                    warnings.append(
                        f"I_COLOR_RMSE_RELAX:{arm_name}:{color_rmse:.2f}:{color_relax:.3f}:{seed_count}"
                    )

        # Strong color-lock mode when color model is reliable and dense enough:
        # keep candidates near the arm's color signature and demote off-color pixels.
        # When the color model is highly reliable, augment the structural candidate mask
        # with color-confirmed pixels to compensate for weak ridge detection on certain colors.
        if model.reliability >= 0.90 and ref_lab is not None:
            color_augment = (
                (color_like_raw >= 0.45)
                & (sat_chan >= HSV_SAT_MIN)
                & (axis_pen_f < min(0.85, candidate_axis_thresh + 0.25))
                & (text_pen < min(0.90, CANDIDATE_TEXT_THRESH + 0.30))
                & (frame_pen < 0.75)
            )
            arm_mask = np.logical_or(candidate_mask, color_augment)
        else:
            arm_mask = candidate_mask.copy()

        # Color-first lock for synthetic benchmark: strict saturation + hue gating.
        ref_hsv = _reference_hsv_from_lab(ref_lab)
        if ref_hsv is not None:
            ref_h, _, _ = ref_hsv
            sat_min = HSV_SAT_MIN if model.reliability >= COLOR_STRICT_RELIABILITY else HSV_SAT_MIN_LOW_RELIABILITY
            sat_mask = sat_chan >= sat_min

            hue_center, hue_thr, hue_seed_n, hue_mad, hue_seed_thr = _dynamic_hue_gate(
                hue_channel=hue_chan,
                sat_channel=sat_chan,
                candidate_mask=candidate_mask,
                color_mix=color_mix,
                ref_hue=ref_h,
                reliability=float(model.reliability),
            )
            if hue_center is not None:
                hue_dist = _hue_distance(hue_chan, hue_center)
                hue_mask = hue_dist <= float(hue_thr)
                hsv_mask = arm_mask & sat_mask & hue_mask
                hsv_density = float(np.mean(hsv_mask))
                hsv_cov = _mask_column_coverage(hsv_mask.astype(np.bool_))
                sat_density = float(np.mean(arm_mask & sat_mask))
                sat_cov = _mask_column_coverage((arm_mask & sat_mask).astype(np.bool_))
                warnings.append(
                    f"I_HSV_STRICT_LOCK:{arm_name}:{hsv_density:.4f}:{sat_density:.4f}:{hsv_cov:.3f}:{sat_cov:.3f}:{hue_center:.1f}:{hue_thr:.2f}:{hue_seed_n}:{hue_mad:.2f}:{hue_seed_thr:.3f}:{sat_min:.1f}"
                )
                if hsv_density >= HSV_STRICT_MIN_DENSITY and hsv_cov >= HSV_STRICT_MIN_COLUMN_COVERAGE:
                    arm_mask = np.logical_and(arm_mask, hsv_mask)
                    base = np.where(
                        hsv_mask,
                        base,
                        base - (2.80 * (1.0 - 0.30 * color_relax)),
                    ).astype(np.float32)
                elif sat_density >= HSV_STRICT_MIN_DENSITY and sat_cov >= HSV_STRICT_MIN_COLUMN_COVERAGE:
                    sat_only = arm_mask & sat_mask
                    arm_mask = np.logical_and(arm_mask, sat_only)
                    base = np.where(
                        sat_only,
                        base,
                        base - (2.10 * (1.0 - 0.25 * color_relax)),
                    ).astype(np.float32)
                    warnings.append(f"W_HSV_STRICT_HUE_SPARSE:{arm_name}:{hsv_density:.4f}:{hsv_cov:.3f}")
                else:
                    warnings.append(
                        f"W_HSV_STRICT_SPARSE:{arm_name}:{hsv_density:.4f}:{sat_density:.4f}:{hsv_cov:.3f}:{sat_cov:.3f}"
                    )

        # Optional grayscale fallback (disabled by default).
        if _env_bool(
            "KM_DIGITIZER_V5_GRAY_GATE",
            default=_env_bool("KM_DIGITIZER_V3_GRAY_GATE", default=APPLY_DYNAMIC_GRAY_GATE_DEFAULT),
        ):
            ref_gray = _reference_gray_from_lab(ref_lab)
            gray_center, gray_thr, gray_seed_n, gray_mad, gray_seed_thr = _dynamic_gray_gate(
                gray_f=gray_f,
                candidate_mask=candidate_mask,
                color_mix=color_mix,
                ref_gray=ref_gray,
                reliability=float(model.reliability),
            )
            if gray_center is not None:
                gray_mask = np.abs(gray_f - float(gray_center)) <= float(gray_thr)
                gray_density = float(np.mean(gray_mask & candidate_mask))
                warnings.append(
                    f"I_GRAY_DYNAMIC_LOCK:{arm_name}:{gray_density:.4f}:{gray_center:.1f}:{gray_thr:.2f}:{gray_seed_n}:{gray_mad:.2f}:{gray_seed_thr:.3f}"
                )
                if gray_density >= GRAY_DYNAMIC_MIN_DENSITY:
                    arm_mask = np.logical_and(arm_mask, gray_mask)
                    base = np.where(
                        gray_mask,
                        base,
                        base - (2.20 * (1.0 - 0.25 * color_relax)),
                    ).astype(np.float32)
                else:
                    warnings.append(
                        f"W_GRAY_DYNAMIC_SPARSE:{arm_name}:{gray_density:.4f}:{gray_thr:.2f}"
                    )

        if model.reliability >= COLOR_STRICT_RELIABILITY:
            candidate_vals = color_mix[candidate_mask]
            if candidate_vals.size > 0:
                q_thr = float(np.percentile(candidate_vals, COLOR_STRICT_QUANTILE))
                strict_floor = max(
                    0.45,
                    COLOR_STRICT_MIN_LIKELIHOOD - (COLOR_RMSE_RELAX_MAX * color_relax),
                )
                strict_thr = max(strict_floor, q_thr - (0.10 * color_relax))
            else:
                strict_thr = max(
                    0.45,
                    COLOR_STRICT_MIN_LIKELIHOOD - (COLOR_RMSE_RELAX_MAX * color_relax),
                )
            strict_mask = color_mix >= strict_thr
            strict_density = float(np.mean(strict_mask & candidate_mask))
            strict_cov = _mask_column_coverage((strict_mask & candidate_mask).astype(np.bool_))
            if strict_density >= COLOR_STRICT_MIN_DENSITY and strict_cov >= COLOR_STRICT_MIN_COLUMN_COVERAGE:
                arm_mask = candidate_mask & strict_mask
                base = np.where(
                    arm_mask,
                    base,
                    base - (COLOR_STRICT_OFF_PENALTY * (1.0 - 0.35 * color_relax)),
                ).astype(np.float32)
                warnings.append(
                    f"I_COLOR_STRICT_LOCK:{arm_name}:{strict_density:.4f}:{strict_cov:.3f}:{strict_thr:.3f}"
                )
            else:
                warnings.append(
                    f"W_COLOR_STRICT_SPARSE_SKIP:{arm_name}:{strict_density:.4f}:{strict_cov:.3f}:{strict_thr:.3f}"
                )

        # Hard lock for reliable color arms: do not allow off-color pixels.
        if model.reliability >= COLOR_HARD_LOCK_RELIABILITY:
            hard_floor = max(
                0.45,
                COLOR_HARD_LOCK_MIN_LIKELIHOOD - (0.12 * color_relax),
            )
            hard_anchor = max(
                0.45,
                (COLOR_STRICT_MIN_LIKELIHOOD - 0.05) - (0.12 * color_relax),
            )
            hard_thr = max(hard_floor, hard_anchor)
            hard_mask = candidate_mask & (color_mix >= hard_thr)
            hard_density = float(np.mean(hard_mask))
            hard_cov = _mask_column_coverage(hard_mask.astype(np.bool_))
            hard_cov_min = max(
                0.05,
                COLOR_HARD_LOCK_MIN_COLUMN_COVERAGE - (0.02 * color_relax),
            )
            if (
                hard_density >= max(0.0002, COLOR_STRICT_MIN_DENSITY * 0.6)
                and hard_cov >= hard_cov_min
            ):
                arm_mask = hard_mask
                base = np.where(
                    arm_mask,
                    base,
                    base - (1.80 * (1.0 - 0.30 * color_relax)),
                ).astype(np.float32)
                warnings.append(
                    f"I_COLOR_HARD_LOCK:{arm_name}:{hard_density:.4f}:{hard_cov:.3f}:{hard_thr:.3f}:{hard_cov_min:.3f}"
                )
            else:
                warnings.append(
                    f"W_COLOR_HARD_LOCK_SPARSE:{arm_name}:{hard_density:.4f}:{hard_cov:.3f}:{hard_thr:.3f}:{hard_cov_min:.3f}"
                )

        # Hardpoint corridor lock: keep tracing near clinically anchored landmarks.
        guide_mask = _hardpoint_corridor_mask(
            shape=arm_mask.shape,
            guides=(hardpoint_guides or {}).get(arm_name),
        )
        if guide_mask is not None:
            guide_keep = arm_mask & guide_mask
            guide_density = float(np.mean(guide_keep))
            guide_cov = _mask_column_coverage(guide_keep.astype(np.bool_))
            if (
                guide_density >= HARDPOINT_WINDOW_MIN_DENSITY
                and guide_cov >= HARDPOINT_WINDOW_MIN_COLUMN_COVERAGE
            ):
                arm_mask = guide_keep
                base = np.where(guide_mask, base, base - HARDPOINT_WINDOW_OFF_PENALTY).astype(np.float32)
                warnings.append(f"I_HARDPOINT_WINDOW_LOCK:{arm_name}:{guide_density:.4f}:{guide_cov:.3f}")
            else:
                warnings.append(f"W_HARDPOINT_WINDOW_SPARSE:{arm_name}:{guide_density:.4f}:{guide_cov:.3f}")

        # Bridge small gaps between adjacent KM step fragments before pruning.
        # The arm_mask is already color-locked so closing only connects same-arm steps.
        # For very sparse masks (thin lines + noise), use a larger closing kernel.
        arm_mask_density = float(np.mean(arm_mask))
        sparse_arm = arm_mask_density < 0.003
        close_ratio = 0.04 if sparse_arm else 0.02
        close_k = max(3, int(round(roi.shape[1] * close_ratio)))
        if close_k % 2 == 0:
            close_k += 1
        close_kern = cv2.getStructuringElement(cv2.MORPH_RECT, (close_k, 3))
        arm_mask_u8 = cv2.morphologyEx(
            arm_mask.astype(np.uint8) * 255, cv2.MORPH_CLOSE, close_kern,
        )
        arm_mask = arm_mask_u8 > 0

        # Connectivity prune: drop disconnected UI/text components that are not
        # plausible curve fragments.
        # Skip pruning for very sparse masks — the color lock already filtered to
        # arm-specific pixels, and pruning destroys the little signal we have.
        if sparse_arm:
            warnings.append(
                f"I_SPARSE_ARM_PRUNE_SKIP:{arm_name}:{arm_mask_density:.5f}"
            )
        else:
            pruned_mask, kept_components, dropped_components = _prune_curve_components(
                arm_mask.astype(np.bool_),
                direction=direction,
            )
            if kept_components > 0:
                arm_mask = pruned_mask
                warnings.append(
                    f"I_CURVE_COMPONENT_PRUNE:{arm_name}:{kept_components}:{dropped_components}"
                )
            elif dropped_components > 0:
                warnings.append(f"W_CURVE_COMPONENT_PRUNE_EMPTY:{arm_name}:{dropped_components}")

            # Enforce one component per arm to prevent curve hopping.
            if model.reliability >= COLOR_STRICT_RELIABILITY:
                primary_mask, selected, xspan = _select_primary_component(
                    arm_mask.astype(np.bool_),
                    direction=direction,
                )
                if selected:
                    arm_mask = primary_mask
                    warnings.append(f"I_PRIMARY_COMPONENT_LOCK:{arm_name}:{xspan:.3f}")
                else:
                    warnings.append(f"W_PRIMARY_COMPONENT_LOCK_SKIPPED:{arm_name}")

        # Ridge-first candidates: attenuate non-candidates when dense enough.
        if cand_density >= 0.01:
            base = np.where(arm_mask, base, base - 0.20)

        arm_maps[arm_name] = _normalize01(base.astype(np.float32))
        arm_candidate_masks[arm_name] = arm_mask.astype(np.bool_)
        if model.reliability <= 0.05:
            warnings.append(f"W_ARM_COLOR_UNINFORMATIVE:{arm_name}")

    # Enforce exclusive per-pixel ownership across arms when evidence margin is strong.
    # This prevents multiple arms from sharing the same candidate band and collapsing
    # onto a single traced line.
    if len(arm_maps) >= 2:
        stack_names = sorted(arm_maps)
        stack = np.stack([arm_maps[name] for name in stack_names], axis=0).astype(np.float32)
        best_idx = np.argmax(stack, axis=0)
        best = np.max(stack, axis=0)
        second = np.partition(stack, kth=-2, axis=0)[-2]
        margin = (best - second).astype(np.float32)
        owned_any = candidate_mask & (margin >= ARM_EXCLUSIVE_MIN_MARGIN)

        for i, arm_name in enumerate(stack_names):
            owned = owned_any & (best_idx == i)
            base_mask = arm_candidate_masks.get(arm_name, candidate_mask)
            exclusive_mask = np.logical_and(base_mask, owned)
            ex_density = float(np.mean(exclusive_mask))
            ex_cov = _mask_column_coverage(exclusive_mask.astype(np.bool_))
            if (
                ex_density >= ARM_EXCLUSIVE_MIN_DENSITY
                and ex_cov >= ARM_EXCLUSIVE_MIN_COLUMN_COVERAGE
            ):
                arm_candidate_masks[arm_name] = exclusive_mask.astype(np.bool_)
                owned_other = owned_any & (best_idx != i)
                adjusted = np.where(
                    owned_other,
                    arm_maps[arm_name] - 0.35,
                    arm_maps[arm_name],
                ).astype(np.float32)
                arm_maps[arm_name] = _normalize01(adjusted)
                warnings.append(
                    f"I_ARM_EXCLUSIVE_LOCK:{arm_name}:{ex_density:.4f}:{ex_cov:.3f}:{ARM_EXCLUSIVE_MIN_MARGIN:.3f}"
                )
            else:
                warnings.append(
                    f"W_ARM_EXCLUSIVE_SPARSE:{arm_name}:{ex_density:.4f}:{ex_cov:.3f}:{ARM_EXCLUSIVE_MIN_MARGIN:.3f}"
                )

    if not arm_maps:
        warnings.append("W_NO_ARM_SCORE_MAPS")
        ambiguity = np.zeros_like(ridge, dtype=np.float32)
    elif len(arm_maps) == 1:
        only = next(iter(arm_maps.values()))
        ambiguity = only.copy()
    else:
        stack = np.stack([arm_maps[name] for name in sorted(arm_maps)], axis=0)
        sorted_stack = np.sort(stack, axis=0)
        best = sorted_stack[-1]
        second = sorted_stack[-2]
        ambiguity = (best - second).astype(np.float32)
        ambiguity = _normalize01(ambiguity)
        if overlap_consensus_map.size and np.any(overlap_consensus_map > 0):
            ambiguity = (
                (1.0 - CONSENSUS_BLEND_WEIGHT) * ambiguity
                + CONSENSUS_BLEND_WEIGHT * overlap_consensus_map
            ).astype(np.float32)
            ambiguity = _normalize01(ambiguity)

    return EvidenceCube(
        ridge_map=ridge,
        edge_map=edge,
        text_penalty_map=text_pen,
        text_region_penalty_map=text_region_pen,
        line_penalty_map=line_pen,
        axis_penalty_map=axis_pen_f,
        structure_map=structure_map,
        overlap_consensus_map=overlap_consensus_map,
        candidate_mask=candidate_mask.astype(np.bool_),
        arm_candidate_masks=arm_candidate_masks,
        arm_score_maps=arm_maps,
        ambiguity_map=ambiguity,
        warning_codes=tuple(warnings),
    )
