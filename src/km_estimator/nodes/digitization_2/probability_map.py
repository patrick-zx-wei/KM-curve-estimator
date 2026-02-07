"""Shared evidence cube and per-arm score maps for digitization_v2."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from numpy.typing import NDArray

from .axis_map import PlotModel
from .legend_color import ArmColorModel

RIDGE_WEIGHT = 0.30
EDGE_WEIGHT = 0.20
COLOR_WEIGHT = 0.45
AXIS_PENALTY_WEIGHT = 0.85
AXIS_PENALTY_WEIGHT_UPWARD = 0.45
AXIS_PENALTY_WEIGHT_UNKNOWN = 0.65
TEXT_PENALTY_WEIGHT = 0.35
COLOR_GOOD_DISTANCE = 42.0
CANDIDATE_AXIS_THRESH = 0.25
CANDIDATE_AXIS_THRESH_UPWARD = 0.55
CANDIDATE_AXIS_THRESH_UNKNOWN = 0.40


def _normalize01(arr: NDArray[np.float32]) -> NDArray[np.float32]:
    lo = float(np.min(arr))
    hi = float(np.max(arr))
    if hi <= lo + 1e-9:
        return np.zeros_like(arr, dtype=np.float32)
    return (arr - lo) / (hi - lo)


def _ridge_response(gray: NDArray[np.uint8]) -> NDArray[np.float32]:
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    lap = cv2.Laplacian(blur, cv2.CV_32F, ksize=3)
    ridge = np.abs(lap).astype(np.float32)
    return _normalize01(ridge)


def _edge_response(gray: NDArray[np.uint8]) -> NDArray[np.float32]:
    edges = cv2.Canny(gray, 35, 110).astype(np.float32) / 255.0
    edges = cv2.GaussianBlur(edges, (3, 3), 0)
    return _normalize01(edges.astype(np.float32))


def _text_penalty(gray: NDArray[np.uint8]) -> NDArray[np.float32]:
    """Approximate text-like regions as a soft penalty map."""
    inv = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        25,
        6,
    )
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(inv, connectivity=8)
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
        # Text glyph-ish blobs: compact-ish and not large bars.
        if aspect > 7.5:
            continue
        out[labels == idx] = 255
    out = cv2.dilate(out, np.ones((2, 2), dtype=np.uint8), iterations=1)
    return (out.astype(np.float32) / 255.0).astype(np.float32)


def _color_likelihood(
    roi_lab: NDArray[np.float32],
    reference_lab: tuple[float, float, float] | None,
    reliability: float,
) -> NDArray[np.float32]:
    if reference_lab is None or reliability <= 0.0:
        return np.zeros(roi_lab.shape[:2], dtype=np.float32)

    ref = np.asarray(reference_lab, dtype=np.float32)
    dist = np.linalg.norm(roi_lab - ref[None, None, :], axis=2).astype(np.float32)
    # Saturating positive-only color contribution.
    likelihood = np.clip((COLOR_GOOD_DISTANCE - dist) / COLOR_GOOD_DISTANCE, 0.0, 1.0)
    return likelihood.astype(np.float32)


@dataclass(frozen=True)
class EvidenceCube:
    ridge_map: NDArray[np.float32]
    edge_map: NDArray[np.float32]
    text_penalty_map: NDArray[np.float32]
    axis_penalty_map: NDArray[np.float32]
    structure_map: NDArray[np.float32]
    candidate_mask: NDArray[np.bool_]
    arm_score_maps: dict[str, NDArray[np.float32]]
    ambiguity_map: NDArray[np.float32]
    warning_codes: tuple[str, ...]


def build_evidence_cube(
    image: NDArray[np.uint8],
    plot_model: PlotModel,
    color_models: dict[str, ArmColorModel],
) -> EvidenceCube:
    """Compute shared evidence once, then derive per-arm score maps."""
    warnings: list[str] = []
    x0, y0, x1, y1 = plot_model.plot_region
    roi = image[y0:y1, x0:x1]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    roi_lab = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB).astype(np.float32)

    ridge = _ridge_response(gray)
    edge = _edge_response(gray)
    text_pen = _text_penalty(gray)

    axis_pen = cv2.bitwise_or(plot_model.axis_mask, plot_model.tick_mask)
    axis_pen_f = (axis_pen.astype(np.float32) / 255.0).astype(np.float32)
    axis_pen_f = cv2.GaussianBlur(axis_pen_f, (5, 5), 0)
    axis_pen_f = _normalize01(axis_pen_f)

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
        - axis_weight * axis_pen_for_structure
        - TEXT_PENALTY_WEIGHT * text_pen
    ).astype(np.float32)
    structure_map = _normalize01(structure_base)

    candidate_mask = (
        (ridge > 0.20)
        & (axis_pen_f < candidate_axis_thresh)
        & (text_pen < 0.55)
    )
    cand_density = float(np.mean(candidate_mask))
    if cand_density < 0.003:
        warnings.append(f"W_RIDGE_CANDIDATES_SPARSE:{cand_density:.4f}")
        candidate_mask = np.ones_like(candidate_mask, dtype=np.bool_)

    arm_maps: dict[str, NDArray[np.float32]] = {}
    for arm_name in sorted(color_models):
        model = color_models[arm_name]
        color_like = _color_likelihood(
            roi_lab=roi_lab,
            reference_lab=model.reference_lab(),
            reliability=model.reliability,
        )
        color_term = (COLOR_WEIGHT * model.reliability * color_like).astype(np.float32)
        base = (structure_base + color_term).astype(np.float32)

        # Ridge-first candidates: attenuate non-candidates when dense enough.
        if cand_density >= 0.01:
            base = np.where(candidate_mask, base, base - 0.20)

        arm_maps[arm_name] = _normalize01(base.astype(np.float32))
        if model.reliability <= 0.05:
            warnings.append(f"W_ARM_COLOR_UNINFORMATIVE:{arm_name}")

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

    return EvidenceCube(
        ridge_map=ridge,
        edge_map=edge,
        text_penalty_map=text_pen,
        axis_penalty_map=axis_pen_f,
        structure_map=structure_map,
        candidate_mask=candidate_mask.astype(np.bool_),
        arm_score_maps=arm_maps,
        ambiguity_map=ambiguity,
        warning_codes=tuple(warnings),
    )
