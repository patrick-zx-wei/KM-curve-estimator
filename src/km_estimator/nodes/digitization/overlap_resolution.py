"""Overlap resolution with joint curve tracing and identity priors."""

from __future__ import annotations

from bisect import bisect_left
from itertools import product
from math import prod
from typing import Any, Literal

import cv2
import numpy as np
from numpy.typing import NDArray

from .axis_calibration import AxisMapping

CurveDirection = Literal["downward", "upward"]

MAX_CANDIDATES_PER_CURVE = 3
MAX_STATES_PER_X = 128
MAX_STATES_PER_X_HIGH_AMBIGUITY = 320
MAX_CANDIDATES_PER_CURVE_AMBIGUOUS = 9
MAX_CANDIDATES_PER_CURVE_MODERATE = 5
MIN_DIVERSE_STATES_PER_SIGNATURE = 2
MAX_DIVERSE_STATES_PER_SIGNATURE = 8
MAX_STATE_COMBINATIONS_PER_X = 1800

MEDIAN_ATTRACTION_WEIGHT = 0.08
MISSING_COLUMN_PENALTY = 0.45
MISSING_TRANSITION_PENALTY = 0.22
COLOR_PRIOR_WEIGHT = 0.8
COLOR_PATCH_RADIUS = 2
MIN_COLOR_WEIGHT_FACTOR = 0.2
SATURATION_CONFIDENCE_SCALE = 80.0
COLUMN_DEFER_CONFIDENCE_THRESHOLD = 0.32
COLUMN_DEFER_AMBIGUITY_THRESHOLD = 0.58
COLUMN_DEFER_BASE_PENALTY = 0.28
COLUMN_DEFER_CONFIDENCE_REWARD = 0.18
LOW_CONFIDENCE_MISSING_TRANSITION_FACTOR = 0.65
HIGH_CONFIDENCE_MISSING_TRANSITION_FACTOR = 1.2
MISSING_COLUMN_ONE_SIDED_MAX_GAP_FACTOR = 2.0
MISSING_COLUMN_SPAN_MAX_GAP_FACTOR = 2.5

SMOOTHNESS_WEIGHT = 0.15
UPWARD_MOVE_PENALTY = 8.0
ALLOWED_UPWARD_MOVE_PX = 2

OVERLAP_PENALTY = 3.0
MIN_VERTICAL_SEPARATION_PX = 2
SWAP_PENALTY = 0.35
CROSSING_MARGIN_PX = 2
COVERAGE_MISSING_STATE_PENALTY = 0.55
IDENTITY_DRIFT_WEIGHT = 0.04
LAB_DISTANCE_NORMALIZER = 180.0

AMBIGUITY_UNIQUE_Y_THRESHOLD = 5
AMBIGUITY_SPREAD_PX_THRESHOLD = 8.0
AMBIGUITY_DENSITY_THRESHOLD = 8
AMBIGUITY_MODERATE_THRESHOLD = 0.5
AMBIGUITY_HIGH_THRESHOLD = 0.85
CROSSING_RELAXED_SWAP_FACTOR = 0.35
CROSSING_LOCKED_SWAP_FACTOR = 1.25
NON_CROSSING_SWAP_FACTOR = 1.65
NON_CROSSING_STRONG_LOCK_FACTOR = 2.15
ORDER_LOCK_PENALTY = 0.42
ORDER_LOCK_STRONG_MULTIPLIER = 1.35
EARLY_ENVELOPE_REF_QUANTILE = 0.18
EARLY_ENVELOPE_MIN_SPAN_PX = 8
EARLY_ENVELOPE_MIN_DELTA_PX = 2.0
EARLY_ENVELOPE_TOLERANCE_PX = 2.0
EARLY_ENVELOPE_PENALTY = 0.07
COLOR_CALIBRATION_MIN_POINTS = 8
COLOR_CALIBRATION_MAX_POINTS = 24
COLOR_CALIBRATION_AMBIGUITY_MAX = 0.45
COLOR_CALIBRATION_BLEND = 0.35
COLOR_CALIBRATION_CONFIDENCE_FLOOR = 0.2


def _column_stats(
    ys: list[int],
) -> tuple[NDArray[np.int32], NDArray[np.float32], float, float, int]:
    """Compute reusable per-column stats once."""
    arr = np.asarray(ys, dtype=np.int32)
    unique, counts = np.unique(arr, return_counts=True)
    median_y = float(np.median(arr)) if arr.size else 0.0
    spread = float(unique[-1] - unique[0]) if unique.size > 1 else 0.0
    density = int(arr.size)
    return unique, counts.astype(np.float32), median_y, spread, density


def enforce_step_function(
    points: list[tuple[float, float]],
    time_tolerance: float = 0.001,
    direction: CurveDirection = "downward",
) -> list[tuple[float, float]]:
    """
    Convert digitized points to proper step function format.

    KM curves are step functions: survival stays constant between events,
    then drops vertically at event times. This function ensures the output
    follows this pattern by:
    1. Sorting by time
    2. For each survival drop, adding a horizontal segment to the drop point
    3. Then adding the vertical drop

    Args:
        points: List of (time, survival) tuples
        time_tolerance: Minimum time difference to consider points distinct

    Returns:
        List of (time, survival) tuples in proper step function format
    """
    if len(points) < 2:
        return points

    # Sort by time
    sorted_pts = sorted(points, key=lambda p: p[0])

    # Remove near-duplicates (same time within tolerance)
    deduped: list[tuple[float, float]] = [sorted_pts[0]]
    for t, s in sorted_pts[1:]:
        prev_t, prev_s = deduped[-1]
        if abs(t - prev_t) > time_tolerance:
            deduped.append((t, s))
        elif s < prev_s:  # Same time but lower survival - keep the drop
            deduped[-1] = (t, s)

    if len(deduped) < 2:
        return deduped

    # Build step function with direction-aware monotonicity:
    # - downward: non-increasing
    # - upward: non-decreasing
    step_points: list[tuple[float, float]] = [deduped[0]]
    monotone_bound = deduped[0][1]

    for i in range(1, len(deduped)):
        curr_t, curr_s = deduped[i]
        prev_t, prev_s = step_points[-1]

        if direction == "downward":
            # Clamp any survival increase to previous level (noise correction).
            if curr_s > monotone_bound:
                curr_s = monotone_bound

            if curr_s < prev_s:
                # Horizontal segment: carry previous survival to current time.
                step_points.append((curr_t, prev_s))
                # Vertical drop: drop to current survival at current time.
                step_points.append((curr_t, curr_s))
                monotone_bound = curr_s
            else:
                # Survival stayed same - extend horizontal.
                step_points.append((curr_t, curr_s))
        else:
            # Clamp any decrease to previous level (noise correction for upward curves).
            if curr_s < monotone_bound:
                curr_s = monotone_bound

            if curr_s > prev_s:
                # Horizontal segment: carry previous level to current time.
                step_points.append((curr_t, prev_s))
                # Vertical rise at current time.
                step_points.append((curr_t, curr_s))
                monotone_bound = curr_s
            else:
                step_points.append((curr_t, curr_s))

    return step_points


def _select_y_candidates(
    ys: list[int],
    x: int,
    color_maps: tuple[NDArray[np.float32], NDArray[np.float32]] | None,
    expected_lab: tuple[float, float, float] | None,
    stats: tuple[NDArray[np.int32], NDArray[np.float32], float, float, int] | None = None,
    max_candidates: int = MAX_CANDIDATES_PER_CURVE,
) -> list[int]:
    """Select representative y candidates for one x-column."""
    if stats is None:
        unique, counts, median_y, spread, _ = _column_stats(ys)
    else:
        unique, counts, median_y, spread, _ = stats
    if unique.size <= max_candidates:
        return unique.tolist()

    spread_scale = max(1.0, spread)

    density_scores = counts.copy()
    if density_scores.max() > 0:
        density_scores /= density_scores.max()

    centrality_scores = 1.0 / (1.0 + (np.abs(unique - median_y) / spread_scale))

    color_scores = np.ones_like(density_scores, dtype=np.float32)
    if color_maps is not None and expected_lab is not None:
        labs, sat_conf = _sample_color_prior_many(color_maps, x, unique)
        expected = np.asarray(expected_lab, dtype=np.float32)
        color_dist = np.linalg.norm(labs - expected[None, :], axis=1) / LAB_DISTANCE_NORMALIZER
        color_weight = MIN_COLOR_WEIGHT_FACTOR + (1.0 - MIN_COLOR_WEIGHT_FACTOR) * sat_conf
        # Higher is better for ranking.
        color_scores = 1.0 / (1.0 + color_dist * color_weight)

    ranking = density_scores * 0.5 + color_scores * 0.35 + centrality_scores * 0.15
    ranked_indices = np.argsort(-ranking)
    selected: list[int] = []

    # Preserve branch extremes before filling with ranked candidates.
    for y_anchor in (int(unique[0]), int(unique[-1]), int(round(median_y))):
        if y_anchor not in selected:
            selected.append(y_anchor)
            if len(selected) >= max_candidates:
                return sorted(selected)

    for idx in ranked_indices:
        y_val = int(unique[idx])
        if y_val in selected:
            continue
        selected.append(y_val)
        if len(selected) >= max_candidates:
            break

    return sorted(selected)


def _column_ambiguity(
    ys: list[int],
    x: int,
    color_maps: tuple[NDArray[np.float32], NDArray[np.float32]] | None,
    expected_lab: tuple[float, float, float] | None,
    stats: tuple[NDArray[np.int32], NDArray[np.float32], float, float, int] | None = None,
) -> float:
    """Estimate ambiguity level for one x-column on one curve in [0, 1]."""
    if not ys:
        return 0.4

    if stats is None:
        unique, _, _, spread, density = _column_stats(ys)
    else:
        unique, _, _, spread, density = stats
    unique_count = int(unique.size)

    score = 0.0
    if unique_count >= AMBIGUITY_UNIQUE_Y_THRESHOLD:
        score += min(0.45, (unique_count - AMBIGUITY_UNIQUE_Y_THRESHOLD + 1) * 0.08)
    if spread >= AMBIGUITY_SPREAD_PX_THRESHOLD:
        score += min(0.35, (spread - AMBIGUITY_SPREAD_PX_THRESHOLD + 1.0) * 0.03)
    if density >= AMBIGUITY_DENSITY_THRESHOLD:
        score += min(0.2, (density - AMBIGUITY_DENSITY_THRESHOLD + 1) * 0.02)

    if color_maps is not None and expected_lab is not None and unique_count > 1:
        y_probe = unique[: min(unique_count, 16)]
        labs, sat_conf = _sample_color_prior_many(color_maps, x, y_probe)
        expected = np.asarray(expected_lab, dtype=np.float32)
        color_dist = np.linalg.norm(labs - expected[None, :], axis=1) / LAB_DISTANCE_NORMALIZER
        weighted = color_dist * (
            MIN_COLOR_WEIGHT_FACTOR + (1.0 - MIN_COLOR_WEIGHT_FACTOR) * sat_conf
        )
        if weighted.size >= 2:
            sorted_d = np.sort(weighted)
            # If color separation between nearby branches is weak, ambiguity is higher.
            if float(sorted_d[1] - sorted_d[0]) < 0.04:
                score += 0.2

    return float(np.clip(score, 0.0, 1.0))


def _adaptive_candidate_limit(ambiguity: float, n_curves: int) -> int:
    """Scale per-curve candidate count by local ambiguity."""
    if n_curves >= 4:
        if ambiguity >= AMBIGUITY_HIGH_THRESHOLD:
            return min(MAX_CANDIDATES_PER_CURVE_AMBIGUOUS, 6)
        if ambiguity >= AMBIGUITY_MODERATE_THRESHOLD:
            return MAX_CANDIDATES_PER_CURVE_MODERATE
        return MAX_CANDIDATES_PER_CURVE
    if n_curves == 3:
        if ambiguity >= AMBIGUITY_HIGH_THRESHOLD:
            return min(MAX_CANDIDATES_PER_CURVE_AMBIGUOUS, 8)
        if ambiguity >= AMBIGUITY_MODERATE_THRESHOLD:
            return min(MAX_CANDIDATES_PER_CURVE_MODERATE, 5)
        return MAX_CANDIDATES_PER_CURVE

    if ambiguity >= AMBIGUITY_HIGH_THRESHOLD:
        return MAX_CANDIDATES_PER_CURVE_AMBIGUOUS
    if ambiguity >= AMBIGUITY_MODERATE_THRESHOLD:
        return MAX_CANDIDATES_PER_CURVE_MODERATE
    return MAX_CANDIDATES_PER_CURVE


def _adaptive_state_beam(ambiguity: float, n_curves: int) -> int:
    """Scale DP beam width by column ambiguity and number of curves."""
    beam = MAX_STATES_PER_X
    if n_curves >= 4:
        beam += 16
    if ambiguity >= AMBIGUITY_HIGH_THRESHOLD:
        beam = min(MAX_STATES_PER_X_HIGH_AMBIGUITY, beam + 96)
    elif ambiguity >= AMBIGUITY_MODERATE_THRESHOLD:
        beam = min(MAX_STATES_PER_X_HIGH_AMBIGUITY, beam + 48)
    return beam


def _trim_joint_candidates_for_budget(
    curve_candidates: list[list[int | None]],
    curve_costs: list[list[float]],
    max_combinations: int = MAX_STATE_COMBINATIONS_PER_X,
) -> tuple[list[list[int | None]], list[list[float]]]:
    """Bound cartesian state explosion by trimming worst local candidates."""
    if not curve_candidates:
        return curve_candidates, curve_costs

    lengths = [len(c) for c in curve_candidates]
    total = prod(lengths)
    if total <= max_combinations:
        return curve_candidates, curve_costs

    while total > max_combinations:
        reducible_idx = -1
        reducible_len = -1
        for idx, cands in enumerate(curve_candidates):
            min_len = 2 if any(v is None for v in cands) else 3
            if len(cands) > min_len and len(cands) > reducible_len:
                reducible_len = len(cands)
                reducible_idx = idx
        if reducible_idx < 0:
            break

        costs = curve_costs[reducible_idx]
        worst_idx = int(np.argmax(np.asarray(costs, dtype=np.float32)))
        curve_candidates[reducible_idx].pop(worst_idx)
        curve_costs[reducible_idx].pop(worst_idx)
        total = prod(len(c) for c in curve_candidates)

    return curve_candidates, curve_costs


def _ordering_signature(ys: tuple[int | None, ...]) -> tuple[int, ...]:
    """Encode pairwise vertical ordering to preserve identity-path diversity."""
    signature: list[int] = []
    n = len(ys)
    for i in range(n):
        for j in range(i + 1, n):
            yi = ys[i]
            yj = ys[j]
            if yi is None or yj is None:
                signature.append(0)
            elif abs(yi - yj) <= CROSSING_MARGIN_PX:
                signature.append(0)
            elif yi < yj:
                signature.append(-1)
            else:
                signature.append(1)
    return tuple(signature)


def _prune_states_with_identity_diversity(
    states: list[tuple[tuple[int | None, ...], float]],
    beam_size: int,
) -> list[tuple[tuple[int | None, ...], float]]:
    """Keep low-cost states while preserving ordering diversity for crossings."""
    if len(states) <= beam_size:
        return states

    sorted_states = sorted(states, key=lambda s: s[1])
    elite = max(8, beam_size // 4)
    kept: list[tuple[tuple[int | None, ...], float]] = sorted_states[:elite]
    kept_idx: set[int] = set(range(elite))

    max_per_signature = int(
        np.clip(beam_size // 12, MIN_DIVERSE_STATES_PER_SIGNATURE, MAX_DIVERSE_STATES_PER_SIGNATURE)
    )
    signature_counts: dict[tuple[int, ...], int] = {}
    for state in kept:
        sig = _ordering_signature(state[0])
        signature_counts[sig] = signature_counts.get(sig, 0) + 1

    for idx in range(elite, len(sorted_states)):
        if len(kept) >= beam_size:
            break
        state = sorted_states[idx]
        sig = _ordering_signature(state[0])
        if signature_counts.get(sig, 0) >= max_per_signature:
            continue
        kept.append(state)
        kept_idx.add(idx)
        signature_counts[sig] = signature_counts.get(sig, 0) + 1

    if len(kept) < beam_size:
        for idx, state in enumerate(sorted_states):
            if len(kept) >= beam_size:
                break
            if idx in kept_idx:
                continue
            kept.append(state)

    return kept


def _transition_cost(
    prev_y: int,
    curr_y: int,
    x_gap: int,
    curve_direction: CurveDirection = "downward",
) -> float:
    """
    Transition penalty between adjacent x-columns.

    The graph is left-to-right in x, with strong penalty for y decreases
    (which would imply survival increasing in KM semantics).
    """
    if curve_direction == "downward":
        adverse_move = max(0, prev_y - curr_y - ALLOWED_UPWARD_MOVE_PX)
    else:
        adverse_move = max(0, curr_y - prev_y - ALLOWED_UPWARD_MOVE_PX)
    smooth = abs(curr_y - prev_y) / max(1, x_gap)
    return adverse_move * UPWARD_MOVE_PENALTY + smooth * SMOOTHNESS_WEIGHT


def _find_nearest_x(xs_sorted: list[int], target_x: int) -> int | None:
    """Return nearest x from sorted list."""
    if not xs_sorted:
        return None
    idx = bisect_left(xs_sorted, target_x)
    if idx <= 0:
        return xs_sorted[0]
    if idx >= len(xs_sorted):
        return xs_sorted[-1]
    left = xs_sorted[idx - 1]
    right = xs_sorted[idx]
    if abs(target_x - left) <= abs(right - target_x):
        return left
    return right


def _neighboring_x_bounds(xs_sorted: list[int], target_x: int) -> tuple[int | None, int | None]:
    """Return closest observed x on left and right of target."""
    if not xs_sorted:
        return None, None
    idx = bisect_left(xs_sorted, target_x)
    left = xs_sorted[idx - 1] if idx > 0 else None
    right = xs_sorted[idx] if idx < len(xs_sorted) else None
    return left, right


def _median_y_at_x(x_map: dict[int, list[int]], x: int | None) -> int | None:
    """Robust median y for one observed x-column."""
    if x is None:
        return None
    ys = x_map.get(x)
    if not ys:
        return None
    return int(round(float(np.median(np.asarray(ys, dtype=np.float32)))))


def _interpolate_y_between_bounds(
    x_map: dict[int, list[int]],
    x: int,
    left_x: int,
    right_x: int,
) -> int | None:
    """Linear y interpolation between left/right observed columns."""
    left_y = _median_y_at_x(x_map, left_x)
    right_y = _median_y_at_x(x_map, right_x)
    if left_y is None or right_y is None or right_x <= left_x:
        return None
    ratio = float(x - left_x) / float(right_x - left_x)
    return int(round(float(left_y) + ratio * float(right_y - left_y)))


def _rgb01_to_lab(rgb: tuple[float, float, float]) -> tuple[float, float, float]:
    """Convert normalized RGB (0-1) to OpenCV LAB."""
    r = int(np.clip(round(rgb[0] * 255), 0, 255))
    g = int(np.clip(round(rgb[1] * 255), 0, 255))
    b = int(np.clip(round(rgb[2] * 255), 0, 255))
    pixel = np.array([[[b, g, r]]], dtype=np.uint8)
    lab = cv2.cvtColor(pixel, cv2.COLOR_BGR2LAB)[0, 0].astype(np.float32)
    return float(lab[0]), float(lab[1]), float(lab[2])


def _build_color_prior_maps(
    image: NDArray[Any],
    radius: int = COLOR_PATCH_RADIUS,
) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    """
    Precompute local color and saturation maps for all pixels.

    Returns tuple of:
    - local LAB map (float32, shape [H,W,3], OpenCV LAB channel order)
    - local saturation confidence map (float32 in [0, 1], shape [H,W])
    """
    kernel = max(1, radius * 2 + 1)

    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32)
    local_lab = cv2.blur(lab, (kernel, kernel))

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    sat = hsv[:, :, 1].astype(np.float32)
    local_sat = cv2.blur(sat, (kernel, kernel))
    sat_confidence = np.clip(local_sat / SATURATION_CONFIDENCE_SCALE, 0.0, 1.0).astype(
        np.float32, copy=False
    )

    return local_lab, sat_confidence


def _sample_color_prior(
    color_maps: tuple[NDArray[np.float32], NDArray[np.float32]],
    x: int,
    y: int,
) -> tuple[tuple[float, float, float], float]:
    """Sample precomputed color and confidence maps at clamped image coordinates."""
    lab_map, sat_map = color_maps
    h, w = lab_map.shape[:2]
    cx = int(np.clip(x, 0, w - 1))
    cy = int(np.clip(y, 0, h - 1))
    lab = lab_map[cy, cx]
    return (float(lab[0]), float(lab[1]), float(lab[2])), float(sat_map[cy, cx])


def _sample_color_prior_many(
    color_maps: tuple[NDArray[np.float32], NDArray[np.float32]],
    x: int,
    ys: NDArray[np.int32],
) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    """Vectorized color prior sampling for many y-values at one x."""
    lab_map, sat_map = color_maps
    h, w = lab_map.shape[:2]
    cx = int(np.clip(x, 0, w - 1))
    cy = np.clip(ys.astype(np.int32, copy=False), 0, h - 1)
    labs = lab_map[cy, cx]
    sat = sat_map[cy, cx]
    return labs.astype(np.float32, copy=False), sat.astype(np.float32, copy=False)


def _candidate_set_confidence(
    candidates: list[int | None],
    costs: list[float],
    ambiguity: float,
) -> float:
    """Estimate confidence of selecting one candidate y in this column."""
    if not candidates or not costs:
        return 0.1

    sorted_costs = np.sort(np.asarray(costs, dtype=np.float32))
    if sorted_costs.size <= 1:
        margin = 1.0
    else:
        denom = max(0.25, abs(float(sorted_costs[0])) + 0.25)
        margin = float(np.clip((sorted_costs[1] - sorted_costs[0]) / denom, 0.0, 1.0))

    has_missing = any(val is None for val in candidates)
    base = 0.25 + 0.55 * margin + 0.2 * (1.0 - ambiguity)
    if has_missing:
        base -= 0.12
    if has_missing and len(candidates) == 1:
        base = min(base, 0.22)
    return float(np.clip(base, 0.05, 1.0))


def _calibrate_expected_lab(
    x_map: dict[int, list[int]],
    sorted_keys: list[int],
    expected_lab: tuple[float, float, float] | None,
    color_maps: tuple[NDArray[np.float32], NDArray[np.float32]] | None,
) -> tuple[float, float, float] | None:
    """
    Calibrate legend-derived color priors from confident local curve patches.

    This keeps semantic color identity while adapting to antialiasing/compression shifts.
    """
    if expected_lab is None or color_maps is None or len(sorted_keys) < COLOR_CALIBRATION_MIN_POINTS:
        return expected_lab

    max_points = min(COLOR_CALIBRATION_MAX_POINTS, len(sorted_keys))
    step = max(1, len(sorted_keys) // max_points)

    samples: list[NDArray[np.float32]] = []
    weights: list[float] = []
    for x in sorted_keys[::step]:
        ys = x_map.get(x)
        if not ys:
            continue
        stats = _column_stats(ys)
        ambiguity = _column_ambiguity(
            ys,
            x=x,
            color_maps=color_maps,
            expected_lab=expected_lab,
            stats=stats,
        )
        if ambiguity > COLOR_CALIBRATION_AMBIGUITY_MAX:
            continue

        y_med = int(round(float(np.median(np.asarray(ys, dtype=np.float32)))))
        lab, sat_conf = _sample_color_prior(color_maps, x, y_med)
        if sat_conf < COLOR_CALIBRATION_CONFIDENCE_FLOOR:
            continue
        samples.append(np.asarray(lab, dtype=np.float32))
        weights.append(max(0.05, sat_conf * (1.05 - ambiguity)))

    if len(samples) < COLOR_CALIBRATION_MIN_POINTS:
        return expected_lab

    observed = np.average(np.stack(samples, axis=0), axis=0, weights=np.asarray(weights))
    expected = np.asarray(expected_lab, dtype=np.float32)
    blended = expected * (1.0 - COLOR_CALIBRATION_BLEND) + observed * COLOR_CALIBRATION_BLEND
    return float(blended[0]), float(blended[1]), float(blended[2])


def _state_order_lock_penalty(
    ys: tuple[int | None, ...],
    refs: list[tuple[bool, float | None, float]],
    crossing_window: bool,
    ambiguity: float,
) -> float:
    """Penalize pairwise identity order flips outside likely crossing windows."""
    if crossing_window:
        return 0.0

    penalty = 0.0
    for i in range(len(ys)):
        yi = ys[i]
        in_span_i, ref_i, _ = refs[i]
        if yi is None or ref_i is None or not in_span_i:
            continue
        for j in range(i + 1, len(ys)):
            yj = ys[j]
            in_span_j, ref_j, _ = refs[j]
            if yj is None or ref_j is None or not in_span_j:
                continue
            ref_diff = float(ref_i - ref_j)
            if abs(ref_diff) <= float(CROSSING_MARGIN_PX + 1):
                continue
            curr_diff = float(yi - yj)
            if abs(curr_diff) <= float(CROSSING_MARGIN_PX):
                continue
            if np.sign(ref_diff) != np.sign(curr_diff):
                penalty += ORDER_LOCK_PENALTY

    if ambiguity < AMBIGUITY_MODERATE_THRESHOLD:
        penalty *= ORDER_LOCK_STRONG_MULTIPLIER
    return float(penalty)


def _state_overlap_penalty(y_values: tuple[int | None, ...]) -> float:
    """Penalize curves collapsing to near-identical y at same x."""
    penalty = 0.0
    present = [y for y in y_values if y is not None]
    n = len(present)
    for i in range(n):
        for j in range(i + 1, n):
            dy = abs(present[i] - present[j])
            if dy < MIN_VERTICAL_SEPARATION_PX:
                penalty += OVERLAP_PENALTY * (MIN_VERTICAL_SEPARATION_PX - dy)
    return penalty


def _swap_penalty(
    prev_y: tuple[int | None, ...],
    curr_y: tuple[int | None, ...],
) -> float:
    """Penalize abrupt pairwise order swaps between curves."""
    penalty = 0.0
    n = len(curr_y)
    for i in range(n):
        for j in range(i + 1, n):
            if (
                prev_y[i] is None
                or prev_y[j] is None
                or curr_y[i] is None
                or curr_y[j] is None
            ):
                continue
            prev_diff = prev_y[i] - prev_y[j]
            curr_diff = curr_y[i] - curr_y[j]
            if (
                abs(prev_diff) > CROSSING_MARGIN_PX
                and abs(curr_diff) > CROSSING_MARGIN_PX
                and np.sign(prev_diff) != np.sign(curr_diff)
            ):
                penalty += SWAP_PENALTY
    return penalty


def _build_curve_x_maps(
    raw_curves: dict[str, list[tuple[int, int]]],
) -> tuple[dict[str, dict[int, list[int]]], list[int]]:
    """Build x->ys maps for each curve and global union x-values."""
    maps: dict[str, dict[int, list[int]]] = {}
    union_x: set[int] = set()

    for name, pixels in raw_curves.items():
        x_map: dict[int, list[int]] = {}
        for px, py in pixels:
            x_map.setdefault(px, []).append(py)
            union_x.add(px)
        maps[name] = x_map

    return maps, sorted(union_x)


def _detect_crossing_window_mask(
    curve_names: list[str],
    x_maps: dict[str, dict[int, list[int]]],
    x_values: list[int],
    dilation: int = 8,
) -> list[bool]:
    """Detect likely crossing windows and dilate around them."""
    if len(curve_names) < 2 or not x_values:
        return [False] * len(x_values)

    pair_prev_sign: dict[tuple[int, int], int] = {}
    crossing_indices: set[int] = set()

    for idx, x in enumerate(x_values):
        medians: list[float | None] = []
        for name in curve_names:
            ys = x_maps[name].get(x)
            medians.append(float(np.median(ys)) if ys else None)

        for i in range(len(curve_names)):
            yi = medians[i]
            if yi is None:
                continue
            for j in range(i + 1, len(curve_names)):
                yj = medians[j]
                if yj is None:
                    continue
                diff = yi - yj
                key = (i, j)
                if abs(diff) <= (CROSSING_MARGIN_PX + 1):
                    crossing_indices.add(idx)
                    continue
                sign = 1 if diff > 0 else -1
                prev_sign = pair_prev_sign.get(key)
                if prev_sign is not None and sign != prev_sign:
                    crossing_indices.add(idx)
                pair_prev_sign[key] = sign

    if not crossing_indices:
        return [False] * len(x_values)

    mask = [False] * len(x_values)
    for idx in crossing_indices:
        lo = max(0, idx - dilation)
        hi = min(len(mask) - 1, idx + dilation)
        for k in range(lo, hi + 1):
            mask[k] = True
    return mask


def _curve_candidates_for_x(
    x_map: dict[int, list[int]],
    sorted_keys: list[int],
    x: int,
    color_maps: tuple[NDArray[np.float32], NDArray[np.float32]] | None,
    expected_lab: tuple[float, float, float] | None,
    max_fallback_gap_px: int,
    n_curves: int,
    curve_direction: CurveDirection,
    early_envelope: tuple[int, int, float, float] | None,
) -> tuple[list[int | None], list[float], float, float]:
    """Generate y-candidates and local costs for one curve at one x."""
    ys = x_map.get(x)
    if ys:
        stats = _column_stats(ys)
        _, _, median_y, _, _ = stats
        ambiguity = _column_ambiguity(ys, x, color_maps, expected_lab, stats=stats)
        candidate_limit = _adaptive_candidate_limit(ambiguity, n_curves=n_curves)
        candidates = _select_y_candidates(
            ys,
            x=x,
            color_maps=color_maps,
            expected_lab=expected_lab,
            stats=stats,
            max_candidates=candidate_limit,
        )
        base_costs = [
            abs(c - median_y) * MEDIAN_ATTRACTION_WEIGHT
            for c in candidates
        ]
    else:
        left_x, right_x = _neighboring_x_bounds(sorted_keys, x)
        in_span = left_x is not None and right_x is not None
        left_gap = abs(x - left_x) if left_x is not None else 10**9
        right_gap = abs(right_x - x) if right_x is not None else 10**9
        nearest_gap = min(left_gap, right_gap)

        if (
            not in_span
            and nearest_gap > int(max_fallback_gap_px * MISSING_COLUMN_ONE_SIDED_MAX_GAP_FACTOR)
        ):
            return [None], [MISSING_COLUMN_PENALTY * 1.9], 0.35, 0.18

        if in_span and (
            left_gap > int(max_fallback_gap_px * MISSING_COLUMN_SPAN_MAX_GAP_FACTOR)
            or right_gap > int(max_fallback_gap_px * MISSING_COLUMN_SPAN_MAX_GAP_FACTOR)
        ):
            return [None], [MISSING_COLUMN_PENALTY * 1.7], 0.42, 0.20

        if in_span and left_x is not None and right_x is not None:
            interp_y = _interpolate_y_between_bounds(x_map, x, left_x, right_x)
            left_y = _median_y_at_x(x_map, left_x)
            right_y = _median_y_at_x(x_map, right_x)
            span_gap = max(1.0, float(left_gap + right_gap))
            gap_ratio = float(span_gap / max(1.0, 2.0 * max_fallback_gap_px))

            candidates: list[int | None] = [None]
            base_costs: list[float] = [
                MISSING_COLUMN_PENALTY
                * (0.72 + 0.22 * float(np.clip(gap_ratio, 0.0, 1.8)))
            ]
            if interp_y is not None:
                candidates.append(interp_y)
                base_costs.append(
                    MISSING_COLUMN_PENALTY
                    * (0.95 + 0.45 * float(np.clip(gap_ratio, 0.0, 2.2)))
                )
            if left_y is not None and left_gap <= max_fallback_gap_px and left_y not in candidates:
                candidates.append(left_y)
                base_costs.append(MISSING_COLUMN_PENALTY * (1.06 + 0.52 * gap_ratio))
            if right_y is not None and right_gap <= max_fallback_gap_px and right_y not in candidates:
                candidates.append(right_y)
                base_costs.append(MISSING_COLUMN_PENALTY * (1.06 + 0.52 * gap_ratio))

            ambiguity = float(np.clip(0.48 + 0.25 * gap_ratio, 0.35, 0.86))
        else:
            nearest_x = left_x if left_x is not None else right_x
            nearest_y = _median_y_at_x(x_map, nearest_x)
            if nearest_y is None:
                return [None], [MISSING_COLUMN_PENALTY * 1.9], 0.38, 0.18

            gap_ratio = float(
                np.clip(nearest_gap / max(1.0, float(max_fallback_gap_px)), 0.0, 3.0)
            )
            candidates = [None, nearest_y]
            base_costs = [
                MISSING_COLUMN_PENALTY * (0.78 + 0.28 * gap_ratio),
                MISSING_COLUMN_PENALTY * (1.45 + 0.95 * gap_ratio),
            ]
            ambiguity = 0.62 if nearest_gap <= max_fallback_gap_px else 0.38

    if color_maps is None or expected_lab is None:
        if early_envelope is not None:
            base_costs = [
                cost + _early_envelope_penalty(y, x, early_envelope, curve_direction)
                if y is not None
                else cost
                for y, cost in zip(candidates, base_costs)
            ]
        confidence = _candidate_set_confidence(candidates, base_costs, ambiguity)
        if (
            ambiguity >= COLUMN_DEFER_AMBIGUITY_THRESHOLD
            and confidence <= COLUMN_DEFER_CONFIDENCE_THRESHOLD
        ):
            defer_cost = (
                COLUMN_DEFER_BASE_PENALTY
                + ambiguity * 0.25
                - confidence * COLUMN_DEFER_CONFIDENCE_REWARD
            )
            if None not in candidates:
                candidates.append(None)
                base_costs.append(float(defer_cost))
            else:
                none_idx = candidates.index(None)
                base_costs[none_idx] = min(base_costs[none_idx], float(defer_cost))
        return candidates, base_costs, ambiguity, confidence

    costs = []
    for candidate_y, base in zip(candidates, base_costs):
        if candidate_y is None:
            costs.append(base)
            continue
        lab, sat_confidence = _sample_color_prior(color_maps, x, candidate_y)
        color_dist = float(np.sqrt(sum((a - b) ** 2 for a, b in zip(lab, expected_lab))))
        color_dist /= LAB_DISTANCE_NORMALIZER
        color_weight = COLOR_PRIOR_WEIGHT * (
            MIN_COLOR_WEIGHT_FACTOR + (1.0 - MIN_COLOR_WEIGHT_FACTOR) * sat_confidence
        )
        penalty = 0.0
        if early_envelope is not None:
            penalty = _early_envelope_penalty(candidate_y, x, early_envelope, curve_direction)
        costs.append(base + color_dist * color_weight + penalty)
    confidence = _candidate_set_confidence(candidates, costs, ambiguity)
    if (
        ambiguity >= COLUMN_DEFER_AMBIGUITY_THRESHOLD
        and confidence <= COLUMN_DEFER_CONFIDENCE_THRESHOLD
    ):
        defer_cost = (
            COLUMN_DEFER_BASE_PENALTY
            + ambiguity * 0.25
            - confidence * COLUMN_DEFER_CONFIDENCE_REWARD
        )
        if None not in candidates:
            candidates.append(None)
            costs.append(float(defer_cost))
        else:
            none_idx = candidates.index(None)
            costs[none_idx] = min(costs[none_idx], float(defer_cost))
        confidence = min(confidence, COLUMN_DEFER_CONFIDENCE_THRESHOLD)

    return candidates, costs, ambiguity, confidence


def _build_early_envelope(
    x_map: dict[int, list[int]],
    sorted_keys: list[int],
    curve_direction: CurveDirection,
) -> tuple[int, int, float, float] | None:
    """Build a soft early-time envelope from observed pixels, if trend is informative."""
    if len(sorted_keys) < 6:
        return None
    first_x = int(sorted_keys[0])
    ref_idx = min(
        len(sorted_keys) - 1,
        max(2, int(round(len(sorted_keys) * EARLY_ENVELOPE_REF_QUANTILE))),
    )
    ref_x = int(sorted_keys[ref_idx])
    if ref_x - first_x < EARLY_ENVELOPE_MIN_SPAN_PX:
        return None

    first_y = float(np.median(x_map[first_x]))
    ref_y = float(np.median(x_map[ref_x]))
    if curve_direction == "downward":
        delta = ref_y - first_y
    else:
        delta = first_y - ref_y
    if delta < EARLY_ENVELOPE_MIN_DELTA_PX:
        return None

    slope = float(delta / max(1, ref_x - first_x))
    return (first_x, ref_x, first_y, slope)


def _early_envelope_penalty(
    y_val: int,
    x: int,
    envelope: tuple[int, int, float, float],
    curve_direction: CurveDirection,
) -> float:
    """Penalize implausible upper-envelope excursions in early ambiguous columns."""
    start_x, end_x, start_y, slope = envelope
    if x < start_x or x > end_x:
        return 0.0

    if curve_direction == "downward":
        expected_y = start_y + slope * float(x - start_x)
        violation = (expected_y - EARLY_ENVELOPE_TOLERANCE_PX) - float(y_val)
    else:
        expected_y = start_y - slope * float(x - start_x)
        violation = float(y_val) - (expected_y + EARLY_ENVELOPE_TOLERANCE_PX)

    if violation <= 0.0:
        return 0.0
    return float(violation * EARLY_ENVELOPE_PENALTY)


def _precompute_identity_refs(
    x: int,
    curve_names: list[str],
    sorted_keys: dict[str, list[int]],
    x_maps: dict[str, dict[int, list[int]]],
) -> list[tuple[bool, float | None, float]]:
    """
    Precompute identity references for one x-column.

    Returns per-curve tuples:
    - in_span: whether x is within observed curve span
    - ref_y: nearest observed median y (or None)
    - drift_weight: precomputed multiplier for |y - ref_y|
    """
    refs: list[tuple[bool, float | None, float]] = []
    for name in curve_names:
        keys = sorted_keys[name]
        if not keys:
            refs.append((False, None, 0.0))
            continue
        min_x = keys[0]
        max_x = keys[-1]
        in_span = min_x <= x <= max_x

        nearest_x = _find_nearest_x(keys, x)
        if nearest_x is None:
            refs.append((in_span, None, 0.0))
            continue

        ref_y = float(np.median(x_maps[name][nearest_x]))
        x_gap = abs(nearest_x - x)
        drift_weight = IDENTITY_DRIFT_WEIGHT / max(1, x_gap + 1)
        refs.append((in_span, ref_y, drift_weight))
    return refs


def _state_identity_penalty(
    ys: tuple[int | None, ...],
    refs: list[tuple[bool, float | None, float]],
) -> float:
    """Coverage-aware identity penalty to reduce swaps/truncation in overlaps."""
    penalty = 0.0
    for y, (in_span, ref_y, drift_weight) in zip(ys, refs):
        if y is None:
            if in_span:
                penalty += COVERAGE_MISSING_STATE_PENALTY
            continue
        if ref_y is None:
            continue
        penalty += abs(y - ref_y) * drift_weight
    return penalty


def _states_to_dense_arrays(
    states: list[tuple[tuple[int | None, ...], float]],
    n_curves: int,
) -> tuple[NDArray[np.float32], NDArray[np.bool_], NDArray[np.float32]]:
    """Convert states to dense arrays for vectorized DP transitions."""
    n_states = len(states)
    ys = np.zeros((n_states, n_curves), dtype=np.float32)
    missing = np.zeros((n_states, n_curves), dtype=np.bool_)
    data_cost = np.zeros(n_states, dtype=np.float32)

    for i, (state_y, cost) in enumerate(states):
        data_cost[i] = float(cost)
        for j, val in enumerate(state_y):
            if val is None:
                missing[i, j] = True
            else:
                ys[i, j] = float(val)

    return ys, missing, data_cost


def _transition_matrix_vectorized(
    prev_y: NDArray[np.float32],
    prev_missing: NDArray[np.bool_],
    curr_y: NDArray[np.float32],
    curr_missing: NDArray[np.bool_],
    x_gap: int,
    swap_penalty: float = SWAP_PENALTY,
    missing_transition_penalty: float = MISSING_TRANSITION_PENALTY,
    curve_direction: CurveDirection = "downward",
) -> NDArray[np.float32]:
    """
    Compute transition costs between all prev/curr states in vectorized form.

    Shapes:
    - prev_y: [P, C]
    - curr_y: [Q, C]
    Returns:
    - transition: [P, Q]
    """
    prev_exp = prev_y[:, None, :]  # [P,1,C]
    curr_exp = curr_y[None, :, :]  # [1,Q,C]
    prev_m = prev_missing[:, None, :]
    curr_m = curr_missing[None, :, :]

    both_missing = prev_m & curr_m
    one_missing = prev_m ^ curr_m
    valid = ~(both_missing | one_missing)

    x_gap_safe = float(max(1, x_gap))
    if curve_direction == "downward":
        adverse_move = np.maximum(0.0, prev_exp - curr_exp - float(ALLOWED_UPWARD_MOVE_PX))
    else:
        adverse_move = np.maximum(0.0, curr_exp - prev_exp - float(ALLOWED_UPWARD_MOVE_PX))
    smooth = np.abs(curr_exp - prev_exp) / x_gap_safe
    base = adverse_move * float(UPWARD_MOVE_PENALTY) + smooth * float(SMOOTHNESS_WEIGHT)
    base = np.where(valid, base, 0.0).sum(axis=2, dtype=np.float32)

    transition = base
    transition += both_missing.sum(axis=2, dtype=np.float32) * float(missing_transition_penalty * 0.5)
    transition += one_missing.sum(axis=2, dtype=np.float32) * float(missing_transition_penalty)

    n_curves = prev_y.shape[1]
    if n_curves >= 2:
        for i in range(n_curves):
            for j in range(i + 1, n_curves):
                valid_pair = ~(prev_m[:, :, i] | prev_m[:, :, j] | curr_m[:, :, i] | curr_m[:, :, j])
                prev_diff = prev_exp[:, :, i] - prev_exp[:, :, j]
                curr_diff = curr_exp[:, :, i] - curr_exp[:, :, j]
                swap_mask = (
                    valid_pair
                    & (np.abs(prev_diff) > float(CROSSING_MARGIN_PX))
                    & (np.abs(curr_diff) > float(CROSSING_MARGIN_PX))
                    & (np.sign(prev_diff) != np.sign(curr_diff))
                )
                transition += swap_mask.astype(np.float32) * float(swap_penalty)

    return transition.astype(np.float32, copy=False)


def _trace_curves_joint(
    raw_curves: dict[str, list[tuple[int, int]]],
    x_values: list[int],
    image: NDArray[Any] | None,
    curve_color_priors: dict[str, tuple[float, float, float] | None] | None,
    max_fallback_gap_px: int,
    crossing_relaxed: bool = False,
    curve_direction: CurveDirection = "downward",
) -> dict[str, list[tuple[int, int]]]:
    """Jointly trace all curves with state exclusivity and identity priors."""
    if not x_values:
        return {name: [] for name in raw_curves}

    curve_names = list(raw_curves.keys())
    x_maps, _ = _build_curve_x_maps(raw_curves)
    sorted_keys = {name: sorted(x_maps[name].keys()) for name in curve_names}
    crossing_window_mask = _detect_crossing_window_mask(
        curve_names=curve_names,
        x_maps=x_maps,
        x_values=x_values,
    )
    color_maps = _build_color_prior_maps(image) if image is not None else None
    expected_labs: dict[str, tuple[float, float, float] | None] = {
        name: (
            _rgb01_to_lab(curve_color_priors[name])
            if curve_color_priors and curve_color_priors.get(name) is not None
            else None
        )
        for name in curve_names
    }
    if color_maps is not None:
        expected_labs = {
            name: _calibrate_expected_lab(
                x_map=x_maps[name],
                sorted_keys=sorted_keys[name],
                expected_lab=expected_labs.get(name),
                color_maps=color_maps,
            )
            for name in curve_names
        }
    early_envelopes: dict[str, tuple[int, int, float, float] | None] = {
        name: _build_early_envelope(
            x_map=x_maps[name],
            sorted_keys=sorted_keys[name],
            curve_direction=curve_direction,
        )
        for name in curve_names
    }

    states_by_x: list[list[tuple[tuple[int | None, ...], float]]] = []
    state_y_by_x: list[NDArray[np.float32]] = []
    state_missing_by_x: list[NDArray[np.bool_]] = []
    state_data_cost_by_x: list[NDArray[np.float32]] = []
    ambiguity_by_x: list[float] = []
    confidence_by_x: list[float] = []
    for x_idx, x in enumerate(x_values):
        curve_candidates: list[list[int | None]] = []
        curve_costs: list[list[float]] = []
        curve_ambiguities: list[float] = []
        curve_confidences: list[float] = []
        n_curves = len(curve_names)
        for name in curve_names:
            expected_lab = expected_labs.get(name)
            candidates, costs, ambiguity, confidence = _curve_candidates_for_x(
                x_maps[name],
                sorted_keys[name],
                x,
                color_maps,
                expected_lab,
                max_fallback_gap_px,
                n_curves=n_curves,
                curve_direction=curve_direction,
                early_envelope=early_envelopes.get(name),
            )
            curve_candidates.append(candidates)
            curve_costs.append(costs)
            curve_ambiguities.append(ambiguity)
            curve_confidences.append(confidence)

        x_ambiguity = float(np.mean(curve_ambiguities)) if curve_ambiguities else 0.0
        x_confidence = float(np.mean(curve_confidences)) if curve_confidences else 0.5

        curve_candidates, curve_costs = _trim_joint_candidates_for_budget(
            curve_candidates, curve_costs
        )
        refs = _precompute_identity_refs(x, curve_names, sorted_keys, x_maps)

        combos = product(*(range(len(cands)) for cands in curve_candidates))
        states: list[tuple[tuple[int | None, ...], float]] = []
        for combo in combos:
            ys = tuple(curve_candidates[i][choice] for i, choice in enumerate(combo))
            cost = sum(curve_costs[i][choice] for i, choice in enumerate(combo))
            cost += _state_overlap_penalty(ys)
            cost += _state_identity_penalty(ys, refs)
            cost += _state_order_lock_penalty(
                ys=ys,
                refs=refs,
                crossing_window=bool(crossing_window_mask[x_idx]),
                ambiguity=x_ambiguity,
            )
            states.append((ys, float(cost)))

        beam_size = _adaptive_state_beam(x_ambiguity, len(curve_names))
        if crossing_relaxed and x_ambiguity >= AMBIGUITY_MODERATE_THRESHOLD:
            beam_size = min(MAX_STATES_PER_X_HIGH_AMBIGUITY, beam_size + 48)
        pruned_states = _prune_states_with_identity_diversity(states, beam_size)
        states_by_x.append(pruned_states)
        y_arr, missing_arr, data_cost_arr = _states_to_dense_arrays(pruned_states, n_curves)
        state_y_by_x.append(y_arr)
        state_missing_by_x.append(missing_arr)
        state_data_cost_by_x.append(data_cost_arr)
        ambiguity_by_x.append(x_ambiguity)
        confidence_by_x.append(x_confidence)

    dp_costs: list[np.ndarray] = []
    parents: list[np.ndarray] = []

    first_costs = state_data_cost_by_x[0]
    dp_costs.append(first_costs)
    parents.append(np.full(len(first_costs), -1, dtype=np.int32))

    for i in range(1, len(x_values)):
        prev_cost = dp_costs[i - 1]
        prev_y = state_y_by_x[i - 1]
        prev_missing = state_missing_by_x[i - 1]
        curr_y = state_y_by_x[i]
        curr_missing = state_missing_by_x[i]
        curr_data_cost = state_data_cost_by_x[i]
        n_curr = curr_y.shape[0]
        curr_cost = np.full(n_curr, np.inf, dtype=np.float32)
        curr_parent = np.full(n_curr, -1, dtype=np.int32)
        x_gap = x_values[i] - x_values[i - 1]
        local_ambiguity = max(ambiguity_by_x[i - 1], ambiguity_by_x[i])
        local_confidence = min(confidence_by_x[i - 1], confidence_by_x[i])
        swap_penalty = SWAP_PENALTY
        in_crossing_window = bool(crossing_window_mask[i] or crossing_window_mask[i - 1])
        if in_crossing_window:
            if crossing_relaxed:
                swap_penalty = SWAP_PENALTY * CROSSING_RELAXED_SWAP_FACTOR
        else:
            # Outside crossing windows, keep identity assignment stable.
            swap_penalty = SWAP_PENALTY * NON_CROSSING_SWAP_FACTOR
            if crossing_relaxed:
                swap_penalty *= CROSSING_LOCKED_SWAP_FACTOR
            if local_ambiguity < AMBIGUITY_MODERATE_THRESHOLD:
                swap_penalty = max(
                    swap_penalty,
                    SWAP_PENALTY * NON_CROSSING_STRONG_LOCK_FACTOR,
                )

        missing_transition_penalty = MISSING_TRANSITION_PENALTY
        if local_confidence <= COLUMN_DEFER_CONFIDENCE_THRESHOLD:
            missing_transition_penalty *= LOW_CONFIDENCE_MISSING_TRANSITION_FACTOR
        elif local_confidence >= 0.72:
            missing_transition_penalty *= HIGH_CONFIDENCE_MISSING_TRANSITION_FACTOR

        transition = _transition_matrix_vectorized(
            prev_y=prev_y,
            prev_missing=prev_missing,
            curr_y=curr_y,
            curr_missing=curr_missing,
            x_gap=x_gap,
            swap_penalty=swap_penalty,
            missing_transition_penalty=missing_transition_penalty,
            curve_direction=curve_direction,
        )
        total = prev_cost[:, None] + transition + curr_data_cost[None, :]
        best_parent = np.argmin(total, axis=0).astype(np.int32)
        curr_cost = total[best_parent, np.arange(n_curr, dtype=np.int32)].astype(np.float32)
        curr_parent = best_parent

        dp_costs.append(curr_cost)
        parents.append(curr_parent)

    best_indices = [0] * len(x_values)
    best_indices[-1] = int(np.argmin(dp_costs[-1]))
    for i in range(len(x_values) - 1, 0, -1):
        best_indices[i - 1] = int(parents[i][best_indices[i]])

    traced_by_curve: dict[str, list[tuple[int, int]]] = {name: [] for name in curve_names}
    for x, state_idx, states in zip(x_values, best_indices, states_by_x):
        ys = states[state_idx][0]
        for curve_idx, name in enumerate(curve_names):
            y_val = ys[curve_idx]
            if y_val is not None:
                traced_by_curve[name].append((x, y_val))

    return traced_by_curve


def _enforce_monotone_pixels(
    points: list[tuple[int, int]],
    curve_direction: CurveDirection = "downward",
) -> list[tuple[int, int]]:
    """Enforce monotone pixel trend by curve direction."""
    if not points:
        return points
    monotone: list[tuple[int, int]] = []
    if curve_direction == "downward":
        max_y = points[0][1]
        for px, py in points:
            if py < max_y:
                py = max_y
            else:
                max_y = py
            monotone.append((px, py))
    else:
        min_y = points[0][1]
        for px, py in points:
            if py > min_y:
                py = min_y
            else:
                min_y = py
            monotone.append((px, py))
    return monotone


def _fill_small_gaps(
    points: list[tuple[int, int]],
    gap_threshold: int,
    curve_direction: CurveDirection = "downward",
) -> list[tuple[int, int]]:
    """Fill small x-gaps by interpolation to keep curve continuity."""
    if len(points) < 2:
        return points

    filled: list[tuple[int, int]] = []
    for i, (px, py) in enumerate(points):
        filled.append((px, py))
        if i >= len(points) - 1:
            continue
        next_px, next_py = points[i + 1]
        gap = next_px - px
        if 1 < gap <= gap_threshold:
            for fill_x in range(px + 1, next_px):
                ratio = (fill_x - px) / gap
                fill_y = int(py + ratio * (next_py - py))
                # Keep monotone trend in pixel space.
                if curve_direction == "downward":
                    fill_y = max(fill_y, py)
                else:
                    fill_y = min(fill_y, py)
                filled.append((fill_x, fill_y))

    return filled


def resolve_overlaps(
    raw_curves: dict[str, list[tuple[int, int]]],
    mapping: AxisMapping,
    image: NDArray[Any] | None = None,
    curve_color_priors: dict[str, tuple[float, float, float] | None] | None = None,
    crossing_relaxed: bool = False,
    curve_direction: CurveDirection = "downward",
) -> dict[str, list[tuple[int, int]]]:
    """Resolve overlaps by jointly tracing all curves with identity priors."""
    clean: dict[str, list[tuple[int, int]]] = {}
    x0, _, x1, _ = mapping.plot_region
    x_range = x1 - x0
    gap_threshold = max(1, int(x_range * 0.05))  # 5% of x-range
    max_fallback_gap_px = max(3, int(x_range * 0.01))  # 1% of x-range

    # Build global x-grid from union of all curve pixels.
    x_union: set[int] = set()
    for pixels in raw_curves.values():
        for px, _ in pixels:
            x_union.add(px)
    x_values = sorted(x_union)

    traced_by_curve = _trace_curves_joint(
        raw_curves=raw_curves,
        x_values=x_values,
        image=image,
        curve_color_priors=curve_color_priors,
        max_fallback_gap_px=max_fallback_gap_px,
        crossing_relaxed=crossing_relaxed,
        curve_direction=curve_direction,
    )

    for name, traced in traced_by_curve.items():
        if not traced:
            clean[name] = []
            continue
        monotone = _enforce_monotone_pixels(traced, curve_direction=curve_direction)
        clean[name] = _fill_small_gaps(monotone, gap_threshold, curve_direction=curve_direction)

    return clean
