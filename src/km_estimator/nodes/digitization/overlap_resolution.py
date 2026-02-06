"""Overlap resolution with joint curve tracing and identity priors."""

from __future__ import annotations

from bisect import bisect_left
from itertools import product
from typing import Any

import cv2
import numpy as np
from numpy.typing import NDArray

from .axis_calibration import AxisMapping

MAX_CANDIDATES_PER_CURVE = 3
MAX_STATES_PER_X = 128
MAX_STATES_PER_X_HIGH_AMBIGUITY = 320
MAX_CANDIDATES_PER_CURVE_AMBIGUOUS = 9
MAX_CANDIDATES_PER_CURVE_MODERATE = 5
MIN_DIVERSE_STATES_PER_SIGNATURE = 2
MAX_DIVERSE_STATES_PER_SIGNATURE = 8

MEDIAN_ATTRACTION_WEIGHT = 0.08
MISSING_COLUMN_PENALTY = 0.45
MISSING_TRANSITION_PENALTY = 0.22
COLOR_PRIOR_WEIGHT = 0.8
COLOR_PATCH_RADIUS = 2
MIN_COLOR_WEIGHT_FACTOR = 0.2
SATURATION_CONFIDENCE_SCALE = 80.0

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


def enforce_step_function(
    points: list[tuple[float, float]],
    time_tolerance: float = 0.001,
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

    # Build step function: for each drop, add horizontal then vertical
    # Track minimum survival seen so far (KM curves are monotonically non-increasing)
    step_points: list[tuple[float, float]] = [deduped[0]]
    min_survival = deduped[0][1]

    for i in range(1, len(deduped)):
        curr_t, curr_s = deduped[i]
        prev_t, prev_s = step_points[-1]

        # Clamp any survival increase to previous level (noise correction)
        if curr_s > min_survival:
            curr_s = min_survival

        # If survival dropped, add the step shape
        if curr_s < prev_s:
            # Horizontal segment: carry previous survival to current time
            step_points.append((curr_t, prev_s))
            # Vertical drop: drop to current survival at current time
            step_points.append((curr_t, curr_s))
            min_survival = curr_s
        else:
            # Survival stayed same - extend horizontal
            step_points.append((curr_t, curr_s))

    return step_points


def _select_y_candidates(
    ys: list[int],
    x: int,
    color_maps: tuple[NDArray[np.float32], NDArray[np.float32]] | None,
    expected_lab: tuple[float, float, float] | None,
    max_candidates: int = MAX_CANDIDATES_PER_CURVE,
) -> list[int]:
    """Select representative y candidates for one x-column."""
    arr = np.asarray(ys, dtype=np.float32)
    unique, counts = np.unique(arr.astype(np.int32), return_counts=True)
    if unique.size <= max_candidates:
        return unique.tolist()

    median_y = float(np.median(arr))
    spread = float(np.percentile(arr, 90) - np.percentile(arr, 10))
    spread_scale = max(1.0, spread)

    density_scores = counts.astype(np.float32)
    if density_scores.max() > 0:
        density_scores /= density_scores.max()

    centrality_scores = 1.0 / (1.0 + (np.abs(unique - median_y) / spread_scale))

    color_scores = np.ones_like(density_scores, dtype=np.float32)
    if color_maps is not None and expected_lab is not None:
        for idx, y_val in enumerate(unique):
            lab, sat_confidence = _sample_color_prior(color_maps, x, int(y_val))
            color_dist = float(np.sqrt(sum((a - b) ** 2 for a, b in zip(lab, expected_lab))))
            color_norm = color_dist / LAB_DISTANCE_NORMALIZER
            # Higher is better for ranking.
            color_scores[idx] = 1.0 / (
                1.0
                + color_norm
                * (MIN_COLOR_WEIGHT_FACTOR + (1.0 - MIN_COLOR_WEIGHT_FACTOR) * sat_confidence)
            )

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
) -> float:
    """Estimate ambiguity level for one x-column on one curve in [0, 1]."""
    if not ys:
        return 0.4

    arr = np.asarray(ys, dtype=np.float32)
    unique = np.unique(arr.astype(np.int32))
    unique_count = int(unique.size)
    spread = float(np.percentile(arr, 90) - np.percentile(arr, 10)) if arr.size > 1 else 0.0
    density = int(arr.size)

    score = 0.0
    if unique_count >= AMBIGUITY_UNIQUE_Y_THRESHOLD:
        score += min(0.45, (unique_count - AMBIGUITY_UNIQUE_Y_THRESHOLD + 1) * 0.08)
    if spread >= AMBIGUITY_SPREAD_PX_THRESHOLD:
        score += min(0.35, (spread - AMBIGUITY_SPREAD_PX_THRESHOLD + 1.0) * 0.03)
    if density >= AMBIGUITY_DENSITY_THRESHOLD:
        score += min(0.2, (density - AMBIGUITY_DENSITY_THRESHOLD + 1) * 0.02)

    if color_maps is not None and expected_lab is not None and unique_count > 1:
        dists = []
        for y_val in unique[: min(unique_count, 16)]:
            lab, sat_confidence = _sample_color_prior(color_maps, x, int(y_val))
            color_dist = float(np.sqrt(sum((a - b) ** 2 for a, b in zip(lab, expected_lab))))
            color_norm = color_dist / LAB_DISTANCE_NORMALIZER
            weighted = color_norm * (
                MIN_COLOR_WEIGHT_FACTOR + (1.0 - MIN_COLOR_WEIGHT_FACTOR) * sat_confidence
            )
            dists.append(weighted)
        if dists:
            dists_arr = np.asarray(dists, dtype=np.float32)
            # If color separation between nearby branches is weak, ambiguity is higher.
            if dists_arr.size >= 2:
                sorted_d = np.sort(dists_arr)
                if float(sorted_d[1] - sorted_d[0]) < 0.04:
                    score += 0.2

    return float(np.clip(score, 0.0, 1.0))


def _adaptive_candidate_limit(ambiguity: float) -> int:
    """Scale per-curve candidate count by local ambiguity."""
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


def _transition_cost(prev_y: int, curr_y: int, x_gap: int) -> float:
    """
    Transition penalty between adjacent x-columns.

    The graph is left-to-right in x, with strong penalty for y decreases
    (which would imply survival increasing in KM semantics).
    """
    upward_move = max(0, prev_y - curr_y - ALLOWED_UPWARD_MOVE_PX)
    smooth = abs(curr_y - prev_y) / max(1, x_gap)
    return upward_move * UPWARD_MOVE_PENALTY + smooth * SMOOTHNESS_WEIGHT


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


def _curve_candidates_for_x(
    x_map: dict[int, list[int]],
    sorted_keys: list[int],
    x: int,
    color_maps: tuple[NDArray[np.float32], NDArray[np.float32]] | None,
    expected_lab: tuple[float, float, float] | None,
    max_fallback_gap_px: int,
) -> tuple[list[int | None], list[float], float]:
    """Generate y-candidates and local costs for one curve at one x."""
    ys = x_map.get(x)
    if ys:
        median_y = float(np.median(ys))
        ambiguity = _column_ambiguity(ys, x, color_maps, expected_lab)
        candidate_limit = _adaptive_candidate_limit(ambiguity)
        candidates = _select_y_candidates(
            ys,
            x=x,
            color_maps=color_maps,
            expected_lab=expected_lab,
            max_candidates=candidate_limit,
        )
        base_costs = [
            abs(c - median_y) * MEDIAN_ATTRACTION_WEIGHT
            for c in candidates
        ]
    else:
        nearest_x = _find_nearest_x(sorted_keys, x)
        if nearest_x is None or abs(nearest_x - x) > max_fallback_gap_px:
            # No reliable nearby evidence for this x-column.
            return [None], [MISSING_COLUMN_PENALTY * 2.0], 0.35
        nearest_gap = abs(nearest_x - x)
        gap_ratio = nearest_gap / max(1, max_fallback_gap_px)
        nearest_ys = x_map[nearest_x]
        nearest_y = int(np.median(nearest_ys))
        in_span = bool(sorted_keys) and sorted_keys[0] <= x <= sorted_keys[-1]
        candidates = [nearest_y, None]
        base_costs = [
            MISSING_COLUMN_PENALTY * (1.0 + gap_ratio),
            MISSING_COLUMN_PENALTY * (1.3 + 0.4 * gap_ratio)
            + (COVERAGE_MISSING_STATE_PENALTY if in_span else 0.0),
        ]
        ambiguity = 0.6 if in_span else 0.35

    if color_maps is None or expected_lab is None:
        return candidates, base_costs, ambiguity

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
        costs.append(base + color_dist * color_weight)
    return candidates, costs, ambiguity


def _state_identity_penalty(
    x: int,
    ys: tuple[int | None, ...],
    curve_names: list[str],
    sorted_keys: dict[str, list[int]],
    x_maps: dict[str, dict[int, list[int]]],
) -> float:
    """Coverage-aware identity penalty to reduce swaps/truncation in overlaps."""
    penalty = 0.0
    for idx, name in enumerate(curve_names):
        y = ys[idx]
        keys = sorted_keys[name]
        if not keys:
            continue
        min_x = keys[0]
        max_x = keys[-1]
        in_span = min_x <= x <= max_x

        if y is None:
            if in_span:
                penalty += COVERAGE_MISSING_STATE_PENALTY
            continue

        nearest_x = _find_nearest_x(keys, x)
        if nearest_x is None:
            continue
        ref_y = int(np.median(x_maps[name][nearest_x]))
        x_gap = abs(nearest_x - x)
        penalty += IDENTITY_DRIFT_WEIGHT * (abs(y - ref_y) / max(1, x_gap + 1))
    return penalty


def _trace_curves_joint(
    raw_curves: dict[str, list[tuple[int, int]]],
    x_values: list[int],
    image: NDArray[Any] | None,
    curve_color_priors: dict[str, tuple[float, float, float] | None] | None,
    max_fallback_gap_px: int,
) -> dict[str, list[tuple[int, int]]]:
    """Jointly trace all curves with state exclusivity and identity priors."""
    if not x_values:
        return {name: [] for name in raw_curves}

    curve_names = list(raw_curves.keys())
    x_maps, _ = _build_curve_x_maps(raw_curves)
    sorted_keys = {name: sorted(x_maps[name].keys()) for name in curve_names}
    color_maps = _build_color_prior_maps(image) if image is not None else None
    expected_labs: dict[str, tuple[float, float, float] | None] = {
        name: (
            _rgb01_to_lab(curve_color_priors[name])
            if curve_color_priors and curve_color_priors.get(name) is not None
            else None
        )
        for name in curve_names
    }

    states_by_x: list[list[tuple[tuple[int | None, ...], float]]] = []
    for x in x_values:
        curve_candidates: list[list[int | None]] = []
        curve_costs: list[list[float]] = []
        curve_ambiguities: list[float] = []
        for name in curve_names:
            expected_lab = expected_labs.get(name)
            candidates, costs, ambiguity = _curve_candidates_for_x(
                x_maps[name],
                sorted_keys[name],
                x,
                color_maps,
                expected_lab,
                max_fallback_gap_px,
            )
            curve_candidates.append(candidates)
            curve_costs.append(costs)
            curve_ambiguities.append(ambiguity)

        combos = product(*(range(len(cands)) for cands in curve_candidates))
        states: list[tuple[tuple[int | None, ...], float]] = []
        for combo in combos:
            ys = tuple(curve_candidates[i][choice] for i, choice in enumerate(combo))
            cost = sum(curve_costs[i][choice] for i, choice in enumerate(combo))
            cost += _state_overlap_penalty(ys)
            cost += _state_identity_penalty(x, ys, curve_names, sorted_keys, x_maps)
            states.append((ys, float(cost)))

        x_ambiguity = float(np.mean(curve_ambiguities)) if curve_ambiguities else 0.0
        beam_size = _adaptive_state_beam(x_ambiguity, len(curve_names))
        states_by_x.append(_prune_states_with_identity_diversity(states, beam_size))

    dp_costs: list[np.ndarray] = []
    parents: list[np.ndarray] = []

    first_costs = np.asarray([cost for _, cost in states_by_x[0]], dtype=np.float32)
    dp_costs.append(first_costs)
    parents.append(np.full(len(first_costs), -1, dtype=np.int32))

    for i in range(1, len(x_values)):
        prev_states = states_by_x[i - 1]
        curr_states = states_by_x[i]
        prev_cost = dp_costs[i - 1]
        curr_cost = np.full(len(curr_states), np.inf, dtype=np.float32)
        curr_parent = np.full(len(curr_states), -1, dtype=np.int32)
        x_gap = x_values[i] - x_values[i - 1]

        for curr_idx, (curr_y, curr_data_cost) in enumerate(curr_states):
            best_cost = np.inf
            best_parent = -1
            for prev_idx, (prev_y, _) in enumerate(prev_states):
                transition = 0.0
                for curve_idx in range(len(curve_names)):
                    prev_val = prev_y[curve_idx]
                    curr_val = curr_y[curve_idx]
                    if prev_val is None and curr_val is None:
                        transition += MISSING_TRANSITION_PENALTY * 0.5
                        continue
                    if prev_val is None or curr_val is None:
                        transition += MISSING_TRANSITION_PENALTY
                        continue
                    transition += _transition_cost(prev_val, curr_val, x_gap)
                transition += _swap_penalty(prev_y, curr_y)

                total = prev_cost[prev_idx] + transition
                if total < best_cost:
                    best_cost = total
                    best_parent = prev_idx
            curr_cost[curr_idx] = best_cost + curr_data_cost
            curr_parent[curr_idx] = best_parent

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


def _enforce_monotone_pixels(points: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Enforce monotone non-decreasing y in pixel coordinates."""
    if not points:
        return points
    monotone: list[tuple[int, int]] = []
    max_y = points[0][1]
    for px, py in points:
        if py < max_y:
            py = max_y
        else:
            max_y = py
        monotone.append((px, py))
    return monotone


def _fill_small_gaps(
    points: list[tuple[int, int]],
    gap_threshold: int,
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
                # Keep monotone in pixel space.
                fill_y = max(fill_y, py)
                filled.append((fill_x, fill_y))

    return filled


def resolve_overlaps(
    raw_curves: dict[str, list[tuple[int, int]]],
    mapping: AxisMapping,
    image: NDArray[Any] | None = None,
    curve_color_priors: dict[str, tuple[float, float, float] | None] | None = None,
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
    )

    for name, traced in traced_by_curve.items():
        if not traced:
            clean[name] = []
            continue
        monotone = _enforce_monotone_pixels(traced)
        clean[name] = _fill_small_gaps(monotone, gap_threshold)

    return clean
