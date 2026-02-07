"""Curve digitization pipeline."""

from bisect import bisect_right
from itertools import permutations

import numpy as np

from km_estimator.models import PipelineState, ProcessingError, ProcessingStage
from km_estimator.utils import cv_utils

from .axis_calibration import (
    AxisMapping,
    calibrate_axes,
    calculate_anchors_from_risk_table,
    validate_against_anchors,
    validate_axis_bounds,
    validate_axis_config,
)
from .censoring_detection import detect_censoring
from .curve_isolation import (
    _all_curves_have_distinct_colors,
    _assign_by_expected_color,
    _coverage_issue,
    _extract_curve_mask,
    isolate_curves,
    parse_curve_color,
)
from .overlap_resolution import enforce_step_function, resolve_overlaps

MAX_IDENTITY_PERMUTATION_CURVES = 7
IDENTITY_REASSIGN_MIN_IMPROVEMENT = 0.35
RESCUE_LOCAL_MARGIN = 0.10
RESCUE_GLOBAL_MARGIN = 0.15
CATASTROPHIC_CURVE_SCORE = 3.0
CATASTROPHIC_IMPROVEMENT_MARGIN = 0.6
DUAL_PATH_AMBIGUITY_THRESHOLD = 0.35
DUAL_PATH_SELECTION_MARGIN = 0.12
DUAL_PATH_REAL_SCORE_WEIGHT = 1.8
OVERLAP_COLLAPSE_PX = 2


def _resolved_curve_direction(raw_direction: str | None) -> str:
    """Normalize direction metadata to supported values."""
    if raw_direction in ("downward", "upward"):
        return raw_direction
    return "downward"


def _curve_pixel_trend(points: list[tuple[int, int]]) -> float:
    """Estimate y-trend over x for one pixel curve (positive means down in image)."""
    if len(points) < 5:
        return 0.0
    ordered = sorted((int(x), int(y)) for x, y in points)
    x_to_y: dict[int, list[int]] = {}
    for x, y in ordered:
        x_to_y.setdefault(x, []).append(y)
    xs = sorted(x_to_y.keys())
    if len(xs) < 4:
        return 0.0
    start_x = xs[max(0, len(xs) // 8)]
    end_x = xs[min(len(xs) - 1, len(xs) - 1 - len(xs) // 8)]
    if end_x <= start_x:
        return 0.0
    start_y = float(np.median(x_to_y[start_x]))
    end_y = float(np.median(x_to_y[end_x]))
    return end_y - start_y


def _resolve_runtime_curve_direction(
    metadata_direction: str,
    raw_curves: dict[str, list[tuple[int, int]]],
) -> str:
    """
    Resolve curve direction from metadata + observed pixel trend.

    In image coordinates: downward survival trends to larger y over time.
    """
    deltas = [
        _curve_pixel_trend(points)
        for points in raw_curves.values()
        if len(points) >= 5
    ]
    if not deltas:
        return metadata_direction
    arr = np.asarray(deltas, dtype=np.float32)
    median_delta = float(np.median(arr))
    observed = "downward" if median_delta >= 0.0 else "upward"
    if metadata_direction == observed:
        return metadata_direction

    magnitudes = np.abs(arr)
    noise_floor = max(0.35, float(np.percentile(magnitudes, 30)) * 0.8)
    if arr.size == 1:
        return observed if float(magnitudes[0]) >= noise_floor else metadata_direction

    observed_sign = 1.0 if observed == "downward" else -1.0
    strong_observed = np.count_nonzero((arr * observed_sign) >= noise_floor)
    strong_opposed = np.count_nonzero((arr * observed_sign) <= -noise_floor)
    support_ratio = strong_observed / float(arr.size)
    oppose_ratio = strong_opposed / float(arr.size)

    if support_ratio >= 0.67 and support_ratio >= oppose_ratio + 0.25:
        return observed
    if abs(median_delta) >= max(0.7, noise_floor * 1.4):
        return observed
    return metadata_direction


def _to_survival_space(
    coords: list[tuple[float, float]],
    y_start: float,
    y_end: float,
    curve_direction: str,
) -> list[tuple[float, float]]:
    """
    Convert plotted curve coordinates to survival space.

    The downstream reconstruction stack assumes survival probabilities in [0, 1].
    For upward cumulative-incidence plots, convert using survival = 1 - incidence.
    """
    if curve_direction != "upward" or not coords:
        return coords

    y_abs_max = float(max(abs(y_start), abs(y_end)))
    percent_scale = 100.0 if (1.5 < y_abs_max <= 100.5) else 1.0

    reflected: list[tuple[float, float]] = []
    for t, y in coords:
        incidence = float(y) / percent_scale
        y_surv = float(np.clip(1.0 - incidence, 0.0, 1.0))
        reflected.append((float(t), y_surv))
    return reflected


def _validate_curves_not_empty(
    raw_curves: dict[str, list[tuple[int, int]]],
    min_points: int = 5,
) -> tuple[list[str], list[str]]:
    """
    Check for empty or near-empty curves.

    Returns:
        Tuple of (warnings, empty_curve_names)
    """
    warnings: list[str] = []
    empty_names: list[str] = []

    for name, pixels in raw_curves.items():
        if len(pixels) < min_points:
            empty_names.append(name)
            warnings.append(
                f"Curve '{name}' has only {len(pixels)} pixels (minimum: {min_points})"
            )

    return warnings, empty_names


def _validate_curve_shape(
    curves: dict[str, list[tuple[float, float]]],
) -> list[str]:
    """
    Validate curves have expected KM shape (not flat, generally decreasing).

    Returns:
        List of warning messages
    """
    warnings: list[str] = []

    for name, points in curves.items():
        if len(points) < 5:
            continue

        y_values = [p[1] for p in points]

        # Check for flat curves (artifact from failed isolation)
        unique_y = len(set(round(y, 3) for y in y_values))
        if unique_y < 3:
            warnings.append(
                f"Curve '{name}' appears flat with only {unique_y} distinct y-values"
            )

        # Check survival is generally decreasing (KM curves shouldn't increase overall)
        if len(y_values) >= 2 and y_values[0] < y_values[-1] - 0.05:
            warnings.append(
                f"Curve '{name}' survival increases from "
                f"{y_values[0]:.3f} to {y_values[-1]:.3f} (invalid)"
            )

    return warnings


def _anchor_lower_bound(
    anchor_points: list[tuple[float, float]],
    t: float,
) -> float:
    """Interpolate anchor lower bound at time t."""
    if not anchor_points:
        return 0.0

    points = sorted(anchor_points, key=lambda p: p[0])
    times = [p[0] for p in points]
    values = [p[1] for p in points]

    if t <= times[0]:
        return values[0]
    if t >= times[-1]:
        return values[-1]

    idx = bisect_right(times, t)
    t0, s0 = times[idx - 1], values[idx - 1]
    t1, s1 = times[idx], values[idx]
    if t1 == t0:
        return min(s0, s1)

    ratio = (t - t0) / (t1 - t0)
    return float(s0 + ratio * (s1 - s0))


def _apply_anchor_constraints(
    digitized_curves: dict[str, list[tuple[float, float]]],
    anchors: dict[str, list[tuple[float, float]]],
    tolerance: float = 0.01,
) -> tuple[dict[str, list[tuple[float, float]]], list[str]]:
    """
    Enforce anchor lower bounds directly on digitized curves.

    Curves are projected to satisfy:
    - survival(t) >= interpolated_anchor(t) - tolerance
    - survival is monotonically non-increasing over time
    """
    adjusted_curves: dict[str, list[tuple[float, float]]] = {}
    warnings: list[str] = []

    for curve_name, points in digitized_curves.items():
        anchor_points = anchors.get(curve_name)
        if not points or not anchor_points:
            adjusted_curves[curve_name] = points
            continue

        adjusted: list[list[float]] = []
        changed = 0
        for t, s in points:
            lower = _anchor_lower_bound(anchor_points, t) - tolerance
            new_s = max(s, lower)
            new_s = min(max(new_s, 0.0), 1.0)
            if abs(new_s - s) > 1e-6:
                changed += 1
            adjusted.append([t, new_s])

        # Enforce monotone non-increasing survival after anchor projection.
        for i in range(len(adjusted) - 2, -1, -1):
            if adjusted[i][1] < adjusted[i + 1][1]:
                adjusted[i][1] = adjusted[i + 1][1]

        adjusted_curves[curve_name] = [(float(t), float(s)) for t, s in adjusted]
        if changed > 0:
            warnings.append(
                f"{curve_name}: anchor constraints adjusted {changed}/{len(points)} points"
            )

    return adjusted_curves, warnings


def _restore_km_origin(
    curves: dict[str, list[tuple[float, float]]],
    x_start: float,
    x_end: float,
    y_max: float,
    start_tolerance_ratio: float = 0.05,
    min_origin_gap: float = 0.02,
) -> tuple[dict[str, list[tuple[float, float]]], list[str]]:
    """
    Restore KM origin when early curve starts implausibly low due digitization loss.

    KM curves should begin near the maximum survival level at study start.
    """
    updated: dict[str, list[tuple[float, float]]] = {}
    warnings: list[str] = []
    x_range = max(1e-6, x_end - x_start)
    start_window = x_start + x_range * start_tolerance_ratio

    for curve_name, points in curves.items():
        if not points:
            updated[curve_name] = points
            continue

        ordered = sorted(points, key=lambda p: p[0])
        t0, s0 = ordered[0]
        if t0 <= start_window and s0 < (y_max - min_origin_gap):
            injected = [(x_start, y_max)]
            if t0 > x_start:
                injected.append((t0, y_max))
            injected.extend(ordered)
            updated[curve_name] = enforce_step_function(injected)
            warnings.append(
                f"{curve_name}: restored KM origin from {s0:.3f} to {y_max:.3f} at study start"
            )
        else:
            updated[curve_name] = ordered

    return updated, warnings


def _anchor_violation_ratio(
    points: list[tuple[float, float]],
    anchor_points: list[tuple[float, float]] | None,
    tolerance: float = 0.01,
) -> float:
    """Fraction of points violating anchor lower bound."""
    if not points or not anchor_points:
        return 0.0
    violations = 0
    for t, s in points:
        if s + 1e-9 < (_anchor_lower_bound(anchor_points, t) - tolerance):
            violations += 1
    return violations / max(1, len(points))


def _curve_rescue_score(
    curve_points: list[tuple[float, float]],
    y_max: float,
    anchor_points: list[tuple[float, float]] | None,
) -> float:
    """
    Lower is better.

    Penalizes collapse/flatness, low start, and anchor violations.
    """
    if not curve_points:
        return 1e9

    ys = [s for _, s in curve_points]
    unique_y = len(set(round(v, 3) for v in ys))
    tail = ys[-120:] if len(ys) > 120 else ys
    tail_unique = len(set(round(v, 3) for v in tail))
    start_gap = max(0.0, y_max - ys[0])
    anchor_viol = _anchor_violation_ratio(curve_points, anchor_points)

    score = 0.0
    if unique_y < 3:
        score += 3.0
    if tail_unique < 2:
        score += 2.0
    score += min(2.0, start_gap * 5.0)
    score += anchor_viol * 4.0
    return score


def _global_curve_set_score(
    curves: dict[str, list[tuple[float, float]]],
    anchors: dict[str, list[tuple[float, float]]],
    y_max: float,
) -> float:
    """Aggregate multi-curve rescue score (lower is better)."""
    return sum(
        _curve_rescue_score(points, y_max=y_max, anchor_points=anchors.get(name))
        for name, points in curves.items()
    )


def _curve_overlap_ambiguity(
    curves: dict[str, list[tuple[int, int]]],
    mapping: AxisMapping,
) -> float:
    """
    Estimate overlap ambiguity in [0,1] from coverage gaps and column collisions.

    Higher values indicate higher risk that one identity track collapses in overlaps.
    """
    if not curves:
        return 1.0

    x0, _, x1, _ = mapping.plot_region
    n_curves = max(1, len(curves))
    coverage_issues = sum(
        1 for points in curves.values() if _coverage_issue(points, x0, x1)
    ) / n_curves

    x_to_curve_y: dict[int, list[int]] = {}
    for points in curves.values():
        by_x: dict[int, list[int]] = {}
        for px, py in points:
            by_x.setdefault(int(px), []).append(int(py))
        for px, ys in by_x.items():
            x_to_curve_y.setdefault(px, []).append(int(round(float(np.median(ys)))))

    overlap_cols = 0
    collapsed_cols = 0
    for ys in x_to_curve_y.values():
        if len(ys) < 2:
            continue
        overlap_cols += 1
        min_sep = min(abs(a - b) for i, a in enumerate(ys) for b in ys[i + 1 :])
        if min_sep <= OVERLAP_COLLAPSE_PX:
            collapsed_cols += 1

    overlap_ratio = overlap_cols / max(1, len(x_to_curve_y))
    collapse_ratio = collapsed_cols / max(1, overlap_cols)

    score = 0.45 * coverage_issues + 0.3 * overlap_ratio + 0.25 * collapse_ratio
    return float(np.clip(score, 0.0, 1.0))


def _pixel_curve_set_score(
    curves: dict[str, list[tuple[int, int]]],
    mapping: AxisMapping,
) -> float:
    """Lower-is-better score for selecting between overlap-tracing strategies."""
    x0, _, x1, _ = mapping.plot_region
    x_span = max(1, x1 - x0)
    score = 0.0

    for points in curves.values():
        if not points:
            score += 8.0
            continue
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        min_x = min(xs)
        max_x = max(xs)
        coverage_ratio = (max_x - min_x) / x_span
        score += max(0.0, 0.85 - coverage_ratio) * 6.0
        score += max(0.0, (min_x - x0) / x_span) * 3.0
        score += max(0.0, (x1 - max_x) / x_span) * 3.0

        unique_y = len(set(ys))
        tail = ys[-80:] if len(ys) > 80 else ys
        tail_unique = len(set(tail))
        if unique_y < 3:
            score += 2.5
        if tail_unique < 2:
            score += 1.5

    x_to_curve_y: dict[int, list[int]] = {}
    for points in curves.values():
        by_x: dict[int, list[int]] = {}
        for px, py in points:
            by_x.setdefault(int(px), []).append(int(py))
        for px, ys in by_x.items():
            x_to_curve_y.setdefault(px, []).append(int(round(float(np.median(ys)))))

    for ys in x_to_curve_y.values():
        if len(ys) < 2:
            continue
        min_sep = min(abs(a - b) for i, a in enumerate(ys) for b in ys[i + 1 :])
        if min_sep <= OVERLAP_COLLAPSE_PX:
            score += (OVERLAP_COLLAPSE_PX - min_sep + 1) * 0.2

    return float(score)


def _project_pixel_curves_to_survival(
    pixel_curves: dict[str, list[tuple[int, int]]],
    mapping: AxisMapping,
    curve_direction: str,
    y_start: float,
    y_end: float,
) -> dict[str, list[tuple[float, float]]]:
    """Project pixel curves into survival-space coordinates for quality scoring."""
    projected: dict[str, list[tuple[float, float]]] = {}
    for name, pixels in pixel_curves.items():
        real_coords = [mapping.px_to_real(px, py) for px, py in pixels]
        directed_coords = enforce_step_function(real_coords, direction=curve_direction)
        survival_coords = _to_survival_space(
            directed_coords,
            y_start=y_start,
            y_end=y_end,
            curve_direction=curve_direction,
        )
        projected[name] = enforce_step_function(survival_coords, direction="downward")
    return projected


def _overlap_candidate_score(
    pixel_curves: dict[str, list[tuple[int, int]]],
    mapping: AxisMapping,
    image: np.ndarray,
    curve_order: list[str],
    expected_colors: dict[str, tuple[float, float, float] | None],
    anchors: dict[str, list[tuple[float, float]]],
    curve_direction: str,
    x_start: float,
    x_end: float,
    y_start: float,
    y_end: float,
) -> tuple[float, float, float]:
    """
    Composite candidate score for overlap-path selection.

    Returns (composite_score, pixel_score, real_score); lower is better.
    """
    pixel_score = _pixel_curve_set_score(pixel_curves, mapping)
    projected = _project_pixel_curves_to_survival(
        pixel_curves,
        mapping=mapping,
        curve_direction=curve_direction,
        y_start=y_start,
        y_end=y_end,
    )
    projected, _, _ = _optimize_curve_identity_assignment(
        digitized_curves=projected,
        pixel_curves=pixel_curves,
        image=image,
        curve_order=curve_order,
        expected_colors=expected_colors,
        anchors=anchors,
        y_max=y_end,
    )
    projected, _, _ = _postprocess_digitized_curves(
        projected,
        anchors=anchors,
        x_start=x_start,
        x_end=x_end,
        y_start=y_start,
        y_max=y_end,
    )
    real_score = _global_curve_set_score(projected, anchors, y_max=y_end)
    composite = pixel_score + DUAL_PATH_REAL_SCORE_WEIGHT * real_score
    return float(composite), float(pixel_score), float(real_score)


def _sample_curve_rgb(
    image: np.ndarray,
    pixel_points: list[tuple[int, int]],
    max_samples: int = 160,
) -> tuple[float, float, float] | None:
    """Sample representative RGB color from a traced pixel curve."""
    if image.size == 0 or not pixel_points:
        return None

    h, w = image.shape[:2]
    step = max(1, len(pixel_points) // max_samples)
    samples: list[tuple[float, float, float]] = []
    for px, py in pixel_points[::step]:
        cx = min(max(int(px), 0), w - 1)
        cy = min(max(int(py), 0), h - 1)
        b, g, r = image[cy, cx]
        samples.append((float(r) / 255.0, float(g) / 255.0, float(b) / 255.0))

    if not samples:
        return None

    arr = np.asarray(samples, dtype=np.float32)
    median_rgb = np.median(arr, axis=0)
    return float(median_rgb[0]), float(median_rgb[1]), float(median_rgb[2])


def _identity_pair_score(
    curve_points: list[tuple[float, float]],
    sampled_rgb: tuple[float, float, float] | None,
    expected_rgb: tuple[float, float, float] | None,
    anchor_points: list[tuple[float, float]] | None,
    y_max: float,
) -> float:
    """Per-pair identity score combining color + anchors + continuity."""
    score = _curve_rescue_score(
        curve_points,
        y_max=y_max,
        anchor_points=anchor_points,
    ) * 0.7
    if expected_rgb is None:
        return score
    if sampled_rgb is None:
        return score + 1.0

    color_dist = float(np.sqrt(sum((a - b) ** 2 for a, b in zip(sampled_rgb, expected_rgb))))
    return score + color_dist * 2.8


def _optimize_curve_identity_assignment(
    digitized_curves: dict[str, list[tuple[float, float]]],
    pixel_curves: dict[str, list[tuple[int, int]]],
    image: np.ndarray,
    curve_order: list[str],
    expected_colors: dict[str, tuple[float, float, float] | None],
    anchors: dict[str, list[tuple[float, float]]],
    y_max: float,
    min_improvement: float = IDENTITY_REASSIGN_MIN_IMPROVEMENT,
) -> tuple[
    dict[str, list[tuple[float, float]]],
    dict[str, list[tuple[int, int]]],
    list[str],
]:
    """
    Globally optimize curve identity assignment over all curves via permutation search.

    Uses color fit and anchor/continuity quality to avoid one-curve-collapse identity failures.
    """
    expected_names = [name for name in curve_order if name in digitized_curves]
    track_names = [name for name in digitized_curves if name in expected_names]

    if (
        len(expected_names) < 2
        or len(track_names) != len(expected_names)
        or len(expected_names) > MAX_IDENTITY_PERMUTATION_CURVES
    ):
        return digitized_curves, pixel_curves, []

    track_colors = {
        name: _sample_curve_rgb(image, pixel_curves.get(name, []))
        for name in track_names
    }

    baseline_assignment: dict[str, str] = {}
    for expected in expected_names:
        if expected in track_names:
            baseline_assignment[expected] = expected
    remaining = [name for name in track_names if name not in baseline_assignment.values()]
    for expected in expected_names:
        if expected not in baseline_assignment and remaining:
            baseline_assignment[expected] = remaining.pop(0)

    def assignment_score(assignment: dict[str, str]) -> float:
        return sum(
            _identity_pair_score(
                digitized_curves.get(track_name, []),
                track_colors.get(track_name),
                expected_colors.get(expected_name),
                anchors.get(expected_name),
                y_max=y_max,
            )
            for expected_name, track_name in assignment.items()
        )

    baseline_score = assignment_score(baseline_assignment)
    best_assignment = dict(baseline_assignment)
    best_score = baseline_score

    for perm in permutations(track_names, len(expected_names)):
        assignment = dict(zip(expected_names, perm))
        score = assignment_score(assignment)
        if score < best_score:
            best_score = score
            best_assignment = assignment

    if best_assignment == baseline_assignment or best_score + min_improvement >= baseline_score:
        return digitized_curves, pixel_curves, []

    remapped_digitized: dict[str, list[tuple[float, float]]] = {}
    remapped_pixels: dict[str, list[tuple[int, int]]] = {}
    for expected_name in expected_names:
        source_name = best_assignment[expected_name]
        remapped_digitized[expected_name] = digitized_curves.get(source_name, [])
        remapped_pixels[expected_name] = pixel_curves.get(source_name, [])

    for name, points in digitized_curves.items():
        if name not in remapped_digitized:
            remapped_digitized[name] = points
    for name, points in pixel_curves.items():
        if name not in remapped_pixels:
            remapped_pixels[name] = points

    warnings = [
        "Applied global identity reassignment "
        f"(score {baseline_score:.2f} -> {best_score:.2f})"
    ]
    return remapped_digitized, remapped_pixels, warnings


def _identify_rescue_candidates(
    digitized: dict[str, list[tuple[float, float]]],
    anchors: dict[str, list[tuple[float, float]]],
    y_min: float,
    y_max: float,
) -> set[str]:
    """Find per-curve rescue candidates based on collapse and anchor inconsistency."""
    flagged: set[str] = set()
    for curve_name, points in digitized.items():
        if not points:
            flagged.add(curve_name)
            continue
        ys = [s for _, s in points]
        unique_y = len(set(round(v, 3) for v in ys))
        tail = ys[-120:] if len(ys) > 120 else ys
        tail_unique = len(set(round(v, 3) for v in tail))
        if unique_y < 3 or tail_unique < 2:
            flagged.add(curve_name)
            continue
        # Large single-step drops are usually identity/collapse artifacts.
        if any((ys[i] - ys[i + 1]) > 0.18 for i in range(len(ys) - 1)):
            flagged.add(curve_name)
            continue
        y_range = max(1e-6, y_max - y_min)
        low_start_margin = max(0.02, y_range * 0.12)
        if ys[0] < (y_max - low_start_margin):
            flagged.add(curve_name)
            continue
        if _anchor_violation_ratio(points, anchors.get(curve_name)) > 0.05:
            flagged.add(curve_name)
    return flagged


def _postprocess_digitized_curves(
    digitized_curves: dict[str, list[tuple[float, float]]],
    anchors: dict[str, list[tuple[float, float]]],
    x_start: float,
    x_end: float,
    y_start: float,
    y_max: float,
) -> tuple[dict[str, list[tuple[float, float]]], list[str], list[str]]:
    """
    Apply anchor constraints + KM origin restoration + step normalization.

    Returns:
        (curves, anchor_constraint_warnings, origin_warnings)
    """
    curves = dict(digitized_curves)
    anchor_constraint_warnings: list[str] = []
    if anchors:
        curves, anchor_constraint_warnings = _apply_anchor_constraints(curves, anchors)
        for curve_name, curve_points in curves.items():
            curves[curve_name] = enforce_step_function(curve_points)

    origin_warnings: list[str] = []
    curves, origin_warnings = _restore_km_origin(
        curves,
        x_start=x_start,
        x_end=x_end,
        y_max=y_max,
    )
    for curve_name, curve_points in curves.items():
        curves[curve_name] = enforce_step_function(curve_points)

    # Tail stabilization and range completion:
    # 1) If a curve ends early but still above baseline, extend horizontal tail to x_end.
    # 2) If anchor exists at x_end, prevent implausible terminal collapse below anchor floor.
    tail_warnings: list[str] = []
    x_range = max(1e-6, x_end - x_start)
    min_tail_extension_gap = x_range * 0.04
    for curve_name, curve_points in list(curves.items()):
        if not curve_points:
            continue
        ordered = sorted(curve_points, key=lambda p: p[0])
        last_t, last_s = ordered[-1]
        baseline_margin = max(0.01, (y_max - y_start) * 0.02)

        if last_t < x_end - min_tail_extension_gap and last_s > (y_start + baseline_margin):
            ordered.append((x_end, last_s))
            tail_warnings.append(
                f"{curve_name}: extended tail to x_end={x_end:.2f} at survival {last_s:.3f}"
            )

        anchor_points = anchors.get(curve_name)
        if anchor_points:
            floor = _anchor_lower_bound(anchor_points, x_end) - 0.02
            floor = min(max(floor, y_start), y_max)
            if ordered[-1][1] < floor:
                ordered[-1] = (ordered[-1][0], floor)
                if ordered[-1][0] < x_end:
                    ordered.append((x_end, floor))
                tail_warnings.append(
                    f"{curve_name}: lifted terminal tail floor to anchor-compatible {floor:.3f}"
                )

        curves[curve_name] = enforce_step_function(ordered)

    origin_warnings.extend(tail_warnings)
    return curves, anchor_constraint_warnings, origin_warnings


def digitize(state: PipelineState) -> PipelineState:
    """
    Digitize curves from preprocessed image using MMPU metadata.

    Steps:
    1. Axis calibration (pixel↔unit mapping)
    2. Curve isolation (k-medoids clustering)
    3. Overlap resolution (graph-based clean traces)
    4. Censoring detection (+ marks)
    5. Convert to real coordinates and enforce KM step shape
    6. Apply risk-table anchor constraints (if available)
    """
    if state.plot_metadata is None:
        return state.model_copy(
            update={
                "errors": state.errors
                + [
                    ProcessingError(
                        stage=ProcessingStage.DIGITIZE,
                        error_type="no_metadata",
                        recoverable=False,
                        message="PlotMetadata required for digitization",
                    )
                ]
            }
        )

    image_path = state.preprocessed_image_path or state.image_path
    image = cv_utils.load_image(image_path, stage=ProcessingStage.DIGITIZE)
    if isinstance(image, ProcessingError):
        return state.model_copy(update={"errors": state.errors + [image]})

    # Step 1: Axis calibration
    mapping = calibrate_axes(image, state.plot_metadata)
    if isinstance(mapping, ProcessingError):
        return state.model_copy(update={"errors": state.errors + [mapping]})

    curve_direction = _resolved_curve_direction(
        getattr(state.plot_metadata, "curve_direction", "downward")
    )

    # Step 1b: Validate axis configurations
    axis_config_warnings = validate_axis_config(state.plot_metadata.x_axis, "x_axis")
    axis_config_warnings.extend(validate_axis_config(state.plot_metadata.y_axis, "y_axis"))
    curve_names = [c.name for c in state.plot_metadata.curves]
    anchors = calculate_anchors_from_risk_table(
        state.plot_metadata.risk_table, curve_names
    )
    expected_colors = {
        curve.name: parse_curve_color(curve.color_description)
        for curve in state.plot_metadata.curves
    }

    # Step 2: Curve isolation
    raw_curves = isolate_curves(image, state.plot_metadata, mapping)
    if isinstance(raw_curves, ProcessingError):
        return state.model_copy(update={"errors": state.errors + [raw_curves]})

    # Step 2b: Validate no empty curves
    empty_warnings, empty_names = _validate_curves_not_empty(raw_curves)
    if empty_names:
        # Remove empty curves but continue with valid ones
        raw_curves = {k: v for k, v in raw_curves.items() if k not in empty_names}
        if not raw_curves:
            return state.model_copy(
                update={
                    "errors": state.errors + [
                        ProcessingError(
                            stage=ProcessingStage.DIGITIZE,
                            error_type="all_curves_empty",
                            recoverable=True,
                            message=f"All curves are empty: {empty_names}",
                            details={"empty_curves": empty_names},
                        )
                    ]
                }
            )

    curve_direction_runtime = _resolve_runtime_curve_direction(curve_direction, raw_curves)
    effective_y_start = state.plot_metadata.y_axis.start
    effective_y_end = state.plot_metadata.y_axis.end
    if curve_direction_runtime == "upward":
        effective_y_start = 0.0
        effective_y_end = 1.0
    direction_warnings: list[str] = []
    if curve_direction_runtime != curve_direction:
        direction_warnings.append(
            "Digitization direction overridden from "
            f"{curve_direction} to {curve_direction_runtime} by pixel trend"
        )

    dual_path_warnings: list[str] = []
    ambiguity_score = _curve_overlap_ambiguity(raw_curves, mapping)

    # Step 3: Clean up overlaps with joint tracing and color identity priors.
    clean_curves = resolve_overlaps(
        raw_curves,
        mapping,
        image=image,
        curve_color_priors=expected_colors,
        crossing_relaxed=False,
        curve_direction=curve_direction_runtime,
    )
    has_color_priors = any(color is not None for color in expected_colors.values())
    if ambiguity_score >= DUAL_PATH_AMBIGUITY_THRESHOLD:
        candidate_paths: list[tuple[str, dict[str, list[tuple[int, int]]], float]] = []
        baseline_name = "color-prior" if has_color_priors else "default"
        baseline_score = _pixel_curve_set_score(clean_curves, mapping)
        candidate_paths.append((baseline_name, clean_curves, baseline_score))

        if has_color_priors:
            neutral_clean_curves = resolve_overlaps(
                raw_curves,
                mapping,
                image=image,
                curve_color_priors=None,
                crossing_relaxed=False,
                curve_direction=curve_direction_runtime,
            )
            neutral_score = _pixel_curve_set_score(neutral_clean_curves, mapping)
            candidate_paths.append(("neutral", neutral_clean_curves, neutral_score))

        crossing_clean_curves = resolve_overlaps(
            raw_curves,
            mapping,
            image=image,
            curve_color_priors=expected_colors if has_color_priors else None,
            crossing_relaxed=True,
            curve_direction=curve_direction_runtime,
        )
        crossing_score = _pixel_curve_set_score(crossing_clean_curves, mapping)
        candidate_paths.append(("crossing-relaxed", crossing_clean_curves, crossing_score))

        candidate_composite: dict[str, tuple[float, float, float]] = {}
        for name, curves_candidate, _ in candidate_paths:
            candidate_composite[name] = _overlap_candidate_score(
                curves_candidate,
                mapping=mapping,
                image=image,
                curve_order=curve_names,
                expected_colors=expected_colors,
                anchors=anchors,
                curve_direction=curve_direction_runtime,
                x_start=state.plot_metadata.x_axis.start,
                x_end=state.plot_metadata.x_axis.end,
                y_start=effective_y_start,
                y_end=effective_y_end,
            )

        best_name = min(candidate_paths, key=lambda item: candidate_composite[item[0]][0])[0]
        best_curves = next(curves for name, curves, _ in candidate_paths if name == best_name)
        baseline_composite = candidate_composite[baseline_name][0]
        best_composite = candidate_composite[best_name][0]

        if best_name != baseline_name and (
            best_composite + DUAL_PATH_SELECTION_MARGIN < baseline_composite
        ):
            clean_curves = best_curves
            best_pixel, best_real = candidate_composite[best_name][1], candidate_composite[best_name][2]
            base_pixel, base_real = candidate_composite[baseline_name][1], candidate_composite[baseline_name][2]
            dual_path_warnings.append(
                "Ambiguous overlap: selected "
                f"{best_name} tracing (composite {baseline_composite:.2f} -> {best_composite:.2f}; "
                f"pixel {base_pixel:.2f}->{best_pixel:.2f}, real {base_real:.2f}->{best_real:.2f})"
            )
        else:
            score_summary = ", ".join(
                (
                    f"{name}=C{candidate_composite[name][0]:.2f}"
                    f"/P{candidate_composite[name][1]:.2f}"
                    f"/R{candidate_composite[name][2]:.2f}"
                )
                for name, _, _ in candidate_paths
            )
            dual_path_warnings.append(
                f"Ambiguous overlap: kept {baseline_name} tracing "
                f"({score_summary})"
            )

    # Step 4: Convert to real coordinates
    digitized: dict[str, list[tuple[float, float]]] = {}
    for name, pixels in clean_curves.items():
        real_coords = [mapping.px_to_real(px, py) for px, py in pixels]
        directed_coords = enforce_step_function(real_coords, direction=curve_direction_runtime)
        survival_coords = _to_survival_space(
            directed_coords,
            y_start=state.plot_metadata.y_axis.start,
            y_end=state.plot_metadata.y_axis.end,
            curve_direction=curve_direction_runtime,
        )
        # Reconstruction expects survival-style (decreasing) curves.
        digitized[name] = enforce_step_function(survival_coords, direction="downward")

    digitized, clean_curves, identity_warnings = _optimize_curve_identity_assignment(
        digitized_curves=digitized,
        pixel_curves=clean_curves,
        image=image,
        curve_order=curve_names,
        expected_colors=expected_colors,
        anchors=anchors,
        y_max=effective_y_end,
    )

    # Step 5: Censoring detection
    censoring = detect_censoring(image, clean_curves, mapping)

    # Step 6: Postprocess curves with anchors + origin constraints
    digitized, anchor_constraint_warnings, origin_warnings = _postprocess_digitized_curves(
        digitized,
        anchors=anchors,
        x_start=state.plot_metadata.x_axis.start,
        x_end=state.plot_metadata.x_axis.end,
        y_start=effective_y_start,
        y_max=effective_y_end,
    )

    rescue_warnings: list[str] = []
    rescue_candidates = _identify_rescue_candidates(
        digitized,
        anchors,
        y_min=effective_y_start,
        y_max=effective_y_end,
    )

    # Per-curve rescue: only replace flagged curves when color-guided path is better.
    if rescue_candidates:
        x0, y0, x1, y1 = mapping.plot_region
        roi = image[y0:y1, x0:x1]
        roi_area = max(1, roi.shape[0] * roi.shape[1])
        min_pixels = max(5, int(roi_area * 0.00005))
        mask = _extract_curve_mask(roi, min_pixels)
        ys, xs = (mask > 0).nonzero()
        all_colors_ok, named_colors = _all_curves_have_distinct_colors(state.plot_metadata)

        if all_colors_ok and len(xs) >= len(state.plot_metadata.curves):
            color_raw = _assign_by_expected_color(roi, xs, ys, named_colors, x0, y0)
            color_clean = resolve_overlaps(
                color_raw,
                mapping,
                image=image,
                curve_color_priors=expected_colors,
                curve_direction=curve_direction_runtime,
            )
            color_digitized: dict[str, list[tuple[float, float]]] = {}
            for name, pixels in color_clean.items():
                real_coords = [mapping.px_to_real(px, py) for px, py in pixels]
                directed_coords = enforce_step_function(
                    real_coords,
                    direction=curve_direction_runtime,
                )
                survival_coords = _to_survival_space(
                    directed_coords,
                    y_start=state.plot_metadata.y_axis.start,
                    y_end=state.plot_metadata.y_axis.end,
                    curve_direction=curve_direction_runtime,
                )
                color_digitized[name] = enforce_step_function(
                    survival_coords,
                    direction="downward",
                )
            color_digitized, _, _ = _optimize_curve_identity_assignment(
                digitized_curves=color_digitized,
                pixel_curves=color_clean,
                image=image,
                curve_order=curve_names,
                expected_colors=expected_colors,
                anchors=anchors,
                y_max=effective_y_end,
            )
            color_digitized, _, _ = _postprocess_digitized_curves(
                color_digitized,
                anchors=anchors,
                x_start=state.plot_metadata.x_axis.start,
                x_end=state.plot_metadata.x_axis.end,
                y_start=effective_y_start,
                y_max=effective_y_end,
            )

            proposed_digitized = dict(digitized)
            proposed_swaps: list[tuple[str, float, float]] = []
            for curve_name in sorted(rescue_candidates):
                base_points = digitized.get(curve_name, [])
                alt_points = color_digitized.get(curve_name, [])
                if not alt_points:
                    continue
                base_score = _curve_rescue_score(
                    base_points,
                    y_max=effective_y_end,
                    anchor_points=anchors.get(curve_name),
                )
                alt_score = _curve_rescue_score(
                    alt_points,
                    y_max=effective_y_end,
                    anchor_points=anchors.get(curve_name),
                )
                if alt_score + RESCUE_LOCAL_MARGIN < base_score:
                    proposed_digitized[curve_name] = alt_points
                    proposed_swaps.append((curve_name, base_score, alt_score))

            if proposed_swaps:
                base_global_score = _global_curve_set_score(
                    digitized, anchors, y_max=effective_y_end
                )
                proposed_global_score = _global_curve_set_score(
                    proposed_digitized,
                    anchors,
                    y_max=effective_y_end,
                )
                catastrophic_recovery = any(
                    base_score >= CATASTROPHIC_CURVE_SCORE
                    and (base_score - alt_score) >= CATASTROPHIC_IMPROVEMENT_MARGIN
                    for _, base_score, alt_score in proposed_swaps
                )
                accept_by_global = proposed_global_score + RESCUE_GLOBAL_MARGIN < base_global_score
                allow_catastrophic = catastrophic_recovery and (
                    proposed_global_score <= base_global_score + 0.10
                )

                if accept_by_global or allow_catastrophic:
                    digitized = proposed_digitized
                    for curve_name, base_score, alt_score in proposed_swaps:
                        rescue_warnings.append(
                            f"{curve_name}: applied color-guided rescue "
                            f"(score {base_score:.2f} -> {alt_score:.2f})"
                        )
                    if allow_catastrophic and not accept_by_global:
                        rescue_warnings.append(
                            "Rescue accepted for catastrophic curve recovery "
                            f"(global {base_global_score:.2f} -> {proposed_global_score:.2f})"
                        )
                    else:
                        rescue_warnings.append(
                            "Rescue accepted by global score gate "
                            f"({base_global_score:.2f} -> {proposed_global_score:.2f})"
                        )
                else:
                    rescue_warnings.append(
                        "Skipped rescue: global score did not improve enough "
                        f"({base_global_score:.2f} -> {proposed_global_score:.2f})"
                    )

    # Step 7: Validate curve shapes (not flat, generally decreasing)
    validation_warnings: list[str] = (
        list(axis_config_warnings)
        + list(empty_warnings)
        + list(direction_warnings)
        + list(dual_path_warnings)
        + list(identity_warnings)
        + list(anchor_constraint_warnings)
        + list(origin_warnings)
        + list(rescue_warnings)
    )
    shape_warnings = _validate_curve_shape(digitized)
    validation_warnings.extend(shape_warnings)

    # Step 8: Validate against anchors from risk table (if available)
    if anchors:
        anchor_warnings = validate_against_anchors(digitized, anchors)
        validation_warnings.extend(anchor_warnings)

    # Step 9: Validate against axis bounds
    bounds_axis = state.plot_metadata.y_axis
    if curve_direction_runtime == "upward":
        bounds_axis = bounds_axis.model_copy(update={"start": 0.0, "end": 1.0})
    bounds_warnings = validate_axis_bounds(digitized, bounds_axis)
    validation_warnings.extend(bounds_warnings)
    if curve_direction_runtime == "upward":
        validation_warnings.append(
            "Converted upward cumulative-incidence curves into survival space before reconstruction"
        )

    # Combine with existing warnings
    all_warnings = list(state.mmpu_warnings) + validation_warnings

    return state.model_copy(
        update={
            "digitized_curves": digitized,
            "censoring_marks": censoring,
            "mmpu_warnings": all_warnings,
        }
    )


__all__ = ["digitize", "AxisMapping"]
