"""Curve digitization pipeline."""

from bisect import bisect_right

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
    _extract_curve_mask,
    isolate_curves,
    parse_curve_color,
)
from .overlap_resolution import enforce_step_function, resolve_overlaps


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


def _identify_rescue_candidates(
    digitized: dict[str, list[tuple[float, float]]],
    anchors: dict[str, list[tuple[float, float]]],
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
        if ys[0] < (y_max - 0.12):
            flagged.add(curve_name)
            continue
        if _anchor_violation_ratio(points, anchors.get(curve_name)) > 0.08:
            flagged.add(curve_name)
    return flagged


def _postprocess_digitized_curves(
    digitized_curves: dict[str, list[tuple[float, float]]],
    anchors: dict[str, list[tuple[float, float]]],
    x_start: float,
    x_end: float,
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

    # Step 1b: Validate axis configurations
    axis_config_warnings = validate_axis_config(state.plot_metadata.x_axis, "x_axis")
    axis_config_warnings.extend(validate_axis_config(state.plot_metadata.y_axis, "y_axis"))

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

    # Step 3: Clean up overlaps with joint tracing and color identity priors.
    color_priors = {
        curve.name: parse_curve_color(curve.color_description)
        for curve in state.plot_metadata.curves
    }
    clean_curves = resolve_overlaps(
        raw_curves,
        mapping,
        image=image,
        curve_color_priors=color_priors,
    )

    # Step 4: Censoring detection
    censoring = detect_censoring(image, clean_curves, mapping)

    # Step 5: Convert to real coordinates
    digitized: dict[str, list[tuple[float, float]]] = {}
    for name, pixels in clean_curves.items():
        real_coords = [mapping.px_to_real(px, py) for px, py in pixels]
        # Step 5b: Enforce proper step function format
        digitized[name] = enforce_step_function(real_coords)

    curve_names = [c.name for c in state.plot_metadata.curves]
    anchors = calculate_anchors_from_risk_table(
        state.plot_metadata.risk_table, curve_names
    )
    digitized, anchor_constraint_warnings, origin_warnings = _postprocess_digitized_curves(
        digitized,
        anchors=anchors,
        x_start=state.plot_metadata.x_axis.start,
        x_end=state.plot_metadata.x_axis.end,
        y_max=state.plot_metadata.y_axis.end,
    )

    rescue_warnings: list[str] = []
    rescue_candidates = _identify_rescue_candidates(
        digitized,
        anchors,
        y_max=state.plot_metadata.y_axis.end,
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
            color_priors = {
                curve.name: parse_curve_color(curve.color_description)
                for curve in state.plot_metadata.curves
            }
            color_clean = resolve_overlaps(
                color_raw,
                mapping,
                image=image,
                curve_color_priors=color_priors,
            )
            color_digitized: dict[str, list[tuple[float, float]]] = {}
            for name, pixels in color_clean.items():
                color_digitized[name] = enforce_step_function(
                    [mapping.px_to_real(px, py) for px, py in pixels]
                )
            color_digitized, _, _ = _postprocess_digitized_curves(
                color_digitized,
                anchors=anchors,
                x_start=state.plot_metadata.x_axis.start,
                x_end=state.plot_metadata.x_axis.end,
                y_max=state.plot_metadata.y_axis.end,
            )

            for curve_name in sorted(rescue_candidates):
                base_points = digitized.get(curve_name, [])
                alt_points = color_digitized.get(curve_name, [])
                if not alt_points:
                    continue
                base_score = _curve_rescue_score(
                    base_points,
                    y_max=state.plot_metadata.y_axis.end,
                    anchor_points=anchors.get(curve_name),
                )
                alt_score = _curve_rescue_score(
                    alt_points,
                    y_max=state.plot_metadata.y_axis.end,
                    anchor_points=anchors.get(curve_name),
                )
                if alt_score + 0.4 < base_score:
                    digitized[curve_name] = alt_points
                    rescue_warnings.append(
                        f"{curve_name}: applied color-guided rescue "
                        f"(score {base_score:.2f} -> {alt_score:.2f})"
                    )

    # Step 6: Validate curve shapes (not flat, generally decreasing)
    validation_warnings: list[str] = (
        list(axis_config_warnings)
        + list(empty_warnings)
        + list(anchor_constraint_warnings)
        + list(origin_warnings)
        + list(rescue_warnings)
    )
    shape_warnings = _validate_curve_shape(digitized)
    validation_warnings.extend(shape_warnings)

    # Step 7: Validate against anchors from risk table (if available)
    if anchors:
        anchor_warnings = validate_against_anchors(digitized, anchors)
        validation_warnings.extend(anchor_warnings)

    # Step 8: Validate against axis bounds
    bounds_warnings = validate_axis_bounds(digitized, state.plot_metadata.y_axis)
    validation_warnings.extend(bounds_warnings)

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
