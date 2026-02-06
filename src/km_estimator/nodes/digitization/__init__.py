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
from .curve_isolation import isolate_curves, parse_curve_color
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
    anchor_constraint_warnings: list[str] = []
    if anchors:
        digitized, anchor_constraint_warnings = _apply_anchor_constraints(digitized, anchors)
        for curve_name, curve_points in digitized.items():
            digitized[curve_name] = enforce_step_function(curve_points)

    # Step 6: Validate curve shapes (not flat, generally decreasing)
    validation_warnings: list[str] = (
        list(axis_config_warnings)
        + list(empty_warnings)
        + list(anchor_constraint_warnings)
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
