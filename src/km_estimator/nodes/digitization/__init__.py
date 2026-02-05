"""Curve digitization pipeline."""

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
from .curve_isolation import isolate_curves
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
            warnings.append(f"Curve '{name}' has only {len(pixels)} pixels (minimum: {min_points})")

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
                f"Curve '{name}' survival increases from {y_values[0]:.3f} to {y_values[-1]:.3f} (invalid)"
            )

    return warnings


def digitize(state: PipelineState) -> PipelineState:
    """
    Digitize curves from preprocessed image using MMPU metadata.

    Steps:
    1. Axis calibration (pixel↔unit mapping)
    2. Curve isolation (k-medoids clustering)
    3. Overlap resolution (clean traces)
    4. Censoring detection (+ marks)
    5. Convert to real coordinates
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

    # Step 3: Clean up overlaps
    clean_curves = resolve_overlaps(raw_curves, mapping)

    # Step 4: Censoring detection
    censoring = detect_censoring(image, clean_curves, mapping)

    # Step 5: Convert to real coordinates
    digitized: dict[str, list[tuple[float, float]]] = {}
    for name, pixels in clean_curves.items():
        real_coords = [mapping.px_to_real(px, py) for px, py in pixels]
        # Step 5b: Enforce proper step function format
        digitized[name] = enforce_step_function(real_coords)

    # Step 6: Validate curve shapes (not flat, generally decreasing)
    validation_warnings: list[str] = list(axis_config_warnings) + list(empty_warnings)
    shape_warnings = _validate_curve_shape(digitized)
    validation_warnings.extend(shape_warnings)

    # Step 7: Validate against anchors from risk table (if available)
    curve_names = [c.name for c in state.plot_metadata.curves]
    anchors = calculate_anchors_from_risk_table(
        state.plot_metadata.risk_table, curve_names
    )
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
