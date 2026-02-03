"""Curve digitization pipeline."""

from km_estimator.models import PipelineState, ProcessingError, ProcessingStage
from km_estimator.utils import cv_utils

from .axis_calibration import AxisMapping, calibrate_axes
from .censoring_detection import detect_censoring
from .curve_isolation import isolate_curves
from .overlap_resolution import resolve_overlaps


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

    # Step 2: Curve isolation
    raw_curves = isolate_curves(image, state.plot_metadata, mapping)
    if isinstance(raw_curves, ProcessingError):
        return state.model_copy(update={"errors": state.errors + [raw_curves]})

    # Step 3: Clean up overlaps
    clean_curves = resolve_overlaps(raw_curves, mapping)

    # Step 4: Censoring detection
    censoring = detect_censoring(image, clean_curves, mapping)

    # Step 5: Convert to real coordinates
    digitized: dict[str, list[tuple[float, float]]] = {}
    for name, pixels in clean_curves.items():
        digitized[name] = [mapping.px_to_real(px, py) for px, py in pixels]

    return state.model_copy(
        update={
            "digitized_curves": digitized,
            "censoring_marks": censoring,
        }
    )


__all__ = ["digitize", "AxisMapping"]
