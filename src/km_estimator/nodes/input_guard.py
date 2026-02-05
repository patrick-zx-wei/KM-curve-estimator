"""Input guard node for validating KM curve images."""

from km_estimator.models import PipelineState, ProcessingError, ProcessingStage
from km_estimator.utils.gemini_client import validate_image


def input_guard(state: PipelineState) -> PipelineState:
    """
    Validate that the image is a proper Kaplan-Meier survival curve.

    Uses LLM to check for required elements (axes, curves, ticks).
    Retries up to max_input_guard_retries times on failure.

    Updates state with:
    - validation_result: ValidationResult from LLM
    - input_guard_retries: Number of retries used
    - errors: Any processing errors encountered
    """
    cfg = state.config

    # Use preprocessed image if available, otherwise original
    image_path = state.preprocessed_image_path or state.image_path

    # Retry only on API errors, not on invalid images
    for attempt in range(1, cfg.max_input_guard_retries + 1):
        result = validate_image(
            image_path,
            timeout=cfg.api_timeout_seconds,
            max_retries=cfg.api_max_retries,
        )

        if isinstance(result, ProcessingError):
            if attempt == cfg.max_input_guard_retries:
                return state.model_copy(update={
                    "input_guard_retries": attempt,
                    "errors": state.errors + [result],
                })
            continue  # Retry on API error

        # Got a ValidationResult - don't retry, trust the LLM's judgment
        if result.valid:
            return state.model_copy(update={
                "validation_result": result,
                "input_guard_retries": attempt,
            })

        # Image is not a valid KM curve - fail immediately, don't retry
        return state.model_copy(update={
            "validation_result": result,
            "input_guard_retries": attempt,
            "errors": state.errors + [ProcessingError(
                stage=ProcessingStage.INPUT_GUARD,
                error_type="invalid_image",
                recoverable=False,
                message=f"Not a valid KM curve: {result.feedback}",
                details={
                    "axes_present": result.axes_present,
                    "curves_present": result.curves_present,
                    "ticks_readable": result.ticks_readable,
                },
            )],
        })

    # All retries exhausted due to API errors
    return state.model_copy(update={
        "input_guard_retries": cfg.max_input_guard_retries,
        "errors": state.errors + [ProcessingError(
            stage=ProcessingStage.INPUT_GUARD,
            error_type="max_retries_exceeded",
            recoverable=False,
            message="Validation failed after max retries",
        )],
    })
