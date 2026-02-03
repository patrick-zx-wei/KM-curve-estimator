"""MMPU (Multi-Model Processing Unit) node for LLM-based extraction."""

from km_estimator.models import PipelineState, ProcessingError, ProcessingStage
from km_estimator.utils.gemini_client import extract_metadata_dual, extract_ocr_dual


def mmpu(state: PipelineState) -> PipelineState:
    """
    Extract OCR tokens and plot metadata using dual-model convergence.

    Stage 1: OCR extraction (axis labels, tick labels, legend, risk table text)
    Stage 2: Structured analysis (axis configs, curve info, risk table parsing)

    Updates state with:
    - ocr_tokens: RawOCRTokens from Stage 1
    - plot_metadata: PlotMetadata from Stage 2
    - mmpu_retries: Total retries across both stages
    - errors: Any processing errors encountered
    """
    cfg = state.config

    # Use preprocessed image if available, otherwise original
    image_path = state.preprocessed_image_path or state.image_path

    warnings: list[str] = []
    total_attempts = 0

    # Stage 1: OCR Extraction
    ocr_result = extract_ocr_dual(
        image_path,
        max_attempts=cfg.max_mmpu_retries,
        timeout=cfg.api_timeout_seconds,
        single_model_mode=cfg.single_model_mode,
    )

    total_attempts += ocr_result.attempts
    warnings.extend(ocr_result.warnings)

    if ocr_result.error is not None:
        return state.model_copy(
            update={
                "mmpu_retries": total_attempts,
                "errors": state.errors + [ocr_result.error],
            }
        )

    if ocr_result.result is None:
        error = ProcessingError(
            stage=ProcessingStage.MMPU,
            error_type="ocr_failed",
            recoverable=False,
            message="OCR extraction returned no result",
        )
        return state.model_copy(
            update={
                "mmpu_retries": total_attempts,
                "errors": state.errors + [error],
            }
        )

    ocr_tokens = ocr_result.result

    if not ocr_result.converged:
        warnings.append("OCR stage did not converge between models")

    # Stage 2: Structured Analysis
    metadata_result = extract_metadata_dual(
        image_path,
        ocr_tokens,
        max_attempts=cfg.max_mmpu_retries,
        timeout=cfg.api_timeout_seconds,
        single_model_mode=cfg.single_model_mode,
    )

    total_attempts += metadata_result.attempts
    warnings.extend(metadata_result.warnings)

    if metadata_result.error is not None:
        # Still save OCR tokens even if metadata fails
        return state.model_copy(
            update={
                "ocr_tokens": ocr_tokens,
                "mmpu_retries": total_attempts,
                "errors": state.errors + [metadata_result.error],
            }
        )

    if metadata_result.result is None:
        error = ProcessingError(
            stage=ProcessingStage.MMPU,
            error_type="metadata_failed",
            recoverable=False,
            message="Metadata extraction returned no result",
        )
        return state.model_copy(
            update={
                "ocr_tokens": ocr_tokens,
                "mmpu_retries": total_attempts,
                "errors": state.errors + [error],
            }
        )

    plot_metadata = metadata_result.result

    if not metadata_result.converged:
        warnings.append("Metadata stage did not converge between models")

    # Validate: at least one curve must be detected
    if len(plot_metadata.curves) == 0:
        error = ProcessingError(
            stage=ProcessingStage.MMPU,
            error_type="zero_curves",
            recoverable=False,
            message="No curves detected in the image",
            details={"warnings": warnings},
        )
        return state.model_copy(
            update={
                "ocr_tokens": ocr_tokens,
                "plot_metadata": plot_metadata,
                "mmpu_retries": total_attempts,
                "errors": state.errors + [error],
            }
        )

    return state.model_copy(
        update={
            "ocr_tokens": ocr_tokens,
            "plot_metadata": plot_metadata,
            "mmpu_retries": total_attempts,
            "mmpu_warnings": warnings,
        }
    )
