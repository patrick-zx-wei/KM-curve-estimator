"""MMPU node - tiered extraction with GPT-5 Mini primary, Gemini Flash verification."""

from km_estimator import config
from km_estimator.models import PipelineState, ProcessingError, ProcessingStage
from km_estimator.utils.tiered_extractor import (
    extract_metadata_tiered,
    extract_metadata_tiered_async,
    extract_ocr_tiered,
    extract_ocr_tiered_async,
)


def mmpu(state: PipelineState) -> PipelineState:
    """
    Extract OCR tokens and plot metadata using tiered extraction.

    Stage 1: OCR extraction (GPT-5 Mini primary, Gemini Flash verification)
    Stage 2: Structured analysis (GPT-5 Mini primary, Gemini Flash verification)

    Updates state with:
    - ocr_tokens: RawOCRTokens from Stage 1
    - plot_metadata: PlotMetadata from Stage 2
    - extraction_route: Route taken (gpt_only, gpt_verified, gemini_override, gemini_fallback)
    - gpt_confidence: GPT confidence score
    - verification_similarity: Similarity score when Gemini verification was used
    - flagged_for_review: True if GPT/Gemini disagreed
    - extraction_cost_usd: Estimated cost in USD
    - errors: Any processing errors encountered
    """
    cfg = state.config
    image_path = state.preprocessed_image_path or state.image_path
    warnings: list[str] = []
    total_cost = 0.0

    # Stage 1: Tiered OCR Extraction (with retry for recoverable errors)
    ocr_result = None
    for attempt in range(cfg.max_mmpu_retries + 1):
        ocr_result = extract_ocr_tiered(
            image_path,
            confidence_threshold=cfg.tiered_confidence_threshold,
            similarity_threshold=cfg.tiered_similarity_threshold,
            timeout=cfg.api_timeout_seconds,
            skip_verification=cfg.single_model_mode,
        )
        if ocr_result.error and ocr_result.error.recoverable:
            if attempt < cfg.max_mmpu_retries:
                warnings.append(f"OCR retry {attempt + 1}/{cfg.max_mmpu_retries}")
                continue
        break

    warnings.extend(ocr_result.warnings)
    if ocr_result.gpt_tokens:
        total_cost += _calculate_cost(ocr_result.gpt_tokens, ocr_result.gemini_used)

    if ocr_result.error:
        return state.model_copy(
            update={
                "errors": state.errors + [ocr_result.error],
                "extraction_route": ocr_result.route.value,
                "extraction_cost_usd": total_cost,
                "mmpu_warnings": warnings,
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
                "errors": state.errors + [error],
                "extraction_route": ocr_result.route.value,
                "extraction_cost_usd": total_cost,
                "mmpu_warnings": warnings,
            }
        )

    ocr_tokens = ocr_result.result

    # Stage 2: Tiered Metadata Extraction (with retry for recoverable errors)
    metadata_result = None
    for attempt in range(cfg.max_mmpu_retries + 1):
        metadata_result = extract_metadata_tiered(
            image_path,
            ocr_tokens,
            confidence_threshold=cfg.tiered_confidence_threshold,
            similarity_threshold=cfg.tiered_similarity_threshold,
            timeout=cfg.api_timeout_seconds,
            skip_verification=cfg.single_model_mode,
        )
        if metadata_result.error and metadata_result.error.recoverable:
            if attempt < cfg.max_mmpu_retries:
                warnings.append(f"Metadata retry {attempt + 1}/{cfg.max_mmpu_retries}")
                continue
        break

    warnings.extend(metadata_result.warnings)
    if metadata_result.gpt_tokens:
        total_cost += _calculate_cost(metadata_result.gpt_tokens, metadata_result.gemini_used)

    if metadata_result.error:
        return state.model_copy(
            update={
                "ocr_tokens": ocr_tokens,
                "errors": state.errors + [metadata_result.error],
                "extraction_route": metadata_result.route.value,
                "gpt_confidence": metadata_result.gpt_confidence,
                "extraction_cost_usd": total_cost,
                "mmpu_warnings": warnings,
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
                "errors": state.errors + [error],
                "extraction_route": metadata_result.route.value,
                "gpt_confidence": metadata_result.gpt_confidence,
                "extraction_cost_usd": total_cost,
                "mmpu_warnings": warnings,
            }
        )

    plot_metadata = metadata_result.result

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
                "errors": state.errors + [error],
                "extraction_route": metadata_result.route.value,
                "gpt_confidence": metadata_result.gpt_confidence,
                "extraction_cost_usd": total_cost,
                "mmpu_warnings": warnings,
            }
        )

    return state.model_copy(
        update={
            "ocr_tokens": ocr_tokens,
            "plot_metadata": plot_metadata,
            "extraction_route": metadata_result.route.value,
            "gpt_confidence": metadata_result.gpt_confidence,
            "verification_similarity": metadata_result.similarity_score,
            "flagged_for_review": ocr_result.flagged_for_review or metadata_result.flagged_for_review,
            "extraction_cost_usd": total_cost,
            "gpt_tokens_used": metadata_result.gpt_tokens,
            "mmpu_warnings": warnings,
        }
    )


def _calculate_cost(gpt_tokens: tuple[int, int], gemini_used: bool) -> float:
    """Calculate extraction cost in USD."""
    cost = (
        gpt_tokens[0] / 1000 * config.GPT5_MINI_COST_INPUT
        + gpt_tokens[1] / 1000 * config.GPT5_MINI_COST_OUTPUT
    )
    if gemini_used:
        # Estimate Gemini tokens (similar to GPT)
        cost += (
            gpt_tokens[0] / 1000 * config.GEMINI_FLASH_COST_INPUT
            + gpt_tokens[1] / 1000 * config.GEMINI_FLASH_COST_OUTPUT
        )
    return cost


async def mmpu_async(state: PipelineState) -> PipelineState:
    """
    Async: Extract OCR tokens and plot metadata using tiered extraction.

    Stage 1: OCR extraction (GPT-5 Mini primary, Gemini Flash verification)
    Stage 2: Structured analysis (GPT-5 Mini primary, Gemini Flash verification)

    Updates state with:
    - ocr_tokens: RawOCRTokens from Stage 1
    - plot_metadata: PlotMetadata from Stage 2
    - extraction_route: Route taken (gpt_only, gpt_verified, gemini_override, gemini_fallback)
    - gpt_confidence: GPT confidence score
    - verification_similarity: Similarity score when Gemini verification was used
    - flagged_for_review: True if GPT/Gemini disagreed
    - extraction_cost_usd: Estimated cost in USD
    - errors: Any processing errors encountered
    """
    cfg = state.config
    image_path = state.preprocessed_image_path or state.image_path
    warnings: list[str] = []
    total_cost = 0.0

    # Stage 1: Tiered OCR Extraction (with retry for recoverable errors)
    ocr_result = None
    for attempt in range(cfg.max_mmpu_retries + 1):
        ocr_result = await extract_ocr_tiered_async(
            image_path,
            confidence_threshold=cfg.tiered_confidence_threshold,
            similarity_threshold=cfg.tiered_similarity_threshold,
            timeout=cfg.api_timeout_seconds,
            skip_verification=cfg.single_model_mode,
        )
        if ocr_result.error and ocr_result.error.recoverable:
            if attempt < cfg.max_mmpu_retries:
                warnings.append(f"OCR retry {attempt + 1}/{cfg.max_mmpu_retries}")
                continue
        break

    warnings.extend(ocr_result.warnings)
    if ocr_result.gpt_tokens:
        total_cost += _calculate_cost(ocr_result.gpt_tokens, ocr_result.gemini_used)

    if ocr_result.error:
        return state.model_copy(
            update={
                "errors": state.errors + [ocr_result.error],
                "extraction_route": ocr_result.route.value,
                "extraction_cost_usd": total_cost,
                "mmpu_warnings": warnings,
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
                "errors": state.errors + [error],
                "extraction_route": ocr_result.route.value,
                "extraction_cost_usd": total_cost,
                "mmpu_warnings": warnings,
            }
        )

    ocr_tokens = ocr_result.result

    # Stage 2: Tiered Metadata Extraction (with retry for recoverable errors)
    metadata_result = None
    for attempt in range(cfg.max_mmpu_retries + 1):
        metadata_result = await extract_metadata_tiered_async(
            image_path,
            ocr_tokens,
            confidence_threshold=cfg.tiered_confidence_threshold,
            similarity_threshold=cfg.tiered_similarity_threshold,
            timeout=cfg.api_timeout_seconds,
            skip_verification=cfg.single_model_mode,
        )
        if metadata_result.error and metadata_result.error.recoverable:
            if attempt < cfg.max_mmpu_retries:
                warnings.append(f"Metadata retry {attempt + 1}/{cfg.max_mmpu_retries}")
                continue
        break

    warnings.extend(metadata_result.warnings)
    if metadata_result.gpt_tokens:
        total_cost += _calculate_cost(metadata_result.gpt_tokens, metadata_result.gemini_used)

    if metadata_result.error:
        return state.model_copy(
            update={
                "ocr_tokens": ocr_tokens,
                "errors": state.errors + [metadata_result.error],
                "extraction_route": metadata_result.route.value,
                "gpt_confidence": metadata_result.gpt_confidence,
                "extraction_cost_usd": total_cost,
                "mmpu_warnings": warnings,
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
                "errors": state.errors + [error],
                "extraction_route": metadata_result.route.value,
                "gpt_confidence": metadata_result.gpt_confidence,
                "extraction_cost_usd": total_cost,
                "mmpu_warnings": warnings,
            }
        )

    plot_metadata = metadata_result.result

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
                "errors": state.errors + [error],
                "extraction_route": metadata_result.route.value,
                "gpt_confidence": metadata_result.gpt_confidence,
                "extraction_cost_usd": total_cost,
                "mmpu_warnings": warnings,
            }
        )

    return state.model_copy(
        update={
            "ocr_tokens": ocr_tokens,
            "plot_metadata": plot_metadata,
            "extraction_route": metadata_result.route.value,
            "gpt_confidence": metadata_result.gpt_confidence,
            "verification_similarity": metadata_result.similarity_score,
            "flagged_for_review": ocr_result.flagged_for_review or metadata_result.flagged_for_review,
            "extraction_cost_usd": total_cost,
            "gpt_tokens_used": metadata_result.gpt_tokens,
            "mmpu_warnings": warnings,
        }
    )
