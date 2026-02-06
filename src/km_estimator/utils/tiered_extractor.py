"""Tiered extraction: GPT-5 Mini primary, Gemini Flash verification."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Generic, TypeVar

from pydantic import BaseModel

from km_estimator import config
from km_estimator.models import (
    AxisConfig,
    PlotMetadata,
    ProcessingError,
    RawOCRTokens,
)
from km_estimator.utils.gemini_client import (
    extract_metadata,
    extract_metadata_async,
    extract_ocr,
    extract_ocr_async,
)
from km_estimator.utils.openai_client import (
    GPTResult,
    extract_metadata_gpt,
    extract_metadata_gpt_async,
    extract_ocr_gpt,
    extract_ocr_gpt_async,
)

T = TypeVar("T", bound=BaseModel)


class ExtractionRoute(str, Enum):
    """Route taken during tiered extraction."""

    GPT_ONLY = "gpt_only"
    GPT_VERIFIED = "gpt_verified"
    GEMINI_OVERRIDE = "gemini_override"
    GEMINI_FALLBACK = "gemini_fallback"


@dataclass
class TieredResult(Generic[T]):
    """Result from tiered extraction with full tracking."""

    result: T | None
    error: ProcessingError | None
    route: ExtractionRoute
    gpt_confidence: float | None = None
    similarity_score: float | None = None
    gpt_tokens: tuple[int, int] | None = None
    gemini_used: bool = False
    warnings: list[str] = field(default_factory=list)
    flagged_for_review: bool = False


def _parse_numeric(s: str) -> float | None:
    """Try to parse a string as a number, return None if not numeric."""
    try:
        return float(s.replace(",", ""))  # Handle "1,000" format
    except ValueError:
        return None


def _normalize_text(s: str) -> str:
    """Normalize text for comparison: strip whitespace, lowercase."""
    return s.strip().lower()


def _tick_label_similarity(a: list[str], b: list[str]) -> float:
    """Compare tick labels with numeric tolerance for numbers, exact match for text."""
    if len(a) != len(b):
        # Different lengths - partial credit based on overlap
        if not a or not b:
            return 0.0
        # Compare what we can
        min_len = min(len(a), len(b))
        max_len = max(len(a), len(b))
        length_penalty = min_len / max_len
        a, b = a[:min_len], b[:min_len]
    else:
        length_penalty = 1.0

    if not a:
        return 1.0  # Both empty

    matches = 0
    for x, y in zip(a, b):
        x_num = _parse_numeric(x)
        y_num = _parse_numeric(y)

        if x_num is not None and y_num is not None:
            # Both numeric - use tolerance
            if abs(x_num - y_num) < config.FLOAT_TOLERANCE:
                matches += 1
        else:
            # Text comparison - normalized
            if _normalize_text(x) == _normalize_text(y):
                matches += 1

    return (matches / len(a)) * length_penalty


def _text_list_similarity(a: list[str], b: list[str]) -> float:
    """Compare text lists with normalization."""
    if len(a) != len(b):
        return 0.0
    if not a:
        return 1.0  # Both empty

    a_norm = [_normalize_text(s) for s in a]
    b_norm = [_normalize_text(s) for s in b]

    matches = sum(1 for x, y in zip(a_norm, b_norm) if x == y)
    return matches / len(a)


def _calculate_ocr_similarity(a: RawOCRTokens, b: RawOCRTokens) -> float:
    """Calculate similarity score between two OCR results.

    Uses type-aware comparison:
    - Tick labels: numeric tolerance for numbers, normalized text for strings
    - Axis/legend labels: normalized text comparison (strip, lowercase)
    - Proportional matching instead of all-or-nothing
    """
    # Weights for each field (must sum to 1.0)
    x_tick_weight = 0.35
    y_tick_weight = 0.35
    axis_label_weight = 0.25
    legend_weight = 0.05

    score = 0.0
    score += x_tick_weight * _tick_label_similarity(a.x_tick_labels, b.x_tick_labels)
    score += y_tick_weight * _tick_label_similarity(a.y_tick_labels, b.y_tick_labels)
    score += axis_label_weight * _text_list_similarity(a.axis_labels, b.axis_labels)
    score += legend_weight * _text_list_similarity(a.legend_labels, b.legend_labels)

    return score


def _axis_match(a: AxisConfig, b: AxisConfig) -> bool:
    """Check if two axis configs match within tolerance."""
    if len(a.tick_values) != len(b.tick_values):
        return False
    return (
        all(abs(x - y) < config.FLOAT_TOLERANCE for x, y in zip(a.tick_values, b.tick_values))
        and abs(a.start - b.start) < config.FLOAT_TOLERANCE
        and abs(a.end - b.end) < config.FLOAT_TOLERANCE
    )


def _calculate_metadata_similarity(a: PlotMetadata, b: PlotMetadata) -> float:
    """Calculate similarity score between two metadata results."""
    score = 0.0
    if _axis_match(a.x_axis, b.x_axis):
        score += 0.4
    if _axis_match(a.y_axis, b.y_axis):
        score += 0.4
    if len(a.curves) == len(b.curves):
        score += 0.2
    return score


def extract_ocr_tiered(
    path: str,
    confidence_threshold: float = config.TIERED_CONFIDENCE_THRESHOLD,
    similarity_threshold: float = config.TIERED_SIMILARITY_THRESHOLD,
    timeout: int = config.API_TIMEOUT_SECONDS,
    max_retries: int = config.API_MAX_RETRIES,
    skip_verification: bool = False,
) -> TieredResult[RawOCRTokens]:
    """Tiered OCR: GPT-5 Mini primary, Gemini Flash verification if low confidence."""
    warnings: list[str] = []

    # Step 1: GPT-5 Mini primary extraction
    gpt_result = extract_ocr_gpt(path, timeout=timeout, max_retries=max_retries)
    gpt_tokens = (gpt_result.input_tokens, gpt_result.output_tokens)

    if gpt_result.error:
        # GPT failed, fall back to Gemini (unless skip_verification is set)
        if skip_verification:
            return TieredResult(
                None,
                gpt_result.error,
                ExtractionRoute.GPT_ONLY,
                gpt_tokens=gpt_tokens,
                warnings=["GPT extraction failed, verification skipped"],
            )
        warnings.append(f"GPT extraction failed: {gpt_result.error.message}")
        gemini_result = extract_ocr(path, timeout=timeout, max_retries=max_retries)
        if isinstance(gemini_result, ProcessingError):
            return TieredResult(
                None,
                gemini_result,
                ExtractionRoute.GEMINI_FALLBACK,
                gpt_tokens=gpt_tokens,
                gemini_used=True,
                warnings=warnings + ["Gemini fallback also failed"],
            )
        return TieredResult(
            gemini_result,
            None,
            ExtractionRoute.GEMINI_FALLBACK,
            gpt_tokens=gpt_tokens,
            gemini_used=True,
            warnings=warnings + ["Using Gemini fallback"],
        )

    # Step 2: Check confidence (or skip verification if requested)
    if skip_verification or gpt_result.confidence >= confidence_threshold:
        return TieredResult(
            gpt_result.result,
            None,
            ExtractionRoute.GPT_ONLY,
            gpt_confidence=gpt_result.confidence,
            gpt_tokens=gpt_tokens,
        )

    # Step 3: Low confidence - verify with Gemini
    warnings.append(f"Low GPT confidence ({gpt_result.confidence:.2f}), verifying with Gemini")
    gemini_result = extract_ocr(path, timeout=timeout, max_retries=max_retries)

    if isinstance(gemini_result, ProcessingError):
        warnings.append("Gemini verification failed, using low-confidence GPT result")
        return TieredResult(
            gpt_result.result,
            None,
            ExtractionRoute.GPT_ONLY,
            gpt_confidence=gpt_result.confidence,
            gpt_tokens=gpt_tokens,
            warnings=warnings,
        )

    # Step 4: Compare results
    similarity = _calculate_ocr_similarity(gpt_result.result, gemini_result)

    if similarity >= similarity_threshold:
        return TieredResult(
            gpt_result.result,
            None,
            ExtractionRoute.GPT_VERIFIED,
            gpt_confidence=gpt_result.confidence,
            similarity_score=similarity,
            gpt_tokens=gpt_tokens,
            gemini_used=True,
            warnings=warnings,
        )
    else:
        warnings.append(f"GPT/Gemini disagreement (similarity={similarity:.2f}), using Gemini")
        return TieredResult(
            gemini_result,
            None,
            ExtractionRoute.GEMINI_OVERRIDE,
            gpt_confidence=gpt_result.confidence,
            similarity_score=similarity,
            gpt_tokens=gpt_tokens,
            gemini_used=True,
            flagged_for_review=True,
            warnings=warnings,
        )


def extract_metadata_tiered(
    path: str,
    ocr: RawOCRTokens,
    confidence_threshold: float = config.TIERED_CONFIDENCE_THRESHOLD,
    similarity_threshold: float = config.TIERED_SIMILARITY_THRESHOLD,
    timeout: int = config.API_TIMEOUT_SECONDS,
    max_retries: int = config.API_MAX_RETRIES,
    skip_verification: bool = False,
) -> TieredResult[PlotMetadata]:
    """Tiered metadata: GPT-5 Mini primary, Gemini Flash verification if low confidence."""
    warnings: list[str] = []

    # Step 1: GPT-5 Mini primary extraction
    gpt_result = extract_metadata_gpt(
        path, ocr, timeout=timeout, max_retries=max_retries
    )
    gpt_tokens = (gpt_result.input_tokens, gpt_result.output_tokens)

    if gpt_result.error:
        # GPT failed, fall back to Gemini (unless skip_verification is set)
        if skip_verification:
            return TieredResult(
                None,
                gpt_result.error,
                ExtractionRoute.GPT_ONLY,
                gpt_tokens=gpt_tokens,
                warnings=["GPT metadata extraction failed, verification skipped"],
            )
        warnings.append(f"GPT metadata extraction failed: {gpt_result.error.message}")
        gemini_result = extract_metadata(
            path, ocr, timeout=timeout, max_retries=max_retries
        )
        if isinstance(gemini_result, ProcessingError):
            return TieredResult(
                None,
                gemini_result,
                ExtractionRoute.GEMINI_FALLBACK,
                gpt_tokens=gpt_tokens,
                gemini_used=True,
                warnings=warnings + ["Gemini fallback also failed"],
            )
        return TieredResult(
            gemini_result,
            None,
            ExtractionRoute.GEMINI_FALLBACK,
            gpt_tokens=gpt_tokens,
            gemini_used=True,
            warnings=warnings + ["Using Gemini fallback"],
        )

    # Step 2: Check confidence (use OCR confidence as proxy since PlotMetadata doesn't have it)
    # For metadata, we use a heuristic: if GPT produced valid axis configs, confidence is high
    metadata_confidence = gpt_result.confidence
    if gpt_result.result and len(gpt_result.result.curves) > 0:
        metadata_confidence = max(metadata_confidence, config.METADATA_MIN_CONFIDENCE_BOOST)

    if skip_verification or metadata_confidence >= confidence_threshold:
        return TieredResult(
            gpt_result.result,
            None,
            ExtractionRoute.GPT_ONLY,
            gpt_confidence=metadata_confidence,
            gpt_tokens=gpt_tokens,
        )

    # Step 3: Low confidence - verify with Gemini
    warnings.append(
        f"Low GPT metadata confidence ({metadata_confidence:.2f}), verifying with Gemini"
    )
    gemini_result = extract_metadata(
        path, ocr, timeout=timeout, max_retries=max_retries
    )

    if isinstance(gemini_result, ProcessingError):
        warnings.append("Gemini verification failed, using low-confidence GPT result")
        return TieredResult(
            gpt_result.result,
            None,
            ExtractionRoute.GPT_ONLY,
            gpt_confidence=metadata_confidence,
            gpt_tokens=gpt_tokens,
            warnings=warnings,
        )

    # Step 4: Compare results
    similarity = _calculate_metadata_similarity(gpt_result.result, gemini_result)

    if similarity >= similarity_threshold:
        return TieredResult(
            gpt_result.result,
            None,
            ExtractionRoute.GPT_VERIFIED,
            gpt_confidence=metadata_confidence,
            similarity_score=similarity,
            gpt_tokens=gpt_tokens,
            gemini_used=True,
            warnings=warnings,
        )
    else:
        warnings.append(
            f"GPT/Gemini metadata disagreement (similarity={similarity:.2f}), using Gemini"
        )
        return TieredResult(
            gemini_result,
            None,
            ExtractionRoute.GEMINI_OVERRIDE,
            gpt_confidence=metadata_confidence,
            similarity_score=similarity,
            gpt_tokens=gpt_tokens,
            gemini_used=True,
            flagged_for_review=True,
            warnings=warnings,
        )


# --- Async Tiered Extraction ---


async def extract_ocr_tiered_async(
    path: str,
    confidence_threshold: float = config.TIERED_CONFIDENCE_THRESHOLD,
    similarity_threshold: float = config.TIERED_SIMILARITY_THRESHOLD,
    timeout: int = config.API_TIMEOUT_SECONDS,
    max_retries: int = config.API_MAX_RETRIES,
    skip_verification: bool = False,
) -> TieredResult[RawOCRTokens]:
    """Async tiered OCR: GPT-5 Mini primary, Gemini Flash verification if low confidence."""
    warnings: list[str] = []

    # Step 1: GPT-5 Mini primary extraction
    gpt_result = await extract_ocr_gpt_async(
        path, timeout=timeout, max_retries=max_retries
    )
    gpt_tokens = (gpt_result.input_tokens, gpt_result.output_tokens)

    if gpt_result.error:
        # GPT failed, fall back to Gemini (unless skip_verification is set)
        if skip_verification:
            return TieredResult(
                None,
                gpt_result.error,
                ExtractionRoute.GPT_ONLY,
                gpt_tokens=gpt_tokens,
                warnings=["GPT extraction failed, verification skipped"],
            )
        warnings.append(f"GPT extraction failed: {gpt_result.error.message}")
        gemini_result = await extract_ocr_async(
            path, timeout=timeout, max_retries=max_retries
        )
        if isinstance(gemini_result, ProcessingError):
            return TieredResult(
                None,
                gemini_result,
                ExtractionRoute.GEMINI_FALLBACK,
                gpt_tokens=gpt_tokens,
                gemini_used=True,
                warnings=warnings + ["Gemini fallback also failed"],
            )
        return TieredResult(
            gemini_result,
            None,
            ExtractionRoute.GEMINI_FALLBACK,
            gpt_tokens=gpt_tokens,
            gemini_used=True,
            warnings=warnings + ["Using Gemini fallback"],
        )

    # Step 2: Check confidence (or skip verification if requested)
    if skip_verification or gpt_result.confidence >= confidence_threshold:
        return TieredResult(
            gpt_result.result,
            None,
            ExtractionRoute.GPT_ONLY,
            gpt_confidence=gpt_result.confidence,
            gpt_tokens=gpt_tokens,
        )

    # Step 3: Low confidence - verify with Gemini
    warnings.append(f"Low GPT confidence ({gpt_result.confidence:.2f}), verifying with Gemini")
    gemini_result = await extract_ocr_async(
        path, timeout=timeout, max_retries=max_retries
    )

    if isinstance(gemini_result, ProcessingError):
        warnings.append("Gemini verification failed, using low-confidence GPT result")
        return TieredResult(
            gpt_result.result,
            None,
            ExtractionRoute.GPT_ONLY,
            gpt_confidence=gpt_result.confidence,
            gpt_tokens=gpt_tokens,
            warnings=warnings,
        )

    # Step 4: Compare results
    similarity = _calculate_ocr_similarity(gpt_result.result, gemini_result)

    if similarity >= similarity_threshold:
        return TieredResult(
            gpt_result.result,
            None,
            ExtractionRoute.GPT_VERIFIED,
            gpt_confidence=gpt_result.confidence,
            similarity_score=similarity,
            gpt_tokens=gpt_tokens,
            gemini_used=True,
            warnings=warnings,
        )
    else:
        warnings.append(f"GPT/Gemini disagreement (similarity={similarity:.2f}), using Gemini")
        return TieredResult(
            gemini_result,
            None,
            ExtractionRoute.GEMINI_OVERRIDE,
            gpt_confidence=gpt_result.confidence,
            similarity_score=similarity,
            gpt_tokens=gpt_tokens,
            gemini_used=True,
            flagged_for_review=True,
            warnings=warnings,
        )


async def extract_metadata_tiered_async(
    path: str,
    ocr: RawOCRTokens,
    confidence_threshold: float = config.TIERED_CONFIDENCE_THRESHOLD,
    similarity_threshold: float = config.TIERED_SIMILARITY_THRESHOLD,
    timeout: int = config.API_TIMEOUT_SECONDS,
    max_retries: int = config.API_MAX_RETRIES,
    skip_verification: bool = False,
) -> TieredResult[PlotMetadata]:
    """Async tiered metadata: GPT-5 Mini primary, Gemini Flash verification if low confidence."""
    warnings: list[str] = []

    # Step 1: GPT-5 Mini primary extraction
    gpt_result = await extract_metadata_gpt_async(
        path, ocr, timeout=timeout, max_retries=max_retries
    )
    gpt_tokens = (gpt_result.input_tokens, gpt_result.output_tokens)

    if gpt_result.error:
        # GPT failed, fall back to Gemini (unless skip_verification is set)
        if skip_verification:
            return TieredResult(
                None,
                gpt_result.error,
                ExtractionRoute.GPT_ONLY,
                gpt_tokens=gpt_tokens,
                warnings=["GPT metadata extraction failed, verification skipped"],
            )
        warnings.append(f"GPT metadata extraction failed: {gpt_result.error.message}")
        gemini_result = await extract_metadata_async(
            path, ocr, timeout=timeout, max_retries=max_retries
        )
        if isinstance(gemini_result, ProcessingError):
            return TieredResult(
                None,
                gemini_result,
                ExtractionRoute.GEMINI_FALLBACK,
                gpt_tokens=gpt_tokens,
                gemini_used=True,
                warnings=warnings + ["Gemini fallback also failed"],
            )
        return TieredResult(
            gemini_result,
            None,
            ExtractionRoute.GEMINI_FALLBACK,
            gpt_tokens=gpt_tokens,
            gemini_used=True,
            warnings=warnings + ["Using Gemini fallback"],
        )

    # Step 2: Check confidence (use OCR confidence as proxy since PlotMetadata doesn't have it)
    # For metadata, we use a heuristic: if GPT produced valid axis configs, confidence is high
    metadata_confidence = gpt_result.confidence
    if gpt_result.result and len(gpt_result.result.curves) > 0:
        metadata_confidence = max(metadata_confidence, config.METADATA_MIN_CONFIDENCE_BOOST)

    if skip_verification or metadata_confidence >= confidence_threshold:
        return TieredResult(
            gpt_result.result,
            None,
            ExtractionRoute.GPT_ONLY,
            gpt_confidence=metadata_confidence,
            gpt_tokens=gpt_tokens,
        )

    # Step 3: Low confidence - verify with Gemini
    warnings.append(
        f"Low GPT metadata confidence ({metadata_confidence:.2f}), verifying with Gemini"
    )
    gemini_result = await extract_metadata_async(
        path, ocr, timeout=timeout, max_retries=max_retries
    )

    if isinstance(gemini_result, ProcessingError):
        warnings.append("Gemini verification failed, using low-confidence GPT result")
        return TieredResult(
            gpt_result.result,
            None,
            ExtractionRoute.GPT_ONLY,
            gpt_confidence=metadata_confidence,
            gpt_tokens=gpt_tokens,
            warnings=warnings,
        )

    # Step 4: Compare results
    similarity = _calculate_metadata_similarity(gpt_result.result, gemini_result)

    if similarity >= similarity_threshold:
        return TieredResult(
            gpt_result.result,
            None,
            ExtractionRoute.GPT_VERIFIED,
            gpt_confidence=metadata_confidence,
            similarity_score=similarity,
            gpt_tokens=gpt_tokens,
            gemini_used=True,
            warnings=warnings,
        )
    else:
        warnings.append(
            f"GPT/Gemini metadata disagreement (similarity={similarity:.2f}), using Gemini"
        )
        return TieredResult(
            gemini_result,
            None,
            ExtractionRoute.GEMINI_OVERRIDE,
            gpt_confidence=metadata_confidence,
            similarity_score=similarity,
            gpt_tokens=gpt_tokens,
            gemini_used=True,
            flagged_for_review=True,
            warnings=warnings,
        )
