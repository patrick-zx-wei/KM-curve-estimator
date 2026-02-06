"""MMPU node - tiered extraction with GPT-5 Mini primary, Gemini Flash verification."""

import re
import uuid
from pathlib import Path
from tempfile import gettempdir

from km_estimator import config
from km_estimator.models import (
    PipelineState,
    ProcessingError,
    ProcessingStage,
    RawOCRTokens,
    RiskGroup,
    RiskTable,
)
from km_estimator.utils import cv_utils
from km_estimator.utils.tiered_extractor import (
    extract_metadata_tiered,
    extract_metadata_tiered_async,
    extract_ocr_tiered,
    extract_ocr_tiered_async,
)


_NUM_RE = re.compile(r"-?\d+(?:\.\d+)?")


def _extract_numeric_tokens(cell: str) -> list[float]:
    """Extract numeric tokens from OCR cell text."""
    values: list[float] = []
    for match in _NUM_RE.findall(cell.replace(",", "")):
        try:
            values.append(float(match))
        except ValueError:
            continue
    return values


def _row_numbers(row: list[str]) -> list[float]:
    vals: list[float] = []
    for cell in row:
        vals.extend(_extract_numeric_tokens(cell))
    return vals


def _normalize_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", name.lower()).strip()


def _parse_risk_table_from_ocr(
    ocr_tokens: RawOCRTokens,
    curve_names: list[str],
) -> RiskTable | None:
    """Build RiskTable from OCR risk_table_text if possible."""
    table = ocr_tokens.risk_table_text
    if not table:
        return None

    rows = [[str(c).strip() for c in row if str(c).strip()] for row in table]
    rows = [row for row in rows if row]
    if len(rows) < 2:
        return None

    # Pick header row with the most numeric entries as time points.
    header_idx = -1
    header_numbers: list[float] = []
    for i, row in enumerate(rows):
        nums = _row_numbers(row)
        if len(nums) > len(header_numbers):
            header_numbers = nums
            header_idx = i
    if len(header_numbers) < 2:
        return None

    # Ensure non-decreasing time points and cap implausible length.
    time_points = header_numbers[:12]
    if any(time_points[i] > time_points[i + 1] for i in range(len(time_points) - 1)):
        time_points = sorted(time_points)
    if len(time_points) < 2:
        return None

    normalized_curves = [_normalize_name(n) for n in curve_names]
    used_row_idxs: set[int] = {header_idx}
    groups: list[RiskGroup] = []

    # First pass: name-matched rows.
    for curve_name, norm_name in zip(curve_names, normalized_curves):
        match_idx = None
        for i, row in enumerate(rows):
            if i in used_row_idxs:
                continue
            row_text = _normalize_name(" ".join(row))
            if norm_name and norm_name in row_text:
                nums = _row_numbers(row)
                if len(nums) >= len(time_points):
                    match_idx = i
                    break
        if match_idx is None:
            continue
        used_row_idxs.add(match_idx)
        nums = _row_numbers(rows[match_idx])[:len(time_points)]
        counts = [max(0, int(round(v))) for v in nums]
        groups.append(RiskGroup(name=curve_name, counts=counts))

    # Second pass: numeric fallback rows for missing groups.
    missing = [name for name in curve_names if name not in {g.name for g in groups}]
    if missing:
        candidate_rows: list[int] = []
        for i, row in enumerate(rows):
            if i in used_row_idxs:
                continue
            nums = _row_numbers(row)
            if len(nums) >= len(time_points):
                candidate_rows.append(i)
        for curve_name, row_idx in zip(missing, candidate_rows):
            used_row_idxs.add(row_idx)
            nums = _row_numbers(rows[row_idx])[:len(time_points)]
            counts = [max(0, int(round(v))) for v in nums]
            groups.append(RiskGroup(name=curve_name, counts=counts))

    if len(groups) != len(curve_names):
        return None

    return RiskTable(time_points=[float(t) for t in time_points], groups=groups)


def _extract_risk_table_from_cropped_region(
    image_path: str,
    curve_names: list[str],
    confidence_threshold: float,
    similarity_threshold: float,
    timeout: int,
    skip_verification: bool,
) -> tuple[RiskTable | None, list[str], tuple[int, int] | None, bool]:
    """Second-pass risk table extraction from lower image region."""
    warnings: list[str] = []
    image = cv_utils.load_image(image_path, stage=ProcessingStage.MMPU)
    if isinstance(image, ProcessingError):
        return None, warnings, None, False

    h = image.shape[0]
    y0 = max(0, int(h * 0.52))
    crop = image[y0:, :]
    if crop.size == 0:
        return None, warnings, None, False

    tmp_dir = Path(gettempdir()) / "km_estimator"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    crop_path = tmp_dir / f"risk_table_crop_{uuid.uuid4().hex[:10]}.png"
    save = cv_utils.save_image(crop, crop_path, stage=ProcessingStage.MMPU)
    if isinstance(save, ProcessingError):
        return None, warnings, None, False

    try:
        ocr_result = extract_ocr_tiered(
            str(crop_path),
            confidence_threshold=confidence_threshold,
            similarity_threshold=similarity_threshold,
            timeout=timeout,
            skip_verification=skip_verification,
        )
        warnings.extend(ocr_result.warnings)
        if ocr_result.error or ocr_result.result is None:
            return None, warnings, ocr_result.gpt_tokens, ocr_result.gemini_used

        rt = _parse_risk_table_from_ocr(ocr_result.result, curve_names)
        if rt is None:
            warnings.append("Cropped risk-table OCR could not be parsed into structured table")
        return rt, warnings, ocr_result.gpt_tokens, ocr_result.gemini_used
    finally:
        try:
            crop_path.unlink(missing_ok=True)
        except OSError:
            pass


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

    # Risk-table recovery pass:
    # 1) Parse from OCR risk_table_text when metadata lacks table.
    # 2) If still missing, OCR bottom crop and parse table-focused content.
    if plot_metadata.risk_table is None and ocr_tokens is not None:
        curve_names = [c.name for c in plot_metadata.curves]
        parsed_rt = _parse_risk_table_from_ocr(ocr_tokens, curve_names)
        if parsed_rt is not None:
            plot_metadata = plot_metadata.model_copy(update={"risk_table": parsed_rt})
            warnings.append("Recovered risk table from OCR tokens")
        else:
            crop_rt, crop_warnings, crop_tokens, crop_gemini_used = (
                _extract_risk_table_from_cropped_region(
                    image_path=image_path,
                    curve_names=curve_names,
                    confidence_threshold=cfg.tiered_confidence_threshold,
                    similarity_threshold=cfg.tiered_similarity_threshold,
                    timeout=cfg.api_timeout_seconds,
                    skip_verification=cfg.single_model_mode,
                )
            )
            warnings.extend(crop_warnings)
            if crop_tokens:
                total_cost += _calculate_cost(crop_tokens, crop_gemini_used)
            if crop_rt is not None:
                plot_metadata = plot_metadata.model_copy(update={"risk_table": crop_rt})
                warnings.append("Recovered risk table from cropped lower-region OCR")

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

    # Async risk-table recovery mirrors sync logic.
    if plot_metadata.risk_table is None and ocr_tokens is not None:
        curve_names = [c.name for c in plot_metadata.curves]
        parsed_rt = _parse_risk_table_from_ocr(ocr_tokens, curve_names)
        if parsed_rt is not None:
            plot_metadata = plot_metadata.model_copy(update={"risk_table": parsed_rt})
            warnings.append("Recovered risk table from OCR tokens")
        else:
            image = cv_utils.load_image(image_path, stage=ProcessingStage.MMPU)
            if not isinstance(image, ProcessingError):
                h = image.shape[0]
                y0 = max(0, int(h * 0.52))
                crop = image[y0:, :]
                if crop.size > 0:
                    tmp_dir = Path(gettempdir()) / "km_estimator"
                    tmp_dir.mkdir(parents=True, exist_ok=True)
                    crop_path = tmp_dir / f"risk_table_crop_{uuid.uuid4().hex[:10]}.png"
                    save = cv_utils.save_image(crop, crop_path, stage=ProcessingStage.MMPU)
                    if not isinstance(save, ProcessingError):
                        try:
                            ocr_crop = await extract_ocr_tiered_async(
                                str(crop_path),
                                confidence_threshold=cfg.tiered_confidence_threshold,
                                similarity_threshold=cfg.tiered_similarity_threshold,
                                timeout=cfg.api_timeout_seconds,
                                skip_verification=cfg.single_model_mode,
                            )
                            warnings.extend(ocr_crop.warnings)
                            if ocr_crop.gpt_tokens:
                                total_cost += _calculate_cost(
                                    ocr_crop.gpt_tokens, ocr_crop.gemini_used
                                )
                            if ocr_crop.result is not None and ocr_crop.error is None:
                                crop_rt = _parse_risk_table_from_ocr(ocr_crop.result, curve_names)
                                if crop_rt is not None:
                                    plot_metadata = plot_metadata.model_copy(
                                        update={"risk_table": crop_rt}
                                    )
                                    warnings.append(
                                        "Recovered risk table from cropped lower-region OCR"
                                    )
                        finally:
                            try:
                                crop_path.unlink(missing_ok=True)
                            except OSError:
                                pass

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
