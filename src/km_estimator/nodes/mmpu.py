"""MMPU node - tiered extraction with GPT-5 Mini primary, Gemini Flash verification."""

import math
import re
import uuid
from pathlib import Path
from tempfile import gettempdir

from km_estimator import config
from km_estimator.models import (
    PipelineState,
    PlotMetadata,
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
FAST_PHASE_TIMEOUT_SECONDS = 12
FAST_PHASE_MAX_RETRIES = 1
RISK_TABLE_CROP_RATIOS = (0.35, 0.45, 0.52, 0.60, 0.70)


def _mmpu_retry_phases(api_timeout_seconds: int, api_max_retries: int) -> list[tuple[str, int, int]]:
    """Bounded two-phase retry policy: fast attempt, then one recovery attempt."""
    fast_timeout = max(8, min(FAST_PHASE_TIMEOUT_SECONDS, api_timeout_seconds))
    recovery_timeout = max(fast_timeout, api_timeout_seconds)
    recovery_retries = max(1, api_max_retries)
    return [
        ("fast", fast_timeout, FAST_PHASE_MAX_RETRIES),
        ("recovery", recovery_timeout, recovery_retries),
    ]


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


def _best_non_decreasing_run(nums: list[float]) -> list[float]:
    """Return longest contiguous non-decreasing run from nums."""
    if len(nums) < 2:
        return []
    best: list[float] = []
    curr: list[float] = [nums[0]]
    for value in nums[1:]:
        if value + 1e-6 >= curr[-1]:
            curr.append(value)
        else:
            if len(curr) > len(best):
                best = curr
            curr = [value]
    if len(curr) > len(best):
        best = curr
    return best if len(best) >= 2 else []


def _select_time_points(rows: list[list[str]]) -> tuple[list[float], int]:
    """Select a plausible risk-table time row and return (time_points, header_idx)."""
    best_idx = -1
    best_times: list[float] = []
    best_score = -1.0

    for i, row in enumerate(rows):
        nums = [v for v in _row_numbers(row) if v >= 0]
        run = _best_non_decreasing_run(nums)
        if len(run) < 2:
            continue
        capped = run[:12]
        span = capped[-1] - capped[0]
        # Prefer longer monotone rows with larger time span.
        score = len(capped) * 10.0 + max(0.0, span)
        if score > best_score:
            best_score = score
            best_idx = i
            best_times = capped

    return best_times, best_idx


def _select_count_sequence(nums: list[float], target_len: int) -> list[int] | None:
    """Extract most plausible at-risk count sequence from OCR numbers."""
    if target_len <= 0:
        return None
    if not nums:
        return None

    vals = [v for v in nums if v >= 0]
    min_required = max(2, int(math.ceil(target_len * 0.6)))
    if len(vals) < min_required:
        return None

    best_window: list[float] = []
    best_score = -1e9
    window_len = min(target_len, len(vals))
    max_start = max(0, len(vals) - window_len)

    for start in range(max_start + 1):
        window = vals[start : start + window_len]
        if not window:
            continue
        non_increasing = 0
        for j in range(len(window) - 1):
            if window[j] + 1e-6 >= window[j + 1]:
                non_increasing += 1
        monotone_score = non_increasing / max(1, len(window) - 1)
        first_bias = window[0] * 0.002
        integer_penalty = sum(abs(x - round(x)) for x in window) * 0.01
        score = monotone_score * 5.0 + first_bias - integer_penalty
        if score > best_score:
            best_score = score
            best_window = window

    if not best_window:
        return None

    counts = [max(0, int(round(v))) for v in best_window]
    if len(counts) < target_len:
        counts.extend([counts[-1]] * (target_len - len(counts)))
    return counts[:target_len]


def _name_matches_curve(row_text: str, curve_name: str) -> bool:
    """Fuzzy row-to-curve name match for OCR-degraded labels."""
    row_norm = _normalize_name(row_text)
    curve_norm = _normalize_name(curve_name)
    if not row_norm or not curve_norm:
        return False
    if curve_norm in row_norm or row_norm in curve_norm:
        return True
    row_tokens = set(row_norm.split())
    curve_tokens = [t for t in curve_norm.split() if t]
    if not curve_tokens:
        return False
    overlap = sum(1 for t in curve_tokens if t in row_tokens)
    return overlap >= max(1, len(curve_tokens) - 1)


def _risk_table_quality(rt: RiskTable | None) -> float:
    """Simple quality score for selecting best recovered risk table."""
    if rt is None:
        return -1.0
    times = rt.time_points
    monotone = 1.0 if all(times[i] <= times[i + 1] for i in range(len(times) - 1)) else 0.0
    return len(rt.groups) * 100.0 + len(times) * 2.0 + monotone


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

    time_points, header_idx = _select_time_points(rows)
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
            row_text = " ".join(row)
            if norm_name and _name_matches_curve(row_text, curve_name):
                counts = _select_count_sequence(_row_numbers(row), len(time_points))
                if counts is not None:
                    match_idx = i
                    break
        if match_idx is None:
            continue
        used_row_idxs.add(match_idx)
        counts = _select_count_sequence(_row_numbers(rows[match_idx]), len(time_points))
        if counts is not None:
            groups.append(RiskGroup(name=curve_name, counts=counts))

    # Second pass: numeric fallback rows for missing groups.
    missing = [name for name in curve_names if name not in {g.name for g in groups}]
    if missing:
        candidate_rows: list[int] = []
        for i, row in enumerate(rows):
            if i in used_row_idxs:
                continue
            nums = _row_numbers(row)
            if _select_count_sequence(nums, len(time_points)) is not None:
                candidate_rows.append(i)
        for curve_name, row_idx in zip(missing, candidate_rows):
            used_row_idxs.add(row_idx)
            counts = _select_count_sequence(_row_numbers(rows[row_idx]), len(time_points))
            if counts is not None:
                groups.append(RiskGroup(name=curve_name, counts=counts))

    if not groups:
        return None

    return RiskTable(time_points=[float(t) for t in time_points], groups=groups)


def _ocr_x_tick_values(ocr_tokens: RawOCRTokens | None) -> list[float]:
    """Parse numeric x-axis tick labels from OCR output."""
    if ocr_tokens is None:
        return []
    values: list[float] = []
    for label in ocr_tokens.x_tick_labels:
        for value in _extract_numeric_tokens(str(label)):
            if value >= 0:
                values.append(float(value))
    return sorted(set(values))


def _ocr_y_tick_values(ocr_tokens: RawOCRTokens | None) -> list[float]:
    """Parse numeric y-axis tick labels from OCR output."""
    if ocr_tokens is None:
        return []
    values: list[float] = []
    for label in ocr_tokens.y_tick_labels:
        for value in _extract_numeric_tokens(str(label)):
            if -0.1 <= value <= 1.5:
                values.append(float(value))
    return sorted(set(values))


def _maybe_correct_y_axis_start(
    plot_metadata: PlotMetadata,
    ocr_tokens: RawOCRTokens | None,
    warnings: list[str],
) -> PlotMetadata:
    """Correct y_axis.start when OCR/tick evidence suggests a nearby better value."""
    y_axis = plot_metadata.y_axis
    candidates: list[float] = []

    if y_axis.tick_values:
        candidates.append(min(float(v) for v in y_axis.tick_values))

    ocr_y_ticks = _ocr_y_tick_values(ocr_tokens)
    if ocr_y_ticks:
        candidates.append(min(ocr_y_ticks))

    if not candidates:
        return plot_metadata

    corrected_start = min(candidates)
    if not (-0.1 <= corrected_start <= y_axis.end):
        return plot_metadata

    # Only apply conservative nearby corrections to avoid large accidental shifts.
    if abs(corrected_start - y_axis.start) < 0.02 or abs(corrected_start - y_axis.start) > 0.12:
        return plot_metadata

    warnings.append(f"Corrected y_axis.start from {y_axis.start} to {corrected_start}")
    return plot_metadata.model_copy(
        update={"y_axis": y_axis.model_copy(update={"start": corrected_start})}
    )


def _maybe_correct_x_axis_end(
    plot_metadata: PlotMetadata,
    ocr_tokens: RawOCRTokens | None,
    warnings: list[str],
) -> PlotMetadata:
    """Correct x_axis.end when metadata likely used the last tick instead of true endpoint."""
    x_axis = plot_metadata.x_axis
    if not x_axis.tick_values:
        return plot_metadata

    last_tick = max(x_axis.tick_values)
    if abs(x_axis.end - last_tick) > 1e-6:
        return plot_metadata

    corrected_end = x_axis.end
    source = ""

    if plot_metadata.risk_table and plot_metadata.risk_table.time_points:
        rt_end = max(plot_metadata.risk_table.time_points)
        if rt_end > corrected_end + 1e-6:
            corrected_end = float(rt_end)
            source = "risk table time points"

    ocr_x_ticks = _ocr_x_tick_values(ocr_tokens)
    if ocr_x_ticks:
        ocr_end = max(ocr_x_ticks)
        if ocr_end > corrected_end + 1e-6:
            corrected_end = float(ocr_end)
            source = "OCR x-axis ticks"

    # Conservative fallback: only extrapolate one tick when OCR found more x ticks than metadata.
    sorted_ticks = sorted(set(float(t) for t in x_axis.tick_values))
    spacing_matches_interval = False
    if len(sorted_ticks) >= 2 and x_axis.tick_interval is not None and x_axis.tick_interval > 0:
        last_gap = sorted_ticks[-1] - sorted_ticks[-2]
        spacing_matches_interval = abs(last_gap - x_axis.tick_interval) < 1e-3

    if (
        not source
        and x_axis.tick_interval is not None
        and x_axis.tick_interval > 0
        and spacing_matches_interval
        and len(ocr_x_ticks) > len(set(x_axis.tick_values))
    ):
        corrected_end = float(last_tick + x_axis.tick_interval)
        source = "tick-interval extrapolation"

    if corrected_end <= x_axis.end + 1e-6:
        return plot_metadata

    warnings.append(f"Corrected x_axis.end from {x_axis.end} to {corrected_end} via {source}")
    return plot_metadata.model_copy(
        update={"x_axis": x_axis.model_copy(update={"end": corrected_end})}
    )


def _extract_risk_table_from_cropped_region(
    image_path: str,
    curve_names: list[str],
    confidence_threshold: float,
    similarity_threshold: float,
    timeout: int,
    max_retries: int,
    skip_verification: bool,
) -> tuple[RiskTable | None, list[str], tuple[int, int] | None, bool]:
    """Second-pass risk table extraction from lower image region."""
    warnings: list[str] = []
    image = cv_utils.load_image(image_path, stage=ProcessingStage.MMPU)
    if isinstance(image, ProcessingError):
        return None, warnings, None, False

    h = image.shape[0]
    tmp_dir = Path(gettempdir()) / "km_estimator"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    best_rt: RiskTable | None = None
    best_tokens: tuple[int, int] | None = None
    best_gemini_used = False
    best_score = -1.0

    # Sweep multiple lower-region windows to handle diverse risk-table placements.
    for ratio in RISK_TABLE_CROP_RATIOS:
        y0 = max(0, int(h * ratio))
        crop = image[y0:, :]
        if crop.size == 0:
            continue
        crop_path = tmp_dir / f"risk_table_crop_{uuid.uuid4().hex[:10]}.png"
        save = cv_utils.save_image(crop, crop_path, stage=ProcessingStage.MMPU)
        if isinstance(save, ProcessingError):
            continue

        try:
            ocr_result = extract_ocr_tiered(
                str(crop_path),
                confidence_threshold=confidence_threshold,
                similarity_threshold=similarity_threshold,
                timeout=timeout,
                max_retries=max_retries,
                skip_verification=skip_verification,
            )
            warnings.extend(ocr_result.warnings)
            if ocr_result.error or ocr_result.result is None:
                continue

            rt = _parse_risk_table_from_ocr(ocr_result.result, curve_names)
            score = _risk_table_quality(rt)
            if score > best_score and rt is not None:
                best_score = score
                best_rt = rt
                best_tokens = ocr_result.gpt_tokens
                best_gemini_used = ocr_result.gemini_used
        finally:
            try:
                crop_path.unlink(missing_ok=True)
            except OSError:
                pass

    if best_rt is None:
        warnings.append("Cropped risk-table OCR could not be parsed into structured table")
    return best_rt, warnings, best_tokens, best_gemini_used


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

    # Stage 1: Tiered OCR extraction with bounded two-phase retry policy.
    ocr_result = None
    ocr_phases = _mmpu_retry_phases(cfg.api_timeout_seconds, cfg.api_max_retries)
    for phase_idx, (phase_name, phase_timeout, phase_retries) in enumerate(ocr_phases):
        ocr_result = extract_ocr_tiered(
            image_path,
            confidence_threshold=cfg.tiered_confidence_threshold,
            similarity_threshold=cfg.tiered_similarity_threshold,
            timeout=phase_timeout,
            max_retries=phase_retries,
            skip_verification=cfg.single_model_mode,
        )
        if not (ocr_result.error and ocr_result.error.recoverable):
            break
        if phase_idx < len(ocr_phases) - 1:
            warnings.append(
                f"OCR {phase_name} phase failed ({ocr_result.error.error_type}); escalating"
            )

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

    # Stage 2: Tiered metadata extraction with bounded two-phase retry policy.
    metadata_result = None
    metadata_phases = _mmpu_retry_phases(cfg.api_timeout_seconds, cfg.api_max_retries)
    for phase_idx, (phase_name, phase_timeout, phase_retries) in enumerate(metadata_phases):
        metadata_result = extract_metadata_tiered(
            image_path,
            ocr_tokens,
            confidence_threshold=cfg.tiered_confidence_threshold,
            similarity_threshold=cfg.tiered_similarity_threshold,
            timeout=phase_timeout,
            max_retries=phase_retries,
            skip_verification=cfg.single_model_mode,
        )
        if not (metadata_result.error and metadata_result.error.recoverable):
            break
        if phase_idx < len(metadata_phases) - 1:
            warnings.append(
                f"Metadata {phase_name} phase failed "
                f"({metadata_result.error.error_type}); escalating"
            )

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
                    timeout=min(cfg.api_timeout_seconds, FAST_PHASE_TIMEOUT_SECONDS),
                    max_retries=FAST_PHASE_MAX_RETRIES,
                    skip_verification=cfg.single_model_mode,
                )
            )
            warnings.extend(crop_warnings)
            if crop_tokens:
                total_cost += _calculate_cost(crop_tokens, crop_gemini_used)
            if crop_rt is not None:
                plot_metadata = plot_metadata.model_copy(update={"risk_table": crop_rt})
                warnings.append("Recovered risk table from cropped lower-region OCR")

    plot_metadata = _maybe_correct_y_axis_start(plot_metadata, ocr_tokens, warnings)
    plot_metadata = _maybe_correct_x_axis_end(plot_metadata, ocr_tokens, warnings)

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

    # Stage 1: Tiered OCR extraction with bounded two-phase retry policy.
    ocr_result = None
    ocr_phases = _mmpu_retry_phases(cfg.api_timeout_seconds, cfg.api_max_retries)
    for phase_idx, (phase_name, phase_timeout, phase_retries) in enumerate(ocr_phases):
        ocr_result = await extract_ocr_tiered_async(
            image_path,
            confidence_threshold=cfg.tiered_confidence_threshold,
            similarity_threshold=cfg.tiered_similarity_threshold,
            timeout=phase_timeout,
            max_retries=phase_retries,
            skip_verification=cfg.single_model_mode,
        )
        if not (ocr_result.error and ocr_result.error.recoverable):
            break
        if phase_idx < len(ocr_phases) - 1:
            warnings.append(
                f"OCR {phase_name} phase failed ({ocr_result.error.error_type}); escalating"
            )

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

    # Stage 2: Tiered metadata extraction with bounded two-phase retry policy.
    metadata_result = None
    metadata_phases = _mmpu_retry_phases(cfg.api_timeout_seconds, cfg.api_max_retries)
    for phase_idx, (phase_name, phase_timeout, phase_retries) in enumerate(metadata_phases):
        metadata_result = await extract_metadata_tiered_async(
            image_path,
            ocr_tokens,
            confidence_threshold=cfg.tiered_confidence_threshold,
            similarity_threshold=cfg.tiered_similarity_threshold,
            timeout=phase_timeout,
            max_retries=phase_retries,
            skip_verification=cfg.single_model_mode,
        )
        if not (metadata_result.error and metadata_result.error.recoverable):
            break
        if phase_idx < len(metadata_phases) - 1:
            warnings.append(
                f"Metadata {phase_name} phase failed "
                f"({metadata_result.error.error_type}); escalating"
            )

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
                tmp_dir = Path(gettempdir()) / "km_estimator"
                tmp_dir.mkdir(parents=True, exist_ok=True)
                best_rt: RiskTable | None = None
                best_score = -1.0
                for ratio in RISK_TABLE_CROP_RATIOS:
                    y0 = max(0, int(h * ratio))
                    crop = image[y0:, :]
                    if crop.size == 0:
                        continue
                    crop_path = tmp_dir / f"risk_table_crop_{uuid.uuid4().hex[:10]}.png"
                    save = cv_utils.save_image(crop, crop_path, stage=ProcessingStage.MMPU)
                    if isinstance(save, ProcessingError):
                        continue
                    try:
                        ocr_crop = await extract_ocr_tiered_async(
                            str(crop_path),
                            confidence_threshold=cfg.tiered_confidence_threshold,
                            similarity_threshold=cfg.tiered_similarity_threshold,
                            timeout=min(cfg.api_timeout_seconds, FAST_PHASE_TIMEOUT_SECONDS),
                            max_retries=FAST_PHASE_MAX_RETRIES,
                            skip_verification=cfg.single_model_mode,
                        )
                        warnings.extend(ocr_crop.warnings)
                        if ocr_crop.gpt_tokens:
                            total_cost += _calculate_cost(
                                ocr_crop.gpt_tokens, ocr_crop.gemini_used
                            )
                        if ocr_crop.result is not None and ocr_crop.error is None:
                            crop_rt = _parse_risk_table_from_ocr(ocr_crop.result, curve_names)
                            score = _risk_table_quality(crop_rt)
                            if crop_rt is not None and score > best_score:
                                best_score = score
                                best_rt = crop_rt
                    finally:
                        try:
                            crop_path.unlink(missing_ok=True)
                        except OSError:
                            pass
                if best_rt is not None:
                    plot_metadata = plot_metadata.model_copy(update={"risk_table": best_rt})
                    warnings.append("Recovered risk table from cropped lower-region OCR")

    plot_metadata = _maybe_correct_y_axis_start(plot_metadata, ocr_tokens, warnings)
    plot_metadata = _maybe_correct_x_axis_end(plot_metadata, ocr_tokens, warnings)

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
