"""MMPU node - tiered extraction with GPT-5 Mini primary, Gemini Flash verification."""

import math
import re
import uuid
from pathlib import Path
from tempfile import gettempdir

import cv2
import numpy as np

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
RISK_TABLE_REPLACEMENT_MARGIN = 0.25
AXIS_X_CROP_START_RATIO = 0.80
AXIS_Y_CROP_END_RATIO = 0.24
UPWARD_DIRECTION_HINTS = (
    "cumulative incidence",
    "incidence",
    "cumulative event",
    "event probability",
    "failure probability",
    "recurrence",
    "mortality",
)
DOWNWARD_DIRECTION_HINTS = (
    "survival",
    "overall survival",
    "progression-free survival",
    "disease-free survival",
    "event-free survival",
    "survival probability",
)


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


def _infer_curve_direction(
    plot_metadata: PlotMetadata,
    ocr_tokens: RawOCRTokens | None,
    warnings: list[str],
) -> PlotMetadata:
    """Infer curve direction from extracted text and reconcile metadata."""
    texts: list[str] = []
    if plot_metadata.title:
        texts.append(str(plot_metadata.title))
    texts.extend(str(a) for a in plot_metadata.annotations)
    if plot_metadata.x_axis.label:
        texts.append(str(plot_metadata.x_axis.label))
    if plot_metadata.y_axis.label:
        texts.append(str(plot_metadata.y_axis.label))
    if ocr_tokens is not None:
        texts.extend(str(a) for a in ocr_tokens.axis_labels)
        texts.extend(str(a) for a in ocr_tokens.annotations)
        if ocr_tokens.title:
            texts.append(str(ocr_tokens.title))

    combined = " ".join(texts).lower()
    upward_hits = sum(1 for hint in UPWARD_DIRECTION_HINTS if hint in combined)
    downward_hits = sum(1 for hint in DOWNWARD_DIRECTION_HINTS if hint in combined)
    explicit_upward = any(
        phrase in combined
        for phrase in ("cumulative incidence", "incidence", "cumulative event")
    )
    explicit_downward = any(
        phrase in combined
        for phrase in (
            "overall survival",
            "progression-free survival",
            "disease-free survival",
            "event-free survival",
            "survival probability",
        )
    )

    inferred: str | None = None
    if explicit_upward and not explicit_downward:
        inferred = "upward"
    elif explicit_downward and not explicit_upward:
        inferred = "downward"
    elif upward_hits >= 2 and upward_hits >= downward_hits + 2:
        inferred = "upward"
    elif downward_hits >= 2 and downward_hits >= upward_hits + 2:
        inferred = "downward"

    current = plot_metadata.curve_direction if plot_metadata.curve_direction in ("downward", "upward") else "downward"

    # Avoid overcorrecting explicit incidence/survival metadata unless opposite evidence is strong.
    if (
        current == "upward"
        and explicit_upward
        and not (explicit_downward and downward_hits >= upward_hits + 2)
    ):
        return plot_metadata
    if (
        current == "downward"
        and explicit_downward
        and not (explicit_upward and upward_hits >= downward_hits + 2)
    ):
        return plot_metadata

    if inferred is not None and inferred != current:
        warnings.append(
            f"Adjusted curve_direction from {current} to {inferred} based on labels/annotations"
        )
        return plot_metadata.model_copy(update={"curve_direction": inferred})

    if plot_metadata.curve_direction not in ("downward", "upward"):
        resolved = inferred or "downward"
        warnings.append(f"Defaulted invalid curve_direction to {resolved}")
        return plot_metadata.model_copy(update={"curve_direction": resolved})

    return plot_metadata


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
    group_quality = 0.0
    for group in rt.groups:
        group_quality += _counts_monotone_score(group.counts)
    return len(rt.groups) * 100.0 + len(times) * 2.0 + monotone + group_quality


def _risk_table_needs_recovery(rt: RiskTable | None, curve_names: list[str]) -> bool:
    """Decide whether to run OCR recovery for risk table."""
    if rt is None:
        return True
    if len(rt.time_points) < 2:
        return True
    expected = set(curve_names)
    got = {g.name for g in rt.groups}
    if expected and len(expected & got) < len(expected):
        return True
    quality_floor = len(curve_names) * 100.0 + 8.0
    return _risk_table_quality(rt) < quality_floor


def _parse_risk_table_from_ocr(
    ocr_tokens: RawOCRTokens,
    curve_names: list[str],
) -> RiskTable | None:
    """Build RiskTable from OCR output, with fallback to unstructured OCR text."""
    rows: list[list[str]] = []
    table = ocr_tokens.risk_table_text
    if table:
        rows.extend([[str(c).strip() for c in row if str(c).strip()] for row in table])

    # Fallback rows derived from OCR text when structured table extraction fails.
    unstructured_sources: list[str] = []
    unstructured_sources.extend(str(v) for v in ocr_tokens.annotations)
    unstructured_sources.extend(str(v) for v in ocr_tokens.axis_labels)
    unstructured_sources.extend(str(v) for v in ocr_tokens.legend_labels)
    if ocr_tokens.title:
        unstructured_sources.append(str(ocr_tokens.title))
    for text in unstructured_sources:
        for chunk in re.split(r"[\n;|]+", text):
            tokens = [tok for tok in chunk.strip().split() if tok]
            if len(tokens) >= 2 and len(_extract_numeric_tokens(chunk)) >= 2:
                rows.append(tokens)

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
                if counts is not None and _is_plausible_risk_counts(counts):
                    match_idx = i
                    break
        if match_idx is None:
            continue
        used_row_idxs.add(match_idx)
        counts = _select_count_sequence(_row_numbers(rows[match_idx]), len(time_points))
        if counts is not None and _is_plausible_risk_counts(counts):
            groups.append(RiskGroup(name=curve_name, counts=counts))

    # Second pass: numeric fallback rows for missing groups.
    missing = [name for name in curve_names if name not in {g.name for g in groups}]
    if missing:
        candidate_rows: list[int] = []
        for i, row in enumerate(rows):
            if i in used_row_idxs:
                continue
            nums = _row_numbers(row)
            seq = _select_count_sequence(nums, len(time_points))
            if seq is not None and _is_plausible_risk_counts(seq):
                candidate_rows.append(i)
        for curve_name, row_idx in zip(missing, candidate_rows):
            used_row_idxs.add(row_idx)
            counts = _select_count_sequence(_row_numbers(rows[row_idx]), len(time_points))
            if counts is not None and _is_plausible_risk_counts(counts):
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
    raw_values: list[float] = []
    for label in ocr_tokens.y_tick_labels:
        for value in _extract_numeric_tokens(str(label)):
            raw_values.append(float(value))
    return _normalize_y_tick_scale(raw_values)


def _counts_monotone_score(counts: list[int]) -> float:
    """Score monotone non-increasing behavior in a risk-table count sequence."""
    if len(counts) < 2:
        return 0.0
    non_increasing = 0
    for i in range(len(counts) - 1):
        if counts[i] >= counts[i + 1]:
            non_increasing += 1
    return non_increasing / max(1, len(counts) - 1)


def _is_plausible_risk_counts(counts: list[int]) -> bool:
    """Basic sanity filter for OCR-derived risk-table counts."""
    if len(counts) < 2:
        return False
    if any(c < 0 for c in counts):
        return False
    if counts[0] <= 0:
        return False
    if counts[0] < counts[-1]:
        return False
    return _counts_monotone_score(counts) >= 0.65


def _score_y_tick_candidate(values: list[float]) -> float:
    """Score y-axis tick candidate list quality in survival-probability space."""
    if len(values) < 2:
        return -1e9
    ordered = sorted(set(float(v) for v in values))
    span = ordered[-1] - ordered[0]
    if span <= 0:
        return -1e9

    diffs = [ordered[i + 1] - ordered[i] for i in range(len(ordered) - 1)]
    median_step = sorted(diffs)[len(diffs) // 2] if diffs else 0.0

    score = float(len(ordered))
    if 0.0 <= ordered[0] <= 0.5:
        score += 1.0
    if 0.75 <= ordered[-1] <= 1.05:
        score += 1.5
    if 0.2 <= span <= 1.1:
        score += 1.0
    if 0.02 <= median_step <= 0.35:
        score += 0.5
    if ordered[-1] > 1.2:
        score -= 1.0
    return score


def _normalize_y_tick_scale(raw_values: list[float]) -> list[float]:
    """
    Normalize OCR y-axis ticks into probability scale.

    Handles both 0-1 and 0-100 style labels while keeping only plausible values.
    """
    if not raw_values:
        return []

    direct = [v for v in raw_values if -0.1 <= v <= 1.5]
    percent_like = [v / 100.0 for v in raw_values if 0.0 <= v <= 100.0]
    direct = sorted(set(float(v) for v in direct))
    percent_like = sorted(set(float(v) for v in percent_like if -0.1 <= v <= 1.5))

    direct_score = _score_y_tick_candidate(direct)
    percent_score = _score_y_tick_candidate(percent_like)

    if percent_score > direct_score + 0.4:
        return percent_like
    return direct


def _cluster_axis_positions(values: list[float], tolerance: float = 3.0) -> list[float]:
    """Cluster nearby line positions and return cluster medians."""
    if not values:
        return []
    ordered = sorted(float(v) for v in values)
    clusters: list[list[float]] = [[ordered[0]]]
    for value in ordered[1:]:
        if abs(value - clusters[-1][-1]) <= tolerance:
            clusters[-1].append(value)
        else:
            clusters.append([value])
    return [float(np.median(np.asarray(cluster, dtype=np.float64))) for cluster in clusters]


def _infer_y_axis_start_from_geometry(
    image_path: str,
    y_axis_end: float,
    y_tick_interval: float | None,
) -> tuple[float, float] | None:
    """
    Estimate y-axis floor from horizontal line geometry.

    Returns (estimated_start, confidence) when geometry is coherent.
    """
    if y_tick_interval is None or y_tick_interval <= 0:
        return None

    image = cv_utils.load_image(image_path, stage=ProcessingStage.MMPU)
    if isinstance(image, ProcessingError):
        return None

    h, w = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)
    lines = cv2.HoughLinesP(
        edges,
        1,
        np.pi / 180,
        threshold=90,
        minLineLength=max(30, int(w * 0.35)),
        maxLineGap=10,
    )
    if lines is None:
        return None

    y_positions: list[float] = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        if abs(y2 - y1) > 2:
            continue
        span = abs(x2 - x1)
        if span < int(w * 0.3):
            continue
        y_mid = 0.5 * (float(y1) + float(y2))
        if y_mid < h * 0.04 or y_mid > h * 0.82:
            continue
        y_positions.append(y_mid)

    clustered = _cluster_axis_positions(y_positions, tolerance=3.0)
    if len(clustered) < 4:
        return None

    clustered = sorted(clustered)
    diffs = np.diff(np.asarray(clustered, dtype=np.float64))
    valid_diffs = diffs[(diffs >= 6.0) & (diffs <= h * 0.35)]
    if len(valid_diffs) < 2:
        return None

    spacing = float(np.median(valid_diffs))
    if spacing <= 0:
        return None

    top_y = float(clustered[0])
    bottom_y = float(clustered[-1])
    interval_span = (bottom_y - top_y) / spacing
    if interval_span < 2.0 or interval_span > 10.0:
        return None

    estimated_start = float(y_axis_end - interval_span * float(y_tick_interval))
    if not (-0.1 <= estimated_start <= y_axis_end):
        return None

    mad = float(np.median(np.abs(valid_diffs - spacing)))
    rel_dispersion = mad / max(1.0, spacing)
    confidence = float(np.clip(1.0 - rel_dispersion * 3.0, 0.0, 1.0))
    return estimated_start, confidence


def _maybe_correct_y_axis_start(
    plot_metadata: PlotMetadata,
    ocr_tokens: RawOCRTokens | None,
    image_path: str,
    warnings: list[str],
) -> PlotMetadata:
    """Conservatively correct y_axis.start when multiple evidence sources agree."""
    y_axis = plot_metadata.y_axis
    evidence: list[tuple[float, str, float]] = []
    geometry_candidate: tuple[float, float] | None = None

    if y_axis.tick_values:
        min_tick = min(float(v) for v in y_axis.tick_values)
        evidence.append((min_tick, "metadata ticks", 1.0))

    ocr_y_ticks = _ocr_y_tick_values(ocr_tokens)
    if ocr_y_ticks:
        min_ocr_tick = min(ocr_y_ticks)
        evidence.append((min_ocr_tick, "OCR y-ticks", 1.0))

    geometry_candidate = _infer_y_axis_start_from_geometry(
        image_path=image_path,
        y_axis_end=float(y_axis.end),
        y_tick_interval=y_axis.tick_interval,
    )
    if geometry_candidate is not None:
        geom_value, geom_conf = geometry_candidate
        if 0.02 <= abs(geom_value - y_axis.start) <= 0.15:
            evidence.append((geom_value, "geometry", 1.0 + 0.5 * geom_conf))

    if not evidence:
        return plot_metadata

    valid_evidence = [
        (value, source, weight)
        for value, source, weight in evidence
        if -0.1 <= value <= y_axis.end
    ]
    if not valid_evidence:
        return plot_metadata

    # Apply only nearby corrections to avoid large accidental shifts.
    nearby_candidates = [
        (value, source, weight)
        for value, source, weight in valid_evidence
        if 0.02 <= abs(value - y_axis.start) <= 0.12
    ]
    if not nearby_candidates:
        return plot_metadata

    consensus_tol = 0.03
    candidate_values = sorted({round(value, 4) for value, _, _ in nearby_candidates})
    scored_candidates: list[tuple[float, float]] = []
    for candidate in candidate_values:
        support = sum(
            weight
            for value, _, weight in valid_evidence
            if abs(value - candidate) <= consensus_tol
        )
        scored_candidates.append((support, candidate))

    best_support, corrected_start = max(
        scored_candidates,
        key=lambda item: (item[0], -abs(item[1] - y_axis.start)),
    )
    if best_support < 2.0:
        if geometry_candidate is None:
            return plot_metadata
        geom_value, geom_conf = geometry_candidate
        if not (geom_conf >= 0.88 and 0.02 <= abs(geom_value - y_axis.start) <= 0.12):
            return plot_metadata
        corrected_start = float(geom_value)
        source = f"geometry (conf={geom_conf:.2f})"
        warnings.append(
            f"Corrected y_axis.start from {y_axis.start} to {corrected_start} ({source})"
        )
        return plot_metadata.model_copy(
            update={"y_axis": y_axis.model_copy(update={"start": corrected_start})}
        )

    source = "consensus ticks/OCR"
    warnings.append(
        f"Corrected y_axis.start from {y_axis.start} to {corrected_start} ({source})"
    )
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
        crop = cv_utils.preprocess_risk_table_ocr_region(crop)
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


def _merge_text_labels(primary: list[str], supplemental: list[str]) -> list[str]:
    """Merge OCR labels while preserving order and dropping exact duplicates."""
    merged: list[str] = []
    seen: set[str] = set()
    for label in [*primary, *supplemental]:
        text = str(label).strip()
        if not text:
            continue
        key = text.lower()
        if key in seen:
            continue
        seen.add(key)
        merged.append(text)
    return merged


def _recover_axis_ticks_from_crops(
    image_path: str,
    confidence_threshold: float,
    similarity_threshold: float,
    timeout: int,
    max_retries: int,
    skip_verification: bool,
) -> tuple[list[str], list[str], list[str], tuple[int, int] | None, bool]:
    """Axis-specific OCR recovery using global-thresholded axis crops."""
    warnings: list[str] = []
    image = cv_utils.load_image(image_path, stage=ProcessingStage.MMPU)
    if isinstance(image, ProcessingError):
        return [], [], warnings, None, False

    h, w = image.shape[:2]
    x_crop = image[max(0, int(h * AXIS_X_CROP_START_RATIO)) :, :]
    y_crop = image[:, : max(1, int(w * AXIS_Y_CROP_END_RATIO))]
    crops = [("x", x_crop), ("y", y_crop)]

    tmp_dir = Path(gettempdir()) / "km_estimator"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    recovered_x: list[str] = []
    recovered_y: list[str] = []
    total_tokens = [0, 0]
    gemini_used_any = False

    for axis_name, crop in crops:
        if crop.size == 0:
            continue
        processed_crop = cv_utils.preprocess_axis_ocr_region(crop)
        crop_path = tmp_dir / f"axis_{axis_name}_crop_{uuid.uuid4().hex[:10]}.png"
        save = cv_utils.save_image(processed_crop, crop_path, stage=ProcessingStage.MMPU)
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
            if ocr_result.gpt_tokens:
                total_tokens[0] += int(ocr_result.gpt_tokens[0])
                total_tokens[1] += int(ocr_result.gpt_tokens[1])
            gemini_used_any = gemini_used_any or bool(ocr_result.gemini_used)
            if ocr_result.error or ocr_result.result is None:
                continue
            if axis_name == "x":
                recovered_x.extend([str(v) for v in ocr_result.result.x_tick_labels])
            else:
                recovered_y.extend([str(v) for v in ocr_result.result.y_tick_labels])
        finally:
            try:
                crop_path.unlink(missing_ok=True)
            except OSError:
                pass

    token_tuple: tuple[int, int] | None = None
    if total_tokens[0] > 0 or total_tokens[1] > 0:
        token_tuple = (total_tokens[0], total_tokens[1])
    return recovered_x, recovered_y, warnings, token_tuple, gemini_used_any


async def _recover_axis_ticks_from_crops_async(
    image_path: str,
    confidence_threshold: float,
    similarity_threshold: float,
    timeout: int,
    max_retries: int,
    skip_verification: bool,
) -> tuple[list[str], list[str], list[str], tuple[int, int] | None, bool]:
    """Async axis-specific OCR recovery using global-thresholded axis crops."""
    warnings: list[str] = []
    image = cv_utils.load_image(image_path, stage=ProcessingStage.MMPU)
    if isinstance(image, ProcessingError):
        return [], [], warnings, None, False

    h, w = image.shape[:2]
    x_crop = image[max(0, int(h * AXIS_X_CROP_START_RATIO)) :, :]
    y_crop = image[:, : max(1, int(w * AXIS_Y_CROP_END_RATIO))]
    crops = [("x", x_crop), ("y", y_crop)]

    tmp_dir = Path(gettempdir()) / "km_estimator"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    recovered_x: list[str] = []
    recovered_y: list[str] = []
    total_tokens = [0, 0]
    gemini_used_any = False

    for axis_name, crop in crops:
        if crop.size == 0:
            continue
        processed_crop = cv_utils.preprocess_axis_ocr_region(crop)
        crop_path = tmp_dir / f"axis_{axis_name}_crop_{uuid.uuid4().hex[:10]}.png"
        save = cv_utils.save_image(processed_crop, crop_path, stage=ProcessingStage.MMPU)
        if isinstance(save, ProcessingError):
            continue
        try:
            ocr_result = await extract_ocr_tiered_async(
                str(crop_path),
                confidence_threshold=confidence_threshold,
                similarity_threshold=similarity_threshold,
                timeout=timeout,
                max_retries=max_retries,
                skip_verification=skip_verification,
            )
            warnings.extend(ocr_result.warnings)
            if ocr_result.gpt_tokens:
                total_tokens[0] += int(ocr_result.gpt_tokens[0])
                total_tokens[1] += int(ocr_result.gpt_tokens[1])
            gemini_used_any = gemini_used_any or bool(ocr_result.gemini_used)
            if ocr_result.error or ocr_result.result is None:
                continue
            if axis_name == "x":
                recovered_x.extend([str(v) for v in ocr_result.result.x_tick_labels])
            else:
                recovered_y.extend([str(v) for v in ocr_result.result.y_tick_labels])
        finally:
            try:
                crop_path.unlink(missing_ok=True)
            except OSError:
                pass

    token_tuple: tuple[int, int] | None = None
    if total_tokens[0] > 0 or total_tokens[1] > 0:
        token_tuple = (total_tokens[0], total_tokens[1])
    return recovered_x, recovered_y, warnings, token_tuple, gemini_used_any


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
    if len(ocr_tokens.x_tick_labels) < 2 or len(ocr_tokens.y_tick_labels) < 2:
        recovered_x, recovered_y, axis_warnings, axis_tokens, axis_gemini_used = (
            _recover_axis_ticks_from_crops(
                image_path=image_path,
                confidence_threshold=cfg.tiered_confidence_threshold,
                similarity_threshold=cfg.tiered_similarity_threshold,
                timeout=min(cfg.api_timeout_seconds, FAST_PHASE_TIMEOUT_SECONDS),
                max_retries=FAST_PHASE_MAX_RETRIES,
                skip_verification=cfg.single_model_mode,
            )
        )
        warnings.extend(axis_warnings)
        if axis_tokens is not None:
            total_cost += _calculate_cost(axis_tokens, axis_gemini_used)
        merged_x = _merge_text_labels(ocr_tokens.x_tick_labels, recovered_x)
        merged_y = _merge_text_labels(ocr_tokens.y_tick_labels, recovered_y)
        if merged_x != ocr_tokens.x_tick_labels or merged_y != ocr_tokens.y_tick_labels:
            warnings.append("Recovered additional axis tick labels from axis-specific OCR crops")
            ocr_tokens = ocr_tokens.model_copy(
                update={
                    "x_tick_labels": merged_x,
                    "y_tick_labels": merged_y,
                }
            )

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
    plot_metadata = _infer_curve_direction(plot_metadata, ocr_tokens, warnings)

    # Risk-table recovery pass (quality-aware):
    # 1) Always attempt OCR parsing when tokens are available.
    # 2) Replace metadata table only when OCR quality is materially better.
    # 3) If still weak/missing, run cropped lower-region OCR.
    if ocr_tokens is not None:
        curve_names = [c.name for c in plot_metadata.curves]
        best_rt = plot_metadata.risk_table
        best_score = _risk_table_quality(best_rt)

        parsed_rt = _parse_risk_table_from_ocr(ocr_tokens, curve_names)
        parsed_score = _risk_table_quality(parsed_rt)
        if parsed_rt is not None and parsed_score > best_score + RISK_TABLE_REPLACEMENT_MARGIN:
            best_rt = parsed_rt
            best_score = parsed_score
            if plot_metadata.risk_table is None:
                warnings.append("Recovered risk table from OCR tokens")
            else:
                warnings.append("Replaced low-quality metadata risk table using OCR tokens")

        if _risk_table_needs_recovery(best_rt, curve_names):
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
            crop_score = _risk_table_quality(crop_rt)
            if crop_rt is not None and crop_score > best_score + RISK_TABLE_REPLACEMENT_MARGIN:
                best_rt = crop_rt
                best_score = crop_score
                if plot_metadata.risk_table is None and parsed_rt is None:
                    warnings.append("Recovered risk table from cropped lower-region OCR")
                else:
                    warnings.append("Replaced risk table using cropped lower-region OCR")

        if best_rt is not None and best_rt is not plot_metadata.risk_table:
            plot_metadata = plot_metadata.model_copy(update={"risk_table": best_rt})

    plot_metadata = _maybe_correct_y_axis_start(plot_metadata, ocr_tokens, image_path, warnings)
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
    if len(ocr_tokens.x_tick_labels) < 2 or len(ocr_tokens.y_tick_labels) < 2:
        recovered_x, recovered_y, axis_warnings, axis_tokens, axis_gemini_used = (
            await _recover_axis_ticks_from_crops_async(
                image_path=image_path,
                confidence_threshold=cfg.tiered_confidence_threshold,
                similarity_threshold=cfg.tiered_similarity_threshold,
                timeout=min(cfg.api_timeout_seconds, FAST_PHASE_TIMEOUT_SECONDS),
                max_retries=FAST_PHASE_MAX_RETRIES,
                skip_verification=cfg.single_model_mode,
            )
        )
        warnings.extend(axis_warnings)
        if axis_tokens is not None:
            total_cost += _calculate_cost(axis_tokens, axis_gemini_used)
        merged_x = _merge_text_labels(ocr_tokens.x_tick_labels, recovered_x)
        merged_y = _merge_text_labels(ocr_tokens.y_tick_labels, recovered_y)
        if merged_x != ocr_tokens.x_tick_labels or merged_y != ocr_tokens.y_tick_labels:
            warnings.append("Recovered additional axis tick labels from axis-specific OCR crops")
            ocr_tokens = ocr_tokens.model_copy(
                update={
                    "x_tick_labels": merged_x,
                    "y_tick_labels": merged_y,
                }
            )

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
    plot_metadata = _infer_curve_direction(plot_metadata, ocr_tokens, warnings)

    # Async risk-table recovery mirrors sync quality-aware behavior.
    if ocr_tokens is not None:
        curve_names = [c.name for c in plot_metadata.curves]
        best_rt = plot_metadata.risk_table
        best_score = _risk_table_quality(best_rt)

        parsed_rt = _parse_risk_table_from_ocr(ocr_tokens, curve_names)
        parsed_score = _risk_table_quality(parsed_rt)
        if parsed_rt is not None and parsed_score > best_score + RISK_TABLE_REPLACEMENT_MARGIN:
            best_rt = parsed_rt
            best_score = parsed_score
            if plot_metadata.risk_table is None:
                warnings.append("Recovered risk table from OCR tokens")
            else:
                warnings.append("Replaced low-quality metadata risk table using OCR tokens")

        if _risk_table_needs_recovery(best_rt, curve_names):
            image = cv_utils.load_image(image_path, stage=ProcessingStage.MMPU)
            if not isinstance(image, ProcessingError):
                h = image.shape[0]
                tmp_dir = Path(gettempdir()) / "km_estimator"
                tmp_dir.mkdir(parents=True, exist_ok=True)
                crop_best_rt: RiskTable | None = None
                crop_best_score = -1.0
                for ratio in RISK_TABLE_CROP_RATIOS:
                    y0 = max(0, int(h * ratio))
                    crop = image[y0:, :]
                    if crop.size == 0:
                        continue
                    crop = cv_utils.preprocess_risk_table_ocr_region(crop)
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
                            if crop_rt is not None and score > crop_best_score:
                                crop_best_score = score
                                crop_best_rt = crop_rt
                    finally:
                        try:
                            crop_path.unlink(missing_ok=True)
                        except OSError:
                            pass
                if (
                    crop_best_rt is not None
                    and crop_best_score > best_score + RISK_TABLE_REPLACEMENT_MARGIN
                ):
                    best_rt = crop_best_rt
                    best_score = crop_best_score
                    if plot_metadata.risk_table is None and parsed_rt is None:
                        warnings.append("Recovered risk table from cropped lower-region OCR")
                    else:
                        warnings.append("Replaced risk table using cropped lower-region OCR")

        if best_rt is not None and best_rt is not plot_metadata.risk_table:
            plot_metadata = plot_metadata.model_copy(update={"risk_table": best_rt})

    plot_metadata = _maybe_correct_y_axis_start(plot_metadata, ocr_tokens, image_path, warnings)
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
