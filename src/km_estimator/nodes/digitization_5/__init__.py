"""Digitization v5 pipeline.

Design goals:
- single shared PlotModel (no downstream axis guessing)
- shared evidence cube
- joint tracing for 2-arm plots
- confidence gating (abstain over low-confidence outputs)
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from km_estimator.models import PipelineState, ProcessingError, ProcessingStage
from km_estimator.utils import cv_utils

from .axis_map import build_plot_model
from .debug import write_debug_artifacts
from .legend_color import build_color_models
from .path_trace import TraceConfig, trace_curves
from .postprocess import convert_pixel_curves_to_survival
from .probability_map import build_evidence_cube

LOW_CONFIDENCE_FAIL_THRESHOLD = 0.30
LOW_CONFIDENCE_REVIEW_THRESHOLD = 0.48
HARD_FAIL_MIN_CONFIDENCE = 0.55
EXTREME_FAIL_THRESHOLD = 0.12


def _dedupe_ordered(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out


def _seed_upstream_warnings(prev: list[str]) -> list[str]:
    """
    Keep only upstream/non-coded warnings across retries.

    Digitizer-coded warnings (I_/W_/E_) are regenerated each run and should
    not compound confidence penalties on retry.
    """
    seeded: list[str] = []
    for w in prev:
        if not isinstance(w, str):
            continue
        if w.startswith(("I_", "W_", "E_")):
            continue
        seeded.append(w)
    return _dedupe_ordered(seeded)


def _gate_confidence(
    trace_confidence: float,
    arm_confidences: dict[str, float],
    arm_diagnostics: dict[str, dict[str, float]],
    warning_codes: list[str],
    pixel_curves: dict[str, list[tuple[int, int]]],
    plot_region: tuple[int, int, int, int],
) -> tuple[float, list[str], bool]:
    warnings: list[str] = []
    unique_warning_codes = _dedupe_ordered([w for w in warning_codes if isinstance(w, str)])
    if not arm_confidences:
        return 0.0, ["W_NO_ARM_CONFIDENCE"], True

    low_arms = [name for name, score in arm_confidences.items() if score < 0.45]
    low_frac = float(len(low_arms)) / float(max(1, len(arm_confidences)))

    x0, _, x1, _ = plot_region
    width = max(1, x1 - x0)
    span_scores: list[float] = []
    for name, pts in pixel_curves.items():
        if not pts:
            continue
        xs = [px for px, _ in pts]
        span = float(max(xs) - min(xs)) / float(width)
        span_scores.append(span)
        if span < 0.55:
            warnings.append(f"W_LOW_X_COVERAGE:{name}:{span:.3f}")
    span_mean = float(np.mean(np.asarray(span_scores, dtype=np.float32))) if span_scores else 0.0

    adjusted = float(trace_confidence)
    adjusted *= max(0.20, 1.0 - 0.30 * low_frac)
    adjusted *= max(0.20, 0.60 + 0.40 * span_mean)

    hard_fail = False
    diag_penalty = 0.0
    n_diag = 0
    for name, diag in arm_diagnostics.items():
        n_diag += 1
        axis_cap = float(diag.get("axis_capture_frac", 0.0))
        low_margin = float(diag.get("low_margin_frac", 0.0))
        jump_rate = float(diag.get("jump_rate", 0.0))
        max_jump_ratio = float(diag.get("max_jump_ratio", 0.0))
        extreme_jump_frac = float(diag.get("extreme_jump_frac", 0.0))
        mono_mass = float(diag.get("monotone_violation_mass", 0.0))
        overlap_conf = float(diag.get("overlap_conflict_frac", 0.0))

        if axis_cap > 0.03:
            hard_fail = True
            warnings.append(f"E_HARD_FAIL_AXIS_CAPTURE:{name}:{axis_cap:.3f}")
        if low_margin > 0.35:
            hard_fail = True
            warnings.append(f"E_HARD_FAIL_LOW_MARGIN:{name}:{low_margin:.3f}")
        if mono_mass > 0.08:
            hard_fail = True
            warnings.append(f"E_HARD_FAIL_MONOTONE_MASS:{name}:{mono_mass:.3f}")
        if extreme_jump_frac > 0.003 or max_jump_ratio > 0.12:
            hard_fail = True
            warnings.append(
                f"E_HARD_FAIL_EXTREME_JUMP:{name}:{extreme_jump_frac:.4f}:{max_jump_ratio:.3f}"
            )

        diag_penalty += (
            0.60 * axis_cap
            + 0.35 * low_margin
            + 0.25 * jump_rate
            + 0.65 * extreme_jump_frac
            + 0.35 * max_jump_ratio
            + 0.55 * mono_mass
            + 0.25 * overlap_conf
        )
    if n_diag > 0:
        adjusted -= diag_penalty / float(n_diag)

    # Penalize known low-trust warning conditions.
    rejected = [w for w in unique_warning_codes if w.startswith("W_COLOR_PRIOR_REJECTED:")]
    if rejected:
        adjusted -= min(0.20, 0.05 * len(rejected))
        warnings.append(f"W_COLOR_REJECT_PENALTY:{len(rejected)}")
    if any(w.startswith("W_COLOR_UNINFORMATIVE") for w in unique_warning_codes):
        adjusted -= 0.15
        warnings.append("W_COLOR_UNINFORMATIVE_PENALTY")
    if any(w.startswith("W_CROSSING_DISABLED_CAP:") for w in unique_warning_codes):
        hard_fail = True
        warnings.append("E_HARD_FAIL_CROSSING_OVERFIRE")
    adjusted = float(np.clip(adjusted, 0.0, 1.0))

    if low_arms:
        warnings.append("W_LOW_ARM_CONFIDENCE_SET:" + ",".join(sorted(low_arms)))

    # Do not block downstream evaluation for most low-confidence traces.
    # Reserve hard failure for critically unreliable traces with poor coverage.
    should_fail = bool(adjusted < EXTREME_FAIL_THRESHOLD and (span_mean < 0.35 or low_frac >= 0.95))
    if hard_fail and not should_fail:
        warnings.append("W_HARD_FAIL_REVIEW_ONLY")
    if adjusted < LOW_CONFIDENCE_REVIEW_THRESHOLD:
        warnings.append(f"W_NEEDS_MANUAL_REVIEW:{adjusted:.3f}")
    return adjusted, warnings, should_fail


def _normalize01(arr: np.ndarray) -> np.ndarray:
    lo = float(np.min(arr))
    hi = float(np.max(arr))
    if hi <= lo + 1e-9:
        return np.zeros_like(arr, dtype=np.float32)
    return ((arr - lo) / (hi - lo)).astype(np.float32)


def _normalize_group_name(name: str) -> str:
    return "".join(ch for ch in name.lower().strip() if ch.isalnum())


def _match_risk_group_name(curve_name: str, risk_group_names: list[str]) -> str | None:
    """Match curve name to risk-table group name with deterministic fuzzy fallback."""
    if curve_name in risk_group_names:
        return curve_name
    curve_norm = _normalize_group_name(curve_name)
    for group_name in risk_group_names:
        if _normalize_group_name(group_name) == curve_norm:
            return group_name
    for group_name in risk_group_names:
        group_norm = _normalize_group_name(group_name)
        if curve_norm in group_norm or group_norm in curve_norm:
            return group_name
    return None


def _build_risk_table_constraints(
    state: PipelineState,
) -> tuple[dict[str, list[tuple[float, float]]], list[str]]:
    """
    Build curve constraints from extracted risk table only.

    These points are survival lower-bounds derived from n(t)/n0.
    """
    warnings: list[str] = []
    meta = state.plot_metadata
    if meta is None or meta.risk_table is None:
        warnings.append("I_HARDPOINT_SOURCE_RISK_TABLE:none")
        return {}, warnings

    rt = meta.risk_table
    if not rt.time_points or not rt.groups:
        warnings.append("W_HARDPOINT_SOURCE_RISK_TABLE_EMPTY")
        return {}, warnings

    group_map = {g.name: g for g in rt.groups}
    group_names = list(group_map)
    constraints: dict[str, list[tuple[float, float]]] = {}
    unmatched: list[str] = []
    invalid_groups = 0

    for curve in meta.curves:
        curve_name = curve.name
        matched_group_name = _match_risk_group_name(curve_name, group_names)
        if matched_group_name is None:
            unmatched.append(curve_name)
            continue

        group = group_map[matched_group_name]
        counts = list(group.counts)
        if len(counts) != len(rt.time_points) or not counts or int(counts[0]) <= 0:
            invalid_groups += 1
            warnings.append(
                f"W_HARDPOINT_RISK_GROUP_INVALID:{curve_name}:{matched_group_name}:"
                f"{len(counts)}vs{len(rt.time_points)}"
            )
            continue

        n0 = float(counts[0])
        pts: list[tuple[float, float]] = []
        for t, c in zip(rt.time_points, counts):
            if not isinstance(t, (int, float)) or not isinstance(c, (int, float)):
                continue
            s_floor = float(np.clip(float(c) / n0, 0.0, 1.0))
            pts.append((float(t), s_floor))
        if pts:
            pts.sort(key=lambda p: p[0])
            constraints[curve_name] = pts

    warnings.append(f"I_HARDPOINT_SOURCE_RISK_TABLE:groups={len(constraints)}")
    if unmatched:
        warnings.append("W_HARDPOINT_RISK_GROUP_MISSING:" + ",".join(sorted(unmatched)))
    if invalid_groups > 0:
        warnings.append(f"W_HARDPOINT_RISK_GROUP_INVALID_COUNT:{invalid_groups}")
    return constraints, warnings


def _survival_to_plot_y(
    survival: float,
    y_start: float,
    y_end: float,
) -> float:
    s = float(np.clip(float(survival), 0.0, 1.0))
    y_abs_max = float(max(abs(y_start), abs(y_end)))
    use_percent = y_abs_max > 1.5
    if use_percent:
        return float(y_start + s * (y_end - y_start))
    return float(np.clip(s, min(y_start, y_end), max(y_start, y_end)))


def _build_hardpoint_guides(
    plot_model,
    hardpoints: dict[str, list[tuple[float, float]]],
) -> tuple[dict[str, tuple[tuple[int, int], ...]], list[str]]:
    warnings: list[str] = []
    guides: dict[str, tuple[tuple[int, int], ...]] = {}
    if not hardpoints:
        return guides, warnings

    x0, y0, x1, y1 = plot_model.plot_region
    w = max(1, x1 - x0)
    h = max(1, y1 - y0)
    y_start = float(plot_model.mapping.y_axis.start)
    y_end = float(plot_model.mapping.y_axis.end)

    for name, pts in hardpoints.items():
        if not pts:
            continue
        by_col: dict[int, list[int]] = {}
        for t, s in pts:
            y_plot = _survival_to_plot_y(s, y_start, y_end)
            px, py = plot_model.real_to_px(float(t), float(y_plot))
            cx = int(np.clip(px - x0, 0, w - 1))
            cy = int(np.clip(py - y0, 0, h - 1))
            by_col.setdefault(cx, []).append(cy)
        if not by_col:
            continue
        anchors = tuple(
            (int(col), int(round(float(np.median(np.asarray(rows, dtype=np.float32))))))
            for col, rows in sorted(by_col.items())
        )
        guides[name] = anchors
        warnings.append(f"I_HARDPOINT_GUIDE:{name}:{len(anchors)}")

    return guides, warnings


def _trim_unsupported_right_edge(
    pixel_curves: dict[str, list[tuple[int, int]]],
    arm_candidate_masks: dict[str, np.ndarray],
    max_gap: int = 5,
    min_coverage: float = 0.80,
) -> dict[str, list[tuple[int, int]]]:
    """Remove traced points beyond the last column with arm mask support.

    At the right edge of full-box plots, frame lines and JPEG compression can
    destroy curve pixels so the arm mask has zero coverage.  The tracer still
    produces rows for those columns, but they are unreliable.  Trimming them
    lets downstream evaluation fall back to the last well-supported value.

    Only applied when the mask has broad coverage (>= min_coverage of columns)
    but drops out at the very end — this indicates a frame/edge artifact, not a
    sparse curve.
    """
    trimmed: dict[str, list[tuple[int, int]]] = {}
    for name, pts in pixel_curves.items():
        mask = arm_candidate_masks.get(name)
        if mask is None or not pts:
            trimmed[name] = pts
            continue
        col_has_mask = np.any(mask, axis=0)
        cols_with_mask = np.where(col_has_mask)[0]
        if cols_with_mask.size == 0:
            trimmed[name] = pts
            continue
        total_cols = mask.shape[1]
        coverage = float(cols_with_mask.size) / float(max(1, total_cols))
        last_supported = int(cols_with_mask[-1])
        right_gap = total_cols - 1 - last_supported
        # Only trim when the mask covers most columns but a small gap exists at
        # the right edge (frame line / JPEG damage).  Don't trim sparse curves.
        if coverage >= min_coverage and right_gap > 0 and right_gap <= int(total_cols * 0.03):
            cutoff = last_supported + max_gap
            trimmed[name] = [(c, r) for c, r in pts if c <= cutoff]
        else:
            trimmed[name] = pts
    return trimmed


def digitize_v5(state: PipelineState) -> PipelineState:
    """Run the new digitization_5 pipeline."""
    if state.plot_metadata is None:
        err = ProcessingError(
            stage=ProcessingStage.DIGITIZE,
            error_type="no_metadata",
            recoverable=False,
            message="PlotMetadata required for digitization_v5",
        )
        return state.model_copy(update={"errors": state.errors + [err]})

    image_path = state.preprocessed_image_path or state.image_path
    image = cv_utils.load_image(image_path, stage=ProcessingStage.DIGITIZE)
    if isinstance(image, ProcessingError):
        return state.model_copy(update={"errors": state.errors + [image]})

    all_warnings: list[str] = _seed_upstream_warnings(list(state.mmpu_warnings))

    plot_model = build_plot_model(
        image,
        state.plot_metadata,
        state.ocr_tokens,
    )
    if isinstance(plot_model, ProcessingError):
        return state.model_copy(update={"errors": state.errors + [plot_model]})
    all_warnings.extend(plot_model.warning_codes)

    # Keep risk-table hardpoint extraction for diagnostics only.
    # Do not guide tracing with these points: n(t)/n0 are lower bounds, not exact curve points.
    _hardpoint_constraints, hardpoint_warnings = _build_risk_table_constraints(state)
    all_warnings.extend(hardpoint_warnings)
    all_warnings.append("I_HARDPOINT_GUIDANCE_DISABLED_FOR_TRACING")

    color_models, color_warnings = build_color_models(image, state.plot_metadata, plot_model)
    all_warnings.extend(color_warnings)

    evidence = build_evidence_cube(
        image=image,
        plot_model=plot_model,
        color_models=color_models,
        hardpoint_guides=None,
    )
    all_warnings.extend(evidence.warning_codes)

    x0, y0, _, _ = plot_model.plot_region
    trace = trace_curves(
        arm_score_maps=evidence.arm_score_maps,
        evidence=evidence,
        x0=x0,
        y0=y0,
        candidate_mask=evidence.candidate_mask,
        arm_candidate_masks=evidence.arm_candidate_masks,
        hardpoint_guides=None,
    )
    all_warnings.extend(trace.warning_codes)

    # Trim traced points beyond the last column with arm mask support.
    trimmed_pixel_curves = _trim_unsupported_right_edge(
        trace.pixel_curves,
        evidence.arm_candidate_masks,
    )

    # Convert to reconstruction-compatible survival coordinates.
    digitized_curves, post_warnings = convert_pixel_curves_to_survival(
        trimmed_pixel_curves,
        plot_model=plot_model,
    )
    all_warnings.extend(post_warnings)

    # One deterministic retrace when validators scream.
    retrace_trigger = (
        any(w.startswith("W_CROSSING_DISABLED_CAP:") for w in all_warnings)
        or any(w.startswith("E_NEEDS_RETRACE_MONOTONE:") for w in all_warnings)
        or any(
            float(diag.get("axis_capture_frac", 0.0)) > 0.03
            for diag in trace.arm_diagnostics.values()
        )
    )
    if retrace_trigger:
        all_warnings.append("I_DETERMINISTIC_RETRACE_TRIGGERED")
        retrace_maps: dict[str, np.ndarray] = {}
        for name, arm_map in evidence.arm_score_maps.items():
            blended = 0.30 * arm_map + 0.70 * evidence.structure_map
            retrace_maps[name] = _normalize01(blended.astype(np.float32))
        retrace = trace_curves(
            arm_score_maps=retrace_maps,
            evidence=evidence,
            x0=x0,
            y0=y0,
            trace_config=TraceConfig(
                axis_multiplier=2.0,
                smoothness_multiplier=1.5,
                direction_multiplier=1.2,
                order_lock_multiplier=1.6,
                swap_multiplier=0.7,
            ),
            candidate_mask=evidence.candidate_mask,
            arm_candidate_masks=evidence.arm_candidate_masks,
            hardpoint_guides=None,
        )
        all_warnings.extend(retrace.warning_codes)
        retrace_trimmed = _trim_unsupported_right_edge(
            retrace.pixel_curves,
            evidence.arm_candidate_masks,
        )
        retrace_curves, retrace_post_warnings = convert_pixel_curves_to_survival(
            retrace_trimmed,
            plot_model=plot_model,
        )
        all_warnings.extend(retrace_post_warnings)
        if retrace.plot_confidence >= trace.plot_confidence:
            trace = retrace
            digitized_curves = retrace_curves
            all_warnings.append("I_DETERMINISTIC_RETRACE_ACCEPTED")
        else:
            all_warnings.append("I_DETERMINISTIC_RETRACE_SKIPPED")

    # Confidence gating: abstain over low-confidence traces.
    final_conf, gate_warnings, should_fail = _gate_confidence(
        trace_confidence=trace.plot_confidence,
        arm_confidences=trace.confidence_by_arm,
        arm_diagnostics=trace.arm_diagnostics,
        warning_codes=all_warnings,
        pixel_curves=trace.pixel_curves,
        plot_region=plot_model.plot_region,
    )
    all_warnings.extend(gate_warnings)
    all_warnings = _dedupe_ordered(all_warnings)

    if should_fail:
        err = ProcessingError(
            stage=ProcessingStage.DIGITIZE,
            error_type="digitization_low_confidence",
            recoverable=True,
            message=f"digitization_v5 confidence too low ({final_conf:.3f})",
            details={"confidence": final_conf},
        )
        return state.model_copy(
            update={
                "errors": state.errors + [err],
                "mmpu_warnings": all_warnings,
                "flagged_for_review": True,
            }
        )

    try:
        from km_estimator.nodes.digitization_5.censoring_detection import detect_censoring

        censoring = detect_censoring(
            image=image,
            curves=trace.pixel_curves,
            mapping=plot_model.mapping,
        )
    except Exception as exc:  # pragma: no cover - optional dependency path
        censoring = {name: [] for name in trace.pixel_curves}
        all_warnings.append(f"W_CENSORING_DETECTION_UNAVAILABLE:{exc}")

    # Optional debug artifacts.
    debug_enabled = os.getenv("KM_DIGITIZER_V5_DEBUG", "").lower() in {"1", "true", "yes"}
    if debug_enabled:
        out_dir = Path(os.getenv("KM_DIGITIZER_V5_DEBUG_DIR", "/tmp/km_digitization_v5"))
        prefix = Path(state.image_path).stem
        all_warnings.extend(
            write_debug_artifacts(
                image=image,
                plot_model=plot_model,
                evidence=evidence,
                trace=trace,
                out_dir=out_dir,
                prefix=prefix,
            )
        )

    flagged = state.flagged_for_review or (final_conf < LOW_CONFIDENCE_REVIEW_THRESHOLD)
    all_warnings.append(f"I_DIGITIZATION_V5_CONFIDENCE:{final_conf:.3f}")
    all_warnings = _dedupe_ordered(all_warnings)
    return state.model_copy(
        update={
            "digitized_curves": digitized_curves,
            "isolated_curve_pixels": trace.pixel_curves,
            "censoring_marks": censoring,
            "mmpu_warnings": all_warnings,
            "flagged_for_review": flagged,
        }
    )


__all__ = ["digitize_v5"]
