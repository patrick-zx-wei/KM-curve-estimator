"""Digitization v2 pipeline.

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


def _gate_confidence(
    trace_confidence: float,
    arm_confidences: dict[str, float],
    arm_diagnostics: dict[str, dict[str, float]],
    warning_codes: list[str],
    pixel_curves: dict[str, list[tuple[int, int]]],
    plot_region: tuple[int, int, int, int],
) -> tuple[float, list[str], bool]:
    warnings: list[str] = []
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

        diag_penalty += (
            0.60 * axis_cap
            + 0.35 * low_margin
            + 0.25 * jump_rate
            + 0.55 * mono_mass
            + 0.25 * overlap_conf
        )
    if n_diag > 0:
        adjusted -= diag_penalty / float(n_diag)

    # Penalize known low-trust warning conditions.
    rejected = [w for w in warning_codes if w.startswith("W_COLOR_PRIOR_REJECTED:")]
    if rejected:
        adjusted -= min(0.20, 0.05 * len(rejected))
        warnings.append(f"W_COLOR_REJECT_PENALTY:{len(rejected)}")
    if any(w.startswith("W_COLOR_UNINFORMATIVE") for w in warning_codes):
        adjusted -= 0.15
        warnings.append("W_COLOR_UNINFORMATIVE_PENALTY")
    if any(w.startswith("W_DIRECTION_AMBIGUOUS_TEXT") for w in warning_codes):
        adjusted -= 0.18
        warnings.append("W_DIRECTION_AMBIGUOUS_PENALTY")
    if any(w.startswith("I_DIRECTION_FALLBACK_METADATA:unknown") for w in warning_codes):
        hard_fail = True
        warnings.append("E_HARD_FAIL_DIRECTION_UNKNOWN")
    if any(w.startswith("W_CROSSING_DISABLED_CAP:") for w in warning_codes):
        hard_fail = True
        warnings.append("E_HARD_FAIL_CROSSING_OVERFIRE")
    adjusted = float(np.clip(adjusted, 0.0, 1.0))

    if low_arms:
        warnings.append("W_LOW_ARM_CONFIDENCE_SET:" + ",".join(sorted(low_arms)))

    should_fail = (adjusted < EXTREME_FAIL_THRESHOLD) or (
        hard_fail and adjusted < HARD_FAIL_MIN_CONFIDENCE
    ) or (adjusted < LOW_CONFIDENCE_FAIL_THRESHOLD)
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


def digitize_v2(state: PipelineState) -> PipelineState:
    """Run the new digitization_2 pipeline."""
    if state.plot_metadata is None:
        err = ProcessingError(
            stage=ProcessingStage.DIGITIZE,
            error_type="no_metadata",
            recoverable=False,
            message="PlotMetadata required for digitization_v2",
        )
        return state.model_copy(update={"errors": state.errors + [err]})

    image_path = state.preprocessed_image_path or state.image_path
    image = cv_utils.load_image(image_path, stage=ProcessingStage.DIGITIZE)
    if isinstance(image, ProcessingError):
        return state.model_copy(update={"errors": state.errors + [image]})

    all_warnings: list[str] = list(state.mmpu_warnings)

    plot_model = build_plot_model(
        image,
        state.plot_metadata,
        state.ocr_tokens,
        source_image_path=state.image_path,
    )
    if isinstance(plot_model, ProcessingError):
        return state.model_copy(update={"errors": state.errors + [plot_model]})
    all_warnings.extend(plot_model.warning_codes)

    color_models, color_warnings = build_color_models(image, state.plot_metadata, plot_model)
    all_warnings.extend(color_warnings)

    evidence = build_evidence_cube(image, plot_model, color_models)
    all_warnings.extend(evidence.warning_codes)

    x0, y0, _, _ = plot_model.plot_region
    trace = trace_curves(
        arm_score_maps=evidence.arm_score_maps,
        evidence=evidence,
        direction=plot_model.curve_direction,
        direction_confidence=plot_model.direction_confidence,
        x0=x0,
        y0=y0,
        candidate_mask=evidence.candidate_mask,
    )
    all_warnings.extend(trace.warning_codes)

    # Convert to reconstruction-compatible survival coordinates.
    digitized_curves, post_warnings = convert_pixel_curves_to_survival(
        trace.pixel_curves,
        plot_model=plot_model,
        direction=plot_model.curve_direction,
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
            direction=plot_model.curve_direction,
            direction_confidence=plot_model.direction_confidence,
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
        )
        all_warnings.extend(retrace.warning_codes)
        retrace_curves, retrace_post_warnings = convert_pixel_curves_to_survival(
            retrace.pixel_curves,
            plot_model=plot_model,
            direction=plot_model.curve_direction,
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

    if should_fail:
        err = ProcessingError(
            stage=ProcessingStage.DIGITIZE,
            error_type="digitization_low_confidence",
            recoverable=True,
            message=f"digitization_v2 confidence too low ({final_conf:.3f})",
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
        from km_estimator.nodes.digitization.censoring_detection import detect_censoring

        censoring = detect_censoring(
            image=image,
            curves=trace.pixel_curves,
            mapping=plot_model.mapping,
        )
    except Exception as exc:  # pragma: no cover - optional dependency path
        censoring = {name: [] for name in trace.pixel_curves}
        all_warnings.append(f"W_CENSORING_DETECTION_UNAVAILABLE:{exc}")

    # Optional debug artifacts.
    if os.getenv("KM_DIGITIZER_V2_DEBUG", "").lower() in {"1", "true", "yes"}:
        out_dir = Path(os.getenv("KM_DIGITIZER_V2_DEBUG_DIR", "/tmp/km_digitization_v2"))
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
    all_warnings.append(f"I_DIGITIZATION_V2_CONFIDENCE:{final_conf:.3f}")
    return state.model_copy(
        update={
            "digitized_curves": digitized_curves,
            "isolated_curve_pixels": trace.pixel_curves,
            "censoring_marks": censoring,
            "mmpu_warnings": all_warnings,
            "flagged_for_review": flagged,
        }
    )


__all__ = ["digitize_v2"]
