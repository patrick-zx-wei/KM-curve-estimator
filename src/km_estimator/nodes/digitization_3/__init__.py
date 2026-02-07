"""Digitization v3 pipeline.

Design goals:
- single shared PlotModel (no downstream axis guessing)
- shared evidence cube
- joint tracing for 2-arm plots
- confidence gating (abstain over low-confidence outputs)
"""

from __future__ import annotations

import json
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
DIGITIZER_HARDPOINT_ENABLE_ENV = "KM_DIGITIZER_HARDPOINTS_ENABLE"
DIGITIZER_HARDPOINT_PATH_ENV = "KM_DIGITIZER_HARDPOINTS_PATH"


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


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
    if any(w.startswith("W_DIRECTION_AMBIGUOUS_TEXT") for w in unique_warning_codes):
        adjusted -= 0.18
        warnings.append("W_DIRECTION_AMBIGUOUS_PENALTY")
    if any(w.startswith("I_DIRECTION_FALLBACK_METADATA:unknown") for w in unique_warning_codes):
        hard_fail = True
        warnings.append("E_HARD_FAIL_DIRECTION_UNKNOWN")
    if any(w.startswith("W_CROSSING_DISABLED_CAP:") for w in unique_warning_codes):
        hard_fail = True
        warnings.append("E_HARD_FAIL_CROSSING_OVERFIRE")
    adjusted = float(np.clip(adjusted, 0.0, 1.0))

    if low_arms:
        warnings.append("W_LOW_ARM_CONFIDENCE_SET:" + ",".join(sorted(low_arms)))

    # Do not block downstream evaluation for most low-confidence traces.
    # Reserve hard failure for critically unreliable traces with poor coverage.
    should_fail = bool(
        adjusted < EXTREME_FAIL_THRESHOLD
        and (span_mean < 0.35 or low_frac >= 0.95)
    )
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


def _load_hardpoint_constraints(state: PipelineState) -> tuple[dict[str, list[tuple[float, float]]], list[str]]:
    """Load optional hardpoint constraints for digitization-stage anchoring."""
    warnings: list[str] = []
    enabled = _env_bool(DIGITIZER_HARDPOINT_ENABLE_ENV, default=False)

    env_path = os.getenv(DIGITIZER_HARDPOINT_PATH_ENV)
    hp_path = Path(env_path) if env_path else Path(state.image_path).with_name("hard_points.json")

    if not enabled and not hp_path.exists():
        return {}, warnings
    if not hp_path.exists():
        warnings.append(f"W_HARDPOINT_FILE_MISSING:{hp_path}")
        return {}, warnings
    if not enabled:
        warnings.append(f"I_HARDPOINT_AUTO_ENABLED:{hp_path}")

    try:
        payload = json.loads(hp_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        warnings.append(f"W_HARDPOINT_LOAD_FAILED:{exc}")
        return {}, warnings
    if not isinstance(payload, dict):
        warnings.append("W_HARDPOINT_INVALID_PAYLOAD")
        return {}, warnings

    constraints: dict[str, list[tuple[float, float]]] = {}
    for group_name, group_payload in payload.items():
        if not isinstance(group_name, str) or not isinstance(group_payload, dict):
            continue
        landmarks = group_payload.get("landmarks", [])
        if not isinstance(landmarks, list):
            continue
        pts: list[tuple[float, float]] = []
        for lm in landmarks:
            if not isinstance(lm, dict):
                continue
            t = lm.get("time")
            s = lm.get("survival")
            if not isinstance(t, (int, float)) or not isinstance(s, (int, float)):
                continue
            pts.append((float(t), float(np.clip(float(s), 0.0, 1.0))))
        if pts:
            pts.sort(key=lambda p: p[0])
            constraints[group_name] = pts
    warnings.append(f"I_HARDPOINT_LOADED_GROUPS:{len(constraints)}")
    return constraints, warnings


def _survival_to_plot_y(
    survival: float,
    curve_direction: str,
    y_start: float,
    y_end: float,
) -> float:
    s = float(np.clip(float(survival), 0.0, 1.0))
    y_abs_max = float(max(abs(y_start), abs(y_end)))
    use_percent = y_abs_max > 1.5
    if curve_direction == "upward":
        incidence = 1.0 - s
        if use_percent:
            return float(y_start + incidence * (y_end - y_start))
        return float(np.clip(incidence, min(y_start, y_end), max(y_start, y_end)))
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
    direction = plot_model.curve_direction

    for name, pts in hardpoints.items():
        if not pts:
            continue
        by_col: dict[int, list[int]] = {}
        for t, s in pts:
            y_plot = _survival_to_plot_y(s, direction, y_start, y_end)
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


def digitize_v3(state: PipelineState) -> PipelineState:
    """Run the new digitization_3 pipeline."""
    if state.plot_metadata is None:
        err = ProcessingError(
            stage=ProcessingStage.DIGITIZE,
            error_type="no_metadata",
            recoverable=False,
            message="PlotMetadata required for digitization_v3",
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

    color_models, color_warnings = build_color_models(image, state.plot_metadata, plot_model)
    all_warnings.extend(color_warnings)

    evidence = build_evidence_cube(image, plot_model, color_models)
    all_warnings.extend(evidence.warning_codes)

    hardpoint_constraints, hardpoint_warnings = _load_hardpoint_constraints(state)
    all_warnings.extend(hardpoint_warnings)
    hardpoint_guides, guide_warnings = _build_hardpoint_guides(
        plot_model=plot_model,
        hardpoints=hardpoint_constraints,
    )
    all_warnings.extend(guide_warnings)

    x0, y0, _, _ = plot_model.plot_region
    trace = trace_curves(
        arm_score_maps=evidence.arm_score_maps,
        evidence=evidence,
        direction=plot_model.curve_direction,
        direction_confidence=plot_model.direction_confidence,
        x0=x0,
        y0=y0,
        candidate_mask=evidence.candidate_mask,
        arm_candidate_masks=evidence.arm_candidate_masks,
        hardpoint_guides=hardpoint_guides,
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
            arm_candidate_masks=evidence.arm_candidate_masks,
            hardpoint_guides=hardpoint_guides,
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
    all_warnings = _dedupe_ordered(all_warnings)

    if should_fail:
        err = ProcessingError(
            stage=ProcessingStage.DIGITIZE,
            error_type="digitization_low_confidence",
            recoverable=True,
            message=f"digitization_v3 confidence too low ({final_conf:.3f})",
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
    debug_enabled = os.getenv("KM_DIGITIZER_V3_DEBUG", "").lower() in {"1", "true", "yes"}
    if not debug_enabled:
        debug_enabled = os.getenv("KM_DIGITIZER_V2_DEBUG", "").lower() in {"1", "true", "yes"}
    if debug_enabled:
        out_dir = Path(
            os.getenv(
                "KM_DIGITIZER_V3_DEBUG_DIR",
                os.getenv("KM_DIGITIZER_V2_DEBUG_DIR", "/tmp/km_digitization_v3"),
            )
        )
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
    all_warnings.append(f"I_DIGITIZATION_V3_CONFIDENCE:{final_conf:.3f}")
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


__all__ = ["digitize_v3"]
