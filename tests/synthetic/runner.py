"""Test runner that interfaces with the actual KM pipeline.

Feeds synthetic graph.png images through run_pipeline() and compares
the output against ground truth at each pipeline stage.

Requires OPENAI_API_KEY and GEMINI_API_KEY environment variables.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import as_completed
import json
from pathlib import Path

import cv2
import numpy as np

from .data_gen import _km_from_ipd
from .ground_truth import (
    compare_digitized_curves,
    compare_hard_points,
    load_manifest,
    load_test_case,
)

DEFAULT_FIXED_CASES = [
    "case_016",
    "case_033",
    "case_050",
    "case_066",
    "case_083",
    "case_100",
]


def _tick_values_mae(detected: list[float], expected: list[float]) -> float | None:
    det = sorted(float(v) for v in detected)
    exp = sorted(float(v) for v in expected)
    n = min(len(det), len(exp))
    if n == 0:
        return None
    arr = np.abs(
        np.asarray(det[:n], dtype=np.float64) - np.asarray(exp[:n], dtype=np.float64)
    )
    return float(np.mean(arr))


def _tick_coverage(ticks: list[float], axis_start: float, axis_end: float) -> float | None:
    if len(ticks) < 2:
        return None
    lo = min(float(axis_start), float(axis_end))
    hi = max(float(axis_start), float(axis_end))
    span = hi - lo
    if span <= 1e-9:
        return None
    vals = sorted(float(v) for v in ticks)
    return float(np.clip((vals[-1] - vals[0]) / span, 0.0, 1.5))


def _overlay_curve_to_plot_space(
    coords: list[tuple[float, float]],
    curve_direction: str,
    y_start: float,
    y_end: float,
    assume_survival_space: bool,
) -> list[tuple[float, float]]:
    """Map curve coordinates into plotted y-space for visualization overlays."""
    if not coords:
        return []

    ordered = sorted((float(t), float(s)) for t, s in coords)
    y_abs_max = float(max(abs(y_start), abs(y_end)))
    percent_scale = 100.0 if (1.5 < y_abs_max <= 100.5) else 1.0
    y_lo = float(min(y_start, y_end))
    y_hi = float(max(y_start, y_end))

    def normalize_survival_like(val: float) -> float:
        if percent_scale > 1.0 and val > 1.5:
            return float(np.clip(val / percent_scale, 0.0, 1.0))
        return float(np.clip(val, 0.0, 1.0))

    out: list[tuple[float, float]] = []
    if curve_direction == "upward":
        # Upward plots are incidence-like: y_plot = 1 - S.
        # Reconstructed curves are in survival-space by construction.
        net = ordered[-1][1] - ordered[0][1]
        treat_as_survival = bool(assume_survival_space or (net < -0.02))
        for t, y in ordered:
            if treat_as_survival:
                s_norm = normalize_survival_like(y)
                y_plot = (1.0 - s_norm) * percent_scale
            else:
                if percent_scale > 1.0 and y <= 1.5:
                    y_plot = float(y * percent_scale)
                elif percent_scale <= 1.0 and y > 1.5:
                    y_plot = float(y / 100.0)
                else:
                    y_plot = float(y)
            out.append((float(t), float(np.clip(y_plot, y_lo, y_hi))))
        return out

    # Downward survival plot.
    for t, y in ordered:
        if percent_scale > 1.0 and y <= 1.5:
            y_plot = float(y * percent_scale)
        elif percent_scale <= 1.0 and y > 1.5:
            y_plot = float(y / 100.0)
        else:
            y_plot = float(y)
        out.append((float(t), float(np.clip(y_plot, y_lo, y_hi))))
    return out


def _curve_to_polyline_pixels(
    coords: list[tuple[float, float]],
    mapping,
    image_shape: tuple[int, int],
) -> np.ndarray | None:
    """Convert real-space curve coordinates into clipped image pixel polyline."""
    if len(coords) < 2:
        return None
    h, w = image_shape
    pts: list[tuple[int, int]] = []
    for t, y in coords:
        px, py = mapping.real_to_px(float(t), float(y))
        cx = int(np.clip(px, 0, w - 1))
        cy = int(np.clip(py, 0, h - 1))
        if not pts or (cx, cy) != pts[-1]:
            pts.append((cx, cy))
    if len(pts) < 2:
        return None
    return np.asarray(pts, dtype=np.int32).reshape(-1, 1, 2)


def _draw_axis_tick_overlay(
    overlay: np.ndarray,
    mapping,
    plot_metadata,
    x_anchor_px: list[int] | None = None,
    y_anchor_py: list[int] | None = None,
) -> None:
    """Draw calibrated axes and where the pipeline thinks tick marks are."""
    h, w = overlay.shape[:2]
    x_tick_color = (200, 0, 255)   # magenta
    y_tick_color = (0, 180, 255)   # orange
    axis_color = (70, 70, 70)

    x_axis = plot_metadata.x_axis
    y_axis = plot_metadata.y_axis
    x0, y_base = mapping.real_to_px(float(x_axis.start), float(y_axis.start))
    x1, _ = mapping.real_to_px(float(x_axis.end), float(y_axis.start))
    _, y1 = mapping.real_to_px(float(x_axis.start), float(y_axis.end))

    x0 = int(np.clip(x0, 0, w - 1))
    x1 = int(np.clip(x1, 0, w - 1))
    y_base = int(np.clip(y_base, 0, h - 1))
    y1 = int(np.clip(y1, 0, h - 1))

    axis_thickness = max(1, int(round(min(h, w) * 0.0025)))
    tick_len = max(6, int(round(min(h, w) * 0.011)))
    x_lo, x_hi = sorted((x0, x1))
    y_lo, y_hi = sorted((y1, y_base))

    # Draw calibrated axis spines used for px<->real mapping.
    cv2.line(
        overlay,
        (x_lo, y_base),
        (x_hi, y_base),
        axis_color,
        axis_thickness,
        cv2.LINE_AA,
    )
    cv2.line(
        overlay,
        (x0, min(y_base, y1)),
        (x0, max(y_base, y1)),
        axis_color,
        axis_thickness,
        cv2.LINE_AA,
    )

    x_positions: list[int] = []
    if x_anchor_px:
        x_positions = [int(np.clip(px, x_lo, x_hi)) for px in x_anchor_px]
    else:
        for xv in x_axis.tick_values:
            px, _ = mapping.real_to_px(float(xv), float(y_axis.start))
            x_positions.append(int(np.clip(px, x_lo, x_hi)))
    x_positions = sorted(set(x_positions))

    # X-axis ticks: redraw inward (upward) to avoid overlapping x tick labels.
    for i, cx in enumerate(x_positions):
        cy = y_base
        local_len = tick_len if 0 < i < len(x_positions) - 1 else max(4, int(round(tick_len * 0.65)))
        y_in = int(np.clip(cy - local_len, y_lo, y_hi))
        cv2.line(
            overlay,
            (int(cx), int(cy)),
            (int(cx), y_in),
            x_tick_color,
            max(1, axis_thickness),
            cv2.LINE_AA,
        )
        cv2.circle(overlay, (cx, cy), 2, x_tick_color, -1, cv2.LINE_AA)

    y_positions: list[int] = []
    if y_anchor_py:
        y_positions = [int(np.clip(py, y_lo, y_hi)) for py in y_anchor_py]
    else:
        for yv in y_axis.tick_values:
            _, py = mapping.real_to_px(float(x_axis.start), float(yv))
            y_positions.append(int(np.clip(py, y_lo, y_hi)))
    y_positions = sorted(set(y_positions), reverse=True)

    # Y-axis ticks: redraw inward (rightward) to avoid overlapping y tick labels.
    for i, cy in enumerate(y_positions):
        cx = x0
        local_len = tick_len if 0 < i < len(y_positions) - 1 else max(4, int(round(tick_len * 0.65)))
        x_in = int(np.clip(cx + local_len, x_lo, x_hi))
        cv2.line(
            overlay,
            (int(cx), int(cy)),
            (x_in, int(cy)),
            y_tick_color,
            max(1, axis_thickness),
            cv2.LINE_AA,
        )
        cv2.circle(overlay, (cx, cy), 2, y_tick_color, -1, cv2.LINE_AA)


def _write_overlay_artifact(state, case_dir: Path) -> dict:
    """
    Write overlay image showing digitized and reconstructed curves over graph.png.

    Returns artifact metadata for results.json.
    """
    if state.plot_metadata is None:
        return {"overlay_results": None, "error": "no plot_metadata"}

    image_path = case_dir / "graph.png"
    image = cv2.imread(str(image_path))
    if image is None:
        return {"overlay_results": None, "error": f"failed to load image: {image_path}"}

    # Lazy import keeps generation-only tasks lightweight.
    from km_estimator.models import ProcessingError
    from km_estimator.nodes.digitization.axis_calibration import calibrate_axes

    mapping = calibrate_axes(image, state.plot_metadata)
    if isinstance(mapping, ProcessingError):
        return {"overlay_results": None, "error": f"axis calibration failed: {mapping.message}"}

    x_tick_anchor_px: list[int] | None = None
    y_tick_anchor_py: list[int] | None = None
    try:
        import os

        digitizer = os.getenv("KM_DIGITIZER", "").strip().lower()
        if digitizer in {"v4", "4", "digitization_4"}:
            from km_estimator.nodes.digitization_4.axis_map import build_plot_model
        elif digitizer in {"v3", "3", "digitization_3"}:
            from km_estimator.nodes.digitization_3.axis_map import build_plot_model
        else:
            from km_estimator.nodes.digitization_2.axis_map import build_plot_model

        plot_model = build_plot_model(image=image, meta=state.plot_metadata, ocr_tokens=state.ocr_tokens)
        if not isinstance(plot_model, ProcessingError):
            if len(plot_model.x_tick_anchors) >= 2:
                x_tick_anchor_px = [int(px) for px, _ in plot_model.x_tick_anchors]
            if len(plot_model.y_tick_anchors) >= 2:
                y_tick_anchor_py = [int(py) for py, _ in plot_model.y_tick_anchors]
    except Exception:
        # Debug overlay should not fail the test run.
        pass

    overlay = image.copy()
    curve_direction = (
        state.plot_metadata.curve_direction
        if state.plot_metadata.curve_direction in ("downward", "upward")
        else "downward"
    )
    y_start = float(state.plot_metadata.y_axis.start)
    y_end = float(state.plot_metadata.y_axis.end)

    # OpenCV BGR colors.
    digitized_color = (255, 220, 0)      # cyan-ish
    reconstructed_color = (0, 200, 40)   # green

    if isinstance(state.digitized_curves, dict):
        for coords in state.digitized_curves.values():
            plot_coords = _overlay_curve_to_plot_space(
                coords=coords,
                curve_direction=curve_direction,
                y_start=y_start,
                y_end=y_end,
                assume_survival_space=False,
            )
            poly = _curve_to_polyline_pixels(plot_coords, mapping, overlay.shape[:2])
            if poly is not None:
                cv2.polylines(overlay, [poly], False, digitized_color, 1, cv2.LINE_AA)

    if state.output is not None:
        for curve in state.output.curves:
            km_coords = _km_from_ipd(curve.patients)
            plot_coords = _overlay_curve_to_plot_space(
                coords=km_coords,
                curve_direction=curve_direction,
                y_start=y_start,
                y_end=y_end,
                assume_survival_space=True,
            )
            poly = _curve_to_polyline_pixels(plot_coords, mapping, overlay.shape[:2])
            if poly is not None:
                cv2.polylines(overlay, [poly], False, reconstructed_color, 2, cv2.LINE_AA)

    _draw_axis_tick_overlay(
        overlay=overlay,
        mapping=mapping,
        plot_metadata=state.plot_metadata,
        x_anchor_px=x_tick_anchor_px,
        y_anchor_py=y_tick_anchor_py,
    )

    # Minimal legend
    x_tick_color = (200, 0, 255)   # magenta
    y_tick_color = (0, 180, 255)   # orange
    cv2.rectangle(overlay, (12, 12), (320, 98), (255, 255, 255), -1)
    cv2.rectangle(overlay, (12, 12), (320, 98), (40, 40, 40), 1)
    cv2.line(overlay, (24, 30), (58, 30), digitized_color, 1, cv2.LINE_AA)
    cv2.putText(overlay, "digitized", (66, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (20, 20, 20), 1, cv2.LINE_AA)
    cv2.line(overlay, (24, 48), (58, 48), reconstructed_color, 2, cv2.LINE_AA)
    cv2.putText(overlay, "reconstructed", (66, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (20, 20, 20), 1, cv2.LINE_AA)
    cv2.circle(overlay, (41, 66), 3, x_tick_color, -1, cv2.LINE_AA)
    cv2.line(overlay, (41, 59), (41, 73), x_tick_color, 1, cv2.LINE_AA)
    cv2.putText(overlay, "x-axis tick redraw", (66, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (20, 20, 20), 1, cv2.LINE_AA)
    cv2.circle(overlay, (41, 84), 3, y_tick_color, -1, cv2.LINE_AA)
    cv2.line(overlay, (34, 84), (48, 84), y_tick_color, 1, cv2.LINE_AA)
    cv2.putText(overlay, "y-axis tick redraw", (66, 88), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (20, 20, 20), 1, cv2.LINE_AA)

    out_path = case_dir / "overlay_results.png"
    ok = cv2.imwrite(str(out_path), overlay)
    if not ok:
        return {"overlay_results": None, "error": f"failed to write overlay: {out_path}"}
    return {"overlay_results": str(out_path.name)}

def run_case(
    case_name: str,
    fixtures_dir: str | Path = "tests/fixtures/standard",
    write_results: bool = True,
) -> dict:
    """Run the pipeline on a single test case and compare against ground truth.

    Args:
        case_name: Name of the case subfolder (e.g. "case_001", "low_res_overlap")
        fixtures_dir: Path to the fixtures profile directory
        write_results: If True, write results.json to the case folder

    Returns:
        Results dict with per-stage comparison metrics.
    """
    print(f"[RUN] starting {case_name}", flush=True)
    case_dir = Path(fixtures_dir) / case_name
    if not case_dir.exists():
        return {"error": f"Case directory not found: {case_dir}"}

    test_case = load_test_case(case_dir)
    expected_curve_direction = "downward"
    metadata_path = case_dir / "metadata.json"
    if metadata_path.exists():
        try:
            with open(metadata_path) as f:
                case_metadata = json.load(f)
            raw_direction = str(case_metadata.get("curve_direction", "downward")).lower()
            if raw_direction in ("downward", "upward"):
                expected_curve_direction = raw_direction
        except (OSError, json.JSONDecodeError):
            pass

    # Lazy import — requires LLM API keys and full pipeline deps
    from km_estimator.pipeline import run_pipeline

    # Feed graph.png through the pipeline
    image_path = case_dir / "graph.png"
    if not image_path.exists():
        return {"error": f"graph.png not found in {case_dir}"}

    try:
        state = run_pipeline(str(image_path))
    except Exception as e:
        return {
            "case_name": case_name,
            "difficulty": test_case.difficulty,
            "n_curves": len(test_case.curves),
            "pipeline_errors": [{
                "stage": "runtime",
                "message": f"Unhandled exception: {type(e).__name__}: {e}",
            }],
            "mmpu": {"error": "pipeline exception"},
            "digitize": {"error": "pipeline exception"},
            "hard_points": {"error": "pipeline exception"},
            "reconstruction": {"error": "pipeline exception"},
            "validation": {"error": "pipeline exception"},
            "passed": False,
        }

    results: dict = {
        "case_name": case_name,
        "difficulty": test_case.difficulty,
        "n_curves": len(test_case.curves),
        "pipeline_errors": [
            {
                "stage": e.stage.value,
                "message": e.message,
                "details": e.details,
            }
            for e in state.errors
        ],
    }

    # Stage 1: MMPU — compare plot metadata
    if state.plot_metadata:
        pm = state.plot_metadata
        gt_x = test_case.x_axis
        gt_y = test_case.y_axis
        results["mmpu"] = {
            "x_axis_start_error": abs(pm.x_axis.start - gt_x.start),
            "x_axis_end_error": abs(pm.x_axis.end - gt_x.end),
            "y_axis_start_error": abs(pm.y_axis.start - gt_y.start),
            "y_axis_end_error": abs(pm.y_axis.end - gt_y.end),
            "x_tick_interval_error": (
                abs(float(pm.x_axis.tick_interval) - float(gt_x.tick_interval))
                if pm.x_axis.tick_interval is not None and gt_x.tick_interval is not None
                else None
            ),
            "y_tick_interval_error": (
                abs(float(pm.y_axis.tick_interval) - float(gt_y.tick_interval))
                if pm.y_axis.tick_interval is not None and gt_y.tick_interval is not None
                else None
            ),
            "x_tick_count_detected": len(pm.x_axis.tick_values),
            "x_tick_count_expected": len(gt_x.tick_values),
            "y_tick_count_detected": len(pm.y_axis.tick_values),
            "y_tick_count_expected": len(gt_y.tick_values),
            "x_tick_values_mae": _tick_values_mae(pm.x_axis.tick_values, gt_x.tick_values),
            "y_tick_values_mae": _tick_values_mae(pm.y_axis.tick_values, gt_y.tick_values),
            "x_tick_coverage_detected": _tick_coverage(pm.x_axis.tick_values, pm.x_axis.start, pm.x_axis.end),
            "x_tick_coverage_expected": _tick_coverage(gt_x.tick_values, gt_x.start, gt_x.end),
            "y_tick_coverage_detected": _tick_coverage(pm.y_axis.tick_values, pm.y_axis.start, pm.y_axis.end),
            "y_tick_coverage_expected": _tick_coverage(gt_y.tick_values, gt_y.start, gt_y.end),
            "n_curves_detected": len(pm.curves),
            "n_curves_expected": len(test_case.curves),
            "curves_match": len(pm.curves) == len(test_case.curves),
            "curve_direction_detected": getattr(pm, "curve_direction", "downward"),
            "curve_direction_expected": expected_curve_direction,
            "curve_direction_match": (
                getattr(pm, "curve_direction", "downward") == expected_curve_direction
            ),
            "risk_table_detected": pm.risk_table is not None,
            "risk_table_expected": test_case.risk_table is not None,
        }
    else:
        results["mmpu"] = {"error": "no plot_metadata extracted"}

    # Stage 2: Digitize — compare curve coordinates
    if state.digitized_curves:
        expected_curves = {
            c.group_name: c.step_coords for c in test_case.curves
        }
        results["digitize"] = compare_digitized_curves(
            state.digitized_curves, expected_curves
        )
    else:
        results["digitize"] = {"error": "no digitized_curves produced"}

    # Stage 3: Reconstruct — compare IPD via hard points
    if state.output:
        hp_path = case_dir / "hard_points.json"
        if hp_path.exists():
            with open(hp_path) as f:
                hard_points = json.load(f)
            results["hard_points"] = compare_hard_points(
                state.output, hard_points
            )
        else:
            results["hard_points"] = {"error": "hard_points.json not found"}

        # Also compare reconstructed KM curves
        recon_curves = {}
        for curve in state.output.curves:
            recon_km = _km_from_ipd(curve.patients)
            recon_curves[curve.group_name] = recon_km
        expected_curves = {
            c.group_name: c.step_coords for c in test_case.curves
        }
        results["reconstruction"] = compare_digitized_curves(
            recon_curves, expected_curves
        )
    else:
        results["hard_points"] = {"error": "no pipeline output"}
        results["reconstruction"] = {"error": "no pipeline output"}

    # Stage 4: Validate — pipeline's own internal validation
    if state.output:
        results["validation"] = {
            curve.group_name: {
                "mae": curve.validation_mae,
                "dtw": curve.validation_dtw,
                "rmse": curve.validation_rmse,
            }
            for curve in state.output.curves
        }
        results["reconstruction_meta"] = {
            "mode": state.output.reconstruction_mode.value,
            "patient_counts": {
                curve.group_name: len(curve.patients)
                for curve in state.output.curves
            },
            "warnings": list(state.output.warnings),
        }
    else:
        results["validation"] = {"error": "no pipeline output"}
        results["reconstruction_meta"] = {
            "error": "no pipeline output",
            "warnings": list(state.mmpu_warnings),
            "flagged_for_review": bool(state.flagged_for_review),
        }

    # Stage diagnostics: attribute failures to digitization vs reconstruction per arm.
    results["benchmark_track"] = _classify_benchmark_track(results)
    results["diagnostics"] = _build_case_diagnostics(results)
    results["stage_mae"] = _build_stage_mae_decomposition(results)
    results["interval_debug"] = _build_interval_debug_artifacts(state, test_case, results)
    results["artifacts"] = _write_overlay_artifact(state, case_dir)

    # Overall pass/fail
    results["passed"] = _check_pass(results)

    if write_results:
        results_path = case_dir / "results.json"
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2, default=_json_default)

    return results


def run_all(
    fixtures_dir: str | Path = "tests/fixtures/standard",
    write_results: bool = True,
    max_workers: int = 1,
) -> dict:
    """Run the pipeline on all cases in a profile and produce a summary report."""
    fixtures_dir = Path(fixtures_dir)
    manifest = load_manifest(fixtures_dir)

    all_results = _run_manifest_cases(
        manifest=manifest,
        fixtures_dir=fixtures_dir,
        write_results=write_results,
        max_workers=max_workers,
    )

    return _build_summary(all_results, fixtures_dir, write_results)


def run_filtered(
    fixtures_dir: str | Path = "tests/fixtures/standard",
    difficulty_range: tuple[int, int] | None = None,
    names: list[str] | None = None,
    write_results: bool = True,
    max_workers: int = 1,
) -> dict:
    """Run the pipeline on a filtered subset of cases."""
    fixtures_dir = Path(fixtures_dir)
    manifest = load_manifest(fixtures_dir)

    filtered = manifest
    if difficulty_range:
        lo, hi = difficulty_range
        filtered = [e for e in filtered if lo <= e.get("difficulty", 0) <= hi]
    if names:
        name_set = set(names)
        filtered = [e for e in filtered if e["name"] in name_set]

    all_results = _run_manifest_cases(
        manifest=filtered,
        fixtures_dir=fixtures_dir,
        write_results=write_results,
        max_workers=max_workers,
    )

    return _build_summary(all_results, fixtures_dir, write_results)


def run_fixed_suite(
    fixtures_dir: str | Path = "tests/fixtures/standard",
    case_names: list[str] | None = None,
    write_results: bool = True,
    max_workers: int = 1,
) -> dict:
    """
    Run a fixed-case regression suite for deterministic tuning.
    """
    names = list(case_names) if case_names else list(DEFAULT_FIXED_CASES)
    summary = run_filtered(
        fixtures_dir=fixtures_dir,
        names=names,
        write_results=write_results,
        max_workers=max_workers,
    )
    summary["fixed_suite"] = {
        "case_names": names,
    }
    return summary


def _run_manifest_cases(
    manifest: list[dict],
    fixtures_dir: Path,
    write_results: bool,
    max_workers: int,
) -> list[dict]:
    """Run manifest entries serially or with bounded thread parallelism."""
    if max_workers <= 1:
        all_results: list[dict] = []
        for entry in manifest:
            case_name = entry["name"]
            result = run_case(case_name, fixtures_dir, write_results=write_results)
            all_results.append(result)
        return all_results

    # Preserve manifest order in outputs for deterministic summaries.
    ordered_results: list[dict | None] = [None] * len(manifest)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_idx = {
            executor.submit(
                run_case,
                entry["name"],
                fixtures_dir,
                write_results,
            ): idx
            for idx, entry in enumerate(manifest)
        }
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            case_name = str(manifest[idx].get("name", f"case_{idx:03d}"))
            difficulty = int(manifest[idx].get("difficulty", 0))
            try:
                ordered_results[idx] = future.result()
            except Exception as exc:
                ordered_results[idx] = {
                    "case_name": case_name,
                    "difficulty": difficulty,
                    "n_curves": 0,
                    "pipeline_errors": [{
                        "stage": "runtime",
                        "message": f"Unhandled exception: {type(exc).__name__}: {exc}",
                    }],
                    "mmpu": {"error": "runner exception"},
                    "digitize": {"error": "runner exception"},
                    "hard_points": {"error": "runner exception"},
                    "reconstruction": {"error": "runner exception"},
                    "validation": {"error": "runner exception"},
                    "passed": False,
                }

    return [result for result in ordered_results if result is not None]


def _survival_at(coords: list[tuple[float, float]], t: float) -> float:
    """Evaluate a KM step function at time t."""
    if not coords:
        return 1.0
    ordered = sorted(coords, key=lambda p: p[0])
    if t < ordered[0][0]:
        return 1.0
    for i in range(len(ordered) - 1, -1, -1):
        if ordered[i][0] <= t:
            return float(ordered[i][1])
    return float(ordered[0][1])


def _build_interval_debug_artifacts(state, test_case, results: dict) -> dict:
    """Build per-interval reconstruction diagnostics for failed-case debugging."""
    if not state.output:
        return {"error": "no pipeline output"}

    expected_curves = {c.group_name: c.step_coords for c in test_case.curves}
    hard_points = results.get("hard_points", {})
    recon_metrics = results.get("reconstruction", {})

    risk_table = None
    if state.plot_metadata is not None:
        risk_table = state.plot_metadata.risk_table
    risk_time_points = list(risk_table.time_points) if risk_table is not None else []
    risk_counts: dict[str, list[int]] = {}
    if risk_table is not None:
        risk_counts = {group.name: list(group.counts) for group in risk_table.groups}

    per_curve: dict[str, dict] = {}
    for curve in state.output.curves:
        name = curve.group_name
        patients = list(curve.patients)
        recon_coords = _km_from_ipd(patients)
        digitized_coords = state.digitized_curves.get(name, []) if state.digitized_curves else []
        expected_coords = expected_curves.get(name, [])

        interval_ledger: list[dict] = []
        counts = risk_counts.get(name, [])
        if len(risk_time_points) >= 2 and len(counts) == len(risk_time_points):
            last_interval_idx = len(risk_time_points) - 2
            for j in range(len(risk_time_points) - 1):
                t_start = float(risk_time_points[j])
                t_end = float(risk_time_points[j + 1])
                n_start = int(counts[j])
                n_end = int(counts[j + 1])
                events = 0
                censors = 0
                terminal_carryover_censors = 0
                for p in patients:
                    pt = float(p.time)
                    if not (t_start < pt <= t_end):
                        continue
                    if bool(p.event):
                        events += 1
                        continue
                    # Final-time right-censored survivors are denominator carryover.
                    if j == last_interval_idx and abs(pt - t_end) <= 1e-9:
                        terminal_carryover_censors += 1
                        continue
                    censors += 1
                interval_ledger.append({
                    "t_start": t_start,
                    "t_end": t_end,
                    "n_start": n_start,
                    "n_end": n_end,
                    "risk_table_loss": int(n_start - n_end),
                    "events_reconstructed": int(events),
                    "censors_reconstructed": int(censors),
                    "terminal_carryover_censors": int(terminal_carryover_censors),
                    "digitized_s_start": round(_survival_at(digitized_coords, t_start), 4),
                    "digitized_s_end": round(_survival_at(digitized_coords, t_end), 4),
                    "reconstructed_s_end": round(_survival_at(recon_coords, t_end), 4),
                    "expected_s_end": round(_survival_at(expected_coords, t_end), 4),
                })

        hardpoint_deltas: list[dict] = []
        hp_group = hard_points.get(name, {}) if isinstance(hard_points, dict) else {}
        if isinstance(hp_group, dict):
            for landmark in hp_group.get("landmarks", []):
                hardpoint_deltas.append({
                    "time": landmark.get("time"),
                    "expected": landmark.get("expected"),
                    "actual": landmark.get("actual"),
                    "error": landmark.get("error"),
                    "pass": bool(landmark.get("pass", False)),
                })

        rec_mae = None
        metric_group = recon_metrics.get(name, {}) if isinstance(recon_metrics, dict) else {}
        if isinstance(metric_group, dict):
            rec_mae = metric_group.get("mae")

        per_curve[name] = {
            "reconstruction_mae": rec_mae,
            "intervals": interval_ledger,
            "hardpoint_deltas": hardpoint_deltas,
        }

    return {"per_curve": per_curve}


def _classify_benchmark_track(results: dict) -> str:
    """Classify case into full (risk-table constrained) vs cv-only track."""
    mmpu = results.get("mmpu", {})
    recon_meta = results.get("reconstruction_meta", {})
    expected = bool(mmpu.get("risk_table_expected")) if isinstance(mmpu, dict) else False
    detected = bool(mmpu.get("risk_table_detected")) if isinstance(mmpu, dict) else False
    mode = str(recon_meta.get("mode", "")) if isinstance(recon_meta, dict) else ""
    if expected and detected and mode == "full":
        return "full_track"
    return "cv_only_track"


def _check_pass(results: dict) -> bool:
    """Determine if a case passed based on key metrics."""
    # Check if pipeline produced output
    if "error" in results.get("digitize", {}):
        return False
    if "error" in results.get("hard_points", {}):
        return False

    # Check digitization MAE for each curve
    digitize = results.get("digitize", {})
    for curve_name, metrics in digitize.items():
        if isinstance(metrics, dict) and "mae" in metrics:
            if metrics["mae"] > 0.05:
                return False

    # Check hard points
    hp = results.get("hard_points", {})
    for curve_name, hp_result in hp.items():
        if isinstance(hp_result, dict) and not hp_result.get("all_pass", True):
            return False

    return True


def _build_case_diagnostics(results: dict) -> dict:
    """Build per-arm debugging diagnostics for stage attribution."""
    digitize = results.get("digitize", {})
    reconstruction = results.get("reconstruction", {})
    hard_points = results.get("hard_points", {})

    arm_names: set[str] = set()
    if isinstance(digitize, dict):
        arm_names.update(k for k, v in digitize.items() if isinstance(v, dict) and "mae" in v)
    if isinstance(reconstruction, dict):
        arm_names.update(k for k, v in reconstruction.items() if isinstance(v, dict) and "mae" in v)
    if isinstance(hard_points, dict):
        arm_names.update(k for k, v in hard_points.items() if isinstance(v, dict) and "landmarks" in v)

    per_arm: dict[str, dict] = {}
    stage_counts: dict[str, int] = {}
    hard_fail_times: dict[str, int] = {}

    for arm in sorted(arm_names):
        d = digitize.get(arm, {}) if isinstance(digitize, dict) else {}
        r = reconstruction.get(arm, {}) if isinstance(reconstruction, dict) else {}
        h = hard_points.get(arm, {}) if isinstance(hard_points, dict) else {}

        dig_mae = d.get("mae") if isinstance(d, dict) else None
        rec_mae = r.get("mae") if isinstance(r, dict) else None
        delta_rec_minus_dig = (
            rec_mae - dig_mae
            if isinstance(dig_mae, (int, float)) and isinstance(rec_mae, (int, float))
            else None
        )

        failed_landmarks = []
        if isinstance(h, dict):
            for lm in h.get("landmarks", []):
                if lm.get("pass", True):
                    continue
                bias = lm["actual"] - lm["expected"]
                failed_landmarks.append({
                    "time": lm["time"],
                    "expected": lm["expected"],
                    "actual": lm["actual"],
                    "error": lm["error"],
                    "bias": round(bias, 4),
                })
                hard_fail_times[str(lm["time"])] = hard_fail_times.get(str(lm["time"]), 0) + 1

        hard_fail_count = len(failed_landmarks)
        digitization_bad = isinstance(dig_mae, (int, float)) and dig_mae > 0.05
        reconstruction_drift = (
            isinstance(delta_rec_minus_dig, (int, float)) and delta_rec_minus_dig > 0.02
        )

        if hard_fail_count == 0:
            if digitization_bad:
                stage = "digitization_warning"
            elif reconstruction_drift:
                stage = "reconstruction_drift"
            else:
                stage = "clean"
        else:
            if digitization_bad and reconstruction_drift:
                stage = "mixed"
            elif digitization_bad:
                stage = "digitization_limited"
            else:
                stage = "reconstruction_limited"

        stage_counts[stage] = stage_counts.get(stage, 0) + 1
        per_arm[arm] = {
            "digitize_mae": dig_mae,
            "reconstruction_mae": rec_mae,
            "delta_reconstruction_minus_digitize": delta_rec_minus_dig,
            "hard_fail_count": hard_fail_count,
            "stage_attribution": stage,
            "failed_landmarks": failed_landmarks,
        }

    return {
        "per_arm": per_arm,
        "stage_attribution_counts": stage_counts,
        "hard_fail_times": hard_fail_times,
    }


def _build_stage_mae_decomposition(results: dict) -> dict:
    """Per-case MAE decomposition by stage and arm."""
    diagnostics = results.get("diagnostics", {})
    per_arm = diagnostics.get("per_arm", {}) if isinstance(diagnostics, dict) else {}
    stage_by_arm: dict[str, dict] = {}
    dig_values: list[float] = []
    rec_values: list[float] = []
    delta_values: list[float] = []

    if isinstance(per_arm, dict):
        for arm, payload in per_arm.items():
            if not isinstance(payload, dict):
                continue
            dig = payload.get("digitize_mae")
            rec = payload.get("reconstruction_mae")
            delta = payload.get("delta_reconstruction_minus_digitize")
            stage_by_arm[arm] = {
                "digitize_mae": dig,
                "reconstruction_mae": rec,
                "delta_reconstruction_minus_digitize": delta,
                "stage_attribution": payload.get("stage_attribution"),
            }
            if isinstance(dig, (int, float)):
                dig_values.append(float(dig))
            if isinstance(rec, (int, float)):
                rec_values.append(float(rec))
            if isinstance(delta, (int, float)):
                delta_values.append(float(delta))

    return {
        "arms": stage_by_arm,
        "mean_digitize_mae": (float(np.mean(dig_values)) if dig_values else None),
        "mean_reconstruction_mae": (float(np.mean(rec_values)) if rec_values else None),
        "mean_delta_reconstruction_minus_digitize": (
            float(np.mean(delta_values)) if delta_values else None
        ),
    }


def _build_track_summary(all_results: list[dict]) -> dict[str, dict]:
    """Aggregate pass-rate/MAE by benchmark track."""
    buckets: dict[str, dict[str, object]] = {
        "full_track": {"cases": 0, "passed": 0, "mae_values": []},
        "cv_only_track": {"cases": 0, "passed": 0, "mae_values": []},
    }
    for result in all_results:
        track = str(result.get("benchmark_track", _classify_benchmark_track(result)))
        if track not in buckets:
            track = "cv_only_track"
        bucket = buckets[track]
        bucket["cases"] = int(bucket["cases"]) + 1
        if result.get("passed", False):
            bucket["passed"] = int(bucket["passed"]) + 1

        reconstruction = result.get("reconstruction", {})
        if isinstance(reconstruction, dict):
            for metrics in reconstruction.values():
                if isinstance(metrics, dict) and isinstance(metrics.get("mae"), (int, float)):
                    cast_list = bucket["mae_values"]
                    if isinstance(cast_list, list):
                        cast_list.append(float(metrics["mae"]))

    summary: dict[str, dict] = {}
    for track, bucket in buckets.items():
        cases = int(bucket["cases"])
        passed = int(bucket["passed"])
        mae_values = bucket["mae_values"] if isinstance(bucket["mae_values"], list) else []
        summary[track] = {
            "cases": cases,
            "passed": passed,
            "pass_rate": round(passed / max(1, cases), 4),
            "mean_reconstruction_mae": (
                round(float(np.mean(mae_values)), 4) if mae_values else None
            ),
            "median_reconstruction_mae": (
                round(float(np.median(mae_values)), 4) if mae_values else None
            ),
        }
    return summary


def _build_interval_bias_dashboard(all_results: list[dict]) -> dict:
    """Aggregate time-indexed reconstruction and hard-point bias diagnostics."""
    interval_bins: dict[float, dict[str, list[float]]] = {}
    hardpoint_bins: dict[float, dict[str, list[float]]] = {}

    for result in all_results:
        interval_debug = result.get("interval_debug", {})
        if isinstance(interval_debug, dict):
            per_curve = interval_debug.get("per_curve", {})
            if isinstance(per_curve, dict):
                for curve_payload in per_curve.values():
                    if not isinstance(curve_payload, dict):
                        continue
                    intervals = curve_payload.get("intervals", [])
                    if not isinstance(intervals, list):
                        continue
                    for interval in intervals:
                        if not isinstance(interval, dict):
                            continue
                        t_end = float(interval.get("t_end", 0.0))
                        rec_s = interval.get("reconstructed_s_end")
                        exp_s = interval.get("expected_s_end")
                        dig_s = interval.get("digitized_s_end")
                        if not isinstance(rec_s, (int, float)) or not isinstance(exp_s, (int, float)):
                            continue
                        bucket = interval_bins.setdefault(
                            t_end,
                            {
                                "reconstruction_bias": [],
                                "digitization_bias": [],
                            },
                        )
                        bucket["reconstruction_bias"].append(float(rec_s - exp_s))
                        if isinstance(dig_s, (int, float)):
                            bucket["digitization_bias"].append(float(dig_s - exp_s))

        hard_points = result.get("hard_points", {})
        if isinstance(hard_points, dict):
            for hp_payload in hard_points.values():
                if not isinstance(hp_payload, dict):
                    continue
                landmarks = hp_payload.get("landmarks", [])
                if not isinstance(landmarks, list):
                    continue
                for lm in landmarks:
                    if not isinstance(lm, dict):
                        continue
                    t = lm.get("time")
                    exp_s = lm.get("expected")
                    act_s = lm.get("actual")
                    passed = bool(lm.get("pass", False))
                    if not isinstance(t, (int, float)):
                        continue
                    if not isinstance(exp_s, (int, float)) or not isinstance(act_s, (int, float)):
                        continue
                    bucket = hardpoint_bins.setdefault(
                        float(t),
                        {"bias": [], "abs_error": [], "fail": []},
                    )
                    bias = float(act_s - exp_s)
                    bucket["bias"].append(bias)
                    bucket["abs_error"].append(abs(bias))
                    bucket["fail"].append(0.0 if passed else 1.0)

    interval_summary: dict[str, dict] = {}
    for t_end, payload in sorted(interval_bins.items()):
        rec_bias = payload["reconstruction_bias"]
        dig_bias = payload["digitization_bias"]
        interval_summary[str(round(t_end, 3))] = {
            "n": len(rec_bias),
            "mean_reconstruction_bias": round(float(np.mean(rec_bias)), 4) if rec_bias else None,
            "mean_abs_reconstruction_bias": (
                round(float(np.mean(np.abs(np.asarray(rec_bias, dtype=np.float64)))), 4)
                if rec_bias
                else None
            ),
            "mean_digitization_bias": round(float(np.mean(dig_bias)), 4) if dig_bias else None,
        }

    hardpoint_summary: dict[str, dict] = {}
    for t, payload in sorted(hardpoint_bins.items()):
        bias = payload["bias"]
        abs_error = payload["abs_error"]
        fail = payload["fail"]
        hardpoint_summary[str(round(t, 3))] = {
            "n": len(bias),
            "mean_bias": round(float(np.mean(bias)), 4) if bias else None,
            "mean_abs_error": round(float(np.mean(abs_error)), 4) if abs_error else None,
            "fail_rate": round(float(np.mean(fail)), 4) if fail else None,
        }

    return {
        "interval_end_bias": interval_summary,
        "hardpoint_bias": hardpoint_summary,
    }


def _build_stage_mae_dashboard(all_results: list[dict]) -> dict:
    """Aggregate per-stage MAE decomposition across run set."""
    dig: list[float] = []
    rec: list[float] = []
    delta: list[float] = []

    for result in all_results:
        stage = result.get("stage_mae", {})
        if not isinstance(stage, dict):
            continue
        d = stage.get("mean_digitize_mae")
        r = stage.get("mean_reconstruction_mae")
        q = stage.get("mean_delta_reconstruction_minus_digitize")
        if isinstance(d, (int, float)):
            dig.append(float(d))
        if isinstance(r, (int, float)):
            rec.append(float(r))
        if isinstance(q, (int, float)):
            delta.append(float(q))

    return {
        "n_cases_with_metrics": len(rec),
        "mean_digitize_mae": round(float(np.mean(dig)), 4) if dig else None,
        "median_digitize_mae": round(float(np.median(dig)), 4) if dig else None,
        "mean_reconstruction_mae": round(float(np.mean(rec)), 4) if rec else None,
        "median_reconstruction_mae": round(float(np.median(rec)), 4) if rec else None,
        "mean_delta_reconstruction_minus_digitize": (
            round(float(np.mean(delta)), 4) if delta else None
        ),
    }


def _build_summary(
    all_results: list[dict],
    output_dir: Path,
    write_results: bool,
) -> dict:
    """Build an aggregate summary report."""
    total = len(all_results)
    passed = sum(1 for r in all_results if r.get("passed", False))
    failed = total - passed

    # Collect MAE values across all curves
    mae_values = []
    for r in all_results:
        digitize = r.get("digitize", {})
        for curve_name, metrics in digitize.items():
            if isinstance(metrics, dict) and "mae" in metrics:
                mae_values.append(metrics["mae"])

    # Worst cases
    failed_cases = [r["case_name"] for r in all_results if not r.get("passed", False)]

    failure_breakdown = {
        "hard_points_failed": 0,
        "digitization_mae_failed": 0,
        "risk_table_missed": 0,
        "x_axis_end_misread": 0,
        "curve_count_mismatch": 0,
        "pipeline_crash": 0,
    }
    stage_attribution: dict[str, int] = {}
    for r in all_results:
        if r.get("passed", False):
            continue

        mmpu = r.get("mmpu", {})
        digitize = r.get("digitize", {})
        hard_points = r.get("hard_points", {})

        if r.get("pipeline_errors"):
            failure_breakdown["pipeline_crash"] += 1

        if isinstance(mmpu, dict):
            if mmpu.get("x_axis_end_error", 0) > 0:
                failure_breakdown["x_axis_end_misread"] += 1
            if mmpu.get("risk_table_expected") and not mmpu.get("risk_table_detected"):
                failure_breakdown["risk_table_missed"] += 1
            if mmpu.get("n_curves_detected") != mmpu.get("n_curves_expected"):
                failure_breakdown["curve_count_mismatch"] += 1

        if isinstance(digitize, dict):
            has_mae_fail = any(
                isinstance(v, dict) and v.get("mae", 0) > 0.05
                for v in digitize.values()
            )
            if has_mae_fail:
                failure_breakdown["digitization_mae_failed"] += 1

        if isinstance(hard_points, dict):
            has_hard_fail = any(
                isinstance(v, dict) and not v.get("all_pass", True)
                for v in hard_points.values()
            )
            if has_hard_fail:
                failure_breakdown["hard_points_failed"] += 1

        diagnostics = r.get("diagnostics", {})
        if isinstance(diagnostics, dict):
            counts = diagnostics.get("stage_attribution_counts", {})
            if isinstance(counts, dict):
                for k, v in counts.items():
                    stage_attribution[k] = stage_attribution.get(k, 0) + int(v)

    # By difficulty
    by_difficulty: dict[int, dict[str, int]] = {}
    for r in all_results:
        d = r.get("difficulty", 0)
        if d not in by_difficulty:
            by_difficulty[d] = {"n": 0, "pass": 0}
        by_difficulty[d]["n"] += 1
        if r.get("passed", False):
            by_difficulty[d]["pass"] += 1

    summary = {
        "total": total,
        "passed": passed,
        "failed": failed,
        "mean_mae": round(float(np.mean(mae_values)), 4) if mae_values else None,
        "median_mae": round(float(np.median(mae_values)), 4) if mae_values else None,
        "worst_cases": failed_cases[:10],
        "failure_breakdown": failure_breakdown,
        "stage_attribution": stage_attribution,
        "stage_mae_dashboard": _build_stage_mae_dashboard(all_results),
        "benchmark_tracks": _build_track_summary(all_results),
        "metrics_dashboard": _build_interval_bias_dashboard(all_results),
        "by_difficulty": {
            str(k): v for k, v in sorted(by_difficulty.items())
        },
    }

    if write_results:
        summary_path = output_dir / "summary.json"
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2, default=_json_default)

    return summary


def _json_default(obj):
    """JSON serializer for objects not serializable by default."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return str(obj)
