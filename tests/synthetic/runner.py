"""Test runner that interfaces with the actual KM pipeline.

Feeds synthetic graph.png images through run_pipeline() and compares
the output against ground truth at each pipeline stage.

Requires OPENAI_API_KEY and GEMINI_API_KEY environment variables.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .data_gen import _km_from_ipd
from .ground_truth import (
    compare_digitized_curves,
    compare_hard_points,
    load_manifest,
    load_test_case,
)


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
            {"stage": e.stage.value, "message": e.message}
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
        results["reconstruction_meta"] = {"error": "no pipeline output"}

    # Stage diagnostics: attribute failures to digitization vs reconstruction per arm.
    results["diagnostics"] = _build_case_diagnostics(results)
    results["interval_debug"] = _build_interval_debug_artifacts(state, test_case, results)

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
) -> dict:
    """Run the pipeline on all cases in a profile and produce a summary report."""
    fixtures_dir = Path(fixtures_dir)
    manifest = load_manifest(fixtures_dir)

    all_results = []
    for entry in manifest:
        case_name = entry["name"]
        result = run_case(case_name, fixtures_dir, write_results=write_results)
        all_results.append(result)

    return _build_summary(all_results, fixtures_dir, write_results)


def run_filtered(
    fixtures_dir: str | Path = "tests/fixtures/standard",
    difficulty_range: tuple[int, int] | None = None,
    names: list[str] | None = None,
    write_results: bool = True,
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

    all_results = []
    for entry in filtered:
        case_name = entry["name"]
        result = run_case(case_name, fixtures_dir, write_results=write_results)
        all_results.append(result)

    return _build_summary(all_results, fixtures_dir, write_results)


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
            for j in range(len(risk_time_points) - 1):
                t_start = float(risk_time_points[j])
                t_end = float(risk_time_points[j + 1])
                n_start = int(counts[j])
                n_end = int(counts[j + 1])
                events = sum(
                    1 for p in patients if (t_start < float(p.time) <= t_end and bool(p.event))
                )
                censors = sum(
                    1 for p in patients if (t_start < float(p.time) <= t_end and not bool(p.event))
                )
                interval_ledger.append({
                    "t_start": t_start,
                    "t_end": t_end,
                    "n_start": n_start,
                    "n_end": n_end,
                    "risk_table_loss": int(n_start - n_end),
                    "events_reconstructed": int(events),
                    "censors_reconstructed": int(censors),
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
