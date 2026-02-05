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

    # Lazy import — requires LLM API keys and full pipeline deps
    from km_estimator.pipeline import run_pipeline

    # Feed graph.png through the pipeline
    image_path = case_dir / "graph.png"
    if not image_path.exists():
        return {"error": f"graph.png not found in {case_dir}"}

    state = run_pipeline(str(image_path))

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
    else:
        results["validation"] = {"error": "no pipeline output"}

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
