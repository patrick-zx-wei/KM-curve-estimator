"""Full-pipeline benchmark: MMPU + digitize_v5 end-to-end.

Runs through the entire pipeline (preprocess → input_guard → mmpu → digitize → reconstruct → validate)
using real LLM calls for metadata extraction. Requires OPENAI_API_KEY and GOOGLE_API_KEY in .env.
"""
from __future__ import annotations

import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from difflib import SequenceMatcher
from pathlib import Path

import numpy as np
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

# Load API keys from .env
load_dotenv(ROOT / ".env")

# langchain-google-genai expects GOOGLE_API_KEY
if not os.environ.get("GOOGLE_API_KEY") and os.environ.get("GEMINI_API_KEY"):
    os.environ["GOOGLE_API_KEY"] = os.environ["GEMINI_API_KEY"]

# Force v5 digitizer
os.environ["KM_DIGITIZER"] = "v5"

from km_estimator.pipeline import run_pipeline
from km_estimator.models import PipelineConfig

FIXTURE_DIR = ROOT / "tests" / "fixtures" / "standard"
CASES = ["case_001", "case_010", "case_020", "case_040", "case_050",
         "case_060", "case_070", "case_080", "case_086", "case_095"]
MAX_WORKERS = 2


def _fuzzy_match_arms(
    digitized_names: list[str],
    hard_point_names: list[str],
) -> dict[str, str]:
    """Map hard_point arm names to the closest digitized arm name."""
    mapping: dict[str, str] = {}
    used: set[str] = set()
    for hp_name in hard_point_names:
        best_score = -1.0
        best_dig = None
        for dig_name in digitized_names:
            if dig_name in used:
                continue
            score = SequenceMatcher(None, hp_name.lower(), dig_name.lower()).ratio()
            if score > best_score:
                best_score = score
                best_dig = dig_name
        if best_dig is not None and best_score > 0.4:
            mapping[hp_name] = best_dig
            used.add(best_dig)
    return mapping


def _compute_arm_mae(
    digitized: dict[str, list[tuple[float, float]]],
    hard_points: dict,
    y_start: float,
) -> tuple[dict[str, float], dict[str, str]]:
    """Compute per-arm MAE with fuzzy name matching. Returns (mae_dict, name_mapping)."""
    dig_names = list(digitized.keys())
    hp_names = list(hard_points.keys())
    name_map = _fuzzy_match_arms(dig_names, hp_names)

    results = {}
    for hp_name, hp_data in hard_points.items():
        landmarks = hp_data.get("landmarks", [])
        if not landmarks:
            results[hp_name] = float("nan")
            continue

        dig_name = name_map.get(hp_name)
        if dig_name is None:
            print(f"    WARNING: No match for '{hp_name}' in digitized arms: {dig_names}")
            results[hp_name] = float("nan")
            continue

        dig_pts = digitized.get(dig_name, [])
        if not dig_pts:
            results[hp_name] = float("nan")
            continue

        times = np.array([p[0] for p in dig_pts])
        values = np.array([p[1] for p in dig_pts])
        errors = []
        for lm in landmarks:
            t = lm["time"]
            expected = lm["survival"]
            if expected < y_start:
                continue
            if len(times) == 0:
                errors.append(abs(expected))
                continue
            idx = np.searchsorted(times, t, side="right") - 1
            idx = max(0, min(idx, len(values) - 1))
            predicted = values[idx]
            errors.append(abs(predicted - expected))
        results[hp_name] = float(np.mean(errors)) if errors else float("nan")
    return results, name_map


def main():
    # Verify API keys
    if not os.environ.get("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY not set. Check .env file.")
        sys.exit(1)
    if not os.environ.get("GOOGLE_API_KEY"):
        print("ERROR: GOOGLE_API_KEY / GEMINI_API_KEY not set. Check .env file.")
        sys.exit(1)

    print(f"Running full pipeline (MMPU + digitize_v5) on {len(CASES)} cases, {MAX_WORKERS} workers")
    print(f"KM_DIGITIZER = {os.environ.get('KM_DIGITIZER')}")
    print()

    config = PipelineConfig()
    all_results = {}
    timings = {}

    def _run_one(case_name: str) -> tuple[str, dict, float, list[str]]:
        """Run pipeline for one case. Returns (case_name, mae_dict, elapsed, log_lines)."""
        log = []
        case_dir = FIXTURE_DIR / case_name
        meta = json.loads((case_dir / "metadata.json").read_text())
        hard_points = json.loads((case_dir / "hard_points.json").read_text())
        image_path = str(case_dir / "graph.png")

        log.append(f"{'='*60}")
        log.append(f"Running {case_name} ({len(meta['groups'])} arms: {meta['groups']})...")

        t0 = time.time()
        try:
            result = run_pipeline(image_path, config)
        except Exception as e:
            log.append(f"  PIPELINE ERROR: {e}")
            return case_name, {"error": str(e)}, time.time() - t0, log
        elapsed = time.time() - t0

        log.append(f"  Time: {elapsed:.1f}s")
        log.append(f"  Extraction route: {result.extraction_route}")
        if result.plot_metadata:
            pm = result.plot_metadata
            extracted_groups = [c.name for c in pm.curves]
            log.append(f"  Extracted groups: {extracted_groups}")
            log.append(f"  X-axis: [{pm.x_axis.start}, {pm.x_axis.end}]")
            log.append(f"  Y-axis: [{pm.y_axis.start}, {pm.y_axis.end}]")
            log.append(f"  Direction: {pm.curve_direction}")

            gt_groups = meta["groups"]
            if set(g.lower() for g in extracted_groups) != set(g.lower() for g in gt_groups):
                log.append(f"  WARNING: Group mismatch! GT={gt_groups}, Extracted={extracted_groups}")
            if abs(pm.y_axis.start - meta["y_axis"]["start"]) > 0.05:
                log.append(f"  WARNING: Y-start mismatch! GT={meta['y_axis']['start']}, Extracted={pm.y_axis.start}")
            if abs(pm.x_axis.end - meta["x_axis"]["end"]) > 1.0:
                log.append(f"  WARNING: X-end mismatch! GT={meta['x_axis']['end']}, Extracted={pm.x_axis.end}")

        if result.errors:
            non_recoverable = [e for e in result.errors if not e.recoverable]
            if non_recoverable:
                log.append(f"  ERRORS: {[e.message for e in non_recoverable]}")
                return case_name, {"error": True}, elapsed, log

        if result.digitized_curves is None or len(result.digitized_curves) == 0:
            log.append(f"  No digitized curves!")
            return case_name, {"error": True}, elapsed, log

        log.append(f"  Digitized arms: {list(result.digitized_curves.keys())}")
        arm_mae, name_map = _compute_arm_mae(
            result.digitized_curves,
            hard_points,
            meta["y_axis"]["start"],
        )
        for hp_name, mae in arm_mae.items():
            dig_name = name_map.get(hp_name, "???")
            match_note = f" (matched to '{dig_name}')" if dig_name != hp_name else ""
            log.append(f"  {hp_name}: MAE={mae:.4f}{match_note}")
        return case_name, arm_mae, elapsed, log

    wall_start = time.time()
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {pool.submit(_run_one, c): c for c in CASES}
        for fut in as_completed(futures):
            case_name, result_dict, elapsed, log_lines = fut.result()
            for line in log_lines:
                print(line)
            all_results[case_name] = result_dict
            timings[case_name] = elapsed
    wall_elapsed = time.time() - wall_start

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    total_mae = []
    for case_name, res in all_results.items():
        if isinstance(res, dict) and "error" not in res:
            for arm, mae in res.items():
                if not np.isnan(mae):
                    total_mae.append(mae)
                    status = "OK" if mae < 0.05 else "WARN" if mae < 0.10 else "BAD"
                    print(f"  {case_name}/{arm}: {mae:.4f} [{status}]")
    if total_mae:
        print(f"\n  Mean MAE: {np.mean(total_mae):.4f}")
        print(f"  Median MAE: {np.median(total_mae):.4f}")
        ok = sum(1 for m in total_mae if m < 0.05)
        print(f"  Arms < 0.05 MAE: {ok}/{len(total_mae)} ({100*ok/len(total_mae):.0f}%)")
        cpu_total = sum(timings.values())
        print(f"  Wall time: {wall_elapsed:.1f}s")
        print(f"  CPU time: {cpu_total:.1f}s "
              f"({np.mean(list(timings.values())):.1f}s avg/case)")
    else:
        print("  No successful results!")


if __name__ == "__main__":
    main()
