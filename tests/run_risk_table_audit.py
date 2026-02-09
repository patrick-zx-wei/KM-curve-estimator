"""Risk table OCR accuracy audit.

Runs the MMPU pipeline on selected cases and compares extracted risk tables
against ground truth from risk_table_data.csv files.

Usage:
    python tests/run_risk_table_audit.py
"""

from __future__ import annotations

import csv
import json
import re
import sys
from pathlib import Path

import numpy as np
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")
sys.path.insert(0, str(ROOT / "src"))

from km_estimator.models import PipelineState  # noqa: E402
from km_estimator.nodes.input_guard import input_guard  # noqa: E402
from km_estimator.nodes.mmpu import mmpu  # noqa: E402
from km_estimator.nodes.preprocessing import preprocess  # noqa: E402

FIXTURE_DIR = ROOT / "tests" / "fixtures" / "standard"

CASES = [
    "case_001",
    "case_002",
    "case_050",
]


def _load_ground_truth(csv_path: Path) -> dict:
    """Load ground truth risk table from CSV.

    Returns dict with 'time_points' and 'groups': {name: [counts]}.
    """
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        columns = reader.fieldnames
        if not columns:
            return {"time_points": [], "groups": {}}

        time_col = columns[0]  # 'time'
        group_cols = columns[1:]

        time_points = []
        groups = {col: [] for col in group_cols}

        for row in reader:
            time_points.append(float(row[time_col]))
            for col in group_cols:
                groups[col].append(int(float(row[col])))

    return {"time_points": time_points, "groups": groups}


def _normalize_name(name: str) -> str:
    """Normalize group name for fuzzy matching."""
    return re.sub(r"[^a-z0-9]+", " ", name.lower()).strip()


def _fuzzy_match(query: str, candidates: list[str]) -> str | None:
    """Find best fuzzy match for a group name."""
    nq = _normalize_name(query)
    # Exact normalized match
    for c in candidates:
        if _normalize_name(c) == nq:
            return c
    # Substring match
    for c in candidates:
        nc = _normalize_name(c)
        if nq in nc or nc in nq:
            return c
    # Token overlap
    q_tokens = set(nq.split())
    best, best_score = None, 0
    for c in candidates:
        c_tokens = set(_normalize_name(c).split())
        overlap = len(q_tokens & c_tokens)
        if overlap > best_score:
            best, best_score = c, overlap
    return best if best_score > 0 else None


def _compare_risk_table(extracted_rt, gt: dict, case_name: str) -> dict:
    """Compare extracted RiskTable against ground truth."""
    gt_times = gt["time_points"]
    gt_groups = gt["groups"]
    n_gt_groups = len(gt_groups)

    if extracted_rt is None:
        return {
            "case": case_name,
            "extracted": False,
            "n_times_gt": len(gt_times),
            "n_times_ext": 0,
            "n_groups_gt": n_gt_groups,
            "n_groups_ext": 0,
            "time_match": False,
            "groups_found": 0,
            "count_mae": float("nan"),
            "count_max_err": float("nan"),
            "pct_exact": 0.0,
            "errors_detail": [],
        }

    ext_times = extracted_rt.time_points
    ext_groups = {g.name: g.counts for g in extracted_rt.groups}

    # Time point comparison
    time_match = len(ext_times) == len(gt_times) and all(
        abs(a - b) < 0.5 for a, b in zip(ext_times, gt_times)
    )

    # Time point errors
    time_errors = []
    for i, (gt_t, ext_t) in enumerate(
        zip(gt_times, ext_times[: len(gt_times)])
    ):
        if abs(gt_t - ext_t) >= 0.5:
            time_errors.append(f"  t[{i}]: gt={gt_t} ext={ext_t}")

    # Group matching and count comparison
    groups_found = 0
    all_count_errors = []
    errors_detail = []
    ext_candidates = list(ext_groups.keys())

    for gt_name, gt_counts in gt_groups.items():
        matched = _fuzzy_match(gt_name, ext_candidates)
        if matched is None:
            errors_detail.append(f"  Group '{gt_name}' not found in extracted")
            continue

        groups_found += 1
        ext_counts = ext_groups[matched]
        ext_candidates.remove(matched)  # don't reuse

        # Align to min length (compare what overlaps)
        n_compare = min(len(gt_counts), len(ext_counts))
        for i in range(n_compare):
            err = abs(gt_counts[i] - ext_counts[i])
            all_count_errors.append(err)
            if err > 0:
                errors_detail.append(
                    f"  {gt_name}[t={gt_times[i] if i < len(gt_times) else '?'}]: "
                    f"gt={gt_counts[i]} ext={ext_counts[i]} err={err}"
                )

    return {
        "case": case_name,
        "extracted": True,
        "n_times_gt": len(gt_times),
        "n_times_ext": len(ext_times),
        "n_groups_gt": n_gt_groups,
        "n_groups_ext": len(ext_groups),
        "time_match": time_match,
        "time_errors": time_errors,
        "groups_found": groups_found,
        "count_mae": float(np.mean(all_count_errors)) if all_count_errors else float("nan"),
        "count_max_err": float(max(all_count_errors)) if all_count_errors else float("nan"),
        "pct_exact": (
            sum(1 for e in all_count_errors if e == 0) / len(all_count_errors) * 100
            if all_count_errors
            else 0.0
        ),
        "n_cells": len(all_count_errors),
        "errors_detail": errors_detail,
    }


def main():
    results = []
    for case_name in CASES:
        case_dir = FIXTURE_DIR / case_name
        image_path = str(case_dir / "graph.png")
        gt_csv = case_dir / "risk_table_data.csv"
        meta = json.loads((case_dir / "metadata.json").read_text())

        if not gt_csv.exists():
            print(f"\n{case_name}: No ground truth CSV, skipping")
            continue

        gt = _load_ground_truth(gt_csv)

        print(f"\n{'=' * 60}")
        print(f"Running MMPU on {case_name} ({meta['n_groups']} arms, tier={meta.get('tier', '?')})...")

        # Run pipeline: preprocess -> input_guard -> mmpu
        state = PipelineState(image_path=image_path)
        state = preprocess(state)
        state = input_guard(state)

        # Check for fatal errors
        fatal = [e for e in state.errors if not e.recoverable]
        if fatal:
            print(f"  Input guard failed: {[e.message for e in fatal]}")
            results.append({
                "case": case_name,
                "extracted": False,
                "error": "input_guard_failed",
            })
            continue

        state = mmpu(state)

        fatal = [e for e in state.errors if not e.recoverable]
        if fatal:
            print(f"  MMPU failed: {[e.message for e in fatal]}")
            results.append({
                "case": case_name,
                "extracted": False,
                "error": "mmpu_failed",
            })
            continue

        extracted_rt = state.plot_metadata.risk_table if state.plot_metadata else None
        comparison = _compare_risk_table(extracted_rt, gt, case_name)
        results.append(comparison)

        # Print per-case summary
        if not comparison["extracted"]:
            print("  Risk table NOT extracted!")
        else:
            tm = "YES" if comparison["time_match"] else "NO"
            print(
                f"  Times: {comparison['n_times_ext']}/{comparison['n_times_gt']} [{tm}] | "
                f"Groups: {comparison['groups_found']}/{comparison['n_groups_gt']} | "
                f"Count MAE: {comparison['count_mae']:.1f} | "
                f"Exact: {comparison['pct_exact']:.0f}% | "
                f"Max err: {comparison['count_max_err']:.0f}"
            )
            if comparison.get("time_errors"):
                for te in comparison["time_errors"]:
                    print(te)
            # Show worst count errors (up to 5)
            detail = comparison.get("errors_detail", [])
            if detail:
                print(f"  Count errors ({len(detail)} cells wrong):")
                for d in detail[:8]:
                    print(d)
                if len(detail) > 8:
                    print(f"  ... and {len(detail) - 8} more")

    # Aggregate summary
    print(f"\n{'=' * 60}")
    print("RISK TABLE OCR AUDIT SUMMARY")
    print(f"{'=' * 60}")

    extracted = [r for r in results if r.get("extracted", False)]
    not_extracted = [r for r in results if not r.get("extracted", False)]

    print(f"\n  Cases run: {len(results)}")
    print(f"  Risk tables extracted: {len(extracted)}/{len(results)}")
    if not_extracted:
        print(f"  Failed cases: {[r['case'] for r in not_extracted]}")

    if extracted:
        time_correct = sum(1 for r in extracted if r["time_match"])
        all_groups_found = sum(
            1 for r in extracted if r["groups_found"] == r["n_groups_gt"]
        )
        maes = [r["count_mae"] for r in extracted if not np.isnan(r["count_mae"])]
        max_errs = [r["count_max_err"] for r in extracted if not np.isnan(r["count_max_err"])]
        pct_exacts = [r["pct_exact"] for r in extracted if r.get("n_cells", 0) > 0]

        print(f"\n  Time points correct: {time_correct}/{len(extracted)} ({100 * time_correct / len(extracted):.0f}%)")
        print(f"  All groups found: {all_groups_found}/{len(extracted)} ({100 * all_groups_found / len(extracted):.0f}%)")
        if maes:
            print(f"\n  Count MAE  - mean: {np.mean(maes):.1f}, median: {np.median(maes):.1f}")
        if max_errs:
            print(f"  Max error  - mean: {np.mean(max_errs):.0f}, worst: {max(max_errs):.0f}")
        if pct_exacts:
            print(f"  Exact cells - mean: {np.mean(pct_exacts):.0f}%, min: {min(pct_exacts):.0f}%")


if __name__ == "__main__":
    main()
