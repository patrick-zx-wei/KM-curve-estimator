"""Reconstruction benchmark.

Digitizes with ground-truth metadata, injects ground-truth risk table,
runs reconstruct() + validate(), and measures MAE.

Usage:
    python tests/run_recon_bench.py                  # use cached digitization (fast)
    python tests/run_recon_bench.py --generate-cache # digitize all cases & save cache
    python tests/run_recon_bench.py --no-cache       # force re-digitization (no save)
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from km_estimator.models import PipelineState, RiskGroup, RiskTable  # noqa: E402
from km_estimator.models.plot_metadata import (  # noqa: E402
    AxisConfig,
    CurveInfo,
    PlotMetadata,
)
from km_estimator.nodes.digitization_5 import digitize_v5  # noqa: E402
from km_estimator.nodes.reconstruction import reconstruct, validate  # noqa: E402

FIXTURE_DIR = ROOT / "tests" / "fixtures" / "standard"
CACHE_FILENAME = "digitized_cache.json"

CASES = [
    f"case_{i:03d}" for i in range(1, 101)
]

COLOR_NAMES = ["blue", "orange", "green", "red", "purple"]
LINE_STYLES = ["solid", "dashed"]


def _save_cache(case_dir: Path, state: PipelineState) -> None:
    cache = {
        "digitized_curves": {
            name: [[t, s] for t, s in pts]
            for name, pts in (state.digitized_curves or {}).items()
        },
        "censoring_marks": {
            name: list(times)
            for name, times in (state.censoring_marks or {}).items()
        },
        "isolated_curve_pixels": {
            name: [[x, y] for x, y in pts]
            for name, pts in (state.isolated_curve_pixels or {}).items()
        },
    }
    (case_dir / CACHE_FILENAME).write_text(json.dumps(cache))


def _load_cache(case_dir: Path) -> dict | None:
    path = case_dir / CACHE_FILENAME
    if not path.exists():
        return None
    return json.loads(path.read_text())


def _apply_cache(state: PipelineState, cache: dict) -> PipelineState:
    digitized = {
        name: [(pt[0], pt[1]) for pt in pts]
        for name, pts in cache["digitized_curves"].items()
    }
    censoring = {
        name: list(times)
        for name, times in cache.get("censoring_marks", {}).items()
    }
    pixels = {
        name: [(pt[0], pt[1]) for pt in pts]
        for name, pts in cache.get("isolated_curve_pixels", {}).items()
    } if cache.get("isolated_curve_pixels") else None
    return state.model_copy(update={
        "digitized_curves": digitized,
        "censoring_marks": censoring,
        "isolated_curve_pixels": pixels,
    })


def _load_risk_table(csv_path: Path) -> RiskTable | None:
    with open(csv_path) as f:
        reader = csv.reader(f)
        header = next(reader)
        if len(header) < 2:
            return None
        group_names = header[1:]
        time_points: list[float] = []
        counts: dict[str, list[int]] = {g: [] for g in group_names}
        for row in reader:
            time_points.append(float(row[0]))
            for i, g in enumerate(group_names):
                counts[g].append(int(float(row[i + 1])))
    return RiskTable(
        time_points=time_points,
        groups=[RiskGroup(name=g, counts=counts[g]) for g in group_names],
    )


def _build_metadata(meta: dict, risk_table: RiskTable | None) -> PlotMetadata:
    groups = meta["groups"]
    curves = []
    for i, g in enumerate(groups):
        curves.append(
            CurveInfo(
                name=g,
                color_description=f"{LINE_STYLES[i % len(LINE_STYLES)]} {COLOR_NAMES[i % len(COLOR_NAMES)]}",
                line_style=LINE_STYLES[i % len(LINE_STYLES)],
            )
        )
    x = meta["x_axis"]
    y = meta["y_axis"]
    return PlotMetadata(
        x_axis=AxisConfig(
            label=x.get("label"),
            start=x["start"],
            end=x["end"],
            tick_interval=x.get("tick_interval"),
            tick_values=x["tick_values"],
            scale=x.get("scale", "linear"),
        ),
        y_axis=AxisConfig(
            label=y.get("label"),
            start=y["start"],
            end=y["end"],
            tick_interval=y.get("tick_interval"),
            tick_values=y["tick_values"],
            scale=y.get("scale", "linear"),
        ),
        curves=curves,
        risk_table=risk_table,
        curve_direction=meta.get("curve_direction", "downward"),
        title=meta.get("title"),
        annotations=meta.get("annotations", []),
    )


def _km_from_ipd(patients):
    """Reconstruct KM curve from patient records."""
    events = sorted([(p.time, p.event) for p in patients], key=lambda x: x[0])
    if not events:
        return [(0.0, 1.0)]
    n = len(events)
    coords = [(0.0, 1.0)]
    survival = 1.0
    at_risk = n
    i = 0
    while i < n:
        t = events[i][0]
        d = 0
        c = 0
        while i < n and events[i][0] == t:
            if events[i][1]:
                d += 1
            else:
                c += 1
            i += 1
        if d > 0 and at_risk > 0:
            survival *= 1.0 - d / at_risk
            coords.append((t, survival))
        at_risk -= d + c
    return coords


def _mae(curve1, curve2):
    """MAE between two step-function curves."""
    from bisect import bisect_right

    def build(c):
        s = sorted(c, key=lambda p: p[0])
        return [p[0] for p in s], [p[1] for p in s]

    def surv_at(lookup, t):
        times, vals = lookup
        idx = bisect_right(times, t) - 1
        if idx < 0:
            return vals[0] if vals else 1.0
        return vals[idx]

    l1, l2 = build(curve1), build(curve2)
    all_t = sorted(set(l1[0] + l2[0]))
    if len(all_t) < 2:
        return 0.0
    errors = [abs(surv_at(l1, t) - surv_at(l2, t)) for t in all_t]
    return float(np.mean(errors))


def main():
    parser = argparse.ArgumentParser(description="Reconstruction benchmark")
    parser.add_argument("--generate-cache", action="store_true",
                        help="Run digitization and save cache for all cases")
    parser.add_argument("--no-cache", action="store_true",
                        help="Force re-digitization (don't load or save cache)")
    args = parser.parse_args()

    use_cache = not args.generate_cache and not args.no_cache
    save_cache = args.generate_cache

    all_val_mae = []
    all_gt_mae = []
    all_valid_gt_mae = []
    cache_hits = 0
    cache_misses = 0

    for case_name in CASES:
        case_dir = FIXTURE_DIR / case_name
        if not case_dir.exists():
            continue
        meta = json.loads((case_dir / "metadata.json").read_text())
        image_path = str(case_dir / "graph.png")

        # Load ground-truth risk table
        rt_path = case_dir / "risk_table_data.csv"
        risk_table = _load_risk_table(rt_path) if rt_path.exists() else None

        # Load ground-truth curves
        gt_curves: dict[str, list[tuple[float, float]]] = {}
        gt_csv = case_dir / "ground_truth.csv"
        if gt_csv.exists():
            with open(gt_csv) as f:
                reader = csv.DictReader(f)
                for row in reader:
                    group = row["group"]
                    if group not in gt_curves:
                        gt_curves[group] = []
                    gt_curves[group].append((float(row["time"]), float(row["survival_probability"])))

        # Build metadata WITH risk table
        plot_meta = _build_metadata(meta, risk_table)

        # Digitize (or load from cache)
        state = PipelineState(
            image_path=image_path,
            preprocessed_image_path=image_path,
            plot_metadata=plot_meta,
        )

        cached = _load_cache(case_dir) if use_cache else None
        if cached is not None:
            state = _apply_cache(state, cached)
            cache_hits += 1
        else:
            cache_misses += 1
            state = digitize_v5(state)
            if save_cache and not state.errors and state.digitized_curves:
                _save_cache(case_dir, state)

        if state.errors or not state.digitized_curves:
            print(f"\n{case_name}: digitization failed")
            continue

        # Reconstruct
        state = reconstruct(state)
        if not state.output:
            print(f"\n{case_name}: reconstruction failed")
            continue

        # Validate
        state = validate(state)

        # Check for digitization issues (initial point far from 1.0)
        dig_ok = True
        for name, coords in state.digitized_curves.items():
            if coords and abs(coords[0][1] - 1.0) > 0.2:
                dig_ok = False

        print(f"\n{case_name} ({len(meta['groups'])} arms, mode={state.output.reconstruction_mode.value})" +
              (" [DIG_ERR]" if not dig_ok else ""))

        for curve in state.output.curves:
            recon_km = _km_from_ipd(curve.patients)
            val_mae = curve.validation_mae or 0.0

            # Compare vs ground truth
            gt_mae = float("nan")
            if curve.group_name in gt_curves:
                gt_mae = _mae(recon_km, gt_curves[curve.group_name])

            # Compare digitized vs ground truth (for reference)
            dig_mae = float("nan")
            if curve.group_name in gt_curves and curve.group_name in state.digitized_curves:
                dig_mae = _mae(
                    state.digitized_curves[curve.group_name],
                    gt_curves[curve.group_name],
                )

            all_val_mae.append(val_mae)
            if not np.isnan(gt_mae):
                all_gt_mae.append(gt_mae)
                if dig_ok:
                    all_valid_gt_mae.append(gt_mae)

            marker = " ***" if gt_mae > 0.01 else ""
            print(
                f"  {curve.group_name}: "
                f"val_MAE={val_mae:.4f} | "
                f"recon_vs_GT={gt_mae:.4f} | "
                f"dig_vs_GT={dig_mae:.4f} | "
                f"n={len(curve.patients)}{marker}"
            )

    # Summary
    print(f"\n{'=' * 60}")
    print(f"Cache: {cache_hits} hits, {cache_misses} misses")
    print("SUMMARY (all arms)")
    if all_val_mae:
        print(f"  Validation MAE (recon vs digitized): mean={np.mean(all_val_mae):.4f} median={np.median(all_val_mae):.4f}")
    if all_gt_mae:
        print(f"  Reconstruction MAE (recon vs GT):     mean={np.mean(all_gt_mae):.4f} median={np.median(all_gt_mae):.4f}")
        ok = sum(1 for m in all_gt_mae if m < 0.01)
        print(f"  Arms < 0.01 MAE: {ok}/{len(all_gt_mae)}")
    if all_valid_gt_mae:
        print(f"\nSUMMARY (valid digitization only)")
        print(f"  Reconstruction MAE (recon vs GT):     mean={np.mean(all_valid_gt_mae):.4f} median={np.median(all_valid_gt_mae):.4f}")
        ok = sum(1 for m in all_valid_gt_mae if m < 0.01)
        print(f"  Arms < 0.01 MAE: {ok}/{len(all_valid_gt_mae)}")
        ok2 = sum(1 for m in all_valid_gt_mae if m < 0.02)
        print(f"  Arms < 0.02 MAE: {ok2}/{len(all_valid_gt_mae)}")


if __name__ == "__main__":
    main()
