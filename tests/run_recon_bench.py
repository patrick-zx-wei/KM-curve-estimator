"""Reconstruction benchmark.

Digitizes with ground-truth metadata, injects ground-truth risk table,
runs reconstruct() + validate(), and measures MAE.

Usage:
    python tests/run_recon_bench.py                  # use cached digitization (fast)
    python tests/run_recon_bench.py --generate-cache # digitize all cases & save cache
    python tests/run_recon_bench.py --no-cache       # force re-digitization (no save)
    python tests/run_recon_bench.py --generate-cache --workers 3  # parallel digitization
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
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


@dataclass
class ArmResult:
    group_name: str
    val_mae: float
    gt_mae: float
    gt_iae: float
    gt_rmse: float
    dig_mae: float
    n_patients: int


@dataclass
class CaseResult:
    case_name: str
    n_groups: int
    mode: str = ""
    dig_ok: bool = True
    failed: bool = False
    fail_reason: str = ""
    cache_hit: bool = False
    arms: list[ArmResult] = field(default_factory=list)


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
    out_dir = case_dir / "output"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / CACHE_FILENAME).write_text(json.dumps(cache))


def _load_cache(case_dir: Path) -> dict | None:
    path = case_dir / "output" / CACHE_FILENAME
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


def _build_step_lookup(curve):
    from bisect import bisect_right
    s = sorted(curve, key=lambda p: p[0])
    times = [p[0] for p in s]
    vals = [p[1] for p in s]

    def surv_at(t):
        idx = bisect_right(times, t) - 1
        if idx < 0:
            return vals[0] if vals else 1.0
        return vals[idx]

    return times, vals, surv_at


def _mae(curve1, curve2):
    """MAE between two step-function curves."""
    t1, _, f1 = _build_step_lookup(curve1)
    t2, _, f2 = _build_step_lookup(curve2)
    all_t = sorted(set(t1 + t2))
    if len(all_t) < 2:
        return 0.0
    errors = [abs(f1(t) - f2(t)) for t in all_t]
    return float(np.mean(errors))


def _iae(curve1, curve2):
    """Integrated Absolute Error between two step-function curves, normalised by time span."""
    t1, _, f1 = _build_step_lookup(curve1)
    t2, _, f2 = _build_step_lookup(curve2)
    all_t = sorted(set(t1 + t2))
    if len(all_t) < 2:
        return 0.0
    total = 0.0
    for i in range(len(all_t) - 1):
        dt = all_t[i + 1] - all_t[i]
        total += abs(f1(all_t[i]) - f2(all_t[i])) * dt
    span = all_t[-1] - all_t[0]
    return float(total / max(1e-9, span))


def _rmse(curve1, curve2):
    """RMSE between two step-function curves."""
    t1, _, f1 = _build_step_lookup(curve1)
    t2, _, f2 = _build_step_lookup(curve2)
    all_t = sorted(set(t1 + t2))
    if len(all_t) < 2:
        return 0.0
    sq_errors = [(f1(t) - f2(t)) ** 2 for t in all_t]
    return float(np.sqrt(np.mean(sq_errors)))


def _process_case(
    case_name: str,
    use_cache: bool,
    save_cache: bool,
) -> CaseResult:
    """Process a single benchmark case. Safe to call from a worker process."""
    case_dir = FIXTURE_DIR / case_name
    if not case_dir.exists():
        return CaseResult(case_name=case_name, n_groups=0, failed=True, fail_reason="missing")

    meta = json.loads((case_dir / "ground_truth" / "metadata.json").read_text())
    image_path = str(case_dir / "input" / "graph.png")
    n_groups = len(meta["groups"])

    # Load ground-truth risk table
    rt_path = case_dir / "ground_truth" / "risk_table_data.csv"
    risk_table = _load_risk_table(rt_path) if rt_path.exists() else None

    # Load ground-truth curves
    gt_curves: dict[str, list[tuple[float, float]]] = {}
    gt_csv = case_dir / "ground_truth" / "ground_truth.csv"
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
    cache_hit = cached is not None
    if cache_hit:
        state = _apply_cache(state, cached)
    else:
        state = digitize_v5(state)
        if save_cache and not state.errors and state.digitized_curves:
            _save_cache(case_dir, state)

    if state.errors or not state.digitized_curves:
        return CaseResult(
            case_name=case_name, n_groups=n_groups,
            failed=True, fail_reason="digitization", cache_hit=cache_hit,
        )

    # Reconstruct
    state = reconstruct(state)
    if not state.output:
        return CaseResult(
            case_name=case_name, n_groups=n_groups,
            failed=True, fail_reason="reconstruction", cache_hit=cache_hit,
        )

    # Validate
    state = validate(state)

    # Check for digitization issues (initial point far from 1.0)
    dig_ok = True
    for name, coords in state.digitized_curves.items():
        if coords and abs(coords[0][1] - 1.0) > 0.2:
            dig_ok = False

    result = CaseResult(
        case_name=case_name,
        n_groups=n_groups,
        mode=state.output.reconstruction_mode.value,
        dig_ok=dig_ok,
        cache_hit=cache_hit,
    )

    for curve in state.output.curves:
        recon_km = _km_from_ipd(curve.patients)
        val_mae = curve.validation_mae or 0.0

        gt_mae = float("nan")
        gt_iae = float("nan")
        gt_rmse = float("nan")
        if curve.group_name in gt_curves:
            gt_ref = gt_curves[curve.group_name]
            gt_mae = _mae(recon_km, gt_ref)
            gt_iae = _iae(recon_km, gt_ref)
            gt_rmse = _rmse(recon_km, gt_ref)

        dig_mae = float("nan")
        if curve.group_name in gt_curves and curve.group_name in state.digitized_curves:
            dig_mae = _mae(
                state.digitized_curves[curve.group_name],
                gt_curves[curve.group_name],
            )

        result.arms.append(ArmResult(
            group_name=curve.group_name,
            val_mae=val_mae,
            gt_mae=gt_mae,
            gt_iae=gt_iae,
            gt_rmse=gt_rmse,
            dig_mae=dig_mae,
            n_patients=len(curve.patients),
        ))

    return result


def main():
    parser = argparse.ArgumentParser(description="Reconstruction benchmark")
    parser.add_argument("--generate-cache", action="store_true",
                        help="Run digitization and save cache for all cases")
    parser.add_argument("--no-cache", action="store_true",
                        help="Force re-digitization (don't load or save cache)")
    parser.add_argument("--workers", type=int, default=1,
                        help="Number of parallel workers (default: 1)")
    args = parser.parse_args()

    use_cache = not args.generate_cache and not args.no_cache
    save_cache = args.generate_cache
    workers = max(1, args.workers)

    # Collect results
    results: list[CaseResult] = []

    if workers == 1:
        for case_name in CASES:
            r = _process_case(case_name, use_cache, save_cache)
            _print_case_result(r)
            results.append(r)
    else:
        print(f"Running with {workers} parallel workers...")
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(_process_case, case_name, use_cache, save_cache): case_name
                for case_name in CASES
            }
            for future in as_completed(futures):
                r = future.result()
                _print_case_result(r)
                results.append(r)
        # Sort by case name for consistent summary
        results.sort(key=lambda r: r.case_name)

    # Summary
    _print_summary(results)


def _print_case_result(r: CaseResult) -> None:
    if r.failed:
        print(f"\n{r.case_name}: {r.fail_reason} failed")
        return

    tag = " [DIG_ERR]" if not r.dig_ok else ""
    print(f"\n{r.case_name} ({r.n_groups} arms, mode={r.mode}){tag}")
    for arm in r.arms:
        marker = " ***" if arm.gt_mae > 0.01 else ""
        print(
            f"  {arm.group_name}: "
            f"MAE={arm.gt_mae:.4f} | "
            f"IAE={arm.gt_iae:.4f} | "
            f"RMSE={arm.gt_rmse:.4f} | "
            f"dig_MAE={arm.dig_mae:.4f} | "
            f"n={arm.n_patients}{marker}"
        )


def _print_summary(results: list[CaseResult]) -> None:
    cache_hits = sum(1 for r in results if r.cache_hit)
    cache_misses = sum(1 for r in results if not r.cache_hit)

    all_val_mae: list[float] = []
    all_gt_mae: list[float] = []
    all_gt_iae: list[float] = []
    all_gt_rmse: list[float] = []

    for r in results:
        if r.failed:
            continue
        for arm in r.arms:
            all_val_mae.append(arm.val_mae)
            if not np.isnan(arm.gt_mae):
                all_gt_mae.append(arm.gt_mae)
            if not np.isnan(arm.gt_iae):
                all_gt_iae.append(arm.gt_iae)
            if not np.isnan(arm.gt_rmse):
                all_gt_rmse.append(arm.gt_rmse)

    print(f"\n{'=' * 60}")
    print(f"Cache: {cache_hits} hits, {cache_misses} misses")
    print(f"SUMMARY ({len(all_gt_mae)} arms)")
    if all_val_mae:
        print(f"  Validation MAE (recon vs digitized): mean={np.mean(all_val_mae):.4f} median={np.median(all_val_mae):.4f}")
    if all_gt_mae:
        print(f"  MAE  (recon vs GT): mean={np.mean(all_gt_mae):.4f} median={np.median(all_gt_mae):.4f}")
        ok = sum(1 for m in all_gt_mae if m < 0.01)
        print(f"    Arms < 1% MAE: {ok}/{len(all_gt_mae)}")
        ok2 = sum(1 for m in all_gt_mae if m < 0.02)
        print(f"    Arms < 2% MAE: {ok2}/{len(all_gt_mae)}")
    if all_gt_iae:
        print(f"  IAE  (recon vs GT): mean={np.mean(all_gt_iae):.4f} median={np.median(all_gt_iae):.4f}")
    if all_gt_rmse:
        print(f"  RMSE (recon vs GT): mean={np.mean(all_gt_rmse):.4f} median={np.median(all_gt_rmse):.4f}")


if __name__ == "__main__":
    main()
