"""Quick 6-case diagnostic benchmark for digitization_v5.

Bypasses MMPU by constructing PlotMetadata directly from fixture metadata.json.
Computes per-arm MAE at hard-point landmarks.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from km_estimator.models import PipelineState
from km_estimator.models.plot_metadata import AxisConfig, CurveInfo, PlotMetadata, RiskTable, RiskGroup
from km_estimator.nodes.digitization_5 import digitize_v5

FIXTURE_DIR = ROOT / "tests" / "fixtures" / "standard"
CASES = [
    # All 3-arm cases (orange + green curves — key test for these changes)
    "case_002", "case_012", "case_014", "case_017", "case_023", "case_026",
    "case_029", "case_037", "case_049", "case_060", "case_067", "case_073",
    "case_087", "case_093", "case_096", "case_100",
    # 2-arm sample across difficulties
    "case_004", "case_015", "case_018", "case_032", "case_036",
    "case_050", "case_070", "case_080", "case_082", "case_095",
]

COLOR_NAMES = ["blue", "orange", "green", "red", "purple"]
LINE_STYLES = ["solid", "dashed"]


def _build_metadata(meta: dict) -> PlotMetadata:
    groups = meta["groups"]
    curves = []
    for i, g in enumerate(groups):
        color_name = COLOR_NAMES[i % len(COLOR_NAMES)]
        line_style = LINE_STYLES[i % len(LINE_STYLES)]
        curves.append(CurveInfo(
            name=g,
            color_description=f"{line_style} {color_name}",
            line_style=line_style,
        ))
    x = meta["x_axis"]
    y = meta["y_axis"]
    # Build risk table if available
    risk_table = None
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


def _compute_arm_mae(
    digitized: dict[str, list[tuple[float, float]]],
    hard_points: dict,
    curve_direction: str,
    y_start: float,
    y_end: float,
) -> dict[str, float]:
    results = {}
    for arm_name, hp_data in hard_points.items():
        landmarks = hp_data.get("landmarks", [])
        if not landmarks:
            results[arm_name] = float("nan")
            continue
        dig_pts = digitized.get(arm_name, [])
        if not dig_pts:
            results[arm_name] = float("nan")
            continue
        times = np.array([p[0] for p in dig_pts])
        values = np.array([p[1] for p in dig_pts])
        errors = []
        for lm in landmarks:
            t = lm["time"]
            expected = lm["survival"]
            # Skip landmarks where the expected value is below the visible
            # Y-axis range — these cannot be digitized from the plot image.
            if expected < y_start:
                continue
            if len(times) == 0:
                errors.append(abs(expected))
                continue
            idx = np.searchsorted(times, t, side="right") - 1
            idx = max(0, min(idx, len(values) - 1))
            predicted = values[idx]
            errors.append(abs(predicted - expected))
        results[arm_name] = float(np.mean(errors))
    return results


def main():
    all_results = {}
    for case_name in CASES:
        case_dir = FIXTURE_DIR / case_name
        meta = json.loads((case_dir / "metadata.json").read_text())
        hard_points = json.loads((case_dir / "hard_points.json").read_text())
        image_path = str(case_dir / "graph.png")

        plot_meta = _build_metadata(meta)
        state = PipelineState(
            image_path=image_path,
            preprocessed_image_path=image_path,
            plot_metadata=plot_meta,
        )
        print(f"\n{'='*60}")
        print(f"Running {case_name} ({len(meta['groups'])} arms)...")
        result = digitize_v5(state)

        if result.errors:
            print(f"  ERRORS: {[e.message for e in result.errors]}")
            all_results[case_name] = {"error": True}
            continue

        if result.digitized_curves is None:
            print(f"  No digitized curves!")
            all_results[case_name] = {"error": True}
            continue

        arm_mae = _compute_arm_mae(
            result.digitized_curves,
            hard_points,
            meta.get("curve_direction", "downward"),
            meta["y_axis"]["start"],
            meta["y_axis"]["end"],
        )
        all_results[case_name] = arm_mae
        for arm, mae in arm_mae.items():
            print(f"  {arm}: MAE={mae:.4f}")

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


if __name__ == "__main__":
    main()
