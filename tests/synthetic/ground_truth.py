"""Ground truth construction, persistence, and comparison utilities.

Handles the 7-file-per-case format:
  metadata.json, raw_survival_data.csv, risk_table_data.csv,
  ground_truth.csv, hard_points.json, graph_draft.png, graph.png
"""

from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path

import numpy as np

from km_estimator.models.ipd_output import (
    CurveIPD,
    IPDOutput,
    PatientRecord,
    ReconstructionMode,
)
from km_estimator.models.plot_metadata import (
    AxisConfig,
    CurveInfo,
    PlotMetadata,
    RiskGroup,
    RiskTable,
)
from km_estimator.utils.shape_metrics import (
    dtw_distance,
    frechet_distance,
    max_error,
    rmse,
)

from .data_gen import SyntheticCurveData, SyntheticTestCase, _compute_greenwood_ci, _km_from_ipd
from .modifiers import (
    BackgroundStyle,
    CurveDirection,
    FontTypography,
    FrameLayout,
    Modifier,
    get_modifier_names,
)


def _extract_style_profile(modifiers: list[Modifier]) -> dict[str, str]:
    """Extract style directives from figure modifiers for metadata/debugging."""
    profile = {
        "background_style": "white",
        "curve_direction": "downward",
        "frame_layout": "full_box",
        "font_typography": "sans",
    }
    for mod in modifiers:
        if isinstance(mod, BackgroundStyle):
            profile["background_style"] = mod.style
        elif isinstance(mod, CurveDirection):
            profile["curve_direction"] = mod.direction
        elif isinstance(mod, FrameLayout):
            profile["frame_layout"] = mod.layout
        elif isinstance(mod, FontTypography):
            profile["font_typography"] = mod.family
    return profile


def build_ground_truth_metadata(test_case: SyntheticTestCase) -> PlotMetadata:
    """Build the PlotMetadata the pipeline should ideally extract."""
    style_profile = _extract_style_profile(test_case.modifiers)
    curve_direction = style_profile["curve_direction"]
    if not test_case.modifiers and test_case.curve_direction in ("downward", "upward"):
        curve_direction = test_case.curve_direction
    curves = [
        CurveInfo(
            name=c.group_name,
            color_description=f"{c.line_style} {c.color_name}",
            line_style=c.line_style,
        )
        for c in test_case.curves
    ]
    return PlotMetadata(
        x_axis=test_case.x_axis,
        y_axis=test_case.y_axis,
        curves=curves,
        risk_table=test_case.risk_table,
        title=test_case.title,
        annotations=test_case.annotations,
        curve_direction=curve_direction,
    )


def build_ground_truth_ipd(test_case: SyntheticTestCase) -> IPDOutput:
    """Build the IPDOutput the pipeline should ideally produce."""
    curve_ipds = []
    for c in test_case.curves:
        curve_ipds.append(
            CurveIPD(
                group_name=c.group_name,
                patients=c.patients,
                censoring_times=c.censoring_times,
                digitization_confidence=1.0,
            )
        )
    return IPDOutput(
        metadata=build_ground_truth_metadata(test_case),
        curves=curve_ipds,
        reconstruction_mode=(
            ReconstructionMode.FULL if test_case.risk_table else ReconstructionMode.ESTIMATED
        ),
    )


def compute_hard_points(test_case: SyntheticTestCase) -> dict:
    """Compute landmark survival probabilities and median/quartile survival times."""
    result = {}
    direction = (
        test_case.curve_direction
        if test_case.curve_direction in ("downward", "upward")
        else "downward"
    )
    plot_space = "incidence" if direction == "upward" else "survival"

    for curve in test_case.curves:
        step_coords = curve.step_coords

        # The curve is only drawn up to the last event time
        last_event_t = step_coords[-1][0] if len(step_coords) > 1 else 0.0

        # Standard clinical time points (only those within the drawn curve)
        candidate_times = [6, 12, 24, 36, 48, 60, 96, 120]
        time_labels = {
            6: "6-month value",
            12: "1-year value",
            24: "2-year value",
            36: "3-year value",
            48: "4-year value",
            60: "5-year value",
            96: "8-year value",
            120: "10-year value",
        }

        landmarks = []
        for t in candidate_times:
            if t <= last_event_t:
                s = _get_survival_at_step(step_coords, float(t))
                plot_y = (1.0 - s) if direction == "upward" else s
                landmarks.append(
                    {
                        "time": t,
                        "survival": round(s, 4),
                        "plot_y": round(plot_y, 4),
                        "description": time_labels[t],
                    }
                )

        # Median and quartile survival times
        quartiles = {}
        for pct, label in [(0.25, "25pct"), (0.50, "50pct"), (0.75, "75pct")]:
            t = _find_survival_time(step_coords, 1.0 - pct)
            if t is not None:
                quartiles[label] = round(t, 2)

        median = quartiles.get("50pct")

        result[curve.group_name] = {
            "landmarks": landmarks,
            "median_survival": median,
            "quartiles": quartiles,
            "curve_direction": direction,
            "plot_space": plot_space,
            "evaluation_space": "survival",
        }

    return result


def _get_survival_at_step(coords: list[tuple[float, float]], t: float) -> float:
    """Step-function lookup."""
    if not coords:
        return 1.0
    if t < coords[0][0]:
        return 1.0
    for i in range(len(coords) - 1, -1, -1):
        if coords[i][0] <= t:
            return coords[i][1]
    return coords[0][1]


def _find_survival_time(coords: list[tuple[float, float]], target_survival: float) -> float | None:
    """Find the first time survival drops to or below target_survival."""
    for t, s in coords:
        if s <= target_survival:
            return t
    return None


# --- Persistence: save/load 7-file format ---


def save_test_case(test_case: SyntheticTestCase, output_dir: Path) -> None:
    """Write all 7 files for a test case to output_dir."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    style_profile = _extract_style_profile(test_case.modifiers)
    # 1. metadata.json
    metadata = {
        "name": test_case.name,
        "seed": test_case.seed,
        "n_groups": len(test_case.curves),
        "n_patients_per_arm": (len(test_case.curves[0].patients) if test_case.curves else 0),
        "groups": [c.group_name for c in test_case.curves],
        "max_time": test_case.x_axis.end,
        "y_axis_start": test_case.y_axis.start,
        "difficulty": test_case.difficulty,
        "tier": test_case.tier,
        "gap_pattern": test_case.gap_pattern,
        "background_style": style_profile["background_style"],
        "curve_direction": style_profile["curve_direction"],
        "frame_layout": style_profile["frame_layout"],
        "font_typography": style_profile["font_typography"],
        "dpi": 150,
        "modifiers": get_modifier_names(test_case.modifiers),
        "title": test_case.title,
        "annotations": test_case.annotations,
        "x_axis": test_case.x_axis.model_dump(),
        "y_axis": test_case.y_axis.model_dump(),
        "total_patients": sum(len(c.patients) for c in test_case.curves),
        "total_events": sum(sum(1 for p in c.patients if p.event) for c in test_case.curves),
        "hard_points_space": "survival",
        "plot_space_for_upward": "incidence",
    }
    _write_json(output_dir / "metadata.json", metadata)

    # 2. raw_survival_data.csv
    with open(output_dir / "raw_survival_data.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["patient_id", "group", "time", "event"])
        pid = 0
        for curve in test_case.curves:
            for patient in curve.patients:
                writer.writerow([pid, curve.group_name, patient.time, int(patient.event)])
                pid += 1

    # 3. risk_table_data.csv
    if test_case.risk_table:
        rt = test_case.risk_table
        with open(output_dir / "risk_table_data.csv", "w", newline="") as f:
            writer = csv.writer(f)
            header = ["time"] + [g.name for g in rt.groups]
            writer.writerow(header)
            for i, t in enumerate(rt.time_points):
                row = [t] + [g.counts[i] for g in rt.groups]
                writer.writerow(row)
    else:
        # Write empty file
        with open(output_dir / "risk_table_data.csv", "w", newline="") as f:
            f.write("time\n")

    # 4. ground_truth.csv (dense KM coords with Greenwood CI)
    with open(output_dir / "ground_truth.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["time", "group", "survival_probability", "ci_lower", "ci_upper"])
        for curve in test_case.curves:
            # Evaluate at 1000 points
            eval_times = np.linspace(0, test_case.x_axis.end, 1000)
            ci_data = _compute_greenwood_ci(curve.patients, eval_times)
            for t, (s, ci_lo, ci_hi) in zip(eval_times, ci_data):
                writer.writerow(
                    [
                        round(float(t), 6),
                        curve.group_name,
                        round(s, 6),
                        round(ci_lo, 6),
                        round(ci_hi, 6),
                    ]
                )

    # 5. hard_points.json
    hard_points = compute_hard_points(test_case)
    _write_json(output_dir / "hard_points.json", hard_points)

    # 6 & 7. graph_draft.png and graph.png are written by renderer.py
    # (they should already exist at this point)


def load_test_case(case_dir: Path) -> SyntheticTestCase:
    """Load a test case from disk."""
    case_dir = Path(case_dir)

    # Load metadata
    with open(case_dir / "metadata.json") as f:
        metadata = json.load(f)

    # Load raw survival data
    patients_by_group: dict[str, list[PatientRecord]] = {}
    with open(case_dir / "raw_survival_data.csv") as f:
        reader = csv.DictReader(f)
        for row in reader:
            group = row["group"]
            if group not in patients_by_group:
                patients_by_group[group] = []
            patients_by_group[group].append(
                PatientRecord(time=float(row["time"]), event=bool(int(row["event"])))
            )

    # Load risk table
    risk_table = None
    rt_path = case_dir / "risk_table_data.csv"
    if rt_path.exists():
        with open(rt_path) as f:
            reader = csv.reader(f)
            header = next(reader)
            if len(header) > 1:  # has actual groups
                group_names_rt = header[1:]
                time_points = []
                counts: dict[str, list[int]] = {g: [] for g in group_names_rt}
                for row_data in reader:
                    time_points.append(float(row_data[0]))
                    for i, g in enumerate(group_names_rt):
                        counts[g].append(int(row_data[i + 1]))
                risk_table = RiskTable(
                    time_points=time_points,
                    groups=[RiskGroup(name=g, counts=counts[g]) for g in group_names_rt],
                )

    # Build curves
    group_order = metadata.get("groups", list(patients_by_group.keys()))
    from .data_gen import COLOR_NAMES, COLORS, LINE_STYLES, _compute_n_at_risk

    curves = []
    for i, group_name in enumerate(group_order):
        patients = patients_by_group.get(group_name, [])
        step_coords = _km_from_ipd(patients)
        censoring_times = [p.time for p in patients if not p.event]
        censoring_times.sort()

        rt_time_points = risk_table.time_points if risk_table else []
        n_at_risk = _compute_n_at_risk(patients, rt_time_points)

        curves.append(
            SyntheticCurveData(
                group_name=group_name,
                patients=patients,
                step_coords=step_coords,
                censoring_times=censoring_times,
                n_at_risk=n_at_risk,
                color=COLORS[i % len(COLORS)],
                color_name=COLOR_NAMES[i % len(COLOR_NAMES)],
                line_style=LINE_STYLES[i % len(LINE_STYLES)],
            )
        )

    x_axis_data = metadata.get("x_axis")
    y_axis_data = metadata.get("y_axis")
    if isinstance(x_axis_data, dict) and isinstance(y_axis_data, dict):
        x_axis = AxisConfig.model_validate(x_axis_data)
        y_axis = AxisConfig.model_validate(y_axis_data)
    else:
        max_time = metadata.get("max_time", 60.0)
        y_start = metadata.get("y_axis_start", 0.0)

        x_ticks_interval = max_time / 6
        x_ticks = [round(t, 2) for t in np.arange(0, max_time + 0.01, x_ticks_interval)]
        if not any(abs(t - max_time) < 1e-6 for t in x_ticks):
            x_ticks.append(round(max_time, 2))
        x_ticks = sorted(set(x_ticks))
        y_ticks = [round(v, 1) for v in np.arange(y_start, 1.01, 0.2)]
        if 1.0 not in y_ticks:
            y_ticks.append(1.0)

        x_axis = AxisConfig(
            label="Time (months)",
            start=0.0,
            end=max_time,
            tick_interval=x_ticks_interval,
            tick_values=x_ticks,
            scale="linear",
        )
        y_axis = AxisConfig(
            label="Survival Probability",
            start=y_start,
            end=1.0,
            tick_interval=0.2,
            tick_values=y_ticks,
            scale="linear",
        )

    # Reconstruct modifier list from names (for metadata only)
    modifiers: list[Modifier] = []

    graph_path = case_dir / "graph.png"
    draft_path = case_dir / "graph_draft.png"

    return SyntheticTestCase(
        name=metadata.get("name", case_dir.name),
        seed=metadata.get("seed", 0),
        curves=curves,
        x_axis=x_axis,
        y_axis=y_axis,
        risk_table=risk_table,
        title=metadata.get("title"),
        annotations=metadata.get("annotations", []),
        modifiers=modifiers,
        difficulty=metadata.get("difficulty", 1),
        tier=metadata.get("tier", "standard"),
        gap_pattern=metadata.get("gap_pattern"),
        curve_direction=(
            str(metadata.get("curve_direction", "downward")).lower()
            if str(metadata.get("curve_direction", "downward")).lower() in ("downward", "upward")
            else "downward"
        ),
        image_path=str(graph_path) if graph_path.exists() else None,
        draft_image_path=str(draft_path) if draft_path.exists() else None,
    )


def save_manifest(
    cases: list[SyntheticTestCase],
    output_dir: Path,
) -> None:
    """Write manifest.json indexing all cases."""
    manifest = {
        "generated_at": datetime.now().isoformat(),
        "n_cases": len(cases),
        "cases": [
            {
                "name": c.name,
                "seed": c.seed,
                "difficulty": c.difficulty,
                "tier": c.tier,
                "gap_pattern": c.gap_pattern,
                "background_style": style["background_style"],
                "curve_direction": style["curve_direction"],
                "frame_layout": style["frame_layout"],
                "font_typography": style["font_typography"],
                "modifiers": get_modifier_names(c.modifiers),
                "n_curves": len(c.curves),
                "dir": f"{c.name}/",
            }
            for c in cases
            for style in [_extract_style_profile(c.modifiers)]
        ],
    }
    _write_json(Path(output_dir) / "manifest.json", manifest)


def load_manifest(profile_dir: Path) -> list[dict]:
    """Load manifest and return list of case entries."""
    manifest_path = Path(profile_dir) / "manifest.json"
    with open(manifest_path) as f:
        data = json.load(f)
    return data.get("cases", [])


# --- Comparison utilities ---


def compare_digitized_curves(
    actual: dict[str, list[tuple[float, float]]],
    expected: dict[str, list[tuple[float, float]]],
) -> dict[str, dict[str, float]]:
    """Compare pipeline-output digitized curves against ground truth."""
    results = {}
    for name in expected:
        if name not in actual:
            results[name] = {"error": float("inf"), "missing": True}
            continue
        a = actual[name]
        e = expected[name]
        results[name] = {
            "mae": _calculate_mae(e, a),
            "rmse": rmse(e, a),
            "dtw": dtw_distance(e, a),
            "frechet": frechet_distance(e, a),
            "max_error": max_error(e, a),
        }
    return results


def compare_hard_points(
    actual_ipd: IPDOutput,
    hard_points: dict,
    tolerance: float = 0.05,
) -> dict[str, dict]:
    """Check if reconstructed IPD matches landmark survival values."""
    results = {}

    direction = (
        actual_ipd.metadata.curve_direction
        if actual_ipd.metadata.curve_direction in ("downward", "upward")
        else "downward"
    )

    for curve in actual_ipd.curves:
        group = curve.group_name
        if group not in hard_points:
            results[group] = {"error": "group not in hard_points"}
            continue

        expected = hard_points[group]
        reconstructed_km = _km_from_ipd(curve.patients)

        landmark_results = []
        for lm in expected.get("landmarks", []):
            t = lm["time"]
            expected_s = lm["survival"]
            expected_plot_y = lm.get(
                "plot_y", expected_s if direction == "downward" else 1.0 - expected_s
            )
            actual_s = _get_survival_at_step_local(reconstructed_km, float(t))
            actual_plot_y = actual_s if direction == "downward" else (1.0 - actual_s)
            error = abs(actual_s - expected_s)
            plot_error = abs(float(actual_plot_y) - float(expected_plot_y))
            landmark_results.append(
                {
                    "time": t,
                    "expected": expected_s,
                    "actual": round(actual_s, 4),
                    "error": round(error, 4),
                    "expected_plot_y": round(float(expected_plot_y), 4),
                    "actual_plot_y": round(float(actual_plot_y), 4),
                    "plot_error": round(float(plot_error), 4),
                    "pass": error <= tolerance,
                }
            )

        results[group] = {
            "landmarks": landmark_results,
            "all_pass": all(lr["pass"] for lr in landmark_results),
            "evaluation_space": "survival",
            "plot_space": ("incidence" if direction == "upward" else "survival"),
        }

    return results


def _get_survival_at_step_local(coords: list[tuple[float, float]], t: float) -> float:
    if not coords:
        return 1.0
    if t < coords[0][0]:
        return 1.0
    for i in range(len(coords) - 1, -1, -1):
        if coords[i][0] <= t:
            return coords[i][1]
    return coords[0][1]


def _calculate_mae(
    curve1: list[tuple[float, float]],
    curve2: list[tuple[float, float]],
) -> float:
    """Calculate MAE between two step-function curves."""
    if not curve1 or not curve2:
        return 1.0
    errors = []
    for t, s1 in curve1:
        s2 = _get_survival_at_step_local(curve2, t)
        errors.append(abs(s1 - s2))
    return float(np.mean(errors)) if errors else 0.0


def _write_json(path: Path, data: dict) -> None:
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
