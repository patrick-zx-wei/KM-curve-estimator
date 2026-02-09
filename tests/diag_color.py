"""Diagnostic: dump color model details for 6 cases.

Shows what observed LAB centers are found, reliability scores, and
which color source (legend vs observed) is used per arm.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from km_estimator.models.plot_metadata import AxisConfig, CurveInfo, PlotMetadata
from km_estimator.models import PipelineState
from km_estimator.nodes.digitization_5.axis_map import build_plot_model
from km_estimator.nodes.digitization_5.legend_color import (
    build_color_models,
    _collect_observed_centers,
    _rgb01_to_lab,
)
from km_estimator.nodes.digitization.curve_isolation import parse_curve_color
from km_estimator.nodes.digitization_5 import cv_utils
from km_estimator.models.ipd_output import ProcessingStage

FIXTURE_DIR = ROOT / "tests" / "fixtures" / "standard"
CASES = ["case_001", "case_020", "case_040", "case_060", "case_080", "case_100"]
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
    return PlotMetadata(
        x_axis=AxisConfig(label=x.get("label"), start=x["start"], end=x["end"],
                          tick_interval=x.get("tick_interval"), tick_values=x["tick_values"],
                          scale=x.get("scale", "linear")),
        y_axis=AxisConfig(label=y.get("label"), start=y["start"], end=y["end"],
                          tick_interval=y.get("tick_interval"), tick_values=y["tick_values"],
                          scale=y.get("scale", "linear")),
        curves=curves, risk_table=None,
        curve_direction=meta.get("curve_direction", "downward"),
        title=meta.get("title"), annotations=meta.get("annotations", []),
    )


def main():
    for case_name in CASES:
        case_dir = FIXTURE_DIR / case_name
        meta = json.loads((case_dir / "metadata.json").read_text())
        image_path = str(case_dir / "graph.png")
        plot_meta = _build_metadata(meta)

        image = cv_utils.load_image(image_path, stage=ProcessingStage.DIGITIZE)
        plot_model = build_plot_model(image, plot_meta, None)

        print(f"\n{'='*60}")
        print(f"{case_name} ({len(meta['groups'])} arms)")
        print(f"{'='*60}")

        # Show expected LAB from color descriptions
        for curve in plot_meta.curves:
            rgb = parse_curve_color(curve.color_description)
            if rgb is not None:
                lab = _rgb01_to_lab(rgb)
                print(f"  {curve.name}: color_desc='{curve.color_description}' -> RGB={tuple(round(v,3) for v in rgb)} -> LAB=({lab[0]:.1f}, {lab[1]:.1f}, {lab[2]:.1f})")
            else:
                print(f"  {curve.name}: color_desc='{curve.color_description}' -> parse FAILED")

        # Build color models and show details
        models, warnings = build_color_models(image, plot_meta, plot_model)
        print(f"\n  Color warnings: {warnings}")
        print(f"\n  Per-arm color models:")
        for name, m in models.items():
            exp_str = f"({m.expected_lab[0]:.1f}, {m.expected_lab[1]:.1f}, {m.expected_lab[2]:.1f})" if m.expected_lab else "None"
            obs_str = f"({m.observed_lab[0]:.1f}, {m.observed_lab[1]:.1f}, {m.observed_lab[2]:.1f})" if m.observed_lab else "None"
            ref = m.reference_lab()
            ref_str = f"({ref[0]:.1f}, {ref[1]:.1f}, {ref[2]:.1f})" if ref else "None"
            print(f"    {name}: source={m.source} valid={m.valid} reliability={m.reliability:.2f}")
            print(f"      expected={exp_str}  observed={obs_str}  reference={ref_str}")
            print(f"      warnings={m.warning_codes}")

        # Also show raw k-means centers
        x0, y0, x1, y1 = plot_model.plot_region
        roi = image[y0:y1, x0:x1]
        exclude_mask = cv2.bitwise_or(plot_model.axis_mask, plot_model.tick_mask)
        n_curves = len(meta["groups"])
        centers = _collect_observed_centers(roi, exclude_mask, n_centers=max(1, n_curves))
        print(f"\n  Raw k-means centers (K={max(1, n_curves)}):")
        for i, c in enumerate(centers):
            print(f"    Center {i}: LAB=({c[0]:.1f}, {c[1]:.1f}, {c[2]:.1f})")


if __name__ == "__main__":
    main()
