"""Diagnostic: inspect evidence cube and candidate masks for problem cases."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from km_estimator.models.plot_metadata import AxisConfig, CurveInfo, PlotMetadata
from km_estimator.models.ipd_output import ProcessingStage
from km_estimator.nodes.digitization_5.axis_map import build_plot_model
from km_estimator.nodes.digitization_5.legend_color import build_color_models
from km_estimator.nodes.digitization_5.probability_map import build_evidence_cube
from km_estimator.nodes.digitization_5 import cv_utils

FIXTURE_DIR = ROOT / "tests" / "fixtures" / "standard"
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
    for case_name in ["case_020", "case_060", "case_100"]:
        case_dir = FIXTURE_DIR / case_name
        meta = json.loads((case_dir / "metadata.json").read_text())
        image_path = str(case_dir / "graph.png")
        plot_meta = _build_metadata(meta)

        image = cv_utils.load_image(image_path, stage=ProcessingStage.DIGITIZE)
        plot_model = build_plot_model(image, plot_meta, None)
        color_models, color_warnings = build_color_models(image, plot_meta, plot_model)

        print(f"\n{'='*60}")
        print(f"{case_name}")
        print(f"{'='*60}")
        print(f"  Color warnings: {color_warnings}")
        for name, m in color_models.items():
            print(f"  {name}: reliability={m.reliability:.2f} source={m.source}")

        evidence = build_evidence_cube(
            image=image,
            plot_model=plot_model,
            color_models=color_models,
            hardpoint_guides=None,
        )
        print(f"  Evidence warnings: {evidence.warning_codes}")

        # Analyze candidate masks
        print(f"\n  Candidate masks:")
        for arm_name, mask in evidence.arm_candidate_masks.items():
            n_pixels = int(np.sum(mask > 0))
            total = mask.shape[0] * mask.shape[1]
            print(f"    {arm_name}: {n_pixels} pixels ({100*n_pixels/total:.2f}%)")

        # Analyze score maps
        print(f"\n  Score map statistics:")
        for arm_name, score_map in evidence.arm_score_maps.items():
            valid = score_map[score_map > 0]
            if valid.size > 0:
                print(f"    {arm_name}: mean={valid.mean():.3f} p50={np.percentile(valid,50):.3f} p90={np.percentile(valid,90):.3f} p99={np.percentile(valid,99):.3f} max={valid.max():.3f} n_pos={valid.size}")
            else:
                print(f"    {arm_name}: ALL ZERO scores")

        # Check if arm candidate masks overlap
        if len(evidence.arm_candidate_masks) >= 2:
            masks = list(evidence.arm_candidate_masks.values())
            names = list(evidence.arm_candidate_masks.keys())
            print(f"\n  Candidate mask overlap:")
            for i in range(len(masks)):
                for j in range(i+1, len(masks)):
                    overlap = int(np.sum((masks[i] > 0) & (masks[j] > 0)))
                    print(f"    {names[i]} & {names[j]}: {overlap} overlapping pixels")


if __name__ == "__main__":
    main()
