"""Diagnostic: dump per-arm pipeline stage details for problem cases.

Shows candidate mask density, per-arm mask stats at each stage,
and evidence cube warnings.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from km_estimator.models import PipelineState
from km_estimator.models.plot_metadata import AxisConfig, CurveInfo, PlotMetadata
from km_estimator.models.ipd_output import ProcessingStage
from km_estimator.nodes.digitization_5.axis_map import build_plot_model
from km_estimator.nodes.digitization_5.legend_color import build_color_models
from km_estimator.nodes.digitization_5.probability_map import build_evidence_cube
from km_estimator.utils import cv_utils

FIXTURE_DIR = ROOT / "tests" / "fixtures" / "standard"
CASES = ["case_020", "case_060"]
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


def _mask_column_coverage(mask: np.ndarray) -> float:
    if mask.ndim != 2 or mask.shape[1] == 0:
        return 0.0
    return float(np.mean(np.any(mask, axis=0)))


def main():
    for case_name in CASES:
        case_dir = FIXTURE_DIR / case_name
        meta = json.loads((case_dir / "metadata.json").read_text())
        image_path = str(case_dir / "graph.png")
        plot_meta = _build_metadata(meta)

        image = cv_utils.load_image(image_path, stage=ProcessingStage.DIGITIZE)
        plot_model = build_plot_model(image, plot_meta, None)
        color_models, color_warnings = build_color_models(image, plot_meta, plot_model)

        print(f"\n{'='*70}")
        print(f"{case_name} ({len(meta['groups'])} arms)")
        print(f"{'='*70}")

        # Color model info
        print("\n  Color models:")
        for name, m in color_models.items():
            ref = m.reference_lab()
            ref_str = f"({ref[0]:.1f},{ref[1]:.1f},{ref[2]:.1f})" if ref else "None"
            print(f"    {name}: reliability={m.reliability:.2f} source={m.source} ref_lab={ref_str}")
        if color_warnings:
            print(f"  Color warnings: {color_warnings}")

        # Build evidence cube
        evidence = build_evidence_cube(image, plot_model, color_models)

        # Global candidate mask stats
        cand = evidence.candidate_mask
        cand_density = float(np.mean(cand))
        cand_pixels = int(np.count_nonzero(cand))
        cand_cov = _mask_column_coverage(cand)
        print(f"\n  Global candidate mask:")
        print(f"    pixels={cand_pixels}  density={cand_density:.6f}  col_coverage={cand_cov:.3f}")

        # Ridge stats on the ROI
        x0, y0, x1, y1 = plot_model.plot_region
        roi = image[y0:y1, x0:x1]
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        roi_hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        sat = roi_hsv[:, :, 1]

        # Show ridge response stats for saturated pixels (curve pixels)
        sat_mask = sat >= 56
        ridge = evidence.ridge_map
        print(f"\n  Ridge stats:")
        print(f"    Overall: mean={float(np.mean(ridge)):.4f} p50={float(np.median(ridge)):.4f} p90={float(np.percentile(ridge, 90)):.4f}")
        if np.any(sat_mask):
            ridge_sat_vals = ridge[sat_mask]
            print(f"    Saturated pixels (sat>=56): n={int(np.count_nonzero(sat_mask))}"
                  f"  ridge_mean={float(np.mean(ridge_sat_vals)):.4f}"
                  f"  ridge_p50={float(np.median(ridge_sat_vals)):.4f}"
                  f"  ridge_p75={float(np.percentile(ridge_sat_vals, 75)):.4f}"
                  f"  ridge>=0.24={int(np.count_nonzero(ridge_sat_vals >= 0.24))}")
        # Saturation-channel ridge response
        from km_estimator.nodes.digitization_5.probability_map import _ridge_response
        sat_uint8 = np.clip(roi_hsv[:, :, 1], 0, 255).astype(np.uint8)
        ridge_sat_map = _ridge_response(sat_uint8)
        print(f"    Sat-ridge overall: mean={float(np.mean(ridge_sat_map)):.4f}"
              f"  p50={float(np.median(ridge_sat_map)):.4f}"
              f"  p90={float(np.percentile(ridge_sat_map, 90)):.4f}"
              f"  max={float(np.max(ridge_sat_map)):.4f}")
        if np.any(sat_mask):
            ridge_sat_at_sat = ridge_sat_map[sat_mask]
            print(f"    Sat-ridge at saturated pixels: n={int(np.count_nonzero(sat_mask))}"
                  f"  mean={float(np.mean(ridge_sat_at_sat)):.4f}"
                  f"  p50={float(np.median(ridge_sat_at_sat)):.4f}"
                  f"  p75={float(np.percentile(ridge_sat_at_sat, 75)):.4f}"
                  f"  >=0.24={int(np.count_nonzero(ridge_sat_at_sat >= 0.24))}")
            # How many would pass the boost criteria
            axis_pen_map = evidence.axis_penalty_map
            text_pen_map = evidence.text_penalty_map
            frame_pen_map = cv2.GaussianBlur(
                np.zeros_like(gray, dtype=np.float32), (5, 5), 0)  # placeholder
            boost_count = int(np.count_nonzero(
                (ridge_sat_map > 0.24)
                & (sat.astype(np.float32) >= 56.0)
                & (axis_pen_map < 0.25)
                & (text_pen_map < 0.65)
            ))
            print(f"    Sat-ridge boost potential (ridge_sat>0.24 & sat>=56 & axis<0.25 & text<0.65): {boost_count}")

        # Per-arm candidate mask stats
        print(f"\n  Per-arm candidate masks:")
        for arm_name in sorted(evidence.arm_candidate_masks):
            mask = evidence.arm_candidate_masks[arm_name]
            pixels = int(np.count_nonzero(mask))
            density = float(np.mean(mask))
            col_cov = _mask_column_coverage(mask)
            print(f"    {arm_name}: pixels={pixels}  density={density:.6f}  col_coverage={col_cov:.3f}")

        # Evidence cube warnings (filter for informative ones)
        print(f"\n  Evidence cube warnings:")
        for w in evidence.warning_codes:
            if w.startswith(("W_", "I_RIDGE", "I_HSV", "I_COLOR", "I_CURVE",
                            "I_PRIMARY", "I_ARM_EXCLUSIVE", "W_ARM",
                            "I_COLOR_RMSE")):
                print(f"    {w}")


if __name__ == "__main__":
    main()
