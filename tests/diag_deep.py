"""Deep diagnostic: trace every pipeline stage for case_060.

Shows exactly where candidates are lost and what penalties block them.
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
from km_estimator.models.ipd_output import ProcessingStage
from km_estimator.nodes.digitization_5.axis_map import build_plot_model
from km_estimator.nodes.digitization_5.legend_color import build_color_models
from km_estimator.nodes.digitization_5.probability_map import (
    _ridge_response, _edge_response, _text_penalty, _text_region_penalty,
    _frame_penalty, _straight_line_penalty, _horizontal_support, _normalize01,
    CANDIDATE_RIDGE_THRESH, CANDIDATE_TEXT_THRESH, CANDIDATE_TEXT_REGION_THRESH,
    CANDIDATE_LINE_THRESH, CANDIDATE_AXIS_THRESH, HORIZONTAL_SUPPORT_MIN,
    MIN_CANDIDATE_SATURATION,
)
from km_estimator.utils import cv_utils

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
            name=g, color_description=f"{line_style} {color_name}", line_style=line_style,
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
    for case_name in ["case_060", "case_100"]:
        case_dir = FIXTURE_DIR / case_name
        meta = json.loads((case_dir / "metadata.json").read_text())
        image_path = str(case_dir / "graph.png")
        plot_meta = _build_metadata(meta)

        image = cv_utils.load_image(image_path, stage=ProcessingStage.DIGITIZE)
        plot_model = build_plot_model(image, plot_meta, None)

        x0, y0, x1, y1 = plot_model.plot_region
        print(f"\n{'='*70}")
        print(f"{case_name}")
        print(f"{'='*70}")
        print(f"  Image shape: {image.shape}")
        print(f"  Plot region: ({x0},{y0})-({x1},{y1})  size=({x1-x0},{y1-y0})")
        print(f"  Direction: {plot_model.curve_direction}")

        roi = image[y0:y1, x0:x1]
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        roi_hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV).astype(np.float32)
        sat_chan = roi_hsv[:, :, 1]
        h, w = gray.shape
        total = h * w
        print(f"  ROI pixels: {total}")

        # Compute all penalty maps
        ridge = _ridge_response(gray)
        edge = _edge_response(gray)
        text_pen_raw = _text_penalty(gray)
        text_region_pen = _text_region_penalty(text_pen_raw)
        text_pen = np.maximum(text_pen_raw, 0.75 * text_region_pen).astype(np.float32)
        frame_pen, has_top_frame, has_right_frame = _frame_penalty(gray)
        axis_pen = cv2.bitwise_or(plot_model.axis_mask, plot_model.tick_mask)
        axis_pen_f = _normalize01(cv2.GaussianBlur(
            (axis_pen.astype(np.float32) / 255.0).astype(np.float32), (5, 5), 0))
        line_pen, line_count = _straight_line_penalty(gray, axis_mask=axis_pen)

        # Find curve pixels: saturated, non-background
        curve_pixels = sat_chan >= 56  # saturated = likely curve
        n_curve = int(np.count_nonzero(curve_pixels))
        print(f"\n  Saturated pixels (sat>=56): {n_curve} ({100*n_curve/total:.1f}%)")

        # What blocks curve pixels from being candidates?
        # Primary candidate criteria
        ridge_pass = ridge > CANDIDATE_RIDGE_THRESH  # 0.24
        axis_pass = axis_pen_f < CANDIDATE_AXIS_THRESH  # 0.25
        text_pass = text_pen < CANDIDATE_TEXT_THRESH  # 0.35
        text_region_pass = text_region_pen < CANDIDATE_TEXT_REGION_THRESH  # 0.42
        line_pass = line_pen < CANDIDATE_LINE_THRESH  # 0.55
        frame_pass = frame_pen < 0.40
        prelim = ridge > max(0.10, CANDIDATE_RIDGE_THRESH * 0.7)
        h_support = _horizontal_support(
            (prelim & (text_pen < 0.55) & (text_region_pen < 0.70) & (frame_pen < 0.65)).astype(np.bool_)
        )
        hsup_pass = h_support > HORIZONTAL_SUPPORT_MIN  # 0.14
        sat_pass = sat_chan >= MIN_CANDIDATE_SATURATION  # 18

        # Relaxed criteria
        ridge_relax = ridge > max(0.08, CANDIDATE_RIDGE_THRESH * 0.60)  # 0.144
        axis_relax = axis_pen_f < min(0.85, CANDIDATE_AXIS_THRESH + 0.25)  # 0.50
        text_relax = text_pen < min(0.90, CANDIDATE_TEXT_THRESH + 0.30)  # 0.65
        text_region_relax = text_region_pen < min(0.92, CANDIDATE_TEXT_REGION_THRESH + 0.30)
        line_relax = line_pen < min(0.92, CANDIDATE_LINE_THRESH + 0.30)
        frame_relax = frame_pen < 0.75

        # For curve pixels, show which filter blocks them most
        print(f"\n  Curve pixels passing each PRIMARY filter:")
        print(f"    ridge > {CANDIDATE_RIDGE_THRESH}: {int(np.count_nonzero(curve_pixels & ridge_pass))} ({100*np.mean(curve_pixels & ridge_pass)/np.mean(curve_pixels):.1f}%)")
        print(f"    axis_pen < {CANDIDATE_AXIS_THRESH}: {int(np.count_nonzero(curve_pixels & axis_pass))} ({100*np.mean(curve_pixels & axis_pass)/np.mean(curve_pixels):.1f}%)")
        print(f"    text_pen < {CANDIDATE_TEXT_THRESH}: {int(np.count_nonzero(curve_pixels & text_pass))} ({100*np.mean(curve_pixels & text_pass)/np.mean(curve_pixels):.1f}%)")
        print(f"    text_region < {CANDIDATE_TEXT_REGION_THRESH}: {int(np.count_nonzero(curve_pixels & text_region_pass))} ({100*np.mean(curve_pixels & text_region_pass)/np.mean(curve_pixels):.1f}%)")
        print(f"    line_pen < {CANDIDATE_LINE_THRESH}: {int(np.count_nonzero(curve_pixels & line_pass))} ({100*np.mean(curve_pixels & line_pass)/np.mean(curve_pixels):.1f}%)")
        print(f"    frame_pen < 0.40: {int(np.count_nonzero(curve_pixels & frame_pass))} ({100*np.mean(curve_pixels & frame_pass)/np.mean(curve_pixels):.1f}%)")
        print(f"    h_support > {HORIZONTAL_SUPPORT_MIN}: {int(np.count_nonzero(curve_pixels & hsup_pass))} ({100*np.mean(curve_pixels & hsup_pass)/np.mean(curve_pixels):.1f}%)")
        print(f"    sat >= {MIN_CANDIDATE_SATURATION}: {int(np.count_nonzero(curve_pixels & sat_pass))} ({100*np.mean(curve_pixels & sat_pass)/np.mean(curve_pixels):.1f}%)")

        # Combined primary
        primary_mask = ridge_pass & axis_pass & text_pass & text_region_pass & line_pass & frame_pass & hsup_pass & sat_pass
        print(f"    ALL PRIMARY: {int(np.count_nonzero(curve_pixels & primary_mask))} ({100*np.mean(curve_pixels & primary_mask)/np.mean(curve_pixels):.1f}%)")

        print(f"\n  Curve pixels passing each RELAXED filter:")
        print(f"    ridge > {max(0.08, CANDIDATE_RIDGE_THRESH*0.60):.3f}: {int(np.count_nonzero(curve_pixels & ridge_relax))} ({100*np.mean(curve_pixels & ridge_relax)/np.mean(curve_pixels):.1f}%)")
        print(f"    axis_pen < 0.50: {int(np.count_nonzero(curve_pixels & axis_relax))} ({100*np.mean(curve_pixels & axis_relax)/np.mean(curve_pixels):.1f}%)")
        print(f"    text_pen < 0.65: {int(np.count_nonzero(curve_pixels & text_relax))} ({100*np.mean(curve_pixels & text_relax)/np.mean(curve_pixels):.1f}%)")
        relaxed_mask = ridge_relax & axis_relax & text_relax & text_region_relax & line_relax & frame_relax
        print(f"    ALL RELAXED: {int(np.count_nonzero(curve_pixels & relaxed_mask))} ({100*np.mean(curve_pixels & relaxed_mask)/np.mean(curve_pixels):.1f}%)")

        # Cumulative filter analysis: which filter eliminates the most curve pixels?
        print(f"\n  Cumulative filter elimination (on curve pixels only):")
        cp = curve_pixels.copy()
        filters = [
            ("ridge_relax", ridge_relax),
            ("axis_relax", axis_relax),
            ("text_relax", text_relax),
            ("text_region_relax", text_region_relax),
            ("line_relax", line_relax),
            ("frame_relax", frame_relax),
        ]
        remaining = int(np.count_nonzero(cp))
        for fname, fmask in filters:
            cp = cp & fmask
            new_remaining = int(np.count_nonzero(cp))
            lost = remaining - new_remaining
            print(f"    After {fname}: {new_remaining} (lost {lost}, {100*lost/max(1,remaining):.1f}%)")
            remaining = new_remaining

        # Penalty map statistics at curve pixel locations
        print(f"\n  Penalty map stats AT curve pixels (sat>=56):")
        for name, pmap in [("ridge", ridge), ("axis_pen", axis_pen_f), ("text_pen", text_pen),
                           ("text_region", text_region_pen), ("line_pen", line_pen),
                           ("frame_pen", frame_pen), ("h_support", h_support)]:
            vals = pmap[curve_pixels]
            print(f"    {name}: mean={float(np.mean(vals)):.3f} p25={float(np.percentile(vals,25)):.3f} "
                  f"p50={float(np.median(vals)):.3f} p75={float(np.percentile(vals,75)):.3f} "
                  f"p90={float(np.percentile(vals,90)):.3f}")

        # Save visual debug: candidate mask overlay
        debug_dir = ROOT / "tests" / "debug_output"
        debug_dir.mkdir(exist_ok=True)

        # Create overlay showing candidate mask vs curve pixels
        overlay = roi.copy()
        # Red = curve pixels not in candidate mask
        # Green = curve pixels that are candidates
        cand_primary = primary_mask
        cand_relaxed = relaxed_mask
        curve_not_cand = curve_pixels & ~cand_relaxed
        curve_and_cand = curve_pixels & cand_relaxed
        overlay[curve_not_cand] = [0, 0, 200]  # red = lost curve pixels
        overlay[curve_and_cand] = [0, 200, 0]  # green = kept curve pixels
        cv2.imwrite(str(debug_dir / f"{case_name}_candidates.png"), overlay)

        # Also save penalty maps as heatmaps
        for name, pmap in [("ridge", ridge), ("text_pen", text_pen),
                           ("line_pen", line_pen), ("axis_pen", axis_pen_f)]:
            heatmap = (np.clip(pmap, 0, 1) * 255).astype(np.uint8)
            heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
            cv2.imwrite(str(debug_dir / f"{case_name}_{name}.png"), heatmap_color)

        print(f"\n  Debug images saved to {debug_dir}/")


if __name__ == "__main__":
    main()
