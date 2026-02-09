"""Diagnose case_075 Treatment arm mask sparsity."""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from km_estimator.models import PipelineState
from km_estimator.models.plot_metadata import AxisConfig, CurveInfo, PlotMetadata
from km_estimator.nodes.digitization_5.axis_map import build_plot_model
from km_estimator.nodes.digitization_5.legend_color import build_color_models
from km_estimator.nodes.digitization_5.probability_map import build_evidence_cube

FIXTURE = ROOT / "tests" / "fixtures" / "standard" / "case_075"
meta = json.loads((FIXTURE / "metadata.json").read_text())

COLOR_NAMES = ["blue", "orange"]
LINE_STYLES = ["solid", "dashed"]
curves = []
for i, g in enumerate(meta["groups"]):
    curves.append(CurveInfo(
        name=g,
        color_description=f"{LINE_STYLES[i % 2]} {COLOR_NAMES[i % 5]}",
        line_style=LINE_STYLES[i % 2],
    ))

x = meta["x_axis"]
y = meta["y_axis"]
plot_meta = PlotMetadata(
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

image_path = str(FIXTURE / "graph.png")
image = cv2.imread(image_path)
plot_model = build_plot_model(image, plot_meta, ocr_tokens=None)
color_models, color_warnings = build_color_models(image, plot_meta, plot_model)

print("=== Color Warnings ===")
for w in color_warnings:
    print(f"  {w}")

for name, model in color_models.items():
    print(f"\n=== {name} ===")
    print(f"  expected_lab: {model.expected_lab}")
    print(f"  observed_lab: {model.observed_lab}")
    print(f"  reliability: {model.reliability}")
    print(f"  source: {model.source}")
    print(f"  warning_codes: {model.warning_codes}")
    print(f"  reference_lab: {model.reference_lab()}")

# Build evidence cube to see per-arm diagnostics
print("\n=== Building Evidence Cube ===")
ev = build_evidence_cube(image, plot_model, color_models)

print("\n=== Evidence Cube Warnings ===")
for w in ev.warning_codes:
    if "Treatment" in w or "RIDGE" in w or "FRAME" in w or "SPARSE" in w:
        print(f"  {w}")

# Check the arm candidate masks
for name in sorted(ev.arm_candidate_masks):
    mask = ev.arm_candidate_masks[name]
    density = float(np.mean(mask))
    cov = float(np.mean(np.any(mask, axis=0)))
    total_px = int(np.count_nonzero(mask))
    print(f"\n  {name}: density={density:.5f}, col_coverage={cov:.3f}, total_px={total_px}")

# Look at score distribution for Treatment in the right part of the image
x0, y0, x1, y1 = plot_model.plot_region
roi = image[y0:y1, x0:x1]
H, W = roi.shape[:2]
treatment_map = ev.arm_score_maps["Treatment"]
treatment_mask = ev.arm_candidate_masks["Treatment"]

# Check last 20% of columns
right_start = int(W * 0.80)
right_map = treatment_map[:, right_start:]
right_mask = treatment_mask[:, right_start:]
right_density = float(np.mean(right_mask))
right_max = float(np.max(right_map))
right_mean = float(np.mean(right_map[right_mask])) if np.any(right_mask) else 0
print(f"\n  Treatment right 20%: density={right_density:.5f}, max_score={right_max:.3f}, mean_score_in_mask={right_mean:.3f}")

# Check what the orange pixels look like in the right side
roi_hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV).astype(np.float32)
roi_lab = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB).astype(np.float32)
sat_chan = roi_hsv[:, :, 1]
hue_chan = roi_hsv[:, :, 0]

# Orange hue range: roughly 5-25 in OpenCV
orange_region = roi[:, right_start:]
orange_hsv = roi_hsv[:, right_start:]
orange_lab = roi_lab[:, right_start:]

# Find high-saturation pixels in orange hue range in the right region
sat_ok = orange_hsv[:, :, 1] >= 50
hue_ok = (orange_hsv[:, :, 0] >= 5) & (orange_hsv[:, :, 0] <= 25)
orange_px = sat_ok & hue_ok
orange_count = int(np.count_nonzero(orange_px))
print(f"\n  Orange-like pixels (hue 5-25, sat>=50) in right 20%: {orange_count}")

if orange_count > 0:
    ys, xs = np.where(orange_px)
    # Show some sample pixel LAB values
    sample_idx = np.linspace(0, len(ys)-1, min(5, len(ys)), dtype=int)
    ref_lab = color_models["Treatment"].reference_lab()
    ref_arr = np.asarray(ref_lab, dtype=np.float32) if ref_lab else None
    for i in sample_idx:
        lab_val = orange_lab[ys[i], xs[i]]
        dist = float(np.linalg.norm(lab_val - ref_arr)) if ref_arr is not None else -1
        print(f"    px ({ys[i]},{xs[i]+right_start}): LAB=({lab_val[0]:.1f},{lab_val[1]:.1f},{lab_val[2]:.1f}), dist_to_ref={dist:.1f}")

# Check the column where t=36 maps to
# t=36 is the end of x-axis (x_end=36)
print(f"\n  Plot ROI: {W}w x {H}h")
print(f"  Treatment expected at t=36: survival=0.5867, row={int(H*(1-0.5867))}")

# What's the score at the expected position?
expected_row = int(H * (1 - 0.5867))
expected_col = W - 1  # t=36 is at the right edge
print(f"  Score at expected (row={expected_row}, col={expected_col}): {treatment_map[expected_row, expected_col]:.4f}")
print(f"  In mask at expected: {treatment_mask[expected_row, expected_col]}")

# Find the max score in last few columns
for col_offset in [W-1, W-2, W-3, W-5, W-8, W-10, W-15, W-20, W-50]:
    if col_offset < 0:
        continue
    col_scores = treatment_map[:, col_offset]
    best_row = int(np.argmax(col_scores))
    best_score = float(col_scores[best_row])
    in_mask = bool(treatment_mask[best_row, col_offset])
    y_real = 1.0 - best_row / H
    mask_count = int(np.count_nonzero(treatment_mask[:, col_offset]))
    print(f"  Col {col_offset}: best_row={best_row}, y_real={y_real:.3f}, score={best_score:.4f}, in_mask={in_mask}, mask_px={mask_count}")

# Check saturation and color values at the right edge
print("\n=== Right edge pixel properties ===")
ref_lab = color_models["Treatment"].reference_lab()
ref_arr = np.asarray(ref_lab, dtype=np.float32)
for c in range(W-1, max(W-15, 0), -1):
    # Check the pixel at the expected Treatment position (~row 209)
    for r in range(205, 215):
        sat_val = sat_chan[r, c]
        hue_val = hue_chan[r, c]
        lab_val = roi_lab[r, c]
        dist = float(np.linalg.norm(lab_val - ref_arr))
        bgr_val = roi[r, c]
        if sat_val >= 30:
            print(f"  ({r},{c}): SAT={sat_val:.0f}, HUE={hue_val:.0f}, LAB=({lab_val[0]:.0f},{lab_val[1]:.0f},{lab_val[2]:.0f}), dist={dist:.1f}, BGR=({bgr_val[0]},{bgr_val[1]},{bgr_val[2]}), frame_pen_approx={'in_band' if c >= W-8 else 'out'}")
