"""Detailed case_080 diagnostic - check arm masks and trace behavior."""
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

FIXTURE = ROOT / "tests" / "fixtures" / "standard" / "case_080"
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

image = cv2.imread(str(FIXTURE / "graph.png"))
plot_model = build_plot_model(image, plot_meta, ocr_tokens=None)

# Build evidence cube
color_models, _ = build_color_models(image, plot_meta, plot_model)
ev = build_evidence_cube(image, plot_model, color_models)

x0, y0, x1, y1 = plot_model.plot_region
roi = image[y0:y1, x0:x1]
H, W = roi.shape[:2]

print(f"Plot ROI: {W}w x {H}h")
print(f"Y-axis range: {meta['y_axis']['start']} to {meta['y_axis']['end']}")

# Check arm mask coverage at key time points
for arm_name in ["Control", "Treatment"]:
    mask = ev.arm_candidate_masks[arm_name]
    score = ev.arm_score_maps[arm_name]
    density = float(np.mean(mask))
    cov = float(np.mean(np.any(mask, axis=0)))
    print(f"\n=== {arm_name} mask ===")
    print(f"  density={density:.5f}, col_coverage={cov:.3f}")

    # Check at specific time columns
    for t_val in [6, 12, 18, 24, 30, 36]:
        col = int(round(W * t_val / 36.0))
        col = min(col, W - 1)
        col_mask = mask[:, col]
        col_score = score[:, col]
        mask_rows = np.where(col_mask)[0]
        mask_count = len(mask_rows)
        if mask_count > 0:
            y_real_min = 1.0 - mask_rows[-1] / H  # lowest mask pixel (in survival space)
            y_real_max = 1.0 - mask_rows[0] / H   # highest mask pixel
            # y_real is in [0,1] of the plot range, need to convert to survival
            surv_min = meta['y_axis']['start'] + y_real_min * (meta['y_axis']['end'] - meta['y_axis']['start'])
            surv_max = meta['y_axis']['start'] + y_real_max * (meta['y_axis']['end'] - meta['y_axis']['start'])
            best_row = mask_rows[np.argmax(col_score[mask_rows])]
            best_score = float(col_score[best_row])
            surv_best = meta['y_axis']['start'] + (1.0 - best_row / H) * (meta['y_axis']['end'] - meta['y_axis']['start'])
            print(f"  t={t_val}: mask_px={mask_count}, surv_range=[{surv_min:.3f}, {surv_max:.3f}], best_score={best_score:.3f} at surv={surv_best:.3f}")
        else:
            best_row = int(np.argmax(col_score))
            best_score = float(col_score[best_row])
            surv_best = meta['y_axis']['start'] + (1.0 - best_row / H) * (meta['y_axis']['end'] - meta['y_axis']['start'])
            print(f"  t={t_val}: NO MASK, best_score={best_score:.3f} at surv={surv_best:.3f}")

# Check relevant warnings
print("\n=== Key Warnings ===")
for w in ev.warning_codes:
    if any(k in w for k in ["SPARSE", "PRUNE", "FRAME", "SWAP", "CROSS", "Control", "Treatment", "STRICT", "HARD_LOCK"]):
        print(f"  {w}")
