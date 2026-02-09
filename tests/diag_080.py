"""Diagnose case_080 Control arm MAE."""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from km_estimator.models import PipelineState
from km_estimator.models.plot_metadata import AxisConfig, CurveInfo, PlotMetadata
from km_estimator.nodes.digitization_5 import digitize_v5

FIXTURE = ROOT / "tests" / "fixtures" / "standard" / "case_080"
meta = json.loads((FIXTURE / "metadata.json").read_text())
hard_points = json.loads((FIXTURE / "hard_points.json").read_text())

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

state = PipelineState(
    image_path=str(FIXTURE / "graph.png"),
    preprocessed_image_path=str(FIXTURE / "graph.png"),
    plot_metadata=plot_meta,
)
result = digitize_v5(state)

if result.digitized_curves:
    for arm_name, hp_data in hard_points.items():
        print(f"\n=== {arm_name} ===")
        dig_pts = result.digitized_curves.get(arm_name, [])
        times = np.array([p[0] for p in dig_pts])
        values = np.array([p[1] for p in dig_pts])
        print(f"  Digitized range: t=[{times[0]:.2f}, {times[-1]:.2f}], s=[{values.min():.4f}, {values.max():.4f}]")
        print(f"  Points: {len(dig_pts)}")
        for lm in hp_data.get("landmarks", []):
            t = lm["time"]
            expected = lm["survival"]
            idx = np.searchsorted(times, t, side="right") - 1
            idx = max(0, min(idx, len(values) - 1))
            predicted = values[idx]
            error = abs(predicted - expected)
            print(f"  t={t:5.1f}: expected={expected:.4f}, predicted={predicted:.4f}, error={error:.4f}")
else:
    print("No digitized curves!")
    if result.errors:
        for e in result.errors:
            print(f"  Error: {e.message}")
