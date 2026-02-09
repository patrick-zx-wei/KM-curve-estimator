"""Diagnostic: analyze pixel color distributions in case_020."""
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
from km_estimator.nodes.digitization_5.legend_color import build_color_models, _rgb01_to_lab
from km_estimator.nodes.digitization_5 import cv_utils
from km_estimator.nodes.digitization_5.probability_map import _color_likelihood, COLOR_GOOD_DISTANCE

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
    case_name = "case_020"
    case_dir = FIXTURE_DIR / case_name
    meta = json.loads((case_dir / "metadata.json").read_text())
    image_path = str(case_dir / "graph.png")
    plot_meta = _build_metadata(meta)

    image = cv_utils.load_image(image_path, stage=ProcessingStage.DIGITIZE)
    plot_model = build_plot_model(image, plot_meta, None)
    color_models, _ = build_color_models(image, plot_meta, plot_model)

    x0, y0, x1, y1 = plot_model.plot_region
    roi = image[y0:y1, x0:x1]

    roi_lab = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB).astype(np.float32)
    roi_hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    sat = roi_hsv[:, :, 1].astype(np.float32)
    hue = roi_hsv[:, :, 0].astype(np.float32)

    treatment_model = color_models["Treatment"]
    ref_lab = treatment_model.reference_lab()
    print(f"Treatment reference LAB: {ref_lab}")
    print(f"Treatment reliability: {treatment_model.reliability}")

    # Compute color_like for the whole ROI
    ref = np.asarray(ref_lab, dtype=np.float32)
    dist = np.linalg.norm(roi_lab - ref[None, None, :], axis=2).astype(np.float32)
    good_dist = COLOR_GOOD_DISTANCE
    color_like = np.clip((good_dist - dist) / good_dist, 0.0, 1.0)

    # Analyze pixels by color_like threshold
    for thr in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]:
        mask = color_like >= thr
        n = int(np.sum(mask))
        if n > 0:
            s_vals = sat[mask]
            h_vals = hue[mask]
            print(f"\n  color_like >= {thr}: {n} pixels")
            print(f"    sat: mean={s_vals.mean():.1f} p10={np.percentile(s_vals,10):.0f} p50={np.percentile(s_vals,50):.0f} p90={np.percentile(s_vals,90):.0f}")
            print(f"    hue: mean={h_vals.mean():.1f} p10={np.percentile(h_vals,10):.0f} p50={np.percentile(h_vals,50):.0f} p90={np.percentile(h_vals,90):.0f}")
            print(f"    sat >= 56: {int(np.sum(s_vals >= 56))} pixels ({100*np.sum(s_vals>=56)/n:.1f}%)")
            print(f"    sat >= 30: {int(np.sum(s_vals >= 30))} pixels ({100*np.sum(s_vals>=30)/n:.1f}%)")
        else:
            print(f"\n  color_like >= {thr}: 0 pixels")

    # Also check: where are the high-saturation orange pixels?
    orange_hue_mask = np.abs(hue - 14.0) <= 12.0
    print(f"\n  Orange hue (14±12): {int(np.sum(orange_hue_mask))} pixels")
    for sat_thr in [20, 30, 40, 50, 56, 70]:
        combined = orange_hue_mask & (sat >= sat_thr)
        n = int(np.sum(combined))
        cov = float(np.mean(np.any(combined, axis=0)))
        print(f"    + sat>={sat_thr}: {n} pixels, col_coverage={cov:.3f}")

    # Check the axis/frame penalty at orange pixel locations
    from km_estimator.nodes.digitization_5.probability_map import _detect_axis_mask
    axis_mask = plot_model.axis_mask
    tick_mask = plot_model.tick_mask
    print(f"\n  Axis mask density in ROI: {float(np.mean(axis_mask)):.3f}")
    print(f"  Tick mask density in ROI: {float(np.mean(tick_mask)):.3f}")

    # Ridge detection
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    print(f"\n  Orange pixel grayscale intensity:")
    orange_pixels = orange_hue_mask & (sat >= 40)
    if np.sum(orange_pixels) > 0:
        g_vals = gray[orange_pixels]
        print(f"    mean={g_vals.mean():.3f} min={g_vals.min():.3f} max={g_vals.max():.3f}")

    blue_pixels = (np.abs(hue - 102.0) <= 12.0) & (sat >= 40)
    if np.sum(blue_pixels) > 0:
        g_vals = gray[blue_pixels]
        print(f"  Blue pixel grayscale intensity:")
        print(f"    mean={g_vals.mean():.3f} min={g_vals.min():.3f} max={g_vals.max():.3f}")

    # Background intensity
    bg_mask = sat < 15
    if np.sum(bg_mask) > 0:
        bg_vals = gray[bg_mask]
        print(f"  Background (sat<15) grayscale intensity: mean={bg_vals.mean():.3f}")


if __name__ == "__main__":
    main()
