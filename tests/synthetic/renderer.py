"""Matplotlib rendering of synthetic KM curves.

Produces graph_draft.png (clean) and graph.png (after post-render modifiers).
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import cv2
import numpy as np

from .data_gen import SyntheticTestCase
from .modifiers import (
    Annotations,
    BackgroundStyle,
    CensoringMarks,
    CompressedTimeAxis,
    CurveDirection,
    FontTypography,
    FrameLayout,
    GaussianBlur,
    GridLines,
    JPEGArtifacts,
    LegendPlacement,
    LowResolution,
    Modifier,
    ModifierStage,
    NoisyBackground,
    RiskTableDisplay,
    ThickLines,
    ThinLines,
    TruncatedYAxis,
)


def _get_linewidth(modifiers: list[Modifier]) -> float:
    """Get line width from modifiers, defaulting to 2.6."""
    for m in modifiers:
        if isinstance(m, ThickLines):
            return m.linewidth
        if isinstance(m, ThinLines):
            return m.linewidth
    return 2.6


def _get_survival_at(
    coords: list[tuple[float, float]], t: float
) -> float:
    """Step-function interpolation of survival at time t."""
    if not coords:
        return 1.0
    if t < coords[0][0]:
        return 1.0
    for i in range(len(coords) - 1, -1, -1):
        if coords[i][0] <= t:
            return coords[i][1]
    return coords[0][1]


def _extract_style_directives(modifiers: list[Modifier]) -> tuple[str, str, str]:
    """Extract (background_style, direction, frame_layout) directives."""
    background = "white"
    direction = "downward"
    frame_layout = "full_box"
    for mod in modifiers:
        if isinstance(mod, BackgroundStyle):
            background = mod.style
        elif isinstance(mod, CurveDirection):
            direction = mod.direction
        elif isinstance(mod, FrameLayout):
            frame_layout = mod.layout
    return background, direction, frame_layout


def _apply_font_directive(matplotlib, modifiers: list[Modifier]) -> None:
    """Apply font family style to matplotlib rcParams for this render."""
    font_family = "sans"
    for mod in modifiers:
        if isinstance(mod, FontTypography):
            font_family = mod.family
            break

    if font_family == "serif":
        matplotlib.rcParams["font.family"] = "serif"
        matplotlib.rcParams["font.serif"] = ["Times New Roman", "Times", "DejaVu Serif"]
    else:
        matplotlib.rcParams["font.family"] = "sans-serif"
        matplotlib.rcParams["font.sans-serif"] = ["Arial", "Helvetica", "DejaVu Sans"]


def _apply_background_style(
    ax,
    ax_table,
    style: str,
    has_explicit_grid: bool,
) -> None:
    """Apply panel/canvas style presets."""
    if style == "sas_gray":
        ax.figure.patch.set_facecolor("white")
        ax.set_facecolor("#E5E5E5")
        if not has_explicit_grid:
            ax.grid(True, which="major", color="white", linewidth=1.0, alpha=1.0)
    elif style == "ggplot_gray":
        ax.figure.patch.set_facecolor("white")
        ax.set_facecolor("#EBEBEB")
        if not has_explicit_grid:
            ax.grid(True, which="major", color="white", linewidth=1.2, alpha=1.0)
    else:
        ax.figure.patch.set_facecolor("white")
        ax.set_facecolor("white")

    if ax_table is not None:
        ax_table.set_facecolor("white")


def _apply_frame_layout(ax, layout: str) -> None:
    """Apply L-axis vs full-box spine visibility."""
    if layout == "l_axis":
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_visible(True)
        ax.spines["bottom"].set_visible(True)
        ax.tick_params(top=False, right=False)
    else:
        for side in ("top", "right", "left", "bottom"):
            ax.spines[side].set_visible(True)
        ax.tick_params(top=False, right=False)


def _clip_int(v: float, lo: int, hi: int) -> int:
    return int(np.clip(int(round(v)), lo, hi))


def _build_oracle_axes_payload(
    fig,
    ax,
    test_case: SyntheticTestCase,
    dpi: int,
) -> dict:
    """
    Build exact pixel anchors for the saved draft image (bbox_inches='tight').

    Coordinates are stored in graph_draft.png pixel space (top-left origin).
    """
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    tight_bbox_in = fig.get_tightbbox(renderer)

    import matplotlib

    pad_inches = float(matplotlib.rcParams.get("savefig.pad_inches", 0.1))
    out_w = int(round((float(tight_bbox_in.width) + 2.0 * pad_inches) * dpi))
    out_h = int(round((float(tight_bbox_in.height) + 2.0 * pad_inches) * dpi))

    x_offset_px = (float(tight_bbox_in.x0) - pad_inches) * dpi
    y_offset_px = (float(tight_bbox_in.y0) - pad_inches) * dpi

    def disp_to_saved(x_disp: float, y_disp: float) -> tuple[int, int]:
        x_px = x_disp - x_offset_px
        y_from_bottom = y_disp - y_offset_px
        y_px = out_h - y_from_bottom
        x_i = _clip_int(x_px, 0, max(0, out_w - 1))
        y_i = _clip_int(y_px, 0, max(0, out_h - 1))
        return x_i, y_i

    x_start = float(test_case.x_axis.start)
    x_end = float(test_case.x_axis.end)
    y_start = float(test_case.y_axis.start)
    y_end = float(test_case.y_axis.end)

    left_bottom = ax.transData.transform((x_start, y_start))
    right_bottom = ax.transData.transform((x_end, y_start))
    left_top = ax.transData.transform((x_start, y_end))
    x0, y1 = disp_to_saved(float(left_bottom[0]), float(left_bottom[1]))
    x1, _ = disp_to_saved(float(right_bottom[0]), float(right_bottom[1]))
    _, y0 = disp_to_saved(float(left_top[0]), float(left_top[1]))

    plot_region = [
        int(min(x0, x1)),
        int(min(y0, y1)),
        int(max(x0, x1)),
        int(max(y0, y1)),
    ]

    x_tick_anchors: list[dict] = []
    for xv in test_case.x_axis.tick_values:
        xp, yp = ax.transData.transform((float(xv), y_start))
        px, _ = disp_to_saved(float(xp), float(yp))
        x_tick_anchors.append({"px": int(px), "value": float(xv)})

    y_tick_anchors: list[dict] = []
    for yv in test_case.y_axis.tick_values:
        xp, yp = ax.transData.transform((x_start, float(yv)))
        _, py = disp_to_saved(float(xp), float(yp))
        y_tick_anchors.append({"py": int(py), "value": float(yv)})

    return {
        "version": 1,
        "source_image": "graph.png",
        "image_width": int(out_w),
        "image_height": int(out_h),
        "plot_region": plot_region,
        "x_tick_anchors": x_tick_anchors,
        "y_tick_anchors": y_tick_anchors,
        "x_axis": {
            "start": x_start,
            "end": x_end,
            "scale": test_case.x_axis.scale,
        },
        "y_axis": {
            "start": y_start,
            "end": y_end,
            "scale": test_case.y_axis.scale,
        },
    }


def _transform_oracle_for_post_modifiers(payload: dict, modifiers: list[Modifier]) -> dict:
    """Apply geometric post-render transforms to oracle anchors."""
    out = json.loads(json.dumps(payload))
    width = int(out.get("image_width", 0))
    height = int(out.get("image_height", 0))
    if width <= 0 or height <= 0:
        return out

    for mod in modifiers:
        if isinstance(mod, LowResolution):
            scale = float(mod.target_width) / float(max(1, width))
            new_w = int(mod.target_width)
            new_h = int(height * scale)

            def sx(v: int) -> int:
                return int(round(float(v) * scale))

            pr = out.get("plot_region", [0, 0, width - 1, height - 1])
            if isinstance(pr, list) and len(pr) == 4:
                out["plot_region"] = [
                    _clip_int(sx(int(pr[0])), 0, max(0, new_w - 1)),
                    _clip_int(sx(int(pr[1])), 0, max(0, new_h - 1)),
                    _clip_int(sx(int(pr[2])), 0, max(0, new_w - 1)),
                    _clip_int(sx(int(pr[3])), 0, max(0, new_h - 1)),
                ]

            xt = out.get("x_tick_anchors", [])
            if isinstance(xt, list):
                for item in xt:
                    if isinstance(item, dict) and "px" in item:
                        item["px"] = _clip_int(sx(int(item["px"])), 0, max(0, new_w - 1))
            yt = out.get("y_tick_anchors", [])
            if isinstance(yt, list):
                for item in yt:
                    if isinstance(item, dict) and "py" in item:
                        item["py"] = _clip_int(sx(int(item["py"])), 0, max(0, new_h - 1))

            width, height = new_w, new_h
            out["image_width"] = int(width)
            out["image_height"] = int(height)

    return out


def render_test_case(
    test_case: SyntheticTestCase,
    output_dir: Path,
    dpi: int = 150,
    figsize: tuple[float, float] = (10, 7),
) -> tuple[Path, Path]:
    """Render a test case to graph_draft.png and graph.png.

    Returns (draft_path, final_path).
    """
    import matplotlib
    matplotlib.use("Agg")
    _apply_font_directive(matplotlib, test_case.modifiers)
    import matplotlib.pyplot as plt

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Check for figure-stage modifiers
    figure_mods = [m for m in test_case.modifiers if m.stage == ModifierStage.FIGURE]
    post_mods = [m for m in test_case.modifiers if m.stage == ModifierStage.POST_RENDER]

    # Check if risk table display is requested.
    has_risk_table = any(isinstance(m, RiskTableDisplay) for m in figure_mods)

    if has_risk_table and test_case.risk_table:
        fig, (ax, ax_table) = plt.subplots(
            2, 1,
            figsize=figsize,
            gridspec_kw={"height_ratios": [4, 1]},
        )
    else:
        fig, ax = plt.subplots(figsize=figsize)
        ax_table = None

    linewidth = _get_linewidth(test_case.modifiers)
    background_style, curve_direction, frame_layout = _extract_style_directives(figure_mods)
    has_explicit_grid = any(isinstance(m, GridLines) for m in figure_mods)
    _apply_background_style(ax, ax_table, background_style, has_explicit_grid)

    # Plot curves
    for curve in test_case.curves:
        times = [c[0] for c in curve.step_coords]
        survivals = [c[1] for c in curve.step_coords]
        if curve_direction == "upward":
            survivals = [1.0 - float(s) for s in survivals]

        ax.step(
            times,
            survivals,
            where="post",
            label=curve.group_name,
            color=curve.color,
            linewidth=linewidth,
            linestyle=curve.line_style,
        )

    # Apply figure-stage modifiers
    for mod in figure_mods:
        if isinstance(mod, TruncatedYAxis):
            test_case.y_axis = test_case.y_axis.model_copy(
                update={"start": mod.y_start}
            )
            # Update tick values
            y_ticks = [
                round(v, 1)
                for v in np.arange(mod.y_start, 1.01, 0.2)
            ]
            if 1.0 not in y_ticks:
                y_ticks.append(1.0)
            test_case.y_axis = test_case.y_axis.model_copy(
                update={"tick_values": y_ticks}
            )

        elif isinstance(mod, GridLines):
            ax.grid(True, which="major", alpha=mod.alpha, linestyle="--")
            if mod.minor:
                ax.minorticks_on()
                ax.grid(True, which="minor", alpha=mod.alpha * 0.5, linestyle=":")

        elif isinstance(mod, CensoringMarks):
            for curve in test_case.curves:
                if curve.censoring_times:
                    censor_survivals = [
                        _get_survival_at(curve.step_coords, t)
                        for t in curve.censoring_times
                    ]
                    if curve_direction == "upward":
                        censor_survivals = [1.0 - float(s) for s in censor_survivals]
                    ax.plot(
                        curve.censoring_times,
                        censor_survivals,
                        "|",
                        color=curve.color,
                        markersize=mod.marker_size,
                        markeredgewidth=1.5,
                    )

        elif isinstance(mod, LegendPlacement):
            pass  # handled below when setting legend

        elif isinstance(mod, Annotations):
            for i, text in enumerate(mod.texts):
                ax.annotate(
                    text,
                    xy=(0.6, 0.85 - i * 0.06),
                    xycoords="axes fraction",
                    fontsize=9,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="wheat", alpha=0.5),
                )

        elif isinstance(mod, CompressedTimeAxis):
            x_ticks = list(
                np.linspace(
                    test_case.x_axis.start, test_case.x_axis.end, mod.n_ticks
                )
            )
            test_case.x_axis = test_case.x_axis.model_copy(
                update={"tick_values": [round(t, 1) for t in x_ticks]}
            )

    # Axis configuration
    ax.set_xlim(test_case.x_axis.start, test_case.x_axis.end)
    ax.set_ylim(test_case.y_axis.start, test_case.y_axis.end)
    ax.set_xticks(test_case.x_axis.tick_values)
    ax.set_yticks(test_case.y_axis.tick_values)
    ax.set_xlabel(test_case.x_axis.label or "Time (months)")
    y_label = test_case.y_axis.label or "Survival Probability"
    if curve_direction == "upward" and "survival" in y_label.lower():
        y_label = "Cumulative Incidence"
        test_case.y_axis = test_case.y_axis.model_copy(update={"label": y_label})
    ax.set_ylabel(y_label)
    _apply_frame_layout(ax, frame_layout)

    if test_case.title:
        ax.set_title(test_case.title)

    # Legend
    legend_loc = "best"
    for mod in figure_mods:
        if isinstance(mod, LegendPlacement):
            legend_loc = mod.location
            break
    ax.legend(loc=legend_loc)

    # Risk table below plot
    if ax_table is not None and test_case.risk_table:
        rt = test_case.risk_table
        ax_table.axis("off")
        ax_table.set_xlim(ax.get_xlim())

        n_groups = len(rt.groups)
        row_height = 1.0 / (n_groups + 1.5)

        ax_table.text(
            -0.12, 1.0 - row_height * 0.5, "No. at Risk",
            transform=ax_table.transAxes,
            fontsize=9, fontweight="bold",
            verticalalignment="center", ha="left",
        )

        for i, group in enumerate(rt.groups):
            color = (
                test_case.curves[i].color
                if i < len(test_case.curves)
                else "black"
            )
            y_pos = 1.0 - row_height * (i + 1.5)

            # Group name (placed to the left of the plot area)
            ax_table.text(
                -0.12, y_pos,
                group.name,
                transform=ax_table.transAxes,
                fontsize=9, color=color, fontweight="bold",
                verticalalignment="center", ha="left",
            )

            # Counts at each time point
            for t, count in zip(rt.time_points, group.counts):
                ax_table.text(
                    t, y_pos,
                    str(count),
                    transform=ax_table.get_xaxis_transform(),
                    fontsize=8,
                    ha="center", va="center",
                )

    fig.tight_layout()
    oracle_payload = _build_oracle_axes_payload(fig, ax, test_case, dpi=dpi)

    # Save draft
    draft_path = output_dir / "graph_draft.png"
    fig.savefig(str(draft_path), dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    # Copy to graph.png, then apply post-render modifiers
    graph_path = output_dir / "graph.png"
    shutil.copy2(draft_path, graph_path)

    if post_mods:
        _apply_post_render_modifiers(graph_path, post_mods)
        oracle_payload = _transform_oracle_for_post_modifiers(oracle_payload, post_mods)

    # Persist exact calibration anchors for benchmark-oracle mode.
    oracle_path = output_dir / "oracle_axes.json"
    with open(oracle_path, "w") as f:
        json.dump(oracle_payload, f, indent=2)

    test_case.draft_image_path = str(draft_path)
    test_case.image_path = str(graph_path)

    return draft_path, graph_path


def _apply_post_render_modifiers(image_path: Path, modifiers: list[Modifier]) -> None:
    """Apply post-render modifiers to an image file in-place."""
    img = cv2.imread(str(image_path))
    if img is None:
        return

    for mod in modifiers:
        if isinstance(mod, LowResolution):
            h, w = img.shape[:2]
            scale = mod.target_width / w
            new_h = int(h * scale)
            img = cv2.resize(
                img, (mod.target_width, new_h), interpolation=cv2.INTER_AREA
            )

        elif isinstance(mod, JPEGArtifacts):
            encode_param: tuple[int, int] = (int(cv2.IMWRITE_JPEG_QUALITY), int(mod.quality))
            ok, encoded = cv2.imencode(".jpg", img, encode_param)
            if ok:
                decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
                if decoded is not None:
                    img = decoded

        elif isinstance(mod, NoisyBackground):
            rng = np.random.default_rng(42)
            noise = rng.normal(0, mod.sigma, img.shape).astype(np.float32)
            img = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)

        elif isinstance(mod, GaussianBlur):
            k = mod.kernel_size
            if k % 2 == 0:
                k += 1
            img = cv2.GaussianBlur(img, (k, k), 0)

    cv2.imwrite(str(image_path), img)
