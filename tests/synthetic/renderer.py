"""Matplotlib rendering of synthetic KM curves.

Produces graph_draft.png (clean) and graph.png (after post-render modifiers).
"""

from __future__ import annotations

import shutil
from pathlib import Path

import cv2
import numpy as np

from .data_gen import SyntheticTestCase
from .modifiers import (
    Annotations,
    CensoringMarks,
    CompressedTimeAxis,
    GaussianBlur,
    GridLines,
    JPEGArtifacts,
    LegendPlacement,
    LowResolution,
    Modifier,
    ModifierStage,
    NoisyBackground,
    ThickLines,
    ThinLines,
    TruncatedYAxis,
)


def _get_linewidth(modifiers: list[Modifier]) -> float:
    """Get line width from modifiers, defaulting to 2.0."""
    for m in modifiers:
        if isinstance(m, ThickLines):
            return m.linewidth
        if isinstance(m, ThinLines):
            return m.linewidth
    return 2.0


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
    import matplotlib.pyplot as plt

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Check for figure-stage modifiers
    figure_mods = [m for m in test_case.modifiers if m.stage == ModifierStage.FIGURE]
    post_mods = [m for m in test_case.modifiers if m.stage == ModifierStage.POST_RENDER]

    # Check if risk table display is requested
    has_risk_table = any(
        isinstance(m, type) and m.__class__.__name__ == "RiskTableDisplay"
        for m in figure_mods
    )
    # More direct check
    from .modifiers import RiskTableDisplay
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

    # Plot curves
    for curve in test_case.curves:
        times = [c[0] for c in curve.step_coords]
        survivals = [c[1] for c in curve.step_coords]

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
    ax.set_ylabel(test_case.y_axis.label or "Survival Probability")

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
            for j, (t, count) in enumerate(zip(rt.time_points, group.counts)):
                ax_table.text(
                    t, y_pos,
                    str(count),
                    transform=ax_table.get_xaxis_transform(),
                    fontsize=8,
                    ha="center", va="center",
                )

    fig.tight_layout()

    # Save draft
    draft_path = output_dir / "graph_draft.png"
    fig.savefig(str(draft_path), dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    # Copy to graph.png, then apply post-render modifiers
    graph_path = output_dir / "graph.png"
    shutil.copy2(draft_path, graph_path)

    if post_mods:
        _apply_post_render_modifiers(graph_path, post_mods)

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
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), mod.quality]
            _, encoded = cv2.imencode(".jpg", img, encode_param)
            img = cv2.imdecode(encoded, cv2.IMREAD_COLOR)

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
