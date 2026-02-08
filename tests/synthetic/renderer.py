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


def _simplify_step_coords(
    coords: list[tuple[float, float]], max_steps: int = 100
) -> list[tuple[float, float]]:
    """Merge closely-spaced steps for cleaner visual rendering.

    When a curve has hundreds of tiny steps (high patient counts), the
    rendered line looks jagged with vertical-line artifacts.  This merges
    steps closer than ``time_range / max_steps`` apart, keeping the first
    time in each group and the final survival value.
    """
    if len(coords) <= max_steps:
        return coords

    time_range = coords[-1][0] - coords[0][0]
    if time_range <= 0:
        return coords

    min_gap = time_range / max_steps

    result = [coords[0]]
    for i in range(1, len(coords)):
        if coords[i][0] - result[-1][0] >= min_gap:
            result.append(coords[i])
        else:
            # Keep first time, update to latest survival
            result[-1] = (result[-1][0], coords[i][1])

    # Always include the last point
    if result[-1] != coords[-1]:
        result.append(coords[-1])

    return result


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


def _uniform_ticks(start: float, end: float, step: float, ndigits: int = 2) -> list[float]:
    """Generate deterministic ticks with rounding-safe arithmetic."""
    if step <= 0:
        return [round(float(start), ndigits), round(float(end), ndigits)]
    lo = float(min(start, end))
    hi = float(max(start, end))
    n_steps = int(np.floor((hi - lo) / step + 1e-9))
    vals = [round(lo + i * step, ndigits) for i in range(n_steps + 1)]
    if not vals or abs(float(vals[-1]) - hi) > 1e-6:
        vals.append(round(hi, ndigits))
    return sorted(set(float(v) for v in vals))


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
    test_case.curve_direction = curve_direction

    # Simplify step coordinates for cleaner rendering
    simplified: dict[str, list[tuple[float, float]]] = {}
    for curve in test_case.curves:
        simplified[curve.group_name] = _simplify_step_coords(curve.step_coords)

    # Plot curves
    for curve in test_case.curves:
        coords = simplified[curve.group_name]
        times = [c[0] for c in coords]
        survivals = [c[1] for c in coords]
        if curve_direction == "upward":
            survivals = [1.0 - float(s) for s in survivals]

        step_kwargs: dict = dict(
            where="post",
            label=curve.group_name,
            color=curve.color,
            linewidth=linewidth,
        )
        if curve.line_style == "dashed":
            step_kwargs["linestyle"] = "--"
            step_kwargs["dashes"] = (5, 0.5)
        else:
            step_kwargs["linestyle"] = curve.line_style

        ax.step(times, survivals, **step_kwargs)

    # Apply figure-stage modifiers
    for mod in figure_mods:
        if isinstance(mod, TruncatedYAxis):
            test_case.y_axis = test_case.y_axis.model_copy(
                update={"start": mod.y_start}
            )
            # Update tick values
            y_ticks = _uniform_ticks(start=float(mod.y_start), end=1.0, step=0.2, ndigits=1)
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
                    simple = simplified.get(curve.group_name, curve.step_coords)
                    censor_survivals = [
                        _get_survival_at(simple, t)
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
        if curve_direction == "upward" and "survival" in test_case.title.lower():
            test_case.title = "Cumulative Incidence Curve"
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

            # Counts at each time point (skip groups with 0 at risk)
            for t, count in zip(rt.time_points, group.counts):
                if int(count) > 0:
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
