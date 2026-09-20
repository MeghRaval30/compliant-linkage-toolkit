"""Every figure the README and the paper use, from one command.

``cmtool figures --out docs/figures`` regenerates all of them. The point is that
when the coupons are measured and the first footage exists, nothing has to be
redrawn by hand: the same command produces the same figures with the measured
series filled in, and the manifest records exactly which inputs each one used.

The rule about missing data
---------------------------
A figure whose data does not exist yet is **not invented**. It is either

* drawn without that series, carrying a visible note naming what is missing, or
* skipped, with a reason recorded in the manifest and printed by the CLI.

So a README built today shows real predictions with the measured curve honestly
absent, and the same README built after A6 shows it present. Nothing in between
is ever a stand-in.

Every figure also carries the placeholder caveat when one applies, plus the code
commit and config hash in small print, so a figure lifted out of the repository
still says where it came from.

Built for a journal column
--------------------------
Every figure is laid out at its **final printed width**, one column of a
two-column journal by default, so nothing is legible only because it is being
looked at four times the size it will be published. That is what decides the
layouts: panels stack rather than sitting side by side, and the strain figure's
per-joint panels wrap into a grid. ``column="double"`` gives the full-width
version for a talk or a poster, with the panels beside each other and larger
type.

Three encodings carry identity, not one. Each series has its own hue, its own
dash pattern and its own marker; filled marks take a hatch, since a fill has no
dash to carry. So the figures survive a greyscale print, a bad projector and a
colour-blind reader. The hues themselves were picked to separate in greyscale as
well as in colour -- see :mod:`cmtool.viz.palette`.
"""

from __future__ import annotations

import json
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from cmtool.core.provenance import Provenance
from cmtool.viz.palette import (
    HATCH,
    INK,
    ORDINAL_RAMP_DARK,
    ORDINAL_RAMP_LIGHT,
    SERIES,
    STATUS,
)
from cmtool.viz.scene import ViewerScene

if TYPE_CHECKING:  # pragma: no cover - typing only
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

    from cmtool.convert.base import CompliantMechanism
    from cmtool.metrics.paths import MeasuredPath

#: Designs drawn when the caller names none.
DEFAULT_DESIGNS: tuple[str, ...] = (
    "examples/designs/fb_02_0052.json",
    "examples/designs/fb_02_0090.json",
    "examples/designs/fb_02_0203.json",
)

#: Flexure thicknesses swept in the design-rule figure, in mm.
DESIGN_RULE_THICKNESS_MM: tuple[float, ...] = (0.4, 0.6, 0.8, 1.0)

#: Dash patterns for the ordinal thickness curves, getting denser with thickness
#: so the order reads without colour. ``None`` is solid.
DESIGN_RULE_DASHES: tuple[list[float] | None, ...] = (
    [1.0, 2.0],
    [4.0, 2.0],
    [7.0, 2.0, 1.5, 2.0],
    None,
)

#: Width of one column in a two-column journal, in inches. Figures are laid out
#: at their final printed size, so a label that fits here fits on the page.
SINGLE_COLUMN_IN = 3.4

#: Width across both columns, for a talk, a poster or a full-width plate.
DOUBLE_COLUMN_IN = 7.0


@dataclass(frozen=True)
class FigureLayout:
    """Page geometry and type sizes for one figure width.

    Point sizes are absolute, and the figure is created at its final printed
    width, so what is set here is what a reader gets on paper. Shrinking a
    7-inch figure into a 3.4-inch column would take 9 pt type down to 4.4 pt;
    laying it out at 3.4 inches in the first place is the only way to know.
    """

    column: str
    width_in: float
    base_pt: float
    title_pt: float
    caption_pt: float
    footer_pt: float
    marker_pt: float
    #: Whether two-panel figures stack vertically rather than sitting side by side.
    stacked: bool

    def height(self, single: float, double: float) -> float:
        """Pick a height for this layout."""
        return single if self.column == "single" else double

    def wrap_chars(self, font_pt: float | None = None) -> int:
        """How many characters of caption fit across the figure.

        Matplotlib does no wrapping of its own, and ``savefig`` uses a tight
        bounding box, so a caption line wider than the axes silently widens the
        saved figure past the column it was laid out for. The width is therefore
        computed from the figure width and the type size rather than guessed:
        0.58 em is the mean advance of this sans at text sizes, and the margin
        keeps the longest line inside the plot area.
        """
        size = font_pt or self.caption_pt
        return max(40, int(self.width_in * 72.0 * 0.96 / (size * 0.58)))


#: The two supported widths.
LAYOUTS: dict[str, FigureLayout] = {
    "single": FigureLayout(
        column="single",
        width_in=SINGLE_COLUMN_IN,
        base_pt=7.0,
        title_pt=8.0,
        caption_pt=6.2,
        footer_pt=5.4,
        marker_pt=3.4,
        stacked=True,
    ),
    "double": FigureLayout(
        column="double",
        width_in=DOUBLE_COLUMN_IN,
        base_pt=9.0,
        title_pt=10.5,
        caption_pt=7.5,
        footer_pt=6.5,
        marker_pt=4.6,
        stacked=False,
    ),
}


def layout_for(column: str) -> FigureLayout:
    """Return the layout for ``column``, or explain what the choices are."""
    try:
        return LAYOUTS[column]
    except KeyError:
        raise ValueError(
            f"unknown column width {column!r}; choose from {sorted(LAYOUTS)}"
        ) from None


@dataclass(frozen=True)
class FigureResult:
    """What became of one figure.

    Attributes
    ----------
    produced
        Whether a file was written. ``False`` always comes with a ``reason``.
    missing
        Series that were left out because their data does not exist yet. Present
        on a produced figure too: the figure is real, one of its curves is not
        there, and the caption says so.
    """

    name: str
    produced: bool
    path: Path | None = None
    reason: str | None = None
    missing: list[str] = field(default_factory=list)
    inputs: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable summary."""
        return {
            "name": self.name,
            "produced": self.produced,
            "path": str(self.path) if self.path else None,
            "reason": self.reason,
            "missing": list(self.missing),
            "inputs": self.inputs,
        }


@dataclass
class FigureInputs:
    """Everything the figures draw from, resolved once.

    Building this is the expensive part -- one beam FEA sweep per design -- so it
    is done once and shared, rather than each figure re-solving.
    """

    scenes: dict[str, ViewerScene]
    mechanisms: dict[str, CompliantMechanism]
    provenance: Provenance
    measured: MeasuredPath | None = None
    torque_readings: list[Any] = field(default_factory=list)
    uncertainty: dict[str, Any] | None = None
    primary: str = ""

    @property
    def primary_scene(self) -> ViewerScene:
        """The design the single-design figures use."""
        return self.scenes[self.primary or next(iter(self.scenes))]

    @property
    def primary_mechanism(self) -> CompliantMechanism:
        """The converted mechanism behind :attr:`primary_scene`."""
        return self.mechanisms[self.primary or next(iter(self.mechanisms))]


# --------------------------------------------------------------- style ------


def _apply_style(theme: str, layout: FigureLayout) -> None:
    """Set the rcParams every figure shares: recessive chrome, one sans face."""
    import matplotlib as mpl

    ink = INK[theme]
    mpl.rcParams.update(
        {
            "figure.facecolor": ink["surface"],
            "axes.facecolor": ink["surface"],
            "savefig.facecolor": ink["surface"],
            "text.color": ink["primary"],
            "axes.labelcolor": ink["secondary"],
            "axes.edgecolor": ink["axis"],
            "xtick.color": ink["muted"],
            "ytick.color": ink["muted"],
            "xtick.labelcolor": ink["secondary"],
            "ytick.labelcolor": ink["secondary"],
            "grid.color": ink["grid"],
            "grid.linewidth": 0.8,
            "axes.grid": True,
            "axes.axisbelow": True,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
            "font.size": layout.base_pt,
            "axes.titlesize": layout.title_pt,
            "axes.titleweight": "semibold",
            "axes.titlelocation": "left",
            "axes.labelsize": layout.base_pt,
            "xtick.labelsize": layout.base_pt - 0.5,
            "ytick.labelsize": layout.base_pt - 0.5,
            "legend.fontsize": layout.base_pt - 0.5,
            "lines.markersize": layout.marker_pt,
            "hatch.linewidth": 0.6,
            "figure.dpi": 300,
            "savefig.bbox": "tight",
        }
    )


def _new_figure(layout: FigureLayout, height_in: float) -> Figure:
    """Return a figure that renders without a display.

    Deliberately not ``pyplot``: importing it binds whatever interactive backend
    the machine happens to offer, which on a desktop means matplotlib tries to
    open a Tk window to draw a file that is only ever going to be saved. That
    fails outright on a headless CI runner and fails intermittently everywhere
    else. The object-oriented API touches no global state and needs no display,
    which is what a library writing files should be doing anyway.

    Attaching an Agg canvas does not restrict the output format: ``savefig``
    swaps in the right canvas for the extension, so SVG and PDF still work.
    """
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure as MplFigure

    figure = MplFigure(figsize=(layout.width_in, height_in))
    FigureCanvasAgg(figure)
    return figure


def _line_kwargs(
    key: str, theme: str, *, n_points: int | None = None, markers: bool = True
) -> dict[str, Any]:
    """Colour, width, dash pattern and marker for one model's line.

    Three channels rather than one. Colour alone fails a greyscale print and a
    colour-blind reader; the dash pattern fails where two curves lie on top of
    each other and one is drawn over the other; the marker survives both.

    ``n_points`` thins the markers to about six across the curve, because a
    marker on every sample is a solid band rather than an encoding.
    """
    style = SERIES[key]
    kwargs: dict[str, Any] = {
        "color": style.colour(theme),
        "linewidth": style.width,
        "label": style.label,
        "solid_capstyle": "round",
    }
    if style.mpl_dash:
        kwargs["dashes"] = list(style.mpl_dash)
    if markers:
        kwargs["marker"] = style.marker
        kwargs["markerfacecolor"] = "none"
        kwargs["markeredgewidth"] = 1.0
        if n_points:
            kwargs["markevery"] = max(1, n_points // 6)
    return kwargs


def _comparison_line(comparison: dict[str, Any]) -> str:
    """One line of a path-comparison caption, leaving absent numbers absent."""
    parts = [f"{comparison['a']} vs {comparison['b']}:"]
    for label, key in (("mean", "mean_mm"), ("max", "max_mm"), ("Frechet", "frechet_mm")):
        value = comparison.get(key)
        parts.append(f"{label} -" if value is None else f"{label} {value:.3f} mm")
    return " ".join(parts[:1]) + " " + ", ".join(parts[1:])


def _wrap(text: str, width: int) -> str:
    """Hard-wrap a caption line; matplotlib does no wrapping of its own."""
    import textwrap

    return "\n".join(textwrap.wrap(text, width))


def _caption(
    figure: Figure,
    lines: list[str],
    theme: str,
    layout: FigureLayout,
    *,
    top: float = -0.02,
) -> float:
    """Draw a caption beneath the axes and return where the footer should start.

    Captions live outside the axes because the figures they belong to are curves
    that sweep the full plot height -- a text box inside would sit on the data.
    The returned ``y`` accounts for how many lines the text actually wrapped to,
    so the footer never lands on top of it.
    """
    if not lines:
        return top
    text = "\n".join(_wrap(line, layout.wrap_chars()) for line in lines)
    figure.text(
        0.0,
        top,
        text,
        fontsize=layout.caption_pt,
        va="top",
        ha="left",
        color=INK[theme]["secondary"],
        transform=figure.transFigure,
    )
    line_fraction = 1.4 * layout.caption_pt / 72.0 / figure.get_figheight()
    return top - line_fraction * (text.count("\n") + 1) - 0.01


def _footer(
    figure: Figure,
    provenance: Provenance,
    theme: str,
    layout: FigureLayout,
    extra: str | None = None,
    *,
    y: float = -0.015,
) -> None:
    """Stamp the caveat and the provenance along the bottom of a figure.

    ``y`` is in figure coordinates and may be negative: ``savefig`` uses a tight
    bounding box, so text below the axes is kept rather than clipped. Push it
    further down when a figure carries its own caption above the footer.
    """
    ink = INK[theme]
    lines = []
    caveat = provenance.caveat()
    if caveat:
        lines.append(caveat)
    if extra:
        lines.append(extra)
    commit = provenance.code_commit[:12] if provenance.code_commit else "unknown"
    lines.append(
        f"cmtool {provenance.version} @ {commit} - config "
        f"{(provenance.config_hash or 'n/a')[:12]} - {provenance.created_utc}"
    )
    figure.text(
        0.0,
        y,
        "\n".join(_wrap(line, layout.wrap_chars(layout.footer_pt)) for line in lines),
        fontsize=layout.footer_pt,
        color=STATUS["critical"] if caveat else ink["muted"],
        va="top",
        ha="left",
        transform=figure.transFigure,
    )


def _title(axes: Axes, text: str, layout: FigureLayout) -> None:
    """Set an axes title, wrapped to the figure width.

    A title is one long line by default, and ``savefig``'s tight bounding box
    widens the saved figure to fit it -- so an unwrapped title quietly pushes a
    figure out of the column it was laid out for.
    """
    axes.set_title(_wrap(text, layout.wrap_chars(layout.title_pt)))


def _direct_label(
    axes: Axes,
    x: float,
    y: float,
    text: str,
    colour: str,
    layout: FigureLayout,
    *,
    offset: tuple[float, float] = (4.0, 4.0),
) -> None:
    """Label a curve at its own end, so identity never rests on colour alone."""
    axes.annotate(
        text,
        xy=(x, y),
        xytext=offset,
        textcoords="offset points",
        fontsize=layout.base_pt - 0.5,
        color=colour,
        fontweight="semibold",
    )


def _save(figure: Figure, out: Path, name: str, formats: tuple[str, ...]) -> Path:
    """Write a figure in every requested format; return the first one's path."""
    out.mkdir(parents=True, exist_ok=True)
    paths = []
    for suffix in formats:
        path = out / f"{name}.{suffix}"
        figure.savefig(path, pad_inches=0.02)
        paths.append(path)
    return paths[0]


# ------------------------------------------------------------- figures ------


def report_is_measured(mechanism: CompliantMechanism) -> bool:
    """Whether this conversion used a measured allowable strain."""
    return not any(
        name.endswith("allowable_strain") for name in mechanism.provenance.placeholders_used
    )


def fig_path_overlay(
    inputs: FigureInputs,
    out: Path,
    *,
    theme: str = "light",
    formats: tuple[str, ...] = ("png",),
    column: str = "single",
) -> FigureResult:
    """Draw the coupler path four ways, and how far apart the four actually are.

    The headline A6 figure, in two panels because one will not do the job. The
    paths agree to within a millimetre over a part 140 mm across, so drawn on top
    of each other they are a single curve -- which is itself the finding, and is
    why the second panel exists. Separation is the quantity the Phase A go/no-go
    is stated against, so it gets an axis of its own rather than an inset.
    """
    scene = inputs.primary_scene
    layout = layout_for(column)
    _apply_style(theme, layout)
    ink = INK[theme]

    figure = _new_figure(layout, layout.height(5.1, 4.0))
    if layout.stacked:
        left, right = figure.subplots(2, 1, gridspec_kw={"height_ratios": [1.1, 1.0]})
    else:
        left, right = figure.subplots(1, 2, gridspec_kw={"width_ratios": [1.05, 1.0]})
    present = [k for k in ("rigid", "prbm", "fea", "measured") if k in scene.models]
    missing = [] if "measured" in present else ["measured"]

    # --- left: the path itself, to scale ---------------------------------
    for key in present:
        path = np.asarray(scene.models[key].path_mm, dtype=float)
        left.plot(path[:, 0], path[:, 1], **_line_kwargs(key, theme, n_points=len(path)))
    left.set_aspect("equal", adjustable="box")
    left.set_xlabel("x (mm)")
    left.set_ylabel("y (mm)")
    _title(left, f"Coupler point {scene.output_name} of {scene.name}", layout)
    left.legend(loc="best")
    left.text(
        0.02,
        0.02,
        f"all {len(present)} curves, to scale",
        transform=left.transAxes,
        fontsize=layout.base_pt - 0.8,
        color=ink["muted"],
    )

    # --- right: where they differ ----------------------------------------
    reference = scene.deviation_reference
    drawn_any = False
    for key in present:
        series = scene.models[key]
        if key == reference or not series.deviation_mm:
            continue
        x = np.asarray(series.input_angles_deg, dtype=float)
        y = np.asarray(series.deviation_mm, dtype=float)
        right.plot(x, y, **_line_kwargs(key, theme, n_points=len(x)))
        _direct_label(
            right,
            x[-1],
            y[-1],
            SERIES[key].label,
            SERIES[key].colour(theme),
            layout,
            offset=(-2.0, 5.0),
        )
        drawn_any = True

    if not drawn_any:
        right.text(
            0.5,
            0.5,
            "only one model in this scene, so there is nothing to compare",
            transform=right.transAxes,
            ha="center",
            fontsize=layout.base_pt,
            color=ink["muted"],
        )

    right.set_xlabel("input angle (deg)")
    right.set_ylabel(f"distance from the {reference} path (mm)")
    _title(right, "How far apart the predictions are", layout)
    right.set_ylim(bottom=0.0)

    # The numbers go in a caption rather than inside the axes: this panel's whole
    # job is a curve that sweeps the full height, and a text box would sit on it.
    summary = [_comparison_line(c) for c in scene.comparisons]
    flat = scene.models.get("prbm")
    if flat is not None and flat.deviation_mm and max(flat.deviation_mm) < 1e-9:
        summary.append(
            "PRBM sits exactly on the rigid path: with a prescribed input and pivot-matched "
            "placement, stiffness cannot move a one-degree-of-freedom path. Only the FEA, "
            "which lets the flexures stretch and shear, departs from it."
        )
    if missing:
        summary.append("No measured path yet -- that series is absent, not zero.")

    figure.tight_layout()
    _footer(figure, scene.provenance, theme, layout, y=_caption(figure, summary, theme, layout))
    written = _save(figure, out, "path_overlay", formats)
    return FigureResult(
        name="path_overlay",
        produced=True,
        path=written,
        missing=missing,
        inputs={"design": scene.name, "states": scene.n_frames},
    )


def fig_torque_curves(
    inputs: FigureInputs,
    out: Path,
    *,
    theme: str = "light",
    formats: tuple[str, ...] = ("png",),
    column: str = "single",
) -> FigureResult:
    """Input torque against input angle: the measurement that tests stiffness.

    With a prescribed input the coupler path is fixed by geometry, so the path
    cannot discriminate between stiffness models at all. Torque can, and the two
    models differ here by about a third -- far more than a set of kitchen weights
    resolves, which is the whole argument for the dead-weight rig.
    """
    scene = inputs.primary_scene
    layout = layout_for(column)
    _apply_style(theme, layout)

    figure = _new_figure(layout, layout.height(2.7, 3.8))
    axes = figure.subplots()
    angles = np.asarray(scene.input_angles_deg, dtype=float)
    drawn = []
    for key in ("prbm", "fea"):
        series = scene.models.get(key)
        if series is None or series.torque_nmm is None:
            continue
        torque = np.abs(np.asarray(series.torque_nmm, dtype=float))
        axes.plot(angles, torque, **_line_kwargs(key, theme, n_points=len(angles)))
        _direct_label(
            axes,
            angles[-1],
            torque[-1],
            SERIES[key].label,
            SERIES[key].colour(theme),
            layout,
            offset=(-4.0, 4.0),
        )
        drawn.append((key, float(np.max(torque))))

    missing: list[str] = []
    if inputs.torque_readings:
        loading = [r for r in inputs.torque_readings if getattr(r, "direction", "") == "loading"]
        unloading = [
            r for r in inputs.torque_readings if getattr(r, "direction", "") == "unloading"
        ]
        for subset, marker, label in (
            (loading, "o", "measured, loading"),
            (unloading, "s", "measured, unloading"),
        ):
            if not subset:
                continue
            axes.scatter(
                [r.input_angle_deg for r in subset],
                [abs(r.torque_nmm) for r in subset],
                marker=marker,
                s=layout.marker_pt**2 * 2.4,
                facecolors="none",
                edgecolors=SERIES["measured"].colour(theme),
                linewidths=1.3,
                label=label,
                zorder=5,
            )
    else:
        missing.append("measured")

    caption: list[str] = []
    if len(drawn) == 2:
        (first_key, first), (second_key, second) = drawn
        higher, lower = (first, second) if first >= second else (second, first)
        higher_key = first_key if first >= second else second_key
        gap = (higher / lower - 1.0) * 100.0 if lower > 0 else float("nan")
        caption.append(
            f"Peak torque {first:.1f} N·mm ({SERIES[first_key].label}) against "
            f"{second:.1f} N·mm ({SERIES[second_key].label}): the "
            f"{SERIES[higher_key].label} is {gap:.0f}% higher. The path cannot tell these "
            "two apart at all, because with a prescribed input it is fixed by geometry."
        )
    if missing:
        caption.append(
            "No torque readings yet: fill in the template written beside the print sheet "
            "and pass it with --torque. Record loading and unloading separately -- the gap "
            "between them is hysteresis plus rig friction, not hysteresis alone."
        )

    axes.set_xlabel("input angle (deg)")
    axes.set_ylabel("input torque magnitude (N·mm)")
    _title(axes, f"Torque to hold {scene.name} at each input angle", layout)
    axes.legend(loc="lower right")

    figure.tight_layout()
    _footer(figure, scene.provenance, theme, layout, y=_caption(figure, caption, theme, layout))
    path = _save(figure, out, "torque_curves", formats)
    return FigureResult(
        name="torque_curves",
        produced=True,
        path=path,
        missing=missing,
        inputs={"design": scene.name, "n_readings": len(inputs.torque_readings)},
    )


def fig_strain(
    inputs: FigureInputs,
    out: Path,
    *,
    theme: str = "light",
    formats: tuple[str, ...] = ("png",),
    column: str = "single",
) -> FigureResult:
    """Draw peak flexure strain through the arc, one panel per joint.

    Each flexure is printed unstressed at mid-arc, so every curve is a V: zero in
    the middle, largest at both ends. That shape is the factor of two that
    decision buys, drawn.

    One panel per joint rather than four curves on one axes. Four identities in
    one plot would need four hues that separate for every pair in both light and
    dark, and no four of this palette's hues do -- so identity is carried by
    position and a panel title instead, which does not depend on colour vision at
    all. It also makes the real result legible: the four joints sit almost on top
    of each other, and none of them is the outlier.
    """
    scene = inputs.primary_scene
    fea = scene.models.get("fea")
    if fea is None or not fea.frames:
        return FigureResult(
            name="strain",
            produced=False,
            reason="the scene has no beam FEA, so there is no strain to plot",
        )

    layout = layout_for(column)
    _apply_style(theme, layout)
    sweep = np.asarray(scene.input_sweep_deg, dtype=float)
    joints = scene.joints
    allowable = scene.allowable_strain * 100.0

    # One column wide cannot take four panels in a row, so they wrap into a
    # grid; a full-width plate keeps them in one row.
    if layout.stacked:
        columns = 2 if len(joints) > 2 else len(joints)
        rows = -(-len(joints) // columns)
        figure = _new_figure(layout, 1.5 * rows + 1.0)
    else:
        columns, rows = len(joints), 1
        figure = _new_figure(layout, 3.4)
    panels = figure.subplots(rows, columns, sharey=True, sharex=True)
    axes_list = list(np.atleast_1d(panels).ravel())
    for spare in axes_list[len(joints) :]:
        spare.set_visible(False)
    peaks: dict[str, float] = {}

    for axes, joint in zip(axes_list, joints, strict=False):
        values = (
            np.asarray([max(frame.flexure_strain[joint]) for frame in fea.frames], dtype=float)
            * 100.0
        )
        peaks[joint] = float(values.max())
        axes.plot(
            sweep,
            values,
            color=SERIES["fea"].colour(theme),
            linewidth=SERIES["fea"].width,
            marker=SERIES["fea"].marker,
            markerfacecolor="none",
            markeredgewidth=1.0,
            markevery=max(1, len(sweep) // 5),
        )
        axes.axhline(allowable, color=STATUS["critical"], linewidth=1.3, dashes=[5, 3])
        axes.axhspan(
            allowable,
            allowable * 2.0,
            facecolor="none",
            edgecolor=STATUS["critical"],
            hatch="xxx",
            linewidth=0.0,
            alpha=0.45,
        )
        axes.set_title(f"joint {joint}", fontsize=layout.base_pt + 0.5)
        axes.set_ylim(0, allowable * 1.25)

    for axes in axes_list[: len(joints)]:
        if axes.get_subplotspec().is_last_row() or layout.stacked is False:
            axes.set_xlabel("deg")
        if axes.get_subplotspec().is_first_col():
            axes.set_ylabel("strain (%)")
    axes_list[0].text(
        sweep[0],
        allowable,
        " allowable",
        fontsize=layout.base_pt - 0.8,
        va="bottom",
        color=STATUS["critical"],
    )
    figure.suptitle(
        _wrap(f"Flexure strain through the arc, {scene.name}", layout.wrap_chars(layout.title_pt)),
        fontsize=layout.title_pt,
        x=0.01,
        ha="left",
    )

    worst = max(peaks, key=lambda j: peaks[j])
    caption = [
        "x axis: input rotation from the as-printed state (deg). Printed unstressed at "
        "mid-arc, so each flexure swings symmetrically about zero and its peak bend is "
        "half what printing at one end of the arc would give.",
        "Peak strain: "
        + ", ".join(f"{j} {v:.3f}%" for j, v in peaks.items())
        + f" -- worst is joint {worst} at {peaks[worst] / allowable * 100:.0f}% of the allowable.",
    ]
    if scene.allowable_strain_is_placeholder:
        caption.append(
            f"The allowable strain drawn here ({scene.allowable_strain:g}) is a PLACEHOLDER, "
            "not a measurement. The strain curves are real solver output; what they are "
            "being compared against is not yet. The strain coupon is what replaces it."
        )
    figure.tight_layout()
    _footer(figure, scene.provenance, theme, layout, y=_caption(figure, caption, theme, layout))
    written = _save(figure, out, "strain", formats)
    return FigureResult(
        name="strain",
        produced=True,
        path=written,
        missing=["measured_allowable_strain"] if scene.allowable_strain_is_placeholder else [],
        inputs={"design": scene.name, "allowable_strain": scene.allowable_strain, "peaks": peaks},
    )


def fig_feasibility(
    inputs: FigureInputs,
    out: Path,
    *,
    theme: str = "light",
    formats: tuple[str, ...] = ("png",),
    column: str = "single",
) -> FigureResult:
    """Per joint: the flexure strain demands, the flexure that fits, and the ratio.

    The two hard limits side by side, in the same units, for every joint of every
    pilot design. The joint whose bars are closest together is the one limiting
    the design -- the number that says which joint to fix.
    """
    layout = layout_for(column)
    _apply_style(theme, layout)
    ink = INK[theme]
    mechanisms = inputs.mechanisms
    labels: list[str] = []
    strain_mm: list[float] = []
    geometric_mm: list[float] = []
    utilisation: list[float] = []
    binding: list[bool] = []

    for name, mechanism in mechanisms.items():
        worst = mechanism.feasibility.binding_joint
        for joint, sized in mechanism.sizing.items():
            labels.append(f"{name.replace('fb_02_', '')}:{joint}")
            strain_mm.append(sized.min_length_strain_mm)
            geometric_mm.append(sized.max_length_geometric_mm)
            utilisation.append(sized.utilisation)
            binding.append(joint == worst)

    figure = _new_figure(layout, 0.20 * len(labels) + 1.7)
    axes = figure.subplots()
    positions = np.arange(len(labels), dtype=float)
    # A bar has no dash pattern to carry, so the second channel is a hatch.
    axes.barh(
        positions + 0.19,
        geometric_mm,
        height=0.34,
        facecolor=SERIES["fea"].colour(theme),
        edgecolor=ink["surface"],
        hatch=HATCH["fea"],
        linewidth=0.5,
        label="longest that fits (geometry)",
    )
    axes.barh(
        positions - 0.19,
        strain_mm,
        height=0.34,
        facecolor=SERIES["rigid"].colour(theme),
        edgecolor=ink["primary"],
        hatch=HATCH["rigid"],
        linewidth=0.5,
        label="shortest that survives the bend (strain)",
    )
    for position, value, ratio, is_binding in zip(
        positions, geometric_mm, utilisation, binding, strict=True
    ):
        axes.text(
            value + 0.4,
            position,
            f"{ratio:.2f}" + ("  <-- binding" if is_binding else ""),
            va="center",
            fontsize=layout.base_pt - 0.8,
            color=STATUS["critical"] if ratio >= 1.0 else ink["secondary"],
            fontweight="semibold" if is_binding else "normal",
        )

    axes.set_yticks(positions)
    axes.set_yticklabels(labels, fontsize=layout.base_pt - 1.0)
    axes.invert_yaxis()
    axes.set_xlim(0, max(geometric_mm) * 1.28)
    axes.set_xlabel("flexure length (mm)")
    _title(axes, "Both hard limits per joint; the label is utilisation", layout)
    axes.grid(axis="y", visible=False)
    # Below the axes, not above: the title wraps to the column width and a
    # legend anchored over it lands on top of the second line.
    axes.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.10 if layout.stacked else -0.16),
        ncol=1 if layout.stacked else 2,
    )
    figure.tight_layout()
    _footer(
        figure,
        inputs.provenance,
        theme,
        layout,
        y=_caption(
            figure,
            [
                "Utilisation is strain-required length over geometry-allowed length; above "
                "1.0 the joint cannot be built. PRBM model validity is metadata and is "
                "deliberately not a limit here."
            ],
            theme,
            layout,
        ),
    )
    path = _save(figure, out, "feasibility", formats)
    return FigureResult(
        name="feasibility",
        produced=True,
        path=path,
        inputs={"designs": list(mechanisms)},
    )


def fig_design_rule(
    inputs: FigureInputs,
    out: Path,
    *,
    theme: str = "light",
    formats: tuple[str, ...] = ("png",),
    column: str = "single",
) -> FigureResult:
    """Draw the governing bound: usable rotation against adjacent link length.

    ``theta_max = 2 f l eps_allow / (t SF)``. The counter-intuitive part is that
    the adjacent link length sets the bound -- so a crank-rocker with a stubby
    crank is the wrong shape for a compliant mechanism at any flexure thickness.

    Flexure thickness is an *ordered* quantity, not an identity, so the four
    curves take an ordinal one-hue ramp rather than four separate hues. The pilot
    designs are marked at their own binding joint, which is where the rule stops
    being a plot and starts being a constraint on a real part.
    """
    from cmtool.convert.limits import max_bend_deg

    layout = layout_for(column)
    _apply_style(theme, layout)
    ink = INK[theme]
    mechanism = inputs.primary_mechanism
    report = mechanism.feasibility
    allowable = report.allowable_strain
    fraction = report.max_length_fraction
    safety = report.strain_safety_factor

    figure = _new_figure(layout, layout.height(3.1, 4.2))
    axes = figure.subplots()
    lengths = np.linspace(5.0, 90.0, 200)
    ramp = ORDINAL_RAMP_DARK if theme == "dark" else ORDINAL_RAMP_LIGHT
    top = 90.0

    for index, thickness in enumerate(DESIGN_RULE_THICKNESS_MM):
        excursion = np.array(
            [
                max_bend_deg(
                    length,
                    thickness,
                    allowable,
                    max_length_fraction=fraction,
                    strain_safety_factor=safety,
                ).max_excursion_deg
                for length in lengths
            ]
        )
        colour = ramp[min(index, len(ramp) - 1)]
        # An ordinal ramp is already ordered by lightness, so it survives one ink;
        # the dash pattern makes the order readable rather than merely present.
        dashes = DESIGN_RULE_DASHES[index % len(DESIGN_RULE_DASHES)]
        curve_style: dict[str, Any] = {
            "color": colour,
            "linewidth": 1.9,
            "label": f"t = {thickness} mm",
        }
        if dashes:
            # Matplotlib rejects dashes=None; solid is the absence of the keyword.
            curve_style["dashes"] = dashes
        axes.plot(lengths, excursion, **curve_style)
        # Label where the curve leaves the frame, not at its last computed point,
        # which for the thin flexures is far above the top of the axes.
        inside = np.nonzero(excursion <= top * 0.93)[0]
        if inside.size:
            last = int(inside[-1])
            _direct_label(axes, lengths[last], excursion[last], f"{thickness} mm", colour, layout)

    marked: list[dict[str, Any]] = []
    for offset, (name, converted) in enumerate(inputs.mechanisms.items()):
        worst = converted.feasibility.binding_joint
        sized = converted.sizing[worst]
        axes.scatter(
            [sized.shortest_adjacent_link_mm],
            [sized.excursion_deg],
            s=layout.marker_pt**2 * 3.0,
            marker=SERIES["measured"].marker,
            facecolors=ink["surface"],
            edgecolors=SERIES["measured"].colour(theme),
            linewidths=1.3,
            zorder=5,
            label="pilot, binding joint" if offset == 0 else None,
        )
        axes.annotate(
            f"{name.replace('fb_02_', '')} ({worst})",
            xy=(sized.shortest_adjacent_link_mm, sized.excursion_deg),
            # The pilots cluster: the arc fit drives every design to the same
            # excursion target, so their markers land on top of one another.
            # Fan the labels out and lead each one back to its own marker.
            xytext=(12, 24 - 15 * offset),
            textcoords="offset points",
            fontsize=layout.base_pt - 0.8,
            color=ink["secondary"],
            arrowprops={
                "arrowstyle": "-",
                "color": ink["axis"],
                "linewidth": 0.8,
                "shrinkA": 0,
                "shrinkB": 3,
            },
        )
        marked.append(
            {
                "design": name,
                "joint": worst,
                "link_mm": sized.shortest_adjacent_link_mm,
                "excursion_deg": sized.excursion_deg,
            }
        )

    axes.set_xlabel("shorter adjacent link length (mm)")
    axes.set_ylabel("usable peak-to-peak joint excursion (deg)")
    axes.set_xlim(0, 92)
    axes.set_ylim(0, top)
    _title(axes, "The bound that governs: usable rotation is set by the adjacent link", layout)
    axes.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.17),
        ncol=2 if layout.stacked else 5,
        columnspacing=1.0,
        handlelength=1.8,
    )

    caption = [
        "theta_max = 2 f l eps_allow / (t SF), peak-to-peak, printed unstressed at mid-arc, "
        f"with f = {fraction}, SF = {safety}, eps_allow = {allowable:g}"
        + ("" if report_is_measured(mechanism) else " (PLACEHOLDER -- not measured)")
        + ".",
        "Every pilot sits far below its own ceiling, which is the point: the search picked "
        "link lengths that can carry the rotation, rather than flexures thin enough to "
        "survive links that cannot.",
    ]
    figure.tight_layout()
    _footer(figure, mechanism.provenance, theme, layout, y=_caption(figure, caption, theme, layout))
    written = _save(figure, out, "design_rule", formats)
    return FigureResult(
        name="design_rule",
        produced=True,
        path=written,
        missing=[] if report_is_measured(mechanism) else ["measured_allowable_strain"],
        inputs={"allowable_strain": allowable, "designs": marked},
    )


def fig_conversion_artefact(
    inputs: FigureInputs,
    out: Path,
    *,
    theme: str = "light",
    formats: tuple[str, ...] = ("png",),
    column: str = "single",
) -> FigureResult:
    """Draw pivot-matched against unmatched placement: an artefact, not a physics gap.

    A small-length flexure pivots about its centre, not its ends. Drop flexures
    in without accounting for that and every effective link length shifts by part
    of a flexure length, moving the coupler path before any physics enters. This
    figure is how the size of that artefact gets reported rather than guessed at
    -- and why it must never be mistaken for a simulation-to-reality gap.

    Both curves here are *rigid* simulations. No material, no stiffness, no
    solver difference: the only thing that changed is where the flexures were
    put.
    """
    from cmtool.api import convert, simulate
    from cmtool.core.graph import Linkage
    from cmtool.metrics.paths import compare_paths

    scene = inputs.primary_scene
    mechanism = inputs.primary_mechanism
    linkage: Linkage = mechanism.base
    angles = np.asarray(scene.input_angles_deg, dtype=float)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        unmatched = convert(
            linkage,
            material=mechanism.material_name,
            printer=mechanism.printer_name,
            input_range_deg=mechanism.input_range_deg,
            placement="unmatched",
        )
        shifted = simulate(unmatched.effective_linkage(), solver="rigid", input_angles_deg=angles)

    matched = np.asarray(scene.models["rigid"].path_mm, dtype=float)
    moved = shifted.path(scene.output_name)
    gap = compare_paths(matched, moved, labels=("pivot_matched", "unmatched"))
    separation = np.linalg.norm(matched - moved, axis=1)

    layout = layout_for(column)
    _apply_style(theme, layout)
    figure = _new_figure(layout, layout.height(5.1, 4.0))
    if layout.stacked:
        left, right = figure.subplots(2, 1, gridspec_kw={"height_ratios": [1.1, 1.0]})
    else:
        left, right = figure.subplots(1, 2, gridspec_kw={"width_ratios": [1.05, 1.0]})

    matched_colour = SERIES["fea"].colour(theme)
    moved_colour = SERIES["rigid"].colour(theme)
    step = max(1, len(matched) // 6)
    left.plot(
        matched[:, 0],
        matched[:, 1],
        color=matched_colour,
        linewidth=2.1,
        marker=SERIES["fea"].marker,
        markerfacecolor="none",
        markeredgewidth=1.0,
        markevery=step,
        label="pivot-matched (default)",
    )
    left.plot(
        moved[:, 0],
        moved[:, 1],
        color=moved_colour,
        linewidth=1.9,
        dashes=[7, 3],
        marker=SERIES["rigid"].marker,
        markerfacecolor="none",
        markeredgewidth=1.0,
        markevery=step,
        label="unmatched placement",
    )
    left.set_aspect("equal", adjustable="box")
    left.set_xlabel("x (mm)")
    left.set_ylabel("y (mm)")
    _title(left, f"Coupler path of {scene.name}, both placements", layout)
    left.legend(loc="best")

    right.plot(
        angles,
        separation,
        color=moved_colour,
        linewidth=1.9,
        dashes=[7, 3],
        marker=SERIES["rigid"].marker,
        markerfacecolor="none",
        markeredgewidth=1.0,
        markevery=step,
    )
    right.fill_between(
        angles,
        0.0,
        separation,
        facecolor="none",
        edgecolor=moved_colour,
        hatch=HATCH["rigid"],
        linewidth=0.0,
        alpha=0.35,
    )
    right.set_xlabel("input angle (deg)")
    right.set_ylabel("path shift from placement alone (mm)")
    _title(right, "How much the placement rule moves it", layout)
    right.set_ylim(bottom=0.0)

    shortest = min(s.geometry.length_mm for s in mechanism.sizing.values())
    longest = max(s.geometry.length_mm for s in mechanism.sizing.values())
    fea_gap = next(
        (c for c in scene.comparisons if {c["a"], c["b"]} == {"rigid", "fea"}),
        None,
    )
    caption = [
        f"Mean {gap.mean_mm:.3f} mm, max {gap.max_mm:.3f} mm of path shift, with flexures "
        f"{shortest:.1f}-{longest:.1f} mm long. Both curves are rigid simulations: no "
        "material, no stiffness, no solver difference. The only change is where the "
        "flexures were placed.",
    ]
    if fea_gap is not None and fea_gap.get("mean_mm") is not None:
        caption.append(
            f"For scale, the PRBM-to-FEA disagreement on the same design is "
            f"{fea_gap['mean_mm']:.3f} mm mean -- the same order. Left unmatched, this "
            "artefact would be indistinguishable from the physics the project exists to "
            "measure, which is why placement is matched by default and the unmatched "
            "variant is kept only to size the effect."
        )
    figure.tight_layout()
    _footer(figure, mechanism.provenance, theme, layout, y=_caption(figure, caption, theme, layout))
    written = _save(figure, out, "conversion_artefact", formats)
    return FigureResult(
        name="conversion_artefact",
        produced=True,
        path=written,
        inputs={
            "design": scene.name,
            "mean_mm": gap.mean_mm,
            "max_mm": gap.max_mm,
            "flexure_length_mm": [shortest, longest],
        },
    )


def fig_prbm_variant(
    out: Path,
    *,
    provenance: Provenance | None = None,
    theme: str = "light",
    formats: tuple[str, ...] = ("png",),
    column: str = "single",
    load_ratios: tuple[float, ...] | None = None,
) -> FigureResult:
    """Draw which long-segment PRBM variant applies, decided by our own beam FEA.

    Howell gives different constants for different end loadings, **and they do
    not share a stiffness formula**. Which one a flexure joint needs is an
    empirical question; this is the answer, fitted across tip-load ratios.

    Two panels rather than two y-axes: gamma is dimensionless and K is in units
    of EI/L, and a dual-axis plot of them would invite a comparison that means
    nothing.

    Note on provenance: unlike every other figure here, this one carries **no
    placeholder caveat**, and that is not an oversight. Both fitted constants are
    dimensionless -- gamma is geometric and the stiffness is reported as a
    multiple of EI/L -- so the modulus cancels and no config quantity enters. It
    is a physical result today.
    """
    from cmtool.flexures.prbm_models import PRBM_MODELS, LongSegmentModel
    from cmtool.solvers.prbm_study import DEFAULT_LOAD_RATIOS, JOINT_LOAD_RATIO, sweep

    layout = layout_for(column)
    _apply_style(theme, layout)
    ink = INK[theme]
    ratios = load_ratios or DEFAULT_LOAD_RATIOS
    fitted = sweep(ratios)

    figure = _new_figure(layout, layout.height(4.7, 3.8))
    if layout.stacked:
        left, right = figure.subplots(2, 1)
    else:
        left, right = figure.subplots(1, 2)
    x = np.array([f.load_ratio for f in fitted])
    panels = (
        (left, np.array([f.gamma for f in fitted]), "gamma", "gamma"),
        (
            right,
            np.array([f.stiffness_multiple for f in fitted]),
            "K / (EI/L)",
            "stiffness_multiple",
        ),
    )
    references: dict[str, dict[str, float]] = {}
    for axes, values, label, attribute in panels:
        axes.plot(
            x,
            values,
            color=SERIES["fea"].colour(theme),
            linewidth=SERIES["fea"].width,
            marker=SERIES["fea"].marker,
            markerfacecolor="none",
            markeredgewidth=1.0,
            label="fitted to our beam FEA",
        )
        for variant, dash, series_key in (
            ("end_moment", [6, 3], "prbm"),
            ("end_force", [2, 2], "rigid"),
        ):
            model = PRBM_MODELS.get(f"long_segment_{variant}")
            assert isinstance(model, LongSegmentModel)
            reference = model.gamma() if attribute == "gamma" else model.stiffness_multiple()
            references.setdefault(variant, {})[attribute] = float(reference)
            axes.axhline(
                reference,
                color=SERIES[series_key].colour(theme),
                linewidth=1.4,
                dashes=dash,
                label=f"Howell, {variant.replace('_', ' ')}",
            )
        axes.axvline(JOINT_LOAD_RATIO, color=ink["muted"], linewidth=1.0, dashes=[1, 3])
        # symlog keeps the interesting decade near zero readable while still
        # reaching lambda = 5; the limit stops it drawing a negative arm that a
        # load ratio can never occupy.
        axes.set_xscale("symlog", linthresh=0.05, linscale=0.4)
        axes.set_xlim(0.0, max(x) * 1.15)
        # symlog's own ticks put "0" and "10^-2" on top of each other at the
        # linear-to-log seam; name the ones a reader actually wants instead.
        chosen = [v for v in (0.0, 0.1, 1.0, 5.0) if v <= max(x)]
        axes.set_xticks(chosen)
        axes.set_xticklabels([f"{v:g}" for v in chosen])
        axes.set_xlabel("tip load ratio  lambda = P L / M")
        axes.set_ylabel(label)
        _title(axes, label, layout)
        axes.annotate(
            "a flexure joint",
            xy=(JOINT_LOAD_RATIO, axes.get_ylim()[1]),
            xytext=(4, -9),
            textcoords="offset points",
            fontsize=layout.base_pt - 0.8,
            color=ink["secondary"],
            va="top",
        )

    joint = next(f for f in fitted if abs(f.load_ratio - JOINT_LOAD_RATIO) < 1e-9)
    errors = {
        variant: abs(values["stiffness_multiple"] - joint.stiffness_multiple)
        / joint.stiffness_multiple
        for variant, values in references.items()
    }
    best = min(errors, key=lambda v: errors[v])

    handles, labels = left.get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.02),
        ncol=1 if layout.stacked else 3,
    )
    figure.suptitle(
        _wrap(
            "A flexure joint is moment-dominated, so the end-moment variant is in use",
            layout.wrap_chars(layout.title_pt),
        ),
        fontsize=layout.title_pt,
        x=0.01,
        ha="left",
    )

    caption = [
        f"At lambda = {JOINT_LOAD_RATIO} the beam FEA gives gamma = {joint.gamma:.4f} and "
        f"K = {joint.stiffness_multiple:.4f} EI/L. The {best.replace('_', '-')} variant is "
        f"within {errors[best] * 100:.1f}% on stiffness; the other is off by "
        f"{max(errors.values()) * 100:.0f}%.",
        "Why lambda ~ 0.1 is the joint's own ratio: under no external load the only forces "
        "in a compliant four-bar are those bending the other flexures, so the moment a "
        "flexure carries is of order K.dphi while the transverse force is that over a link "
        "length -- their ratio is the flexure's length ratio L/l.",
        "Both fitted constants are dimensionless, so no material data enters and this "
        "figure is a physical result even while the material configs are placeholders.",
    ]
    figure.tight_layout()
    _footer(
        figure,
        provenance or Provenance(notes={"figure": "prbm_variant"}),
        theme,
        layout,
        y=_caption(figure, caption, theme, layout, top=-0.16 if layout.stacked else -0.12),
    )
    written = _save(figure, out, "prbm_variant", formats)
    return FigureResult(
        name="prbm_variant",
        produced=True,
        path=written,
        inputs={
            "load_ratios": list(ratios),
            "joint_load_ratio": JOINT_LOAD_RATIO,
            "fitted_at_joint": joint.to_dict(),
            "recommended": best,
            "stiffness_errors": errors,
        },
    )


def fig_uncertainty(
    inputs: FigureInputs,
    out: Path,
    *,
    theme: str = "light",
    formats: tuple[str, ...] = ("png",),
    column: str = "single",
) -> FigureResult:
    """Draw the Phase A go/no-go: uncertainty against the signal it has to resolve.

    Skipped entirely when no tracked take exists. There is no honest way to draw
    this figure from simulation -- the synthetic harness validates the software,
    not the rig, and putting its numbers here would answer the go/no-go with the
    wrong measurement.
    """
    report = inputs.uncertainty
    if not report:
        return FigureResult(
            name="uncertainty",
            produced=False,
            reason=(
                "no tracking uncertainty report: run 'cmtool uncertainty --static/--circle "
                "--json out/uncertainty.json' on real footage and pass it with --uncertainty"
            ),
            missing=["measured"],
        )

    layout = layout_for(column)
    _apply_style(theme, layout)
    ink = INK[theme]
    estimates = report.get("estimates", [])
    signal = float(report.get("signal_mm", 0.46))

    figure = _new_figure(layout, layout.height(2.5, 3.4))
    axes = figure.subplots()
    names = [e["method"] for e in estimates]
    positions = np.arange(len(names), dtype=float)
    axes.barh(
        positions,
        [e["sigma_mm"] for e in estimates],
        height=0.46,
        facecolor=SERIES["fea"].colour(theme),
        edgecolor=ink["surface"],
        linewidth=0.5,
        label="sigma",
    )
    axes.scatter(
        [e["max_deviation_mm"] for e in estimates],
        positions,
        marker="|",
        s=layout.marker_pt**2 * 14,
        linewidths=1.8,
        color=SERIES["measured"].colour(theme),
        label="worst deviation",
        zorder=5,
    )

    from matplotlib.transforms import blended_transform_factory

    # Threshold labels run vertically beside their own line: laid out flat they
    # collide with each other and with the title, because 5:1 and 3:1 of the same
    # signal are close together by construction.
    along = blended_transform_factory(axes.transData, axes.transAxes)
    for label, divisor, colour, dash in (
        ("signal to resolve", 1.0, ink["secondary"], [6.0, 2.0]),
        ("3:1 marginal", 3.0, STATUS["warning"], [2.0, 2.0]),
        ("5:1 go", 5.0, STATUS["good"], [1.0, 1.5]),
    ):
        axes.axvline(signal / divisor, color=colour, linewidth=1.3, dashes=dash)
        axes.text(
            signal / divisor,
            0.99,
            f"{label}  {signal / divisor:.3f} mm ",
            transform=along,
            rotation=90,
            fontsize=layout.base_pt - 1.0,
            color=colour,
            va="top",
            ha="right",
        )

    verdict = str(report.get("verdict", "unknown"))
    verdict_colour = {
        "go": STATUS["good"],
        "marginal": STATUS["warning"],
        "no-go": STATUS["critical"],
    }.get(verdict, ink["secondary"])
    axes.set_yticks(positions)
    axes.set_yticklabels(names)
    axes.set_xlabel("uncertainty (mm)")
    _title(
        axes,
        f"Tracking uncertainty against the {signal:.2f} mm PRBM-vs-FEA signal",
        layout,
    )
    axes.grid(axis="y", visible=False)
    axes.legend(loc="lower right")

    caption = [
        f"VERDICT: {verdict.upper()}. The verdict takes the worst of the methods used, not "
        "the most complete: a circle fit absorbs uniform offset and scale that static "
        "jitter never sees, and jitter ignores everything systematic. Neither contains the "
        "other.",
        "The circle residual is a shape check -- a wrong scale fits a circle perfectly -- so "
        "the known-motion test wants a caliper-measured radius and combines the radius "
        "error with the shape residual in quadrature.",
    ]
    figure.tight_layout()
    figure.text(
        0.0,
        1.01,
        f"VERDICT: {verdict.upper()}",
        fontsize=layout.title_pt,
        fontweight="semibold",
        color=verdict_colour,
        va="bottom",
        ha="left",
        transform=figure.transFigure,
    )
    _footer(
        figure,
        inputs.provenance,
        theme,
        layout,
        y=_caption(figure, caption, theme, layout),
    )
    written = _save(figure, out, "uncertainty", formats)
    return FigureResult(
        name="uncertainty",
        produced=True,
        path=written,
        inputs={"verdict": verdict, "signal_mm": signal},
    )


# ------------------------------------------------------------- the driver ---


def build_inputs(
    designs: tuple[str, ...] = DEFAULT_DESIGNS,
    *,
    n_steps: int = 41,
    measured_csv: str | Path | None = None,
    torque_csv: str | Path | None = None,
    uncertainty_json: str | Path | None = None,
    material: str = "PLA",
    printer: str = "kobra2_neo",
    target_excursion_deg: float = 22.0,
) -> FigureInputs:
    """Convert and simulate every design once, and load whatever measurements exist.

    This is where the cost is: one beam FEA sweep per design. Doing it here means
    the figures share the result instead of each re-solving it, and means every
    figure in a run is drawn from exactly the same states.
    """
    from cmtool.api import convert
    from cmtool.convert.arc import fit_input_arc
    from cmtool.core.graph import Linkage
    from cmtool.metrics.paths import read_path_csv
    from cmtool.viz.scene import build_scene

    scenes: dict[str, ViewerScene] = {}
    mechanisms: dict[str, CompliantMechanism] = {}
    measured = read_path_csv(measured_csv) if measured_csv else None

    for source in designs:
        linkage = Linkage.from_json(source)
        arc = fit_input_arc(linkage, max_joint_excursion_deg=target_excursion_deg)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mechanism = convert(
                linkage,
                material=material,
                printer=printer,
                input_range_deg=arc.input_range_deg,
            )
            scene = build_scene(mechanism, n_steps=n_steps, measured=measured)
        mechanisms[linkage.name] = mechanism
        scenes[linkage.name] = scene

    readings: list[Any] = []
    if torque_csv:
        from cmtool.metrics.torque import read_measurements

        readings = [r for r in read_measurements(torque_csv) if r.torque_nmm is not None]

    report = None
    if uncertainty_json:
        report = json.loads(Path(uncertainty_json).read_text(encoding="utf-8"))

    provenance = Provenance(notes={"figures": list(scenes)})
    for scene in scenes.values():
        provenance.merge(scene.provenance)
        provenance.config_hash = provenance.config_hash or scene.provenance.config_hash

    return FigureInputs(
        scenes=scenes,
        mechanisms=mechanisms,
        provenance=provenance,
        measured=measured,
        torque_readings=readings,
        uncertainty=report,
        primary=next(iter(scenes)) if scenes else "",
    )


def generate_all(
    out: str | Path,
    inputs: FigureInputs,
    *,
    theme: str = "light",
    formats: tuple[str, ...] = ("png",),
    only: tuple[str, ...] | None = None,
    column: str = "single",
) -> list[FigureResult]:
    """Draw every figure into ``out`` and write a manifest beside them.

    Parameters
    ----------
    only
        Draw just these figures by name. Everything by default.
    column
        ``"single"`` lays each figure out at one journal column, its final
        printed width, which is the default because that is where these end up.
        ``"double"`` gives the full-width version for a talk or a poster.

    Returns
    -------
    list of FigureResult
        Including the ones that were skipped, each with its reason. The manifest
        ``figures.json`` records the same, so a README that is missing an image
        can always be traced to the measurement that has not happened yet.
    """
    directory = Path(out)
    layout_for(column)  # fail early on a bad width rather than mid-run
    builders = {
        "path_overlay": fig_path_overlay,
        "torque_curves": fig_torque_curves,
        "strain": fig_strain,
        "feasibility": fig_feasibility,
        "design_rule": fig_design_rule,
        "conversion_artefact": fig_conversion_artefact,
        "uncertainty": fig_uncertainty,
    }
    wanted = set(only) if only else set(builders) | {"prbm_variant"}
    results: list[FigureResult] = []

    for name, builder in builders.items():
        if name not in wanted:
            continue
        results.append(builder(inputs, directory, theme=theme, formats=formats, column=column))

    if "prbm_variant" in wanted:
        # Deliberately NOT inputs.provenance: this figure's constants are
        # dimensionless, so none of the run's placeholder quantities reach it and
        # stamping their caveat on it would be a false disclaimer.
        results.append(fig_prbm_variant(directory, theme=theme, formats=formats, column=column))

    manifest = {
        "generated": [r.to_dict() for r in results],
        "provenance": inputs.provenance.to_dict(),
        "theme": theme,
        "column": column,
        "figure_width_in": layout_for(column).width_in,
        "formats": list(formats),
        "designs": list(inputs.scenes),
        "measured_path": inputs.measured.source if inputs.measured else None,
        "n_torque_readings": len(inputs.torque_readings),
        "has_uncertainty_report": inputs.uncertainty is not None,
    }
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "figures.json").write_text(
        json.dumps(manifest, indent=2, default=str) + "\n", encoding="utf-8"
    )
    return results
