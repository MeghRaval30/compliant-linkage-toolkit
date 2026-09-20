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


def _apply_style(theme: str) -> None:
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
            "font.size": 9,
            "axes.titlesize": 10.5,
            "axes.titleweight": "semibold",
            "axes.titlelocation": "left",
            "figure.dpi": 150,
            "savefig.bbox": "tight",
        }
    )


def _new_figure(**kwargs: Any) -> Figure:
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

    figure = MplFigure(**kwargs)
    FigureCanvasAgg(figure)
    return figure


def _line_kwargs(key: str, theme: str) -> dict[str, Any]:
    """Colour, width and dash pattern for one model's line."""
    style = SERIES[key]
    kwargs: dict[str, Any] = {
        "color": style.colour(theme),
        "linewidth": style.width,
        "label": style.label,
        "solid_capstyle": "round",
    }
    if style.mpl_dash:
        kwargs["dashes"] = list(style.mpl_dash)
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


#: Caption and footer font sizes, in points.
_CAPTION_PT = 7.5
_FOOTER_PT = 6.5


def _caption(
    figure: Figure,
    lines: list[str],
    theme: str,
    *,
    width: int = 108,
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
    text = "\n".join(_wrap(line, width) for line in lines)
    figure.text(
        0.0,
        top,
        text,
        fontsize=_CAPTION_PT,
        va="top",
        ha="left",
        color=INK[theme]["secondary"],
        transform=figure.transFigure,
    )
    line_fraction = 1.4 * _CAPTION_PT / 72.0 / figure.get_figheight()
    return top - line_fraction * (text.count("\n") + 1) - 0.01


def _footer(
    figure: Figure,
    provenance: Provenance,
    theme: str,
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
        "\n".join(lines),
        fontsize=6.5,
        color=STATUS["critical"] if caveat else ink["muted"],
        va="top",
        ha="left",
        transform=figure.transFigure,
    )


def _direct_label(axes: Axes, x: float, y: float, text: str, colour: str) -> None:
    """Label a curve at its own end, so identity never rests on colour alone."""
    axes.annotate(
        text,
        xy=(x, y),
        xytext=(4, 4),
        textcoords="offset points",
        fontsize=8,
        color=colour,
        fontweight="semibold",
    )


def _save(figure: Figure, out: Path, name: str, formats: tuple[str, ...]) -> Path:
    """Write a figure in every requested format; return the first one's path."""
    out.mkdir(parents=True, exist_ok=True)
    paths = []
    for suffix in formats:
        path = out / f"{name}.{suffix}"
        figure.savefig(path)
        paths.append(path)
    return paths[0]


# ------------------------------------------------------------- figures ------


def report_is_measured(mechanism: CompliantMechanism) -> bool:
    """Whether this conversion used a measured allowable strain."""
    return not any(
        name.endswith("allowable_strain") for name in mechanism.provenance.placeholders_used
    )


def fig_path_overlay(
    inputs: FigureInputs, out: Path, *, theme: str = "light", formats: tuple[str, ...] = ("png",)
) -> FigureResult:
    """Draw the coupler path four ways, and how far apart the four actually are.

    The headline A6 figure, in two panels because one will not do the job. The
    paths agree to within a millimetre over a part 140 mm across, so drawn on top
    of each other they are a single curve -- which is itself the finding, and is
    why the second panel exists. Separation is the quantity the Phase A go/no-go
    is stated against, so it gets an axis of its own rather than an inset.
    """
    scene = inputs.primary_scene
    _apply_style(theme)
    ink = INK[theme]

    figure = _new_figure(figsize=(9.0, 4.0))
    left, right = figure.subplots(1, 2, gridspec_kw={"width_ratios": [1.05, 1.0]})
    present = [k for k in ("rigid", "prbm", "fea", "measured") if k in scene.models]
    missing = [] if "measured" in present else ["measured"]

    # --- left: the path itself, to scale ---------------------------------
    for key in present:
        path = np.asarray(scene.models[key].path_mm, dtype=float)
        left.plot(path[:, 0], path[:, 1], **_line_kwargs(key, theme))
    left.set_aspect("equal", adjustable="box")
    left.set_xlabel("x (mm)")
    left.set_ylabel("y (mm)")
    left.set_title(f"Coupler point {scene.output_name} of {scene.name}")
    left.legend(loc="best", fontsize=8)
    left.text(
        0.02,
        0.02,
        f"all {len(present)} curves, to scale",
        transform=left.transAxes,
        fontsize=7.5,
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
        right.plot(x, y, **_line_kwargs(key, theme))
        _direct_label(right, x[-1], y[-1], SERIES[key].label, SERIES[key].colour(theme))
        drawn_any = True

    if not drawn_any:
        right.text(
            0.5,
            0.5,
            "only one model in this scene, so there is nothing to compare",
            transform=right.transAxes,
            ha="center",
            fontsize=8,
            color=ink["muted"],
        )

    right.set_xlabel("input angle (deg)")
    right.set_ylabel(f"distance from the {reference} path (mm)")
    right.set_title("How far apart the predictions are")
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
    _footer(figure, scene.provenance, theme, y=_caption(figure, summary, theme, width=140))
    written = _save(figure, out, "path_overlay", formats)
    return FigureResult(
        name="path_overlay",
        produced=True,
        path=written,
        missing=missing,
        inputs={"design": scene.name, "states": scene.n_frames},
    )


def fig_torque_curves(
    inputs: FigureInputs, out: Path, *, theme: str = "light", formats: tuple[str, ...] = ("png",)
) -> FigureResult:
    """Input torque against input angle: the measurement that tests stiffness.

    With a prescribed input the coupler path is fixed by geometry, so the path
    cannot discriminate between stiffness models at all. Torque can, and the two
    models differ here by about a third -- far more than a set of kitchen weights
    resolves, which is the whole argument for the dead-weight rig.
    """
    scene = inputs.primary_scene
    _apply_style(theme)

    figure = _new_figure(figsize=(6.4, 3.8))
    axes = figure.subplots()
    angles = np.asarray(scene.input_angles_deg, dtype=float)
    drawn = []
    for key in ("prbm", "fea"):
        series = scene.models.get(key)
        if series is None or series.torque_nmm is None:
            continue
        torque = np.abs(np.asarray(series.torque_nmm, dtype=float))
        axes.plot(angles, torque, **_line_kwargs(key, theme))
        _direct_label(axes, angles[-1], torque[-1], SERIES[key].label, SERIES[key].colour(theme))
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
                s=34,
                facecolors="none",
                edgecolors=SERIES["measured"].colour(theme),
                linewidths=1.4,
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
    axes.set_title(f"Torque to hold {scene.name} at each input angle")
    axes.legend(loc="lower right", fontsize=8)

    figure.tight_layout()
    _footer(figure, scene.provenance, theme, y=_caption(figure, caption, theme))
    path = _save(figure, out, "torque_curves", formats)
    return FigureResult(
        name="torque_curves",
        produced=True,
        path=path,
        missing=missing,
        inputs={"design": scene.name, "n_readings": len(inputs.torque_readings)},
    )


def fig_strain(
    inputs: FigureInputs, out: Path, *, theme: str = "light", formats: tuple[str, ...] = ("png",)
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

    _apply_style(theme)
    sweep = np.asarray(scene.input_sweep_deg, dtype=float)
    joints = scene.joints
    allowable = scene.allowable_strain * 100.0

    figure = _new_figure(figsize=(2.1 * len(joints) + 0.8, 3.4))
    panels = figure.subplots(1, len(joints), sharey=True)
    axes_list = list(np.atleast_1d(panels))
    peaks: dict[str, float] = {}

    for axes, joint in zip(axes_list, joints, strict=True):
        values = (
            np.asarray([max(frame.flexure_strain[joint]) for frame in fea.frames], dtype=float)
            * 100.0
        )
        peaks[joint] = float(values.max())
        axes.plot(sweep, values, color=SERIES["fea"].colour(theme), linewidth=2.0)
        axes.axhline(allowable, color=STATUS["critical"], linewidth=1.3, dashes=[5, 3])
        axes.axhspan(allowable, allowable * 2.0, color=STATUS["critical"], alpha=0.07)
        axes.set_title(f"joint {joint}", fontsize=9.5)
        axes.set_xlabel("deg")
        axes.set_ylim(0, allowable * 1.25)

    axes_list[0].set_ylabel("peak surface strain (%)")
    axes_list[0].text(
        sweep[0],
        allowable,
        " allowable",
        fontsize=7.5,
        va="bottom",
        color=STATUS["critical"],
    )
    figure.suptitle(
        f"Flexure strain through the arc, {scene.name}", fontsize=10.5, x=0.01, ha="left"
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
    _footer(figure, scene.provenance, theme, y=_caption(figure, caption, theme, width=118))
    written = _save(figure, out, "strain", formats)
    return FigureResult(
        name="strain",
        produced=True,
        path=written,
        missing=["measured_allowable_strain"] if scene.allowable_strain_is_placeholder else [],
        inputs={"design": scene.name, "allowable_strain": scene.allowable_strain, "peaks": peaks},
    )


def fig_feasibility(
    inputs: FigureInputs, out: Path, *, theme: str = "light", formats: tuple[str, ...] = ("png",)
) -> FigureResult:
    """Per joint: the flexure strain demands, the flexure that fits, and the ratio.

    The two hard limits side by side, in the same units, for every joint of every
    pilot design. The joint whose bars are closest together is the one limiting
    the design -- the number that says which joint to fix.
    """
    _apply_style(theme)
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

    figure = _new_figure(figsize=(7.0, 0.32 * len(labels) + 1.9))
    axes = figure.subplots()
    positions = np.arange(len(labels), dtype=float)
    axes.barh(
        positions + 0.19,
        geometric_mm,
        height=0.34,
        color=SERIES["fea"].colour(theme),
        label="longest flexure that fits (geometry)",
    )
    axes.barh(
        positions - 0.19,
        strain_mm,
        height=0.34,
        color=SERIES["rigid"].colour(theme),
        label="shortest flexure that survives the bend (strain)",
    )
    for position, value, ratio, is_binding in zip(
        positions, geometric_mm, utilisation, binding, strict=True
    ):
        axes.text(
            value + 0.25,
            position,
            f"{ratio:.2f}" + ("  <-- binding" if is_binding else ""),
            va="center",
            fontsize=7.5,
            color=STATUS["critical"] if ratio >= 1.0 else ink["secondary"],
            fontweight="semibold" if is_binding else "normal",
        )

    axes.set_yticks(positions)
    axes.set_yticklabels(labels, fontsize=7.5)
    axes.invert_yaxis()
    axes.set_xlabel("flexure length (mm)")
    axes.set_title("Both hard limits, per joint, with utilisation = strain / geometry")
    axes.grid(axis="y", visible=False)
    axes.legend(loc="lower left", fontsize=8, bbox_to_anchor=(0.0, 1.02), ncol=2)
    _footer(
        figure,
        inputs.provenance,
        theme,
        "utilisation above 1.0 cannot be built; PRBM model validity is metadata and "
        "is deliberately not a limit here",
    )
    path = _save(figure, out, "feasibility", formats)
    return FigureResult(
        name="feasibility",
        produced=True,
        path=path,
        inputs={"designs": list(mechanisms)},
    )


def fig_design_rule(
    inputs: FigureInputs, out: Path, *, theme: str = "light", formats: tuple[str, ...] = ("png",)
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

    _apply_style(theme)
    ink = INK[theme]
    mechanism = inputs.primary_mechanism
    report = mechanism.feasibility
    allowable = report.allowable_strain
    fraction = report.max_length_fraction
    safety = report.strain_safety_factor

    figure = _new_figure(figsize=(6.6, 4.2))
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
        axes.plot(lengths, excursion, color=colour, linewidth=2.0, label=f"t = {thickness} mm")
        # Label where the curve leaves the frame, not at its last computed point,
        # which for the thin flexures is far above the top of the axes.
        inside = np.nonzero(excursion <= top * 0.93)[0]
        if inside.size:
            last = int(inside[-1])
            _direct_label(axes, lengths[last], excursion[last], f"{thickness} mm", colour)

    marked: list[dict[str, Any]] = []
    for offset, (name, converted) in enumerate(inputs.mechanisms.items()):
        worst = converted.feasibility.binding_joint
        sized = converted.sizing[worst]
        axes.scatter(
            [sized.shortest_adjacent_link_mm],
            [sized.excursion_deg],
            s=44,
            marker="D",
            facecolors=ink["surface"],
            edgecolors=SERIES["measured"].colour(theme),
            linewidths=1.4,
            zorder=5,
            label="pilot design, at its binding joint" if offset == 0 else None,
        )
        axes.annotate(
            f"{name.replace('fb_02_', '')} ({worst})",
            xy=(sized.shortest_adjacent_link_mm, sized.excursion_deg),
            # The pilots cluster: the arc fit drives every design to the same
            # excursion target, so their markers land on top of one another.
            # Fan the labels out and lead each one back to its own marker.
            xytext=(14, 26 - 18 * offset),
            textcoords="offset points",
            fontsize=7.5,
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
    axes.set_title("The bound that governs: usable rotation is set by the neighbouring link")
    axes.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.14),
        fontsize=8,
        ncol=5,
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
    _footer(figure, mechanism.provenance, theme, y=_caption(figure, caption, theme, width=112))
    written = _save(figure, out, "design_rule", formats)
    return FigureResult(
        name="design_rule",
        produced=True,
        path=written,
        missing=[] if report_is_measured(mechanism) else ["measured_allowable_strain"],
        inputs={"allowable_strain": allowable, "designs": marked},
    )


def fig_conversion_artefact(
    inputs: FigureInputs, out: Path, *, theme: str = "light", formats: tuple[str, ...] = ("png",)
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

    _apply_style(theme)
    figure = _new_figure(figsize=(9.0, 4.0))
    left, right = figure.subplots(1, 2, gridspec_kw={"width_ratios": [1.05, 1.0]})

    matched_colour = SERIES["fea"].colour(theme)
    moved_colour = SERIES["rigid"].colour(theme)
    left.plot(
        matched[:, 0],
        matched[:, 1],
        color=matched_colour,
        linewidth=2.2,
        label="pivot-matched (the default)",
    )
    left.plot(
        moved[:, 0],
        moved[:, 1],
        color=moved_colour,
        linewidth=2.0,
        dashes=[7, 3],
        label="unmatched placement",
    )
    left.set_aspect("equal", adjustable="box")
    left.set_xlabel("x (mm)")
    left.set_ylabel("y (mm)")
    left.set_title(f"Coupler path of {scene.name}, both placements")
    left.legend(loc="best", fontsize=8)

    right.plot(angles, separation, color=moved_colour, linewidth=2.0)
    right.fill_between(angles, 0.0, separation, color=moved_colour, alpha=0.12)
    right.set_xlabel("input angle (deg)")
    right.set_ylabel("path shift from placement alone (mm)")
    right.set_title("How much the placement rule moves it")
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
    _footer(figure, mechanism.provenance, theme, y=_caption(figure, caption, theme, width=140))
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

    _apply_style(theme)
    ink = INK[theme]
    ratios = load_ratios or DEFAULT_LOAD_RATIOS
    fitted = sweep(ratios)

    figure = _new_figure(figsize=(7.6, 3.8))
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
            linewidth=2.0,
            marker="o",
            markersize=4,
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
        axes.set_title(label)
        axes.annotate(
            "a flexure joint",
            xy=(JOINT_LOAD_RATIO, axes.get_ylim()[1]),
            xytext=(4, -10),
            textcoords="offset points",
            fontsize=7.5,
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
        ncol=3,
        fontsize=8,
    )
    figure.suptitle(
        "A flexure joint is moment-dominated, so the end-moment variant is the one in use",
        fontsize=10.5,
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
        y=_caption(figure, caption, theme, width=116, top=-0.12),
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
    inputs: FigureInputs, out: Path, *, theme: str = "light", formats: tuple[str, ...] = ("png",)
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

    _apply_style(theme)
    ink = INK[theme]
    estimates = report.get("estimates", [])
    signal = float(report.get("signal_mm", 0.46))

    figure = _new_figure(figsize=(6.4, 3.4))
    axes = figure.subplots()
    names = [e["method"] for e in estimates]
    positions = np.arange(len(names), dtype=float)
    axes.barh(
        positions,
        [e["sigma_mm"] for e in estimates],
        height=0.46,
        color=SERIES["fea"].colour(theme),
        label="sigma",
    )
    axes.scatter(
        [e["max_deviation_mm"] for e in estimates],
        positions,
        marker="|",
        s=220,
        linewidths=2,
        color=SERIES["measured"].colour(theme),
        label="worst deviation",
        zorder=5,
    )

    from matplotlib.transforms import blended_transform_factory

    # Threshold labels run vertically beside their own line: laid out flat they
    # collide with each other and with the title, because 5:1 and 3:1 of the same
    # signal are close together by construction.
    along = blended_transform_factory(axes.transData, axes.transAxes)
    for label, divisor, colour in (
        ("signal to resolve", 1.0, ink["secondary"]),
        ("3:1 marginal", 3.0, STATUS["warning"]),
        ("5:1 go", 5.0, STATUS["good"]),
    ):
        axes.axvline(signal / divisor, color=colour, linewidth=1.4, dashes=[5, 3])
        axes.text(
            signal / divisor,
            0.99,
            f"{label}  {signal / divisor:.3f} mm ",
            transform=along,
            rotation=90,
            fontsize=7.5,
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
    axes.set_title(f"Tracking uncertainty against the {signal:.2f} mm PRBM-vs-FEA signal")
    axes.grid(axis="y", visible=False)
    axes.legend(loc="lower right", fontsize=8)

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
        fontsize=10.5,
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
        y=_caption(figure, caption, theme, width=118),
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
) -> list[FigureResult]:
    """Draw every figure into ``out`` and write a manifest beside them.

    Parameters
    ----------
    only
        Draw just these figures by name. Everything by default.

    Returns
    -------
    list of FigureResult
        Including the ones that were skipped, each with its reason. The manifest
        ``figures.json`` records the same, so a README that is missing an image
        can always be traced to the measurement that has not happened yet.
    """
    directory = Path(out)
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
        results.append(builder(inputs, directory, theme=theme, formats=formats))

    if "prbm_variant" in wanted:
        # Deliberately NOT inputs.provenance: this figure's constants are
        # dimensionless, so none of the run's placeholder quantities reach it and
        # stamping their caveat on it would be a false disclaimer.
        results.append(fig_prbm_variant(directory, theme=theme, formats=formats))

    manifest = {
        "generated": [r.to_dict() for r in results],
        "provenance": inputs.provenance.to_dict(),
        "theme": theme,
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
