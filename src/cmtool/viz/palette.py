"""Colours and line styles, defined once for the viewer and the figures.

Both outputs draw the same four series, so they share one definition: a reader
who sees orange in the HTML viewer and orange in the README figure is looking at
the same model.

Why these colours
-----------------
Each model keeps **one hue everywhere** -- its coupler path, its ghost outline in
the viewer, its torque curve. The measured series is deliberately *not* a hue: it
is drawn in primary ink, because it is the reference the others are judged
against rather than a fourth opinion.

The three hues were chosen by running the palette validator over candidate
triples rather than by eye, on the all-pairs test (the paths overlap in one
plane, so every pair has to separate, not just adjacent ones). Orange / aqua /
violet passes every gate in both light and dark: worst all-pairs colour-vision
separation dE 9.2 light and 9.4 dark against a target of 8, worst normal-vision
separation 27.6 and 24.6 against a floor of 15.

Aqua falls below 3:1 contrast on the light surface, so the relief rule applies:
**every series carries a direct label or a legend entry and a distinct dash
pattern**, and identity is never left to colour alone.

Strain is a different job -- continuous magnitude, not identity -- so it gets a
sequential one-hue blue ramp, light to dark, with the anchor flipped in dark mode
so "near zero" recedes toward the surface in both. Blue is kept out of the series
set so the ramp can never be mistaken for a model. Anything over the allowable
strain leaves the ramp entirely for the reserved ``critical`` status colour, and
is labelled in words.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SeriesStyle:
    """How one model is drawn, in both themes."""

    key: str
    label: str
    light: str
    dark: str
    #: SVG ``stroke-dasharray``; empty means solid.
    dash: str
    #: Matplotlib linestyle for the same series.
    mpl_dash: tuple[float, ...] | None
    width: float = 2.0

    def colour(self, theme: str = "light") -> str:
        """Return the hex colour for ``theme``."""
        return self.dark if theme == "dark" else self.light


#: Drawing style per model. Order is the order they are listed in legends.
SERIES: dict[str, SeriesStyle] = {
    "rigid": SeriesStyle(
        key="rigid",
        label="rigid",
        light="#eb6834",
        dark="#d95926",
        dash="7 3",
        mpl_dash=(7.0, 3.0),
    ),
    "prbm": SeriesStyle(
        key="prbm",
        label="PRBM",
        light="#1baf7a",
        dark="#199e70",
        dash="9 3 2 3",
        mpl_dash=(9.0, 3.0, 2.0, 3.0),
    ),
    "fea": SeriesStyle(
        key="fea",
        label="beam FEA",
        light="#4a3aa7",
        dark="#9085e9",
        dash="",
        mpl_dash=None,
    ),
    "measured": SeriesStyle(
        key="measured",
        label="measured",
        light="#0b0b0b",
        dark="#ffffff",
        dash="",
        mpl_dash=None,
        width=2.6,
    ),
}

#: Sequential blue ramp for strain, lightest (near zero) first.
STRAIN_RAMP_LIGHT: tuple[str, ...] = ("#cde2fb", "#9ec5f4", "#5598e7", "#2a78d6", "#1c5cab")

#: The same ramp re-anchored for a dark surface: near zero recedes into the dark.
STRAIN_RAMP_DARK: tuple[str, ...] = ("#0d366b", "#184f95", "#256abf", "#3987e5", "#86b6ef")

#: Ordinal ramp for a small set of *ordered* categories -- flexure thickness, a
#: tier, a stage. Unlike the sequential ramp it never approaches the surface: its
#: lightest step still clears 2:1 contrast, because an ordinal mark has to be
#: visible in its own right rather than reading as "near zero".
ORDINAL_RAMP_LIGHT: tuple[str, ...] = ("#86b6ef", "#3987e5", "#256abf", "#184f95")

#: The same ramp for a dark surface: lightness increases with magnitude there.
ORDINAL_RAMP_DARK: tuple[str, ...] = ("#184f95", "#256abf", "#3987e5", "#86b6ef")

#: Reserved status colours. Never reused for a series.
STATUS: dict[str, str] = {
    "good": "#0ca30c",
    "warning": "#fab219",
    "serious": "#ec835a",
    "critical": "#d03b3b",
}

#: Chart chrome, per theme.
INK: dict[str, dict[str, str]] = {
    "light": {
        "surface": "#fcfcfb",
        "plane": "#f9f9f7",
        "primary": "#0b0b0b",
        "secondary": "#52514e",
        "muted": "#898781",
        "grid": "#e1e0d9",
        "axis": "#c3c2b7",
        "border": "rgba(11,11,11,0.10)",
    },
    "dark": {
        "surface": "#1a1a19",
        "plane": "#0d0d0d",
        "primary": "#ffffff",
        "secondary": "#c3c2b7",
        "muted": "#898781",
        "grid": "#2c2c2a",
        "axis": "#383835",
        "border": "rgba(255,255,255,0.10)",
    },
}

#: UI typeface. One family everywhere, including large figures.
FONT_STACK = 'system-ui, -apple-system, "Segoe UI", sans-serif'


def strain_colour(utilisation: float, theme: str = "light") -> str:
    """Return the ramp colour for a strain ``utilisation`` (strain / allowable).

    Parameters
    ----------
    utilisation
        Peak strain divided by the allowable strain. ``1.0`` is the limit.
    theme
        ``"light"`` or ``"dark"``.

    Returns
    -------
    str
        A hex colour from the sequential ramp, or the reserved ``critical``
        status colour at or above 1.0 -- which is a different statement, not a
        darker shade of the same one.
    """
    if utilisation >= 1.0:
        return STATUS["critical"]
    ramp = STRAIN_RAMP_DARK if theme == "dark" else STRAIN_RAMP_LIGHT
    clamped = max(0.0, float(utilisation))
    position = clamped * (len(ramp) - 1)
    lower = min(int(position), len(ramp) - 2)
    return _mix(ramp[lower], ramp[lower + 1], position - lower)


def _mix(first: str, second: str, fraction: float) -> str:
    """Blend two hex colours in sRGB. Good enough between adjacent ramp steps."""
    a = _rgb(first)
    b = _rgb(second)
    blended = [round(x + (y - x) * fraction) for x, y in zip(a, b, strict=True)]
    return "#" + "".join(f"{value:02x}" for value in blended)


def _rgb(value: str) -> tuple[int, int, int]:
    text = value.lstrip("#")
    return int(text[0:2], 16), int(text[2:4], 16), int(text[4:6], 16)
