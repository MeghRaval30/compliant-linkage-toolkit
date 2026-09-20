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

Three gates had to pass at once, and each was checked by computation rather than
by eye:

1. **Colour vision**, all-pairs -- the curves overlap in one plane, so every pair
   has to separate, not just adjacent ones. Magenta / green / violet gives worst
   all-pairs dE 17.6 light and 13.0 dark against a target of 8, and worst
   normal-vision separation 33.9 and 19.7 against a floor of 15.
2. **Greyscale**, because these figures go to a journal that may print in one
   ink. Reduced to luminance the three sit at 156, 112 and 77 out of 255, with
   the measured series at 19: four levels a reader can separate on paper. This is
   what ruled out the first choice -- orange and aqua are a fine colour pair and
   land ten grey levels apart, which is no pair at all in print.
3. **Contrast against the surface.** Magenta falls below 3:1 on the light
   surface, so the relief rule applies.

Relief and the journal requirement turn out to be the same thing: **every series
carries a distinct dash pattern and a distinct marker as well as its hue**, plus
a direct label or a legend entry. Identity never rests on colour alone, so the
figures read in one ink, on a bad projector, and to a colour-blind reader. Filled
marks have no dash to carry, so they take a hatch instead.

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
    #: Matplotlib dash pattern for the same series; ``None`` means solid.
    mpl_dash: tuple[float, ...] | None
    #: Matplotlib marker: the second redundant channel after the dash pattern, so
    #: a greyscale print and a colour-blind reader both still separate the series
    #: -- and so do two curves that happen to lie on top of each other.
    marker: str
    width: float = 2.0

    def colour(self, theme: str = "light") -> str:
        """Return the hex colour for ``theme``."""
        return self.dark if theme == "dark" else self.light

    @property
    def grey(self) -> int:
        """Luminance of the light-theme colour, 0-255, as a greyscale print shows it."""
        return round(relative_luminance(self.light) ** (1.0 / 2.2) * 255)


#: Drawing style per model. Order is the order they are listed in legends.
SERIES: dict[str, SeriesStyle] = {
    "rigid": SeriesStyle(
        key="rigid",
        label="rigid",
        light="#e87ba4",
        dark="#d55181",
        dash="7 3",
        mpl_dash=(7.0, 3.0),
        marker="o",
    ),
    "prbm": SeriesStyle(
        key="prbm",
        label="PRBM",
        light="#008300",
        dark="#008300",
        dash="9 3 2 3",
        mpl_dash=(9.0, 3.0, 2.0, 3.0),
        marker="s",
    ),
    "fea": SeriesStyle(
        key="fea",
        label="beam FEA",
        light="#4a3aa7",
        dark="#9085e9",
        dash="",
        mpl_dash=None,
        marker="^",
    ),
    "measured": SeriesStyle(
        key="measured",
        label="measured",
        light="#0b0b0b",
        dark="#ffffff",
        dash="2 2",
        mpl_dash=(2.0, 2.0),
        marker="D",
        width=2.4,
    ),
}

#: Hatch per series, for marks that are filled rather than stroked. A fill has no
#: dash pattern to carry, so texture is what makes a bar chart survive one ink.
#: Angles are 45 degrees and its mirror only, never a third direction.
HATCH: dict[str, str] = {
    "rigid": "///",
    "prbm": "\\" * 3,
    "fea": "",
    "measured": "xxx",
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


def relative_luminance(colour: str) -> float:
    """Return the WCAG relative luminance of a hex colour, 0 to 1.

    This is what a greyscale print reduces the colour to, so it is the number the
    palette's journal requirement is checked against.
    """
    channels = []
    for value in _rgb(colour):
        fraction = value / 255.0
        channels.append(
            fraction / 12.92 if fraction <= 0.04045 else ((fraction + 0.055) / 1.055) ** 2.4
        )
    red, green, blue = channels
    return 0.2126 * red + 0.7152 * green + 0.0722 * blue


def greyscale_contrast(first: str, second: str) -> float:
    """Contrast ratio between two colours once both are reduced to grey."""
    a, b = relative_luminance(first), relative_luminance(second)
    return (max(a, b) + 0.05) / (min(a, b) + 0.05)
