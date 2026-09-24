"""A whole mechanism viewer in one HTML file.

The point of this renderer is that the output has **no dependencies at all**: no
CDN, no fonts to fetch, no Python on the far end. The scene is serialised into a
``<script type="application/json">`` block and drawn with plain SVG, so the file
can be mailed to a supervisor, opened on a phone with no signal, or dropped into
a repository and viewed straight from disk.

A test asserts that: the generated document is scanned for any external
reference, and finding one fails the build. A viewer that silently needs the
network is not a viewer you can take to a lab bench.

What it shows
-------------
The beam FEA's deformed part, flexures coloured by strain, with the rigid and
PRBM outlines over it as ghosts at the same input angle, and all the coupler
paths underneath. A slider steps through the precomputed states -- nothing is
solved in the browser, so every number on screen came from the same run that
wrote the file.

Honesty rules this renderer follows
-----------------------------------
* When any input was a placeholder, a banner says so, it is not dismissible, and
  it names the quantities.
* The strain ramp is normalised against the allowable strain. While that is a
  placeholder the legend says the scale is relative and means nothing physical.
* A missing measured path is a missing series, never a flat line at zero.
* Flexures are 0.6 mm beside 8 mm links, so by default they are drawn at a
  minimum on-screen width to keep the strain colour legible. The legend states
  the true thickness and a toggle turns the exaggeration off.
"""

from __future__ import annotations

import json
from pathlib import Path

from cmtool.viz.palette import FONT_STACK, INK, SERIES, STATUS, STRAIN_RAMP_DARK
from cmtool.viz.palette import STRAIN_RAMP_LIGHT as RAMP_LIGHT
from cmtool.viz.scene import ViewerScene

#: Smallest on-screen flexure width, in scene millimetres, when exaggeration is on.
DEFAULT_MIN_FLEXURE_MM = 1.6


def render_html(
    scene: ViewerScene,
    *,
    title: str | None = None,
    min_flexure_mm: float = DEFAULT_MIN_FLEXURE_MM,
) -> str:
    """Return a complete, self-contained HTML document for ``scene``.

    Parameters
    ----------
    scene
        Built by :func:`cmtool.viz.scene.build_scene`.
    title
        Document title. Defaults to the mechanism's name.
    min_flexure_mm
        Minimum drawn flexure width in scene units, so a 0.6 mm strip beside an
        8 mm link still shows its strain colour. Set to ``0`` for true scale.

    Returns
    -------
    str
        The document. Nothing in it is fetched from anywhere.
    """
    payload = {
        "scene": scene.to_dict(),
        "style": {
            "series": {
                key: {
                    "label": style.label,
                    "light": style.light,
                    "dark": style.dark,
                    "dash": style.dash,
                    "width": style.width,
                }
                for key, style in SERIES.items()
            },
            "rampLight": list(RAMP_LIGHT),
            "rampDark": list(STRAIN_RAMP_DARK),
            "critical": STATUS["critical"],
            "minFlexureMm": float(min_flexure_mm),
        },
    }
    # </script> inside the payload would close the block early; < cannot.
    data = json.dumps(payload, allow_nan=False).replace("<", "\\u003c")

    return (
        _TEMPLATE.replace("__SCENE_DRAW__", _scene_draw_js())
        .replace("__TITLE__", _escape(title or f"cmtool viewer - {scene.name}"))
        .replace("__NAME__", _escape(scene.name))
        .replace("__FONT__", FONT_STACK)
        .replace("__INK_LIGHT__", _css_vars(INK["light"]))
        .replace("__INK_DARK__", _css_vars(INK["dark"]))
        .replace("__DATA__", data)
    )


def _scene_draw_js() -> str:
    """Return the shared drawing module's source, for inlining.

    The standalone viewer must work with no network at all, so the module is
    baked into the file rather than linked. It is still the *same file* the
    local UI serves: one implementation, two delivery mechanisms.
    """
    from importlib import resources

    module = resources.files("cmtool.viz") / "static" / "scene_draw.js"
    return module.read_text(encoding="utf-8")


def write_html(
    scene: ViewerScene,
    path: str | Path,
    *,
    title: str | None = None,
    min_flexure_mm: float = DEFAULT_MIN_FLEXURE_MM,
) -> Path:
    """Render ``scene`` and write it to ``path``, creating parent directories."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render_html(scene, title=title, min_flexure_mm=min_flexure_mm), encoding="utf-8")
    return out


#: XML namespace URIs, which look like URLs but are identifiers and are never
#: fetched. ``createElementNS`` needs the SVG one by name.
_NAMESPACE_URIS: tuple[str, ...] = (
    "http://www.w3.org/2000/svg",
    "http://www.w3.org/1999/xhtml",
    "http://www.w3.org/1999/xlink",
)


def external_references(document: str) -> list[str]:
    """Return any external resource the document would fetch.

    XML namespace URIs are excluded: they are identifiers a parser compares
    against, not addresses a browser resolves.

    Used by the test that keeps this renderer honest, and worth calling from any
    script that generates a viewer for someone else to open offline.
    """
    found: list[str] = []
    lowered = document.lower()
    for marker in ("http://", "https://", 'src="//', "src='//", "@import", "url(http"):
        start = 0
        while (index := lowered.find(marker, start)) != -1:
            excerpt = document[index : index + 80]
            if not any(excerpt.startswith(uri) for uri in _NAMESPACE_URIS):
                found.append(excerpt)
            start = index + 1
    return found


def _escape(text: str) -> str:
    return (
        text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")
    )


def _css_vars(tokens: dict[str, str]) -> str:
    return "\n      ".join(f"--{name}: {value};" for name, value in tokens.items())


_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TITLE__</title>
<style>
  :root {
      color-scheme: light dark;
      __INK_LIGHT__
      --critical: #d03b3b;
      --font: __FONT__;
  }
  @media (prefers-color-scheme: dark) {
    :root:not([data-theme="light"]) {
      __INK_DARK__
    }
  }
  * { box-sizing: border-box; }
  body {
    margin: 0;
    padding: 16px;
    background: var(--plane);
    color: var(--primary);
    font-family: var(--font);
    font-size: 15px;
    line-height: 1.45;
    overflow-x: hidden;
  }
  .wrap { max-width: 860px; margin: 0 auto; }
  h1 { font-size: 1.15rem; margin: 0 0 2px; }
  .sub { color: var(--secondary); font-size: 0.85rem; margin: 0 0 14px; }
  .card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 14px;
    margin-bottom: 14px;
  }
  .banner {
    border-left: 4px solid var(--critical);
    background: var(--surface);
    border-radius: 8px;
    padding: 10px 12px;
    margin-bottom: 14px;
    font-size: 0.86rem;
  }
  .banner strong { color: var(--critical); }
  .banner code { font-size: 0.8rem; color: var(--secondary); word-break: break-all; }
  svg { display: block; width: 100%; height: auto; touch-action: manipulation; }
  h2.chart-title { font-size: 0.95rem; margin: 0 0 2px; font-weight: 600; }
  .controls { display: flex; align-items: center; gap: 12px; flex-wrap: wrap; }
  input[type=range] {
    flex: 1 1 200px; min-width: 160px; height: 44px; accent-color: var(--secondary);
  }
  button {
    font: inherit; font-size: 0.9rem; min-height: 44px; min-width: 64px;
    padding: 0 14px; border-radius: 8px; cursor: pointer;
    border: 1px solid var(--border); background: var(--plane); color: var(--primary);
  }
  button[aria-pressed="true"] { border-color: var(--secondary); }
  .angle { font-variant-numeric: tabular-nums; font-size: 0.95rem; min-width: 128px; }
  .angle b { font-size: 1.25rem; }
  .toggles { display: flex; gap: 8px; flex-wrap: wrap; margin-top: 10px; }
  .readouts {
    display: grid; gap: 10px; margin-top: 12px;
    grid-template-columns: repeat(auto-fit, minmax(130px, 1fr));
  }
  .stat { border-top: 2px solid var(--axis); padding-top: 6px; }
  .stat .k { font-size: 0.72rem; color: var(--muted); text-transform: uppercase;
             letter-spacing: 0.03em; }
  .stat .v { font-size: 1.15rem; font-variant-numeric: tabular-nums; }
  .stat .u { font-size: 0.78rem; color: var(--secondary); }
  .legend { display: flex; gap: 14px; flex-wrap: wrap; margin-top: 10px;
            font-size: 0.82rem; color: var(--secondary); }
  .legend span { display: inline-flex; align-items: center; gap: 6px; }
  .swatch { width: 22px; height: 0; border-top-width: 3px; border-top-style: solid; }
  .note { font-size: 0.8rem; color: var(--secondary); margin-top: 8px; }
  table { border-collapse: collapse; width: 100%; font-size: 0.8rem;
          font-variant-numeric: tabular-nums; }
  th, td { text-align: right; padding: 3px 6px; border-bottom: 1px solid var(--grid); }
  th:first-child, td:first-child { text-align: left; }
  th { color: var(--muted); font-weight: 600; }
  details summary { cursor: pointer; min-height: 44px; display: flex; align-items: center;
                    font-size: 0.9rem; }
  .scroll { overflow-x: auto; }
  @media (max-width: 460px) {
    body { padding: 12px; }
    button { min-width: 0; padding: 0 10px; font-size: 0.85rem; }
    .angle { min-width: 104px; }
  }
  footer { font-size: 0.75rem; color: var(--muted); margin-top: 6px; word-break: break-all; }
  footer code { font-size: 0.72rem; }
</style>
</head>
<body>
<div class="wrap">
  <h1>__NAME__</h1>
  <p class="sub" id="subtitle"></p>
  <div id="banner"></div>

  <div class="card">
    <svg id="view" role="img" aria-label="compliant mechanism at the selected input angle"></svg>
    <div class="legend" id="legend"></div>
    <p class="note" id="scale-note"></p>
  </div>

  <div class="card" id="deviation-card">
    <h2 class="chart-title" id="deviation-title"></h2>
    <svg id="deviation" role="img"
         aria-label="how far each model's coupler path sits from the reference path"></svg>
    <p class="note" id="deviation-note"></p>
  </div>

  <div class="card">
    <div class="controls">
      <button id="play" aria-pressed="false">Play</button>
      <input id="slider" type="range" min="0" value="0" step="1"
             aria-label="input angle step">
      <div class="angle">
        <b id="angle">-</b> deg<br>
        <span class="u" id="sweep"></span>
      </div>
    </div>
    <div class="toggles" id="toggles"></div>
    <div class="readouts" id="readouts"></div>
  </div>

  <div class="card">
    <details>
      <summary>Numbers behind the picture (table view)</summary>
      <div class="scroll"><table id="table"></table></div>
    </details>
    <details>
      <summary>Path disagreement between the models</summary>
      <div class="scroll"><table id="gaps"></table></div>
    </details>
    <footer id="prov"></footer>
  </div>
</div>

<script type="application/json" id="payload">__DATA__</script>
<script>__SCENE_DRAW__</script>
<script>
(function () {
  "use strict";
  const PAYLOAD = JSON.parse(document.getElementById("payload").textContent);
  const S = PAYLOAD.scene;
  const ST = PAYLOAD.style;
  const SVGNS = "http://www.w3.org/2000/svg";

  const ink = (name) => getComputedStyle(document.documentElement)
    .getPropertyValue("--" + name).trim();

  // Read the theme off the rendered surface rather than the media query, so the
  // SVG colours follow whatever CSS actually applied.
  const dark = () => {
    const c = rgb(ink("surface") || "#ffffff");
    return (c[0] + c[1] + c[2]) / 3 < 128;
  };
  const colour = (key) => {
    const s = ST.series[key];
    return dark() ? s.dark : s.light;
  };
  const ramp = () => (dark() ? ST.rampDark : ST.rampLight);

  function strainColour(utilisation) {
    if (!isFinite(utilisation)) return ST.critical;
    if (utilisation >= 1) return ST.critical;
    const r = D.ramp(ST);
    const pos = Math.max(0, utilisation) * (r.length - 1);
    const lo = Math.min(Math.floor(pos), r.length - 2);
    return mix(r[lo], r[lo + 1], pos - lo);
  }
  function mix(a, b, f) {
    const pa = rgb(a), pb = rgb(b);
    return "#" + [0, 1, 2]
      .map((i) => Math.round(pa[i] + (pb[i] - pa[i]) * f).toString(16).padStart(2, "0"))
      .join("");
  }
  function rgb(hex) {
    const h = hex.replace("#", "");
    return [0, 2, 4].map((i) => parseInt(h.slice(i, i + 2), 16));
  }

  // --- drawing ---------------------------------------------------------
  // Every mark on this page is drawn by CmtoolScene, the same module the local
  // UI loads from the server. It is inlined above rather than fetched, so this
  // file still needs nothing from the network -- but it is the same source, so
  // the two renderers cannot drift apart.
  const D = window.CmtoolScene;
  const inkOf = (name) =>
    getComputedStyle(document.documentElement).getPropertyValue("--" + name).trim();

  const view = document.getElementById("view");
  const devSvg = document.getElementById("deviation");
  const has = (k) => Object.prototype.hasOwnProperty.call(S.models, k);

  const show = { rigid: has("rigid"), prbm: has("prbm"), fea: has("fea"),
                 measured: has("measured"), paths: true, trueScale: false };

  let devSeries = [];

  function draw(i) {
    D.drawMechanism(view, S, ST, i, { show: show });
  }

  function drawDeviation(i) {
    const card = document.getElementById("deviation-card");
    devSeries = D.drawDeviation(devSvg, S, ST, i, {
      width: devSvg.clientWidth || 520,
    });
    card.style.display = devSeries.length ? "" : "none";
  }


  // --- readouts --------------------------------------------------------
  const fmt = (v, d) => (v === null || v === undefined || !isFinite(v) ? "-" : v.toFixed(d));

  function worstStrain(i) {
    const frame = frameOf("fea", i);
    if (!frame) return null;
    let best = { joint: null, value: -1 };
    for (const joint in frame.strain) {
      const peak = Math.max.apply(null, frame.strain[joint]);
      if (peak > best.value) best = { joint: joint, value: peak };
    }
    return best;
  }

  function stat(key, value, unit) {
    return '<div class="stat"><div class="k">' + key + '</div><div class="v">' +
           value + '</div><div class="u">' + (unit || "") + '</div></div>';
  }

  function readouts(i) {
    const parts = [];
    for (const key of ["prbm", "fea"]) {
      const m = S.models[key];
      if (!m || !m.torque_nmm) continue;
      parts.push(stat(ST.series[key].label + " torque", fmt(m.torque_nmm[i], 1), "N&middot;mm"));
    }
    const worst = D.worstStrain(S, i);
    if (worst && worst.joint) {
      const util = worst.utilisation;
      parts.push(stat("peak strain", (worst.value * 100).toFixed(3) + "%",
                      "joint " + worst.joint));
      parts.push(stat("of allowable", (util * 100).toFixed(0) + "%",
                      S.allowable_strain_is_placeholder ? "against a placeholder" : "measured"));
    }
    document.getElementById("readouts").innerHTML = parts.join("");
  }

  function update(i) {
    draw(i);
    drawDeviation(i);
    readouts(i);
    document.getElementById("angle").textContent = S.input_angles_deg[i].toFixed(1);
    document.getElementById("sweep").textContent =
      (S.input_sweep_deg[i] >= 0 ? "+" : "") + S.input_sweep_deg[i].toFixed(1) +
      " deg from as-printed";
  }

  // --- chrome ----------------------------------------------------------
  function legend() {
    const parts = [];
    for (const key of ["rigid", "prbm", "fea", "measured"]) {
      if (!has(key)) continue;
      parts.push('<span><i class="swatch" style="border-top-color:' + D.colourOf(ST, key) +
        ";border-top-style:" + (ST.series[key].dash ? "dashed" : "solid") +
        '"></i>' + ST.series[key].label + " path</span>");
    }
    const r = D.ramp(ST);
    const bar = 'linear-gradient(90deg,' + r.join(",") + ')';
    parts.push('<span><i style="width:56px;height:10px;border-radius:2px;background:' +
      bar + '"></i>flexure strain, 0 to allowable' +
      (S.allowable_strain_is_placeholder ? " (placeholder: relative scale only)" : "") +
      "</span>");
    parts.push('<span><i class="swatch" style="border-top-color:' + ST.critical +
      '"></i>over allowable</span>');
    document.getElementById("legend").innerHTML = parts.join("");
  }

  function toggles() {
    const box = document.getElementById("toggles");
    const items = [];
    for (const key of ["rigid", "prbm", "fea", "measured"]) {
      if (has(key)) items.push([key, ST.series[key].label]);
    }
    items.push(["paths", "paths"]);
    items.push(["trueScale", "true flexure width"]);
    box.innerHTML = items.map(function (it) {
      return '<button data-key="' + it[0] + '" aria-pressed="' + (show[it[0]] ? "true" : "false") +
        '">' + it[1] + "</button>";
    }).join("");
    box.addEventListener("click", function (event) {
      const button = event.target.closest("button[data-key]");
      if (!button) return;
      const key = button.dataset.key;
      show[key] = !show[key];
      button.setAttribute("aria-pressed", show[key] ? "true" : "false");
      update(Number(slider.value));
      scaleNote();
    });
  }

  function scaleNote() {
    const thick = Object.keys(S.flexure_thickness_mm)
      .map(function (j) { return S.flexure_thickness_mm[j]; });
    const min = Math.min.apply(null, thick);
    document.getElementById("scale-note").textContent = show.trueScale
      ? "Flexures drawn at true width (" + min.toFixed(2) + " mm). Links are " +
        S.link_width_mm.toFixed(0) + " mm."
      : "Flexures are " + min.toFixed(2) + " mm thick beside " +
        S.link_width_mm.toFixed(0) + " mm links, so they are drawn at a minimum of " +
        ST.minFlexureMm.toFixed(1) + " mm to keep the strain colour legible. " +
        "Everything else is to scale; toggle \\u201ctrue flexure width\\u201d for the honest one.";
  }

  function tables() {
    const joints = S.joints;
    const head = ["step", "angle (deg)", "sweep (deg)"];
    if (S.models.prbm && S.models.prbm.torque_nmm) head.push("PRBM T (N\\u00b7mm)");
    if (S.models.fea && S.models.fea.torque_nmm) head.push("FEA T (N\\u00b7mm)");
    const feaFrames = S.models.fea && S.models.fea.frames;
    if (feaFrames) for (const j of joints) head.push("strain " + j + " (%)");

    let html = "<thead><tr>" + head.map((h) => "<th>" + h + "</th>").join("") +
               "</tr></thead><tbody>";
    for (let i = 0; i < S.input_angles_deg.length; i++) {
      const row = [i, S.input_angles_deg[i].toFixed(2), S.input_sweep_deg[i].toFixed(2)];
      const tp = S.models.prbm && S.models.prbm.torque_nmm;
      const tf = S.models.fea && S.models.fea.torque_nmm;
      if (tp) row.push(tp[i].toFixed(2));
      if (tf) row.push(tf[i].toFixed(2));
      if (feaFrames) {
        for (const j of joints) {
          row.push((Math.max.apply(null, feaFrames[i].strain[j]) * 100).toFixed(3));
        }
      }
      html += "<tr>" + row.map((c) => "<td>" + c + "</td>").join("") + "</tr>";
    }
    document.getElementById("table").innerHTML = html + "</tbody>";

    let gaps = "<thead><tr><th>pair</th><th>mean (mm)</th><th>max (mm)</th>" +
               "<th>Frechet (mm)</th><th>samples</th></tr></thead><tbody>";
    for (const c of S.comparisons) {
      gaps += "<tr><td>" + c.a + " vs " + c.b + "</td><td>" + fmt(c.mean_mm, 3) +
        "</td><td>" + fmt(c.max_mm, 3) + "</td><td>" + fmt(c.frechet_mm, 3) +
        "</td><td>" + c.n_points + "</td></tr>";
    }
    document.getElementById("gaps").innerHTML = gaps + "</tbody>";
  }

  function chrome() {
    const meta = S.meta || {};
    document.getElementById("subtitle").textContent =
      S.joints.length + " flexures, " + S.input_angles_deg.length + " precomputed states over " +
      meta.input_range_deg[0].toFixed(1) + " to " + meta.input_range_deg[1].toFixed(1) +
      " deg; printed unstressed at " + meta.reference_input_deg.toFixed(1) + " deg (" +
      meta.material + " on " + meta.printer + ")";

    const banner = document.getElementById("banner");
    let html = "";
    if (S.caveat) {
      html += '<div class="banner"><strong>Not a physical prediction.</strong> ' +
        "Placeholder inputs were used, so these numbers show that the pipeline runs, " +
        "not what the printed part will do. Waiting on: <code>" +
        S.provenance.placeholders_used.join(", ") + "</code></div>";
    }
    for (const note of S.notes || []) {
      html += '<div class="banner" style="border-left-color:var(--muted)">' + note + "</div>";
    }
    banner.innerHTML = html;

    const p = S.provenance;
    document.getElementById("prov").innerHTML =
      "cmtool " + p.version + " &middot; commit <code>" + p.code_commit + "</code>" +
      " &middot; config <code>" + (p.config_hash || "n/a") + "</code>" +
      " &middot; generated " + p.created_utc;
  }

  // --- playback --------------------------------------------------------
  const slider = document.getElementById("slider");
  slider.max = String(S.input_angles_deg.length - 1);
  slider.addEventListener("input", function () { update(Number(slider.value)); });

  const playButton = document.getElementById("play");
  let timer = null, direction = 1;
  playButton.addEventListener("click", function () {
    if (timer) {
      clearInterval(timer);
      timer = null;
      playButton.textContent = "Play";
      playButton.setAttribute("aria-pressed", "false");
      return;
    }
    playButton.textContent = "Pause";
    playButton.setAttribute("aria-pressed", "true");
    timer = setInterval(function () {
      let next = Number(slider.value) + direction;
      // Bounce rather than jump: the arc is limited, and a flexure mechanism
      // snapping from one end back to the other is not a motion it can make.
      if (next > Number(slider.max)) { direction = -1; next = Number(slider.max) - 1; }
      if (next < 0) { direction = 1; next = 1; }
      slider.value = String(next);
      update(next);
    }, 70);
  });

  window.addEventListener("resize", function () { update(Number(slider.value)); });

  window.matchMedia("(prefers-color-scheme: dark)")
    .addEventListener("change", function () {
      legend();
      update(Number(slider.value));
    });

  document.getElementById("deviation-title").textContent =
    "Coupler path, distance from the " + (S.deviation_reference || "reference") +
    " path (mm)";
  document.getElementById("deviation-note").textContent = devSeries.length
    ? "A flat line at zero is not a missing series: with a prescribed input and " +
      "pivot-matched placement the PRBM path is the rigid path exactly, because " +
      "stiffness cannot move a one-degree-of-freedom path. Only the FEA, which lets " +
      "the flexures stretch and shear, departs from it."
    : "";

  chrome();
  legend();
  toggles();
  scaleNote();
  tables();
  update(0);
})();
</script>
</body>
</html>
"""
