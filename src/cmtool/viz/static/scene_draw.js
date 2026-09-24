/*
 * Drawing a ViewerScene, in one place.
 *
 * Two things render scenes: the self-contained HTML viewer (cmtool view), which
 * inlines this file so the output needs nothing from the network, and the local
 * UI (cmtool ui), which serves it. Sharing the file is the point -- the scene
 * model exists so the two cannot disagree about what was solved, and that would
 * be undone by two copies of the drawing code drifting apart.
 *
 * Nothing here touches the DOM beyond the <svg> element it is handed, reads no
 * globals, and fetches nothing. Every function takes (svg, scene, style, ...)
 * and returns nothing.
 */
(function (root) {
  "use strict";

  var SVGNS = "http://www.w3.org/2000/svg";

  function el(name, attrs) {
    var node = document.createElementNS(SVGNS, name);
    for (var k in attrs) {
      if (attrs[k] !== null && attrs[k] !== undefined) node.setAttribute(k, attrs[k]);
    }
    return node;
  }

  function clear(svg) {
    while (svg.firstChild) svg.removeChild(svg.firstChild);
  }

  function rgb(hex) {
    var h = String(hex).replace("#", "");
    return [0, 2, 4].map(function (i) {
      return parseInt(h.slice(i, i + 2), 16);
    });
  }

  function mix(a, b, f) {
    var pa = rgb(a);
    var pb = rgb(b);
    return (
      "#" +
      [0, 1, 2]
        .map(function (i) {
          return Math.round(pa[i] + (pb[i] - pa[i]) * f)
            .toString(16)
            .padStart(2, "0");
        })
        .join("")
    );
  }

  /* Read the theme off the rendered surface rather than a media query, so the
   * drawing follows whatever CSS actually applied -- including a page that
   * pins its own theme. */
  function ink(name, host) {
    var target = host || document.documentElement;
    return getComputedStyle(target).getPropertyValue("--" + name).trim();
  }

  function isDark(host) {
    var surface = ink("surface", host) || "#ffffff";
    var c = rgb(surface);
    return (c[0] + c[1] + c[2]) / 3 < 128;
  }

  function colourOf(style, key, host) {
    var s = style.series[key];
    return isDark(host) ? s.dark : s.light;
  }

  function ramp(style, host) {
    return isDark(host) ? style.rampDark : style.rampLight;
  }

  /* Strain as a fraction of the allowable. At or over the limit the mark leaves
   * the ramp entirely for the reserved critical colour: that is a different
   * statement, not a darker shade of the same one. */
  function strainColour(utilisation, style, host) {
    if (!isFinite(utilisation)) return style.critical;
    if (utilisation >= 1) return style.critical;
    var r = ramp(style, host);
    var pos = Math.max(0, utilisation) * (r.length - 1);
    var lo = Math.min(Math.floor(pos), r.length - 2);
    return mix(r[lo], r[lo + 1], pos - lo);
  }

  /* --- the mechanism ------------------------------------------------- */

  /* Millimetres with y up. SVG y runs down, so every y is reflected about the
   * midline of the bounds. Done in code rather than with a transform, so stroke
   * widths stay in millimetres and text never comes out mirrored. */
  function frameOf(scene, key, index) {
    var m = scene.models[key];
    return m && m.frames && m.frames[index] ? m.frames[index] : null;
  }

  function has(scene, key) {
    return Object.prototype.hasOwnProperty.call(scene.models, key);
  }

  function drawMechanism(svg, scene, style, index, options) {
    var opts = options || {};
    var show = opts.show || { rigid: true, prbm: true, fea: true, paths: true, trueScale: false };
    var host = opts.host;
    var b = scene.bounds_mm;
    var x0 = b[0], y0 = b[1], x1 = b[2], y1 = b[3];
    var flip = y0 + y1;
    var X = function (v) { return v; };
    var Y = function (v) { return flip - v; };
    var pts = function (poly) {
      return poly
        .map(function (p) { return X(p[0]) + "," + Y(p[1]); })
        .join(" ");
    };

    clear(svg);
    svg.setAttribute("viewBox", x0 + " " + y0 + " " + (x1 - x0) + " " + (y1 - y0));
    svg.setAttribute("preserveAspectRatio", "xMidYMid meet");
    /* An inline SVG with only a viewBox has no intrinsic height, so height:auto
     * resolves to whatever the box gives it. Stating the ratio pins it. */
    svg.style.aspectRatio = (x1 - x0) + " / " + (y1 - y0);

    if (show.paths) {
      ["rigid", "prbm", "fea", "measured"].forEach(function (key) {
        if (!has(scene, key) || !show[key]) return;
        svg.appendChild(
          el("polyline", {
            points: pts(scene.models[key].path),
            fill: "none",
            stroke: colourOf(style, key, host),
            "stroke-width": 0.7,
            "stroke-dasharray": style.series[key].dash || null,
            "stroke-linecap": "round",
            opacity: 0.95,
          })
        );
      });
    }

    if (scene.ground_polyline_mm && scene.ground_polyline_mm.length === 2) {
      svg.appendChild(
        el("polyline", {
          points: pts(scene.ground_polyline_mm),
          fill: "none",
          stroke: ink("axis", host),
          "stroke-width": scene.link_width_mm,
          "stroke-linecap": "round",
          opacity: 0.55,
        })
      );
    }

    /* The FEA part: links as true-width bars with a hairline ring, then the
     * flexures element by element so each wears its own strain. */
    var fea = show.fea ? frameOf(scene, "fea", index) : null;
    if (fea) {
      Object.keys(fea.links).forEach(function (body) {
        svg.appendChild(
          el("polyline", {
            points: pts(fea.links[body]),
            fill: "none",
            stroke: ink("axis", host),
            "stroke-width": scene.link_width_mm + 1.2,
            "stroke-linecap": "round",
            "stroke-linejoin": "round",
          })
        );
        svg.appendChild(
          el("polyline", {
            points: pts(fea.links[body]),
            fill: "none",
            stroke: ink("grid", host),
            "stroke-width": scene.link_width_mm,
            "stroke-linecap": "round",
            "stroke-linejoin": "round",
          })
        );
      });
      Object.keys(fea.flexures).forEach(function (joint) {
        var poly = fea.flexures[joint];
        var strain = fea.strain[joint];
        var trueWidth = scene.flexure_thickness_mm[joint];
        var width = show.trueScale
          ? trueWidth
          : Math.max(trueWidth, style.minFlexureMm);
        for (var e = 0; e < strain.length; e++) {
          svg.appendChild(
            el("line", {
              x1: X(poly[e][0]),
              y1: Y(poly[e][1]),
              x2: X(poly[e + 1][0]),
              y2: Y(poly[e + 1][1]),
              stroke: strainColour(strain[e] / scene.allowable_strain, style, host),
              "stroke-width": width,
              "stroke-linecap": "round",
            })
          );
        }
      });
    }

    /* Ghost outlines of the pin-jointed models at the same input angle. */
    ["rigid", "prbm"].forEach(function (key) {
      var frame = show[key] ? frameOf(scene, key, index) : null;
      if (!frame) return;
      var dash = style.series[key].dash || null;
      Object.keys(frame.links).forEach(function (body) {
        if (body === scene.ground_body) return;
        svg.appendChild(
          el("polyline", {
            points: pts(frame.links[body]),
            fill: "none",
            stroke: colourOf(style, key, host),
            "stroke-width": 1.1,
            "stroke-dasharray": dash,
            "stroke-linecap": "round",
          })
        );
      });
      Object.keys(frame.links).forEach(function (body) {
        frame.links[body].forEach(function (p) {
          svg.appendChild(
            el("circle", {
              cx: X(p[0]),
              cy: Y(p[1]),
              r: 1.2,
              fill: "none",
              stroke: colourOf(style, key, host),
              "stroke-width": 0.8,
            })
          );
        });
      });
    });

    ["rigid", "prbm", "fea"].forEach(function (key) {
      var frame = show[key] ? frameOf(scene, key, index) : null;
      if (!frame) return;
      svg.appendChild(
        el("circle", {
          cx: X(frame.output[0]),
          cy: Y(frame.output[1]),
          r: key === "fea" ? 2.2 : 1.6,
          fill: colourOf(style, key, host),
          stroke: ink("surface", host),
          "stroke-width": 0.6,
        })
      );
    });
  }

  /* --- line charts ---------------------------------------------------- */

  function chartFrame(svg, opts) {
    var width = Math.max(320, Math.min(760, opts.width || 520));
    var height = Math.round(width * (opts.ratio || 0.42)) + 40;
    svg.setAttribute("viewBox", "0 0 " + width + " " + height);
    svg.style.aspectRatio = width + " / " + height;
    return { w: width, h: height, l: 56, r: 16, t: 14, b: 34 };
  }

  function axes(svg, box, scales, opts, host) {
    var ticks = [0, 0.25, 0.5, 0.75, 1].map(function (f) {
      return scales.lo + f * (scales.top - scales.lo);
    });
    ticks.forEach(function (value) {
      svg.appendChild(
        el("line", {
          x1: box.l,
          y1: scales.py(value),
          x2: box.w - box.r,
          y2: scales.py(value),
          stroke: value === 0 ? ink("axis", host) : ink("grid", host),
          "stroke-width": 1,
        })
      );
      var label = el("text", {
        x: box.l - 8,
        y: scales.py(value) + 4,
        "text-anchor": "end",
        fill: ink("muted", host),
        "font-size": 11,
      });
      label.textContent = opts.format ? opts.format(value) : value.toFixed(2);
      svg.appendChild(label);
    });
    var xLabel = el("text", {
      x: (box.l + box.w - box.r) / 2,
      y: box.h - 8,
      "text-anchor": "middle",
      fill: ink("muted", host),
      "font-size": 11,
    });
    xLabel.textContent = opts.xLabel || "input angle (deg)";
    svg.appendChild(xLabel);
  }

  function makeScales(box, angles, lo, top) {
    var a0 = Math.min(angles[0], angles[angles.length - 1]);
    var a1 = Math.max(angles[0], angles[angles.length - 1]);
    return {
      lo: lo,
      top: top,
      px: function (a) {
        return box.l + ((a - a0) / (a1 - a0 || 1)) * (box.w - box.l - box.r);
      },
      py: function (v) {
        return (
          box.h -
          box.b -
          ((v - lo) / (top - lo || 1)) * (box.h - box.t - box.b)
        );
      },
    };
  }

  function series(svg, scales, xs, ys, colour, dash, width) {
    var points = ys
      .map(function (v, i) { return scales.px(xs[i]) + "," + scales.py(v); })
      .join(" ");
    svg.appendChild(
      el("polyline", {
        points: points,
        fill: "none",
        stroke: colour,
        "stroke-width": width || 2,
        "stroke-dasharray": dash || null,
        "stroke-linecap": "round",
        "stroke-linejoin": "round",
      })
    );
  }

  function crosshair(svg, box, scales, x, host) {
    svg.appendChild(
      el("line", {
        x1: scales.px(x),
        y1: box.t,
        x2: scales.px(x),
        y2: box.h - box.b,
        stroke: ink("secondary", host),
        "stroke-width": 1,
        "stroke-dasharray": "3 3",
      })
    );
  }

  function tag(svg, x, y, text, colour) {
    var node = el("text", {
      x: x - 4,
      y: y - 7,
      "text-anchor": "end",
      fill: colour,
      "font-size": 11,
      "font-weight": "600",
    });
    node.textContent = text;
    svg.appendChild(node);
  }

  /* How far each model's coupler path sits from the reference path. The three
   * predictions differ by tenths of a millimetre across a part 140 mm wide, so
   * drawn on top of each other they are one curve -- this is the chart that
   * shows the difference, and it is the quantity the go/no-go is stated
   * against. */
  function drawDeviation(svg, scene, style, index, options) {
    var opts = options || {};
    var host = opts.host;
    clear(svg);
    var keys = ["prbm", "fea", "measured"].filter(function (k) {
      return (
        has(scene, k) &&
        scene.models[k].deviation_mm &&
        k !== scene.deviation_reference
      );
    });
    if (!keys.length) return keys;

    var box = chartFrame(svg, opts);
    var top = 0;
    keys.forEach(function (k) {
      scene.models[k].deviation_mm.forEach(function (v) {
        top = Math.max(top, v);
      });
    });
    var scales = makeScales(box, scene.input_angles_deg, 0, top > 0 ? top * 1.15 : 1);
    axes(svg, box, scales, { xLabel: opts.xLabel }, host);

    keys.forEach(function (key) {
      var m = scene.models[key];
      var xs = key === "measured" ? m.input_angles_deg : scene.input_angles_deg;
      series(
        svg,
        scales,
        xs,
        m.deviation_mm,
        colourOf(style, key, host),
        style.series[key].dash,
        key === "measured" ? style.series[key].width : 2
      );
      var last = m.deviation_mm.length - 1;
      tag(
        svg,
        scales.px(xs[last]),
        scales.py(m.deviation_mm[last]),
        style.series[key].label,
        colourOf(style, key, host)
      );
    });

    crosshair(svg, box, scales, scene.input_angles_deg[index], host);
    keys.forEach(function (key) {
      if (key === "measured") return;
      svg.appendChild(
        el("circle", {
          cx: scales.px(scene.input_angles_deg[index]),
          cy: scales.py(scene.models[key].deviation_mm[index]),
          r: 4,
          fill: colourOf(style, key, host),
          stroke: ink("surface", host),
          "stroke-width": 1.5,
        })
      );
    });
    return keys;
  }

  /* Input torque. With a prescribed input the path is fixed by geometry, so the
   * path cannot discriminate between stiffness models at all -- torque can, and
   * this is where the two differ by about half. */
  function drawTorque(svg, scene, style, index, options) {
    var opts = options || {};
    var host = opts.host;
    clear(svg);
    var keys = ["prbm", "fea"].filter(function (k) {
      return has(scene, k) && scene.models[k].torque_nmm;
    });
    if (!keys.length) return keys;

    var box = chartFrame(svg, opts);
    var top = 0;
    keys.forEach(function (k) {
      scene.models[k].torque_nmm.forEach(function (v) {
        top = Math.max(top, Math.abs(v));
      });
    });
    var scales = makeScales(box, scene.input_angles_deg, 0, top > 0 ? top * 1.15 : 1);
    axes(svg, box, scales, { xLabel: opts.xLabel, format: function (v) { return v.toFixed(0); } }, host);

    keys.forEach(function (key) {
      var values = scene.models[key].torque_nmm.map(Math.abs);
      series(
        svg,
        scales,
        scene.input_angles_deg,
        values,
        colourOf(style, key, host),
        style.series[key].dash
      );
      var last = values.length - 1;
      tag(
        svg,
        scales.px(scene.input_angles_deg[last]),
        scales.py(values[last]),
        style.series[key].label,
        colourOf(style, key, host)
      );
    });

    crosshair(svg, box, scales, scene.input_angles_deg[index], host);
    keys.forEach(function (key) {
      svg.appendChild(
        el("circle", {
          cx: scales.px(scene.input_angles_deg[index]),
          cy: scales.py(Math.abs(scene.models[key].torque_nmm[index])),
          r: 4,
          fill: colourOf(style, key, host),
          stroke: ink("surface", host),
          "stroke-width": 1.5,
        })
      );
    });
    return keys;
  }

  /* Peak flexure strain through the arc, against the allowable. Each flexure is
   * printed unstressed at mid-arc, so every curve is a V. */
  function drawStrain(svg, scene, style, index, options) {
    var opts = options || {};
    var host = opts.host;
    clear(svg);
    var fea = scene.models.fea;
    if (!fea || !fea.frames || !fea.frames.length) return [];

    var box = chartFrame(svg, opts);
    var allowable = scene.allowable_strain * 100;
    var curves = {};
    var top = allowable;
    scene.joints.forEach(function (joint) {
      curves[joint] = fea.frames.map(function (frame) {
        return Math.max.apply(null, frame.strain[joint]) * 100;
      });
      top = Math.max(top, Math.max.apply(null, curves[joint]));
    });
    var scales = makeScales(box, scene.input_angles_deg, 0, top * 1.18);
    axes(
      svg,
      box,
      scales,
      { xLabel: opts.xLabel, format: function (v) { return v.toFixed(2); } },
      host
    );

    svg.appendChild(
      el("line", {
        x1: box.l,
        y1: scales.py(allowable),
        x2: box.w - box.r,
        y2: scales.py(allowable),
        stroke: style.critical,
        "stroke-width": 1.6,
        "stroke-dasharray": "6 3",
      })
    );
    var limit = el("text", {
      x: box.l + 4,
      y: scales.py(allowable) - 4,
      fill: style.critical,
      "font-size": 11,
    });
    limit.textContent =
      "allowable" + (scene.allowable_strain_is_placeholder ? " (placeholder)" : "");
    svg.appendChild(limit);

    var r = ramp(style, host);
    scene.joints.forEach(function (joint, i) {
      var colour = r[Math.min(i + 1, r.length - 1)];
      series(svg, scales, scene.input_angles_deg, curves[joint], colour, null, 1.8);
      var last = curves[joint].length - 1;
      tag(
        svg,
        scales.px(scene.input_angles_deg[last]),
        scales.py(curves[joint][last]),
        joint,
        colour
      );
    });
    crosshair(svg, box, scales, scene.input_angles_deg[index], host);
    return scene.joints;
  }

  /* Worst flexure at one state, and how much of the allowable it is using. */
  function worstStrain(scene, index) {
    var frame = frameOf(scene, "fea", index);
    if (!frame) return null;
    var best = { joint: null, value: -1 };
    Object.keys(frame.strain).forEach(function (joint) {
      var peak = Math.max.apply(null, frame.strain[joint]);
      if (peak > best.value) best = { joint: joint, value: peak };
    });
    if (best.joint === null) return null;
    best.utilisation = best.value / scene.allowable_strain;
    best.margin = scene.allowable_strain / (best.value || Infinity);
    return best;
  }

  root.CmtoolScene = {
    drawMechanism: drawMechanism,
    drawDeviation: drawDeviation,
    drawTorque: drawTorque,
    drawStrain: drawStrain,
    strainColour: strainColour,
    worstStrain: worstStrain,
    isDark: isDark,
    colourOf: colourOf,
    ramp: ramp,
    el: el,
    clear: clear,
  };
})(typeof window !== "undefined" ? window : this);
