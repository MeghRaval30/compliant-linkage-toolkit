/*
 * The local UI.
 *
 * All drawing goes through CmtoolScene (/shared/scene_draw.js), the same module
 * the standalone viewer inlines, so the two cannot disagree about what a scene
 * looks like. Everything numeric comes from the server, which runs the same
 * convert/simulate/build_scene path the CLI does. Nothing is computed here that
 * a solver could compute instead.
 */
(function () {
  "use strict";

  var S = window.CmtoolScene;
  var meta = null;
  var scene = null;
  var result = null;
  var frame = 0;
  var timer = null;
  var direction = 1;
  var poll = null;
  var lastSolved = null;

  var show = { rigid: true, prbm: true, fea: true, measured: true, paths: true, trueScale: false };

  var NUMBER_FIELDS = [
    "ground_mm", "input_mm", "coupler_mm", "output_mm",
    "coupler_x_mm", "coupler_y_mm", "input_angle_deg",
    "target_excursion_deg", "thickness_mm", "n_steps",
    "arc_start_deg", "arc_end_deg",
  ];
  var TEXT_FIELDS = ["material", "printer", "name"];

  var $ = function (id) { return document.getElementById(id); };

  /* ---------------------------------------------------------- params */

  function readParams() {
    var params = Object.assign({}, meta ? meta.defaults : {});
    NUMBER_FIELDS.forEach(function (id) {
      var raw = $(id).value.trim();
      params[id] = raw === "" ? null : Number(raw);
    });
    TEXT_FIELDS.forEach(function (id) { params[id] = $(id).value.trim(); });
    params.flexure_type = $("flexure_type").value;
    params.solvers = Array.prototype.slice
      .call($("solvers").querySelectorAll("input:checked"))
      .map(function (b) { return b.value; });
    return params;
  }

  function writeParams(params) {
    NUMBER_FIELDS.forEach(function (id) {
      var v = params[id];
      $(id).value = v === null || v === undefined ? "" : v;
    });
    TEXT_FIELDS.forEach(function (id) {
      if (params[id] !== undefined) $(id).value = params[id];
    });
    if (params.flexure_type) $("flexure_type").value = params.flexure_type;
    if (params.solvers) {
      Array.prototype.forEach.call($("solvers").querySelectorAll("input"), function (box) {
        box.checked = params.solvers.indexOf(box.value) !== -1;
      });
    }
  }

  /* ------------------------------------------------------------ solve */

  function setProgress(fraction, text) {
    var box = $("progress");
    box.hidden = false;
    $("bar-fill").style.width = Math.round(fraction * 100) + "%";
    $("progress-text").textContent = text;
  }

  function clearProgress() { $("progress").hidden = true; }

  function showError(message) {
    var box = $("error");
    box.hidden = false;
    box.textContent = message;
    $("verdict").className = "verdict bad";
    $("verdict").textContent = "could not solve";
  }

  function clearError() { $("error").hidden = true; }

  function solve() {
    clearError();
    if (poll) { clearInterval(poll); poll = null; }
    var params = readParams();
    lastSolved = params;
    $("solve").disabled = true;
    setProgress(0.04, "submitting");

    fetch("/api/solve", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(params),
    })
      .then(function (response) {
        return response.json().then(function (body) {
          if (!response.ok) throw new Error(body.detail || response.statusText);
          return body;
        });
      })
      .then(function (job) {
        if (job.state === "done") return collect(job.id);
        setProgress(0.1, job.cached ? "from cache" : "solving");
        /* Poll rather than block. A beam FEA sweep is seconds, and the page has
         * to stay responsive through it. */
        poll = setInterval(function () { checkJob(job.id); }, 250);
        return null;
      })
      .catch(function (err) {
        $("solve").disabled = false;
        clearProgress();
        showError(String(err.message || err));
      });
  }

  function checkJob(id) {
    fetch("/api/job/" + id)
      .then(function (r) { return r.json(); })
      .then(function (job) {
        if (job.state === "running") {
          /* The server reports elapsed time, not fake progress: show it as
           * elapsed rather than pretending to know the fraction. */
          setProgress(
            Math.min(0.92, 0.1 + job.elapsed_s / 8),
            "solving — " + job.elapsed_s.toFixed(1) + " s"
          );
          return;
        }
        clearInterval(poll);
        poll = null;
        if (job.state === "error") {
          $("solve").disabled = false;
          clearProgress();
          showError(job.error + (job.detail ? "\n\n" + job.detail : ""));
          return;
        }
        collect(id);
      })
      .catch(function (err) {
        clearInterval(poll);
        poll = null;
        $("solve").disabled = false;
        clearProgress();
        showError(String(err.message || err));
      });
  }

  function collect(id) {
    setProgress(0.97, "drawing");
    return fetch("/api/job/" + id + "/result")
      .then(function (response) {
        return response.json().then(function (body) {
          if (!response.ok) throw new Error(body.detail || response.statusText);
          return body;
        });
      })
      .then(function (payload) {
        result = payload;
        scene = payload.scene;
        frame = 0;
        $("solve").disabled = false;
        clearProgress();
        render();
      })
      .catch(function (err) {
        $("solve").disabled = false;
        clearProgress();
        showError(String(err.message || err));
      });
  }

  /* ----------------------------------------------------------- render */

  function render() {
    if (!scene) return;
    var slider = $("slider");
    slider.max = String(scene.input_angles_deg.length - 1);
    if (Number(slider.value) > Number(slider.max)) slider.value = "0";

    banners();
    verdict();
    feasibilityTable();
    legend();
    toggles();
    provenance();
    update(Number(slider.value));

    var cad = meta && meta.cad_available;
    ["dl-stl", "dl-step", "dl-sheet"].forEach(function (id) {
      $(id).disabled = !cad;
      $(id).title = cad ? "" : "needs cadquery: uv sync --extra cad";
    });
  }

  function update(index) {
    frame = index;
    S.drawMechanism($("view"), scene, meta.style, index, { show: show });
    addCouplerHandle($("view"), index);
    var devKeys = S.drawDeviation($("deviation"), scene, meta.style, index, {
      width: $("deviation").clientWidth || 520,
    });
    S.drawTorque($("torque"), scene, meta.style, index, {
      width: $("torque").clientWidth || 520,
    });
    S.drawStrain($("strain"), scene, meta.style, index, {
      width: $("strain").clientWidth || 520,
    });

    $("dev-title").textContent =
      "Coupler path, distance from the " + (scene.deviation_reference || "reference") + " path (mm)";
    $("dev-note").textContent = devKeys.length
      ? "A flat line at zero is not a missing series: with a prescribed input and " +
        "pivot-matched placement the PRBM path is the rigid path exactly, because " +
        "stiffness cannot move a one-degree-of-freedom path."
      : "Only one model in this run, so there is nothing to compare.";

    var gap = (result.feasibility && result.strain_margin) || {};
    $("torque-note").textContent =
      "The path is fixed by geometry under a prescribed input, so only torque can " +
      "discriminate between the stiffness models.";
    $("strain-note").textContent = gap.allowable_is_placeholder
      ? "The allowable strain is a PLACEHOLDER, so the margin below is relative, not physical."
      : "";

    $("angle").textContent = scene.input_angles_deg[index].toFixed(1);
    $("sweep").textContent =
      (scene.input_sweep_deg[index] >= 0 ? "+" : "") +
      scene.input_sweep_deg[index].toFixed(1) + " deg from as-printed";

    readouts(index);
  }

  function readouts(index) {
    var parts = [];
    ["prbm", "fea"].forEach(function (key) {
      var m = scene.models[key];
      if (!m || !m.torque_nmm) return;
      parts.push(stat(meta.style.series[key].label + " torque",
        Math.abs(m.torque_nmm[index]).toFixed(1), "N&middot;mm"));
    });
    var worst = S.worstStrain(scene, index);
    if (worst) {
      parts.push(stat("peak strain", (worst.value * 100).toFixed(3) + "%", "joint " + worst.joint));
      parts.push(stat("of allowable", (worst.utilisation * 100).toFixed(0) + "%",
        scene.allowable_strain_is_placeholder ? "against a placeholder" : "measured"));
    }
    var margin = result.strain_margin;
    if (margin && margin.available) {
      parts.push(stat("worst-case margin", margin.margin.toFixed(2) + "x",
        "over the whole arc, joint " + margin.worst_joint));
    }
    var comparison = (scene.comparisons || []).filter(function (c) {
      return c.a === "rigid" && c.b === "fea";
    })[0];
    if (comparison && comparison.mean_mm !== null) {
      parts.push(stat("rigid vs FEA path", comparison.mean_mm.toFixed(3), "mm mean"));
    }
    $("readouts").innerHTML = parts.join("");
  }

  function stat(key, value, unit) {
    return '<div class="stat"><div class="k">' + key + '</div><div class="v">' +
      value + '</div><div class="u">' + (unit || "") + "</div></div>";
  }

  function verdict() {
    var f = result.feasibility;
    var node = $("verdict");
    node.className = "verdict " + (f.feasible ? "ok" : "bad");
    node.textContent = f.feasible
      ? "buildable — worst joint " + f.binding_joint +
        " at " + (f.max_utilisation * 100).toFixed(0) + "%"
      : "not buildable — joint " + f.binding_joint;
    $("why").textContent = f.why;
  }

  function feasibilityTable() {
    var f = result.feasibility;
    var head = ["joint", "excursion", "L strain min", "L fits", "L chosen",
                "utilisation", "peak strain", "PRBM model"];
    var html = "<thead><tr>" + head.map(function (h) { return "<th>" + h + "</th>"; }).join("") +
      "</tr></thead><tbody>";
    Object.keys(f.joints).forEach(function (name) {
      var j = f.joints[name];
      var over = j.utilisation >= 1 ? ' class="over"' : "";
      html += "<tr" + (name === f.binding_joint ? ' class="binding"' : "") + ">" +
        "<td>" + name + "</td>" +
        "<td>" + j.excursion_deg.toFixed(2) + "&deg;</td>" +
        "<td>" + j.min_length_strain_mm.toFixed(2) + "</td>" +
        "<td>" + j.max_length_geometric_mm.toFixed(2) + "</td>" +
        "<td>" + j.length_mm.toFixed(2) + "</td>" +
        "<td" + over + ">" + j.utilisation.toFixed(3) + "</td>" +
        "<td>" + (j.peak_strain * 100).toFixed(3) + "%</td>" +
        "<td>" + j.prbm_model + "</td></tr>";
    });
    $("joints").innerHTML = html + "</tbody>";
  }

  function banners() {
    var html = "";
    if (result.caveat) {
      html += '<div class="banner"><strong>Not a physical prediction.</strong> ' +
        "Placeholder inputs were used, so these numbers show that the pipeline runs, " +
        "not what a printed part will do. Waiting on: <code>" +
        result.placeholders.join(", ") + "</code></div>";
    }
    (scene.notes || []).forEach(function (note) {
      html += '<div class="banner note">' + note + "</div>";
    });
    /* One line, not one per joint: four identical sentences differing only in a
     * joint name push everything else off the screen. */
    var notes = result.feasibility.prbm_notes || [];
    if (notes.length) {
      var models = {};
      Object.keys(result.feasibility.joints).forEach(function (j) {
        var m = result.feasibility.joints[j].prbm_model;
        (models[m] = models[m] || []).push(j);
      });
      var summary = Object.keys(models).map(function (m) {
        return models[m].join("/") + ": " + m;
      }).join(" · ");
      html += '<div class="banner note"><strong>PRBM model per joint.</strong> ' +
        summary + ". Where the small-length model does not apply the beam FEA is the " +
        "reference, not the PRBM. This is metadata, not a feasibility limit." +
        '<details><summary>per-joint detail</summary>' +
        notes.map(function (n) { return "<div>" + n + "</div>"; }).join("") +
        "</details></div>";
    }
    $("banner").innerHTML = html;
  }

  function legend() {
    var parts = [];
    ["rigid", "prbm", "fea", "measured"].forEach(function (key) {
      if (!scene.models[key]) return;
      parts.push('<span><i class="swatch" style="border-top-color:' +
        S.colourOf(meta.style, key) + ";border-top-style:" +
        (meta.style.series[key].dash ? "dashed" : "solid") + '"></i>' +
        meta.style.series[key].label + "</span>");
    });
    var r = S.ramp(meta.style);
    parts.push('<span><i style="width:52px;height:10px;border-radius:2px;background:' +
      "linear-gradient(90deg," + r.join(",") + ')"></i>flexure strain, 0 to allowable' +
      (scene.allowable_strain_is_placeholder ? " (placeholder)" : "") + "</span>");
    $("legend").innerHTML = parts.join("");
  }

  function toggles() {
    var items = [];
    ["rigid", "prbm", "fea", "measured"].forEach(function (key) {
      if (scene.models[key]) items.push([key, meta.style.series[key].label]);
    });
    items.push(["paths", "paths"]);
    items.push(["trueScale", "true flexure width"]);
    $("toggles").innerHTML = items.map(function (it) {
      return '<button data-key="' + it[0] + '" aria-pressed="' +
        (show[it[0]] ? "true" : "false") + '">' + it[1] + "</button>";
    }).join("");
  }

  function provenance() {
    var p = scene.provenance;
    $("prov").textContent =
      "cmtool " + p.version + " · commit " + p.code_commit +
      " · config " + (p.config_hash || "n/a") +
      " · solved " + p.created_utc +
      " · " + scene.input_angles_deg.length + " states";
  }

  /* ------------------------------------------------- coupler drag handle */

  /* The coupler point is the one input worth dragging: it is the thing the
   * mechanism draws with, and typing coordinates for it is guesswork. The drag
   * happens at whatever frame is on screen, so the grabbed position has to be
   * carried back through the coupler's own frame to the reference pose, which is
   * the pose `coupler_x_mm` / `coupler_y_mm` are expressed in. */

  function couplerFrame(index) {
    var f = scene.models.fea || scene.models.rigid || scene.models.prbm;
    if (!f || !f.frames || !f.frames[index]) return null;
    var links = f.frames[index].links;
    var poly = links[couplerBody()];
    if (!poly || poly.length < 2) return null;
    var o = poly[0];
    var far = poly[poly.length - 1];
    return { o: o, angle: Math.atan2(far[1] - o[1], far[0] - o[0]) };
  }

  function couplerBody() {
    /* The body carrying the tracked point: every body except ground and the two
     * that touch it. Falls back to the literal name. */
    var f = scene.models.rigid || scene.models.fea || scene.models.prbm;
    if (f && f.frames && f.frames[0]) {
      var names = Object.keys(f.frames[0].links);
      if (names.indexOf("coupler") !== -1) return "coupler";
      return names[0];
    }
    return "coupler";
  }

  function referenceIndex() {
    var target = result.reference_input_deg;
    var best = 0;
    var bestGap = Infinity;
    scene.input_angles_deg.forEach(function (a, i) {
      var gap = Math.abs(a - target);
      if (gap < bestGap) { bestGap = gap; best = i; }
    });
    return best;
  }

  function addCouplerHandle(svg, index) {
    var f = scene.models.fea || scene.models.rigid || scene.models.prbm;
    if (!f || !f.frames || !f.frames[index]) return;
    var point = f.frames[index].output;
    var b = scene.bounds_mm;
    var flip = b[1] + b[3];

    var ring = S.el("circle", {
      cx: point[0],
      cy: flip - point[1],
      r: 3.4,
      fill: "none",
      stroke: S.colourOf(meta.style, "measured"),
      "stroke-width": 0.8,
      "stroke-dasharray": "1.5 1.5",
      class: "handle",
    });
    /* A fat invisible target: 3.4 mm of SVG is a hard thing to hit with a mouse
     * and impossible with a finger. */
    var grab = S.el("circle", {
      cx: point[0],
      cy: flip - point[1],
      r: 7,
      fill: "transparent",
      class: "handle",
    });
    svg.appendChild(ring);
    svg.appendChild(grab);

    grab.addEventListener("pointerdown", function (event) {
      event.preventDefault();
      grab.setPointerCapture(event.pointerId);
      var cur = couplerFrame(index);
      var ref = couplerFrame(referenceIndex());
      if (!cur || !ref) return;

      function move(ev) {
        var p = svgPoint(svg, ev);
        var world = [p.x, flip - p.y];
        /* Into the coupler's local frame at the frame on screen... */
        var dx = world[0] - cur.o[0];
        var dy = world[1] - cur.o[1];
        var ca = Math.cos(-cur.angle), sa = Math.sin(-cur.angle);
        var lx = dx * ca - dy * sa;
        var ly = dx * sa + dy * ca;
        /* ...and back out through the reference pose. */
        var ra = Math.cos(ref.angle), rb = Math.sin(ref.angle);
        $("coupler_x_mm").value = round(ref.o[0] + lx * ra - ly * rb);
        $("coupler_y_mm").value = round(ref.o[1] + lx * rb + ly * ra);
        ring.setAttribute("cx", world[0]);
        ring.setAttribute("cy", flip - world[1]);
        grab.setAttribute("cx", world[0]);
        grab.setAttribute("cy", flip - world[1]);
      }

      function up() {
        svg.removeEventListener("pointermove", move);
        svg.removeEventListener("pointerup", up);
        svg.removeEventListener("pointercancel", up);
        $("preset").value = "";
        solve();
      }

      svg.addEventListener("pointermove", move);
      svg.addEventListener("pointerup", up);
      svg.addEventListener("pointercancel", up);
    });
  }

  function svgPoint(svg, event) {
    var pt = svg.createSVGPoint();
    pt.x = event.clientX;
    pt.y = event.clientY;
    return pt.matrixTransform(svg.getScreenCTM().inverse());
  }

  /* --------------------------------------------------------- downloads */

  function download(url, params, fallbackName) {
    fetch(url, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(params),
    })
      .then(function (response) {
        if (!response.ok) {
          return response.json().then(function (body) {
            throw new Error(body.detail || response.statusText);
          });
        }
        var name = fallbackName;
        var disposition = response.headers.get("Content-Disposition") || "";
        var match = /filename="?([^";]+)"?/.exec(disposition);
        if (match) name = match[1];
        return response.blob().then(function (blob) { save(blob, name); });
      })
      .catch(function (err) { showError(String(err.message || err)); });
  }

  function save(blob, name) {
    var url = URL.createObjectURL(blob);
    var anchor = document.createElement("a");
    anchor.href = url;
    anchor.download = name;
    document.body.appendChild(anchor);
    anchor.click();
    document.body.removeChild(anchor);
    URL.revokeObjectURL(url);
  }

  /* ------------------------------------------------------------- wiring */

  function wire() {
    $("solve").addEventListener("click", solve);

    $("slider").addEventListener("input", function () {
      update(Number(this.value));
    });

    $("play").addEventListener("click", function () {
      var slider = $("slider");
      if (timer) {
        clearInterval(timer);
        timer = null;
        this.textContent = "Play";
        this.setAttribute("aria-pressed", "false");
        return;
      }
      this.textContent = "Pause";
      this.setAttribute("aria-pressed", "true");
      timer = setInterval(function () {
        var next = Number(slider.value) + direction;
        /* Bounce rather than jump: the arc is limited, and a flexure mechanism
         * snapping from one end back to the other is not a motion it can make. */
        if (next > Number(slider.max)) { direction = -1; next = Number(slider.max) - 1; }
        if (next < 0) { direction = 1; next = 1; }
        slider.value = String(next);
        update(next);
      }, 70);
    });

    $("toggles").addEventListener("click", function (event) {
      var button = event.target.closest("button[data-key]");
      if (!button) return;
      var key = button.dataset.key;
      show[key] = !show[key];
      button.setAttribute("aria-pressed", show[key] ? "true" : "false");
      update(frame);
    });

    $("preset").addEventListener("change", function () {
      if (!this.value) return;
      fetch("/api/preset/" + encodeURIComponent(this.value))
        .then(function (r) { return r.json(); })
        .then(function (params) { writeParams(params); solve(); })
        .catch(function (err) { showError(String(err.message || err)); });
    });

    $("dl-stl").addEventListener("click", function () {
      download("/api/export/stl", lastSolved || readParams(), "design.stl");
    });
    $("dl-step").addEventListener("click", function () {
      download("/api/export/step", lastSolved || readParams(), "design.step");
    });
    $("dl-sheet").addEventListener("click", function () {
      download("/api/export/sheet", lastSolved || readParams(), "print_sheet.md");
    });
    $("dl-json").addEventListener("click", function () {
      download("/api/design.json", readParams(), "design.json");
    });

    $("load-json").addEventListener("click", function () { $("file").click(); });
    $("file").addEventListener("change", function () {
      var file = this.files && this.files[0];
      if (!file) return;
      file.text().then(function (text) {
        try {
          loadDesign(JSON.parse(text));
        } catch (err) {
          showError("could not read that file: " + err.message);
        }
      });
      this.value = "";
    });

    window.addEventListener("resize", function () { if (scene) update(frame); });
  }

  /* Accepts either a saved parameter set or a linkage JSON as the CLI writes it. */
  function loadDesign(data) {
    if (data.joints && data.bodies) {
      var joints = data.joints;
      var length = function (a, b) {
        var pa = joints[a].xy_mm, pb = joints[b].xy_mm;
        return Math.hypot(pb[0] - pa[0], pb[1] - pa[1]);
      };
      var out = data.outputs && data.outputs[Object.keys(data.outputs)[0]];
      writeParams({
        ground_mm: round(length("A", "D")),
        input_mm: round(length("A", "B")),
        coupler_mm: round(length("B", "C")),
        output_mm: round(length("C", "D")),
        coupler_x_mm: out ? round(out.xy_mm[0]) : null,
        coupler_y_mm: out ? round(out.xy_mm[1]) : null,
        input_angle_deg: (data.meta && data.meta.reference_input_deg) || 90,
        arc_start_deg: data.input_range_deg ? round(data.input_range_deg[0]) : null,
        arc_end_deg: data.input_range_deg ? round(data.input_range_deg[1]) : null,
        name: data.name || "loaded",
      });
    } else {
      writeParams(data);
    }
    $("preset").value = "";
    solve();
  }

  function round(v) { return Math.round(v * 100) / 100; }

  /* ---------------------------------------------------------- start up */

  fetch("/api/meta")
    .then(function (r) { return r.json(); })
    .then(function (payload) {
      meta = payload;
      $("build").textContent =
        "local · offline · cmtool " + meta.version + " @ " + meta.commit.slice(0, 10);
      meta.presets.forEach(function (name) {
        var option = document.createElement("option");
        option.value = name;
        option.textContent = name;
        $("preset").appendChild(option);
      });
      meta.flexures.forEach(function (name) {
        var option = document.createElement("option");
        option.value = name;
        option.textContent = name;
        $("flexure_type").appendChild(option);
      });
      writeParams(meta.defaults);
      $("verdict").className = "verdict idle";
      $("verdict").textContent = "not solved yet";
      wire();
      $("preset").value = "demo_pair";
      $("preset").dispatchEvent(new Event("change"));
    })
    .catch(function (err) {
      showError("could not reach the local server: " + (err.message || err));
    });
})();
