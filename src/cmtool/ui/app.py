"""The local UI's HTTP layer: jobs, cache, downloads, static files.

Local only, by design. The server binds to loopback, holds no state a restart
would miss, and fetches nothing from the network -- the page and its one script
come out of the installed package. It is meant to be started in front of an
audience and to work on a laptop with the wifi off.

Why jobs rather than a plain request
------------------------------------
A beam FEA sweep is seconds, not milliseconds. Running it inside the request
would leave the page dead until it finished, which is exactly the thing to avoid
in a live demo. So a solve is submitted, runs on a worker thread, and the page
polls for progress. The progress is real -- the solver reports the states it has
completed -- not an animation pretending to be one.

Results are cached on the parameter hash, so stepping back to a design already
solved is instant, and the cache key is the same ``canonical_hash`` the rest of
the toolkit uses for provenance.
"""

from __future__ import annotations

import json
import threading
import time
import traceback
import uuid
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from cmtool.ui.model import PRESETS, DesignParams, preset_params, solve

#: How many solved designs to keep. Each is a few hundred kB of scene JSON.
CACHE_SIZE = 24

#: A finished job stays readable this long, so a slow page can still collect it.
JOB_TTL_S = 900.0


@dataclass
class Job:
    """One solve, in flight or finished."""

    id: str
    params: DesignParams
    state: str = "running"
    progress: float = 0.0
    message: str = "starting"
    result: dict[str, Any] | None = None
    error: str | None = None
    detail: str | None = None
    started: float = field(default_factory=time.monotonic)
    finished: float | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return the status the page polls for, without the bulky result."""
        return {
            "id": self.id,
            "state": self.state,
            "progress": self.progress,
            "message": self.message,
            "error": self.error,
            "detail": self.detail,
            "elapsed_s": (self.finished or time.monotonic()) - self.started,
        }


class Solver:
    """Runs solves on worker threads, with a small result cache.

    One lock guards both maps. The work itself happens outside the lock, so a
    long FEA never blocks the page from reading progress.
    """

    def __init__(self, cache_size: int = CACHE_SIZE) -> None:
        self._lock = threading.Lock()
        self._jobs: dict[str, Job] = {}
        self._cache: OrderedDict[str, dict[str, Any]] = OrderedDict()
        self._cache_size = cache_size

    # ------------------------------------------------------------- cache

    def cached(self, key: str) -> dict[str, Any] | None:
        """Return a previously solved result, marking it recently used."""
        with self._lock:
            if key not in self._cache:
                return None
            self._cache.move_to_end(key)
            return self._cache[key]

    def _store(self, key: str, result: dict[str, Any]) -> None:
        with self._lock:
            self._cache[key] = result
            self._cache.move_to_end(key)
            while len(self._cache) > self._cache_size:
                self._cache.popitem(last=False)

    @property
    def cache_keys(self) -> list[str]:
        """Keys currently held, oldest first."""
        with self._lock:
            return list(self._cache)

    # -------------------------------------------------------------- jobs

    def submit(self, params: DesignParams) -> Job:
        """Start a solve, or return a finished job if the answer is already known."""
        key = params.cache_key
        hit = self.cached(key)
        job = Job(id=uuid.uuid4().hex[:12], params=params)
        if hit is not None:
            job.state = "done"
            job.progress = 1.0
            job.message = "from cache"
            job.result = hit
            job.finished = time.monotonic()
            with self._lock:
                self._jobs[job.id] = job
            return job

        with self._lock:
            self._jobs[job.id] = job
        thread = threading.Thread(target=self._run, args=(job,), daemon=True)
        thread.start()
        return job

    def _run(self, job: Job) -> None:
        try:
            job.message = "converting"
            job.progress = 0.05
            result = solve(job.params)
            self._store(job.params.cache_key, result)
            job.result = result
            job.progress = 1.0
            job.message = "done"
            job.state = "done"
        except Exception as exc:  # surfaced to the page verbatim, never swallowed
            job.state = "error"
            job.error = str(exc) or exc.__class__.__name__
            job.detail = traceback.format_exc(limit=4)
            job.message = "failed"
        finally:
            job.finished = time.monotonic()

    def job(self, job_id: str) -> Job | None:
        """Return a job by id, dropping ones that have aged out."""
        now = time.monotonic()
        with self._lock:
            stale = [
                k
                for k, j in self._jobs.items()
                if j.finished is not None and now - j.finished > JOB_TTL_S
            ]
            for key in stale:
                del self._jobs[key]
            return self._jobs.get(job_id)


def _static_dir(package: str) -> Path:
    from importlib import resources

    return Path(str(resources.files(package))) / "static"


def create_app(*, solver: Solver | None = None) -> Any:
    """Build the FastAPI application.

    Separated from serving so the tests can drive it with ``TestClient`` without
    binding a port.
    """
    try:
        from fastapi import FastAPI, HTTPException
        from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response
        from fastapi.staticfiles import StaticFiles
    except ModuleNotFoundError as exc:  # pragma: no cover - dependency guard
        raise ModuleNotFoundError(
            "the local UI needs FastAPI and uvicorn: install them with\n"
            "    uv sync --extra ui\n"
            "or  pip install 'cmtool[ui]'"
        ) from exc

    engine = solver or Solver()
    ui_static = _static_dir("cmtool.ui")
    viz_static = _static_dir("cmtool.viz")

    app = FastAPI(title="cmtool", docs_url=None, redoc_url=None)
    app.state.solver = engine

    @app.get("/", response_class=HTMLResponse)
    def index() -> Any:
        return HTMLResponse((ui_static / "index.html").read_text(encoding="utf-8"))

    @app.get("/api/meta")
    def meta() -> Any:
        """Everything the page needs once: presets, options, drawing style."""
        from cmtool import __version__
        from cmtool.api import available_flexures
        from cmtool.core.provenance import git_commit
        from cmtool.viz.palette import SERIES, STATUS, STRAIN_RAMP_DARK, STRAIN_RAMP_LIGHT

        return {
            "version": __version__,
            "commit": git_commit(),
            "presets": sorted(PRESETS),
            "flexures": available_flexures(),
            "defaults": DesignParams().to_dict(),
            "cad_available": _cad_available(),
            "style": {
                "series": {
                    k: {
                        "label": s.label,
                        "light": s.light,
                        "dark": s.dark,
                        "dash": s.dash,
                        "width": s.width,
                        "marker": s.marker,
                    }
                    for k, s in SERIES.items()
                },
                "rampLight": list(STRAIN_RAMP_LIGHT),
                "rampDark": list(STRAIN_RAMP_DARK),
                "critical": STATUS["critical"],
                "good": STATUS["good"],
                "warning": STATUS["warning"],
                "minFlexureMm": 1.6,
            },
        }

    @app.get("/api/preset/{name}")
    def preset(name: str) -> Any:
        try:
            return preset_params(name).to_dict()
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except FileNotFoundError as exc:
            raise HTTPException(status_code=410, detail=str(exc)) from exc

    @app.post("/api/solve")
    def start_solve(payload: dict[str, Any]) -> Any:
        try:
            params = DesignParams.from_dict(payload)
        except (TypeError, ValueError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        job = engine.submit(params)
        return job.to_dict() | {"cached": job.state == "done"}

    @app.get("/api/job/{job_id}")
    def job_status(job_id: str) -> Any:
        job = engine.job(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="no such job; solve again")
        return job.to_dict()

    @app.get("/api/job/{job_id}/result")
    def job_result(job_id: str) -> Any:
        job = engine.job(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="no such job; solve again")
        if job.state == "error":
            raise HTTPException(status_code=500, detail=job.error or "solve failed")
        if job.result is None:
            raise HTTPException(status_code=409, detail="still running")
        return JSONResponse(job.result)

    @app.post("/api/design.json")
    def design_json(payload: dict[str, Any]) -> Any:
        """Return the linkage as the CLI would read it, for Save design JSON."""
        try:
            params = DesignParams.from_dict(payload)
            linkage = params.to_linkage()
        except (TypeError, ValueError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        body = json.dumps(linkage.to_dict(), indent=2) + "\n"
        return Response(
            body,
            media_type="application/json",
            headers={"Content-Disposition": f'attachment; filename="{params.name}.json"'},
        )

    @app.post("/api/export/{kind}")
    def export(kind: str, payload: dict[str, Any]) -> Any:
        """Build and return an STL, a STEP or a print sheet."""
        if kind not in {"stl", "step", "sheet"}:
            raise HTTPException(status_code=404, detail=f"unknown export {kind!r}")
        if not _cad_available():
            raise HTTPException(
                status_code=503,
                detail="CAD export needs cadquery: uv sync --extra cad",
            )
        try:
            params = DesignParams.from_dict(payload)
            path, media = _build_export(params, kind)
        except (TypeError, ValueError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return FileResponse(path, media_type=media, filename=path.name)

    app.mount("/static", StaticFiles(directory=str(ui_static)), name="static")
    app.mount("/shared", StaticFiles(directory=str(viz_static)), name="shared")
    return app


def _cad_available() -> bool:
    """Whether cadquery is importable, so the page can grey out what it cannot do."""
    from importlib.util import find_spec

    return find_spec("cadquery") is not None


def _build_export(params: DesignParams, kind: str) -> tuple[Path, str]:
    """Build the requested artefact into a temp directory and return its path."""
    import tempfile
    import warnings

    from cmtool.api import convert
    from cmtool.cad import check_printability, export_solid, write_print_sheet
    from cmtool.cad.mechanism import MechanismCadSpec, build_mechanism
    from cmtool.materials import Material, Printer

    linkage = params.to_linkage()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mechanism = convert(
            linkage,
            flexures=params.flexure_type,
            material=params.material,
            printer=params.printer,
            input_range_deg=linkage.input_range_deg,
            thickness_mm=params.thickness_mm,
            placement=params.placement,
            unstressed_at=params.unstressed_at,
        )
        printer_cfg = Printer.load(params.printer)
        material_cfg = Material.load(params.material)
        spec = MechanismCadSpec(link_width_mm=10.0, pen_hole_diameter_mm=5.0)
        solid, layout = build_mechanism(mechanism, spec=spec, printer=printer_cfg)

    out = Path(tempfile.mkdtemp(prefix="cmtool_ui_"))
    stem = params.name or "design"
    if kind in {"stl", "step"}:
        paths = export_solid(solid, out, stem, formats=(kind,))
        return paths[kind], "application/octet-stream"

    thinnest = min(s.geometry.thickness_mm for s in mechanism.sizing.values())
    check = check_printability(solid, printer_cfg, min_feature_mm=thinnest)
    sheet = write_print_sheet(
        out / f"{stem}_print_sheet.md",
        title=f"compliant mechanism {stem}",
        printer=printer_cfg,
        material=material_cfg,
        purpose="Exported from the cmtool UI.",
        check=check,
        details={
            "input arc (deg)": (
                f"{mechanism.input_range_deg[0]:.2f} to {mechanism.input_range_deg[1]:.2f}"
            ),
            "printed unstressed at (deg)": f"{mechanism.reference_input_deg:.2f}",
            "flexure thickness (mm)": thinnest,
            "flexure lengths (mm)": {
                n: round(s.geometry.length_mm, 2) for n, s in mechanism.sizing.items()
            },
            "binding joint": mechanism.feasibility.binding_joint,
            "max utilisation": round(mechanism.feasibility.max_utilisation, 3),
            "PRBM model per joint": mechanism.feasibility.prbm_models,
            "pen hole at coupler point (mm)": (
                f"5.0 dia at ({layout.coupler_pad_mm[0]:.1f}, {layout.coupler_pad_mm[1]:.1f})"
                if layout.coupler_pad_mm
                else "none"
            ),
        },
        instructions=[
            "Print flat on the bed, mechanism plane parallel to it, so the flexures bend in-plane.",
            "No supports. OrcaSlicer, 0.2 mm layers, Arachne wall generator.",
            "Check every flexure under a bright light before flexing anything.",
        ],
    )
    return sheet, "text/markdown"
