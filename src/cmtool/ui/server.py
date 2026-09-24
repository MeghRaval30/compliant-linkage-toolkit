"""Starting the local UI: bind a loopback port, open a browser, serve.

Everything here is about the one requirement that matters for a demo -- it has
to come up with one command on a machine you have not seen before, and it has to
keep working when the wifi does not.
"""

from __future__ import annotations

import socket
import threading
import webbrowser
from typing import Any

#: Bound to loopback only. This serves a filesystem and builds CAD on request;
#: it has no authentication and is not meant to be reachable from anywhere else.
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8765


def find_port(host: str = DEFAULT_HOST, preferred: int = DEFAULT_PORT) -> int:
    """Return a usable port, preferring ``preferred``.

    Falls back to whatever the OS gives rather than failing, because "port 8765
    is busy" is not a useful thing to discover in front of an audience.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            probe.bind((host, preferred))
        except OSError:
            pass
        else:
            return preferred
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind((host, 0))
        return int(probe.getsockname()[1])


def serve(
    *,
    host: str = DEFAULT_HOST,
    port: int | None = None,
    open_browser: bool = True,
    log_level: str = "warning",
) -> None:
    """Run the UI until interrupted.

    Parameters
    ----------
    open_browser
        Open the default browser once the server is listening. The delay is
        deliberate: opening before uvicorn binds gives a connection-refused page
        and someone reloading in front of the room.
    """
    try:
        import uvicorn
    except ModuleNotFoundError as exc:  # pragma: no cover - dependency guard
        raise ModuleNotFoundError(
            "the local UI needs FastAPI and uvicorn: install them with\n"
            "    uv sync --extra ui\n"
            "or  pip install 'cmtool[ui]'"
        ) from exc

    from cmtool.ui.app import create_app

    chosen = port or find_port(host)
    url = f"http://{host}:{chosen}/"

    if open_browser:
        threading.Timer(1.0, lambda: _open(url)).start()

    config = uvicorn.Config(
        create_app(),
        host=host,
        port=chosen,
        log_level=log_level,
        access_log=False,
    )
    uvicorn.Server(config).run()


def _open(url: str) -> Any:
    """Open a browser, ignoring the case where there is no browser to open."""
    try:
        return webbrowser.open(url)
    except Exception:  # pragma: no cover - headless machines
        return None
