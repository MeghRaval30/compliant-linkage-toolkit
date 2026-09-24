"""The local single-page UI: `cmtool ui`.

Local only and offline by construction. The page, its stylesheet and its one
script ship inside the package; nothing is fetched from the network, and the
server binds loopback. It is meant to be started on a laptop in front of people
and to keep working with the wifi off.

:mod:`cmtool.ui.model` is the whole of the logic and imports nothing web-shaped,
so the solve path is testable without starting a server. :mod:`cmtool.ui.app`
adds HTTP, jobs and a result cache; :mod:`cmtool.ui.server` binds a port and
opens a browser.
"""

from cmtool.ui.model import PRESETS, DesignParams, preset_params, solve

__all__ = ["PRESETS", "DesignParams", "preset_params", "solve"]
