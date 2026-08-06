"""Local stdio MCP server: a filesystem over time-lapse microscopy pixels.

Read-only except for annotate(), which appends to a CSV of its own. Serves a bundle
built by scripts/build_bundle.py -- indexed frame PNGs, PNG-16 label maps, a
per-frame track table, and a manifest carrying calibration read from the ND2 at
build time. Nothing here touches an ND2, a GPU, torch, or Cellpose, so the install
stays pure-python.

Point it at a bundle with the CELL_BUNDLE_DIR environment variable.

Docstrings and type hints ARE the schema the model sees, so they are written for
someone who has never used a microscope.

This file is a thin entrypoint. The implementation lives in cell_mcp_server/,
split into server.py (the MCPServer object + shared constants), io.py (bundle
loading/caching), render.py (crops/tiles/filmstrips), and tools_*.py (the actual
@server.tool() functions, grouped by what they're for). Everything is re-exported
here so `import cell_mcp; cell_mcp.whatever` keeps working -- tests and scripts
that reach into private helpers (e.g. cell_mcp._manifest) are unaffected.
"""

import sys

# Run directly (as the MCP launcher does: `python cell_mcp.py`), this module loads
# as `__main__`, not `cell_mcp` -- so cell_mcp_server/io.py's `import cell_mcp as _cm`
# (needed so tests can monkeypatch cell_mcp._manifest etc. and have it take effect
# everywhere) finds nothing cached under that name and re-executes this whole file as
# a second, independent module. That second execution re-enters this same import
# chain and hits render.py's `from .io import _frame_at_offset_min` while io.py is
# still mid-import -- a circular-import crash that only shows up when run as a script,
# never when imported normally (e.g. by tests). Registering the real name up front
# lets io.py's import find this module instead of re-running it.
if __name__ == "__main__":
    sys.modules.setdefault("cell_mcp", sys.modules[__name__])

from cell_mcp_server.server import (
    server, BUNDLE, MAX_IMAGES, MAX_IMAGES_PAGE,
    _WINDOW_BEFORE_MIN, _WINDOW_AFTER_MIN, _STRIDE_MIN, _UPSCALE_TO, _HDR_SEP,
)
from cell_mcp_server.io import *
from cell_mcp_server.render import *
from cell_mcp_server.tools_browse import *
from cell_mcp_server.tools_candidates import *
from cell_mcp_server.tools_filmstrip import *
from cell_mcp_server.tools_output import *

if __name__ == "__main__":
    if not BUNDLE.is_dir():
        print(f"warning: CELL_BUNDLE_DIR={BUNDLE} does not exist", file=sys.stderr)
    server.run(transport="stdio")
