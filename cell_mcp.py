"""Local stdio MCP server: a filesystem over time-lapse microscopy pixels.

Read-only except for annotate(), which appends to a CSV of its own. Serves a bundle
built by scripts/build_bundle.py -- indexed frame PNGs, PNG-16 label maps, a
per-frame track table, and a manifest carrying calibration read from the ND2 at
build time. Nothing here touches an ND2, a GPU, torch, or Cellpose, so the install
stays pure-python.

Point it at a bundle with the CELL_BUNDLE_DIR environment variable.

Docstrings and type hints ARE the schema the model sees, so they are written for
someone who has never used a microscope.

This file is a thin launcher. The implementation lives in cell_mcp_server/, split
into server.py (the MCPServer object + shared constants), io.py (bundle
loading/caching), render.py (crops/tiles/filmstrips), and tools_*.py (the actual
@server.tool() functions, grouped by what they're for). cell_mcp_server/__init__.py
re-exports all of it, and this file re-exports that in turn, so `import cell_mcp;
cell_mcp.whatever` keeps working -- tests and scripts that reach into private
helpers (e.g. cell_mcp._manifest) are unaffected.
"""

import sys

from cell_mcp_server import *
from cell_mcp_server import server, BUNDLE

if __name__ == "__main__":
    if not BUNDLE.is_dir():
        print(f"warning: CELL_BUNDLE_DIR={BUNDLE} does not exist", file=sys.stderr)
    server.run(transport="stdio")
