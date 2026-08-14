"""Flat namespace for the cell-microscopy MCP server package.

Collects every submodule's public API (including the underscore-prefixed private
helpers -- see each submodule's own __all__) into `cell_mcp_server` itself, so
`import cell_mcp_server; cell_mcp_server.whatever` works from outside the package,
and so each submodule can self-import the package (`import cell_mcp_server as
_cm`) to reach mutable state and sibling functions dynamically -- see the note at
the top of io.py for why that indirection exists (it's what makes
`monkeypatch.setattr` visible everywhere the patched name is used).

That self-import is safe specifically BECAUSE it targets this package's own real,
permanent dotted name. Python registers `cell_mcp_server` in sys.modules before
running this file, so a submodule importing it mid-init always gets back the
(partially built) module already in progress -- never a re-execution. The bug
this replaced came from submodules instead self-importing the flat top-level
entrypoint script (`cell_mcp.py`) by its bare name: run directly, that script
loads as `__main__`, not `cell_mcp`, so the self-import found nothing cached and
re-ran the whole file, recursing back into this same import chain. Self-importing
the package instead of the launcher script means this can't happen no matter what
the launcher is called or how it's invoked.
"""

# Aliased: server.py's own __all__ includes the name "server" (the MCPServer
# instance), which would collide with the submodule name below and silently shadow
# one or the other depending on import order.
from . import server as _server_mod, io as _io_mod, render as _render_mod
from . import tools_browse as _tools_browse_mod, tools_candidates as _tools_candidates_mod
from . import tools_filmstrip as _tools_filmstrip_mod, tools_output as _tools_output_mod
from . import tools_fiji as _tools_fiji_mod

from .server import *
from .io import *
from .render import *
from .tools_browse import *
from .tools_candidates import *
from .tools_filmstrip import *
from .tools_output import *
from .tools_fiji import *

__all__ = (
    _server_mod.__all__ + _io_mod.__all__ + _render_mod.__all__ + _tools_browse_mod.__all__
    + _tools_candidates_mod.__all__ + _tools_filmstrip_mod.__all__ + _tools_output_mod.__all__
    + _tools_fiji_mod.__all__
)
