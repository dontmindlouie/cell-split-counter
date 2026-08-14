"""open_in_fiji: launch Fiji pointed at the exact frame/channel/location a report
page or filmstrip is already showing, so a researcher cross-checking a call
against the raw .nd2 doesn't have to manually scrub sliders to find it.

Split out as its own module (rather than folded into tools_output.py) because it
shells out to an external application and resolves paths outside the bundle --
a different failure mode (missing exe, missing raw file, OOM in Fiji itself)
than every other tool here, which only ever touches the bundle it was given.
"""

import os
import subprocess
import tempfile
from pathlib import Path

from .server import server

import cell_mcp_server as _cm

# Same env-var-with-default pattern as server.py's BUNDLE / src/config.py's
# SHARED_DATA_DIR: a real default for this machine, overridable for anyone whose
# Fiji or raw-data layout differs.
FIJI_EXE = Path(os.environ.get("CELL_MCP_FIJI_EXE", r"G:\Fiji\fiji-windows-x64.exe"))
RAW_ND2_ROOT = Path(os.environ.get("CELL_MCP_RAW_ND2_ROOT", r"G:\Projects"))


def _find_raw_nd2(well: str) -> Path:
    """Locate the raw .nd2 for `well` under RAW_ND2_ROOT.

    Matched by prefix, not exact name -- wells are bundle/run names (e.g.
    "20251016_ACTB_M2"), but the raw file on disk sometimes carries a suffix the
    bundle name drops (e.g. "20251016_ACTB_M2_red.nd2"). Raw files live scattered
    across per-experiment subfolders under the root, not one flat directory, so
    this searches recursively.
    """
    matches = sorted(RAW_ND2_ROOT.rglob(f"{well}*.nd2"))
    if not matches:
        raise ValueError(
            f"no .nd2 found for well {well!r} under {RAW_ND2_ROOT} "
            f"(searched recursively for '{well}*.nd2'). If the raw file lives "
            f"elsewhere, set CELL_MCP_RAW_ND2_ROOT."
        )
    if len(matches) > 1:
        raise ValueError(
            f"multiple .nd2 files match well {well!r} under {RAW_ND2_ROOT}: "
            f"{[str(m) for m in matches]} -- ambiguous, narrow CELL_MCP_RAW_ND2_ROOT."
        )
    return matches[0]


@server.tool()
def open_in_fiji(well: str, frame: int, cx: int | None = None,
                  cy: int | None = None, crop_um: float = 60.0) -> str:
    """Launch Fiji, opened directly on a specific frame (and, if given, a specific
    location) of a well's raw .nd2 -- so a call from a report page or filmstrip can
    be cross-checked against the real file without hand-scrubbing sliders to find
    the same spot.

    Opens as a virtual stack (reads planes from disk on demand) rather than
    loading the whole series into RAM -- several of these files are 10GB+ and will
    hit a Java heap OutOfMemoryError otherwise (seen firsthand on ACTB_M3, a 13.8GB
    series). Opens in Composite color mode with all channels visible, matching
    what a multi-channel bundle's own composite render shows.

    Args:
        well: well name from list_wells() -- also used to find the matching raw
            .nd2 under CELL_MCP_RAW_ND2_ROOT (default G:\\Projects).
        frame: 0-indexed frame number, as shown in a filmstrip label ("f428").
            Converted to Fiji's 1-indexed T position internally.
        cx, cy: pixel coordinates, as shown in a filmstrip label ("@(1055, 427)").
            When given, draws a rectangular ROI of width `crop_um` centred there
            and zooms the view to it -- the same region a report page's crop shows.
            Omit both to just land on the right frame, full field of view.
        crop_um: width of the ROI/zoom box in micrometres, when cx/cy are given.
            Defaults to 60.0, matching show_cells_in_browser's report crop.

    Returns a confirmation line once Fiji has been launched (does not wait for the
    user to close it).
    """
    if not FIJI_EXE.is_file():
        raise ValueError(
            f"Fiji executable not found at {FIJI_EXE}. Set CELL_MCP_FIJI_EXE to "
            f"its actual path."
        )
    nd2_path = _find_raw_nd2(well)
    m = _cm._manifest(well)
    n_frames = m["n_frames"]
    if not (0 <= frame < n_frames):
        raise ValueError(f"frame {frame} not in {well} (has frames 0-{n_frames - 1})")
    t_pos = frame + 1  # Fiji/ImageJ stack positions are 1-indexed

    n_channels = m["nd2_sizes"]["C"]
    macro_lines = [
        f'open_path = "{nd2_path.as_posix()}";',
        'run("Bio-Formats Importer", "open=[" + open_path + "] autoscale color_mode=Composite '
        'view=Hyperstack stack_order=XYCZT use_virtual_stack");',
        f'Stack.setPosition(1, 1, {t_pos});',
        # Bio-Formats' own import-time autoscale is a single heuristic applied across
        # all channels -- fine for a bright one, but leaves a dim/sparse channel
        # (e.g. punctate signal against a mostly-black field) scaled so its real
        # signal sits below visible black, i.e. looks like the channel is off even
        # though its checkbox is ticked. Re-autoscale each channel individually,
        # same as clicking Auto in the B&C dialog once per channel by hand.
    ] + [
        f'Stack.setChannel({c}); run("Enhance Contrast", "saturated=0.35");'
        for c in range(1, n_channels + 1)
    ] + ['Stack.setChannel(1);']  # leave the slider back where a fresh open would land
    if cx is not None and cy is not None:
        um_px = m["pixel_size_um"]
        half_px = max(8, int(round(crop_um / um_px / 2)))
        x0 = max(0, int(cx) - half_px)
        y0 = max(0, int(cy) - half_px)
        w = h = half_px * 2
        macro_lines += [
            f'makeRectangle({x0}, {y0}, {w}, {h});',
            'run("To Selection");',  # Image > Zoom > To Selection -- actually zooms in,
                                      # unlike "To Bounding Box" (resizes the window only)
        ]

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".ijm", delete=False, encoding="utf-8"
    ) as f:
        f.write("\n".join(macro_lines) + "\n")
        macro_path = f.name

    subprocess.Popen([str(FIJI_EXE), "-macro", macro_path])
    loc = f" at ({cx}, {cy})" if cx is not None else ""
    return (f"Launched Fiji on {nd2_path.name}, frame {frame}{loc}, "
            f"crop_um={crop_um}.")


__all__ = ["FIJI_EXE", "RAW_ND2_ROOT", "open_in_fiji"]
