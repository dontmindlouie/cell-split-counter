"""Fiji pre-clicks: put the detector's mitosis candidates on the ORIGINAL ND2 in Fiji, as
ROI Manager markers the researcher keeps or deletes, instead of clicking from scratch.

The researcher's own platform and the full raw movie, not our crops -- the review is of
the detector, so it happens where she already trusts what she sees (2026-10-28: "doesn't
trust the AI despite the numbers").

    uv run python scripts/fiji_preclick.py <well> --nd2 <path to the ND2> [--scan DIR] [--out DIR]

Writes two macros (drag each onto Fiji):
    <well>_preclick.ijm     opens the ND2 (asks for it if the path has moved) and loads one
                            yellow circle per candidate, on the candidate's first plate frame,
                            named "<rank> c<cand> <hints>" -- likely errors (fate hints) first,
                            then by plate brightness. Double-click a name to jump to it.
    save_preclick_review.ijm   writes whatever is in the ROI Manager to a CSV in Downloads:
                            kept circles = confirmed, deleted = rejected, added points =
                            mitoses the detector missed, renamed ones carry her note.

Coordinates are the ND2's own pixels and Fiji frames are 1-based (bundle frame + 1).
"""
import argparse, json, sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from review_events import suggestions  # noqa: E402

SAVE_MACRO = r"""// Save the ROI Manager to a CSV for cell-split-counter (pre-click review).
// Circles still here = kept candidates; points = mitoses added by hand; names carry notes.
title = getTitle();
dir = getDirectory("home") + "Downloads" + File.separator;
path = dir + "preclick_review_" + replace(title, "[^A-Za-z0-9_.-]", "_") + ".csv";
n = roiManager("count");
s = "name,x,y,fiji_frame,shape\n";
for (i = 0; i < n; i++) {
    roiManager("select", i);
    name = replace(Roi.getName, ",", ";");
    Roi.getPosition(c, z, t);
    if (selectionType() == 10) {   // point / multipoint: one row per point
        getSelectionCoordinates(xs, ys);
        for (j = 0; j < xs.length; j++)
            s = s + name + "," + xs[j] + "," + ys[j] + "," + t + ",point\n";
    } else {
        Roi.getBounds(x, y, w, h);
        s = s + name + "," + (x + w / 2) + "," + (y + h / 2) + "," + t + ",circle\n";
    }
}
File.saveString(s, path);
showMessage("Saved " + n + " ROIs to\n" + path);
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("well")
    ap.add_argument("--nd2", required=True, help="the original ND2 (any path; the macro asks if it moved)")
    ap.add_argument("--scan", help="mitosis_scan output dir (default data/mitosis_scan/<well>)")
    ap.add_argument("--out", help="output dir (default data/fiji_preclick/<well>)")
    ap.add_argument("--circle-um", type=float, default=20.0)
    a = ap.parse_args()
    scan = Path(a.scan or REPO / "data" / "mitosis_scan" / a.well)
    out = Path(a.out or REPO / "data" / "fiji_preclick" / a.well)
    out.mkdir(parents=True, exist_ok=True)
    um = float(json.loads((scan / "report.json").read_text(encoding="utf-8"))["um_px"])
    c = pd.read_csv(scan / "fate.csv") if (scan / "fate.csv").exists() else \
        pd.read_csv(scan / "candidates.csv").reset_index(names="cand")
    c["hint"] = [suggestions(r) for r in c.to_dict("records")]
    c = c.sort_values(["hint", "peak"], ascending=[True, False], key=lambda s: s.eq("") if s.name == "hint" else s)
    d = max(4, round(a.circle_um / um))
    nd2 = str(Path(a.nd2).resolve()).replace("\\", "/")
    lines = [
        f"// Pre-clicks for {a.well}: {len(c)} detector candidates (cell-split-counter).",
        "// Delete the ones that aren't mitoses, add any it missed (point tool, then Add [t]),",
        "// rename to note an error (e.g. add ' cytokinesis failure'). Then run save_preclick_review.ijm.",
        f'p = "{nd2}";',
        'if (!File.exists(p)) p = File.openDialog("Open the original ND2 for ' + a.well + '");',
        'run("Bio-Formats Importer", "open=[" + p + "] autoscale color_mode=Composite view=Hyperstack stack_order=XYCZT");',
        'roiManager("reset");',
        'setBatchMode(true);',
    ]
    for rank, r in enumerate(c.itertuples(), 1):
        name = f"{rank:03d} c{int(r.cand)}" + (f" {r.hint}" if r.hint else "")
        lines += [f"makeOval({round(r.x - d / 2)}, {round(r.y - d / 2)}, {d}, {d});",
                  f"Roi.setPosition(0, 1, {int(r.first) + 1}); Roi.setStrokeColor(\"yellow\");",
                  f'roiManager("add"); roiManager("select", {rank - 1}); roiManager("rename", "{name}");']
    lines += ['setBatchMode(false);', 'roiManager("deselect");',
              'roiManager("Associate", "true"); roiManager("UseNames", "true");',
              'roiManager("Show All with labels");']
    mac = out / f"{a.well}_preclick.ijm"
    mac.write_text("\n".join(lines) + "\n", encoding="utf-8")
    (out / "save_preclick_review.ijm").write_text(SAVE_MACRO, encoding="utf-8")
    c.assign(rank=range(1, len(c) + 1), fiji_frame=c["first"] + 1)[
        ["rank", "cand", "fiji_frame", "x", "y", "peak", "hint"]].to_csv(out / f"{a.well}_preclick_key.csv", index=False)
    print(f"{len(c)} candidates ({int((c.hint != '').sum())} with error hints, listed first) -> {mac}")


if __name__ == "__main__":
    main()
