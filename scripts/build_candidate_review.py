"""Build a yes/no review page for the mitosis_scan candidates that matched NO hand click.

Each one is either a false alarm (measures precision) or a mitosis the clicker missed (measures how
complete the clicks are). Every candidate gets two lossless PNG frame series, from BEFORE frames ahead
of its first hit to AFTER frames past its last, so the whole prophase -> plate -> daughters sequence is
there:
  s<frame>.png  close crop (2*CLOSE px across) shown as a tile grid on the page
  x<frame>.png  wide crop (2*WIDE px across) shown in the lightbox, to see the neighbours
Pixels are written at native resolution and scaled up by the browser (pixelated), so there is no
resampling blur or JPEG noise; annotations are drawn by the page, not burned in.

    uv run python scripts/build_candidate_review.py <nd2> <scan out dir> <clicks csv> <out dir> [--channel 1]

Matching uses the same rule as mitosis_scan.recall (within fold_min*2 minutes and fold_um*1.3 um).
"""
import argparse, json
from pathlib import Path

import nd2
import numpy as np
import pandas as pd
from PIL import Image

CLOSE = 40     # half-width px of the tile crop (~70 um across at 0.86 um/px)
WIDE = 128     # half-width px of the lightbox crop (~220 um across)
BEFORE, AFTER = 8, 25   # frames (80 min before the first hit, ~4 h after the last at 10-min frames;
                        # 10 after was too few to tell a plate that resolves from one that never divides)


def crop(im, x, y, half):
    y0, x0 = int(round(y)) - half, int(round(x)) - half
    c = np.zeros((2 * half, 2 * half), im.dtype)
    sy = slice(max(0, y0), min(im.shape[0], y0 + 2 * half))
    sx = slice(max(0, x0), min(im.shape[1], x0 + 2 * half))
    c[sy.start - y0:sy.stop - y0, sx.start - x0:sx.stop - x0] = im[sy, sx]
    return c.astype(float)


def to8(a, lo, hi):
    return Image.fromarray((np.clip((a - lo) / max(hi - lo, 1), 0, 1) * 255).astype(np.uint8))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("nd2"); ap.add_argument("scan"); ap.add_argument("clicks"); ap.add_argument("out")
    ap.add_argument("--channel", type=int, default=1); ap.add_argument("--position", type=int, default=0)
    ap.add_argument("--all", action="store_true", help="include candidates that matched a click too")
    ap.add_argument("--prior", help="an exported review CSV: its notes are shown on each card")
    ap.add_argument("--only", help="with --prior: review only rows whose note contains this text (e.g. 'more frame')")
    a = ap.parse_args()
    scan, out = Path(a.scan), Path(a.out)
    rep = json.loads((scan / "report.json").read_text())
    p = rep["profile"]
    cand = pd.read_csv(scan / "candidates.csv").reset_index(names="cand")
    cl = pd.read_csv(a.clicks)

    f = nd2.ND2File(a.nd2)
    dims, dask = list(f.sizes), f.to_dask()
    um_px, n = f.voxel_size().x, f.sizes["T"]
    t_s = pd.DataFrame(f.events()).drop_duplicates("T Index")["Time [s]"].to_numpy()[:n]
    minutes = (t_s - t_s[0]) / 60
    pick = {"P": a.position, "C": a.channel}
    cache = {}

    def frame(t):
        if t not in cache:
            if len(cache) > 64:
                cache.clear()
            cache[t] = np.asarray(dask[tuple(t if d == "T" else pick[d] for d in dims if d not in ("Y", "X"))])
        return cache[t]

    cl = cl[cl.frame < n]
    def matched(r):
        return any(abs(minutes[r.first] - minutes[c.frame]) <= p["fold_min"] * 2
                   and np.hypot(r.x - c.x, r.y - c.y) * um_px < p["fold_um"] * 1.3 for c in cl.itertuples())
    cand["matched"] = [matched(r) for r in cand.itertuples()]
    todo = cand if a.all else cand[~cand.matched]
    prior = {}
    if a.prior:
        pr = pd.read_csv(a.prior).fillna("")
        prior = dict(zip(pr.cand, pr.note))
        if a.only:
            todo = todo[todo.cand.isin(pr[pr.note.str.lower().str.contains(a.only.lower())].cand)]
    # frame order keeps the ND2 reads local (the cache), the page re-sorts by peak
    items = []
    for r in todo.sort_values("first").itertuples():
        ts = list(range(max(0, r.first - BEFORE), min(n, r.last + AFTER + 1)))
        d = out / "img" / f"c{r.cand:04d}"
        d.mkdir(parents=True, exist_ok=True)
        close = [crop(frame(t), r.x, r.y, CLOSE) for t in ts]
        wide = [crop(frame(t), r.x, r.y, WIDE) for t in ts]
        # contrast from the close crops: the candidate, not a bright cell across the wide field, sets it
        lo, hi = np.percentile(np.stack(close), [1, 99.95])
        for t, s, w in zip(ts, close, wide):
            to8(s, lo, hi).save(d / f"s{t + 1}.png", optimize=True)
            to8(w, lo, hi).save(d / f"x{t + 1}.png", optimize=True)
        items.append({"id": int(r.cand), "dir": f"img/c{r.cand:04d}", "frames": [t + 1 for t in ts],
                      "first": int(r.first) + 1, "last": int(r.last) + 1,
                      "x": round(r.x), "y": round(r.y), "x_um": round(r.x * um_px, 1), "y_um": round(r.y * um_px, 1),
                      "peak": round(r.peak, 2), "matched": bool(r.matched),
                      "prior": str(prior.get(r.cand, "")).strip()})
    f.close()
    items.sort(key=lambda it: -it["peak"])
    field = Path(a.nd2).stem
    todo.to_csv(out / "review_items.csv", index=False)
    html = (Path(__file__).with_name("candidate_review_template.html").read_text(encoding="utf-8")
            .replace("__PAGEKEY__", f"{field}:{out.name}")
            .replace("__FIELD__", field).replace("__ITEMS__", json.dumps(items))
            .replace("__CLOSE__", str(CLOSE)).replace("__WIDE__", str(WIDE))
            .replace("__UMPX__", f"{um_px:.4f}")
            .replace("__SUMMARY__", f"{len(cand)} candidates, {int(cand.matched.sum())} matched a click, "
                                    f"{int((~cand.matched).sum())} did not; {len(cl)} clicks"))
    (out / "index.html").write_text(html, encoding="utf-8")
    print(f"{len(items)} items -> {out / 'index.html'}")


if __name__ == "__main__":
    main()
