"""Gallery of hand-clicked mitoses that mitosis_scan did NOT find, with the reason for each.

For every missed click, re-segment the frames around it and read the object under the click:
  no mask      - the click is on background in every frame near it (segmentation)
  no baseline  - an object is there but has no 30-60 min history (lineage broke)
  not brighter - it has a baseline but never reaches the profile's cond
  too big      - bright enough, but area > profile's area x baseline (merged with a neighbour)
  folded       - passed both cutoffs, but was folded into an event that starts too far away/earlier
Each strip shows the segmentation outlines (green), the object under the click (magenta), and
per-frame cond / area ratio of that object.

    uv run python scripts/build_miss_gallery.py <nd2> <scan out dir> <clicks csv> <out dir> [--channel 1]
"""
import argparse, json, sys
from pathlib import Path

import nd2
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
from skimage.segmentation import find_boundaries

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.mitosis_scan import Profile, segment  # noqa: E402

HALF, SCALE, BEFORE, AFTER = 32, 2, 5, 5


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("nd2"); ap.add_argument("scan"); ap.add_argument("clicks"); ap.add_argument("out")
    ap.add_argument("--channel", type=int, default=1); ap.add_argument("--position", type=int, default=0)
    a = ap.parse_args()
    scan, out = Path(a.scan), Path(a.out)
    (out / "img").mkdir(parents=True, exist_ok=True)
    rep = json.loads((scan / "report.json").read_text())
    p = Profile(**{k: v for k, v in rep["profile"].items()})
    cand = pd.read_csv(scan / "candidates.csv")
    obj = pd.read_csv(scan / "objects.csv.gz").set_index(["frame", "label"])
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
            im = np.asarray(dask[tuple(t if d == "T" else pick[d] for d in dims if d not in ("Y", "X"))])
            cache[t] = (im, segment(im, p, um_px))
        return cache[t]

    cl = cl[cl.frame < n]
    near = lambda c, r: (abs(minutes[int(r.first)] - minutes[c.frame]) <= p.fold_min * 2
                         and np.hypot(r.x - c.x, r.y - c.y) * um_px < p.fold_um * 1.3)
    misses = [c for c in cl.itertuples() if not any(near(c, r) for r in cand.itertuples())]

    items = []
    for c in misses:
        ts = list(range(max(0, c.frame - BEFORE), min(n, c.frame + AFTER + 1)))
        per = []
        for t in ts:
            im, lab = frame(t)
            l = int(lab[c.y, c.x])
            if l and (t, l) in obj.index:
                o = obj.loc[(t, l)]
                per.append((t, l, o.cond, o.aratio))
            else:
                per.append((t, l, np.nan, np.nan))
        core = [r for r in per if abs(r[0] - c.frame) <= 3]
        if not any(r[1] for r in core):
            why = "no mask"
        elif all(np.isnan(r[2]) for r in core if r[1]):
            why = "no baseline"
        elif not any(r[2] >= p.cond for r in core if r[1]):
            why = "not brighter"
        elif not any(r[2] >= p.cond and r[3] <= p.area for r in core if r[1]):
            why = "too big"
        else:
            why = "folded"

        w = 2 * HALF * SCALE
        img = Image.new("RGB", (w * len(ts), w + 30), (20, 20, 20))
        d = ImageDraw.Draw(img)
        crops = []
        for t in ts:
            im, lab = frame(t)
            y0, x0 = c.y - HALF, c.x - HALF
            pad = lambda arr: np.pad(arr, HALF)[y0 + HALF:y0 + 3 * HALF, x0 + HALF:x0 + 3 * HALF]
            crops.append((pad(im.astype(float)), pad(lab)))
        allpx = np.stack([cr[0] for cr in crops])
        lo, hi = np.percentile(allpx, 1), np.percentile(allpx, 99.8)
        for i, ((g, lb), (t, l, cnd, ar)) in enumerate(zip(crops, per)):
            g = (np.clip((g - lo) / max(hi - lo, 1), 0, 1) * 255).astype(np.uint8)
            rgb = np.stack([g, g, g], -1)
            rgb[find_boundaries(lb, mode="inner")] = (60, 200, 90)
            if l:
                rgb[find_boundaries(lb == l, mode="inner")] = (255, 60, 220)
            tile = Image.fromarray(rgb).resize((w, w), Image.NEAREST)
            img.paste(tile, (i * w, 16))
            clicked = t == c.frame
            d.text((i * w + 4, 2), f"f{t + 1}", fill=(255, 200, 60) if clicked else (170, 170, 170))
            if clicked:
                d.rectangle([i * w, 16, i * w + w - 1, w + 15], outline=(255, 200, 60), width=2)
            txt = "no obj" if not l else ("no base" if np.isnan(cnd) else f"{cnd:.2f}x  a{ar:.2f}")
            ok = l and not np.isnan(cnd) and cnd >= p.cond and ar <= p.area
            d.text((i * w + 4, w + 17), txt, fill=(120, 230, 120) if ok else (200, 200, 200))
            m = w // 2
            d.line([i * w + m - 10, 16 + m, i * w + m - 5, 16 + m], fill=(80, 200, 255))
            d.line([i * w + m + 5, 16 + m, i * w + m + 10, 16 + m], fill=(80, 200, 255))
        name = f"click{c.click:03d}.jpg"
        img.save(out / "img" / name, quality=88)
        items.append({"click": int(c.click), "frame": int(c.frame) + 1, "x": int(c.x), "y": int(c.y),
                      "why": why, "img": f"img/{name}"})
    f.close()

    df = pd.DataFrame(items)
    df.drop(columns="img").to_csv(out / "misses.csv", index=False)
    counts = df.why.value_counts().to_dict() if len(df) else {}
    order = ["no mask", "no baseline", "not brighter", "too big", "folded"]
    field = Path(a.nd2).stem
    cards = "\n".join(
        f'<div class="card"><div class="meta"><b>{it["why"]}</b> &middot; click {it["click"]} &middot; frame {it["frame"]}'
        f' &middot; px ({it["x"]}, {it["y"]})</div><div class="strip"><img loading="lazy" src="{it["img"]}"></div></div>'
        for it in sorted(items, key=lambda it: (order.index(it["why"]), it["frame"])))
    summary = ", ".join(f"{k}: {counts.get(k, 0)}" for k in order)
    (out / "index.html").write_text(f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>Missed clicks: {field}</title>
<style>body{{background:#121212;color:#e8e8e8;font:14px system-ui,sans-serif;margin:0;padding:0 16px 60px}}
header{{position:sticky;top:0;background:#121212;padding:10px 0;border-bottom:1px solid #333}}h1{{font-size:16px;margin:0 0 4px}}
.sum{{color:#999;font-size:12px}}.card{{background:#1d1d1d;margin:12px 0;padding:8px;border-radius:6px}}
.meta{{color:#bbb;font-size:12px;margin-bottom:4px}}.strip{{overflow-x:auto}}.strip img{{display:block;max-width:none}}</style></head>
<body><header><h1>Missed clicks: {field}</h1>
<div class="sum">{len(items)} of {len(cl)} clicks had no candidate. {summary}.<br>
Green = segmentation outlines, magenta = object under the click, yellow box = clicked frame.
Under each frame: that object's brightness vs its 30-60 min baseline and area ratio (cutoffs: &ge;{p.cond}x, area &le;{p.area}).</div>
</header>{cards}</body></html>""", encoding="utf-8")
    print(f"{len(items)} misses: {summary}")


if __name__ == "__main__":
    main()
