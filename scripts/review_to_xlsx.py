"""Exported review CSVs (show_cells_in_browser "Export CSV") -> one Excel workbook.

    uv run python scripts/review_to_xlsx.py <out.xlsx> <review.csv> [<review.csv> ...]
        [--events <events.csv> ...] [--sheet-by crop] [--verdict verdict --yes division]

Each review CSV row is one card. With --events (scripts/review_events.py output) the
rows get the card's source, Fiji frame, real minutes and suggestions back by event_id.
Sheets: one per --sheet-by group (default: the crop, from the well name), rows sorted by
field then frame; then a "summary" sheet of per-error percentages over the cards marked
as divisions, by day group (D0/D2/D4 from the well name) and by any pick-one field such as
HLA-G. Tick boxes stay 1/blank so Excel can sum them.

Refuses to overwrite an existing workbook: a sheet someone has edited by hand is not
regenerated over. Write to a new name instead.
"""
import argparse, re
from pathlib import Path

import pandas as pd

ID_COLS = {"event", "event_id", "label", "kind", "track_ids", "x", "y", "start_frame", "end_frame", "marked"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out"); ap.add_argument("reviews", nargs="+")
    ap.add_argument("--events", nargs="*", default=[])
    ap.add_argument("--sheet-by", default="crop", choices=["crop", "field", "day"])
    ap.add_argument("--verdict", default="verdict"); ap.add_argument("--yes", default="division")
    a = ap.parse_args()
    out = Path(a.out)
    if out.exists():
        raise SystemExit(f"{out} exists -- not overwriting a workbook that may have been edited; pick a new name")

    rv = pd.concat([pd.read_csv(p, encoding="utf-8-sig", dtype={"event_id": str}) for p in a.reviews],
                   ignore_index=True)
    fields = [c for c in rv.columns if c not in ID_COLS]
    if a.events:
        ev = pd.concat([pd.read_csv(p, dtype={"event_id": str}) for p in a.events], ignore_index=True)
        keep = [c for c in ["event_id", "source", "fiji_frame", "minutes", "x_um", "y_um", "suggest", "ids"] if c in ev]
        rv = rv.merge(ev[keep], on="event_id", how="left")
    rv["field"] = rv.event_id.str.split("#").str[0]
    rv["crop"] = rv.field.str.extract(r"(crop\d+)", expand=False).fillna(rv.field)
    rv["day"] = rv.field.str.extract(r"_(D\d)_", expand=False).fillna("")
    sort = ["field"] + (["fiji_frame"] if "fiji_frame" in rv else ["start_frame"])
    front = [c for c in ["event_id", "field", "day", "source", "fiji_frame", "minutes", "x_um", "y_um", "suggest"] if c in rv]
    rv = rv[front + fields + [c for c in rv.columns if c not in front + fields]].sort_values(sort)

    checks = [c for c in fields if c != "notes" and set(rv[c].dropna().unique()) <= {0, 1, 1.0, 0.0}]
    picks = [c for c in fields if c not in checks and c not in ("notes", a.verdict)
             and not pd.api.types.is_numeric_dtype(rv[c])]
    div = rv[rv.get(a.verdict, pd.Series(dtype=object)) == a.yes] if a.verdict in rv else rv
    summary = []
    for by in ["day"] + picks:
        for g, d in div.groupby(by):
            row = {"grouped by": by, "group": g, "divisions": len(d)}
            row.update({f"% {c}": round(100 * d[c].fillna(0).astype(float).mean(), 1) for c in checks})
            summary.append(row)
    with pd.ExcelWriter(out, engine="openpyxl") as xw:
        for g, d in rv.groupby(a.sheet_by):
            d.drop(columns=["crop"]).to_excel(xw, sheet_name=re.sub(r"[\[\]:*?/\\]", "_", str(g))[:31] or "all", index=False)
        pd.DataFrame(summary).to_excel(xw, sheet_name="summary", index=False)
    print(f"{len(rv)} cards ({len(div)} marked {a.yes!r}) -> {out}")


if __name__ == "__main__":
    main()
