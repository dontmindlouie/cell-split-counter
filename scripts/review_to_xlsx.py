"""Exported review CSVs (show_cells_in_browser "Export CSV") -> one Excel workbook.

    uv run python scripts/review_to_xlsx.py <out.xlsx> <review.csv> [<review.csv> ...]
        [--events <events.csv> ...] [--group NAME=REGEX ...] [--sheet-by field|NAME]
        [--verdict verdict --yes division]

Each review CSV row is one card. With --events (scripts/review_events.py output) the
rows get the card's source, Fiji frame, real minutes and suggestions back by event_id.
Sheets: one per --sheet-by value (default: the field, i.e. the well), rows sorted by field
then frame; then a "summary" sheet of per-tick-box percentages over the cards whose
--verdict is --yes, broken down by every pick-one field (e.g. HLA-G) and by any --group.

--group derives a column from the well name with one regex capture group, so grouping by
experiment-specific naming is opt-in rather than built in -- for the 20260822 EVT crops:
    --group day=_(D\\d)_ --group crop=(crop\\d+) --sheet-by crop
Tick boxes stay 1/blank so Excel can sum them.

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
    ap.add_argument("--group", action="append", default=[], help="NAME=REGEX on the well name (one capture group)")
    ap.add_argument("--sheet-by", default="field", help="field, or the NAME of a --group")
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
    groups = []
    for g in a.group:
        name, rx = g.split("=", 1)
        rv[name] = rv.field.str.extract(rx, expand=False).fillna("")
        groups.append(name)
    if a.sheet_by not in ["field"] + groups:
        raise SystemExit(f"--sheet-by {a.sheet_by!r}: use field or one of the --group names {groups}")
    sort = ["field"] + (["fiji_frame"] if "fiji_frame" in rv else ["start_frame"])
    front = [c for c in ["event_id", "field", *groups, "source", "fiji_frame", "minutes", "x_um", "y_um", "suggest"]
             if c in rv]
    rv = rv[front + fields + [c for c in rv.columns if c not in front + fields]].sort_values(sort)

    checks = [c for c in fields if c != "notes" and set(rv[c].dropna().unique()) <= {0, 1, 1.0, 0.0}]
    picks = [c for c in fields if c not in checks and c not in ("notes", a.verdict)
             and not pd.api.types.is_numeric_dtype(rv[c])]
    div = rv[rv[a.verdict] == a.yes] if a.verdict in rv else rv
    summary = []
    for by in groups + picks:
        for g, d in div.groupby(by):
            row = {"grouped by": by, "group": g, "divisions": len(d)}
            row.update({f"% {c}": round(100 * d[c].fillna(0).astype(float).mean(), 1) for c in checks})
            summary.append(row)
    with pd.ExcelWriter(out, engine="openpyxl") as xw:
        for g, d in rv.groupby(a.sheet_by):
            d.to_excel(xw, sheet_name=re.sub(r"[\[\]:*?/\\]", "_", str(g))[:31] or "all", index=False)
        pd.DataFrame(summary).to_excel(xw, sheet_name="summary", index=False)
    print(f"{len(rv)} cards ({len(div)} marked {a.yes!r}) -> {out}")


if __name__ == "__main__":
    main()
