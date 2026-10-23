"""Tests for show_cells_in_browser's review marking (notes + Export CSV on every page,
plus the caller-chosen review_form fields).

Added 2026-10-22: the researcher's Claude kept writing its own scoring sheets next to
the browser pages because the pages were view-only. The fields are the calling
session's choice per task, not a fixed list, so the tests only pin the mechanics.
"""

import json
import re
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cell_mcp_server  # noqa: E402
from test_show_cells_in_browser_xy import fake  # noqa: E402,F401

POINT = {"kind": "point", "x": 100.0, "y": 100.0, "start_frame": 0, "end_frame": 1}


def _review(html):
    return json.loads(re.search(r"var REVIEW = (.*?);\n", html).group(1))


def test_every_page_has_notes_and_export(fake, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    html = Path(cell_mcp_server.show_cells_in_browser(
        fake, events=[POINT]).splitlines()[0]).read_text(encoding="utf-8")
    assert 'data-f="notes"' in html and "exportReview()" in html
    rv = _review(html)
    assert rv["form"] == {"checks": [], "choices": {}, "numbers": [], "text": []}
    assert rv["events"][0]["x"] == 100.0


def test_review_form_fields_render(fake, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    form = {"checks": ["not a division", "lagging"],
            "choices": {"HLA-G": ["+", "-", "unsure"]}, "numbers": ["metaphase frame"]}
    html = Path(cell_mcp_server.show_cells_in_browser(
        fake, events=[dict(POINT, id="C1-007")], review_form=form)
        .splitlines()[0]).read_text(encoding="utf-8")
    for f in ("not a division", "lagging", "HLA-G", "metaphase frame"):
        assert f'data-f="{f}"' in html
    assert html.count('type=radio name="ev0-HLA-G"') == 3
    assert _review(html)["events"][0]["event_id"] == "C1-007"


def test_every_part_carries_the_whole_set(fake, tmp_path, monkeypatch):
    # Export from any part must write every event, so each part embeds the full list
    # under one shared storage key.
    monkeypatch.chdir(tmp_path)
    lines = cell_mcp_server.show_cells_in_browser(fake, events=[POINT] * 13).splitlines()
    pages = [l for l in lines[1:] if not l.startswith("file:")]
    rvs = [_review(Path(p).read_text(encoding="utf-8")) for p in pages]
    assert len(rvs) == 2
    assert all(len(r["events"]) == 13 for r in rvs)
    assert rvs[0]["key"] == rvs[1]["key"]


@pytest.mark.parametrize("form, msg", [
    ({"boxes": ["x"]}, "unknown keys"),
    ({"checks": ["a", "a"]}, "unique"),
    ({"checks": ["notes"]}, "unique"),
    ({"choices": {"HLA-G": ["+"]}}, "at least 2"),
])
def test_bad_review_form_rejected(fake, form, msg):
    with pytest.raises(ValueError, match=msg):
        cell_mcp_server.show_cells_in_browser(fake, events=[POINT], review_form=form)
