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


def test_parts_listed_and_keyboard_wired(fake, tmp_path, monkeypatch):
    # Every part links every other part (jump to where you left off), and the page
    # knows its own first event so j/k and the lightbox map to global event numbers.
    monkeypatch.chdir(tmp_path)
    lines = cell_mcp_server.show_cells_in_browser(fake, events=[POINT] * 13).splitlines()
    pages = [Path(l) for l in lines[1:] if not l.startswith("file:")]
    second = pages[1].read_text(encoding="utf-8")
    assert second.count("data-part=") == 2 * 2          # top and bottom pager
    assert f'href="{pages[0].name}" data-part=1' in second
    assert "var FIRST = 12," in second
    assert "markFirstChoice" in second and 'id="lb-play"' in second


def test_context_um_widens_only_the_lightbox(fake, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    calls = []
    real = cell_mcp_server.tools_output._cm._fixed_point_frames

    def spy(*a, **k):
        calls.append(k["crop_um"])
        return real(*a, **k)

    monkeypatch.setattr(cell_mcp_server.tools_output._cm, "_fixed_point_frames", spy)
    cell_mcp_server.show_cells_in_browser(fake, events=[dict(POINT, crop_um=60, context_um=200)])
    assert calls == [60.0, 200.0]                        # filmstrip, then lightbox


def test_channel_layers_validated(fake, monkeypatch):
    # Layers are named from the bundle's own channel list; a typo should say what exists.
    real = cell_mcp_server._manifest
    monkeypatch.setattr(cell_mcp_server, "_manifest", lambda w: dict(
        real(w), channels=[{"name": "AF594", "dir": "channels/AF594", "color_rgb": [255, 0, 0]}]))
    with pytest.raises(ValueError, match="no channel"):
        cell_mcp_server.show_cells_in_browser(fake, events=[POINT], channels=["AF999"])
    with pytest.raises(ValueError, match="channels_off"):
        cell_mcp_server.show_cells_in_browser(fake, events=[POINT], channels=["AF594"],
                                              channels_off=["AF488"])


def test_events_per_part_and_frame_cap(fake, tmp_path, monkeypatch):
    # Fewer events per part is what pays for longer windows; the cap is the per-event ceiling.
    monkeypatch.chdir(tmp_path)
    lines = cell_mcp_server.show_cells_in_browser(fake, events=[POINT] * 5, events_per_part=2).splitlines()
    assert len([l for l in lines[1:] if not l.startswith("file:")]) == 3
    seen = []
    real = cell_mcp_server.tools_output._cm._fixed_point_frames
    monkeypatch.setattr(cell_mcp_server.tools_output._cm, "_fixed_point_frames",
                        lambda *a, **k: seen.append(k["cap"]) or real(*a, **k))
    cell_mcp_server.show_cells_in_browser(fake, events=[POINT], max_frames=90)
    assert seen == [90, 90]
    with pytest.raises(ValueError, match="events_per_part"):
        cell_mcp_server.show_cells_in_browser(fake, events=[POINT], events_per_part=0)
