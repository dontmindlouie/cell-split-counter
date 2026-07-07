"""Tests for the self-contained HTML QC report renderers (src/reports)."""

import json
import re

from src.reports.html_chart import histogram_bins, render_bar_html, render_scatter_html


def test_scatter_html_embeds_points_and_fit_line(tmp_path):
    points = [(1.0, 0.1), (2.0, 0.2), (3.0, 0.3)]
    out = render_scatter_html(
        points,
        out_path=tmp_path / "scatter.html",
        title="test scatter",
        x_label="x",
        y_label="y",
        fit_line=(0.1, 0.0),
        fit_label="y = 0.1x",
    )
    html = out.read_text(encoding="utf-8")
    assert "<title>test scatter</title>" in html
    m = re.search(r"var points = (\[.*?\]);", html)
    assert json.loads(m.group(1)) == [list(p) for p in points]
    assert "y = 0.1x" in html
    # basic structural sanity -- every opened div is closed
    assert html.count("<div") == html.count("</div>")


def test_scatter_html_without_fit_line_or_callout(tmp_path):
    out = render_scatter_html(
        [(0.0, 0.0), (1.0, 1.0)],
        out_path=tmp_path / "scatter.html",
        title="no fit",
    )
    html = out.read_text(encoding="utf-8")
    assert "var fit = null;" in html
    assert '<div class="callout">' not in html  # no empty callout box when none was given


def test_scatter_html_empty_points_does_not_crash(tmp_path):
    out = render_scatter_html([], out_path=tmp_path / "empty.html", title="empty")
    assert out.exists()
    assert "var points = [];" in out.read_text(encoding="utf-8")


def test_bar_html_embeds_categories_and_values(tmp_path):
    out = render_bar_html(
        ["a", "b", "c"], [1, 5, 2],
        out_path=tmp_path / "bar.html",
        title="test bar",
        y_label="count",
    )
    html = out.read_text(encoding="utf-8")
    assert "<title>test bar</title>" in html
    assert json.loads(re.search(r"var categories = (\[.*?\]);", html).group(1)) == ["a", "b", "c"]
    assert json.loads(re.search(r"var values = (\[.*?\]);", html).group(1)) == [1, 5, 2]


def test_bar_html_requires_parallel_lists(tmp_path):
    try:
        render_bar_html(["a", "b"], [1], out_path=tmp_path / "bad.html", title="bad")
        assert False, "expected AssertionError for mismatched lengths"
    except AssertionError:
        pass


def test_callout_and_stats_render(tmp_path):
    out = render_scatter_html(
        [(1.0, 1.0)],
        out_path=tmp_path / "scatter.html",
        title="with callout",
        callout_html="<strong>heads up</strong>",
        stats={"n": "1", "pearson r": "1.0"},
    )
    html = out.read_text(encoding="utf-8")
    assert "heads up" in html
    assert 'class="stat-value">1<' in html
    assert 'class="stat-value">1.0<' in html


def test_histogram_bins_basic():
    labels, counts = histogram_bins([0.0, 0.5, 0.9, 1.0], n_bins=2)
    assert len(labels) == 2
    assert sum(counts) == 4


def test_histogram_bins_empty():
    labels, counts = histogram_bins([], n_bins=5)
    assert labels == [] and counts == []


def test_histogram_bins_single_distinct_value():
    labels, counts = histogram_bins([3.0, 3.0, 3.0], n_bins=5)
    assert counts == [3]
