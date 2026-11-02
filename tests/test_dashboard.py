import json
import re
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]
DATA = json.loads((ROOT / "dashboard" / "data" / "results.json").read_text())


def test_snapshot_adds_up():
    routes = DATA["routes"]
    assert sum(r["requests"] for r in routes.values()) == DATA["requests"]
    assert sum(r["cost_usd"] for r in routes.values()) == pytest.approx(DATA["baseline_usd"])
    # What's left of the bill is the paid route; the saving is everything else.
    saved = routes["cache"]["cost_usd"] + routes["local"]["cost_usd"]
    assert saved / DATA["baseline_usd"] == pytest.approx(DATA["current"]["savings"], abs=0.001)


def test_dashboard_renders_and_shows_the_measured_numbers():
    app = AppTest.from_file(str(ROOT / "dashboard" / "app.py"), default_timeout=30).run()

    assert not app.exception
    page = " ".join(m.value for m in app.markdown)
    assert f"{DATA['current']['savings']:.0%}" in page
    assert [h.value for h in app.header] == [
        "Where the requests went, and where the money goes",
        "First version compared with the current version",
        "Old free model compared with the new one",
        "How long people wait",
        "What to keep in mind",
    ]


def test_dashboard_shows_no_dates():
    app = AppTest.from_file(str(ROOT / "dashboard" / "app.py"), default_timeout=30).run()
    text = " ".join(e.value for e in [*app.markdown, *app.caption, *app.title, *app.header])
    assert "measured_on" not in DATA
    assert not re.search(
        r"\b(19|20)\d\d\b|January|February|March|April|June|July|August|"
        r"September|October|November|December",
        text,
    )
