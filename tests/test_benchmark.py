import httpx
import pytest
import respx

from benchmark import calibrate, evaluate, replay, report, workload
from benchmark.replay import Pacer, ReplayResult
from benchmark.workload import Mix, WorkItem


@pytest.fixture
def fake_datasets(monkeypatch):
    rows = {
        "dolly": [
            {"_row": i, "instruction": f"Task {i}", "context": "ctx" if i % 2 else "",
             "category": cat, "response": ""}
            for i, cat in enumerate(sorted(workload.SIMPLE_DOLLY | workload.OPEN_ENDED_DOLLY) * 5)
        ],
        "gsm8k": [{"_row": i, "question": f"Q{i}?", "answer": f"work\n#### 1,00{i}"}
                  for i in range(10)],
        "humaneval": [{"_row": i, "prompt": f"def f{i}():"} for i in range(10)],
        "qqp": [
            {"_row": i, "question1": f"how do i learn python fast {i}",
             "question2": f"how can i learn python quickly {i}" if i % 4 == 1
             else f"how do i learn python slowly {i}", "label": 1 if i % 4 == 1 else 0}
            for i in range(200)
        ],
    }  # fmt: skip
    monkeypatch.setattr(workload.datasets, "load", lambda name: rows[name])


SMALL = Mix(dolly=16, gsm8k=3, humaneval=2, paraphrase_pairs=4, near_miss_pairs=3, popular=3,
            repeats=6)  # fmt: skip


def test_workload_mix_and_ordering(fake_datasets):
    items = workload.build(SMALL, seed=1)
    positions = {item.id: i for i, item in enumerate(items)}

    assert len(items) == 16 + 3 + 2 + 2 * 4 + 2 * 3 + 6
    assert len(positions) == len(items)  # ids are unique
    for item in items:
        if item.original_id:  # repeats and second halves always follow their original
            assert positions[item.original_id] < items.index(item)
    assert {i.reference for i in items if i.source == "gsm8k"} <= {f"100{n}" for n in range(10)}
    assert all(i.expected_tier == "premium" for i in items if i.source in ("gsm8k", "humaneval"))


def test_workload_is_reproducible(fake_datasets):
    first = [i.id for i in workload.build(SMALL, seed=3)]
    assert first == [i.id for i in workload.build(SMALL, seed=3)]
    assert first != [i.id for i in workload.build(SMALL, seed=4)]


def test_qqp_split_is_disjoint():
    rows = [{"_row": i} for i in range(10)]
    cal, work = workload.qqp_split(rows, "calibration"), workload.qqp_split(rows, "workload")
    assert {r["_row"] for r in cal}.isdisjoint({r["_row"] for r in work})
    assert len(cal) + len(work) == 10


def test_calibration_sweep_and_choice():
    # (similarity, guards match, is duplicate)
    pairs = [(0.99, True, True), (0.97, True, True), (0.95, True, False), (0.93, False, True),
             (0.90, True, False), (0.85, True, True)]  # fmt: skip
    points = {p.threshold: p for p in calibrate.sweep(pairs)}

    assert points[0.97].precision == 1.0 and points[0.97].hits == 2
    assert points[0.95].precision == pytest.approx(2 / 3, abs=1e-3)
    assert points[0.93].hits_no_guard == 4 and points[0.93].hits == 3  # guard filtered one
    assert calibrate.choose(list(points.values()), 0.9).threshold == 0.96


def test_choose_prefers_lower_threshold_when_precision_ties():
    points = [calibrate.Point(t, p, 0, 0, 10, p, 10) for t, p in [(0.95, 0.87), (0.96, 0.893),
              (0.97, 0.896)]]  # fmt: skip
    assert calibrate.choose(points, target_precision=0.99).threshold == 0.96


@pytest.mark.parametrize(
    ("text", "number"),
    [("so the answer is\n18", "18"), ("Total: $1,234.50.", "1234.5"), ("x = -3", "-3"),
     ("no digits", None), ("18.00", "18")],
)  # fmt: skip
def test_final_number(text, number):
    assert evaluate.final_number(text) == number


@pytest.mark.parametrize(
    ("text", "verdict"),
    [('{"verdict": "pass", "reason": "ok"}', "pass"), ('Sure! {"verdict":"FAIL"}', "fail"),
     ("verdict: pass", "pass"), ("no idea", "error")],
)  # fmt: skip
def test_parse_verdict(text, verdict):
    assert evaluate.parse_verdict(text)[0] == verdict


def test_pacer_spaces_requests():
    now, slept = [0.0], []
    pacer = Pacer(
        max_rpm=30,
        clock=lambda: now[0],
        sleep=lambda s: (slept.append(s), now.__setitem__(0, now[0] + s)),
    )  # noqa: E501
    pacer.wait()
    now[0] += 0.5
    pacer.wait()
    assert slept == [pytest.approx(1.5)]


def test_send_retries_rate_limits():
    item = WorkItem(id="x", prompt="hi", kind="unique", source="dolly", category="open_qa")
    ok = httpx.Response(
        200,
        json={"choices": [{"message": {"content": "hello"}}]},
        headers={"x-finops-route": "local", "x-finops-cache": "miss"},
    )
    with respx.mock(base_url="http://proxy.test") as mock:
        mock.post("/v1/chat/completions").side_effect = [
            httpx.Response(429, headers={"retry-after": "1"}),
            ok,
        ]
        with httpx.Client(base_url="http://proxy.test") as client:
            result = replay.send(client, item, 0, "bench-t", Pacer(0), sleep=lambda s: None)
    assert (result.status, result.route, result.answer, result.attempts) == (
        200,
        "local",
        "hello",
        2,
    )


def _result(id_, route, cache="miss", escalated=None, answer="a"):
    return ReplayResult(id_, 0, 200, route, cache, None, None, escalated, "r", 1.0, answer, 1)


def test_routing_accuracy_and_cache_behaviour():
    items = {
        "s1": WorkItem("s1", "p", "unique", "dolly", "classification", "local"),
        "s2": WorkItem("s2", "p", "unique", "dolly", "summarization", "local"),
        "c1": WorkItem("c1", "p", "unique", "gsm8k", "math", "premium"),
        "p1": WorkItem("p1", "p", "paraphrase", "qqp", "paraphrase", original_id="x"),
        "n1": WorkItem("n1", "p", "near_miss", "qqp", "near_miss", original_id="y"),
    }
    results = {
        "s1": _result("s1", "local"),
        "s2": _result("s2", "premium", escalated="local_truncated"),  # decided local
        "c1": _result("c1", "local"),
        "p1": _result("p1", "cache", cache="hit_semantic"),
        "n1": _result("n1", "local"),
    }
    routing = report.routing_accuracy(items, results)
    cache = report.cache_behaviour(items, results)

    assert routing["n"] == 3 and routing["accuracy"] == pytest.approx(2 / 3)
    assert routing["expected_premium_got_local"] == 1
    assert cache["paraphrases"]["hit_rate"] == 1.0
    assert cache["lookalikes"]["hit_rate"] == 0.0


def test_render_report_smoke():
    from finops.ledger.store import build_summary as ledger_summary

    ledger = ledger_summary(
        {"cache": {"requests": 2, "p50_ms": 10, "baseline_usd": 1, "saved_usd": 1},
         "premium": {"requests": 2, "p50_ms": 800, "baseline_usd": 1, "cost_usd": 1}},
        errors=0,
    )  # fmt: skip
    ledger["pricing"] = {"version": "v", "baseline_model": "b", "premium_priced_as": "b"}
    items = {"a": WorkItem("a", "p", "unique", "dolly", "classification", "local")}
    summary = report.build_summary("run1", ledger, items, {"a": _result("a", "local")}, None, None)

    text = report.render(summary)

    assert "# Benchmark report: run1" in text
    assert "80x faster" in text
    assert "Resume-ready lines" in text
