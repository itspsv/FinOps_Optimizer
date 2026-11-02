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
        "c1": WorkItem("c1", "p", "unique", "humaneval", "code", "premium"),
        "g1": WorkItem("g1", "p", "unique", "gsm8k", "math", "premium"),
        "p1": WorkItem("p1", "p", "paraphrase", "qqp", "paraphrase", original_id="x"),
        "n1": WorkItem("n1", "p", "near_miss", "qqp", "near_miss", original_id="y"),
    }
    results = {
        "s1": _result("s1", "local"),
        "s2": _result("s2", "premium", escalated="local_truncated"),  # decided local
        "c1": _result("c1", "local"),
        "g1": _result("g1", "local"),  # either route is fine for GSM8K: not counted
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


def test_token_budget_waits_for_the_window_to_clear():
    now, slept = [0.0], []

    def sleep(s):
        slept.append(s)
        now[0] += s

    budget = evaluate.TokenBudget(8000, clock=lambda: now[0], sleep=sleep)
    budget.reserve(5000)
    now[0] = 10
    budget.reserve(2000)  # 7,000 in window: fits
    assert slept == []
    budget.correct_last(2500)
    budget.reserve(1000)  # 8,500 would exceed: wait until the first entry expires
    assert slept and now[0] >= 60


def test_collect_references_from_earlier_runs(tmp_path):
    run = tmp_path / "run-a"
    run.mkdir()
    (run / "eval.jsonl").write_text(
        '{"id": "x", "check": "local_vs_reference", "verdict": "pass", "reason": "", '
        '"reference": "ref x"}\n'
        '{"id": "y", "check": "semantic_hit", "verdict": "pass", "reason": "", '
        '"reference": null}\n'
    )
    assert evaluate.collect_references(tmp_path) == {"x": "ref x"}


def test_compare_render():
    from benchmark import compare
    from finops.ledger.store import build_summary as ledger_summary

    def summary(local_requests):
        ledger = ledger_summary({"local": {"requests": local_requests, "p50_ms": 9000}}, 0)
        return {
            "ledger": ledger,
            "shares": {"cache": 0.0, "local": 1.0, "premium": 0.0},
            "routing": {"accuracy": 0.5},
            "evaluation": {"local_quality": {"n": 4, "pass_rate": 0.75}},
        }

    text = compare.render({"v1": summary(3), "v2": summary(1)})
    assert "| Metric | v1 | v2 |" in text
    assert "75.0% (n=4)" in text


def test_localeval_summary():
    from benchmark import localeval
    from benchmark.localeval import LocalAnswer

    answers = {
        ("math", "a"): LocalAnswer("a", "math", "18", "stop", 10, 1000.0, correct=True),
        ("math", "b"): LocalAnswer("b", "math", "7", "length", 20, 2000.0, correct=False),
        ("knowledge", "c"): LocalAnswer("c", "knowledge", "x", "stop", 5, 500.0, verdict="pass"),
        ("knowledge", "d"): LocalAnswer("d", "knowledge", None, None, 0, 0.0, error="502"),
    }
    summary = localeval.summarize("m", answers)
    assert summary["math"]["accuracy"] == 0.5
    assert summary["math"]["truncated"] == 1
    assert summary["knowledge"]["n"] == 1  # errors excluded
    assert summary["knowledge"]["pass_rate"] == 1.0


async def test_daily_limit_stops_retries():
    from finops.providers import ProviderError

    class Capped:
        calls = 0

        async def chat(self, payload):
            self.calls += 1
            raise ProviderError(429, {"error": {"message": "tokens per day (TPD): Limit 200000"}})

    provider = Capped()
    groq = evaluate.GroqClient(provider, evaluate.TokenBudget(8000), sleep=lambda s: None)
    with pytest.raises(evaluate.DailyLimitReached):
        await groq.ask([{"role": "user", "content": "hi"}], max_tokens=5)
    assert provider.calls == 1


def test_replay_stops_at_groq_daily_cap(tmp_path, monkeypatch):
    monkeypatch.setattr(replay, "RESULTS_DIR", tmp_path)
    items = [
        WorkItem(id=i, prompt="hi", kind="unique", source="dolly", category="open_qa")
        for i in ("a", "b", "c")
    ]
    capped = httpx.Response(429, json={"error": {"message": "on tokens per day (TPD)"}})
    with respx.mock(base_url="http://proxy.test") as mock:
        route = mock.post("/v1/chat/completions").mock(return_value=capped)
        with httpx.Client(base_url="http://proxy.test") as client:
            path = replay.replay(items, "r1", client, Pacer(0))
    assert route.call_count == 1  # no retries and no further requests
    assert replay.load_results(path)["a"].daily_cap is True
    assert replay.load_results(path)["a"].status == 429


def test_subset_keeps_groups_mix_and_order(fake_datasets):
    items = workload.build(SMALL, seed=1)
    half = workload.subset(items, 0.5)
    ids = {i.id for i in half}

    assert 0.3 < len(half) / len(items) < 0.7
    assert all(i.original_id in ids for i in half if i.original_id)  # dependents keep originals
    assert [i.id for i in half] == [i.id for i in items if i.id in ids]  # same order
    assert {i.source for i in half} == {i.source for i in items}
    assert sum(i.kind == "repeat" for i in half) == 3  # 6 repeats in the full mix: stays 50%
    assert [i.id for i in workload.subset(items, 0.5)] == [i.id for i in half]  # reproducible


def _baseline_fixture():
    from finops.ledger.pricing import ModelPrice, PriceTable

    prices = PriceTable(
        version="t",
        baseline_model="b",
        premium_priced_as="b",
        models={"b": ModelPrice(input_per_1m=2.0, output_per_1m=10.0)},
    )
    items = {
        "a": WorkItem("a", "premium prompt", "unique", "dolly", "open_qa"),
        "b": WorkItem("b", "local prompt", "unique", "dolly", "summarization"),
        "c": WorkItem("c", "local prompt", "repeat", "dolly", "summarization", original_id="b"),
    }
    results = {i: _result(i, "x") for i in items}
    for key, result in results.items():
        result.request_id = f"r-{key}"
    ledger = {
        "r-a": {"route": "premium", "tokens": (100, 1000), "cost_usd": 0.0102},
        "r-b": {"route": "local", "tokens": (50, 10), "cost_usd": 0.0},
        "r-c": {"route": "cache", "tokens": (50, 10), "cost_usd": 0.0},
    }
    return prices, items, results, ledger


def test_baseline_measures_each_non_premium_prompt_once():
    from benchmark import baseline

    _, items, results, ledger = _baseline_fixture()
    known = baseline.premium_tokens_by_prompt(results, ledger, items)
    assert known == {"premium prompt": (100, 1000)}
    assert baseline.prompts_to_measure(items, items, known) == ["local prompt"]


def test_baseline_is_the_same_for_every_run():
    from benchmark import baseline

    prices, items, results, ledger = _baseline_fixture()
    tokens = {"premium prompt": (100, 1000), "local prompt": (50, 500)}
    # A second policy sent everything local: same requests, no premium cost.
    other = {k: {**v, "route": "local", "cost_usd": 0.0} for k, v in ledger.items()}

    out = baseline.compute(
        list(items), items, tokens, prices, {"p1": (results, ledger), "p2": (results, other)}
    )

    per_request_local = (50 * 2 + 500 * 10) / 1e6
    assert out["complete"] is True
    assert out["baseline_usd"] == pytest.approx(0.0102 + 2 * per_request_local)
    assert out["runs"]["p1"]["savings_pct"] == pytest.approx(1 - 0.0102 / out["baseline_usd"])
    assert out["runs"]["p2"]["savings_pct"] == 1.0
    assert out["runs"]["p1"]["premium_requests"] == 1


def test_incomplete_baseline_is_flagged_and_not_reported():
    from benchmark import baseline

    prices, items, results, ledger = _baseline_fixture()
    out = baseline.compute(
        list(items), items, {"premium prompt": (100, 1000)}, prices, {"p1": (results, ledger)}
    )
    assert out["complete"] is False
    assert out["unmeasured_requests"] == 2
