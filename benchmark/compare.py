"""Side-by-side comparison of benchmark runs (e.g. router v1 vs v2).

make bench-compare RUNS="full-1 full-2"   ->  benchmark/results/compare-full-1-vs-full-2.md
"""

import argparse
import json

from benchmark.replay import RESULTS_DIR


def _pct(x):
    return f"{x:.1%}" if x is not None else "n/a"


def _ms(x):
    return f"{x:,.0f} ms" if x is not None else "n/a"


ROWS = [
    ("Requests kept off premium", lambda s: _pct(s["ledger"]["deflection_rate"])),
    ("Cost saved vs. all-premium", lambda s: _pct(s["ledger"]["cost"]["savings_pct"])),
    ("Local share of traffic", lambda s: _pct(s["shares"]["local"])),
    ("Cache share of traffic", lambda s: _pct(s["shares"]["cache"])),
    ("Premium share of traffic", lambda s: _pct(s["shares"]["premium"])),
    (
        "Local answers judged acceptable",
        lambda s: (
            f"{_pct(((s['evaluation'] or {}).get('local_quality') or {}).get('pass_rate'))}"
            f" (n={((s['evaluation'] or {}).get('local_quality') or {}).get('n', 0)})"
        ),
    ),
    ("Router matched expected tier", lambda s: _pct(s["routing"]["accuracy"])),
    ("Escalations (local rejected)", lambda s: str(s["ledger"]["escalations"])),
    ("Cache p50", lambda s: _ms(s["ledger"]["routes"]["cache"]["p50_ms"])),
    ("Local p50", lambda s: _ms(s["ledger"]["routes"]["local"]["p50_ms"])),
    ("Premium p50", lambda s: _ms(s["ledger"]["routes"]["premium"]["p50_ms"])),
    ("Billed by Groq", lambda s: f"${s['ledger']['cost']['billed_usd']:.4f}"),
]


def render(runs: dict[str, dict]) -> str:
    names = list(runs)
    lines = [
        f"# Benchmark comparison: {' vs '.join(names)}",
        "",
        "| Metric | " + " | ".join(names) + " |",
        "|---|" + "---|" * len(names),
    ]
    for label, get in ROWS:
        lines.append(f"| {label} | " + " | ".join(get(runs[n]) for n in names) + " |")
    lines += [
        "",
        "Quality is the share of sampled local answers an LLM judge rated correct and "
        "responsive. Savings and quality trade off: sending more traffic local saves more "
        "but risks worse answers.",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="+")
    args = parser.parse_args()
    runs = {r: json.loads((RESULTS_DIR / r / "summary.json").read_text()) for r in args.runs}
    text = render(runs)
    out = RESULTS_DIR / f"compare-{'-vs-'.join(args.runs)}.md"
    out.write_text(text)
    print(text)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
