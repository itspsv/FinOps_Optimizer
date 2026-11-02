"""One-page results dashboard for people who don't work on the project.

    make dashboard

Everything shown comes from dashboard/data/results.json (see scripts/export_dashboard_data.py).
"""

import json
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

DATA_FILE = Path(__file__).parent / "data" / "results.json"

INK = "#0b0b0b"
MUTED = "#52514e"
GRID = "#e4e3df"
SURFACE = "#fcfcfb"
FIRST_VERSION = "#b4b3ad"
CURRENT_VERSION = "#2a78d6"
NO_OPTIMIZER = "#6f6e68"
FIRST_LABEL = "First version"
CURRENT_LABEL = "Current version"

# Same three colors everywhere, so a route keeps its color from chart to chart.
ROUTES = {
    "cache": ("Answered from memory", "#2a78d6"),
    "local": ("Free model on a laptop", "#eb6834"),
    "premium": ("Paid AI service", "#1baf7a"),
}
ROUTE_LABELS = [label for label, _ in ROUTES.values()]
ROUTE_COLORS = [color for _, color in ROUTES.values()]
MODEL_NAMES = {"gpt-6.1-sol": "OpenAI's GPT-6.1 Sol"}


def pct(x: float) -> str:
    return f"{x:.0%}"


def _base_axis(**kwargs) -> alt.Axis:
    return alt.Axis(
        grid=False, domain=False, ticks=False, labelColor=INK, labelFontSize=14,
        labelLimit=220, labelPadding=10, title=None, **kwargs,
    )  # fmt: skip


def stacked_chart(rows: list[dict], row_order: list[str]) -> alt.LayerChart:
    """One 100% bar per row, split by route, in the route colors."""
    base = (
        alt.Chart(pd.DataFrame(rows))
        .transform_stack(
            stack="share", groupby=["row"], sort=[alt.SortField("order")], as_=["start", "end"]
        )
        .transform_calculate(middle="(datum.start + datum.end) / 2")
    )
    y = alt.Y("row:N", axis=_base_axis(), sort=row_order)
    bars = base.mark_bar(height=30, cornerRadius=3, stroke=SURFACE, strokeWidth=2).encode(
        y=y,
        x=alt.X("start:Q", axis=None, scale=alt.Scale(domain=[0, 1])),
        x2="end:Q",
        color=alt.Color(
            "route:N",
            scale=alt.Scale(domain=ROUTE_LABELS, range=ROUTE_COLORS),
            legend=alt.Legend(
                orient="top", title=None, labelColor=INK, labelFontSize=13,
                symbolType="square", columnPadding=18, offset=12,
            ),
        ),
        tooltip=[
            alt.Tooltip("route:N", title="Route"),
            alt.Tooltip("detail:N", title="Amount"),
            alt.Tooltip("share:Q", title="Share", format=".0%"),
        ],
    )  # fmt: skip
    # A very thin slice has no room for its number; the tooltip and the table still show it.
    labels = (
        base.transform_filter("datum.share >= 0.05")
        .mark_text(fontSize=14, fontWeight=600, color=INK)
        .encode(
            y=y,
            x=alt.X("middle:Q", scale=alt.Scale(domain=[0, 1])),
            text=alt.Text("share:Q", format=".0%"),
        )
    )
    return (bars + labels).properties(height=alt.Step(42))


def split_chart(data: dict) -> alt.LayerChart:
    """Share of requests next to share of cost: most of the bill is in one route."""
    rows = []
    for order, (key, (label, _)) in enumerate(ROUTES.items()):
        route = data["routes"][key]
        requests, cost = route["requests"], route["cost_usd"]
        rows += [
            dict(row="Share of requests", route=label, order=order,
                 share=requests / data["requests"],
                 detail=f"{requests} of {data['requests']} requests"),
            dict(row="Share of cost", route=label, order=order,
                 share=cost / data["baseline_usd"],
                 detail=f"${cost:.2f} of ${data['baseline_usd']:.2f}"),
        ]  # fmt: skip
    return stacked_chart(rows, ["Share of requests", "Share of cost"])


def routes_by_version_chart(data: dict) -> alt.LayerChart:
    """The same requests, split very differently by each version."""
    versions = {
        FIRST_LABEL: data["first"]["route_requests"],
        CURRENT_LABEL: {k: v["requests"] for k, v in data["routes"].items()},
    }
    rows = [
        dict(row=version, route=label, order=order, share=counts[key] / data["requests"],
             detail=f"{counts[key]} of {data['requests']} requests")
        for version, counts in versions.items()
        for order, (key, (label, _)) in enumerate(ROUTES.items())
    ]  # fmt: skip
    return stacked_chart(rows, list(versions))


def bar_chart(bars_data: list[tuple[str, float, str]], value_format: str = ".0%") -> alt.LayerChart:
    """Labeled horizontal bars. Each item is (label, value, color)."""
    df = pd.DataFrame(bars_data, columns=["label", "value", "color"])
    order = list(df["label"])
    y = alt.Y("label:N", axis=_base_axis(), sort=order)
    bars = (
        alt.Chart(df)
        .mark_bar(height=26, cornerRadiusEnd=4)
        .encode(
            y=y,
            x=alt.X(
                "value:Q", axis=None, scale=alt.Scale(domain=[0, float(df["value"].max()) * 1.3])
            ),
            color=alt.Color("color:N", scale=None, legend=None),
            tooltip=[
                alt.Tooltip("label:N", title=" "),
                alt.Tooltip("value:Q", title="Result", format=value_format),
            ],
        )
    )
    labels = (
        alt.Chart(df)
        .mark_text(align="left", dx=6, fontSize=14, fontWeight=600, color=INK)
        .encode(y=y, x="value:Q", text=alt.Text("value:Q", format=value_format))
    )
    return (bars + labels).properties(height=alt.Step(38))


def versions_chart(first: float, current: float) -> alt.LayerChart:
    return bar_chart(
        [(FIRST_LABEL, first, FIRST_VERSION), (CURRENT_LABEL, current, CURRENT_VERSION)]
    )


def models_chart(result: dict) -> alt.LayerChart:
    return bar_chart(
        [
            ("Old free model", result["old"]["passed"] / result["old"]["n"], FIRST_VERSION),
            ("New free model", result["new"]["passed"] / result["new"]["n"], CURRENT_VERSION),
        ]
    )


def speed_label(seconds: float) -> str:
    return f"{seconds:.3f} seconds" if seconds < 0.1 else f"{seconds:.1f} seconds"


def speed_chart(data: dict) -> alt.LayerChart:
    df = pd.DataFrame(
        [
            {
                "route": label,
                "seconds": data["routes"][key]["p50_seconds"],
                "text": speed_label(data["routes"][key]["p50_seconds"]),
            }
            for key, (label, _) in ROUTES.items()
        ]
    )
    y = alt.Y("route:N", axis=_base_axis(), sort=ROUTE_LABELS)
    longest = float(df["seconds"].max())
    bars = (
        alt.Chart(df)
        .mark_bar(height=22, cornerRadiusEnd=4)
        .encode(
            y=y,
            x=alt.X("seconds:Q", axis=None, scale=alt.Scale(domain=[0, longest * 1.75])),
            color=alt.Color(
                "route:N", scale=alt.Scale(domain=ROUTE_LABELS, range=ROUTE_COLORS), legend=None
            ),
            tooltip=[
                alt.Tooltip("route:N", title="Route"),
                alt.Tooltip("text:N", title="Typical wait"),
            ],
        )
    )
    labels = (
        alt.Chart(df)
        .mark_text(align="left", dx=6, fontSize=14, color=INK)
        .encode(y=y, x="seconds:Q", text="text:N")
    )
    return (bars + labels).properties(height=alt.Step(36))


def numbers_table(data: dict) -> pd.DataFrame:
    first, current = data["first"], data["current"]

    def quality(q: dict) -> str:
        return f"{pct(q['rate'])} ({q['passed']} of {q['n']})"

    return pd.DataFrame(
        [
            ["Bill saved", pct(first["savings"]), pct(current["savings"])],
            ["Cost of the test", f"${first['cost_usd']:.2f}", f"${current['cost_usd']:.2f}"],
            [
                "Requests kept off the paid service",
                pct(first["kept_off_paid"]),
                pct(current["kept_off_paid"]),
            ],
            [
                "Free model answers that were acceptable",
                quality(first["quality"]),
                quality(current["quality"]),
            ],
        ],  # fmt: skip
        columns=["", "First version", "Current version"],
    ).set_index("")


def main() -> None:
    st.set_page_config(page_title="AI FinOps Optimizer: test results", layout="centered")
    st.markdown(
        """
        <style>
          .block-container { padding-top: 3rem; max-width: 720px; }
          div[data-testid="stMarkdownContainer"] p.big-number {
            font-size: 64px; font-weight: 600; line-height: 1; margin: 0.25rem 0 0.5rem;
          }
          h2 {
            font-size: 1.3rem !important; font-weight: 600 !important;
            padding-top: 1.75rem !important;
          }
          .chart-title { font-weight: 600; margin-bottom: 0; }
        </style>
        """,
        unsafe_allow_html=True,
    )  # fmt: skip

    data = json.loads(DATA_FILE.read_text())
    current, first = data["current"], data["first"]
    routes, baseline = data["routes"], data["baseline_usd"]
    quality = current["quality"]
    cache = data["cache"]
    saved_by_cache = routes["cache"]["cost_usd"] / baseline
    saved_by_local = routes["local"]["cost_usd"] / baseline
    per_10k = round(current["savings"] * 10_000, -2)
    model_name = MODEL_NAMES.get(data["baseline_model"], data["baseline_model"])

    st.title("AI FinOps Optimizer: test results")
    st.caption(f"{data['requests']} test requests, run on a MacBook Air")

    st.markdown(
        "Apps pay for every question they send to an AI service. The optimizer sits in the "
        "middle and answers what it safely can for free, either from memory or with a small "
        "model on a laptop. It only pays for the rest."
    )

    st.markdown(f'<p class="big-number">{pct(current["savings"])}</p>', unsafe_allow_html=True)
    st.markdown(
        f"**of the bill saved in this test.** About {round(quality['rate'] * 10)} in 10 answers "
        "from the free model were acceptable. If real traffic looked like this test, a "
        f"\\$10,000 monthly bill would drop by about \\${per_10k:,.0f}."
    )

    st.header("Where the requests went, and where the money goes")
    st.altair_chart(split_chart(data), width="stretch")
    st.markdown(
        f"The paid service handled {pct(routes['premium']['requests'] / data['requests'])} of "
        f"requests, but they make up {pct(routes['premium']['cost_usd'] / baseline)} of the "
        "cost, because they tend to be the long ones. The saving comes from the rest: about "
        f"{pct(saved_by_local)} of the bill from the free model and {pct(saved_by_cache)} "
        "from memory."
    )

    st.header("First version compared with the current version")
    st.markdown(
        "Our first version saved far more, but about half of the free model's answers were "
        "unacceptable, with wrong facts and made-up details. So we switched to a better free "
        "model and stopped sending it general knowledge questions. Both versions handled the "
        "same requests."
    )
    st.markdown('<p class="chart-title">Where the requests went</p>', unsafe_allow_html=True)
    st.altair_chart(routes_by_version_chart(data), width="stretch")
    left, right = st.columns(2)
    with left:
        st.markdown('<p class="chart-title">Bill saved</p>', unsafe_allow_html=True)
        st.altair_chart(versions_chart(first["savings"], current["savings"]), width="stretch")
    with right:
        st.markdown(
            '<p class="chart-title">Free model answers that were acceptable</p>',
            unsafe_allow_html=True,
        )
        st.altair_chart(versions_chart(first["quality"]["rate"], quality["rate"]), width="stretch")
    st.caption(
        "Each version sent different questions to its free model, so these two quality scores "
        "don't cover the same questions. The next section compares the models on identical ones."
    )
    st.markdown(
        f'<p class="chart-title">Cost of the {data["requests"]} requests</p>',
        unsafe_allow_html=True,
    )
    st.altair_chart(
        bar_chart(
            [
                ("Without the optimizer", baseline, NO_OPTIMIZER),
                (FIRST_LABEL, first["cost_usd"], FIRST_VERSION),
                (CURRENT_LABEL, current["cost_usd"], CURRENT_VERSION),
            ],
            value_format="$.2f",
        ),
        width="stretch",
    )

    models = data["free_models"]
    st.header("Old free model compared with the new one")
    st.markdown(
        "We gave Llama 3.2 (old) and Qwen3 (new) the same questions. The new model does better "
        "on both, which is why we switched, though it is still not reliable enough for general "
        "knowledge questions. It also runs about a quarter slower."
    )
    left, right = st.columns(2)
    with left:
        st.markdown(
            '<p class="chart-title">Math problems solved correctly</p>', unsafe_allow_html=True
        )
        st.altair_chart(models_chart(models["math"]), width="stretch")
    with right:
        st.markdown(
            '<p class="chart-title">General questions answered acceptably</p>',
            unsafe_allow_html=True,
        )
        st.altair_chart(models_chart(models["knowledge"]), width="stretch")
    st.caption(
        f"Math: {models['math']['new']['n']} problems checked against known answers "
        f"(the old model completed {models['math']['old']['n']}). General questions: "
        f"{models['knowledge']['new']['n']} questions graded by another AI model."
    )

    st.header("How long people wait")
    st.altair_chart(speed_chart(data), width="stretch")
    st.markdown(
        "Repeat questions come back almost instantly. New questions are slower on the free "
        "model than on the paid service, because it runs on a laptop. That is the cost of "
        "getting them for free."
    )

    st.header("What to keep in mind")
    st.markdown(
        f"""
- This was a test on public questions, not real company traffic. Savings depend heavily on how
  often questions repeat, and we chose how often that was.
- Costs are simulated. Calls ran on a low-cost service (Groq) but are priced at the list
  price of {model_name}, to show what a company paying that rate would see.
- Answers were graded by another AI model, not by people. With {quality["n"]} answers graded,
  the {pct(quality["rate"])} could be about {quality["margin"] * 100:.0f} points higher or lower.
- Memory only worked for exact repeats ({cache["repeats"]["hits"]} of {cache["repeats"]["n"]}).
  Questions asked in different words matched {cache["paraphrases"]["hits"]} of
  {cache["paraphrases"]["n"]} times, and different questions that only look alike wrongly
  matched {cache["lookalikes"]["hits"]} of {cache["lookalikes"]["n"]} times.
"""
    )

    with st.expander("See the numbers"):
        st.table(numbers_table(data))


main()
