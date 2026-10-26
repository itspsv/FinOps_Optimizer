"""Build a reproducible request stream that looks like real application traffic.

Mix (defaults, ~330 requests):
  120 unique Dolly tasks across 8 categories      mostly simple
   30 GSM8K math problems                         complex
   20 HumanEval coding problems                   complex
   40 Quora duplicate pairs  (80 requests)        2nd question should hit the cache
   20 Quora look-alike pairs (40 requests)        2nd question must NOT hit the cache
   40 repeats of 10 "popular" questions           exact cache hits, Zipf-skewed

Dependent requests (repeats, second halves of pairs) always come after their original.
"""

import argparse
import json
import random
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path

from benchmark import datasets

WORKLOAD_FILE = datasets.DATA_DIR / "workload.jsonl"

SIMPLE_DOLLY = {"classification", "closed_qa", "information_extraction", "summarization", "open_qa"}
OPEN_ENDED_DOLLY = {"brainstorming", "creative_writing", "general_qa"}


@dataclass
class Mix:
    dolly: int = 120
    gsm8k: int = 30
    humaneval: int = 20
    paraphrase_pairs: int = 40
    near_miss_pairs: int = 20
    popular: int = 10
    repeats: int = 40


@dataclass
class WorkItem:
    id: str
    prompt: str
    kind: str  # unique | repeat | pair_first | paraphrase | near_miss
    source: str
    category: str
    expected_tier: str | None = None  # local | premium | None when it's a judgment call
    original_id: str | None = None  # for repeats and second halves of pairs
    reference: str | None = None  # known answer, e.g. GSM8K's final number
    meta: dict = field(default_factory=dict)


def build(mix: Mix, seed: int = 42) -> list[WorkItem]:
    rng = random.Random(seed)
    base = _dolly(mix.dolly, rng) + _gsm8k(mix.gsm8k, rng) + _humaneval(mix.humaneval, rng)
    firsts, seconds = _qqp_pairs(mix.paraphrase_pairs, mix.near_miss_pairs, rng)
    base += firsts
    rng.shuffle(base)

    popular = rng.sample([i for i in base if i.expected_tier == "local"], mix.popular)
    weights = [1 / rank for rank in range(1, len(popular) + 1)]  # Zipf: a few are very popular
    repeats = [
        WorkItem(
            id=f"repeat-{n:03d}",
            prompt=original.prompt,
            kind="repeat",
            source=original.source,
            category=original.category,
            expected_tier=original.expected_tier,
            original_id=original.id,
            reference=original.reference,
        )
        for n, original in enumerate(rng.choices(popular, weights=weights, k=mix.repeats))
    ]

    ordered = list(base)
    for item in seconds + repeats:
        after = next(i for i, x in enumerate(ordered) if x.id == item.original_id)
        ordered.insert(rng.randint(after + 1, len(ordered)), item)
    return ordered


def _dolly(n: int, rng: random.Random) -> list[WorkItem]:
    rows = datasets.load("dolly")
    categories = sorted(SIMPLE_DOLLY | OPEN_ENDED_DOLLY)
    per_category = -(-n // len(categories))
    items = []
    for category in categories:
        pool = [r for r in rows if r["category"] == category]
        for row in rng.sample(pool, min(per_category, len(pool))):
            prompt = row["instruction"].strip()
            if row.get("context"):
                prompt += "\n\n" + row["context"].strip()
            items.append(
                WorkItem(
                    id=f"dolly-{row['_row']}",
                    prompt=prompt,
                    kind="unique",
                    source="dolly",
                    category=category,
                    expected_tier="local" if category in SIMPLE_DOLLY else None,
                )
            )
    return items[:n]


def _gsm8k(n: int, rng: random.Random) -> list[WorkItem]:
    return [
        WorkItem(
            id=f"gsm8k-{row['_row']}",
            prompt=row["question"] + "\n\nGive the final answer as a number on the last line.",
            kind="unique",
            source="gsm8k",
            category="math",
            expected_tier="premium",
            reference=row["answer"].split("####")[-1].strip().replace(",", ""),
        )
        for row in rng.sample(datasets.load("gsm8k"), n)
    ]


def _humaneval(n: int, rng: random.Random) -> list[WorkItem]:
    return [
        WorkItem(
            id=f"humaneval-{row['_row']}",
            prompt="Complete this Python function:\n\n" + row["prompt"],
            kind="unique",
            source="humaneval",
            category="code",
            expected_tier="premium",
        )
        for row in rng.sample(datasets.load("humaneval"), n)
    ]


def qqp_split(rows: list[dict], part: str) -> list[dict]:
    """Calibration and workload use disjoint QQP rows (even vs odd row index)."""
    parity = 0 if part == "calibration" else 1
    return [r for r in rows if r["_row"] % 2 == parity]


def jaccard(a: str, b: str) -> float:
    wa, wb = set(re.findall(r"\w+", a.lower())), set(re.findall(r"\w+", b.lower()))
    return len(wa & wb) / len(wa | wb) if wa | wb else 0.0


def _qqp_pairs(
    n_paraphrase: int, n_near_miss: int, rng: random.Random
) -> tuple[list[WorkItem], list[WorkItem]]:
    rows = qqp_split(datasets.load("qqp"), "workload")
    duplicates = [r for r in rows if r["label"] == 1 and r["question1"] != r["question2"]]
    # Look-alikes: different meaning but heavy word overlap, the hard case for a cache.
    lookalikes = [
        r for r in rows if r["label"] == 0 and jaccard(r["question1"], r["question2"]) >= 0.5
    ]
    firsts, seconds = [], []
    for kind, pool, n in (
        ("paraphrase", duplicates, n_paraphrase),
        ("near_miss", lookalikes, n_near_miss),
    ):
        for row in rng.sample(pool, n):
            first = WorkItem(
                id=f"qqp-{row['_row']}-a",
                prompt=row["question1"],
                kind="pair_first",
                source="qqp",
                category=kind,
            )
            firsts.append(first)
            seconds.append(
                WorkItem(
                    id=f"qqp-{row['_row']}-b",
                    prompt=row["question2"],
                    kind=kind,
                    source="qqp",
                    category=kind,
                    original_id=first.id,
                    meta={"jaccard": round(jaccard(row["question1"], row["question2"]), 3)},
                )
            )
    return firsts, seconds


def save(items: list[WorkItem], path: Path = WORKLOAD_FILE) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(asdict(i), ensure_ascii=False) + "\n" for i in items))
    return path


def load(path: Path = WORKLOAD_FILE) -> list[WorkItem]:
    return [WorkItem(**json.loads(line)) for line in path.read_text().splitlines() if line]


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the benchmark workload.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--scale", type=float, default=1.0, help="e.g. 0.2 for a quick run")
    args = parser.parse_args()
    mix = Mix(**{k: max(1, round(v * args.scale)) for k, v in asdict(Mix()).items()})
    items = build(mix, args.seed)
    path = save(items)
    kinds: dict[str, int] = {}
    for item in items:
        kinds[item.kind] = kinds.get(item.kind, 0) + 1
    print(f"{len(items)} requests -> {path}")
    for kind, count in sorted(kinds.items()):
        print(f"  {kind:<11} {count:>4}")


if __name__ == "__main__":
    main()
