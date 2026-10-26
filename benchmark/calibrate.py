"""Pick the semantic-cache similarity threshold from labeled paraphrase pairs.

Offline and free: embeds Quora question pairs with the same local model the cache uses,
then measures, for each threshold, how often a "hit" would be a true duplicate.

A false hit serves the wrong answer, so we favour precision: the chosen threshold is
the lowest one whose precision reaches the target. QQP labels are human and somewhat
noisy, so precision near 100% isn't reachable; the target is a deliberate trade-off.
"""

import argparse
import asyncio
import json
import math
from dataclasses import asdict, dataclass

from benchmark import datasets
from benchmark.workload import qqp_split
from finops.cache.embeddings import FastEmbedEmbedder
from finops.cache.guards import fingerprint
from finops.core.config import get_settings

RESULT_FILE = datasets.DATA_DIR / "calibration.json"
THRESHOLDS = [round(0.80 + 0.01 * i, 2) for i in range(20)]  # 0.80 .. 0.99


@dataclass
class Point:
    threshold: float
    precision: float
    recall: float
    f1: float
    hits: int
    precision_no_guard: float
    hits_no_guard: int


def cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    return dot / (math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b)))


def sweep(pairs: list[tuple[float, bool, bool]]) -> list[Point]:
    """pairs: (similarity, guards_match, is_duplicate)."""
    positives = sum(dup for _, _, dup in pairs)
    points = []
    for t in THRESHOLDS:
        hits = [(g, dup) for s, g, dup in pairs if s >= t]
        guarded = [dup for g, dup in hits if g]
        tp = sum(guarded)
        precision = tp / len(guarded) if guarded else 1.0
        recall = tp / positives if positives else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        points.append(
            Point(
                threshold=t,
                precision=round(precision, 4),
                recall=round(recall, 4),
                f1=round(f1, 4),
                hits=len(guarded),
                precision_no_guard=round(sum(d for _, d in hits) / len(hits), 4) if hits else 1.0,
                hits_no_guard=len(hits),
            )
        )
    return points


def choose(points: list[Point], target_precision: float, tolerance: float = 0.005) -> Point:
    """Lowest threshold meeting the target; if none does, the lowest one whose precision
    is within `tolerance` of the best (higher thresholds then only cost recall)."""
    candidates = [p for p in points if p.hits > 0]
    meeting = [p for p in candidates if p.precision >= target_precision]
    if not meeting:
        best = max(p.precision for p in candidates)
        meeting = [p for p in candidates if p.precision >= best - tolerance]
    return min(meeting, key=lambda p: p.threshold)


async def embed_pairs(rows: list[dict]) -> list[tuple[float, bool, bool]]:
    settings = get_settings().cache
    embedder = FastEmbedEmbedder(
        settings.embedding_model, cache_dir=str(settings.model_cache_dir.expanduser())
    )
    pairs = []
    for i, row in enumerate(rows, 1):
        a, b = row["question1"], row["question2"]
        sim = cosine(await embedder.embed(a), await embedder.embed(b))
        pairs.append((sim, fingerprint(a) == fingerprint(b), row["label"] == 1))
        if i % 250 == 0:
            print(f"  embedded {i}/{len(rows)} pairs")
    return pairs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", type=int, default=2000)
    parser.add_argument("--target-precision", type=float, default=0.90)
    args = parser.parse_args()

    rows = qqp_split(datasets.load("qqp"), "calibration")[: args.pairs]
    print(f"Embedding {len(rows)} QQP pairs ({sum(r['label'] for r in rows)} duplicates)...")
    points = sweep(asyncio.run(embed_pairs(rows)))
    best = choose(points, args.target_precision)

    print(f"\n{'thresh':>7}{'precision':>11}{'recall':>8}{'hits':>6}   no-guard precision / hits")
    for p in points:
        mark = "  <-- chosen" if p is best else ""
        print(
            f"{p.threshold:>7.2f}{p.precision:>11.3f}{p.recall:>8.3f}{p.hits:>6}"
            f"   {p.precision_no_guard:>8.3f} / {p.hits_no_guard:<5}{mark}"
        )
    datasets.DATA_DIR.mkdir(parents=True, exist_ok=True)
    RESULT_FILE.write_text(
        json.dumps(
            {
                "pairs": len(rows),
                "target_precision": args.target_precision,
                "chosen": asdict(best),
                "sweep": [asdict(p) for p in points],
            },
            indent=2,
        )
    )
    print(f"\nChosen threshold {best.threshold} -> set cache.similarity_threshold in settings.yaml")
    print(f"Saved {RESULT_FILE}")


if __name__ == "__main__":
    main()
