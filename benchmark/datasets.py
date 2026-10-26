"""Public datasets, fetched as JSON from the Hugging Face rows API and cached locally.

  dolly      databricks/databricks-dolly-15k   everyday tasks, mostly simple   CC BY-SA 3.0
  gsm8k      openai/gsm8k (test)               math word problems (complex)    MIT
  humaneval  openai/openai_humaneval           coding problems (complex)       MIT
  qqp        nyu-mll/glue qqp (validation)     human paraphrase pairs          Quora terms

Only small samples are fetched; nothing here is committed to git.
"""

import json
import random
from pathlib import Path
from typing import Any

import httpx

DATA_DIR = Path(__file__).resolve().parent / "data"
ROWS_API = "https://datasets-server.huggingface.co/rows"
PAGE = 100  # the rows API's maximum page size

SOURCES = {
    "dolly": ("databricks/databricks-dolly-15k", "default", "train", 15011),
    "gsm8k": ("openai/gsm8k", "main", "test", 1319),
    "humaneval": ("openai/openai_humaneval", "openai_humaneval", "test", 164),
    "qqp": ("nyu-mll/glue", "qqp", "validation", 40430),
}

# Rows fetched per dataset: enough to sample from, small enough to download in seconds.
SAMPLE_SIZES = {"dolly": 1500, "gsm8k": 300, "humaneval": 164, "qqp": 6000}


def fetch(name: str, rows: int, seed: int = 7, client: httpx.Client | None = None) -> list[dict]:
    """Fetch `rows` rows as random pages spread across the dataset (deterministic per seed)."""
    dataset, config, split, total = SOURCES[name]
    pages = list(range(0, total, PAGE))
    random.Random(seed).shuffle(pages)
    owns_client = client is None
    client = client or httpx.Client(timeout=30)
    out: list[dict] = []
    try:
        for offset in sorted(pages[: -(-rows // PAGE)]):
            response = client.get(
                ROWS_API,
                params={
                    "dataset": dataset,
                    "config": config,
                    "split": split,
                    "offset": offset,
                    "length": PAGE,
                },
            )
            response.raise_for_status()
            out += [r["row"] | {"_row": r["row_idx"]} for r in response.json()["rows"]]
    finally:
        if owns_client:
            client.close()
    return out[:rows]


def path_for(name: str) -> Path:
    return DATA_DIR / f"{name}.jsonl"


def load(name: str) -> list[dict[str, Any]]:
    path = path_for(name)
    if not path.exists():
        raise FileNotFoundError(f"{path} missing; run `make bench-data` first")
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def save(name: str, rows: list[dict]) -> Path:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    path = path_for(name)
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    return path


def main() -> None:
    for name, size in SAMPLE_SIZES.items():
        if path_for(name).exists():
            print(f"{name:<10} cached   {path_for(name)}")
            continue
        rows = fetch(name, size)
        print(f"{name:<10} {len(rows):>5} rows -> {save(name, rows)}")


if __name__ == "__main__":
    main()
