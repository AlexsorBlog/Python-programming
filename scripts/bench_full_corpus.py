"""Benchmark a few configurations on the whole corpus.

uv run --group bench python scripts/bench_full_corpus.py data/shards_full
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import psutil

from findex.parallel import build_parallel

CONFIGS: list[tuple[str, int]] = [
    ("serial", 1),
    ("threads", 8),
    ("processes", 8),
    ("processes", 16),
]


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="Бенчмарк на повному корпусі")
    parser.add_argument("root", type=Path)
    parser.add_argument("--out", type=Path, default=Path("bench_full_corpus.json"))
    args = parser.parse_args()

    rows: list[dict[str, Any]] = []
    for executor, workers in CONFIGS:
        proc = psutil.Process()
        index, timings = build_parallel(
            args.root,
            workers,
            executor,  # type: ignore[arg-type]  # from CONFIGS
            repr_kind="array",
        )
        info: Any = proc.memory_info()
        rss = float(getattr(info, "peak_wset", info.rss)) / 1048576
        rows.append(
            {
                "executor": executor,
                "workers": workers,
                "wall": timings.wall,
                "cpu": timings.cpu,
                "merge": timings.merge,
                "rss": rss,
                "terms": len(index),
                "docs": index.num_docs,
            }
        )
        print(
            f"{executor:<10}{workers:>3}  wall {timings.wall:7.2f}  "
            f"cpu {timings.cpu:7.2f}  merge {timings.merge:5.2f}  "
            f"rss {rss:7.0f} МБ  термів {len(index):,}",
            flush=True,
        )
        del index

    base = rows[0]["wall"]
    for row in rows:
        row["speedup"] = base / row["wall"]
    args.out.write_text(
        json.dumps({"results": rows}, ensure_ascii=False, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    # without this guard every spawned worker would re-run the whole benchmark
    main()
