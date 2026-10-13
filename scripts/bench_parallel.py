"""Benchmark serial / threads / processes on the sharded corpus.

    uv run --group bench python scripts/bench_parallel.py data/shards --repeats 3

Writes bench_parallel.json (raw numbers), prints a markdown table and
saves speedup.png next to the README.
"""

import argparse
import json
import os
import statistics
import sys
from pathlib import Path
from typing import Any

import psutil

from findex.parallel import build_parallel


def run_once(
    root: Path, workers: int, executor: str, repr_kind: str = "slots"
) -> dict[str, float]:
    proc = psutil.Process()
    _, timings = build_parallel(
        root,
        workers,
        executor,  # type: ignore[arg-type]  # str from argv
        repr_kind=repr_kind,  # type: ignore[arg-type]  # str from argv
    )
    info: Any = proc.memory_info()
    # peak_wset on Windows, rss elsewhere: peak memory of the parent only
    peak = float(getattr(info, "peak_wset", info.rss))
    return {
        "wall": timings.wall,
        "cpu": timings.cpu,
        "merge": timings.merge,
        "rss": peak,
        "chunks": float(timings.chunks),
    }


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="Бенчмарк паралельної індексації")
    parser.add_argument("root", type=Path)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--out", type=Path, default=Path("bench_parallel.json"))
    parser.add_argument("--repr", dest="repr_kind", default="slots")
    args = parser.parse_args()

    cpus = os.cpu_count() or 1
    worker_counts = sorted({1, 2, 4, 8, cpus})
    configs: list[tuple[str, int]] = [("serial", 1)]
    for executor in ("threads", "processes"):
        configs += [(executor, n) for n in worker_counts]

    results: list[dict[str, Any]] = []
    for executor, workers in configs:
        runs = [
            run_once(args.root, workers, executor, args.repr_kind)
            for _ in range(args.repeats)
        ]
        row: dict[str, Any] = {
            "executor": executor,
            "workers": workers,
            "wall": statistics.median(r["wall"] for r in runs),
            "wall_min": min(r["wall"] for r in runs),
            "wall_max": max(r["wall"] for r in runs),
            "cpu": statistics.median(r["cpu"] for r in runs),
            "merge": statistics.median(r["merge"] for r in runs),
            "rss": max(r["rss"] for r in runs),
            "chunks": runs[0]["chunks"],
        }
        results.append(row)
        print(
            f"{executor:<10} {workers:>3} воркер(ів): "
            f"wall {row['wall']:6.2f} с, CPU {row['cpu']:7.2f} с, "
            f"merge {row['merge']:5.2f} с, RSS {row['rss'] / 1024 / 1024:7.1f} МБ",
            flush=True,
        )

    baseline = results[0]["wall"]
    for row in results:
        row["speedup"] = baseline / row["wall"]

    args.out.write_text(
        json.dumps(
            {"cpus": cpus, "repr": args.repr_kind, "results": results},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    header = (
        "| Executor | Воркерів | Wall (с, медіана) | CPU (с) | "
        "Пік RSS | Merge (с) | Прискорення |"
    )
    print()
    print(header)
    print("|---|---|---|---|---|---|---|")
    for row in results:
        print(
            f"| {row['executor']} | {row['workers']} | {row['wall']:.2f} | "
            f"{row['cpu']:.2f} | {row['rss'] / 1024 / 1024:.0f} МБ | "
            f"{row['merge']:.2f} | {row['speedup']:.2f}× |"
        )


if __name__ == "__main__":
    main()
