"""Threaded indexing on the free-threaded build (no psutil needed).

.venv-ft/Scripts/python scripts/bench_freethreaded.py data/shards --repeats 3
"""

import argparse
import json
import os
import statistics
import sys
from pathlib import Path
from typing import Any

from findex.parallel import build_parallel


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="Бенчмарк потоків без GIL")
    parser.add_argument("root", type=Path)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--out", type=Path, default=Path("bench_freethreaded.json"))
    args = parser.parse_args()

    gil = sys._is_gil_enabled() if hasattr(sys, "_is_gil_enabled") else True
    cpus = os.cpu_count() or 1
    print(f"Python {sys.version.split()[0]}, GIL увімкнений: {gil}")

    results: list[dict[str, Any]] = []
    for workers in sorted({1, 2, 4, 8, cpus}):
        walls: list[float] = []
        cpu_times: list[float] = []
        merges: list[float] = []
        for _ in range(args.repeats):
            _, timings = build_parallel(args.root, workers, "threads")
            walls.append(timings.wall)
            cpu_times.append(timings.cpu)
            merges.append(timings.merge)
        row = {
            "executor": "threads (3.13t)",
            "workers": workers,
            "wall": statistics.median(walls),
            "cpu": statistics.median(cpu_times),
            "merge": statistics.median(merges),
        }
        results.append(row)
        print(
            f"threads {workers:>3} воркер(ів): wall {row['wall']:6.2f} с, "
            f"CPU {row['cpu']:7.2f} с, merge {row['merge']:5.2f} с",
            flush=True,
        )

    args.out.write_text(
        json.dumps({"gil": gil, "cpus": cpus, "results": results}, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
