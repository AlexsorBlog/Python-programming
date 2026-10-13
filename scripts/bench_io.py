"""The counter-experiment: pure I/O, where threads do scale.

    uv run python scripts/bench_io.py data/shards --repeats 3

Each task only reads a shard off disk (no tokenizing), so the GIL is
released for most of the work.
"""

import argparse
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from findex.corpus import corpus_paths


def read_bytes(path: Path) -> int:
    total = 0
    with path.open("rb") as f:
        while chunk := f.read(1 << 20):
            total += len(chunk)
    return total


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="Бенчмарк вводу-виводу")
    parser.add_argument("root", type=Path)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()

    paths = corpus_paths(args.root)
    print(f"файлів: {len(paths)}")

    baseline = 0.0
    print("| Воркерів | Wall (с, медіана) | Прискорення |")
    print("|---|---|---|")
    for workers in (1, 2, 4, 8, 16):
        walls: list[float] = []
        for _ in range(args.repeats):
            start = time.perf_counter()
            if workers == 1:
                for path in paths:
                    read_bytes(path)
            else:
                with ThreadPoolExecutor(max_workers=workers) as pool:
                    list(pool.map(read_bytes, paths))
            walls.append(time.perf_counter() - start)
        wall = statistics.median(walls)
        baseline = baseline or wall
        print(f"| {workers} | {wall:.2f} | {baseline / wall:.2f}× |", flush=True)


if __name__ == "__main__":
    main()
