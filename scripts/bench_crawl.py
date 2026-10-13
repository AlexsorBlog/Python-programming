"""Crawl the same seed at three concurrency levels.

uv run --group bench python scripts/bench_crawl.py https://books.toscrape.com/
"""

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path
from typing import Any

import psutil

from findex.crawler import CrawlStats, crawl


async def one_run(seed: str, max_pages: int, concurrency: int) -> dict[str, Any]:
    stats = CrawlStats()
    proc = psutil.Process()
    start = time.perf_counter()
    pages = 0
    async for _page in crawl(
        [seed],
        max_pages=max_pages,
        concurrency=concurrency,
        per_host=min(concurrency, 4),
        host_delay=0.1,
        stats=stats,
    ):
        pages += 1
    wall = time.perf_counter() - start
    info: Any = proc.memory_info()
    rss = float(getattr(info, "peak_wset", info.rss)) / 1048576
    return {
        "concurrency": concurrency,
        "pages": pages,
        "wall": wall,
        "pps": pages / wall if wall else 0.0,
        "errors": stats.errors,
        "rss": rss,
    }


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="Бенчмарк краулера")
    parser.add_argument("seed")
    parser.add_argument("--max-pages", type=int, default=200)
    parser.add_argument("--out", type=Path, default=Path("bench_crawl.json"))
    args = parser.parse_args()

    rows: list[dict[str, Any]] = []
    for concurrency in (1, 5, 20):
        row = asyncio.run(one_run(args.seed, args.max_pages, concurrency))
        rows.append(row)
        print(
            f"concurrency {concurrency:>3}: {row['pages']} сторінок за "
            f"{row['wall']:6.1f} с  ({row['pps']:5.2f} стор/с), "
            f"помилок {row['errors']}, RSS {row['rss']:.0f} МБ",
            flush=True,
        )

    base = rows[0]["wall"]
    for row in rows:
        row["speedup"] = base / row["wall"]
    args.out.write_text(
        json.dumps({"seed": args.seed, "results": rows}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print()
    print("| Конкурентність | Сторінок | Wall (с) | Сторінок/с | Помилок | Пік RSS |")
    print("|---|---|---|---|---|---|")
    for row in rows:
        print(
            f"| {row['concurrency']} | {row['pages']} | {row['wall']:.1f} | "
            f"{row['pps']:.2f} | {row['errors']} | {row['rss']:.0f} МБ |"
        )


if __name__ == "__main__":
    main()
