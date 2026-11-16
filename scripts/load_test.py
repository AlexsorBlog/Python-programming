"""A small oha-like load test: requests/s and p50/p95/p99.

    uv run python scripts/load_test.py http://127.0.0.1:8000/search -c 50 --clients 4

Rotates through a list of queries, so the lru_cache on the query path does
not turn the test into "the same answer from memory" over and over.

--clients matters: one Python client process tops out around 160 requests/s
on my machine (it parses every JSON reply), so a fast server would look slow.
With several client processes the load generator stops being the bottleneck.
"""

import argparse
import asyncio
import statistics
import sys
import time
from multiprocessing import Pool

import httpx

QUERIES = [
    "chernobyl disaster",
    "united states president",
    "music rock band",
    "solar system planet",
    "world war",
    "ancient rome empire",
    "computer software",
    "football world cup",
    "mount everest",
    "python programming language",
    "the city",
    "river OR lake",
]


def percentile(sorted_values: list[float], p: float) -> float:
    if not sorted_values:
        return 0.0
    i = min(len(sorted_values) - 1, max(0, round(p / 100 * len(sorted_values)) - 1))
    return sorted_values[i]


async def collect(
    urls: list[str], duration: float, concurrency: int, offset: int
) -> tuple[list[float], int, float]:
    latencies: list[float] = []
    errors = 0
    counter = offset
    limits = httpx.Limits(
        max_connections=concurrency, max_keepalive_connections=concurrency
    )

    async def worker(client: httpx.AsyncClient, stop_at: float) -> None:
        nonlocal errors, counter
        while time.perf_counter() < stop_at:
            q = QUERIES[counter % len(QUERIES)]
            # several URLs = several server processes, picked round-robin
            url = urls[counter % len(urls)]
            counter += 1
            start = time.perf_counter()
            try:
                r = await client.get(url, params={"q": q, "k": 10})
                if r.status_code != 200:
                    errors += 1
                    continue
            except httpx.HTTPError:
                errors += 1
                continue
            latencies.append((time.perf_counter() - start) * 1000)

    async with httpx.AsyncClient(timeout=60, limits=limits) as client:
        for url in urls:
            await client.get(url, params={"q": QUERIES[0]})
        started = time.perf_counter()
        stop_at = started + duration
        await asyncio.gather(*(worker(client, stop_at) for _ in range(concurrency)))
        elapsed = time.perf_counter() - started
    return latencies, errors, elapsed


def _client(args: tuple[list[str], float, int, int]) -> tuple[list[float], int, float]:
    return asyncio.run(collect(*args))


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="Навантажувальний тест")
    parser.add_argument("url", help="одна адреса або кілька через кому")
    parser.add_argument("--duration", type=float, default=20)
    parser.add_argument("-c", "--concurrency", type=int, default=50)
    parser.add_argument("--clients", type=int, default=4, help="процесів-клієнтів")
    args = parser.parse_args()

    urls = args.url.split(",")
    per_client = max(1, args.concurrency // args.clients)
    jobs = [(urls, args.duration, per_client, i * 7) for i in range(args.clients)]
    with Pool(args.clients) as pool:
        parts = pool.map(_client, jobs)

    latencies = sorted(x for part in parts for x in part[0])
    errors = sum(part[1] for part in parts)
    elapsed = max(part[2] for part in parts)

    print(f"URL:           {', '.join(urls)}")
    print(f"одночасно:     {per_client * args.clients} ({args.clients} клієнти)")
    print(f"тривалість:    {elapsed:.1f} с")
    print(f"запитів:       {len(latencies)} (помилок {errors})")
    print(f"запитів/с:     {len(latencies) / elapsed:.1f}")
    print(f"p50:           {percentile(latencies, 50):.1f} мс")
    print(f"p95:           {percentile(latencies, 95):.1f} мс")
    print(f"p99:           {percentile(latencies, 99):.1f} мс")
    if latencies:
        print(f"середнє:       {statistics.fmean(latencies):.1f} мс")


if __name__ == "__main__":
    main()
