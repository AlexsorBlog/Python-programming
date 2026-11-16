"""20 concurrent /search requests: def route vs async def route.

    INDEX_PATH=data/index.pkl uv run python scripts/bench_def_vs_async.py

The server runs in a SEPARATE process. My first version ran uvicorn in the
same event loop as the httpx client, and then a blocking async route froze
the client too, so the /health probe was sent late and looked instant.
"""

import asyncio
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

import httpx

QUERY = "the OR of OR and OR city"
HEAVY = "the OR of OR and OR in OR a OR is OR to OR was"
BURST = 20
PORT = 8765


async def burst(client: httpx.AsyncClient, path: str) -> dict[str, float]:
    async def one() -> float:
        start = time.perf_counter()
        r = await client.get(path, params={"q": QUERY, "k": 10})
        r.raise_for_status()
        return (time.perf_counter() - start) * 1000

    async def health_probe() -> float:
        await asyncio.sleep(0.05)
        start = time.perf_counter()
        await client.get("/health")
        return (time.perf_counter() - start) * 1000

    start = time.perf_counter()
    probe = asyncio.create_task(health_probe())
    latencies = sorted(await asyncio.gather(*(one() for _ in range(BURST))))
    wall = time.perf_counter() - start
    return {
        "wall": wall,
        "p50": statistics.median(latencies),
        "p95": latencies[int(len(latencies) * 0.95) - 1],
        "max": latencies[-1],
        "health": await probe,
    }


async def single_heavy(client: httpx.AsyncClient, path: str) -> dict[str, float]:
    """One slow request, and a /health probe sent 20 ms after it."""

    async def heavy() -> float:
        start = time.perf_counter()
        r = await client.get(path, params={"q": HEAVY, "k": 50})
        r.raise_for_status()
        return (time.perf_counter() - start) * 1000

    async def probe() -> float:
        await asyncio.sleep(0.02)
        start = time.perf_counter()
        await client.get("/health")
        return (time.perf_counter() - start) * 1000

    search_ms, health_ms = await asyncio.gather(heavy(), probe())
    return {"search": search_ms, "health": health_ms}


async def measure() -> None:
    async with httpx.AsyncClient(
        base_url=f"http://127.0.0.1:{PORT}", timeout=120
    ) as client:
        # warm up both paths so the first burst doesn't pay for imports
        await client.get("/search", params={"q": QUERY})
        await client.get("/search_async", params={"q": QUERY})

        print(f"запит: {QUERY!r}, {BURST} одночасних запитів, медіана з 3 прогонів\n")
        print(
            "| Маршрут | Wall (с) | p50 (мс) | p95 (мс) | max (мс) "
            "| /health під час (мс) |"
        )
        print("|---|---|---|---|---|---|")
        for label, path in (("def", "/search"), ("async def", "/search_async")):
            runs = [await burst(client, path) for _ in range(3)]
            mid = sorted(runs, key=lambda r: r["wall"])[1]
            print(
                f"| `{label}` | {mid['wall']:.2f} | {mid['p50']:.0f} | "
                f"{mid['p95']:.0f} | {mid['max']:.0f} | {mid['health']:.0f} |",
                flush=True,
            )

        print("\nОдин важкий запит (k=50) і проба /health через 20 мс, медіана з 5:\n")
        print("| Маршрут | важкий запит (мс) | /health під час (мс) |")
        print("|---|---|---|")
        for label, path in (("def", "/search"), ("async def", "/search_async")):
            runs = [await single_heavy(client, path) for _ in range(5)]
            mid = sorted(runs, key=lambda r: r["health"])[2]
            print(
                f"| `{label}` | {mid['search']:.0f} | {mid['health']:.1f} |",
                flush=True,
            )


def wait_ready() -> None:
    deadline = time.time() + 180
    url = f"http://127.0.0.1:{PORT}/health"
    while time.time() < deadline:
        try:
            if httpx.get(url, timeout=2).status_code == 200:
                return
        except httpx.HTTPError:
            pass
        time.sleep(0.5)
    raise SystemExit("сервер не піднявся")


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    scripts = Path(__file__).parent
    server = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "uvicorn",
            "bench_app:app",
            "--app-dir",
            str(scripts),
            "--port",
            str(PORT),
            "--log-level",
            "warning",
        ],
        env=os.environ.copy(),
    )
    try:
        wait_ready()
        asyncio.run(measure())
    finally:
        server.terminate()
        server.wait(timeout=30)


if __name__ == "__main__":
    main()
