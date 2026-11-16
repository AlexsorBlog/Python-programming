"""Run the local rows of the load table one after another.

    INDEX_PATH=data/demo.pkl uv run python scripts/load_matrix.py

Each row gets a fresh uvicorn process, so nothing leaks between them.
"""

import os
import subprocess
import sys
import time
from pathlib import Path

import httpx

PORT = 8800
# on Windows `uvicorn --workers N` half-crashes (WinError 10022 on
# sock.listen in the spawned children), so N processes means N separate
# uvicorn instances on N ports, and the load test spreads requests over them
ROWS = [
    ("1 воркер, async def", 1, "/search_async"),
    ("1 воркер, def", 1, "/search"),
    ("4 процеси, def", 4, "/search"),
]


def wait_ready(port: int) -> None:
    deadline = time.time() + 180
    while time.time() < deadline:
        try:
            if (
                httpx.get(f"http://127.0.0.1:{port}/health", timeout=2).status_code
                == 200
            ):
                return
        except httpx.HTTPError:
            pass
        time.sleep(0.5)
    raise SystemExit("сервер не піднявся")


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    scripts = Path(__file__).parent
    for label, processes, path in ROWS:
        print(f"\n=== {label}", flush=True)
        ports = [PORT + i for i in range(processes)]
        servers = [
            subprocess.Popen(
                [
                    sys.executable,
                    "-m",
                    "uvicorn",
                    "bench_app:app",
                    "--app-dir",
                    str(scripts),
                    "--port",
                    str(port),
                    "--log-level",
                    "warning",
                ],
                env=os.environ.copy(),
            )
            for port in ports
        ]
        try:
            for port in ports:
                wait_ready(port)
            urls = ",".join(f"http://127.0.0.1:{p}{path}" for p in ports)
            subprocess.run(
                [
                    sys.executable,
                    str(scripts / "load_test.py"),
                    urls,
                    "--duration",
                    "20",
                    "-c",
                    "50",
                ],
                check=True,
            )
        finally:
            for server in servers:
                server.terminate()
            for server in servers:
                server.wait(timeout=30)
            time.sleep(2)


if __name__ == "__main__":
    main()
