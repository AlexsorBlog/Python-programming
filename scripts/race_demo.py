"""Show that the GIL does not protect my own invariants.

    uv run python scripts/race_demo.py
    .venv-ft/Scripts/python scripts/race_demo.py   # free-threaded 3.13t

Eight threads each add 50_000 to the same counter, once without a lock
and once with one. The expected total is 400_000.
"""

import sys
import threading
from collections import Counter

THREADS = 8
PER_THREAD = 50_000
KEY = "kyiv"


def without_lock(shared: Counter[str]) -> None:
    for _ in range(PER_THREAD):
        # read, add, store: three steps, and a thread switch between
        # them loses somebody's increment
        shared[KEY] = shared[KEY] + 1


def with_lock(shared: Counter[str], lock: threading.Lock) -> None:
    for _ in range(PER_THREAD):
        with lock:
            shared[KEY] = shared[KEY] + 1


def run(target, *args) -> int:
    threads = [threading.Thread(target=target, args=args) for _ in range(THREADS)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    return args[0][KEY]


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    gil = sys._is_gil_enabled() if hasattr(sys, "_is_gil_enabled") else True
    expected = THREADS * PER_THREAD

    print(f"Python {sys.version.split()[0]}")
    print(f"sys._is_gil_enabled(): {gil}")
    print(f"очікую: {expected:,}")

    lost = run(without_lock, Counter())
    print(f"без Lock: {lost:,}  (втрачено {expected - lost:,})")

    locked = run(with_lock, Counter(), threading.Lock())
    print(f"з Lock:   {locked:,}  (втрачено {expected - locked:,})")


if __name__ == "__main__":
    main()
