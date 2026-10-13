"""Speedup vs workers plot.

uv run --group bench python scripts/plot_speedup.py
"""

import json
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # backend must be set first

SOURCES = [
    ("bench_parallel.json", "{executor} (slots)"),
    ("bench_parallel_array.json", "{executor} (array)"),
    ("bench_freethreaded.json", "threads 3.13t без GIL (slots)"),
]


def load(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    payload: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return payload


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    series: dict[str, list[tuple[int, float]]] = {}
    baselines: dict[str, float] = {}

    for name, label in SOURCES:
        payload = load(Path(name))
        if not payload:
            continue
        rows: list[dict[str, Any]] = payload["results"]
        serial = next((r["wall"] for r in rows if r["executor"] == "serial"), None)
        if serial is None:
            # the free-threaded run has no serial row: compare with its own 1 thread
            serial = next(r["wall"] for r in rows if r["workers"] == 1)
        baselines[name] = serial

        for row in rows:
            if row["executor"] == "serial":
                continue
            key = label.format(executor=row["executor"])
            series.setdefault(key, []).append((row["workers"], serial / row["wall"]))

    if not series:
        raise SystemExit("спочатку запусти scripts/bench_parallel.py")

    fig, ax = plt.subplots(figsize=(8, 5.5))
    workers = sorted({w for points in series.values() for w, _ in points})
    ax.plot(workers, workers, "--", color="gray", label="ідеальне лінійне")

    for key in sorted(series):
        points = sorted(series[key])
        ax.plot([w for w, _ in points], [s for _, s in points], "o-", label=key)

    ax.set_xlabel("воркерів")
    ax.set_ylabel("прискорення відносно серійного")
    ax.set_title(
        "Паралельна індексація 20 000 документів\ni7-12650H, 10 ядер / 16 потоків"
    )
    ax.set_xticks(workers)
    ax.set_yscale("log", base=2)
    ax.set_yticks([0.5, 1, 2, 4, 8, 16])
    ax.get_yaxis().set_major_formatter(plt.ScalarFormatter())
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig("speedup.png", dpi=130)
    print("speedup.png збережено")


if __name__ == "__main__":
    main()
