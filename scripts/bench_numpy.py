"""Lab 3 scorer (Python loops over Posting objects) vs NumPy arrays.

    uv run python scripts/bench_numpy.py data/full.pkl data/full_np

Both scorers get the same candidate set from the same boolean parser, so the
table measures only scoring + top-k. Results are checked to be identical.
"""

import statistics
import sys
import time
import timeit
from pathlib import Path

import numpy as np

from findex.numpy_index import (
    from_index,
    is_numpy_index,
    load_numpy,
    save_numpy,
    score_numpy,
    top_k_numpy,
)
from findex.scoring import BM25, score_documents, top_k
from findex.search import matching_docs
from findex.store import load

QUERIES = [
    ("частий терм", "the"),
    ("рідкісний терм", "kyiv"),
    ("три терми", "united states president"),
]


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    pickle_path, np_dir = Path(sys.argv[1]), Path(sys.argv[2])

    start = time.perf_counter()
    index = load(pickle_path)
    pickle_load = time.perf_counter() - start
    print(f"pickle.load повного індексу: {pickle_load:.1f} с ({index!r})", flush=True)

    if not is_numpy_index(np_dir):
        start = time.perf_counter()
        save_numpy(from_index(index), np_dir)
        print(f"перетворення + збереження .npy: {time.perf_counter() - start:.1f} с")

    loads: list[float] = []
    for _ in range(5):
        start = time.perf_counter()
        nidx = load_numpy(np_dir)
        loads.append(time.perf_counter() - start)
    size = sum(f.stat().st_size for f in np_dir.iterdir()) / 2**20
    print(
        f"load_numpy (mmap): {statistics.median(loads) * 1000:.0f} мс, "
        f"на диску {size:.0f} МБ ({nidx!r})\n",
        flush=True,
    )

    bm25 = BM25()
    print("| Запит | Кандидатів | Постингів | Лаба 3 (мс) | NumPy (мс) | Прискорення |")
    print("|---|---|---|---|---|---|")
    for label, query in QUERIES:
        doc_ids, terms = matching_docs(index, query)
        postings = sum(index.df(t) for t in terms)

        def lab3(terms=terms, doc_ids=doc_ids) -> list[tuple[int, float]]:
            scores = score_documents(index, list(terms), set(doc_ids), bm25)
            return [(r.doc_id, r.score) for r in top_k(scores, index, 10)]

        def lab8(terms=terms, doc_ids=doc_ids) -> list[tuple[int, float]]:
            scores, touched = score_numpy(nidx, terms)
            cands = np.asarray(doc_ids, dtype=np.int32)
            return top_k_numpy(scores, cands[touched[cands]], 10)

        assert lab3() == lab8(), f"порядок розійшовся на {query!r}"
        slow = min(timeit.repeat(lab3, number=1, repeat=3)) * 1000
        fast = min(timeit.repeat(lab8, number=1, repeat=7)) * 1000
        print(
            f"| {label}: `{query}` | {len(doc_ids):,} | {postings:,} | "
            f"{slow:.1f} | {fast:.2f} | {slow / fast:.0f}× |",
            flush=True,
        )


if __name__ == "__main__":
    main()
