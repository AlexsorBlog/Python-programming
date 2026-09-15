import argparse
import sys
import time
import tracemalloc
from collections import Counter
from dataclasses import dataclass
from itertools import islice
from pathlib import Path

from findex.corpus import iter_documents
from findex.tokenize import tokenize


@dataclass
class Stats:
    documents: int
    tokens: int
    counts: Counter


def lazy_stats(root: Path, limit: int | None = None) -> Stats:
    counts: Counter = Counter()
    n_docs = 0
    for doc in islice(iter_documents(root), limit):
        n_docs += 1
        counts.update(tokenize(doc.text))
    return Stats(n_docs, counts.total(), counts)


def eager_stats(root: Path, limit: int | None = None) -> Stats:
    # lists on purpose, to compare with the lazy version
    docs = list(islice(iter_documents(root), limit))
    tokenized = [list(tokenize(doc.text)) for doc in docs]
    all_tokens = [tok for doc_tokens in tokenized for tok in doc_tokens]
    return Stats(len(docs), len(all_tokens), Counter(all_tokens))


def human(n: float) -> str:
    for unit in ("Б", "КБ", "МБ", "ГБ"):
        if n < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} ТБ"


def main() -> None:
    parser = argparse.ArgumentParser(description="Статистика корпусу")
    parser.add_argument("root", type=Path, help="файл або папка з корпусом")
    parser.add_argument(
        "--limit", type=int, default=None, help="тільки перші N документів"
    )
    parser.add_argument(
        "--top", type=int, default=50, help="скільки топ-термів вивести"
    )
    parser.add_argument("--eager", action="store_true", help="версія через списки")
    args = parser.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")

    run = eager_stats if args.eager else lazy_stats

    tracemalloc.start()
    start = time.perf_counter()
    stats = run(args.root, args.limit)
    elapsed = time.perf_counter() - start
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    print(f"режим:          {'eager (списки)' if args.eager else 'lazy (генератори)'}")
    print(f"документів:     {stats.documents:,}")
    print(f"токенів:        {stats.tokens:,}")
    print(f"словник:        {len(stats.counts):,}")
    print(f"час:            {elapsed:.2f} с")
    print(f"пікова пам'ять: {human(peak)}")
    print(f"\nтоп-{args.top} термів:")
    for rank, (term, count) in enumerate(stats.counts.most_common(args.top), start=1):
        print(f"{rank:>3}. {term:<15} {count:,}")


if __name__ == "__main__":
    main()
