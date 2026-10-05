import sys
import timeit
from pathlib import Path

from findex.search import search
from findex.store import load


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    index = load(Path(sys.argv[1]))

    by_length = sorted(index.postings, key=lambda t: len(index.doc_ids(t)))
    rare = by_length[:2]
    common = by_length[-2:]

    print(f"{'запит':<28}{'документів':>12}{'merge, мс':>12}{'set, мс':>12}")
    for pair in (common, rare, [common[0], rare[0]]):
        query = " ".join(pair)
        hits = len(search(index, query))
        times = {
            engine: min(
                timeit.repeat(
                    lambda e=engine, q=query: search(index, q, e), number=5, repeat=5
                )
            )
            / 5
            * 1000
            for engine in ("merge", "set")
        }
        sizes = "+".join(str(len(index.doc_ids(t))) for t in pair)
        print(f"{query:<28}{sizes:>12}{times['merge']:>12.3f}{times['set']:>12.3f}")
        print(f"{'  (знайдено ' + str(hits) + ')':<28}")


if __name__ == "__main__":
    main()
