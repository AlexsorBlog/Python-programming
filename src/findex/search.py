import argparse
import sys
import time
import tracemalloc
from pathlib import Path

from findex.index import Index
from findex.tokenize import tokenize


def merge_and(a: list[int], b: list[int]) -> list[int]:
    out: list[int] = []
    i = j = 0
    while i < len(a) and j < len(b):
        if a[i] == b[j]:
            out.append(a[i])
            i += 1
            j += 1
        elif a[i] < b[j]:
            i += 1
        else:
            j += 1
    return out


def merge_or(a: list[int], b: list[int]) -> list[int]:
    out: list[int] = []
    i = j = 0
    while i < len(a) and j < len(b):
        if a[i] == b[j]:
            out.append(a[i])
            i += 1
            j += 1
        elif a[i] < b[j]:
            out.append(a[i])
            i += 1
        else:
            out.append(b[j])
            j += 1
    out.extend(a[i:])
    out.extend(b[j:])
    return out


def merge_not(a: list[int], b: list[int]) -> list[int]:
    out: list[int] = []
    i = j = 0
    while i < len(a) and j < len(b):
        if a[i] == b[j]:
            i += 1
            j += 1
        elif a[i] < b[j]:
            out.append(a[i])
            i += 1
        else:
            j += 1
    out.extend(a[i:])
    return out


def _set_op(op: str, a: list[int], b: list[int]) -> list[int]:
    sa, sb = set(a), set(b)
    result = {"AND": sa & sb, "OR": sa | sb, "NOT": sa - sb}[op]
    return sorted(result)


def parse_query(query: str) -> list[tuple[str, str]]:
    """Query -> list of (operator, term). AND is implicit."""
    steps: list[tuple[str, str]] = []
    op = "AND"
    for word in query.split():
        if word.upper() in ("AND", "OR", "NOT"):
            op = word.upper()
            continue
        terms = list(tokenize(word))
        if not terms:
            continue
        steps.append((op, terms[0]))
        op = "AND"
    return steps


def search(index: Index, query: str, engine: str = "merge") -> list[int]:
    steps = parse_query(query)
    if not steps:
        return []

    first_op, first_term = steps[0]
    result = index.doc_ids(first_term)
    if first_op == "NOT":
        result = merge_not(sorted(index.doc_meta), result)

    for op, term in steps[1:]:
        other = index.doc_ids(term)
        if engine == "set":
            result = _set_op(op, result, other)
        elif op == "AND":
            result = merge_and(result, other)
        elif op == "OR":
            result = merge_or(result, other)
        else:
            result = merge_not(result, other)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Пошук у збереженому індексі")
    parser.add_argument("index", type=Path, help="файл індексу")
    parser.add_argument("query", help="запит, напр. 'kyiv OR lviv NOT river'")
    parser.add_argument("--engine", choices=("merge", "set"), default="merge")
    parser.add_argument(
        "--top", type=int, default=10, help="скільки результатів показати"
    )
    args = parser.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")

    from findex.stats import human
    from findex.store import load

    tracemalloc.start()
    start = time.perf_counter()
    index = load(args.index)
    load_time = time.perf_counter() - start

    start = time.perf_counter()
    hits = search(index, args.query, args.engine)
    search_time = time.perf_counter() - start
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    print(f"запит:          {args.query}  (engine={args.engine})")
    print(f"знайдено:       {len(hits):,} документів")
    print(f"час завантаження індексу: {load_time:.2f} с")
    print(f"час пошуку:     {search_time * 1000:.2f} мс")
    print(f"пікова пам'ять: {human(peak)}")
    print()
    for doc_id in hits[: args.top]:
        print(f"  {doc_id:>7}  {index.doc_meta[doc_id].title}")


if __name__ == "__main__":
    main()
