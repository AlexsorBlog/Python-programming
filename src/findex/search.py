import argparse
import logging
import sys
from functools import lru_cache
from pathlib import Path

from findex.decorators import timed
from findex.index import Index
from findex.merge import merge_and, merge_not, merge_or
from findex.query import parse
from findex.scoring import BM25, SearchResult, TfIdf, score_documents, top_k
from findex.snippet import document_text, make_snippet
from findex.tokenize import tokenize

__all__ = [
    "merge_and",
    "merge_not",
    "merge_or",
    "ranked_search",
    "search",
]

log = logging.getLogger("findex.search")
SCORERS = {"bm25": BM25, "tfidf": TfIdf}


def _set_op(op: str, a: list[int], b: list[int]) -> list[int]:
    sa, sb = set(a), set(b)
    result = {"AND": sa & sb, "OR": sa | sb, "NOT": sa - sb}[op]
    return sorted(result)


def parse_query(query: str) -> list[tuple[str, str]]:
    """Flat query -> list of (operator, term). Used by the lab 2 engines."""
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


@lru_cache(maxsize=256)
def _cached_matches(
    index: Index, query: str
) -> tuple[tuple[int, ...], tuple[str, ...]]:
    node = parse(query)
    if node is None:
        return (), ()
    return tuple(node.evaluate(index)), tuple(dict.fromkeys(node.terms()))


@timed
def matching_docs(index: Index, query: str) -> tuple[tuple[int, ...], tuple[str, ...]]:
    result = _cached_matches(index, query)
    log.debug("кеш запитів: %s", _cached_matches.cache_info())
    return result


@timed
def ranked_search(
    index: Index,
    query: str,
    scorer=None,
    k: int = 10,
    snippets: bool = False,
) -> list[SearchResult]:
    scorer = scorer if scorer is not None else BM25()
    doc_ids, terms = matching_docs(index, query)
    if not doc_ids:
        return []

    scores = score_documents(index, list(terms), set(doc_ids), scorer)
    results = top_k(scores, index, k)
    if not snippets:
        return results

    with_snippets = []
    for result in results:
        meta = index.doc_meta[result.doc_id]
        text = document_text(meta)
        with_snippets.append(
            SearchResult(
                result.score,
                result.doc_id,
                result.title,
                make_snippet(text, list(terms)),
            )
        )
    return with_snippets


def main() -> None:
    parser = argparse.ArgumentParser(description="Пошук у збереженому індексі")
    parser.add_argument("index", type=Path, help="файл індексу")
    parser.add_argument("query", help="запит, напр. python AND (async OR await)")
    parser.add_argument("--scorer", choices=tuple(SCORERS), default="bm25")
    parser.add_argument("--k", type=int, default=10, help="скільки результатів")
    parser.add_argument("--no-snippets", action="store_true", help="без сніпетів")
    parser.add_argument(
        "--repeat", action="store_true", help="виконати запит двічі, щоб побачити кеш"
    )
    parser.add_argument(
        "--boolean", action="store_true", help="старий Boolean-пошук без рейтингу"
    )
    parser.add_argument("--engine", choices=("merge", "set"), default="merge")
    parser.add_argument("--verbose", action="store_true", help="показати лог таймінгів")
    args = parser.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.WARNING,
        format="%(name)s | %(message)s",
    )

    from findex.store import open_index

    with open_index(args.index) as index:
        print(f"{index!r}")

        if args.boolean:
            hits = search(index, args.query, args.engine)
            print(f"запит: {args.query}  (engine={args.engine})")
            print(f"знайдено: {len(hits):,} документів")
            for doc_id in hits[: args.k]:
                print(f"  {doc_id:>7}  {index.doc_meta[doc_id].title}")
            return

        for run in range(2 if args.repeat else 1):
            if run:
                print("\n-- той самий запит ще раз (має влучити в кеш) --")
            results = ranked_search(
                index,
                args.query,
                SCORERS[args.scorer](),
                args.k,
                snippets=not args.no_snippets,
            )
            print(f"запит: {args.query}  (scorer={args.scorer})")
            if not results:
                print("нічого не знайдено")
            for result in results:
                print(result)


if __name__ == "__main__":
    main()
