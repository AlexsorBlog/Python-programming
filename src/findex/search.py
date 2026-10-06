import logging
from functools import lru_cache
from typing import Literal

from findex.decorators import timed
from findex.index import DocId, Index
from findex.merge import merge_and, merge_not, merge_or
from findex.query import parse
from findex.scoring import BM25, Scorer, SearchResult, score_documents, top_k
from findex.snippet import document_text, make_snippet

log = logging.getLogger(__name__)

type ScorerName = Literal["bm25", "tfidf"]
type Engine = Literal["merge", "set"]
type BoolOp = Literal["AND", "OR", "NOT"]

__all__ = [
    "boolean_search",
    "matching_docs",
    "ranked_search",
]


def _set_op(op: BoolOp, a: list[DocId], b: list[DocId]) -> list[DocId]:
    sa, sb = set(a), set(b)
    result = {"AND": sa & sb, "OR": sa | sb, "NOT": sa - sb}[op]
    return sorted(result)


def parse_flat(query: str) -> list[tuple[BoolOp, str]]:
    """Flat query -> list of (operator, term). Used by the lab 2 engines."""
    from findex.tokenize import tokenize

    steps: list[tuple[BoolOp, str]] = []
    op: BoolOp = "AND"
    for word in query.split():
        upper = word.upper()
        if upper in ("AND", "OR", "NOT"):
            op = upper  # pyright: ignore[reportAssignmentType]  # upper is one of the three
            continue
        terms = list(tokenize(word))
        if not terms:
            continue
        steps.append((op, terms[0]))
        op = "AND"
    return steps


def boolean_search(index: Index, query: str, engine: Engine = "merge") -> list[DocId]:
    steps = parse_flat(query)
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
) -> tuple[tuple[DocId, ...], tuple[str, ...]]:
    node = parse(query)
    if node is None:
        return (), ()
    return tuple(node.evaluate(index)), tuple(dict.fromkeys(node.terms()))


@timed
def matching_docs(
    index: Index, query: str
) -> tuple[tuple[DocId, ...], tuple[str, ...]]:
    result = _cached_matches(index, query)
    log.debug("кеш запитів: %s", _cached_matches.cache_info())
    return result


@timed
def ranked_search(
    index: Index,
    query: str,
    scorer: Scorer | None = None,
    k: int = 10,
    snippets: bool = False,
    marker: tuple[str, str] = ("[", "]"),
) -> list[SearchResult]:
    scorer = scorer if scorer is not None else BM25()
    doc_ids, terms = matching_docs(index, query)
    if not doc_ids:
        return []

    scores = score_documents(index, list(terms), set(doc_ids), scorer)
    results = top_k(scores, index, k)
    if not snippets:
        return results

    with_snippets: list[SearchResult] = []
    for result in results:
        meta = index.doc_meta[result.doc_id]
        text = document_text(meta)
        with_snippets.append(
            SearchResult(
                result.score,
                result.doc_id,
                result.title,
                make_snippet(text, list(terms), marker=marker),
            )
        )
    return with_snippets


# the old name from labs 2-3, kept so older commands and scripts still work
search = boolean_search
