import heapq
import math
from collections import defaultdict
from dataclasses import dataclass
from operator import attrgetter
from typing import Protocol

from findex.index import Index, Posting


class Scorer(Protocol):
    def score(self, term: str, posting: Posting, index: Index) -> float: ...


@dataclass(frozen=True, slots=True)
class TfIdf:
    def score(self, term: str, posting: Posting, index: Index) -> float:
        df = index.df(term)
        if not df:
            return 0.0
        idf = math.log(index.num_docs / df)
        return (1 + math.log(posting.tf)) * idf

    __call__ = score


@dataclass(frozen=True, slots=True)
class BM25:
    k1: float = 1.5
    b: float = 0.75

    def score(self, term: str, posting: Posting, index: Index) -> float:
        df = index.df(term)
        if not df:
            return 0.0
        idf = math.log(1 + (index.num_docs - df + 0.5) / (df + 0.5))
        dl = index.doc_length(posting.doc_id)
        norm = 1 - self.b + self.b * dl / index.avg_doc_length
        tf = posting.tf
        return idf * tf * (self.k1 + 1) / (tf + self.k1 * norm)

    __call__ = score


@dataclass(order=True, frozen=True, slots=True)
class SearchResult:
    # score goes first on purpose: then sorted(results) orders by score
    score: float
    doc_id: int
    title: str = ""
    snippet: str = ""

    def __str__(self) -> str:
        line = f"{self.score:7.3f}  {self.title}"
        return f"{line}\n          {self.snippet}" if self.snippet else line


def score_documents(
    index: Index, terms: list[str], candidates: set[int], scorer: Scorer
) -> dict[int, float]:
    scores: dict[int, float] = defaultdict(float)
    for term in terms:
        for posting in index.iter_postings(term):
            if posting.doc_id in candidates:
                scores[posting.doc_id] += scorer.score(term, posting, index)
    return scores


def top_k(scores: dict[int, float], index: Index, k: int = 10) -> list[SearchResult]:
    results = [
        SearchResult(score, doc_id, index.doc_meta[doc_id].title)
        for doc_id, score in scores.items()
    ]
    return heapq.nlargest(k, results, key=attrgetter("score"))
