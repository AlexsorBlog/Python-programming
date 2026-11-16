"""mode=semantic and mode=hybrid on top of either index type."""

from functools import lru_cache
from pathlib import Path

from findex.index import DocId, Index
from findex.numpy_index import NumpyIndex
from findex.scoring import BM25, SearchResult
from findex.search import ranked_search, ranked_search_numpy
from findex.semantic import Mode, SemanticIndex, rrf
from findex.snippet import document_text, make_snippet
from findex.tokenize import tokenize

# how deep each list goes before fusion: RRF needs more than the final k
FUSION_DEPTH = 100


@lru_cache(maxsize=4)
def load_semantic(folder: str) -> SemanticIndex:
    return SemanticIndex.load(Path(folder))


def lexical_ids(index: Index | NumpyIndex, query: str, k: int) -> list[DocId]:
    if isinstance(index, NumpyIndex):
        return [r.doc_id for r in ranked_search_numpy(index, query, "bm25", k=k)]
    return [r.doc_id for r in ranked_search(index, query, BM25(), k=k)]


def semantic_results(
    index: Index | NumpyIndex,
    embeddings: Path,
    query: str,
    mode: Mode,
    k: int = 10,
    snippets: bool = False,
    marker: tuple[str, str] = ("[", "]"),
) -> list[SearchResult]:
    sem = load_semantic(str(embeddings))
    if sem.n_docs != index.num_docs:
        raise ValueError(
            f"ембединги для {sem.n_docs} документів, а індекс має {index.num_docs}: "
            "їх рахували з іншого корпусу"
        )

    if mode == "semantic":
        ranked = sem.search(query, k)
    else:
        try:
            lexical = lexical_ids(index, query, FUSION_DEPTH)
        except ValueError:
            # a query the parser can't read still has a meaning to embed
            lexical = []
        semantic = [doc_id for doc_id, _ in sem.search(query, FUSION_DEPTH)]
        ranked = rrf([lexical, semantic], k=k)

    # a meaning query has no boolean terms; highlight its words, minus "in", "of"
    terms = [t for t in dict.fromkeys(tokenize(query)) if len(t) > 2]
    results: list[SearchResult] = []
    for doc_id, score in ranked:
        meta = index.doc_meta[doc_id]
        snippet = (
            make_snippet(document_text(meta), terms, marker=marker) if snippets else ""
        )
        results.append(SearchResult(score, doc_id, meta.title, snippet))
    return results
