"""Semantic search: embeddings of document fragments and cosine similarity.

    findex embed demo/simplewiki-4000.jsonl --out data/emb_demo

Every document is cut into fragments of ~CHUNK_WORDS words. Each fragment
becomes a 384-dim vector (all-MiniLM-L6-v2), normalized to length 1, so the
cosine similarity with the query is just a dot product: matrix @ query.
A document scores as its best fragment.
"""

import json
import logging
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from functools import lru_cache
from itertools import islice
from pathlib import Path
from typing import Any, Literal, Protocol

import numpy as np
import numpy.typing as npt

from findex.corpus import iter_documents
from findex.index import DocId

log = logging.getLogger(__name__)

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_WORDS = 80
MAX_CHUNKS = 2
RRF_K = 60

type Mode = Literal["lexical", "semantic", "hybrid"]


class _Embedder(Protocol):
    def embed(
        self, documents: str | Iterable[str], batch_size: int = ...
    ) -> Iterable[Any]: ...


@lru_cache(maxsize=1)
def embedder() -> _Embedder:
    """Loaded once per process; the first call downloads the model (~90 MB)."""
    from fastembed import TextEmbedding

    model: _Embedder = TextEmbedding(MODEL_NAME)
    return model


def chunks(text: str, words: int = CHUNK_WORDS, limit: int = MAX_CHUNKS) -> list[str]:
    tokens = text.split()
    if not tokens:
        return []
    return [
        " ".join(tokens[i : i + words])
        for i in range(0, min(len(tokens), words * limit), words)
    ]


def normalize_rows(matrix: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (matrix / norms).astype(np.float32)


def embed_texts(texts: Sequence[str], batch_size: int = 64) -> npt.NDArray[np.float32]:
    """Embed in order of length, then put rows back where they belong.

    A batch is padded to its longest text, so mixing a 40-token fragment with
    a 300-token one makes the short one pay for the long one. Sorting first
    turned out to be most of the speed (see the README).
    """
    if not texts:
        return np.zeros((0, 384), dtype=np.float32)
    order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
    vectors = np.array(
        list(embedder().embed([texts[i] for i in order], batch_size=batch_size)),
        dtype=np.float32,
    )
    result = np.empty_like(vectors)
    result[order] = vectors
    return normalize_rows(result)


def build_embeddings(
    root: Path, out: Path, limit: int | None = None, batch: int = 512
) -> tuple[int, int]:
    """Stream the corpus, embed fragments batch by batch, save .npy files."""
    out.mkdir(parents=True, exist_ok=True)
    parts: list[npt.NDArray[np.float32]] = []
    owners: list[int] = []
    pending: list[str] = []
    pending_owner: list[int] = []
    n_docs = 0

    def flush() -> None:
        if pending:
            parts.append(embed_texts(pending))
            owners.extend(pending_owner)
            pending.clear()
            pending_owner.clear()

    for doc_id, doc in enumerate(islice(iter_documents(root), limit)):
        n_docs += 1
        for piece in chunks(doc.text):
            pending.append(piece)
            pending_owner.append(doc_id)
        if len(pending) >= batch:
            flush()
            log.info("документів: %d, фрагментів: %d", n_docs, len(owners))
    flush()

    matrix = np.concatenate(parts) if parts else np.zeros((0, 384), dtype=np.float32)
    np.save(out / "embeddings.npy", matrix)
    np.save(out / "chunk_doc.npy", np.asarray(owners, dtype=np.int32))
    (out / "meta.json").write_text(
        json.dumps({"model": MODEL_NAME, "documents": n_docs, "chunks": len(owners)}),
        "utf-8",
    )
    return n_docs, len(owners)


@dataclass(eq=False)
class SemanticIndex:
    matrix: npt.NDArray[np.float32]
    chunk_doc: npt.NDArray[np.int32]
    n_docs: int

    @classmethod
    def load(cls, folder: Path) -> "SemanticIndex":
        meta = json.loads((folder / "meta.json").read_text("utf-8"))
        return cls(
            np.load(folder / "embeddings.npy", mmap_mode="r"),
            np.load(folder / "chunk_doc.npy"),
            int(meta["documents"]),
        )

    def doc_scores(self, query: str) -> npt.NDArray[np.float32]:
        """Best-fragment cosine similarity for every document (-1 if none)."""
        q = embed_texts([query])[0]
        sims = self.matrix @ q
        best = np.full(self.n_docs, -1.0, dtype=np.float32)
        # chunk_doc is sorted, so this is "max per group" without a Python loop
        np.maximum.at(best, self.chunk_doc, sims)
        return best

    def search(self, query: str, k: int = 10) -> list[tuple[DocId, float]]:
        best = self.doc_scores(query)
        k = min(k, len(best))
        if k <= 0:
            return []
        picked = np.argpartition(-best, k - 1)[:k]
        order = np.lexsort((picked, -best[picked]))
        return [(int(picked[i]), float(best[picked[i]])) for i in order]


def rrf(
    rankings: Sequence[Sequence[DocId]], k: int = 10, c: int = RRF_K
) -> list[tuple[DocId, float]]:
    """Reciprocal Rank Fusion: score(d) = sum over lists of 1 / (c + rank)."""
    fused: dict[DocId, float] = {}
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking, start=1):
            fused[doc_id] = fused.get(doc_id, 0.0) + 1.0 / (c + rank)
    return sorted(fused.items(), key=lambda item: (-item[1], item[0]))[:k]
