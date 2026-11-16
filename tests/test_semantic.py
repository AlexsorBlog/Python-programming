"""Semantic search without the real model: a fake embedder keeps it offline."""

import hashlib
from pathlib import Path

import numpy as np
import pytest

from findex import semantic
from findex.hybrid import semantic_results
from findex.index import Index, build_index
from findex.semantic import SemanticIndex, build_embeddings, chunks, rrf

# meaning groups: words in one group get the same direction in vector space,
# so "capital" lands near "kyiv" even though the strings share nothing
GROUPS = {
    "ukraine": ["kyiv", "lviv", "ukraine", "capital", "ukrainian"],
    "france": ["paris", "france", "french"],
    "water": ["river", "sea", "lake", "water"],
}


def fake_embed(texts, batch_size: int = 64):
    out = np.zeros((len(texts), 8), dtype=np.float32)
    for row, text in enumerate(texts):
        for word in text.casefold().split():
            word = word.strip(".,")
            for g, (_, members) in enumerate(GROUPS.items()):
                if word in members:
                    out[row, g] += 1.0
            # a little noise so unrelated words are not all zero
            h = int(hashlib.md5(word.encode()).hexdigest(), 16)
            out[row, 3 + h % 5] += 0.05
    return semantic.normalize_rows(out)


@pytest.fixture(autouse=True)
def offline(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(semantic, "embed_texts", fake_embed)


@pytest.fixture
def emb_dir(corpus: Path, tmp_path: Path) -> Path:
    out = tmp_path / "emb"
    build_embeddings(corpus, out)
    return out


def test_chunks_split_and_cap():
    text = " ".join(f"w{i}" for i in range(1000))
    pieces = chunks(text, words=100, limit=4)
    assert len(pieces) == 4
    assert pieces[0].split()[0] == "w0"
    assert pieces[1].split()[0] == "w100"
    assert chunks("") == []


def test_rows_are_unit_length():
    m = semantic.normalize_rows(np.array([[3.0, 4.0], [0.0, 0.0]], dtype=np.float32))
    assert np.allclose(np.linalg.norm(m[0]), 1.0)
    assert not m[1].any()


def test_embeddings_cover_every_document(emb_dir: Path, index: Index):
    sem = SemanticIndex.load(emb_dir)
    assert sem.n_docs == index.num_docs
    assert set(sem.chunk_doc.tolist()) == set(range(index.num_docs))
    assert np.allclose(np.linalg.norm(sem.matrix, axis=1), 1.0)


def test_semantic_finds_meaning_not_words(emb_dir: Path, index: Index):
    # "capital" appears in doc 0 (Kyiv) but the point is the group:
    # a query with no word from the Lviv doc still ranks Ukrainian docs first
    sem = SemanticIndex.load(emb_dir)
    top = [doc for doc, _ in sem.search("ukrainian capital", k=2)]
    assert set(top) == {0, 1}


def test_rrf_rewards_agreement():
    fused = rrf([[1, 2, 3], [3, 1, 4]], k=4)
    order = [doc for doc, _ in fused]
    assert order[0] == 1  # ranks 1 and 2
    assert order[1] == 3  # ranks 3 and 1
    assert set(order) == {1, 2, 3, 4}


def test_rrf_ties_are_stable():
    assert rrf([[5], [7]], k=2) == rrf([[7], [5]], k=2)


@pytest.mark.parametrize("mode", ["semantic", "hybrid"])
def test_semantic_results_through_index(emb_dir: Path, index: Index, mode: str):
    results = semantic_results(index, emb_dir, "french city", mode, k=3)  # type: ignore[arg-type]
    assert results
    assert results[0].title == "Paris"


def test_hybrid_survives_a_query_the_parser_rejects(emb_dir: Path, index: Index):
    results = semantic_results(index, emb_dir, "paris )", "hybrid", k=3)
    assert results[0].title == "Paris"


def test_embeddings_from_another_corpus_are_refused(emb_dir: Path, tmp_path: Path):
    other = tmp_path / "other.jsonl"
    other.write_text('{"id": 1, "text": "one doc"}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="іншого корпусу"):
        semantic_results(build_index(other), emb_dir, "x", "semantic")
