"""pytest-benchmark: run in CI so a slowdown shows up in the very next run.

uv run pytest tests/test_benchmarks.py --benchmark-only
"""

import json
from pathlib import Path

import pytest

from findex.index import Index, build_index
from findex.numpy_index import NumpyIndex, from_index
from findex.scoring import BM25
from findex.search import ranked_search, ranked_search_numpy

WORDS = [
    "city",
    "river",
    "war",
    "kyiv",
    "lviv",
    "sea",
    "the",
    "a",
    "of",
    "music",
    "rock",
    "band",
    "planet",
    "star",
]


@pytest.fixture(scope="module")
def corpus(tmp_path_factory: pytest.TempPathFactory) -> Path:
    path = tmp_path_factory.mktemp("bench") / "docs.jsonl"
    with path.open("w", encoding="utf-8") as f:
        for i in range(2000):
            words = [WORDS[(i * 7 + j * 3) % len(WORDS)] for j in range(60)]
            f.write(json.dumps({"id": i, "title": f"doc {i}", "text": " ".join(words)}))
            f.write("\n")
    return path


@pytest.fixture(scope="module")
def index(corpus: Path) -> Index:
    return build_index(corpus)


@pytest.fixture(scope="module")
def nidx(index: Index) -> NumpyIndex:
    return from_index(index)


def test_bench_build_index(benchmark, corpus: Path):
    built = benchmark.pedantic(build_index, args=(corpus,), rounds=3, iterations=1)
    assert built.num_docs == 2000


def test_bench_search_lab3(benchmark, index: Index):
    results = benchmark(ranked_search, index, "city OR river OR the", BM25(), 10)
    assert len(results) == 10


def test_bench_search_numpy(benchmark, nidx: NumpyIndex):
    results = benchmark(ranked_search_numpy, nidx, "city OR river OR the", "bm25", 10)
    assert len(results) == 10
