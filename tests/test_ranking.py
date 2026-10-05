import json
from pathlib import Path

import pytest

from findex.index import build_index
from findex.query import And, Not, Or, Phrase, Term, parse
from findex.scoring import BM25, SearchResult, TfIdf
from findex.search import ranked_search
from findex.snippet import make_snippet
from findex.store import open_index, save

DOCS = [
    {"id": 0, "title": "Kyiv", "text": "Kyiv is the capital of Ukraine"},
    {"id": 1, "title": "Lviv", "text": "Lviv is a city in Ukraine the the the"},
    {"id": 2, "title": "Paris", "text": "Paris is a city in France city event loop"},
]


@pytest.fixture
def index(tmp_path: Path):
    f = tmp_path / "docs.jsonl"
    f.write_text("\n".join(json.dumps(d) for d in DOCS), encoding="utf-8")
    return build_index(f, positions=True)


def test_index_is_a_mapping(index):
    assert len(index) == len(index.postings)
    assert "kyiv" in index
    assert "nothinghere" not in index
    assert index["kyiv"][0].doc_id == 0
    assert sorted(index)[:2] == sorted(index.postings)[:2]
    assert "Index(terms=" in repr(index)
    with pytest.raises(KeyError):
        index["nothinghere"]


def test_index_properties(index):
    assert index.num_docs == 3
    assert index.df("ukraine") == 2
    assert index.doc_length(0) == 7
    assert index.avg_doc_length == sum(index.doc_lengths.values()) / 3
    # cached_property keeps the value on the instance
    assert "avg_doc_length" in index.__dict__


def test_open_index_releases_on_error(tmp_path: Path, index):
    path = tmp_path / "index.pkl"
    save(index, path)
    with pytest.raises(RuntimeError), open_index(path) as opened:
        assert len(opened) > 0
        kept = opened
        raise RuntimeError("boom")
    assert len(kept) == 0


def test_parse_builds_tree():
    assert parse("a OR b c") == Or(Term("a"), And(Term("b"), Term("c")))
    assert parse("a AND b") == And(Term("a"), Term("b"))
    assert parse("NOT java") == Not(Term("java"))
    assert parse('python AND (async OR await) NOT java "event loop"') == And(
        And(
            And(Term("python"), Or(Term("async"), Term("await"))),
            Not(Term("java")),
        ),
        Phrase(("event", "loop")),
    )


def test_operators_build_same_tree():
    assert Term("a") & Term("b") == parse("a b")
    assert Term("a") | Term("b") == parse("a OR b")
    assert ~Term("a") == parse("NOT a")


def test_phrase_needs_order(index):
    assert Phrase(("event", "loop")).evaluate(index) == [2]
    assert Phrase(("loop", "event")).evaluate(index) == []


def test_boolean_tree_evaluation(index):
    assert parse("city ukraine").evaluate(index) == [1]
    assert parse("kyiv OR paris").evaluate(index) == [0, 2]
    assert parse("city NOT france").evaluate(index) == [1]


def test_rare_term_beats_common(index):
    results = ranked_search(index, "kyiv OR the", BM25(), k=5)
    assert results[0].doc_id == 0


def test_saturation_and_length(index):
    bm25 = BM25()
    short = bm25.score("city", next(index.iter_postings("city")), index)
    assert short > 0
    tenfold = type(next(index.iter_postings("city")))(1, 20)
    once = type(next(index.iter_postings("city")))(1, 1)
    assert bm25.score("city", tenfold, index) < 4 * bm25.score("city", once, index)


def test_scorers_are_interchangeable(index):
    for scorer in (BM25(), TfIdf()):
        results = ranked_search(index, "city", scorer, k=5)
        assert [r.doc_id for r in results] == [2, 1]


def test_results_sortable_and_printable():
    a = SearchResult(1.5, 1, "A")
    b = SearchResult(2.5, 2, "B")
    assert sorted([b, a]) == [a, b]
    assert "B" in str(b)


def test_snippet_highlights():
    text = "Kyiv is the capital of Ukraine and a very old city on the Dnipro river"
    snippet = make_snippet(text, ["kyiv", "dnipro"], width=30)
    assert "[kyiv]" in snippet.casefold()
    assert len(snippet) < len(text) + 20


def test_snippets_in_results(index):
    results = ranked_search(index, "kyiv", k=1, snippets=True)
    assert "[" in results[0].snippet
