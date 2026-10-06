import json
from pathlib import Path

import pytest

from findex.index import Index, Posting, build_index
from findex.query import And, Not, Or, Phrase, Term, parse
from findex.scoring import BM25, SearchResult, TfIdf
from findex.search import ranked_search
from findex.snippet import make_snippet
from findex.store import open_index


def test_index_is_a_mapping(index: Index):
    assert len(index) == len(index.postings)
    assert "kyiv" in index
    assert "nothinghere" not in index
    assert index.doc_ids("kyiv") == [0]
    assert sorted(index)[:2] == sorted(index.postings)[:2]
    assert "Index(terms=" in repr(index)
    with pytest.raises(KeyError):
        index["nothinghere"]


def test_index_properties(index: Index):
    assert index.num_docs == 3
    assert index.df("ukraine") == 2
    assert index.doc_length(0) == 7
    assert index.avg_doc_length == sum(index.doc_lengths.values()) / 3
    # cached_property keeps the value on the instance
    assert "avg_doc_length" in index.__dict__


def test_open_index_releases_on_error(index_file: Path):
    with pytest.raises(RuntimeError), open_index(index_file) as opened:
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


def test_broken_query_raises():
    with pytest.raises(ValueError, match="розібрати"):
        parse("kyiv )")


def test_phrase_needs_order(index: Index):
    assert Phrase(("event", "loop")).evaluate(index) == [2]
    assert Phrase(("loop", "event")).evaluate(index) == []


def test_phrase_needs_positions(corpus: Path):
    without = build_index(corpus)
    with pytest.raises(ValueError, match="positions"):
        Phrase(("event", "loop")).evaluate(without)


def test_boolean_tree_evaluation(index: Index):
    node = parse("city ukraine")
    assert node is not None
    assert node.evaluate(index) == [1]


def test_rare_term_beats_common(index: Index):
    results = ranked_search(index, "kyiv OR the", BM25(), k=5)
    assert results[0].doc_id == 0


def test_bm25_saturation(index: Index):
    bm25 = BM25()
    once = bm25.score("city", Posting(1, 1), index)
    twenty = bm25.score("city", Posting(1, 20), index)
    assert once < twenty < 4 * once


def test_bm25_prefers_short_document(index: Index):
    bm25 = BM25()
    short = bm25.score("city", Posting(1, 1), index)
    long_doc_id = max(index.doc_lengths, key=lambda d: index.doc_length(d))
    long = bm25.score("city", Posting(long_doc_id, 1), index)
    assert index.doc_length(long_doc_id) >= index.doc_length(1)
    assert short >= long


def test_scorers_are_interchangeable(index: Index):
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


def test_snippets_in_results(index: Index):
    results = ranked_search(index, "kyiv", k=1, snippets=True)
    assert "[" in results[0].snippet


@pytest.mark.slow
def test_build_on_a_bigger_corpus(tmp_path: Path):
    path = tmp_path / "big.jsonl"
    with path.open("w", encoding="utf-8") as f:
        for i in range(4000):
            words = " ".join(f"word{i % 97} term{i % 31} filler{j}" for j in range(12))
            f.write(json.dumps({"id": i, "title": f"doc {i}", "text": words}) + "\n")

    built = build_index(path)
    assert built.num_docs == 4000
    assert built.doc_ids("word5") == sorted(built.doc_ids("word5"))
    results = ranked_search(built, "word5 term10", BM25(), k=5)
    assert results
