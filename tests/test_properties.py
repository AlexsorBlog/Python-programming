import json
from pathlib import Path

from hypothesis import given, settings
from hypothesis import strategies as st

from findex.index import build_index
from findex.merge import merge_and, merge_not, merge_or
from findex.store import load, save
from findex.tokenize import tokenize

sorted_ids = st.lists(st.integers(0, 50), unique=True).map(sorted)
texts = st.text(max_size=120)


@given(a=sorted_ids, b=sorted_ids)
def test_merge_matches_sets(a: list[int], b: list[int]):
    assert merge_and(a, b) == sorted(set(a) & set(b))
    assert merge_or(a, b) == sorted(set(a) | set(b))
    assert merge_not(a, b) == sorted(set(a) - set(b))


@given(docs=st.lists(texts, min_size=1, max_size=8))
@settings(max_examples=30, deadline=None)
def test_postings_always_sorted(docs: list[str], tmp_path_factory):
    path = Path(tmp_path_factory.mktemp("corpus")) / "docs.jsonl"
    path.write_text(
        "\n".join(json.dumps({"id": i, "text": t}) for i, t in enumerate(docs)),
        encoding="utf-8",
    )
    index = build_index(path)
    for term in index:
        ids = index.doc_ids(term)
        assert ids == sorted(ids)
        assert len(ids) == len(set(ids))


@given(docs=st.lists(texts, min_size=1, max_size=6))
@settings(max_examples=20, deadline=None)
def test_save_load_round_trip(docs: list[str], tmp_path_factory):
    folder = Path(tmp_path_factory.mktemp("corpus"))
    path = folder / "docs.jsonl"
    path.write_text(
        "\n".join(json.dumps({"id": i, "text": t}) for i, t in enumerate(docs)),
        encoding="utf-8",
    )
    index = build_index(path)
    for name in ("index.pkl", "index.json"):
        target = folder / name
        save(index, target)
        back = load(target)
        assert back.postings == index.postings
        assert back.doc_lengths == index.doc_lengths
        assert back.doc_meta == index.doc_meta


@given(text=texts)
def test_tokenize_is_idempotent(text: str):
    once = list(tokenize(text))
    assert list(tokenize(" ".join(once))) == once
    assert all(tok for tok in once)
