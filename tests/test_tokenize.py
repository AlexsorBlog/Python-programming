import inspect
from pathlib import Path

from findex.corpus import iter_documents
from findex.tokenize import tokenize


def toks(text):
    return list(tokenize(text))


def test_is_generator():
    assert inspect.isgeneratorfunction(tokenize)
    assert inspect.isgeneratorfunction(iter_documents)


def test_mixed_case():
    assert toks("Hello WORLD hElLo") == ["hello", "world", "hello"]
    assert toks("Straße") == toks("STRASSE") == ["strasse"]


def test_cyrillic():
    assert toks("Київ — столиця України") == ["київ", "столиця", "україни"]


def test_combining_accent():
    assert toks("café") == toks("café") == ["café"]


def test_punctuation():
    assert toks("Hi, there! How... are_you?") == ["hi", "there", "how", "are", "you"]


def test_apostrophes():
    assert toks("don't 'quoted' п’ять пʼять") == ["don't", "quoted", "п'ять", "п'ять"]


def test_hyphens():
    assert toks("e-mail state-of-the-art - trailing-") == [
        "e-mail",
        "state-of-the-art",
        "trailing",
    ]


def test_digits():
    assert toks("In 1999 pi was 3.14") == ["in", "1999", "pi", "was", "3", "14"]


def test_empty():
    assert toks("") == []
    assert toks("  ...  ") == []


def test_bad_lines_skipped(tmp_path: Path):
    f = tmp_path / "docs.jsonl"
    f.write_bytes(
        b'{"id": 1, "text": "a"}\nnot json\n\xff\xfe\n{"id": 2, "text": "b"}\n'
    )
    assert [d.doc_id for d in iter_documents(f)] == ["1", "2"]
