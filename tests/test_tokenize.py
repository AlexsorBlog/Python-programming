import inspect
from pathlib import Path

import pytest

from findex.corpus import iter_documents
from findex.tokenize import tokenize


def toks(text: str) -> list[str]:
    return list(tokenize(text))


def test_is_generator():
    assert inspect.isgeneratorfunction(tokenize)
    assert inspect.isgeneratorfunction(iter_documents)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Hello WORLD hElLo", ["hello", "world", "hello"]),
        ("Straße", ["strasse"]),
        ("STRASSE", ["strasse"]),
        ("Київ — столиця України", ["київ", "столиця", "україни"]),
        ("café", ["café"]),
        ("café", ["café"]),
        ("Hi, there! How... are_you?", ["hi", "there", "how", "are", "you"]),
        ("don't 'quoted'", ["don't", "quoted"]),
        ("п’ять пʼять", ["п'ять", "п'ять"]),
        (
            "e-mail state-of-the-art - trailing-",
            ["e-mail", "state-of-the-art", "trailing"],
        ),
        ("In 1999 pi was 3.14", ["in", "1999", "pi", "was", "3", "14"]),
        ("", []),
        ("  ...  ", []),
        # NFC (not NFKC) keeps fullwidth letters as they are, casefold only lowercases
        ("ＦＵＬＬ", ["ｆｕｌｌ"]),
    ],
)
def test_tokenize_rules(text: str, expected: list[str]):
    assert toks(text) == expected


def test_bad_lines_skipped(tmp_path: Path):
    f = tmp_path / "docs.jsonl"
    f.write_bytes(
        b'{"id": 1, "text": "a"}\nnot json\n\xff\xfe\n{"id": 2, "text": "b"}\n'
    )
    assert [d.doc_id for d in iter_documents(f)] == ["1", "2"]


def test_offsets_point_at_the_right_line(tmp_path: Path):
    f = tmp_path / "docs.jsonl"
    f.write_text(
        '{"id": 1, "text": "first"}\n{"id": 2, "text": "second"}\n', encoding="utf-8"
    )
    docs = list(iter_documents(f))
    with f.open("rb") as handle:
        handle.seek(docs[1].offset)
        assert b"second" in handle.readline()
