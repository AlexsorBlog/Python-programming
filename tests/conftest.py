import json
from pathlib import Path

import pytest

from findex.index import Index, build_index
from findex.store import save

DOCS = [
    {"id": 0, "title": "Kyiv", "text": "Kyiv is the capital of Ukraine"},
    {"id": 1, "title": "Lviv", "text": "Lviv is a city in Ukraine the the the"},
    {"id": 2, "title": "Paris", "text": "Paris is a city in France city event loop"},
]


@pytest.fixture
def corpus(tmp_path: Path) -> Path:
    path = tmp_path / "docs.jsonl"
    path.write_text("\n".join(json.dumps(d) for d in DOCS), encoding="utf-8")
    return path


@pytest.fixture
def index(corpus: Path) -> Index:
    return build_index(corpus, positions=True)


@pytest.fixture
def index_file(index: Index, tmp_path: Path) -> Path:
    path = tmp_path / "index.pkl"
    save(index, path)
    return path
