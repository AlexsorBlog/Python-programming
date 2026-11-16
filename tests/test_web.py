from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from findex.index import Index
from findex.web.app import create_app, get_index_optional, get_settings
from findex.web.settings import Settings


@pytest.fixture
def app(index: Index, tmp_path: Path) -> Iterator[FastAPI]:
    """The tiny fixture index from conftest, no index file on disk at all."""
    application = create_app()
    application.dependency_overrides[get_settings] = lambda: Settings(
        index_path=tmp_path / "немає.pkl"
    )
    application.dependency_overrides[get_index_optional] = lambda: index
    yield application
    application.dependency_overrides.clear()


@pytest.fixture
def client(app: FastAPI) -> Iterator[TestClient]:
    with TestClient(app) as c:
        yield c


def test_search_ok(client: TestClient):
    r = client.get("/search", params={"q": "city ukraine"})
    assert r.status_code == 200
    body = r.json()
    assert body["total"] == 1
    assert body["hits"][0]["title"] == "Lviv"
    assert body["hits"][0]["snippet"]
    assert "x-request-id" in r.headers


def test_search_pagination(client: TestClient):
    first = client.get("/search", params={"q": "city", "k": 1, "page": 1}).json()
    second = client.get("/search", params={"q": "city", "k": 1, "page": 2}).json()
    assert first["total"] == 2
    assert first["pages"] == 2
    assert first["hits"][0]["doc_id"] != second["hits"][0]["doc_id"]


def test_search_scorer_tfidf(client: TestClient):
    r = client.get("/search", params={"q": "city", "scorer": "tfidf"})
    assert r.status_code == 200
    assert r.json()["scorer"] == "tfidf"


@pytest.mark.parametrize(
    "params",
    [
        {},
        {"q": ""},
        {"q": "city", "k": 0},
        {"q": "city", "k": 999},
        {"q": "city", "page": 0},
        {"q": "city", "scorer": "pagerank"},
    ],
)
def test_search_validation_is_422(client: TestClient, params: dict[str, object]):
    assert client.get("/search", params=params).status_code == 422


def test_broken_query_is_422_not_500(client: TestClient):
    r = client.get("/search", params={"q": "kyiv )"})
    assert r.status_code == 422


def test_doc_found_and_404(client: TestClient):
    ok = client.get("/docs/0")
    assert ok.status_code == 200
    assert ok.json()["title"] == "Kyiv"
    missing = client.get("/docs/999")
    assert missing.status_code == 404


def test_stats(client: TestClient):
    body = client.get("/stats").json()
    assert body["documents"] == 3
    assert body["positions"] is True


def test_health_ok(client: TestClient):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_health_503_when_index_missing(tmp_path: Path):
    application = create_app()
    application.dependency_overrides[get_settings] = lambda: Settings(
        index_path=tmp_path / "немає.pkl"
    )
    application.dependency_overrides[get_index_optional] = lambda: None
    with TestClient(application) as c:
        assert c.get("/health").status_code == 503
        assert c.get("/search", params={"q": "kyiv"}).status_code == 503


def test_unexpected_error_is_500_without_stack(app: FastAPI):
    def broken() -> Index:
        raise RuntimeError("секретна деталь реалізації")

    app.dependency_overrides[get_index_optional] = broken
    with TestClient(app, raise_server_exceptions=False) as c:
        r = c.get("/stats")
    assert r.status_code == 500
    assert "request_id" in r.json()
    assert "секретна" not in r.text
    assert "Traceback" not in r.text


def test_html_page_renders_results(client: TestClient):
    r = client.get("/", params={"q": "city ukraine"})
    assert r.status_code == 200
    assert "<mark>" in r.text
    assert "Lviv" in r.text


def test_html_home_without_query(client: TestClient):
    r = client.get("/")
    assert r.status_code == 200
    assert "Пошук" in r.text


def test_html_escapes_user_input(client: TestClient):
    r = client.get("/", params={"q": "<script>alert(1)</script>"})
    assert "<script>alert(1)</script>" not in r.text


def test_html_doc_page(client: TestClient):
    r = client.get("/doc/1", params={"q": "lviv"})
    assert r.status_code == 200
    assert "Lviv" in r.text
