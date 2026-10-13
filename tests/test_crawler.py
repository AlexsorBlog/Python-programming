import asyncio
import json
from pathlib import Path

import httpx
import pytest

from findex.corpus import iter_documents
from findex.crawler import (
    RobotsCache,
    canonicalize,
    crawl,
    extract_links,
    extract_text,
    fetch,
)
from findex.crawler.fetch import backoff_delay

PAGES = {
    "/": "<html><head><title>Головна</title></head><body>"
    "<p>вітаю</p><a href='/a'>a</a><a href='/b/'>b</a>"
    "<a href='/a#fragment'>той самий a</a></body></html>",
    "/a": "<html><head><title>A</title></head><body><p>сторінка a</p>"
    "<a href='https://other.example/x'>чужий хост</a></body></html>",
    "/b": "<html><head><title>B</title></head><body><p>сторінка b</p>"
    "<script>var x = 1;</script><a href='/'>назад</a></body></html>",
}


def site_handler(request: httpx.Request) -> httpx.Response:
    path = request.url.path.rstrip("/") or "/"
    if path == "/robots.txt":
        return httpx.Response(404)
    if path in PAGES:
        return httpx.Response(
            200, html=PAGES[path], headers={"content-type": "text/html"}
        )
    return httpx.Response(404)


def make_client(handler) -> httpx.AsyncClient:
    return httpx.AsyncClient(
        transport=httpx.MockTransport(handler), base_url="https://site.example"
    )


# --- канонізація ----------------------------------------------------------


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("HTTPS://Site.Example/Path", "https://site.example/Path"),
        ("https://site.example/path#section", "https://site.example/path"),
        ("https://site.example/path/", "https://site.example/path"),
        ("https://site.example", "https://site.example/"),
        ("https://site.example/?b=2&a=1", "https://site.example/?a=1&b=2"),
        ("https://site.example:443/x", "https://site.example/x"),
        ("http://site.example:80/x", "http://site.example/x"),
    ],
)
def test_canonicalize(raw: str, expected: str):
    assert canonicalize(raw) == expected


def test_canonicalize_relative():
    assert canonicalize("../b", "https://site.example/a/c") == "https://site.example/b"


# --- розбір html ----------------------------------------------------------


def test_extract_text_drops_scripts():
    title, text = extract_text(PAGES["/b"])
    assert title == "B"
    assert "сторінка b" in text
    assert "var x" not in text


def test_extract_links_absolutizes():
    links = extract_links(PAGES["/"], "https://site.example/")
    assert "https://site.example/a" in links
    assert "https://site.example/b" in links


# --- fetch ----------------------------------------------------------------


@pytest.mark.asyncio
async def test_fetch_ok():
    async with make_client(site_handler) as client:
        result = await fetch(client, "https://site.example/a")
    assert result.ok
    assert result.status == 200
    assert result.attempts == 1
    assert result.bytes > 0


@pytest.mark.asyncio
async def test_fetch_retries_429_and_honours_retry_after():
    calls: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(1)
        if len(calls) == 1:
            return httpx.Response(429, headers={"Retry-After": "0"})
        return httpx.Response(200, html="<html><title>ок</title></html>")

    async with make_client(handler) as client:
        result = await fetch(client, "https://site.example/")
    assert result.ok
    assert result.attempts == 2
    assert len(calls) == 2


@pytest.mark.asyncio
async def test_fetch_does_not_retry_404():
    calls: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(1)
        return httpx.Response(404)

    async with make_client(handler) as client:
        result = await fetch(client, "https://site.example/нема")
    assert not result.ok
    assert len(calls) == 1


@pytest.mark.asyncio
async def test_fetch_gives_up_after_three_attempts():
    calls: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(1)
        return httpx.Response(503)

    async with make_client(handler) as client:
        result = await fetch(client, "https://site.example/", max_attempts=3)
    assert not result.ok
    assert len(calls) == 3
    assert result.status == 503


@pytest.mark.asyncio
async def test_fetch_timeout_is_an_error_not_a_crash():
    async def handler(request: httpx.Request) -> httpx.Response:
        await asyncio.sleep(5)
        return httpx.Response(200)

    async with make_client(handler) as client:
        result = await fetch(
            client, "https://site.example/", timeout=0.05, max_attempts=2
        )
    assert not result.ok
    assert result.error


def test_backoff_grows_and_respects_retry_after():
    assert backoff_delay(0) < backoff_delay(3)
    assert backoff_delay(0, "2") == 2.0
    # an HTTP-date Retry-After is not a number: fall back to the usual backoff
    assert backoff_delay(0, "Wed, 21 Oct 2026 07:28:00 GMT") > 0


# --- robots ---------------------------------------------------------------


@pytest.mark.asyncio
async def test_robots_cache_blocks_and_caches():
    hits: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        hits.append(request.url.path)
        if request.url.path == "/robots.txt":
            return httpx.Response(200, text="User-agent: *\nDisallow: /private\n")
        return httpx.Response(200, html="<html></html>")

    async with make_client(handler) as client:
        robots = RobotsCache(client)
        assert await robots.allowed("https://site.example/public")
        assert not await robots.allowed("https://site.example/private/secret")
        assert await robots.allowed("https://site.example/other")

    # robots.txt is read once per host
    assert hits.count("/robots.txt") == 1


@pytest.mark.asyncio
async def test_missing_robots_means_allowed():
    async with make_client(site_handler) as client:
        robots = RobotsCache(client)
        assert await robots.allowed("https://site.example/a")


# --- crawl ----------------------------------------------------------------


@pytest.mark.asyncio
async def test_crawl_visits_every_page_once():
    async with make_client(site_handler) as client:
        pages = [
            page
            async for page in crawl(
                ["https://site.example/"],
                max_pages=10,
                concurrency=3,
                host_delay=0,
                client=client,
            )
        ]

    urls = [p.url for p in pages]
    assert sorted(urls) == [
        "https://site.example/",
        "https://site.example/a",
        "https://site.example/b",
    ]
    assert len(urls) == len(set(urls))


@pytest.mark.asyncio
async def test_crawl_respects_max_pages():
    async with make_client(site_handler) as client:
        pages = [
            page
            async for page in crawl(
                ["https://site.example/"],
                max_pages=2,
                concurrency=1,
                host_delay=0,
                client=client,
            )
        ]
    assert len(pages) <= 2


@pytest.mark.asyncio
async def test_crawl_stays_inside_allowed_domains():
    async with make_client(site_handler) as client:
        pages = [
            page
            async for page in crawl(
                ["https://site.example/"],
                max_pages=10,
                concurrency=2,
                host_delay=0,
                client=client,
            )
        ]
    assert all("other.example" not in p.url for p in pages)


@pytest.mark.asyncio
async def test_crawl_obeys_robots():
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/robots.txt":
            return httpx.Response(200, text="User-agent: *\nDisallow: /a\n")
        return site_handler(request)

    async with make_client(handler) as client:
        pages = [
            page
            async for page in crawl(
                ["https://site.example/"],
                max_pages=10,
                concurrency=2,
                host_delay=0,
                client=client,
            )
        ]
    assert all(not p.url.endswith("/a") for p in pages)


@pytest.mark.asyncio
async def test_crawl_output_feeds_iter_documents(tmp_path: Path):
    out = tmp_path / "crawl.jsonl"
    async with make_client(site_handler) as client:
        with out.open("w", encoding="utf-8") as f:
            async for page in crawl(
                ["https://site.example/"],
                max_pages=5,
                concurrency=2,
                host_delay=0,
                client=client,
            ):
                f.write(
                    json.dumps(
                        {"id": page.url, "title": page.title, "text": page.text},
                        ensure_ascii=False,
                    )
                    + "\n"
                )

    docs = list(iter_documents(out))
    assert len(docs) == 3
    assert any("вітаю" in d.text for d in docs)
