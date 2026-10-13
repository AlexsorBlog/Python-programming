import asyncio
import contextlib
import logging
import time
from collections.abc import AsyncIterator, Callable, Iterable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from urllib.parse import parse_qsl, urldefrag, urlencode, urljoin, urlsplit, urlunsplit

import httpx
from selectolax.lexbor import LexborHTMLParser

from findex.crawler.fetch import USER_AGENT, FetchResult, fetch
from findex.crawler.robots import RobotsCache

log = logging.getLogger(__name__)
url_log = logging.getLogger("findex.crawl.urls")


@dataclass(frozen=True, slots=True)
class Page:
    url: str
    title: str
    text: str
    fetched_at: str
    status: int
    bytes: int
    elapsed: float


@dataclass
class CrawlStats:
    fetched: int = 0
    errors: int = 0
    in_flight: int = 0
    queued: int = 0
    skipped: int = 0
    started: float = field(default_factory=time.perf_counter)

    @property
    def pages_per_second(self) -> float:
        elapsed = time.perf_counter() - self.started
        return self.fetched / elapsed if elapsed > 0 else 0.0


def canonicalize(url: str, base: str | None = None) -> str:
    """Same page -> same string: lowercase scheme/host, no fragment, sorted query."""
    if base:
        url = urljoin(base, url)
    url, _ = urldefrag(url)
    parts = urlsplit(url)

    scheme = parts.scheme.lower()
    host = parts.hostname or ""
    port = parts.port
    if port and not (
        (scheme == "http" and port == 80) or (scheme == "https" and port == 443)
    ):
        netloc = f"{host}:{port}"
    else:
        netloc = host

    path = parts.path or "/"
    if path != "/" and path.endswith("/"):
        path = path.rstrip("/")
    query = urlencode(sorted(parse_qsl(parts.query, keep_blank_values=True)))
    return urlunsplit((scheme, netloc, path, query, ""))


def host_of(url: str) -> str:
    return urlsplit(url).netloc.lower()


def extract_links(html: str, base_url: str) -> list[str]:
    tree = LexborHTMLParser(html)
    links: list[str] = []
    for node in tree.css("a[href]"):
        href = node.attributes.get("href")
        if not href or href.startswith(("mailto:", "javascript:", "tel:")):
            continue
        try:
            links.append(canonicalize(href, base_url))
        except ValueError:
            continue
    return links


def extract_text(html: str) -> tuple[str, str]:
    tree = LexborHTMLParser(html)
    for tag in ("script", "style", "noscript"):
        for node in tree.css(tag):
            node.decompose()
    title_node = tree.css_first("title")
    title = title_node.text(strip=True) if title_node else ""
    body = tree.body
    text = body.text(separator=" ", strip=True) if body else ""
    return title, " ".join(text.split())


class _HostLimiter:
    """Per-host concurrency and a minimum gap between requests to that host."""

    def __init__(self, per_host: int, delay: float) -> None:
        self._per_host = per_host
        self._delay = delay
        self._sems: dict[str, asyncio.Semaphore] = {}
        self._last: dict[str, float] = {}

    def semaphore(self, host: str) -> asyncio.Semaphore:
        sem = self._sems.get(host)
        if sem is None:
            sem = asyncio.Semaphore(self._per_host)
            self._sems[host] = sem
        return sem

    async def wait_turn(self, host: str) -> None:
        if self._delay <= 0:
            return
        now = time.monotonic()
        last = self._last.get(host)
        if last is not None:
            gap = self._delay - (now - last)
            if gap > 0:
                await asyncio.sleep(gap)
        self._last[host] = time.monotonic()


async def crawl(
    seeds: Sequence[str],
    *,
    max_pages: int = 200,
    concurrency: int = 10,
    per_host: int = 2,
    host_delay: float = 0.2,
    allowed_domains: Iterable[str] | None = None,
    timeout: float = 10.0,
    client: httpx.AsyncClient | None = None,
    stats: CrawlStats | None = None,
    on_result: Callable[[FetchResult], None] | None = None,
) -> AsyncIterator[Page]:
    """Stream pages as they arrive. Workers live inside a TaskGroup."""
    stats = stats if stats is not None else CrawlStats()
    seeds = [canonicalize(url) for url in seeds]
    domains = (
        {d.lower() for d in allowed_domains}
        if allowed_domains
        else {host_of(url) for url in seeds}
    )

    owns_client = client is None
    client = client or httpx.AsyncClient(
        headers={"User-Agent": USER_AGENT},
        timeout=timeout,
        follow_redirects=True,
    )
    robots = RobotsCache(client)

    frontier: asyncio.Queue[str] = asyncio.Queue()
    pages: asyncio.Queue[Page | None] = asyncio.Queue(maxsize=concurrency * 2)
    seen: set[str] = set()
    limiter = _HostLimiter(per_host, host_delay)
    gate = asyncio.Semaphore(concurrency)
    done = asyncio.Event()

    for url in seeds:
        if url not in seen:
            seen.add(url)
            frontier.put_nowait(url)
    stats.queued = frontier.qsize()

    async def handle(url: str) -> None:
        if stats.fetched >= max_pages:
            return
        if not await robots.allowed(url):
            stats.skipped += 1
            url_log.info("SKIP robots %s", url)
            return

        host = host_of(url)
        async with gate, limiter.semaphore(host):
            await limiter.wait_turn(host)
            stats.in_flight += 1
            try:
                result = await fetch(client, url, timeout=timeout)
            finally:
                stats.in_flight -= 1

        if on_result is not None:
            on_result(result)
        url_log.info(
            "%s %s %d байт %.3f с спроб=%d %s",
            result.status if result.status is not None else "ERR",
            result.url,
            result.bytes,
            result.elapsed,
            result.attempts,
            result.error,
        )

        if not result.ok:
            stats.errors += 1
            return

        content_type = result.headers.get("content-type", "")
        if "html" not in content_type and content_type:
            stats.skipped += 1
            return

        title, text = extract_text(result.text)
        page = Page(
            url=result.url,
            title=title,
            text=text,
            fetched_at=datetime.now(UTC).isoformat(timespec="seconds"),
            status=result.status or 0,
            bytes=result.bytes,
            elapsed=result.elapsed,
        )
        stats.fetched += 1
        await pages.put(page)

        if stats.fetched >= max_pages:
            done.set()
            return

        for link in extract_links(result.text, result.url):
            if link in seen or host_of(link) not in domains:
                continue
            seen.add(link)
            frontier.put_nowait(link)
        stats.queued = frontier.qsize()

    async def worker(name: int) -> None:
        while not done.is_set():
            try:
                url = await asyncio.wait_for(frontier.get(), timeout=0.5)
            except TimeoutError:
                if frontier.empty() and stats.in_flight == 0:
                    done.set()
                continue
            try:
                await handle(url)
            except asyncio.CancelledError:
                raise
            except Exception:
                stats.errors += 1
                log.exception("воркер %d впав на %s", name, url)
            finally:
                frontier.task_done()
                stats.queued = frontier.qsize()

    async def supervisor() -> None:
        try:
            async with asyncio.TaskGroup() as group:
                workers = [group.create_task(worker(i)) for i in range(concurrency)]
                await done.wait()
                for task in workers:
                    task.cancel()
        except* asyncio.CancelledError:
            pass
        finally:
            await pages.put(None)

    runner = asyncio.create_task(supervisor())
    try:
        while True:
            page = await pages.get()
            if page is None:
                break
            yield page
    finally:
        done.set()
        runner.cancel()
        with contextlib.suppress(asyncio.CancelledError, ExceptionGroup):
            await runner
        if owns_client:
            await client.aclose()
