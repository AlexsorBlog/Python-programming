import asyncio
import logging
from urllib.parse import urlsplit, urlunsplit
from urllib.robotparser import RobotFileParser

import httpx

from findex.crawler.fetch import USER_AGENT

log = logging.getLogger(__name__)


class RobotsCache:
    """One robots.txt per host, fetched once and remembered."""

    def __init__(self, client: httpx.AsyncClient, agent: str = USER_AGENT) -> None:
        self._client = client
        self._agent = agent
        self._parsers: dict[str, RobotFileParser | None] = {}
        self._locks: dict[str, asyncio.Lock] = {}

    def _lock_for(self, host_key: str) -> asyncio.Lock:
        lock = self._locks.get(host_key)
        if lock is None:
            lock = asyncio.Lock()
            self._locks[host_key] = lock
        return lock

    async def _parser_for(self, url: str) -> RobotFileParser | None:
        parts = urlsplit(url)
        host_key = f"{parts.scheme}://{parts.netloc}"

        # one lock per host, so twenty workers don't fetch the same robots.txt
        async with self._lock_for(host_key):
            if host_key in self._parsers:
                return self._parsers[host_key]

            robots_url = urlunsplit((parts.scheme, parts.netloc, "/robots.txt", "", ""))
            parser: RobotFileParser | None = None
            try:
                response = await self._client.get(robots_url, timeout=10.0)
                if response.status_code < 400:
                    parser = RobotFileParser()
                    parser.parse(response.text.splitlines())
                    log.info("robots.txt прочитано для %s", host_key)
                else:
                    # no robots.txt means nothing is forbidden
                    log.info(
                        "robots.txt для %s дав %d, вважаю що все дозволено",
                        host_key,
                        response.status_code,
                    )
            except httpx.HTTPError as e:
                log.warning("не вдалось прочитати robots.txt для %s (%s)", host_key, e)

            self._parsers[host_key] = parser
            return parser

    async def allowed(self, url: str) -> bool:
        parser = await self._parser_for(url)
        if parser is None:
            return True
        return parser.can_fetch(self._agent, url)

    async def crawl_delay(self, url: str) -> float | None:
        parser = await self._parser_for(url)
        if parser is None:
            return None
        value = parser.crawl_delay(self._agent)
        return float(value) if value is not None else None
