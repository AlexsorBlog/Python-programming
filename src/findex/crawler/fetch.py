import asyncio
import logging
import random
import time
from dataclasses import dataclass, field

import httpx

log = logging.getLogger(__name__)

# HTTP headers must be ASCII, so this one string stays in English
USER_AGENT = (
    "findex-crawler/0.6 (student project; "
    "+https://github.com/AlexsorBlog/findex; sorokopud22@gmail.com)"
)

RETRY_STATUSES = frozenset({429, 500, 502, 503, 504})
MAX_ATTEMPTS = 3
BASE_DELAY = 0.5
DELAY_CAP = 8.0


@dataclass(frozen=True, slots=True)
class FetchResult:
    url: str
    status: int | None
    bytes: int
    elapsed: float
    text: str = ""
    error: str = ""
    attempts: int = 1
    headers: dict[str, str] = field(default_factory=lambda: {})

    @property
    def ok(self) -> bool:
        return self.status is not None and 200 <= self.status < 300


def backoff_delay(attempt: int, retry_after: str | None = None) -> float:
    """Exponential backoff with jitter; Retry-After wins when the server sends it."""
    if retry_after:
        try:
            return min(float(retry_after), DELAY_CAP)
        except ValueError:
            # Retry-After can also be an HTTP date; fall back to the usual backoff
            pass
    return min(DELAY_CAP, BASE_DELAY * 2**attempt) + random.uniform(0, 0.5)


async def fetch(
    client: httpx.AsyncClient,
    url: str,
    *,
    timeout: float = 10.0,
    max_attempts: int = MAX_ATTEMPTS,
) -> FetchResult:
    start = time.perf_counter()
    last_error = ""
    status: int | None = None

    for attempt in range(max_attempts):
        try:
            async with asyncio.timeout(timeout):
                response = await client.get(url, follow_redirects=True)
        except asyncio.CancelledError:
            # CancelledError is a BaseException on purpose: never swallow it
            raise
        except (httpx.HTTPError, TimeoutError) as e:
            last_error = f"{type(e).__name__}: {e}"
            log.warning(
                "спроба %d для %s не вдалась (%s)", attempt + 1, url, last_error
            )
            if attempt + 1 < max_attempts:
                await asyncio.sleep(backoff_delay(attempt))
                continue
            break

        status = response.status_code
        if status in RETRY_STATUSES and attempt + 1 < max_attempts:
            delay = backoff_delay(attempt, response.headers.get("Retry-After"))
            log.warning(
                "%s відповів %d, повтор через %.2f с (спроба %d)",
                url,
                status,
                delay,
                attempt + 1,
            )
            await asyncio.sleep(delay)
            continue

        body = response.text
        return FetchResult(
            url=str(response.url),
            status=status,
            bytes=len(response.content),
            elapsed=time.perf_counter() - start,
            text=body,
            attempts=attempt + 1,
            headers={k.lower(): v for k, v in response.headers.items()},
        )

    return FetchResult(
        url=url,
        status=status,
        bytes=0,
        elapsed=time.perf_counter() - start,
        error=last_error or f"status {status}",
        attempts=max_attempts,
    )
