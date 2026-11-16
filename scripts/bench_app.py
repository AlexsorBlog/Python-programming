"""The app under test for bench_def_vs_async.py, run by uvicorn in its own process.

The real app only keeps the def route. Here an async twin is bolted on that
does exactly the same work, so the two can be compared.
"""

from typing import Annotated

from fastapi import Depends

from findex.index import Index
from findex.web.app import create_app, get_index, search_response
from findex.web.models import ScorerName, SearchResponse

app = create_app()


@app.get("/search_async")
async def search_async(
    index: Annotated[Index, Depends(get_index)], q: str, k: int = 10
) -> SearchResponse:
    # same work as /search, but inside the event loop
    return search_response(index, q, k, 1, ScorerName.bm25)
