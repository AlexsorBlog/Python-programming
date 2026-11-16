import asyncio
import logging
import math
import time
import uuid
from collections.abc import AsyncGenerator, Awaitable, Callable
from contextlib import asynccontextmanager
from functools import lru_cache
from pathlib import Path
from typing import Annotated

from fastapi import Depends, FastAPI, HTTPException, Query, Request, Response
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from markupsafe import Markup, escape

from findex.index import Index
from findex.scoring import BM25, Scorer, SearchResult, TfIdf
from findex.search import matching_docs, ranked_search
from findex.snippet import document_text, make_snippet
from findex.store import load
from findex.web.models import (
    DocResponse,
    HealthResponse,
    Hit,
    ScorerName,
    SearchResponse,
    StatsResponse,
)
from findex.web.settings import Settings

log = logging.getLogger("findex.web")
HERE = Path(__file__).parent
templates = Jinja2Templates(directory=HERE / "templates")

# private markers: the snippet gets them, then they become <mark> after escaping
_OPEN, _CLOSE = "⟦", "⟧"


@lru_cache
def get_settings() -> Settings:
    return Settings()


def get_index_optional(request: Request) -> Index | None:
    """The one place routes get the index from; tests override this."""
    index: Index | None = getattr(request.app.state, "index", None)
    return index


def get_index(
    index: Annotated[Index | None, Depends(get_index_optional)],
) -> Index:
    if index is None:
        raise HTTPException(status_code=503, detail="індекс ще не завантажено")
    return index


def _scorer(name: ScorerName) -> Scorer:
    return BM25() if name is ScorerName.bm25 else TfIdf()


def run_search(
    index: Index, q: str, k: int, page: int, scorer: ScorerName
) -> tuple[list[SearchResult], int, list[str]]:
    """One page of results plus the total number of matches."""
    doc_ids, terms = matching_docs(index, q)
    total = len(doc_ids)
    if not total:
        return [], 0, []
    ranked = ranked_search(index, q, _scorer(scorer), k=k * page)
    return ranked[(page - 1) * k : page * k], total, list(terms)


def snippet_for(index: Index, doc_id: int, terms: list[str]) -> str:
    text = document_text(index.doc_meta[doc_id])
    return make_snippet(text, terms, marker=(_OPEN, _CLOSE))


def highlight(snippet: str) -> Markup:
    """Escape the article text first, then turn my markers into <mark>."""
    safe = str(escape(snippet))
    return Markup(safe.replace(_OPEN, "<mark>").replace(_CLOSE, "</mark>"))


def search_response(
    index: Index, q: str, k: int, page: int, scorer: ScorerName
) -> SearchResponse:
    """The whole /search job, shared by the route and the def/async benchmark."""
    start = time.perf_counter()
    try:
        results, total, terms = run_search(index, q, k, page, scorer)
    except ValueError as e:
        # a broken query or a phrase without positions is the user's input
        raise HTTPException(status_code=422, detail=str(e)) from None
    hits = [
        Hit(
            doc_id=r.doc_id,
            score=round(r.score, 4),
            title=r.title,
            snippet=snippet_for(index, r.doc_id, terms)
            .replace(_OPEN, "")
            .replace(_CLOSE, ""),
        )
        for r in results
    ]
    return SearchResponse(
        query=q,
        scorer=scorer,
        page=page,
        k=k,
        total=total,
        pages=max(1, math.ceil(total / k)),
        hits=hits,
        took_ms=round((time.perf_counter() - start) * 1000, 2),
    )


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None]:
    # lifespan is not a route, so Depends doesn't reach it: honour the
    # override by hand, otherwise tests would quietly load the real index
    settings = app.dependency_overrides.get(get_settings, get_settings)()
    app.state.index = None
    started = time.perf_counter()
    try:
        # loading is plain CPU + disk work, so it goes to a thread
        app.state.index = await asyncio.to_thread(load, settings.index_path)
        log.info(
            "індекс %s завантажено за %.2f с: %r",
            settings.index_path,
            time.perf_counter() - started,
            app.state.index,
        )
    except FileNotFoundError:
        log.error("немає файлу індексу %s, /health віддає 503", settings.index_path)
    yield
    if app.state.index is not None:
        app.state.index.close()
        app.state.index = None


def create_app() -> FastAPI:
    app = FastAPI(
        title="findex",
        version="0.7.0",
        lifespan=lifespan,
        docs_url="/api/docs",
        redoc_url=None,
    )
    app.mount("/static", StaticFiles(directory=HERE / "static"), name="static")

    @app.middleware("http")
    async def request_id(
        request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        rid = request.headers.get("x-request-id") or uuid.uuid4().hex[:12]
        request.state.request_id = rid
        start = time.perf_counter()
        try:
            response = await call_next(request)
        except Exception:
            # the log keeps the stack, the client gets only the id
            log.exception(
                "запит %s %s впав (id=%s)", request.method, request.url.path, rid
            )
            response = JSONResponse(
                {"detail": "внутрішня помилка сервера", "request_id": rid},
                status_code=500,
            )
        response.headers["x-request-id"] = rid
        log.info(
            "%s %s -> %d за %.1f мс (id=%s)",
            request.method,
            request.url.path,
            response.status_code,
            (time.perf_counter() - start) * 1000,
            rid,
        )
        return response

    # async def on purpose: no CPU work here, so it answers straight from the
    # event loop and doesn't queue behind searches in the thread pool
    @app.get("/health", response_model=HealthResponse)
    async def health(
        index: Annotated[Index | None, Depends(get_index_optional)],
    ) -> JSONResponse:
        if index is None:
            return JSONResponse({"status": "starting", "documents": 0}, status_code=503)
        return JSONResponse({"status": "ok", "documents": index.num_docs})

    # plain def on purpose: search blocks (CPU plus a disk read per snippet),
    # and FastAPI runs a def route in its thread pool instead of the event loop
    @app.get("/search", response_model=SearchResponse)
    def search(
        index: Annotated[Index, Depends(get_index)],
        settings: Annotated[Settings, Depends(get_settings)],
        q: Annotated[str, Query(min_length=1, max_length=200)],
        k: Annotated[int, Query(ge=1, le=50)] = 10,
        page: Annotated[int, Query(ge=1, le=100)] = 1,
        scorer: ScorerName = ScorerName.bm25,
    ) -> SearchResponse:
        return search_response(index, q, min(k, settings.max_k), page, scorer)

    @app.get("/docs/{doc_id}", response_model=DocResponse)
    def get_doc(
        doc_id: int, index: Annotated[Index, Depends(get_index)]
    ) -> DocResponse:
        meta = index.doc_meta.get(doc_id)
        if meta is None:
            raise HTTPException(status_code=404, detail=f"документа {doc_id} немає")
        return DocResponse(
            doc_id=doc_id,
            title=meta.title,
            length=index.doc_length(doc_id),
            text=document_text(meta),
        )

    @app.get("/stats", response_model=StatsResponse)
    def stats(index: Annotated[Index, Depends(get_index)]) -> StatsResponse:
        return StatsResponse(
            terms=len(index),
            documents=index.num_docs,
            avg_doc_length=round(index.avg_doc_length, 1),
            positions=index.has_positions,
            repr=index.repr_kind,
        )

    @app.get("/", response_class=HTMLResponse)
    def home(
        request: Request,
        settings: Annotated[Settings, Depends(get_settings)],
        index: Annotated[Index | None, Depends(get_index_optional)],
        q: Annotated[str, Query(max_length=200)] = "",
        page: Annotated[int, Query(ge=1, le=100)] = 1,
        scorer: ScorerName = ScorerName.bm25,
    ) -> HTMLResponse:
        context: dict[str, object] = {
            "q": q,
            "page": page,
            "scorer": scorer.value,
            "results": [],
            "total": 0,
            "pages": 1,
            "took_ms": 0.0,
            "error": "",
            "docs": index.num_docs if index else 0,
            "ready": index is not None,
        }
        if index is not None and q.strip():
            k = settings.default_k
            start = time.perf_counter()
            try:
                results, total, terms = run_search(index, q, k, page, scorer)
                context["results"] = [
                    {
                        "doc_id": r.doc_id,
                        "score": f"{r.score:.2f}",
                        "title": r.title,
                        "snippet": highlight(snippet_for(index, r.doc_id, terms)),
                    }
                    for r in results
                ]
                context["total"] = total
                context["pages"] = max(1, math.ceil(total / k))
            except ValueError as e:
                context["error"] = str(e)
            context["took_ms"] = round((time.perf_counter() - start) * 1000, 1)
        return templates.TemplateResponse(request, "index.html", context)

    @app.get("/doc/{doc_id}", response_class=HTMLResponse)
    def doc_page(
        request: Request,
        doc_id: int,
        index: Annotated[Index, Depends(get_index)],
        q: str = "",
    ) -> HTMLResponse:
        meta = index.doc_meta.get(doc_id)
        if meta is None:
            raise HTTPException(status_code=404, detail=f"документа {doc_id} немає")
        text = document_text(meta)
        body = text.split("\n", 1)[1] if "\n" in text else text
        return templates.TemplateResponse(
            request,
            "doc.html",
            {
                "title": meta.title,
                "doc_id": doc_id,
                "length": index.doc_length(doc_id),
                "paragraphs": [p for p in body.split("\n") if p.strip()],
                "q": q,
            },
        )

    return app


app = create_app()
