import asyncio
import enum
import io
import json
import logging
import os
import sys
import time
import tracemalloc
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.markup import escape
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn
from rich.table import Table

from findex.crawler import CrawlStats
from findex.parallel import build_parallel
from findex.scoring import BM25, TfIdf
from findex.search import ranked_search
from findex.stats import eager_stats, human, lazy_stats
from findex.store import open_index, save


# typer builds options from enums, not from Literal aliases,
# so the library keeps Literal and the CLI keeps these
class ReprChoice(enum.StrEnum):
    slots = "slots"
    plain = "plain"
    array = "array"


class ScorerChoice(enum.StrEnum):
    bm25 = "bm25"
    tfidf = "tfidf"


class ExecutorChoice(enum.StrEnum):
    serial = "serial"
    threads = "threads"
    processes = "processes"


# Windows consoles default to cp1252 and crash on Ukrainian output.
# This has to run at import time: typer prints --help before any callback.
for _stream in (sys.stdout, sys.stderr):
    if isinstance(_stream, io.TextIOWrapper):
        _stream.reconfigure(encoding="utf-8")


app = typer.Typer(
    no_args_is_help=True,
    add_completion=True,
    help="findex — пошук по власному корпусу текстів",
)
log = logging.getLogger("findex")

# private markers: the snippet gets them, then _highlight swaps in rich tags
_OPEN, _CLOSE = "⟦", "⟧"

# rich Console: stdout for data, stderr for diagnostics
out = Console()
err = Console(stderr=True)


def _fail(message: str) -> typer.Exit:
    """One clear line to stderr, no traceback."""
    err.print(f"[red]помилка:[/red] {message}")
    return typer.Exit(code=1)


@app.callback()
def main(
    verbose: Annotated[
        int,
        typer.Option("-v", "--verbose", count=True, help="-v: INFO, -vv: DEBUG"),
    ] = 0,
) -> None:
    level = {0: logging.WARNING, 1: logging.INFO}.get(verbose, logging.DEBUG)
    logging.basicConfig(
        level=level,
        format="%(levelname)s %(name)s | %(message)s",
        stream=sys.stderr,
    )


@app.command()
def index(
    corpus: Annotated[Path, typer.Argument(help="файл або папка з корпусом")],
    out_path: Annotated[Path, typer.Option("--out", help="куди зберегти індекс")],
    limit: Annotated[int | None, typer.Option(help="тільки перші N документів")] = None,
    positions: Annotated[bool, typer.Option(help="зберігати позиції токенів")] = False,
    repr_kind: Annotated[
        ReprChoice, typer.Option("--repr", help="спосіб зберігання постингів")
    ] = ReprChoice.slots,
    workers: Annotated[
        int, typer.Option("--workers", help="скільки воркерів (0 = os.cpu_count())")
    ] = 1,
    executor: Annotated[
        ExecutorChoice, typer.Option("--executor", help="як розпаралелити збірку")
    ] = ExecutorChoice.serial,
) -> None:
    """Побудувати індекс і зберегти його у файл."""
    if not corpus.exists():
        raise _fail(f"немає такого корпусу: {corpus}")
    if repr_kind is ReprChoice.array and positions:
        raise _fail("позиції не підтримуються для --repr array")

    jobs = workers if workers > 0 else (os.cpu_count() or 1)

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        console=err,
    ) as progress:
        progress.add_task(f"індексую ({executor.value}, {jobs} воркер(ів))", total=None)
        built, timings = build_parallel(
            corpus,
            jobs,
            executor.value,
            limit,
            positions,
            repr_kind.value,
        )

    out.print(f"{built!r}")
    out.print(
        f"час побудови: {timings.wall:.2f} с (merge {timings.merge:.2f} с), "
        f"CPU {timings.cpu:.2f} с, шматків: {timings.chunks}"
    )

    start = time.perf_counter()
    save(built, out_path)
    size = out_path.stat().st_size
    out.print(
        f"збережено в {out_path} за {time.perf_counter() - start:.2f} с ({human(size)})"
    )


@app.command()
def search(
    index_path: Annotated[Path, typer.Argument(help="файл індексу")],
    query: Annotated[str, typer.Argument(help='запит, напр. kyiv AND "nuclear power"')],
    k: Annotated[int, typer.Option("--k", help="скільки результатів")] = 10,
    scorer: Annotated[
        ScorerChoice, typer.Option(help="формула рейтингу")
    ] = ScorerChoice.bm25,
    as_json: Annotated[
        bool, typer.Option("--json", help="вивести JSON-рядки в stdout")
    ] = False,
    snippets: Annotated[bool, typer.Option(help="показувати сніпети")] = True,
) -> None:
    """Знайти документи у збереженому індексі."""
    if not index_path.exists():
        raise _fail(f"немає файлу індексу: {index_path}")

    with open_index(index_path) as opened:
        try:
            results = ranked_search(
                opened,
                query,
                BM25() if scorer is ScorerChoice.bm25 else TfIdf(),
                k,
                snippets=snippets and not as_json,
                marker=(_OPEN, _CLOSE),
            )
        except ValueError as e:
            raise _fail(str(e)) from None

        if as_json:
            for result in results:
                line = {
                    "doc_id": result.doc_id,
                    "score": round(result.score, 4),
                    "title": result.title,
                }
                # plain print, not rich: stdout must stay machine-readable
                print(json.dumps(line, ensure_ascii=False))
            return

        if not results:
            out.print("нічого не знайдено")
            return

        table = Table(title=f"{query}  (scorer={scorer.value})", title_justify="left")
        table.add_column("бал", justify="right", style="cyan")
        table.add_column("заголовок", style="bold")
        if snippets:
            table.add_column("сніпет", overflow="fold")
        for result in results:
            row = [f"{result.score:.3f}", result.title]
            if snippets:
                row.append(_highlight(result.snippet))
            table.add_row(*row)
        out.print(table)


def _highlight(snippet: str) -> str:
    """Escape the article text first, then turn my own markers into rich tags."""
    return escape(snippet).replace(_OPEN, "[yellow]").replace(_CLOSE, "[/yellow]")


@app.command()
def stats(
    path: Annotated[Path, typer.Argument(help="корпус або файл індексу")],
    limit: Annotated[int | None, typer.Option(help="тільки перші N документів")] = None,
    top: Annotated[int, typer.Option(help="скільки топ-термів показати")] = 10,
    eager: Annotated[bool, typer.Option(help="жадібна версія через списки")] = False,
) -> None:
    """Статистика корпусу або готового індексу."""
    if not path.exists():
        raise _fail(f"немає такого файлу: {path}")

    table = Table(show_header=False)
    table.add_column(style="dim")
    table.add_column(justify="right")

    if path.suffix in (".pkl", ".json"):
        with open_index(path) as opened:
            table.add_row("термів", f"{len(opened):,}")
            table.add_row("документів", f"{opened.num_docs:,}")
            table.add_row("середня довжина", f"{opened.avg_doc_length:.1f}")
            table.add_row("позиції", "так" if opened.has_positions else "ні")
        out.print(table)
        return

    tracemalloc.start()
    start = time.perf_counter()
    result = eager_stats(path, limit) if eager else lazy_stats(path, limit)
    elapsed = time.perf_counter() - start
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    table.add_row("режим", "eager (списки)" if eager else "lazy (генератори)")
    table.add_row("документів", f"{result.documents:,}")
    table.add_row("токенів", f"{result.tokens:,}")
    table.add_row("словник", f"{len(result.counts):,}")
    table.add_row("час", f"{elapsed:.2f} с")
    table.add_row("пікова пам'ять", human(peak))
    out.print(table)

    if top:
        terms = Table(title=f"топ-{top} термів", title_justify="left")
        terms.add_column("#", justify="right", style="dim")
        terms.add_column("терм")
        terms.add_column("к-сть", justify="right", style="cyan")
        for rank, (term, count) in enumerate(result.counts.most_common(top), start=1):
            terms.add_row(str(rank), term, f"{count:,}")
        out.print(terms)


__all__ = ["ReprChoice", "ScorerChoice", "app"]


@app.command()
def crawl(
    seed: Annotated[str, typer.Argument(help="початковий URL")],
    out_path: Annotated[
        Path, typer.Option("--out", help="куди писати сторінки (jsonl)")
    ] = Path("data/crawl.jsonl"),
    max_pages: Annotated[int, typer.Option(help="стеля на число сторінок")] = 200,
    concurrency: Annotated[int, typer.Option(help="одночасних запитів усього")] = 10,
    per_host: Annotated[int, typer.Option(help="одночасних запитів на хост")] = 2,
    host_delay: Annotated[
        float, typer.Option(help="мінімальна пауза між запитами до хоста, с")
    ] = 0.2,
    log_path: Annotated[
        Path, typer.Option("--log", help="лог статусів по кожному URL")
    ] = Path("crawl.log"),
    debug_loop: Annotated[
        bool, typer.Option("--debug-loop", help="asyncio у режимі debug")
    ] = False,
) -> None:
    """Обійти сайт і зберегти сторінки у jsonl (індексація - окремою командою)."""
    if not seed.startswith(("http://", "https://")):
        raise _fail("URL має починатися з http:// або https://")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    url_logger = logging.getLogger("findex.crawl.urls")
    url_logger.setLevel(logging.INFO)
    url_logger.propagate = False
    handler = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(message)s"))
    url_logger.addHandler(handler)

    stats = CrawlStats()
    started = time.perf_counter()

    try:
        asyncio.run(
            _crawl_to_file(
                seed, out_path, max_pages, concurrency, per_host, host_delay, stats
            ),
            debug=debug_loop,
        )
    except KeyboardInterrupt:
        err.print("[yellow]перервано[/yellow]")
    finally:
        handler.close()
        url_logger.removeHandler(handler)

    elapsed = time.perf_counter() - started
    table = Table(show_header=False)
    table.add_column(style="dim")
    table.add_column(justify="right")
    table.add_row("сторінок", f"{stats.fetched:,}")
    table.add_row("час", f"{elapsed:.1f} с")
    table.add_row("сторінок/с", f"{stats.fetched / elapsed if elapsed else 0:.2f}")
    table.add_row("помилок", str(stats.errors))
    table.add_row("пропущено", str(stats.skipped))
    table.add_row("файл", str(out_path))
    table.add_row("лог", str(log_path))
    out.print(table)


async def _crawl_to_file(
    seed: str,
    out_path: Path,
    max_pages: int,
    concurrency: int,
    per_host: int,
    host_delay: float,
    stats: CrawlStats,
) -> None:
    from rich.live import Live

    from findex.crawler import crawl as crawl_pages

    def render() -> Table:
        table = Table(show_header=False, box=None)
        table.add_column(style="dim")
        table.add_column(justify="right")
        table.add_row("зібрано", f"{stats.fetched}/{max_pages}")
        table.add_row("сторінок/с", f"{stats.pages_per_second:.2f}")
        table.add_row("в польоті", str(stats.in_flight))
        table.add_row("у черзі", str(stats.queued))
        table.add_row("помилок", str(stats.errors))
        return table

    # pages are written as they arrive: the whole crawl never sits in memory
    with (
        out_path.open("w", encoding="utf-8") as f,
        Live(render(), console=err, refresh_per_second=4) as live,
    ):
        async for page in crawl_pages(
            [seed],
            max_pages=max_pages,
            concurrency=concurrency,
            per_host=per_host,
            host_delay=host_delay,
            stats=stats,
        ):
            f.write(
                json.dumps(
                    {
                        "id": page.url,
                        "title": page.title,
                        "text": page.text,
                        "fetched_at": page.fetched_at,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            live.update(render())


@app.command()
def serve(
    index_path: Annotated[
        Path | None,
        typer.Option("--index", help="файл індексу (інакше INDEX_PATH)"),
    ] = None,
    host: Annotated[str, typer.Option(help="адреса")] = "127.0.0.1",
    port: Annotated[int, typer.Option(help="порт")] = 8000,
    workers: Annotated[int, typer.Option(help="скільки процесів uvicorn")] = 1,
) -> None:
    """Запустити веб-сервіс (FastAPI + uvicorn)."""
    import uvicorn

    if index_path is not None:
        # settings come only from the environment, so the flag just sets it
        os.environ["INDEX_PATH"] = str(index_path)
    path = Path(os.environ.get("INDEX_PATH", "data/demo.pkl"))
    if not path.exists():
        raise _fail(f"немає файлу індексу: {path}")

    err.print(f"findex на http://{host}:{port}  (індекс {path}, воркерів {workers})")
    uvicorn.run(
        "findex.web.app:app",
        host=host,
        port=port,
        workers=workers,
        log_level="info",
    )
