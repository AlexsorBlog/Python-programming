import enum
import io
import json
import logging
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

from findex.index import build_index
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
) -> None:
    """Побудувати індекс і зберегти його у файл."""
    if not corpus.exists():
        raise _fail(f"немає такого корпусу: {corpus}")
    if repr_kind is ReprChoice.array and positions:
        raise _fail("позиції не підтримуються для --repr array")

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed:,} документів"),
        console=err,
    ) as progress:
        task = progress.add_task("індексую", total=limit)
        tracemalloc.start()
        start = time.perf_counter()
        built = build_index(
            corpus,
            limit,
            positions,
            repr_kind.value,
            on_progress=lambda done: progress.update(task, completed=done),
        )
        elapsed = time.perf_counter() - start
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

    out.print(f"{built!r}")
    out.print(f"час побудови: {elapsed:.2f} с, пікова пам'ять: {human(peak)}")

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
