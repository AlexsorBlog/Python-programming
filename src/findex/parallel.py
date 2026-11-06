import logging
import time
from array import array
from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from concurrent.futures import Executor, ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, NamedTuple

from findex.corpus import corpus_paths, iter_documents
from findex.index import (
    ArrayEntry,
    DocId,
    DocMeta,
    Index,
    PlainPosting,
    Posting,
    PostingLike,
    PostingsEntry,
    ReprKind,
    title_of,
)
from findex.tokenize import tokenize

log = logging.getLogger(__name__)

type ExecutorKind = Literal["serial", "threads", "processes"]
EXECUTORS: tuple[ExecutorKind, ...] = ("serial", "threads", "processes")


@dataclass(frozen=True, slots=True)
class Chunk:
    """One unit of work: file paths plus the doc_id range reserved for them."""

    paths: tuple[Path, ...]
    first_doc_id: int
    positions: bool = False
    repr_kind: ReprKind = "slots"
    max_docs: int | None = None


@dataclass(frozen=True, slots=True)
class Partial:
    # not a NamedTuple: tuple already has an .index() method
    part: Index
    cpu_seconds: float
    documents: int


class Timings(NamedTuple):
    wall: float
    cpu: float
    merge: float
    chunks: int


def count_documents(path: Path) -> int:
    """Count lines without decoding them."""
    if path.suffix != ".jsonl":
        return 1
    with path.open("rb") as f:
        return sum(1 for _ in f)


def plan_chunks(
    root: Path,
    workers: int,
    limit: int | None = None,
    positions: bool = False,
    repr_kind: ReprKind = "slots",
) -> list[Chunk]:
    """Split paths into chunks and hand each one its own doc_id range."""
    paths = corpus_paths(root)
    sizes = [count_documents(p) for p in paths]

    if limit is not None:
        kept: list[Path] = []
        kept_sizes: list[int] = []
        left = limit
        for path, size in zip(paths, sizes, strict=True):
            if left <= 0:
                break
            kept.append(path)
            kept_sizes.append(min(size, left))
            left -= size
        paths, sizes = kept, kept_sizes

    groups = max(1, min(workers, len(paths)))
    per_group = -(-len(paths) // groups)

    chunks: list[Chunk] = []
    next_id = 0
    for i in range(0, len(paths), per_group):
        group = paths[i : i + per_group]
        taken = sum(sizes[i : i + per_group])
        chunks.append(
            Chunk(
                tuple(group),
                next_id,
                positions,
                repr_kind,
                taken if limit is not None else None,
            )
        )
        next_id += taken
    return chunks


def build_partial(chunk: Chunk) -> Partial:
    """Index one chunk. Module level so a process pool can pickle it."""
    if chunk.repr_kind == "array" and chunk.positions:
        raise ValueError("позиції не підтримуються для --repr array")

    cpu_start = time.process_time()
    cls = Posting if chunk.repr_kind == "slots" else PlainPosting
    objects: defaultdict[str, list[PostingLike]] = defaultdict(list)
    arrays: dict[str, ArrayEntry] = {}
    doc_lengths: dict[DocId, int] = {}
    doc_meta: dict[DocId, DocMeta] = {}

    doc_id = chunk.first_doc_id
    budget = chunk.max_docs
    for path in chunk.paths:
        for doc in iter_documents(path):
            if budget is not None and doc_id - chunk.first_doc_id >= budget:
                break
            doc_meta[doc_id] = DocMeta(
                doc_id, str(doc.path), title_of(doc.text), doc.offset
            )
            if chunk.positions:
                where: defaultdict[str, list[int]] = defaultdict(list)
                for pos, term in enumerate(tokenize(doc.text)):
                    where[term].append(pos)
                doc_lengths[doc_id] = sum(len(v) for v in where.values())
                for term, found in where.items():
                    objects[term].append(cls(doc_id, len(found), tuple(found)))
            else:
                counts = Counter(tokenize(doc.text))
                doc_lengths[doc_id] = counts.total()
                if chunk.repr_kind == "array":
                    for term, tf in counts.items():
                        entry = arrays.get(term)
                        if entry is None:
                            entry = (array("I"), array("I"))
                            arrays[term] = entry
                        entry[0].append(doc_id)
                        entry[1].append(tf)
                else:
                    for term, tf in counts.items():
                        objects[term].append(cls(doc_id, tf))
            doc_id += 1

    postings: dict[str, PostingsEntry] = (
        dict(arrays) if chunk.repr_kind == "array" else dict(objects)
    )
    index = Index(postings, doc_lengths, doc_meta, chunk.repr_kind, chunk.positions)
    return Partial(index, time.process_time() - cpu_start, len(doc_meta))


def merge(partials: Iterable[Index]) -> Index:
    """Concatenate postings. Chunks own increasing doc_id ranges, so the
    result stays sorted as long as the partials come in chunk order."""
    merged: dict[str, PostingsEntry] = {}
    doc_lengths: dict[DocId, int] = {}
    doc_meta: dict[DocId, DocMeta] = {}
    repr_kind: ReprKind = "slots"
    positions = False

    for part in partials:
        repr_kind = part.repr_kind
        positions = positions or part.has_positions
        doc_lengths.update(part.doc_lengths)
        doc_meta.update(part.doc_meta)

        for term, entry in part.postings.items():
            current = merged.get(term)
            if current is None:
                merged[term] = entry
            elif isinstance(entry, tuple) and isinstance(current, tuple):
                current[0].extend(entry[0])
                current[1].extend(entry[1])
            elif not isinstance(entry, tuple) and not isinstance(current, tuple):
                current.extend(entry)

    return Index(merged, doc_lengths, doc_meta, repr_kind, positions)


def _make_executor(kind: ExecutorKind, workers: int) -> Executor:
    if kind == "threads":
        return ThreadPoolExecutor(max_workers=workers)
    return ProcessPoolExecutor(max_workers=workers)


def build_parallel(
    root: Path,
    workers: int = 1,
    executor: ExecutorKind = "serial",
    limit: int | None = None,
    positions: bool = False,
    repr_kind: ReprKind = "slots",
) -> tuple[Index, Timings]:
    chunks = plan_chunks(root, workers, limit, positions, repr_kind)
    log.info("шматків: %d, воркерів: %d, executor=%s", len(chunks), workers, executor)

    wall_start = time.perf_counter()
    cpu_start = time.process_time()
    parts: Sequence[Partial]

    if executor == "serial":
        parts = [build_partial(chunk) for chunk in chunks]
    else:
        with _make_executor(executor, workers) as pool:
            # map keeps chunk order, and .result() re-raises a worker's
            # exception here with its own traceback instead of hanging
            parts = list(pool.map(build_partial, chunks))

    worker_cpu = sum(p.cpu_seconds for p in parts)
    merge_start = time.perf_counter()
    index = merge(p.part for p in parts)
    merge_time = time.perf_counter() - merge_start

    wall = time.perf_counter() - wall_start
    parent_cpu = time.process_time() - cpu_start
    # on threads and serial the worker CPU is already inside parent_cpu
    cpu = parent_cpu if executor != "processes" else parent_cpu + worker_cpu
    log.info("готово за %.2f с (merge %.2f с), CPU %.2f с", wall, merge_time, cpu)
    return index, Timings(wall, cpu, merge_time, len(chunks))
