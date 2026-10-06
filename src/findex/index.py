import logging
from array import array
from collections import Counter, defaultdict
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass
from functools import cached_property
from itertools import islice
from pathlib import Path
from typing import Literal, Protocol, runtime_checkable

from findex.corpus import iter_documents
from findex.decorators import timed
from findex.tokenize import tokenize

log = logging.getLogger(__name__)

type ReprKind = Literal["slots", "plain", "array"]
REPRS: tuple[ReprKind, ...] = ("slots", "plain", "array")

type DocId = int
type ArrayEntry = tuple[array[int], array[int]]
type PostingsEntry = list[PostingLike] | ArrayEntry


@runtime_checkable
class PostingLike(Protocol):
    """Both Posting and PlainPosting satisfy this without inheriting it."""

    @property
    def doc_id(self) -> DocId: ...
    @property
    def tf(self) -> int: ...
    @property
    def positions(self) -> tuple[int, ...]: ...


@dataclass(frozen=True, slots=True)
class Posting:
    doc_id: DocId
    tf: int
    positions: tuple[int, ...] = ()


@dataclass(frozen=True)
class PlainPosting:
    doc_id: DocId
    tf: int
    positions: tuple[int, ...] = ()


@dataclass(frozen=True, slots=True)
class DocMeta:
    doc_id: DocId
    path: str
    title: str
    offset: int = 0


class Index(Mapping[str, PostingsEntry]):
    """Term -> postings. Behaves like a read-only dict of terms."""

    def __init__(
        self,
        postings: dict[str, PostingsEntry] | None = None,
        doc_lengths: dict[DocId, int] | None = None,
        doc_meta: dict[DocId, DocMeta] | None = None,
        repr_kind: ReprKind = "slots",
        has_positions: bool = False,
    ) -> None:
        self.postings: dict[str, PostingsEntry] = postings if postings else {}
        self.doc_lengths: dict[DocId, int] = doc_lengths if doc_lengths else {}
        self.doc_meta: dict[DocId, DocMeta] = doc_meta if doc_meta else {}
        self.repr_kind: ReprKind = repr_kind
        self.has_positions: bool = has_positions

    def __len__(self) -> int:
        return len(self.postings)

    def __iter__(self) -> Iterator[str]:
        return iter(self.postings)

    def __getitem__(self, term: str) -> PostingsEntry:
        return self.postings[term]

    # Mapping declares __hash__ = None, but lru_cache needs a hashable key,
    # so I take the identity hash back: one index object = one cache key
    def __hash__(self) -> int:  # pyright: ignore[reportIncompatibleMethodOverride]
        return id(self)

    def __repr__(self) -> str:
        return (
            f"Index(terms={len(self):_}, docs={self.num_docs:_}, "
            f"repr={self.repr_kind!r}, positions={self.has_positions})"
        )

    @property
    def num_docs(self) -> int:
        return len(self.doc_lengths)

    @cached_property
    def avg_doc_length(self) -> float:
        if not self.doc_lengths:
            return 0.0
        return sum(self.doc_lengths.values()) / len(self.doc_lengths)

    def doc_length(self, doc_id: DocId) -> int:
        return self.doc_lengths.get(doc_id, 0)

    def df(self, term: str) -> int:
        entry = self.postings.get(term)
        if entry is None:
            return 0
        if isinstance(entry, tuple):
            return len(entry[0])
        return len(entry)

    def iter_postings(self, term: str) -> Iterator[PostingLike]:
        entry = self.postings.get(term)
        if entry is None:
            return
        if isinstance(entry, tuple):
            for doc_id, tf in zip(entry[0], entry[1], strict=True):
                yield Posting(doc_id, tf)
        else:
            yield from entry

    def doc_ids(self, term: str) -> list[DocId]:
        entry = self.postings.get(term)
        if entry is None:
            return []
        if isinstance(entry, tuple):
            return list(entry[0])
        return [p.doc_id for p in entry]

    def term_frequencies(self, term: str) -> dict[DocId, int]:
        return {p.doc_id: p.tf for p in self.iter_postings(term)}

    def positions(self, term: str, doc_id: DocId) -> tuple[int, ...]:
        for posting in self.iter_postings(term):
            if posting.doc_id == doc_id:
                return posting.positions
        return ()

    def close(self) -> None:
        self.postings = {}
        self.doc_lengths = {}
        self.doc_meta = {}
        self.__dict__.pop("avg_doc_length", None)


def _title_of(text: str) -> str:
    head = text.lstrip().split("\n", 1)[0]
    return head[:80] if head else "(без назви)"


@timed
def build_index(
    root: Path,
    limit: int | None = None,
    positions: bool = False,
    repr_kind: ReprKind = "slots",
    on_progress: Callable[[int], None] | None = None,
) -> Index:
    if repr_kind == "array" and positions:
        raise ValueError("позиції не підтримуються для --repr array")

    cls = Posting if repr_kind == "slots" else PlainPosting
    objects: defaultdict[str, list[PostingLike]] = defaultdict(list)
    arrays: dict[str, ArrayEntry] = {}
    doc_lengths: dict[DocId, int] = {}
    doc_meta: dict[DocId, DocMeta] = {}

    # doc_id grows with the stream, so postings come out sorted by doc_id already
    for doc_id, doc in enumerate(islice(iter_documents(root), limit)):
        doc_meta[doc_id] = DocMeta(
            doc_id, str(doc.path), _title_of(doc.text), doc.offset
        )

        if positions:
            where: defaultdict[str, list[int]] = defaultdict(list)
            for pos, term in enumerate(tokenize(doc.text)):
                where[term].append(pos)
            doc_lengths[doc_id] = sum(len(v) for v in where.values())
            for term, found in where.items():
                objects[term].append(cls(doc_id, len(found), tuple(found)))
        else:
            counts = Counter(tokenize(doc.text))
            doc_lengths[doc_id] = counts.total()
            if repr_kind == "array":
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

        if on_progress is not None and doc_id % 200 == 0:
            on_progress(doc_id)

    if on_progress is not None:
        on_progress(len(doc_meta))

    postings: dict[str, PostingsEntry] = (
        dict(arrays) if repr_kind == "array" else dict(objects)
    )
    log.info("індекс готовий: %d термів, %d документів", len(postings), len(doc_meta))
    return Index(postings, doc_lengths, doc_meta, repr_kind, positions)
