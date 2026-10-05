import argparse
import sys
import time
import tracemalloc
from array import array
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from itertools import islice
from pathlib import Path

from findex.corpus import iter_documents
from findex.tokenize import tokenize

REPRS = ("slots", "plain", "array")


@dataclass(frozen=True, slots=True)
class Posting:
    doc_id: int
    tf: int
    positions: tuple[int, ...] = ()


@dataclass(frozen=True)
class PlainPosting:
    doc_id: int
    tf: int
    positions: tuple[int, ...] = ()


@dataclass(frozen=True, slots=True)
class DocMeta:
    doc_id: int
    path: str
    title: str


@dataclass
class Index:
    postings: dict = field(default_factory=dict)
    doc_lengths: dict[int, int] = field(default_factory=dict)
    doc_meta: dict[int, DocMeta] = field(default_factory=dict)
    repr_kind: str = "slots"
    has_positions: bool = False

    def doc_ids(self, term: str) -> list[int]:
        entry = self.postings.get(term)
        if entry is None:
            return []
        if self.repr_kind == "array":
            return list(entry[0])
        return [p.doc_id for p in entry]

    def term_frequencies(self, term: str) -> dict[int, int]:
        entry = self.postings.get(term)
        if entry is None:
            return {}
        if self.repr_kind == "array":
            return dict(zip(entry[0], entry[1], strict=True))
        return {p.doc_id: p.tf for p in entry}


def _title_of(text: str) -> str:
    head = text.lstrip().split("\n", 1)[0]
    return head[:80] if head else "(без назви)"


def build_index(
    root: Path,
    limit: int | None = None,
    positions: bool = False,
    repr_kind: str = "slots",
) -> Index:
    if repr_kind == "array" and positions:
        raise SystemExit("позиції не підтримуються для --repr array")

    index = Index(
        postings=defaultdict(list) if repr_kind != "array" else {},
        repr_kind=repr_kind,
        has_positions=positions,
    )
    cls = Posting if repr_kind == "slots" else PlainPosting

    # doc_id grows with the stream, so postings come out sorted by doc_id already
    for doc_id, doc in enumerate(islice(iter_documents(root), limit)):
        index.doc_meta[doc_id] = DocMeta(doc_id, str(doc.path), _title_of(doc.text))

        if positions:
            where: dict[str, list[int]] = defaultdict(list)
            for pos, term in enumerate(tokenize(doc.text)):
                where[term].append(pos)
            index.doc_lengths[doc_id] = sum(len(v) for v in where.values())
            for term, poss in where.items():
                index.postings[term].append(cls(doc_id, len(poss), tuple(poss)))
            continue

        counts = Counter(tokenize(doc.text))
        index.doc_lengths[doc_id] = counts.total()
        if repr_kind == "array":
            for term, tf in counts.items():
                entry = index.postings.get(term)
                if entry is None:
                    entry = (array("I"), array("I"))
                    index.postings[term] = entry
                entry[0].append(doc_id)
                entry[1].append(tf)
        else:
            for term, tf in counts.items():
                index.postings[term].append(cls(doc_id, tf))

    if isinstance(index.postings, defaultdict):
        index.postings = dict(index.postings)
    return index


def main() -> None:
    parser = argparse.ArgumentParser(description="Побудова інвертованого індексу")
    parser.add_argument("root", type=Path, help="файл або папка з корпусом")
    parser.add_argument("--out", type=Path, default=None, help="куди зберегти індекс")
    parser.add_argument(
        "--limit", type=int, default=None, help="тільки перші N документів"
    )
    parser.add_argument(
        "--positions", action="store_true", help="зберігати позиції токенів"
    )
    parser.add_argument(
        "--repr", choices=REPRS, default="slots", help="спосіб зберігання постингів"
    )
    args = parser.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")

    from findex.stats import human

    tracemalloc.start()
    start = time.perf_counter()
    index = build_index(args.root, args.limit, args.positions, args.repr)
    build_time = time.perf_counter() - start
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    print(f"постинги:       {args.repr}" + (" + позиції" if args.positions else ""))
    print(f"документів:     {len(index.doc_meta):,}")
    print(f"термів:         {len(index.postings):,}")
    print(f"час побудови:   {build_time:.2f} с")
    print(f"пікова пам'ять: {human(peak)}")

    if args.out:
        from findex.store import save

        start = time.perf_counter()
        save(index, args.out)
        print(f"збережено в {args.out} за {time.perf_counter() - start:.2f} с")
        print(f"розмір файлу:   {human(args.out.stat().st_size)}")


if __name__ == "__main__":
    # import again so the dataclasses belong to findex.index and not to __main__,
    # otherwise pickle writes classes that findex.search cannot find
    from findex.index import main as entry

    entry()
