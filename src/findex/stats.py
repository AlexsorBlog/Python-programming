from collections import Counter
from dataclasses import dataclass
from itertools import islice
from pathlib import Path

from findex.corpus import iter_documents
from findex.decorators import timed
from findex.tokenize import tokenize


@dataclass(frozen=True, slots=True)
class Stats:
    documents: int
    tokens: int
    counts: Counter[str]


@timed
def lazy_stats(root: Path, limit: int | None = None) -> Stats:
    counts: Counter[str] = Counter()
    n_docs = 0
    for doc in islice(iter_documents(root), limit):
        n_docs += 1
        counts.update(tokenize(doc.text))
    return Stats(n_docs, counts.total(), counts)


@timed
def eager_stats(root: Path, limit: int | None = None) -> Stats:
    # lists on purpose, to compare with the lazy version
    docs = list(islice(iter_documents(root), limit))
    tokenized = [list(tokenize(doc.text)) for doc in docs]
    all_tokens = [tok for doc_tokens in tokenized for tok in doc_tokens]
    return Stats(len(docs), len(all_tokens), Counter(all_tokens))


def human(n: float) -> str:
    for unit in ("Б", "КБ", "МБ", "ГБ"):
        if n < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} ТБ"
