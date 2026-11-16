"""precision@5 for lexical / semantic / hybrid.

    uv run --group semantic python scripts/eval_modes.py data/index.pkl data/emb_20k

10 hand-labelled queries from lab 3 plus 5 paraphrases that avoid the key
words of the documents they should find.
"""

import sys
from pathlib import Path

from eval_queries import LABELS

from findex.hybrid import semantic_results
from findex.scoring import BM25
from findex.search import ranked_search
from findex.store import load

PLANETS = LABELS["solar system planet"]
EVEREST = LABELS["mount everest mountain"]
EINSTEIN = LABELS["albert einstein physics"]
PRESIDENT = LABELS["united states president"]
ROCK = LABELS["music rock band"]

# the paraphrase must not lean on the obvious words: no "planet", "everest",
# "einstein", "president", "rock" or "band"
PARAPHRASES: dict[str, set[str]] = {
    "huge balls of gas and stone that circle the sun": PLANETS,
    "tallest summit on earth between nepal and tibet": EVEREST,
    "physicist who said that time slows down when you move fast": EINSTEIN,
    "the person who leads the american government from the white house": PRESIDENT,
    "loud groups of musicians with electric guitars and drums": ROCK,
}


def precision(titles: list[str], relevant: set[str]) -> float:
    return sum(t in relevant for t in titles[:5]) / 5


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    index = load(Path(sys.argv[1]))
    emb = Path(sys.argv[2])

    def lexical(q: str) -> list[str]:
        try:
            return [r.title for r in ranked_search(index, q, BM25(), k=5)]
        except ValueError:
            return []

    def neural(q: str, mode: str) -> list[str]:
        return [r.title for r in semantic_results(index, emb, q, mode, k=5)]  # type: ignore[arg-type]

    rows = [(q, rel, "з лаби 3") for q, rel in LABELS.items()]
    rows += [(q, rel, "перефраз") for q, rel in PARAPHRASES.items()]

    totals = {"lexical": [0.0, 0.0], "semantic": [0.0, 0.0], "hybrid": [0.0, 0.0]}
    print("| Запит | Тип | BM25 | Семантика | Гібрид (RRF) |")
    print("|---|---|---|---|---|")
    for q, relevant, kind in rows:
        p = {
            "lexical": precision(lexical(q), relevant),
            "semantic": precision(neural(q, "semantic"), relevant),
            "hybrid": precision(neural(q, "hybrid"), relevant),
        }
        slot = 0 if kind == "з лаби 3" else 1
        for mode, value in p.items():
            totals[mode][slot] += value
        cells = " | ".join(f"{p[m]:.1f}" for m in ("lexical", "semantic", "hybrid"))
        print(f"| {q} | {kind} | {cells} |", flush=True)

    for slot, label, n in (
        (0, "лаба 3", len(LABELS)),
        (1, "перефрази", len(PARAPHRASES)),
    ):
        cells = " | ".join(
            f"**{totals[m][slot] / n:.2f}**" for m in ("lexical", "semantic", "hybrid")
        )
        print(f"| **середнє, {label}** | | {cells} |")


if __name__ == "__main__":
    main()
