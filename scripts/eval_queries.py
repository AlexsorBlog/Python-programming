"""precision@5 for TF-IDF and BM25 on 10 queries I labelled by hand."""

import sys
from pathlib import Path

from findex.scoring import BM25, TfIdf
from findex.search import ranked_search
from findex.store import open_index

# query -> titles I decided should be in the top 5
LABELS: dict[str, set[str]] = {
    "chernobyl disaster": {
        "Chernobyl disaster",
        "Chernobyl",
        "Pripyat",
        "Nuclear meltdown",
        "International Nuclear Event Scale",
    },
    "solar system planet": {
        "Solar System",
        "Planet",
        "List of planets",
        "Dwarf planet",
        "Jupiter",
        "Mars",
        "Neptune",
        "Saturn",
        "Mercury (planet)",
    },
    "python programming language": {
        "Python (programming language)",
        "Programming language",
        "C (programming language)",
        "Computer programming",
        "Object-oriented programming",
        "PHP",
    },
    "ukraine capital city": {"Kyiv", "Ukraine", "Lviv", "Odesa", "Sevastopol"},
    "albert einstein physics": {
        "Albert Einstein",
        "Theory of relativity",
        "General relativity",
        "Special relativity",
        "Relativity",
        "Physics",
        "Spacetime",
    },
    "mount everest mountain": {
        "Mount Everest",
        "Himalayas",
        "Edmund Hillary",
        "Tenzing Norgay",
        "K2",
        "Mountain",
        "Nepal",
    },
    "united states president": {
        "President of the United States",
        "Vice President of the United States",
        "United States presidential line of succession",
        "Ronald Reagan",
        "Jimmy Carter",
        "Donald Trump",
        "George H. W. Bush",
        "Millard Fillmore",
    },
    "music rock band": {
        "Rock band",
        "Southern rock",
        "Thin Lizzy",
        "Iron Maiden",
        "Linkin Park",
        "Coldplay",
        "Megadeth",
        "Deep Purple",
        "Eagles (band)",
        "Green Day",
        "Red Hot Chili Peppers",
        "Muse (band)",
        "Journey (band)",
        "Dinosaur Jr",
        "Motörhead",
        "Midnight Oil",
    },
    "computer software": {
        "Software",
        "Software engineering",
        "Free software",
        "Computer hardware",
        "Computer science",
        "Computer",
        "Programmer",
    },
    "ancient rome empire": {
        "Ancient Rome",
        "Roman Empire",
        "Military of ancient Rome",
        "Roman Italy",
        "Rome",
        "Ancient history",
        "Byzantine Empire",
    },
}


def precision_at_5(index, query: str, scorer) -> float:
    titles = [r.title for r in ranked_search(index, query, scorer, k=5)]
    return sum(t in LABELS[query] for t in titles) / 5


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    scorers = {"tfidf": TfIdf(), "bm25": BM25()}
    totals = dict.fromkeys(scorers, 0.0)

    with open_index(Path(sys.argv[1] if len(sys.argv) > 1 else "data/index.pkl")) as ix:
        print(f"| {'запит':<30} | tfidf P@5 | bm25 P@5 |")
        print("|---|---|---|")
        for query in LABELS:
            row = {}
            for name, scorer in scorers.items():
                row[name] = precision_at_5(ix, query, scorer)
                totals[name] += row[name]
            print(f"| {query:<30} | {row['tfidf']:.1f} | {row['bm25']:.1f} |")
        n = len(LABELS)
        tfidf_avg = totals["tfidf"] / n
        bm25_avg = totals["bm25"] / n
        print(f"| **середнє** | **{tfidf_avg:.2f}** | **{bm25_avg:.2f}** |")


if __name__ == "__main__":
    main()
