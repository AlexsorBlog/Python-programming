import sys
from pathlib import Path

from findex.index import Posting
from findex.scoring import BM25
from findex.search import ranked_search
from findex.store import open_index


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    bm25 = BM25()

    with open_index(Path(sys.argv[1])) as index:
        print("1) рідкісний терм важить більше за частий")
        print(f"   df(kyiv)={index.df('kyiv')}, df(the)={index.df('the')}")
        for result in ranked_search(index, "kyiv OR the", bm25, k=3):
            print(f"   {result.score:7.3f}  {result.title}")

        print("\n2) насичення: 20-те повторення майже нічого не додає")
        doc_id = index.doc_ids("kyiv")[0]
        print(
            f"   документ {doc_id!r} ({index.doc_length(doc_id)} токенів), терм 'kyiv'"
        )
        for tf in (1, 2, 3, 5, 20, 100):
            score = bm25.score("kyiv", Posting(doc_id, tf), index)
            print(f"   tf={tf:>4}  bal={score:7.3f}")

        print("\n3) короткий документ з одним входженням вище довгого")
        singles = [p.doc_id for p in index.iter_postings("kyiv") if p.tf == 1]
        singles.sort(key=index.doc_length)
        short, long = singles[0], singles[-1]
        for doc_id in (short, long):
            score = bm25.score("kyiv", Posting(doc_id, 1), index)
            meta = index.doc_meta[doc_id]
            print(
                f"   {score:7.3f}  {index.doc_length(doc_id):>6} токенів  {meta.title}"
            )


if __name__ == "__main__":
    main()
