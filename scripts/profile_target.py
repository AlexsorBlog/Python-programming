"""Small entry points for the profilers (Scalene wants a script, not -m).

uv run python scripts/profile_target.py search data/full.pkl "united states president"
uv run python scripts/profile_target.py index data/simplewiki.jsonl 20000
"""

import sys
from pathlib import Path

from findex.index import build_index
from findex.scoring import BM25
from findex.search import ranked_search
from findex.store import load, save


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    what = sys.argv[1]
    if what == "search":
        index = load(Path(sys.argv[2]))
        for result in ranked_search(index, sys.argv[3], BM25(), k=10):
            print(f"{result.score:7.3f}  {result.title}")
    elif what == "index":
        index = build_index(Path(sys.argv[2]), limit=int(sys.argv[3]))
        save(index, Path("data/profile_tmp.pkl"))
        Path("data/profile_tmp.pkl").unlink()
    else:
        raise SystemExit("search або index")


if __name__ == "__main__":
    main()
