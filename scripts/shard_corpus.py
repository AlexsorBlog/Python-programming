"""Split one big jsonl corpus into N shard files.

    uv run python scripts/shard_corpus.py data/simplewiki.jsonl data/shards 32

Workers get paths, so the corpus has to be more than one file.
"""

import argparse
import sys
from pathlib import Path


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="Поділити корпус на шарди")
    parser.add_argument("source", type=Path)
    parser.add_argument("out_dir", type=Path)
    parser.add_argument("shards", type=int)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for old in args.out_dir.glob("*.jsonl"):
        old.unlink()

    total = 0
    with args.source.open("rb") as src:
        lines: list[bytes] = []
        for raw in src:
            lines.append(raw)
            total += 1
            if args.limit is not None and total >= args.limit:
                break

    per_shard = -(-len(lines) // args.shards)
    written = 0
    for i in range(args.shards):
        part = lines[i * per_shard : (i + 1) * per_shard]
        if not part:
            break
        # zero padded so sorted() keeps shard order the same as line order
        target = args.out_dir / f"shard-{i:03d}.jsonl"
        with target.open("wb") as out:
            out.writelines(part)
        written += 1

    print(f"{total} документів -> {written} шардів у {args.out_dir}")


if __name__ == "__main__":
    main()
