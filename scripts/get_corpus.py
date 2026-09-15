import json
import sys
from pathlib import Path

import pyarrow.parquet as pq


def main() -> None:
    src, dst = Path(sys.argv[1]), Path(sys.argv[2])
    pf = pq.ParquetFile(src)
    count = 0
    with dst.open("w", encoding="utf-8") as out:
        for batch in pf.iter_batches(batch_size=1000, columns=["id", "title", "text"]):
            for row in batch.to_pylist():
                out.write(json.dumps(row, ensure_ascii=False) + "\n")
                count += 1
    print(f"записано {count} статей у {dst}")


if __name__ == "__main__":
    main()
