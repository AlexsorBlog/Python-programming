import json
from pathlib import Path

import pytest

from findex.index import build_index
from findex.parallel import (
    Chunk,
    build_parallel,
    build_partial,
    count_documents,
    merge,
    plan_chunks,
)
from findex.search import boolean_search


@pytest.fixture
def shards(tmp_path: Path) -> Path:
    """Four shard files, three documents each."""
    folder = tmp_path / "shards"
    folder.mkdir()
    doc_id = 0
    for shard in range(4):
        lines: list[str] = []
        for _ in range(3):
            lines.append(
                json.dumps(
                    {
                        "id": doc_id,
                        "title": f"doc {doc_id}",
                        "text": f"term{doc_id} shared word city",
                    }
                )
            )
            doc_id += 1
        (folder / f"shard-{shard:03d}.jsonl").write_text(
            "\n".join(lines) + "\n", encoding="utf-8"
        )
    return folder


def test_count_documents(shards: Path):
    assert count_documents(next(iter(sorted(shards.glob("*.jsonl"))))) == 3


def test_plan_chunks_reserves_id_ranges(shards: Path):
    chunks = plan_chunks(shards, workers=4)
    assert [c.first_doc_id for c in chunks] == [0, 3, 6, 9]
    assert sum(len(c.paths) for c in chunks) == 4


def test_plan_chunks_with_fewer_workers_than_files(shards: Path):
    chunks = plan_chunks(shards, workers=2)
    assert len(chunks) == 2
    assert [c.first_doc_id for c in chunks] == [0, 6]


def test_serial_merge_matches_lab4_build(shards: Path):
    reference = build_index(shards)
    one_chunk = build_partial(Chunk(tuple(sorted(shards.glob("*.jsonl"))), 0))
    merged = merge([one_chunk.part])

    assert merged.postings == reference.postings
    assert merged.doc_lengths == reference.doc_lengths
    assert merged.doc_meta == reference.doc_meta


@pytest.mark.parametrize("executor", ["serial", "threads", "processes"])
def test_all_executors_give_the_same_index(shards: Path, executor: str):
    reference = build_index(shards)
    built, timings = build_parallel(shards, workers=4, executor=executor)  # type: ignore[arg-type]  # parametrized literal

    assert built.postings == reference.postings
    assert built.doc_lengths == reference.doc_lengths
    assert built.doc_meta == reference.doc_meta
    assert timings.wall > 0
    if executor != "serial":
        assert timings.chunks == 4


def test_doc_ids_stay_unique_and_sorted(shards: Path):
    built, _ = build_parallel(shards, workers=4, executor="threads")
    assert sorted(built.doc_meta) == list(range(12))
    for term in built:
        ids = built.doc_ids(term)
        assert ids == sorted(ids)
    assert boolean_search(built, "shared") == list(range(12))


def test_positions_survive_parallel_build(shards: Path):
    built, _ = build_parallel(shards, workers=2, executor="threads", positions=True)
    reference = build_index(shards, positions=True)
    assert built.postings == reference.postings


def test_array_repr_survives_parallel_build(shards: Path):
    built, _ = build_parallel(shards, workers=2, executor="threads", repr_kind="array")
    reference = build_index(shards, repr_kind="array")
    assert built.postings == reference.postings


def test_worker_error_propagates(shards: Path):
    # array + positions is rejected inside the worker, and the pool must
    # re-raise it here instead of hanging
    with pytest.raises(ValueError, match="array"):
        build_parallel(
            shards, workers=2, executor="processes", positions=True, repr_kind="array"
        )


def test_limit_is_respected(shards: Path):
    built, _ = build_parallel(shards, workers=4, executor="serial", limit=6)
    assert built.num_docs == 6
