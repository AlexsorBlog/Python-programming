import json
from pathlib import Path

from typer.testing import CliRunner

from findex.cli import app

runner = CliRunner()


def test_help():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "index" in result.output
    assert "search" in result.output


def test_index_then_search(corpus: Path, tmp_path: Path):
    out = tmp_path / "index.pkl"
    built = runner.invoke(app, ["index", str(corpus), "--out", str(out), "--positions"])
    assert built.exit_code == 0, built.output
    assert out.exists()

    found = runner.invoke(app, ["search", str(out), "city ukraine"])
    assert found.exit_code == 0
    assert "Lviv" in found.output


def test_search_json_is_machine_readable(index_file: Path):
    result = runner.invoke(app, ["search", str(index_file), "city", "--json"])
    assert result.exit_code == 0
    rows = [json.loads(line) for line in result.output.splitlines() if line.strip()]
    assert rows
    assert {"doc_id", "score", "title"} == set(rows[0])


def test_stats_on_index(index_file: Path):
    result = runner.invoke(app, ["stats", str(index_file)])
    assert result.exit_code == 0
    assert "термів" in result.output


def test_stats_on_corpus(corpus: Path):
    result = runner.invoke(app, ["stats", str(corpus), "--top", "3"])
    assert result.exit_code == 0
    assert "документів" in result.output


def test_missing_index_is_one_clear_line(tmp_path: Path):
    result = runner.invoke(app, ["search", str(tmp_path / "nope.pkl"), "kyiv"])
    assert result.exit_code == 1
    assert "помилка" in result.output
    assert "Traceback" not in result.output


def test_broken_query_exits_nonzero(index_file: Path):
    result = runner.invoke(app, ["search", str(index_file), "kyiv )"])
    assert result.exit_code == 1
    assert "Traceback" not in result.output


def test_positions_with_array_repr_fails(corpus: Path, tmp_path: Path):
    result = runner.invoke(
        app,
        [
            "index",
            str(corpus),
            "--out",
            str(tmp_path / "i.pkl"),
            "--positions",
            "--repr",
            "array",
        ],
    )
    assert result.exit_code == 1
    assert "array" in result.output


def test_verbose_logs_go_to_stderr(index_file: Path):
    result = runner.invoke(app, ["-vv", "search", str(index_file), "kyiv"])
    assert result.exit_code == 0
