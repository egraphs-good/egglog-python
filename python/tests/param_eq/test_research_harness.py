from __future__ import annotations

import csv
import hashlib
import json
import os
import subprocess
import tomllib
from pathlib import Path
from unittest.mock import Mock

import pytest
from experiments.param_eq import run as param_eq_run
from experiments.param_eq.aggregate import (
    COMPARISON_COLUMNS,
    PAPER_COLUMNS,
    build_paper_replication,
    build_representation_comparison,
    generate_aggregates,
    load_raw,
)
from experiments.param_eq.corpus import (
    RAW_ALGORITHMS,
    ArchiveLayoutError,
    CorpusRow,
    external_archive_hash,
    load_corpus_rows,
)
from experiments.param_eq.run import RAW_COLUMNS, _build_haskell_program, _parse_haskell_output

from egglog.exp.param_eq import PaperPipelineReport


@pytest.fixture(autouse=True)
def _stable_repository_state(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("experiments.param_eq.aggregate._repo_state", lambda: ("d" * 40, True))


def _archive(root: Path) -> Path:
    results = root / "results"
    (results / "exprs").mkdir(parents=True)
    (results / "exprs_simpl").mkdir()
    for dataset in ("pagie", "kotanchek"):
        for algorithm in RAW_ALGORITHMS:
            payload = "2.3*x0\n9.9*x0\n3.7*x0\n" if dataset == "pagie" and algorithm == "Bingo" else ""
            (results / "exprs" / f"{algorithm}_exprs_{dataset}").write_text(payload, encoding="utf-8")
            (results / "exprs_simpl" / f"{algorithm}_exprs_{dataset}").write_text(payload, encoding="utf-8")
        (results / f"{dataset}_results").write_text(
            "algorithm,expr,expr_sympy\n"
            + ("Bingo,2.3*x0,2.3*x0\nBingo,9.9*x0,9.9*x0\nBingo,3.7*x0,3.7*x0\n" if dataset == "pagie" else ""),
            encoding="utf-8",
        )
        (results / f"{dataset}_table_counts.csv").write_text(
            (
                ",orig_nodes,orig_params,simpl_nodes,simpl_params,orig_nodes_sympy,orig_params_sympy,"
                "simpl_nodes_sympy,simpl_params_sympy,algorithm,n_params,n_rank\n"
                + (
                    "0,3,1,3,1,3,1,3,1,Bingo,1,1\n2,3,1,3,1,3,1,3,1,Bingo,1,\n3,3,1,3,1,3,1,3,1,Bingo,1,1\n"
                    if dataset == "pagie"
                    else ""
                )
            ),
            encoding="utf-8",
        )
    return root


def _raw_row(row_id: str, variant: str, *, status: str = "saturated") -> dict[str, str]:
    values = {
        "row_id": row_id,
        "dataset": "pagie",
        "algorithm": "Bingo",
        "input_kind": "original",
        "implementation": "egglog",
        "variant": variant,
        "external_archive_sha256": "a" * 64,
        "source_n_rank": "1",
        "timeout_sec": "60",
        "memory_limit_mb": "2048",
        "sample_interval_sec": "0.2",
        "execution_mode": "release",
        "workers_requested": "2",
        "workers_effective": "2",
        "ordering_seed": "param-eq-variant-order-v1",
        "egglog_python_commit": "d" * 40,
        "egglog_python_clean": "true",
        "egglog_core_commit": "b" * 40,
        "egglog_core_clean": "true",
        "egglog_experimental_commit": "c" * 40,
        "egglog_experimental_clean": "true",
        "egglog_bindings_sha256": "e" * 64,
        "python_version": "3.13.7",
        "platform": "test-platform",
        "rust_version": "rustc 1.91.0",
        "cpu": "test-cpu",
        "status": status,
        "runtime_ms": "2.0" if variant == "binary" else "1.0",
        "peak_rss_mb": "10.0",
        "passes": "2",
        "total_size": "20" if variant == "binary" else "10",
        "before_nodes": "11",
        "before_params": "4",
        "after_nodes": "7",
        "after_params": "2",
    }
    if status != "saturated":
        for key in (
            "runtime_ms",
            "passes",
            "total_size",
            "before_nodes",
            "before_params",
            "after_nodes",
            "after_params",
        ):
            values[key] = ""
    return values


def _write_raw(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=RAW_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


class _ExitOnUnpickle:
    """Make a spawned worker hard-exit while restoring its process target."""

    def __reduce__(self) -> tuple[object, tuple[int]]:
        return os._exit, (17,)


def test_haskell_program_forces_results_and_parser_keeps_expression_text_out() -> None:
    row = CorpusRow(
        row_id="pagie/0/Bingo/1/original",
        dataset="pagie",
        raw_index=0,
        algorithm_raw="Bingo",
        algorithm="Bingo",
        algorithm_row=1,
        input_kind="original",
        source="2.3*x0",
        source_n_rank=1.0,
    )
    program = _build_haskell_program()
    assert "beforeNodes <- evaluate (countNodes expr)" in program
    assert "beforeParams <- evaluate (recountParams (replaceConstsWithParams expr))" in program
    assert "afterNodes <- evaluate (countNodes simplified)" in program
    assert "afterParams <- evaluate (recountParams (replaceConstsWithParams simplified))" in program
    assert row.source not in program
    assert "[dataset, inputKind, algorithm, rowIndex]" in program
    assert "lookupExpr dataset inputKind algorithm (read rowIndex)" in program

    parsed = _parse_haskell_output("11\t4\t7\t2\t1.25\n")
    assert parsed == {
        "status": "saturated",
        "runtime_ms": 1.25,
        "passes": "",
        "total_size": "",
        "before_nodes": 11.0,
        "before_params": 4.0,
        "after_nodes": 7.0,
        "after_params": 2.0,
    }
    with pytest.raises(ValueError, match="five Haskell output fields"):
        _parse_haskell_output("11\t4\n")
    with pytest.raises(ValueError, match="invalid counts or runtime"):
        _parse_haskell_output("11\t4\t7\t2\tnan\n")


def test_haskell_results_reuse_the_expression_free_raw_schema(tmp_path: Path) -> None:
    path = tmp_path / "haskell.csv"
    row = _raw_row("pagie/0/Bingo/1/original", "binary")
    row["implementation"] = "haskell"
    row["passes"] = row["total_size"] = ""
    _write_raw(path, [row])

    assert load_raw(path, expected_variant="binary", expected_implementation="haskell") == [row]
    with pytest.raises(ValueError, match="No egglog/binary rows"):
        load_raw(path, expected_variant="binary")


def test_iteration_limited_worker_result_has_no_publishable_metrics(monkeypatch: pytest.MonkeyPatch) -> None:
    report = PaperPipelineReport(
        status="iteration_limit",
        passes=1,
        total_sec=0.1,
        total_size=10,
        before_nodes=3,
        before_params=1,
        extracted="x0",
        extracted_nodes=1,
        extracted_params=0,
    )
    monkeypatch.setattr("egglog.exp.param_eq.run_paper_pipeline", lambda _expr: report)
    connection = Mock()

    param_eq_run._worker(connection, "x0", "binary")

    connection.send.assert_called_once_with({"status": "iteration_limit"})
    connection.close.assert_called_once_with()


def test_run_rows_records_error_when_worker_exits_without_sending(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    row = CorpusRow(
        row_id="pagie/0/Bingo/1/original",
        dataset="pagie",
        raw_index=0,
        algorithm_raw="Bingo",
        algorithm="Bingo",
        algorithm_row=1,
        input_kind="original",
        source="2.3*x0",
        source_n_rank=1.0,
    )
    expected = _raw_row(row.row_id, "binary", status="error")
    provenance = {
        column: expected[column]
        for column in RAW_COLUMNS[RAW_COLUMNS.index("execution_mode") : RAW_COLUMNS.index("status")]
    }
    monkeypatch.setattr(param_eq_run, "_worker", _ExitOnUnpickle())

    results = param_eq_run.run_rows(
        [row],
        implementation="egglog",
        variant="binary",
        archive_root=tmp_path,
        haskell_executable=None,
        external_archive_sha256="a" * 64,
        workers=1,
        timeout_sec=5.0,
        memory_limit_mb=2048,
        sample_interval_sec=0.01,
        provenance=provenance,
    )

    assert len(results) == 1
    assert set(results[0]) == set(RAW_COLUMNS)
    assert results[0]["status"] == "error"
    assert all(
        results[0][column] == ""
        for column in (
            "runtime_ms",
            "passes",
            "total_size",
            "before_nodes",
            "before_params",
            "after_nodes",
            "after_params",
        )
    )


def test_run_provenance_hashes_the_loaded_native_extension(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    repository = tmp_path / "egglog-python"
    core = tmp_path / "egglog"
    experimental = tmp_path / "egglog-experimental"
    artifact = tmp_path / "bindings.so"
    artifact.write_bytes(b"loaded native extension")
    metadata = {
        "packages": [
            {"name": "egglog", "manifest_path": str(core / "Cargo.toml")},
            {"name": "egglog-experimental", "manifest_path": str(experimental / "Cargo.toml")},
        ]
    }

    def run_metadata(command: list[str], **_kwargs: object) -> subprocess.CompletedProcess[str]:
        if command[:2] == ["cargo", "metadata"]:
            return subprocess.CompletedProcess(command, 0, json.dumps(metadata), "")
        if command[:3] == ["git", "rev-parse", "HEAD"]:
            return subprocess.CompletedProcess(command, 0, "d" * 40 + "\n", "")
        if command[:3] == ["git", "status", "--porcelain"]:
            return subprocess.CompletedProcess(command, 0, "", "")
        if command[:2] == ["rustc", "--version"]:
            return subprocess.CompletedProcess(command, 0, "rustc test\n", "")
        if command[:2] == ["sysctl", "-n"]:
            return subprocess.CompletedProcess(command, 0, "test cpu\n", "")
        raise AssertionError(f"Unexpected command: {command}")

    monkeypatch.setattr(param_eq_run, "REPO_ROOT", repository)
    monkeypatch.setattr(param_eq_run.egglog_bindings, "__file__", str(artifact))
    monkeypatch.setattr(param_eq_run.subprocess, "run", run_metadata)
    monkeypatch.setattr(param_eq_run.platform, "processor", lambda: "test cpu")
    monkeypatch.setattr(param_eq_run.platform, "system", lambda: "Test")
    monkeypatch.setattr(param_eq_run.platform, "platform", lambda: "test platform")

    provenance = param_eq_run._collect_run_provenance(
        execution_mode="release", workers_requested=2, workers_effective=1
    )

    assert provenance["egglog_bindings_sha256"] == hashlib.sha256(artifact.read_bytes()).hexdigest()


def test_external_loader_is_explicit_and_expression_data_stays_in_memory(tmp_path: Path) -> None:
    with pytest.raises(ArchiveLayoutError, match="incomplete"):
        load_corpus_rows(tmp_path)
    root = _archive(tmp_path)
    rows = load_corpus_rows(root)
    assert [(row.row_id, row.input_kind) for row in rows] == [
        ("pagie/0/Bingo/1/original", "original"),
        ("pagie/0/Bingo/1/sympy", "sympy"),
        ("pagie/3/Bingo/3/original", "original"),
        ("pagie/3/Bingo/3/sympy", "sympy"),
    ]
    assert [row.source for row in rows] == ["2.3*x0", "2.3*x0", "3.7*x0", "3.7*x0"]


def test_external_hash_is_stable_and_content_sensitive(tmp_path: Path) -> None:
    root = _archive(tmp_path)
    first = external_archive_hash(root)
    assert external_archive_hash(root) == first
    target = root / "results" / "pagie_results"
    target.write_text(target.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    assert external_archive_hash(root) != first


def test_aggregate_outputs_have_only_aggregate_schemas(tmp_path: Path) -> None:
    binary = tmp_path / "binary.csv"
    container = tmp_path / "container.csv"
    output = tmp_path / "published"
    _write_raw(binary, [_raw_row("pagie/0/Bingo/1/original", "binary")])
    _write_raw(container, [_raw_row("pagie/0/Bingo/1/original", "container")])
    generate_aggregates(binary, container, output)
    with (output / "paper-replication.csv").open(newline="", encoding="utf-8") as handle:
        assert tuple(csv.DictReader(handle).fieldnames or ()) == PAPER_COLUMNS
    with (output / "representation-comparison.csv").open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        assert tuple(reader.fieldnames or ()) == COMPARISON_COLUMNS
        all_runtime = next(row for row in reader if row["slice"] == "all" and row["metric"] == "runtime_ms")
    assert all_runtime["container_better"] == "1"
    combined = "\n".join(path.read_text(encoding="utf-8") for path in output.iterdir())
    assert str(tmp_path) not in combined
    assert "expression" not in combined
    assert "external_archive_sha256" not in combined
    assert "a" * 64 not in combined


def test_paired_raw_file_drives_the_exact_aggregate_path(tmp_path: Path) -> None:
    paired = tmp_path / "paired.csv"
    output = tmp_path / "published"
    _write_raw(
        paired,
        [
            _raw_row("pagie/0/Bingo/1/original", "binary"),
            _raw_row("pagie/0/Bingo/1/original", "container"),
        ],
    )

    generate_aggregates(paired, paired, output)

    manifest = tomllib.loads((output / "manifest.toml").read_text(encoding="utf-8"))
    assert manifest["schema_version"] == 5
    assert manifest["egglog_bindings_sha256"] == "e" * 64
    assert manifest["python"] == "3.13.7"
    assert manifest["platform"] == "test-platform"
    assert manifest["raw_layout"] == "paired-single-file"
    assert manifest["binary_raw_sha256"] == manifest["container_raw_sha256"]
    assert manifest["binary_rows"] == manifest["container_rows"] == 1
    assert manifest["binary_status_saturated"] == manifest["container_status_saturated"] == 1
    assert manifest["recommended_full_rerun"] == "make -C experiments/param_eq aggregate"
    assert "command" not in manifest


def test_paper_aggregate_keeps_failure_statuses_distinct() -> None:
    statuses = ("saturated", "iteration_limit", "timeout", "memory_limit", "error")
    rows = [
        _raw_row(f"pagie/{index}/Bingo/{index + 1}/original", "binary", status=status)
        for index, status in enumerate(statuses)
    ]

    aggregate = next(row for row in build_paper_replication(rows) if row["metric"] == "final_params")

    assert aggregate["n_total"] == 5
    assert aggregate["n_success"] == 1
    assert aggregate["n_iteration_limit"] == 1
    assert aggregate["n_timeout"] == 1
    assert aggregate["n_memory_limit"] == 1
    assert aggregate["n_error"] == 1


def test_ratio_summary_discloses_zero_denominator_omissions() -> None:
    binary = _raw_row("pagie/0/Bingo/1/original", "binary")
    container = _raw_row("pagie/0/Bingo/1/original", "container")
    binary["after_params"] = "0"
    container["after_params"] = "1"

    comparison = build_representation_comparison([binary], [container])
    aggregate = next(row for row in comparison if row["slice"] == "all" and row["metric"] == "after_params")

    assert aggregate["n_pairs"] == 1
    assert aggregate["n_ratio"] == 0
    assert aggregate["container_worse"] == 1
    assert aggregate["ratio_median"] == ""


def test_aggregate_refuses_mismatched_producer_environment(tmp_path: Path) -> None:
    binary_row = _raw_row("pagie/0/Bingo/1/original", "binary")
    container_row = _raw_row("pagie/0/Bingo/1/original", "container")
    container_row["platform"] = "other-platform"
    binary = tmp_path / "binary.csv"
    container = tmp_path / "container.csv"
    _write_raw(binary, [binary_row])
    _write_raw(container, [container_row])

    with pytest.raises(ValueError, match="nonempty platform"):
        generate_aggregates(binary, container, tmp_path / "published")


@pytest.mark.parametrize(
    ("column", "value", "message"),
    [
        ("egglog_python_clean", "false", "dirty repository state"),
        ("egglog_python_commit", "abcdef0", "invalid egglog_python_commit"),
        ("egglog_core_clean", "false", "dirty repository state"),
        ("egglog_core_commit", "unknown", "invalid egglog_core_commit"),
        ("egglog_bindings_sha256", "not-a-hash", "invalid egglog_bindings_sha256"),
    ],
)
def test_aggregate_refuses_unpublishable_dependency_provenance(
    tmp_path: Path, column: str, value: str, message: str
) -> None:
    binary_row = _raw_row("pagie/0/Bingo/1/original", "binary")
    container_row = _raw_row("pagie/0/Bingo/1/original", "container")
    binary_row[column] = value
    container_row[column] = value
    binary = tmp_path / "binary.csv"
    container = tmp_path / "container.csv"
    _write_raw(binary, [binary_row])
    _write_raw(container, [container_row])
    with pytest.raises(ValueError, match=message):
        generate_aggregates(binary, container, tmp_path / "published")


def test_aggregate_refuses_a_dirty_egglog_python_worktree(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    binary = tmp_path / "binary.csv"
    container = tmp_path / "container.csv"
    output = tmp_path / "published"
    _write_raw(binary, [_raw_row("pagie/0/Bingo/1/original", "binary")])
    _write_raw(container, [_raw_row("pagie/0/Bingo/1/original", "container")])
    monkeypatch.setattr("experiments.param_eq.aggregate._repo_state", lambda: ("d" * 40, False))

    with pytest.raises(ValueError, match="dirty egglog-python worktree"):
        generate_aggregates(binary, container, output)
    assert not output.exists()


def test_aggregate_refuses_raw_from_a_different_egglog_python_commit(tmp_path: Path) -> None:
    binary_row = _raw_row("pagie/0/Bingo/1/original", "binary")
    container_row = _raw_row("pagie/0/Bingo/1/original", "container")
    binary_row["egglog_python_commit"] = "e" * 40
    container_row["egglog_python_commit"] = "e" * 40
    binary = tmp_path / "binary.csv"
    container = tmp_path / "container.csv"
    output = tmp_path / "published"
    _write_raw(binary, [binary_row])
    _write_raw(container, [container_row])

    with pytest.raises(ValueError, match="produced by a different egglog-python commit"):
        generate_aggregates(binary, container, output)
    assert not output.exists()


@pytest.mark.parametrize(
    ("status", "column", "value", "message"),
    [
        ("saturated", "runtime_ms", "nan", "invalid runtime_ms"),
        ("saturated", "after_nodes", "-1", "invalid after_nodes"),
        ("saturated", "before_params", "1.5", "noninteger before_params"),
        ("saturated", "passes", "0", "no completed passes"),
        ("timeout", "runtime_ms", "1", "unexpectedly has result metrics"),
    ],
)
def test_aggregate_rejects_invalid_or_status_inconsistent_metrics(
    tmp_path: Path, status: str, column: str, value: str, message: str
) -> None:
    binary_row = _raw_row("pagie/0/Bingo/1/original", "binary", status=status)
    container_row = _raw_row("pagie/0/Bingo/1/original", "container", status=status)
    binary_row[column] = value
    container_row[column] = value
    binary = tmp_path / "binary.csv"
    container = tmp_path / "container.csv"
    output = tmp_path / "published"
    _write_raw(binary, [binary_row])
    _write_raw(container, [container_row])

    with pytest.raises(ValueError, match=message):
        generate_aggregates(binary, container, output)
    assert not output.exists()


def test_aggregate_validates_all_publication_inputs_before_writing(tmp_path: Path) -> None:
    binary_row = _raw_row("pagie/0/Bingo/1/original", "binary")
    container_row = _raw_row("pagie/0/Bingo/1/original", "container")
    binary_row["external_archive_sha256"] = "not-a-hash"
    container_row["external_archive_sha256"] = "not-a-hash"
    binary = tmp_path / "binary.csv"
    container = tmp_path / "container.csv"
    output = tmp_path / "published"
    _write_raw(binary, [binary_row])
    _write_raw(container, [container_row])

    with pytest.raises(ValueError, match="external_archive_sha256"):
        generate_aggregates(binary, container, output)
    assert not output.exists()


def test_aggregate_rejects_private_algorithm_label_without_leaking_or_writing(tmp_path: Path) -> None:
    sentinel = "PRIVATE_EXPR_SENTINEL(x0)"
    binary_row = _raw_row("pagie/0/Bingo/1/original", "binary")
    container_row = _raw_row("pagie/0/Bingo/1/original", "container")
    binary_row["algorithm"] = sentinel
    container_row["algorithm"] = sentinel
    binary = tmp_path / "binary.csv"
    container = tmp_path / "container.csv"
    output = tmp_path / "published"
    _write_raw(binary, [binary_row])
    _write_raw(container, [container_row])

    with pytest.raises(ValueError, match="invalid algorithm label") as error:
        generate_aggregates(binary, container, output)

    assert sentinel not in str(error.value)
    assert not output.exists()


def test_write_local_raw_requires_an_untracked_ignored_target(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    repo = tmp_path / "repo"
    raw_results = repo / "results" / "raw"
    raw_results.mkdir(parents=True)
    ignore_file = raw_results / ".gitignore"
    ignore_contents = "*\n!.gitignore\n"
    ignore_file.write_text(ignore_contents, encoding="utf-8")
    subprocess.run(["git", "init", "--quiet"], cwd=repo, check=True)  # noqa: S607
    subprocess.run(
        ["git", "add", "--", "results/raw/.gitignore"],  # noqa: S607
        cwd=repo,
        check=True,
    )
    monkeypatch.setattr(param_eq_run, "REPO_ROOT", repo)
    monkeypatch.setattr(param_eq_run, "RAW_RESULTS_DIR", raw_results)
    row = _raw_row("pagie/0/Bingo/1/original", "binary")

    with pytest.raises(ValueError, match="untracked, Git-ignored"):
        param_eq_run.write_local_raw([row], ignore_file)
    assert ignore_file.read_text(encoding="utf-8") == ignore_contents

    output = raw_results / "binary.csv"
    param_eq_run.write_local_raw([row], output)
    assert load_raw(output, expected_variant="binary") == [row]


@pytest.mark.parametrize(
    ("expected_hash", "message"),
    [
        (None, "Set EGGLOG_PARAM_EQ_EXPECTED_ARCHIVE_SHA256"),
        ("b" * 64, "does not match EGGLOG_PARAM_EQ_EXPECTED_ARCHIVE_SHA256"),
    ],
)
def test_runner_requires_the_privately_recorded_archive_hash(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    expected_hash: str | None,
    message: str,
) -> None:
    monkeypatch.setattr(param_eq_run, "external_archive_root", lambda: tmp_path)
    monkeypatch.setattr(param_eq_run, "load_corpus_rows", lambda *args, **kwargs: [])
    monkeypatch.setattr(param_eq_run, "external_archive_hash", lambda root: "a" * 64)
    if expected_hash is None:
        monkeypatch.delenv("EGGLOG_PARAM_EQ_EXPECTED_ARCHIVE_SHA256", raising=False)
    else:
        monkeypatch.setenv("EGGLOG_PARAM_EQ_EXPECTED_ARCHIVE_SHA256", expected_hash)

    with pytest.raises(SystemExit, match="2"):
        param_eq_run.main(["--execution-mode", "debug"])

    stderr = capsys.readouterr().err
    assert message in stderr
    assert "a" * 64 not in stderr


def test_runner_rejects_filters_that_match_no_rows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(param_eq_run, "external_archive_root", lambda: tmp_path)
    monkeypatch.setattr(param_eq_run, "load_corpus_rows", lambda *args, **kwargs: [])
    monkeypatch.setattr(param_eq_run, "external_archive_hash", lambda root: "a" * 64)
    monkeypatch.setenv("EGGLOG_PARAM_EQ_EXPECTED_ARCHIVE_SHA256", "a" * 64)

    with pytest.raises(SystemExit, match="2"):
        param_eq_run.main(["--execution-mode", "debug", "--algorithm", "typo"])

    assert "selected corpus filters matched no rows" in capsys.readouterr().err


def test_raw_schema_rejects_expression_columns_and_unpaired_rows(tmp_path: Path) -> None:
    bad = tmp_path / "bad.csv"
    bad.write_text(",".join([*RAW_COLUMNS, "expression"]) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="forbidden columns"):
        load_raw(bad, expected_variant="binary")
    with pytest.raises(ValueError, match="Unpaired raw inputs"):
        build_representation_comparison(
            [_raw_row("pagie/0/Bingo/1/original", "binary")],
            [_raw_row("pagie/3/Bingo/3/original", "container")],
        )
