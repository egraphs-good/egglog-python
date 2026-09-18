"""Validate local row results and publish expression-free aggregate artifacts."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import subprocess
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from experiments.param_eq.corpus import ALGORITHM_RENAMES, DATASETS, RAW_ALGORITHMS
from experiments.param_eq.run import RAW_COLUMNS, VARIANT_ORDER_SEED

PAPER_COLUMNS = (
    "implementation",
    "dataset",
    "algorithm",
    "input_kind",
    "metric",
    "n_total",
    "n_success",
    "n_iteration_limit",
    "n_timeout",
    "n_memory_limit",
    "n_error",
    "value",
    "min",
    "q1",
    "median",
    "q3",
    "max",
)
COMPARISON_COLUMNS = (
    "slice",
    "metric",
    "n_pairs",
    "n_ratio",
    "n_binary_missing",
    "n_container_missing",
    "container_better",
    "same",
    "container_worse",
    "ratio_p10",
    "ratio_p25",
    "ratio_median",
    "ratio_p75",
    "ratio_p90",
)
FORBIDDEN_COLUMNS = {"source", "expression", "rendered", "orig_expr", "simpl_expr", "private_path"}
RAW_STATUSES = {"saturated", "iteration_limit", "timeout", "memory_limit", "error"}
PROVENANCE_COLUMNS = (
    "execution_mode",
    "workers_requested",
    "workers_effective",
    "ordering_seed",
    "egglog_python_commit",
    "egglog_python_clean",
    "egglog_core_commit",
    "egglog_core_clean",
    "egglog_experimental_commit",
    "egglog_experimental_clean",
    "egglog_bindings_sha256",
    "python_version",
    "platform",
    "rust_version",
    "cpu",
)
PUBLIC_ALGORITHMS = frozenset(ALGORITHM_RENAMES.get(name, name) for name in RAW_ALGORITHMS if name != "FEAT")
INPUT_KINDS = frozenset({"original", "sympy"})


def _validate_public_identity(row: dict[str, str], filename: str) -> None:
    """Reject labels that could carry private data into aggregate artifacts."""
    if row["dataset"] not in DATASETS:
        raise ValueError(f"Raw row in {filename} has an invalid dataset label")
    if row["algorithm"] not in PUBLIC_ALGORITHMS:
        raise ValueError(f"Raw row in {filename} has an invalid algorithm label")
    if row["input_kind"] not in INPUT_KINDS:
        raise ValueError(f"Raw row in {filename} has an invalid input-kind label")
    identity = row["row_id"].split("/")
    if (
        len(identity) != 5
        or identity[0] != row["dataset"]
        or identity[2] != row["algorithm"]
        or identity[4] != row["input_kind"]
        or not identity[1].isdecimal()
        or not identity[3].isdecimal()
        or int(identity[3]) < 1
    ):
        raise ValueError(f"Raw row in {filename} has an invalid or inconsistent row identity")


def _number(row: dict[str, str], key: str) -> float:
    """Return one required finite, nonnegative raw measurement."""
    value = row[key]
    if value == "":
        raise ValueError(f"Raw row {row['row_id']} has no {key}")
    try:
        number = float(value)
    except ValueError:
        raise ValueError(f"Raw row {row['row_id']} has an invalid {key}") from None
    if not math.isfinite(number) or number < 0.0:
        raise ValueError(f"Raw row {row['row_id']} has an invalid {key}")
    return number


def _validate_result_measurements(row: dict[str, str], filename: str) -> None:
    """Enforce the result fields allowed by a row's status and implementation."""
    result_fields = (
        "runtime_ms",
        "passes",
        "total_size",
        "before_nodes",
        "before_params",
        "after_nodes",
        "after_params",
    )
    if row["status"] == "saturated":
        required = ("runtime_ms", "before_nodes", "before_params", "after_nodes", "after_params")
        if row["implementation"] == "egglog":
            required += ("passes", "total_size")
        elif row["passes"] or row["total_size"]:
            raise ValueError(f"Haskell row in {filename} unexpectedly has Egglog-only metrics")
        for key in required:
            number = _number(row, key)
            if key != "runtime_ms" and not number.is_integer():
                raise ValueError(f"Raw row in {filename} has a noninteger {key}")
            if key == "passes" and number == 0.0:
                raise ValueError(f"Raw row in {filename} has no completed passes")
    elif any(row[key] for key in result_fields):
        raise ValueError(f"Unsuccessful row in {filename} unexpectedly has result metrics")


def _validate_measurements(row: dict[str, str], filename: str) -> None:
    """Enforce numeric and status-dependent invariants for one raw row."""
    source_rank = _number(row, "source_n_rank")
    timeout = _number(row, "timeout_sec")
    memory_limit = _number(row, "memory_limit_mb")
    sample_interval = _number(row, "sample_interval_sec")
    if timeout == 0.0 or sample_interval == 0.0:
        raise ValueError(f"Raw row in {filename} has an invalid timeout/sample configuration")
    if memory_limit == 0.0 or not memory_limit.is_integer():
        raise ValueError(f"Raw row in {filename} has an invalid memory limit")
    if not source_rank.is_integer():
        raise ValueError(f"Raw row in {filename} has a noninteger source_n_rank")
    _validate_result_measurements(row, filename)
    if row["peak_rss_mb"]:
        _number(row, "peak_rss_mb")


def load_raw(path: Path, *, expected_variant: str, expected_implementation: str = "egglog") -> list[dict[str, str]]:
    """Load a local raw result and reject provenance/schema mismatches."""
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fields = tuple(reader.fieldnames or ())
        if fields != RAW_COLUMNS:
            extras = FORBIDDEN_COLUMNS.intersection(fields)
            detail = f" forbidden columns={sorted(extras)}" if extras else ""
            raise ValueError(f"Unexpected raw-result schema in {path.name}.{detail}")
        all_rows = list(reader)
    variants = {row["variant"] for row in all_rows}
    if not variants.issubset({"binary", "container"}):
        raise ValueError(f"Unknown variants in {path.name}: {sorted(variants)}")
    implementations = {row["implementation"] for row in all_rows}
    if not implementations.issubset({"egglog", "haskell"}):
        raise ValueError(f"Unknown implementations in {path.name}: {sorted(implementations)}")
    statuses = {row["status"] for row in all_rows}
    if not statuses.issubset(RAW_STATUSES):
        raise ValueError(f"Unknown statuses in {path.name}: {sorted(statuses)}")
    for row in all_rows:
        _validate_public_identity(row, path.name)
        _validate_measurements(row, path.name)
    rows = [
        row
        for row in all_rows
        if row["variant"] == expected_variant and row["implementation"] == expected_implementation
    ]
    if not rows:
        raise ValueError(f"No {expected_implementation}/{expected_variant} rows in {path.name}")
    seen: set[str] = set()
    for row in rows:
        if row["row_id"] in seen:
            raise ValueError(f"Duplicate row identity in {path.name}: {row['row_id']}")
        seen.add(row["row_id"])
    return rows


def _validated_provenance(rows: list[dict[str, str]]) -> dict[str, str]:
    """Require one complete, clean dependency/runtime configuration."""
    provenance: dict[str, str] = {}
    for column in PROVENANCE_COLUMNS:
        values = {row[column] for row in rows}
        if len(values) != 1 or not next(iter(values)):
            raise ValueError(f"Raw rows do not share one nonempty {column}")
        provenance[column] = next(iter(values))
    for column, length in (
        ("egglog_python_commit", 40),
        ("egglog_core_commit", 40),
        ("egglog_experimental_commit", 40),
        ("egglog_bindings_sha256", 64),
    ):
        identity = provenance[column]
        if len(identity) != length or any(character not in "0123456789abcdef" for character in identity.lower()):
            raise ValueError(f"Raw rows have an invalid {column}")
    for column in ("egglog_python_clean", "egglog_core_clean", "egglog_experimental_clean"):
        if provenance[column] != "true":
            raise ValueError(f"Refusing aggregate publication with dirty repository state: {column}")
    if provenance["execution_mode"] not in {"debug", "release"}:
        msg = "Raw rows have an invalid execution_mode"
        raise ValueError(msg)
    requested = int(provenance["workers_requested"])
    effective = int(provenance["workers_effective"])
    if requested < 1 or effective < 1 or effective > requested:
        msg = "Raw rows have invalid worker counts"
        raise ValueError(msg)
    if provenance["ordering_seed"] != VARIANT_ORDER_SEED:
        msg = "Raw rows use an unknown ordering seed"
        raise ValueError(msg)
    return provenance


def _quantile(values: Iterable[float], probability: float) -> float | str:
    ordered = sorted(values)
    if not ordered:
        return ""
    position = (len(ordered) - 1) * probability
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _slice_predicates() -> dict[str, Callable[[dict[str, str]], bool]]:
    return {
        "all": lambda _row: True,
        "original": lambda row: row["input_kind"] == "original",
        "sympy": lambda row: row["input_kind"] == "sympy",
        "pagie": lambda row: row["dataset"] == "pagie",
        "kotanchek": lambda row: row["dataset"] == "kotanchek",
    }


def build_representation_comparison(  # noqa: C901
    binary_rows: list[dict[str, str]], container_rows: list[dict[str, str]]
) -> list[dict[str, Any]]:
    """Build paired lower-is-better outcome and ratio summaries."""
    binary = {row["row_id"]: row for row in binary_rows}
    container = {row["row_id"]: row for row in container_rows}
    if set(binary) != set(container):
        missing_binary = sorted(set(container) - set(binary))
        missing_container = sorted(set(binary) - set(container))
        raise ValueError(
            f"Unpaired raw inputs: missing_binary={len(missing_binary)} missing_container={len(missing_container)}"
        )
    archive_hashes = {row["external_archive_sha256"] for row in [*binary_rows, *container_rows]}
    if len(archive_hashes) != 1:
        msg = "Binary and container rows do not share one external-input hash"
        raise ValueError(msg)
    for row_id, left in binary.items():
        right = container[row_id]
        for key in (
            "dataset",
            "algorithm",
            "input_kind",
            "source_n_rank",
            "timeout_sec",
            "memory_limit_mb",
            "sample_interval_sec",
            *PROVENANCE_COLUMNS,
        ):
            if left[key] != right[key]:
                raise ValueError(f"Paired metadata mismatch for {row_id}: {key}")

    output: list[dict[str, Any]] = []
    metrics = ("after_params", "after_nodes", "runtime_ms", "total_size")
    for slice_name, predicate in _slice_predicates().items():
        identities = [row_id for row_id, row in binary.items() if predicate(row)]
        for metric in metrics:
            better = same = worse = 0
            ratios: list[float] = []
            binary_missing = container_missing = 0
            for row_id in identities:
                left, right = binary[row_id], container[row_id]
                left_ok = left["status"] == "saturated"
                right_ok = right["status"] == "saturated"
                binary_missing += int(not left_ok)
                container_missing += int(not right_ok)
                if not (left_ok and right_ok):
                    continue
                binary_value = _number(left, metric)
                container_value = _number(right, metric)
                better += int(container_value < binary_value)
                same += int(container_value == binary_value)
                worse += int(container_value > binary_value)
                if binary_value != 0.0:
                    ratios.append(container_value / binary_value)
            output.append({
                "slice": slice_name,
                "metric": metric,
                "n_pairs": len(identities),
                "n_ratio": len(ratios),
                "n_binary_missing": binary_missing,
                "n_container_missing": container_missing,
                "container_better": better,
                "same": same,
                "container_worse": worse,
                "ratio_p10": _quantile(ratios, 0.10),
                "ratio_p25": _quantile(ratios, 0.25),
                "ratio_median": _quantile(ratios, 0.50),
                "ratio_p75": _quantile(ratios, 0.75),
                "ratio_p90": _quantile(ratios, 0.90),
            })
    return output


def build_paper_replication(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    """Build aggregate Egglog rows without source identities or expressions."""
    groups: defaultdict[tuple[str, str, str, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        groups[(row["implementation"], row["variant"], row["dataset"], row["algorithm"], row["input_kind"])].append(row)
    output: list[dict[str, Any]] = []
    for (implementation, variant, dataset, algorithm, input_kind), group in sorted(groups.items()):
        successes = [row for row in group if row["status"] == "saturated"]
        status_counts: defaultdict[str, int] = defaultdict(int)
        for row in group:
            status_counts[row["status"]] += 1
        metrics = {
            "final_params": [_number(row, "after_params") for row in successes],
            "parameter_reduction": [_number(row, "before_params") - _number(row, "after_params") for row in successes],
            "parameter_rank_gap": [_number(row, "after_params") - _number(row, "source_n_rank") for row in successes],
        }
        for metric, values in metrics.items():
            output.append({
                "implementation": f"egglog-{variant}" if implementation == "egglog" else implementation,
                "dataset": dataset,
                "algorithm": algorithm,
                "input_kind": input_kind,
                "metric": metric,
                "n_total": len(group),
                "n_success": len(successes),
                "n_iteration_limit": status_counts["iteration_limit"],
                "n_timeout": status_counts["timeout"],
                "n_memory_limit": status_counts["memory_limit"],
                "n_error": status_counts["error"],
                "value": "",
                "min": min(values) if values else "",
                "q1": _quantile(values, 0.25),
                "median": _quantile(values, 0.50),
                "q3": _quantile(values, 0.75),
                "max": max(values) if values else "",
            })
    return output


def _write_csv(path: Path, columns: tuple[str, ...], rows: Iterable[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def _repo_state() -> tuple[str, bool]:
    root = Path(__file__).resolve().parents[2]
    commit_process = subprocess.run(
        ["git", "rev-parse", "HEAD"],  # noqa: S607
        cwd=root,
        capture_output=True,
        check=False,
        text=True,
    )
    status_process = subprocess.run(
        ["git", "status", "--porcelain"],  # noqa: S607
        cwd=root,
        capture_output=True,
        check=False,
        text=True,
    )
    if commit_process.returncode != 0 or status_process.returncode != 0:
        msg = "Could not resolve the egglog-python Git state"
        raise RuntimeError(msg)
    return commit_process.stdout.strip(), not status_process.stdout.strip()


def generate_aggregates(binary_path: Path, container_path: Path, output_dir: Path) -> None:
    """Validate paired rows and write the only publishable research artifacts."""
    binary_rows = load_raw(binary_path, expected_variant="binary")
    container_rows = load_raw(container_path, expected_variant="container")
    all_rows = [*binary_rows, *container_rows]
    provenance = _validated_provenance(all_rows)
    comparison = build_representation_comparison(binary_rows, container_rows)
    paper = build_paper_replication(all_rows)
    repository_commit, repository_clean = _repo_state()
    if not repository_clean:
        msg = "Refusing aggregate publication from a dirty egglog-python worktree"
        raise ValueError(msg)
    if len(repository_commit) != 40 or any(
        character not in "0123456789abcdef" for character in repository_commit.lower()
    ):
        msg = "Could not identify the egglog-python commit"
        raise ValueError(msg)
    if repository_commit != provenance["egglog_python_commit"]:
        msg = "Refusing aggregate publication from raw rows produced by a different egglog-python commit"
        raise ValueError(msg)
    archive_hashes = {row["external_archive_sha256"] for row in all_rows}
    archive_hash = next(iter(archive_hashes)) if len(archive_hashes) == 1 else "mismatch"
    if len(archive_hash) != 64 or any(character not in "0123456789abcdef" for character in archive_hash.lower()):
        msg = "Raw rows have an invalid external_archive_sha256"
        raise ValueError(msg)
    configurations = {(row["timeout_sec"], row["memory_limit_mb"], row["sample_interval_sec"]) for row in all_rows}
    if len(configurations) != 1:
        msg = "Raw rows do not share one timeout/memory/sample configuration"
        raise ValueError(msg)
    timeout_sec, memory_limit_mb, sample_interval_sec = next(iter(configurations))

    output_dir.mkdir(parents=True, exist_ok=True)
    paper_path = output_dir / "paper-replication.csv"
    comparison_path = output_dir / "representation-comparison.csv"
    _write_csv(paper_path, PAPER_COLUMNS, paper)
    _write_csv(comparison_path, COMPARISON_COLUMNS, comparison)
    binary_statuses = Counter(row["status"] for row in binary_rows)
    container_statuses = Counter(row["status"] for row in container_rows)
    raw_layout = "paired-single-file" if binary_path.resolve() == container_path.resolve() else "split-files"
    manifest = "\n".join([
        "schema_version = 5",
        f"generated_utc = {json.dumps(datetime.now(UTC).isoformat())}",
        f"repository_commit = {json.dumps(repository_commit)}",
        f"repository_clean = {str(repository_clean).lower()}",
        f"egglog_core_commit = {json.dumps(provenance['egglog_core_commit'])}",
        f"egglog_experimental_commit = {json.dumps(provenance['egglog_experimental_commit'])}",
        f"egglog_bindings_sha256 = {json.dumps(provenance['egglog_bindings_sha256'])}",
        f"python = {json.dumps(provenance['python_version'])}",
        f"rust = {json.dumps(provenance['rust_version'])}",
        f"platform = {json.dumps(provenance['platform'])}",
        f"cpu = {json.dumps(provenance['cpu'])}",
        f"execution_mode = {json.dumps(provenance['execution_mode'])}",
        f"workers_requested = {provenance['workers_requested']}",
        f"workers_effective = {provenance['workers_effective']}",
        f"ordering_seed = {json.dumps(provenance['ordering_seed'])}",
        f"raw_layout = {json.dumps(raw_layout)}",
        f"binary_raw_sha256 = {json.dumps(hashlib.sha256(binary_path.read_bytes()).hexdigest())}",
        f"container_raw_sha256 = {json.dumps(hashlib.sha256(container_path.read_bytes()).hexdigest())}",
        f"timeout_sec = {json.dumps(timeout_sec)}",
        f"memory_limit_mb = {json.dumps(memory_limit_mb)}",
        f"sample_interval_sec = {json.dumps(sample_interval_sec)}",
        f"paper_replication_sha256 = {json.dumps(hashlib.sha256(paper_path.read_bytes()).hexdigest())}",
        f"representation_comparison_sha256 = {json.dumps(hashlib.sha256(comparison_path.read_bytes()).hexdigest())}",
        f"binary_rows = {len(binary_rows)}",
        f"container_rows = {len(container_rows)}",
        *[f"binary_status_{status} = {binary_statuses[status]}" for status in sorted(RAW_STATUSES)],
        *[f"container_status_{status} = {container_statuses[status]}" for status in sorted(RAW_STATUSES)],
        'ratio_definition = "container / binary; lower is better; zero binary values omitted; n_ratio counts the remaining ratios"',
        'recommended_full_rerun = "make -C experiments/param_eq aggregate"',
        "",
    ])
    (output_dir / "manifest.toml").write_text(manifest, encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--container", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent / "results")
    args = parser.parse_args(argv)
    generate_aggregates(args.binary, args.container, args.output_dir)
    print(f"wrote aggregate-only artifacts to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
