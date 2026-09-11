"""Run external Param-Eq corpus rows in isolated local workers."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import shutil
import subprocess
import tempfile
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import ExitStack
from multiprocessing import get_context
from multiprocessing.connection import Connection
from pathlib import Path
from typing import Any

from experiments.param_eq.corpus import CorpusRow, external_archive_hash, external_archive_root, load_corpus_rows
from experiments.param_eq.resource_guard import (
    DEFAULT_MEMORY_LIMIT_MB,
    DEFAULT_SAMPLE_INTERVAL_SEC,
    cap_workers_for_memory,
    watch_process,
    watch_subprocess,
)

from egglog import bindings as egglog_bindings

RESULTS_DIR = Path(__file__).resolve().parent / "results"
RAW_RESULTS_DIR = RESULTS_DIR / "raw"
REPO_ROOT = Path(__file__).resolve().parents[2]
VARIANT_ORDER_SEED = "param-eq-variant-order-v1"
DEFAULT_HASKELL_BUILD_TIMEOUT_SEC = 600.0
RAW_COLUMNS = (
    "row_id",
    "dataset",
    "algorithm",
    "input_kind",
    "implementation",
    "variant",
    "external_archive_sha256",
    "source_n_rank",
    "timeout_sec",
    "memory_limit_mb",
    "sample_interval_sec",
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
    "status",
    "runtime_ms",
    "peak_rss_mb",
    "passes",
    "total_size",
    "before_nodes",
    "before_params",
    "after_nodes",
    "after_params",
)


def _validate_haskell_checkout(archive_root: Path) -> None:
    """Require the source modules and Stack configuration used by live mode."""
    required = [
        archive_root / "stack.yaml",
        *(
            archive_root / "src" / name
            for name in (
                "FixTree.hs",
                "KotanchekSR.hs",
                "KotanchekSympy.hs",
                "PagieSR.hs",
                "PagieSympy.hs",
                "Reparam.hs",
            )
        ),
    ]
    missing = [path.relative_to(archive_root).as_posix() for path in required if not path.is_file()]
    if missing:
        raise ValueError(f"External Haskell checkout is missing: {', '.join(missing)}")


def _build_haskell_program() -> str:
    """Build a reusable forced live-Haskell runner without embedding source text."""
    return "\n".join([  # noqa: FLY002
        "import Control.Exception (evaluate)",
        "import Data.List (intercalate)",
        "import qualified Data.Map as M",
        "import Data.SRTree",
        "import FixTree",
        "import KotanchekSR (kotanchekSR)",
        "import KotanchekSympy (kotanchekSympy)",
        "import PagieSR (pagieSR)",
        "import PagieSympy (pagieSympy)",
        "import Reparam (replaceConstsWithParams)",
        "import Data.Time.Clock.POSIX (getPOSIXTime)",
        "import System.Environment (getArgs)",
        "",
        "lookupExpr :: String -> String -> String -> Int -> SRTree Int Double",
        "lookupExpr dataset inputKind algorithm rowIndex = case (dataset, inputKind) of",
        '  ("pagie", "original") -> (pagieSR M.! algorithm) !! rowIndex',
        '  ("pagie", "sympy") -> (pagieSympy M.! algorithm) !! rowIndex',
        '  ("kotanchek", "original") -> (kotanchekSR M.! algorithm) !! rowIndex',
        '  ("kotanchek", "sympy") -> (kotanchekSympy M.! algorithm) !! rowIndex',
        '  _ -> error "unknown dataset/input kind"',
        "",
        "emitExpr :: SRTree Int Double -> IO ()",
        "emitExpr expr = do",
        "  beforeNodes <- evaluate (countNodes expr)",
        "  beforeParams <- evaluate (recountParams (replaceConstsWithParams expr))",
        "  start <- getPOSIXTime",
        "  let simplified = simplifyE expr",
        "  afterNodes <- evaluate (countNodes simplified)",
        "  afterParams <- evaluate (recountParams (replaceConstsWithParams simplified))",
        "  end <- getPOSIXTime",
        "  let runtimeMs = (realToFrac (end - start) :: Double) * 1000.0",
        "      fields = map show [fromIntegral beforeNodes, fromIntegral beforeParams,",
        "                         fromIntegral afterNodes, fromIntegral afterParams, runtimeMs]",
        '  putStrLn (intercalate "\\t" fields)',
        "",
        "main :: IO ()",
        "main = do",
        "  args <- getArgs",
        "  case args of",
        "    [dataset, inputKind, algorithm, rowIndex] ->",
        "      emitExpr (lookupExpr dataset inputKind algorithm (read rowIndex))",
        '    _ -> error "expected dataset input-kind algorithm zero-based-row-index"',
        "",
    ])


def _parse_haskell_output(stdout: str) -> dict[str, Any]:
    """Parse the expression-free, fully forced Haskell result row."""
    lines = [line for line in stdout.splitlines() if line.strip()]
    if len(lines) != 1:
        raise ValueError(f"Expected one Haskell output row, got {len(lines)}")
    fields = lines[0].split("\t")
    if len(fields) != 5:
        raise ValueError(f"Expected five Haskell output fields, got {len(fields)}")
    before_nodes, before_params, after_nodes, after_params, runtime_ms = map(float, fields)
    counts = (before_nodes, before_params, after_nodes, after_params)
    if (
        any(not math.isfinite(value) or value < 0.0 or not value.is_integer() for value in counts)
        or not math.isfinite(runtime_ms)
        or runtime_ms < 0.0
    ):
        msg = "Haskell output contains invalid counts or runtime"
        raise ValueError(msg)
    return {
        "status": "saturated",
        "runtime_ms": runtime_ms,
        "passes": "",
        "total_size": "",
        "before_nodes": before_nodes,
        "before_params": before_params,
        "after_nodes": after_nodes,
        "after_params": after_params,
    }


def _compile_haskell_runner(
    archive_root: Path,
    build_root: Path,
    *,
    execution_mode: str,
    timeout_sec: float,
    memory_limit_mb: int,
    sample_interval_sec: float,
) -> Path:
    """Compile the generated runner once so per-row timings exclude startup."""
    source_path = build_root / "ParamEqRunner.hs"
    object_dir = build_root / "objects"
    executable = build_root / "param-eq-haskell-runner"
    object_dir.mkdir()
    source_path.write_text(_build_haskell_program(), encoding="utf-8")
    optimization = "-O2" if execution_mode == "release" else "-O0"
    stack = shutil.which("stack")
    if stack is None:
        msg = "The live-Haskell runner requires Stack on PATH"
        raise RuntimeError(msg)
    process = subprocess.Popen(
        [
            stack,
            "ghc",
            "--",
            optimization,
            "-rtsopts",
            "-isrc",
            "-outputdir",
            str(object_dir),
            "-o",
            str(executable),
            str(source_path),
        ],
        cwd=archive_root,
        text=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    watched = watch_subprocess(
        process,
        timeout_sec=timeout_sec,
        memory_limit_mb=memory_limit_mb,
        sample_interval_sec=sample_interval_sec,
    )
    if watched.status != "completed":
        raise RuntimeError(f"Haskell runner compilation reached the {watched.status} guard")
    if process.returncode != 0 or not executable.is_file():
        msg = "Could not compile the live-Haskell runner; run `stack build` in the external checkout first"
        raise RuntimeError(msg)
    return executable


def _collect_run_provenance(*, execution_mode: str, workers_requested: int, workers_effective: int) -> dict[str, str]:
    """Resolve the local source checkouts, loaded extension, and machine state."""
    metadata_process = subprocess.run(
        ["cargo", "metadata", "--format-version", "1", "--locked"],  # noqa: S607
        cwd=REPO_ROOT,
        capture_output=True,
        check=False,
        text=True,
    )
    if metadata_process.returncode != 0:
        raise RuntimeError(f"Could not resolve Cargo dependencies: {metadata_process.stderr.strip()}")
    packages = json.loads(metadata_process.stdout)["packages"]
    manifests = {package["name"]: Path(package["manifest_path"]) for package in packages}
    repositories = {"egglog_python": REPO_ROOT}
    for package_name, prefix in (("egglog", "egglog_core"), ("egglog-experimental", "egglog_experimental")):
        try:
            repositories[prefix] = manifests[package_name].parent
        except KeyError as error:
            raise RuntimeError(f"Cargo metadata has no {package_name!r} dependency") from error
    dependency_values: dict[str, str] = {}
    for prefix, worktree in repositories.items():
        commit_process = subprocess.run(
            ["git", "rev-parse", "HEAD"],  # noqa: S607
            cwd=worktree,
            capture_output=True,
            check=False,
            text=True,
        )
        status_process = subprocess.run(
            ["git", "status", "--porcelain"],  # noqa: S607
            cwd=worktree,
            capture_output=True,
            check=False,
            text=True,
        )
        if commit_process.returncode != 0 or status_process.returncode != 0:
            raise RuntimeError(f"Could not resolve Git state for {prefix!r}")
        dependency_values[f"{prefix}_commit"] = commit_process.stdout.strip()
        dependency_values[f"{prefix}_clean"] = str(not status_process.stdout.strip()).lower()

    rust_process = subprocess.run(
        ["rustc", "--version"],  # noqa: S607
        capture_output=True,
        check=False,
        text=True,
    )
    if rust_process.returncode != 0 or not rust_process.stdout.strip():
        msg = "Could not resolve the Rust compiler version"
        raise RuntimeError(msg)
    bindings_file = getattr(egglog_bindings, "__file__", None)
    if bindings_file is None or not Path(bindings_file).is_file():
        msg = "Could not identify the loaded egglog native extension"
        raise RuntimeError(msg)
    with Path(bindings_file).open("rb") as artifact:
        bindings_sha256 = hashlib.file_digest(artifact, "sha256").hexdigest()
    cpu = platform.processor()
    if platform.system() == "Darwin":
        cpu_process = subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"],  # noqa: S607
            capture_output=True,
            check=False,
            text=True,
        )
        if cpu_process.returncode == 0 and cpu_process.stdout.strip():
            cpu = cpu_process.stdout.strip()
    return {
        "execution_mode": execution_mode,
        "workers_requested": str(workers_requested),
        "workers_effective": str(workers_effective),
        "ordering_seed": VARIANT_ORDER_SEED,
        **dependency_values,
        "egglog_bindings_sha256": bindings_sha256,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "rust_version": rust_process.stdout.strip(),
        "cpu": cpu or platform.machine(),
    }


def _worker(connection: Connection, source: str, variant: str) -> None:
    try:
        from egglog.exp.param_eq import (  # noqa: PLC0415
            binary_to_containers,
            parse_expression,
            run_paper_pipeline,
            run_paper_pipeline_container,
        )

        report = (
            run_paper_pipeline(parse_expression(source))
            if variant == "binary"
            else run_paper_pipeline_container(binary_to_containers(parse_expression(source)))
        )
        payload: dict[str, Any] = {"status": report.status}
        if report.status == "saturated":
            payload.update({
                "runtime_ms": report.total_sec * 1000.0,
                "passes": report.passes,
                "total_size": report.total_size,
                "before_nodes": report.before_nodes,
                "before_params": report.before_params,
                "after_nodes": report.extracted_nodes,
                "after_params": report.extracted_params,
            })
        connection.send(payload)
    except BaseException:  # worker errors are accounted for without publishing private input text
        connection.send({"status": "error"})
    finally:
        connection.close()


def _raw_result(
    row: CorpusRow,
    implementation: str,
    variant: str,
    *,
    external_archive_sha256: str,
    timeout_sec: float,
    memory_limit_mb: int,
    sample_interval_sec: float,
    provenance: Mapping[str, str],
    payload: Mapping[str, Any],
    peak_rss_mb: float | None,
) -> dict[str, Any]:
    """Build the expression-free raw schema shared by both implementations."""
    return {
        "row_id": row.row_id,
        "dataset": row.dataset,
        "algorithm": row.algorithm,
        "input_kind": row.input_kind,
        "implementation": implementation,
        "variant": variant,
        "external_archive_sha256": external_archive_sha256,
        "source_n_rank": row.source_n_rank,
        "timeout_sec": timeout_sec,
        "memory_limit_mb": memory_limit_mb,
        "sample_interval_sec": sample_interval_sec,
        **provenance,
        "status": payload["status"],
        "runtime_ms": payload.get("runtime_ms", ""),
        "peak_rss_mb": peak_rss_mb if peak_rss_mb is not None else "",
        "passes": payload.get("passes", ""),
        "total_size": payload.get("total_size", ""),
        "before_nodes": payload.get("before_nodes", ""),
        "before_params": payload.get("before_params", ""),
        "after_nodes": payload.get("after_nodes", ""),
        "after_params": payload.get("after_params", ""),
    }


def _run_egglog_one(
    row: CorpusRow,
    variant: str,
    *,
    external_archive_sha256: str,
    timeout_sec: float,
    memory_limit_mb: int,
    sample_interval_sec: float,
    provenance: Mapping[str, str],
) -> dict[str, Any]:
    context = get_context("spawn")
    parent, child = context.Pipe(duplex=False)
    process = context.Process(target=_worker, args=(child, row.source, variant))
    process.start()
    child.close()
    watched = watch_process(
        process,
        timeout_sec=timeout_sec,
        memory_limit_mb=memory_limit_mb,
        sample_interval_sec=sample_interval_sec,
    )
    payload: dict[str, Any]
    try:
        if watched.status != "completed":
            payload = {"status": watched.status}
        elif parent.poll():
            try:
                payload = parent.recv()
            except EOFError:
                payload = {"status": "error"}
        else:
            payload = {"status": "error"}
    finally:
        parent.close()
    return _raw_result(
        row,
        "egglog",
        variant,
        external_archive_sha256=external_archive_sha256,
        timeout_sec=timeout_sec,
        memory_limit_mb=memory_limit_mb,
        sample_interval_sec=sample_interval_sec,
        provenance=provenance,
        payload=payload,
        peak_rss_mb=watched.peak_rss_mb,
    )


def _run_haskell_one(
    row: CorpusRow,
    *,
    archive_root: Path,
    executable: Path,
    external_archive_sha256: str,
    timeout_sec: float,
    memory_limit_mb: int,
    sample_interval_sec: float,
    provenance: Mapping[str, str],
) -> dict[str, Any]:
    peak_rss_mb = None
    payload: dict[str, Any] = {"status": "error"}
    try:
        process = subprocess.Popen(
            [
                str(executable),
                row.dataset,
                row.input_kind,
                row.algorithm_raw,
                str(row.algorithm_row - 1),
                "+RTS",
                "-K3G",
                "-RTS",
            ],
            cwd=archive_root,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            start_new_session=True,
        )
        watched = watch_subprocess(
            process,
            timeout_sec=timeout_sec,
            memory_limit_mb=memory_limit_mb,
            sample_interval_sec=sample_interval_sec,
        )
        stdout, _stderr = process.communicate()
        peak_rss_mb = watched.peak_rss_mb
        if watched.status != "completed":
            payload = {"status": watched.status}
        elif process.returncode == 0:
            payload = _parse_haskell_output(stdout)
    except (OSError, ValueError):
        pass
    return _raw_result(
        row,
        "haskell",
        "binary",
        external_archive_sha256=external_archive_sha256,
        timeout_sec=timeout_sec,
        memory_limit_mb=memory_limit_mb,
        sample_interval_sec=sample_interval_sec,
        provenance=provenance,
        payload=payload,
        peak_rss_mb=peak_rss_mb,
    )


def _variant_order(row_id: str, requested: str) -> tuple[str, ...]:
    if requested != "both":
        return (requested,)
    digest = hashlib.sha256(f"{VARIANT_ORDER_SEED}\0{row_id}".encode()).digest()
    return ("binary", "container") if digest[0] % 2 == 0 else ("container", "binary")


def _run_row(
    row: CorpusRow,
    *,
    implementation: str,
    variant: str,
    archive_root: Path,
    haskell_executable: Path | None,
    external_archive_sha256: str,
    timeout_sec: float,
    memory_limit_mb: int,
    sample_interval_sec: float,
    provenance: Mapping[str, str],
) -> list[dict[str, Any]]:
    """Keep paired variants sequential while parallelizing independent rows."""
    if implementation == "haskell":
        if haskell_executable is None:
            msg = "The Haskell implementation requires a compiled runner"
            raise ValueError(msg)
        return [
            _run_haskell_one(
                row,
                archive_root=archive_root,
                executable=haskell_executable,
                external_archive_sha256=external_archive_sha256,
                timeout_sec=timeout_sec,
                memory_limit_mb=memory_limit_mb,
                sample_interval_sec=sample_interval_sec,
                provenance=provenance,
            )
        ]
    return [
        _run_egglog_one(
            row,
            current_variant,
            external_archive_sha256=external_archive_sha256,
            timeout_sec=timeout_sec,
            memory_limit_mb=memory_limit_mb,
            sample_interval_sec=sample_interval_sec,
            provenance=provenance,
        )
        for current_variant in _variant_order(row.row_id, variant)
    ]


def run_rows(
    rows: Iterable[CorpusRow],
    *,
    implementation: str,
    variant: str,
    archive_root: Path,
    haskell_executable: Path | None,
    external_archive_sha256: str,
    workers: int,
    timeout_sec: float,
    memory_limit_mb: int,
    sample_interval_sec: float = DEFAULT_SAMPLE_INTERVAL_SEC,
    provenance: Mapping[str, str],
) -> list[dict[str, Any]]:
    """Run rows with stable order balancing and explicit failure accounting."""
    results: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(
                _run_row,
                row,
                implementation=implementation,
                variant=variant,
                archive_root=archive_root,
                haskell_executable=haskell_executable,
                external_archive_sha256=external_archive_sha256,
                timeout_sec=timeout_sec,
                memory_limit_mb=memory_limit_mb,
                sample_interval_sec=sample_interval_sec,
                provenance=provenance,
            ): row.row_id
            for row in rows
        }
        for future in as_completed(futures):
            results.extend(future.result())
    return sorted(results, key=lambda item: (str(item["row_id"]), str(item["implementation"]), str(item["variant"])))


def write_local_raw(rows: Iterable[dict[str, Any]], path: Path) -> None:
    """Write untracked row metrics; expressions are never included."""
    resolved_path = path.expanduser().resolve()
    resolved_parent = resolved_path.parent
    if resolved_parent != RAW_RESULTS_DIR.resolve():
        msg = f"Row-level output must stay in the ignored directory {RAW_RESULTS_DIR}"
        raise ValueError(msg)
    relative_path = resolved_path.relative_to(REPO_ROOT.resolve()).as_posix()
    git = shutil.which("git")
    if git is None:
        msg = "Git is required to verify that row-level output stays untracked"
        raise RuntimeError(msg)
    is_tracked = (
        subprocess.run(
            [git, "ls-files", "--error-unmatch", "--", relative_path],
            cwd=REPO_ROOT,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        ).returncode
        == 0
    )
    is_ignored = (
        subprocess.run(
            [git, "check-ignore", "--quiet", "--", relative_path],
            cwd=REPO_ROOT,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        ).returncode
        == 0
    )
    if is_tracked or not is_ignored:
        msg = "Row-level output must be an untracked, Git-ignored file"
        raise ValueError(msg)
    resolved_parent.mkdir(parents=True, exist_ok=True)
    with resolved_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=RAW_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--implementation", choices=("egglog", "haskell"), default="egglog")
    parser.add_argument("--variant", choices=("binary", "container", "both"), default="both")
    parser.add_argument("--dataset", choices=("pagie", "kotanchek"))
    parser.add_argument("--algorithm")
    parser.add_argument("--input-kind", choices=("original", "sympy"))
    parser.add_argument("--limit", type=int)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--timeout-sec", type=float, default=60.0)
    parser.add_argument("--memory-limit-mb", type=int, default=DEFAULT_MEMORY_LIMIT_MB)
    parser.add_argument("--haskell-build-timeout-sec", type=float, default=DEFAULT_HASKELL_BUILD_TIMEOUT_SEC)
    parser.add_argument("--execution-mode", choices=("debug", "release"), required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    nonpositive = next(
        (
            name
            for name, value in {
                "--workers": args.workers,
                "--limit": args.limit,
                "--timeout-sec": args.timeout_sec,
                "--memory-limit-mb": args.memory_limit_mb,
                "--haskell-build-timeout-sec": args.haskell_build_timeout_sec,
            }.items()
            if value is not None and value <= 0
        ),
        None,
    )
    if nonpositive is not None:
        parser.error(f"{nonpositive} must be positive")
    if args.implementation == "haskell" and args.variant != "binary":
        parser.error("The live-Haskell implementation supports only --variant binary")
    output = args.output or RAW_RESULTS_DIR / f"{args.implementation}-{args.variant}.csv"
    archive_root = external_archive_root()
    if args.implementation == "haskell":
        try:
            _validate_haskell_checkout(archive_root)
        except ValueError as error:
            parser.error(str(error))
    rows = load_corpus_rows(
        archive_root,
        dataset=args.dataset,
        algorithm=args.algorithm,
        input_kind=args.input_kind,
        limit=args.limit,
    )
    archive_sha256 = external_archive_hash(archive_root)
    expected_archive_sha256 = os.environ.get("EGGLOG_PARAM_EQ_EXPECTED_ARCHIVE_SHA256", "").lower()
    if len(expected_archive_sha256) != 64 or any(
        character not in "0123456789abcdef" for character in expected_archive_sha256
    ):
        parser.error("Set EGGLOG_PARAM_EQ_EXPECTED_ARCHIVE_SHA256 to the privately recorded archive hash")
    if archive_sha256.lower() != expected_archive_sha256:
        parser.error("The external Param-Eq archive does not match EGGLOG_PARAM_EQ_EXPECTED_ARCHIVE_SHA256")
    if not rows:
        parser.error("The selected corpus filters matched no rows")
    effective_workers = cap_workers_for_memory(args.workers, memory_limit_mb=args.memory_limit_mb)
    provenance = _collect_run_provenance(
        execution_mode=args.execution_mode,
        workers_requested=args.workers,
        workers_effective=effective_workers,
    )
    with ExitStack() as stack:
        haskell_executable = None
        if args.implementation == "haskell":
            build_root = Path(stack.enter_context(tempfile.TemporaryDirectory(prefix="param-eq-haskell-")))
            try:
                haskell_executable = _compile_haskell_runner(
                    archive_root,
                    build_root,
                    execution_mode=args.execution_mode,
                    timeout_sec=args.haskell_build_timeout_sec,
                    memory_limit_mb=args.memory_limit_mb,
                    sample_interval_sec=DEFAULT_SAMPLE_INTERVAL_SEC,
                )
            except (OSError, RuntimeError) as error:
                parser.error(str(error))
        results = run_rows(
            rows,
            implementation=args.implementation,
            variant=args.variant,
            archive_root=archive_root,
            haskell_executable=haskell_executable,
            external_archive_sha256=archive_sha256,
            workers=effective_workers,
            timeout_sec=args.timeout_sec,
            memory_limit_mb=args.memory_limit_mb,
            provenance=provenance,
        )
    write_local_raw(results, output)
    counts = Counter(str(row["status"]) for row in results)
    print(
        f"wrote {len(results)} rows to {output}; workers={effective_workers}/{args.workers}; "
        f"statuses={dict(sorted(counts.items()))}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
