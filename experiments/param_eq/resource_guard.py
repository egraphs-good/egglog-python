"""Memory and timeout guards for isolated Param-Eq corpus workers."""

from __future__ import annotations

import os
import signal
import subprocess
import time
from contextlib import suppress
from dataclasses import dataclass
from multiprocessing.process import BaseProcess

SAFE_MEMORY_FRACTION = 0.75
DEFAULT_MEMORY_LIMIT_MB = 2048
DEFAULT_SAMPLE_INTERVAL_SEC = 0.2


@dataclass(frozen=True)
class WatchResult:
    status: str
    peak_rss_mb: float | None


def total_system_memory_bytes() -> int:
    if "SC_PAGE_SIZE" in os.sysconf_names and "SC_PHYS_PAGES" in os.sysconf_names:
        return int(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES"))
    return int(subprocess.check_output(["sysctl", "-n", "hw.memsize"], text=True).strip())  # noqa: S607


def cap_workers_for_memory(
    requested_workers: int,
    *,
    memory_limit_mb: int = DEFAULT_MEMORY_LIMIT_MB,
    total_memory_bytes_value: int | None = None,
    safe_memory_fraction: float = SAFE_MEMORY_FRACTION,
) -> int:
    total = total_system_memory_bytes() if total_memory_bytes_value is None else total_memory_bytes_value
    allowed_mb = total / (1024.0 * 1024.0) * safe_memory_fraction
    return max(1, min(requested_workers, max(1, int(allowed_mb // memory_limit_mb))))


def _rss_mb(pid: int) -> float | None:
    completed = subprocess.run(
        ["ps", "-o", "rss=", "-p", str(pid)],  # noqa: S607
        capture_output=True,
        check=False,
        text=True,
    )
    if completed.returncode != 0 or not completed.stdout.strip():
        return None
    try:
        return float(completed.stdout.strip()) / 1024.0
    except ValueError:
        return None


def _process_tree_rss_mb(root_pid: int) -> float | None:
    """Return aggregate RSS for a subprocess and all of its descendants."""
    completed = subprocess.run(
        ["ps", "-axo", "pid=,ppid=,rss="],  # noqa: S607
        capture_output=True,
        check=False,
        text=True,
    )
    if completed.returncode != 0:
        return None
    parents: dict[int, int] = {}
    rss_kb: dict[int, int] = {}
    for line in completed.stdout.splitlines():
        try:
            pid_text, parent_text, rss_text = line.split()
            pid = int(pid_text)
            parents[pid] = int(parent_text)
            rss_kb[pid] = int(rss_text)
        except ValueError:
            continue
    descendants = {root_pid}
    changed = True
    while changed:
        changed = False
        for pid, parent in parents.items():
            if parent in descendants and pid not in descendants:
                descendants.add(pid)
                changed = True
    measured = [rss_kb[pid] for pid in descendants if pid in rss_kb]
    return sum(measured) / 1024.0 if measured else None


def _kill_subprocess_group(process: subprocess.Popen[str]) -> None:
    """Kill a subprocess started in its own session, including descendants."""
    with suppress(ProcessLookupError):
        os.killpg(process.pid, signal.SIGKILL)
    process.wait(timeout=1.0)


def watch_process(
    process: BaseProcess,
    *,
    timeout_sec: float,
    memory_limit_mb: int,
    sample_interval_sec: float = DEFAULT_SAMPLE_INTERVAL_SEC,
) -> WatchResult:
    start = time.monotonic()
    peak_rss_mb = None
    while process.is_alive():
        if process.pid is not None:
            rss_mb = _rss_mb(process.pid)
            if rss_mb is not None:
                peak_rss_mb = rss_mb if peak_rss_mb is None else max(peak_rss_mb, rss_mb)
                if rss_mb > memory_limit_mb:
                    process.kill()
                    process.join(timeout=1.0)
                    return WatchResult("memory_limit", peak_rss_mb)
        if time.monotonic() - start > timeout_sec:
            process.kill()
            process.join(timeout=1.0)
            return WatchResult("timeout", peak_rss_mb)
        time.sleep(sample_interval_sec)
    process.join(timeout=1.0)
    return WatchResult("completed", peak_rss_mb)


def watch_subprocess(
    process: subprocess.Popen[str],
    *,
    timeout_sec: float,
    memory_limit_mb: int,
    sample_interval_sec: float = DEFAULT_SAMPLE_INTERVAL_SEC,
) -> WatchResult:
    """Apply a timeout/RSS boundary to an external process tree."""
    start = time.monotonic()
    peak_rss_mb = None
    while process.poll() is None:
        rss_mb = _process_tree_rss_mb(process.pid)
        if rss_mb is not None:
            peak_rss_mb = rss_mb if peak_rss_mb is None else max(peak_rss_mb, rss_mb)
            if rss_mb > memory_limit_mb:
                _kill_subprocess_group(process)
                return WatchResult("memory_limit", peak_rss_mb)
        if time.monotonic() - start > timeout_sec:
            _kill_subprocess_group(process)
            return WatchResult("timeout", peak_rss_mb)
        time.sleep(sample_interval_sec)
    return WatchResult("completed", peak_rss_mb)
