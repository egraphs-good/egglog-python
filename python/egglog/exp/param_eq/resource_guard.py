"""Resource-guard helpers for `param_eq` row runners."""

from __future__ import annotations

import os
import subprocess
import time
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
    page_size_name = "SC_PAGE_SIZE"
    phys_pages_name = "SC_PHYS_PAGES"
    if page_size_name in os.sysconf_names and phys_pages_name in os.sysconf_names:
        return int(os.sysconf(page_size_name) * os.sysconf(phys_pages_name))
    return int(subprocess.check_output(["sysctl", "-n", "hw.memsize"], text=True).strip())  # noqa: S607


def total_system_memory_mb() -> float:
    return total_system_memory_bytes() / (1024.0 * 1024.0)


def cap_workers_for_memory(
    requested_workers: int,
    *,
    memory_limit_mb: int = DEFAULT_MEMORY_LIMIT_MB,
    total_memory_bytes_value: int | None = None,
    safe_memory_fraction: float = SAFE_MEMORY_FRACTION,
) -> int:
    total_memory_bytes_value = (
        total_system_memory_bytes() if total_memory_bytes_value is None else total_memory_bytes_value
    )
    allowed_budget_mb = (total_memory_bytes_value / (1024.0 * 1024.0)) * safe_memory_fraction
    max_workers = max(1, int(allowed_budget_mb // memory_limit_mb))
    return max(1, min(requested_workers, max_workers))


def _rss_mb(pid: int) -> float | None:
    completed = subprocess.run(
        ["ps", "-o", "rss=", "-p", str(pid)],
        capture_output=True,
        check=False,
        text=True,
    )
    if completed.returncode != 0:
        return None
    output = completed.stdout.strip()
    if not output:
        return None
    try:
        return float(output) / 1024.0
    except ValueError:
        return None


def watch_subprocess(
    process: subprocess.Popen[str],
    *,
    timeout_sec: float,
    memory_limit_mb: int,
    sample_interval_sec: float = DEFAULT_SAMPLE_INTERVAL_SEC,
) -> WatchResult:
    start = time.monotonic()
    peak_rss_mb = None
    while process.poll() is None:
        rss_mb = _rss_mb(process.pid)
        if rss_mb is not None:
            peak_rss_mb = rss_mb if peak_rss_mb is None else max(peak_rss_mb, rss_mb)
            if rss_mb > memory_limit_mb:
                process.kill()
                return WatchResult(status="memory_limit", peak_rss_mb=peak_rss_mb)
        if time.monotonic() - start > timeout_sec:
            process.kill()
            return WatchResult(status="timeout", peak_rss_mb=peak_rss_mb)
        time.sleep(sample_interval_sec)
    rss_mb = _rss_mb(process.pid)
    if rss_mb is not None:
        peak_rss_mb = rss_mb if peak_rss_mb is None else max(peak_rss_mb, rss_mb)
    return WatchResult(status="completed", peak_rss_mb=peak_rss_mb)


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
                    return WatchResult(status="memory_limit", peak_rss_mb=peak_rss_mb)
        if time.monotonic() - start > timeout_sec:
            process.kill()
            process.join(timeout=1.0)
            return WatchResult(status="timeout", peak_rss_mb=peak_rss_mb)
        time.sleep(sample_interval_sec)
    if process.pid is not None:
        rss_mb = _rss_mb(process.pid)
        if rss_mb is not None:
            peak_rss_mb = rss_mb if peak_rss_mb is None else max(peak_rss_mb, rss_mb)
    process.join(timeout=1.0)
    return WatchResult(status="completed", peak_rss_mb=peak_rss_mb)
