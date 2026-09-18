from __future__ import annotations

import subprocess
import sys
from unittest.mock import MagicMock

import pytest
from experiments.param_eq import resource_guard


def test_cap_workers_for_memory_respects_safe_memory_budget() -> None:
    gib = 1024**3

    assert (
        resource_guard.cap_workers_for_memory(
            8,
            memory_limit_mb=1024,
            total_memory_bytes_value=8 * gib,
            safe_memory_fraction=0.5,
        )
        == 4
    )


def test_process_tree_rss_includes_descendants_only(monkeypatch: pytest.MonkeyPatch) -> None:
    completed = subprocess.CompletedProcess(
        ["ps"],
        0,
        stdout="100 1 1024\n101 100 2048\n102 101 3072\n200 1 4096\nmalformed\n",
    )
    monkeypatch.setattr(resource_guard.subprocess, "run", lambda *args, **kwargs: completed)

    assert resource_guard._process_tree_rss_mb(100) == 6.0


def test_watch_process_kills_multiprocessing_worker_at_memory_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    process = MagicMock()
    process.pid = 1234
    process.is_alive.return_value = True
    monkeypatch.setattr(resource_guard, "_rss_mb", lambda pid: 65.0)

    result = resource_guard.watch_process(process, timeout_sec=10.0, memory_limit_mb=64)

    assert result == resource_guard.WatchResult("memory_limit", 65.0)
    process.kill.assert_called_once_with()
    process.join.assert_called_once_with(timeout=1.0)


def test_watch_subprocess_kills_process_group_at_memory_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    process = MagicMock()
    process.pid = 5678
    process.poll.return_value = None
    kill_process_group = MagicMock()
    monkeypatch.setattr(resource_guard, "_process_tree_rss_mb", lambda pid: 129.0)
    monkeypatch.setattr(resource_guard, "_kill_subprocess_group", kill_process_group)

    result = resource_guard.watch_subprocess(process, timeout_sec=10.0, memory_limit_mb=128)

    assert result == resource_guard.WatchResult("memory_limit", 129.0)
    kill_process_group.assert_called_once_with(process)


def test_watch_subprocess_timeout_terminates_real_process() -> None:
    process = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(10)"],
        text=True,
        start_new_session=True,
    )
    try:
        result = resource_guard.watch_subprocess(
            process,
            timeout_sec=0.05,
            memory_limit_mb=1_000_000_000,
            sample_interval_sec=0.01,
        )

        assert result.status == "timeout"
        assert process.poll() is not None
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=1.0)
