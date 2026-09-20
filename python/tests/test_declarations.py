import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from threading import Event, RLock, get_ident
from weakref import WeakValueDictionary

import pytest

from egglog import declarations
from egglog.declarations import CallDecl, FunctionRef, Ident


@pytest.mark.parametrize("published", [False, True], ids=["cache-miss", "uninitialized-publication"])
def test_call_decl_interning_serializes_initialization(monkeypatch: pytest.MonkeyPatch, published: bool) -> None:
    paused, resume, second_entered = Event(), Event(), Event()
    second_thread: int | None = None
    lock = RLock()

    class ObservedLock:
        def __enter__(self) -> None:
            if get_ident() == second_thread:
                second_entered.set()
            lock.acquire()

        def __exit__(self, *args: object) -> None:
            lock.release()

    class PausingPool(WeakValueDictionary):
        def __getitem__(self, key) -> CallDecl:
            try:
                return super().__getitem__(key)
            except KeyError:
                if not published and not paused.is_set():
                    paused.set()
                    assert resume.wait(10)
                raise

    original_init = CallDecl.__init__

    def pausing_init(self, *args, **kwargs) -> None:
        if published and not paused.is_set():
            # __new__ has published the object, but dataclass initialization has not begun.
            paused.set()
            assert resume.wait(10)
            assert not hasattr(self, "callable")
        original_init(self, *args, **kwargs)

    monkeypatch.setattr(declarations, "_CALL_DECL_LOCK", ObservedLock(), raising=False)
    monkeypatch.setattr(CallDecl, "_args_to_value", PausingPool())
    monkeypatch.setattr(CallDecl, "__init__", pausing_init)
    callable_ref = FunctionRef(Ident("concurrent_interning"))

    def second_call() -> CallDecl:
        nonlocal second_thread
        second_thread = get_ident()
        try:
            return CallDecl(callable_ref)
        finally:
            # Without locking, completion releases the first constructor instead.
            second_entered.set()

    with ThreadPoolExecutor(max_workers=2) as executor:
        first = executor.submit(CallDecl, callable_ref)
        try:
            assert paused.wait(10)
            second = executor.submit(second_call)
            assert second_entered.wait(10)
        finally:
            resume.set()
        call = first.result(timeout=10)
        assert second.result(timeout=10) is call
    assert (call.callable, call.args, call.bound_tp_params) == (callable_ref, (), ())


def test_call_decl_interning_allows_reentrant_literal_hashing() -> None:
    subprocess.run(
        [
            sys.executable,
            "-c",
            """
from egglog import i64

class ReentrantInt(int):
    def __hash__(self):
        i64(1) + 2
        return super().__hash__()

assert str(i64(ReentrantInt(3)) + 4) == "i64(3) + 4"
""",
        ],
        check=True,
        timeout=10,
    )
