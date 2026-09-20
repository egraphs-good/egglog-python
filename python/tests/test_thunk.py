from concurrent.futures import ThreadPoolExecutor, TimeoutError
from threading import Event

import pytest

from egglog.thunk import Thunk


def test_recursive_resolution_still_raises() -> None:
    thunk = Thunk.fn(lambda: thunk())

    with pytest.raises(ValueError, match="Recursively resolving thunk") as first:
        thunk()
    with pytest.raises(ValueError, match="Recursively resolving thunk") as second:
        thunk()
    assert second.value is first.value


@pytest.mark.parametrize("error", [None, ValueError("resolution failed"), KeyboardInterrupt("resolution interrupted")])
@pytest.mark.parametrize("partial", [False, True])
def test_concurrent_resolution_is_cached(error: BaseException | None, partial: bool) -> None:
    entered = Event()
    release = Event()
    calling = Event()
    value = object()
    partial_value = object()
    calls = []

    def resolve() -> object:
        calls.append(None)
        if partial:
            thunk.set_partial(partial_value)
            assert thunk() is partial_value
        entered.set()
        assert release.wait(10)
        if error is not None:
            raise error
        return value

    thunk = Thunk.fn(resolve)

    def concurrent_call() -> object:
        calling.set()
        return thunk()

    with ThreadPoolExecutor(max_workers=2) as executor:
        first = executor.submit(thunk)
        try:
            assert entered.wait(10)
            second = executor.submit(concurrent_call)
            assert calling.wait(10)
            with pytest.raises(TimeoutError):
                second.result(timeout=0.1)
        finally:
            release.set()

        if error is not None:
            assert first.exception(timeout=10) is error
            assert second.exception(timeout=10) is error
            with pytest.raises(type(error)) as cached:
                thunk()
            assert cached.value is error
        else:
            assert first.result(timeout=10) is value
            assert second.result(timeout=10) is value
            assert thunk() is value
    assert len(calls) == 1


def test_partial_value_requires_active_resolution() -> None:
    with pytest.raises(ValueError, match="Cannot set a partial value outside thunk resolution"):
        Thunk.value(1).set_partial(2)
