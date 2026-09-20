from concurrent.futures import ThreadPoolExecutor
from threading import Event, Lock, Thread
from typing import Any, cast

import pytest

from egglog import Expr, conversion, convert, converter, expr_parts
from egglog.declarations import ClassDecl, Declarations, DelayedDeclarations, Ident, JustTypeRef, TypeRefWithVars
from egglog.runtime import RuntimeClass, RuntimeExpr
from egglog.thunk import Thunk

ConversionKey = tuple[type | JustTypeRef, JustTypeRef]
ConversionValue = tuple[int, object]


def test_retrieve_conversion_decls_does_not_drop_concurrent_types() -> None:
    class Pending(Expr): ...

    conversion.retrieve_conversion_decls()
    resolving = Event()
    queued = Event()
    queued_while_resolving = []

    def resolve_blocking_declarations() -> Declarations:
        resolving.set()
        queued_while_resolving.append(queued.wait(10))
        return Declarations()

    conversion._TO_PROCESS_DECLS.append(DelayedDeclarations(resolve_blocking_declarations))

    def queue_pending_type() -> None:
        conversion.process_tp(Pending)
        queued.set()

    with ThreadPoolExecutor(max_workers=2) as executor:
        retrieval = executor.submit(conversion.retrieve_conversion_decls)
        assert resolving.wait(10)
        queueing = executor.submit(queue_pending_type)
        retrieval.result(timeout=10)
        queueing.result(timeout=10)

    assert queued_while_resolving == [True]
    assert any(cast("object", item) is Pending for item in conversion._TO_PROCESS_DECLS)
    conversion.retrieve_conversion_decls()
    assert not conversion._TO_PROCESS_DECLS
    assert conversion._CONVERSION_DECLS_OWNER is None
    assert not conversion._CONVERSION_DECLS_IN_FLIGHT
    assert not conversion._CONVERSION_DECLS_OWNER_PENDING


def test_retrieve_conversion_decls_can_reenter_from_resolving_thread() -> None:
    inner_results: list[Declarations] = []

    def resolve_declarations() -> Declarations:
        partial = Declarations()
        thunk.set_partial(partial)
        inner_results.append(conversion.retrieve_conversion_decls())
        return partial

    thunk = Thunk.fn(resolve_declarations)
    conversion._TO_PROCESS_DECLS.append(DelayedDeclarations(thunk))

    result = conversion.retrieve_conversion_decls()

    assert inner_results
    assert inner_results[0] is not result
    assert not conversion._TO_PROCESS_DECLS
    assert conversion._CONVERSION_DECLS_OWNER is None
    assert not conversion._CONVERSION_DECLS_IN_FLIGHT
    assert not conversion._CONVERSION_DECLS_OWNER_PENDING


def test_failed_reentrant_retrieval_does_not_publish_partial_declarations(monkeypatch) -> None:
    conversion.retrieve_conversion_decls()
    monkeypatch.setattr(conversion, "_CONVERSION_DECLS", conversion._CONVERSION_DECLS.copy())
    marker = Ident("FailedReentrantConversion", __name__)
    marker_decl = ClassDecl()
    inner_results: list[Declarations] = []
    error = RuntimeError("declaration resolution failed")

    def resolve_declarations() -> Declarations:
        partial = Declarations(_classes={marker: marker_decl})
        thunk.set_partial(partial)
        inner_results.append(conversion.retrieve_conversion_decls())
        raise error

    thunk = Thunk.fn(resolve_declarations)
    pending = DelayedDeclarations(thunk)
    conversion._TO_PROCESS_DECLS.append(pending)

    with pytest.raises(RuntimeError, match="declaration resolution failed") as raised:
        conversion.retrieve_conversion_decls()

    assert raised.value is error
    assert inner_results[0].get_class_decl(marker) is marker_decl
    assert marker not in conversion._CONVERSION_DECLS._classes
    assert any(item is pending for item in conversion._TO_PROCESS_DECLS)
    assert conversion._CONVERSION_DECLS_OWNER is None
    assert not conversion._CONVERSION_DECLS_IN_FLIGHT
    assert not conversion._CONVERSION_DECLS_OWNER_PENDING

    conversion._TO_PROCESS_DECLS.remove(pending)
    assert not conversion._TO_PROCESS_DECLS


def test_reentrant_retrieval_ignores_other_threads_pending_declarations(monkeypatch) -> None:
    conversion.retrieve_conversion_decls()
    monkeypatch.setattr(conversion, "_CONVERSION_DECLS", conversion._CONVERSION_DECLS.copy())
    marker = Ident("ConcurrentReentrantConversion", __name__)
    marker_decl = ClassDecl()
    resolving = Event()
    unrelated_queued = Event()
    inner_results: list[Declarations] = []
    unrelated_calls = []
    unrelated_error = LookupError("unrelated declaration failed")

    def resolve_declarations() -> Declarations:
        partial = Declarations(_classes={marker: marker_decl})
        outer_thunk.set_partial(partial)
        resolving.set()
        assert unrelated_queued.wait(10)
        inner_results.append(conversion.retrieve_conversion_decls())
        return partial

    def fail_unrelated_declarations() -> Declarations:
        unrelated_calls.append(None)
        raise unrelated_error

    outer_thunk = Thunk.fn(resolve_declarations)
    outer_pending = DelayedDeclarations(outer_thunk)
    unrelated = RuntimeClass(
        Thunk.fn(fail_unrelated_declarations), TypeRefWithVars(Ident("UnrelatedFailedConversion", __name__))
    )
    conversion._TO_PROCESS_DECLS.append(outer_pending)

    def queue_unrelated() -> None:
        conversion.process_tp(unrelated)
        unrelated_queued.set()

    with ThreadPoolExecutor(max_workers=2) as executor:
        retrieval = executor.submit(conversion.retrieve_conversion_decls)
        assert resolving.wait(10)
        queueing = executor.submit(queue_unrelated)
        queueing.result(timeout=10)
        result = retrieval.result(timeout=10)

    assert inner_results[0].get_class_decl(marker) is marker_decl
    assert result.get_class_decl(marker) is marker_decl
    assert not unrelated_calls
    assert any(item is unrelated for item in conversion._TO_PROCESS_DECLS)

    with pytest.raises(LookupError, match="unrelated declaration failed") as raised:
        conversion.retrieve_conversion_decls()
    assert raised.value is unrelated_error
    assert unrelated_calls == [None]
    assert any(item is unrelated for item in conversion._TO_PROCESS_DECLS)
    assert conversion._CONVERSION_DECLS_OWNER is None
    assert not conversion._CONVERSION_DECLS_IN_FLIGHT
    assert not conversion._CONVERSION_DECLS_OWNER_PENDING

    conversion._TO_PROCESS_DECLS.remove(unrelated)
    assert not conversion._TO_PROCESS_DECLS


class _CoordinatedConversions(dict[ConversionKey, ConversionValue]):
    def __init__(self, key: ConversionKey) -> None:
        super().__init__()
        self._key = key
        self._contains_lock = Lock()
        self._is_first_check = True
        self._first_checked = Event()
        self._release_first = Event()
        self._cheap_written = Event()

    def __contains__(self, key: object) -> bool:
        present = super().__contains__(key)
        if key != self._key or present:
            return present
        with self._contains_lock:
            first_check = self._is_first_check
            self._is_first_check = False
        if first_check:
            self._first_checked.set()
            assert self._release_first.wait(10)
        return False

    def __setitem__(self, key: ConversionKey, value: ConversionValue) -> None:
        cost, _ = value
        super().__setitem__(key, value)
        if key == self._key and cost == 1:
            self._cheap_written.set()


def test_concurrent_registration_keeps_lowest_cost(monkeypatch) -> None:
    class Source: ...

    target = JustTypeRef(Ident("ThreadedTarget"))
    key = (Source, target)
    conversions = _CoordinatedConversions(key)
    monkeypatch.setattr(conversion, "CONVERSIONS", conversions)

    def convert_source(value: Any) -> RuntimeExpr:
        raise AssertionError(value)

    def register(cost: int) -> None:
        conversion._register_converter(Source, target, convert_source, cost)

    with ThreadPoolExecutor(max_workers=2) as executor:
        expensive = executor.submit(register, 10)
        assert conversions._first_checked.wait(10)
        cheap_started = Event()

        def register_cheap() -> None:
            cheap_started.set()
            register(1)

        cheap = executor.submit(register_cheap)
        assert cheap_started.wait(10)

        registration_locked = not conversion._CONVERSION_LOCK.acquire(blocking=False)
        if not registration_locked:
            conversion._CONVERSION_LOCK.release()
            assert conversions._cheap_written.wait(10)
        conversions._release_first.set()

        expensive.result(timeout=10)
        cheap.result(timeout=10)

    registered = conversion._lookup_conversion(Source, target)
    assert registered is not None
    assert registered[0] == 1
    assert registration_locked


def test_converter_callback_runs_without_registration_lock() -> None:
    class Source: ...

    class AdditionalSource: ...

    class Target(Expr):
        def __init__(self) -> None: ...

    registered = Event()

    def register_additional_converter() -> None:
        converter(AdditionalSource, Target, lambda _: Target())
        registered.set()

    def convert_source(_: Source) -> Target:
        registration = Thread(target=register_additional_converter)
        registration.start()
        try:
            assert registered.wait(10)
        finally:
            registration.join(timeout=10)
        return Target()

    converter(Source, Target, convert_source)

    assert expr_parts(convert(Source(), Target)) == expr_parts(Target())
