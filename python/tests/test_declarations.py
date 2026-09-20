from concurrent.futures import ThreadPoolExecutor, TimeoutError
from dataclasses import replace
from threading import Event
from weakref import WeakValueDictionary

import pytest

from egglog.declarations import CallDecl, ClassMethodRef, FunctionRef, Ident, JustTypeRef, LitDecl, TypedExprDecl


def test_call_decl_interns_normalized_arguments() -> None:
    callable = FunctionRef(Ident("interned"))
    call = CallDecl(callable)
    assert CallDecl(callable=callable, args=(), bound_tp_params=()) is call

    args = (TypedExprDecl(JustTypeRef(Ident.builtin("i64")), LitDecl(1)),)
    with_args = CallDecl(callable, args)
    assert CallDecl(callable=callable, args=args) is with_args
    equal_args = (TypedExprDecl(args[0].tp, LitDecl(1)),)
    assert CallDecl(callable, equal_args) is with_args
    assert with_args.args is args
    assert replace(call, args=args) is with_args
    assert with_args != call
    match with_args:
        case CallDecl(matched_callable, matched_args, matched_bound_tp_params):
            assert matched_callable == callable
            assert matched_args == args
            assert matched_bound_tp_params == ()
        case _:
            pytest.fail("CallDecl did not support positional pattern matching")


def test_call_decl_validates_bound_type_parameters() -> None:
    tp = JustTypeRef(Ident.builtin("i64"))
    for _ in range(2):
        with pytest.raises(ValueError, match="Cannot bind type parameters to a non-class method callable"):
            CallDecl(FunctionRef(Ident("invalid_bound_parameters")), bound_tp_params=(tp,))

    callable = ClassMethodRef(Ident("Generic"), "create")
    call = CallDecl(callable, bound_tp_params=(tp,))
    assert CallDecl(callable, (), (tp,)) is call


def test_call_decl_publishes_initialized_canonical_instance(monkeypatch: pytest.MonkeyPatch) -> None:
    publishing = Event()
    release = Event()
    calling = Event()
    published_fields = []
    callable = FunctionRef(Ident("concurrent_interning"))

    class PausingPool(WeakValueDictionary):
        def __setitem__(self, key, value) -> None:
            published_fields.append((
                getattr(value, "callable", None),
                getattr(value, "args", None),
                getattr(value, "bound_tp_params", None),
            ))
            super().__setitem__(key, value)
            publishing.set()
            assert release.wait(10)

    monkeypatch.setattr(CallDecl, "_args_to_value", PausingPool())

    def concurrent_call() -> CallDecl:
        calling.set()
        return CallDecl(callable)

    with ThreadPoolExecutor(max_workers=2) as executor:
        first = executor.submit(CallDecl, callable)
        try:
            assert publishing.wait(10)
            second = executor.submit(concurrent_call)
            assert calling.wait(10)
            with pytest.raises(TimeoutError):
                second.result(timeout=0.1)
        finally:
            release.set()

        call = first.result(timeout=10)
        assert second.result(timeout=10) is call
    assert published_fields == [(callable, (), ())]
