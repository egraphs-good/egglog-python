import sys
import typing

__all__ = ["monkeypatch_forward_ref"]


def monkeypatch_forward_ref():
    """
    Monkeypatch to backport https://github.com/python/cpython/pull/21553.
    Removed recursive gaurd for simplicity
    Can be removed once Pytho 3.8 is no longer supported
    """
    if sys.version_info >= (3, 9):
        return
    typing.ForwardRef._evaluate = _evaluate_monkeypatch  # type: ignore


def _evaluate_monkeypatch(self, globalns, localns):
    if not self.__forward_evaluated__ or localns is not globalns:
        if globalns is None and localns is None:
            globalns = localns = {}
        elif globalns is None:
            globalns = localns
        elif localns is None:
            localns = globalns
        type_ = typing._type_check(  # type: ignore
            eval(self.__forward_code__, globalns, localns),
            "Forward references must evaluate to types.",
            is_argument=self.__forward_is_argument__,
        )
        self.__forward_value__ = typing._eval_type(type_, globalns, localns)  # type: ignore
        self.__forward_evaluated__ = True
    return self.__forward_value__
