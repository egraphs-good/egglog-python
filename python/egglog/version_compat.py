import collections
import sys
import types
import typing

BEFORE_3_11 = sys.version_info < (3, 11)

__all__ = ["add_note"]


def add_note(message: str, exc: BaseException) -> BaseException:
    """
    Backwards compatible add_note for Python <= 3.10
    """
    if BEFORE_3_11:
        return exc
    exc.add_note(message)
    return exc


# For Python version 3.10 need to monkeypatch this function so that RuntimeClass type parameters
# will be collected as typevars
if BEFORE_3_11:

    @typing.no_type_check
    def _collect_type_vars_monkeypatch(types_, typevar_types=None):
        """
        Collect all type variable contained
        in types in order of first appearance (lexicographic order). For example::

            _collect_type_vars((T, List[S, T])) == (T, S)
        """
        from .runtime import RuntimeClass

        if typevar_types is None:
            typevar_types = typing.TypeVar
        tvars = []
        for t in types_:
            if isinstance(t, typevar_types) and t not in tvars:
                tvars.append(t)
            # **MONKEYPATCH CHANGE HERE TO ADD RuntimeClass**
            if isinstance(t, (typing._GenericAlias, typing.GenericAlias, types.UnionType, RuntimeClass)):  # type: ignore[name-defined]
                tvars.extend([t for t in t.__parameters__ if t not in tvars])
        return tuple(tvars)

    typing._collect_type_vars = _collect_type_vars_monkeypatch  # type: ignore[attr-defined]

    @typing.no_type_check
    @typing._tp_cache
    def __getitem__monkeypatch(self, params):  # noqa: C901, PLR0912
        from .runtime import RuntimeClass

        if self.__origin__ in (typing.Generic, typing.Protocol):
            # Can't subscript Generic[...] or Protocol[...].
            raise TypeError(f"Cannot subscript already-subscripted {self}")
        if not isinstance(params, tuple):
            params = (params,)
        params = tuple(typing._type_convert(p) for p in params)
        if self._paramspec_tvars and any(isinstance(t, typing.ParamSpec) for t in self.__parameters__):
            params = typing._prepare_paramspec_params(self, params)
        else:
            typing._check_generic(self, params, len(self.__parameters__))

        subst = dict(zip(self.__parameters__, params, strict=False))
        new_args = []
        for arg in self.__args__:
            if isinstance(arg, self._typevar_types):
                if isinstance(arg, typing.ParamSpec):
                    arg = subst[arg]  # noqa: PLW2901
                    if not typing._is_param_expr(arg):
                        raise TypeError(f"Expected a list of types, an ellipsis, ParamSpec, or Concatenate. Got {arg}")
                else:
                    arg = subst[arg]  # noqa: PLW2901
            # **MONKEYPATCH CHANGE HERE TO ADD RuntimeClass**
            elif isinstance(arg, (typing._GenericAlias, typing.GenericAlias, types.UnionType, RuntimeClass)):
                subparams = arg.__parameters__
                if subparams:
                    subargs = tuple(subst[x] for x in subparams)
                    arg = arg[subargs]  # noqa: PLW2901
            # Required to flatten out the args for CallableGenericAlias
            if self.__origin__ == collections.abc.Callable and isinstance(arg, tuple):
                new_args.extend(arg)
            else:
                new_args.append(arg)
        return self.copy_with(tuple(new_args))

    typing._GenericAlias.__getitem__ = __getitem__monkeypatch  # type: ignore[attr-defined]
