"""
Provides a class for solving type constraints.


Usages:

When trying to resolve a literal to a value

"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import chain, repeat
from typing import TYPE_CHECKING, assert_never

from .declarations import *

if TYPE_CHECKING:
    from collections.abc import Collection, Iterable


__all__ = ["TypeConstraintError", "TypeConstraintSolver"]


class TypeConstraintError(RuntimeError):
    """Typing error when trying to infer the return type."""


@dataclass
class TypeConstraintSolver:
    """
    Given some typevars and types, solves the constraints to resolve the typevars.
    """

    # Mapping of typevar index to inferred type for each class
    _typevar_to_type: dict[Ident, JustTypeRef] = field(default_factory=dict, init=False)

    def bind_class(self, ref: JustTypeRef, decls: Declarations) -> None:
        """
        Bind the typevars of a class to the given types.
        Used for a situation like Map[int, str].create().

        This is the same as binding the typevars of the class to the given types.
        """
        cls_typevars = decls.get_class_decl(ref.ident).type_vars
        for typevar, arg in zip(cls_typevars, ref.args, strict=True):
            self.infer_typevars(typevar, arg)

    def infer_arg_types(
        self,
        fn_args: Collection[TypeOrVarRef],
        fn_return: TypeOrVarRef,
        fn_var_args: TypeOrVarRef | None,
        return_: JustTypeRef,
    ) -> Iterable[JustTypeRef]:
        """
        Given a return type, infer the argument types. If there is a variable arg, it returns an infinite iterable.
        """
        self.infer_typevars(fn_return, return_)
        arg_types = [self.substitute_typevars(fn_arg) for fn_arg in fn_args]
        if fn_var_args is None:
            return arg_types
        var_arg_type = self.substitute_typevars(fn_var_args)
        return chain(arg_types, repeat(var_arg_type))

    def infer_typevars(self, fn_arg: TypeOrVarRef, arg: JustTypeRef) -> None:
        """
        Infer typevars from a function argument and a given type, raises TypeConstraintError if they are incompatible.
        """
        match fn_arg:
            case TypeRefWithVars(cls_ident, fn_args):
                if cls_ident != arg.ident:
                    raise TypeConstraintError(f"Expected {cls_ident}, got {arg.ident}")
                for inner_fn_arg, inner_arg in zip(fn_args, arg.args, strict=True):
                    self.infer_typevars(inner_fn_arg, inner_arg)
            case TypeVarRef(typevar_ident):
                if typevar_ident in self._typevar_to_type:
                    if self._typevar_to_type[typevar_ident] != arg:
                        raise TypeConstraintError(f"Expected {self._typevar_to_type[typevar_ident]}, got {arg}")
                else:
                    self._typevar_to_type[typevar_ident] = arg
            case _:
                assert_never(fn_arg)

    def substitute_typevars(self, tp: TypeOrVarRef) -> JustTypeRef:
        """
        Substitute typevars in a type with their inferred types, raises TypeConstraintError if a typevar is unresolved.
        """
        match tp:
            case TypeVarRef(typevar_ident):
                try:
                    return self._typevar_to_type[typevar_ident]
                except KeyError as e:
                    raise TypeConstraintError(f"Unresolved type variable: {typevar_ident}") from e
            case TypeRefWithVars(name, args):
                return JustTypeRef(name, tuple(self.substitute_typevars(arg) for arg in args))
        assert_never(tp)

    def substitute_typevars_try_function(
        self, tp: TypeOrVarRef, value: Callable, decls: Callable[[], Declarations]
    ) -> JustTypeRef:
        """
        Try to substitute typevars in a type with their inferred types.

        If this fails and we have an UnstableFn type and a function value, we can try to infer the typevars by calling
        it with the input types, if we can resolve those
        """
        from .runtime import RuntimeExpr  # noqa: PLC0415

        try:
            return self.substitute_typevars(tp)
        except TypeConstraintError:
            if isinstance(tp, TypeVarRef) or tp.ident != Ident.builtin("UnstableFn") or not callable(value):
                raise
        dummy_args = [
            RuntimeExpr.__from_values__(decls(), TypedExprDecl(self.substitute_typevars(arg_tp), DummyDecl()))
            for arg_tp in tp.args[1:]
        ]
        try:
            result = value(*dummy_args)
        except Exception as e:
            raise TypeConstraintError(
                f"Function {value} raised an exception when called with dummy args to infer return type: {e}"
            ) from e
        if not isinstance(result, RuntimeExpr):
            raise TypeConstraintError(
                f"Function {value} did not return a RuntimeExpr, got {type(result)}, so cannot infer return type"
            )
        self.infer_typevars(tp.args[0], result.__egg_typed_expr__.tp)
        return self.substitute_typevars(tp)
