"""Provides a class for solving type constraints."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from itertools import chain, repeat
from typing import TYPE_CHECKING

from typing_extensions import assert_never

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

    _decls: Declarations = field(repr=False)
    # Mapping of class name to mapping of bound class typevar to type
    _cls_typevar_index_to_type: defaultdict[str, dict[ClassTypeVarRef, JustTypeRef]] = field(
        default_factory=lambda: defaultdict(dict)
    )

    def bind_class(self, ref: JustTypeRef) -> None:
        """
        Bind the typevars of a class to the given types.
        Used for a situation like Map[int, str].create().
        """
        name = ref.name
        cls_typevars = self._decls.get_class_decl(name).type_vars
        if len(cls_typevars) != len(ref.args):
            raise TypeConstraintError(f"Mismatch of typevars {cls_typevars} and {ref}")
        bound_typevars = self._cls_typevar_index_to_type[name]
        for i, arg in enumerate(ref.args):
            bound_typevars[cls_typevars[i]] = arg

    def infer_arg_types(
        self,
        fn_args: Collection[TypeOrVarRef],
        fn_return: TypeOrVarRef,
        fn_var_args: TypeOrVarRef | None,
        return_: JustTypeRef,
        cls_name: str | None,
    ) -> tuple[Iterable[JustTypeRef], tuple[JustTypeRef, ...] | None]:
        """
        Given a return type, infer the argument types. If there is a variable arg, it returns an infinite iterable.

        Also returns the bound type params if the class name is passed in.
        """
        self.infer_typevars(fn_return, return_, cls_name)
        arg_types: Iterable[JustTypeRef] = [self.substitute_typevars(a, cls_name) for a in fn_args]
        if fn_var_args:
            # Need to be generator so it can be infinite for variable args
            arg_types = chain(arg_types, repeat(self.substitute_typevars(fn_var_args, cls_name)))
        bound_typevars = (
            tuple(
                v
                # Sort by the index of the typevar in the class
                for _, v in sorted(
                    self._cls_typevar_index_to_type[cls_name].items(),
                    key=lambda kv: self._decls.get_class_decl(cls_name).type_vars.index(kv[0]),
                )
            )
            if cls_name
            else None
        )
        return arg_types, bound_typevars

    def infer_typevars(self, fn_arg: TypeOrVarRef, arg: JustTypeRef, cls_name: str | None = None) -> None:
        match fn_arg:
            case TypeRefWithVars(cls_name, fn_args):
                if cls_name != arg.name:
                    raise TypeConstraintError(f"Expected {cls_name}, got {arg.name}")
                for inner_fn_arg, inner_arg in zip(fn_args, arg.args, strict=True):
                    self.infer_typevars(inner_fn_arg, inner_arg, cls_name)
            case ClassTypeVarRef():
                if cls_name is None:
                    msg = "Cannot infer typevar without class name"
                    raise RuntimeError(msg)

                class_typevars = self._cls_typevar_index_to_type[cls_name]
                if fn_arg in class_typevars:
                    if class_typevars[fn_arg] != arg:
                        raise TypeConstraintError(f"Expected {class_typevars[fn_arg]}, got {arg}")
                else:
                    class_typevars[fn_arg] = arg
            case _:
                assert_never(fn_arg)

    def substitute_typevars(self, tp: TypeOrVarRef, cls_name: str | None = None) -> JustTypeRef:
        match tp:
            case ClassTypeVarRef():
                assert cls_name is not None
                try:
                    return self._cls_typevar_index_to_type[cls_name][tp]
                except KeyError as e:
                    raise TypeConstraintError(f"Not enough bound typevars for {tp!r} in class {cls_name}") from e
            case TypeRefWithVars(name, args):
                return JustTypeRef(name, tuple(self.substitute_typevars(arg, cls_name) for arg in args))
        assert_never(tp)
