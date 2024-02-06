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

__all__ = ["TypeConstraintSolver", "TypeConstraintError"]


class TypeConstraintError(RuntimeError):
    """Typing error when trying to infer the return type."""


@dataclass
class TypeConstraintSolver:
    """
    Given some typevars and types, solves the constraints to resolve the typevars.
    """

    _decls: Declarations
    # Mapping of class name to mapping of bound class typevar to type
    _cls_typevar_index_to_type: defaultdict[str, dict[str, JustTypeRef]] = field(
        default_factory=lambda: defaultdict(dict)
    )

    def bind_class(self, ref: JustTypeRef) -> None:
        """
        Bind the typevars of a class to the given types.
        Used for a situation like Map[int, str].create().
        """
        cls_typevars = self._decls.get_class_decl(ref.name).type_vars
        if len(cls_typevars) != len(ref.args):
            raise TypeConstraintError(f"Mismatch of typevars {cls_typevars} and {ref}")
        bound_typevars = self._cls_typevar_index_to_type[ref.name]
        for i, arg in enumerate(ref.args):
            bound_typevars[cls_typevars[i]] = arg

    def infer_return_type(
        self,
        fn_args: Collection[TypeOrVarRef],
        fn_return: TypeOrVarRef,
        fn_var_args: TypeOrVarRef | None,
        args: Collection[JustTypeRef],
        cls_name: str | None,
    ) -> JustTypeRef:
        """
        Given some arg types, infer the return type.

        cls_name should be the class name if this is a classmethod, so we can lookup typevars.
        """
        # Infer the type of each type variable based on the actual types of the arguments
        self._infer_typevars_zip(fn_args, fn_var_args, args, cls_name)
        # Substitute the type variables with their inferred types
        return self._subtitute_typevars(fn_return, cls_name)

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
        self._infer_typevars(fn_return, return_, cls_name)
        arg_types = (
            self._subtitute_typevars(a, cls_name) for a in chain(fn_args, repeat(fn_var_args) if fn_var_args else [])
        )
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

    def _infer_typevars_zip(
        self,
        fn_args: Collection[TypeOrVarRef],
        fn_var_args: TypeOrVarRef | None,
        args: Collection[JustTypeRef],
        cls_name: str | None,
    ) -> None:
        if len(fn_args) != len(args) if fn_var_args is None else len(fn_args) > len(args):
            raise TypeConstraintError(f"Mismatch of args {fn_args} and {args}")
        all_fn_args = fn_args if fn_var_args is None else chain(fn_args, repeat(fn_var_args))
        for fn_arg, arg in zip(all_fn_args, args, strict=False):
            self._infer_typevars(fn_arg, arg, cls_name)

    def _infer_typevars(self, fn_arg: TypeOrVarRef, arg: JustTypeRef, cls_name: str | None) -> None:
        match fn_arg:
            case TypeRefWithVars(cls_name, fn_args):
                if cls_name != arg.name:
                    raise TypeConstraintError(f"Expected {cls_name}, got {arg.name}")
                self._infer_typevars_zip(fn_args, None, arg.args, cls_name)
            case ClassTypeVarRef(typevar):
                if cls_name is None:
                    msg = "Cannot infer typevar without class name"
                    raise RuntimeError(msg)
                class_typevars = self._cls_typevar_index_to_type[cls_name]
                if typevar in class_typevars:
                    if class_typevars[typevar] != arg:
                        raise TypeConstraintError(f"Expected {class_typevars[typevar]}, got {arg}")
                else:
                    class_typevars[typevar] = arg
            case _:
                assert_never(fn_arg)

    def _subtitute_typevars(self, tp: TypeOrVarRef, cls_name: str | None) -> JustTypeRef:
        match tp:
            case ClassTypeVarRef(name):
                try:
                    assert cls_name is not None
                    return self._cls_typevar_index_to_type[cls_name][name]
                except KeyError as e:
                    raise TypeConstraintError(f"Not enough bound typevars for {tp}") from e
            case TypeRefWithVars(name, args):
                return JustTypeRef(name, tuple(self._subtitute_typevars(arg, name) for arg in args))
        assert_never(tp)
