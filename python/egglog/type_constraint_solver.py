from __future__ import annotations

from dataclasses import dataclass, field
from itertools import chain, repeat
from typing import Collection, Optional

from .declarations import *

__all__ = ["TypeConstraintSolver", "TypeConstraintError"]


class TypeConstraintError(RuntimeError):
    pass


@dataclass
class TypeConstraintSolver:
    """
    Given some typevars and types, solves the constraints to resolve the typevars.
    """

    # Mapping of class typevar index to their types
    _cls_typevar_index_to_type: dict[int, JustTypeRef] = field(default_factory=dict)

    @classmethod
    def from_type_parameters(cls, tps: Collection[JustTypeRef]) -> TypeConstraintSolver:
        """
        Create a TypeInference from a parameterized class, given as a bound typevar.

        Used for a situation like Map[int, str].create()
        """
        cs = cls()
        # For each param in the class, use this as the ith bound class typevar
        for i, tp in enumerate(tps):
            cs._cls_typevar_index_to_type[i] = tp
        return cs

    def infer_return_type(
        self,
        fn_args: Collection[TypeOrVarRef],
        fn_return: TypeOrVarRef,
        fn_var_args: Optional[TypeOrVarRef],
        args: Collection[JustTypeRef],
    ) -> JustTypeRef:
        # Infer the type of each type variable based on the actual types of the arguments
        self._infer_typevars_zip(fn_args, fn_var_args, args)
        # Substitute the type variables with their inferred types
        return self._subtitute_typevars(fn_return)

    def _infer_typevars_zip(
        self, fn_args: Collection[TypeOrVarRef], fn_var_args: Optional[TypeOrVarRef], args: Collection[JustTypeRef]
    ) -> None:
        if len(fn_args) != len(args) if fn_var_args is None else len(fn_args) > len(args):
            raise TypeConstraintError(f"Mismatch of args {fn_args} and {args}")
        all_fn_args = fn_args if fn_var_args is None else chain(fn_args, repeat(fn_var_args))
        for fn_arg, arg in zip(all_fn_args, args):
            self._infer_typevars(fn_arg, arg)

    def _infer_typevars(self, fn_arg: TypeOrVarRef, arg: JustTypeRef) -> None:
        if isinstance(fn_arg, TypeRefWithVars):
            if fn_arg.name != arg.name:
                raise TypeConstraintError(f"Expected {fn_arg.name}, got {arg.name}")
            self._infer_typevars_zip(fn_arg.args, None, arg.args)
        elif fn_arg.index not in self._cls_typevar_index_to_type:
            self._cls_typevar_index_to_type[fn_arg.index] = arg
        elif self._cls_typevar_index_to_type[fn_arg.index] != arg:
            raise TypeConstraintError(f"Expected {fn_arg}, got {arg}")

    def _subtitute_typevars(self, tp: TypeOrVarRef) -> JustTypeRef:
        if isinstance(tp, ClassTypeVarRef):
            try:
                return self._cls_typevar_index_to_type[tp.index]
            except KeyError:
                raise TypeConstraintError(f"Not enough bound typevars for {tp}")
        elif isinstance(tp, TypeRefWithVars):
            return JustTypeRef(
                tp.name,
                tuple(self._subtitute_typevars(arg) for arg in tp.args),
            )
