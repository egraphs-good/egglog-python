from __future__ import annotations

from dataclasses import dataclass, field
from typing import Collection

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
        args: Collection[JustTypeRef],
    ) -> JustTypeRef:
        self._infer_typevars_zip(fn_args, args)
        return self._subtitute_typevars(fn_return)

    def _infer_typevars_zip(
        self, fn_args: Collection[TypeOrVarRef], args: Collection[JustTypeRef]
    ) -> None:
        if len(fn_args) != len(args):
            raise TypeConstraintError(f"Expected {len(fn_args)} args, got {len(args)}")
        for fn_arg, arg in zip(fn_args, args):
            self._infer_typevars(fn_arg, arg)

    def _infer_typevars(self, fn_arg: TypeOrVarRef, arg: JustTypeRef) -> None:
        if isinstance(fn_arg, TypeRefWithVars):
            if fn_arg.name != arg.name:
                raise TypeConstraintError(f"Expected {fn_arg.name}, got {arg.name}")
            self._infer_typevars_zip(fn_arg.args, arg.args)
        elif fn_arg not in self._cls_typevar_index_to_type:
            self._cls_typevar_index_to_type[fn_arg.index] = arg
        elif self._cls_typevar_index_to_type[fn_arg.index] != arg:
            raise TypeConstraintError(f"Expected {fn_arg}, got {arg}")

    def _subtitute_typevars(self, tp: TypeOrVarRef) -> JustTypeRef:
        if isinstance(tp, ClassTypeVarRef):
            return self._cls_typevar_index_to_type[tp.index]
        elif isinstance(tp, TypeRefWithVars):
            return JustTypeRef(
                tp.name,
                tuple(self._subtitute_typevars(arg) for arg in tp.args),
            )


def test_type_inference() -> None:
    import pytest

    i64 = TypeRefWithVars("i64")
    unit = TypeRefWithVars("Unit")
    K, V = ClassTypeVarRef(0), ClassTypeVarRef(1)
    map = TypeRefWithVars("Map", (K, V))
    map_i64_unit = TypeRefWithVars("Map", (i64, unit))

    cs = TypeConstraintSolver()
    assert cs.infer_return_type([i64], i64, [i64.to_just()]) == i64.to_just()
    with pytest.raises(TypeError):
        cs.infer_return_type([i64], i64, [unit.to_just()])
    with pytest.raises(TypeError):
        cs.infer_return_type([], i64, [unit.to_just()])

    assert (
        cs.infer_return_type([map, K], V, [map_i64_unit.to_just(), i64.to_just()])
        == unit.to_just()
    )

    with pytest.raises(TypeError):
        cs.infer_return_type([map, K], V, [map_i64_unit.to_just(), unit.to_just()])

    bound_cs = TypeConstraintSolver.from_type_parameters(
        [i64.to_just(), unit.to_just()]
    )
    assert bound_cs.infer_return_type([K], V, [i64.to_just()]) == unit.to_just()

    with pytest.raises(TypeError):
        bound_cs.infer_return_type([K], V, [unit.to_just()])
