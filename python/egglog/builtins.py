"""Catalog-generated builtin wrappers and handwritten Python value protocols."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast

from egglog_proto.egglog.v1 import egglog_pb as pb
from typing_extensions import deprecated

from ._catalog import builtin_catalog
from .conversion import converter
from .runtime import RuntimeClass, RuntimeExpr, RuntimeFunction


@dataclass
class ExprValueError(AttributeError):
    """A symbolic expression is not in a locally inspectable value form."""

    expr: object
    allowed: str

    def __str__(self) -> str:
        return f"Cannot get Python value of {self.expr}, must be of form {self.allowed}. Try calling `extract` on it to get the underlying value."


def _scalar_value(expr: RuntimeExpr) -> object:
    from .deconstruct import get_literal_value  # noqa: PLC0415

    value = get_literal_value(expr)
    if value is not None:
        return value
    node: pb.Node = expr.__egg_ref__.read()
    from .runtime import _class_for_sort  # noqa: PLC0415

    cls = _class_for_sort(expr.__egg_ref__.owner.ref("sorts", node.sort_id))
    raise ExprValueError(expr, str(cls))


@deprecated("use .value")
def _eval(expr: RuntimeExpr) -> object:
    return expr.value


def _integer(expr: RuntimeExpr) -> int:
    return int(cast("Any", expr.value))


def _float(expr: RuntimeExpr) -> float:
    return float(cast("Any", expr.value))


# These conversions/codecs are Python value protocol policy. No callable input,
# output, overload, or generic signature is duplicated here.
_LITERAL_PYTHON_TYPES: dict[str, tuple[type, ...]] = {
    "i64": (int,),
    "f64": (float, int),
    "String": (str,),
    "bool": (bool,),
}
__all__ = ["ExprValueError"]
_catalog = builtin_catalog()
for _definition in _catalog.definitions.values():
    _declaration: pb.Declaration = _definition.read()
    _kind = _declaration.kind
    if _kind is None or not isinstance(_kind.value, pb.HostSortFamily | pb.EqSort):
        continue
    _binding = _kind.value.bindings
    _path = (
        tuple(_binding.python.path)
        if _binding is not None and _binding.python is not None
        else (*__name__.split("."), _kind.value.name)
    )
    if _path[:-1] != tuple(__name__.split(".")):
        continue
    _name = _path[-1]
    if not _name.isidentifier() or _name in globals():
        raise ValueError(f"Cannot generate Python builtin type {_path!r}")
    _class = RuntimeClass(_definition)
    globals()[_name] = _class
    __all__ += [_name]  # noqa: PLE0604 -- names come from the generated catalog
    if _kind.value.name in _LITERAL_PYTHON_TYPES:
        _class.__egg_hooks__.update(value=property(_scalar_value), eval=_eval)
        _literal_types = _LITERAL_PYTHON_TYPES[_kind.value.name]
        _like: Any = _class
        for _literal_type in _literal_types:
            _like = _like | _literal_type
            if _kind.value.name == "f64":
                converter(_literal_type, _class, lambda value, target=_class: target(float(value)))
            else:
                converter(_literal_type, _class, _class)
        globals()[f"{_name}Like"] = _like
        __all__ += [f"{_name}Like"]  # noqa: PLE0604
        if _kind.value.name == "i64":
            _class.__egg_hooks__.update(__index__=_integer, __int__=_integer)
        if _kind.value.name == "f64":
            _class.__egg_hooks__.update(__float__=_float, __int__=_integer)

for _path, (_definition, _view) in _catalog.functions.items():
    if _path[:-1] == tuple(__name__.split(".")):
        if _path[-1] in globals():
            raise ValueError(f"Python builtin binding collision: {_path!r}")
        globals()[_path[-1]] = RuntimeFunction(_definition, _view)
        __all__ += [_path[-1]]  # noqa: PLE0604
