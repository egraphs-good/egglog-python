"""Python-only builtin surface policy; semantic signatures come from the native catalog."""

from typing import Any

_CALLABLES: dict[str, list[dict[str, Any]]] = {}
_OPERATORS = {
    "add": "__add__",
    "sub": "__sub__",
    "mul": "__mul__",
    "div": "__truediv__",
    "rem": "__mod__",
    "pow": "__pow__",
    "ge": "__ge__",
    "gt": "__gt__",
    "le": "__le__",
    "lt": "__lt__",
    "bitand": "__and__",
    "bitor": "__or__",
    "bitxor": "__xor__",
    "shl": "__lshift__",
    "shr": "__rshift__",
    "and": "__and__",
    "or": "__or__",
    "xor": "__xor__",
    "intersection": "__and__",
    "subtract": "__sub__",
    "sum": "__add__",
    "diff": "__sub__",
    "intersect": "__and__",
    "union": "__or__",
}
_NUMERIC_OPERATORS = "add div ge gt le lt mul sub"
for _namespace, _owner, _names in (
    ("egglog.core.bigint", 0, _NUMERIC_OPERATORS + " bitand bitor bitxor rem shl shr"),
    ("egglog.core.bigrat", 1, _NUMERIC_OPERATORS + " pow"),
    ("egglog.core.bool", 2, "and or xor"),
    ("egglog.core.f64", 3, _NUMERIC_OPERATORS + " pow rem"),
    ("egglog.core.i64", 4, _NUMERIC_OPERATORS + " bitand bitor bitxor rem shl shr"),
    ("egglog.core.multiset", 6, "intersection subtract sum"),
    ("egglog.core.set", 8, "diff intersect union"),
    ("egglog.core.vec", 10, "union"),
    ("egglog.experimental.rational", 12, _NUMERIC_OPERATORS + " pow"),
):
    for _name in _names.split():
        _CALLABLES[f"{_namespace}.{_name}"] = [
            {
                "form": "METHOD",
                "owner": _owner,
                "params": [{"name": "other"}],
                "path": [_OPERATORS[_name]],
            }
        ]
_CALLABLES["egglog.core.bool.or"][0]["tailRepetitions"] = 2

for _namespace, _owner, _names in (
    ("egglog.core.bigint", 0, "bool_eq bool_ge bool_gt bool_le bool_lt max min"),
    ("egglog.core.bigrat", 1, "max min"),
    ("egglog.core.bool", 2, "implies"),
    ("egglog.core.f64", 3, "max min"),
    ("egglog.core.i64", 4, "bool_ge bool_gt bool_le bool_lt max min"),
    ("egglog.experimental.rational", 12, "max min"),
):
    for _name in _names.split():
        _CALLABLES[f"{_namespace}.{_name}"] = [{"form": "METHOD", "owner": _owner, "params": [{"name": "other"}]}]

for _namespace, _owner, _names in (
    ("egglog.core.bigint", 0, "bits not to_string"),
    ("egglog.core.bigrat", 1, "abs cbrt ceil floor log neg round sqrt to_f64 to_i64"),
    ("egglog.core.bool", 2, "not"),
    ("egglog.core.f64", 3, "abs exp log neg sqrt to_i64 to_string"),
    ("egglog.core.i64", 4, "abs log2 not to_string"),
    ("egglog.core.map", 5, "length"),
    ("egglog.core.multiset", 6, "length pick pick_max reset_counts union_values"),
    ("egglog.core.set", 8, "length"),
    ("egglog.core.vec", 10, "length pop"),
    ("egglog.experimental.f64", 3, "is_finite"),
    ("egglog.experimental.maybe", 11, "unwrap"),
    ("egglog.experimental.rational", 12, "abs cbrt ceil floor log neg round sqrt to_f64"),
    ("egglog.python.pyobject", 13, "to_bool to_string"),
):
    for _name in _names.split():
        _view: dict[str, Any] = {"form": "METHOD", "owner": _owner}
        _alias = {"abs": "__abs__", "neg": "__neg__", "not": "__invert__"}.get(_name)
        if _alias is not None:
            _view["path"] = [_alias]
        _CALLABLES[f"{_namespace}.{_name}"] = [_view]

for _namespace, _owner, _names in (
    ("egglog.core.multiset", 6, "contains count insert not_contains remove"),
    ("egglog.core.set", 8, "contains insert not_contains remove"),
    ("egglog.core.vec", 10, "contains not_contains push"),
):
    for _name in _names.split():
        _CALLABLES[f"{_namespace}.{_name}"] = [{"form": "METHOD", "owner": _owner, "params": [{"name": "value"}]}]

for _name in ["contains", "not_contains", "remove"]:
    _CALLABLES[f"egglog.core.map.{_name}"] = [{"form": "METHOD", "owner": 5, "params": [{"name": "key"}]}]
for _namespace, _owner, _names in (
    ("egglog.core.multiset", 6, "filter filter_not map"),
    ("egglog.core.vec", 10, "map"),
):
    for _name in _names.split():
        _CALLABLES[f"{_namespace}.{_name}"] = [
            {"form": "METHOD", "owner": _owner, "params": [{"name": "f"}], "receiver": 1}
        ]
for _name in ["clear_index", "fill_index"]:
    _CALLABLES[f"egglog.core.multiset.{_name}"] = [{"form": "METHOD", "owner": 6, "params": [{"name": "f"}]}]
for _name in ["contains_swapped", "not_contains_swapped", "remove_swapped", "subtract_swapped"]:
    _CALLABLES[f"egglog.core.multiset.{_name}"] = [
        {
            "form": "FUNCTION",
            "params": [{"name": "x"}, {"name": "xs"}],
            "path": ["egglog", "builtins", f"multiset_{_name}"],
        }
    ]

for _namespace, _owner in (("egglog.core.multiset", 6), ("egglog.core.set", 8), ("egglog.core.vec", 10)):
    _CALLABLES[f"{_namespace}.of"] = [{"form": "INITIALIZER", "owner": _owner, "params": [{"name": "args"}]}]
for _namespace, _owner in (("egglog.core.map", 5), ("egglog.core.set", 8), ("egglog.core.vec", 10)):
    _CALLABLES[f"{_namespace}.empty"] = [{"form": "CLASS_METHOD", "owner": _owner}]
for _namespace, _owner in (("egglog.core.bigrat", 1), ("egglog.experimental.rational", 12)):
    for _name in ("denom", "numer"):
        _CALLABLES[f"{_namespace}.{_name}"] = [{"form": "PROPERTY", "owner": _owner}]
    _CALLABLES[f"{_namespace}.make"] = [
        {"form": "INITIALIZER", "owner": _owner, "params": [{"name": "num"}, {"name": "den"}]}
    ]
for _name, _default in (("eval", 0), ("exec", 1)):
    _CALLABLES[f"egglog.python.pyobject.{_name}"] = [
        {
            "form": "FUNCTION",
            "params": [
                {"name": "code"},
                {"default": _default, "name": "globals_"},
                {"default": _default, "name": "locals_"},
            ],
            "path": ["egglog", "builtins", f"py_{_name}"],
        }
    ]

# Nonuniform aliases and parameter layouts remain explicit.
_CALLABLES.update({
    "egglog.core.bigint.from_i64": [{"form": "INITIALIZER", "owner": 0, "params": [{"name": "value"}]}],
    "egglog.core.bigint.from_string": [{"form": "CLASS_METHOD", "owner": 0, "params": [{"name": "s"}]}],
    "egglog.core.f64.from_i64": [{"form": "CLASS_METHOD", "owner": 3, "params": [{"name": "i"}]}],
    "egglog.core.map.get": [{"form": "METHOD", "owner": 5, "params": [{"name": "key"}], "path": ["__getitem__"]}],
    "egglog.core.map.insert": [{"form": "METHOD", "owner": 5, "params": [{"name": "key"}, {"name": "value"}]}],
    "egglog.core.map.of": [
        {"form": "CLASS_METHOD", "owner": 5, "params": [{"name": "entries"}], "path": ["from_pairs"]}
    ],
    "egglog.core.multiset.flat_map": [
        {
            "form": "FUNCTION",
            "params": [{"name": "f"}, {"name": "xs"}],
            "path": ["egglog", "builtins", "multiset_flat_map"],
        }
    ],
    "egglog.core.multiset.reduce": [
        {
            "form": "FUNCTION",
            "params": [{"name": "f"}, {"name": "initial"}, {"name": "xs"}],
            "path": ["egglog", "builtins", "multiset_fold"],
        }
    ],
    "egglog.core.multiset.single": [{"form": "CLASS_METHOD", "owner": 6, "params": [{"name": "x"}, {"name": "i"}]}],
    "egglog.core.multiset.sum_multisets": [{"form": "CLASS_METHOD", "owner": 6, "params": [{"name": "xs"}]}],
    "egglog.core.pair.first": [{"form": "PROPERTY", "owner": 7, "path": ["left"]}],
    "egglog.core.pair.make": [{"form": "INITIALIZER", "owner": 7, "params": [{"name": "left"}, {"name": "right"}]}],
    "egglog.core.pair.second": [{"form": "PROPERTY", "owner": 7, "path": ["right"]}],
    "egglog.core.string.concat": [
        {"form": "FUNCTION", "params": [{"name": "strings"}], "path": ["egglog", "builtins", "join"]},
        {"form": "METHOD", "owner": 9, "params": [{"name": "other"}], "path": ["__add__"], "tailRepetitions": 2},
    ],
    "egglog.core.string.count_matches": [
        {
            "form": "FUNCTION",
            "params": [{"name": "s"}, {"name": "pattern"}],
            "path": ["egglog", "builtins", "count_matches"],
        }
    ],
    "egglog.core.string.replace": [{"form": "METHOD", "owner": 9, "params": [{"name": "old"}, {"name": "new"}]}],
    "egglog.core.vec.append": [{"form": "METHOD", "owner": 10, "params": [{"name": "others"}]}],
    "egglog.core.vec.get": [{"form": "METHOD", "owner": 10, "params": [{"name": "index"}], "path": ["__getitem__"]}],
    "egglog.core.vec.range": [{"form": "METHOD", "owner": 4}],
    "egglog.core.vec.remove": [{"form": "METHOD", "owner": 10, "params": [{"name": "index"}]}],
    "egglog.core.vec.set": [{"form": "METHOD", "owner": 10, "params": [{"name": "index"}, {"name": "value"}]}],
    "egglog.experimental.map.fold_kv": [
        {
            "form": "FUNCTION",
            "params": [{"name": "f"}, {"name": "initial"}, {"name": "xs"}],
            "path": ["egglog", "builtins", "map_fold_kv"],
        }
    ],
    "egglog.experimental.maybe.catch": [
        {"form": "FUNCTION", "params": [{"name": "f"}], "path": ["egglog", "builtins", "catch"]}
    ],
    "egglog.experimental.maybe.match": [{"form": "METHOD", "owner": 11, "params": [{"name": "f"}, {"name": "n"}]}],
    "egglog.experimental.maybe.none": [{"form": "CLASS_METHOD", "owner": 11}],
    "egglog.experimental.maybe.some": [{"form": "CLASS_METHOD", "owner": 11, "params": [{"name": "value"}]}],
    "egglog.experimental.maybe.unwrap_or": [{"form": "METHOD", "owner": 11, "params": [{"name": "default"}]}],
    "egglog.python.pyobject.call": [
        {"form": "METHOD", "owner": 13, "params": [{"name": "args"}], "path": ["__call__"]}
    ],
    "egglog.python.pyobject.call_extended": [
        {"form": "METHOD", "owner": 13, "params": [{"name": "args"}, {"name": "kwargs"}]}
    ],
    "egglog.python.pyobject.dict": [{"form": "CLASS_METHOD", "owner": 13, "params": [{"name": "keys_and_values"}]}],
    "egglog.python.pyobject.dict_update": [{"form": "METHOD", "owner": 13, "params": [{"name": "keys_and_values"}]}],
    "egglog.python.pyobject.from_int": [{"form": "CLASS_METHOD", "owner": 13, "params": [{"name": "i"}]}],
    "egglog.python.pyobject.from_string": [{"form": "CLASS_METHOD", "owner": 13, "params": [{"name": "s"}]}],
})

_TYPE_ORDER = [
    "BigInt",
    "bool",
    "i64",
    "String",
    "Unit",
    "BigRat",
    "f64",
    "Map",
    "Pair",
    "Vec",
    "MultiSet",
    "Set",
    "Maybe",
    "Rational",
    "PyObject",
]
_TYPE_PARAMS = {"Map": "K V", "Maybe": "T", "MultiSet": "T", "Pair": "L R", "Set": "T", "Vec": "T"}
# These positions are stable local owner/default roots, not native signature indices.
_SORTS = [
    "BigInt",
    "BigRat",
    "bool",
    "f64",
    "i64",
    "Map",
    "MultiSet",
    "Pair",
    "Set",
    "String",
    "Vec",
    "Maybe",
    "Rational",
    "PyObject",
]
DATA: dict[str, Any] = {
    "typeOrder": _TYPE_ORDER,
    "callables": dict(sorted(_CALLABLES.items())),
    "fragment": {
        "nodes": [{"call": {"func": "egglog.python.pyobject.dict"}, "sortId": 13} for _ in range(2)],
        "sorts": [
            {
                "primitive": {
                    **(
                        {"args": list(range(len(_SORTS), len(_SORTS) + len(_TYPE_PARAMS[_name].split())))}
                        if _name in _TYPE_PARAMS
                        else {}
                    ),
                    "name": _name,
                }
            }
            for _name in _SORTS
        ]
        + [{"var": 0}, {"var": 1}],
    },
    "types": {
        _name: {
            "path": ["egglog", "builtins", "Bool" if _name == "bool" else _name],
            **({"typeParams": _TYPE_PARAMS[_name].split()} if _name in _TYPE_PARAMS else {}),
        }
        for _name in sorted(_TYPE_ORDER)
    },
}
