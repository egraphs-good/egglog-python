"""Python-only builtin surface policy; semantic signatures come from the native catalog."""

import json

DATA = json.loads(r"""{
  "typeOrder": ["BigInt", "bool", "i64", "String", "Unit", "BigRat", "f64", "Map", "Pair", "Vec", "MultiSet", "Set", "Maybe", "Rational", "PyObject"],
  "callables": {
    "egglog.core.bigint.add": [
      {
        "form": "METHOD",
        "owner": 0,
        "params": [
          {
            "name": "other"
          }
        ],
        "path": [
          "__add__"
        ]
      }
    ],
    "egglog.core.bigint.bitand": [
      {
        "form": "METHOD",
        "owner": 0,
        "params": [
          {
            "name": "other"
          }
        ],
        "path": [
          "__and__"
        ]
      }
    ],
    "egglog.core.bigint.bitor": [
      {
        "form": "METHOD",
        "owner": 0,
        "params": [
          {
            "name": "other"
          }
        ],
        "path": [
          "__or__"
        ]
      }
    ],
    "egglog.core.bigint.bits": [
      {
        "form": "METHOD",
        "owner": 0
      }
    ],
    "egglog.core.bigint.bitxor": [
      {
        "form": "METHOD",
        "owner": 0,
        "params": [
          {
            "name": "other"
          }
        ],
        "path": [
          "__xor__"
        ]
      }
    ],
    "egglog.core.bigint.bool_eq": [
      {
        "form": "METHOD",
        "owner": 0,
        "params": [
          {
            "name": "other"
          }
        ]
      }
    ],
    "egglog.core.bigint.bool_ge": [
      {
        "form": "METHOD",
        "owner": 0,
        "params": [
          {
            "name": "other"
          }
        ]
      }
    ],
    "egglog.core.bigint.bool_gt": [
      {
        "form": "METHOD",
        "owner": 0,
        "params": [
          {
            "name": "other"
          }
        ]
      }
    ],
    "egglog.core.bigint.bool_le": [
      {
        "form": "METHOD",
        "owner": 0,
        "params": [
          {
            "name": "other"
          }
        ]
      }
    ],
    "egglog.core.bigint.bool_lt": [
      {
        "form": "METHOD",
        "owner": 0,
        "params": [
          {
            "name": "other"
          }
        ]
      }
    ],
    "egglog.core.bigint.div": [
      {
        "form": "METHOD",
        "owner": 0,
        "params": [
          {
            "name": "other"
          }
        ],
        "path": [
          "__truediv__"
        ]
      }
    ],
    "egglog.core.bigint.from_i64": [
      {
        "form": "INITIALIZER",
        "owner": 0,
        "params": [
          {
            "name": "value"
          }
        ]
      }
    ],
    "egglog.core.bigint.from_string": [
      {
        "form": "CLASS_METHOD",
        "owner": 0,
        "params": [
          {
            "name": "s"
          }
        ]
      }
    ],
    "egglog.core.bigint.ge": [
      {
        "form": "METHOD",
        "owner": 0,
        "params": [
          {
            "name": "other"
          }
        ],
        "path": [
          "__ge__"
        ]
      }
    ],
    "egglog.core.bigint.gt": [
      {
        "form": "METHOD",
        "owner": 0,
        "params": [
          {
            "name": "other"
          }
        ],
        "path": [
          "__gt__"
        ]
      }
    ],
    "egglog.core.bigint.le": [
      {
        "form": "METHOD",
        "owner": 0,
        "params": [
          {
            "name": "other"
          }
        ],
        "path": [
          "__le__"
        ]
      }
    ],
    "egglog.core.bigint.lt": [
      {
        "form": "METHOD",
        "owner": 0,
        "params": [
          {
            "name": "other"
          }
        ],
        "path": [
          "__lt__"
        ]
      }
    ],
    "egglog.core.bigint.max": [
      {
        "form": "METHOD",
        "owner": 0,
        "params": [
          {
            "name": "other"
          }
        ]
      }
    ],
    "egglog.core.bigint.min": [
      {
        "form": "METHOD",
        "owner": 0,
        "params": [
          {
            "name": "other"
          }
        ]
      }
    ],
    "egglog.core.bigint.mul": [
      {
        "form": "METHOD",
        "owner": 0,
        "params": [
          {
            "name": "other"
          }
        ],
        "path": [
          "__mul__"
        ]
      }
    ],
    "egglog.core.bigint.not": [
      {
        "form": "METHOD",
        "owner": 0,
        "path": [
          "__invert__"
        ]
      }
    ],
    "egglog.core.bigint.rem": [
      {
        "form": "METHOD",
        "owner": 0,
        "params": [
          {
            "name": "other"
          }
        ],
        "path": [
          "__mod__"
        ]
      }
    ],
    "egglog.core.bigint.shl": [
      {
        "form": "METHOD",
        "owner": 0,
        "params": [
          {
            "name": "other"
          }
        ],
        "path": [
          "__lshift__"
        ]
      }
    ],
    "egglog.core.bigint.shr": [
      {
        "form": "METHOD",
        "owner": 0,
        "params": [
          {
            "name": "other"
          }
        ],
        "path": [
          "__rshift__"
        ]
      }
    ],
    "egglog.core.bigint.sub": [
      {
        "form": "METHOD",
        "owner": 0,
        "params": [
          {
            "name": "other"
          }
        ],
        "path": [
          "__sub__"
        ]
      }
    ],
    "egglog.core.bigint.to_string": [
      {
        "form": "METHOD",
        "owner": 0
      }
    ],
    "egglog.core.bigrat.abs": [
      {
        "form": "METHOD",
        "owner": 1,
        "path": [
          "__abs__"
        ]
      }
    ],
    "egglog.core.bigrat.add": [
      {
        "form": "METHOD",
        "owner": 1,
        "params": [
          {
            "name": "other"
          }
        ],
        "path": [
          "__add__"
        ]
      }
    ],
    "egglog.core.bigrat.cbrt": [
      {
        "form": "METHOD",
        "owner": 1
      }
    ],
    "egglog.core.bigrat.ceil": [
      {
        "form": "METHOD",
        "owner": 1
      }
    ],
    "egglog.core.bigrat.denom": [
      {
        "form": "PROPERTY",
        "owner": 1
      }
    ],
    "egglog.core.bigrat.div": [
      {
        "form": "METHOD",
        "owner": 1,
        "params": [
          {
            "name": "other"
          }
        ],
        "path": [
          "__truediv__"
        ]
      }
    ],
    "egglog.core.bigrat.floor": [
      {
        "form": "METHOD",
        "owner": 1
      }
    ],
    "egglog.core.bigrat.ge": [
      {
        "form": "METHOD",
        "owner": 1,
        "params": [
          {
            "name": "other"
          }
        ],
        "path": [
          "__ge__"
        ]
      }
    ],
    "egglog.core.bigrat.gt": [
      {
        "form": "METHOD",
        "owner": 1,
        "params": [
          {
            "name": "other"
          }
        ],
        "path": [
          "__gt__"
        ]
      }
    ],
    "egglog.core.bigrat.le": [
      {
        "form": "METHOD",
        "owner": 1,
        "params": [
          {
            "name": "other"
          }
        ],
        "path": [
          "__le__"
        ]
      }
    ],
    "egglog.core.bigrat.log": [
      {
        "form": "METHOD",
        "owner": 1
      }
    ],
    "egglog.core.bigrat.lt": [
      {
        "form": "METHOD",
        "owner": 1,
        "params": [
          {
            "name": "other"
          }
        ],
        "path": [
          "__lt__"
        ]
      }
    ],
    "egglog.core.bigrat.make": [
      {
        "form": "INITIALIZER",
        "owner": 1,
        "params": [
          {
            "name": "num"
          },
          {
            "name": "den"
          }
        ]
      }
    ],
    "egglog.core.bigrat.max": [
      {
        "form": "METHOD",
        "owner": 1,
        "params": [
          {
            "name": "other"
          }
        ]
      }
    ],
    "egglog.core.bigrat.min": [
      {
        "form": "METHOD",
        "owner": 1,
        "params": [
          {
            "name": "other"
          }
        ]
      }
    ],
    "egglog.core.bigrat.mul": [
      {
        "form": "METHOD",
        "owner": 1,
        "params": [
          {
            "name": "other"
          }
        ],
        "path": [
          "__mul__"
        ]
      }
    ],
    "egglog.core.bigrat.neg": [
      {
        "form": "METHOD",
        "owner": 1,
        "path": [
          "__neg__"
        ]
      }
    ],
    "egglog.core.bigrat.numer": [
      {
        "form": "PROPERTY",
        "owner": 1
      }
    ],
    "egglog.core.bigrat.pow": [
      {
        "form": "METHOD",
        "owner": 1,
        "params": [
          {
            "name": "other"
          }
        ],
        "path": [
          "__pow__"
        ]
      }
    ],
    "egglog.core.bigrat.round": [
      {
        "form": "METHOD",
        "owner": 1
      }
    ],
    "egglog.core.bigrat.sqrt": [
      {
        "form": "METHOD",
        "owner": 1
      }
    ],
    "egglog.core.bigrat.sub": [
      {
        "form": "METHOD",
        "owner": 1,
        "params": [
          {
            "name": "other"
          }
        ],
        "path": [
          "__sub__"
        ]
      }
    ],
    "egglog.core.bigrat.to_f64": [
      {
        "form": "METHOD",
        "owner": 1
      }
    ],
    "egglog.core.bigrat.to_i64": [
      {
        "form": "METHOD",
        "owner": 1
      }
    ],
    "egglog.core.bool.and": [
      {
        "form": "METHOD",
        "owner": 2,
        "params": [
          {
            "name": "other"
          }
        ],
        "path": [
          "__and__"
        ]
      }
    ],
    "egglog.core.bool.implies": [
      {
        "form": "METHOD",
        "owner": 2,
        "params": [
          {
            "name": "other"
          }
        ]
      }
    ],
    "egglog.core.bool.not": [
      {
        "form": "METHOD",
        "owner": 2,
        "path": [
          "__invert__"
        ]
      }
    ],
    "egglog.core.bool.or": [
      {
        "form": "METHOD",
        "owner": 2,
        "params": [
          {
            "name": "other"
          }
        ],
        "path": [
          "__or__"
        ],
        "tailRepetitions": 2
      }
    ],
    "egglog.core.bool.xor": [
      {
        "form": "METHOD",
        "owner": 2,
        "params": [
          {
            "name": "other"
          }
        ],
        "path": [
          "__xor__"
        ]
      }
    ],
    "egglog.core.f64.abs": [
      {
        "form": "METHOD",
        "owner": 3,
        "path": [
          "__abs__"
        ]
      }
    ],
    "egglog.core.f64.add": [
      {
        "form": "METHOD",
        "owner": 3,
        "params": [
          {
            "name": "other"
          }
        ],
        "path": [
          "__add__"
        ]
      }
    ],
    "egglog.core.f64.div": [
      {
        "form": "METHOD",
        "owner": 3,
        "params": [
          {
            "name": "other"
          }
        ],
        "path": [
          "__truediv__"
        ]
      }
    ],
    "egglog.core.f64.exp": [
      {
        "form": "METHOD",
        "owner": 3
      }
    ],
    "egglog.core.f64.from_i64": [
      {
        "form": "CLASS_METHOD",
        "owner": 3,
        "params": [
          {
            "name": "i"
          }
        ]
      }
    ],
    "egglog.core.f64.ge": [
      {
        "form": "METHOD",
        "owner": 3,
        "params": [
          {
            "name": "other"
          }
        ],
        "path": [
          "__ge__"
        ]
      }
    ],
    "egglog.core.f64.gt": [
      {
        "form": "METHOD",
        "owner": 3,
        "params": [
          {
            "name": "other"
          }
        ],
        "path": [
          "__gt__"
        ]
      }
    ],
    "egglog.core.f64.le": [
      {
        "form": "METHOD",
        "owner": 3,
        "params": [
          {
            "name": "other"
          }
        ],
        "path": [
          "__le__"
        ]
      }
    ],
    "egglog.core.f64.log": [
      {
        "form": "METHOD",
        "owner": 3
      }
    ],
    "egglog.core.f64.lt": [
      {
        "form": "METHOD",
        "owner": 3,
        "params": [
          {
            "name": "other"
          }
        ],
        "path": [
          "__lt__"
        ]
      }
    ],
    "egglog.core.f64.max": [
      {
        "form": "METHOD",
        "owner": 3,
        "params": [
          {
            "name": "other"
          }
        ]
      }
    ],
    "egglog.core.f64.min": [
      {
        "form": "METHOD",
        "owner": 3,
        "params": [
          {
            "name": "other"
          }
        ]
      }
    ],
    "egglog.core.f64.mul": [
      {
        "form": "METHOD",
        "owner": 3,
        "params": [
          {
            "name": "other"
          }
        ],
        "path": [
          "__mul__"
        ]
      }
    ],
    "egglog.core.f64.neg": [
      {
        "form": "METHOD",
        "owner": 3,
        "path": [
          "__neg__"
        ]
      }
    ],
    "egglog.core.f64.pow": [
      {
        "form": "METHOD",
        "owner": 3,
        "params": [
          {
            "name": "other"
          }
        ],
        "path": [
          "__pow__"
        ]
      }
    ],
    "egglog.core.f64.rem": [
      {
        "form": "METHOD",
        "owner": 3,
        "params": [
          {
            "name": "other"
          }
        ],
        "path": [
          "__mod__"
        ]
      }
    ],
    "egglog.core.f64.sqrt": [
      {
        "form": "METHOD",
        "owner": 3
      }
    ],
    "egglog.core.f64.sub": [
      {
        "form": "METHOD",
        "owner": 3,
        "params": [
          {
            "name": "other"
          }
        ],
        "path": [
          "__sub__"
        ]
      }
    ],
    "egglog.core.f64.to_i64": [
      {
        "form": "METHOD",
        "owner": 3
      }
    ],
    "egglog.core.f64.to_string": [
      {
        "form": "METHOD",
        "owner": 3
      }
    ],
    "egglog.core.i64.abs": [
      {
        "form": "METHOD",
        "owner": 4,
        "path": [
          "__abs__"
        ]
      }
    ],
    "egglog.core.i64.add": [
      {
        "form": "METHOD",
        "owner": 4,
        "params": [
          {
            "name": "other"
          }
        ],
        "path": [
          "__add__"
        ]
      }
    ],
    "egglog.core.i64.bitand": [
      {
        "form": "METHOD",
        "owner": 4,
        "params": [
          {
            "name": "other"
          }
        ],
        "path": [
          "__and__"
        ]
      }
    ],
    "egglog.core.i64.bitor": [
      {
        "form": "METHOD",
        "owner": 4,
        "params": [
          {
            "name": "other"
          }
        ],
        "path": [
          "__or__"
        ]
      }
    ],
    "egglog.core.i64.bitxor": [
      {
        "form": "METHOD",
        "owner": 4,
        "params": [
          {
            "name": "other"
          }
        ],
        "path": [
          "__xor__"
        ]
      }
    ],
    "egglog.core.i64.bool_ge": [
      {
        "form": "METHOD",
        "owner": 4,
        "params": [
          {
            "name": "other"
          }
        ]
      }
    ],
    "egglog.core.i64.bool_gt": [
      {
        "form": "METHOD",
        "owner": 4,
        "params": [
          {
            "name": "other"
          }
        ]
      }
    ],
    "egglog.core.i64.bool_le": [
      {
        "form": "METHOD",
        "owner": 4,
        "params": [
          {
            "name": "other"
          }
        ]
      }
    ],
    "egglog.core.i64.bool_lt": [
      {
        "form": "METHOD",
        "owner": 4,
        "params": [
          {
            "name": "other"
          }
        ]
      }
    ],
    "egglog.core.i64.div": [
      {
        "form": "METHOD",
        "owner": 4,
        "params": [
          {
            "name": "other"
          }
        ],
        "path": [
          "__truediv__"
        ]
      }
    ],
    "egglog.core.i64.ge": [
      {
        "form": "METHOD",
        "owner": 4,
        "params": [
          {
            "name": "other"
          }
        ],
        "path": [
          "__ge__"
        ]
      }
    ],
    "egglog.core.i64.gt": [
      {
        "form": "METHOD",
        "owner": 4,
        "params": [
          {
            "name": "other"
          }
        ],
        "path": [
          "__gt__"
        ]
      }
    ],
    "egglog.core.i64.le": [
      {
        "form": "METHOD",
        "owner": 4,
        "params": [
          {
            "name": "other"
          }
        ],
        "path": [
          "__le__"
        ]
      }
    ],
    "egglog.core.i64.log2": [
      {
        "form": "METHOD",
        "owner": 4
      }
    ],
    "egglog.core.i64.lt": [
      {
        "form": "METHOD",
        "owner": 4,
        "params": [
          {
            "name": "other"
          }
        ],
        "path": [
          "__lt__"
        ]
      }
    ],
    "egglog.core.i64.max": [
      {
        "form": "METHOD",
        "owner": 4,
        "params": [
          {
            "name": "other"
          }
        ]
      }
    ],
    "egglog.core.i64.min": [
      {
        "form": "METHOD",
        "owner": 4,
        "params": [
          {
            "name": "other"
          }
        ]
      }
    ],
    "egglog.core.i64.mul": [
      {
        "form": "METHOD",
        "owner": 4,
        "params": [
          {
            "name": "other"
          }
        ],
        "path": [
          "__mul__"
        ]
      }
    ],
    "egglog.core.i64.not": [
      {
        "form": "METHOD",
        "owner": 4,
        "path": [
          "__invert__"
        ]
      }
    ],
    "egglog.core.i64.rem": [
      {
        "form": "METHOD",
        "owner": 4,
        "params": [
          {
            "name": "other"
          }
        ],
        "path": [
          "__mod__"
        ]
      }
    ],
    "egglog.core.i64.shl": [
      {
        "form": "METHOD",
        "owner": 4,
        "params": [
          {
            "name": "other"
          }
        ],
        "path": [
          "__lshift__"
        ]
      }
    ],
    "egglog.core.i64.shr": [
      {
        "form": "METHOD",
        "owner": 4,
        "params": [
          {
            "name": "other"
          }
        ],
        "path": [
          "__rshift__"
        ]
      }
    ],
    "egglog.core.i64.sub": [
      {
        "form": "METHOD",
        "owner": 4,
        "params": [
          {
            "name": "other"
          }
        ],
        "path": [
          "__sub__"
        ]
      }
    ],
    "egglog.core.i64.to_string": [
      {
        "form": "METHOD",
        "owner": 4
      }
    ],
    "egglog.core.map.contains": [
      {
        "form": "METHOD",
        "owner": 5,
        "params": [
          {
            "name": "key"
          }
        ]
      }
    ],
    "egglog.core.map.empty": [
      {
        "form": "CLASS_METHOD",
        "owner": 5
      }
    ],
    "egglog.core.map.get": [
      {
        "form": "METHOD",
        "owner": 5,
        "params": [
          {
            "name": "key"
          }
        ],
        "path": [
          "__getitem__"
        ]
      }
    ],
    "egglog.core.map.insert": [
      {
        "form": "METHOD",
        "owner": 5,
        "params": [
          {
            "name": "key"
          },
          {
            "name": "value"
          }
        ]
      }
    ],
    "egglog.core.map.length": [
      {
        "form": "METHOD",
        "owner": 5
      }
    ],
    "egglog.core.map.not_contains": [
      {
        "form": "METHOD",
        "owner": 5,
        "params": [
          {
            "name": "key"
          }
        ]
      }
    ],
    "egglog.core.map.of": [
      {
        "form": "CLASS_METHOD",
        "owner": 5,
        "params": [
          {
            "name": "entries"
          }
        ],
        "path": [
          "from_pairs"
        ]
      }
    ],
    "egglog.core.map.remove": [
      {
        "form": "METHOD",
        "owner": 5,
        "params": [
          {
            "name": "key"
          }
        ]
      }
    ],
    "egglog.core.multiset.clear_index": [
      {
        "form": "METHOD",
        "owner": 6,
        "params": [
          {
            "name": "f"
          }
        ]
      }
    ],
    "egglog.core.multiset.contains": [
      {
        "form": "METHOD",
        "owner": 6,
        "params": [
          {
            "name": "value"
          }
        ]
      }
    ],
    "egglog.core.multiset.contains_swapped": [
      {
        "form": "FUNCTION",
        "params": [
          {
            "name": "x"
          },
          {
            "name": "xs"
          }
        ],
        "path": [
          "egglog",
          "builtins",
          "multiset_contains_swapped"
        ]
      }
    ],
    "egglog.core.multiset.count": [
      {
        "form": "METHOD",
        "owner": 6,
        "params": [
          {
            "name": "value"
          }
        ]
      }
    ],
    "egglog.core.multiset.fill_index": [
      {
        "form": "METHOD",
        "owner": 6,
        "params": [
          {
            "name": "f"
          }
        ]
      }
    ],
    "egglog.core.multiset.filter": [
      {
        "form": "METHOD",
        "owner": 6,
        "params": [
          {
            "name": "f"
          }
        ],
        "receiver": 1
      }
    ],
    "egglog.core.multiset.filter_not": [
      {
        "form": "METHOD",
        "owner": 6,
        "params": [
          {
            "name": "f"
          }
        ],
        "receiver": 1
      }
    ],
    "egglog.core.multiset.flat_map": [
      {
        "form": "FUNCTION",
        "params": [
          {
            "name": "f"
          },
          {
            "name": "xs"
          }
        ],
        "path": [
          "egglog",
          "builtins",
          "multiset_flat_map"
        ]
      }
    ],
    "egglog.core.multiset.insert": [
      {
        "form": "METHOD",
        "owner": 6,
        "params": [
          {
            "name": "value"
          }
        ]
      }
    ],
    "egglog.core.multiset.intersection": [
      {
        "form": "METHOD",
        "owner": 6,
        "params": [
          {
            "name": "other"
          }
        ],
        "path": [
          "__and__"
        ]
      }
    ],
    "egglog.core.multiset.length": [
      {
        "form": "METHOD",
        "owner": 6
      }
    ],
    "egglog.core.multiset.map": [
      {
        "form": "METHOD",
        "owner": 6,
        "params": [
          {
            "name": "f"
          }
        ],
        "receiver": 1
      }
    ],
    "egglog.core.multiset.not_contains": [
      {
        "form": "METHOD",
        "owner": 6,
        "params": [
          {
            "name": "value"
          }
        ]
      }
    ],
    "egglog.core.multiset.not_contains_swapped": [
      {
        "form": "FUNCTION",
        "params": [
          {
            "name": "x"
          },
          {
            "name": "xs"
          }
        ],
        "path": [
          "egglog",
          "builtins",
          "multiset_not_contains_swapped"
        ]
      }
    ],
    "egglog.core.multiset.of": [
      {
        "form": "INITIALIZER",
        "owner": 6,
        "params": [
          {
            "name": "args"
          }
        ]
      }
    ],
    "egglog.core.multiset.pick": [
      {
        "form": "METHOD",
        "owner": 6
      }
    ],
    "egglog.core.multiset.pick_max": [
      {
        "form": "METHOD",
        "owner": 6
      }
    ],
    "egglog.core.multiset.reduce": [
      {
        "form": "FUNCTION",
        "params": [
          {
            "name": "f"
          },
          {
            "name": "initial"
          },
          {
            "name": "xs"
          }
        ],
        "path": [
          "egglog",
          "builtins",
          "multiset_fold"
        ]
      }
    ],
    "egglog.core.multiset.remove": [
      {
        "form": "METHOD",
        "owner": 6,
        "params": [
          {
            "name": "value"
          }
        ]
      }
    ],
    "egglog.core.multiset.remove_swapped": [
      {
        "form": "FUNCTION",
        "params": [
          {
            "name": "x"
          },
          {
            "name": "xs"
          }
        ],
        "path": [
          "egglog",
          "builtins",
          "multiset_remove_swapped"
        ]
      }
    ],
    "egglog.core.multiset.reset_counts": [
      {
        "form": "METHOD",
        "owner": 6
      }
    ],
    "egglog.core.multiset.single": [
      {
        "form": "CLASS_METHOD",
        "owner": 6,
        "params": [
          {
            "name": "x"
          },
          {
            "name": "i"
          }
        ]
      }
    ],
    "egglog.core.multiset.subtract": [
      {
        "form": "METHOD",
        "owner": 6,
        "params": [
          {
            "name": "other"
          }
        ],
        "path": [
          "__sub__"
        ]
      }
    ],
    "egglog.core.multiset.subtract_swapped": [
      {
        "form": "FUNCTION",
        "params": [
          {
            "name": "x"
          },
          {
            "name": "xs"
          }
        ],
        "path": [
          "egglog",
          "builtins",
          "multiset_subtract_swapped"
        ]
      }
    ],
    "egglog.core.multiset.sum": [
      {
        "form": "METHOD",
        "owner": 6,
        "params": [
          {
            "name": "other"
          }
        ],
        "path": [
          "__add__"
        ]
      }
    ],
    "egglog.core.multiset.sum_multisets": [
      {
        "form": "CLASS_METHOD",
        "owner": 6,
        "params": [
          {
            "name": "xs"
          }
        ]
      }
    ],
    "egglog.core.multiset.union_values": [
      {
        "form": "METHOD",
        "owner": 6
      }
    ],
    "egglog.core.pair.first": [
      {
        "form": "PROPERTY",
        "owner": 7,
        "path": [
          "left"
        ]
      }
    ],
    "egglog.core.pair.make": [
      {
        "form": "INITIALIZER",
        "owner": 7,
        "params": [
          {
            "name": "left"
          },
          {
            "name": "right"
          }
        ]
      }
    ],
    "egglog.core.pair.second": [
      {
        "form": "PROPERTY",
        "owner": 7,
        "path": [
          "right"
        ]
      }
    ],
    "egglog.core.set.contains": [
      {
        "form": "METHOD",
        "owner": 8,
        "params": [
          {
            "name": "value"
          }
        ]
      }
    ],
    "egglog.core.set.diff": [
      {
        "form": "METHOD",
        "owner": 8,
        "params": [
          {
            "name": "other"
          }
        ],
        "path": [
          "__sub__"
        ]
      }
    ],
    "egglog.core.set.empty": [
      {
        "form": "CLASS_METHOD",
        "owner": 8
      }
    ],
    "egglog.core.set.insert": [
      {
        "form": "METHOD",
        "owner": 8,
        "params": [
          {
            "name": "value"
          }
        ]
      }
    ],
    "egglog.core.set.intersect": [
      {
        "form": "METHOD",
        "owner": 8,
        "params": [
          {
            "name": "other"
          }
        ],
        "path": [
          "__and__"
        ]
      }
    ],
    "egglog.core.set.length": [
      {
        "form": "METHOD",
        "owner": 8
      }
    ],
    "egglog.core.set.not_contains": [
      {
        "form": "METHOD",
        "owner": 8,
        "params": [
          {
            "name": "value"
          }
        ]
      }
    ],
    "egglog.core.set.of": [
      {
        "form": "INITIALIZER",
        "owner": 8,
        "params": [
          {
            "name": "args"
          }
        ]
      }
    ],
    "egglog.core.set.remove": [
      {
        "form": "METHOD",
        "owner": 8,
        "params": [
          {
            "name": "value"
          }
        ]
      }
    ],
    "egglog.core.set.union": [
      {
        "form": "METHOD",
        "owner": 8,
        "params": [
          {
            "name": "other"
          }
        ],
        "path": [
          "__or__"
        ]
      }
    ],
    "egglog.core.string.concat": [
      {
        "form": "FUNCTION",
        "params": [
          {
            "name": "strings"
          }
        ],
        "path": [
          "egglog",
          "builtins",
          "join"
        ]
      },
      {
        "form": "METHOD",
        "owner": 9,
        "params": [
          {
            "name": "other"
          }
        ],
        "path": [
          "__add__"
        ],
        "tailRepetitions": 2
      }
    ],
    "egglog.core.string.count_matches": [
      {
        "form": "FUNCTION",
        "params": [
          {
            "name": "s"
          },
          {
            "name": "pattern"
          }
        ],
        "path": [
          "egglog",
          "builtins",
          "count_matches"
        ]
      }
    ],
    "egglog.core.string.replace": [
      {
        "form": "METHOD",
        "owner": 9,
        "params": [
          {
            "name": "old"
          },
          {
            "name": "new"
          }
        ]
      }
    ],
    "egglog.core.vec.append": [
      {
        "form": "METHOD",
        "owner": 10,
        "params": [
          {
            "name": "others"
          }
        ]
      }
    ],
    "egglog.core.vec.contains": [
      {
        "form": "METHOD",
        "owner": 10,
        "params": [
          {
            "name": "value"
          }
        ]
      }
    ],
    "egglog.core.vec.empty": [
      {
        "form": "CLASS_METHOD",
        "owner": 10
      }
    ],
    "egglog.core.vec.get": [
      {
        "form": "METHOD",
        "owner": 10,
        "params": [
          {
            "name": "index"
          }
        ],
        "path": [
          "__getitem__"
        ]
      }
    ],
    "egglog.core.vec.length": [
      {
        "form": "METHOD",
        "owner": 10
      }
    ],
    "egglog.core.vec.map": [
      {
        "form": "METHOD",
        "owner": 10,
        "params": [
          {
            "name": "f"
          }
        ],
        "receiver": 1
      }
    ],
    "egglog.core.vec.not_contains": [
      {
        "form": "METHOD",
        "owner": 10,
        "params": [
          {
            "name": "value"
          }
        ]
      }
    ],
    "egglog.core.vec.of": [
      {
        "form": "INITIALIZER",
        "owner": 10,
        "params": [
          {
            "name": "args"
          }
        ]
      }
    ],
    "egglog.core.vec.pop": [
      {
        "form": "METHOD",
        "owner": 10
      }
    ],
    "egglog.core.vec.push": [
      {
        "form": "METHOD",
        "owner": 10,
        "params": [
          {
            "name": "value"
          }
        ]
      }
    ],
    "egglog.core.vec.range": [
      {
        "form": "METHOD",
        "owner": 4
      }
    ],
    "egglog.core.vec.remove": [
      {
        "form": "METHOD",
        "owner": 10,
        "params": [
          {
            "name": "index"
          }
        ]
      }
    ],
    "egglog.core.vec.set": [
      {
        "form": "METHOD",
        "owner": 10,
        "params": [
          {
            "name": "index"
          },
          {
            "name": "value"
          }
        ]
      }
    ],
    "egglog.core.vec.union": [
      {
        "form": "METHOD",
        "owner": 10,
        "params": [
          {
            "name": "other"
          }
        ],
        "path": [
          "__or__"
        ]
      }
    ],
    "egglog.experimental.f64.is_finite": [
      {
        "form": "METHOD",
        "owner": 3
      }
    ],
    "egglog.experimental.map.fold_kv": [
      {
        "form": "FUNCTION",
        "params": [
          {
            "name": "f"
          },
          {
            "name": "initial"
          },
          {
            "name": "xs"
          }
        ],
        "path": [
          "egglog",
          "builtins",
          "map_fold_kv"
        ]
      }
    ],
    "egglog.experimental.maybe.catch": [
      {
        "form": "FUNCTION",
        "params": [
          {
            "name": "f"
          }
        ],
        "path": [
          "egglog",
          "builtins",
          "catch"
        ]
      }
    ],
    "egglog.experimental.maybe.match": [
      {
        "form": "METHOD",
        "owner": 11,
        "params": [
          {
            "name": "f"
          },
          {
            "name": "n"
          }
        ]
      }
    ],
    "egglog.experimental.maybe.none": [
      {
        "form": "CLASS_METHOD",
        "owner": 11
      }
    ],
    "egglog.experimental.maybe.some": [
      {
        "form": "CLASS_METHOD",
        "owner": 11,
        "params": [
          {
            "name": "value"
          }
        ]
      }
    ],
    "egglog.experimental.maybe.unwrap": [
      {
        "form": "METHOD",
        "owner": 11
      }
    ],
    "egglog.experimental.maybe.unwrap_or": [
      {
        "form": "METHOD",
        "owner": 11,
        "params": [
          {
            "name": "default"
          }
        ]
      }
    ],
    "egglog.experimental.rational.abs": [
      {
        "form": "METHOD",
        "owner": 12,
        "path": [
          "__abs__"
        ]
      }
    ],
    "egglog.experimental.rational.add": [
      {
        "form": "METHOD",
        "owner": 12,
        "params": [
          {
            "name": "other"
          }
        ],
        "path": [
          "__add__"
        ]
      }
    ],
    "egglog.experimental.rational.cbrt": [
      {
        "form": "METHOD",
        "owner": 12
      }
    ],
    "egglog.experimental.rational.ceil": [
      {
        "form": "METHOD",
        "owner": 12
      }
    ],
    "egglog.experimental.rational.denom": [
      {
        "form": "PROPERTY",
        "owner": 12
      }
    ],
    "egglog.experimental.rational.div": [
      {
        "form": "METHOD",
        "owner": 12,
        "params": [
          {
            "name": "other"
          }
        ],
        "path": [
          "__truediv__"
        ]
      }
    ],
    "egglog.experimental.rational.floor": [
      {
        "form": "METHOD",
        "owner": 12
      }
    ],
    "egglog.experimental.rational.ge": [
      {
        "form": "METHOD",
        "owner": 12,
        "params": [
          {
            "name": "other"
          }
        ],
        "path": [
          "__ge__"
        ]
      }
    ],
    "egglog.experimental.rational.gt": [
      {
        "form": "METHOD",
        "owner": 12,
        "params": [
          {
            "name": "other"
          }
        ],
        "path": [
          "__gt__"
        ]
      }
    ],
    "egglog.experimental.rational.le": [
      {
        "form": "METHOD",
        "owner": 12,
        "params": [
          {
            "name": "other"
          }
        ],
        "path": [
          "__le__"
        ]
      }
    ],
    "egglog.experimental.rational.log": [
      {
        "form": "METHOD",
        "owner": 12
      }
    ],
    "egglog.experimental.rational.lt": [
      {
        "form": "METHOD",
        "owner": 12,
        "params": [
          {
            "name": "other"
          }
        ],
        "path": [
          "__lt__"
        ]
      }
    ],
    "egglog.experimental.rational.make": [
      {
        "form": "INITIALIZER",
        "owner": 12,
        "params": [
          {
            "name": "num"
          },
          {
            "name": "den"
          }
        ]
      }
    ],
    "egglog.experimental.rational.max": [
      {
        "form": "METHOD",
        "owner": 12,
        "params": [
          {
            "name": "other"
          }
        ]
      }
    ],
    "egglog.experimental.rational.min": [
      {
        "form": "METHOD",
        "owner": 12,
        "params": [
          {
            "name": "other"
          }
        ]
      }
    ],
    "egglog.experimental.rational.mul": [
      {
        "form": "METHOD",
        "owner": 12,
        "params": [
          {
            "name": "other"
          }
        ],
        "path": [
          "__mul__"
        ]
      }
    ],
    "egglog.experimental.rational.neg": [
      {
        "form": "METHOD",
        "owner": 12,
        "path": [
          "__neg__"
        ]
      }
    ],
    "egglog.experimental.rational.numer": [
      {
        "form": "PROPERTY",
        "owner": 12
      }
    ],
    "egglog.experimental.rational.pow": [
      {
        "form": "METHOD",
        "owner": 12,
        "params": [
          {
            "name": "other"
          }
        ],
        "path": [
          "__pow__"
        ]
      }
    ],
    "egglog.experimental.rational.round": [
      {
        "form": "METHOD",
        "owner": 12
      }
    ],
    "egglog.experimental.rational.sqrt": [
      {
        "form": "METHOD",
        "owner": 12
      }
    ],
    "egglog.experimental.rational.sub": [
      {
        "form": "METHOD",
        "owner": 12,
        "params": [
          {
            "name": "other"
          }
        ],
        "path": [
          "__sub__"
        ]
      }
    ],
    "egglog.experimental.rational.to_f64": [
      {
        "form": "METHOD",
        "owner": 12
      }
    ],
    "egglog.python.pyobject.call": [
      {
        "form": "METHOD",
        "owner": 13,
        "params": [
          {
            "name": "args"
          }
        ],
        "path": [
          "__call__"
        ]
      }
    ],
    "egglog.python.pyobject.call_extended": [
      {
        "form": "METHOD",
        "owner": 13,
        "params": [
          {
            "name": "args"
          },
          {
            "name": "kwargs"
          }
        ]
      }
    ],
    "egglog.python.pyobject.dict": [
      {
        "form": "CLASS_METHOD",
        "owner": 13,
        "params": [
          {
            "name": "keys_and_values"
          }
        ]
      }
    ],
    "egglog.python.pyobject.dict_update": [
      {
        "form": "METHOD",
        "owner": 13,
        "params": [
          {
            "name": "keys_and_values"
          }
        ]
      }
    ],
    "egglog.python.pyobject.eval": [
      {
        "form": "FUNCTION",
        "params": [
          {
            "name": "code"
          },
          {
            "default": 0,
            "name": "globals_"
          },
          {
            "default": 0,
            "name": "locals_"
          }
        ],
        "path": [
          "egglog",
          "builtins",
          "py_eval"
        ]
      }
    ],
    "egglog.python.pyobject.exec": [
      {
        "form": "FUNCTION",
        "params": [
          {
            "name": "code"
          },
          {
            "default": 1,
            "name": "globals_"
          },
          {
            "default": 1,
            "name": "locals_"
          }
        ],
        "path": [
          "egglog",
          "builtins",
          "py_exec"
        ]
      }
    ],
    "egglog.python.pyobject.from_int": [
      {
        "form": "CLASS_METHOD",
        "owner": 13,
        "params": [
          {
            "name": "i"
          }
        ]
      }
    ],
    "egglog.python.pyobject.from_string": [
      {
        "form": "CLASS_METHOD",
        "owner": 13,
        "params": [
          {
            "name": "s"
          }
        ]
      }
    ],
    "egglog.python.pyobject.to_bool": [
      {
        "form": "METHOD",
        "owner": 13
      }
    ],
    "egglog.python.pyobject.to_string": [
      {
        "form": "METHOD",
        "owner": 13
      }
    ]
  },
  "fragment": {
    "nodes": [
      {
        "call": {
          "func": "egglog.python.pyobject.dict"
        },
        "sortId": 13
      },
      {
        "call": {
          "func": "egglog.python.pyobject.dict"
        },
        "sortId": 13
      }
    ],
    "sorts": [
      {
        "primitive": {
          "name": "BigInt"
        }
      },
      {
        "primitive": {
          "name": "BigRat"
        }
      },
      {
        "primitive": {
          "name": "bool"
        }
      },
      {
        "primitive": {
          "name": "f64"
        }
      },
      {
        "primitive": {
          "name": "i64"
        }
      },
      {
        "primitive": {
          "args": [
            14,
            15
          ],
          "name": "Map"
        }
      },
      {
        "primitive": {
          "args": [
            14
          ],
          "name": "MultiSet"
        }
      },
      {
        "primitive": {
          "args": [
            14,
            15
          ],
          "name": "Pair"
        }
      },
      {
        "primitive": {
          "args": [
            14
          ],
          "name": "Set"
        }
      },
      {
        "primitive": {
          "name": "String"
        }
      },
      {
        "primitive": {
          "args": [
            14
          ],
          "name": "Vec"
        }
      },
      {
        "primitive": {
          "args": [
            14
          ],
          "name": "Maybe"
        }
      },
      {
        "primitive": {
          "name": "Rational"
        }
      },
      {
        "primitive": {
          "name": "PyObject"
        }
      },
      {
        "var": 0
      },
      {
        "var": 1
      }
    ]
  },
  "types": {
    "BigInt": {
      "path": [
        "egglog",
        "builtins",
        "BigInt"
      ]
    },
    "BigRat": {
      "path": [
        "egglog",
        "builtins",
        "BigRat"
      ]
    },
    "Map": {
      "path": [
        "egglog",
        "builtins",
        "Map"
      ],
      "typeParams": [
        "K",
        "V"
      ]
    },
    "Maybe": {
      "path": [
        "egglog",
        "builtins",
        "Maybe"
      ],
      "typeParams": [
        "T"
      ]
    },
    "MultiSet": {
      "path": [
        "egglog",
        "builtins",
        "MultiSet"
      ],
      "typeParams": [
        "T"
      ]
    },
    "Pair": {
      "path": [
        "egglog",
        "builtins",
        "Pair"
      ],
      "typeParams": [
        "L",
        "R"
      ]
    },
    "PyObject": {
      "path": [
        "egglog",
        "builtins",
        "PyObject"
      ]
    },
    "Rational": {
      "path": [
        "egglog",
        "builtins",
        "Rational"
      ]
    },
    "Set": {
      "path": [
        "egglog",
        "builtins",
        "Set"
      ],
      "typeParams": [
        "T"
      ]
    },
    "String": {
      "path": [
        "egglog",
        "builtins",
        "String"
      ]
    },
    "Unit": {
      "path": [
        "egglog",
        "builtins",
        "Unit"
      ]
    },
    "Vec": {
      "path": [
        "egglog",
        "builtins",
        "Vec"
      ],
      "typeParams": [
        "T"
      ]
    },
    "bool": {
      "path": [
        "egglog",
        "builtins",
        "Bool"
      ]
    },
    "f64": {
      "path": [
        "egglog",
        "builtins",
        "f64"
      ]
    },
    "i64": {
      "path": [
        "egglog",
        "builtins",
        "i64"
      ]
    }
  }
}""")
