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
            "coreInput": 1,
            "name": "other"
          }
        ],
        "path": [
          "__add__"
        ],
        "receiver": 0
      }
    ],
    "egglog.core.bigint.bitand": [
      {
        "form": "METHOD",
        "owner": 0,
        "params": [
          {
            "coreInput": 1,
            "name": "other"
          }
        ],
        "path": [
          "__and__"
        ],
        "receiver": 0
      }
    ],
    "egglog.core.bigint.bitor": [
      {
        "form": "METHOD",
        "owner": 0,
        "params": [
          {
            "coreInput": 1,
            "name": "other"
          }
        ],
        "path": [
          "__or__"
        ],
        "receiver": 0
      }
    ],
    "egglog.core.bigint.bits": [
      {
        "form": "METHOD",
        "owner": 0,
        "path": [
          "bits"
        ],
        "receiver": 0
      }
    ],
    "egglog.core.bigint.bitxor": [
      {
        "form": "METHOD",
        "owner": 0,
        "params": [
          {
            "coreInput": 1,
            "name": "other"
          }
        ],
        "path": [
          "__xor__"
        ],
        "receiver": 0
      }
    ],
    "egglog.core.bigint.bool_eq": [
      {
        "form": "METHOD",
        "owner": 0,
        "params": [
          {
            "coreInput": 1,
            "name": "other"
          }
        ],
        "path": [
          "bool_eq"
        ],
        "receiver": 0
      }
    ],
    "egglog.core.bigint.bool_ge": [
      {
        "form": "METHOD",
        "owner": 0,
        "params": [
          {
            "coreInput": 1,
            "name": "other"
          }
        ],
        "path": [
          "bool_ge"
        ],
        "receiver": 0
      }
    ],
    "egglog.core.bigint.bool_gt": [
      {
        "form": "METHOD",
        "owner": 0,
        "params": [
          {
            "coreInput": 1,
            "name": "other"
          }
        ],
        "path": [
          "bool_gt"
        ],
        "receiver": 0
      }
    ],
    "egglog.core.bigint.bool_le": [
      {
        "form": "METHOD",
        "owner": 0,
        "params": [
          {
            "coreInput": 1,
            "name": "other"
          }
        ],
        "path": [
          "bool_le"
        ],
        "receiver": 0
      }
    ],
    "egglog.core.bigint.bool_lt": [
      {
        "form": "METHOD",
        "owner": 0,
        "params": [
          {
            "coreInput": 1,
            "name": "other"
          }
        ],
        "path": [
          "bool_lt"
        ],
        "receiver": 0
      }
    ],
    "egglog.core.bigint.div": [
      {
        "form": "METHOD",
        "owner": 0,
        "params": [
          {
            "coreInput": 1,
            "name": "other"
          }
        ],
        "path": [
          "__truediv__"
        ],
        "receiver": 0
      }
    ],
    "egglog.core.bigint.from_i64": [
      {
        "form": "INITIALIZER",
        "owner": 0,
        "params": [
          {
            "coreInput": 0,
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
            "coreInput": 0,
            "name": "s"
          }
        ],
        "path": [
          "from_string"
        ]
      }
    ],
    "egglog.core.bigint.ge": [
      {
        "form": "METHOD",
        "owner": 0,
        "params": [
          {
            "coreInput": 1,
            "name": "other"
          }
        ],
        "path": [
          "__ge__"
        ],
        "receiver": 0
      }
    ],
    "egglog.core.bigint.gt": [
      {
        "form": "METHOD",
        "owner": 0,
        "params": [
          {
            "coreInput": 1,
            "name": "other"
          }
        ],
        "path": [
          "__gt__"
        ],
        "receiver": 0
      }
    ],
    "egglog.core.bigint.le": [
      {
        "form": "METHOD",
        "owner": 0,
        "params": [
          {
            "coreInput": 1,
            "name": "other"
          }
        ],
        "path": [
          "__le__"
        ],
        "receiver": 0
      }
    ],
    "egglog.core.bigint.lt": [
      {
        "form": "METHOD",
        "owner": 0,
        "params": [
          {
            "coreInput": 1,
            "name": "other"
          }
        ],
        "path": [
          "__lt__"
        ],
        "receiver": 0
      }
    ],
    "egglog.core.bigint.max": [
      {
        "form": "METHOD",
        "owner": 0,
        "params": [
          {
            "coreInput": 1,
            "name": "other"
          }
        ],
        "path": [
          "max"
        ],
        "receiver": 0
      }
    ],
    "egglog.core.bigint.min": [
      {
        "form": "METHOD",
        "owner": 0,
        "params": [
          {
            "coreInput": 1,
            "name": "other"
          }
        ],
        "path": [
          "min"
        ],
        "receiver": 0
      }
    ],
    "egglog.core.bigint.mul": [
      {
        "form": "METHOD",
        "owner": 0,
        "params": [
          {
            "coreInput": 1,
            "name": "other"
          }
        ],
        "path": [
          "__mul__"
        ],
        "receiver": 0
      }
    ],
    "egglog.core.bigint.not": [
      {
        "form": "METHOD",
        "owner": 0,
        "path": [
          "__invert__"
        ],
        "receiver": 0
      }
    ],
    "egglog.core.bigint.rem": [
      {
        "form": "METHOD",
        "owner": 0,
        "params": [
          {
            "coreInput": 1,
            "name": "other"
          }
        ],
        "path": [
          "__mod__"
        ],
        "receiver": 0
      }
    ],
    "egglog.core.bigint.shl": [
      {
        "form": "METHOD",
        "owner": 0,
        "params": [
          {
            "coreInput": 1,
            "name": "other"
          }
        ],
        "path": [
          "__lshift__"
        ],
        "receiver": 0
      }
    ],
    "egglog.core.bigint.shr": [
      {
        "form": "METHOD",
        "owner": 0,
        "params": [
          {
            "coreInput": 1,
            "name": "other"
          }
        ],
        "path": [
          "__rshift__"
        ],
        "receiver": 0
      }
    ],
    "egglog.core.bigint.sub": [
      {
        "form": "METHOD",
        "owner": 0,
        "params": [
          {
            "coreInput": 1,
            "name": "other"
          }
        ],
        "path": [
          "__sub__"
        ],
        "receiver": 0
      }
    ],
    "egglog.core.bigint.to_string": [
      {
        "form": "METHOD",
        "owner": 0,
        "path": [
          "to_string"
        ],
        "receiver": 0
      }
    ],
    "egglog.core.bigrat.abs": [
      {
        "form": "METHOD",
        "owner": 1,
        "path": [
          "__abs__"
        ],
        "receiver": 0
      }
    ],
    "egglog.core.bigrat.add": [
      {
        "form": "METHOD",
        "owner": 1,
        "params": [
          {
            "coreInput": 1,
            "name": "other"
          }
        ],
        "path": [
          "__add__"
        ],
        "receiver": 0
      }
    ],
    "egglog.core.bigrat.cbrt": [
      {
        "form": "METHOD",
        "owner": 1,
        "path": [
          "cbrt"
        ],
        "receiver": 0
      }
    ],
    "egglog.core.bigrat.ceil": [
      {
        "form": "METHOD",
        "owner": 1,
        "path": [
          "ceil"
        ],
        "receiver": 0
      }
    ],
    "egglog.core.bigrat.denom": [
      {
        "form": "PROPERTY",
        "owner": 1,
        "path": [
          "denom"
        ],
        "receiver": 0
      }
    ],
    "egglog.core.bigrat.div": [
      {
        "form": "METHOD",
        "owner": 1,
        "params": [
          {
            "coreInput": 1,
            "name": "other"
          }
        ],
        "path": [
          "__truediv__"
        ],
        "receiver": 0
      }
    ],
    "egglog.core.bigrat.floor": [
      {
        "form": "METHOD",
        "owner": 1,
        "path": [
          "floor"
        ],
        "receiver": 0
      }
    ],
    "egglog.core.bigrat.ge": [
      {
        "form": "METHOD",
        "owner": 1,
        "params": [
          {
            "coreInput": 1,
            "name": "other"
          }
        ],
        "path": [
          "__ge__"
        ],
        "receiver": 0
      }
    ],
    "egglog.core.bigrat.gt": [
      {
        "form": "METHOD",
        "owner": 1,
        "params": [
          {
            "coreInput": 1,
            "name": "other"
          }
        ],
        "path": [
          "__gt__"
        ],
        "receiver": 0
      }
    ],
    "egglog.core.bigrat.le": [
      {
        "form": "METHOD",
        "owner": 1,
        "params": [
          {
            "coreInput": 1,
            "name": "other"
          }
        ],
        "path": [
          "__le__"
        ],
        "receiver": 0
      }
    ],
    "egglog.core.bigrat.log": [
      {
        "form": "METHOD",
        "owner": 1,
        "path": [
          "log"
        ],
        "receiver": 0
      }
    ],
    "egglog.core.bigrat.lt": [
      {
        "form": "METHOD",
        "owner": 1,
        "params": [
          {
            "coreInput": 1,
            "name": "other"
          }
        ],
        "path": [
          "__lt__"
        ],
        "receiver": 0
      }
    ],
    "egglog.core.bigrat.make": [
      {
        "form": "INITIALIZER",
        "owner": 1,
        "params": [
          {
            "coreInput": 0,
            "name": "num"
          },
          {
            "coreInput": 1,
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
            "coreInput": 1,
            "name": "other"
          }
        ],
        "path": [
          "max"
        ],
        "receiver": 0
      }
    ],
    "egglog.core.bigrat.min": [
      {
        "form": "METHOD",
        "owner": 1,
        "params": [
          {
            "coreInput": 1,
            "name": "other"
          }
        ],
        "path": [
          "min"
        ],
        "receiver": 0
      }
    ],
    "egglog.core.bigrat.mul": [
      {
        "form": "METHOD",
        "owner": 1,
        "params": [
          {
            "coreInput": 1,
            "name": "other"
          }
        ],
        "path": [
          "__mul__"
        ],
        "receiver": 0
      }
    ],
    "egglog.core.bigrat.neg": [
      {
        "form": "METHOD",
        "owner": 1,
        "path": [
          "__neg__"
        ],
        "receiver": 0
      }
    ],
    "egglog.core.bigrat.numer": [
      {
        "form": "PROPERTY",
        "owner": 1,
        "path": [
          "numer"
        ],
        "receiver": 0
      }
    ],
    "egglog.core.bigrat.pow": [
      {
        "form": "METHOD",
        "owner": 1,
        "params": [
          {
            "coreInput": 1,
            "name": "other"
          }
        ],
        "path": [
          "__pow__"
        ],
        "receiver": 0
      }
    ],
    "egglog.core.bigrat.round": [
      {
        "form": "METHOD",
        "owner": 1,
        "path": [
          "round"
        ],
        "receiver": 0
      }
    ],
    "egglog.core.bigrat.sqrt": [
      {
        "form": "METHOD",
        "owner": 1,
        "path": [
          "sqrt"
        ],
        "receiver": 0
      }
    ],
    "egglog.core.bigrat.sub": [
      {
        "form": "METHOD",
        "owner": 1,
        "params": [
          {
            "coreInput": 1,
            "name": "other"
          }
        ],
        "path": [
          "__sub__"
        ],
        "receiver": 0
      }
    ],
    "egglog.core.bigrat.to_f64": [
      {
        "form": "METHOD",
        "owner": 1,
        "path": [
          "to_f64"
        ],
        "receiver": 0
      }
    ],
    "egglog.core.bigrat.to_i64": [
      {
        "form": "METHOD",
        "owner": 1,
        "path": [
          "to_i64"
        ],
        "receiver": 0
      }
    ],
    "egglog.core.bool.and": [
      {
        "form": "METHOD",
        "owner": 2,
        "params": [
          {
            "coreInput": 1,
            "name": "other"
          }
        ],
        "path": [
          "__and__"
        ],
        "receiver": 0
      }
    ],
    "egglog.core.bool.implies": [
      {
        "form": "METHOD",
        "owner": 2,
        "params": [
          {
            "coreInput": 1,
            "name": "other"
          }
        ],
        "path": [
          "implies"
        ],
        "receiver": 0
      }
    ],
    "egglog.core.bool.not": [
      {
        "form": "METHOD",
        "owner": 2,
        "path": [
          "__invert__"
        ],
        "receiver": 0
      }
    ],
    "egglog.core.bool.or": [
      {
        "form": "METHOD",
        "owner": 2,
        "params": [
          {
            "coreInput": 1,
            "name": "other"
          }
        ],
        "path": [
          "__or__"
        ],
        "receiver": 0,
        "tailRepetitions": 2
      }
    ],
    "egglog.core.bool.xor": [
      {
        "form": "METHOD",
        "owner": 2,
        "params": [
          {
            "coreInput": 1,
            "name": "other"
          }
        ],
        "path": [
          "__xor__"
        ],
        "receiver": 0
      }
    ],
    "egglog.core.f64.abs": [
      {
        "form": "METHOD",
        "owner": 3,
        "path": [
          "__abs__"
        ],
        "receiver": 0
      }
    ],
    "egglog.core.f64.add": [
      {
        "form": "METHOD",
        "owner": 3,
        "params": [
          {
            "coreInput": 1,
            "name": "other"
          }
        ],
        "path": [
          "__add__"
        ],
        "receiver": 0
      }
    ],
    "egglog.core.f64.div": [
      {
        "form": "METHOD",
        "owner": 3,
        "params": [
          {
            "coreInput": 1,
            "name": "other"
          }
        ],
        "path": [
          "__truediv__"
        ],
        "receiver": 0
      }
    ],
    "egglog.core.f64.exp": [
      {
        "form": "METHOD",
        "owner": 3,
        "path": [
          "exp"
        ],
        "receiver": 0
      }
    ],
    "egglog.core.f64.from_i64": [
      {
        "form": "CLASS_METHOD",
        "owner": 3,
        "params": [
          {
            "coreInput": 0,
            "name": "i"
          }
        ],
        "path": [
          "from_i64"
        ]
      }
    ],
    "egglog.core.f64.ge": [
      {
        "form": "METHOD",
        "owner": 3,
        "params": [
          {
            "coreInput": 1,
            "name": "other"
          }
        ],
        "path": [
          "__ge__"
        ],
        "receiver": 0
      }
    ],
    "egglog.core.f64.gt": [
      {
        "form": "METHOD",
        "owner": 3,
        "params": [
          {
            "coreInput": 1,
            "name": "other"
          }
        ],
        "path": [
          "__gt__"
        ],
        "receiver": 0
      }
    ],
    "egglog.core.f64.le": [
      {
        "form": "METHOD",
        "owner": 3,
        "params": [
          {
            "coreInput": 1,
            "name": "other"
          }
        ],
        "path": [
          "__le__"
        ],
        "receiver": 0
      }
    ],
    "egglog.core.f64.log": [
      {
        "form": "METHOD",
        "owner": 3,
        "path": [
          "log"
        ],
        "receiver": 0
      }
    ],
    "egglog.core.f64.lt": [
      {
        "form": "METHOD",
        "owner": 3,
        "params": [
          {
            "coreInput": 1,
            "name": "other"
          }
        ],
        "path": [
          "__lt__"
        ],
        "receiver": 0
      }
    ],
    "egglog.core.f64.max": [
      {
        "form": "METHOD",
        "owner": 3,
        "params": [
          {
            "coreInput": 1,
            "name": "other"
          }
        ],
        "path": [
          "max"
        ],
        "receiver": 0
      }
    ],
    "egglog.core.f64.min": [
      {
        "form": "METHOD",
        "owner": 3,
        "params": [
          {
            "coreInput": 1,
            "name": "other"
          }
        ],
        "path": [
          "min"
        ],
        "receiver": 0
      }
    ],
    "egglog.core.f64.mul": [
      {
        "form": "METHOD",
        "owner": 3,
        "params": [
          {
            "coreInput": 1,
            "name": "other"
          }
        ],
        "path": [
          "__mul__"
        ],
        "receiver": 0
      }
    ],
    "egglog.core.f64.neg": [
      {
        "form": "METHOD",
        "owner": 3,
        "path": [
          "__neg__"
        ],
        "receiver": 0
      }
    ],
    "egglog.core.f64.pow": [
      {
        "form": "METHOD",
        "owner": 3,
        "params": [
          {
            "coreInput": 1,
            "name": "other"
          }
        ],
        "path": [
          "__pow__"
        ],
        "receiver": 0
      }
    ],
    "egglog.core.f64.rem": [
      {
        "form": "METHOD",
        "owner": 3,
        "params": [
          {
            "coreInput": 1,
            "name": "other"
          }
        ],
        "path": [
          "__mod__"
        ],
        "receiver": 0
      }
    ],
    "egglog.core.f64.sqrt": [
      {
        "form": "METHOD",
        "owner": 3,
        "path": [
          "sqrt"
        ],
        "receiver": 0
      }
    ],
    "egglog.core.f64.sub": [
      {
        "form": "METHOD",
        "owner": 3,
        "params": [
          {
            "coreInput": 1,
            "name": "other"
          }
        ],
        "path": [
          "__sub__"
        ],
        "receiver": 0
      }
    ],
    "egglog.core.f64.to_i64": [
      {
        "form": "METHOD",
        "owner": 3,
        "path": [
          "to_i64"
        ],
        "receiver": 0
      }
    ],
    "egglog.core.f64.to_string": [
      {
        "form": "METHOD",
        "owner": 3,
        "path": [
          "to_string"
        ],
        "receiver": 0
      }
    ],
    "egglog.core.i64.abs": [
      {
        "form": "METHOD",
        "owner": 4,
        "path": [
          "__abs__"
        ],
        "receiver": 0
      }
    ],
    "egglog.core.i64.add": [
      {
        "form": "METHOD",
        "owner": 4,
        "params": [
          {
            "coreInput": 1,
            "name": "other"
          }
        ],
        "path": [
          "__add__"
        ],
        "receiver": 0
      }
    ],
    "egglog.core.i64.bitand": [
      {
        "form": "METHOD",
        "owner": 4,
        "params": [
          {
            "coreInput": 1,
            "name": "other"
          }
        ],
        "path": [
          "__and__"
        ],
        "receiver": 0
      }
    ],
    "egglog.core.i64.bitor": [
      {
        "form": "METHOD",
        "owner": 4,
        "params": [
          {
            "coreInput": 1,
            "name": "other"
          }
        ],
        "path": [
          "__or__"
        ],
        "receiver": 0
      }
    ],
    "egglog.core.i64.bitxor": [
      {
        "form": "METHOD",
        "owner": 4,
        "params": [
          {
            "coreInput": 1,
            "name": "other"
          }
        ],
        "path": [
          "__xor__"
        ],
        "receiver": 0
      }
    ],
    "egglog.core.i64.bool_ge": [
      {
        "form": "METHOD",
        "owner": 4,
        "params": [
          {
            "coreInput": 1,
            "name": "other"
          }
        ],
        "path": [
          "bool_ge"
        ],
        "receiver": 0
      }
    ],
    "egglog.core.i64.bool_gt": [
      {
        "form": "METHOD",
        "owner": 4,
        "params": [
          {
            "coreInput": 1,
            "name": "other"
          }
        ],
        "path": [
          "bool_gt"
        ],
        "receiver": 0
      }
    ],
    "egglog.core.i64.bool_le": [
      {
        "form": "METHOD",
        "owner": 4,
        "params": [
          {
            "coreInput": 1,
            "name": "other"
          }
        ],
        "path": [
          "bool_le"
        ],
        "receiver": 0
      }
    ],
    "egglog.core.i64.bool_lt": [
      {
        "form": "METHOD",
        "owner": 4,
        "params": [
          {
            "coreInput": 1,
            "name": "other"
          }
        ],
        "path": [
          "bool_lt"
        ],
        "receiver": 0
      }
    ],
    "egglog.core.i64.div": [
      {
        "form": "METHOD",
        "owner": 4,
        "params": [
          {
            "coreInput": 1,
            "name": "other"
          }
        ],
        "path": [
          "__truediv__"
        ],
        "receiver": 0
      }
    ],
    "egglog.core.i64.ge": [
      {
        "form": "METHOD",
        "owner": 4,
        "params": [
          {
            "coreInput": 1,
            "name": "other"
          }
        ],
        "path": [
          "__ge__"
        ],
        "receiver": 0
      }
    ],
    "egglog.core.i64.gt": [
      {
        "form": "METHOD",
        "owner": 4,
        "params": [
          {
            "coreInput": 1,
            "name": "other"
          }
        ],
        "path": [
          "__gt__"
        ],
        "receiver": 0
      }
    ],
    "egglog.core.i64.le": [
      {
        "form": "METHOD",
        "owner": 4,
        "params": [
          {
            "coreInput": 1,
            "name": "other"
          }
        ],
        "path": [
          "__le__"
        ],
        "receiver": 0
      }
    ],
    "egglog.core.i64.log2": [
      {
        "form": "METHOD",
        "owner": 4,
        "path": [
          "log2"
        ],
        "receiver": 0
      }
    ],
    "egglog.core.i64.lt": [
      {
        "form": "METHOD",
        "owner": 4,
        "params": [
          {
            "coreInput": 1,
            "name": "other"
          }
        ],
        "path": [
          "__lt__"
        ],
        "receiver": 0
      }
    ],
    "egglog.core.i64.max": [
      {
        "form": "METHOD",
        "owner": 4,
        "params": [
          {
            "coreInput": 1,
            "name": "other"
          }
        ],
        "path": [
          "max"
        ],
        "receiver": 0
      }
    ],
    "egglog.core.i64.min": [
      {
        "form": "METHOD",
        "owner": 4,
        "params": [
          {
            "coreInput": 1,
            "name": "other"
          }
        ],
        "path": [
          "min"
        ],
        "receiver": 0
      }
    ],
    "egglog.core.i64.mul": [
      {
        "form": "METHOD",
        "owner": 4,
        "params": [
          {
            "coreInput": 1,
            "name": "other"
          }
        ],
        "path": [
          "__mul__"
        ],
        "receiver": 0
      }
    ],
    "egglog.core.i64.not": [
      {
        "form": "METHOD",
        "owner": 4,
        "path": [
          "__invert__"
        ],
        "receiver": 0
      }
    ],
    "egglog.core.i64.rem": [
      {
        "form": "METHOD",
        "owner": 4,
        "params": [
          {
            "coreInput": 1,
            "name": "other"
          }
        ],
        "path": [
          "__mod__"
        ],
        "receiver": 0
      }
    ],
    "egglog.core.i64.shl": [
      {
        "form": "METHOD",
        "owner": 4,
        "params": [
          {
            "coreInput": 1,
            "name": "other"
          }
        ],
        "path": [
          "__lshift__"
        ],
        "receiver": 0
      }
    ],
    "egglog.core.i64.shr": [
      {
        "form": "METHOD",
        "owner": 4,
        "params": [
          {
            "coreInput": 1,
            "name": "other"
          }
        ],
        "path": [
          "__rshift__"
        ],
        "receiver": 0
      }
    ],
    "egglog.core.i64.sub": [
      {
        "form": "METHOD",
        "owner": 4,
        "params": [
          {
            "coreInput": 1,
            "name": "other"
          }
        ],
        "path": [
          "__sub__"
        ],
        "receiver": 0
      }
    ],
    "egglog.core.i64.to_string": [
      {
        "form": "METHOD",
        "owner": 4,
        "path": [
          "to_string"
        ],
        "receiver": 0
      }
    ],
    "egglog.core.map.contains": [
      {
        "form": "METHOD",
        "owner": 5,
        "params": [
          {
            "coreInput": 1,
            "name": "key"
          }
        ],
        "path": [
          "contains"
        ],
        "receiver": 0
      }
    ],
    "egglog.core.map.empty": [
      {
        "form": "CLASS_METHOD",
        "owner": 5,
        "path": [
          "empty"
        ]
      }
    ],
    "egglog.core.map.get": [
      {
        "form": "METHOD",
        "owner": 5,
        "params": [
          {
            "coreInput": 1,
            "name": "key"
          }
        ],
        "path": [
          "__getitem__"
        ],
        "receiver": 0
      }
    ],
    "egglog.core.map.insert": [
      {
        "form": "METHOD",
        "owner": 5,
        "params": [
          {
            "coreInput": 1,
            "name": "key"
          },
          {
            "coreInput": 2,
            "name": "value"
          }
        ],
        "path": [
          "insert"
        ],
        "receiver": 0
      }
    ],
    "egglog.core.map.length": [
      {
        "form": "METHOD",
        "owner": 5,
        "path": [
          "length"
        ],
        "receiver": 0
      }
    ],
    "egglog.core.map.not_contains": [
      {
        "form": "METHOD",
        "owner": 5,
        "params": [
          {
            "coreInput": 1,
            "name": "key"
          }
        ],
        "path": [
          "not_contains"
        ],
        "receiver": 0
      }
    ],
    "egglog.core.map.of": [
      {
        "form": "CLASS_METHOD",
        "owner": 5,
        "params": [
          {
            "coreInput": 0,
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
            "coreInput": 1,
            "name": "key"
          }
        ],
        "path": [
          "remove"
        ],
        "receiver": 0
      }
    ],
    "egglog.core.multiset.clear_index": [
      {
        "form": "METHOD",
        "owner": 6,
        "params": [
          {
            "coreInput": 1,
            "name": "f"
          }
        ],
        "path": [
          "clear_index"
        ],
        "receiver": 0
      }
    ],
    "egglog.core.multiset.contains": [
      {
        "form": "METHOD",
        "owner": 6,
        "params": [
          {
            "coreInput": 1,
            "name": "value"
          }
        ],
        "path": [
          "contains"
        ],
        "receiver": 0
      }
    ],
    "egglog.core.multiset.contains_swapped": [
      {
        "form": "FUNCTION",
        "params": [
          {
            "coreInput": 0,
            "name": "x"
          },
          {
            "coreInput": 1,
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
            "coreInput": 1,
            "name": "value"
          }
        ],
        "path": [
          "count"
        ],
        "receiver": 0
      }
    ],
    "egglog.core.multiset.fill_index": [
      {
        "form": "METHOD",
        "owner": 6,
        "params": [
          {
            "coreInput": 1,
            "name": "f"
          }
        ],
        "path": [
          "fill_index"
        ],
        "receiver": 0
      }
    ],
    "egglog.core.multiset.filter": [
      {
        "form": "METHOD",
        "owner": 6,
        "params": [
          {
            "coreInput": 0,
            "name": "f"
          }
        ],
        "path": [
          "filter"
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
            "coreInput": 0,
            "name": "f"
          }
        ],
        "path": [
          "filter_not"
        ],
        "receiver": 1
      }
    ],
    "egglog.core.multiset.flat_map": [
      {
        "form": "FUNCTION",
        "params": [
          {
            "coreInput": 0,
            "name": "f"
          },
          {
            "coreInput": 1,
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
            "coreInput": 1,
            "name": "value"
          }
        ],
        "path": [
          "insert"
        ],
        "receiver": 0
      }
    ],
    "egglog.core.multiset.intersection": [
      {
        "form": "METHOD",
        "owner": 6,
        "params": [
          {
            "coreInput": 1,
            "name": "other"
          }
        ],
        "path": [
          "__and__"
        ],
        "receiver": 0
      }
    ],
    "egglog.core.multiset.length": [
      {
        "form": "METHOD",
        "owner": 6,
        "path": [
          "length"
        ],
        "receiver": 0
      }
    ],
    "egglog.core.multiset.map": [
      {
        "form": "METHOD",
        "owner": 6,
        "params": [
          {
            "coreInput": 0,
            "name": "f"
          }
        ],
        "path": [
          "map"
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
            "coreInput": 1,
            "name": "value"
          }
        ],
        "path": [
          "not_contains"
        ],
        "receiver": 0
      }
    ],
    "egglog.core.multiset.not_contains_swapped": [
      {
        "form": "FUNCTION",
        "params": [
          {
            "coreInput": 0,
            "name": "x"
          },
          {
            "coreInput": 1,
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
            "coreInput": 0,
            "name": "args"
          }
        ]
      }
    ],
    "egglog.core.multiset.pick": [
      {
        "form": "METHOD",
        "owner": 6,
        "path": [
          "pick"
        ],
        "receiver": 0
      }
    ],
    "egglog.core.multiset.pick_max": [
      {
        "form": "METHOD",
        "owner": 6,
        "path": [
          "pick_max"
        ],
        "receiver": 0
      }
    ],
    "egglog.core.multiset.reduce": [
      {
        "form": "FUNCTION",
        "params": [
          {
            "coreInput": 0,
            "name": "f"
          },
          {
            "coreInput": 1,
            "name": "initial"
          },
          {
            "coreInput": 2,
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
            "coreInput": 1,
            "name": "value"
          }
        ],
        "path": [
          "remove"
        ],
        "receiver": 0
      }
    ],
    "egglog.core.multiset.remove_swapped": [
      {
        "form": "FUNCTION",
        "params": [
          {
            "coreInput": 0,
            "name": "x"
          },
          {
            "coreInput": 1,
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
        "owner": 6,
        "path": [
          "reset_counts"
        ],
        "receiver": 0
      }
    ],
    "egglog.core.multiset.single": [
      {
        "form": "CLASS_METHOD",
        "owner": 6,
        "params": [
          {
            "coreInput": 0,
            "name": "x"
          },
          {
            "coreInput": 1,
            "name": "i"
          }
        ],
        "path": [
          "single"
        ]
      }
    ],
    "egglog.core.multiset.subtract": [
      {
        "form": "METHOD",
        "owner": 6,
        "params": [
          {
            "coreInput": 1,
            "name": "other"
          }
        ],
        "path": [
          "__sub__"
        ],
        "receiver": 0
      }
    ],
    "egglog.core.multiset.subtract_swapped": [
      {
        "form": "FUNCTION",
        "params": [
          {
            "coreInput": 0,
            "name": "x"
          },
          {
            "coreInput": 1,
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
            "coreInput": 1,
            "name": "other"
          }
        ],
        "path": [
          "__add__"
        ],
        "receiver": 0
      }
    ],
    "egglog.core.multiset.sum_multisets": [
      {
        "form": "CLASS_METHOD",
        "owner": 6,
        "params": [
          {
            "coreInput": 0,
            "name": "xs"
          }
        ],
        "path": [
          "sum_multisets"
        ]
      }
    ],
    "egglog.core.multiset.union_values": [
      {
        "form": "METHOD",
        "owner": 6,
        "path": [
          "union_values"
        ],
        "receiver": 0
      }
    ],
    "egglog.core.pair.first": [
      {
        "form": "PROPERTY",
        "owner": 7,
        "path": [
          "left"
        ],
        "receiver": 0
      }
    ],
    "egglog.core.pair.make": [
      {
        "form": "INITIALIZER",
        "owner": 7,
        "params": [
          {
            "coreInput": 0,
            "name": "left"
          },
          {
            "coreInput": 1,
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
        ],
        "receiver": 0
      }
    ],
    "egglog.core.set.contains": [
      {
        "form": "METHOD",
        "owner": 8,
        "params": [
          {
            "coreInput": 1,
            "name": "value"
          }
        ],
        "path": [
          "contains"
        ],
        "receiver": 0
      }
    ],
    "egglog.core.set.diff": [
      {
        "form": "METHOD",
        "owner": 8,
        "params": [
          {
            "coreInput": 1,
            "name": "other"
          }
        ],
        "path": [
          "__sub__"
        ],
        "receiver": 0
      }
    ],
    "egglog.core.set.empty": [
      {
        "form": "CLASS_METHOD",
        "owner": 8,
        "path": [
          "empty"
        ]
      }
    ],
    "egglog.core.set.insert": [
      {
        "form": "METHOD",
        "owner": 8,
        "params": [
          {
            "coreInput": 1,
            "name": "value"
          }
        ],
        "path": [
          "insert"
        ],
        "receiver": 0
      }
    ],
    "egglog.core.set.intersect": [
      {
        "form": "METHOD",
        "owner": 8,
        "params": [
          {
            "coreInput": 1,
            "name": "other"
          }
        ],
        "path": [
          "__and__"
        ],
        "receiver": 0
      }
    ],
    "egglog.core.set.length": [
      {
        "form": "METHOD",
        "owner": 8,
        "path": [
          "length"
        ],
        "receiver": 0
      }
    ],
    "egglog.core.set.not_contains": [
      {
        "form": "METHOD",
        "owner": 8,
        "params": [
          {
            "coreInput": 1,
            "name": "value"
          }
        ],
        "path": [
          "not_contains"
        ],
        "receiver": 0
      }
    ],
    "egglog.core.set.of": [
      {
        "form": "INITIALIZER",
        "owner": 8,
        "params": [
          {
            "coreInput": 0,
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
            "coreInput": 1,
            "name": "value"
          }
        ],
        "path": [
          "remove"
        ],
        "receiver": 0
      }
    ],
    "egglog.core.set.union": [
      {
        "form": "METHOD",
        "owner": 8,
        "params": [
          {
            "coreInput": 1,
            "name": "other"
          }
        ],
        "path": [
          "__or__"
        ],
        "receiver": 0
      }
    ],
    "egglog.core.string.concat": [
      {
        "form": "FUNCTION",
        "params": [
          {
            "coreInput": 0,
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
            "coreInput": 1,
            "name": "other"
          }
        ],
        "path": [
          "__add__"
        ],
        "receiver": 0,
        "tailRepetitions": 2
      }
    ],
    "egglog.core.string.count_matches": [
      {
        "form": "FUNCTION",
        "params": [
          {
            "coreInput": 0,
            "name": "s"
          },
          {
            "coreInput": 1,
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
            "coreInput": 1,
            "name": "old"
          },
          {
            "coreInput": 2,
            "name": "new"
          }
        ],
        "path": [
          "replace"
        ],
        "receiver": 0
      }
    ],
    "egglog.core.vec.append": [
      {
        "form": "METHOD",
        "owner": 10,
        "params": [
          {
            "coreInput": 1,
            "name": "others"
          }
        ],
        "path": [
          "append"
        ],
        "receiver": 0
      }
    ],
    "egglog.core.vec.contains": [
      {
        "form": "METHOD",
        "owner": 10,
        "params": [
          {
            "coreInput": 1,
            "name": "value"
          }
        ],
        "path": [
          "contains"
        ],
        "receiver": 0
      }
    ],
    "egglog.core.vec.empty": [
      {
        "form": "CLASS_METHOD",
        "owner": 10,
        "path": [
          "empty"
        ]
      }
    ],
    "egglog.core.vec.get": [
      {
        "form": "METHOD",
        "owner": 10,
        "params": [
          {
            "coreInput": 1,
            "name": "index"
          }
        ],
        "path": [
          "__getitem__"
        ],
        "receiver": 0
      }
    ],
    "egglog.core.vec.length": [
      {
        "form": "METHOD",
        "owner": 10,
        "path": [
          "length"
        ],
        "receiver": 0
      }
    ],
    "egglog.core.vec.map": [
      {
        "form": "METHOD",
        "owner": 10,
        "params": [
          {
            "coreInput": 0,
            "name": "f"
          }
        ],
        "path": [
          "map"
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
            "coreInput": 1,
            "name": "value"
          }
        ],
        "path": [
          "not_contains"
        ],
        "receiver": 0
      }
    ],
    "egglog.core.vec.of": [
      {
        "form": "INITIALIZER",
        "owner": 10,
        "params": [
          {
            "coreInput": 0,
            "name": "args"
          }
        ]
      }
    ],
    "egglog.core.vec.pop": [
      {
        "form": "METHOD",
        "owner": 10,
        "path": [
          "pop"
        ],
        "receiver": 0
      }
    ],
    "egglog.core.vec.push": [
      {
        "form": "METHOD",
        "owner": 10,
        "params": [
          {
            "coreInput": 1,
            "name": "value"
          }
        ],
        "path": [
          "push"
        ],
        "receiver": 0
      }
    ],
    "egglog.core.vec.range": [
      {
        "form": "METHOD",
        "owner": 4,
        "path": [
          "range"
        ],
        "receiver": 0
      }
    ],
    "egglog.core.vec.remove": [
      {
        "form": "METHOD",
        "owner": 10,
        "params": [
          {
            "coreInput": 1,
            "name": "index"
          }
        ],
        "path": [
          "remove"
        ],
        "receiver": 0
      }
    ],
    "egglog.core.vec.set": [
      {
        "form": "METHOD",
        "owner": 10,
        "params": [
          {
            "coreInput": 1,
            "name": "index"
          },
          {
            "coreInput": 2,
            "name": "value"
          }
        ],
        "path": [
          "set"
        ],
        "receiver": 0
      }
    ],
    "egglog.core.vec.union": [
      {
        "form": "METHOD",
        "owner": 10,
        "params": [
          {
            "coreInput": 1,
            "name": "other"
          }
        ],
        "path": [
          "__or__"
        ],
        "receiver": 0
      }
    ],
    "egglog.experimental.f64.is_finite": [
      {
        "form": "METHOD",
        "owner": 3,
        "path": [
          "is_finite"
        ],
        "receiver": 0
      }
    ],
    "egglog.experimental.map.fold_kv": [
      {
        "form": "FUNCTION",
        "params": [
          {
            "coreInput": 0,
            "name": "f"
          },
          {
            "coreInput": 1,
            "name": "initial"
          },
          {
            "coreInput": 2,
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
            "coreInput": 0,
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
            "coreInput": 1,
            "name": "f"
          },
          {
            "coreInput": 2,
            "name": "n"
          }
        ],
        "path": [
          "match"
        ],
        "receiver": 0
      }
    ],
    "egglog.experimental.maybe.none": [
      {
        "form": "CLASS_METHOD",
        "owner": 11,
        "path": [
          "none"
        ]
      }
    ],
    "egglog.experimental.maybe.some": [
      {
        "form": "CLASS_METHOD",
        "owner": 11,
        "params": [
          {
            "coreInput": 0,
            "name": "value"
          }
        ],
        "path": [
          "some"
        ]
      }
    ],
    "egglog.experimental.maybe.unwrap": [
      {
        "form": "METHOD",
        "owner": 11,
        "path": [
          "unwrap"
        ],
        "receiver": 0
      }
    ],
    "egglog.experimental.maybe.unwrap_or": [
      {
        "form": "METHOD",
        "owner": 11,
        "params": [
          {
            "coreInput": 1,
            "name": "default"
          }
        ],
        "path": [
          "unwrap_or"
        ],
        "receiver": 0
      }
    ],
    "egglog.experimental.rational.abs": [
      {
        "form": "METHOD",
        "owner": 12,
        "path": [
          "__abs__"
        ],
        "receiver": 0
      }
    ],
    "egglog.experimental.rational.add": [
      {
        "form": "METHOD",
        "owner": 12,
        "params": [
          {
            "coreInput": 1,
            "name": "other"
          }
        ],
        "path": [
          "__add__"
        ],
        "receiver": 0
      }
    ],
    "egglog.experimental.rational.cbrt": [
      {
        "form": "METHOD",
        "owner": 12,
        "path": [
          "cbrt"
        ],
        "receiver": 0
      }
    ],
    "egglog.experimental.rational.ceil": [
      {
        "form": "METHOD",
        "owner": 12,
        "path": [
          "ceil"
        ],
        "receiver": 0
      }
    ],
    "egglog.experimental.rational.denom": [
      {
        "form": "PROPERTY",
        "owner": 12,
        "path": [
          "denom"
        ],
        "receiver": 0
      }
    ],
    "egglog.experimental.rational.div": [
      {
        "form": "METHOD",
        "owner": 12,
        "params": [
          {
            "coreInput": 1,
            "name": "other"
          }
        ],
        "path": [
          "__truediv__"
        ],
        "receiver": 0
      }
    ],
    "egglog.experimental.rational.floor": [
      {
        "form": "METHOD",
        "owner": 12,
        "path": [
          "floor"
        ],
        "receiver": 0
      }
    ],
    "egglog.experimental.rational.ge": [
      {
        "form": "METHOD",
        "owner": 12,
        "params": [
          {
            "coreInput": 1,
            "name": "other"
          }
        ],
        "path": [
          "__ge__"
        ],
        "receiver": 0
      }
    ],
    "egglog.experimental.rational.gt": [
      {
        "form": "METHOD",
        "owner": 12,
        "params": [
          {
            "coreInput": 1,
            "name": "other"
          }
        ],
        "path": [
          "__gt__"
        ],
        "receiver": 0
      }
    ],
    "egglog.experimental.rational.le": [
      {
        "form": "METHOD",
        "owner": 12,
        "params": [
          {
            "coreInput": 1,
            "name": "other"
          }
        ],
        "path": [
          "__le__"
        ],
        "receiver": 0
      }
    ],
    "egglog.experimental.rational.log": [
      {
        "form": "METHOD",
        "owner": 12,
        "path": [
          "log"
        ],
        "receiver": 0
      }
    ],
    "egglog.experimental.rational.lt": [
      {
        "form": "METHOD",
        "owner": 12,
        "params": [
          {
            "coreInput": 1,
            "name": "other"
          }
        ],
        "path": [
          "__lt__"
        ],
        "receiver": 0
      }
    ],
    "egglog.experimental.rational.make": [
      {
        "form": "INITIALIZER",
        "owner": 12,
        "params": [
          {
            "coreInput": 0,
            "name": "num"
          },
          {
            "coreInput": 1,
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
            "coreInput": 1,
            "name": "other"
          }
        ],
        "path": [
          "max"
        ],
        "receiver": 0
      }
    ],
    "egglog.experimental.rational.min": [
      {
        "form": "METHOD",
        "owner": 12,
        "params": [
          {
            "coreInput": 1,
            "name": "other"
          }
        ],
        "path": [
          "min"
        ],
        "receiver": 0
      }
    ],
    "egglog.experimental.rational.mul": [
      {
        "form": "METHOD",
        "owner": 12,
        "params": [
          {
            "coreInput": 1,
            "name": "other"
          }
        ],
        "path": [
          "__mul__"
        ],
        "receiver": 0
      }
    ],
    "egglog.experimental.rational.neg": [
      {
        "form": "METHOD",
        "owner": 12,
        "path": [
          "__neg__"
        ],
        "receiver": 0
      }
    ],
    "egglog.experimental.rational.numer": [
      {
        "form": "PROPERTY",
        "owner": 12,
        "path": [
          "numer"
        ],
        "receiver": 0
      }
    ],
    "egglog.experimental.rational.pow": [
      {
        "form": "METHOD",
        "owner": 12,
        "params": [
          {
            "coreInput": 1,
            "name": "other"
          }
        ],
        "path": [
          "__pow__"
        ],
        "receiver": 0
      }
    ],
    "egglog.experimental.rational.round": [
      {
        "form": "METHOD",
        "owner": 12,
        "path": [
          "round"
        ],
        "receiver": 0
      }
    ],
    "egglog.experimental.rational.sqrt": [
      {
        "form": "METHOD",
        "owner": 12,
        "path": [
          "sqrt"
        ],
        "receiver": 0
      }
    ],
    "egglog.experimental.rational.sub": [
      {
        "form": "METHOD",
        "owner": 12,
        "params": [
          {
            "coreInput": 1,
            "name": "other"
          }
        ],
        "path": [
          "__sub__"
        ],
        "receiver": 0
      }
    ],
    "egglog.experimental.rational.to_f64": [
      {
        "form": "METHOD",
        "owner": 12,
        "path": [
          "to_f64"
        ],
        "receiver": 0
      }
    ],
    "egglog.python.pyobject.call": [
      {
        "form": "METHOD",
        "owner": 13,
        "params": [
          {
            "coreInput": 1,
            "name": "args"
          }
        ],
        "path": [
          "__call__"
        ],
        "receiver": 0
      }
    ],
    "egglog.python.pyobject.call_extended": [
      {
        "form": "METHOD",
        "owner": 13,
        "params": [
          {
            "coreInput": 1,
            "name": "args"
          },
          {
            "coreInput": 2,
            "name": "kwargs"
          }
        ],
        "path": [
          "call_extended"
        ],
        "receiver": 0
      }
    ],
    "egglog.python.pyobject.dict": [
      {
        "form": "CLASS_METHOD",
        "owner": 13,
        "params": [
          {
            "coreInput": 0,
            "name": "keys_and_values"
          }
        ],
        "path": [
          "dict"
        ]
      }
    ],
    "egglog.python.pyobject.dict_update": [
      {
        "form": "METHOD",
        "owner": 13,
        "params": [
          {
            "coreInput": 1,
            "name": "keys_and_values"
          }
        ],
        "path": [
          "dict_update"
        ],
        "receiver": 0
      }
    ],
    "egglog.python.pyobject.eval": [
      {
        "form": "FUNCTION",
        "params": [
          {
            "coreInput": 0,
            "name": "code"
          },
          {
            "coreInput": 1,
            "default": 0,
            "name": "globals_"
          },
          {
            "coreInput": 2,
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
            "coreInput": 0,
            "name": "code"
          },
          {
            "coreInput": 1,
            "default": 1,
            "name": "globals_"
          },
          {
            "coreInput": 2,
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
            "coreInput": 0,
            "name": "i"
          }
        ],
        "path": [
          "from_int"
        ]
      }
    ],
    "egglog.python.pyobject.from_string": [
      {
        "form": "CLASS_METHOD",
        "owner": 13,
        "params": [
          {
            "coreInput": 0,
            "name": "s"
          }
        ],
        "path": [
          "from_string"
        ]
      }
    ],
    "egglog.python.pyobject.to_bool": [
      {
        "form": "METHOD",
        "owner": 13,
        "path": [
          "to_bool"
        ],
        "receiver": 0
      }
    ],
    "egglog.python.pyobject.to_string": [
      {
        "form": "METHOD",
        "owner": 13,
        "path": [
          "to_string"
        ],
        "receiver": 0
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
        "application": {
          "name": "BigInt"
        }
      },
      {
        "application": {
          "name": "BigRat"
        }
      },
      {
        "application": {
          "name": "bool"
        }
      },
      {
        "application": {
          "name": "f64"
        }
      },
      {
        "application": {
          "name": "i64"
        }
      },
      {
        "application": {
          "args": [
            14,
            15
          ],
          "name": "Map"
        }
      },
      {
        "application": {
          "args": [
            14
          ],
          "name": "MultiSet"
        }
      },
      {
        "application": {
          "args": [
            14,
            15
          ],
          "name": "Pair"
        }
      },
      {
        "application": {
          "args": [
            14
          ],
          "name": "Set"
        }
      },
      {
        "application": {
          "name": "String"
        }
      },
      {
        "application": {
          "args": [
            14
          ],
          "name": "Vec"
        }
      },
      {
        "application": {
          "args": [
            14
          ],
          "name": "Maybe"
        }
      },
      {
        "application": {
          "name": "Rational"
        }
      },
      {
        "application": {
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
