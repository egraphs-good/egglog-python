---
file_format: mystnb
kernelspec:
  name: python3
---

# Compared to Rust implementation

This package serves two purposes:

1. Exposing the machinery of the Rust implementation to Python.
2. Providing a Pythonic interface to build e-graphs.

Instead of skipping directly to the second step, we first expose the primitives
from the rust library as close as we can to the original API. Then we build a
higher level API on top of that, that is more friendly to use.

We can show show these two APIs starting witht he eqsat_basic example. The
egg text version of this from the tests is:

```lisp
(datatype Math
  (Num i64)
  (Var String)
  (Add Math Math)
  (Mul Math Math))

;; expr1 = 2 * (x + 3)
(define expr1 (Mul (Num 2) (Add (Var "x") (Num 3))))
;; expr2 = 6 + 2 * x
(define expr2 (Add (Num 6) (Mul (Num 2) (Var "x"))))


;; (rule ((= __root (Add a b)))
;;       ((union __root (Add b a)))
(rewrite (Add a b)
         (Add b a))
(rewrite (Mul a (Add b c))
         (Add (Mul a b) (Mul a c)))
(rewrite (Add (Num a) (Num b))
         (Num (+ a b)))
(rewrite (Mul (Num a) (Num b))
         (Num (* a b)))

(run 10)
(check (= expr1 expr2))
```

## Text API

One way to run this in Python is to parse the text and run it similar to how the
egg CLI works:

```{code-cell} python
from egg_smol.bindings import *

eqsat_basic = """(datatype Math
  (Num i64)
  (Var String)
  (Add Math Math)
  (Mul Math Math))

;; expr1 = 2 * (x + 3)
(define expr1 (Mul (Num 2) (Add (Var "x") (Num 3))))
;; expr2 = 6 + 2 * x
(define expr2 (Add (Num 6) (Mul (Num 2) (Var "x"))))


;; (rule ((= __root (Add a b)))
;;       ((union __root (Add b a)))
(rewrite (Add a b)
         (Add b a))
(rewrite (Mul a (Add b c))
         (Add (Mul a b) (Mul a c)))
(rewrite (Add (Num a) (Num b))
         (Num (+ a b)))
(rewrite (Mul (Num a) (Num b))
         (Num (* a b)))

(run 10)
(check (= expr1 expr2))"""

egraph = EGraph()
egraph.parse_and_run_program(eqsat_basic)
```

## Low level bindings API

However, this isn't the most friendly for Python users. Instead, we can use the
low level APIs that mirror the rust APIs to build the same egraph:

```{code-cell} python
egraph = EGraph()
egraph.declare_sort("Math")
egraph.declare_constructor(Variant("Num", ["i64"]), "Math")
egraph.declare_constructor(Variant("Var", ["String"]), "Math")
egraph.declare_constructor(Variant("Add", ["Math", "Math"]), "Math")
egraph.declare_constructor(Variant("Mul", ["Math", "Math"]), "Math")

# (define expr1 (Mul (Num 2) (Add (Var "x") (Num 3))))
egraph.define(
    "expr1",
    Call(
        "Mul",
        [
            Call(
                "Num",
                [
                    Lit(Int(2)),
                ],
            ),
            Call(
                "Add",
                [
                    Call(
                        "Var",
                        [
                            Lit(String("x")),
                        ],
                    ),
                    Call(
                        "Num",
                        [
                            Lit(Int(3)),
                        ],
                    ),
                ],
            ),
        ],
    ),
)
# (define expr2 (Add (Num 6) (Mul (Num 2) (Var "x"))))
egraph.define(
    "expr2",
    Call(
        "Add",
        [
            Call(
                "Num",
                [
                    Lit(Int(6)),
                ],
            ),
            Call(
                "Mul",
                [
                    Call(
                        "Num",
                        [
                            Lit(Int(2)),
                        ],
                    ),
                    Call(
                        "Var",
                        [
                            Lit(String("x")),
                        ],
                    ),
                ],
            ),
        ],
    ),
)
# (rewrite (Add a b)
#          (Add b a))
egraph.add_rewrite(
    Rewrite(
        Call(
            "Add",
            [
                Var("a"),
                Var("b"),
            ],
        ),
        Call(
            "Add",
            [
                Var("b"),
                Var("a"),
            ],
        ),
    )
)
# (rewrite (Mul a (Add b c))
#          (Add (Mul a b) (Mul a c)))
egraph.add_rewrite(
    Rewrite(
        Call(
            "Mul",
            [
                Var("a"),
                Call(
                    "Add",
                    [
                        Var("b"),
                        Var("c"),
                    ],
                ),
            ],
        ),
        Call(
            "Add",
            [
                Call(
                    "Mul",
                    [
                        Var("a"),
                        Var("b"),
                    ],
                ),
                Call(
                    "Mul",
                    [
                        Var("a"),
                        Var("c"),
                    ],
                ),
            ],
        ),
    )
)

# (rewrite (Add (Num a) (Num b))
#          (Num (+ a b)))
lhs = Call(
    "Add",
    [
        Call(
            "Num",
            [
                Var("a"),
            ],
        ),
        Call(
            "Num",
            [
                Var("b"),
            ],
        ),
    ],
)
rhs = Call(
    "Num",
    [
        Call(
            "+",
            [
                Var("a"),
                Var("b"),
            ],
        )
    ],
)
egraph.add_rewrite(Rewrite(lhs, rhs))

# (rewrite (Mul (Num a) (Num b))
#          (Num (* a b)))
lhs = Call(
    "Mul",
    [
        Call(
            "Num",
            [
                Var("a"),
            ],
        ),
        Call(
            "Num",
            [
                Var("b"),
            ],
        ),
    ],
)
rhs = Call(
    "Num",
    [
        Call(
            "*",
            [
                Var("a"),
                Var("b"),
            ],
        )
    ],
)
egraph.add_rewrite(Rewrite(lhs, rhs))

egraph.run_rules(10)
egraph.check_fact(
    Eq(
        [
            Var("expr1"),
            Var("expr2"),
        ]
    )
)
```

This has a couple of advantages over the text version. Users now know what types
of functions are available to them and also it can be statically type checked with MyPy,
to make sure that the types are correct.

However, it is much more verbose than the text version!

## High level API

So would it be possible to make an API that:

1. Statically type checks as much as possible with MyPy
2. Is concise to write
3. Feels "pythonic"

TODO: Finish this section
