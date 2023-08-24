---
file_format: mystnb
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

## Low Level API

One way to run this in Python is to parse the text and run it similar to how the
egglog CLI works:

```{code-cell} python
from egglog.bindings import *

eqsat_basic = """(datatype Math
  (Num i64)
  (Var String)
  (Add Math Math)
  (Mul Math Math))

;; expr1 = 2 * (x + 3)
(let expr1 (Mul (Num 2) (Add (Var "x") (Num 3))))
;; expr2 = 6 + 2 * x
(let expr2 (Add (Num 6) (Mul (Num 2) (Var "x"))))


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
commands = egraph.parse_program(eqsat_basic)
egraph.run_program(*commands)
```

The commands are a representation which is close the AST of the egglog text language. We
can see this by printing the commands:

```{code-cell} python
for command in commands:
    print(command)
```

## High level API

The high level API builds on this API and is designed to:

1. Statically type checks as much as possible with MyPy
2. Be concise to write
3. Feels "pythonic"

Here is the same example using the high level API:

```{code-cell} python
from __future__ import annotations

from egglog import *

egraph = EGraph()

@egraph.class_
class Math(Expr):
    def __init__(self, value: i64Like) -> None:
        ...

    @classmethod
    def var(cls, v: StringLike) -> Math:
        ...

    def __add__(self, other: Math) -> Math:
        ...

    def __mul__(self, other: Math) -> Math:
        ...

# expr1 = 2 * (x + 3)
expr1 = egraph.let("expr1", Math(2) * (Math.var("x") + Math(3)))

# expr2 = 6 + 2 * x
expr2 = egraph.let("expr2", Math(6) + Math(2) * Math.var("x"))

a, b, c = vars_("a b c", Math)
x, y = vars_("x y", i64)

egraph.register(
    rewrite(a + b).to(b + a),
    rewrite(a * (b + c)).to((a * b) + (a * c)),
    rewrite(Math(x) + Math(y)).to(Math(x + y)),
    rewrite(Math(x) * Math(y)).to(Math(x * y)),
)

egraph.run(10)

egraph.check(eq(expr1).to(expr2))
```
