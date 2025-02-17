---
file_format: mystnb
---

# Low Level: `egglog.bindings`

This modules contains bindings for the rust `egglog` library which are meant to be as close to the source as possible. This might result in not the most ergonomic API, if so, we can build higher level abstractions on top of it.

Example:

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

```{eval-rst}
.. automodule:: egglog.bindings
   :members:
   :undoc-members:
```
